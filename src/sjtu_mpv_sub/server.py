"""Ephemeral FastAPI companion server with idle auto-exit and rotating file logging."""

from __future__ import annotations

import asyncio
import logging
import logging.handlers
import socket
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from sjtu_mpv_sub import __version__
from sjtu_mpv_sub.config import (
    AppConfig,
    get_course_preference,
    get_root_dir,
    load_config,
    upsert_course_preference,
)
from sjtu_mpv_sub.models import (
    CoursePreference,
    CoursePreferenceUpdate,
    JobStatus,
    PlayRequest,
)
from sjtu_mpv_sub.pipeline import LecturePipeline, make_session_key

logger = logging.getLogger(__name__)


def setup_logging(root_dir: Path) -> None:
    """Configure rotating file logs and terminal logging when interactive."""
    log_dir = root_dir / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "server.log"

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    root_logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    if not Path(sys.executable).name.lower().startswith("pythonw"):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        root_logger.addHandler(stream_handler)


def should_idle_exit(
    *,
    active_jobs: int,
    alive_mpv_count: int,
    idle_seconds: float,
    idle_limit_seconds: int,
) -> bool:
    """Return True when no jobs or mpv windows are active beyond the idle limit."""
    if idle_limit_seconds <= 0:
        return False
    return active_jobs == 0 and alive_mpv_count == 0 and idle_seconds >= float(idle_limit_seconds)


class ServerRuntime:
    """In-memory runtime state for jobs, child mpv processes, and idle tracking."""

    def __init__(self, root_dir: Path, config: AppConfig) -> None:
        self.root_dir = root_dir
        self.config = config
        self.jobs: dict[str, JobStatus] = {}
        self.active_session_jobs: dict[str, str] = {}
        self.job_tasks: dict[str, asyncio.Task[Any]] = {}
        self.tasks: set[asyncio.Task[Any]] = set()
        self.mpv_procs: list[subprocess.Popen[bytes]] = []
        self.last_activity_at = time.monotonic()
        self.shutdown_requested = False
        self.uvicorn_server: uvicorn.Server | None = None

    def touch(self) -> None:
        self.last_activity_at = time.monotonic()

    def register_mpv_proc(self, proc: subprocess.Popen[bytes]) -> None:
        self.mpv_procs.append(proc)
        self.touch()

    def alive_mpv_count(self) -> int:
        self.mpv_procs = [p for p in self.mpv_procs if p.poll() is None]
        return len(self.mpv_procs)

    def active_jobs_count(self) -> int:
        return sum(
            1
            for job in self.jobs.values()
            if job.state in {"queued", "probing", "running", "playing"}
        )

    def cancel_job(self, job_id: str, reason_prefix: str = "已手动停止后台转录") -> JobStatus | None:
        """Cancel a running playback job if still active."""
        job = self.jobs.get(job_id)
        if job is None:
            return None
        task = self.job_tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
        if job.state not in {"completed", "canceled", "failed"}:
            job.state = "canceled"
            job.message = f"{reason_prefix} ({job.chunks_done}/{job.chunks_total})"
        self.touch()
        return job

    def request_shutdown(self) -> None:
        self.shutdown_requested = True
        if self.uvicorn_server is not None:
            self.uvicorn_server.should_exit = True


def create_app(
    *,
    root_dir: Path | None = None,
    config: AppConfig | None = None,
    pipeline_factory: Any = None,
    ephemeral: bool = False,
) -> FastAPI:
    """Create the FastAPI companion application."""
    resolved_root = root_dir or get_root_dir()
    resolved_config = config or load_config(resolved_root)
    runtime = ServerRuntime(resolved_root, resolved_config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        watchdog_task: asyncio.Task[None] | None = None

        async def _watchdog() -> None:
            while not runtime.shutdown_requested:
                await asyncio.sleep(15.0)
                active_jobs = runtime.active_jobs_count()
                alive_mpv = runtime.alive_mpv_count()
                if active_jobs > 0 or alive_mpv > 0:
                    runtime.touch()
                    continue
                idle_elapsed = time.monotonic() - runtime.last_activity_at
                if should_idle_exit(
                    active_jobs=active_jobs,
                    alive_mpv_count=alive_mpv,
                    idle_seconds=idle_elapsed,
                    idle_limit_seconds=runtime.config.server.idle_exit_seconds,
                ):
                    logger.info("Idle limit (%ss) reached; exiting ephemeral server.", runtime.config.server.idle_exit_seconds)
                    runtime.request_shutdown()
                    break

        if ephemeral:
            watchdog_task = asyncio.create_task(_watchdog())
        try:
            yield
        finally:
            if watchdog_task is not None:
                watchdog_task.cancel()
                try:
                    await watchdog_task
                except asyncio.CancelledError:
                    pass
            for task in list(runtime.tasks):
                if not task.done():
                    task.cancel()

    app = FastAPI(title="sjtu-mpv-sub", version=__version__, lifespan=lifespan)
    app.state.runtime = runtime

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://v.sjtu.edu.cn",
            "http://127.0.0.1",
            "http://localhost",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _touch_activity(request: Request, call_next):
        path = request.url.path
        skip_touch = False
        if request.method == "GET" and path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/").strip("/")
            job = runtime.jobs.get(job_id)
            if job is not None and job.state in {"completed", "canceled", "failed"}:
                skip_touch = True
        if not skip_touch:
            runtime.touch()
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        cfg = load_config(runtime.root_dir)
        runtime.config = cfg
        return {
            "ok": True,
            "version": __version__,
            "api_key_configured": bool(cfg.effective_dashscope_api_key()),
            "active_jobs": runtime.active_jobs_count(),
        }

    @app.post("/api/play")
    async def play(req: PlayRequest) -> dict[str, str]:
        cfg = load_config(runtime.root_dir)
        runtime.config = cfg
        session_key = make_session_key(req)

        existing_id = runtime.active_session_jobs.get(session_key)
        if existing_id and existing_id in runtime.jobs:
            existing = runtime.jobs[existing_id]
            if existing.state == "playing" and runtime.alive_mpv_count() == 0:
                runtime.cancel_job(existing_id, reason_prefix="mpv 已关闭，已停止旧后台任务")
            elif existing.state in {"queued", "probing", "running", "playing"}:
                return {"job_id": existing.job_id, "session_key": existing.session_key}

        job_id = uuid.uuid4().hex[:12]
        job = JobStatus(
            job_id=job_id,
            session_key=session_key,
            state="queued",
            message="任务已排队，准备处理...",
            chunks_done=0,
            chunks_total=0,
            mpv_launched=False,
            cache_hit=False,
        )
        runtime.jobs[job_id] = job
        runtime.active_session_jobs[session_key] = job_id

        pipeline = (
            pipeline_factory(cfg, runtime.root_dir, runtime.register_mpv_proc)
            if pipeline_factory is not None
            else LecturePipeline(
                config=cfg,
                root_dir=runtime.root_dir,
                on_mpv_started=runtime.register_mpv_proc,
            )
        )

        async def _runner() -> None:
            try:
                await pipeline.run_job(job, req)
            except asyncio.CancelledError:
                pass
            finally:
                runtime.job_tasks.pop(job_id, None)
                runtime.touch()

        task = asyncio.create_task(_runner())
        runtime.job_tasks[job_id] = task
        runtime.tasks.add(task)
        task.add_done_callback(runtime.tasks.discard)
        return {"job_id": job_id, "session_key": session_key}

    @app.get("/api/jobs/{job_id}", response_model=JobStatus)
    async def get_job(job_id: str) -> JobStatus:
        job = runtime.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.post("/api/jobs/{job_id}/cancel", response_model=JobStatus)
    async def cancel_job(job_id: str) -> JobStatus:
        job = runtime.cancel_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.get("/api/courses/{tecl_id}", response_model=CoursePreference)
    async def get_course(tecl_id: int) -> CoursePreference:
        return get_course_preference(tecl_id, root_dir=runtime.root_dir)

    @app.put("/api/courses/{tecl_id}", response_model=CoursePreference)
    async def put_course(tecl_id: int, update: CoursePreferenceUpdate) -> CoursePreference:
        return upsert_course_preference(tecl_id, update, root_dir=runtime.root_dir)

    @app.post("/api/shutdown")
    async def shutdown() -> dict[str, bool]:
        runtime.request_shutdown()
        return {"ok": True}

    return app


def is_server_already_running(host: str, port: int) -> bool:
    """Check whether another sjtu-mpv-sub companion instance is already serving /health."""
    try:
        with httpx.Client(trust_env=False, timeout=0.6) as client:
            resp = client.get(f"http://{host}:{port}/health")
            return resp.status_code == 200 and bool(resp.json().get("ok"))
    except Exception:
        return False


def _can_bind_port(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def run_server(*, ephemeral: bool = True, root_dir: Path | None = None) -> None:
    """Start the companion server, exiting silently if an instance already runs."""
    resolved_root = root_dir or get_root_dir()
    setup_logging(resolved_root)
    config = load_config(resolved_root)
    host = config.server.host
    port = config.server.port

    if ephemeral and (is_server_already_running(host, port) or not _can_bind_port(host, port)):
        if is_server_already_running(host, port):
            logger.info("Server already running on %s:%d; reusing existing instance.", host, port)
            return

    app = create_app(root_dir=resolved_root, config=config, ephemeral=ephemeral)
    uv_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_config=None,
        access_log=False,
    )
    uv_server = uvicorn.Server(uv_config)
    app.state.runtime.uvicorn_server = uv_server

    try:
        uv_server.run()
    except SystemExit as exc:
        if ephemeral and is_server_already_running(host, port):
            logger.info("Port %d already bound by healthy instance; exiting cleanly.", port)
            return
        raise exc
