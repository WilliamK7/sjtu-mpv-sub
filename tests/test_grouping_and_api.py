"""Tests for lecture chunk planning, FastAPI endpoints, caching, and idle exit."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from sjtu_mpv_sub import media, pipeline as pipeline_mod
from sjtu_mpv_sub.config import AppConfig, load_config
from sjtu_mpv_sub.media import ResolvedSegment
from sjtu_mpv_sub.models import AsrSentence, BilingualCue, PlayRequest, SegmentInput, StreamSource
from sjtu_mpv_sub.pipeline import LecturePipeline, build_chunk_plan, make_session_key
from sjtu_mpv_sub.server import create_app, should_idle_exit


def test_chunk_plan_accumulates_period_offsets() -> None:
    resolved = [
        ResolvedSegment(
            course_id=1,
            title="P1",
            video_url="https://example.com/1.mp4",
            audio_url="https://example.com/1.mp4",
            separate_audio=False,
            duration_sec=500.0,
            ppt_slices=[],
        ),
        ResolvedSegment(
            course_id=2,
            title="P2",
            video_url="https://example.com/2.mp4",
            audio_url="https://example.com/2.mp4",
            separate_audio=False,
            duration_sec=300.0,
            ppt_slices=[],
        ),
    ]
    chunks = build_chunk_plan(resolved, first_chunk_seconds=180, chunk_seconds=420)
    assert len(chunks) == 3
    assert chunks[0].period_offset_ms == 0 and chunks[0].start_sec == 0.0 and chunks[0].duration_sec == 180.0
    assert chunks[1].period_offset_ms == 0 and chunks[1].start_sec == 180.0 and chunks[1].duration_sec == 320.0
    assert chunks[2].period_offset_ms == 500000 and chunks[2].start_sec == 0.0 and chunks[2].duration_sec == 300.0


def test_make_session_key_orders_chronologically() -> None:
    req = PlayRequest(
        tecl_id=99,
        course_name="Physics",
        segments=[
            SegmentInput(course_id=202, title="第2节", begin_time="2026-11-23 09:00:00", streams=[StreamSource(view_num=1, url="u2")]),
            SegmentInput(course_id=201, title="第1节", begin_time="2026-11-23 08:00:00", streams=[StreamSource(view_num=1, url="u1")]),
        ],
    )
    assert make_session_key(req) == "tecl_99_201_202"


def test_should_idle_exit_logic() -> None:
    assert should_idle_exit(active_jobs=0, alive_mpv_count=0, idle_seconds=305, idle_limit_seconds=300) is True
    assert should_idle_exit(active_jobs=1, alive_mpv_count=0, idle_seconds=400, idle_limit_seconds=300) is False
    assert should_idle_exit(active_jobs=0, alive_mpv_count=1, idle_seconds=400, idle_limit_seconds=300) is False
    assert should_idle_exit(active_jobs=0, alive_mpv_count=0, idle_seconds=120, idle_limit_seconds=300) is False


def test_api_courses_and_progressive_play_with_cache_hit(tmp_path, monkeypatch) -> None:
    cfg = load_config(tmp_path)
    cfg.dashscope.api_key = "sk-test-key"
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    async def fake_resolve(segment, preferred_view="ppt", referer="", ffprobe_path="ffprobe"):
        return ResolvedSegment(
            course_id=segment.course_id,
            title=segment.title,
            video_url=segment.streams[0].url,
            audio_url=segment.streams[0].url,
            separate_audio=False,
            duration_sec=250.0,
            ppt_slices=segment.ppt_slices,
        )

    async def fake_extract(audio_url, start_sec, duration_sec, output_audio, referer="", ffmpeg_path="ffmpeg", timeout=120.0):
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        output_audio.write_bytes(b"ID3fake")
        return output_audio

    monkeypatch.setattr(pipeline_mod, "resolve_segment_tracks_async", fake_resolve)
    monkeypatch.setattr(pipeline_mod, "extract_audio_chunk_async", fake_extract)
    monkeypatch.setattr(
        pipeline_mod,
        "resolve_tool_paths",
        lambda _cfg: {"mpv": "mock-mpv.exe", "ffmpeg": "mock-ffmpeg", "ffprobe": "mock-ffprobe"},
    )

    asr_calls = 0
    launch_calls: list[Path] = []
    reload_calls = 0

    class FakeASR:
        async def transcribe_chunk(
            self,
            audio_path,
            language_hints,
            *,
            vocabulary_id=None,
            hotwords=None,
            context_messages=None,
        ):
            nonlocal asr_calls
            asr_calls += 1
            return [AsrSentence(start_ms=500, end_ms=2000, text=f"Chunk line {asr_calls}")]

        async def close(self):
            return None

    class FakeTranslator:
        async def refine_and_translate(self, sentences, **_kwargs):
            return [BilingualCue(start_ms=s.start_ms, end_ms=s.end_ms, en=s.text, zh=f"翻译:{s.text}") for s in sentences]

        async def close(self):
            return None

    class FakeProc:
        def poll(self):
            return None

    class FakeIpc:
        def __init__(self, _pipe):
            pass

        def wait_ready(self, timeout=10.0):
            return True

        def reload_subtitles(self):
            nonlocal reload_calls
            reload_calls += 1
            return True

        def show_osd(self, text, duration_ms=3000):
            return True

    def fake_launcher(**kwargs):
        launch_calls.append(kwargs["video_edl_path"])
        return FakeProc()

    def build_test_pipeline(config: AppConfig, root: Path, on_mpv_started):
        return LecturePipeline(
            config=config,
            root_dir=root,
            asr_factory=lambda _c: FakeASR(),
            translator_factory=lambda _c: FakeTranslator(),
            mpv_launcher=fake_launcher,
            ipc_factory=FakeIpc,
            on_mpv_started=on_mpv_started,
        )

    app = create_app(root_dir=tmp_path, config=cfg, pipeline_factory=build_test_pipeline, ephemeral=False)
    with TestClient(app) as client:
        assert client.get("/health").json()["api_key_configured"] is True

        put_resp = client.put(
            "/api/courses/888",
            json={"name": "AI Systems", "preferred_view": "camera", "glossary": ["Transformer"]},
        )
        assert put_resp.status_code == 200
        assert put_resp.json()["preferred_view"] == "camera"
        assert client.get("/api/courses/888").json()["glossary"] == ["Transformer"]

        payload = {
            "tecl_id": 888,
            "course_name": "AI Systems",
            "segments": [
                {
                    "course_id": 5001,
                    "title": "第1节",
                    "begin_time": "2026-11-23 08:00:00",
                    "streams": [{"view_num": 5, "url": "https://live.sjtu.edu.cn/5001.mp4?key=first"}],
                }
            ],
        }
        job1 = client.post("/api/play", json=payload).json()
        for _ in range(40):
            status = client.get(f"/api/jobs/{job1['job_id']}").json()
            if status["state"] in ("completed", "failed"):
                break
            time.sleep(0.05)

        assert status["state"] == "completed"
        assert status["cache_hit"] is False
        assert status["chunks_total"] == 2
        assert status["chunks_done"] == 2
        assert asr_calls == 2
        assert reload_calls == 1
        assert len(launch_calls) == 1

        # Second call with new signed URL hits subtitle cache and skips ASR while refreshing EDL.
        payload["segments"][0]["streams"][0]["url"] = "https://live.sjtu.edu.cn/5001.mp4?key=refreshed"
        job2 = client.post("/api/play", json=payload).json()
        for _ in range(40):
            status2 = client.get(f"/api/jobs/{job2['job_id']}").json()
            if status2["state"] in ("completed", "failed"):
                break
            time.sleep(0.05)

        assert status2["state"] == "completed"
        assert status2["cache_hit"] is True
        assert asr_calls == 2  # No extra ASR calls!
        edl_text = (tmp_path / "data" / "cache" / "tecl_888_5001" / "session.edl").read_text(encoding="utf-8")
        assert "key=refreshed" in edl_text
        meta = json.loads((tmp_path / "data" / "cache" / "tecl_888_5001" / "meta.json").read_text(encoding="utf-8"))
        assert meta["complete"] is True


def test_pipeline_handles_silent_chunk_zero_and_reloads_on_chunk_one(tmp_path, monkeypatch) -> None:
    cfg = load_config(tmp_path)
    cfg.dashscope.api_key = "sk-test-key"
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    async def fake_resolve(segment, preferred_view="ppt", referer="", ffprobe_path="ffprobe"):
        return ResolvedSegment(
            course_id=segment.course_id,
            title=segment.title,
            video_url=segment.streams[0].url,
            audio_url=segment.streams[0].url,
            separate_audio=False,
            duration_sec=250.0,
            ppt_slices=segment.ppt_slices,
        )

    async def fake_extract(audio_url, start_sec, duration_sec, output_audio, referer="", ffmpeg_path="ffmpeg", timeout=120.0):
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        output_audio.write_bytes(b"ID3fake")
        return output_audio

    monkeypatch.setattr(pipeline_mod, "resolve_segment_tracks_async", fake_resolve)
    monkeypatch.setattr(pipeline_mod, "extract_audio_chunk_async", fake_extract)
    monkeypatch.setattr(
        pipeline_mod,
        "resolve_tool_paths",
        lambda _cfg: {"mpv": "mock-mpv.exe", "ffmpeg": "mock-ffmpeg", "ffprobe": "mock-ffprobe"},
    )

    launch_calls: list[Path] = []
    reload_calls = 0

    class SilentChunkZeroASR:
        async def transcribe_chunk(
            self,
            audio_path,
            language_hints,
            *,
            vocabulary_id=None,
            hotwords=None,
            context_messages=None,
        ):
            if audio_path.name == "chunk_000.mp3":
                return []
            return [AsrSentence(start_ms=1000, end_ms=3000, text="Lecture begins now.")]

        async def close(self):
            return None

    class FakeTranslator:
        async def refine_and_translate(self, sentences, **_kwargs):
            return [BilingualCue(start_ms=s.start_ms, end_ms=s.end_ms, en=s.text, zh="课程开始了。") for s in sentences]

        async def close(self):
            return None

    class FakeProc:
        def poll(self):
            return None

    class FakeIpc:
        def __init__(self, _pipe):
            pass

        def wait_ready(self, timeout=10.0):
            return True

        def reload_subtitles(self):
            nonlocal reload_calls
            reload_calls += 1
            return True

        def show_osd(self, text, duration_ms=3000):
            return True

    def fake_launcher(**kwargs):
        launch_calls.append(kwargs["video_edl_path"])
        return FakeProc()

    def build_test_pipeline(config: AppConfig, root: Path, on_mpv_started):
        return LecturePipeline(
            config=config,
            root_dir=root,
            asr_factory=lambda _c: SilentChunkZeroASR(),
            translator_factory=lambda _c: FakeTranslator(),
            mpv_launcher=fake_launcher,
            ipc_factory=FakeIpc,
            on_mpv_started=on_mpv_started,
        )

    app = create_app(root_dir=tmp_path, config=cfg, pipeline_factory=build_test_pipeline, ephemeral=False)
    with TestClient(app) as client:
        payload = {
            "tecl_id": 889,
            "course_name": "Physics II",
            "segments": [
                {
                    "course_id": 5002,
                    "title": "第1节",
                    "begin_time": "2026-11-23 08:00:00",
                    "streams": [{"view_num": 1, "url": "https://live.sjtu.edu.cn/5002.mp4?key=first"}],
                }
            ],
        }
        job = client.post("/api/play", json=payload).json()
        for _ in range(40):
            status = client.get(f"/api/jobs/{job['job_id']}").json()
            if status["state"] in ("completed", "failed"):
                break
            time.sleep(0.05)

        assert status["state"] == "completed"
        assert status["chunks_total"] == 2
        assert status["chunks_done"] == 2
        assert len(launch_calls) == 1
        assert reload_calls == 1

        sub_text = (tmp_path / "data" / "cache" / "tecl_889_5002" / "bilingual.ass").read_text(encoding="utf-8-sig")
        assert "Lecture begins now." in sub_text
        assert "课程开始了。" in sub_text
        assert "双语字幕加载中" not in sub_text

        meta = json.loads((tmp_path / "data" / "cache" / "tecl_889_5002" / "meta.json").read_text(encoding="utf-8"))
        assert meta["complete"] is True


def _patch_fake_pipeline_media(monkeypatch, *, duration_sec: float) -> None:
    async def fake_resolve(segment, preferred_view="ppt", referer="", ffprobe_path="ffprobe"):
        return ResolvedSegment(
            course_id=segment.course_id,
            title=segment.title,
            video_url=segment.streams[0].url,
            audio_url=segment.streams[0].url,
            separate_audio=False,
            duration_sec=duration_sec,
            ppt_slices=segment.ppt_slices,
        )

    async def fake_extract(audio_url, start_sec, duration_sec, output_audio, referer="", ffmpeg_path="ffmpeg", timeout=120.0):
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        output_audio.write_bytes(b"ID3fake")
        return output_audio

    monkeypatch.setattr(pipeline_mod, "resolve_segment_tracks_async", fake_resolve)
    monkeypatch.setattr(pipeline_mod, "extract_audio_chunk_async", fake_extract)
    monkeypatch.setattr(
        pipeline_mod,
        "resolve_tool_paths",
        lambda _cfg: {"mpv": "mock-mpv.exe", "ffmpeg": "mock-ffmpeg", "ffprobe": "mock-ffprobe"},
    )


def _write_test_config(tmp_path: Path, *, max_concurrency: int = 1) -> AppConfig:
    cfg = load_config(tmp_path)
    cfg.dashscope.api_key = "sk-test-key"
    cfg.pipeline.max_concurrency = max_concurrency
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return cfg


def test_pipeline_stops_background_chunks_when_mpv_exits(tmp_path, monkeypatch) -> None:
    cfg = _write_test_config(tmp_path, max_concurrency=1)
    _patch_fake_pipeline_media(monkeypatch, duration_sec=700.0)

    seen_chunks: list[str] = []
    procs: list[object] = []

    class SlowChunkASR:
        async def transcribe_chunk(self, audio_path, language_hints, **_kwargs):
            seen_chunks.append(audio_path.name)
            if audio_path.name == "chunk_000.mp3":
                return [AsrSentence(start_ms=0, end_ms=1000, text="Intro line")]
            if procs:
                procs[-1].exited = True
            await asyncio.sleep(5.0)
            return [AsrSentence(start_ms=0, end_ms=1000, text="Should be canceled")]

        async def close(self):
            return None

    class FakeTranslator:
        async def refine_and_translate(self, sentences, **_kwargs):
            return [BilingualCue(start_ms=s.start_ms, end_ms=s.end_ms, en=s.text, zh="译文") for s in sentences]

        async def close(self):
            return None

    class FakeProc:
        def __init__(self) -> None:
            self.exited = False

        def poll(self):
            return 0 if self.exited else None

    class FakeIpc:
        def __init__(self, _pipe):
            pass

        def wait_ready(self, timeout=10.0):
            return True

        def reload_subtitles(self):
            return True

        def show_osd(self, text, duration_ms=3000):
            return True

    def fake_launcher(**_kwargs):
        proc = FakeProc()
        procs.append(proc)
        return proc

    def build_test_pipeline(config: AppConfig, root: Path, on_mpv_started):
        return LecturePipeline(
            config=config,
            root_dir=root,
            asr_factory=lambda _c: SlowChunkASR(),
            translator_factory=lambda _c: FakeTranslator(),
            mpv_launcher=fake_launcher,
            ipc_factory=FakeIpc,
            on_mpv_started=on_mpv_started,
            mpv_exit_poll_interval=0.02,
        )

    app = create_app(root_dir=tmp_path, config=cfg, pipeline_factory=build_test_pipeline, ephemeral=False)
    with TestClient(app) as client:
        payload = {
            "tecl_id": 890,
            "course_name": "Electromagnetism",
            "segments": [
                {
                    "course_id": 5003,
                    "title": "第1节",
                    "begin_time": "2026-11-23 08:00:00",
                    "streams": [{"view_num": 1, "url": "https://live.sjtu.edu.cn/5003.mp4?key=first"}],
                }
            ],
        }
        job1 = client.post("/api/play", json=payload).json()
        for _ in range(60):
            status = client.get(f"/api/jobs/{job1['job_id']}").json()
            if status["state"] in ("completed", "canceled", "failed"):
                break
            time.sleep(0.05)

        assert status["state"] == "canceled"
        assert "chunk_002.mp3" not in seen_chunks
        meta_path = tmp_path / "data" / "cache" / "tecl_890_5003" / "meta.json"
        assert not meta_path.exists()

        # Re-launching the same session after mpv closed starts a brand-new job cleanly.
        job2 = client.post("/api/play", json=payload).json()
        assert job2["job_id"] != job1["job_id"]
        for _ in range(60):
            status2 = client.get(f"/api/jobs/{job2['job_id']}").json()
            if status2["state"] in ("completed", "canceled", "failed"):
                break
            time.sleep(0.05)
        assert status2["state"] == "canceled"


def test_api_cancel_job_endpoint(tmp_path, monkeypatch) -> None:
    cfg = _write_test_config(tmp_path, max_concurrency=1)
    _patch_fake_pipeline_media(monkeypatch, duration_sec=250.0)

    asr_started = threading.Event()

    class BlockingASR:
        async def transcribe_chunk(self, audio_path, language_hints, **_kwargs):
            asr_started.set()
            await asyncio.sleep(10.0)
            return []

        async def close(self):
            return None

    class FakeTranslator:
        async def refine_and_translate(self, sentences, **_kwargs):
            return []

        async def close(self):
            return None

    def build_test_pipeline(config: AppConfig, root: Path, on_mpv_started):
        return LecturePipeline(
            config=config,
            root_dir=root,
            asr_factory=lambda _c: BlockingASR(),
            translator_factory=lambda _c: FakeTranslator(),
            on_mpv_started=on_mpv_started,
            mpv_exit_poll_interval=0.02,
        )

    app = create_app(root_dir=tmp_path, config=cfg, pipeline_factory=build_test_pipeline, ephemeral=False)
    with TestClient(app) as client:
        payload = {
            "tecl_id": 891,
            "course_name": "Cancel Test",
            "segments": [
                {
                    "course_id": 5004,
                    "title": "第1节",
                    "begin_time": "2026-11-23 08:00:00",
                    "streams": [{"view_num": 1, "url": "https://live.sjtu.edu.cn/5004.mp4?key=first"}],
                }
            ],
        }
        job = client.post("/api/play", json=payload).json()
        assert asr_started.wait(timeout=3.0)

        cancel_resp = client.post(f"/api/jobs/{job['job_id']}/cancel")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["state"] == "canceled"

        for _ in range(30):
            health = client.get("/health").json()
            if health["active_jobs"] == 0:
                break
            time.sleep(0.05)
        assert client.get("/health").json()["active_jobs"] == 0
        assert client.get(f"/api/jobs/{job['job_id']}").json()["state"] == "canceled"
