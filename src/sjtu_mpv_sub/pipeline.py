"""Progressive ASR + translation pipeline with session cache and live mpv reload."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from sjtu_mpv_sub.asr import (
    ASRProvider,
    QwenAudioFiletransASR,
    build_asr_context_messages,
    build_instant_vocabulary,
)
from sjtu_mpv_sub.config import (
    AppConfig,
    get_course_preference,
    resolve_tool_paths,
    upsert_course_preference,
)
from sjtu_mpv_sub.media import (
    ResolvedSegment,
    extract_audio_chunk_async,
    resolve_segment_tracks_async,
)
from sjtu_mpv_sub.models import (
    BilingualCue,
    CoursePreferenceUpdate,
    JobStatus,
    PlayRequest,
    PptSlice,
    SegmentInput,
)
from sjtu_mpv_sub.mpv_controller import (
    EdlSegment,
    MpvIpcClient,
    build_edl,
    build_mpv_pipe_path,
    launch_mpv,
    write_edl,
)
from sjtu_mpv_sub.subtitle import render_ass, write_ass_atomic
from sjtu_mpv_sub.translator import LLMTranslator, collect_ppt_window

logger = logging.getLogger(__name__)


class ChunkTask(BaseModel):
    """A single audio chunk to transcribe, translate, and merge."""

    index: int
    segment_idx: int
    course_id: int
    audio_url: str
    start_sec: float
    duration_sec: float
    period_offset_ms: int
    ppt_slices: list[PptSlice]


def sort_segments(segments: list[SegmentInput]) -> list[SegmentInput]:
    """Order class periods chronologically within a lecture."""
    return sorted(
        segments,
        key=lambda s: (s.begin_time or "", s.leti_number if s.leti_number is not None else 0, s.course_id),
    )


def make_session_key(req: PlayRequest) -> str:
    """Build deterministic cache key for a teaching class + ordered period set."""
    ordered = sort_segments(req.segments)
    ids = "_".join(str(seg.course_id) for seg in ordered)
    raw = f"tecl_{req.tecl_id}_{ids}"
    return "".join(c if c.isalnum() or c == "_" else "_" for c in raw)


def build_chunk_plan(
    segments: list[ResolvedSegment],
    first_chunk_seconds: int = 180,
    chunk_seconds: int = 420,
) -> list[ChunkTask]:
    """Partition multi-period segments into priority chunk 0 + parallel subsequent chunks."""
    tasks: list[ChunkTask] = []
    period_offset_ms = 0
    chunk_idx = 0

    for seg_idx, seg in enumerate(segments):
        dur = max(0.1, float(seg.duration_sec))
        cursor = 0.0
        while cursor < dur - 0.5 or (cursor == 0.0 and dur > 0):
            win = float(first_chunk_seconds if chunk_idx == 0 else chunk_seconds)
            span = min(win, max(0.1, dur - cursor))
            tasks.append(
                ChunkTask(
                    index=chunk_idx,
                    segment_idx=seg_idx,
                    course_id=seg.course_id,
                    audio_url=seg.audio_url,
                    start_sec=cursor,
                    duration_sec=span,
                    period_offset_ms=period_offset_ms,
                    ppt_slices=seg.ppt_slices,
                )
            )
            chunk_idx += 1
            cursor += span
            if dur - cursor < 0.5:
                break

        # Match EDL formatting precision (%.3f sec = 1 ms) so offsets stay locked to chapters.
        period_offset_ms += int(round(dur * 1000.0))

    return tasks


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class LecturePipeline:
    """Coordinates probing, caching, progressive ASR/LLM translation, and mpv IPC reload."""

    def __init__(
        self,
        *,
        config: AppConfig,
        root_dir: Path,
        asr_factory: Callable[[AppConfig], ASRProvider] | None = None,
        translator_factory: Callable[[AppConfig], LLMTranslator] | None = None,
        mpv_launcher: Callable[..., subprocess.Popen[bytes]] = launch_mpv,
        ipc_factory: Callable[[str], MpvIpcClient] = MpvIpcClient,
        on_mpv_started: Callable[[subprocess.Popen[bytes]], None] | None = None,
        mpv_exit_poll_interval: float = 0.5,
    ) -> None:
        self.config = config
        self.root_dir = root_dir
        self._asr_factory = asr_factory or self._default_asr_factory
        self._translator_factory = translator_factory or self._default_translator_factory
        self._mpv_launcher = mpv_launcher
        self._ipc_factory = ipc_factory
        self._on_mpv_started = on_mpv_started
        self._mpv_exit_poll_interval = max(0.02, float(mpv_exit_poll_interval))

    @staticmethod
    def _default_asr_factory(cfg: AppConfig) -> ASRProvider:
        return QwenAudioFiletransASR(
            api_key=cfg.effective_dashscope_api_key(),
            model=cfg.dashscope.asr_model,
            uploads_base_url=cfg.dashscope.uploads_base_url,
            asr_api_base_url=cfg.effective_asr_api_base_url(),
            poll_interval_seconds=cfg.dashscope.asr_poll_interval_seconds,
            timeout_seconds=cfg.dashscope.asr_timeout_seconds,
        )

    @staticmethod
    def _default_translator_factory(cfg: AppConfig) -> LLMTranslator:
        return LLMTranslator(
            api_key=cfg.effective_llm_api_key(),
            base_url=cfg.effective_llm_base_url(),
            model=cfg.dashscope.llm_model,
            proxy=cfg.dashscope.llm_proxy,
            enable_thinking=cfg.dashscope.llm_enable_thinking,
        )

    async def run_job(self, job: JobStatus, req: PlayRequest) -> None:
        """Execute a full playback job, updating JobStatus in place."""
        tools = resolve_tool_paths(self.config)
        mpv_path = tools.get("mpv")
        ffmpeg_path = tools.get("ffmpeg")
        ffprobe_path = tools.get("ffprobe")

        if not mpv_path:
            job.state = "failed"
            job.error = "mpv.exe not found. Please check tools.mpv_path in config.yaml."
            job.message = job.error
            return
        if not ffmpeg_path or not ffprobe_path:
            job.state = "failed"
            job.error = "ffmpeg or ffprobe not found on PATH or config.yaml."
            job.message = job.error
            return

        ordered_inputs = sort_segments(req.segments)
        session_key = job.session_key
        cache_dir = self.root_dir / "data" / "cache" / session_key
        temp_dir = self.root_dir / "data" / "tmp" / job.job_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        video_edl_path = cache_dir / "session.edl"
        audio_edl_path = cache_dir / "audio.edl"
        sub_path = cache_dir / "bilingual.ass"
        meta_path = cache_dir / "meta.json"
        cues_path = cache_dir / "cues.json"

        course_pref = get_course_preference(req.tecl_id, root_dir=self.root_dir)
        effective_view = req.preferred_view or course_pref.preferred_view
        course_pref = upsert_course_preference(
            req.tecl_id,
            CoursePreferenceUpdate(
                name=req.course_name or course_pref.name,
                preferred_view=effective_view,
            ),
            root_dir=self.root_dir,
        )

        asr_provider: ASRProvider | None = None
        translator: LLMTranslator | None = None
        try:
            job.state = "probing"
            job.message = f"正在探测 {len(ordered_inputs)} 节课程视频流..."

            resolved_segments = await asyncio.gather(
                *[
                    resolve_segment_tracks_async(
                        seg,
                        preferred_view=effective_view,
                        referer=self.config.pipeline.referer,
                        ffprobe_path=ffprobe_path,
                    )
                    for seg in ordered_inputs
                ]
            )

            video_edl = build_edl(
                [EdlSegment(url=s.video_url, duration_sec=s.duration_sec, title=s.title) for s in resolved_segments],
                include_chapters=True,
            )
            write_edl(video_edl_path, video_edl)

            use_separate_audio = any(s.separate_audio for s in resolved_segments)
            if use_separate_audio:
                audio_edl = build_edl(
                    [EdlSegment(url=s.audio_url, duration_sec=s.duration_sec) for s in resolved_segments],
                    include_chapters=False,
                )
                write_edl(audio_edl_path, audio_edl)
            elif audio_edl_path.exists():
                audio_edl_path.unlink(missing_ok=True)

            media_title = f"{req.course_name} ({ordered_inputs[0].title})" if ordered_inputs else req.course_name
            pipe_path = build_mpv_pipe_path(job.job_id)

            # Cache hit check: refresh EDLs (fresh signed keys) and reuse cached bilingual subtitles.
            if not req.force_retranscribe and meta_path.is_file() and sub_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    meta = {}
                if meta.get("complete") is True:
                    if cues_path.is_file():
                        raw_cues = json.loads(cues_path.read_text(encoding="utf-8"))
                        cached_cues = [BilingualCue.model_validate(item) for item in raw_cues]
                        write_ass_atomic(sub_path, render_ass(cached_cues, self.config.subtitle))

                    proc = self._mpv_launcher(
                        mpv_path=mpv_path,
                        video_edl_path=video_edl_path,
                        sub_path=sub_path,
                        pipe_path=pipe_path,
                        media_title=media_title,
                        referer=self.config.pipeline.referer,
                        audio_edl_path=audio_edl_path if use_separate_audio else None,
                    )
                    if self._on_mpv_started:
                        self._on_mpv_started(proc)

                    job.cache_hit = True
                    job.mpv_launched = True
                    total = max(1, int(meta.get("chunks_total", 1)))
                    job.chunks_total = total
                    job.chunks_done = total
                    job.state = "completed"
                    job.message = "命中本地字幕缓存，已立即唤醒 mpv 播放"
                    return

            # Cache miss: validate API keys before spawning transcription tasks.
            if not self.config.effective_dashscope_api_key():
                raise RuntimeError(
                    "DashScope API Key 未配置。请在 config.yaml 的 dashscope.api_key 中填写或设置环境变量 DASHSCOPE_API_KEY。"
                )
            if not self.config.effective_llm_api_key():
                raise RuntimeError("LLM API Key 未配置，请检查 config.yaml 中的 dashscope.api_key / llm_api_key。")

            chunks = build_chunk_plan(
                resolved_segments,
                first_chunk_seconds=self.config.pipeline.first_chunk_seconds,
                chunk_seconds=self.config.pipeline.chunk_seconds,
            )
            if not chunks:
                raise RuntimeError("未能生成有效的音频切片计划。")

            job.chunks_total = len(chunks)
            job.chunks_done = 0
            job.state = "running"
            job.message = f"正在优先转录与翻译首段字幕 (1/{len(chunks)})..."

            asr_provider = self._asr_factory(self.config)
            translator = self._translator_factory(self.config)
            cues_by_chunk: dict[int, list[BilingualCue]] = {}
            merge_lock = asyncio.Lock()

            async def process_single_chunk(task: ChunkTask) -> list[BilingualCue]:
                audio_path = temp_dir / f"chunk_{task.index:03d}.mp3"
                try:
                    await extract_audio_chunk_async(
                        task.audio_url,
                        task.start_sec,
                        task.duration_sec,
                        audio_path,
                        referer=self.config.pipeline.referer,
                        ffmpeg_path=ffmpeg_path,
                    )
                    ppt_keywords, ppt_ctx = collect_ppt_window(
                        task.ppt_slices,
                        task.start_sec,
                        task.start_sec + task.duration_sec,
                    )
                    hotwords = build_instant_vocabulary(course_pref.glossary, ppt_keywords)
                    asr_context = build_asr_context_messages(req.course_name, ppt_ctx)
                    sentences = await asr_provider.transcribe_chunk(
                        audio_path,
                        self.config.dashscope.asr_language_hints,
                        vocabulary_id=course_pref.vocabulary_id,
                        hotwords=hotwords or None,
                        context_messages=asr_context or None,
                    )
                    translated = await translator.refine_and_translate(
                        sentences,
                        course_name=req.course_name,
                        glossary=course_pref.glossary,
                        extra_prompt=course_pref.extra_prompt,
                        ppt_context=ppt_ctx,
                        batch_size=self.config.pipeline.translation_batch_size,
                    )
                    base_offset_ms = task.period_offset_ms + int(round(task.start_sec * 1000.0))
                    return [
                        BilingualCue(
                            start_ms=base_offset_ms + cue.start_ms,
                            end_ms=base_offset_ms + cue.end_ms,
                            en=cue.en,
                            zh=cue.zh,
                        )
                        for cue in translated
                    ]
                finally:
                    audio_path.unlink(missing_ok=True)

            # 1. Priority Chunk 0 -> launch mpv ASAP!
            chunk0_cues = await process_single_chunk(chunks[0])
            cues_by_chunk[0] = chunk0_cues
            initial_cues = sorted(chunk0_cues, key=lambda c: (c.start_ms, c.end_ms))
            write_ass_atomic(sub_path, render_ass(initial_cues, self.config.subtitle))

            mpv_proc = self._mpv_launcher(
                mpv_path=mpv_path,
                video_edl_path=video_edl_path,
                sub_path=sub_path,
                pipe_path=pipe_path,
                media_title=media_title,
                referer=self.config.pipeline.referer,
                audio_edl_path=audio_edl_path if use_separate_audio else None,
            )
            if self._on_mpv_started:
                self._on_mpv_started(mpv_proc)

            ipc = self._ipc_factory(pipe_path)
            await asyncio.to_thread(ipc.wait_ready, 10.0)

            job.mpv_launched = True
            job.chunks_done = 1
            job.state = "playing"
            job.message = (
                f"mpv 已开播，后台正在补齐剩余字幕 (1/{len(chunks)})..."
                if len(chunks) > 1
                else "整堂课双语字幕已全部就绪"
            )

            # 2. Remaining chunks in parallel with live hot-reload (stopping if mpv exits).
            stop_on_exit = bool(self.config.pipeline.stop_on_mpv_exit)
            stopped_by_mpv_exit = False
            if len(chunks) > 1:
                if stop_on_exit and mpv_proc.poll() is not None:
                    stopped_by_mpv_exit = True
                else:
                    sem = asyncio.Semaphore(max(1, self.config.pipeline.max_concurrency))

                    async def run_bg_chunk(task: ChunkTask) -> None:
                        async with sem:
                            if stop_on_exit and mpv_proc.poll() is not None:
                                return
                            chunk_cues = await process_single_chunk(task)
                            if stop_on_exit and mpv_proc.poll() is not None:
                                return
                            async with merge_lock:
                                cues_by_chunk[task.index] = chunk_cues
                                job.chunks_done += 1
                                job.message = f"mpv 播放中，后台字幕进度 {job.chunks_done}/{job.chunks_total}..."
                                if chunk_cues:
                                    merged = [
                                        cue
                                        for idx in sorted(cues_by_chunk)
                                        for cue in cues_by_chunk[idx]
                                    ]
                                    merged.sort(key=lambda c: (c.start_ms, c.end_ms))
                                    write_ass_atomic(sub_path, render_ass(merged, self.config.subtitle))
                                    if mpv_proc.poll() is None:
                                        await asyncio.to_thread(ipc.reload_subtitles)

                    bg_tasks = [asyncio.create_task(run_bg_chunk(c)) for c in chunks[1:]]

                    async def _run_bg_all() -> None:
                        await asyncio.gather(*bg_tasks)

                    async def _watch_mpv_exit() -> None:
                        while mpv_proc.poll() is None:
                            await asyncio.sleep(self._mpv_exit_poll_interval)

                    bg_group_task = asyncio.create_task(_run_bg_all())
                    watcher_task: asyncio.Task[None] | None = None
                    try:
                        if stop_on_exit:
                            watcher_task = asyncio.create_task(_watch_mpv_exit())
                            done, _pending = await asyncio.wait(
                                {bg_group_task, watcher_task},
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if watcher_task in done and not bg_group_task.done():
                                stopped_by_mpv_exit = True
                                for t in bg_tasks:
                                    if not t.done():
                                        t.cancel()
                                bg_group_task.cancel()
                                await asyncio.gather(*bg_tasks, bg_group_task, return_exceptions=True)
                            else:
                                watcher_task.cancel()
                                await asyncio.gather(watcher_task, return_exceptions=True)
                                await bg_group_task
                        else:
                            await bg_group_task
                    except asyncio.CancelledError:
                        if watcher_task is not None and not watcher_task.done():
                            watcher_task.cancel()
                        for t in bg_tasks:
                            if not t.done():
                                t.cancel()
                        if not bg_group_task.done():
                            bg_group_task.cancel()
                        await asyncio.gather(
                            *(t for t in (watcher_task, bg_group_task, *bg_tasks) if t is not None),
                            return_exceptions=True,
                        )
                        raise

            if stopped_by_mpv_exit:
                logger.info(
                    "mpv exited for job %s; stopped background transcription at %d/%d chunks",
                    job.job_id,
                    job.chunks_done,
                    job.chunks_total,
                )
                job.state = "canceled"
                job.message = f"mpv 已关闭，已停止后台转录 ({job.chunks_done}/{job.chunks_total})"
                return

            all_cues = [cue for idx in sorted(cues_by_chunk) for cue in cues_by_chunk[idx]]
            all_cues.sort(key=lambda c: (c.start_ms, c.end_ms))
            write_ass_atomic(sub_path, render_ass(all_cues, self.config.subtitle))
            _write_json(cues_path, [c.model_dump(mode="json") for c in all_cues])
            has_cues = bool(all_cues)
            if not has_cues:
                logger.warning(
                    "Playback job %s completed with zero speech cues across %d chunks",
                    job.job_id,
                    len(chunks),
                )
            _write_json(
                meta_path,
                {
                    "complete": has_cues,
                    "tecl_id": req.tecl_id,
                    "course_name": req.course_name,
                    "preferred_view": effective_view,
                    "chunks_total": len(chunks),
                    "segments": [
                        {"course_id": s.course_id, "title": s.title, "duration_sec": s.duration_sec}
                        for s in resolved_segments
                    ],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            completion_msg = (
                "整堂课双语字幕已全部就绪" if has_cues else "处理完成，但整段音频未检测到有效语音"
            )
            if mpv_proc.poll() is None:
                await asyncio.to_thread(ipc.show_osd, completion_msg, 3500)

            job.state = "completed"
            job.message = completion_msg
        except asyncio.CancelledError:
            logger.info("Playback job %s canceled", job.job_id)
            if job.state not in {"completed", "canceled", "failed"}:
                job.state = "canceled"
                job.message = f"已停止后台转录 ({job.chunks_done}/{job.chunks_total})"
            raise
        except Exception as exc:
            logger.exception("Playback job %s failed", job.job_id)
            job.state = "failed"
            job.error = str(exc)
            job.message = f"处理失败: {exc}"
        finally:
            if asr_provider is not None:
                await asr_provider.close()
            if translator is not None:
                await translator.close()
            shutil.rmtree(temp_dir, ignore_errors=True)
