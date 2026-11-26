"""Progressive ASR + translation pipeline with range caching, smart start, and live mpv HUD."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import shutil
import subprocess
import time
from collections.abc import Awaitable, Callable
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
    detect_lecture_start_async,
    extract_audio_chunk_async,
    resolve_segment_tracks_async,
)
from sjtu_mpv_sub.models import (
    AsrSentence,
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
    get_hud_script_path,
    get_thumb_script_path,
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
    global_start_ms: int = 0
    global_end_ms: int = 0
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


def compute_segment_offsets_ms(segments: list[ResolvedSegment]) -> tuple[list[int], int]:
    """Compute cumulative start offsets (ms) matching EDL %.3f formatting precision."""
    offsets: list[int] = []
    cursor_ms = 0
    for seg in segments:
        offsets.append(cursor_ms)
        dur = max(0.1, float(seg.duration_sec))
        cursor_ms += int(round(dur * 1000.0))
    return offsets, cursor_ms


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
            g_start = period_offset_ms + int(round(cursor * 1000.0))
            g_end = period_offset_ms + int(round((cursor + span) * 1000.0))
            tasks.append(
                ChunkTask(
                    index=chunk_idx,
                    segment_idx=seg_idx,
                    course_id=seg.course_id,
                    audio_url=seg.audio_url,
                    start_sec=cursor,
                    duration_sec=span,
                    period_offset_ms=period_offset_ms,
                    global_start_ms=g_start,
                    global_end_ms=g_end,
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


def merge_ranges(
    ranges: list[tuple[int, int]],
    gap_tolerance_ms: int = 500,
) -> list[tuple[int, int]]:
    """Sort and coalesce overlapping or adjacent millisecond intervals."""
    valid = sorted((int(a), int(b)) for a, b in ranges if int(b) > int(a))
    if not valid:
        return []
    merged: list[tuple[int, int]] = [valid[0]]
    for a, b in valid[1:]:
        prev_a, prev_b = merged[-1]
        if a <= prev_b + gap_tolerance_ms:
            merged[-1] = (prev_a, max(prev_b, b))
        else:
            merged.append((a, b))
    return merged


def subtract_ranges(
    base: list[tuple[int, int]],
    covered: list[tuple[int, int]],
    min_span_ms: int = 500,
) -> list[tuple[int, int]]:
    """Return portions of base intervals not covered by covered intervals."""
    base_merged = merge_ranges(base, gap_tolerance_ms=0)
    cov_merged = merge_ranges(covered, gap_tolerance_ms=0)
    if not base_merged:
        return []
    if not cov_merged:
        return [(a, b) for a, b in base_merged if (b - a) >= min_span_ms]

    result: list[tuple[int, int]] = []
    for ba, bb in base_merged:
        cursor = ba
        for ca, cb in cov_merged:
            if cb <= cursor:
                continue
            if ca >= bb:
                break
            if ca > cursor and (ca - cursor) >= min_span_ms:
                result.append((cursor, ca))
            cursor = max(cursor, cb)
            if cursor >= bb:
                break
        if cursor < bb and (bb - cursor) >= min_span_ms:
            result.append((cursor, bb))
    return result


def is_pos_covered(
    pos_ms: int,
    ranges: list[tuple[int, int]],
    tolerance_ms: int = 500,
) -> bool:
    """Return True if pos_ms falls inside any covered interval within tolerance."""
    return any((a - tolerance_ms) <= pos_ms <= (b + tolerance_ms) for a, b in ranges)


def sum_ranges_ms(ranges: list[tuple[int, int]]) -> int:
    """Return total duration in milliseconds across merged intervals."""
    return sum(b - a for a, b in merge_ranges(ranges, gap_tolerance_ms=0))


def format_timestamp_mmss(sec: float) -> str:
    """Format seconds cleanly as MM:SS or H:MM:SS."""
    total = max(0, int(round(sec)))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _is_manual_seek_step(prev_pos_sec: float, cur_pos_sec: float, dt_sec: float) -> bool:
    """Return True when playback position jumped discontinuously rather than advancing normally."""
    elapsed = max(0.05, float(dt_sec))
    return cur_pos_sec < prev_pos_sec - 1.5 or cur_pos_sec > prev_pos_sec + max(3.5, elapsed * 4.5)


def format_ranges_summary(ranges_ms: list[tuple[int, int]], max_parts: int = 2) -> str:
    """Format millisecond intervals into a concise human-readable time summary."""
    merged = merge_ranges(ranges_ms, gap_tolerance_ms=2000)
    if not merged:
        return ""
    parts = [
        f"{format_timestamp_mmss(a / 1000.0)}–{format_timestamp_mmss(b / 1000.0)}"
        for a, b in merged[:max_parts]
    ]
    text = ", ".join(parts)
    if len(merged) > max_parts:
        text += f" 等{len(merged)}段"
    return text


def build_chunk_task_for_global_span(
    start_ms: int,
    max_duration_ms: int,
    segments: list[ResolvedSegment],
    segment_offsets_ms: list[int],
    index: int,
    clamp_end_ms: int | None = None,
) -> ChunkTask | None:
    """Build a single-segment ChunkTask starting at global start_ms, never crossing periods."""
    if not segments or len(segments) != len(segment_offsets_ms):
        return None
    if clamp_end_ms is not None and start_ms >= clamp_end_ms:
        return None

    for seg_idx, (seg, seg_start) in enumerate(zip(segments, segment_offsets_ms)):
        seg_dur_ms = int(round(max(0.1, float(seg.duration_sec)) * 1000.0))
        seg_end = seg_start + seg_dur_ms
        if start_ms >= seg_end:
            continue

        actual_start = max(seg_start, start_ms)
        target_end = actual_start + max(500, int(max_duration_ms))
        if clamp_end_ms is not None:
            target_end = min(target_end, clamp_end_ms)
        end_ms = min(seg_end, target_end)

        # Avoid leaving a tiny <0.5s tail fragment at the end of a period or hole.
        limit_end = min(seg_end, clamp_end_ms) if clamp_end_ms is not None else seg_end
        if 0 < (limit_end - end_ms) < 500:
            end_ms = limit_end

        if end_ms - actual_start < 100:
            return None

        local_start_sec = max(0.0, (actual_start - seg_start) / 1000.0)
        duration_sec = max(0.1, (end_ms - actual_start) / 1000.0)
        return ChunkTask(
            index=index,
            segment_idx=seg_idx,
            course_id=seg.course_id,
            audio_url=seg.audio_url,
            start_sec=local_start_sec,
            duration_sec=duration_sec,
            period_offset_ms=seg_start,
            global_start_ms=actual_start,
            global_end_ms=end_ms,
            ppt_slices=seg.ppt_slices,
        )
    return None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class LecturePipeline:
    """Coordinates probing, caching, smart-start detection, seek-priority ASR/LLM, and mpv HUD."""

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
        start_detector: Callable[..., Awaitable[float]] | None = None,
        hud_script_path: Path | None = None,
        thumb_script_path: Path | None = None,
    ) -> None:
        self.config = config
        self.root_dir = root_dir
        self._asr_factory = asr_factory or self._default_asr_factory
        self._translator_factory = translator_factory or self._default_translator_factory
        self._mpv_launcher = mpv_launcher
        self._ipc_factory = ipc_factory
        self._on_mpv_started = on_mpv_started
        self._mpv_exit_poll_interval = max(0.02, float(mpv_exit_poll_interval))
        self._start_detector = start_detector or detect_lecture_start_async
        self._hud_script_path = hud_script_path if hud_script_path is not None else get_hud_script_path()
        self._thumb_script_path = (
            thumb_script_path
            if thumb_script_path is not None
            else (get_thumb_script_path() if config.pipeline.enable_thumbnails else None)
        )

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
        asr_path = cache_dir / "asr.json"

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

            segment_offsets_ms, total_duration_ms = compute_segment_offsets_ms(resolved_segments)
            if total_duration_ms <= 0:
                raise RuntimeError("未能获取有效的课程视频时长。")
            total_duration_sec = total_duration_ms / 1000.0
            job.total_seconds = total_duration_sec

            media_title = f"{req.course_name} ({ordered_inputs[0].title})" if ordered_inputs else req.course_name
            pipe_path = build_mpv_pipe_path(job.job_id)

            # Load existing full or incremental cache unless force_retranscribe is requested.
            meta: dict[str, Any] = {}
            all_cues: list[BilingualCue] = []
            all_asr: list[AsrSentence] = []
            covered_ranges: list[tuple[int, int]] = []
            saved_lecture_start_sec: float | None = None

            if not req.force_retranscribe and meta_path.is_file():
                try:
                    loaded_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if isinstance(loaded_meta, dict):
                        meta = loaded_meta
                except json.JSONDecodeError:
                    meta = {}

                if cues_path.is_file():
                    try:
                        raw_cues = json.loads(cues_path.read_text(encoding="utf-8"))
                        if isinstance(raw_cues, list):
                            all_cues = [BilingualCue.model_validate(item) for item in raw_cues]
                            all_cues.sort(key=lambda c: (c.start_ms, c.end_ms))
                    except Exception:
                        all_cues = []

                if asr_path.is_file():
                    try:
                        raw_asr = json.loads(asr_path.read_text(encoding="utf-8"))
                        if isinstance(raw_asr, list):
                            all_asr = [AsrSentence.model_validate(item) for item in raw_asr]
                            all_asr.sort(key=lambda s: (s.start_ms, s.end_ms))
                    except Exception:
                        all_asr = []

                raw_start = meta.get("lecture_start_sec")
                if raw_start is not None:
                    try:
                        saved_lecture_start_sec = max(0.0, float(raw_start))
                    except (TypeError, ValueError):
                        saved_lecture_start_sec = None

                raw_ranges = meta.get("covered_ranges_ms")
                if isinstance(raw_ranges, list):
                    parsed_ranges: list[tuple[int, int]] = []
                    for item in raw_ranges:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            a = max(0, min(total_duration_ms, int(item[0])))
                            b = max(0, min(total_duration_ms, int(item[1])))
                            if b > a:
                                parsed_ranges.append((a, b))
                    covered_ranges = merge_ranges(parsed_ranges, gap_tolerance_ms=500)
                elif meta.get("complete") is True:
                    # Legacy cache without covered_ranges_ms means whole lecture was covered.
                    covered_ranges = [(0, total_duration_ms)]

            # Cache hit check: refresh EDLs (fresh signed keys) and reuse cached bilingual subtitles.
            if not req.force_retranscribe and meta.get("complete") is True and sub_path.is_file():
                if all_cues:
                    write_ass_atomic(sub_path, render_ass(all_cues, self.config.subtitle))

                proc = self._mpv_launcher(
                    mpv_path=mpv_path,
                    video_edl_path=video_edl_path,
                    sub_path=sub_path,
                    pipe_path=pipe_path,
                    media_title=media_title,
                    referer=self.config.pipeline.referer,
                    audio_edl_path=audio_edl_path if use_separate_audio else None,
                    script_path=self._hud_script_path,
                    thumb_script_path=self._thumb_script_path,
                )
                if self._on_mpv_started:
                    self._on_mpv_started(proc)

                if saved_lecture_start_sec is not None and saved_lecture_start_sec > 15.0:
                    ipc = self._ipc_factory(pipe_path)
                    ready = await asyncio.to_thread(ipc.wait_ready, 10.0)
                    if ready and proc.poll() is None:
                        cur_pos = await asyncio.to_thread(ipc.get_time_pos)
                        if cur_pos is None or cur_pos <= 5.0:
                            await asyncio.to_thread(ipc.seek_absolute, saved_lecture_start_sec)
                            await asyncio.to_thread(
                                ipc.show_osd,
                                f"命中缓存，已跳至开课点 {format_timestamp_mmss(saved_lecture_start_sec)}",
                                3500,
                            )

                job.cache_hit = True
                job.mpv_launched = True
                total = max(1, int(meta.get("chunks_total", 1)))
                job.chunks_total = total
                job.chunks_done = total
                job.covered_seconds = round(sum_ranges_ms(covered_ranges) / 1000.0, 1) or total_duration_sec
                job.lecture_start_sec = saved_lecture_start_sec
                job.state = "completed"
                job.message = "命中本地字幕缓存，已立即唤醒 mpv 播放"
                return

            # Cache miss or partial incremental cache: validate API keys before spawning transcription.
            if not self.config.effective_dashscope_api_key():
                raise RuntimeError(
                    "DashScope API Key 未配置。请在 config.yaml 的 dashscope.api_key 中填写或设置环境变量 DASHSCOPE_API_KEY。"
                )
            if not self.config.effective_llm_api_key():
                raise RuntimeError("LLM API Key 未配置，请检查 config.yaml 中的 dashscope.api_key / llm_api_key。")

            # Write initial subtitle file (with any cached cues, or loading placeholder) and launch mpv ASAP.
            write_ass_atomic(sub_path, render_ass(all_cues, self.config.subtitle))

            mpv_proc = self._mpv_launcher(
                mpv_path=mpv_path,
                video_edl_path=video_edl_path,
                sub_path=sub_path,
                pipe_path=pipe_path,
                media_title=media_title,
                referer=self.config.pipeline.referer,
                audio_edl_path=audio_edl_path if use_separate_audio else None,
                script_path=self._hud_script_path,
                thumb_script_path=self._thumb_script_path,
            )
            if self._on_mpv_started:
                self._on_mpv_started(mpv_proc)

            ipc = self._ipc_factory(pipe_path)
            await asyncio.to_thread(ipc.wait_ready, 10.0)

            job.mpv_launched = True
            job.state = "playing"
            job.covered_seconds = round(sum_ranges_ms(covered_ranges) / 1000.0, 1)
            job.message = "mpv 已开播，正在智能定位开课点并准备字幕..."

            stop_on_exit = bool(self.config.pipeline.stop_on_mpv_exit)

            def _ranges_to_sec(ranges_ms: list[tuple[int, int]]) -> list[list[float]]:
                return [[round(a / 1000.0, 1), round(b / 1000.0, 1)] for a, b in ranges_ms]

            async def _safe_get_time_pos() -> float | None:
                if mpv_proc.poll() is not None:
                    return None
                return await asyncio.to_thread(ipc.get_time_pos)

            async def _safe_consume_cancel() -> bool:
                if mpv_proc.poll() is not None:
                    return False
                return await asyncio.to_thread(ipc.consume_user_cancel_flag)

            async def _push_hud(
                phase: str,
                *,
                headline: str = "",
                in_flight_map: dict[int, tuple[ChunkTask, bool]] | None = None,
                anchor_ms_val: int = 0,
            ) -> None:
                if mpv_proc.poll() is not None:
                    return
                span_total = max(1, total_duration_ms - anchor_ms_val)
                anchor_cov_ms = sum(
                    max(0, min(b, total_duration_ms) - max(a, anchor_ms_val))
                    for a, b in covered_ranges
                    if b > anchor_ms_val
                )
                pct = min(100, max(0, int(round((anchor_cov_ms / span_total) * 100))))
                active_items = list((in_flight_map or {}).values())
                in_flight_ranges = [(t.global_start_ms, t.global_end_ms) for t, _ in active_items]
                urgent_pair = next(
                    ([round(t.global_start_ms / 1000.0, 1), round(t.global_end_ms / 1000.0, 1)] for t, urg in active_items if urg),
                    None,
                )
                payload: dict[str, Any] = {
                    "phase": phase,
                    "total_sec": round(total_duration_sec, 1),
                    "covered": _ranges_to_sec(covered_ranges),
                    "in_flight": _ranges_to_sec(in_flight_ranges),
                    "urgent": urgent_pair,
                    "covered_summary": format_ranges_summary(covered_ranges),
                    "pct": pct,
                    "headline": headline,
                }
                await asyncio.to_thread(ipc.send_hud_state, payload)

            def _persist_incremental(
                *,
                complete: bool,
                lecture_start_val: float | None,
                chunks_total_val: int,
            ) -> None:
                all_cues.sort(key=lambda c: (c.start_ms, c.end_ms))
                write_ass_atomic(sub_path, render_ass(all_cues, self.config.subtitle))
                _write_json(cues_path, [c.model_dump(mode="json") for c in all_cues])
                if all_asr:
                    all_asr.sort(key=lambda s: (s.start_ms, s.end_ms))
                    _write_json(asr_path, [s.model_dump(mode="json") for s in all_asr])
                _write_json(
                    meta_path,
                    {
                        "complete": complete,
                        "tecl_id": req.tecl_id,
                        "course_name": req.course_name,
                        "preferred_view": effective_view,
                        "lecture_start_sec": lecture_start_val,
                        "total_duration_ms": total_duration_ms,
                        "covered_ranges_ms": [[a, b] for a, b in covered_ranges],
                        "chunks_total": max(1, chunks_total_val),
                        "segments": [
                            {"course_id": s.course_id, "title": s.title, "duration_sec": s.duration_sec}
                            for s in resolved_segments
                        ],
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

            if stop_on_exit and mpv_proc.poll() is not None:
                job.state = "canceled"
                job.message = "mpv 已关闭，已停止后台转录"
                return

            # Determine starting anchor: respect watch_later (>15s), saved lecture start, or auto-detect.
            init_pos = await _safe_get_time_pos()
            anchor_ms = 0
            lecture_start_sec: float = 0.0
            auto_seek_target_sec: float | None = None
            auto_seek_origin_sec: float = init_pos or 0.0
            auto_seek_guard_until = 0.0

            if init_pos is not None and init_pos > 15.0:
                anchor_ms = min(max(0, total_duration_ms - 1000), int(round(init_pos * 1000.0)))
                lecture_start_sec = saved_lecture_start_sec if saved_lecture_start_sec is not None else round(init_pos, 1)
                logger.info(
                    "Job %s resuming from initial mpv position %.1fs (watch_later/seek)",
                    job.job_id,
                    init_pos,
                )
            elif saved_lecture_start_sec is not None and saved_lecture_start_sec > 15.0:
                lecture_start_sec = saved_lecture_start_sec
                anchor_ms = min(max(0, total_duration_ms - 1000), int(round(saved_lecture_start_sec * 1000.0)))
                logger.info(
                    "Job %s reusing saved lecture_start_sec=%.1fs",
                    job.job_id,
                    lecture_start_sec,
                )
                if mpv_proc.poll() is None:
                    await asyncio.to_thread(ipc.seek_absolute, lecture_start_sec)
                    await asyncio.to_thread(
                        ipc.show_osd,
                        f"已跳至上次开课点 {format_timestamp_mmss(lecture_start_sec)}",
                        3500,
                    )
                    auto_seek_target_sec = lecture_start_sec
                    auto_seek_origin_sec = init_pos or 0.0
                    auto_seek_guard_until = time.monotonic() + 6.0
            elif self.config.pipeline.auto_detect_start and not is_pos_covered(0, covered_ranges):
                await _push_hud("detecting", headline="🔍 正在智能定位开课时间...")
                detect_task = asyncio.create_task(
                    self._start_detector(
                        resolved_segments,
                        referer=self.config.pipeline.referer,
                        ffmpeg_path=ffmpeg_path,
                        max_scan_sec=self.config.pipeline.auto_detect_max_scan_seconds,
                        step_sec=self.config.pipeline.auto_detect_step_seconds,
                        silence_db=self.config.pipeline.auto_detect_silence_db,
                        lead_sec=self.config.pipeline.auto_start_lead_seconds,
                    )
                )
                manual_seek_pos: float | None = None
                prev_detect_pos = init_pos or 0.0
                prev_detect_ts = time.monotonic()
                try:
                    while not detect_task.done():
                        await asyncio.wait({detect_task}, timeout=0.25)
                        if detect_task.done():
                            break
                        if stop_on_exit and mpv_proc.poll() is not None:
                            detect_task.cancel()
                            await asyncio.gather(detect_task, return_exceptions=True)
                            job.state = "canceled"
                            job.message = "mpv 已关闭，已停止开课探测与转录"
                            return
                        if await _safe_consume_cancel():
                            detect_task.cancel()
                            await asyncio.gather(detect_task, return_exceptions=True)
                            await _push_hud("canceled")
                            await asyncio.to_thread(ipc.show_osd, "⏹ 已停止后台字幕转录", 2500)
                            job.state = "canceled"
                            job.message = "已从 mpv 停止后台字幕转录"
                            return
                        cur_pos = await _safe_get_time_pos()
                        now_ts = time.monotonic()
                        if cur_pos is not None:
                            if _is_manual_seek_step(prev_detect_pos, cur_pos, now_ts - prev_detect_ts):
                                manual_seek_pos = cur_pos
                                detect_task.cancel()
                                await asyncio.gather(detect_task, return_exceptions=True)
                                break
                            prev_detect_pos = cur_pos
                        prev_detect_ts = now_ts
                except asyncio.CancelledError:
                    if not detect_task.done():
                        detect_task.cancel()
                    await asyncio.gather(detect_task, return_exceptions=True)
                    raise

                if manual_seek_pos is not None:
                    lecture_start_sec = round(manual_seek_pos, 1)
                    anchor_ms = min(max(0, total_duration_ms - 1000), int(round(manual_seek_pos * 1000.0)))
                    logger.info(
                        "Job %s auto-detect canceled by manual user seek to %.1fs",
                        job.job_id,
                        manual_seek_pos,
                    )
                else:
                    detected_sec = await detect_task
                    cur_pos = await _safe_get_time_pos()
                    post_dt = max(0.05, time.monotonic() - prev_detect_ts)
                    if cur_pos is not None and _is_manual_seek_step(prev_detect_pos, cur_pos, post_dt):
                        lecture_start_sec = round(cur_pos, 1)
                        anchor_ms = min(max(0, total_duration_ms - 1000), int(round(cur_pos * 1000.0)))
                        logger.info(
                            "Job %s post-detect manual seek detected at %.1fs (ignoring auto-detect %.1fs)",
                            job.job_id,
                            cur_pos,
                            detected_sec,
                        )
                    elif 15.0 < detected_sec < (total_duration_sec - 5.0):
                        lecture_start_sec = detected_sec
                        anchor_ms = min(max(0, total_duration_ms - 1000), int(round(detected_sec * 1000.0)))
                        logger.info(
                            "Job %s auto-detected lecture start at %.1fs; seeking mpv",
                            job.job_id,
                            lecture_start_sec,
                        )
                        if mpv_proc.poll() is None:
                            await asyncio.to_thread(ipc.seek_absolute, lecture_start_sec)
                            await asyncio.to_thread(
                                ipc.show_osd,
                                f"已自动跳至开课点 {format_timestamp_mmss(lecture_start_sec)}",
                                3500,
                            )
                            auto_seek_target_sec = lecture_start_sec
                            auto_seek_origin_sec = prev_detect_pos
                            auto_seek_guard_until = time.monotonic() + 6.0
                    else:
                        logger.info(
                            "Job %s auto-detected lecture start near beginning (%.1fs); starting at 0.0s",
                            job.job_id,
                            detected_sec,
                        )

            job.lecture_start_sec = lecture_start_sec
            focus_ms = anchor_ms

            asr_provider = self._asr_factory(self.config)
            translator = self._translator_factory(self.config)
            merge_lock = asyncio.Lock()

            async def process_single_chunk(task: ChunkTask) -> tuple[list[AsrSentence], list[BilingualCue]]:
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
                    global_asr = [
                        AsrSentence(
                            start_ms=base_offset_ms + sent.start_ms,
                            end_ms=base_offset_ms + sent.end_ms,
                            text=sent.text,
                        )
                        for sent in sentences
                    ]
                    global_cues = [
                        BilingualCue(
                            start_ms=base_offset_ms + cue.start_ms,
                            end_ms=base_offset_ms + cue.end_ms,
                            en=cue.en,
                            zh=cue.zh,
                        )
                        for cue in translated
                    ]
                    return global_asr, global_cues
                finally:
                    audio_path.unlink(missing_ok=True)

            # Dynamic priority range scheduler + mpv seek/exit monitor.
            bg_limit = max(1, self.config.pipeline.max_concurrency)
            urgent_burst_limit = max(0, self.config.pipeline.urgent_burst_concurrency)
            urgent_ms = max(30_000, int(self.config.pipeline.urgent_chunk_seconds) * 1000)
            chunk_ms = max(60_000, int(self.config.pipeline.chunk_seconds) * 1000)
            seek_poll_sec = max(0.05, float(self.config.pipeline.seek_poll_interval_seconds))
            seek_debounce_sec = max(0.0, float(self.config.pipeline.seek_debounce_seconds))

            in_flight: dict[int, tuple[ChunkTask, bool]] = {}
            worker_tasks: dict[int, asyncio.Task[None]] = {}
            worker_error: BaseException | None = None
            next_chunk_idx = 0
            wake_event = asyncio.Event()

            def _estimate_total_chunks() -> int:
                rem = subtract_ranges([(anchor_ms, total_duration_ms)], covered_ranges)
                rem_chunks = sum(max(1, math.ceil((b - a) / float(chunk_ms))) for a, b in rem)
                if rem and rem_chunks < len(in_flight):
                    rem_chunks = len(in_flight)
                return max(1, job.chunks_done + rem_chunks)

            def _pick_next_task(urgent_only: bool) -> tuple[ChunkTask, bool] | None:
                nonlocal next_chunk_idx
                in_flight_spans = [(t.global_start_ms, t.global_end_ms) for t, _ in in_flight.values()]
                claimed = merge_ranges(covered_ranges + in_flight_spans, gap_tolerance_ms=200)

                min_focus_floor = anchor_ms if focus_ms >= anchor_ms else 0
                focus_floor = max(min_focus_floor, focus_ms - 2000)
                if focus_ms != anchor_ms and focus_floor < total_duration_ms:
                    seek_base = focus_floor
                else:
                    seek_base = min(max(0, focus_ms), total_duration_ms)

                from_focus = subtract_ranges([(seek_base, total_duration_ms)], claimed)
                if from_focus:
                    gap_start, gap_end = from_focus[0]
                    if gap_start <= focus_ms + 3000:
                        task = build_chunk_task_for_global_span(
                            gap_start,
                            urgent_ms,
                            resolved_segments,
                            segment_offsets_ms,
                            next_chunk_idx,
                            clamp_end_ms=gap_end,
                        )
                        if task is not None:
                            next_chunk_idx += 1
                            return task, True

                if urgent_only:
                    return None

                if from_focus:
                    gap_start, gap_end = from_focus[0]
                    span_ms = urgent_ms if (gap_start - focus_ms) <= 30_000 else chunk_ms
                    task = build_chunk_task_for_global_span(
                        gap_start,
                        span_ms,
                        resolved_segments,
                        segment_offsets_ms,
                        next_chunk_idx,
                        clamp_end_ms=gap_end,
                    )
                    if task is not None:
                        next_chunk_idx += 1
                        return task, False

                from_anchor = subtract_ranges([(anchor_ms, total_duration_ms)], claimed)
                if from_anchor:
                    gap_start, gap_end = from_anchor[0]
                    task = build_chunk_task_for_global_span(
                        gap_start,
                        chunk_ms,
                        resolved_segments,
                        segment_offsets_ms,
                        next_chunk_idx,
                        clamp_end_ms=gap_end,
                    )
                    if task is not None:
                        next_chunk_idx += 1
                        return task, False

                return None

            async def _run_worker(task: ChunkTask, is_urgent: bool) -> None:
                nonlocal covered_ranges, worker_error, all_cues, all_asr
                try:
                    chunk_asr, chunk_cues = await process_single_chunk(task)
                    if stop_on_exit and mpv_proc.poll() is not None:
                        return
                    async with merge_lock:
                        g_start = task.global_start_ms
                        g_end = task.global_end_ms
                        if chunk_asr:
                            all_asr = [
                                s
                                for s in all_asr
                                if not (g_start <= ((s.start_ms + s.end_ms) // 2) < g_end)
                            ]
                            all_asr.extend(chunk_asr)
                        if chunk_cues:
                            all_cues = [
                                c
                                for c in all_cues
                                if not (g_start <= ((c.start_ms + c.end_ms) // 2) < g_end)
                            ]
                            all_cues.extend(chunk_cues)
                        covered_ranges = merge_ranges([*covered_ranges, (g_start, g_end)], gap_tolerance_ms=500)
                        job.chunks_done += 1
                        job.chunks_total = _estimate_total_chunks()
                        job.covered_seconds = round(sum_ranges_ms(covered_ranges) / 1000.0, 1)
                        summary = format_ranges_summary(covered_ranges)
                        job.message = f"mpv 播放中，字幕已覆盖 {summary} ({job.chunks_done}/{job.chunks_total})"

                        _persist_incremental(
                            complete=False,
                            lecture_start_val=lecture_start_sec,
                            chunks_total_val=job.chunks_total,
                        )
                    if mpv_proc.poll() is None:
                        if chunk_cues or all_cues:
                            await asyncio.to_thread(ipc.reload_subtitles)
                        if is_urgent and (task.global_start_ms - 5000) <= focus_ms <= (task.global_end_ms + 30_000):
                            await asyncio.to_thread(ipc.show_osd, "✓ 当前位置字幕已加载", 1800)
                        remaining_in_flight = {k: v for k, v in in_flight.items() if k != task.index}
                        await _push_hud("transcribing", in_flight_map=remaining_in_flight, anchor_ms_val=anchor_ms)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    worker_error = exc
                finally:
                    in_flight.pop(task.index, None)
                    worker_tasks.pop(task.index, None)
                    wake_event.set()

            job.chunks_total = _estimate_total_chunks()
            last_polled_pos_ms = anchor_ms
            last_poll_ts = time.monotonic()
            pending_seek_ms: int | None = None
            pending_seek_since = 0.0
            last_hud_push_ts = 0.0

            async def _cancel_all_workers() -> None:
                tasks_to_cancel = [t for t in worker_tasks.values() if not t.done()]
                for t in tasks_to_cancel:
                    t.cancel()
                if tasks_to_cancel:
                    await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                worker_tasks.clear()
                in_flight.clear()

            try:
                while True:
                    wake_event.clear()
                    if worker_error is not None:
                        await _cancel_all_workers()
                        if covered_ranges:
                            _persist_incremental(
                                complete=False,
                                lecture_start_val=lecture_start_sec,
                                chunks_total_val=job.chunks_total,
                            )
                        raise worker_error

                    if stop_on_exit and mpv_proc.poll() is not None:
                        await _cancel_all_workers()
                        if covered_ranges:
                            _persist_incremental(
                                complete=False,
                                lecture_start_val=lecture_start_sec,
                                chunks_total_val=job.chunks_total,
                            )
                        logger.info(
                            "mpv exited for job %s; stopped background transcription at %d chunks",
                            job.job_id,
                            job.chunks_done,
                        )
                        job.state = "canceled"
                        job.message = f"mpv 已关闭，已停止后台转录 ({job.chunks_done}/{job.chunks_total})"
                        return

                    if await _safe_consume_cancel():
                        await _cancel_all_workers()
                        if covered_ranges:
                            _persist_incremental(
                                complete=False,
                                lecture_start_val=lecture_start_sec,
                                chunks_total_val=job.chunks_total,
                            )
                        await _push_hud("canceled", in_flight_map={}, anchor_ms_val=anchor_ms)
                        await asyncio.to_thread(ipc.show_osd, "⏹ 已停止后台字幕转录（已保留缓存）", 2500)
                        job.state = "canceled"
                        job.message = f"已从 mpv 停止后台转录 ({job.chunks_done}/{job.chunks_total})"
                        return

                    now = time.monotonic()
                    cur_pos_sec = await _safe_get_time_pos()
                    if auto_seek_target_sec is not None and cur_pos_sec is not None:
                        if cur_pos_sec >= auto_seek_target_sec - 15.0 or now >= auto_seek_guard_until:
                            auto_seek_target_sec = None
                        elif abs(cur_pos_sec - auto_seek_origin_sec) <= 10.0:
                            # mpv has not completed the initial programmatic HTTP seek yet.
                            cur_pos_sec = None

                    focus_changed = False
                    if cur_pos_sec is not None:
                        cur_pos_ms = min(max(0, total_duration_ms - 500), int(round(cur_pos_sec * 1000.0)))
                        dt = max(0.05, now - last_poll_ts)
                        max_normal_step = max(5000, int(dt * 4500))
                        jumped = (
                            cur_pos_ms < last_polled_pos_ms - 1500
                            or cur_pos_ms > last_polled_pos_ms + max_normal_step
                        )
                        if jumped:
                            if pending_seek_ms is None or abs(cur_pos_ms - pending_seek_ms) > 2500:
                                pending_seek_ms = cur_pos_ms
                                pending_seek_since = now
                        if pending_seek_ms is not None:
                            if (now - pending_seek_since) >= seek_debounce_sec:
                                if abs(cur_pos_ms - focus_ms) > 1500:
                                    focus_ms = cur_pos_ms
                                    focus_changed = True
                                pending_seek_ms = None
                        elif not is_pos_covered(cur_pos_ms, covered_ranges):
                            if abs(cur_pos_ms - focus_ms) > 1500:
                                focus_ms = cur_pos_ms
                                focus_changed = True

                        last_polled_pos_ms = cur_pos_ms
                    last_poll_ts = now

                    # Dispatch background and urgent burst tasks.
                    dispatched = False
                    while len(in_flight) < bg_limit:
                        picked = _pick_next_task(urgent_only=False)
                        if picked is None:
                            break
                        task, is_urg = picked
                        in_flight[task.index] = (task, is_urg)
                        worker_tasks[task.index] = asyncio.create_task(_run_worker(task, is_urg))
                        dispatched = True

                    if len(in_flight) < bg_limit + urgent_burst_limit:
                        picked_urg = _pick_next_task(urgent_only=True)
                        if picked_urg is not None:
                            task, is_urg = picked_urg
                            in_flight[task.index] = (task, is_urg)
                            worker_tasks[task.index] = asyncio.create_task(_run_worker(task, is_urg))
                            dispatched = True

                    if dispatched or focus_changed or (now - last_hud_push_ts) >= 2.0:
                        job.chunks_total = _estimate_total_chunks()
                        await _push_hud("transcribing", in_flight_map=in_flight, anchor_ms_val=anchor_ms)
                        last_hud_push_ts = now

                    remaining_after_anchor = subtract_ranges([(anchor_ms, total_duration_ms)], covered_ranges)
                    if not worker_tasks and not in_flight and not remaining_after_anchor:
                        break

                    try:
                        await asyncio.wait_for(wake_event.wait(), timeout=seek_poll_sec)
                    except asyncio.TimeoutError:
                        pass
            except asyncio.CancelledError:
                await _cancel_all_workers()
                if covered_ranges:
                    _persist_incremental(
                        complete=False,
                        lecture_start_val=lecture_start_sec,
                        chunks_total_val=job.chunks_total,
                    )
                raise

            # Completion!
            has_cues = bool(all_cues)
            final_chunks_total = max(1, job.chunks_done)
            job.chunks_total = final_chunks_total
            job.covered_seconds = round(sum_ranges_ms(covered_ranges) / 1000.0, 1)
            if not has_cues:
                logger.warning(
                    "Playback job %s completed with zero speech cues across %d chunks",
                    job.job_id,
                    job.chunks_done,
                )
            _persist_incremental(
                complete=has_cues,
                lecture_start_val=lecture_start_sec,
                chunks_total_val=final_chunks_total,
            )

            completion_msg = (
                "整堂课双语字幕已全部就绪" if has_cues else "处理完成，但整段音频未检测到有效语音"
            )
            if mpv_proc.poll() is None:
                await _push_hud("completed", in_flight_map={}, anchor_ms_val=anchor_ms)
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
