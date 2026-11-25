"""Media probing and HTTP-range audio chunk extraction using ffprobe and ffmpeg."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import BaseModel

from sjtu_mpv_sub.models import PptSlice, SegmentInput, StreamSource, ViewMode

logger = logging.getLogger(__name__)
_MEAN_VOLUME_RE = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?|-?inf)\s*dB", re.IGNORECASE)


class StreamProbe(BaseModel):
    """Probe metadata for a single media URL or file."""

    duration: float
    has_video: bool
    has_audio: bool


class ResolvedSegment(BaseModel):
    """Resolved video/audio stream assignment and metadata for one period."""

    course_id: int
    title: str
    video_url: str
    audio_url: str
    separate_audio: bool
    duration_sec: float
    ppt_slices: list[PptSlice]


def _no_window_kwargs() -> dict[str, int]:
    """Prevent console windows from flashing when invoked from pythonw.exe on Windows."""
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def _http_header_args(url: str, referer: str) -> list[str]:
    if referer and url.lower().startswith(("http://", "https://")):
        return ["-headers", f"Referer: {referer}\r\n"]
    return []


def probe_stream(
    url: str,
    referer: str = "",
    ffprobe_path: str = "ffprobe",
    timeout: float = 20.0,
) -> StreamProbe:
    """Inspect duration and stream codecs via ffprobe."""
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        *_http_header_args(url, referer),
        "-show_entries",
        "format=duration:stream=codec_type",
        "-of",
        "json",
        url,
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        **_no_window_kwargs(),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout or "{}")
    fmt = payload.get("format") or {}
    streams = payload.get("streams") or []

    duration = 0.0
    raw_duration = fmt.get("duration")
    if raw_duration is not None:
        try:
            duration = max(0.0, float(raw_duration))
        except (TypeError, ValueError):
            duration = 0.0

    codec_types = {str(s.get("codec_type", "")).lower() for s in streams}
    return StreamProbe(
        duration=duration,
        has_video="video" in codec_types,
        has_audio="audio" in codec_types,
    )


def _select_primary_and_alt(
    streams: list[StreamSource],
    preferred_view: ViewMode,
) -> tuple[StreamSource, StreamSource | None]:
    if not streams:
        raise ValueError("Segment contains no media streams")

    # SJTU course_vod_urls_new: viewNum=1 is classroom camera, viewNum=5 is PC/PPT screen capture.
    target_view_num = 5 if preferred_view == "ppt" else 1
    primary = next((s for s in streams if s.view_num == target_view_num), streams[0])
    alt = next((s for s in streams if s.url != primary.url), None)
    return primary, alt


def resolve_segment_tracks(
    segment: SegmentInput,
    preferred_view: ViewMode = "ppt",
    referer: str = "",
    ffprobe_path: str = "ffprobe",
) -> ResolvedSegment:
    """Resolve primary video URL, audio-bearing URL, and exact duration for a segment."""
    primary, alt = _select_primary_and_alt(segment.streams, preferred_view)

    duration_sec = 0.0
    primary_has_audio = True
    try:
        probe = probe_stream(primary.url, referer=referer, ffprobe_path=ffprobe_path)
        duration_sec = probe.duration
        primary_has_audio = probe.has_audio
    except Exception as exc:
        logger.warning(
            "ffprobe failed on primary stream for course_id=%s (%s); falling back to vod_time",
            segment.course_id,
            exc,
        )
        if primary.vod_time and primary.vod_time > 0:
            duration_sec = float(primary.vod_time)
        else:
            raise RuntimeError(f"Unable to probe primary stream for course_id={segment.course_id}: {exc}") from exc

    if duration_sec <= 0 and primary.vod_time and primary.vod_time > 0:
        duration_sec = float(primary.vod_time)

    audio_url = primary.url
    separate_audio = False

    if not primary_has_audio and alt is not None:
        try:
            alt_probe = probe_stream(alt.url, referer=referer, ffprobe_path=ffprobe_path)
            if alt_probe.has_audio:
                audio_url = alt.url
                separate_audio = True
                logger.info(
                    "Primary viewNum=%s has no audio for course_id=%s; attaching audio from viewNum=%s",
                    primary.view_num,
                    segment.course_id,
                    alt.view_num,
                )
        except Exception as exc:
            logger.warning(
                "ffprobe failed on alternate stream for course_id=%s: %s",
                segment.course_id,
                exc,
            )

    return ResolvedSegment(
        course_id=segment.course_id,
        title=segment.title,
        video_url=primary.url,
        audio_url=audio_url,
        separate_audio=separate_audio,
        duration_sec=duration_sec,
        ppt_slices=segment.ppt_slices,
    )


def extract_audio_chunk(
    audio_url: str,
    start_sec: float,
    duration_sec: float,
    output_audio: Path,
    referer: str = "",
    ffmpeg_path: str = "ffmpeg",
    timeout: float = 120.0,
) -> Path:
    """Extract a 16kHz mono 48kbps MP3 chunk using fast HTTP range input seeking."""
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    # Place -ss BEFORE -i so ffmpeg seeks remotely via HTTP 206 byte ranges.
    cmd = [
        ffmpeg_path,
        "-v",
        "error",
        *_http_header_args(audio_url, referer),
        "-ss",
        f"{max(0.0, start_sec):.3f}",
        "-i",
        audio_url,
        "-t",
        f"{max(0.1, duration_sec):.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "48k",
        "-y",
        str(output_audio),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        **_no_window_kwargs(),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed ({proc.returncode}): {proc.stderr.strip()}")
    if not output_audio.is_file() or output_audio.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced empty audio file: {output_audio}")
    return output_audio


async def resolve_segment_tracks_async(
    segment: SegmentInput,
    preferred_view: ViewMode = "ppt",
    referer: str = "",
    ffprobe_path: str = "ffprobe",
) -> ResolvedSegment:
    """Async wrapper for resolve_segment_tracks."""
    return await asyncio.to_thread(
        resolve_segment_tracks,
        segment,
        preferred_view,
        referer,
        ffprobe_path,
    )


async def extract_audio_chunk_async(
    audio_url: str,
    start_sec: float,
    duration_sec: float,
    output_audio: Path,
    referer: str = "",
    ffmpeg_path: str = "ffmpeg",
    timeout: float = 120.0,
) -> Path:
    """Async wrapper for extract_audio_chunk."""
    return await asyncio.to_thread(
        extract_audio_chunk,
        audio_url,
        start_sec,
        duration_sec,
        output_audio,
        referer,
        ffmpeg_path,
        timeout,
    )


def probe_audio_rms_db(
    audio_url: str,
    start_sec: float,
    probe_duration_sec: float = 2.0,
    referer: str = "",
    ffmpeg_path: str = "ffmpeg",
    timeout: float = 15.0,
) -> float | None:
    """Measure speech-band mean volume (dB) over a short HTTP-range audio slice."""
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-nostats",
        "-v",
        "info",
        *_http_header_args(audio_url, referer),
        "-ss",
        f"{max(0.0, start_sec):.3f}",
        "-i",
        audio_url,
        "-t",
        f"{max(0.5, probe_duration_sec):.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "8000",
        "-af",
        "highpass=f=200,lowpass=f=3400,volumedetect",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            **_no_window_kwargs(),
        )
    except Exception as exc:
        logger.debug("ffmpeg volumedetect failed at %.1fs: %s", start_sec, exc)
        return None

    if proc.returncode != 0:
        logger.debug("ffmpeg volumedetect exited %d at %.1fs: %s", proc.returncode, start_sec, proc.stderr.strip()[:200])
        return None

    match = _MEAN_VOLUME_RE.search(proc.stderr or "")
    if not match:
        return None
    raw = match.group(1).lower()
    if "inf" in raw:
        return -91.0
    try:
        return float(raw)
    except ValueError:
        return None


async def probe_audio_rms_db_async(
    audio_url: str,
    start_sec: float,
    probe_duration_sec: float = 2.0,
    referer: str = "",
    ffmpeg_path: str = "ffmpeg",
    timeout: float = 15.0,
) -> float | None:
    """Async wrapper for probe_audio_rms_db."""
    return await asyncio.to_thread(
        probe_audio_rms_db,
        audio_url,
        start_sec,
        probe_duration_sec,
        referer,
        ffmpeg_path,
        timeout,
    )


def locate_global_time(
    segments: list[ResolvedSegment],
    global_sec: float,
) -> tuple[ResolvedSegment, float] | None:
    """Map a global timeline offset (seconds) to (segment, local_offset_sec)."""
    if not segments or global_sec < 0:
        return None
    cursor = 0.0
    for seg in segments:
        dur = max(0.0, float(seg.duration_sec))
        if global_sec < cursor + dur:
            return seg, max(0.0, global_sec - cursor)
        cursor += dur
    return None


async def detect_lecture_start_async(
    segments: list[ResolvedSegment],
    *,
    referer: str = "",
    ffmpeg_path: str = "ffmpeg",
    max_scan_sec: float = 2400.0,
    step_sec: float = 120.0,
    silence_db: float = -38.0,
    lead_sec: float = 10.0,
    probe_fn: Callable[[str, float, float], Awaitable[float | None]] | None = None,
) -> float:
    """Detect when sustained speech begins, returning recommended start offset in seconds."""
    total_dur = sum(max(0.0, float(s.duration_sec)) for s in segments)
    if not segments or total_dur <= 30.0:
        return 0.0

    async def _default_probe(audio_url: str, local_sec: float, probe_dur: float) -> float | None:
        return await probe_audio_rms_db_async(
            audio_url,
            local_sec,
            probe_duration_sec=probe_dur,
            referer=referer,
            ffmpeg_path=ffmpeg_path,
        )

    active_probe = probe_fn or _default_probe

    async def _probe_global(g_sec: float) -> float | None:
        located = locate_global_time(segments, g_sec)
        if located is None:
            return None
        seg, local_sec = located
        rem = max(0.5, min(2.0, float(seg.duration_sec) - local_sec))
        return await active_probe(seg.audio_url, local_sec, rem)

    async def _is_sustained(t_sec: float) -> bool:
        for offset in (15.0, 30.0):
            check_t = t_sec + offset
            if check_t >= total_dur - 2.5:
                return True
            val = await _probe_global(check_t)
            if val is not None and val >= silence_db:
                return True
        return False

    # Wave 0: check near the very start and ~45s in parallel.
    t5_pos = min(5.0, max(0.0, total_dur - 3.0))
    t45_pos = min(45.0, max(0.0, total_dur - 2.5))
    t5_val, t45_val = await asyncio.gather(_probe_global(t5_pos), _probe_global(t45_pos))

    if t5_val is not None and t5_val >= silence_db:
        if total_dur < 60.0 or (t45_val is not None and t45_val >= silence_db):
            return 0.0

    lo: float | None = None
    hi: float | None = None

    if t45_val is not None and t45_val >= silence_db and await _is_sustained(t45_pos):
        lo = t5_pos
        hi = t45_pos
    else:
        step = max(30.0, float(step_sec))
        scan_limit = min(total_dur - 3.0, float(max_scan_sec))
        grid: list[float] = []
        cursor = step
        while cursor <= scan_limit:
            grid.append(cursor)
            cursor += step

        sem = asyncio.Semaphore(6)

        async def _scan_one(t_sec: float) -> tuple[float, float | None]:
            async with sem:
                return t_sec, await _probe_global(t_sec)

        prev_silent = t45_pos
        for wave in (grid[:8], grid[8:]):
            if not wave:
                continue
            results = await asyncio.gather(*(_scan_one(t) for t in wave))
            for t_sec, rms in results:
                if rms is not None and rms >= silence_db:
                    if await _is_sustained(t_sec):
                        lo = prev_silent
                        hi = t_sec
                        break
                else:
                    prev_silent = t_sec
            if hi is not None:
                break

    if hi is None or lo is None:
        return 0.0

    # Wave 2: 3-step binary refinement between last silent point and first confirmed speech.
    for _ in range(3):
        if hi - lo <= 12.0:
            break
        mid = (lo + hi) / 2.0
        rms = await _probe_global(mid)
        if rms is not None and rms >= silence_db:
            hi = mid
        else:
            lo = mid

    return max(0.0, round(hi - max(0.0, float(lead_sec)), 1))
