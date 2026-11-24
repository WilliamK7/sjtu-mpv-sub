"""Media probing and HTTP-range audio chunk extraction using ffprobe and ffmpeg."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

from sjtu_mpv_sub.models import PptSlice, SegmentInput, StreamSource, ViewMode

logger = logging.getLogger(__name__)


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
