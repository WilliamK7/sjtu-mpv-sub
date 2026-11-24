"""Styled bilingual ASS subtitle generator and atomic UTF-8-BOM writer."""

from __future__ import annotations

import os
import time
from pathlib import Path

from sjtu_mpv_sub.config import SubtitleConfig
from sjtu_mpv_sub.models import BilingualCue


def format_ass_timestamp(ms: int) -> str:
    """Format milliseconds as ASS timestamp H:MM:SS.cc (centiseconds)."""
    clamped = max(0, int(ms))
    total_cs = round(clamped / 10.0)
    cs = total_cs % 100
    total_sec = total_cs // 100
    sec = total_sec % 60
    total_min = total_sec // 60
    minutes = total_min % 60
    hours = total_min // 60
    return f"{hours}:{minutes:02d}:{sec:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    """Sanitize raw subtitle text so braces/newlines never break libass rendering."""
    cleaned = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    # Replace ASCII braces with fullwidth braces so math set notation stays visible.
    return cleaned.replace("{", "｛").replace("}", "｝")


def _normalize_cues(cues: list[BilingualCue]) -> list[BilingualCue]:
    ordered = sorted(cues, key=lambda c: (c.start_ms, c.end_ms))
    normalized: list[BilingualCue] = []
    for idx, cue in enumerate(ordered):
        start_ms = max(0, int(cue.start_ms))
        end_ms = max(start_ms + 100, int(cue.end_ms))
        if idx + 1 < len(ordered):
            next_start = max(0, int(ordered[idx + 1].start_ms))
            if next_start > start_ms and end_ms > next_start:
                end_ms = max(start_ms + 100, next_start)
        normalized.append(
            BilingualCue(
                start_ms=start_ms,
                end_ms=end_ms,
                en=_escape_ass_text(cue.en),
                zh=_escape_ass_text(cue.zh),
            )
        )
    return normalized


def render_ass(cues: list[BilingualCue], subtitle_config: SubtitleConfig | None = None) -> str:
    """Render bilingual cues into a styled ASS subtitle document."""
    cfg = subtitle_config or SubtitleConfig()
    zh_top = cfg.bilingual_order == "zh_top"
    default_style = "ZH" if zh_top else "EN"

    header = "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1920",
            "PlayResY: 1080",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: ZH,{cfg.font_zh},{cfg.size_zh},&H00FFFFFF,&H000000FF,&H66000000,&H66000000,"
            "-1,0,0,0,100,100,0,0,3,8,0,2,60,60,42,1",
            f"Style: EN,{cfg.font_en},{cfg.size_en},&H00B8E8FF,&H000000FF,&H66000000,&H66000000,"
            "0,0,0,0,100,100,0,0,3,8,0,2,60,60,42,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )

    lines = [header]
    normalized = _normalize_cues(cues)

    if not normalized:
        lines.append("Dialogue: 0,0:00:00.00,0:00:05.00,ZH,,0,0,0,,双语字幕加载中...")
        return "\n".join(lines) + "\n"

    for cue in normalized:
        start_str = format_ass_timestamp(cue.start_ms)
        end_str = format_ass_timestamp(cue.end_ms)
        if zh_top:
            if cue.zh and cue.en:
                body = f"{cue.zh}\\N{{\\rEN}}{cue.en}"
            else:
                body = cue.zh or f"{{\\rEN}}{cue.en}"
        else:
            if cue.en and cue.zh:
                body = f"{cue.en}\\N{{\\rZH}}{cue.zh}"
            else:
                body = cue.en or f"{{\\rZH}}{cue.zh}"

        if not body:
            continue
        lines.append(f"Dialogue: 0,{start_str},{end_str},{default_style},,0,0,0,,{body}")

    return "\n".join(lines) + "\n"


def write_ass_atomic(path: Path, content: str, retries: int = 5) -> None:
    """Atomically write ASS file with UTF-8 BOM (required to override mpv-lazy GB18030)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".ass.tmp")
    tmp_path.write_text(content, encoding="utf-8-sig")

    last_err: OSError | None = None
    for attempt in range(retries):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError as exc:
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(0.05)

    if last_err is not None:
        raise last_err
