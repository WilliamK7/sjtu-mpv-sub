"""Styled bilingual ASS subtitle generator and atomic UTF-8-BOM writer."""

from __future__ import annotations

import math
import os
import time
import unicodedata
from pathlib import Path

from sjtu_mpv_sub.config import SubtitleConfig
from sjtu_mpv_sub.models import BilingualCue
from sjtu_mpv_sub.translator import sanitize_subtitle_text

_ZH_NO_BREAK_BEFORE = frozenset("。！？；：，、.,!?;:）)]}》」』”’%℃°")
_ZH_NO_BREAK_AFTER = frozenset("（([{《「『“‘")
_ZH_STRONG_BREAK_AFTER = frozenset("。！？；：.!?;:")
_ZH_SOFT_BREAK_AFTER = frozenset("，、,")


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


def _is_fullwidth_char(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in ("W", "F")


def _char_visual_units(ch: str) -> float:
    return 1.0 if _is_fullwidth_char(ch) else 0.58


def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF) or (0x3000 <= code <= 0x303F) or (0xFF00 <= code <= 0xFFEF)


def _find_balanced_zh_split(text: str, target_units: float, max_units: float) -> int | None:
    """Pick the cleanest split index near target_units without stranding orphan prefixes."""
    n = len(text)
    if n < 6:
        return None

    cum_units = [0.0] * (n + 1)
    for i, ch in enumerate(text):
        cum_units[i + 1] = cum_units[i] + _char_visual_units(ch)
    total = cum_units[n]
    if total <= max_units:
        return None

    min_units = max(5.0, target_units * 0.52)
    upper_units = min(max_units, max(min_units + 2.0, target_units * 1.25))

    best_score = -1e9
    best_idx: int | None = None

    for i in range(2, n - 1):
        left_u = cum_units[i]
        right_u = total - left_u
        if left_u < min_units or right_u < 4.5 or left_u > upper_units:
            continue

        prev_ch = text[i - 1]
        next_ch = text[i]
        if next_ch in _ZH_NO_BREAK_BEFORE or prev_ch in _ZH_NO_BREAK_AFTER:
            continue

        priority: float | None = None
        if prev_ch in _ZH_STRONG_BREAK_AFTER:
            priority = 300.0
        elif prev_ch in _ZH_SOFT_BREAK_AFTER:
            priority = 220.0
        elif next_ch == " " or prev_ch == " ":
            priority = 160.0
        elif _is_cjk_char(prev_ch) or _is_cjk_char(next_ch):
            # Allow breaking cleanly between CJK chars or at CJK/symbol boundary, never inside ASCII words.
            if not (prev_ch.isascii() and prev_ch.isalnum() and next_ch.isascii() and next_ch.isalnum()):
                priority = 100.0

        if priority is None:
            continue

        distance_penalty = abs(left_u - target_units) * 6.0
        score = priority - distance_penalty
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is not None:
        return best_idx

    # Fallback: pick nearest breakable position inside max_units.
    for i in range(n - 2, 1, -1):
        if cum_units[i] <= max_units and text[i] not in _ZH_NO_BREAK_BEFORE and text[i - 1] not in _ZH_NO_BREAK_AFTER:
            prev_ch, next_ch = text[i - 1], text[i]
            if not (prev_ch.isascii() and prev_ch.isalnum() and next_ch.isascii() and next_ch.isalnum()):
                return i
    return None


def wrap_zh_for_ass(text: str, size_zh: int = 46) -> str:
    """Wrap long Chinese text with explicit \\N breaks and prevent libass orphan space breaks."""
    if not text:
        return ""

    usable_px = 1620.0
    max_units = max(18.0, usable_px / max(20, int(size_zh)))
    total_units = sum(_char_visual_units(ch) for ch in text)

    lines: list[str] = []
    remaining = text.strip()
    if total_units <= max_units:
        lines = [remaining]
    else:
        n_lines = max(2, math.ceil(total_units / max_units))
        while remaining:
            rem_units = sum(_char_visual_units(ch) for ch in remaining)
            if rem_units <= max_units:
                lines.append(remaining)
                break
            target = rem_units / max(2, math.ceil(rem_units / max_units))
            split_idx = _find_balanced_zh_split(remaining, target, max_units)
            if split_idx is None:
                lines.append(remaining)
                break
            head = remaining[:split_idx].strip()
            tail = remaining[split_idx:].strip()
            if not head or not tail:
                lines.append(remaining)
                break
            lines.append(head)
            remaining = tail
            if len(lines) >= n_lines + 2:
                if remaining:
                    lines.append(remaining)
                break

    # Replace ASCII spaces inside wrapped Chinese lines with NBSP so libass WrapStyle=0
    # never inserts an unwanted orphan break like "关于\Nδ...".
    return "\\N".join(line.replace(" ", "\u00a0") for line in lines if line)


def _escape_ass_text(text: str, *, is_zh: bool = False, size_zh: int = 46) -> str:
    """Sanitize raw subtitle text so control chars, markup, and braces never break libass rendering."""
    cleaned = sanitize_subtitle_text(text, is_zh=is_zh)
    # Replace ASCII braces with fullwidth braces so math set notation stays visible.
    escaped = cleaned.replace("{", "｛").replace("}", "｝")
    if is_zh and escaped:
        return wrap_zh_for_ass(escaped, size_zh=size_zh)
    return escaped


def _normalize_cues(cues: list[BilingualCue], *, size_zh: int = 46) -> list[BilingualCue]:
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
                en=_escape_ass_text(cue.en, is_zh=False, size_zh=size_zh),
                zh=_escape_ass_text(cue.zh, is_zh=True, size_zh=size_zh),
            )
        )
    return normalized


def render_ass(cues: list[BilingualCue], subtitle_config: SubtitleConfig | None = None) -> str:
    """Render bilingual cues into a styled ASS subtitle document."""
    cfg = subtitle_config or SubtitleConfig()
    zh_top = cfg.bilingual_order == "zh_top"
    use_box = cfg.style != "outline"

    if use_box:
        # BorderStyle=4 uses Shadow for unified card padding and BackColour for uniform opacity,
        # avoiding overlapping double-alpha dark seams between lines or wrapped text.
        zh_style = (
            f"Style: ZH,{cfg.font_zh},{cfg.size_zh},&H00FFFFFF,&H000000FF,&H33000000,&H66000000,"
            "-1,0,0,0,100,100,0,0,4,1.2,8,2,60,60,42,1"
        )
        en_style = (
            f"Style: EN,{cfg.font_en},{cfg.size_en},&H00B8E8FF,&H000000FF,&H33000000,&H66000000,"
            "0,0,0,0,100,100,0,0,4,1.2,8,2,60,60,42,1"
        )
    else:
        zh_style = (
            f"Style: ZH,{cfg.font_zh},{cfg.size_zh},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
            "-1,0,0,0,100,100,0,0,1,2.5,1.8,2,60,60,42,1"
        )
        en_style = (
            f"Style: EN,{cfg.font_en},{cfg.size_en},&H00B8E8FF,&H000000FF,&H00000000,&H80000000,"
            "0,0,0,0,100,100,0,0,1,2.5,1.8,2,60,60,42,1"
        )

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
            zh_style,
            en_style,
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )

    lines = [header]
    normalized = _normalize_cues(cues, size_zh=cfg.size_zh)

    if not normalized:
        lines.append("Dialogue: 0,0:00:00.00,0:00:05.00,ZH,,0,0,0,,双语字幕加载中...")
        return "\n".join(lines) + "\n"

    # Inline style overrides keep a single Dialogue style so BorderStyle=4 renders one unified box.
    inline_en = f"{{\\fn{cfg.font_en}\\fs{cfg.size_en}\\b0\\1c&H00B8E8FF&}}"
    inline_zh = f"{{\\fn{cfg.font_zh}\\fs{cfg.size_zh}\\b1\\1c&H00FFFFFF&}}"

    for cue in normalized:
        start_str = format_ass_timestamp(cue.start_ms)
        end_str = format_ass_timestamp(cue.end_ms)
        if cue.zh and cue.en:
            if zh_top:
                style_name = "ZH"
                body = f"{cue.zh}\\N{inline_en}{cue.en}"
            else:
                style_name = "EN"
                body = f"{cue.en}\\N{inline_zh}{cue.zh}"
        elif cue.zh:
            style_name = "ZH"
            body = cue.zh
        elif cue.en:
            style_name = "EN"
            body = cue.en
        else:
            continue
        lines.append(f"Dialogue: 0,{start_str},{end_str},{style_name},,0,0,0,,{body}")

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
