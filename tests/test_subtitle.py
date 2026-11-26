"""Tests for bilingual ASS subtitle rendering, styling modes, wrapping, and UTF-8-BOM output."""

from __future__ import annotations

from sjtu_mpv_sub.config import SubtitleConfig
from sjtu_mpv_sub.models import BilingualCue
from sjtu_mpv_sub.subtitle import format_ass_timestamp, render_ass, wrap_zh_for_ass, write_ass_atomic


def test_format_ass_timestamp() -> None:
    assert format_ass_timestamp(0) == "0:00:00.00"
    assert format_ass_timestamp(3723450) == "1:02:03.45"


def test_render_ass_unified_box_and_outline_styles_and_clamping() -> None:
    cues = [
        BilingualCue(start_ms=1000, end_ms=5000, en="Set {x | x > 0}", zh="集合 {x | x > 0}"),
        BilingualCue(start_ms=4200, end_ms=7000, en="Second line", zh="第二行"),
    ]
    ass_zh_top = render_ass(cues, SubtitleConfig(style="box", bilingual_order="zh_top"))
    assert ",4,1.2,8,2,60,60,42,1" in ass_zh_top
    assert "Dialogue: 0,0:00:01.00,0:00:04.20,ZH,,0,0,0,,集合\u00a0｛x\u00a0|\u00a0x\u00a0>\u00a00｝\\N{\\fnSegoe UI\\fs34\\b0\\1c&H00B8E8FF&}Set ｛x | x > 0｝" in ass_zh_top
    assert "\\rEN" not in ass_zh_top

    ass_en_top = render_ass(cues, SubtitleConfig(style="box", bilingual_order="en_top"))
    assert "Dialogue: 0,0:00:01.00,0:00:04.20,EN,,0,0,0,,Set ｛x | x > 0｝\\N{\\fnMicrosoft YaHei UI\\fs46\\b1\\1c&H00FFFFFF&}集合\u00a0｛x\u00a0|\u00a0x\u00a0>\u00a00｝" in ass_en_top

    ass_outline = render_ass(cues, SubtitleConfig(style="outline", bilingual_order="en_top"))
    assert ",1,2.5,1.8,2,60,60,42,1" in ass_outline


def test_render_ass_sanitizes_control_chars_markup_and_prevents_orphan_wrap() -> None:
    cues = [
        BilingualCue(
            start_ms=100,
            end_ms=2500,
            en="Delta, because it's **not** `$x$` convenient.",
            zh="关于 $\\boldsymbol{\\delta}$（相位常数）。但我们实际上绝不会以那种形式书写波动解，因为那样做在追踪所有相关因子等方面并不方便。",
        ),
        BilingualCue(
            start_ms=3000,
            end_ms=5000,
            en="Corrupt",
            zh="关于 $\x08oldsymbol{\x08eta}$（相位常数）。",
        ),
    ]
    ass = render_ass(cues, SubtitleConfig(bilingual_order="en_top", size_zh=46))
    assert "**" not in ass
    assert "`" not in ass
    assert "$" not in ass
    assert "\x08" not in ass
    assert "关于\u00a0δ（相位常数）。但我们实际上绝不会以那种形式书写波动解，\\N因为那样做在追踪所有相关因子等方面并不方便。" in ass


def test_wrap_zh_for_ass_avoids_short_orphan_first_line() -> None:
    wrapped = wrap_zh_for_ass(
        "关于 δ（相位常数）。但我们实际上绝不会以那种形式书写波动解，因为那样做在追踪所有相关因子等方面并不方便。",
        size_zh=46,
    )
    first_line = wrapped.split("\\N")[0]
    assert len(first_line) > 10
    assert " " not in wrapped


def test_render_ass_zh_only_cue_renders_single_line_without_en_override() -> None:
    ass = render_ass(
        [BilingualCue(start_ms=0, end_ms=2000, en="", zh="好的，我们开始上课。")],
        SubtitleConfig(bilingual_order="en_top"),
    )
    assert "Dialogue: 0,0:00:00.00,0:00:02.00,ZH,,0,0,0,,好的，我们开始上课。" in ass
    assert "\\N" not in ass.splitlines()[-1]
    assert "\\fnSegoe UI" not in ass.splitlines()[-1]


def test_render_ass_empty_cues_produces_placeholder() -> None:
    ass = render_ass([])
    assert "双语字幕加载中..." in ass


def test_write_ass_atomic_writes_utf8_bom(tmp_path) -> None:
    target = tmp_path / "bilingual.ass"
    write_ass_atomic(target, render_ass([BilingualCue(start_ms=0, end_ms=1500, en="Hi", zh="你好")]))
    raw = target.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert not target.with_suffix(".ass.tmp").exists()
