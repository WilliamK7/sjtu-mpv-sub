"""Tests for bilingual ASS subtitle rendering and atomic UTF-8-BOM output."""

from __future__ import annotations

from sjtu_mpv_sub.config import SubtitleConfig
from sjtu_mpv_sub.models import BilingualCue
from sjtu_mpv_sub.subtitle import format_ass_timestamp, render_ass, write_ass_atomic


def test_format_ass_timestamp() -> None:
    assert format_ass_timestamp(0) == "0:00:00.00"
    assert format_ass_timestamp(3723450) == "1:02:03.45"


def test_render_ass_orders_and_escapes_and_clamps_overlap() -> None:
    cues = [
        BilingualCue(start_ms=1000, end_ms=5000, en="Set {x | x > 0}", zh="集合 {x | x > 0}"),
        BilingualCue(start_ms=4200, end_ms=7000, en="Second line", zh="第二行"),
    ]
    ass_zh_top = render_ass(cues, SubtitleConfig(bilingual_order="zh_top"))
    assert "Dialogue: 0,0:00:01.00,0:00:04.20,ZH,,0,0,0,,集合 ｛x | x > 0｝\\N{\\rEN}Set ｛x | x > 0｝" in ass_zh_top
    assert "Dialogue: 0,0:00:04.20,0:00:07.00,ZH,,0,0,0,,第二行\\N{\\rEN}Second line" in ass_zh_top

    ass_en_top = render_ass(cues, SubtitleConfig(bilingual_order="en_top"))
    assert "Dialogue: 0,0:00:01.00,0:00:04.20,EN,,0,0,0,,Set ｛x | x > 0｝\\N{\\rZH}集合 ｛x | x > 0｝" in ass_en_top


def test_render_ass_empty_cues_produces_placeholder() -> None:
    ass = render_ass([])
    assert "双语字幕加载中..." in ass


def test_write_ass_atomic_writes_utf8_bom(tmp_path) -> None:
    target = tmp_path / "bilingual.ass"
    write_ass_atomic(target, render_ass([BilingualCue(start_ms=0, end_ms=1500, en="Hi", zh="你好")]))
    raw = target.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert not target.with_suffix(".ass.tmp").exists()
