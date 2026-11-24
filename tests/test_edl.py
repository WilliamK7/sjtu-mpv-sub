"""Tests for EDL playlist formatting and silent-PPT audio stream resolution."""

from __future__ import annotations

from sjtu_mpv_sub import media
from sjtu_mpv_sub.media import StreamProbe
from sjtu_mpv_sub.models import SegmentInput, StreamSource
from sjtu_mpv_sub.mpv_controller import EdlSegment, build_edl, quote_edl_url, write_edl


def test_quote_edl_url_handles_query_commas_and_utf8() -> None:
    url = "https://live.sjtu.edu.cn:443/mnt/record6/file.mp4?key=abc,123&课程=1"
    quoted = quote_edl_url(url)
    expected_len = len(url.encode("utf-8"))
    assert quoted == f"%{expected_len}%{url}"


def test_build_edl_chapters_and_strict_lf(tmp_path) -> None:
    segs = [
        EdlSegment(
            url="https://live.sjtu.edu.cn:443/mnt/record6/1.mp4?key=aaa,bbb",
            duration_sec=2700.5,
            title="2026-11-23 第11节, 上半场; 测试",
        ),
        EdlSegment(
            url="https://live.sjtu.edu.cn:443/mnt/record6/2.mp4?key=ccc&x=1",
            duration_sec=2700.0,
            title="2026-11-23 第12节",
        ),
    ]
    edl = build_edl(segs, include_chapters=True)
    assert "\r" not in edl
    assert "!no_chapters" not in edl
    lines = edl.strip().split("\n")
    assert lines[0] == "# mpv EDL v0"
    assert ",0,2700.500,title=2026-11-23 第11节 上半场 测试" in lines[1]
    assert ",0,2700.000,title=2026-11-23 第12节" in lines[2]

    out = tmp_path / "video.edl"
    write_edl(out, edl)
    assert b"\r\n" not in out.read_bytes()


def test_build_audio_edl_uses_no_chapters() -> None:
    segs = [EdlSegment(url="https://example.com/a1.mp4?key=1", duration_sec=60.0, title="Ignored")]
    edl = build_edl(segs, include_chapters=False)
    lines = edl.strip().split("\n")
    assert lines[0] == "# mpv EDL v0"
    assert lines[1] == "!no_chapters"
    assert "title=" not in lines[2]


def test_resolve_segment_attaches_camera_audio_for_silent_ppt(monkeypatch) -> None:
    def fake_probe(url: str, referer: str = "", ffprobe_path: str = "ffprobe", timeout: float = 20.0) -> StreamProbe:
        if "ppt" in url:
            return StreamProbe(duration=2705.2, has_video=True, has_audio=False)
        return StreamProbe(duration=2705.4, has_video=True, has_audio=True)

    monkeypatch.setattr(media, "probe_stream", fake_probe)

    segment = SegmentInput(
        course_id=1001,
        title="2026-11-23 第11节",
        streams=[
            StreamSource(view_num=1, url="https://live.sjtu.edu.cn/cam.mp4?key=c1", vod_time=2705.0),
            StreamSource(view_num=5, url="https://live.sjtu.edu.cn/ppt.mp4?key=p1", vod_time=2705.0),
        ],
    )
    resolved = media.resolve_segment_tracks(segment, preferred_view="ppt")
    assert resolved.video_url.endswith("key=p1")
    assert resolved.audio_url.endswith("key=c1")
    assert resolved.separate_audio is True
    assert resolved.duration_sec == 2705.2


def test_resolve_segment_selects_view_num_1_for_camera_and_5_for_ppt(monkeypatch) -> None:
    def fake_probe(url: str, referer: str = "", ffprobe_path: str = "ffprobe", timeout: float = 20.0) -> StreamProbe:
        return StreamProbe(duration=1800.0, has_video=True, has_audio=True)

    monkeypatch.setattr(media, "probe_stream", fake_probe)

    segment = SegmentInput(
        course_id=1002,
        title="2026-11-23 第12节",
        streams=[
            StreamSource(view_num=5, url="https://live.sjtu.edu.cn/ppt.mp4?key=p2", vod_time=1800.0),
            StreamSource(view_num=1, url="https://live.sjtu.edu.cn/cam.mp4?key=c2", vod_time=1800.0),
        ],
    )
    resolved_cam = media.resolve_segment_tracks(segment, preferred_view="camera")
    assert resolved_cam.video_url.endswith("key=c2")
    assert resolved_cam.separate_audio is False

    resolved_ppt = media.resolve_segment_tracks(segment, preferred_view="ppt")
    assert resolved_ppt.video_url.endswith("key=p2")
    assert resolved_ppt.separate_audio is False
