"""Real integration test verifying mpv EDL timeline, separate audio EDL, chapters, and IPC sub-reload."""

from __future__ import annotations

import subprocess
import uuid

import pytest

from sjtu_mpv_sub.config import load_config, resolve_tool_paths
from sjtu_mpv_sub.models import BilingualCue
from sjtu_mpv_sub.mpv_controller import EdlSegment, MpvIpcClient, build_edl, launch_mpv, write_edl
from sjtu_mpv_sub.subtitle import render_ass, write_ass_atomic


def test_real_mpv_edl_chapters_audio_and_sub_reload(tmp_path) -> None:
    cfg = load_config(tmp_path)
    tools = resolve_tool_paths(cfg)
    mpv_exe = tools.get("mpv")
    ffmpeg_exe = tools.get("ffmpeg")
    if not mpv_exe or not ffmpeg_exe:
        pytest.skip("mpv or ffmpeg not available in test environment")

    v1 = tmp_path / "v1.mp4"
    v2 = tmp_path / "v2.mp4"
    a1 = tmp_path / "a1.mp4"
    a2 = tmp_path / "a2.mp4"

    subprocess.run([ffmpeg_exe, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=320x240:d=2", "-y", str(v1)], check=True)
    subprocess.run([ffmpeg_exe, "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2", "-y", str(v2)], check=True)
    subprocess.run([ffmpeg_exe, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", "-y", str(a1)], check=True)
    subprocess.run([ffmpeg_exe, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=660:duration=2", "-c:a", "aac", "-y", str(a2)], check=True)

    video_edl = tmp_path / "session.edl"
    audio_edl = tmp_path / "audio.edl"
    sub_ass = tmp_path / "bilingual.ass"

    write_edl(
        video_edl,
        build_edl(
            [
                EdlSegment(url=str(v1), duration_sec=2.0, title="第11节"),
                EdlSegment(url=str(v2), duration_sec=2.0, title="第12节"),
            ],
            include_chapters=True,
        ),
    )
    write_edl(
        audio_edl,
        build_edl(
            [
                EdlSegment(url=str(a1), duration_sec=2.0),
                EdlSegment(url=str(a2), duration_sec=2.0),
            ],
            include_chapters=False,
        ),
    )

    write_ass_atomic(
        sub_ass,
        render_ass([BilingualCue(start_ms=0, end_ms=1500, en="First cue", zh="第一条字幕")]),
    )

    pipe_path = rf"\\.\pipe\sjtu-mpv-test-{uuid.uuid4().hex[:10]}"
    proc = launch_mpv(
        mpv_path=mpv_exe,
        video_edl_path=video_edl,
        sub_path=sub_ass,
        pipe_path=pipe_path,
        media_title="Test Lecture",
        audio_edl_path=audio_edl,
        extra_args=["--no-config", "--vo=null", "--ao=null", "--pause=yes", "--keep-open=yes"],
    )
    try:
        ipc = MpvIpcClient(pipe_path)
        assert ipc.wait_ready(timeout=12.0) is True

        dur_resp = ipc.send_command(["get_property", "duration"])
        assert dur_resp.get("error") == "success"
        assert float(dur_resp["data"]) == pytest.approx(4.0, abs=0.2)

        chapters_resp = ipc.send_command(["get_property", "chapter-list"])
        assert chapters_resp.get("error") == "success"
        titles = [c.get("title") for c in chapters_resp["data"]]
        assert titles == ["第11节", "第12节"]

        # Hot-reload subtitle file while mpv is open.
        write_ass_atomic(
            sub_ass,
            render_ass(
                [
                    BilingualCue(start_ms=0, end_ms=1500, en="First cue", zh="第一条字幕"),
                    BilingualCue(start_ms=1600, end_ms=3500, en="Reloaded cue", zh="热重载字幕"),
                ]
            ),
        )
        assert ipc.reload_subtitles() is True

        quit_resp = ipc.send_command(["quit"])
        assert quit_resp.get("error") == "success"
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
