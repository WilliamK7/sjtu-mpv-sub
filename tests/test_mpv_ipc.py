"""Real integration test verifying mpv EDL timeline, separate audio EDL, chapters, and IPC sub-reload."""

from __future__ import annotations

import subprocess
import time
import uuid

import pytest

from sjtu_mpv_sub.config import load_config, resolve_tool_paths
from sjtu_mpv_sub.media import probe_audio_rms_db
from sjtu_mpv_sub.models import BilingualCue
from sjtu_mpv_sub.mpv_controller import (
    EdlSegment,
    MpvIpcClient,
    build_edl,
    get_hud_script_path,
    launch_mpv,
    write_edl,
)
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


def test_real_mpv_hud_script_seek_and_user_cancel_ipc(tmp_path) -> None:
    cfg = load_config(tmp_path)
    tools = resolve_tool_paths(cfg)
    mpv_exe = tools.get("mpv")
    ffmpeg_exe = tools.get("ffmpeg")
    if not mpv_exe or not ffmpeg_exe:
        pytest.skip("mpv or ffmpeg not available in test environment")

    vid = tmp_path / "sample.mp4"
    subprocess.run(
        [ffmpeg_exe, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=320x240:d=10", "-y", str(vid)],
        check=True,
    )
    edl_path = tmp_path / "session.edl"
    sub_ass = tmp_path / "bilingual.ass"
    write_edl(edl_path, build_edl([EdlSegment(url=str(vid), duration_sec=10.0, title="第1节")], include_chapters=True))
    write_ass_atomic(sub_ass, render_ass([]))

    hud_script = get_hud_script_path()
    assert hud_script.is_file()

    pipe_path = rf"\\.\pipe\sjtu-mpv-hud-test-{uuid.uuid4().hex[:10]}"
    proc = launch_mpv(
        mpv_path=mpv_exe,
        video_edl_path=edl_path,
        sub_path=sub_ass,
        pipe_path=pipe_path,
        media_title="HUD IPC Test",
        script_path=hud_script,
        extra_args=["--no-config", "--vo=null", "--ao=null", "--pause=yes", "--keep-open=yes"],
    )
    try:
        ipc = MpvIpcClient(pipe_path)
        assert ipc.wait_ready(timeout=12.0) is True

        assert ipc.get_time_pos() is not None
        assert ipc.seek_absolute(4.5) is True
        time.sleep(0.25)
        pos = ipc.get_time_pos()
        assert pos is not None and pos >= 3.5

        assert ipc.send_hud_state(
            {
                "phase": "transcribing",
                "total_sec": 10.0,
                "covered": [[4.0, 6.0]],
                "in_flight": [[6.0, 10.0]],
                "urgent": [4.0, 6.0],
                "covered_summary": "00:04–00:06",
                "pct": 33,
                "headline": "测试 HUD 状态",
            }
        ) is True
        assert ipc.clear_hud() is True

        assert ipc.consume_user_cancel_flag() is False
        set_resp = ipc.send_command(["set_property", "user-data/sjtu-sub/cancel", True])
        assert set_resp.get("error") == "success"
        assert ipc.consume_user_cancel_flag() is True

        quit_resp = ipc.send_command(["quit"])
        assert quit_resp.get("error") == "success"
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


def test_real_ffmpeg_probe_audio_rms_db(tmp_path) -> None:
    cfg = load_config(tmp_path)
    tools = resolve_tool_paths(cfg)
    ffmpeg_exe = tools.get("ffmpeg")
    if not ffmpeg_exe:
        pytest.skip("ffmpeg not available in test environment")

    audio_file = tmp_path / "silence_then_tone.mp4"
    subprocess.run(
        [
            ffmpeg_exe,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono:d=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=500:sample_rate=16000:duration=3",
            "-filter_complex",
            "[0:a][1:a]concat=n=2:v=0:a=1[a]",
            "-map",
            "[a]",
            "-c:a",
            "aac",
            "-y",
            str(audio_file),
        ],
        check=True,
    )

    silent_db = probe_audio_rms_db(str(audio_file), start_sec=0.5, probe_duration_sec=1.5, ffmpeg_path=ffmpeg_exe)
    active_db = probe_audio_rms_db(str(audio_file), start_sec=3.5, probe_duration_sec=1.5, ffmpeg_path=ffmpeg_exe)
    assert silent_db is not None and silent_db < -50.0
    assert active_db is not None and active_db > -25.0
