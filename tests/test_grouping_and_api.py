"""Tests for lecture chunk planning, smart start, seek preemption, incremental cache, and API."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from sjtu_mpv_sub import pipeline as pipeline_mod
from sjtu_mpv_sub.config import AppConfig, load_config
from sjtu_mpv_sub.media import ResolvedSegment, detect_lecture_start_async
from sjtu_mpv_sub.models import AsrSentence, BilingualCue, PlayRequest, SegmentInput, StreamSource
from sjtu_mpv_sub.pipeline import (
    LecturePipeline,
    build_chunk_plan,
    build_chunk_task_for_global_span,
    format_ranges_summary,
    format_timestamp_mmss,
    is_pos_covered,
    make_session_key,
    merge_ranges,
    subtract_ranges,
)
from sjtu_mpv_sub.server import create_app, should_idle_exit


class FakeProc:
    def __init__(self) -> None:
        self.exited = False

    def poll(self):
        return 0 if self.exited else None


class FakeIpcBase:
    def __init__(self, _pipe: str = "") -> None:
        self.time_pos: float | None = 0.0
        self.seeks: list[float] = []
        self.hud_states: list[dict] = []
        self.osd_messages: list[str] = []
        self.user_cancel: bool = False
        self.reload_calls: int = 0

    def wait_ready(self, timeout: float = 10.0) -> bool:
        return True

    def reload_subtitles(self) -> bool:
        self.reload_calls += 1
        return True

    def show_osd(self, text: str, duration_ms: int = 3000) -> bool:
        self.osd_messages.append(text)
        return True

    def get_time_pos(self) -> float | None:
        return self.time_pos

    def seek_absolute(self, seconds: float) -> bool:
        self.seeks.append(seconds)
        self.time_pos = seconds
        return True

    def send_hud_state(self, payload: dict) -> bool:
        self.hud_states.append(payload)
        return True

    def clear_hud(self) -> bool:
        return True

    def consume_user_cancel_flag(self) -> bool:
        return self.user_cancel


async def _zero_start_detector(*_args, **_kwargs) -> float:
    return 0.0


def _patch_fake_pipeline_media(monkeypatch, *, duration_sec: float) -> None:
    async def fake_resolve(segment, preferred_view="ppt", referer="", ffprobe_path="ffprobe"):
        return ResolvedSegment(
            course_id=segment.course_id,
            title=segment.title,
            video_url=segment.streams[0].url,
            audio_url=segment.streams[0].url,
            separate_audio=False,
            duration_sec=duration_sec,
            ppt_slices=segment.ppt_slices,
        )

    async def fake_extract(audio_url, start_sec, duration_sec, output_audio, referer="", ffmpeg_path="ffmpeg", timeout=120.0):
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        output_audio.write_bytes(b"ID3fake")
        return output_audio

    monkeypatch.setattr(pipeline_mod, "resolve_segment_tracks_async", fake_resolve)
    monkeypatch.setattr(pipeline_mod, "extract_audio_chunk_async", fake_extract)
    monkeypatch.setattr(
        pipeline_mod,
        "resolve_tool_paths",
        lambda _cfg: {"mpv": "mock-mpv.exe", "ffmpeg": "mock-ffmpeg", "ffprobe": "mock-ffprobe"},
    )


def _write_test_config(
    tmp_path: Path,
    *,
    max_concurrency: int = 1,
    urgent_burst_concurrency: int = 1,
) -> AppConfig:
    cfg = load_config(tmp_path)
    cfg.dashscope.api_key = "sk-test-key"
    cfg.pipeline.max_concurrency = max_concurrency
    cfg.pipeline.urgent_burst_concurrency = urgent_burst_concurrency
    cfg.pipeline.seek_poll_interval_seconds = 0.05
    cfg.pipeline.seek_debounce_seconds = 0.05
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return cfg


def test_chunk_plan_accumulates_period_offsets() -> None:
    resolved = [
        ResolvedSegment(
            course_id=1,
            title="P1",
            video_url="https://example.com/1.mp4",
            audio_url="https://example.com/1.mp4",
            separate_audio=False,
            duration_sec=500.0,
            ppt_slices=[],
        ),
        ResolvedSegment(
            course_id=2,
            title="P2",
            video_url="https://example.com/2.mp4",
            audio_url="https://example.com/2.mp4",
            separate_audio=False,
            duration_sec=300.0,
            ppt_slices=[],
        ),
    ]
    chunks = build_chunk_plan(resolved, first_chunk_seconds=180, chunk_seconds=420)
    assert len(chunks) == 3
    assert chunks[0].period_offset_ms == 0 and chunks[0].start_sec == 0.0 and chunks[0].duration_sec == 180.0
    assert chunks[1].period_offset_ms == 0 and chunks[1].start_sec == 180.0 and chunks[1].duration_sec == 320.0
    assert chunks[2].period_offset_ms == 500000 and chunks[2].start_sec == 0.0 and chunks[2].duration_sec == 300.0


def test_interval_helpers_and_segment_boundary_slicing() -> None:
    merged = merge_ranges([(1000, 3000), (3200, 5000), (8000, 9000)], gap_tolerance_ms=500)
    assert merged == [(1000, 5000), (8000, 9000)]

    uncovered = subtract_ranges([(0, 12000)], merged)
    assert uncovered == [(0, 1000), (5000, 8000), (9000, 12000)]

    assert is_pos_covered(2500, merged) is True
    assert is_pos_covered(6500, merged) is False
    assert format_timestamp_mmss(1460) == "24:20"
    assert format_timestamp_mmss(3665) == "1:01:05"
    assert format_ranges_summary([(1460_000, 2700_000)]) == "24:20–45:00"

    resolved = [
        ResolvedSegment(
            course_id=10,
            title="P1",
            video_url="v1",
            audio_url="a1",
            separate_audio=False,
            duration_sec=300.0,
            ppt_slices=[],
        ),
        ResolvedSegment(
            course_id=11,
            title="P2",
            video_url="v2",
            audio_url="a2",
            separate_audio=False,
            duration_sec=300.0,
            ppt_slices=[],
        ),
    ]
    offsets = [0, 300_000]
    task1 = build_chunk_task_for_global_span(250_000, 100_000, resolved, offsets, index=0)
    assert task1 is not None
    assert task1.segment_idx == 0
    assert task1.start_sec == 250.0
    assert task1.duration_sec == 50.0  # Clipped to segment 0 end, never crossing into segment 1!
    assert task1.global_end_ms == 300_000

    task2 = build_chunk_task_for_global_span(300_000, 100_000, resolved, offsets, index=1)
    assert task2 is not None
    assert task2.segment_idx == 1
    assert task2.period_offset_ms == 300_000
    assert task2.start_sec == 0.0
    assert task2.duration_sec == 100.0


async def test_detect_lecture_start_skips_leading_silence_and_noise_spikes() -> None:
    seg = ResolvedSegment(
        course_id=1,
        title="P1",
        video_url="u1",
        audio_url="a1",
        separate_audio=False,
        duration_sec=3600.0,
        ppt_slices=[],
    )

    # Lecture starts at 1470s (24:30), with a 1-second cough/noise spike at 360s.
    async def mock_probe(_url: str, start_sec: float, _dur: float) -> float | None:
        if 358.0 <= start_sec <= 362.0:
            return -20.0  # Isolated noise spike; +15s and +30s remain silent!
        if start_sec >= 1470.0:
            return -22.0
        return -60.0

    detected = await detect_lecture_start_async(
        [seg],
        max_scan_sec=2400.0,
        step_sec=120.0,
        silence_db=-38.0,
        lead_sec=10.0,
        probe_fn=mock_probe,
    )
    assert 1445.0 <= detected <= 1475.0

    # Active immediately at start -> returns 0.0.
    async def active_start_probe(_url: str, _start_sec: float, _dur: float) -> float | None:
        return -24.0

    assert await detect_lecture_start_async([seg], probe_fn=active_start_probe) == 0.0

    # All probes fail -> degrades safely to 0.0.
    async def broken_probe(_url: str, _start_sec: float, _dur: float) -> float | None:
        return None

    assert await detect_lecture_start_async([seg], probe_fn=broken_probe) == 0.0


def test_make_session_key_orders_chronologically() -> None:
    req = PlayRequest(
        tecl_id=99,
        course_name="Physics",
        segments=[
            SegmentInput(course_id=202, title="第2节", begin_time="2026-11-23 09:00:00", streams=[StreamSource(view_num=1, url="u2")]),
            SegmentInput(course_id=201, title="第1节", begin_time="2026-11-23 08:00:00", streams=[StreamSource(view_num=1, url="u1")]),
        ],
    )
    assert make_session_key(req) == "tecl_99_201_202"


def test_should_idle_exit_logic() -> None:
    assert should_idle_exit(active_jobs=0, alive_mpv_count=0, idle_seconds=305, idle_limit_seconds=300) is True
    assert should_idle_exit(active_jobs=1, alive_mpv_count=0, idle_seconds=400, idle_limit_seconds=300) is False
    assert should_idle_exit(active_jobs=0, alive_mpv_count=1, idle_seconds=400, idle_limit_seconds=300) is False
    assert should_idle_exit(active_jobs=0, alive_mpv_count=0, idle_seconds=120, idle_limit_seconds=300) is False


def test_api_courses_and_progressive_play_with_cache_hit(tmp_path, monkeypatch) -> None:
    cfg = _write_test_config(tmp_path, max_concurrency=3, urgent_burst_concurrency=1)
    _patch_fake_pipeline_media(monkeypatch, duration_sec=250.0)

    asr_calls = 0
    launch_calls: list[Path] = []
    ipcs: list[FakeIpcBase] = []

    class FakeASR:
        async def transcribe_chunk(self, audio_path, language_hints, **_kwargs):
            nonlocal asr_calls
            asr_calls += 1
            return [AsrSentence(start_ms=500, end_ms=2000, text=f"Chunk line {asr_calls}")]

        async def close(self):
            return None

    class FakeTranslator:
        async def refine_and_translate(self, sentences, **_kwargs):
            return [BilingualCue(start_ms=s.start_ms, end_ms=s.end_ms, en=s.text, zh=f"翻译:{s.text}") for s in sentences]

        async def close(self):
            return None

    def fake_launcher(**kwargs):
        launch_calls.append(kwargs["video_edl_path"])
        return FakeProc()

    def make_ipc(pipe):
        inst = FakeIpcBase(pipe)
        ipcs.append(inst)
        return inst

    def build_test_pipeline(config: AppConfig, root: Path, on_mpv_started):
        return LecturePipeline(
            config=config,
            root_dir=root,
            asr_factory=lambda _c: FakeASR(),
            translator_factory=lambda _c: FakeTranslator(),
            mpv_launcher=fake_launcher,
            ipc_factory=make_ipc,
            on_mpv_started=on_mpv_started,
            start_detector=_zero_start_detector,
        )

    app = create_app(root_dir=tmp_path, config=cfg, pipeline_factory=build_test_pipeline, ephemeral=False)
    with TestClient(app) as client:
        assert client.get("/health").json()["api_key_configured"] is True

        put_resp = client.put(
            "/api/courses/888",
            json={"name": "AI Systems", "preferred_view": "camera", "glossary": ["Transformer"]},
        )
        assert put_resp.status_code == 200
        assert put_resp.json()["preferred_view"] == "camera"
        assert client.get("/api/courses/888").json()["glossary"] == ["Transformer"]

        payload = {
            "tecl_id": 888,
            "course_name": "AI Systems",
            "segments": [
                {
                    "course_id": 5001,
                    "title": "第1节",
                    "begin_time": "2026-11-23 08:00:00",
                    "streams": [{"view_num": 5, "url": "https://live.sjtu.edu.cn/5001.mp4?key=first"}],
                }
            ],
        }
        job1 = client.post("/api/play", json=payload).json()
        for _ in range(40):
            status = client.get(f"/api/jobs/{job1['job_id']}").json()
            if status["state"] in ("completed", "failed"):
                break
            time.sleep(0.05)

        assert status["state"] == "completed"
        assert status["cache_hit"] is False
        assert status["chunks_total"] == 2
        assert status["chunks_done"] == 2
        assert asr_calls == 2
        assert ipcs[0].reload_calls >= 1
        assert len(launch_calls) == 1

        # Second call with new signed URL hits subtitle cache and skips ASR while refreshing EDL.
        payload["segments"][0]["streams"][0]["url"] = "https://live.sjtu.edu.cn/5001.mp4?key=refreshed"
        job2 = client.post("/api/play", json=payload).json()
        for _ in range(40):
            status2 = client.get(f"/api/jobs/{job2['job_id']}").json()
            if status2["state"] in ("completed", "failed"):
                break
            time.sleep(0.05)

        assert status2["state"] == "completed"
        assert status2["cache_hit"] is True
        assert asr_calls == 2  # No extra ASR calls!
        edl_text = (tmp_path / "data" / "cache" / "tecl_888_5001" / "session.edl").read_text(encoding="utf-8")
        assert "key=refreshed" in edl_text
        meta = json.loads((tmp_path / "data" / "cache" / "tecl_888_5001" / "meta.json").read_text(encoding="utf-8"))
        assert meta["complete"] is True
        assert meta["covered_ranges_ms"] == [[0, 250000]]


def test_pipeline_handles_silent_chunk_zero_and_reloads_on_chunk_one(tmp_path, monkeypatch) -> None:
    cfg = _write_test_config(tmp_path, max_concurrency=1, urgent_burst_concurrency=0)
    _patch_fake_pipeline_media(monkeypatch, duration_sec=250.0)

    launch_calls: list[Path] = []
    ipcs: list[FakeIpcBase] = []

    class SilentChunkZeroASR:
        async def transcribe_chunk(self, audio_path, language_hints, **_kwargs):
            if audio_path.name == "chunk_000.mp3":
                return []
            return [AsrSentence(start_ms=1000, end_ms=3000, text="Lecture begins now.")]

        async def close(self):
            return None

    class FakeTranslator:
        async def refine_and_translate(self, sentences, **_kwargs):
            return [BilingualCue(start_ms=s.start_ms, end_ms=s.end_ms, en=s.text, zh="课程开始了。") for s in sentences]

        async def close(self):
            return None

    def fake_launcher(**kwargs):
        launch_calls.append(kwargs["video_edl_path"])
        return FakeProc()

    def make_ipc(pipe):
        inst = FakeIpcBase(pipe)
        ipcs.append(inst)
        return inst

    def build_test_pipeline(config: AppConfig, root: Path, on_mpv_started):
        return LecturePipeline(
            config=config,
            root_dir=root,
            asr_factory=lambda _c: SilentChunkZeroASR(),
            translator_factory=lambda _c: FakeTranslator(),
            mpv_launcher=fake_launcher,
            ipc_factory=make_ipc,
            on_mpv_started=on_mpv_started,
            start_detector=_zero_start_detector,
        )

    app = create_app(root_dir=tmp_path, config=cfg, pipeline_factory=build_test_pipeline, ephemeral=False)
    with TestClient(app) as client:
        payload = {
            "tecl_id": 889,
            "course_name": "Physics II",
            "segments": [
                {
                    "course_id": 5002,
                    "title": "第1节",
                    "begin_time": "2026-11-23 08:00:00",
                    "streams": [{"view_num": 1, "url": "https://live.sjtu.edu.cn/5002.mp4?key=first"}],
                }
            ],
        }
        job = client.post("/api/play", json=payload).json()
        for _ in range(40):
            status = client.get(f"/api/jobs/{job['job_id']}").json()
            if status["state"] in ("completed", "failed"):
                break
            time.sleep(0.05)

        assert status["state"] == "completed"
        assert status["chunks_total"] == 2
        assert status["chunks_done"] == 2
        assert len(launch_calls) == 1
        assert ipcs[0].reload_calls == 1

        sub_text = (tmp_path / "data" / "cache" / "tecl_889_5002" / "bilingual.ass").read_text(encoding="utf-8-sig")
        assert "Lecture begins now." in sub_text
        assert "课程开始了。" in sub_text
        assert "双语字幕加载中" not in sub_text

        meta = json.loads((tmp_path / "data" / "cache" / "tecl_889_5002" / "meta.json").read_text(encoding="utf-8"))
        assert meta["complete"] is True


def test_pipeline_stops_background_chunks_and_preserves_incremental_cache_on_mpv_exit(tmp_path, monkeypatch) -> None:
    cfg = _write_test_config(tmp_path, max_concurrency=1, urgent_burst_concurrency=0)
    _patch_fake_pipeline_media(monkeypatch, duration_sec=700.0)

    seen_spans: list[tuple[float, float]] = []
    procs: list[FakeProc] = []
    extract_calls: list[float] = []

    async def tracking_extract(audio_url, start_sec, duration_sec, output_audio, referer="", ffmpeg_path="ffmpeg", timeout=120.0):
        extract_calls.append(start_sec)
        seen_spans.append((start_sec, duration_sec))
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        output_audio.write_bytes(b"ID3fake")
        return output_audio

    monkeypatch.setattr(pipeline_mod, "extract_audio_chunk_async", tracking_extract)

    class SlowChunkASR:
        async def transcribe_chunk(self, audio_path, language_hints, **_kwargs):
            if audio_path.name == "chunk_000.mp3":
                return [AsrSentence(start_ms=0, end_ms=1000, text="Intro line")]
            if procs:
                procs[-1].exited = True
            await asyncio.sleep(5.0)
            return [AsrSentence(start_ms=0, end_ms=1000, text="Should be canceled")]

        async def close(self):
            return None

    class FakeTranslator:
        async def refine_and_translate(self, sentences, **_kwargs):
            return [BilingualCue(start_ms=s.start_ms, end_ms=s.end_ms, en=s.text, zh="译文") for s in sentences]

        async def close(self):
            return None

    def fake_launcher(**_kwargs):
        proc = FakeProc()
        procs.append(proc)
        return proc

    def build_test_pipeline(config: AppConfig, root: Path, on_mpv_started):
        return LecturePipeline(
            config=config,
            root_dir=root,
            asr_factory=lambda _c: SlowChunkASR(),
            translator_factory=lambda _c: FakeTranslator(),
            mpv_launcher=fake_launcher,
            ipc_factory=FakeIpcBase,
            on_mpv_started=on_mpv_started,
            mpv_exit_poll_interval=0.02,
            start_detector=_zero_start_detector,
        )

    app = create_app(root_dir=tmp_path, config=cfg, pipeline_factory=build_test_pipeline, ephemeral=False)
    with TestClient(app) as client:
        payload = {
            "tecl_id": 890,
            "course_name": "Electromagnetism",
            "segments": [
                {
                    "course_id": 5003,
                    "title": "第1节",
                    "begin_time": "2026-11-23 08:00:00",
                    "streams": [{"view_num": 1, "url": "https://live.sjtu.edu.cn/5003.mp4?key=first"}],
                }
            ],
        }
        job1 = client.post("/api/play", json=payload).json()
        for _ in range(80):
            status = client.get(f"/api/jobs/{job1['job_id']}").json()
            if status["state"] in ("completed", "canceled", "failed"):
                break
            time.sleep(0.05)

        assert status["state"] == "canceled"
        assert len(seen_spans) == 2  # Chunk 0 (0-100s) finished, chunk 1 canceled, chunk 2 never started.
        meta_path = tmp_path / "data" / "cache" / "tecl_890_5003" / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["complete"] is False
        assert meta["covered_ranges_ms"] == [[0, 100000]]

        # Re-launching resumes from 100s without re-transcribing 0-100s!
        extract_calls.clear()
        job2 = client.post("/api/play", json=payload).json()
        assert job2["job_id"] != job1["job_id"]
        for _ in range(80):
            status2 = client.get(f"/api/jobs/{job2['job_id']}").json()
            if status2["state"] in ("completed", "canceled", "failed"):
                break
            time.sleep(0.05)
        assert status2["state"] == "canceled"
        assert extract_calls[0] == 100.0  # Skipped cached 0-100s!


def test_pipeline_instant_launch_auto_seeks_and_preempts_on_manual_seek(tmp_path, monkeypatch) -> None:
    cfg = _write_test_config(tmp_path, max_concurrency=1, urgent_burst_concurrency=1)
    _patch_fake_pipeline_media(monkeypatch, duration_sec=5400.0)

    events: list[str] = []
    extracted_spans: list[tuple[float, float]] = []
    ipcs: list[FakeIpcBase] = []

    async def fake_extract(audio_url, start_sec, duration_sec, output_audio, referer="", ffmpeg_path="ffmpeg", timeout=120.0):
        events.append(f"extract:{start_sec:.1f}:{duration_sec:.1f}")
        extracted_spans.append((start_sec, duration_sec))
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        output_audio.write_bytes(b"ID3fake")
        return output_audio

    monkeypatch.setattr(pipeline_mod, "extract_audio_chunk_async", fake_extract)

    async def detect_24_20(*_args, **_kwargs) -> float:
        events.append("detect_done")
        return 1460.0

    class SeekSimASR:
        def __init__(self) -> None:
            self.calls = 0

        async def transcribe_chunk(self, audio_path, language_hints, **_kwargs):
            self.calls += 1
            if self.calls == 2:
                # Simulate user jumping ahead to 1:10:00 (4200s) while chunk 1 is running!
                ipcs[0].time_pos = 4200.0
                await asyncio.sleep(0.18)
            return [AsrSentence(start_ms=500, end_ms=2000, text=f"Line {self.calls}")]

        async def close(self):
            return None

    class FakeTranslator:
        async def refine_and_translate(self, sentences, **_kwargs):
            return [BilingualCue(start_ms=s.start_ms, end_ms=s.end_ms, en=s.text, zh="译文") for s in sentences]

        async def close(self):
            return None

    def fake_launcher(**kwargs):
        events.append("mpv_launched")
        assert kwargs.get("script_path") is not None
        return FakeProc()

    def make_ipc(pipe):
        inst = FakeIpcBase(pipe)
        ipcs.append(inst)
        return inst

    def build_test_pipeline(config: AppConfig, root: Path, on_mpv_started):
        return LecturePipeline(
            config=config,
            root_dir=root,
            asr_factory=lambda _c: SeekSimASR(),
            translator_factory=lambda _c: FakeTranslator(),
            mpv_launcher=fake_launcher,
            ipc_factory=make_ipc,
            on_mpv_started=on_mpv_started,
            start_detector=detect_24_20,
        )

    app = create_app(root_dir=tmp_path, config=cfg, pipeline_factory=build_test_pipeline, ephemeral=False)
    with TestClient(app) as client:
        payload = {
            "tecl_id": 892,
            "course_name": "Advanced Calculus",
            "segments": [
                {
                    "course_id": 5005,
                    "title": "第1节",
                    "begin_time": "2026-11-23 08:00:00",
                    "streams": [{"view_num": 5, "url": "https://live.sjtu.edu.cn/5005.mp4?key=first"}],
                }
            ],
        }
        job = client.post("/api/play", json=payload).json()
        for _ in range(120):
            status = client.get(f"/api/jobs/{job['job_id']}").json()
            if status["state"] in ("completed", "failed"):
                break
            time.sleep(0.05)

        assert status["state"] == "completed"
        assert status["lecture_start_sec"] == 1460.0
        assert events[0] == "mpv_launched"  # mpv opened BEFORE detect & ASR!
        assert ipcs[0].seeks == [1460.0]

        # First chunk starts at 1460.0s (24:20) with urgent duration 100s, never touching 0..1460s!
        assert extracted_spans[0] == (1460.0, 100.0)
        assert all(start >= 1460.0 for start, _dur in extracted_spans)
        # Urgent preemption carved a 100s micro-chunk around 4200s (4198s..4298s)!
        assert any(4195.0 <= start <= 4205.0 and dur == 100.0 for start, dur in extracted_spans)

        # HUD received detecting, transcribing, and completed phases.
        phases = [h["phase"] for h in ipcs[0].hud_states]
        assert "detecting" in phases
        assert "transcribing" in phases
        assert phases[-1] == "completed"

        meta = json.loads((tmp_path / "data" / "cache" / "tecl_892_5005" / "meta.json").read_text(encoding="utf-8"))
        assert meta["complete"] is True
        assert meta["lecture_start_sec"] == 1460.0
        assert meta["covered_ranges_ms"] == [[1460000, 5400000]]


def test_pipeline_normal_playback_during_detect_does_not_cancel_auto_seek(tmp_path, monkeypatch) -> None:
    cfg = _write_test_config(tmp_path, max_concurrency=1, urgent_burst_concurrency=0)
    _patch_fake_pipeline_media(monkeypatch, duration_sec=3600.0)

    extracted_starts: list[float] = []
    ipcs: list[FakeIpcBase] = []

    async def tracking_extract(audio_url, start_sec, duration_sec, output_audio, referer="", ffmpeg_path="ffmpeg", timeout=120.0):
        extracted_starts.append(start_sec)
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        output_audio.write_bytes(b"ID3fake")
        return output_audio

    monkeypatch.setattr(pipeline_mod, "extract_audio_chunk_async", tracking_extract)

    class AdvancingIpc(FakeIpcBase):
        def __init__(self, pipe: str = "") -> None:
            super().__init__(pipe)
            self._reads = 0

        def get_time_pos(self) -> float | None:
            # Advance smoothly past 5s (0.0 -> 2.0 -> 4.0 -> 6.0 -> 8.0) during detection!
            if not self.seeks:
                val = min(10.0, self._reads * 2.0)
                self._reads += 1
                return val
            return self.time_pos

    async def slow_detector(*_args, **_kwargs) -> float:
        await asyncio.sleep(0.85)
        return 1580.0

    class FastASR:
        async def transcribe_chunk(self, audio_path, language_hints, **_kwargs):
            return [AsrSentence(start_ms=0, end_ms=1000, text="Lecture line")]

        async def close(self):
            return None

    class FakeTranslator:
        async def refine_and_translate(self, sentences, **_kwargs):
            return [BilingualCue(start_ms=s.start_ms, end_ms=s.end_ms, en=s.text, zh="译文") for s in sentences]

        async def close(self):
            return None

    def make_ipc(pipe):
        inst = AdvancingIpc(pipe)
        ipcs.append(inst)
        return inst

    def build_test_pipeline(config: AppConfig, root: Path, on_mpv_started):
        return LecturePipeline(
            config=config,
            root_dir=root,
            asr_factory=lambda _c: FastASR(),
            translator_factory=lambda _c: FakeTranslator(),
            mpv_launcher=lambda **_kwargs: FakeProc(),
            ipc_factory=make_ipc,
            on_mpv_started=on_mpv_started,
            start_detector=slow_detector,
        )

    app = create_app(root_dir=tmp_path, config=cfg, pipeline_factory=build_test_pipeline, ephemeral=False)
    with TestClient(app) as client:
        payload = {
            "tecl_id": 894,
            "course_name": "Linear Algebra",
            "segments": [
                {
                    "course_id": 5007,
                    "title": "第1节",
                    "begin_time": "2026-11-24 18:00:00",
                    "streams": [{"view_num": 5, "url": "https://live.sjtu.edu.cn/5007.mp4?key=first"}],
                }
            ],
        }
        job = client.post("/api/play", json=payload).json()
        for _ in range(120):
            status = client.get(f"/api/jobs/{job['job_id']}").json()
            if status["state"] in ("completed", "failed"):
                break
            time.sleep(0.05)

        assert status["state"] == "completed"
        assert ipcs[0].seeks == [1580.0]
        assert status["lecture_start_sec"] == 1580.0
        assert extracted_starts[0] == 1580.0


def test_pipeline_genuine_manual_seek_during_detect_cancels_auto_seek(tmp_path, monkeypatch) -> None:
    cfg = _write_test_config(tmp_path, max_concurrency=1, urgent_burst_concurrency=0)
    _patch_fake_pipeline_media(monkeypatch, duration_sec=3600.0)

    extracted_starts: list[float] = []
    ipcs: list[FakeIpcBase] = []
    detector_canceled = False

    async def tracking_extract(audio_url, start_sec, duration_sec, output_audio, referer="", ffmpeg_path="ffmpeg", timeout=120.0):
        extracted_starts.append(start_sec)
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        output_audio.write_bytes(b"ID3fake")
        return output_audio

    monkeypatch.setattr(pipeline_mod, "extract_audio_chunk_async", tracking_extract)

    class JumpingIpc(FakeIpcBase):
        def __init__(self, pipe: str = "") -> None:
            super().__init__(pipe)
            self._reads = 0

        def get_time_pos(self) -> float | None:
            self._reads += 1
            if self._reads <= 2:
                return 0.5
            self.time_pos = 900.0
            return 900.0

    async def slow_detector(*_args, **_kwargs) -> float:
        nonlocal detector_canceled
        try:
            await asyncio.sleep(5.0)
            return 1580.0
        except asyncio.CancelledError:
            detector_canceled = True
            raise

    class FastASR:
        async def transcribe_chunk(self, audio_path, language_hints, **_kwargs):
            return [AsrSentence(start_ms=0, end_ms=1000, text="Manual jump line")]

        async def close(self):
            return None

    class FakeTranslator:
        async def refine_and_translate(self, sentences, **_kwargs):
            return [BilingualCue(start_ms=s.start_ms, end_ms=s.end_ms, en=s.text, zh="译文") for s in sentences]

        async def close(self):
            return None

    def make_ipc(pipe):
        inst = JumpingIpc(pipe)
        ipcs.append(inst)
        return inst

    def build_test_pipeline(config: AppConfig, root: Path, on_mpv_started):
        return LecturePipeline(
            config=config,
            root_dir=root,
            asr_factory=lambda _c: FastASR(),
            translator_factory=lambda _c: FakeTranslator(),
            mpv_launcher=lambda **_kwargs: FakeProc(),
            ipc_factory=make_ipc,
            on_mpv_started=on_mpv_started,
            start_detector=slow_detector,
        )

    app = create_app(root_dir=tmp_path, config=cfg, pipeline_factory=build_test_pipeline, ephemeral=False)
    with TestClient(app) as client:
        payload = {
            "tecl_id": 895,
            "course_name": "Manual Seek Test",
            "segments": [
                {
                    "course_id": 5008,
                    "title": "第1节",
                    "begin_time": "2026-11-24 18:00:00",
                    "streams": [{"view_num": 5, "url": "https://live.sjtu.edu.cn/5008.mp4?key=first"}],
                }
            ],
        }
        job = client.post("/api/play", json=payload).json()
        for _ in range(120):
            status = client.get(f"/api/jobs/{job['job_id']}").json()
            if status["state"] in ("completed", "failed"):
                break
            time.sleep(0.05)

        assert status["state"] == "completed"
        assert detector_canceled is True
        assert ipcs[0].seeks == []
        assert status["lecture_start_sec"] == 900.0
        assert extracted_starts[0] == 900.0


def test_pipeline_respects_watch_later_and_mpv_user_cancel(tmp_path, monkeypatch) -> None:
    cfg = _write_test_config(tmp_path, max_concurrency=1, urgent_burst_concurrency=0)
    _patch_fake_pipeline_media(monkeypatch, duration_sec=3600.0)

    detect_called = False
    extracted_starts: list[float] = []
    ipcs: list[FakeIpcBase] = []

    async def tracking_extract(audio_url, start_sec, duration_sec, output_audio, referer="", ffmpeg_path="ffmpeg", timeout=120.0):
        extracted_starts.append(start_sec)
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        output_audio.write_bytes(b"ID3fake")
        return output_audio

    monkeypatch.setattr(pipeline_mod, "extract_audio_chunk_async", tracking_extract)

    async def fail_if_called_detector(*_args, **_kwargs) -> float:
        nonlocal detect_called
        detect_called = True
        return 0.0

    class CancelTriggerASR:
        def __init__(self) -> None:
            self.calls = 0

        async def transcribe_chunk(self, audio_path, language_hints, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return [AsrSentence(start_ms=0, end_ms=1000, text="Resumed line")]
            ipcs[0].user_cancel = True
            await asyncio.sleep(5.0)
            return []

        async def close(self):
            return None

    class FakeTranslator:
        async def refine_and_translate(self, sentences, **_kwargs):
            return [BilingualCue(start_ms=s.start_ms, end_ms=s.end_ms, en=s.text, zh="续播字幕") for s in sentences]

        async def close(self):
            return None

    def make_ipc(pipe):
        inst = FakeIpcBase(pipe)
        inst.time_pos = 1800.0  # Simulate mpv watch_later resuming at 30:00!
        ipcs.append(inst)
        return inst

    def build_test_pipeline(config: AppConfig, root: Path, on_mpv_started):
        return LecturePipeline(
            config=config,
            root_dir=root,
            asr_factory=lambda _c: CancelTriggerASR(),
            translator_factory=lambda _c: FakeTranslator(),
            mpv_launcher=lambda **_kwargs: FakeProc(),
            ipc_factory=make_ipc,
            on_mpv_started=on_mpv_started,
            start_detector=fail_if_called_detector,
        )

    app = create_app(root_dir=tmp_path, config=cfg, pipeline_factory=build_test_pipeline, ephemeral=False)
    with TestClient(app) as client:
        payload = {
            "tecl_id": 893,
            "course_name": "Quantum Mechanics",
            "segments": [
                {
                    "course_id": 5006,
                    "title": "第1节",
                    "begin_time": "2026-11-23 08:00:00",
                    "streams": [{"view_num": 1, "url": "https://live.sjtu.edu.cn/5006.mp4?key=first"}],
                }
            ],
        }
        job = client.post("/api/play", json=payload).json()
        for _ in range(80):
            status = client.get(f"/api/jobs/{job['job_id']}").json()
            if status["state"] in ("completed", "canceled", "failed"):
                break
            time.sleep(0.05)

        assert detect_called is False
        assert ipcs[0].seeks == []
        assert extracted_starts[0] == 1800.0
        assert status["state"] == "canceled"
        assert ipcs[0].hud_states[-1]["phase"] == "canceled"

        meta = json.loads((tmp_path / "data" / "cache" / "tecl_893_5006" / "meta.json").read_text(encoding="utf-8"))
        assert meta["complete"] is False
        assert meta["covered_ranges_ms"] == [[1800000, 1900000]]


def test_api_cancel_job_endpoint(tmp_path, monkeypatch) -> None:
    cfg = _write_test_config(tmp_path, max_concurrency=1, urgent_burst_concurrency=0)
    _patch_fake_pipeline_media(monkeypatch, duration_sec=250.0)

    asr_started = threading.Event()

    class BlockingASR:
        async def transcribe_chunk(self, audio_path, language_hints, **_kwargs):
            asr_started.set()
            await asyncio.sleep(10.0)
            return []

        async def close(self):
            return None

    class FakeTranslator:
        async def refine_and_translate(self, sentences, **_kwargs):
            return []

        async def close(self):
            return None

    def build_test_pipeline(config: AppConfig, root: Path, on_mpv_started):
        return LecturePipeline(
            config=config,
            root_dir=root,
            asr_factory=lambda _c: BlockingASR(),
            translator_factory=lambda _c: FakeTranslator(),
            mpv_launcher=lambda **_kwargs: FakeProc(),
            ipc_factory=FakeIpcBase,
            on_mpv_started=on_mpv_started,
            mpv_exit_poll_interval=0.02,
            start_detector=_zero_start_detector,
        )

    app = create_app(root_dir=tmp_path, config=cfg, pipeline_factory=build_test_pipeline, ephemeral=False)
    with TestClient(app) as client:
        payload = {
            "tecl_id": 891,
            "course_name": "Cancel Test",
            "segments": [
                {
                    "course_id": 5004,
                    "title": "第1节",
                    "begin_time": "2026-11-23 08:00:00",
                    "streams": [{"view_num": 1, "url": "https://live.sjtu.edu.cn/5004.mp4?key=first"}],
                }
            ],
        }
        job = client.post("/api/play", json=payload).json()
        assert asr_started.wait(timeout=3.0)

        cancel_resp = client.post(f"/api/jobs/{job['job_id']}/cancel")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["state"] == "canceled"

        for _ in range(30):
            health = client.get("/health").json()
            if health["active_jobs"] == 0:
                break
            time.sleep(0.05)
        assert client.get("/health").json()["active_jobs"] == 0
        assert client.get(f"/api/jobs/{job['job_id']}").json()["state"] == "canceled"
