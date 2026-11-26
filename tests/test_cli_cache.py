"""Tests for raw asr.json cache persistence and rerender/retranslate CLI operations."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from sjtu_mpv_sub import pipeline as pipeline_mod
from sjtu_mpv_sub.cli import rerender_sessions, retranslate_session_async
from sjtu_mpv_sub.config import AppConfig, load_config
from sjtu_mpv_sub.media import ResolvedSegment
from sjtu_mpv_sub.models import AsrSentence, BilingualCue, PlayRequest, SegmentInput, StreamSource
from sjtu_mpv_sub.mpv_controller import MpvIpcClient
from sjtu_mpv_sub.pipeline import LecturePipeline
from sjtu_mpv_sub.translator import LLMTranslator


@pytest.mark.asyncio
async def test_pipeline_persists_asr_json_and_cli_rerender_retranslate(tmp_path, monkeypatch) -> None:
    (tmp_path / "config.yaml").write_text(
        """
dashscope:
  api_key: "sk-test"
pipeline:
  first_chunk_seconds: 100
  chunk_seconds: 100
  max_concurrency: 1
  translation_batch_size: 12
subtitle:
  style: "box"
  bilingual_order: "en_top"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "courses.yaml").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        pipeline_mod,
        "resolve_tool_paths",
        lambda _cfg: {"mpv": "mpv", "ffmpeg": "ffmpeg", "ffprobe": "ffprobe"},
    )

    async def fake_resolve(segment, preferred_view="ppt", referer="", ffprobe_path="ffprobe"):
        return ResolvedSegment(
            course_id=segment.course_id,
            title=segment.title,
            video_url=segment.streams[0].url,
            audio_url=segment.streams[0].url,
            separate_audio=False,
            duration_sec=90.0,
            ppt_slices=segment.ppt_slices,
        )

    monkeypatch.setattr(pipeline_mod, "resolve_segment_tracks_async", fake_resolve)

    async def fake_extract(audio_url, start_sec, duration_sec, output_audio, referer="", ffmpeg_path="ffmpeg", timeout=120.0):
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        output_audio.write_bytes(b"ID3")
        return output_audio

    monkeypatch.setattr(pipeline_mod, "extract_audio_chunk_async", fake_extract)

    class FakeASR:
        async def transcribe_chunk(self, audio_path, language_hints, **_kwargs):
            return [AsrSentence(start_ms=1000, end_ms=4000, text="Raw ASR eigenvalue line.")]

        async def close(self):
            return None

    class FakeTranslator:
        async def refine_and_translate(self, sentences, **_kwargs):
            return [
                BilingualCue(start_ms=s.start_ms, end_ms=s.end_ms, en="Refined eigenvalue line.", zh="特征值旧译文")
                for s in sentences
            ]

        async def close(self):
            return None

    class FakeProc:
        def poll(self):
            return None

    class FakeIpc(MpvIpcClient):
        def wait_ready(self, timeout=12.0):
            return True

        def get_time_pos(self):
            return 0.0

        def consume_user_cancel_flag(self):
            return False

        def reload_subtitles(self):
            return True

        def show_osd(self, text, duration_ms=3000):
            return True

        def send_hud_state(self, payload):
            return True

    cfg = load_config(tmp_path)
    pipe = LecturePipeline(
        config=cfg,
        root_dir=tmp_path,
        asr_factory=lambda _c: FakeASR(),
        translator_factory=lambda _c: FakeTranslator(),
        mpv_launcher=lambda **_kwargs: FakeProc(),
        ipc_factory=FakeIpc,
        start_detector=lambda *_a, **_kw: __import__("asyncio").sleep(0, result=0.0),
    )

    from sjtu_mpv_sub.models import JobStatus

    req = PlayRequest(
        tecl_id=777,
        course_name="Linear Algebra",
        segments=[
            SegmentInput(
                course_id=6001,
                title="第1节",
                streams=[StreamSource(view_num=5, url="https://live.sjtu.edu.cn/6001.mp4?key=1")],
            )
        ],
    )
    job = JobStatus(
        job_id="job1",
        session_key="tecl_777_6001",
        state="queued",
        message="",
        chunks_done=0,
        chunks_total=1,
        mpv_launched=False,
        cache_hit=False,
    )
    await pipe.run_job(job, req)
    assert job.state == "completed"

    cache_dir = tmp_path / "data" / "cache" / "tecl_777_6001"
    asr_items = json.loads((cache_dir / "asr.json").read_text(encoding="utf-8"))
    assert len(asr_items) == 1
    assert asr_items[0]["text"] == "Raw ASR eigenvalue line."

    # Test rerender switches style in bilingual.ass without touching cues.json.
    (tmp_path / "config.yaml").write_text(
        (tmp_path / "config.yaml").read_text(encoding="utf-8").replace('style: "box"', 'style: "outline"'),
        encoding="utf-8",
    )
    rendered = rerender_sessions(tmp_path, session_key="tecl_777_6001")
    assert rendered == [("tecl_777_6001", 1)]
    ass_after_rerender = (cache_dir / "bilingual.ass").read_text(encoding="utf-8-sig")
    assert ",1,2.5,1.8,2,60,60,42,1" in ass_after_rerender

    # Test retranslate uses raw asr.json (NOT Refined text) via LLMTranslator.
    seen_inputs: list[str] = []

    class FakeCompletions:
        async def create(self, **kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            seen_inputs.extend(seg["text"] for seg in payload["segments"])
            resp_json = json.dumps(
                {"items": [{"i": 0, "en": "Clean eigenvalue line.", "zh": "清晰的特征值新译文。"}]},
                ensure_ascii=False,
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=resp_json))])

    mock_translator = LLMTranslator(
        api_key="sk-test",
        base_url="http://localhost",
        client=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
    )
    count = await retranslate_session_async(tmp_path, "tecl_777_6001", translator=mock_translator)
    assert count == 1
    assert seen_inputs == ["Raw ASR eigenvalue line."]
    new_cues = json.loads((cache_dir / "cues.json").read_text(encoding="utf-8"))
    assert new_cues[0]["zh"] == "清晰的特征值新译文。"
    assert "清晰的特征值新译文。" in (cache_dir / "bilingual.ass").read_text(encoding="utf-8-sig")
