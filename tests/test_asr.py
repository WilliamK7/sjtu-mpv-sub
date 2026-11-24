"""Tests for Qwen-Audio 3.1 Filetrans RESTful ASR, MP3 extraction, and NO_PROXY safety."""

from __future__ import annotations

import json
import subprocess

import httpx
import pytest

from sjtu_mpv_sub.asr import (
    QwenAudioFiletransASR,
    build_asr_context_messages,
    build_instant_vocabulary,
)
from sjtu_mpv_sub.config import load_config, resolve_tool_paths
from sjtu_mpv_sub.media import extract_audio_chunk
from sjtu_mpv_sub.server import is_server_already_running
from sjtu_mpv_sub.translator import LLMTranslator


def test_build_instant_vocabulary_enforces_official_rules() -> None:
    glossary = [
        "Eigenvalue = 特征值",
        "Kirchhoff's Law: 基尔霍夫定律",
        "This English sentence has strictly more than seven separate whitespace words",
        "这是一个长度明显超过了十五个汉字限制的超级长术语词条",
        "A",  # too short (<2)
        "CMOS",
    ]
    ppt_keywords = ["CMOS", "Transformer", "矩阵乘法"]
    vocab = build_instant_vocabulary(glossary, ppt_keywords)
    assert vocab == {
        "Eigenvalue": 4,
        "Kirchhoff's Law": 4,
        "CMOS": 4,
        "Transformer": 4,
        "矩阵乘法": 4,
    }


def test_build_asr_context_messages_truncates_to_400() -> None:
    assert build_asr_context_messages("", "") == []
    msgs = build_asr_context_messages("Linear Algebra", "A " * 300)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    text = msgs[0]["content"][0]["text"]
    assert len(text) <= 400
    assert text.startswith("Course: Linear Algebra. Slides:")


@pytest.mark.asyncio
async def test_qwen_audio_filetrans_full_4_step_flow(tmp_path) -> None:
    audio_file = tmp_path / "chunk_000.mp3"
    audio_file.write_bytes(b"ID3fake-mp3-bytes")

    seen_calls: list[dict] = []
    poll_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal poll_count
        url = str(request.url)
        seen_calls.append({"method": request.method, "url": url, "headers": dict(request.headers)})

        if request.method == "GET" and "/uploads" in url:
            assert request.headers.get("authorization") == "Bearer sk-test-key"
            assert request.url.params.get("action") == "getPolicy"
            assert request.url.params.get("model") == "qwen-audio-3.1-asr-flash-filetrans"
            return httpx.Response(
                200,
                json={
                    "data": {
                        "policy": "pol",
                        "signature": "sig",
                        "upload_dir": "dashscope-instant/u1/2026-11-23/d1",
                        "upload_host": "https://upload-test.oss-cn-beijing.aliyuncs.com",
                        "oss_access_key_id": "ak123",
                        "x_oss_object_acl": "private",
                        "x_oss_forbid_overwrite": "true",
                    }
                },
            )

        if request.method == "POST" and "upload-test.oss-cn-beijing.aliyuncs.com" in url:
            body_bytes = request.content
            assert b"dashscope-instant/u1/2026-11-23/d1/" in body_bytes
            assert b"_chunk_000.mp3" in body_bytes
            return httpx.Response(200, text="")

        if request.method == "POST" and url.endswith("/services/audio/asr/transcription"):
            assert request.headers.get("x-dashscope-async") == "enable"
            assert request.headers.get("x-dashscope-ossresourceresolve") == "enable"
            body = json.loads(request.content.decode("utf-8"))
            assert body["model"] == "qwen-audio-3.1-asr-flash-filetrans"
            assert body["input"]["file_urls"][0].startswith("oss://dashscope-instant/u1/2026-11-23/d1/")
            assert body["input"]["context"][0]["role"] == "user"
            assert body["parameters"]["channel_id"] == [0]
            assert body["parameters"]["vocabulary"] == {"Transformer": 4}
            assert body["parameters"]["language_hints"] == ["en", "zh"]
            return httpx.Response(200, json={"output": {"task_id": "task-abc", "task_status": "PENDING"}})

        if request.method == "GET" and url.endswith("/tasks/task-abc"):
            poll_count += 1
            if poll_count == 1:
                return httpx.Response(200, json={"output": {"task_id": "task-abc", "task_status": "RUNNING"}})
            return httpx.Response(
                200,
                json={
                    "output": {
                        "task_id": "task-abc",
                        "task_status": "SUCCEEDED",
                        "results": [
                            {
                                "subtask_status": "SUCCEEDED",
                                "transcription_url": "https://result-oss.example.com/transcript.json?Expires=999",
                            }
                        ],
                    }
                },
            )

        if request.method == "GET" and "result-oss.example.com" in url:
            assert "authorization" not in request.headers
            return httpx.Response(
                200,
                json={
                    "transcripts": [
                        {
                            "channel_id": 0,
                            "sentences": [
                                {"begin_time": 120, "end_time": 1850, "text": "Today we study Transformer."},
                                {"begin_time": 2000, "end_time": 4200, "text": "注意这里的矩阵乘法。"},
                            ],
                        }
                    ]
                },
            )

        raise AssertionError(f"Unexpected request: {request.method} {url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        asr = QwenAudioFiletransASR(
            api_key="sk-test-key",
            asr_api_base_url="https://ws-demo.cn-beijing.maas.aliyuncs.com/api/v1",
            poll_interval_seconds=0.05,
            client=client,
        )
        sentences = await asr.transcribe_chunk(
            audio_file,
            ["en", "zh"],
            hotwords={"Transformer": 4},
            context_messages=build_asr_context_messages("LLM Course", "Attention is all you need"),
        )

    assert len(sentences) == 2
    assert sentences[0].start_ms == 120
    assert sentences[0].end_ms == 1850
    assert sentences[0].text == "Today we study Transformer."
    assert sentences[1].text == "注意这里的矩阵乘法。"
    assert poll_count == 2


async def _run_silent_filetrans_case(tmp_path, task_id: str, task_payload: dict) -> tuple[list, dict[str, int]]:
    audio_file = tmp_path / "chunk_000.mp3"
    audio_file.write_bytes(b"ID3silent-mp3-bytes")
    counts = {"uploads": 0, "submits": 0, "polls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "GET" and "/uploads" in url:
            counts["uploads"] += 1
            return httpx.Response(
                200,
                json={
                    "data": {
                        "policy": "pol",
                        "signature": "sig",
                        "upload_dir": "dashscope-instant/u1/2026-11-24/d1",
                        "upload_host": "https://upload-test.oss-cn-beijing.aliyuncs.com",
                        "oss_access_key_id": "ak123",
                        "x_oss_object_acl": "private",
                        "x_oss_forbid_overwrite": "true",
                    }
                },
            )
        if request.method == "POST" and "upload-test.oss-cn-beijing.aliyuncs.com" in url:
            return httpx.Response(200, text="")
        if request.method == "POST" and url.endswith("/services/audio/asr/transcription"):
            counts["submits"] += 1
            return httpx.Response(200, json={"output": {"task_id": task_id, "task_status": "PENDING"}})
        if request.method == "GET" and url.endswith(f"/tasks/{task_id}"):
            counts["polls"] += 1
            return httpx.Response(200, json=task_payload)
        raise AssertionError(f"Unexpected request: {request.method} {url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        asr = QwenAudioFiletransASR(
            api_key="sk-test-key",
            asr_api_base_url="https://ws-demo.cn-beijing.maas.aliyuncs.com/api/v1",
            poll_interval_seconds=0.05,
            client=client,
        )
        sentences = await asr.transcribe_chunk(audio_file, ["en", "zh"])
    return sentences, counts


@pytest.mark.asyncio
async def test_qwen_audio_filetrans_treats_no_words_top_level_failed_as_silent_without_retry(tmp_path) -> None:
    payload = {
        "request_id": "req-silent-top",
        "model": "qwen-audio-3.1-asr-flash-filetrans",
        "output": {
            "task_id": "task-silent-top",
            "task_status": "FAILED",
            "code": "ASR_RESPONSE_HAVE_NO_WORDS",
            "message": "ASR_RESPONSE_HAVE_NO_WORDS",
            "subtask_status": "FAILED",
            "results": [{"subtask_status": "FAILED", "results": []}],
        },
        "usage": {},
    }
    sentences, counts = await _run_silent_filetrans_case(tmp_path, "task-silent-top", payload)
    assert sentences == []
    assert counts == {"uploads": 1, "submits": 1, "polls": 1}


@pytest.mark.asyncio
async def test_qwen_audio_filetrans_treats_no_words_subtask_failed_as_silent_without_retry(tmp_path) -> None:
    payload = {
        "request_id": "req-silent-sub",
        "model": "qwen-audio-3.1-asr-flash-filetrans",
        "output": {
            "task_id": "task-silent-sub",
            "task_status": "SUCCEEDED",
            "results": [
                {
                    "subtask_status": "FAILED",
                    "code": "ASR_RESPONSE_HAVE_NO_WORDS",
                    "message": "ASR_RESPONSE_HAVE_NO_WORDS",
                }
            ],
        },
        "usage": {},
    }
    sentences, counts = await _run_silent_filetrans_case(tmp_path, "task-silent-sub", payload)
    assert sentences == []
    assert counts == {"uploads": 1, "submits": 1, "polls": 1}


def test_extract_audio_chunk_produces_valid_mp3(tmp_path) -> None:
    cfg = load_config(tmp_path)
    tools = resolve_tool_paths(cfg)
    ffmpeg_exe = tools.get("ffmpeg")
    ffprobe_exe = tools.get("ffprobe")
    if not ffmpeg_exe or not ffprobe_exe:
        pytest.skip("ffmpeg/ffprobe not available")

    src_mp4 = tmp_path / "src.mp4"
    out_mp3 = tmp_path / "chunk_000.mp3"
    subprocess.run(
        [ffmpeg_exe, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5", "-c:a", "aac", "-y", str(src_mp4)],
        check=True,
    )
    extract_audio_chunk(str(src_mp4), 0.0, 1.0, out_mp3, ffmpeg_path=ffmpeg_exe)
    assert out_mp3.is_file() and out_mp3.stat().st_size > 100

    probe = subprocess.run(
        [ffprobe_exe, "-v", "error", "-show_streams", "-of", "json", str(out_mp3)],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    assert streams[0]["codec_name"] == "mp3"
    assert int(streams[0]["sample_rate"]) == 16000
    assert int(streams[0]["channels"]) == 1


@pytest.mark.asyncio
async def test_no_proxy_ipv6_bracket_does_not_crash_clients(monkeypatch) -> None:
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1,::1,[::1]")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:10808")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:10808")

    asr = QwenAudioFiletransASR(api_key="sk-test")
    translator = LLMTranslator(
        api_key="sk-test",
        base_url="https://ws-demo.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
    try:
        assert is_server_already_running("127.0.0.1", 19999) is False
    finally:
        await asr.close()
        await translator.close()
