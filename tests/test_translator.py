"""Tests for PPT OCR context selection and LLM bilingual translation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sjtu_mpv_sub.models import AsrSentence, PptSlice
from sjtu_mpv_sub.translator import LLMTranslator, build_ppt_context, parse_llm_items


def test_build_ppt_context_includes_active_and_window_slices() -> None:
    slices = [
        PptSlice(seek_sec=10.0, ocr_text="Intro slide", keywords=["Intro"]),
        PptSlice(seek_sec=150.0, ocr_text="Eigenvalue decomposition Ax = lambda x", keywords=["Eigenvalue", "Matrix"]),
        PptSlice(seek_sec=240.0, ocr_text="Orthogonal diagonalization", keywords=["Matrix", "Orthogonal"]),
        PptSlice(seek_sec=900.0, ocr_text="Unrelated future chapter", keywords=["Future"]),
    ]
    ctx = build_ppt_context(slices, chunk_start_sec=180.0, chunk_end_sec=420.0)
    assert "Eigenvalue" in ctx
    assert "Orthogonal diagonalization" in ctx
    assert "Intro slide" not in ctx
    assert "Unrelated future chapter" not in ctx


def test_parse_llm_items_strips_fences() -> None:
    raw = """```json
    {"items": [{"i": 0, "en": "Clean text", "zh": "清晰文本"}]}
    ```"""
    parsed = parse_llm_items(raw)
    assert parsed == {0: ("Clean text", "清晰文本")}


@pytest.mark.asyncio
async def test_llm_translator_batches_and_falls_back_missing_indices() -> None:
    calls: list[dict] = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            batch_index = len(calls)
            if batch_index == 1:
                content = '{"items": [{"i": 0, "en": "Hello world.", "zh": "你好，世界。"}, {"i": 1, "en": "Linear algebra.", "zh": "线性代数。"}]}'
            else:
                # Simulate missing index 0 to test fallback.
                content = '{"items": []}'
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    translator = LLMTranslator(api_key="test", base_url="http://localhost", client=fake_client)

    sentences = [
        AsrSentence(start_ms=0, end_ms=1000, text="helo world"),
        AsrSentence(start_ms=1000, end_ms=2500, text="linear algerba"),
        AsrSentence(start_ms=2500, end_ms=4000, text="Fallback line"),
    ]
    cues = await translator.refine_and_translate(
        sentences,
        course_name="Linear Algebra",
        glossary=["Linear algebra"],
        ppt_context="PPT OCR: Eigenvalue",
        batch_size=2,
    )
    assert len(calls) == 2
    assert calls[0]["extra_body"] == {"enable_thinking": False}
    assert cues[0].en == "Hello world."
    assert cues[0].zh == "你好，世界。"
    assert cues[1].zh == "线性代数。"
    # Fallback keeps original ASR text instead of failing.
    assert cues[2].en == "Fallback line"
    assert cues[2].zh == "Fallback line"
