"""Tests for PPT OCR context selection, subtitle text sanitizer, and hardened LLM translation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sjtu_mpv_sub.models import AsrSentence, PptSlice
from sjtu_mpv_sub.translator import (
    LLMTranslator,
    build_ppt_context,
    is_chinese_spoken,
    parse_llm_items,
    sanitize_subtitle_text,
    validate_translated_batch,
)


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


def test_parse_llm_items_strips_fences_and_repairs_latex_escapes() -> None:
    raw = """```json
    {"items": [{"i": 0, "en": "Clean text", "zh": "$\\lambda^2$ 特征值"}]}
    ```"""
    parsed = parse_llm_items(raw)
    assert parsed == {0: ("Clean text", r"$\lambda^2$ 特征值")}


def test_sanitize_subtitle_text_cleans_latex_markdown_and_control_chars() -> None:
    assert sanitize_subtitle_text("即 $\\lambda^2 + \\frac{R}{L}\\lambda + \\frac{1}{LC} = 0$。", is_zh=True) == "即 λ² + R/Lλ + 1/(LC) = 0。"
    assert sanitize_subtitle_text("求导 $\\frac{d}{dt} x$。") == "求导 d/dt x。"
    assert sanitize_subtitle_text("所以是 $\\\\\\text{det}(\\\\lambda I - A)$。", is_zh=True) == "所以是 det(λ I - A)。"
    assert sanitize_subtitle_text("Write **bold** and `code` with √(4L/C).") == "Write bold and code with √(4L/C)."
    assert sanitize_subtitle_text("关于 $\x08oldsymbol{\x08eta}$（相位）。", is_zh=True) == "关于 oldsymboleta（相位）。"
    assert sanitize_subtitle_text("Set {x | x > 0} and \\{1, 2\\}") == "Set {x | x > 0} and {1, 2}"


def test_validate_translated_batch_catches_defects_and_ignores_cjk_src_for_drift() -> None:
    batch = [
        AsrSentence(start_ms=0, end_ms=1000, text="We have real and repeated roots"),
        AsrSentence(start_ms=1000, end_ms=2000, text="of the characteristic equation."),
    ]
    assert any("punctuation_stub" in r for r in validate_translated_batch(batch, {0: ("We have real and repeated roots", "特征方程有实数重根"), 1: ("of the characteristic equation.", "。")}))
    assert any("meta_or_cot_leak" in r for r in validate_translated_batch(batch[:1], {0: ("We have real and repeated roots", "我们有实数重根。（注：此处可能为口误）")}))
    assert any("editorial_placeholder" in r for r in validate_translated_batch(batch[:1], {0: ("[Background noise/Unrelated conversation]", "[背景噪音/无关对话]")}))

    # CJK / code-switched ASR input must NOT false-positive on English drift check.
    cjk_batch = [AsrSentence(start_ms=0, end_ms=2000, text="这个 matrix 的 eigenvalue 是多少")]
    assert validate_translated_batch(cjk_batch, {0: ("What is the eigenvalue of this matrix?", "这个矩阵的特征值是多少？")}) == []


def test_validate_translated_batch_catches_hallucination_collapse_and_single_severe_drift() -> None:
    short_batch = [AsrSentence(start_ms=0, end_ms=1500, text="OK.")]
    assert any(
        "en_hallucination@0" in r
        for r in validate_translated_batch(
            short_batch,
            {
                0: (
                    "I mean it is completely normal if you think about plugging in the matrix to get zero.",
                    "我的意思是如果代入矩阵得到零这很正常。",
                )
            },
        )
    )

    long_batch = [
        AsrSentence(
            start_ms=0,
            end_ms=4000,
            text="Notice that the minimal polynomial must divide the characteristic polynomial of this matrix.",
        )
    ]
    assert any(
        "en_collapse@0" in r
        for r in validate_translated_batch(long_batch, {0: ("Okay.", "好的。")})
    )

    mixed_batch = [
        AsrSentence(
            start_ms=0,
            end_ms=3000,
            text="Now let us calculate the eigenvalues and eigenvectors of the given three by three matrix.",
        ),
        AsrSentence(start_ms=3000, end_ms=5000, text="So therefore all Jordan blocks have size one."),
    ]
    assert any(
        "en_severe_drift@0" in r
        for r in validate_translated_batch(
            mixed_batch,
            {
                0: (
                    "Please remember to review the second chapter before Friday morning homework.",
                    "请记得在周五早交作业前复习第二章。",
                ),
                1: ("So therefore all Jordan blocks have size one.", "因此所有若尔当块大小为 1。"),
            },
        )
    )

    # Legitimate ASR homophone refinement on a spoken technical line must pass cleanly.
    assert (
        validate_translated_batch(
            [AsrSentence(start_ms=0, end_ms=3000, text="Here we use the Kelly Hamilton theorem to prove the zero matrix.")],
            {0: ("Here we use the Cayley-Hamilton theorem to prove the zero matrix.", "这里我们用凯莱-哈密顿定理证明零矩阵。")},
        )
        == []
    )


@pytest.mark.asyncio
async def test_llm_translator_retries_and_bisects_on_missing_or_shifted_indices() -> None:
    calls: list[dict] = []

    class FakeCompletions:
        async def create(self, **kwargs):
            import json

            calls.append(kwargs)
            payload = json.loads(kwargs["messages"][1]["content"])
            segs = payload["segments"]
            if len(segs) == 4:
                # Both full-batch attempts fail with drifted/missing indices.
                content = '{"items": [{"i": 0, "en": "Completely wrong shifted alpha line one", "zh": "错误1"}, {"i": 1, "en": "Completely wrong shifted beta line two", "zh": "错误2"}, {"i": 2, "en": "Completely wrong shifted gamma line three", "zh": "错误3"}]}'
            else:
                items = [{"i": s["i"], "en": s["text"], "zh": f"译文{s['i']}:{s['text'][:6]}"} for s in segs]
                content = json.dumps({"items": items}, ensure_ascii=False)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    translator = LLMTranslator(api_key="test", base_url="http://localhost", client=fake_client)

    sentences = [
        AsrSentence(start_ms=0, end_ms=1000, text="Now consider the impedance of the circuit."),
        AsrSentence(start_ms=1000, end_ms=2000, text="Notice how inductors oppose rapid current changes."),
        AsrSentence(start_ms=2000, end_ms=3000, text="Meanwhile capacitors resist voltage swings across plates."),
        AsrSentence(start_ms=3000, end_ms=4000, text="Together they determine the resonant frequency."),
    ]
    cues = await translator.refine_and_translate(
        sentences,
        course_name="General Physics",
        glossary=["impedance"],
        ppt_context="PPT OCR: Impedance",
        batch_size=4,
    )
    # 2 failed attempts on size=4, then 2 bisected batches of size=2.
    assert len(calls) == 4
    assert calls[0]["extra_body"] == {"enable_thinking": False}
    assert [c.en for c in cues] == [s.text for s in sentences]
    assert all(c.zh.startswith("译文") for c in cues)


@pytest.mark.asyncio
async def test_llm_translator_single_item_strips_cot_via_no_ppt_retry_and_fallback() -> None:
    calls: list[dict] = []

    class FakeCompletions:
        async def create(self, **kwargs):
            import json

            calls.append(kwargs)
            payload = json.loads(kwargs["messages"][1]["content"])
            if payload["ppt_context"]:
                content = json.dumps(
                    {
                        "items": [
                            {
                                "i": 0,
                                "en": "cosine omega t plus phi where this.",
                                "zh": "等等，PPT OCR有点乱，这里应该是 $V \\cos(\\omega t + \\phi)$ 结合上下文和PPT公式(92)等等，讲师说的是这个振幅……" * 2,
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            else:
                content = json.dumps(
                    {"items": [{"i": 0, "en": "cosine omega t plus phi where this.", "zh": "cos(ωt + φ)，其中这一项……"}]},
                    ensure_ascii=False,
                )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    translator = LLMTranslator(api_key="test", base_url="http://localhost", client=fake_client)

    cues = await translator.refine_and_translate(
        [AsrSentence(start_ms=0, end_ms=2000, text="cosine omega t plus phi where this.")],
        ppt_context="Noisy OCR [624s]",
        batch_size=1,
    )
    assert len(calls) == 3  # 2 failed attempts with PPT context -> 1 clean retry without PPT OCR
    assert cues[0].zh == "cos(ωt + φ)，其中这一项……"


@pytest.mark.asyncio
async def test_llm_translator_falls_back_cleanly_when_model_returns_empty_items() -> None:
    class EmptyCompletions:
        async def create(self, **_kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"items": []}'))])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=EmptyCompletions()))
    translator = LLMTranslator(api_key="test", base_url="http://localhost", client=fake_client)

    cues = await translator.refine_and_translate(
        [AsrSentence(start_ms=2500, end_ms=4000, text="Fallback line")],
        ppt_context="",
        batch_size=2,
    )
    assert cues[0].en == "Fallback line"
    assert cues[0].zh == "Fallback line"


def test_is_chinese_spoken_classification() -> None:
    assert is_chinese_spoken("好的，我们今天继续讲微分方程。") is True
    assert is_chinese_spoken("这是 R = 4L/C 的临界条件") is True
    assert is_chinese_spoken("这个 matrix 的 eigenvalue 是多少") is True
    assert is_chinese_spoken("We call this 特征值.") is False
    assert is_chinese_spoken("In Chinese it is 若尔当.") is False
    assert is_chinese_spoken("Now consider the characteristic equation.") is False


@pytest.mark.asyncio
async def test_llm_translator_chinese_spoken_emits_zh_only_and_strips_accidental_en() -> None:
    import json

    calls: list[dict] = []

    class MixedCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            content = json.dumps(
                {
                    "items": [
                        {"i": 0, "en": "", "zh": "好的，今天我们继续讲特征值分解。"},
                        {"i": 1, "en": "Accidental English translation", "zh": "这个矩阵的 eigenvalue 是多少？"},
                        {"i": 2, "en": "Now notice the determinant is zero.", "zh": "现在注意行列式为零。"},
                    ]
                },
                ensure_ascii=False,
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    translator = LLMTranslator(
        api_key="test",
        base_url="http://localhost",
        client=SimpleNamespace(chat=SimpleNamespace(completions=MixedCompletions())),
    )
    sentences = [
        AsrSentence(start_ms=0, end_ms=1500, text="好的今天我们继续讲特征值分解"),
        AsrSentence(start_ms=1500, end_ms=3000, text="这个矩阵的 eigenvalue 是多少"),
        AsrSentence(start_ms=3000, end_ms=5000, text="Now notice the determinant is zero."),
    ]
    cues = await translator.refine_and_translate(sentences, batch_size=12)
    assert len(calls) == 1  # Validated on first pass with zero retries even with en=""
    assert cues[0].en == ""
    assert cues[0].zh == "好的，今天我们继续讲特征值分解。"
    assert cues[1].en == ""
    assert cues[1].zh == "这个矩阵的 eigenvalue 是多少？"
    assert cues[2].en == "Now notice the determinant is zero."
    assert cues[2].zh == "现在注意行列式为零。"
