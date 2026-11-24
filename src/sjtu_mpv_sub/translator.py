"""Time-aligned PPT OCR context builder and OpenAI-compatible LLM translator."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from openai import AsyncOpenAI

from sjtu_mpv_sub.models import AsrSentence, BilingualCue, PptSlice

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def collect_ppt_window(
    ppt_slices: list[PptSlice],
    chunk_start_sec: float,
    chunk_end_sec: float,
    max_chars: int = 1500,
) -> tuple[list[str], str]:
    """Collect active/in-window PPT keywords and formatted OCR context for a chunk."""
    if not ppt_slices:
        return [], ""

    ordered = sorted(ppt_slices, key=lambda s: s.seek_sec)
    active_before: PptSlice | None = None
    window_slices: list[PptSlice] = []

    for sl in ordered:
        if sl.seek_sec <= chunk_start_sec:
            active_before = sl
        elif sl.seek_sec <= chunk_end_sec + 20.0:
            window_slices.append(sl)

    selected: list[PptSlice] = []
    if active_before is not None:
        selected.append(active_before)
    selected.extend(window_slices)

    if not selected:
        return [], ""

    keywords_seen: list[str] = []
    keywords_set: set[str] = set()
    slide_blocks: list[str] = []

    for sl in selected:
        for kw in sl.keywords:
            cleaned_kw = kw.strip()
            if cleaned_kw and cleaned_kw not in keywords_set:
                keywords_set.add(cleaned_kw)
                keywords_seen.append(cleaned_kw)
        ocr = " ".join(sl.ocr_text.split())
        if ocr:
            slide_blocks.append(f"[{int(sl.seek_sec)}s] {ocr}")

    parts: list[str] = []
    if keywords_seen:
        parts.append("PPT Keywords: " + ", ".join(keywords_seen))
    if slide_blocks:
        parts.append("PPT OCR:\n" + "\n".join(slide_blocks))

    combined = "\n".join(parts)
    if len(combined) > max_chars:
        combined = combined[: max_chars - 3].rstrip() + "..."
    return keywords_seen, combined


def build_ppt_context(
    ppt_slices: list[PptSlice],
    chunk_start_sec: float,
    chunk_end_sec: float,
    max_chars: int = 1500,
) -> str:
    """Collect active and in-window PPT slide OCR text and keywords for a chunk."""
    _, context = collect_ppt_window(ppt_slices, chunk_start_sec, chunk_end_sec, max_chars=max_chars)
    return context


def parse_llm_items(raw_content: str) -> dict[int, tuple[str, str]]:
    """Parse JSON items mapping index -> (en, zh), stripping markdown code fences if needed."""
    text = (raw_content or "").strip()
    if text.startswith("```"):
        text = _CODE_FENCE_RE.sub("", text).strip()

    data: Any = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError("LLM JSON missing 'items' array")

    mapped: dict[int, tuple[str, str]] = {}
    for entry in data["items"]:
        if not isinstance(entry, dict) or "i" not in entry:
            continue
        try:
            idx = int(entry["i"])
        except (TypeError, ValueError):
            continue
        en_val = str(entry.get("en", "")).strip()
        zh_val = str(entry.get("zh", "")).strip()
        mapped[idx] = (en_val, zh_val)
    return mapped


class LLMTranslator:
    """Async batch ASR refiner and bilingual translator via OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "qwen3.8-flash",
        proxy: str = "",
        enable_thinking: bool = False,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self.enable_thinking = enable_thinking
        self._owned_http_client: httpx.AsyncClient | None = None
        if client is not None:
            self._client = client
        else:
            http_kwargs: dict[str, Any] = {"trust_env": False, "timeout": 90.0}
            if proxy.strip():
                http_kwargs["proxy"] = proxy.strip()
            self._owned_http_client = httpx.AsyncClient(**http_kwargs)
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=self._owned_http_client,
                timeout=90.0,
            )

    async def close(self) -> None:
        if self._owned_http_client is not None:
            await self._owned_http_client.aclose()
            self._owned_http_client = None

    async def refine_and_translate(
        self,
        sentences: list[AsrSentence],
        *,
        course_name: str = "",
        glossary: list[str] | None = None,
        extra_prompt: str = "",
        ppt_context: str = "",
        batch_size: int = 28,
    ) -> list[BilingualCue]:
        """Refine ASR typos with context and produce aligned English+Chinese cues."""
        if not sentences:
            return []

        step = max(1, batch_size)
        cues: list[BilingualCue] = []

        for start in range(0, len(sentences), step):
            batch = sentences[start : start + step]
            prev_context = [s.text for s in sentences[max(0, start - 3) : start]]
            batch_cues = await self._translate_batch(
                batch,
                prev_context=prev_context,
                course_name=course_name,
                glossary=glossary or [],
                extra_prompt=extra_prompt,
                ppt_context=ppt_context,
            )
            cues.extend(batch_cues)

        return cues

    async def _translate_batch(
        self,
        batch: list[AsrSentence],
        *,
        prev_context: list[str],
        course_name: str,
        glossary: list[str],
        extra_prompt: str,
        ppt_context: str,
    ) -> list[BilingualCue]:
        system_prompt = (
            "You are an expert bilingual academic lecture subtitle refiner and translator for Shanghai Jiao Tong "
            "University (SJTU) courses.\n"
            "Tasks for each indexed ASR segment:\n"
            "1. Correct ASR speech-recognition typos, homophone errors, and misheard technical terms using the "
            "course name, glossary, and PPT slide OCR.\n"
            "2. Output 'en': accurate, cleaned English caption faithful to the lecture (if a line was spoken in "
            "Chinese/code-switching, preserve technical terms accurately in clear English).\n"
            "3. Output 'zh': natural, concise, accurate Simplified Chinese translation suitable for fast-speed "
            "lecture watching (keep key English abbreviations or symbols in parentheses where helpful).\n"
            "4. Return strict JSON ONLY with schema: "
            '{"items": [{"i": <int>, "en": "<string>", "zh": "<string>"}]} '
            "preserving every input index 1-to-1."
        )

        payload = {
            "course_name": course_name,
            "glossary": glossary,
            "extra_instructions": extra_prompt.strip(),
            "ppt_context": ppt_context,
            "previous_context_read_only": prev_context,
            "segments": [{"i": idx, "text": s.text} for idx, s in enumerate(batch)],
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        mapped: dict[int, tuple[str, str]] = {}
        for attempt in range(2):
            try:
                resp = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    extra_body={"enable_thinking": self.enable_thinking},
                )
                content = resp.choices[0].message.content or ""
                mapped = parse_llm_items(content)
                break
            except Exception as exc:
                logger.warning("LLM translation batch attempt %d failed: %s", attempt + 1, exc)

        result: list[BilingualCue] = []
        for idx, sent in enumerate(batch):
            en_text, zh_text = mapped.get(idx, ("", ""))
            result.append(
                BilingualCue(
                    start_ms=sent.start_ms,
                    end_ms=sent.end_ms,
                    en=en_text or sent.text,
                    zh=zh_text or sent.text,
                )
            )
        return result
