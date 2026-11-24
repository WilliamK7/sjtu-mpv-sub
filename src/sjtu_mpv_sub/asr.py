"""Pure RESTful Aliyun Bailian Qwen-Audio 3.1 Filetrans ASR with temporary OSS upload."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

import httpx

from sjtu_mpv_sub.models import AsrSentence

logger = logging.getLogger(__name__)

_TERM_SPLIT_RE = re.compile(r"\s*[=：:]\s*")
_SILENT_ASR_ERROR_CODES = frozenset({"ASR_RESPONSE_HAVE_NO_WORDS"})


def _is_no_words_error(output: dict[str, Any]) -> bool:
    """Return True when Bailian reports a silent or no-speech audio chunk."""
    top_code = str(output.get("code") or "").strip().upper()
    top_message = str(output.get("message") or "").strip().upper()
    if top_code in _SILENT_ASR_ERROR_CODES or top_message in _SILENT_ASR_ERROR_CODES:
        return True

    results = output.get("results")
    if not isinstance(results, list) or not results:
        return False
    dict_items = [item for item in results if isinstance(item, dict)]
    if len(dict_items) != len(results):
        return False

    for item in dict_items:
        if str(item.get("subtask_status", "")).strip().upper() != "FAILED":
            return False
        sub_code = str(item.get("code") or "").strip().upper()
        sub_message = str(item.get("message") or "").strip().upper()
        if sub_code not in _SILENT_ASR_ERROR_CODES and sub_message not in _SILENT_ASR_ERROR_CODES:
            return False
    return True


class ASRProvider(Protocol):
    """Protocol for async chunked speech recognition providers."""

    async def transcribe_chunk(
        self,
        audio_path: Path,
        language_hints: list[str],
        *,
        vocabulary_id: str | None = None,
        hotwords: dict[str, int] | None = None,
        context_messages: list[dict[str, Any]] | None = None,
    ) -> list[AsrSentence]:
        """Transcribe a single local audio chunk into timestamped sentences."""
        ...

    async def close(self) -> None:
        """Release underlying HTTP client resources if owned."""
        ...


def build_instant_vocabulary(
    glossary: list[str],
    ppt_keywords: list[str],
    max_terms: int = 2000,
) -> dict[str, int]:
    """Build Bailian instant vocabulary dict adhering to official ASCII/non-ASCII length rules."""
    vocab: dict[str, int] = {}

    def _add_raw(raw: str) -> None:
        if len(vocab) >= max_terms:
            return
        candidate = _TERM_SPLIT_RE.split(raw.strip(), maxsplit=1)[0]
        term = " ".join(candidate.split())
        if len(term) < 2 or term in vocab:
            return
        if term.isascii():
            if len(term.split()) > 7:
                return
        else:
            if len(term) > 15:
                return
        vocab[term] = 4

    for item in glossary:
        _add_raw(item)
    for kw in ppt_keywords:
        _add_raw(kw)

    return vocab


def build_asr_context_messages(
    course_name: str,
    ppt_ocr_snippet: str,
    max_chars: int = 400,
) -> list[dict[str, Any]]:
    """Build official input.context message list within Bailian's 400-char per-turn limit."""
    course_clean = " ".join(course_name.split())
    ocr_clean = " ".join(ppt_ocr_snippet.split())

    if course_clean and ocr_clean:
        text = f"Course: {course_clean}. Slides: {ocr_clean}"
    elif course_clean:
        text = f"Course: {course_clean}"
    elif ocr_clean:
        text = ocr_clean
    else:
        return []

    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    if not text:
        return []

    return [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        }
    ]


def parse_filetrans_sentences(transcript_json: dict[str, Any]) -> list[AsrSentence]:
    """Extract normalized AsrSentence items from Bailian transcription result JSON."""
    transcripts = transcript_json.get("transcripts")
    if not isinstance(transcripts, list) or not transcripts:
        return []

    sentences: list[AsrSentence] = []
    first_channel = transcripts[0] if isinstance(transcripts[0], dict) else {}
    raw_sentences = first_channel.get("sentences") or []
    for item in raw_sentences:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        begin_ms = max(0, int(item.get("begin_time", 0) or 0))
        end_ms = max(begin_ms + 100, int(item.get("end_time", begin_ms) or begin_ms))
        sentences.append(AsrSentence(start_ms=begin_ms, end_ms=end_ms, text=text))
    return sentences


class QwenAudioFiletransASR:
    """Async RESTful client for qwen-audio-3.1-asr-flash-filetrans + Bailian temporary OSS."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "qwen-audio-3.1-asr-flash-filetrans",
        uploads_base_url: str = "https://dashscope.aliyuncs.com/api/v1",
        asr_api_base_url: str = "https://dashscope.aliyuncs.com/api/v1",
        poll_interval_seconds: float = 1.5,
        timeout_seconds: float = 300.0,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model
        self.uploads_base_url = uploads_base_url.rstrip("/")
        self.asr_api_base_url = asr_api_base_url.rstrip("/")
        self.poll_interval_seconds = max(0.05, float(poll_interval_seconds))
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self.max_retries = max_retries
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(trust_env=False, timeout=60.0)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get_upload_policy(self) -> dict[str, Any]:
        resp = await self._client.get(
            f"{self.uploads_base_url}/uploads",
            params={"action": "getPolicy", "model": self.model},
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Bailian getPolicy failed ({resp.status_code}): {resp.text[:300]}")
        data = resp.json().get("data")
        if not isinstance(data, dict) or not data.get("upload_host") or not data.get("upload_dir"):
            raise RuntimeError(f"Bailian getPolicy returned malformed payload: {resp.text[:300]}")
        return data

    async def _upload_to_temp_oss(self, policy_data: dict[str, Any], audio_path: Path) -> str:
        unique_name = f"{uuid.uuid4().hex[:8]}_{audio_path.name}"
        upload_dir = str(policy_data["upload_dir"]).strip("/")
        object_key = f"{upload_dir}/{unique_name}"
        audio_bytes = await asyncio.to_thread(audio_path.read_bytes)

        # Keep form fields ordered with 'file' last per Aliyun OSS PostObject requirements.
        multipart_fields = [
            ("OSSAccessKeyId", (None, str(policy_data["oss_access_key_id"]))),
            ("policy", (None, str(policy_data["policy"]))),
            ("Signature", (None, str(policy_data["signature"]))),
            ("x-oss-object-acl", (None, str(policy_data["x_oss_object_acl"]))),
            ("x-oss-forbid-overwrite", (None, str(policy_data["x_oss_forbid_overwrite"]))),
            ("key", (None, object_key)),
            ("success_action_status", (None, "200")),
            ("file", (unique_name, audio_bytes, "audio/mpeg")),
        ]
        resp = await self._client.post(str(policy_data["upload_host"]), files=multipart_fields)
        if resp.status_code not in (200, 201, 204):
            raise RuntimeError(f"Bailian temp OSS upload failed ({resp.status_code}): {resp.text[:300]}")
        return f"oss://{object_key}"

    async def _submit_transcription(
        self,
        oss_url: str,
        language_hints: list[str],
        vocabulary_id: str | None,
        hotwords: dict[str, int] | None,
        context_messages: list[dict[str, Any]] | None,
    ) -> str:
        input_obj: dict[str, Any] = {"file_urls": [oss_url]}
        if context_messages:
            input_obj["context"] = context_messages

        params_obj: dict[str, Any] = {"channel_id": [0]}
        if language_hints:
            params_obj["language_hints"] = list(language_hints[:4])
        if vocabulary_id and vocabulary_id.strip():
            params_obj["vocabulary_id"] = vocabulary_id.strip()
        if hotwords:
            params_obj["vocabulary"] = hotwords

        payload = {
            "model": self.model,
            "input": input_obj,
            "parameters": params_obj,
        }
        resp = await self._client.post(
            f"{self.asr_api_base_url}/services/audio/asr/transcription",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
                "X-DashScope-OssResourceResolve": "enable",
            },
            json=payload,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Bailian ASR submit failed ({resp.status_code}): {resp.text[:400]}")
        output = resp.json().get("output") or {}
        task_id = output.get("task_id")
        if not task_id:
            raise RuntimeError(f"Bailian ASR submit missing task_id: {resp.text[:300]}")
        return str(task_id)

    async def _poll_and_fetch(self, task_id: str) -> list[AsrSentence]:
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            resp = await self._client.get(
                f"{self.asr_api_base_url}/tasks/{task_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Bailian ASR task poll failed ({resp.status_code}): {resp.text[:300]}")

            payload = resp.json()
            output = payload.get("output") or {}
            status = str(output.get("task_status", "")).upper()

            if status == "SUCCEEDED":
                results = output.get("results") or []
                transcription_url: str | None = None
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    if str(item.get("subtask_status", "")).upper() == "SUCCEEDED" and item.get("transcription_url"):
                        transcription_url = str(item["transcription_url"])
                        break
                if not transcription_url and output.get("transcription_url"):
                    transcription_url = str(output["transcription_url"])
                if transcription_url:
                    # Signed OSS result URLs must NOT receive DashScope Bearer Authorization headers.
                    tr_resp = await self._client.get(transcription_url)
                    if tr_resp.status_code != 200:
                        raise RuntimeError(
                            f"Failed to download transcription_url ({tr_resp.status_code}): {tr_resp.text[:200]}"
                        )
                    return parse_filetrans_sentences(tr_resp.json())
                if _is_no_words_error(output):
                    logger.info(
                        "Bailian ASR task %s reported no speech (subtask); treating chunk as silent.",
                        task_id,
                    )
                    return []
                raise RuntimeError(f"Bailian ASR subtask failed: {json.dumps(results, ensure_ascii=False)}")

            if status in ("FAILED", "CANCELED"):
                if _is_no_words_error(output):
                    logger.info(
                        "Bailian ASR task %s reported no speech (%s); treating chunk as silent.",
                        task_id,
                        output.get("code") or output.get("message") or status,
                    )
                    return []
                raise RuntimeError(f"Bailian ASR task {task_id} ended with {status}: {json.dumps(payload, ensure_ascii=False)}")

            await asyncio.sleep(self.poll_interval_seconds)

        raise RuntimeError(f"Bailian ASR task {task_id} timed out after {self.timeout_seconds}s")

    async def _run_once(
        self,
        audio_path: Path,
        language_hints: list[str],
        *,
        vocabulary_id: str | None,
        hotwords: dict[str, int] | None,
        context_messages: list[dict[str, Any]] | None,
    ) -> list[AsrSentence]:
        policy = await self._get_upload_policy()
        oss_url = await self._upload_to_temp_oss(policy, audio_path)
        task_id = await self._submit_transcription(
            oss_url,
            language_hints,
            vocabulary_id,
            hotwords,
            context_messages,
        )
        return await self._poll_and_fetch(task_id)

    async def transcribe_chunk(
        self,
        audio_path: Path,
        language_hints: list[str],
        *,
        vocabulary_id: str | None = None,
        hotwords: dict[str, int] | None = None,
        context_messages: list[dict[str, Any]] | None = None,
    ) -> list[AsrSentence]:
        if not self.api_key:
            raise RuntimeError("DashScope API key is missing. Please configure dashscope.api_key in config.yaml.")

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self._run_once(
                    audio_path,
                    language_hints,
                    vocabulary_id=vocabulary_id,
                    hotwords=hotwords,
                    context_messages=context_messages,
                )
            except Exception as exc:
                last_err = exc
                if attempt < self.max_retries:
                    delay = 1.0 * (2**attempt)
                    logger.warning(
                        "Bailian filetrans ASR attempt %d/%d failed for %s: %s; retrying in %.1fs",
                        attempt + 1,
                        self.max_retries + 1,
                        audio_path.name,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)

        assert last_err is not None
        raise last_err
