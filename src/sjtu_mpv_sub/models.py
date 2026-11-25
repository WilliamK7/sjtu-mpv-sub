"""Pydantic domain models for sjtu-mpv-sub."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ViewMode = Literal["ppt", "camera"]
JobState = Literal["queued", "probing", "running", "playing", "completed", "canceled", "failed"]
BilingualOrder = Literal["zh_top", "en_top"]


class StreamSource(BaseModel):
    """A single video stream source returned by SJTU course_vod_urls_new."""

    view_num: int
    url: str
    vod_time: float | None = None


class PptSlice(BaseModel):
    """A time-aligned PPT slide OCR snapshot from SJTU course/ai/ppt."""

    seek_sec: float
    ocr_text: str = ""
    keywords: list[str] = Field(default_factory=list)


class SegmentInput(BaseModel):
    """One class period (section) belonging to a lecture session."""

    course_id: int
    title: str
    begin_time: str = ""
    end_time: str = ""
    leti_number: int | None = None
    streams: list[StreamSource]
    ppt_slices: list[PptSlice] = Field(default_factory=list)


class PlayRequest(BaseModel):
    """Payload posted from Tampermonkey when launching mpv playback."""

    tecl_id: int
    course_name: str
    preferred_view: ViewMode | None = None
    force_retranscribe: bool = False
    segments: list[SegmentInput] = Field(min_length=1)


class CoursePreference(BaseModel):
    """Per-course persistent view preference and glossary settings."""

    name: str = ""
    preferred_view: ViewMode = "ppt"
    glossary: list[str] = Field(default_factory=list)
    vocabulary_id: str | None = None
    extra_prompt: str = ""


class CoursePreferenceUpdate(BaseModel):
    """Partial or full update payload for PUT /api/courses/{tecl_id}."""

    name: str | None = None
    preferred_view: ViewMode | None = None
    glossary: list[str] | None = None
    vocabulary_id: str | None = None
    extra_prompt: str | None = None


class AsrSentence(BaseModel):
    """A single recognized speech sentence with millisecond bounds."""

    start_ms: int
    end_ms: int
    text: str


class BilingualCue(BaseModel):
    """A refined bilingual subtitle cue with millisecond bounds."""

    start_ms: int
    end_ms: int
    en: str
    zh: str


class JobStatus(BaseModel):
    """Progress and status snapshot for a background playback job."""

    job_id: str
    session_key: str
    state: JobState
    message: str
    chunks_done: int
    chunks_total: int
    mpv_launched: bool
    cache_hit: bool
    covered_seconds: float = 0.0
    total_seconds: float = 0.0
    lecture_start_sec: float | None = None
    error: str | None = None
