"""Configuration, executable resolution, and thread-safe course preferences."""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from sjtu_mpv_sub.models import BilingualOrder, CoursePreference, CoursePreferenceUpdate

_COURSES_LOCK = threading.RLock()
_DEFAULT_MPV_FALLBACK = Path(r"D:\Program Files\mpv-lazy\mpv.exe")


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 18732
    idle_exit_seconds: int = 300


class ToolsConfig(BaseModel):
    mpv_path: str = r"D:\Program Files\mpv-lazy\mpv.exe"
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"


class DashScopeConfig(BaseModel):
    api_key: str = ""
    workspace_id: str = ""
    region: str = "cn-beijing"
    uploads_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    asr_api_base_url: str = ""
    asr_model: str = "qwen-audio-3.1-asr-flash-filetrans"
    asr_language_hints: list[str] = Field(default_factory=lambda: ["en", "zh"])
    asr_poll_interval_seconds: float = 1.5
    asr_timeout_seconds: float = 300.0
    llm_base_url: str = ""
    llm_model: str = "qwen3.8-flash"
    llm_enable_thinking: bool = False
    llm_api_key: str = ""
    llm_proxy: str = ""


class PipelineConfig(BaseModel):
    first_chunk_seconds: int = 180
    chunk_seconds: int = 420
    max_concurrency: int = 3
    translation_batch_size: int = 28
    stop_on_mpv_exit: bool = True
    referer: str = "https://v.sjtu.edu.cn/jy-application-resourcemanage-ui/"


class SubtitleConfig(BaseModel):
    bilingual_order: BilingualOrder = "zh_top"
    font_zh: str = "Microsoft YaHei UI"
    font_en: str = "Segoe UI"
    size_zh: int = 46
    size_en: int = 34


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    dashscope: DashScopeConfig = Field(default_factory=DashScopeConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    subtitle: SubtitleConfig = Field(default_factory=SubtitleConfig)

    def effective_dashscope_api_key(self) -> str:
        """Return DashScope API key from config or DASHSCOPE_API_KEY env var."""
        return (self.dashscope.api_key or os.environ.get("DASHSCOPE_API_KEY", "")).strip()

    def effective_llm_api_key(self) -> str:
        """Return LLM API key, falling back to effective DashScope key."""
        return (self.dashscope.llm_api_key or self.effective_dashscope_api_key()).strip()

    def effective_asr_api_base_url(self) -> str:
        """Return ASR RESTful API base URL, deriving Workspace endpoint when configured."""
        override = self.dashscope.asr_api_base_url.strip().rstrip("/")
        if override:
            return override
        ws = self.dashscope.workspace_id.strip()
        reg = self.dashscope.region.strip() or "cn-beijing"
        if ws:
            return f"https://{ws}.{reg}.maas.aliyuncs.com/api/v1"
        return "https://dashscope.aliyuncs.com/api/v1"

    def effective_llm_base_url(self) -> str:
        """Return OpenAI-compatible LLM base URL, deriving Workspace endpoint when configured."""
        override = self.dashscope.llm_base_url.strip().rstrip("/")
        if override:
            return override
        ws = self.dashscope.workspace_id.strip()
        reg = self.dashscope.region.strip() or "cn-beijing"
        if ws:
            return f"https://{ws}.{reg}.maas.aliyuncs.com/compatible-mode/v1"
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_root_dir() -> Path:
    """Resolve the project workspace root directory."""
    env_home = os.environ.get("SJTU_MPV_SUB_HOME", "").strip()
    if env_home:
        return Path(env_home).resolve()
    return Path(__file__).resolve().parents[2]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def ensure_config_files(root_dir: Path | None = None) -> tuple[Path, Path]:
    """Ensure config.yaml and courses.yaml exist under root_dir."""
    root = root_dir or get_root_dir()
    config_path = root / "config.yaml"
    example_path = root / "config.example.yaml"
    courses_path = root / "courses.yaml"

    if not config_path.exists():
        if example_path.exists():
            _atomic_write_text(config_path, example_path.read_text(encoding="utf-8"))
        else:
            default_yaml = yaml.safe_dump(
                AppConfig().model_dump(mode="json"),
                allow_unicode=True,
                sort_keys=False,
            )
            _atomic_write_text(config_path, default_yaml)

    if not courses_path.exists():
        _atomic_write_text(courses_path, "{}\n")

    return config_path, courses_path


def load_config(root_dir: Path | None = None) -> AppConfig:
    """Load AppConfig from config.yaml, auto-seeding if missing."""
    config_path, _ = ensure_config_files(root_dir)
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raw = {}
    return AppConfig.model_validate(raw)


def resolve_executable(configured: str, fallbacks: list[str | Path] | None = None) -> str | None:
    """Resolve an executable path from explicit path, PATH lookup, or known fallbacks."""
    candidates: list[str | Path] = []
    if configured:
        candidates.append(configured)
    if fallbacks:
        candidates.extend(fallbacks)

    for candidate in candidates:
        candidate_str = str(candidate).strip()
        if not candidate_str:
            continue
        p = Path(candidate_str)
        if p.is_file():
            return str(p)
        resolved = shutil.which(candidate_str)
        if resolved:
            return resolved
    return None


def resolve_tool_paths(config: AppConfig) -> dict[str, str | None]:
    """Resolve mpv, ffmpeg, and ffprobe executable paths."""
    return {
        "mpv": resolve_executable(config.tools.mpv_path, ["mpv", _DEFAULT_MPV_FALLBACK]),
        "ffmpeg": resolve_executable(config.tools.ffmpeg_path, ["ffmpeg"]),
        "ffprobe": resolve_executable(config.tools.ffprobe_path, ["ffprobe"]),
    }


def _load_courses_raw(courses_path: Path) -> dict[str, dict[str, Any]]:
    if not courses_path.exists():
        return {}
    raw = yaml.safe_load(courses_path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def get_course_preference(tecl_id: int, root_dir: Path | None = None) -> CoursePreference:
    """Load per-course preference from courses.yaml, returning defaults if absent."""
    with _COURSES_LOCK:
        _, courses_path = ensure_config_files(root_dir)
        data = _load_courses_raw(courses_path)
        entry = data.get(str(tecl_id))
        if isinstance(entry, dict):
            return CoursePreference.model_validate(entry)
        return CoursePreference()


def upsert_course_preference(
    tecl_id: int,
    update: CoursePreferenceUpdate,
    root_dir: Path | None = None,
) -> CoursePreference:
    """Thread-safely merge and save per-course preference to courses.yaml."""
    with _COURSES_LOCK:
        _, courses_path = ensure_config_files(root_dir)
        data = _load_courses_raw(courses_path)
        key = str(tecl_id)
        current = (
            CoursePreference.model_validate(data[key])
            if isinstance(data.get(key), dict)
            else CoursePreference()
        )
        merged = current.model_copy(
            update={k: v for k, v in update.model_dump(exclude_unset=True).items() if v is not None or k == "vocabulary_id"}
        )
        data[key] = merged.model_dump(mode="json")
        serialized = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        _atomic_write_text(courses_path, serialized)
        return merged
