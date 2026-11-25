"""mpv EDL playlist builder, launcher, and Windows Named Pipe JSON IPC client."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EdlSegment(BaseModel):
    """One virtual segment inside an mpv EDL timeline."""

    url: str
    duration_sec: float
    title: str = ""


def quote_edl_url(url: str) -> str:
    """Length-prefix quote URLs so query strings (?key=...) and commas never break EDL parsing."""
    raw = url.encode("utf-8")
    return f"%{len(raw)}%{url}"


def sanitize_edl_title(title: str) -> str:
    """Remove characters that act as EDL record/field delimiters."""
    cleaned = " ".join(title.replace("\r", " ").replace("\n", " ").split())
    cleaned = cleaned.replace(",", " ").replace(";", " ").replace("%", "")
    return " ".join(cleaned.split()) or "Section"


def build_edl(segments: list[EdlSegment], *, include_chapters: bool = True) -> str:
    """Generate an LF-delimited mpv EDL v0 document."""
    if not segments:
        raise ValueError("Cannot build EDL with zero segments")

    lines = ["# mpv EDL v0"]
    if not include_chapters:
        lines.append("!no_chapters")

    for seg in segments:
        dur = max(0.1, float(seg.duration_sec))
        entry = f"{quote_edl_url(seg.url)},0,{dur:.3f}"
        if include_chapters and seg.title.strip():
            entry += f",title={sanitize_edl_title(seg.title)}"
        lines.append(entry)

    return "\n".join(lines) + "\n"


def write_edl(path: Path, content: str) -> None:
    """Write EDL with strict LF line endings (mpv rejects CRLF on the header line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode("utf-8"))


class MpvIpcClient:
    """Minimal synchronous client for mpv Windows Named Pipe JSON IPC."""

    def __init__(self, pipe_path: str) -> None:
        self.pipe_path = pipe_path

    def send_command(self, command: list[Any]) -> dict[str, Any]:
        """Send one JSON IPC command and return the matching response packet."""
        payload = (json.dumps({"command": command}, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            with open(self.pipe_path, "r+b", buffering=0) as pipe:
                pipe.write(payload)
                while True:
                    line = pipe.readline()
                    if not line:
                        return {"error": "eof"}
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text:
                        continue
                    try:
                        obj = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and "error" in obj:
                        return obj
        except OSError as exc:
            return {"error": f"pipe_unavailable: {exc}"}

    def wait_ready(self, timeout: float = 12.0) -> bool:
        """Wait until mpv pipe is accepting commands and tracks have loaded."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            resp = self.send_command(["get_property", "track-list"])
            if resp.get("error") == "success" and isinstance(resp.get("data"), list) and len(resp["data"]) > 0:
                return True
            time.sleep(0.15)
        return False

    def reload_subtitles(self) -> bool:
        """Ask mpv to reload its external subtitle file."""
        resp = self.send_command(["sub-reload"])
        if resp.get("error") == "success":
            return True
        if resp.get("error") == "error running command":
            time.sleep(0.25)
            resp = self.send_command(["sub-reload"])
            return resp.get("error") == "success"
        logger.debug("mpv sub-reload response: %s", resp)
        return False

    def show_osd(self, text: str, duration_ms: int = 3000) -> bool:
        """Display a brief OSD toast inside mpv."""
        resp = self.send_command(["show-text", text, duration_ms])
        return resp.get("error") == "success"

    def get_property(self, name: str) -> Any:
        """Read a single mpv property via JSON IPC, returning None if unavailable."""
        resp = self.send_command(["get_property", name])
        if resp.get("error") == "success":
            return resp.get("data")
        return None

    def get_time_pos(self) -> float | None:
        """Return current mpv playback timestamp in seconds, or None if unavailable."""
        val = self.get_property("time-pos")
        if val is None:
            return None
        try:
            return max(0.0, float(val))
        except (TypeError, ValueError):
            return None

    def seek_absolute(self, seconds: float) -> bool:
        """Seek mpv immediately to an absolute timestamp in seconds."""
        target = max(0.0, float(seconds))
        resp = self.send_command(["seek", target, "absolute+exact"])
        return resp.get("error") == "success"

    def send_hud_state(self, payload: dict[str, Any]) -> bool:
        """Push structured subtitle progress state to the bundled Lua HUD script."""
        encoded = json.dumps(payload, ensure_ascii=False)
        resp = self.send_command(["script-message-to", "sjtu_sub_hud", "update", encoded])
        return resp.get("error") == "success"

    def clear_hud(self) -> bool:
        """Hide/clear the bundled Lua HUD overlay."""
        resp = self.send_command(["script-message-to", "sjtu_sub_hud", "clear"])
        return resp.get("error") == "success"

    def consume_user_cancel_flag(self) -> bool:
        """Return True if the user triggered background cancellation from inside mpv."""
        return bool(self.get_property("user-data/sjtu-sub/cancel"))


def get_hud_script_path() -> Path:
    """Return the path to the bundled mpv Lua HUD script."""
    return Path(__file__).resolve().parent / "mpv_scripts" / "sjtu_sub_hud.lua"


def build_mpv_pipe_path(job_id: str) -> str:
    """Return a clean Windows named pipe path for a playback job."""
    safe = "".join(c for c in job_id if c.isalnum() or c in ("-", "_"))
    return rf"\\.\pipe\sjtu-mpv-{safe}"


def launch_mpv(
    *,
    mpv_path: str,
    video_edl_path: Path,
    sub_path: Path,
    pipe_path: str,
    media_title: str,
    referer: str = "",
    audio_edl_path: Path | None = None,
    script_path: Path | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.Popen[bytes]:
    """Launch mpv with EDL timeline, UTF-8 ASS subtitles, and IPC server enabled."""
    cmd: list[str] = [
        mpv_path,
        "--autocreate-playlist=no",
        "--sub-codepage=utf8",
        f"--sub-file={sub_path}",
        f"--force-media-title={media_title}",
        f"--input-ipc-server={pipe_path}",
    ]
    if referer.strip():
        cmd.append(f"--http-header-fields=Referer: {referer.strip()}")
    if audio_edl_path is not None:
        cmd.append(f"--audio-file={audio_edl_path}")
    if script_path is not None and script_path.is_file():
        cmd.append(f"--script={script_path}")
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(video_edl_path))

    logger.info("Launching mpv: %s", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
