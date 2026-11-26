"""Command-line interface for sjtu-mpv-sub."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sjtu_mpv_sub.config import get_course_preference, get_root_dir, load_config, resolve_tool_paths
from sjtu_mpv_sub.models import AsrSentence, BilingualCue
from sjtu_mpv_sub.server import run_server
from sjtu_mpv_sub.subtitle import render_ass, write_ass_atomic
from sjtu_mpv_sub.translator import LLMTranslator

PROTOCOL_SCHEME = "sjtu-mpv"


def _protocol_command(root_dir: Path) -> str:
    wake_vbs = root_dir / "wake.vbs"
    return f'wscript.exe "{wake_vbs}" "%1"'


def is_protocol_registered() -> bool:
    """Check whether HKCU\\Software\\Classes\\sjtu-mpv is registered."""
    if sys.platform != "win32":
        return False
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, rf"Software\Classes\{PROTOCOL_SCHEME}\shell\open\command") as key:
            value, _ = winreg.QueryValueEx(key, "")
            return bool(value)
    except FileNotFoundError:
        return False


def install_protocol(root_dir: Path) -> None:
    """Register HKCU\\Software\\Classes\\sjtu-mpv custom URL protocol without Admin rights."""
    if sys.platform != "win32":
        raise RuntimeError("Windows URL protocol registration is only supported on Windows.")
    import winreg

    base_path = rf"Software\Classes\{PROTOCOL_SCHEME}"
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, base_path) as base_key:
        winreg.SetValueEx(base_key, "", 0, winreg.REG_SZ, "URL:SJTU MPV Bilingual Player")
        winreg.SetValueEx(base_key, "URL Protocol", 0, winreg.REG_SZ, "")

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{base_path}\shell\open\command") as cmd_key:
        winreg.SetValueEx(cmd_key, "", 0, winreg.REG_SZ, _protocol_command(root_dir))


def uninstall_protocol() -> None:
    """Remove HKCU\\Software\\Classes\\sjtu-mpv registration."""
    if sys.platform != "win32":
        return
    import winreg

    def _delete_tree(root: int, subkey: str) -> None:
        try:
            with winreg.OpenKey(root, subkey, 0, winreg.KEY_ALL_ACCESS) as key:
                while True:
                    try:
                        child = winreg.EnumKey(key, 0)
                        _delete_tree(root, rf"{subkey}\{child}")
                    except OSError:
                        break
            winreg.DeleteKey(root, subkey)
        except FileNotFoundError:
            return

    _delete_tree(winreg.HKEY_CURRENT_USER, rf"Software\Classes\{PROTOCOL_SCHEME}")


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def rerender_sessions(root_dir: Path, *, session_key: str | None = None) -> list[tuple[str, int]]:
    """Re-render bilingual.ass from cached cues.json using current config.yaml subtitle settings."""
    config = load_config(root_dir)
    cache_root = root_dir / "data" / "cache"
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {cache_root}")

    if session_key:
        targets = [cache_root / session_key]
        if not (targets[0] / "cues.json").is_file():
            raise FileNotFoundError(f"Cached cues.json not found for session: {session_key}")
    else:
        targets = sorted(d for d in cache_root.iterdir() if d.is_dir() and (d / "cues.json").is_file())
        if not targets:
            raise FileNotFoundError(f"No cached sessions with cues.json found under {cache_root}")

    results: list[tuple[str, int]] = []
    for session_dir in targets:
        raw_cues = json.loads((session_dir / "cues.json").read_text(encoding="utf-8"))
        cues = [BilingualCue.model_validate(item) for item in raw_cues]
        ass_text = render_ass(cues, config.subtitle)
        write_ass_atomic(session_dir / "bilingual.ass", ass_text)
        results.append((session_dir.name, len(cues)))
    return results


def _group_sentences_for_retranslate(
    sentences: list[AsrSentence],
    *,
    max_span_ms: int = 360_000,
    max_items: int = 48,
) -> list[list[AsrSentence]]:
    """Group sentences into bounded time/size chunks while preserving chronological order."""
    if not sentences:
        return []
    groups: list[list[AsrSentence]] = []
    current: list[AsrSentence] = []
    group_start_ms = sentences[0].start_ms

    for sent in sentences:
        if current and (len(current) >= max_items or (sent.end_ms - group_start_ms) > max_span_ms):
            groups.append(current)
            current = []
            group_start_ms = sent.start_ms
        current.append(sent)
    if current:
        groups.append(current)
    return groups


async def retranslate_session_async(
    root_dir: Path,
    session_key: str,
    *,
    translator: LLMTranslator | None = None,
) -> int:
    """Re-translate a cached session using raw asr.json (or cues.json fallback) without re-running ASR."""
    config = load_config(root_dir)
    cache_dir = root_dir / "data" / "cache" / session_key
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Session cache directory not found: {cache_dir}")

    asr_path = cache_dir / "asr.json"
    cues_path = cache_dir / "cues.json"
    meta_path = cache_dir / "meta.json"
    sub_path = cache_dir / "bilingual.ass"

    sentences: list[AsrSentence] = []
    preserved_legacy_cues: list[BilingualCue] = []
    if asr_path.is_file():
        raw_asr = json.loads(asr_path.read_text(encoding="utf-8"))
        sentences = [AsrSentence.model_validate(item) for item in raw_asr]
    elif cues_path.is_file():
        raw_cues = json.loads(cues_path.read_text(encoding="utf-8"))
        for item in raw_cues:
            cue = BilingualCue.model_validate(item)
            src_text = cue.en or cue.zh
            if src_text.lstrip().startswith("["):
                preserved_legacy_cues.append(cue)
            else:
                sentences.append(AsrSentence(start_ms=cue.start_ms, end_ms=cue.end_ms, text=src_text))
    else:
        raise FileNotFoundError(f"Neither asr.json nor cues.json exists in {cache_dir}")

    sentences.sort(key=lambda s: (s.start_ms, s.end_ms))

    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            loaded_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded_meta, dict):
                meta = loaded_meta
        except json.JSONDecodeError:
            meta = {}

    tecl_id = meta.get("tecl_id")
    course_name = str(meta.get("course_name") or "")
    glossary: list[str] = []
    extra_prompt = ""
    if isinstance(tecl_id, int):
        pref = get_course_preference(tecl_id, root_dir=root_dir)
        course_name = course_name or pref.name
        glossary = pref.glossary
        extra_prompt = pref.extra_prompt

    owns_translator = translator is None
    if translator is None:
        api_key = config.effective_llm_api_key()
        if not api_key:
            raise RuntimeError("LLM API Key 未配置，请检查 config.yaml 中的 dashscope.api_key / llm_api_key。")
        translator = LLMTranslator(
            api_key=api_key,
            base_url=config.effective_llm_base_url(),
            model=config.dashscope.llm_model,
            proxy=config.dashscope.llm_proxy,
            enable_thinking=config.dashscope.llm_enable_thinking,
        )

    try:
        batch_size = max(1, int(config.pipeline.translation_batch_size))
        groups = _group_sentences_for_retranslate(
            sentences,
            max_span_ms=max(60_000, int(config.pipeline.chunk_seconds) * 1000),
            max_items=max(batch_size, batch_size * 3),
        )
        sem = asyncio.Semaphore(max(1, int(config.pipeline.max_concurrency)))

        async def _run_group(idx: int, grp: list[AsrSentence]) -> tuple[int, list[BilingualCue]]:
            async with sem:
                translated = await translator.refine_and_translate(
                    grp,
                    course_name=course_name,
                    glossary=glossary,
                    extra_prompt=extra_prompt,
                    ppt_context="",
                    batch_size=batch_size,
                )
                return idx, translated

        group_results = await asyncio.gather(*(_run_group(i, g) for i, g in enumerate(groups)))
        group_results.sort(key=lambda pair: pair[0])
        all_cues = [cue for _, cues in group_results for cue in cues]
        if preserved_legacy_cues:
            all_cues.extend(preserved_legacy_cues)
        all_cues.sort(key=lambda c: (c.start_ms, c.end_ms))

        _atomic_write_json(cues_path, [c.model_dump(mode="json") for c in all_cues])
        write_ass_atomic(sub_path, render_ass(all_cues, config.subtitle))
        if meta:
            meta["updated_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_write_json(meta_path, meta)
        return len(all_cues)
    finally:
        if owns_translator:
            await translator.close()


def cmd_check(root_dir: Path) -> int:
    """Validate local executables, protocol registration, and API configuration."""
    config = load_config(root_dir)
    tools = resolve_tool_paths(config)
    api_key_ready = bool(config.effective_dashscope_api_key())
    proto_ready = is_protocol_registered()

    print(f"Workspace : {root_dir}")
    print(f"mpv       : {tools['mpv'] or 'MISSING'}")
    print(f"ffmpeg    : {tools['ffmpeg'] or 'MISSING'}")
    print(f"ffprobe   : {tools['ffprobe'] or 'MISSING'}")
    print(f"Protocol  : {'REGISTERED (sjtu-mpv://)' if proto_ready else 'NOT REGISTERED'}")
    if api_key_ready:
        print("DashScope : CONFIGURED")
    else:
        print("DashScope : NOT CONFIGURED (awaits dashscope.api_key in config.yaml or DASHSCOPE_API_KEY)")

    required_ok = bool(tools["mpv"] and tools["ffmpeg"] and tools["ffprobe"] and proto_ready)
    return 0 if required_ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sjtu-mpv-sub", description="SJTU lecture bilingual mpv companion CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    serve_p = sub.add_parser("serve", help="Start the local FastAPI companion server")
    serve_p.add_argument("--ephemeral", action="store_true", help="Exit automatically when idle and mpv closes")

    sub.add_parser("check", help="Check mpv, ffmpeg, ffprobe, protocol registration, and API config")
    sub.add_parser("install-protocol", help="Register sjtu-mpv:// URL protocol in HKCU")
    sub.add_parser("uninstall-protocol", help="Remove sjtu-mpv:// URL protocol from HKCU")

    rerender_p = sub.add_parser("rerender", help="Re-render bilingual.ass from cached cues.json using config.yaml")
    rerender_p.add_argument("--session", default=None, help="Specific session cache directory name (default: all)")
    rerender_p.add_argument("--all", action="store_true", help="Re-render all cached sessions")

    retranslate_p = sub.add_parser(
        "retranslate",
        help="Re-translate a cached session with LLM only (uses asr.json if available, else cues.json)",
    )
    retranslate_p.add_argument("session", help="Session cache directory name under data/cache/")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root_dir = get_root_dir()

    if args.command == "serve":
        run_server(ephemeral=bool(args.ephemeral), root_dir=root_dir)
        return 0
    if args.command == "check":
        return cmd_check(root_dir)
    if args.command == "install-protocol":
        install_protocol(root_dir)
        print(f"Registered {PROTOCOL_SCHEME}:// -> {_protocol_command(root_dir)}")
        return 0
    if args.command == "uninstall-protocol":
        uninstall_protocol()
        print(f"Unregistered {PROTOCOL_SCHEME}://")
        return 0
    if args.command == "rerender":
        target_session = None if args.all else args.session
        cfg = load_config(root_dir)
        rendered = rerender_sessions(root_dir, session_key=target_session)
        for sess, count in rendered:
            print(
                f"[rerender] {sess}: {count} cues -> bilingual.ass "
                f"(style={cfg.subtitle.style}, order={cfg.subtitle.bilingual_order})"
            )
        return 0
    if args.command == "retranslate":
        t0 = time.monotonic()
        count = asyncio.run(retranslate_session_async(root_dir, args.session))
        elapsed = time.monotonic() - t0
        print(f"[retranslate] {args.session}: {count} cues updated in {elapsed:.1f}s")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
