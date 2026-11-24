"""Command-line interface for sjtu-mpv-sub."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sjtu_mpv_sub.config import get_root_dir, load_config, resolve_tool_paths
from sjtu_mpv_sub.server import run_server

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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
