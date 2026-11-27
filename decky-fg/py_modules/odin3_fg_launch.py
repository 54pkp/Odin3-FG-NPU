#!/usr/bin/python3 -I
"""Self-contained host game launcher for Odin 3 Pure-Neural Frame Generation (OmniShift HTP v79).

Installed to ~/.local/share/odin3-fg/bin/odin3-fg-launch.
Safe after plugin uninstall: if no active runtime or profile multiplier == 0, execs the game untouched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

DONE_ENV = "ODIN3_FG_LAUNCH_DONE"
APPID_KEYS = ("STEAM_COMPAT_APP_ID", "SteamAppId", "SteamGameId")
LOG_MAX = 128 * 1024


def is_assignment(word: str) -> bool:
    name, sep, _ = word.partition("=")
    return bool(sep) and bool(name) and (name[0].isalpha() or name[0] == "_") and all(
        c.isalnum() or c == "_" for c in name
    )


def log_line(home: Path, message: str) -> None:
    try:
        cache_dir = Path(os.environ.get("XDG_CACHE_HOME") or (home / ".cache")) / "odin3-fg"
        cache_dir.mkdir(parents=True, exist_ok=True)
        log_file = cache_dir / "launch.log"
        if log_file.is_file() and log_file.stat().st_size > LOG_MAX:
            log_file.replace(cache_dir / "launch.log.1")
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except OSError:
        pass


def resolve_app_id(env: dict[str, str]) -> int | None:
    for key in APPID_KEYS:
        raw = env.get(key, "").strip()
        if raw.isdigit() and raw != "0":
            val = int(raw)
            if key == "SteamGameId" and val >= 2**32:
                val >>= 32
            if 1 <= val <= 2**32 - 1:
                return val
    compat = env.get("STEAM_COMPAT_DATA_PATH", "").strip().rstrip("/")
    if compat:
        base = os.path.basename(compat)
        if base.isdigit() and 1 <= int(base) <= 2**32 - 1:
            return int(base)
    return None


def sync_vulkan_implicit_manifest(home: Path, active_release: Path) -> None:
    lib_so = active_release / "lib/libmako-render.so"
    if not lib_so.is_file():
        return
    layer_dir = Path(os.environ.get("XDG_DATA_HOME") or (home / ".local/share")) / "vulkan/implicit_layer.d"
    layer_dir.mkdir(parents=True, exist_ok=True)
    target = layer_dir / "VkLayer_MAKO_render.json"
    payload = json.dumps(
        {
            "file_format_version": "1.2.1",
            "layer": {
                "name": "VK_LAYER_MAKO_render",
                "description": "Odin3 Pure-Neural VFI graphics layer (SM8750 HTP v79)",
                "implementation_version": "2",
                "library_path": str(lib_so),
                "library_arch": "64",
                "type": "GLOBAL",
                "api_version": "1.4.365",
                "instance_extensions": [{"name": "VK_EXT_swapchain_colorspace", "spec_version": "4"}],
                "enable_environment": {"ENABLE_MAKO": "1"},
                "disable_environment": {"DISABLE_MAKO": "1"},
            },
        },
        indent=2,
    ) + "\n"
    try:
        if target.is_file() and target.read_text(encoding="utf-8") == payload:
            return
        tmp = layer_dir / f".VkLayer_MAKO_render.{os.getpid()}.tmp"
        tmp.write_text(payload, encoding="utf-8")
        tmp.chmod(0o644)
        tmp.replace(target)
    except OSError:
        pass


def read_xprop_cardinal(prop: str, env: dict[str, str]) -> int | None:
    try:
        out = subprocess.check_output(
            ["xprop", "-display", env.get("DISPLAY", ":0"), "-root", prop],
            env=env,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        tok = out.strip().split()[-1]
        return int(tok) if tok.isdigit() else None
    except Exception:
        return None


def apply_and_watch_display_mode(target_hz: int, home: Path) -> None:
    uid = os.getuid()
    # Always target Gamescope's control XWayland server (:0), NEVER the game XWayland server (:1)
    # passed in DISPLAY by Steam, because touching :1 before Proton starts resets on-demand XWayland!
    ctrl_display = ":0"
    xenv = {
        **os.environ,
        "DISPLAY": ctrl_display,
        "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{uid}",
    }
    prev_fps = read_xprop_cardinal("GAMESCOPE_FPS_LIMIT", xenv)
    prev_dyn_hz = read_xprop_cardinal("GAMESCOPE_DYNAMIC_REFRESH", xenv)
    prev_hdr = read_xprop_cardinal("GAMESCOPE_DISPLAY_HDR_ENABLED", xenv)
    prev_redzone = read_xprop_cardinal("GAMESCOPE_TUNEABLE_VBLANK_REDZONE", xenv)
    fps_limit_arg = "0" if target_hz == 120 else str(target_hz)
    fps_limit_int = int(fps_limit_arg)
    if (
        prev_dyn_hz == target_hz
        and prev_fps == fps_limit_int
        and prev_hdr in (0, None)
        and prev_redzone is None
    ):
        log_line(home, f"display guard already at target on {ctrl_display}: hz={target_hz} fps_limit={fps_limit_arg} hdr=0")
        return
    try:
        for prop_name, prop_val in (
            ("GAMESCOPE_DYNAMIC_REFRESH", str(target_hz)),
            ("GAMESCOPE_FPS_LIMIT", fps_limit_arg),
            ("GAMESCOPE_DISPLAY_HDR_ENABLED", "0"),
        ):
            subprocess.run(
                [
                    "xprop", "-display", ctrl_display, "-root",
                    "-f", prop_name, "32c",
                    "-set", prop_name, prop_val,
                ],
                env=xenv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
                check=False,
            )
        if prev_redzone is not None:
            subprocess.run(["xprop", "-display", ctrl_display, "-root", "-remove", "GAMESCOPE_TUNEABLE_VBLANK_REDZONE"], env=xenv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2.0, check=False)
        subprocess.run(
            ["gamescopectl", "debug_set_fps_limit", fps_limit_arg],
            env=xenv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            check=False,
        )
    except Exception:
        return

    if prev_fps is None and prev_dyn_hz is None and prev_hdr is None:
        return

    watcher_code = """
import os, subprocess, sys, time
pid, disp, rt_dir, p_fps, p_dyn, p_hdr, p_rz = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
env = {"PATH": "/usr/bin:/bin", "DISPLAY": disp, "XDG_RUNTIME_DIR": rt_dir}
while os.path.exists(f"/proc/{pid}"):
    time.sleep(1.0)
time.sleep(0.4)
if p_dyn != "None":
    subprocess.run(["xprop", "-display", disp, "-root", "-f", "GAMESCOPE_DYNAMIC_REFRESH", "32c", "-set", "GAMESCOPE_DYNAMIC_REFRESH", p_dyn], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if p_fps != "None":
    subprocess.run(["xprop", "-display", disp, "-root", "-f", "GAMESCOPE_FPS_LIMIT", "32c", "-set", "GAMESCOPE_FPS_LIMIT", p_fps], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["gamescopectl", "debug_set_fps_limit", p_fps], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if p_hdr != "None":
    subprocess.run(["xprop", "-display", disp, "-root", "-f", "GAMESCOPE_DISPLAY_HDR_ENABLED", "32c", "-set", "GAMESCOPE_DISPLAY_HDR_ENABLED", p_hdr], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if p_rz != "None":
    subprocess.run(["xprop", "-display", disp, "-root", "-f", "GAMESCOPE_TUNEABLE_VBLANK_REDZONE", "32c", "-set", "GAMESCOPE_TUNEABLE_VBLANK_REDZONE", p_rz], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
else:
    subprocess.run(["xprop", "-display", disp, "-root", "-remove", "GAMESCOPE_TUNEABLE_VBLANK_REDZONE"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
"""
    try:
        subprocess.Popen(
            [
                "/usr/bin/python3", "-I", "-c", watcher_code,
                str(os.getpid()), xenv["DISPLAY"], xenv["XDG_RUNTIME_DIR"],
                str(prev_fps), str(prev_dyn_hz), str(prev_hdr), str(prev_redzone),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    except Exception:
        pass
    log_line(
        home,
        f"display guard armed: prev_fps={prev_fps} prev_dyn={prev_dyn_hz} prev_hdr={prev_hdr} "
        f"-> target_hz={target_hz} fps_limit={fps_limit_arg} hdr=0",
    )


def main() -> None:
    args = sys.argv[1:]
    if args[:1] == ["--"]:
        args = args[1:]
    while args and is_assignment(args[0]):
        k, _, v = args.pop(0).partition("=")
        os.environ[k] = v
    if not args:
        raise SystemExit("Usage: odin3-fg-launch [--] GAME_COMMAND [ARGS...]")

    if os.environ.get(DONE_ENV) == "1":
        os.execvp(args[0], args)
    os.environ[DONE_ENV] = "1"

    home = Path(os.environ.get("HOME") or "/var/home/armada")
    data_root = Path(os.environ.get("ODIN3_FG_DATA_ROOT") or (
        Path(os.environ.get("XDG_DATA_HOME") or (home / ".local/share")) / "odin3-fg"
    ))
    config_root = Path(os.environ.get("XDG_CONFIG_HOME") or (home / ".config")) / "odin3-fg"

    app_id = resolve_app_id(os.environ)
    if app_id is None:
        log_line(home, "bypass: no Steam AppID found in environment")
        os.execvp(args[0], args)

    profile_file = Path(os.environ.get("ODIN3_FG_PROFILES_DIR") or (config_root / "profiles")) / f"{app_id}.json"
    try:
        profile = json.loads(profile_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        log_line(home, f"bypass appid={app_id}: no saved profile")
        os.execvp(args[0], args)
    except Exception as exc:
        log_line(home, f"bypass appid={app_id}: unreadable profile ({exc})")
        os.execvp(args[0], args)

    multiplier = profile.get("multiplier", 0)
    if multiplier not in (2, 3):
        log_line(home, f"passthrough appid={app_id}: multiplier={multiplier}")
        os.execvp(args[0], args)

    installer = data_root / "install_runtime.py"
    active_file = data_root / "active.json"
    try:
        active_sha = json.loads(active_file.read_text(encoding="utf-8")).get("active")
        active_release = data_root / "versions" / active_sha if isinstance(active_sha, str) else None
    except Exception:
        active_release = None

    if not installer.is_file() or active_release is None or not active_release.is_dir():
        log_line(home, f"bypass appid={app_id}: active runtime missing in {data_root}")
        os.execvp(args[0], args)

    sync_vulkan_implicit_manifest(home, active_release)

    target_hz = profile.get("target_refresh_hz", 120)
    if target_hz not in (60, 120):
        target_hz = 120
    auto_disp = profile.get("auto_display_mode", True) is not False
    if auto_disp:
        apply_and_watch_display_mode(target_hz, home)

    os.environ["ODIN3_FG_APP_ID"] = str(app_id)
    os.environ["ODIN3_FG_DATA_ROOT"] = str(data_root)
    os.environ.setdefault("ODIN3_FG_REFRESH_HZ", str(target_hz))

    pv_ro = os.environ.get("PRESSURE_VESSEL_FILESYSTEMS_RO", "").strip()
    if str(data_root) not in pv_ro.split(":"):
        os.environ["PRESSURE_VESSEL_FILESYSTEMS_RO"] = f"{pv_ro}:{data_root}" if pv_ro else str(data_root)

    log_line(
        home,
        f"launching appid={app_id} multiplier={multiplier}x cap={profile.get('source_fps_cap')} "
        f"hz={target_hz} auto_disp={auto_disp} release={active_release.name[:12]}",
    )
    os.execv(
        "/usr/bin/python3",
        ["/usr/bin/python3", "-I", str(installer), "--root", str(data_root), "launch", "--", *args],
    )


if __name__ == "__main__":
    main()
