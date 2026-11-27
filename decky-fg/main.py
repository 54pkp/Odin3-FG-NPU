"""Decky system plugin backend for Odin 3 Pure-Neural Frame Generation (OmniShift HTP v79)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys

import decky

PLUGIN_DIR = Path(__file__).resolve().parent
if str(PLUGIN_DIR / "py_modules") not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR / "py_modules"))

import launch_sync  # noqa: E402


def steam_user():
    account = pwd.getpwnam(decky.DECKY_USER)
    if account.pw_uid == 0 or Path(account.pw_dir).resolve() != Path(decky.DECKY_USER_HOME).resolve():
        raise RuntimeError("Decky did not identify the Steam user")
    if os.geteuid() not in (0, account.pw_uid):
        raise RuntimeError("Decky is running as a different user")
    return account


def app_argument(app_id) -> int:
    if type(app_id) is not int or not 1 <= app_id <= 2**32 - 1:
        raise ValueError("Expected a positive Steam AppID")
    return app_id


def ensure_owned_dirs(paths: list[Path], uid: int, gid: int, mode: int = 0o755) -> None:
    for p in paths:
        p.mkdir(mode=mode, parents=True, exist_ok=True)
        try:
            os.chown(p, uid, gid)
        except OSError:
            pass


PANEL_FRAME_LEN_120_TARGET = "56"  # Thermally compensated 120.02-120.14 Hz across 45-80C (factory 76 = 118.96 Hz)


def calibrate_odin3_panel_120hz() -> int | None:
    candidates = [
        Path("/sys/devices/platform/soc@0/ae00000.display-subsystem/ae94000.dsi/ae94000.dsi.0/frame_len_120"),
    ]
    node = next((p for p in candidates if p.is_file()), None)
    if node is None:
        try:
            node = next(Path("/sys/devices/platform").rglob("frame_len_120"), None)
        except Exception:
            node = None
    if node is None:
        return None
    try:
        cur = node.read_text(encoding="utf-8").strip()
        if os.geteuid() == 0 and cur != PANEL_FRAME_LEN_120_TARGET:
            node.write_text(PANEL_FRAME_LEN_120_TARGET + "\n", encoding="utf-8")
            cur = node.read_text(encoding="utf-8").strip()
        if os.geteuid() == 0:
            tmpf = Path("/etc/tmpfiles.d/odin3-fg-panel-120hz.conf")
            rule = f"w {node} - - - - {PANEL_FRAME_LEN_120_TARGET}\n"
            try:
                if not tmpf.is_file() or tmpf.read_text(encoding="utf-8") != rule:
                    tmpf.write_text(rule, encoding="utf-8")
            except OSError:
                pass
        return int(cur) if cur.isdigit() else None
    except Exception as exc:
        decky.logger.warning(f"Panel 120Hz calibration check warning: {exc}")
        return None


def sync_host_integration(account) -> dict:
    panel_frame_len = calibrate_odin3_panel_120hz()
    home = Path(account.pw_dir)
    uid, gid = account.pw_uid, account.pw_gid
    data_root = home / ".local/share/odin3-fg"
    bin_dir = data_root / "bin"
    ensure_owned_dirs([home / ".local", home / ".local/share", data_root, bin_dir], uid, gid)

    # 1. Install self-contained ~/.local/share/odin3-fg/bin/odin3-fg-launch
    launcher_src = PLUGIN_DIR / "py_modules/odin3_fg_launch.py"
    launcher_dst = bin_dir / "odin3-fg-launch"
    if launcher_src.is_file():
        src_bytes = launcher_src.read_bytes()
        need_write = True
        if launcher_dst.is_file():
            try:
                need_write = launcher_dst.read_bytes() != src_bytes or not os.access(launcher_dst, os.X_OK)
            except OSError:
                need_write = True
        if need_write:
            tmp = bin_dir / f".odin3-fg-launch.{os.getpid()}.tmp"
            tmp.write_bytes(src_bytes)
            tmp.chmod(0o755)
            os.chown(tmp, uid, gid)
            tmp.replace(launcher_dst)

    # 2. Check active runtime + pure-neural SM8750 bundle
    active_sha = None
    active_release = None
    try:
        active_data = json.loads((data_root / "active.json").read_text(encoding="utf-8"))
        if isinstance(active_data.get("active"), str):
            cand = data_root / "versions" / active_data["active"]
            if (cand / "lib/libmako-render.so").is_file() and (cand / "odin3-fg-worker").is_file():
                active_sha = active_data["active"]
                active_release = cand
    except Exception:
        pass

    bundle_dir = data_root / "models/omnishift-sm8750"
    if not bundle_dir.is_dir():
        bundle_dir = data_root / "models/gen4-pure-sm8750"
    resolutions = []
    for res in ("360p", "540p", "720p", "900p", "1080p"):
        rdir = bundle_dir / res
        if (
            (rdir / "vfi-motion3x.bin").is_file()
            and (rdir / "phase12.bin").is_file()
            and (rdir / "phase13.bin").is_file()
            and (rdir / "phase23.bin").is_file()
        ):
            resolutions.append(res)

    qairt_ok = (data_root / "qairt/lib/libQnnHtp.so").is_file() and (data_root / "qairt/hexagon/libQnnHtpV79Skel.so").is_file()

    # 3. Sync ~/.local/share/vulkan/implicit_layer.d/VkLayer_MAKO_render.json so pressure-vessel imports it
    if active_release is not None:
        vk_dir = home / ".local/share/vulkan"
        imp_dir = vk_dir / "implicit_layer.d"
        ensure_owned_dirs([vk_dir, imp_dir], uid, gid)
        manifest_path = imp_dir / "VkLayer_MAKO_render.json"
        payload = json.dumps(
            {
                "file_format_version": "1.2.1",
                "layer": {
                    "name": "VK_LAYER_MAKO_render",
                    "description": "Odin3 Pure-Neural VFI graphics layer (SM8750 HTP v79)",
                    "implementation_version": "2",
                    "library_path": str(active_release / "lib/libmako-render.so"),
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
            if not manifest_path.is_file() or manifest_path.read_text(encoding="utf-8") != payload:
                tmp_m = imp_dir / f".VkLayer_MAKO_render.{os.getpid()}.tmp"
                tmp_m.write_text(payload, encoding="utf-8")
                tmp_m.chmod(0o644)
                os.chown(tmp_m, uid, gid)
                tmp_m.replace(manifest_path)
        except OSError as exc:
            decky.logger.error(f"Failed to sync VkLayer_MAKO_render.json: {exc}")

    return {
        "ready": bool(active_release is not None and len(resolutions) == 5 and qairt_ok),
        "active_version": active_sha[:12] if active_sha else None,
        "resolutions": resolutions,
        "qairt_ok": qairt_ok,
        "launch_token": launch_sync.LAUNCHER_TOKEN,
        "panel_frame_len_120": panel_frame_len,
    }


def run_profiles(account, *args: str):
    cli = PLUGIN_DIR / "profiles.py"
    if not cli.is_file():
        raise RuntimeError("profiles.py is missing from the plugin directory")
    env = {
        "HOME": account.pw_dir,
        "USER": account.pw_name,
        "LOGNAME": account.pw_name,
        "PATH": "/usr/bin:/bin",
        "XDG_RUNTIME_DIR": f"/run/user/{account.pw_uid}",
    }
    command = ["/usr/bin/python3", "-I", str(cli), *args]
    if os.geteuid() == 0:
        command = [
            "/usr/bin/runuser", "-u", account.pw_name, "--", "/usr/bin/env", "-i",
            *(f"{key}={value}" for key, value in env.items()), *command,
        ]
    result = subprocess.run(command, env=env, capture_output=True, encoding="utf-8", timeout=5)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "The profile helper failed")
    return json.loads(result.stdout) if result.stdout else None


PRESET_NAMES = ("eco", "saver", "balanced", "auto", "high", "max", "custom", "passthrough")


def read_npu_preset(home: Path) -> str:
    import struct
    best_seq = None
    best_preset = "max"
    for p in (Path("/run/odin3-npu-governor/config.bin"), home / ".config/odin3-npu-governor/config.bin"):
        try:
            raw = p.read_bytes()
            if len(raw) >= 128:
                words = struct.unpack("<32I", raw[:128])
                if words[0] == 0x4F4E5055 and not (words[2] & 1):
                    if best_seq is None or ((words[2] - best_seq) & 0xFFFFFFFF) < 0x80000000:
                        best_seq = words[2]
                        idx = 7 if words[3] == 0 else min(words[4], 7)
                        best_preset = PRESET_NAMES[idx]
        except Exception:
            pass
    return best_preset


def read_tier_runtime(uid: int) -> dict | None:
    import time
    p = Path(f"/run/user/{uid}/odin3-fg-tier.json")
    try:
        if not p.is_file():
            return None
        if time.time() - p.stat().st_mtime > 15.0:
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _status_sync(app_id: int) -> dict:
    account = steam_user()
    engine = sync_host_integration(account)
    engine["npu_preset"] = read_npu_preset(Path(account.pw_dir))
    status = run_profiles(account, "status", str(app_id))
    status["engine"] = engine
    status["tier_runtime"] = read_tier_runtime(account.pw_uid)
    return status


def _save_sync(
    app_id: int,
    multiplier: int,
    source_fps_cap: int | None,
    target_refresh_hz: int,
    auto_display_mode: bool,
    current_launch_options: str | None,
) -> dict:
    account = steam_user()
    engine = sync_host_integration(account)
    args = [
        "set", str(app_id),
        "--multiplier", str(multiplier),
        "--target-refresh-hz", str(target_refresh_hz),
        "--auto-display-mode", "1" if auto_display_mode else "0",
    ]
    if source_fps_cap is not None:
        args.extend(("--source-fps-cap", str(source_fps_cap)))
    run_profiles(account, *args)

    next_opts, warn = launch_sync.sync_launch_options(
        Path(account.pw_dir),
        account.pw_uid,
        account.pw_gid,
        app_id,
        enable=(multiplier in (2, 3)),
        current_options=current_launch_options,
    )
    decky.logger.info(
        f"Saved profile appid={app_id} multiplier={multiplier} cap={source_fps_cap} "
        f"hz={target_refresh_hz} auto_disp={auto_display_mode} "
        f"launch_changed={current_launch_options is not None and next_opts != current_launch_options}"
    )
    return {
        "launch_options": next_opts,
        "launch_changed": (current_launch_options is not None and next_opts != current_launch_options),
        "launch_warning": warn,
        "engine": engine,
    }


class Plugin:
    async def _main(self):
        try:
            info = await asyncio.to_thread(lambda: sync_host_integration(steam_user()))
            decky.logger.info(f"Odin3 FG NPU initialized: {info}")
        except Exception as exc:
            decky.logger.error(f"Odin3 FG NPU init warning: {exc}")

    async def _unload(self):
        decky.logger.info("Odin3 FG NPU unloaded")

    async def game_status(self, app_id):
        app = app_argument(app_id)
        return await asyncio.to_thread(_status_sync, app)

    async def save_game_profile(
        self,
        app_id,
        multiplier,
        source_fps_cap,
        target_refresh_hz=120,
        auto_display_mode=True,
        current_launch_options=None,
    ):
        app = app_argument(app_id)
        if type(multiplier) is not int or multiplier not in (0, 2, 3):
            raise ValueError("Choose Off (0), 2x or 3x")
        if source_fps_cap is not None and (type(source_fps_cap) is not int or not 1 <= source_fps_cap <= 1000):
            raise ValueError("Expected an original frame limit from 1 to 1000 FPS")
        if type(target_refresh_hz) is not int or target_refresh_hz not in (60, 120):
            raise ValueError("Target refresh rate must be 60 or 120 Hz")
        if type(auto_display_mode) is not bool:
            raise ValueError("auto_display_mode must be boolean")
        if current_launch_options is not None and not isinstance(current_launch_options, str):
            current_launch_options = None
        return await asyncio.to_thread(
            _save_sync, app, multiplier, source_fps_cap, target_refresh_hz, auto_display_mode, current_launch_options
        )
