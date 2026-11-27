"""Decky backend for Odin3 NPU Power: performance / power tiers for the SM8750 Hexagon HTP (NPU).

The NPU has no Linux devfreq node: clocks, voltage corners, HMX gating and sleep are voted from user space through
QNN HTP DCVS_V3. Both NPU clients on Odin 3 (odin3-fg-worker = Odin3 FG NPU, vela-npu.service) hard-code the
top corner, so this plugin installs an interposer (libodin3-npu-governor.so, named libQnnHtp.so) in front of Qualcomm's
library inside each client. The tier is written to a seqlock config.bin that every client polls (hot switch, no restart).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import pwd
import shutil
import struct
import subprocess
import time

import decky

PLUGIN_DIR = Path(__file__).resolve().parent
SHIM_SRC = PLUGIN_DIR / "assets/libodin3-npu-governor.so"
MARKER = b"ODIN3_NPU_GOVERNOR_SHIM_V1"

STATE_DIR = Path("/var/lib/odin3-npu-governor")
VELA_LIB_DIR = STATE_DIR / "vela-lib"
VELA_REAL = STATE_DIR / "libQnnHtp.vela.real.so"
GOV_SO = STATE_DIR / "libodin3-npu-governor.so"
RUN_DIR = Path("/run/odin3-npu-governor")
TEL_DIR = RUN_DIR / "telemetry"
TMPFILES = Path("/etc/tmpfiles.d/odin3-npu-governor.conf")
VELA_DROPIN = Path("/etc/systemd/system/vela-npu.service.d/60-odin3-npu-governor.conf")

CFG_MAGIC = 0x4F4E5055
TEL_MAGIC = 0x4F4E544D
CFG_FMT = "<32I"
TEL_FMT = "<IIQ7I4f4xQ32s6I"
assert struct.calcsize(CFG_FMT) == 128 and struct.calcsize(TEL_FMT) == 128

DISABLE, SVS2, SVS, SVS_P, NOM, NOM_P, TURBO, TURBO_P, MAXC = 0x10, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0
PM_ADJUST, PM_SAVER, PM_SAVER_AGG, PM_PERF = 0x01, 0x04, 0x08, 0x10
CORNER_NAMES = {
    0x10: "OFF", 0x20: "MIN", 0x30: "SVS2", 0x40: "SVS", 0x50: "SVS+", 0x60: "NOM", 0x70: "NOM+",
    0x80: "TURBO", 0x90: "TURBO+", 0xA0: "MAX",
}
PRESET_IDS = {"eco": 0, "saver": 1, "balanced": 2, "auto": 3, "high": 4, "max": 5, "custom": 6, "passthrough": 7}
CORE_CHOICES = (SVS, NOM, TURBO, TURBO_P, MAXC)
BUS_CHOICES = (SVS, NOM, TURBO, MAXC)

DEFAULT_SETTINGS = {
    "preset": "auto",
    "core_max": MAXC,
    "bus_max": TURBO_P,
    "sleep_enable": True,
    "auto_target_pct": 75,
}


def steam_user():
    account = pwd.getpwnam(decky.DECKY_USER)
    if account.pw_uid == 0:
        raise RuntimeError("Decky did not identify the Steam user")
    return account


def user_dirs(account):
    home = Path(account.pw_dir)
    return home / ".config/odin3-npu-governor", home / ".cache/odin3-npu-governor/telemetry", home


def corner_name(value: int) -> str:
    return CORNER_NAMES.get(int(value), f"0x{int(value):02X}")


def build_fields(settings: dict) -> dict:
    preset = settings["preset"]
    f = dict(
        mode=1, preset_id=PRESET_IDS[preset], dcvs_enable=1, power_mode=PM_ADJUST, sleep_disable=0,
        sleep_latency_us=600, rpc_latency_us=350, hmx_timeout_us=250,
        core_min=SVS, core_target=TURBO, core_max=TURBO, bus_min=SVS2, bus_target=SVS, bus_max=NOM_P,
        auto_target_pct=int(settings.get("auto_target_pct", 75)),
    )
    if preset == "passthrough":
        f["mode"] = 0
    elif preset == "eco":
        f.update(power_mode=PM_SAVER_AGG, sleep_latency_us=2000, rpc_latency_us=1000, hmx_timeout_us=200,
                 core_min=SVS, core_target=SVS, core_max=SVS_P, bus_min=SVS2, bus_target=SVS, bus_max=SVS)
    elif preset == "saver":
        f.update(power_mode=PM_SAVER, sleep_latency_us=1000, rpc_latency_us=600, hmx_timeout_us=300,
                 core_min=SVS, core_target=NOM_P, core_max=NOM_P, bus_min=SVS, bus_target=NOM, bus_max=NOM)
    elif preset == "balanced":
        pass
    elif preset == "auto":
        f.update(mode=2, core_max=int(settings["core_max"]), bus_max=int(settings["bus_max"]),
                 core_target=int(settings["core_max"]), bus_target=int(settings["bus_max"]))
    elif preset == "high":
        f.update(power_mode=PM_ADJUST, sleep_latency_us=600, rpc_latency_us=350, hmx_timeout_us=200,
                 core_min=SVS, core_target=TURBO, core_max=MAXC, bus_min=SVS2, bus_target=SVS, bus_max=NOM_P)
    elif preset == "max":
        f.update(dcvs_enable=1, power_mode=PM_ADJUST, sleep_disable=0, sleep_latency_us=600, rpc_latency_us=350,
                 hmx_timeout_us=200, core_min=SVS, core_target=TURBO_P, core_max=MAXC,
                 bus_min=SVS2, bus_target=SVS, bus_max=NOM_P)
    elif preset == "custom":
        core, bus = int(settings["core_max"]), int(settings["bus_max"])
        sleep_on = bool(settings.get("sleep_enable", True))
        f.update(power_mode=PM_PERF if core >= TURBO else PM_ADJUST, sleep_disable=0 if sleep_on else 1,
                 sleep_latency_us=150, rpc_latency_us=100, hmx_timeout_us=600 if sleep_on else 0,
                 core_min=min(SVS, core), core_target=core, core_max=core,
                 bus_min=min(SVS, bus), bus_target=bus, bus_max=bus)
    return f


def pack_config(f: dict, seq: int) -> bytes:
    words = [
        CFG_MAGIC, 1, seq, f["mode"], f["preset_id"], f["dcvs_enable"], f["power_mode"], f["sleep_disable"],
        f["sleep_latency_us"], f["rpc_latency_us"], f["hmx_timeout_us"], f["core_min"], f["core_target"], f["core_max"],
        f["bus_min"], f["bus_target"], f["bus_max"], f["auto_target_pct"],
    ]
    words += [0] * (32 - len(words))
    return struct.pack(CFG_FMT, *words)


def read_seq(path: Path) -> int:
    try:
        data = path.read_bytes()[:128]
        if len(data) == 128:
            words = struct.unpack(CFG_FMT, data)
            if words[0] == CFG_MAGIC:
                return words[2] & ~1
    except OSError:
        pass
    return 0


def write_config_inplace(path: Path, f: dict, new_seq: int, uid: int, gid: int) -> None:
    """Seqlock update: odd seq -> payload -> even seq, always in the same inode so mmap readers see it."""
    payload_even = pack_config(f, new_seq)
    if not path.is_file() or path.stat().st_size < 128:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(payload_even)
        os.chmod(tmp, 0o644)
        os.chown(tmp, uid, gid)
        os.replace(tmp, path)
        return
    fd = os.open(path, os.O_RDWR)
    try:
        os.pwrite(fd, struct.pack("<I", new_seq - 1), 8)
        os.pwrite(fd, payload_even[12:], 12)
        os.pwrite(fd, struct.pack("<I", new_seq), 8)
    finally:
        os.close(fd)


def load_settings(cfg_dir: Path) -> dict:
    s = dict(DEFAULT_SETTINGS)
    try:
        data = json.loads((cfg_dir / "settings.json").read_text(encoding="utf-8"))
        if isinstance(data, dict):
            s.update({k: data[k] for k in DEFAULT_SETTINGS if k in data})
    except (OSError, ValueError):
        pass
    if s["preset"] not in PRESET_IDS:
        s["preset"] = "auto"
    if int(s["core_max"]) not in CORE_CHOICES:
        s["core_max"] = MAXC
    if int(s["bus_max"]) not in BUS_CHOICES + (TURBO_P,):
        s["bus_max"] = TURBO_P
    s["sleep_enable"] = bool(s["sleep_enable"])
    s["auto_target_pct"] = DEFAULT_SETTINGS["auto_target_pct"]
    return s


def save_settings(cfg_dir: Path, s: dict, uid: int, gid: int) -> None:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    os.chown(cfg_dir, uid, gid)
    p = cfg_dir / "settings.json"
    tmp = cfg_dir / ".settings.tmp"
    tmp.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
    os.chown(tmp, uid, gid)
    os.replace(tmp, p)


def write_all_configs(account, settings: dict) -> None:
    cfg_dir, _, _ = user_dirs(account)
    fields = build_fields(settings)
    run_cfg, user_cfg = RUN_DIR / "config.bin", cfg_dir / "config.bin"
    new_seq = max(read_seq(run_cfg), read_seq(user_cfg)) + 2
    write_config_inplace(user_cfg, fields, new_seq, account.pw_uid, account.pw_gid)
    write_config_inplace(run_cfg, fields, new_seq, 0, 0)


def is_shim(path: Path) -> bool:
    try:
        return MARKER in path.read_bytes()
    except OSError:
        return False


def atomic_copy(src: Path, dst: Path, mode: int, uid: int = 0, gid: int = 0) -> bool:
    """Returns True when dst changed."""
    data = src.read_bytes()
    try:
        if dst.is_file() and dst.read_bytes() == data:
            os.chmod(dst, mode)
            return False
    except OSError:
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.chmod(tmp, mode)
    os.chown(tmp, uid, gid)
    os.replace(tmp, dst)
    return True


def systemctl(*args: str, timeout: float = 20.0) -> subprocess.CompletedProcess:
    return subprocess.run(["/usr/bin/systemctl", *args], capture_output=True, text=True, timeout=timeout)


def sync_install(account) -> dict:
    uid, gid = account.pw_uid, account.pw_gid
    cfg_dir, cache_tel, home = user_dirs(account)
    info = {"shim_installed": False, "fg_hooked": False, "vela_dropin": False, "vela_restarted": False}
    if not SHIM_SRC.is_file():
        raise RuntimeError(f"missing {SHIM_SRC}")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(STATE_DIR, 0o755)
    atomic_copy(SHIM_SRC, STATE_DIR / "libodin3-npu-governor.so", 0o755)
    shutil.rmtree(VELA_LIB_DIR, ignore_errors=True)  # obsolete LD_LIBRARY_PATH approach
    vela_changed = False
    vela_real_src = Path("/usr/lib64/vela-npu/libQnnHtp.so")
    if vela_real_src.is_file() and not is_shim(vela_real_src):
        vela_changed = atomic_copy(vela_real_src, VELA_REAL, 0o755)
    info["shim_installed"] = True

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(RUN_DIR, 0o755)
    TEL_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(TEL_DIR, 0o1777)
    rule = "d /run/odin3-npu-governor 0755 root root -\nd /run/odin3-npu-governor/telemetry 1777 root root -\n"
    try:
        if not TMPFILES.is_file() or TMPFILES.read_text(encoding="utf-8") != rule:
            TMPFILES.write_text(rule, encoding="utf-8")
    except OSError as exc:
        decky.logger.warning(f"tmpfiles: {exc}")

    cfg_dir.mkdir(parents=True, exist_ok=True)
    cache_tel.mkdir(parents=True, exist_ok=True)
    for d in (cfg_dir, home / ".cache/odin3-npu-governor", cache_tel):
        try:
            os.chown(d, uid, gid)
        except OSError:
            pass

    settings = load_settings(cfg_dir)
    if not (cfg_dir / "settings.json").is_file():
        save_settings(cfg_dir, settings, uid, gid)
    write_all_configs(account, settings)

    # Odin3 FG NPU (odin3-fg-worker) loads qairt/lib/libQnnHtp.so
    qlib = home / ".local/share/odin3-fg/qairt/lib"
    q_so, q_real = qlib / "libQnnHtp.so", qlib / "libQnnHtp.real.so"
    if q_so.is_file():
        if not is_shim(q_so):
            os.replace(q_so, q_real)
        if q_real.is_file():
            atomic_copy(SHIM_SRC, q_so, 0o755, uid, gid)
    info["fg_hooked"] = q_real.is_file() and is_shim(q_so)

    # vela-npu.service: bind our shim over /usr/lib64/vela-npu/libQnnHtp.so inside the service mount namespace
    # (same mechanism as ArmadaOS's own 50-vela-npu-chip.conf); the original is exposed at its /var/lib copy.
    dropin = (
        "[Service]\n"
        f"BindReadOnlyPaths=-{GOV_SO}:/usr/lib64/vela-npu/libQnnHtp.so\n"
        f"BindReadOnlyPaths=-{VELA_REAL}\n"
        f"Environment=ODIN3_REAL_QNN_HTP={VELA_REAL}\n"
        "Environment=ODIN3_NPU_CLIENT_NAME=vela-npu\n"
        "ReadWritePaths=-/run/odin3-npu-governor/telemetry\n"
    )
    dropin_changed = False
    try:
        if not VELA_DROPIN.is_file() or VELA_DROPIN.read_text(encoding="utf-8") != dropin:
            VELA_DROPIN.parent.mkdir(parents=True, exist_ok=True)
            VELA_DROPIN.write_text(dropin, encoding="utf-8")
            dropin_changed = True
        info["vela_dropin"] = True
    except OSError as exc:
        decky.logger.warning(f"vela drop-in: {exc}")
    if dropin_changed:
        systemctl("daemon-reload")
    if (dropin_changed or vela_changed or not vela_hooked()) and systemctl("is-active", "vela-npu.service").stdout.strip() == "active":
        r = systemctl("try-restart", "vela-npu.service", timeout=45.0)
        info["vela_restarted"] = r.returncode == 0
    return info


def vela_pid() -> int:
    try:
        out = systemctl("show", "-p", "MainPID", "--value", "vela-npu.service").stdout.strip()
        return int(out) if out.isdigit() else 0
    except Exception:
        return 0


def vela_hooked() -> bool:
    pid = vela_pid()
    if not pid:
        return False
    try:
        maps = Path(f"/proc/{pid}/maps").read_text(errors="replace")
    except OSError:
        return False
    if "odin3-npu-governor" in maps or "libQnnHtp.vela.real.so" in maps:
        return True
    try:
        ino = str(GOV_SO.stat().st_ino)
    except OSError:
        return False
    for line in maps.splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[4] == ino and parts[5].endswith("libQnnHtp.so"):
            return True
    return False


_thermal_cache: dict = {}


def thermal_zones() -> dict:
    if _thermal_cache:
        return _thermal_cache
    groups = {"hmx": [], "hvx": [], "ddr": []}
    for z in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            t = (z / "type").read_text().strip()
        except OSError:
            continue
        if t.startswith("nsphmx"):
            groups["hmx"].append(z / "temp")
        elif t.startswith("nsphvx"):
            groups["hvx"].append(z / "temp")
        elif t.startswith("ddr"):
            groups["ddr"].append(z / "temp")
    _thermal_cache.update(groups)
    return _thermal_cache


def read_num(path: Path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def max_temp(paths) -> float | None:
    vals = [v for v in (read_num(p) for p in paths) if v is not None and v > -50000]
    return round(max(vals) / 1000.0, 1) if vals else None


def read_sensors() -> dict:
    z = thermal_zones()
    ps = Path("/sys/class/power_supply")
    bat, usb = ps / "battery", ps / "qcom-battmgr-usb"
    bat_v = (read_num(bat / "voltage_now") or 0) / 1e6
    bat_i = abs(read_num(bat / "current_now") or 0) / 1e6
    usb_on = (read_num(usb / "online") or 0) == 1
    usb_w = ((read_num(usb / "voltage_now") or 0) / 1e6) * ((read_num(usb / "current_now") or 0) / 1e6)
    try:
        status = (bat / "status").read_text().strip()
    except OSError:
        status = ""
    if usb_on:
        w = usb_w - (bat_v * bat_i if status == "Charging" else 0.0)
        sys_w = max(0.5, w)
        src = "USB-PD"
    else:
        sys_w = bat_v * bat_i
        src = "Battery"
    return {
        "hmx_temp_c": max_temp(z.get("hmx", [])),
        "hvx_temp_c": max_temp(z.get("hvx", [])),
        "ddr_temp_c": max_temp(z.get("ddr", [])),
        "sys_power_w": round(sys_w, 1),
        "power_source": src,
        "battery_pct": read_num(bat / "capacity"),
    }


def read_clients(account) -> list:
    _, cache_tel, _ = user_dirs(account)
    now = time.monotonic_ns()
    seen: dict[int, dict] = {}
    for d in (TEL_DIR, cache_tel):
        try:
            files = list(d.glob("pid-*.bin"))
        except OSError:
            continue
        for f in files:
            try:
                data = f.read_bytes()[:128]
                if len(data) < 128:
                    continue
                w = struct.unpack(TEL_FMT, data)
            except (OSError, struct.error):
                continue
            magic, _ver, ts, pid, tier, core_t, core_m, bus_t, sleep_dis, idle, fps, avg, p95, duty, total, name, *_ = w
            if magic != TEL_MAGIC or now - ts > 2_500_000_000 or not Path(f"/proc/{pid}").exists():
                continue
            if pid in seen and seen[pid]["_ts"] >= ts:
                continue
            seen[pid] = {
                "_ts": ts, "pid": pid, "name": name.split(b"\0", 1)[0].decode(errors="replace"), "tier": tier,
                "core_corner": corner_name(core_t), "core_max": corner_name(core_m), "bus_corner": corner_name(bus_t),
                "sleep_disabled": bool(sleep_dis), "client_idle": bool(idle), "infer_fps": round(fps, 1),
                "avg_exec_ms": round(avg, 2), "p95_exec_ms": round(p95, 2), "duty_cycle_pct": round(duty, 1),
                "total_inferences": total,
            }
    out = []
    for c in seen.values():
        c.pop("_ts", None)
        out.append(c)
    return out


_service_cache = {"t": 0.0, "v": {}}


def read_services(account) -> dict:
    if time.monotonic() - _service_cache["t"] < 4.0 and _service_cache["v"]:
        return _service_cache["v"]
    _, _, home = user_dirs(account)
    qlib = home / ".local/share/odin3-fg/qairt/lib"
    try:
        active = systemctl("is-active", "vela-npu.service").stdout.strip() == "active"
    except Exception:
        active = False
    v = {
        "vela_npu_active": active,
        "vela_hooked": vela_hooked() if active else False,
        "fg_installed": (qlib / "libQnnHtp.so").is_file(),
        "fg_hooked": (qlib / "libQnnHtp.real.so").is_file() and is_shim(qlib / "libQnnHtp.so"),
        "shim_installed": GOV_SO.is_file(),
    }
    _service_cache.update(t=time.monotonic(), v=v)
    return v


def status_sync() -> dict:
    account = steam_user()
    cfg_dir, _, _ = user_dirs(account)
    settings = load_settings(cfg_dir)
    return {
        "settings": settings,
        "clients": read_clients(account),
        "sensors": read_sensors(),
        "services": read_services(account),
    }


def set_profile_sync(preset: str, custom: dict | None) -> dict:
    if preset not in PRESET_IDS:
        raise ValueError("Unknown preset")
    account = steam_user()
    cfg_dir, _, _ = user_dirs(account)
    s = load_settings(cfg_dir)
    s["preset"] = preset
    if custom:
        if "core_max" in custom:
            if int(custom["core_max"]) not in CORE_CHOICES:
                raise ValueError("Invalid core corner")
            s["core_max"] = int(custom["core_max"])
        if "bus_max" in custom:
            if int(custom["bus_max"]) not in BUS_CHOICES:
                raise ValueError("Invalid bus corner")
            s["bus_max"] = int(custom["bus_max"])
        if "sleep_enable" in custom:
            s["sleep_enable"] = bool(custom["sleep_enable"])
    save_settings(cfg_dir, s, account.pw_uid, account.pw_gid)
    write_all_configs(account, s)
    decky.logger.info(f"NPU profile -> {preset} core_max={corner_name(s['core_max'])} bus_max={corner_name(s['bus_max'])}")
    return status_sync()


def vela_enable_sync(enabled: bool) -> dict:
    if enabled:
        systemctl("start", "vela-npu.service", timeout=45.0)
    else:
        systemctl("stop", "vela-npu.service", timeout=45.0)
    _service_cache["t"] = 0.0
    return status_sync()


class Plugin:
    async def _main(self):
        try:
            info = await asyncio.to_thread(lambda: sync_install(steam_user()))
            decky.logger.info(f"Odin3 NPU Power initialized: {info}")
        except Exception as exc:
            decky.logger.error(f"Odin3 NPU Power init failed: {exc}")

    async def _unload(self):
        decky.logger.info("Odin3 NPU Power unloaded (config stays active)")

    async def get_npu_status(self):
        return await asyncio.to_thread(status_sync)

    async def set_npu_profile(self, preset, custom_opts=None):
        if not isinstance(preset, str):
            raise ValueError("preset must be a string")
        if custom_opts is not None and not isinstance(custom_opts, dict):
            custom_opts = None
        return await asyncio.to_thread(set_profile_sync, preset, custom_opts)

    async def set_vela_service_enabled(self, enabled):
        return await asyncio.to_thread(vela_enable_sync, bool(enabled))

    async def reinstall_governor(self):
        info = await asyncio.to_thread(lambda: sync_install(steam_user()))
        return info
