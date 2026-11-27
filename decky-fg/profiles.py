"""Read and write launch-time profiles for Odin 3 Pure-Neural Frame Generation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import chain
import json
from math import isfinite
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import time


FLOW_STRIDES = {12.5: 8, 25.0: 4, 50.0: 2}
FLOW_DEFAULT = 25.0


def validate_flow_percent(percent: float) -> None:
    if type(percent) not in (int, float) or not min(FLOW_STRIDES) <= percent <= max(FLOW_STRIDES):
        raise ValueError("Flow analysis percentage must be between 12.5 and 50")


def flow_plan(requested: float) -> dict:
    validate_flow_percent(requested)
    applied = min(FLOW_STRIDES, key=lambda value: (abs(value - requested), value))
    stride = FLOW_STRIDES[applied]
    return {
        "requested_percent": requested, "applied_percent": applied, "stride": stride,
        "width": 1280 // stride, "height": 720 // stride,
        "supported_percentages": sorted(FLOW_STRIDES),
    }


@dataclass(frozen=True)
class GameProfile:
    app_id: int
    multiplier: int
    source_fps_cap: int | None = None
    input_mode: str = "standalone"
    version: int = 1
    flow_analysis_percent: float = FLOW_DEFAULT
    target_refresh_hz: int = 120
    auto_display_mode: bool = True


def parse_app_id(text: str) -> int:
    if not text.isascii() or not text.isdecimal() or len(text) > 10:
        raise argparse.ArgumentTypeError("AppID must be a positive Steam application ID")
    app_id = int(text)
    if not 1 <= app_id <= 2**32 - 1:
        raise argparse.ArgumentTypeError("AppID must be a positive Steam application ID")
    return app_id


def profile_path(app_id: int) -> Path:
    override = os.environ.get("ODIN3_FG_PROFILES_DIR")
    if override:
        directory = Path(override).expanduser()
    else:
        config_dir = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
        directory = config_dir / "odin3-fg" / "profiles"
    return directory / f"{app_id}.json"


def validate_profile(profile: GameProfile, expected_app_id: int) -> None:
    if type(profile.version) is not int or profile.version != 1:
        raise ValueError("Unsupported profile version")
    if (type(profile.app_id) is not int or not 1 <= profile.app_id <= 2**32 - 1
            or profile.app_id != expected_app_id):
        raise ValueError("Profile AppID does not match the requested game")
    if type(profile.multiplier) is not int or profile.multiplier not in (0, 2, 3):
        raise ValueError("Profile multiplier must be 0, 2 or 3")
    if profile.input_mode != "standalone":
        raise ValueError("Profile input mode must be standalone")
    validate_flow_percent(profile.flow_analysis_percent)
    cap = profile.source_fps_cap
    if cap is not None and (type(cap) is not int or not 1 <= cap <= 1000):
        raise ValueError("Source FPS cap must be null or an integer from 1 to 1000")
    if type(profile.target_refresh_hz) is not int or profile.target_refresh_hz not in (60, 120):
        raise ValueError("Target refresh rate must be 60 or 120 Hz")
    if type(profile.auto_display_mode) is not bool:
        raise ValueError("auto_display_mode must be a boolean")


def load_profile(app_id: int) -> GameProfile | None:
    try:
        with profile_path(app_id).open(encoding="utf-8") as stream:
            values = json.load(stream)
    except FileNotFoundError:
        return None
    if not isinstance(values, dict) or "version" not in values:
        raise ValueError("Expected a versioned profile object")
    allowed = {"app_id", "multiplier", "source_fps_cap", "input_mode", "version",
               "flow_analysis_percent", "target_refresh_hz", "auto_display_mode"}
    filtered = {k: v for k, v in values.items() if k in allowed}
    profile = GameProfile(**filtered)
    validate_profile(profile, app_id)
    return profile


def save_profile(profile: GameProfile) -> None:
    validate_profile(profile, profile.app_id)
    path = profile_path(profile.app_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=f".{path.stem}-", delete=False)
    pending = Path(stream.name)
    try:
        with stream:
            json.dump(asdict(profile), stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def suggested_source_fps(refresh_hz: float, multiplier: int) -> int | None:
    if not isfinite(refresh_hz) or not 1 <= refresh_hz <= 1000:
        raise ValueError("Refresh rate must be between 1 and 1000 Hz")
    return max(1, round(refresh_hz / multiplier)) if multiplier else None


def runtime_geometry(state: dict, active: bool) -> tuple[dict | None, dict | None]:
    width, height = state.get("frame_width"), state.get("frame_height")
    if any(type(value) is not int or not 1 <= value <= 2**32 - 1 for value in (width, height)):
        return None, None
    frame = {"width": width, "height": height}
    stride = state.get("flow_stride")
    if not active or type(stride) is not int or stride not in FLOW_STRIDES.values() or min(width, height) < stride:
        return frame, None
    requested = None
    raw_request = state.get("flow_request")
    if isinstance(raw_request, str):
        try:
            percent = float(raw_request)
            validate_flow_percent(percent)
            requested = percent
        except ValueError:
            pass
    return frame, {"requested_percent": requested, "applied_percent": 100 / stride,
                   "width": width // stride, "height": height // stride}


def live_runtime(app_id: int) -> dict | None:
    override = os.environ.get("ODIN3_FG_RUNTIME_STATE_DIR")
    if override:
        directories = [Path(override)]
    else:
        directories = [Path.home() / ".cache/odin3-fg/runtime-state"]
        legacy = os.environ.get("XDG_RUNTIME_DIR")
        if legacy:
            directories.append(Path(legacy) / "odin3-fg/runtime-state")
    now = time.time_ns() // 1_000_000
    user_id = os.geteuid() if hasattr(os, "geteuid") else None
    records = []
    paths = chain.from_iterable(directory.glob("*-independent-frame-generation-*.json") for directory in directories)
    for path in paths:
        try:
            if path.is_symlink() or not path.is_file():
                continue
            if user_id is not None and path.stat().st_uid != user_id:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            state = data["independent"]
            pid, ticks = data["pid"], data["process_start_ticks"]
            updated = data["updated_unix_ms"]
            if (data["schema_version"] != 5 or data["role"] != "independent-frame-generation"
                    or state["app_id"] != app_id or state["input_mode"] != "standalone"
                    or type(pid) is not int or pid <= 0 or type(ticks) is not int or ticks <= 0
                    or type(updated) is not int or not 0 <= now - updated <= 3000):
                continue
            proc_path = Path(f"/proc/{pid}/stat")
            if user_id is not None and proc_path.stat().st_uid != user_id:
                continue
            proc = proc_path.read_text(encoding="utf-8")
            if int(proc.rsplit(")", 1)[1].split()[19]) != ticks:
                continue
            active = data["frame_generation_active"] is True
            multiplier = data["requested"]["multiplier"]
            reason = data["reason"]
            original, generated, requested = (
                state[key] for key in ("original_frames", "generated_frames", "requested_generated_frames")
            )
            rates = state["original_submission_fps"], state["total_submission_fps"]
            if (type(multiplier) is not int or multiplier not in (2, 3) or not isinstance(reason, str)
                    or any(type(count) is not int or count < 0 for count in (original, generated, requested))
                    or generated > requested
                    or any(rate is not None and (type(rate) not in (int, float) or not isfinite(rate) or rate < 0)
                           for rate in rates)):
                continue
            frame, analysis = runtime_geometry(state, active)
            records.append({"pid": pid, "updated_unix_ms": updated,
                            "frame_size": frame, "flow_analysis": analysis,
                            "input_mode": "standalone", "active": active, "applied_multiplier": multiplier if active else 0,
                            "reason": reason, "original_frames": original, "generated_frames": generated,
                            "requested_generated_frames": requested,
                            "original_submission_fps": rates[0], "total_submission_fps": rates[1]})
        except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError):
            continue
    return max(records, key=lambda value: (value["updated_unix_ms"], value["generated_frames"]), default=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    read = commands.add_parser("get")
    read.add_argument("app_id", type=parse_app_id)
    formatting = read.add_mutually_exclusive_group()
    formatting.add_argument("--launch", action="store_true")
    formatting.add_argument("--launch-with-flow", action="store_true")
    formatting.add_argument("--refresh-hz", type=float)
    write = commands.add_parser("set")
    write.add_argument("app_id", type=parse_app_id)
    write.add_argument("--multiplier", type=int, choices=(0, 2, 3), required=True)
    write.add_argument("--source-fps-cap", type=int)
    write.add_argument("--flow-analysis-percent", type=float)
    write.add_argument("--target-refresh-hz", type=int, choices=(60, 120))
    write.add_argument("--auto-display-mode", choices=("0", "1"))
    status = commands.add_parser("status")
    status.add_argument("app_id", type=parse_app_id)
    args = parser.parse_args()

    try:
        if args.command == "set":
            previous = load_profile(args.app_id)
            requested = args.flow_analysis_percent if args.flow_analysis_percent is not None else (
                previous.flow_analysis_percent if previous else FLOW_DEFAULT
            )
            target_hz = args.target_refresh_hz if args.target_refresh_hz is not None else (
                previous.target_refresh_hz if previous else 120
            )
            auto_disp = (args.auto_display_mode == "1") if args.auto_display_mode is not None else (
                previous.auto_display_mode if previous else True
            )
            profile = GameProfile(
                args.app_id, args.multiplier, args.source_fps_cap,
                flow_analysis_percent=requested,
                target_refresh_hz=target_hz,
                auto_display_mode=auto_disp,
            )
            save_profile(profile)
        elif args.command == "status":
            stored = load_profile(args.app_id)
            profile = stored or GameProfile(args.app_id, 0)
            print(json.dumps({
                "stored": stored is not None,
                "profile": asdict(profile),
                "runtime": live_runtime(args.app_id),
                "flow_plan": flow_plan(profile.flow_analysis_percent),
            }))
        else:
            stored = load_profile(args.app_id)
            profile = stored or GameProfile(args.app_id, 0)
            if args.launch:
                print(profile.multiplier, profile.source_fps_cap or "-", profile.target_refresh_hz, int(profile.auto_display_mode))
                return
            plan = flow_plan(profile.flow_analysis_percent)
            if args.launch_with_flow:
                print(profile.multiplier, profile.source_fps_cap or "-", plan["requested_percent"],
                      plan["applied_percent"], plan["stride"], plan["width"], plan["height"],
                      profile.target_refresh_hz, int(profile.auto_display_mode))
                return
            result = {**asdict(profile), "stored": stored is not None, "flow_plan": plan}
            if args.refresh_hz is not None:
                result["suggested_source_fps"] = suggested_source_fps(args.refresh_hz, profile.multiplier)
            print(json.dumps(result))
    except (OSError, TypeError, ValueError) as error:
        parser.exit(2, f"Odin 3 FG profile: {error}\n")


if __name__ == "__main__":
    main()
