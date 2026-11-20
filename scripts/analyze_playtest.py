"""Summarize aligned Odin 3 screen and DRM traces from the isolated A/B."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import cv2
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def motion_times(path: Path) -> tuple[int, int]:
    marks = dict(re.findall(r"^motion_(start|stop)_ns (\d+)$", path.read_text(), re.M))
    return int(marks["start"]), int(marks["stop"])


def stats(values) -> dict:
    sample = np.asarray(values, dtype=np.float64)
    return dict(zip(("p05", "p50", "p95", "max"), map(float, np.percentile(sample, (5, 50, 95, 100)))))


def analyze(folder: Path) -> dict:
    start, stop = motion_times(folder / "input.log")
    screen = read_rows(folder / "screen.csv")
    capture = np.asarray([int(row["capture_monotonic_ns"]) for row in screen], dtype=np.int64)
    pts = np.asarray([int(row["pts_clock_ns"]) if row["pts_clock_ns"] else 0 for row in screen], dtype=np.int64)
    if len(screen) == 0 or not (capture[0] < start < stop < capture[-1]):
        raise ValueError("The trace did not cover the entire motion interval")
    image = []
    for row in screen:
        item = cv2.imread(str(folder / f"{int(row['frame']):04d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if item is None:
            raise FileNotFoundError(row["frame"])
        # Exclude permanent HUD borders from the brightness and motion measurements.
        image.append(item[25:-25, 40:-40])
    pixels = np.asarray(image, np.float32)
    brightness = pixels.mean(axis=(1, 2))
    difference = np.mean(np.abs(np.diff(pixels, axis=0)), axis=(1, 2), dtype=np.float64)
    step = (capture[1:] >= start) & (capture[1:] < stop)
    gameplay = (capture >= start) & (capture < stop)
    active = len(capture[gameplay])
    visible_changes = step & (difference > 0.7)
    samples = read_rows(folder / "gpu.csv")
    gpu = [row for row in samples if start <= int(row["monotonic_ns"]) < stop]
    if len(gpu) < 3 or len({row["unique_drm_clients"] for row in gpu}) != 1:
        raise ValueError("Process DRM client population changed inside the comparison interval")
    busy = (int(gpu[-1]["gpu_engine_ns"]) - int(gpu[0]["gpu_engine_ns"])) / (
        int(gpu[-1]["monotonic_ns"]) - int(gpu[0]["monotonic_ns"]))
    priming = (capture[1:] >= start - 400_000_000) & (capture[1:] < start)
    threshold = max(1.5, float(np.percentile(difference[priming], 99)) * 2) if priming.any() else 1.5
    reacted = np.where(step & (difference >= threshold))[0]
    result = {
        "buffers": len(screen), "capture_time_with_pts": int(np.sum(pts != 0)),
        "motion_duration_s": (stop - start) / 1e9,
        "motion_capture_rate": active * 1e9 / (stop - start),
        "motion_visually_distinct_rate": np.count_nonzero(visible_changes) * 1e9 / (stop - start),
        "interval_projection": "PipeWire compositor delivery and JPEG arrival clocks",
        "motion_capture_intervals_ms": stats(np.diff(capture[gameplay]) / 1e6),
        "motion_scene_mean_brightness": stats(brightness[gameplay]),
        "motion_scene_adjacent_pixel_change": stats(difference[step]),
        "motion_process_GPU_engine_busy_pct": busy * 100,
        "first_visible_response_ms": float((capture[reacted[0] + 1] - start) / 1e6) if len(reacted) else None,
        "response_estimator": "first strong change after injected stick motion, compositor-to-JPEG path included",
    }
    pause = re.search(r"^pause_start_ns (\d+)$", (folder / "input.log").read_text(), re.M)
    if pause:
        pause_at = int(pause[1])
        before, after = [], []
        for i in np.flatnonzero((capture >= pause_at - 300_000_000) & (capture <= pause_at + 1_000_000_000)):
            frame = cv2.imread(str(folder / f"{int(screen[i]['frame']):04d}.jpg"))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[25:-25, 40:-40]
            overlay = float(((hsv[:, :, 0] >= 140) & (hsv[:, :, 0] <= 179) & (hsv[:, :, 1] >= 100) &
                            (hsv[:, :, 2] >= 50)).mean())
            (before if capture[i] < pause_at else after).append((int(capture[i]), overlay))
        if not before or len(after) < 2:
            raise ValueError("Pause latency trace has too few samples")
        base = max(v for _, v in before)
        limit = max(0.35, base + 0.25)
        onset = next((t1 for (t1, val1), (_, val2) in zip(after, after[1:]) if min(val1, val2) >= limit), None)
        result["pause_to_compositor_capture_ms"] = (onset - pause_at) / 1e6 if onset is not None else None
        result["pre_pause_magenta_fraction_max"] = base
        result["post_pause_magenta_fraction_max"] = max(v for _, v in after)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--modes", nargs="+", default=["native", "anvil"])
    args = parser.parse_args()
    output = {mode: analyze(args.root / ("measured-" + mode)) for mode in args.modes}
    rendered = json.dumps(output, indent=2) + "\n"
    print(rendered)
    (args.root / "summary.json").write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
