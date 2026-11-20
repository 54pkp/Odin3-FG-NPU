"""Verify the known Vulkan worker signature in isolated Gamescope captures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np

from analyze_playtest import motion_times, read_rows


def has_signature(image: np.ndarray) -> bool:
    height, width = image.shape[:2]
    y, x = height // 2, width // 2
    blue, green, red = image[y - 14:y + 14, x - 18:x + 18].astype(np.int16).transpose(2, 0, 1)
    magenta = (blue > 145) & (red > 145) & (red - green > 70) & (blue - green > 55)
    cyan = (blue > 145) & (green > 145) & (green - red > 70) & (blue - red > 55)
    left_y, left_x = np.nonzero(magenta)
    right_y, right_x = np.nonzero(cyan)
    if min(len(left_x), len(right_x)) < 6:
        return False
    return (
        2 <= right_x.mean() - left_x.mean() <= 12
        and abs(right_y.mean() - left_y.mean()) <= 3
        and abs((right_x.mean() + left_x.mean()) / 2 - 18) <= 4
        and abs((right_y.mean() + left_y.mean()) / 2 - 14) <= 4
    )


def captures(folder: Path) -> dict:
    start, stop = motion_times(folder / "input.log")
    rows = read_rows(folder / "screen.csv")
    marked = []
    motion_frames = 0
    motion_marked = 0
    for row in rows:
        frame = int(row["frame"])
        image = cv2.imread(str(folder / f"{frame:04d}.jpg"))
        if image is None:
            raise ValueError(f"Capture {frame} is missing or unreadable")
        active = start <= int(row["capture_monotonic_ns"]) < stop
        motion_frames += active
        if has_signature(image):
            marked.append(frame)
            motion_marked += active
    return {
        "captures": len(rows),
        "marked_captures": len(marked),
        "motion_captures": motion_frames,
        "marked_motion_captures": motion_marked,
        "marked_capture_examples": marked[:12],
    }


def timing(log: str) -> dict:
    samples = [
        tuple(map(int, row))
        for row in re.findall(
            r"anvil-timing [^\n]*source_submit_us=(\d+) source_ready_us=(-?\d+) "
            r"output_ready_us=(-?\d+) admitted=([01])", log
        )
    ]
    if not samples:
        return {}

    def microseconds(values: list[int]) -> dict:
        if not values:
            return {"samples": 0}
        p50, p95 = np.percentile(values, (50, 95))
        return {"samples": len(values), "p50": float(p50), "p95": float(p95)}

    missed = [row for row in samples if row[2] < 0]
    result = {
        "scheduled": len(samples),
        "admitted": sum(row[3] for row in samples),
        "output_unobserved": len(missed),
        "source_unobserved_among_those": sum(row[1] < 0 for row in missed),
        "submit_to_source_ready_us": microseconds([
            source - submitted for submitted, source, _, _ in samples if source >= 0
        ]),
        "source_to_output_ready_us": microseconds([
            output - source for _, source, output, _ in samples if source >= 0 and output >= 0
        ]),
    }
    boundary = [
        tuple(map(int, row))
        for row in re.findall(
            r"anvil-timing [^\n]*source_submit_us=(\d+) source_ready_us=(-?\d+) "
            r"output_ready_us=(-?\d+) admitted=[01] render_ready_us=(-?\d+) same_queue=([01])",
            log,
        )
    ]
    if boundary:
        result["render_boundary"] = {
            "samples": len(boundary),
            "same_queue_count": sum(row[4] for row in boundary),
            "source_missing": sum(row[1] < 0 for row in boundary),
            "render_also_unobserved_if_source_missing": sum(
                source < 0 and render < 0 for _, source, _, render, _ in boundary
            ),
            "submit_to_render_ready_us": microseconds([
                render - submitted for submitted, _, _, render, _ in boundary if render >= 0
            ]),
            "render_to_source_ready_us": microseconds([
                source - render for _, source, _, render, _ in boundary if render >= 0 and source >= 0
            ]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("copy", type=Path)
    args = parser.parse_args()

    control = captures(args.control)
    copy = captures(args.copy)
    log = (args.copy / "present.log").read_text(encoding="utf-8", errors="replace")
    for kind, operation in (("real", "present-total"), ("copy", "present-generated-image")):
        copy[f"{kind}_presents"] = len(re.findall(
            rf"operation={operation} [^\n]*result=(?:0|1000001003)(?: |$)", log, re.M))
    copy["timeline"] = timing(log)
    output = json.dumps({"control": control, "copy": copy}, indent=2) + "\n"
    (args.copy / "pixel-verification.json").write_text(output, encoding="utf-8")
    print(output)
    if control["marked_captures"] or copy["marked_motion_captures"] < 5 or copy["copy_presents"] < 30:
        raise SystemExit("The GPU signature was not verified against the real-only control")


if __name__ == "__main__":
    main()
