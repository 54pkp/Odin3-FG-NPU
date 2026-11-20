"""Match FFX frame IDs and measure repeated visible game responses."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re

import cv2
import numpy as np


def percentiles(values) -> dict[str, float] | None:
    if not len(values):
        return None
    return dict(zip(("p50", "p90", "max"), map(float, np.percentile(values, (50, 90, 100)))))


def ffx_timeline(log: Path) -> dict:
    frames = defaultdict(dict)
    for stage, frame, tick in re.findall(r"ANVIL TRACE (configure|generate|generated|real) (\d+) (\d+)",
                                         log.read_text(encoding="utf-8")):
        frames[int(frame)].setdefault(stage, int(tick))
    if not frames:
        raise ValueError(f"No trace events in {log}")
    pairs = {}
    for start, end in (("configure", "generate"), ("generate", "real"), ("configure", "real"),
                       ("generate", "generated"), ("generated", "real")):
        values = [(stages[end] - stages[start]) / 1e6 for stages in frames.values()
                  if start in stages and end in stages and stages[end] >= stages[start]]
        pairs[f"{start}_to_{end}"] = {"count": len(values), "ms": percentiles(values)}
    return {"stages": {stage: sum(stage in stages for stages in frames.values())
                       for stage in ("configure", "generate", "generated", "real")}, "deltas": pairs}


def software_response(folder: Path) -> dict:
    with (folder / "screen.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    capture = np.asarray([int(row["capture_monotonic_ns"]) for row in rows], dtype=np.int64)
    source = np.asarray([int(row.get("source_monotonic_ns") or "0") for row in rows], dtype=np.int64)
    observed = source > 0
    # Keep the two clocks separate. Frames that were dropped before the encoder
    # do not have image evidence and cannot be used for overlay detection.
    match = source.copy()
    match[~observed] = capture[~observed]
    stamps = (folder / "input.log").read_text()
    edges = {}
    for edge, trial, tick in re.findall(r"^pause_(down|up) (\d+) (\d+)$", stamps, re.M):
        edges.setdefault(int(trial), {})[edge] = int(tick)
    if len(edges) not in (3, 6) or set(edges) != set(range(len(edges))):
        raise ValueError(f"Expected three or six contiguous button presses in {folder}")
    results = []
    tunic = folder.name.startswith("measured-tunic-")
    for trial, edge in sorted(edges.items()):
        down, up = edge["down"], edge["up"]
        before, after = [], []
        for i in np.flatnonzero((match >= down - 300_000_000) & (match < down + 950_000_000)):
            img = cv2.imread(str(folder / f"{int(rows[i]['frame']):04d}.jpg"))
            if img is None:
                raise FileNotFoundError(rows[i]["frame"])
            crop = img[36:-36, 66:-66] if tunic else img[25:-25, 40:-40]
            if tunic:
                signal = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                signal = float(((hsv[:, :, 0] >= 140) & (hsv[:, :, 0] <= 179) & (hsv[:, :, 1] >= 100) &
                                (hsv[:, :, 2] >= 50)).mean())
            (before if match[i] < down else after).append((i, signal))
        if len(before) < 2 or len(after) < 2:
            raise ValueError(f"Pause {trial} lacks coverage")
        if tunic:
            # The scene is held still before each pause. Exclude both frame-tag regions.
            reference = np.median([image for _, image in before[-8:]], axis=0).astype(np.uint8)
            base = max(float(cv2.absdiff(reference, image).mean()) for _, image in before)
            after = [(i, float(cv2.absdiff(reference, image).mean())) for i, image in after]
            limit = max(6.0, base * 3)
            metric_names = ("baseline_mean_luma_change", "max_mean_luma_change")
        else:
            base = max(ratio for _, ratio in before)
            limit = max(0.35, base + 0.25)
            metric_names = ("baseline_overlay_fraction", "max_overlay_fraction")
        onset = next((i for (i, a), (_, b) in zip(after, after[1:]) if min(a, b) >= limit), None)
        results.append({"trial": trial, metric_names[0]: base,
                        metric_names[1]: max(ratio for _, ratio in after),
                        "down_to_visible_ms": (match[onset] - down) / 1e6 if onset is not None else None,
                        "up_to_visible_ms": (match[onset] - up) / 1e6 if onset is not None else None,
                        "visible_source": "PipeWire arrival" if onset is not None and observed[onset] else "JPEG arrival"})
    return {"frames": len(rows), "frames_with_source_stamp": int(observed.sum()),
            "source_to_jpeg_ms": percentiles((capture[observed] - source[observed]) / 1e6),
            "trials": results,
            "down_to_visible_ms": percentiles([r["down_to_visible_ms"] for r in results if r["down_to_visible_ms"] is not None])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    output = {"ffx": ffx_timeline(args.log), "software_response": software_response(args.folder)}
    result = json.dumps(output, indent=2) + "\n"
    (args.folder / "timeline.json").write_text(result, encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
