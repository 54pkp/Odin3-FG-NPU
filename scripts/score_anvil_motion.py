"""Prepare captured real frame triplets and score the replay against the withheld frame."""
import argparse
from itertools import pairwise
import json
from pathlib import Path

import cv2
import numpy as np

from analyze_playtest import motion_times, read_rows

WIDTH, HEIGHT = 1280, 720


def rgba(image):
    scaled = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(scaled, cv2.COLOR_BGR2RGBA)


def image(folder, frame):
    path = folder / f"{frame:04d}.jpg"
    result = cv2.imread(str(path))
    if result is None:
        raise ValueError(f"Cannot read {path}")
    return result


def prepare(capture, replay, reference_rate):
    start, stop = motion_times(capture / "input.log")
    frames = [(int(row["frame"]), int(row["source_monotonic_ns"])) for row in read_rows(capture / "screen.csv")
              if row["source_monotonic_ns"] and start <= int(row["source_monotonic_ns"]) < stop]
    eligible = []
    period = 1000 / reference_rate
    for i in range(len(frames) - 2):
        triplet = frames[i:i + 3]
        deltas = [(right[1] - left[1]) / 1e6 for left, right in pairwise(triplet)]
        if min(deltas) < .66 * period or max(deltas) > 1.35 * period or max(deltas) / min(deltas) > 1.5:
            continue
        first, middle, last = (image(capture, item[0]) for item in triplet)
        differences = [float(np.mean(cv2.absdiff(a[24:-24, 38:-38], b[24:-24, 38:-38])))
                       for a, b in pairwise((first, middle, last))]
        if min(differences) < 3:
            continue
        eligible.append((sum(differences), triplet, deltas))
    chosen = []
    for _, triplet, deltas in sorted(eligible, reverse=True):
        if any(abs(triplet[1][0] - known[0][1][0]) < 7 for known in chosen):
            continue
        chosen.append((triplet, deltas))
        if len(chosen) == 8:
            break
    if len(chosen) < 4:
        raise ValueError(f"The capture only has {len(chosen)} usable moving triplets")
    replay.mkdir()
    manifest = []

    def save(images, metadata):
        case = replay / f"pair-{len(manifest):02d}"
        case.mkdir()
        for name, value in zip(("first", "middle", "last"), images, strict=True):
            value.tofile(case / f"{name}.rgba")
        manifest.append(metadata)

    for triplet, deltas in sorted(chosen):
        indices = [item[0] for item in triplet]
        save([rgba(image(capture, frame)) for frame in indices],
             {"kind": "recorded_game", "frames": indices, "source_intervals_ms": deltas,
              "reference_fps_target": reference_rate})
    base = rgba(image(capture, sorted(chosen)[0][0][1][0]))
    shifts = (0, 24, 48)
    pan = [cv2.warpAffine(base, np.float32([[1, 0, dx], [0, 1, 0]]), (WIDTH, HEIGHT),
                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT) for dx in shifts]
    save(pan, {"kind": "analytic_pan", "x_offsets": shifts})
    (replay / "fixtures.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{len(manifest)} cases, {len(eligible)} eligible game triplets")


def score(replay):
    cases = json.loads((replay / "fixtures.json").read_text(encoding="utf-8"))
    results = []
    for i, meta in enumerate(cases):
        case = replay / f"pair-{i:02d}"

        def read(name, channels):
            return np.fromfile(case / name, dtype=np.uint8).reshape(HEIGHT, WIDTH, channels).astype(np.float32)

        first, middle, last = (read(f"{role}.rgba", 4)[:, :, :3] for role in ("first", "middle", "last"))
        aligned_a = read("aligned-first.rgba", 4)
        aligned_b = read("aligned-last.rgba", 4)
        # Exclude resampling borders. The motion mask uses original inputs
        # only and is shared by every method for this case.
        interior = (slice(72, -72), slice(96, -96))
        cut = np.mean(np.abs(first - last), axis=2)[interior] >= 18.0
        if not cut.any():
            raise ValueError(f"No motion-region contrast in {case.name}")
        predictions = {
            "last_real": last,
            "raw_blend": (first + last) / 2,
            "unaligned_htp": read("unaligned-htp.rgb", 3),
            "aligned_blend": (aligned_a[:, :, :3] + aligned_b[:, :, :3]) / 2,
            "aligned_htp": read("aligned-htp.rgb", 3),
        }
        scores = {}
        for name, output in predictions.items():
            delta = (output - middle)[interior]
            scores[name] = {
                "psnr_db": float(20 * np.log10(255) - 10 * np.log10(max(np.mean(delta**2), 1e-12))),
                "motion_mae_codes": float(np.mean(np.abs(delta[cut]))),
            }
            if name in ("unaligned_htp", "aligned_htp"):
                target = case / f"{name}.png"
                if not cv2.imwrite(str(target), cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR)):
                    raise OSError(target)
        target = case / "withheld.png"
        if not cv2.imwrite(str(target), cv2.cvtColor(middle.astype(np.uint8), cv2.COLOR_RGB2BGR)):
            raise OSError(target)
        results.append({"case": i, **meta, "motion_fraction": float(cut.mean()),
                        "scores": scores, "aligned_htp_minus_unaligned_db":
                            scores["aligned_htp"]["psnr_db"] - scores["unaligned_htp"]["psnr_db"]})
    game = [item for item in results if item["kind"] == "recorded_game"]
    gains = [item["aligned_htp_minus_unaligned_db"] for item in game]
    data = {"results": results, "game_pairs": len(game), "wins": sum(value > 0 for value in gains),
            "median_gain_db": float(np.median(gains))}
    rendered = json.dumps(data, indent=2) + "\n"
    (replay / "scores.json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--prepare-from", type=Path)
    parser.add_argument("--reference-rate", type=int, choices=(30, 60), default=30)
    args = parser.parse_args()
    if args.prepare_from:
        prepare(args.prepare_from, args.replay, args.reference_rate)
    else:
        score(args.replay)


if __name__ == "__main__":
    main()
