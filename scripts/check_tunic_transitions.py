"""Check captured TUNIC pause boundaries against the actual neighboring game frames."""
import argparse
import json
from pathlib import Path

import cv2

from analyze_mako_anvil import tagged_output
from analyze_playtest import read_rows


def pause_presses(folder):
    return [int(line.split()[2]) for line in (folder / "input.log").read_text().splitlines()
            if line.startswith("pause_down ")]


def screen(folder):
    return [(int(row["source_monotonic_ns"]), int(row["frame"]))
            for row in read_rows(folder / "screen.csv") if row["source_monotonic_ns"]]


def menu_crop(folder, frame):
    path = folder / f"{frame:04d}.jpg"
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image[54:156, 145:335], cv2.COLOR_BGR2GRAY), tagged_output(image)


def difference(left, right):
    return float(cv2.absdiff(left, right).mean())


def audit(folder, template):
    frames = screen(folder)
    transitions = []
    for trial, down in enumerate(pause_presses(folder)):
        previous = None
        generated = []
        boundaries = []
        for stamp, frame in frames:
            if not down - 150_000_000 <= stamp < down + 1_850_000_000:
                continue
            crop, marked = menu_crop(folder, frame)
            if marked:
                generated.append((frame, crop))
                continue
            menu = difference(crop, template) <= 8.0
            if previous and previous[1] != menu:
                first, _, old = previous
                candidates = [{"frame": number, "mae_to_closer_endpoint":
                               min(difference(pixels, old), difference(pixels, crop))}
                              for number, pixels in generated]
                boundaries.append({"action": "open" if menu else "close", "first": first,
                                   "last": frame, "generated": candidates})
            previous = (frame, menu, crop)
            generated.clear()
        if [entry["action"] for entry in boundaries] != ["open", "close"]:
            raise ValueError(f"{folder} trial {trial} lacks an unambiguous open/close capture")
        transitions.extend(boundaries)
    if len(transitions) != 12:
        raise ValueError(f"{folder} needs six complete pause trials")
    artifacts = [candidate for boundary in transitions for candidate in boundary["generated"]
                 if candidate["mae_to_closer_endpoint"] > 12.0]
    return {"transitions": transitions, "visibly_mixed_generated_frames": artifacts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--require-composited", action="store_true",
                        help="Require one clean generated frame at each real-frame transition")
    args = parser.parse_args()
    settled = pause_presses(args.control)[0] + 500_000_000
    _, frame = min(screen(args.control), key=lambda item: abs(item[0] - settled))
    template, marked = menu_crop(args.control, frame)
    if marked:
        raise ValueError("The menu template should come from the real-only control")
    result = audit(args.capture, template)
    print(json.dumps(result, indent=2))
    if (args.require_clean or args.require_composited) and result["visibly_mixed_generated_frames"]:
        raise SystemExit(1)
    if args.require_composited and any(len(item["generated"]) != 1 for item in result["transitions"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
