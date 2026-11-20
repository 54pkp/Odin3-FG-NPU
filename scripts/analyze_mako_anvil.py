"""Summarize a private HTP game run and the matched renderer control."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import cv2
import numpy as np

from analyze_mako_copy import timing
from analyze_playtest import analyze, motion_times, read_rows, stats
from analyze_timeline import software_response


def vk_presents(log, operation):
    return Counter(int(context) for context in re.findall(
        rf"operation={operation} context=(\d+)[^\n]*result=(?:0|1000001003)(?: |$)", log, re.M))


def integer_fields(line):
    return {key: int(value) for key, value in re.findall(r"(\w+)=(-?\d+)(?= |$)", line)}


def tagged_output(image):
    height, width = image.shape[:2]
    if (width, height) != (480, 270):
        raise ValueError("The tag geometry assumes the 480x270 capture")
    blue, green, red = image[15:29, 32:59].astype(np.int16).transpose(2, 0, 1)
    magenta = (blue > 125) & (red > 125) & (red - green > 55) & (blue - green > 48)
    cyan = (blue > 125) & (green > 125) & (green - red > 55) & (blue - red > 48)
    my, mx = np.nonzero(magenta)
    cy, cx = np.nonzero(cyan)
    return (len(mx) >= 6 and len(cx) >= 6 and 5 <= mx.mean() <= 11
            and 14 <= cx.mean() <= 21 and 2 <= my.mean() <= 10
            and abs(my.mean() - cy.mean()) <= 3)


def generated_id(image):
    # Read the centers of the two complementary rows after 1280x720 -> 480x270.
    centers = (96 + np.arange(32) * 8 + 4) * 3 // 8
    top = image[249, centers].mean(axis=1)
    bottom = image[255, centers].mean(axis=1)
    if np.min(np.abs(top - bottom)) < 48:
        return None
    return sum(1 << index for index, value in enumerate(top > bottom) if value)


def capture_provenance(folder):
    start, stop = motion_times(folder / "input.log")
    frames = read_rows(folder / "screen.csv")
    observed = []
    marked = []
    numbered = []
    distinct_timestamps = {"tagged": [], "untagged": []}
    previous_scene = None
    for row in frames:
        frame = int(row["frame"])
        arrival = int(row["source_monotonic_ns"] or row["capture_monotonic_ns"])
        image = cv2.imread(str(folder / f"{frame:04d}.jpg"))
        if image is None:
            raise ValueError(f"Missing capture {frame}")
        is_tagged = tagged_output(image)
        if is_tagged:
            marked.append((frame, arrival))
            marker_id = generated_id(image)
            if marker_id is not None:
                numbered.append((frame, marker_id))
        scene = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[25:-25, 40:-40]
        if start <= arrival < stop:
            observed.append(arrival)
            if previous_scene is not None and cv2.absdiff(scene, previous_scene).mean() > 0.7:
                distinct_timestamps["tagged" if is_tagged else "untagged"].append(arrival)
        previous_scene = scene
    motion = [(frame, arrival) for frame, arrival in marked if start <= arrival < stop]
    seconds = (stop - start) / 1e9
    return {
        "tagged": len(marked), "tagged_motion": len(motion), "motion_buffers": len(observed),
        "average_tagged_motion_hz": len(motion) / seconds,
        "motion_distinct_capture_hz": {
            label: len(timestamps) / seconds for label, timestamps in distinct_timestamps.items()
        },
        "motion_max_gap_between_distinct_tagged_ms": float(max(
            np.diff([start, *distinct_timestamps["tagged"], stop])) / 1e6),
        "motion_examples": [frame for frame, _ in motion[:10]],
        "numbered_captures": numbered,
    }


def combined_gpu(folder):
    start, stop = motion_times(folder / "input.log")
    samples = [row for row in read_rows(folder / "gpu.csv") if start <= int(row["monotonic_ns"]) < stop]
    if len(samples) < 2 or "gpu_combined_clients" not in samples[0]:
        return None
    for field in ("gpu_worker_clients", "gpu_combined_clients"):
        if len({row[field] for row in samples}) != 1:
            raise ValueError(f"DRM client population changed during motion for {field}")
    elapsed = int(samples[-1]["monotonic_ns"]) - int(samples[0]["monotonic_ns"])
    duty = lambda field: 100 * (int(samples[-1][field]) - int(samples[0][field])) / elapsed
    return {"worker_pct": duty("gpu_worker_ns"), "game_and_worker_pct": duty("gpu_combined_ns"),
            "worker_clients_observed": bool(samples[0]["gpu_worker_clients"])}


def bridge_stages(log):
    """Join protocol feedback to exact requests, including source timings when available."""
    requests = {}
    feedback = {}
    admitted = defaultdict(list)
    for line in log.splitlines():
        if "operation=gamescope-bridge-request " in line:
            row = integer_fields(line)
            requests[row["bridge"], row["id"]] = row
        elif "operation=gamescope-bridge-frame " in line:
            row = integer_fields(line)
            feedback[row["bridge"], row["id"]] = row
        elif "anvil-timing " in line and "started_ns=" in line:
            row = integer_fields(line)
            if row["admitted"]:
                admitted[row["frame"]].append(row)
    if not requests:
        return None

    main_bridge = Counter(key[0] for key in requests).most_common(1)[0][0]
    by_frame = defaultdict(list)
    for (bridge, serial), row in requests.items():
        if bridge == main_bridge:
            by_frame[row["source_frame"]].append((serial, row))
    samples = defaultdict(list)
    counts = Counter()
    prediction_kinds = Counter()

    def record(label, stage, begin, end, *, signed=False):
        if begin > 0 and end > 0 and (end >= begin or signed):
            samples[label, stage].append((end - begin) / 1e6)

    for frame, group in by_frame.items():
        group.sort(key=lambda item: item[0])
        if len(group) == 2 and all(row["batch"] == 2 for _, row in group):
            labels = ("generated", "real_in_pair")
        elif len(group) == 1 and group[0][1]["batch"] == 1:
            labels = ("real_only",)
        else:
            counts["ambiguous_requests"] += len(group)
            continue
        for label, (serial, request) in zip(labels, group, strict=True):
            counts[f"{label}_requested"] += 1
            item = feedback.get((main_bridge, serial))
            if item is None or item["desired_ns"] != request["desired_ns"]:
                counts["missing_or_mismatched_feedback"] += 1
                continue
            prediction_kinds[item["timing_protocol"]] += 1
            if item["dropped"]:
                counts[f"{label}_reported_dropped"] += 1
                continue
            counts[f"{label}_with_feedback"] += 1
            submitted, queue_end = item["submitted_ns"], item["queue_end_ns"]
            latch, shown = item["latch_ns"], item["shown_ns"]
            record(label, "planned_to_actual_request_ms", item["planned_submit_ns"], submitted, signed=True)
            record(label, "request_to_queue_end_ms", submitted, queue_end, signed=True)
            record(label, "queue_end_to_latch_ms", queue_end, latch)
            record(label, "latch_to_compositor_reported_output_ms", latch, shown, signed=True)
            record(label, "request_to_compositor_reported_output_ms", submitted, shown)
            record(label, "target_to_compositor_reported_output_ms", item["desired_ns"], shown, signed=True)
            if label == "real_only":
                continue
            matching = [row for row in admitted[frame]
                        if 0 <= submitted - row["started_ns"] <= 100_000_000]
            if len(matching) == 1:
                row = matching[0]
                start = row["started_ns"]
                if row["source_ready_us"] >= 0:
                    record(label, "source_ready_to_compositor_reported_output_ms",
                           start + row["source_ready_us"] * 1000, shown)
                record(label, "generation_ready_to_present_request_ms",
                       start + row["output_ready_us"] * 1000, submitted)

    stages = defaultdict(dict)
    for (label, stage), values in samples.items():
        stages[label][stage] = {"samples": len(values), **stats(values)}
    return {"main_bridge": main_bridge, "counts": dict(counts),
            "compositor_feedback_protocols": dict(prediction_kinds), "stages_ms": dict(stages)}


def movement_admission(log, folder, visible_keys):
    start, stop = motion_times(folder / "input.log")
    rows = [integer_fields(line) for line in log.splitlines()
            if "anvil-timing " in line and "started_ns=" in line]
    if not rows:
        return None
    result = Counter()
    for row in rows:
        if not start <= row["started_ns"] < stop:
            continue
        result["generation_requests"] += 1
        if not row["worker_on_time"]:
            result["generation_deadline_miss"] += 1
        elif row["worker_rejected"]:
            result["motion_rejected"] += 1
        elif not row["admitted"]:
            result["no_generated_swapchain_image"] += 1
        else:
            result["admitted"] += 1
            key = (str(row["context"]), str(row["frame"]))
            if key in visible_keys:
                result["successful_vulkan_submit"] += 1
    return dict(result)


def verify_numbered_generation(log, folder, visible_keys, numbered):
    if not numbered:
        return None
    start, stop = motion_times(folder / "input.log")
    results = defaultdict(list)
    for pid, source, moment in re.findall(
            r"ANVIL fault worker: pid=(\d+) action=anvil-result-ready source=(\d+) monotonic_us=(\d+)", log):
        results[int(source)].append((int(pid), int(moment) * 1000))
    marks = {value for _, value in numbered}
    all_submitted = set()
    submitted_during_motion = set()
    motion_order = []
    unmatched_submissions = 0
    for line in log.splitlines():
        if "anvil-timing " not in line or "started_ns=" not in line:
            continue
        row = integer_fields(line)
        if (str(row["context"]), str(row["frame"])) not in visible_keys:
            continue
        first = row["started_ns"]
        last = first + row["output_ready_us"] * 1000 + 1000
        matched = [pid for pid, moment in results[row["source_value"]] if first <= moment <= last]
        if len(matched) != 1:
            unmatched_submissions += 1
            continue
        key = (matched[0] & 0xffff) << 16 | (row["source_value"] & 0xffff)
        all_submitted.add(key)
        if start <= first < stop:
            submitted_during_motion.add(key)
            motion_order.append((first, key))
    delivery = {
        "identified_submissions": len(all_submitted), "unmatched_submissions": unmatched_submissions,
        "unique_captured_ids": len(marks), "captured_ids_without_matched_submission": len(marks - all_submitted),
        "movement_submitted_ids": len(submitted_during_motion),
        "movement_submitted_ids_seen_in_capture": len(submitted_during_motion & marks),
    }
    before_jpeg = folder / "before-jpeg.csv"
    if before_jpeg.is_file():
        source_frames = read_rows(folder / "source-frames.csv")
        pre_encode = read_rows(before_jpeg)
        raw_ids = [int(row["generated_id"]) if row["generated_id"] else None for row in pre_encode]
        raw_marks = {value for value in raw_ids if value is not None}
        positions = [i for i, value in enumerate(raw_ids) if value in submitted_during_motion]
        source_motion = {int(row["pts_clock_ns"]) for row in source_frames
                         if start <= int(row["monotonic_ns"]) < stop}
        delivery.update(
            pipewire_buffers_during_movement=len(source_motion),
            before_jpeg_buffers_from_movement=sum(int(row["pts_clock_ns"]) in source_motion for row in pre_encode),
            movement_submitted_ids_seen_before_jpeg=len(submitted_during_motion & raw_marks),
            movement_submitted_ids_seen_before_jpeg_but_not_saved=len((submitted_during_motion & raw_marks) - marks),
            before_jpeg_ids_without_matched_submission=len(raw_marks - all_submitted),
            movement_raw_order_matches_submissions=(
                [raw_ids[i] for i in positions] == [key for _, key in sorted(motion_order)]),
            movement_raw_interleaves_one_unmarked_buffer=(
                bool(positions)
                and all(right - left == 2 for left, right in zip(positions, positions[1:]))
                and all(i + 1 < len(raw_ids) and raw_ids[i + 1] is None for i in positions)),
        )
    return delivery


def summarize_run(folder):
    log = (folder / "present.log").read_text(encoding="utf-8", errors="replace")
    real = vk_presents(log, "present-total")
    generated = vk_presents(log, "present-generated-image")
    rejected_keys = set(re.findall(
        r"anvil-timing context=(\d+) frame=(\d+)[^\n]*worker_rejected=1(?: |$)", log, re.M))
    visible_keys = set(re.findall(
        r"operation=present-generated-image context=(\d+)[^\n]*frame=(\d+)"
        r"[^\n]*result=(?:0|1000001003)(?: |$)", log, re.M))
    if rejected_keys & visible_keys:
        raise ValueError("An output rejected by motion analysis was successfully presented")
    display = {"successful_real_callbacks": sum(real.values()), "successful_generated_calls": sum(generated.values())}
    if generated:
        context = generated.most_common(1)[0][0]
        main_log = "\n".join(line for line in log.splitlines() if f"context={context} " in line)
        display["main_generated_context"] = {
            "id": context, "real": real[context], "generated": generated[context],
            "probes": timing(main_log),
        }
        movement = movement_admission(main_log, folder, visible_keys)
        if movement is not None:
            display["main_generated_context"]["motion_admission"] = movement
        measured = [tuple(int(value or 0) for value in row) for row in re.findall(
            r"anvil-timing [^\n]*source_ready_us=(-?\d+) output_ready_us=(-?\d+) admitted=([01])"
            r"[^\n]*deadline_us=(\d+) worker_on_time=([01])"
            r"(?: worker_rejected=([01]))? acquire_us=(\d+)", main_log)]
        gaps = [int(wait) / 1000 for wait in re.findall(r"anvil-pair [^\n]*pre_real_wait_us=(\d+)", main_log)]
        if measured:
            admitted = sum(row[2] and not row[5] for row in measured)
            on_time = sum(row[4] for row in measured)
            rejected = sum(row[5] for row in measured)
            timely_deltas = [(output - source) / 1000
                             for source, output, _, _, timely, cut, _ in measured
                             if timely and not cut and source >= 0 and output >= source]
            stages = {"requests": len(measured), "deadline_miss": len(measured) - on_time,
                      "motion_rejected_on_time": rejected,
                      "no_swapchain_image_after_output_ready": on_time - rejected - admitted,
                      "admitted": admitted,
                      "deadline_ms": stats([row[3] / 1000 for row in measured]),
                      "pre_present_output_ready_ms": stats(
                          [row[1] / 1000 for row in measured if row[2]]) if admitted else None,
                      "acquire_ms": stats([row[6] / 1000 for row in measured if row[4] and not row[5]])
                          if on_time > rejected else None,
                      "source_to_timely_generated_output_ms": stats(timely_deltas) if timely_deltas else None,
                      "gap_before_real_ms": stats(gaps) if gaps else None}
            display["main_generated_context"]["admission_stages"] = stages
            display["main_generated_context"]["probes"]["admitted"] = admitted
    calls = [tuple(map(int, item)) for item in re.findall(
        r"ANVIL HTP worker: pid=(\d+) source=(\d+) exec_us=(\d+) rtt_us=(\d+)", log)]
    pack_starts = {}
    source_to_inference = []
    for pid, action, source, moment in re.findall(
            r"ANVIL fault worker: pid=(\d+) action=(anvil-pack-start|anvil-result-ready) "
            r"source=(\d+) monotonic_us=(\d+)", log):
        key = (int(pid), int(source))
        if action == "anvil-pack-start":
            pack_starts[key] = int(moment)
        elif key in pack_starts:
            source_to_inference.append(int(moment) - pack_starts.pop(key))
    result = {"vulkan": display, "motion_rejected_and_not_presented": len(rejected_keys),
              "motion": analyze(folder), "software_response": software_response(folder),
              "capture_provenance": capture_provenance(folder), "combined_gpu_engine": combined_gpu(folder),
              "htp_jobs": len(calls), "worker_fallback_failures": len(re.findall(
                  r"anvil-deadline action=(?:schedule-failed|worker-lost)", log))}
    slots = bridge_stages(log)
    if slots is not None:
        result["compositor_slots"] = slots
    numbers = verify_numbered_generation(log, folder, visible_keys,
                                         result["capture_provenance"]["numbered_captures"])
    if numbers is not None:
        result["generated_frame_delivery"] = numbers
    queue_file = folder / "proton-queue.log"
    if queue_file.is_file():
        queue_log = queue_file.read_text(encoding="utf-8", errors="replace")
        result["proton_queue_setup"] = {
            "frame_limits_logged": sorted(set(map(int, re.findall(
                r"Ensure maximum latency of (\d+) frames", queue_log)))),
            "present_wait_extension_mentioned": "VK_KHR_present_wait" in queue_log,
        }
    if calls:
        result["htp"] = {
            "qnn_execute_ms": stats([item[2] / 1000 for item in calls]),
            "daemon_roundtrip_ms": stats([item[3] / 1000 for item in calls]),
            "pack_start_to_return_ms": stats([item / 1000 for item in source_to_inference]),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--daemon-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-tags", action="store_true")
    parser.add_argument("--queue-limit", type=int, choices=(1, 2))
    args = parser.parse_args()
    control, model = summarize_run(args.control), summarize_run(args.model)
    daemon = args.daemon_log.read_text(encoding="utf-8")
    clients = [(int(n), int(e)) for n, e in re.findall(r"gone: [^;]+; frames (\d+) errors (\d+)", daemon)]
    proof = {
        "qnn_graph": "backend=qnn graph=anvil_s_float" in daemon,
        "registered_clients": len(clients),
        "logged_inferences": sum(n for n, _ in clients),
        "logged_errors": sum(e for _, e in clients),
    }
    if (control["htp_jobs"] or control["vulkan"]["successful_generated_calls"]
            or not model["htp_jobs"] or not model["vulkan"]["successful_generated_calls"]
            or not proof["qnn_graph"] or not proof["logged_inferences"] or proof["logged_errors"]):
        raise ValueError(f"Failed the run-mode and hardware evidence checks: {proof!r}")
    if args.require_tags and (control["capture_provenance"]["tagged"]
                              or model["capture_provenance"]["tagged_motion"] < 5):
        raise ValueError("The captured generation marker could not be verified against the control")
    if args.queue_limit is not None:
        for run in (control, model):
            queue = run.get("proton_queue_setup", {})
            trials = run["software_response"]["trials"]
            if (queue.get("frame_limits_logged") != [args.queue_limit] or len(trials) != 6
                    or sum(trial["down_to_visible_ms"] is not None for trial in trials) < 5):
                raise ValueError(f"The queue override or matched input trials were not verified: {queue!r}")
    rendered = json.dumps({"control": control, "model": model, "device_daemon": proof}, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
