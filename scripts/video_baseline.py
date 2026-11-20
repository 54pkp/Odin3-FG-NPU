"""Prepare ANVIL's video inputs with its published codec-vector prealignment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import av
from av.sidedata.sidedata import Type
import cv2
import numpy as np


def codec_vectors(frame: av.VideoFrame) -> dict[str, np.ndarray | int] | None:
    side_data = frame.side_data.get(Type.MOTION_VECTORS)
    if side_data is None:
        return None
    vectors = side_data.to_ndarray()
    if not len(vectors):
        return None
    if np.any(vectors["source"] >= 0):
        raise ValueError("Expected motion vectors referring to a previous frame")
    scales = np.unique(vectors["motion_scale"])
    if len(scales) != 1 or int(scales[0]) <= 0:
        raise ValueError("Inconsistent codec motion-vector scale")
    return {target: vectors[key] for target, key in {
        "blockw": "w", "blockh": "h", "dstx": "dst_x", "dsty": "dst_y",
        "motion_x": "motion_x", "motion_y": "motion_y",
    }.items()} | {"motion_scale": int(scales[0])}


def load_prealignment(upstream: Path):
    location = upstream / "anvil_exp01" / "data" / "prealign_v2.py"
    spec = importlib.util.spec_from_file_location("anvil_prealign_v2", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._prealign_med_gauss


def write_image(path: Path, rgb: np.ndarray) -> None:
    pixels = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Unable to write {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--resolution", choices=("1080p", "720p"), default="1080p")
    parser.add_argument("--start", type=int, default=30)
    parser.add_argument("--stride", type=int, default=11)
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    if args.start < 1 or args.stride < 1 or args.pairs < 1:
        parser.error("start, stride, and pairs must be positive")
    width, height = (1920, 1080) if args.resolution == "1080p" else (1280, 720)
    prealign = load_prealignment(args.upstream)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    cv2.setNumThreads(4)
    session = None
    if args.model:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        session = ort.InferenceSession(str(args.model), sess_options=opts, providers=["CPUExecutionProvider"])
        if session.get_inputs()[0].shape != [1, 6, height, width]:
            raise ValueError("ANVIL-S model input shape does not match the selected resolution")
    records, input_list = [], []
    with av.open(str(args.video)) as video:
        stream = video.streams.video[0]
        if stream.codec_context.name != "h264" or stream.codec_context.has_b_frames:
            raise ValueError("Expected an H.264 clip without B frames")
        stream.codec_context.options = {**stream.codec_context.options, "flags2": "+export_mvs"}
        previous = None
        next_index = args.start
        for index, frame in enumerate(video.decode(stream)):
            if index < next_index - 1:
                continue
            if frame.width != 1920 or frame.height != 1080:
                raise ValueError("Expected an official 1080p demo clip")
            rgb = frame.to_ndarray(format="rgb24")
            if previous is not None and index >= next_index:
                if int(frame.pict_type) != 2:
                    raise ValueError(f"Unexpected non-P-frame at frame {index}")
                vectors = codec_vectors(frame)
                if vectors is None:
                    raise ValueError(f"No past motion vectors at frame {index}")
                started = perf_counter()
                first, second = prealign(previous, rgb, vectors)
                align_ms = (perf_counter() - started) * 1000
                if args.resolution == "720p":
                    first = cv2.resize(first, (width, height), interpolation=cv2.INTER_AREA)
                    second = cv2.resize(second, (width, height), interpolation=cv2.INTER_AREA)
                pair = np.concatenate((first, second), axis=2).astype(np.float32) / np.float32(255)
                nchw = np.ascontiguousarray(pair.transpose(2, 0, 1)[None, ...], dtype="<f4")
                raw = output / f"pair-{index:05}.raw"
                nchw.tofile(raw)
                input_list.append(f"{raw.name}")
                result = {"frame": index, "pts": frame.pts, "vectors": len(vectors["motion_x"]),
                          "cpu_align_ms": align_ms, "input": raw.name}
                if session is not None:
                    started = perf_counter()
                    residual = session.run(None, {"input": nchw})[0]
                    result["cpu_onnx_ms"] = (perf_counter() - started) * 1000
                    if residual.shape != (1, 3, height, width) or not np.isfinite(residual).all():
                        raise RuntimeError("Invalid ANVIL-S output")
                    residual = residual[0].transpose(1, 2, 0)
                    residual.tofile(output / f"residual-{index:05}.raw")
                    blend = (pair[:, :, :3] + pair[:, :, 3:]) * np.float32(0.5)
                    write_image(output / f"blend-{index:05}.png", blend)
                    write_image(output / f"anvil-{index:05}.png", blend + residual)
                    result["residual_abs_mean"] = float(np.abs(residual).mean())
                records.append(result)
                print(json.dumps(result), flush=True)
                if len(records) == args.pairs:
                    break
                next_index = index + args.stride
            previous = rgb
    if len(records) != args.pairs:
        raise RuntimeError(f"Only decoded {len(records)} requested frame pairs")
    (output / "input_list.txt").write_text("\n".join(input_list) + "\n", encoding="utf-8")
    report = {"video_sha256": hashlib.file_digest(args.video.open("rb"), "sha256").hexdigest(),
              "model_sha256": hashlib.file_digest(args.model.open("rb"), "sha256").hexdigest() if args.model else None,
              "resolution": args.resolution, "pairs": records}
    (output / "reference.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
