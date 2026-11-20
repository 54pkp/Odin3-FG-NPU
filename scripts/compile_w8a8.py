"""Convert a fixed-shape ANVIL-S ONNX model and calibrate QNN W8A8."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--io-layout", choices=("NCHW", "NHWC"), default="NCHW")
    args = parser.parse_args()
    sdk = args.sdk.resolve(strict=True)
    model = args.model.resolve(strict=True)
    calibration = args.calibration.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    binaries = sdk / "bin" / "x86_64-windows-msvc"
    libraries = sdk / "lib" / "x86_64-windows-msvc"
    variables = ("QAIRT_SDK_ROOT", "QNN_SDK_ROOT", "SNPE_ROOT", "AISW_SDK_ROOT")
    environment = {name: value for name, value in os.environ.items() if not name.upper().startswith("PYTHON")}
    environment.update({name: str(sdk) for name in variables})
    environment.update(PATH=os.pathsep.join(map(str, (binaries, libraries, Path(sys.executable).parent))),
                       PYTHONPATH=str(sdk / "lib" / "python"), PYTHONUTF8="1", PYTHONNOUSERSITE="1")
    calibration_paths = [(calibration / line).resolve(strict=True)
                         for line in (calibration / "input_list.txt").read_text().splitlines() if line.strip()]
    sample_bytes = 1 * 6 * args.height * args.width * 4
    if not calibration_paths or any(path.stat().st_size != sample_bytes for path in calibration_paths):
        raise ValueError("Expected full float32 NCHW calibration pairs")
    if args.io_layout == "NHWC":
        native_dir = output / "calibration-nhwc"
        native_dir.mkdir()
        native_inputs = []
        for path in calibration_paths:
            tensor = np.fromfile(path, dtype="<f4").reshape(1, 6, args.height, args.width)
            target = native_dir / path.name
            tensor.transpose(0, 2, 3, 1).tofile(target)
            native_inputs.append(target)
        calibration_paths = native_inputs
    samples = output / "calibration.txt"
    samples.write_text("".join(f"{path.as_posix()}\n" for path in calibration_paths), encoding="utf-8")
    layout = ["--source_model_input_layout", "input", "NCHW", "--desired_input_layout", "input", args.io_layout,
              "--source_model_output_layout", "output", "NCHW", "--desired_output_layout", "output", args.io_layout]
    shape = f"1,6,{args.height},{args.width}"
    commands = [
        ("qairt-converter", ["--input_network", str(model), "--output_path", "anvil-s.float.dlc",
                             "--target_backend", "HTP", "--source_model_input_shape", "input", shape, *layout]),
        ("qairt-quantizer", ["--input_dlc", "anvil-s.float.dlc", "--output_dlc", "anvil-s.w8a8.dlc",
                             "--input_list", str(samples), "--act_bitwidth", "8", "--weights_bitwidth", "8",
                             "--bias_bitwidth", "32", "--target_backend", "HTP", "--dump_encoding_json"]),
        ("qairt-dlc-info", ["-i", "anvil-s.w8a8.dlc", "-d"]),
    ]
    for name, flags in commands:
        print(f"Running {name}", flush=True)
        with (output / f"{name}.log").open("x", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-B", str(binaries / name), *flags], cwd=output, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, timeout=900, check=True)
    print(output / "anvil-s.w8a8.dlc")


if __name__ == "__main__":
    main()
