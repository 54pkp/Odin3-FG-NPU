"""Export OmniShift PyTorch checkpoint to ONNX and compile 5-scale W8A8 Hexagon HTP v79 context binaries.

Produces `<out-dir>/{360p,540p,720p,900p,1080p}/{vfi-joined.bin,vfi-motion3x.bin,phase12.bin,phase13.bin,phase23.bin}`
for Snapdragon 8 Elite (`SM8750`, `soc_id=69`, `dsp_arch=v79`).

Usage:
  python training/export_and_compile_htp.py \\
      --ckpt runs/omnishift_sm8750/omnishift_vfi_sm8750.pt \\
      --qairt-sdk /path/to/qairt/2.x \\
      --out-dir models/omnishift-sm8750
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import onnx
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from omnishift_vfi import load_omnishift_v7_checkpoint  # noqa: E402

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "360p": (360, 640),
    "540p": (540, 960),
    "720p": (720, 1280),
    "900p": (900, 1600),
    "1080p": (1080, 1920),
}


class Joined2xWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.m = model

    def forward(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
        out = self.m(frame_a.permute(0, 3, 1, 2), frame_b.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class SharedEncoder3xWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.m = model

    def forward(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
        pa64, pb64, _, _, _ = self.m.unpack_endpoints_4x4(frame_a.permute(0, 3, 1, 2), frame_b.permute(0, 3, 1, 2))
        lat = self.m.encode_shared_from_64(pa64, pb64)
        return lat.permute(0, 2, 3, 1)


class PhaseDecoder3xWrapper(nn.Module):
    def __init__(self, model: nn.Module, h: int, w: int):
        super().__init__()
        self.m = model
        self.h = h
        self.w = w

    def forward(self, frame_a: torch.Tensor, frame_b: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        pa64, pb64, _, _, _ = self.m.unpack_endpoints_4x4(frame_a.permute(0, 3, 1, 2), frame_b.permute(0, 3, 1, 2))
        out = self.m.decode_phase_from_64(pa64, pb64, motion.permute(0, 3, 1, 2), self.h, self.w, True)
        return out.permute(0, 2, 3, 1)


def set_onnx_graph_name(onnx_path: Path, graph_name: str) -> None:
    m = onnx.load(str(onnx_path))
    m.graph.name = graph_name
    onnx.save(m, str(onnx_path))


def write_encodings(path: Path, include_motion: bool = False, motion_only: bool = False) -> None:
    enc_01 = [{"bitwidth": 8, "is_symmetric": "false", "max": 1.0, "min": 0.0, "offset": 0.0, "scale": float(np.float32(1.0 / 255.0))}]
    act = {"frame_a": enc_01, "frame_b": enc_01}
    if motion_only:
        act["motion"] = enc_01
    else:
        if include_motion:
            act["motion"] = enc_01
        act["frame"] = enc_01
    path.write_text(json.dumps({"version": "0.6.1", "activation_encodings": act, "param_encodings": {}}, indent=2) + "\n", encoding="utf-8")


def generate_calibration_lists(cal_dir: Path, h: int, w: int, model_eval: nn.Module, seed: int) -> tuple[Path, Path]:
    cal_dir.mkdir(parents=True, exist_ok=True)
    enc_wrap = SharedEncoder3xWrapper(copy.deepcopy(model_eval).eval())
    dev = next(enc_wrap.parameters()).device
    rng = np.random.default_rng(seed)
    lines2, lines3 = [], []

    with torch.no_grad():
        for i in range(8):
            fa = np.ones((h, w, 4), dtype=np.float32)
            fb = np.ones((h, w, 4), dtype=np.float32)
            if i == 0:
                # Pure black / pure white / monochrome split for zero-offset & partition-of-unity calibration
                fa[:, : w // 2, :3] = 0.0
                fa[:, w // 2 :, :3] = 1.0
                fb[...] = fa
            else:
                yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
                xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
                tex = np.clip(0.5 + 0.35 * np.sin(xx * 37.0 + i) * np.cos(yy * 29.0 - i) + rng.normal(0, 0.05, (h, w, 3)).astype(np.float32), 0.0, 1.0)
                dx = int(rng.choice([-48, -24, -8, 0, 8, 24, 48]))
                dy = int(rng.choice([-24, -8, 0, 8, 24]))
                fa[..., :3] = tex
                fb[..., :3] = np.roll(np.roll(tex, shift=dx, axis=1), shift=dy, axis=0)
            pa_f, pb_f = cal_dir / f"a_{i}.raw", cal_dir / f"b_{i}.raw"
            fa.tofile(pa_f)
            fb.tofile(pb_f)
            ln2 = f"frame_a:={pa_f.as_posix()} frame_b:={pb_f.as_posix()}"
            lines2.append(ln2)

            lat = enc_wrap(torch.from_numpy(fa).unsqueeze(0).to(dev), torch.from_numpy(fb).unsqueeze(0).to(dev)).cpu().numpy()
            pm_f = cal_dir / f"m_{i}.raw"
            lat.astype(np.float32).tofile(pm_f)
            lines3.append(f"{ln2} motion:={pm_f.as_posix()}")

    cal2, cal3 = cal_dir / "cal_2x.txt", cal_dir / "cal_3x.txt"
    cal2.write_text("\n".join(lines2) + "\n", encoding="utf-8")
    cal3.write_text("\n".join(lines3) + "\n", encoding="utf-8")
    return cal2, cal3


def host_qairt_subdir() -> tuple[str, str]:
    if platform.system().lower().startswith("win"):
        return "x86_64-windows-msvc", "QnnHtpNetRunExtensions.dll"
    return "x86_64-linux-clang", "libQnnHtpNetRunExtensions.so"


def compile_one_graph(job: dict, sdk: Path, qairt_py: Path) -> dict:
    t0 = time.perf_counter()
    wdir: Path = job["wdir"]
    gname: str = job["gname"]
    stem: str = job["stem"]
    enc_p: Path = job["enc_p"]
    cal_p: Path = job["cal_p"]
    ins: list[str] = job["ins"]
    out_n: str = job["out_n"]
    dst_bin: Path = job["dst_bin"]

    host_arch, ext_lib_name = host_qairt_subdir()
    bins = sdk / "bin" / host_arch
    libs = sdk / "lib" / host_arch
    tmp_d = wdir / "tmp"
    tmp_d.mkdir(parents=True, exist_ok=True)
    (wdir / "float_dlc").mkdir(parents=True, exist_ok=True)

    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
    env.update({k: str(sdk) for k in ("QAIRT_SDK_ROOT", "QNN_SDK_ROOT", "SNPE_ROOT", "AISW_SDK_ROOT")})
    env.update(
        PATH=os.pathsep.join([str(bins), str(libs), str(qairt_py.parent), env.get("PATH", "")]),
        PYTHONPATH=str(sdk / "lib/python"),
        QAIRT_TMP_DIR=str(tmp_d),
        TMP=str(tmp_d),
        TEMP=str(tmp_d),
    )

    layout: list[str] = []
    for n in ins:
        layout.extend(["--source_model_input_layout", n, "NHWC", "--desired_input_layout", n, "NHWC"])
    layout.extend(["--source_model_output_layout", out_n, "NHWC", "--desired_output_layout", out_n, "NHWC"])

    subprocess.run(
        [str(qairt_py), "-B", str(bins / "qairt-converter"), "--input_network", f"{gname}.onnx",
         "--output_path", f"float_dlc/{gname}.dlc", "--target_backend", "HTP",
         "--quantization_overrides", str(enc_p.resolve()), *layout],
        cwd=wdir, env=env, check=True,
    )
    subprocess.run(
        [str(qairt_py), "-B", str(bins / "qairt-quantizer"), "--input_dlc", f"float_dlc/{gname}.dlc",
         "--output_dlc", "w8a8.dlc", "--input_list", str(cal_p.resolve()),
         "--act_bitwidth", "8", "--weights_bitwidth", "8", "--bias_bitwidth", "32", "--target_backend", "HTP"],
        cwd=wdir, env=env, check=True,
    )
    htp_cfg = wdir / "htp.json"
    htp_cfg.write_text(
        json.dumps({
            "devices": [{"dsp_arch": "v79", "soc_id": 69, "device_id": 0, "pd_session": "unsigned"}],
            "graphs": [{"graph_names": [gname], "vtcm_mb": 0, "O": 3}],
        }),
        encoding="utf-8",
    )
    be_cfg = wdir / "backend.json"
    be_cfg.write_text(
        json.dumps({
            "backend_extensions": {
                "shared_library_path": str((libs / ext_lib_name).resolve()),
                "config_file_path": str(htp_cfg.resolve()),
            }
        }),
        encoding="utf-8",
    )
    cb_bin = bins / ("qnn-context-binary-generator.exe" if os.name == "nt" else "qnn-context-binary-generator")
    dlc_lib = libs / ("QnnModelDlc.dll" if os.name == "nt" else "libQnnModelDlc.so")
    htp_lib = libs / ("QnnHtp.dll" if os.name == "nt" else "libQnnHtp.so")
    subprocess.run(
        [str(cb_bin), "--model", str(dlc_lib), "--backend", str(htp_lib),
         "--dlc_path", "w8a8.dlc", "--config_file", str(be_cfg.resolve()),
         "--output_dir", ".", "--binary_file", stem],
        cwd=wdir, env=env, check=True,
    )
    shutil.copy2(wdir / f"{stem}.bin", dst_bin)
    return {"res": job["res"], "stem": stem, "kb": round(dst_bin.stat().st_size / 1024.0, 1), "sec": round(time.perf_counter() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True, help="Trained OmniShift .pt checkpoint")
    ap.add_argument("--qairt-sdk", type=Path, default=Path(os.environ.get("QAIRT_SDK_ROOT", "")))
    ap.add_argument("--qairt-python", type=Path, default=Path(sys.executable))
    ap.add_argument("--out-dir", type=Path, default=Path("models/omnishift-sm8750"))
    ap.add_argument("--scales", nargs="+", default=list(RESOLUTIONS), choices=list(RESOLUTIONS))
    ap.add_argument("--export-only", action="store_true", help="Export ONNX graphs without invoking QAIRT compiler")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model_eval, _ = load_omnishift_v7_checkpoint(args.ckpt, device=dev)
    work = args.out_dir.parent / ".compile-omnishift-work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    enc_2x, enc_mot, enc_dec = work / "enc_2x.json", work / "enc_mot.json", work / "enc_dec.json"
    write_encodings(enc_2x, include_motion=False)
    write_encodings(enc_mot, motion_only=True)
    write_encodings(enc_dec, include_motion=True)

    m_cpu = copy.deepcopy(model_eval).cpu().eval()
    m12 = copy.deepcopy(m_cpu); m12.switch_to_deploy(0.5)
    m13 = copy.deepcopy(m_cpu); m13.switch_to_deploy(1.0 / 3.0)
    m23 = copy.deepcopy(m_cpu); m23.switch_to_deploy(2.0 / 3.0)

    jobs: list[dict] = []
    for idx, res in enumerate(args.scales):
        h, w = RESOLUTIONS[res]
        r_work = work / res
        r_out = args.out_dir / res
        r_work.mkdir(parents=True, exist_ok=True)
        r_out.mkdir(parents=True, exist_ok=True)
        cal2, cal3 = generate_calibration_lists(r_work / "cal", h, w, model_eval, seed=20261100 + idx)
        lat_h, lat_w = ((h + 31) // 32 * 32) // 8, ((w + 31) // 32 * 32) // 8
        da, db, dm = torch.rand(1, h, w, 4), torch.rand(1, h, w, 4), torch.rand(1, lat_h, lat_w, 64)
        specs = [
            ("2x", Joined2xWrapper(m12), (da, db), ["frame_a", "frame_b"], "frame", "fg4_joined_1_2", "vfi-joined", enc_2x, cal2),
            ("mot", SharedEncoder3xWrapper(m13), (da, db), ["frame_a", "frame_b"], "motion", "fg4_motion", "vfi-motion3x", enc_mot, cal2),
            ("p12", PhaseDecoder3xWrapper(m12, h, w), (da, db, dm), ["frame_a", "frame_b", "motion"], "frame", "fg4_phase_1_2", "phase12", enc_dec, cal3),
            ("p13", PhaseDecoder3xWrapper(m13, h, w), (da, db, dm), ["frame_a", "frame_b", "motion"], "frame", "fg4_phase_1_3", "phase13", enc_dec, cal3),
            ("p23", PhaseDecoder3xWrapper(m23, h, w), (da, db, dm), ["frame_a", "frame_b", "motion"], "frame", "fg4_phase_2_3", "phase23", enc_dec, cal3),
        ]
        for sub, mod, inputs, in_names, out_name, gname, stem, enc_p, cal_p in specs:
            gdir = r_work / sub
            gdir.mkdir(parents=True, exist_ok=True)
            onnx_p = gdir / f"{gname}.onnx"
            print(f"[export {res}/{stem}] latent={lat_w}x{lat_h}", flush=True)
            torch.onnx.export(mod, inputs, str(onnx_p), input_names=in_names, output_names=[out_name],
                              opset_version=14, do_constant_folding=True, dynamo=False)
            set_onnx_graph_name(onnx_p, gname)
            jobs.append({"res": res, "wdir": gdir, "gname": gname, "stem": stem, "enc_p": enc_p, "cal_p": cal_p,
                         "ins": in_names, "out_n": out_name, "dst_bin": r_out / f"{stem}.bin"})

    if args.export_only:
        print(f"Exported {len(jobs)} ONNX graphs in {work}", flush=True)
        return

    if not args.qairt_sdk or not args.qairt_sdk.is_dir():
        raise SystemExit("Pass --qairt-sdk /path/to/qairt/2.x (or --export-only)")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(compile_one_graph, j, args.qairt_sdk.resolve(), args.qairt_python.resolve()): j for j in jobs}
        for f in as_completed(futs):
            info = f.result()
            results.append(info)
            print(f"  [OK {info['res']}/{info['stem']}.bin] {info['kb']} KB in {info['sec']}s", flush=True)

    shutil.rmtree(work, ignore_errors=True)
    (args.out_dir / "manifest.json").write_text(
        json.dumps({"model": "OmniShift-HTP-v79-SM8750", "resolutions": args.scales, "binaries": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Compiled {len(results)} HTP v79 binaries into {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
