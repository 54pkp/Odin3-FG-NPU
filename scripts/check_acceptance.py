"""Strict acceptance verifier for Odin 3 Odin3 FG Live Suites (witness, obradinn, gpu_audit)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

G8 = Path(__file__).resolve().parent.parent / 'docs/benchmarks'

WITNESS_ORDER = [
    "g8_witness_0x",
    "g8_witness_2x_max",
    "g8_witness_2x_balanced",
    "g8_witness_2x_eco",
    "g8_witness_3x_max",
    "g8_witness_3x_eco",
    "g8_witness_2x60_max",
    "g8_witness_2x60_eco",
    "g8_witness_3x40_max",
    "g8_witness_3x40_eco",
]


def verify_witness() -> None:
    p = G8 / "live-acceptance-witness.json"
    by = {r["name"]: r for r in json.loads(p.read_text(encoding="utf-8"))}
    rows = [by[k] for k in WITNESS_ORDER]
    p.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    r0 = by["g8_witness_0x"]["real_screen_fps"]
    g0 = by["g8_witness_0x"]["gpu_game_pct"]
    errs: list[str] = []
    print(f"=== Witness Baseline 0x: real={r0:.2f} FPS (97% floor={0.97 * r0:.2f} FPS), gpu_game={g0:.2f}% ===")
    for r in rows[1:]:
        n = r["name"]
        gd = (r["gpu_game_pct"] + r["gpu_worker_pct"]) - g0
        if r["gpu_worker_pct"] != 0.0 or gd >= 2.0:
            errs.append(f"{n}: gpu_delta={gd:+.2f}% worker={r['gpu_worker_pct']}%")
        if r["repeated_frames"] != 0 or not r["saves_unchanged"] or not r["restore_ok"]:
            errs.append(f"{n}: repeated={r['repeated_frames']} saves={r['saves_unchanged']}")
        if r["cap"] == 0:
            pct = 100.0 * r["real_screen_fps"] / r0
            req_g = 60.0 if r["mult"] == 2 else 80.0
            print(
                f"  [UNCAPPED] {n:24s}: real={r['real_screen_fps']:6.2f} ({pct:6.2f}%), "
                f"gen={r['gen_screen_fps']:6.2f} (>{req_g:.0f}), tot={r['total_screen_fps']:6.2f}, "
                f"pairs={r['pairs_in_window']} exp={r['expected_gen']} act={r['actual_gen']} miss={r['missed_gen']}, "
                f"gpu_delta={gd:+.2f}%"
            )
            if pct < 97.0 or r["gen_screen_fps"] <= req_g:
                errs.append(f"{n}: native_pct={pct:.2f}% gen={r['gen_screen_fps']}")
        elif r["cap"] == 60:
            tot = r["total_screen_fps"]
            print(
                f"  [LOCK 2x60] {n:23s}: real={r['real_screen_fps']:6.2f} + gen={r['gen_screen_fps']:6.2f} = {tot:6.2f} FPS, "
                f"share={r['orig_share_pct']}:{r['gen_share_pct']}, exp={r['expected_gen']} act={r['actual_gen']} miss={r['missed_gen']}, "
                f"gpu_delta={gd:+.2f}%"
            )
            if not (118.80 <= tot <= 121.20) or r["missed_gen"] != 0:
                errs.append(f"{n}: tot={tot} missed_gen={r['missed_gen']}")
        elif r["cap"] == 40:
            tot = r["total_screen_fps"]
            print(
                f"  [LOCK 3x40] {n:23s}: real={r['real_screen_fps']:6.2f} + gen={r['gen_screen_fps']:6.2f} = {tot:6.2f} FPS, "
                f"share={r['orig_share_pct']}:{r['gen_share_pct']}, exp={r['expected_gen']} act={r['actual_gen']} miss={r['missed_gen']}, "
                f"gpu_delta={gd:+.2f}%"
            )
            if not (117.60 <= tot <= 122.40) or r["missed_gen"] != 0:
                errs.append(f"{n}: tot={tot} missed_gen={r['missed_gen']}")
    if errs:
        raise SystemExit("WITNESS FAILED:\n  " + "\n  ".join(errs))
    print("ALL 10 WITNESS CASES PASSED 100%!\n")


def verify_obradinn() -> None:
    p = G8 / "live-acceptance-obradinn.json"
    rows = json.loads(p.read_text(encoding="utf-8"))
    by = {r["name"]: r for r in rows}
    g0 = by["g8_obra_0x"]["gpu_game_pct"]
    errs: list[str] = []
    print(f"=== Obra Dinn Baseline 0x: real={by['g8_obra_0x']['real_screen_fps']:.2f} FPS, gpu_game={g0:.2f}% ===")
    for r in rows[1:]:
        n = r["name"]
        gd = (r["gpu_game_pct"] + r["gpu_worker_pct"]) - g0
        tot = r["total_screen_fps"]
        if r["gpu_worker_pct"] != 0.0 or gd >= 2.0:
            errs.append(f"{n}: gpu_delta={gd:+.2f}% worker={r['gpu_worker_pct']}%")
        if r["repeated_frames"] != 0 or not r["saves_unchanged"] or not r["restore_ok"]:
            errs.append(f"{n}: repeated={r['repeated_frames']} saves={r['saves_unchanged']}")
        if r["cap"] == 60 and (not (118.80 <= tot <= 121.20) or r["missed_gen"] != 0):
            errs.append(f"{n}: tot={tot} missed_gen={r['missed_gen']}")
        if r["cap"] == 40 and (not (117.60 <= tot <= 122.40) or r["missed_gen"] != 0):
            errs.append(f"{n}: tot={tot} missed_gen={r['missed_gen']}")
        f = r.get("forensic") or {}
        if case_requires_forensic := (n in ("g8_obra_2x_max", "g8_obra_2x_eco")):
            phases = f.get("phases") or {}
            if not phases:
                errs.append(f"{n}: missing forensic phases capture")
            for m_name, pm in phases.items():
                mc = pm.get("mono_chroma_max_u8", 999.0)
                oob = pm.get("bot4_oob_p99_u8", 999.0)
                pw = pm.get("pure_white_p999_err_u8", 999.0)
                pb = pm.get("pure_black_p999_err_u8", 999.0)
                print(f"    [FORENSIC {n} {m_name}] mono_chroma_max={mc} u8, bot4_oob_p99={oob} u8, pw_p999={pw} u8, pb_p999={pb} u8")
                if mc != 0.0 or oob > 1.0 or pw > 1.0 or pb > 1.0:
                    errs.append(f"{n}/{m_name}: forensic failed {pm}")
        print(
            f"  [OBRA] {n:24s}: real={r['real_screen_fps']:6.2f} + gen={r['gen_screen_fps']:6.2f} = {tot:6.2f} FPS, "
            f"scale={r['matched_scale']}, exp={r['expected_gen']} act={r['actual_gen']} miss={r['missed_gen']}, "
            f"gpu_delta={gd:+.2f}%, forensic={bool(f)}"
        )
    if errs:
        raise SystemExit("OBRADINN FAILED:\n  " + "\n  ".join(errs))
    print("ALL 7 OBRA DINN CASES PASSED 100%!\n")


def verify_gpu_audit() -> None:
    p = G8 / "live-acceptance-gpu_audit.json"
    rows = json.loads(p.read_text(encoding="utf-8"))
    by = {r["name"]: r for r in rows}
    errs: list[str] = []
    for base_name, prefix in (("g8_scarlet_0x", "g8_scarlet_"), ("g8_talos_0x", "g8_talos_")):
        g0 = by[base_name]["gpu_game_pct"]
        print(f"=== {base_name}: real={by[base_name]['real_screen_fps']:.2f} FPS, gpu_game={g0:.2f}% ===")
        for r in rows:
            n = r["name"]
            if not n.startswith(prefix) or n == base_name:
                continue
            gd = (r["gpu_game_pct"] + r["gpu_worker_pct"]) - g0
            tot = r["total_screen_fps"]
            if r["gpu_worker_pct"] != 0.0 or gd >= 2.0:
                errs.append(f"{n}: gpu_delta={gd:+.2f}% worker={r['gpu_worker_pct']}%")
            if r["repeated_frames"] != 0 or not r["saves_unchanged"] or not r["restore_ok"]:
                errs.append(f"{n}: repeated={r['repeated_frames']} saves={r['saves_unchanged']}")
            if r["cap"] == 60 and (not (118.80 <= tot <= 121.20) or r["missed_gen"] != 0):
                errs.append(f"{n}: tot={tot} missed_gen={r['missed_gen']}")
            if r["cap"] == 40 and (not (117.60 <= tot <= 122.40) or r["missed_gen"] != 0):
                errs.append(f"{n}: tot={tot} missed_gen={r['missed_gen']}")
            print(
                f"  [AUDIT] {n:23s}: real={r['real_screen_fps']:6.2f} + gen={r['gen_screen_fps']:6.2f} = {tot:6.2f} FPS, "
                f"scale={r['matched_scale']}, exp={r['expected_gen']} act={r['actual_gen']} miss={r['missed_gen']}, "
                f"gpu_delta={gd:+.2f}%"
            )
    if errs:
        raise SystemExit("GPU_AUDIT FAILED:\n  " + "\n  ".join(errs))
    print("ALL 8 GPU_AUDIT CASES PASSED 100%!\n")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target == "witness_smoke":
        p = G8 / "live-acceptance-witness.json"
        by = {r["name"]: r for r in json.loads(p.read_text(encoding="utf-8"))}
        r0 = by["g8_witness_0x"]["real_screen_fps"]
        g0 = by["g8_witness_0x"]["gpu_game_pct"]
        rb = by["g8_witness_2x_balanced"]
        rm3 = by["g8_witness_3x_max"]
        re3 = by["g8_witness_3x_eco"]
        pct2 = 100.0 * rb["real_screen_fps"] / r0
        pct3m = 100.0 * rm3["real_screen_fps"] / r0
        pct3e = 100.0 * re3["real_screen_fps"] / r0
        gd2 = (rb["gpu_game_pct"] + rb["gpu_worker_pct"]) - g0
        gd3m = (rm3["gpu_game_pct"] + rm3["gpu_worker_pct"]) - g0
        gd3e = (re3["gpu_game_pct"] + re3["gpu_worker_pct"]) - g0
        print(
            f"[SMOKE] 0x={r0:.2f} FPS (gpu={g0:.2f}%) | 2x_bal={rb['real_screen_fps']:.2f} ({pct2:.2f}%, gd={gd2:+.2f}%), gen={rb['gen_screen_fps']:.2f} | "
            f"3x_max={rm3['real_screen_fps']:.2f} ({pct3m:.2f}%, gd={gd3m:+.2f}%), gen={rm3['gen_screen_fps']:.2f} | "
            f"3x_eco={re3['real_screen_fps']:.2f} ({pct3e:.2f}%, gd={gd3e:+.2f}%), gen={re3['gen_screen_fps']:.2f}"
        )
        if (
            pct2 < 97.0 or rb["gen_screen_fps"] <= 60.0 or gd2 >= 2.0
            or pct3m < 97.0 or rm3["gen_screen_fps"] <= 80.0 or gd3m >= 2.0
            or pct3e < 97.0 or re3["gen_screen_fps"] <= 80.0 or gd3e >= 2.0
        ):
            raise SystemExit(
                f"SMOKE FAILED: 2x={pct2:.2f}%/{rb['gen_screen_fps']}/gd={gd2:+.2f}%, 3x_max={pct3m:.2f}%/{rm3['gen_screen_fps']}/gd={gd3m:+.2f}%, 3x_eco={pct3e:.2f}%/{re3['gen_screen_fps']}/gd={gd3e:+.2f}%"
            )
    if target in ("witness", "all"):
        verify_witness()
    if target in ("obradinn", "all"):
        verify_obradinn()
    if target in ("gpu_audit", "all"):
        verify_gpu_audit()
