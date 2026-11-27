"""Build all Odin 3 FG NPU binaries, runtime archive, and Decky plugins from source.

Environment / CLI overrides for external SDK headers & toolchains:
  --zig              or ODIN3_ZIG             : Path to Zig >= 0.14 executable (cross-compiles to aarch64-linux-gnu)
  --vulkan-include   or ODIN3_VULKAN_INCLUDE  : Path to Khronos Vulkan-Headers/include
  --linux-sysroot    or ODIN3_LINUX_SYSROOT   : Path to Linux/X11/XCB/Wayland/DRM usr/include sysroot headers
  --qairt-include    or ODIN3_QAIRT_INCLUDE   : Path to Qualcomm QAIRT SDK include/QNN directory
  --vulkan-stub-so   or ODIN3_VULKAN_STUB_SO  : Path to aarch64 libvulkan.so.1 link stub
  --fastrpc-dir      or ODIN3_FASTRPC_DIR     : Directory or base runtime tar providing open-source cdsprpcd & libcdsprpc.so
  --models-dir       or ODIN3_MODELS_DIR      : Optional directory of compiled 5-scale OmniShift .bin contexts to bundle
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "runtime"))
from install_runtime import EXECUTABLES, LAYER_MANIFEST, MANIFEST, SHARED_INPUT_PROTOCOL, WSI_CAPTURE, verify_archive  # noqa: E402


def file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def find_node() -> Path:
    for cand in (shutil.which("node"), r"C:\\Program Files\\nodejs\\node.exe", r"D:\\Program Files\\nodejs\\node.exe"):
        if cand and Path(cand).is_file():
            return Path(cand)
    for env_key in ("APPDATA", "LOCALAPPDATA"):
        base = os.environ.get(env_key)
        if base and Path(base).exists():
            for hit in Path(base).rglob("node.exe"):
                return hit
    raise FileNotFoundError("node.exe not found; install Node.js >= 18 to build Decky frontend bundles")


def resolve_path(cli_val: Path | None, env_name: str, fallbacks: list[Path], desc: str) -> Path:
    if cli_val and cli_val.exists():
        return cli_val.resolve()
    env_v = os.environ.get(env_name)
    if env_v and Path(env_v).exists():
        return Path(env_v).resolve()
    for fb in fallbacks:
        if fb.exists():
            return fb.resolve()
    raise FileNotFoundError(f"Missing {desc}: pass via CLI or set {env_name}")


def compile_renderer(zig: Path, vk_inc: Path, sysroot_inc: Path, out_dir: Path, jobs: int = 8) -> Path:
    eng = REPO / "engine"
    obj_dir = out_dir / "obj-renderer"
    shutil.rmtree(obj_dir, ignore_errors=True)
    obj_dir.mkdir(parents=True, exist_ok=True)

    # Compile zig_dso_lifetime.c for safe __cxa_finalize on dlclose
    dso_o = obj_dir / "zig_dso_lifetime.o"
    subprocess.run(
        [str(zig), "cc", "-target", "aarch64-linux-gnu.2.39", "-O2", "-fPIC", "-fdebug-compilation-dir=.",
         "-c", str(eng / "wsi-dma-capture/zig_dso_lifetime.c"), "-o", str(dso_o)],
        check=True,
    )

    cpps = sorted(p for p in eng.rglob("*.cpp"))
    common_flags = [
        str(zig), "c++", "-target", "aarch64-linux-gnu.2.39",
        "-std=gnu++20", "-O3", "-DNDEBUG", "-fPIC", "-Wno-nullability-completeness",
        "-D_LIBCPP_ENABLE_EXPERIMENTAL", "-fdebug-compilation-dir=.",
        f"-I{eng / 'wsi-dma-capture'}",
        f"-I{eng / 'mako-common/include'}",
        "-isystem", str(eng / "mako-common/thirdparty/include"),
        "-isystem", str(vk_inc),
        "-isystem", str(sysroot_inc),
    ]

    tasks: list[tuple[Path, list[str]]] = []
    for cpp in cpps:
        rel = cpp.relative_to(eng).as_posix()
        obj = obj_dir / (rel.replace("/", "__") + ".o")
        cmd = list(common_flags)
        if rel.startswith("mako-backend/"):
            cmd += [
                "-DMAKO_FG=1", "-fvisibility=hidden",
                f"-I{eng / 'mako-backend/include'}",
                f"-I{eng / 'mako-backend/src'}",
            ]
        elif rel in (
            "mako-render/src/adaptive_scheduler.cpp",
            "mako-render/src/color_pipeline.cpp",
            "mako-render/src/spatial_scaler.cpp",
        ):
            cmd += [
                "-fvisibility=hidden", "-fvisibility-inlines-hidden",
                f"-I{eng / 'mako-render/src'}",
                f"-I{eng / 'mako-backend/include'}",
            ]
        elif rel.startswith("mako-render/"):
            cmd += [
                '-DMAKO_BUILD_FINGERPRINT="1.0.0"',
                '-DMAKO_BUILD_VERSION="1.0.0"',
                "-Dmako_render_EXPORTS",
                "-fvisibility=hidden",
                "-Wno-unknown-warning-option",
                "-Wno-unsafe-buffer-usage",
                f"-I{eng / 'mako-render/src'}",
                f"-I{eng / 'mako-backend/include'}",
            ]
        cmd += ["-c", str(cpp), "-o", str(obj)]
        tasks.append((obj, cmd))

    print(f"[1/6] Compiling libmako-render.so ({len(tasks)} translation units)...", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(lambda t: subprocess.run(t[1], check=True), tasks))

    out_so = out_dir / "libmako-render.so"
    link_cmd = [
        str(zig), "c++", "-target", "aarch64-linux-gnu.2.39",
        "-shared", "-fPIC", "-O3", "-DNDEBUG", "-s", "-fdebug-compilation-dir=.",
        "-Wl,-soname,libmako-render.so",
        str(dso_o),
        *(str(t[0]) for t in tasks),
        "-ldl",
        "-o", str(out_so),
    ]
    subprocess.run(link_cmd, check=True)
    print(f"      -> {out_so.name} ({out_so.stat().st_size / 1024:.1f} KB)", flush=True)
    return out_so


def compile_wsi_capture(zig: Path, sysroot_inc: Path, out_dir: Path) -> Path:
    print("[2/6] Compiling libodin3-wsi-dma-capture.so...", flush=True)
    eng = REPO / "engine/wsi-dma-capture"
    out_so = out_dir / "libodin3-wsi-dma-capture.so"
    subprocess.run(
        [
            str(zig), "cc", "-target", "aarch64-linux-gnu.2.34",
            "-shared", "-fPIC", "-O2", "-s", "-fdebug-compilation-dir=.",
            f"-I{eng}", "-isystem", str(sysroot_inc),
            str(eng / "odin3_wsi_dma_capture.c"),
            "-ldl", "-lpthread",
            "-o", str(out_so),
        ],
        check=True,
    )
    print(f"      -> {out_so.name} ({out_so.stat().st_size / 1024:.1f} KB)", flush=True)
    return out_so


def compile_worker(zig: Path, vk_inc: Path, sysroot_inc: Path, qnn_inc: Path, vk_stub: Path, out_dir: Path) -> Path:
    print("[3/6] Compiling odin3-fg-worker...", flush=True)
    out_bin = out_dir / "odin3-fg-worker"
    subprocess.run(
        [
            str(zig), "c++", "-target", "aarch64-linux-gnu.2.39",
            "-std=c++20", "-O2", "-s", "-fdebug-compilation-dir=.",
            "-Wall", "-Wextra", "-Wno-missing-field-initializers",
            f"-I{REPO / 'worker'}",
            f"-I{REPO / 'engine/wsi-dma-capture'}",
            f"-I{REPO / 'engine/mako-backend/src'}",
            f"-I{REPO / 'engine/mako-common/include'}",
            "-isystem", str(vk_inc),
            "-isystem", str(sysroot_inc),
            "-isystem", str(qnn_inc),
            str(REPO / "worker/odin3_fg_worker.cpp"),
            str(vk_stub), "-ldl", "-lm", "-lpthread",
            "-o", str(out_bin),
        ],
        check=True,
    )
    print(f"      -> {out_bin.name} ({out_bin.stat().st_size / 1024:.1f} KB)", flush=True)
    return out_bin


def compile_governor_shim(zig: Path, sysroot_inc: Path, qnn_inc: Path, out_dir: Path) -> Path:
    print("[4/6] Compiling libodin3-npu-governor.so...", flush=True)
    gov_dir = REPO / "npu-governor"
    out_so = out_dir / "libodin3-npu-governor.so"
    subprocess.run(
        [
            str(zig), "c++", "-target", "aarch64-linux-gnu.2.34",
            "-shared", "-fPIC", "-O2", "-std=c++20", "-s", "-fdebug-compilation-dir=.",
            "-Wall", "-Wextra", "-Wno-unused-parameter", "-Wno-missing-field-initializers",
            "-fvisibility=hidden",
            "-isystem", str(sysroot_inc),
            "-isystem", str(qnn_inc),
            str(gov_dir / "npu_governor_shim.cpp"),
            "-Wl,-soname,libQnnHtp.so",
            f"-Wl,--version-script={gov_dir / 'exports.map'}",
            "-ldl", "-lpthread",
            "-o", str(out_so),
        ],
        check=True,
    )
    print(f"      -> {out_so.name} ({out_so.stat().st_size / 1024:.1f} KB)", flush=True)
    return out_so


def package_runtime_archive(renderer_so: Path, wsi_so: Path, worker_bin: Path, fastrpc_src: Path, out_dir: Path) -> Path:
    print("[5/6] Packaging odin3-fg-runtime.tar...", flush=True)
    out_tar = out_dir / "odin3-fg-runtime.tar"
    out_tar.unlink(missing_ok=True)

    with TemporaryDirectory(prefix="odin3-rt-") as tmp:
        st = Path(tmp) / "rt"
        (st / "lib").mkdir(parents=True)
        (st / "fastrpc/bin").mkdir(parents=True)
        (st / "fastrpc/lib").mkdir(parents=True)
        (st / "fastrpc/licenses").mkdir(parents=True)
        (st / "share/vulkan/implicit_layer.d").mkdir(parents=True)

        shutil.copy2(renderer_so, st / "lib/libmako-render.so")
        shutil.copy2(wsi_so, st / WSI_CAPTURE)
        shutil.copy2(worker_bin, st / "odin3-fg-worker")
        shutil.copy2(REPO / "runtime/odin3-fg.sh", st / "odin3-fg.sh")
        shutil.copy2(REPO / "runtime/profiles.py", st / "profiles.py")
        shutil.copy2(REPO / "runtime/odin3-fg-cdsp.service", st / "fastrpc/odin3-fg-cdsp.service")
        shutil.copy2(REPO / "runtime/VkLayer_MAKO_render.json", st / LAYER_MANIFEST)
        shutil.copy2(REPO / "licenses/FastRPC.txt", st / "fastrpc/licenses/FastRPC.txt")
        shutil.copy2(REPO / "licenses/LibYAML.txt", st / "fastrpc/licenses/LibYAML.txt")

        if fastrpc_src.is_file() and fastrpc_src.suffix == ".tar":
            with tarfile.open(fastrpc_src, "r:") as tf:
                for member_name in ("fastrpc/bin/cdsprpcd", "fastrpc/lib/libcdsprpc.so", "fastrpc/lib/libcdsp_default_listener.so.1"):
                    (st / member_name).write_bytes(tf.extractfile(member_name).read())
        else:
            for rel in ("bin/cdsprpcd", "lib/libcdsprpc.so", "lib/libcdsp_default_listener.so.1"):
                shutil.copy2(fastrpc_src / rel, st / "fastrpc" / rel)

        sources_meta = json.loads((REPO / "sources.json").read_text(encoding="utf-8"))
        files = sorted(p.relative_to(st).as_posix() for p in st.rglob("*") if p.is_file())
        receipt = {
            "schema_version": 1,
            "architecture": "aarch64",
            "pinned_dependencies": sources_meta,
            "shared_input": {
                "protocol": SHARED_INPUT_PROTOCOL,
                "source": {"slots": 12, "zero_copy_wsi": True},
            },
            "files": {
                name: {"sha256": file_sha256(st / name), "size": (st / name).stat().st_size}
                for name in files
            },
        }
        (st / MANIFEST).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

        all_files = sorted(p.relative_to(st).as_posix() for p in st.rglob("*") if p.is_file())
        with tarfile.open(out_tar, "x", format=tarfile.USTAR_FORMAT) as archive:
            for name in all_files:
                fp = st / name
                ti = tarfile.TarInfo(name)
                ti.mode = 0o755 if name in EXECUTABLES else 0o644
                ti.size = fp.stat().st_size
                with fp.open("rb") as f:
                    archive.addfile(ti, f)

    verify_archive(out_tar)
    digest = file_sha256(out_tar)
    (out_dir / "odin3-fg-runtime.tar.sha256").write_text(f"{digest}  odin3-fg-runtime.tar\n", encoding="utf-8", newline="\n")
    print(f"      -> {out_tar.name} ({out_tar.stat().st_size / (1024 * 1024):.2f} MB, sha256={digest[:16]}...)", flush=True)
    return out_tar


def build_decky_zip(src_dir: Path, zip_root: str, out_zip: Path, nm_cache: Path | None, extra_assets: dict[str, Path] | None = None) -> Path:
    node = find_node()
    env = {**os.environ, "PATH": f"{node.parent}{os.pathsep}{os.environ.get('PATH', '')}"}

    with TemporaryDirectory(prefix=f"decky-{zip_root}-") as tmp:
        work = Path(tmp) / "work"
        shutil.copytree(src_dir, work)
        nm = work / "node_modules"
        used_junction = False
        if nm_cache and nm_cache.exists():
            subprocess.run(["cmd", "/c", "mklink", "/J", str(nm), str(nm_cache)], check=True, stdout=subprocess.DEVNULL)
            used_junction = True
        else:
            npm = node.parent / ("npm.cmd" if os.name == "nt" else "npm")
            subprocess.run([str(npm), "install", "--no-audit", "--no-fund"], cwd=work, env=env, check=True)

        try:
            subprocess.run([str(node), str(nm / "typescript/bin/tsc"), "--noEmit"], cwd=work, env=env, check=True)
            subprocess.run([str(node), str(nm / "rollup/dist/bin/rollup"), "-c"], cwd=work, env=env, check=True)
        finally:
            if used_junction and nm.exists():
                os.rmdir(nm)

        pkg = Path(tmp) / "pkg"
        (pkg / "dist").mkdir(parents=True)
        shutil.copy2(REPO / "LICENSE", pkg / "LICENSE")
        shutil.copy2(work / "plugin.json", pkg / "plugin.json")
        shutil.copy2(work / "package.json", pkg / "package.json")
        shutil.copy2(work / "dist/index.js", pkg / "dist/index.js")
        shutil.copy2(work / "main.py", pkg / "main.py")
        if (work / "profiles.py").is_file():
            shutil.copy2(work / "profiles.py", pkg / "profiles.py")
        if (work / "py_modules").is_dir():
            shutil.copytree(work / "py_modules", pkg / "py_modules")
        for rel_dst, asset_path in (extra_assets or {}).items():
            dst = pkg / rel_dst
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(asset_path, dst)

        out_zip.unlink(missing_ok=True)
        with ZipFile(out_zip, "x") as zf:
            for fp in sorted(pkg.rglob("*")):
                if not fp.is_file():
                    continue
                rel = fp.relative_to(pkg).as_posix()
                zi = ZipInfo(f"{zip_root}/{rel}")
                zi.create_system = 3
                is_exec = rel.endswith((".so", "odin3_fg_launch.py"))
                zi.external_attr = (0o100755 if is_exec else 0o100644) << 16
                zf.writestr(zi, fp.read_bytes(), compress_type=ZIP_DEFLATED)
    return out_zip


def package_models_bundle(models_dir: Path, out_dir: Path) -> Path:
    out_tgz = out_dir / "omnishift-sm8750-models.tar.gz"
    out_tgz.unlink(missing_ok=True)
    scales = ("360p", "540p", "720p", "900p", "1080p")
    manifest_scales = {}
    with tarfile.open(out_tgz, "w:gz") as tf:
        for sc in scales:
            sdir = models_dir / sc
            if not sdir.is_dir():
                raise FileNotFoundError(f"Missing scale directory: {sdir}")
            manifest_scales[sc] = {}
            for fn in sorted(p.name for p in sdir.glob("*.bin")):
                fp = sdir / fn
                manifest_scales[sc][fn] = {"size": fp.stat().st_size, "sha256": file_sha256(fp)}
                ti = tarfile.TarInfo(f"omnishift-sm8750/{sc}/{fn}")
                ti.mode = 0o644
                ti.size = fp.stat().st_size
                with fp.open("rb") as f:
                    tf.addfile(ti, f)
        clean_manifest = json.dumps(
            {
                "model": "OmniShift-HTP-v79-SM8750",
                "soc": "SM8750 (Snapdragon 8 Elite, Hexagon HTP v79)",
                "quantization": "W8A8 (UFIXED_POINT_8 activations, INT8 weights, INT32 bias)",
                "resolutions": list(scales),
                "files": manifest_scales,
            },
            indent=2,
        ).encode("utf-8") + b"\n"
        ti = tarfile.TarInfo("omnishift-sm8750/manifest.json")
        ti.mode = 0o644
        ti.size = len(clean_manifest)
        import io
        tf.addfile(ti, io.BytesIO(clean_manifest))
    print(f"      -> {out_tgz.name} ({out_tgz.stat().st_size / (1024 * 1024):.2f} MB)", flush=True)
    return out_tgz


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=REPO / "dist-release")
    ap.add_argument("--zig", type=Path, default=None)
    ap.add_argument("--vulkan-include", type=Path, default=None)
    ap.add_argument("--linux-sysroot", type=Path, default=None)
    ap.add_argument("--qairt-include", type=Path, default=None)
    ap.add_argument("--vulkan-stub-so", type=Path, default=None)
    ap.add_argument("--fastrpc-dir", type=Path, default=None)
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    zig = resolve_path(args.zig, "ODIN3_ZIG", [REPO / "build/warp-toolchain/zig-x86_64-windows-0.16.0/zig.exe"], "Zig compiler")
    vk_inc = resolve_path(args.vulkan_include, "ODIN3_VULKAN_INCLUDE", [REPO / "local/sources/Vulkan-Headers/include"], "Vulkan-Headers/include")
    sysroot_inc = resolve_path(
        args.linux_sysroot,
        "ODIN3_LINUX_SYSROOT",
        [REPO / "local/sysroot/usr/include"],
        "Linux/X11/XCB usr/include sysroot",
    )
    qnn_inc = resolve_path(
        args.qairt_include,
        "ODIN3_QAIRT_INCLUDE",
        [REPO / "local/qairt/include/QNN"],
        "Qualcomm QAIRT include/QNN headers",
    )
    vk_stub = resolve_path(
        args.vulkan_stub_so,
        "ODIN3_VULKAN_STUB_SO",
        [REPO / "local/sysroot/usr/lib/libvulkan.so.1"],
        "aarch64 libvulkan.so.1 link stub",
    )
    fastrpc_src = resolve_path(
        args.fastrpc_dir,
        "ODIN3_FASTRPC_DIR",
        [REPO / "local/fastrpc"],
        "FastRPC binary directory or base archive",
    )

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_bin = out_dir / ".bin-work"
    tmp_bin.mkdir(parents=True, exist_ok=True)

    renderer_so = compile_renderer(zig, vk_inc, sysroot_inc, tmp_bin, jobs=args.jobs)
    wsi_so = compile_wsi_capture(zig, sysroot_inc, tmp_bin)
    worker_bin = compile_worker(zig, vk_inc, sysroot_inc, qnn_inc, vk_stub, tmp_bin)
    gov_so = compile_governor_shim(zig, sysroot_inc, qnn_inc, tmp_bin)

    rt_tar = package_runtime_archive(renderer_so, wsi_so, worker_bin, fastrpc_src, out_dir)

    print("[6/6] Building Decky plugins...", flush=True)
    nm_cache = REPO / "build/decky-cache/node_modules"
    fg_zip = build_decky_zip(REPO / "decky-fg", "odin3-fg-npu", out_dir / "odin3-fg-npu-decky.zip", nm_cache if nm_cache.exists() else None)
    print(f"      -> {fg_zip.name} ({fg_zip.stat().st_size / 1024:.1f} KB)", flush=True)
    pwr_zip = build_decky_zip(
        REPO / "decky-npu-power",
        "odin3-npu-power",
        out_dir / "odin3-npu-power-decky.zip",
        nm_cache if nm_cache.exists() else None,
        extra_assets={"assets/libodin3-npu-governor.so": gov_so},
    )
    print(f"      -> {pwr_zip.name} ({pwr_zip.stat().st_size / 1024:.1f} KB)", flush=True)

    release_files = [rt_tar, fg_zip, pwr_zip]
    if args.models_dir and args.models_dir.is_dir():
        release_files.append(package_models_bundle(args.models_dir, out_dir))

    shutil.rmtree(tmp_bin, ignore_errors=True)
    sums = [f"{file_sha256(p)}  {p.name}" for p in release_files]
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8", newline="\n")
    print("All release artifacts built successfully in:", out_dir, flush=True)


if __name__ == "__main__":
    main()
