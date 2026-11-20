"""Assemble a relocatable ARM64 runtime directory from the cross-build outputs."""
import argparse
import json
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

from build_inputs import BUILD, ROOT
from build_kfg_anvil_probe import SHADERS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BUILD / "runtime")
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    renderer = BUILD / "mako/mako-render"
    manifest = json.loads((renderer / "VkLayer_MAKO_render.json").read_text(encoding="utf-8"))
    if manifest["layer"]["name"] != "VK_LAYER_MAKO_render":
        raise ValueError("Unexpected Vulkan layer manifest")
    manifest["layer"]["library_path"] = "../../../lib/libmako-render.so"
    with TemporaryDirectory(dir=target.parent, prefix="odin3-stage-") as directory:
        output = Path(directory) / "runtime"
        for relative in ("lib", "shaders", "share/vulkan/implicit_layer.d"):
            (output / relative).mkdir(parents=True, exist_ok=True)
        (output / "share/vulkan/implicit_layer.d/VkLayer_MAKO_render.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        copy2(renderer / "libmako-render.so", output / "lib")
        copy2(BUILD / "worker/mako-anvil-worker", output)
        copy2(BUILD / "daemon/anvil-npud", output)
        for name in SHADERS:
            copy2(BUILD / "motion" / f"{name}.spv", output / "shaders")
        for name in ("start-htp.sh", "odin3-fg.sh"):
            copy2(ROOT / "runtime" / name, output)
        output.rename(target)
    print(target)


if __name__ == "__main__":
    main()
