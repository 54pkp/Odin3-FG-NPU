"""Cross-compile the private MAKO worker with real ANVIL-S HTP support."""
from hashlib import sha256
import struct
import subprocess

from build_inputs import BUILD, GLSL, ROOT, SOURCES, VULKAN_HEADERS, VULKAN_LOADER, ZIG

CLIENT = SOURCES / "vela/npu/src"
TARGET = "aarch64-linux-gnu.2.39"
KFG_ENGINE = SOURCES / "GSF_Kedavra/kedavra/engine"


def run(*args):
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True)


def borrowed_kfg(build):
    source = (KFG_ENGINE / "kfg_vk.c").read_text(encoding="utf-8")
    init = source.index("int kfg_vk_init(")
    start = source.index("\tVkCommandPoolCreateInfo pci", init)
    end = source.index("\n}\n\nvoid kfg_vk_destroy(", start)
    shared = source[start:end]
    source = ("#include <stdint.h>\n" + source[:init] +
              "static int init_resources(kfg_vk *vk, const char *spv_dir)\n{\n" + shared + "\n}\n\n" +
              source[init:start] + "\treturn init_resources(vk, spv_dir);" + source[end:])
    source = source.replace("void kfg_vk_destroy(kfg_vk *vk)", """int kfg_vk_borrow(kfg_vk *vk, VkPhysicalDevice physical, VkDevice device, VkQueue queue,
                  uint32_t family, const char *spv_dir)
{
    memset(vk, 0, sizeof(*vk));
    vk->phys = physical;
    vk->device = device;
    vk->queue = queue;
    vk->queue_family = family;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical, &props);
    vk->ubo_align = props.limits.minUniformBufferOffsetAlignment;
    if (vk->ubo_align < 16)
        vk->ubo_align = 16;
    return init_resources(vk, spv_dir);
}

void kfg_vk_destroy(kfg_vk *vk)""")
    source = source.replace("\tvkDeviceWaitIdle(vk->device);", "\tif (vk->instance) vkDeviceWaitIdle(vk->device);", 1)
    source = source.replace("\tvkDestroyDevice(vk->device, NULL);\n\tvkDestroyInstance(vk->instance, NULL);",
                            "\tif (vk->instance) {\n\t\tvkDestroyDevice(vk->device, NULL);\n"
                            "\t\tvkDestroyInstance(vk->instance, NULL);\n\t}")
    path = build / "kfg_vk_borrowed.c"
    path.write_text(source, encoding="utf-8")
    return path


def main():
    build = BUILD / "worker"
    build.mkdir(parents=True, exist_ok=True)
    shaders = ("pack", "resolve", "resolve_wide")
    options = (("-DPACK=1",), (), ("-DWIDE=1",))
    code = ["#pragma once"]
    for name, defines in zip(shaders, options, strict=True):
        binary = build / f"{name}.spv"
        run(GLSL / "glslangValidator.exe", "-V", "--target-env", "vulkan1.1", "-Os",
            *defines, ROOT / "native/mako_anvil.comp", "-o", binary)
        run(GLSL / "spirv-val.exe", "--target-env", "vulkan1.1", binary)
        raw = binary.read_bytes()
        words = struct.unpack(f"<{len(raw) // 4}I", raw)
        code.append(f"constexpr uint32_t mako_anvil_{name}[] = {{")
        for i in range(0, len(words), 8):
            code.append("    " + ", ".join(f"0x{word:08x}u" for word in words[i:i + 8]) + ",")
        code.append("};")
    (build / "mako_anvil_spv.hpp").write_text("\n".join(code) + "\n", encoding="ascii")

    client = build / "fsr4npu.o"
    run(ZIG, "cc", "-target", TARGET, "-std=gnu17", "-O2", "-c", CLIENT / "fsr4npu.c", "-o", client)
    kfg_objects = []
    for code in (borrowed_kfg(build), ROOT / "native/mako_kfg_motion.c"):
        obj = build / f"{code.stem}.o"
        run(ZIG, "cc", "-target", TARGET, "-std=gnu17", "-O2", "-Wall", "-Wextra",
            "-Wno-missing-field-initializers", "-I", VULKAN_HEADERS,
            "-I", KFG_ENGINE, "-c", code, "-o", obj)
        kfg_objects.append(obj)
    source = SOURCES / "MAKO/engine"
    output = build / "mako-anvil-worker"
    run(ZIG, "c++", "-target", TARGET, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
        "-Wno-missing-field-initializers", "-DMAKO_ANVIL_MODEL=1",
        "-I", source / "mako-backend/src", "-I", source / "mako-common/include",
        "-I", VULKAN_HEADERS, "-I", CLIENT, "-I", build,
        ROOT / "native/mako_fault_worker.cpp", client, *kfg_objects,
        VULKAN_LOADER, "-lm", "-o", output)
    print(sha256(output.read_bytes()).hexdigest(), output)


if __name__ == "__main__":
    main()
