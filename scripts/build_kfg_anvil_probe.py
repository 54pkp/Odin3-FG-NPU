"""Build a device replay probe using Kedavra's motion analysis and the ANVIL-S daemon."""
import argparse
from hashlib import sha256
from pathlib import Path
import subprocess

from build_inputs import BUILD, GLSL, ROOT, SOURCES, VULKAN_HEADERS, VULKAN_LOADER, ZIG
from build_mako_anvil_worker import CLIENT, TARGET

UPSTREAM = SOURCES / "GSF_Kedavra/kedavra"
OUTPUT = BUILD / "motion"
SHADERS = ("kfg_reduce", "kfg_me", "kfg_stats", "kfg_select", "kfg_warp",
           "kfg_post", "kfg_smaa_edge", "kfg_smaa_weight", "kfg_smaa_blend")


def run(*args):
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True)


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"The pinned Kedavra warp source changed at {before!r}")
    return text.replace(before, after)


def endpoint_shader(compose_transitions=False):
    text = (UPSTREAM / "shaders/kfg_warp.comp").read_text(encoding="utf-8")
    text = replace_once(text, "const float CAMERA_MARGIN = 0.12;", """const float CAMERA_MARGIN = 0.12;
// The probe stores both aligned model inputs in the two halves of dst.
const int ANVIL_INPUT_PAIR = 2048;""")
    if compose_transitions:
        text = replace_once(text, "\t\t// Scene cut: never blend two unrelated images.", """        if ((u_flags & ANVIL_INPUT_PAIR) != 0 && texelFetch(s_samplers[3], ivec2(0, 0), 0).r > 1.5) {
            // A composited overlay is an appearance change at fixed screen coordinates.
            // Keep the old content until frame B. Blend only mutually visible,
            // photometrically stable pixels and never apply a motion residual here.
            vec3 a = texelFetch(s_samplers[0], p, 0).rgb;
            vec3 b = texelFetch(s_samplers[1], p, 0).rgb;
            vec3 delta = abs(a - b);
            float disagreement = max(delta.r, max(delta.g, delta.b));
            float agreement = 1.0 - smoothstep(2.0 / 255.0, 8.0 / 255.0, disagreement);
            imageStore(dst, p, vec4(a, 0.0));
            imageStore(dst, p + ivec2(u_fullSize.x, 0), vec4(mix(a, b, agreement), 0.0));
            return;
        }
        // Scene cut: never blend two unrelated images.""")
    text = replace_once(text, "imageStore(dst, p, vec4(c.rgb, 1.0));", """if ((u_flags & ANVIL_INPUT_PAIR) != 0) {
            imageStore(dst, p, vec4(c.rgb, 0.0));
            imageStore(dst, p + ivec2(u_fullSize.x, 0), vec4(c.rgb, 0.0));
        } else {
            imageStore(dst, p, vec4(c.rgb, 1.0));
        }""")
    text = replace_once(text, "\tvec3 c = mix(ca.rgb, cb.rgb, w_b);", """    if ((u_flags & ANVIL_INPUT_PAIR) != 0) {
        // Their mean equals Kedavra's visibility-weighted midpoint. At an
        // occlusion both endpoints use the visible side when possible.
        float residual = (1.0 - smoothstep(0.10, 0.25, sel.w)) * clamp(4.0 * w_b * (1.0 - w_b), 0.0, 1.0);
        vec3 a = mix(ca.rgb, cb.rgb, max(0.0, 2.0 * w_b - 1.0));
        vec3 b = mix(ca.rgb, cb.rgb, min(1.0, 2.0 * w_b));
        imageStore(dst, p, vec4(a, residual));
        imageStore(dst, p + ivec2(u_fullSize.x, 0), vec4(b, residual));
        return;
    }
    vec3 c = mix(ca.rgb, cb.rgb, w_b);""")
    return text


def coverage_stats_shader():
    text = (UPSTREAM / "shaders/kfg_stats.comp").read_text(encoding="utf-8")
    return replace_once(text, "float cut = (frac_bad > u_cutThreshold || unexplained) ? 1.0 : 0.0;",
                        """// Large camera turns can leave a small remainder even when alignment
        // explains most of the change. Retain the absolute bad-block guard.
        bool localRemainder = frac_bad <= 0.10 && mean_mc <= 0.035 && mean_mc <= 0.55 * mean_zero;
        float cut = (frac_bad > u_cutThreshold || (unexplained && !localRemainder)) ? 1.0 : 0.0;""")


def transition_stats_shader(compose_transitions=False):
    text = coverage_stats_shader()
    text = replace_once(text, "\tfloat bad = 0.0, mc = 0.0, zero = 0.0, n = 0.0;", """    float bad = 0.0, mc = 0.0, zero = 0.0, n = 0.0;
    uvec2 strongChange = uvec2(0u);""")
    text = replace_once(text, "\t\tfloat e_zero = abs(a - texelFetch(s_samplers[1], q, 0).r);", """        float directChange = texelFetch(s_samplers[1], q, 0).r - a;
        float e_zero = abs(directChange);
        strongChange += uvec2(directChange >= 0.10 ? 1u : 0u, directChange <= -0.10 ? 1u : 0u);""")
    text = replace_once(text, "shared float s_zero[256];", """shared float s_zero[256];
shared uvec2 s_strongChange[256];""")
    text = replace_once(text, "\ts_zero[lid] = zero;", "\ts_zero[lid] = zero;\n    s_strongChange[lid] = strongChange;")
    text = replace_once(text, "\t\t\ts_zero[lid] += s_zero[lid + s];",
                        "\t\t\ts_zero[lid] += s_zero[lid + s];\n            s_strongChange[lid] += s_strongChange[lid + s];")
    mode = "2.0" if compose_transitions else "1.0"
    return replace_once(text, "float cut = (frac_bad > u_cutThreshold || (unexplained && !localRemainder)) ? 1.0 : 0.0;", """// A newly composited overlay or a fade can change the luma of a large
        // area in one direction even when flow finds similar blocks elsewhere.
        uvec2 votes = s_strongChange[0];
        uint changed = votes.x + votes.y;
        bool overlayOrFade = float(changed) >= 0.25 * total &&
                             10u * max(votes.x, votes.y) >= 9u * changed;
        float cut = overlayOrFade ? OVERLAY_MODE :
                    (frac_bad > u_cutThreshold || (unexplained && !localRemainder)) ? 1.0 : 0.0;""".replace("OVERLAY_MODE", mode))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    cuts = parser.add_mutually_exclusive_group()
    cuts.add_argument("--coverage-cut", action="store_true")
    cuts.add_argument("--transition-cut", action="store_true")
    cuts.add_argument("--transition-compose", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    overlay = output / "kfg_warp.comp"
    compose = args.transition_compose or not args.coverage_cut and not args.transition_cut
    overlay.write_text(endpoint_shader(compose), encoding="utf-8")
    stats = UPSTREAM / "shaders/kfg_stats.comp"
    stats = output / "kfg_stats.comp"
    shader = coverage_stats_shader() if args.coverage_cut else transition_stats_shader(compose)
    stats.write_text(shader, encoding="utf-8")
    for name in SHADERS:
        source = {"kfg_warp": overlay, "kfg_stats": stats}.get(name, UPSTREAM / f"shaders/{name}.comp")
        spv = output / f"{name}.spv"
        run(GLSL / "glslangValidator.exe", "-V", "--target-env", "vulkan1.1",
            f"-I{UPSTREAM / 'shaders'}", source, "-o", spv)
        run(GLSL / "spirv-val.exe", "--target-env", "vulkan1.1", spv)
    worker = output / "kfg-anvil-replay"
    run(ZIG, "cc", "-target", TARGET, "-std=gnu17", "-O2", "-Wall", "-Wextra",
        "-Wno-missing-field-initializers", "-I", VULKAN_HEADERS,
        "-I", UPSTREAM / "engine", "-I", CLIENT, ROOT / "native/kfg_anvil_replay.c",
        UPSTREAM / "engine/kfg_vk.c", CLIENT / "fsr4npu.c",
        VULKAN_LOADER, "-lm", "-o", worker)
    print(sha256(worker.read_bytes()).hexdigest(), worker)


if __name__ == "__main__":
    main()
