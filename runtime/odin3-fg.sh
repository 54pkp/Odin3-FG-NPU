#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "$0")" && pwd)
if [[ ${1:-} = -- ]]; then shift; fi
if [[ $# = 0 ]]; then
    printf 'Usage: %s -- GAME_COMMAND [ARGUMENTS...]\n' "$0" >&2
    exit 2
fi
test "$(uname -m)" = aarch64
test -x "$root/mako-anvil-worker"
test -r "$root/lib/libmako-render.so"
test -r "$root/shaders/kfg_stats.spv"
ss -xln | grep -Fw '@fsr4npud_anvil_s_playtest' >/dev/null

manifest="$root/share/vulkan/implicit_layer.d"
export VK_ADD_IMPLICIT_LAYER_PATH="$manifest${VK_ADD_IMPLICIT_LAYER_PATH:+:$VK_ADD_IMPLICIT_LAYER_PATH}"
export ENABLE_MAKO=1 MAKO_ENV=1 MAKO_FRAME_GENERATION_PROVISIONED=1 MAKO_FRAME_GENERATION_ENABLED=1
export MAKO_FRAME_GENERATION_MODEL=mako-fg MAKO_MULTIPLIER=2 MAKO_ADAPTIVE=0
export DISABLE_GAMESCOPE_WSI=1 ENABLE_GAMESCOPE_WSI=0 MAKO_DISABLE_HDR_EXPOSURE=1 DXVK_HDR=0
export MAKO_SCALING_ENABLED=1 MAKO_SCALING_METHOD=native MAKO_SCALING_FACTOR=1
export DISABLE_LSFGVK=1 DISABLE_LSFG=1 VELA_LAUNCH_DONE=1 DISABLE_VELA_NPU=1
unset VELA_NPU MAKO_ANVIL_PAIR_PACING
export MAKO_ANVIL_DEADLINE=1 MAKO_ANVIL_RENDER_BOUNDARY=1 MAKO_ANVIL_READY_WINDOW=1
export MAKO_ANVIL_ASYNC_SCHEDULE=1 MAKO_ANVIL_COVERAGE_WINDOW=1 MAKO_ANVIL_MOTION=1
export MAKO_FG_HOST="$root/mako-anvil-worker" MAKO_FAULT_WORKER_MODE=anvil MAKO_ANVIL_SPV_DIR="$root/shaders"
exec "$@"
