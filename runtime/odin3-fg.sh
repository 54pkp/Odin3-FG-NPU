#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "$0")" && pwd)
if [[ ${1:-} = -- ]]; then shift; fi
if [[ $# = 0 ]]; then
    printf 'Usage: %s -- GAME_COMMAND [ARGUMENTS...]\n' "$0" >&2
    exit 2
fi
start_original() {
    printf 'Odin 3 FG: %s. Starting the original command.\n' "$1" >&2
    shift
    exec "$@"
}

data_root=${ODIN3_FG_DATA_ROOT:-"${XDG_DATA_HOME:-${HOME:-/var/home/armada}/.local/share}/odin3-fg"}
multiplier=${ODIN3_FG_MULTIPLIER:-}
source_cap=${ODIN3_FG_SOURCE_CAP:--}
flow_requested=25.0
flow_applied=25.0
flow_stride=4
flow_width=320
flow_height=180
profile_refresh_hz=120
profile_auto_disp=1

app_id=${ODIN3_FG_APP_ID:-}
if [[ -z "$app_id" ]]; then
    for candidate in "${STEAM_COMPAT_APP_ID:-}" "${SteamAppId:-}" "${SteamGameId:-}"; do
        if [[ "$candidate" =~ ^[1-9][0-9]{0,9}$ ]]; then
            app_id=$candidate
            break
        elif [[ "$candidate" =~ ^[1-9][0-9]{10,20}$ ]]; then
            app_id=$(( candidate >> 32 ))
            break
        fi
    done
fi
if [[ -z "$app_id" && -n "${STEAM_COMPAT_DATA_PATH:-}" ]]; then
    compat_base=$(basename -- "${STEAM_COMPAT_DATA_PATH%/}")
    if [[ "$compat_base" =~ ^[1-9][0-9]{0,9}$ ]]; then
        app_id=$compat_base
    fi
fi

if [[ -z "$multiplier" ]]; then
    multiplier=0
    if [[ -n "$app_id" ]]; then
        if ! settings=$(python3 -I "$root/profiles.py" get "$app_id" --launch-with-flow); then
            start_original "Could not read the profile for AppID $app_id" "$@"
        fi
        read -r multiplier source_cap flow_requested flow_applied flow_stride flow_width flow_height profile_refresh_hz profile_auto_disp <<< "$settings"
    fi
fi

case "$multiplier" in
    0) exec "$@" ;;
    2|3) ;;
    *) printf 'ODIN3_FG_MULTIPLIER must be 0, 2 or 3.\n' >&2; exit 2 ;;
esac

refresh_hz=${ODIN3_FG_REFRESH_HZ:-$profile_refresh_hz}
case "$refresh_hz" in
    60|120) ;;
    "") refresh_hz=120 ;;
    *) start_original "ODIN3_FG_REFRESH_HZ must be 60 or 120" "$@" ;;
esac

manifest="$root/share/vulkan/implicit_layer.d"
if [[ $(uname -m) != aarch64 ]]; then
    start_original "Independent frame generation requires an ARM64 host" "$@"
fi
if [[ ! -x "$root/odin3-fg-worker" || ! -r "$root/lib/libmako-render.so" ||
      ! -r "$manifest/VkLayer_MAKO_render.json" ]]; then
    start_original "Required runtime files are missing" "$@"
fi

direct_htp=${ODIN3_HTP_DIRECT_HTP:-0}
raw_htp=${ODIN3_HTP_RAW_HTP:-1}
if [[ "$raw_htp" != 1 || "$direct_htp" == 1 ]]; then
    start_original "This runtime requires the full HTP path with directly presentable output" "$@"
fi

unset ODIN3_WSI_DIRECT_INPUT ODIN3_WSI_DMA_PREFER_LINEAR
unset ODIN3_HTP_INDEPENDENT_REPEAT_DIAGNOSTIC ODIN3_HTP_HTP_PRESENT_ORIGINAL_DIAGNOSTIC
unset VELA_NPU
export ODIN3_HTP_RAW_HTP=1
export ODIN3_HTP_DIRECT_HTP=0
export ODIN3_HTP_DIRECT_DRI3=1
export ODIN3_HTP_SHARED_SOURCES=1
export ODIN3_HTP_RAW_LINEAR_INPUT=1
export ODIN3_HTP_RAW_LINEAR_OUTPUT=1
export ODIN3_HTP_RAW_CHANNEL_ORDER=${ODIN3_HTP_RAW_CHANNEL_ORDER:-bgra}
if [[ "$ODIN3_HTP_RAW_CHANNEL_ORDER" != bgra ]]; then
    start_original "The direct-output pure-neural path requires BGRA8 HTP input" "$@"
fi

# Gen-5 smooth zero-drop DRI3, continuous prefetch & dynamic sleep-enabled HTP DCVS defaults
export ODIN3_HTP_PROBE_TOP_POWER=${ODIN3_HTP_PROBE_TOP_POWER:-1}
export ODIN3_HTP_ASYNC_RUNTIME_STATUS=${ODIN3_HTP_ASYNC_RUNTIME_STATUS:-1}
export ODIN3_HTP_KEEP_BYPASSED_HISTORY=${ODIN3_HTP_KEEP_BYPASSED_HISTORY:-1}
export ODIN3_HTP_DRI3_PREFETCH=${ODIN3_HTP_DRI3_PREFETCH:-1}
export ODIN3_HTP_DRI3_CREDIT_GRACE=${ODIN3_HTP_DRI3_CREDIT_GRACE:-1}
export ODIN3_DRI3_RECHECK_CREDIT=${ODIN3_DRI3_RECHECK_CREDIT:-1}
export ODIN3_DRI3_RETRY_CURRENT_PAIR=${ODIN3_DRI3_RETRY_CURRENT_PAIR:-1}
export ODIN3_DRI3_RETRY_PREFETCH=${ODIN3_DRI3_RETRY_PREFETCH:-1}
export ODIN3_DRI3_DRAIN_IDLE=${ODIN3_DRI3_DRAIN_IDLE:-1}
export ODIN3_HTP_INDEPENDENT_FIXED_PACING=${ODIN3_HTP_INDEPENDENT_FIXED_PACING:-1}
export ODIN3_HTP_INDEPENDENT_PRESENT_HIGH_PRIORITY=${ODIN3_HTP_INDEPENDENT_PRESENT_HIGH_PRIORITY:-0}
export ODIN3_DRI3_DRAIN_SOURCE_WAIT=${ODIN3_DRI3_DRAIN_SOURCE_WAIT:-1}
export ODIN3_FG_APP_LOW_PRIORITY=${ODIN3_FG_APP_LOW_PRIORITY:-0}
export ODIN3_HTP_RAW_SIGNAL_ON_HOST=${ODIN3_HTP_RAW_SIGNAL_ON_HOST:-1}
export ODIN3_HTP_RAW_FLOW_STRIDE=4
export ODIN3_HTP_RAW_FLOW_REQUEST=25.0

# Context resolution: explicit env vars OR automatic multi-res bundle (720p / 900p / 1080p)
raw_context_dir=${ODIN3_FG_RAW_CONTEXT_DIR:-}
joined_context=${ODIN3_HTP_RAW_JOINED_CONTEXT:-}
first_context=${ODIN3_HTP_RAW_FIRST_CONTEXT:-}
second_context=${ODIN3_HTP_RAW_SECOND_CONTEXT:-}
pure_bundle_dir=${ODIN3_FG_MODEL_BUNDLE_DIR:-}

if [[ -z "$raw_context_dir" && -z "$joined_context" && -z "$first_context" ]]; then
    if [[ -z "$pure_bundle_dir" ]]; then
        if [[ -d "$data_root/models/omnishift-sm8750" ]]; then
            pure_bundle_dir="$data_root/models/omnishift-sm8750"
        else
            pure_bundle_dir="$data_root/models/gen4-pure-sm8750"
        fi
    fi
fi

if [[ -n "$pure_bundle_dir" && -z "$joined_context" && -z "$first_context" && -z "$raw_context_dir" ]]; then
    if [[ "$pure_bundle_dir" != /* || ! -d "$pure_bundle_dir" ]]; then
        start_original "ODIN3_FG_MODEL_BUNDLE_DIR ($pure_bundle_dir) is missing" "$@"
    fi
    for res_sub in 360p 540p 720p 900p 1080p; do
        if [[ ! -r "$pure_bundle_dir/$res_sub/vfi-motion3x.bin" || ! -r "$pure_bundle_dir/$res_sub/phase12.bin" ]]; then
            start_original "Missing 2x/motion contexts in $pure_bundle_dir/$res_sub" "$@"
        fi
        if [[ "$multiplier" == 3 && ( ! -r "$pure_bundle_dir/$res_sub/phase13.bin" || ! -r "$pure_bundle_dir/$res_sub/phase23.bin" ) ]]; then
            start_original "Missing 3x phase contexts in $pure_bundle_dir/$res_sub" "$@"
        fi
    done
    export ODIN3_FG_MODEL_BUNDLE_DIR="$pure_bundle_dir"
    unset ODIN3_HTP_RAW_JOINED_CONTEXT ODIN3_HTP_RAW_FIRST_CONTEXT ODIN3_HTP_RAW_SECOND_CONTEXT ODIN3_HTP_RAW_MOTION_CONTEXT
else
    unset ODIN3_FG_MODEL_BUNDLE_DIR
    context_vars=()
    if [[ "$multiplier" == 3 ]]; then
        unset ODIN3_HTP_RAW_JOINED_CONTEXT
        context_vars+=(ODIN3_HTP_RAW_FIRST_CONTEXT ODIN3_HTP_RAW_SECOND_CONTEXT)
        if [[ -n "${ODIN3_HTP_RAW_MOTION_CONTEXT:-}" && "${ODIN3_HTP_RAW_MOTION_CONTEXT:-}" != "-" ]]; then
            context_vars+=(ODIN3_HTP_RAW_MOTION_CONTEXT)
        else
            unset ODIN3_HTP_RAW_MOTION_CONTEXT
        fi
    elif [[ -n "$joined_context" ]]; then
        context_vars+=(ODIN3_HTP_RAW_JOINED_CONTEXT)
    elif [[ -n "$raw_context_dir" ]]; then
        export ODIN3_HTP_RAW_MOTION_CONTEXT="$raw_context_dir/fg4_motion.SM8750.bin"
        export ODIN3_HTP_RAW_FIRST_CONTEXT="$raw_context_dir/fg4_phase_1_2.SM8750.bin"
        context_vars+=(ODIN3_HTP_RAW_MOTION_CONTEXT ODIN3_HTP_RAW_FIRST_CONTEXT)
    else
        context_vars+=(ODIN3_HTP_RAW_MOTION_CONTEXT ODIN3_HTP_RAW_FIRST_CONTEXT)
    fi
    for context_var in "${context_vars[@]}"; do
        context_path=${!context_var:-}
        if [[ "$context_path" != /* || ! -f "$context_path" || ! -r "$context_path" ]]; then
            start_original "$context_var must point to a readable context file using an absolute path" "$@"
        fi
        export "$context_var"
    done
fi

ODIN3_QAIRT_LIB_DIR=${ODIN3_QAIRT_LIB_DIR:-"$data_root/qairt/lib"}
if [[ "$ODIN3_QAIRT_LIB_DIR" != /* || "$ODIN3_QAIRT_LIB_DIR" == *:* || ! -d "$ODIN3_QAIRT_LIB_DIR" ||
      ! -r "$ODIN3_QAIRT_LIB_DIR/libQnnHtp.so" || ! -r "$ODIN3_QAIRT_LIB_DIR/libQnnSystem.so" ]]; then
    start_original "ODIN3_QAIRT_LIB_DIR must contain readable libQnnHtp.so and libQnnSystem.so" "$@"
fi
export ODIN3_QAIRT_LIB_DIR

ODIN3_FASTRPC_LIB_DIR=${ODIN3_FASTRPC_LIB_DIR:-"$root/fastrpc/lib"}
if [[ "$ODIN3_FASTRPC_LIB_DIR" != /* || "$ODIN3_FASTRPC_LIB_DIR" == *:* ||
      ! -r "$ODIN3_FASTRPC_LIB_DIR/libcdsprpc.so" ]]; then
    start_original "ODIN3_FASTRPC_LIB_DIR must contain readable libcdsprpc.so" "$@"
fi
export ODIN3_FASTRPC_LIB_DIR

if [[ -z "${ADSP_LIBRARY_PATH:-}" && -d "$data_root/qairt/hexagon" ]]; then
    fw_dir="$data_root/firmware"
    [[ -d "$fw_dir" ]] || fw_dir="/usr/lib/rfsa/adsp"
    export ADSP_LIBRARY_PATH="$data_root/qairt/hexagon;$fw_dir"
    export DSP_LIBRARY_PATH="$data_root/qairt/hexagon;$fw_dir"
fi

user_vk_layer="${XDG_DATA_HOME:-${HOME:-/var/home/armada}/.local/share}/vulkan/implicit_layer.d/VkLayer_MAKO_render.json"
if [[ ! -r "$user_vk_layer" ]]; then
    export VK_ADD_IMPLICIT_LAYER_PATH="$manifest${VK_ADD_IMPLICIT_LAYER_PATH:+:$VK_ADD_IMPLICIT_LAYER_PATH}"
fi

export MAKO_MULTIPLIER="$multiplier"
if [[ "$multiplier" == 3 ]]; then
    export ODIN3_HTP_3X=1
else
    unset ODIN3_HTP_3X ODIN3_HTP_RAW_SECOND_CONTEXT
fi

# Single drift-free phase-locked limiter (never stack DXVK_FRAME_RATE + MAKO_BASE_FPS_CAP in series!).
# When source_cap is '-', apply smart display-ceiling cap (refresh_hz / multiplier: 60 for 2x@120Hz, 40 for 3x@120Hz)
# unless ODIN3_FG_SMART_CEIL=0. Games running below the cap incur 0.00 ms wait overhead!
ceil_cap=$(( refresh_hz / multiplier ))
if [[ "$source_cap" != - && "$source_cap" =~ ^[1-9][0-9]{0,3}$ ]]; then
    effective_cap="$source_cap"
    if (( effective_cap > ceil_cap )) && [[ "${ODIN3_FG_SMART_CEIL:-1}" == "1" ]]; then
        effective_cap="$ceil_cap"
    fi
    export MAKO_BASE_FPS_CAP="$effective_cap"
elif [[ "${ODIN3_FG_SMART_CEIL:-1}" == "1" ]]; then
    export MAKO_BASE_FPS_CAP="$ceil_cap"
else
    unset MAKO_BASE_FPS_CAP
fi

export ODIN3_FG_REFRESH_HZ="$refresh_hz"
export ODIN3_HTP_INDEPENDENT_OUTPUT_HZ="$refresh_hz"

if [[ -n "$app_id" ]]; then
    export ODIN3_FG_RUNTIME_APP_ID="$app_id"
    if [[ -z ${ODIN3_FG_RUNTIME_STATE_DIR:-} && ${HOME:-} == /* ]]; then
        export ODIN3_FG_RUNTIME_STATE_DIR="$HOME/.cache/odin3-fg/runtime-state"
    fi
fi

export ENABLE_MAKO=1 MAKO_ENV=1 MAKO_FRAME_GENERATION_PROVISIONED=1 MAKO_FRAME_GENERATION_ENABLED=0
export MAKO_FRAME_GENERATION_MODEL=mako-fg MAKO_ADAPTIVE=0
export DISABLE_GAMESCOPE_WSI=1 ENABLE_GAMESCOPE_WSI=0 MAKO_DISABLE_HDR_EXPOSURE=1 PROTON_ENABLE_HDR=0 DXVK_HDR=0
export MAKO_SCALING_ENABLED=0
export DISABLE_LSFGVK=1 DISABLE_LSFG=1 VELA_LAUNCH_DONE=1 DISABLE_VELA_NPU=1
unset ODIN3_HTP_PAIR_PACING
export ODIN3_HTP_INDEPENDENT_REAL=1 ODIN3_HTP_ASYNC_REAL=1 ODIN3_HTP_INDEPENDENT_GENERATE=1
export ODIN3_HTP_INDEPENDENT_PREFETCH="${ODIN3_HTP_INDEPENDENT_PREFETCH:-1}"
export ODIN3_HTP_DEADLINE=1 ODIN3_HTP_ASYNC_SCHEDULE=1 ODIN3_HTP_MOTION=1
export ODIN3_FG_WORKER="$root/odin3-fg-worker" ODIN3_FG_WORKER_MODE=htp
export ODIN3_WSI_DIRECT_ZERO_COPY=${ODIN3_WSI_DIRECT_ZERO_COPY:-1}
if [[ "${ODIN3_WSI_DIRECT_ZERO_COPY}" == "1" && -r "$root/lib/libodin3-wsi-dma-capture.so" ]]; then
    export LD_PRELOAD="$root/lib/libodin3-wsi-dma-capture.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi
exec "$@"
