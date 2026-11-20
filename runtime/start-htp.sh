#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "$0")" && pwd)
sdk=${ODIN3_QAIRT_LINUX:-"$root/qairt-linux"}
context=${ODIN3_ANVIL_CONTEXT:-"$root/anvil-s-v79.bin"}
socket=fsr4npud_anvil_s_playtest

test "$(uname -m)" = aarch64
test -x "$root/anvil-npud"
test -r "$context"
test -r "$sdk/lib/libQnnHtp.so"
if ss -xln | grep -Fw "@$socket" >/dev/null; then
    printf 'The Odin 3 FG socket is already in use.\n' >&2
    exit 1
fi
export LD_LIBRARY_PATH="$sdk/lib:/usr/lib64/vela-npu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ADSP_LIBRARY_PATH="$sdk/hexagon;/usr/lib/rfsa/adsp"
exec "$root/anvil-npud" --name "$socket" --backend qnn --alloc rpcmem --ctx "$context"
