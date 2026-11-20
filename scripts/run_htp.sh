#!/usr/bin/bash
set -euo pipefail

lab="$(cd "$(dirname "$0")" && pwd)"
sdk="$lab/qairt-linux"
results="${1:?usage: run_htp.sh NEW_OUTPUT_DIRECTORY [INFERENCES]}"
runs="${2:-30}"
if [[ ! "$runs" =~ ^[1-9][0-9]{0,2}$ ]]; then
    echo 'INFERENCES must be between 1 and 999' >&2
    exit 2
fi
test -r "$lab/anvil-s.w8a8.dlc"
test -r "$lab/pair.raw"
mkdir -- "$results"
results="$(cd "$results" && pwd)"
printf 'input:=%s/pair.raw\n' "$lab" > "$results/input_list.txt"
printf '%s\n' '{"devices":[{"dsp_arch":"v79","soc_id":69,"device_id":0,"pd_session":"unsigned"}]}' \
    > "$results/htp.json"
printf '{"backend_extensions":{"shared_library_path":"%s/lib/libQnnHtpNetRunExtensions.so","config_file_path":"%s/htp.json"}}\n' \
    "$sdk" "$results" > "$results/backend.json"

export LD_LIBRARY_PATH="$sdk/lib:/usr/lib64/vela-npu"
export ADSP_LIBRARY_PATH="$sdk/hexagon;/usr/lib/rfsa/adsp"
cd "$results"
timeout 180 "$sdk/bin/qnn-context-binary-generator" \
    --backend "$sdk/lib/libQnnHtp.so" --model "$sdk/lib/libQnnModelDlc.so" \
    --dlc_path "$lab/anvil-s.w8a8.dlc" --binary_file anvil-s-v79 \
    --output_dir "$results" --config_file "$results/backend.json" > context.log 2>&1

timeout 100 "$sdk/bin/qnn-net-run" \
    --backend "$sdk/lib/libQnnHtp.so" --retrieve_context "$results/anvil-s-v79.bin" \
    --input_list "$results/input_list.txt" --output_dir "$results/output" \
    --config_file "$results/backend.json" --profiling_level basic \
    --num_inferences "$runs" --keep_num_outputs 1 > net-run.log 2>&1

"$sdk/bin/qnn-profile-viewer" --input_log "$results/output/qnn-profiling-data_0.log" \
    > "$results/profile.txt" 2>&1
grep -Fx "inferences_completed: $runs" "$results/output/execution_metadata.yaml"
