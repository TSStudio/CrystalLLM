#!/usr/bin/env bash
# Qwen3 A20B, TP1/PP4/EP8/ETP1/CP1/VPP1, 512 logical ranks.
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
MEGATRON_ROOT="${1:?path to the patched Megatron Lite source is required}"
BENCH="${MEGATRON_ROOT}/experimental/lite/examples/bench/bench.py"
if [[ ! -f "$BENCH" ]]; then
    echo "Megatron Lite benchmark not found: $BENCH" >&2
    exit 1
fi

: "${MASTER_ADDR:?}" "${MASTER_PORT:?}" "${NNODES:?}" "${NODE_RANK:?}"
: "${GPUS_PER_NODE:?}" "${CRYSTAL_TRAIN_ITERS:?}"
export PYTHONPATH="${MEGATRON_ROOT}/experimental/lite:${MEGATRON_ROOT}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

CASE_CONFIG="${2:-${SCRIPT_DIR}/configs/512/A20B_tp_1_pp_4_vpp_1_cp_1_etp_1_ep_8_mbs_1_gbs_1024.json}"
exec python - "$BENCH" "$CASE_CONFIG" <<'PY'
import json
import os
import sys
from pathlib import Path

bench, config_path = sys.argv[1:]
config = json.loads(Path(config_path).read_text())
config["hf_path"] = str((Path(config_path).parent / config["hf_path"]).resolve())
config["steps"] = int(os.environ["CRYSTAL_TRAIN_ITERS"])
command = [
    "crystalllm-torchrun",
    "--master-addr", os.environ["MASTER_ADDR"],
    "--master-port", os.environ["MASTER_PORT"],
    "--nnodes", os.environ["NNODES"],
    "--node-rank", os.environ["NODE_RANK"],
    "--nproc-per-node", os.environ["GPUS_PER_NODE"],
    "--no-python", "crystalllm-numarun", "python", "-u", bench,
]
for key, value in config.items():
    option = "--" + key.replace("_", "-")
    if isinstance(value, bool):
        if value:
            command.append(option)
    else:
        command.extend([option, json.dumps(value, separators=(",", ":")) if isinstance(value, dict) else str(value)])
os.execvp(command[0], command)
PY
