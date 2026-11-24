#!/usr/bin/env bash
# Launch through the CrystalLLM coordinator in capture mode.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CRYSTAL_TORCH_OVERLAY:?set to the node-local patched PyTorch overlay}"
export PYTHONPATH="${CRYSTAL_TORCH_OVERLAY}${PYTHONPATH:+:$PYTHONPATH}"
exec crystalllm-torchrun --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    --nnodes "$NNODES" --node-rank "$NODE_RANK" --nproc-per-node "$GPUS_PER_NODE" \
    "$SCRIPT_DIR/capture_smoke.py"
