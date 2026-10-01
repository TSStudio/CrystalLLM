#!/usr/bin/env bash
# Forward Crystal's training arguments to the reconstructed Pai source.
set -euo pipefail

PAI_ROOT="${1:?path to the patched Pai source is required}"
MODEL_TYPE="${2:?model type is required}"
shift 2

if [[ "$MODEL_TYPE" == GPT3 ]]; then
    ENTRY="$PAI_ROOT/examples/gpt3/run_pretrain_megatron_gpt3_crystal.sh"
else
    ENTRY="$PAI_ROOT/examples/qwen3/run_mcore_qwen3_crystal.sh"
fi
if [[ ! -f "$ENTRY" ]]; then
    echo "Pai training entry not found: $ENTRY" >&2
    exit 1
fi
exec bash "$ENTRY" "$@"
