#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# CrystalLLM batch-step script
#
# This script runs a specific step (capture, planning, emulate) on existing
# log directories.
#
# Usage:
#   bash run_batch_step.sh <MASTER_ADDR> <MASTER_PORT> <MODE>
#
# Example:
#   bash run_batch_step.sh <master-ip> 40001 capture
#   bash run_batch_step.sh <master-ip> 40001 planning
#   bash run_batch_step.sh <master-ip> 40001 emulate
# -----------------------------------------------------------------------------

# Function to handle Ctrl+C
cleanup() {
    echo ""
    echo "Interrupted! Cleaning up..."
    pkill -P $$ 2>/dev/null || true
    exit 130
}
trap cleanup SIGINT SIGTERM

# =============================================================================
# Command Line Arguments
# =============================================================================

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <MASTER_ADDR> <MASTER_PORT> <MODE>"
    echo "  MASTER_ADDR - Master node IP address"
    echo "  MASTER_PORT - Master node port"
    echo "  MODE        - capture, planning, emulate"
    echo ""
    echo "Modes:"
    echo "  capture   - Capture communication traces"
    echo "  planning  - Run planning mode (traverse PP stages with runtime profiling)"
    echo "  emulate   - Emulate with single sandbox node"
    echo ""
    echo "Example:"
    echo "  $0 <master-ip> 40001 capture"
    echo "  $0 <master-ip> 40001 planning"
    echo "  $0 <master-ip> 40001 emulate"
    exit 1
fi

MASTER_ADDR="$1"
MASTER_PORT="$2"
MODE="$3"

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRYSTAL_WORKSPACE="${CRYSTAL_WORKSPACE:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
LOG_ROOT="${CRYSTAL_WORKSPACE}/logs"

export GPUS_PER_NODE=8

# Training config
TRAIN_ITERS=80
ANALYSIS_ITERS=10

# Coordinator server config (this current machine)
SERVER_ADDR="${SERVER_ADDR:-$(hostname -I | awk '{print $1}')}"
SERVER_PORT="${SERVER_PORT:-8888}"

# Override paths for trace/graph directories (useful when referencing capture data
# from outside the current workspace)
CRYSTAL_TRACE_DIR="${CRYSTAL_TRACE_DIR:-}"
CRYSTAL_GRAPH_DIR="${CRYSTAL_GRAPH_DIR:-}"

# Counters
TEST_COUNTER=0
SUCCESS_COUNT=0
FAIL_COUNT=0

# Failed cases list
declare -a FAILED_CASES=()

# =============================================================================
# Parse Directory Name
# =============================================================================

parse_log_dir() {
    local dir_name="$1"

    # Extract MODEL_TYPE (first part before underscore + model size, e.g., "Qwen3" or "GPT3")
    MODEL_TYPE="${dir_name%%_*}"

    # Extract SCALE (between "scale_" and end or timestamp)
    if ! SCALE=$(echo "$dir_name" | grep -oP 'scale_\K[0-9]+'); then
        echo "ERROR: Cannot parse SCALE from directory name: ${dir_name}"
        return 1
    fi

    # Extract CONFIG_NAME (between MODEL_TYPE_ and _scale_)
    local rest="${dir_name#${MODEL_TYPE}_}"
    CONFIG_NAME="${rest%%_scale_*}"

    # Extract MODEL_SIZE from CONFIG_NAME (first part before _tp_)
    MODEL_SIZE="${CONFIG_NAME%%_tp_*}"

    # Extract PP (pipeline parallel size) from directory name
    if ! PP_SIZE=$(echo "$dir_name" | grep -oP '(?<!v)pp_\K[0-9]+'); then
        echo "WARNING: Cannot parse PP_SIZE from directory name, defaulting to 1"
        PP_SIZE=1
    fi

    # Extract EP (expert parallel size) from directory name
    if ! EP_SIZE=$(echo "$dir_name" | grep -oP '(?<![a-z])ep_\K[0-9]+'); then
        echo "WARNING: Cannot parse EP_SIZE from directory name, defaulting to 8"
        EP_SIZE=8
    fi

    # Calculate TARGET_NODES and default AVAILABLE_NODES
    TARGET_NODES=$((SCALE / 8))
    AVAILABLE_NODES="${AVAILABLE_NODES:-${TARGET_NODES}}"

    return 0
}

# =============================================================================
# Run Mode Functions
# =============================================================================

run_capture() {
    echo "[run] capture mode"
    mkdir -p "${LOG_BASE_DIR}/capture"
    crystalllm run capture \
        --crystal-workspace "${CRYSTAL_WORKSPACE}" \
        --log-dir-name "logs/${LOG_NAME}/capture" \
        --target-nodes "${TARGET_NODES}" \
        --available-nodes "${AVAILABLE_NODES}" \
        --train-script-path "${TRAIN_SCRIPT_PATH}" \
        --server-addr "${SERVER_ADDR}" \
        --server-port "${SERVER_PORT}" \
        --master-port "${MASTER_PORT}" \
        --train-iters "${TRAIN_ITERS}" \
        --analysis-iters "${ANALYSIS_ITERS}" \
        --pretrain-args "${PRETRAIN_ARGS}" \
        --pretrain-args-json "${PRETRAIN_ARGS_JSON}" \
        --pretrain-envs-json "${PRETRAIN_ENVS_JSON}"
}

run_emulate() {
    # Calculate sandbox nodes based on EP_SIZE / 8
    local SANDBOX_NODE_COUNT=$((EP_SIZE / 8))
    if [[ $SANDBOX_NODE_COUNT -lt 1 ]]; then
        SANDBOX_NODE_COUNT=1
    fi

    # Build sandbox nodes array: [0, 1, ..., SANDBOX_NODE_COUNT-1]
    local SANDBOX_NODES_ARRAY=()
    for ((i=0; i<SANDBOX_NODE_COUNT; i++)); do
        SANDBOX_NODES_ARRAY+=("$i")
    done

    # Calculate pp-rank-stride = world_size / pp_size
    local WORLD_SIZE=$((TARGET_NODES * GPUS_PER_NODE))
    local PP_RANK_STRIDE=$((WORLD_SIZE / PP_SIZE))

    local DATA_SOURCE_DIR="${LOG_BASE_DIR}"

    echo "[run] emulate mode"
    echo "[run] EP_SIZE: ${EP_SIZE}, sandbox nodes: ${SANDBOX_NODES_ARRAY[*]}"
    echo "[run] Config file: ${PRETRAIN_ARGS_JSON}"
    echo "[run] Output directory: ${LOG_BASE_DIR}/emulate"
    echo "[run] Data source directory: ${DATA_SOURCE_DIR}"

    mkdir -p "${LOG_BASE_DIR}/emulate"
    crystalllm run emulate \
        --crystal-workspace "${CRYSTAL_WORKSPACE}" \
        --log-dir-name "logs/${LOG_NAME}/emulate" \
        --target-nodes "${TARGET_NODES}" \
        --available-nodes "${AVAILABLE_NODES}" \
        --train-script-path "${TRAIN_SCRIPT_PATH}" \
        --server-addr "${SERVER_ADDR}" \
        --server-port "${SERVER_PORT}" \
        --master-port "${MASTER_PORT}" \
        --sandbox-nodes "${SANDBOX_NODES_ARRAY[@]}" \
        --trace-dir "${CRYSTAL_TRACE_DIR:-${DATA_SOURCE_DIR}/capture/trace}" \
        --graph-dir "${CRYSTAL_GRAPH_DIR:-${DATA_SOURCE_DIR}/capture/graph}" \
        --profiler-dir "${DATA_SOURCE_DIR}/planning/profiler/crystal" \
        --pp-rank-stride "${PP_RANK_STRIDE}" \
        --dp-rank-stride "$((SANDBOX_NODE_COUNT * GPUS_PER_NODE))" \
        --train-iters "${TRAIN_ITERS}" \
        --analysis-iters "${ANALYSIS_ITERS}" \
        --pretrain-args "${PRETRAIN_ARGS}" \
        --pretrain-args-json "${PRETRAIN_ARGS_JSON}" \
        --pretrain-envs-json "${PRETRAIN_ENVS_JSON}" \
        --enable-crystal-timer \
        --tensor-provider "${TENSOR_GENERATOR}" \
        --enable-mock-nccl-bootstrap \
        --enable-crystal-monitor
}

run_planning() {
    local PLANNING_EPOCHS="${PLANNING_EPOCHS:-1}"

    # Calculate sandbox nodes based on EP_SIZE / 8
    local SANDBOX_NODE_COUNT=$((EP_SIZE / 8))
    if [[ $SANDBOX_NODE_COUNT -lt 1 ]]; then
        SANDBOX_NODE_COUNT=1
    fi

    # Build sandbox nodes array: [0, 1, ..., SANDBOX_NODE_COUNT-1]
    local SANDBOX_NODES_ARRAY=()
    for ((i=0; i<SANDBOX_NODE_COUNT; i++)); do
        SANDBOX_NODES_ARRAY+=("$i")
    done

    local PLANNING_SANDBOX_NODES_COUNT="${SANDBOX_NODE_COUNT}"
    local TRACE_PATH="${CRYSTAL_TRACE_DIR:-${LOG_BASE_DIR}/capture/trace}"

    # Calculate pp-rank-stride = world_size / pp_size
    local WORLD_SIZE=$((TARGET_NODES * GPUS_PER_NODE))
    local PP_RANK_STRIDE=$((WORLD_SIZE / PP_SIZE))

    echo "[run] planning mode"
    echo "[run] PP_SIZE: ${PP_SIZE}, WORLD_SIZE: ${WORLD_SIZE}"
    echo "[run] PP rank stride: ${PP_RANK_STRIDE} (world_size / pp_size)"
    echo "[run] Sandbox nodes: ${SANDBOX_NODES_ARRAY[*]} (EP_SIZE / 8 = ${SANDBOX_NODE_COUNT})"
    echo "[run] Planning epochs: ${PLANNING_EPOCHS}"
    echo "[run] Sandbox nodes per batch: ${PLANNING_SANDBOX_NODES_COUNT}"

    mkdir -p "${LOG_BASE_DIR}/planning"

    crystalllm run planning \
        --crystal-workspace "${CRYSTAL_WORKSPACE}" \
        --log-dir-name "logs/${LOG_NAME}/planning" \
        --target-nodes "${TARGET_NODES}" \
        --available-nodes "${AVAILABLE_NODES}" \
        --train-script-path "${TRAIN_SCRIPT_PATH}" \
        --server-addr "${SERVER_ADDR}" \
        --server-port "${SERVER_PORT}" \
        --master-port "${MASTER_PORT}" \
        --sandbox-nodes "${SANDBOX_NODES_ARRAY[@]}" \
        --trace-dir "${TRACE_PATH}" \
        --graph-dir "${CRYSTAL_GRAPH_DIR:-${LOG_BASE_DIR}/capture/graph}" \
        --pp-rank-stride "${PP_RANK_STRIDE}" \
        --dp-rank-stride "$((SANDBOX_NODE_COUNT * GPUS_PER_NODE))" \
        --planning-epochs "${PLANNING_EPOCHS}" \
        --planning-sandbox-nodes-count "${PLANNING_SANDBOX_NODES_COUNT}" \
        --train-iters "${TRAIN_ITERS}" \
        --analysis-iters "${ANALYSIS_ITERS}" \
        --pretrain-args "${PRETRAIN_ARGS}" \
        --pretrain-args-json "${PRETRAIN_ARGS_JSON}" \
        --pretrain-envs-json "${PRETRAIN_ENVS_JSON}" \
        --tensor-provider "${TENSOR_GENERATOR}" \
        --enable-mock-nccl-bootstrap || return $?

    echo ""
    echo "[run] planning mode completed"
}

execute_mode() {
    case "${MODE}" in
        capture)
            run_capture
            ;;
        emulate)
            run_emulate
            ;;
        planning)
            run_planning
            ;;
        *)
            echo "ERROR: Unknown mode: ${MODE}"
            return 1
            ;;
    esac
}

# =============================================================================
# Helper Function
# =============================================================================

run_case() {
    local LOG_DIR_NAME=$1

    TEST_COUNTER=$((TEST_COUNTER + 1))

    # Full path to log directory
    LOG_BASE_DIR="${LOG_ROOT}/${LOG_DIR_NAME}"
    LOG_NAME="${LOG_DIR_NAME}"

    # Validate directory exists
    if [[ ! -d "${LOG_BASE_DIR}" ]]; then
        echo "ERROR: Log directory does not exist: ${LOG_BASE_DIR}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_CASES+=("${LOG_DIR_NAME} (directory not found)")
        return 1
    fi

    # Parse directory name to extract configuration
    if ! parse_log_dir "${LOG_DIR_NAME}"; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_CASES+=("${LOG_DIR_NAME} (parse error)")
        return 1
    fi

    # The backend entry forwards to the reconstructed Pai training script.
    TRAIN_SCRIPT_PATH="${SCRIPT_DIR}/train.sh"
    printf -v PRETRAIN_ARGS '%q %q %q' \
        "${CRYSTAL_WORKSPACE}/Pai-Megatron-Patch-Crystal" "${MODEL_TYPE}" "${MODEL_SIZE}"

    # Configs live alongside this driver, whether scripts are copied or linked.
    PRETRAIN_ARGS_JSON="${CRYSTAL_PRETRAIN_ARGS_JSON:-${SCRIPT_DIR}/configs/${SCALE}/${CONFIG_NAME}.json}"
    PRETRAIN_ENVS_JSON="${SCRIPT_DIR}/envs.json"
    TENSOR_GENERATOR="${CRYSTAL_WORKSPACE}/Pai-Megatron-Patch-Crystal/crystalllm/replay/tensor_providers.py"

    # Validate config file exists
    if [[ ! -f "${PRETRAIN_ARGS_JSON}" ]]; then
        echo "ERROR: Config file not found: ${PRETRAIN_ARGS_JSON}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_CASES+=("${LOG_DIR_NAME} (config not found)")
        return 1
    fi

    echo ""
    echo "============================================"
    echo "Test Case $TEST_COUNTER: ${LOG_DIR_NAME}"
    echo "  Model Type: ${MODEL_TYPE}"
    echo "  Config: ${CONFIG_NAME}"
    echo "  Scale: ${SCALE} GPUs (${TARGET_NODES} nodes)"
    echo "  Mode: ${MODE}"
    echo "============================================"

    # Run specified mode
    if execute_mode; then
        echo "Test Case $TEST_COUNTER completed successfully"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        local EXIT_CODE=$?
        echo "Test Case $TEST_COUNTER FAILED with exit code $EXIT_CODE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_CASES+=("${LOG_DIR_NAME}")
    fi

    sleep 5
}

# =============================================================================
# Main
# =============================================================================

echo "============================================"
echo "CrystalLLM Batch Step Runner"
echo "============================================"
echo "MASTER_ADDR  : ${MASTER_ADDR}"
echo "MASTER_PORT  : ${MASTER_PORT}"
echo "MODE         : ${MODE}"
echo "LOG_ROOT     : ${LOG_ROOT}"
echo "============================================"

# Validate mode
case "${MODE}" in
    capture|planning|emulate)
        ;;
    *)
        echo "ERROR: Invalid mode: ${MODE}"
        echo "Valid modes: capture, planning, emulate"
        exit 1
        ;;
esac

# =============================================================================
# Test Cases
# Format: run_case "<LOG_DIR_NAME>"
# The LOG_DIR_NAME is the directory name under ${LOG_ROOT} (logs folder)
# =============================================================================

run_case "Qwen3_A20B_tp_1_pp_4_vpp_0_etp_1_ep_8_mbs_1_gbs_1024_scale_512"

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================"
echo "Batch Step Run Complete"
echo "============================================"
echo "Total test cases: ${TEST_COUNTER}"
echo "Successful: ${SUCCESS_COUNT}"
echo "Failed: ${FAIL_COUNT}"

if [[ ${#FAILED_CASES[@]} -gt 0 ]]; then
    echo ""
    echo "Failed cases:"
    for case in "${FAILED_CASES[@]}"; do
        echo "  - ${case}"
    done
fi

echo "============================================"

# Exit with error if any test failed
if [[ $FAIL_COUNT -gt 0 ]]; then
    exit 1
fi
