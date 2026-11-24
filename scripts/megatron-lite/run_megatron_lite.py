#!/usr/bin/env python3
"""Run one CrystalLLM stage for the bundled Megatron Lite A20B example."""

import argparse
import os
import shlex
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT = SCRIPT_DIR.parents[1]
CASE = "Qwen3_A20B_tp_1_pp_4_vpp_1_cp_1_etp_1_ep_8_mbs_1_gbs_1024_scale_512"
TRAIN_ITERS = 80
ANALYSIS_ITERS = 10


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "planning", "emulate"))
    parser.add_argument(
        "--workspace", type=Path, default=os.getenv("CRYSTAL_WORKSPACE")
    )
    parser.add_argument("--server-addr", required=True)
    parser.add_argument("--server-port", type=int, default=8891)
    parser.add_argument("--master-port", type=int, default=40001)
    parser.add_argument("--available-nodes", type=int, default=2)
    parser.add_argument("--target-nodes", type=int, default=64)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--pp-rank-stride", type=int, default=128)
    parser.add_argument("--dp-rank-stride", type=int, default=8)
    parser.add_argument("--case-name", default=CASE)
    parser.add_argument("--case-config", type=Path)
    parser.add_argument("--envs-json", type=Path, default=SCRIPT_DIR / "envs.json")
    parser.add_argument(
        "--train-iters",
        type=int,
        default=TRAIN_ITERS,
        help="Iterations before the ten-step analysis window (default: 80)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without starting a job",
    )
    args = parser.parse_args(argv)
    if args.workspace is None:
        parser.error("set CRYSTAL_WORKSPACE or pass --workspace")
    args.workspace = args.workspace.expanduser().resolve()
    if args.target_nodes < 2 or not 2 <= args.available_nodes <= args.target_nodes:
        parser.error("available nodes must be between 2 and target nodes")
    if args.gpus_per_node < 1 or args.pp_rank_stride < 1 or args.dp_rank_stride < 1:
        parser.error("GPU count and rank strides must be positive")
    if (args.target_nodes, args.gpus_per_node) != (64, 8) and args.case_config is None:
        parser.error("a custom layout requires --case-config for a compatible workload")
    if args.mode != "capture":
        world_size = args.target_nodes * args.gpus_per_node
        if (args.dp_rank_stride % args.gpus_per_node or
                args.pp_rank_stride % args.dp_rank_stride or
                world_size % args.pp_rank_stride):
            parser.error("rank strides must align with whole nodes and divide the world size")
    if args.case_name != Path(args.case_name).name or args.case_name in {".", ".."}:
        parser.error("case name must be a single directory name")
    if args.train_iters < 3:
        parser.error("--train-iters must be at least 3 for replay warmup")
    for port in (args.server_port, args.master_port):
        if not 1 <= port <= 65535:
            parser.error("ports must be between 1 and 65535")
    return args


def make_command(args):
    base = args.workspace / "logs" / args.case_name
    train_args = shlex.quote(str(args.workspace / "Megatron-LM"))
    if args.case_config is not None:
        train_args += " " + shlex.quote(str(args.case_config.expanduser().resolve()))
    command = [
        "crystalllm",
        "run",
        args.mode,
        "--crystal-workspace",
        str(args.workspace),
        "--log-dir-name",
        f"logs/{args.case_name}/{args.mode}",
        "--target-nodes",
        str(args.target_nodes),
        "--available-nodes",
        str(args.available_nodes),
        "--gpus-per-node",
        str(args.gpus_per_node),
        "--train-script-path",
        str(SCRIPT_DIR / "train.sh"),
        "--server-addr",
        args.server_addr,
        "--server-port",
        str(args.server_port),
        "--master-port",
        str(args.master_port),
        "--train-iters",
        str(args.train_iters),
        "--analysis-iters",
        str(ANALYSIS_ITERS),
        # Crystal joins these values into a shell command before node launch.
        "--pretrain-args",
        train_args,
        "--pretrain-envs-json",
        str(args.envs_json.expanduser().resolve()),
    ]
    if args.mode != "capture":
        command += [
            "--sandbox-nodes",
            "0",
            "--trace-dir",
            str(base / "capture/trace"),
            "--graph-dir",
            str(base / "capture/graph"),
            "--pp-rank-stride",
            str(args.pp_rank_stride),
            "--dp-rank-stride",
            str(args.dp_rank_stride),
            "--tensor-provider",
            str(
                ARTIFACT
                / "crystalllm/src/crystalllm/integration/megatron_lite_tensor_providers.py"
            ),
            "--enable-mock-nccl-bootstrap",
        ]
        if args.mode == "planning":
            command += ["--planning-epochs", "1", "--planning-sandbox-nodes-count", "1"]
        else:
            command += ["--profiler-dir", str(base / "planning/profiler/crystal")]
    return command, base


def main(argv=None):
    args = parse_args(argv)
    command, base = make_command(args)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    required = [
        args.workspace / "venv/bin/python",
        args.workspace / "Megatron-LM/experimental/lite/examples/bench/bench.py",
    ]
    if args.mode != "capture":
        required += [base / "capture/trace", base / "capture/graph"]
    if args.mode == "emulate":
        required += [base / "planning/profiler/crystal"]
    for path in required:
        if not path.exists():
            raise SystemExit(f"Missing prerequisite: {path}")
    output = base / args.mode
    output.mkdir(parents=True, exist_ok=True)
    with (output / "coordinator.log").open("a", encoding="utf-8") as log:
        log.write("\n" + shlex.join(command) + "\n")
        log.flush()
        # Append diagnostics and let Crystal handle in-place resume.
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    (output / "exit_code.txt").write_text(f"{result.returncode}\n", encoding="utf-8")
    print(
        f"{args.mode}: exit={result.returncode}, log={output / 'coordinator.log'}",
        flush=True,
    )
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
