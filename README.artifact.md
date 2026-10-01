# CrystalLLM — Reproduction Artifact

For the project overview and paper, see the [project README](README.md).

CrystalLLM emulates large-scale Megatron-LM MoE training on a small physical
cluster. Instead of running the full GPU cluster for real, one **sandbox** node
runs the actual Megatron training while one or more **assistant** nodes replay
recorded collective traces using a *customized NCCL* — so all process groups
appear fully populated from the sandbox's perspective. The end product is a set
of per-iteration timings that closely match the corresponding real-cluster run.

This guide documents the reproduction workflow. It is meant to be followed top-to-bottom;
log fragments are quoted where useful.

Choose the [Megatron/Pai workflow](#3-reproduction) or the
[Megatron-Lite workflow](#4-megatron-lite-reproduction). Each backend has its
own training entry and environment.

---

## 1. Hardware / Software Prerequisites

CrystalLLM uses **three roles**; the coordinator may share the sandbox host
(two GPU client hosts in total):

| Role | Count | Notes |
|---|---|---|
| Coordinator (Server) | 1 | Owns local Redis, runs `crystalllm run …`. GPU optional for `planning`/`emulate` modes; required if the host also acts as an assistant. |
| Sandbox client (Master) | 1 | 1× GPU node running real Megatron training with `LD_PRELOAD=baseBuild/libnccl.so`. |
| Assistant client (Worker) | 1 | 1× GPU node running trace replay with `LD_PRELOAD=mockBuild/libnccl.so` + local Redis. |

> The GPU client hosts **must be hardware-homogeneous** (same GPU model, driver,
> CPU, and NIC topology). The mock-NCCL `setup_group_neighbors` routine reads
> local topology when computing per-group neighbors; heterogeneous hosts would
> produce divergent neighbor lists.

Other expectations:

- Linux x86_64 with CUDA toolkit and a CUDA-capable PyTorch.
  **The Crystal Patch (`distributed_c10d.py`) is tightly coupled to a specific
  PyTorch version.** Different PyTorch releases change line numbers, function
  signatures, and internal structures (`ncclConfig_t` offsets) that the patch
  depends on. The tested and supported environment is:

  ```
  Docker base:  NVIDIA PyTorch NGC 25.06 (Ubuntu 24.04) — nvcr.io/nvidia/pytorch:25.06-py3
  PyTorch:      2.8.0a0+5228986c39.nv25.06
  CUDA:         12.9
  Python:       3.12.3
  ```

  Running on a different PyTorch version will likely require regenerating
  the patch.
- Python ≥ 3.10 (tested with 3.12.3).
- `redis-server` reachable on each client at `localhost:6379`.
- Outbound TCP between coordinator and clients on `SERVER_PORT` (default `8888`)
  and between clients on `MASTER_PORT` (default `40001`).
- **CPU:** ≥ 32 CPU cores per GPU is recommended for `planning`/`emulate`. Fewer cores still run
  correctly, but may be slower.

## 2. Repository Layout

```
CrystalLLM/
├── crystalllm/            # Python package + setup.py (builds the three libnccl variants)
│   ├── src/crystalllm/    # Library source (entry, coordinator, runner, replay, …)
│   ├── scripts/           # crystalllm-init / crystalllm / crystalllm-numarun / crystalllm-pytorch / crystalllm-torchrun
│   ├── patches/           # Crystal Patches (see Steps 3.1.a and 3.3)
│   │   ├── pytorch-patch/           # Crystal modifications for PyTorch distributed
│   │   │   ├── distributed_c10d.py      # Patched file (ready to deploy)
│   │   │   └── distributed_c10d.patch   # Unified diff vs stock PyTorch 2.8.0a0+nv25.06
│   │   ├── megatron-patch/          # Crystal modifications for Pai-Megatron-Patch
│   │   │   └── crystal.patch        # Unified diff vs upstream commit 7a91081
│   │   ├── megatron-lm-patch/       # Crystal modifications for Megatron-LM
│   │   │   └── megatron_lm.patch    # Unified diff vs upstream commit 6a6cd47
│   │   └── megatron-lite-patch/     # Crystal modifications for Megatron Lite
│   │       └── megatron_lite.patch  # Unified diff vs upstream commit 53f4976
│   ├── vendor/nccl        # symlink → ../../nccl, populated in Step 3.2
│   └── setup.py           # Builds baseBuild / fakeBuild / mockBuild
├── nccl/                  # Vendored NCCL source (the actual library code)
└── scripts/
    ├── megatron-pai/
    │   ├── run_batch_step.sh  # Capture / planning / emulate driver
    │   ├── train.sh           # Calls the patched Pai training entry
    │   ├── envs.json          # Training environment variables
    │   ├── requirements.txt   # Client dependencies
    │   ├── constraints.txt    # Dependency version constraints
    │   └── configs/512/       # Parallelism and batch configuration
    └── megatron-lite/
        ├── run_megatron_lite.py  # Capture / planning / emulate driver
        ├── train.sh             # Calls the native Lite benchmark
        ├── envs.json             # Training environment variables
        ├── requirements.txt     # Client dependencies
        ├── constraints.txt      # Dependency version constraints
        └── configs/             # Case parameters and local model configuration
```

## 3. Reproduction

The flow is:

1. **Step 3.1** — Set up the workspace tree (Server, no GPU required).
2. **Step 3.2** — Install the coordinator on the Server.
3. **Step 3.3** — Install the clients on each client node.
4. **Step 3.4** — Launch the coordinator and clients, run `capture` mode to
   generate trace + graph artifacts.
5. **Step 3.5** — Run `planning` mode against the captured trace to produce
   per-operation runtime profiler data.
6. **Step 3.6** — Run `emulate` mode using the captured trace + planning
   runtime profiler data for full-fidelity replay with profiling.

### 3.1 Workspace bootstrap (Server)

All commands in this section are run from the *parent directory* that contains
this source tree, i.e. the directory in which `CrystalLLM/` sits. The
workspace will be created as a sibling:

```
<parent>/
├── CrystalLLM/             # this repo (contains README.artifact.md)
└── CrystalLLM_workspace/   # created in this step
```

```bash
# cd to the parent directory of CrystalLLM/ before running anything below.
mkdir -p CrystalLLM_workspace/logs
export CRYSTAL_WORKSPACE="$(realpath CrystalLLM_workspace)"
```

#### 3.1.a — Megatron-Patch-Crystal

CrystalLLM trains and replays through a Crystal-modified fork of
[Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch). The
modifications span 12 files and ship as a patch in
`CrystalLLM/crystalllm/patches/megatron-patch/`.

Set up from upstream:

```bash
# Clone upstream and pin to the tested base commit
git clone https://github.com/alibaba/Pai-Megatron-Patch.git \
    CrystalLLM_workspace/Pai-Megatron-Patch-Crystal
cd CrystalLLM_workspace/Pai-Megatron-Patch-Crystal
git checkout 7a910814aea646832a2e3d05e3a53a1ee3703ed6

# Apply the Crystal patch (adds pretrain scripts, tokenizer fix, forward_step hook)
git apply ../../CrystalLLM/crystalllm/patches/megatron-patch/crystal.patch

# Initialize the Megatron-LM backend submodule (from public NVIDIA upstream,
# pinned to 6a6cd47 by the base commit's gitlink), then apply the Crystal patch
git submodule update --init backends/megatron/Megatron-LM-250624
MEGATRON_LM_PATCH=$(realpath ../../CrystalLLM/crystalllm/patches/megatron-lm-patch/megatron_lm.patch)
cd backends/megatron/Megatron-LM-250624
git apply "$MEGATRON_LM_PATCH"
cd ../../..   # back to Pai-Megatron-Patch-Crystal

# Download the Qwen3 tokenizer (used by the A20B config)
bash resource/get_qwen3.sh

cd ../..   # back to parent directory
```

Before Capture, provide a Qwen3-tokenized Megatron MMAP dataset as
`CrystalLLM_workspace/Pai-Megatron-Patch-Crystal/resource/downloads/qwen3/dataset/mmap_qwen3_datasets_text_document.bin`
and the matching `.idx` file. Ensure both files are in place before proceeding.

#### 3.1.b — Scripts and configs

The Pai driver (`run_batch_step.sh`), training entry, environment variables,
dependency files, and pretrain configs ship under
`CrystalLLM/scripts/megatron-pai/`. The workspace
needs access to them — choose one option below with no existing `scripts/` destination:

```bash
# Option 1 (recommended): symlink — edits to the repo copy are picked up immediately
ln -s $(realpath CrystalLLM/scripts) CrystalLLM_workspace/scripts

# Option 2: copy (allows independent edits per workspace)
cp -r CrystalLLM/scripts CrystalLLM_workspace/scripts
```

The driver is available at
`CrystalLLM_workspace/scripts/megatron-pai/run_batch_step.sh`.
It determines its workspace from the `CRYSTAL_WORKSPACE` environment
variable. If unset, it defaults to the directory containing
`scripts/`. Set `CRYSTAL_WORKSPACE` explicitly when running the artifact copy
directly. Network settings in `scripts/megatron-pai/envs.json` should match the clients.

#### 3.1.c — Empty log directory for the target case

`run_batch_step.sh` parses the *directory name* under `logs/` to extract the
model config and training scale. The naming convention is:

```
<Model>_<Size>_<parallelism>_<batch>_scale_<GPUs>_<timestamp>
```

Create the directory before the first run:

```bash
mkdir -p CrystalLLM_workspace/logs/Qwen3_A20B_tp_1_pp_4_vpp_0_etp_1_ep_8_mbs_1_gbs_1024_scale_512
```

> The directory name must match the `run_case` argument at the end of
> `run_batch_step.sh`.

#### 3.1.d — Sanity check

```bash
ls CrystalLLM/crystalllm/setup.py
ls CrystalLLM_workspace/scripts/megatron-pai/run_batch_step.sh
ls CrystalLLM_workspace/Pai-Megatron-Patch-Crystal/examples/qwen3/run_mcore_qwen3_crystal.sh
ls CrystalLLM_workspace/scripts/megatron-pai/configs/512/A20B_tp_1_pp_4_vpp_0_etp_1_ep_8_mbs_1_gbs_1024.json
```

All four paths must exist. Continue to **Step 3.2** below.

### 3.2 Coordinator install (Server)

The Server hosts the `crystalllm run …` coordinator process plus a local Redis
used to publish topology data to the assistants. Building the package compiles
three NCCL variants (`baseBuild`, `fakeBuild`, `mockBuild`), so the build host
must have CUDA toolchain and at least one CUDA
device visible (`setup.py` queries `torch.cuda.get_device_capability(0)` to
pick `-gencode=sm_XX`; alternatively set `CRYSTALLLM_SM_VERSION` explicitly
for the target GPU).

#### 3.2.a — Link the NCCL source into the package

`crystalllm/setup.py` expects to find NCCL source under
`crystalllm/vendor/nccl/`. The repo ships NCCL as a top-level sibling directory
(`CrystalLLM/nccl/`); link it in:

```bash
ln -sfn ../../nccl CrystalLLM/crystalllm/vendor/nccl
ls CrystalLLM/crystalllm/vendor/nccl/makefiles/version.mk   # sanity check
```

#### 3.2.b — System packages

The NCCL build links against `libhiredis` (used by the Crystal Redis-control
extension inside NCCL). Install it once per host:

```bash
apt-get update && apt-get install -y libhiredis-dev redis-server
```

Verify Redis is running locally:

```bash
redis-cli -h localhost -p 6379 PING        # expect: PONG
```

Make sure the CUDA toolkit's `nvcc` is on `PATH` — NVIDIA's PyTorch image
installs CUDA under `/usr/local/cuda/` but does not always add the `bin/`
subdirectory to the default shell PATH:

```bash
which nvcc || export PATH=/usr/local/cuda/bin:$PATH
nvcc --version | tail -3      # confirm a release line is printed
```

Persist the `export` in your shell rc (e.g. `~/.bashrc`) if you intend to run
multiple `pip install` rounds.

#### 3.2.c — Fresh virtual environment

```bash
cd CrystalLLM_workspace
python -m venv venv --system-site-packages
source venv/bin/activate
pip install --upgrade pip
pip install "numpy<2" redis
```

`--system-site-packages` lets the venv re-use the system PyTorch (the NVIDIA
pre-release build in our case), which is required for both compiling and
loading the vendored NCCL — installing a different `torch` in the venv would
mismatch the ABI.

#### 3.2.d — Build and install the `crystalllm` package

With the venv active and CUDA visible:

```bash
# setup.py imports torch at top level; pip's build isolation hides system
# site-packages, so install build deps explicitly and disable isolation.
pip install 'setuptools>70.1' wheel pybind11

# Build the wheel, then install it (two-step needed due to custom wheel naming).
cd ../CrystalLLM/crystalllm
python setup.py bdist_wheel
pip install dist/crystalllm-*.whl
```

This compiles all three NCCL variants in `vendor/nccl/build/{base,fake,mock}Build/`
and packages them under `src/crystalllm/nccl/lib/{baseBuild,fakeBuild,mockBuild}/`.
Expect 10–30 minutes on a single-GPU build host.

#### 3.2.e — Sanity check

```bash
# Three libnccl variants present in the package install:
python -c "
import crystalllm, os
lib = os.path.join(os.path.dirname(crystalllm.__file__), 'nccl', 'lib')
for v in ('baseBuild', 'fakeBuild', 'mockBuild'):
    p = os.path.join(lib, v, 'libnccl.so')
    print(v, os.path.exists(p), os.path.getsize(p) if os.path.exists(p) else '-')
"

# Entry-point script in PATH:
which crystalllm
```

The `crystalllm` shell wrapper resolves its venv relative to the
`--crystal-workspace` argument (defaulting to the current working directory).
So `crystalllm run --help` must either be run from the workspace root or be
given the workspace path explicitly:

```bash
# Option 1: cd into the workspace first
cd "$CRYSTAL_WORKSPACE"
crystalllm run --help | head -5

# Option 2: pass --crystal-workspace
crystalllm run --crystal-workspace "$CRYSTAL_WORKSPACE" --help | head -5
```

Continue to **Step 3.3** below to install the same package onto each client
node.
### 3.3 Client install (each client node)

Each client node runs the `crystalllm client` daemon, which accepts commands
from the coordinator and spawns training (sandbox) or replay (assistant)
subprocesses. The client workspace lives on **local disk** (under
`/root/crystalllm_workspace`) to avoid cross-node file conflicts. Only `logs/`
is symlinked back to the shared workspace so the coordinator can read outputs.

`crystalllm-pytorch replace` patches the system PyTorch's
`distributed_c10d.py` with the Crystal-patched copy. The coordinator (Server)
does not need this patch unless it also acts as a client.

Run the script below on **each** client node from the parent directory containing
`CrystalLLM/` and `CrystalLLM_workspace/`, using the same newly built wheel:

```bash
#!/bin/bash
set -e
set -o pipefail

REMOTE_WORKSPACE="$(realpath CrystalLLM_workspace)"
LOCAL_WORKSPACE="/root/crystalllm_workspace"
WHEEL_PATH="$(ls "$(realpath CrystalLLM/crystalllm/dist)"/crystalllm-*.whl)"

SERVER_ADDR="${SERVER_ADDR:-XXXX}"
SERVER_PORT="${SERVER_PORT:-XXXX}"

# System deps
apt-get update -qq
groupadd -r -g 101 redis 2>/dev/null || true
useradd  -r -g redis -u 101 -s /usr/sbin/nologin redis 2>/dev/null || true
apt-get install -y libhiredis-dev redis-server || dpkg --configure -a 2>/dev/null || true

# Local Redis (bound to all interfaces for cross-node access)
redis-server --daemonize yes --supervised no --pidfile /var/run/redis.pid \
    --protected-mode no --bind 0.0.0.0 || {
    nohup redis-server --daemonize no --protected-mode no --bind 0.0.0.0 \
        > /tmp/redis.log 2>&1 &
    sleep 1
}
redis-cli -h localhost -p 6379 PING

# Local workspace + venv
mkdir -p "$LOCAL_WORKSPACE"
if [ ! -d "$LOCAL_WORKSPACE/venv" ]; then
    python3 -m venv --system-site-packages "$LOCAL_WORKSPACE/venv"
fi
source "$LOCAL_WORKSPACE/venv/bin/activate"
pip install --upgrade pip

# Install the pre-built wheel (avoids recompiling NCCL on every node)
pip install "$WHEEL_PATH" \
    -c "$(realpath CrystalLLM/scripts/megatron-pai/constraints.txt)" \
    -r "$(realpath CrystalLLM/scripts/megatron-pai/requirements.txt)"

# Patch system PyTorch
crystalllm-pytorch replace \
    --source-dir "$(realpath CrystalLLM/crystalllm/patches/pytorch-patch)"

# Symlink logs to shared workspace
ln -sfn "$REMOTE_WORKSPACE/logs" "$LOCAL_WORKSPACE/logs"

# Configure the PyTorch allocator for training.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Start the client daemon (RANK must be 0 on the first client, 1 on the second)
crystalllm client stop --crystal-workspace "$LOCAL_WORKSPACE" 2>/dev/null || true
crystalllm client start \
    --crystal-workspace "$LOCAL_WORKSPACE" \
    --server-addr "$SERVER_ADDR" \
    --server-port "$SERVER_PORT"
```

After all client nodes start this script, the coordinator log on the Server
(started in Step 3.4) will show `Registered client rank N from ('<IP>', …)`
lines — that confirms wiring is complete.

Continue to **Step 3.4** below to capture trace + graph data.
### 3.4 Capture mode

`capture` records every collective communication call made by the training
process, plus the topology graph of each process group. The output
(`trace_<R>.log.gz` + `graph/`) is the input to `planning` mode in the next
step.

#### 3.4.b — Launch capture

Run the umbrella driver with the absolute `CRYSTAL_WORKSPACE` exported in Step 3.1:

```bash
SERVER_ADDR="<server-ip>" SERVER_PORT="<server-port>" AVAILABLE_NODES="<num-client-nodes>" \
    bash "$CRYSTAL_WORKSPACE/scripts/megatron-pai/run_batch_step.sh" \
         "<MASTER_IP>" "<MASTER_PORT>" capture
```

Where:

- `SERVER_ADDR`/`SERVER_PORT`: the coordinator's listening address.
- `AVAILABLE_NODES`: number of physical client nodes available.
- `<MASTER_IP>`: the IP of the sandbox (master) client node.
- `<MASTER_PORT>`: the NCCL master port (any free port ≥ 1024, e.g. `40001`).

The coordinator will print `Waiting for clients with ranks: [0, 1, …]` and
then `Registered client rank N from ('<IP>', …)` as each client connects.
Once all clients are registered the capture begins.

#### 3.4.c — First-run note

The first capture run may fail due to a dataset cache race. If this happens,
delete any 0-byte trace files and re-run the same command:

```bash
find "$CRYSTAL_WORKSPACE/logs/$CASE"/capture/trace/ \
    -name 'trace_*.log.gz' -size 0 -delete
```

#### 3.4.d — Verify capture output

When the driver finishes successfully it prints `Test Case 1 completed successfully`.

Reference timing (2 client nodes, shared filesystem): **~25–30 minutes total**, 32 batches of ~50 s each. The first batch
takes longer (~2 min) because of the Megatron dataset-cache build on rank 0;
subsequent batches are dominated by the training iterations (≈45–55 s).

The output tree should contain one trace and one graph file per global rank:

```bash
CASE=Qwen3_A20B_tp_1_pp_4_vpp_0_etp_1_ep_8_mbs_1_gbs_1024_scale_512
TRACE_DIR="$CRYSTAL_WORKSPACE/logs/$CASE"/capture/trace
GRAPH_DIR="$CRYSTAL_WORKSPACE/logs/$CASE"/capture/graph

ls "$TRACE_DIR"/trace_*.log.gz | wc -l       # expect: 512
ls "$GRAPH_DIR"/graph_*.json.gz | wc -l      # expect: 512
find "$TRACE_DIR" -name 'trace_*.log.gz' -size 0   # expect: empty
```

Spot-check a few traces — sizes should be tens of KB (≈70–85 KB) regardless of
rank, since every rank captures the full set of `NEW_GROUP` records plus its
own collective calls over the configured iterations:

```bash
ls -la "$TRACE_DIR"/trace_0.log.gz "$TRACE_DIR"/trace_256.log.gz "$TRACE_DIR"/trace_511.log.gz
zcat "$TRACE_DIR"/trace_0.log.gz | grep -c NEW_GROUP
```

`trace_<R>.log.gz` contains gzipped text records; `graph_<R>.json.gz` is a
gzipped JSON object. `planning` consumes both: topology loading uses
`trace_0.log.gz`, while each assistant subprocess on the worker node reads its own
`trace_<RANK>.log.gz` for the per-rank replay ops and `graph_<rank>.json.gz`
for the topology graph.

Continue to **Step 3.5** below to run `planning`.
### 3.5 Planning mode

`planning` collects the operation durations used by `emulate`. It re-uses the trace + graph
artifacts from Step 3.4 to drive a mocked-NCCL run: one **sandbox** node
performs real Megatron training (just like in `capture`) while the worker
client spawns assistant subprocesses that impersonate the remaining ranks of
the logical cluster. The Crystal-patched `distributed_c10d` on the sandbox
injects `commName` into each `ncclConfig_t`; the customized `libnccl.so` on
the worker consults Redis to fake the collective topology so the sandbox's
NCCL kernels see a fully-populated world. The output is a per-iteration timing
log and runtime profiler files consumed by `emulate`.

#### 3.5.a — Which sandbox nodes get visited

`planning` does not run all 64 logical nodes (that would defeat the point —
sandbox runs are still real training). Instead the driver walks one sandbox
node per *PP rank stride*:

```
pp_rank_stride        = world_size / pipeline_parallel_size
                      = 512 / 4
                      = 128 ranks/stride
                      = 16 nodes/stride
```

For the A20B scale-512 case (PP=4) this picks **4 sandbox nodes** —
node 0, 16, 32, 48 — each representing one pipeline-parallel stage.
The worker client's assistants fill in the remaining ranks for that sandbox node.

#### 3.5.b — Launch

`planning` always uses mock NCCL bootstrap; the `--enable-mock-nccl-bootstrap`
flag is already wired into `run_planning()` inside `run_batch_step.sh`. Just
re-use the same env-var prefix as `capture`, switching the mode to
`planning`:

```bash
SERVER_ADDR="<server-ip>" SERVER_PORT="<server-port>" AVAILABLE_NODES=2 \
    bash "$CRYSTAL_WORKSPACE/scripts/megatron-pai/run_batch_step.sh" \
         "<MASTER_IP>" "<MASTER_PORT>" planning
```

The Server will register the clients, then iterate through the four
sandbox nodes. For each it sends `TRAIN_START` to the master client and
dispatches `ASSISTANT_START` to the worker client. The worker spawns
assistant subprocesses for the ranks required by the current graph, seeds its local Redis with
per-PG topology, and barrier-syncs with the sandbox.

#### 3.5.c — Resume behaviour

`_collect_completed_planning_sandbox_nodes` (in
`crystalllm/src/crystalllm/entry/run.py`) detects which sandbox nodes already finished
by tailing each `node_<N>.log` for the markers `All processes completed
successfully` and `Exit code: 0`. A partially-written
log will be re-run automatically. You can safely Ctrl-C the driver between
sandbox nodes; the next invocation picks up where the previous one stopped.

#### 3.5.d — Expected output

Output lives under `logs/<case>/planning/epoch_0/`:

```text
logs/<case>/planning/epoch_0/
├── node/
│   ├── node_0.log              # sandbox node 0 — full Megatron stdout + per-iter timings
│   ├── node_16.log             # sandbox node 16
│   ├── node_32.log             # sandbox node 32
│   └── node_48.log             # sandbox node 48
├── assistant_for_0/
│   ├── assistant_rank_8.log    # assistant for rank 8 while node 0 was sandbox
│   ├── assistant_rank_9.log
│   ├── …
│   └── assistant_rank_511.log
├── assistant_for_16/
├── assistant_for_32/
└── assistant_for_48/
```

Each `node_<N>.log` ends with:

```text
All processes completed successfully
Exit code: 0
```

Per-iteration timings appear earlier in the same log, e.g.:

```text
 [2026-09-07 07:35:43] iteration       95/      95 | consumed samples: 97280 |
   elapsed time per iteration (ms): <elapsed_ms> | throughput per GPU (TFLOP/s/GPU): <tflops_per_gpu> | …
```

#### 3.5.e — Reference timing and known caveats

**Reference timing (2 client nodes, 8 GPU each, shared filesystem):**

Each sandbox node takes ~12–14 min. Total planning sweep for 4 nodes: ~50 min.
The resume mechanism skips already-completed nodes on restart.

### 3.6 Emulate mode

`emulate` is the full-fidelity replay mode. It combines:

- **trace + graph** from `capture` (Step 3.4) — provides the collective
  sequence and group topology.
- **runtime profiler data** from `planning` (Step 3.5) — provides measured per-operation
  durations so the emulation can pace itself realistically.

Unlike `planning` (which iterates one sandbox node at a time to *produce*
runtime profiler data), `emulate` runs a fixed set of sandbox nodes in a single shot and
enables richer monitoring (`--enable-crystal-timer`,
`--enable-crystal-monitor`). The output is a complete emulation log with
profiling data — the final artifact for performance analysis.

#### 3.6.a — Prerequisites

`emulate` requires both `capture/` and `planning/profiler/crystal/` to exist under the
case directory. If you ran Steps 3.4 and 3.5 successfully, both are in place:

```bash
CASE=Qwen3_A20B_tp_1_pp_4_vpp_0_etp_1_ep_8_mbs_1_gbs_1024_scale_512
ls "$CRYSTAL_WORKSPACE/logs/$CASE"/capture/trace/trace_0.log.gz  # from Step 3.4
ls "$CRYSTAL_WORKSPACE/logs/$CASE"/planning/profiler/crystal/runtime_profiler_0.json.gz   # from Step 3.5
```

#### 3.6.b — Launch

```bash
SERVER_ADDR="<server-ip>" SERVER_PORT="<server-port>" AVAILABLE_NODES=2 \
    bash "$CRYSTAL_WORKSPACE/scripts/megatron-pai/run_batch_step.sh" \
         "<MASTER_IP>" "<MASTER_PORT>" emulate
```

#### 3.6.d — Expected output

Output lives under `logs/<case>/emulate/`:

```text
logs/<case>/emulate/
├── node/
│   └── node_0.log          # sandbox node 0 — Megatron stdout + per-iter timings
└── assistant_for_0/
    ├── assistant_rank_8.log
    ├── …
    └── assistant_rank_511.log
```

Success is indicated by `All processes completed successfully` + `Exit code: 0`
in `node_0.log`.

#### 3.6.e — Reference timing

**Reference timing (2 client nodes, 8 GPU each, shared filesystem):**

Emulate completes in ~12 min for a single sandbox node.

The emulate run produces a log of approximately 1 MB for `node_0.log`.
The number of assistant logs under `assistant_for_0/` depends on the
actual graph and rank placement.

## 4. Megatron-Lite reproduction

This example runs the same Qwen3 A20B workload with 512 logical ranks as the
Megatron/Pai example, using the Megatron-Lite backend. It uses TP1/PP4/EP8/ETP1,
CP1/VPP1, sequence length 4096, microbatch size 1, and global batch size 1024.
The native `mlite` backend runs with non-THD inputs and the distributed
optimizer. Two homogeneous eight-GPU client nodes are required; the coordinator
may share the first node. The hardware and base image requirements in §1 apply.

### 4.1 Source and workspace

Run from the parent directory containing `CrystalLLM/`. The Lite workspace
has its own `venv/` and `logs/`, separate from the Megatron/Pai workspace in §3.

```bash
set -euo pipefail
export ARTIFACT_ROOT="$(realpath CrystalLLM)"
mkdir -p CrystalLLM_workspace/megatron_lite/logs
export CRYSTAL_WORKSPACE="$(realpath CrystalLLM_workspace/megatron_lite)"

git clone https://github.com/NVIDIA/Megatron-LM.git "$CRYSTAL_WORKSPACE/Megatron-LM"
git -C "$CRYSTAL_WORKSPACE/Megatron-LM" checkout 53f49769013fb487cbd9ca693f96ee9fd7a9bf4a
git -C "$CRYSTAL_WORKSPACE/Megatron-LM" apply \
    "$ARTIFACT_ROOT/crystalllm/patches/megatron-lite-patch/megatron_lite.patch"
```

The case parameters are in
`scripts/megatron-lite/configs/512/A20B_tp_1_pp_4_vpp_1_cp_1_etp_1_ep_8_mbs_1_gbs_1024.json`;
the model configuration ships under `scripts/megatron-lite/configs/qwen3_a20b/`.
The Lite benchmark initializes model weights locally and generates synthetic
token batches; this example needs no model checkpoint, tokenizer download, or
MMAP `.bin/.idx` dataset.

### 4.2 Coordinator and wheel

Install the system packages and configure Redis as in §3.2.b. Build the wheel
once on a GPU host using the NGC 25.06 PyTorch environment. Use a clean build
with no previous CrystalLLM `build/`, `dist/`, `src/crystalllm/nccl/lib/`, or
NCCL `build/` outputs.

```bash
set -euo pipefail
ln -sfn ../../nccl "$ARTIFACT_ROOT/crystalllm/vendor/nccl"
python -m venv --system-site-packages "$CRYSTAL_WORKSPACE/venv"
source "$CRYSTAL_WORKSPACE/venv/bin/activate"
unset PYTHONPATH LD_PRELOAD
python -m pip install 'setuptools>=70.1,<80' wheel pybind11 'numpy<2' redis

cd "$ARTIFACT_ROOT/crystalllm"
python setup.py bdist_wheel
WHEEL_PATH="$(ls "$ARTIFACT_ROOT"/crystalllm/dist/crystalllm-*.whl)"
python -m pip install --no-deps --force-reinstall "$WHEEL_PATH"
```

Use this same wheel on both clients. Lite requires Transformer Engine 2.14.0
and NVRx 0.6.0; the dependency files below preserve the image's PyTorch build
and NumPy 1.x ABI.

### 4.3 Client installation and startup

On each client, set `ARTIFACT_ROOT` and `CRYSTAL_WORKSPACE` to the shared
absolute paths from §4.1. Set `SERVER_ADDR` to the coordinator address and
`RANK=0` on the first client, `RANK=1` on the second. SSH and Redis must be
available as described in §3.3.

```bash
set -euo pipefail
: "${ARTIFACT_ROOT:?}" "${CRYSTAL_WORKSPACE:?}" "${SERVER_ADDR:?}" "${RANK:?}"
export SERVER_ADDR RANK
export SERVER_PORT="${SERVER_PORT:-8891}"
LOCAL_WORKSPACE=/root/CrystalLLM_megatron_lite
WHEEL_PATH="$(ls "$ARTIFACT_ROOT"/crystalllm/dist/crystalllm-*.whl)"

mkdir -p "$LOCAL_WORKSPACE"
python -m venv --system-site-packages "$LOCAL_WORKSPACE/venv"
source "$LOCAL_WORKSPACE/venv/bin/activate"
unset PYTHONPATH LD_PRELOAD PYTORCH_ALLOC_CONF
python -m pip install 'setuptools>=70.1,<80' wheel pybind11
export NVTE_FRAMEWORK=pytorch MAX_JOBS=8 CUDA_PATH=/usr/local/cuda
python -m pip install --no-build-isolation \
    -c "$ARTIFACT_ROOT/scripts/megatron-lite/constraints.txt" \
    -r "$ARTIFACT_ROOT/scripts/megatron-lite/requirements.txt"
python -m pip install --no-deps --force-reinstall "$WHEEL_PATH"
crystalllm-pytorch replace \
    --source-dir "$ARTIFACT_ROOT/crystalllm/patches/pytorch-patch"

ln -sfn "$CRYSTAL_WORKSPACE/logs" "$LOCAL_WORKSPACE/logs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

crystalllm client stop --crystal-workspace "$LOCAL_WORKSPACE" 2>/dev/null || true
crystalllm client start \
    --crystal-workspace "$LOCAL_WORKSPACE" \
    --server-addr "$SERVER_ADDR" --server-port "$SERVER_PORT"
```

Both clients must load the installed wheel and the reconstructed Lite source.
The training entry sets the Lite source path; the environment JSON contains no
source-workspace paths or allocator overrides. The network settings in
`scripts/megatron-lite/envs.json` should match the client interfaces.

### 4.4 Capture, Planning, and Emulate

On the coordinator, activate the Lite workspace environment and run the three
stages in order. `SERVER_ADDR` is the coordinator address used by both clients.

```bash
set -euo pipefail
source "$CRYSTAL_WORKSPACE/venv/bin/activate"
export SERVER_PORT="${SERVER_PORT:-8891}"

python "$ARTIFACT_ROOT/scripts/megatron-lite/run_megatron_lite.py" capture \
    --server-addr "$SERVER_ADDR" --server-port "$SERVER_PORT" --master-port 40001
python "$ARTIFACT_ROOT/scripts/megatron-lite/run_megatron_lite.py" planning \
    --server-addr "$SERVER_ADDR" --server-port "$SERVER_PORT" --master-port 40002
python "$ARTIFACT_ROOT/scripts/megatron-lite/run_megatron_lite.py" emulate \
    --server-addr "$SERVER_ADDR" --server-port "$SERVER_PORT" --master-port 40003
```

The driver defaults to two available nodes and runs 95 iterations per stage,
with analysis over steps 81–90. Use the same `--train-iters N` value across all
three stages to change the number of iterations before the analysis window.
Add `--dry-run` to inspect a command without
launching it. Planning visits sandbox nodes 0, 16, 32, and 48; Emulate uses
sandbox node 0. Both select the Lite tensor providers included in the artifact.

Outputs are written under:

```text
CrystalLLM_workspace/megatron_lite/logs/
└── Qwen3_A20B_tp_1_pp_4_vpp_1_cp_1_etp_1_ep_8_mbs_1_gbs_1024_scale_512/
    ├── capture/
    ├── planning/
    └── emulate/
```

Each stage records `coordinator.log` and `exit_code.txt`. A nonzero exit stops
that invocation; inspect its logs before continuing. Re-running the same stage
uses Crystal's existing resume behavior in place. Use fresh logs when changing
source, model, or training configuration.

### 4.5 Check the results

- **Capture:** all 64 logical nodes complete; `capture/trace/` contains 512
  `trace_<rank>.log.gz` files and `capture/graph/` contains 512
  `graph_<rank>.json.gz` files. Each graph contains the ten analysis steps.
- **Planning:** nodes 0, 16, 32, and 48 complete; `planning/profiler/crystal/`
  contains 32 `runtime_profiler_<rank>.json.gz` files covering the same window.
- **Emulate:** the sandbox completes all 95 steps, and its assistant processes
  finish replay and cleanup. Check the worker output under `emulate/node/`
  and the corresponding assistant logs.

Require successful worker completion as well as `exit_code.txt` equal to `0`.
Benchmark steps are reported as `[MLITE_BENCH_STEP]`; measured latency and
throughput depend on the allocated hardware.
