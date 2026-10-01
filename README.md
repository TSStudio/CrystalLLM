# CrystalLLM

**Faithful emulation of large-scale LLM training on a small GPU cluster.**

CrystalLLM lets researchers and engineers study a training job at its intended
logical scale while executing only selected ranks on physical GPUs. It captures
the job's execution and communication dependencies, runs the ranks of interest
in a **sandbox**, and replays the remaining ranks on **assistant** GPUs. The
sandbox executes the real training stack and communicates with virtual peers,
so it can be inspected with standard profiling and debugging tools.

This repository provides the CrystalLLM runtime, customized NCCL,
training-stack patches, and reproduction scripts.

[Paper](https://doi.org/10.1145/3830418.3843852) ·
[Artifact guide](README.artifact.md) ·
[Changelog](CHANGELOG.md) ·
[Issues](https://github.com/AlibabaResearch/CrystalLLM/issues)

## What you can do with CrystalLLM

- **Evaluate training configurations:** compare parallelism, recomputation, and
  optimizer-offloading choices using the target job's execution structure.
- **Inspect memory behavior:** observe allocation, fragmentation, and
  out-of-memory behavior on the sandbox ranks.
- **Profile and debug the training stack:** use PyTorch Profiler, NVIDIA Nsight
  Systems, and other tools in an isolated, repeatable environment.
- **Investigate performance anomalies:** compare a sandbox node's behavior
  against the job's captured baseline.

## How it works

The implementation exposes three stages:

| Stage | Purpose | Output |
| --- | --- | --- |
| `capture` | Multiplex logical ranks onto the available GPUs and record operations and dependencies. | Per-rank traces and execution graphs. |
| `planning` | Execute selected slices with real sandbox ranks and virtual peers to collect operation timings. | Runtime profiles used by replay. |
| `emulate` | Run the ranks of interest with timed replay of their virtual peers. | Training logs, timings, and profiling data. |

A **coordinator** orchestrates these stages. Sandbox GPUs run training;
assistant GPUs replay virtual ranks using the customized NCCL implementation.
For the bundled 512-rank examples, the setup uses two homogeneous eight-GPU
client nodes; the coordinator can share a client host. Larger tightly coupled
parallelism groups can require additional sandbox and assistant resources.

## Getting started

```bash
git clone https://github.com/AlibabaResearch/CrystalLLM.git
cd CrystalLLM
```

Follow the **[artifact guide](README.artifact.md)** for the complete environment
setup, NCCL build, training-stack patches, client startup, and
`capture → planning → emulate` workflow. The guide uses the clone directory
`CrystalLLM/` and creates a sibling workspace for runtime outputs.

## Scope and assumptions

CrystalLLM targets training workloads whose operation and communication
structure is repeatable across runs. Graphs must be recaptured when that
structure changes. The current implementation supports PyTorch/NCCL workflows
through the integrations shipped here; fused computation–communication kernels
are outside the current prototype's coverage.

Emulation is intended for inspecting selected ranks and evaluating system
behavior over a limited execution window. Workloads with data-dependent changes
to communication structure, studies of training convergence or numerical drift,
and experiments requiring traffic or computation from every real rank need
additional validation or full-scale execution. Replay represents the captured
baseline and does not generate transient stragglers or rare failures.

## Citation

If CrystalLLM supports your research, please cite:

```bibtex
@inproceedings{crystalllm2026,
  author    = {Shaoke Xi and ChonLam Lao and Boyi Jia and Jiaqi Gao and
               Zhipeng Zhang and Jiamin Cao and Brian Sutioso and Erci Xu and
               Minlan Yu and Kui Ren and Yong Li and Zhengping Qian and
               Ennan Zhai and Jingren Zhou},
  title     = {A Few GPUs, A Whole Lotta Scale: Faithful {LLM} Training
               Emulation with {CrystalLLM}},
  booktitle = {Proceedings of the Symposium on Operating Systems Principles},
  year      = {2026},
  doi       = {10.1145/3830418.3843852}
}
```

## Contact

For research questions and collaboration, contact Shaoke Xi ([@ShaokeXi](https://github.com/ShaokeXi), [shaoke.xsk@alibaba-inc.com](mailto:shaoke.xsk@alibaba-inc.com)), ChonLam Lao ([@laochonlam](https://github.com/laochonlam), [chonlam.lao@alibaba-inc.com](mailto:chonlam.lao@alibaba-inc.com)), Boyi Jia ([@ap0stader](https://github.com/ap0stader), [jiaboyi.jby@alibaba-inc.com](mailto:jiaboyi.jby@alibaba-inc.com)), or Jiaqi Gao ([@Gaojiaqi](https://github.com/Gaojiaqi), [jiaqi.g@alibaba-inc.com](mailto:jiaqi.g@alibaba-inc.com)).

For questions, bug reports, and feature discussions, please open a
[GitHub issue](https://github.com/AlibabaResearch/CrystalLLM/issues). For
reproduction issues, include the backend, GPU and network configuration,
software versions, reproduction steps, and relevant logs.

## Contributing

Contributions to documentation, reproducibility, and training-stack integrations
are welcome. Please open an issue to discuss substantial changes before
submitting a pull request.

## License

CrystalLLM-authored code is licensed under the [MIT License](LICENSE).
