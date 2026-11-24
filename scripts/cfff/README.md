# CFFF capture experiment

`capture_smoke.py` is a synthetic MLP used to check CrystalLLM capture. It is
not Qwen, SFT, or a performance benchmark. Its collective operations are
intercepted during capture; it does not synchronize real gradients.

`torch251-capture.patch` is an **experimental capture-only** adaptation for
the PyTorch 2.5.1 source file with SHA-256
`8bb5c7fe4a752781ae427282259aa67cb4a28b0ddc271278d0d903c5fc7de1ca`.
It has only been exercised with this simple workload. In particular it does
not implement the project's coalesced-communication, planning, or emulation
patches, and must not be treated as a complete PyTorch integration. Do not
apply it to the system PyTorch installation. Use a separate per-node overlay
and verify the original file hash before applying it.

On two CFFF instances with two A100 GPUs each, the coordinator captured an
8-node × 2-GPU logical layout in four batches, producing 16 trace files and
16 graph files. Completion of capture alone does not demonstrate replay or
the accuracy of simulated performance.
