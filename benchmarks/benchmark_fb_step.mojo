"""Where does an FB train step actually spend its time?

Written to answer "FB feels slow next to SAC — is that normal, or is there a
bottleneck?" with a number instead of an opinion. Production dims from
`fb_train_gpu.mojo`. Times a full `train_step` against the 8 `PairwiseDot`
[BATCH,BATCH,D] kernels it issues per step (4 forward: two twin-F targets +
measure + ortho; 4 backward: dA/dC for each of measure and ortho).

`pairwise_dot.mojo` says of its naive kernels: "at the sizes involved the
arithmetic is not the cost. Revisit only with a measurement, and only on
NVIDIA." First measurement (Apple M-series, so indicative only):

    full train_step          56.6 ms
    8 PairwiseDot kernels    13.1 ms   -> 23% of the step
    one PairwiseDot forward   2.39 ms  -> 112 GFLOP/s

⚠ That Apple reading UNDERSTATED it badly. The NVIDIA profile put PairwiseDot at
52.6% of GPU kernel time, not 23% — the Apple/NVIDIA inversion this project has
hit before, 2.3x here. The forward has since moved to
`batched_matmul[transpose_b=True]` (the entry point `attention.mojo` uses for
QK^T), and the same Apple benchmark now reads:

    full train_step          37.8 ms   (was 56.6)
    8 PairwiseDot kernels     5.4 ms   -> 14% of the step (was 23%)
    one PairwiseDot forward   0.47 ms  -> 573 GFLOP/s (was 112)

The remaining bulk is the nets, the optimizer, and — the thing that actually
differed from SAC — uncaptured kernel launches plus per-step device allocation
(`cuMemAlloc`/`cuMemFree` were 81% of CUDA API time until the trainer stopped
allocating vjp sinks every step).

⚠ Absolute numbers here do not transfer between Apple and NVIDIA (this project
has measured the two INVERT on conv kernels). Treat the 23/77 SPLIT as
structural and re-run on the target before optimising anything.

Run:
    pixi run -e nvidia mojo run -I . benchmarks/benchmark_fb_step.mojo
    pixi run -e apple  mojo run -I . benchmarks/benchmark_fb_step.mojo
"""

from max.gpu.host import DeviceContext
from std.random import random_float64, seed
from std.time import perf_counter_ns

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.initializer import Deterministic, Xavier
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU, Tanh
from mojo_rl.nn.primitives.layer_norm_no_affine import LayerNormNoAffine
from mojo_rl.nn.primitives.pairwise_dot import PairwiseDot
from mojo_rl.deep_agents.fb.trainer import FBTrainer


comptime NQ = 9
comptime NV = 9
comptime OBS = NQ + NV
comptime NACT = 6
comptime D = 128
comptime BATCH = 1024
comptime HID = 1024

comptime F_IN = OBS + NACT + D
comptime A_IN = OBS + D

comptime FNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, D]]
comptime BNet = Sequential[
    Linear[OBS, 256], ReLU[256], Linear[256, D], LayerNormNoAffine[D]
]
comptime ANet = Sequential[
    Linear[A_IN, HID], ReLU[HID], Linear[HID, NACT], Tanh[NACT]
]
comptime Trainer = FBTrainer[FNet, BNet, ANet, OBS, NACT, D, BATCH, "gpu"]

comptime WARMUP = 5
comptime ITERS = 30


def _rt(n: Int, ctx: DeviceContext) raises -> Tensor:
    var t = Tensor.alloc(n)
    for i in range(n):
        t.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    t.upload(ctx)
    return t^


def main() raises:
    seed(1234)
    var ctx = DeviceContext()
    print("=" * 66)
    print("FB train-step cost breakdown  (OBS", OBS, " D", D, " BATCH", BATCH,
          " HID", HID, ")")
    print("=" * 66)

    var t = Trainer.make[Xavier](lr=1e-4, ctx=Optional(ctx), max_grad_norm=1.0,
                                 bc_weight=1.0)
    var s = _rt(BATCH * OBS, ctx)
    var a = _rt(BATCH * NACT, ctx)
    var sn = _rt(BATCH * OBS, ctx)
    var sp = _rt(BATCH * OBS, ctx)
    var z = _rt(BATCH * D, ctx)
    t.load_batch(s, a, sn, sp, z)

    # ── full train step ──────────────────────────────────────────────────
    for _ in range(WARMUP):
        _ = t.train_step(want_loss=False)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        _ = t.train_step(want_loss=False)
    ctx.synchronize()
    var step_ms = Float64(perf_counter_ns() - t0) / 1e6 / Float64(ITERS)

    # ── the 8 PairwiseDot kernels a step issues ──────────────────────────
    var pd = PairwiseDot[D, BATCH].make["gpu", Deterministic](Optional(ctx))
    var ins = TensorPack[2]()
    ins[0].ensure(BATCH * D)
    ins[1].ensure(BATCH * D)
    for i in range(BATCH * D):
        ins[0].data[i] = Scalar[DT](random_float64())
        ins[1].data[i] = Scalar[DT](random_float64())
    ins[0].upload(ctx)
    ins[1].upload(ctx)
    var m = Tensor()
    m.ensure(BATCH * BATCH)
    m.ensure_gpu(ctx, BATCH * BATCH)
    var go = Tensor()
    go.ensure(BATCH * BATCH)
    go.ensure_gpu(ctx, BATCH * BATCH)
    var grads = TensorPack[2]()
    grads[0].ensure(BATCH * D)
    grads[1].ensure(BATCH * D)
    grads[0].ensure_gpu(ctx, BATCH * D)
    grads[1].ensure_gpu(ctx, BATCH * D)

    for _ in range(WARMUP):
        pd.forward["gpu", BATCH](
            TensorRefs[2, MutAnyOrigin](ins[0], ins[1]), m, Optional(ctx)
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    for _ in range(ITERS):
        # 4 forwards per step: two twin-F targets, measure, ortho.
        for _f in range(4):
            pd.forward["gpu", BATCH](
                TensorRefs[2, MutAnyOrigin](ins[0], ins[1]), m, Optional(ctx)
            )
        # 4 backward kernels: dA + dC for measure and for ortho.
        for _b in range(2):
            pd.vjp["gpu", BATCH](
                TensorRefs[2, MutAnyOrigin](ins[0], ins[1]), go,
                TensorRefs[2, MutAnyOrigin](grads[0], grads[1]), Optional(ctx),
            )
    ctx.synchronize()
    var pd_ms = Float64(perf_counter_ns() - t1) / 1e6 / Float64(ITERS)

    # ── one isolated forward, for the per-kernel number ──────────────────
    ctx.synchronize()
    var t2 = perf_counter_ns()
    for _ in range(ITERS):
        pd.forward["gpu", BATCH](
            TensorRefs[2, MutAnyOrigin](ins[0], ins[1]), m, Optional(ctx)
        )
    ctx.synchronize()
    var fwd_ms = Float64(perf_counter_ns() - t2) / 1e6 / Float64(ITERS)

    var macs = Float64(BATCH) * Float64(BATCH) * Float64(D)
    print("")
    print("  full train_step         ", step_ms, "ms")
    print("  8 PairwiseDot kernels   ", pd_ms, "ms  (",
          pd_ms * 100.0 / step_ms, "% of the step )")
    print("  one PairwiseDot forward ", fwd_ms, "ms  =",
          macs * 2.0 / (fwd_ms * 1e-3) / 1e9, "GFLOP/s")
    print("")
    print("  per-step PairwiseDot work:", 8.0 * macs / 1e9, "GMAC")
    print("  steps/s at this rate     :", 1000.0 / step_ms)
    print("  2 M steps would take     :", 2e6 * step_ms / 1e3 / 3600.0, "hours")
    print("=" * 66)
