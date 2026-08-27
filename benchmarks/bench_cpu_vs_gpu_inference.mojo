"""Is CPU viable for TD-MPC2 inference on Apple? Trunk-level CPU vs GPU.

    pixi run -e apple mojo run -I . benchmarks/bench_cpu_vs_gpu_inference.mojo

The GPU MPC frame is LAUNCH-CHAIN-bound, not FLOP-bound: ~500 dependent
kernels x ~21 us of Metal dispatch. CPU pays no dispatch at all, so the
question is throughput at these shapes, not peak TFLOPS.

A full plan is 15.8 GFLOP and the GPU does it in 53.6 ms = 295 GFLOPS
effective, which is the bar CPU has to clear.

## Measured, M1 Pro, QUIET machine, 2026-08-12

    trunk        CPU              GPU      ratio
    Dynamics    1822 us (232 GF)  695 us    2.6x
    Reward      1509 us (205 GF)  671 us    2.3x
    QNet        1915 us (162 GF)  671 us    2.9x
    Policy      1523 us (186 GF)  618 us    2.5x
    Policy B=1    18.6 us         194 us    CPU 10.4x FASTER   <-- !

    projected CPU plan  79.2 ms   vs  GPU 53.6 ms   -> CPU 1.48x slower
    matmul-only floor  ~27.7 ms                     -> CPU 1.9x FASTER

So: a CPU MPPI is NOT worth building today, but the ceiling is real. The raw
GEMMs already run at 500-900 GFLOPS (`max_matmul` and Accelerate are within
~25% of each other at B=268); what costs the rest is non-GEMM work.

⚠ B=1 IS A DIFFERENT ANSWER FROM B=268. `max_matmul[target="cpu"]` collapses
at M=1 — 300 us vs cblas's 7.8 us, 38x — which is why `Linear.forward` now
calls `cblas_sgemm` directly on Apple fp32 (as `vjp` always did). After that,
single-env acting is 10x FASTER on CPU than on GPU. Any per-env inference path
(a viewer's policy-prior mode, a CPU eval loop) belongs on CPU.

⚠ CPU numbers are the ones contention hurts. Run this idle; a concurrent build
roughly halves the CPU side and leaves the GPU side alone, which silently
biases the whole comparison.
"""


from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor, TensorImpl
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.deep_agents.tdmpc2.nets import (
    TDMPC2Dynamics, TDMPC2Reward, TDMPC2QNet, TDMPC2Policy,
)

comptime LATENT = 512
comptime MLP = 512
comptime BINS = 101
comptime SN = 8
comptime ACTD = 6
comptime ZA = LATENT + ACTD


def time_cpu[
    M: Module, B: Int, IN: Int, OUT: Int
](label: String, reps: Int) raises -> Float64:
    var m = M.make["cpu", INIT=Kaiming]()
    var x = TensorImpl[M.ACT_DT].alloc(B * IN)
    var y = TensorImpl[M.ACT_DT].alloc(B * OUT)
    for i in range(B * IN):
        x.data[i] = Scalar[M.ACT_DT](0.01) * Scalar[M.ACT_DT]((i % 37) - 18)
    for _ in range(3):
        m.forward["cpu", B](TensorRefs[M.ARITY, ADT=M.ACT_DT](x), y, None)
    var t0 = perf_counter_ns()
    for _ in range(reps):
        m.forward["cpu", B](TensorRefs[M.ARITY, ADT=M.ACT_DT](x), y, None)
    var t1 = perf_counter_ns()
    return Float64(t1 - t0) / 1000.0 / Float64(reps)


def time_gpu[
    M: Module, B: Int, IN: Int, OUT: Int
](ctx: DeviceContext, reps: Int) raises -> Float64:
    var m = M.make["gpu", INIT=Kaiming](ctx=ctx)
    var x = TensorImpl[M.ACT_DT].alloc_gpu(ctx, B * IN)
    var y = TensorImpl[M.ACT_DT].alloc_gpu(ctx, B * OUT)
    for _ in range(10):
        m.forward["gpu", B](
            TensorRefs[M.ARITY, ADT=M.ACT_DT](x), y, Optional(ctx)
        )
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(reps):
        m.forward["gpu", B](
            TensorRefs[M.ARITY, ADT=M.ACT_DT](x), y, Optional(ctx)
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    return Float64(t1 - t0) / 1000.0 / Float64(reps)


def pair[
    M: Module, B: Int, IN: Int, OUT: Int
](ctx: DeviceContext, label: String, gflop: Float64, reps: Int) raises:
    var c = time_cpu[M, B, IN, OUT](label, reps)
    var g = time_gpu[M, B, IN, OUT](ctx, reps)
    print(
        "   ", label, " B=", B, ":  cpu ", c, " us (", gflop * 1e6 / c,
        " GFLOPS)   gpu ", g, " us   -> cpu/gpu ", c / g, "x",
        sep="",
    )


def main() raises:
    var ctx = DeviceContext()
    print("TD-MPC2 trunks, CPU vs GPU —", ctx.name())
    print()
    print("== MPPI batch (BATCH_TOTAL = 268) ==")
    # GFLOP per call, 2*M*K*N summed over the trunk's layers.
    pair[TDMPC2Dynamics[LATENT, ACTD, MLP, SN], 268, ZA, LATENT](
        ctx, "Dynamics", 0.422, 30
    )
    pair[TDMPC2Reward[LATENT, ACTD, MLP, BINS], 268, ZA, BINS](
        ctx, "Reward  ", 0.310, 30
    )
    pair[TDMPC2QNet[LATENT, ACTD, MLP, BINS], 268, ZA, BINS](
        ctx, "QNet    ", 0.310, 30
    )
    pair[TDMPC2Policy[LATENT, ACTD, MLP], 268, LATENT, 2 * ACTD](
        ctx, "Policy  ", 0.283, 30
    )
    print()
    print("== single-env acting batch (B = 1, the `prior` path) ==")
    pair[TDMPC2Policy[LATENT, ACTD, MLP], 1, LATENT, 2 * ACTD](
        ctx, "Policy  ", 0.00106, 200
    )
    print()
    print("A full plan is 15.8 GFLOP; the GPU does it in ~70.6 ms.")
    print("CPU wins outright if it sustains >~225 GFLOPS at these shapes.")
