"""Bench `ReLUOp.eval/.vjp` end-to-end after SIMD swap.

Compares the new SIMD-based ReLUOp to a local scalar reference. Validates that:
  1. ReLUOp matches scalar bit-exact (ReLU has no FP reorder noise).
  2. The swap delivers the ~4× speedup the LayoutTensor microbench predicted.

Run:
    pixi run mojo run -I . benchmarks/benchmark_relu_op_inplace.mojo
"""

from std.memory import alloc
from std.random import seed, random_float64
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff.primitives import ReLUOp


comptime BATCH = 256
comptime DIM = 256


def main() raises:
    print("=" * 80)
    print("ReLUOp end-to-end CPU bench (post-SIMD swap)")
    print("  BATCH=", BATCH, "  DIM=", DIM)
    print("=" * 80)

    seed(42)
    var inp_buf = alloc[Scalar[dtype]](BATCH * DIM)
    var out_buf = alloc[Scalar[dtype]](BATCH * DIM)
    var cache_buf = alloc[Scalar[dtype]](BATCH * DIM)
    var go_buf = alloc[Scalar[dtype]](BATCH * DIM)
    var gi_buf = alloc[Scalar[dtype]](BATCH * DIM)
    var ref_out = alloc[Scalar[dtype]](BATCH * DIM)
    var ref_gi = alloc[Scalar[dtype]](BATCH * DIM)
    for i in range(BATCH * DIM):
        inp_buf[i] = Scalar[dtype](random_float64(-1.0, 1.0))
        go_buf[i] = Scalar[dtype](random_float64(-1.0, 1.0))

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](inp_buf)
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](out_buf)
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](cache_buf)
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](go_buf)
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](gi_buf)
    var p_buf = alloc[Scalar[dtype]](1)
    var p_t = LayoutTensor[dtype, Layout.row_major(0), MutAnyOrigin](p_buf)

    # Build scalar reference
    for i in range(BATCH * DIM):
        var v = inp_buf[i]
        ref_out[i] = v if v > 0 else 0
        ref_gi[i] = go_buf[i] if v > 0 else 0

    # Run new SIMD version
    ReLUOp[DIM].eval[BATCH](inp_t, out_t, p_t, cache_t)
    ReLUOp[DIM].vjp[BATCH](go_t, gi_t, p_t, cache_t, p_t)

    var err_fwd: Float64 = 0
    var err_bwd: Float64 = 0
    for i in range(BATCH * DIM):
        var df = Float64(out_buf[i]) - Float64(ref_out[i])
        if df < 0:
            df = -df
        if df > err_fwd:
            err_fwd = df
        var db = Float64(gi_buf[i]) - Float64(ref_gi[i])
        if db < 0:
            db = -db
        if db > err_bwd:
            err_bwd = db

    print(
        "Correctness: fwd_max_diff=",
        err_fwd,
        " bwd_max_diff=",
        err_bwd,
        " (expect 0.0 — ReLU is bit-exact)",
    )

    # Bench
    var iters = 5000
    var t0 = perf_counter_ns()
    for _ in range(iters):
        ReLUOp[DIM].eval[BATCH](inp_t, out_t, p_t, cache_t)
    var t1 = perf_counter_ns()
    var t_fwd = Float64(t1 - t0) / Float64(iters) / 1000.0

    t0 = perf_counter_ns()
    for _ in range(iters):
        ReLUOp[DIM].vjp[BATCH](go_t, gi_t, p_t, cache_t, p_t)
    t1 = perf_counter_ns()
    var t_bwd = Float64(t1 - t0) / Float64(iters) / 1000.0

    print("\nProduction ReLUOp[", DIM, "]:")
    print("  forward =", Int(t_fwd * 100) / 100.0, "us/call")
    print("  vjp     =", Int(t_bwd * 100) / 100.0, "us/call")
    print(
        "  baseline (pre-SIMD) was ~23us/call for fwd — see"
        " benchmark_vectorize_layouttensor.mojo"
    )
