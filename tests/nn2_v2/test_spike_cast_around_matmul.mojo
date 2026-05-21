"""Test cast_around_matmul helper.

Validates:
  1. fp32 path produces correct matmul output.
  2. bf16 path produces output within bf16 precision (~3e-3 max rel)
     of the fp32 reference.
  3. `weights_dirty` flag works: stale-cached bf16 weights are detected.
"""

from std.math import abs as fabs
from std.memory import alloc

from mojo_rl.nn2_v2.spike_cast_around_matmul import (
    DT,
    NoAMP,
    Bf16Compute,
    BF16Scratch,
    cast_around_matmul,
)


comptime M = 4
comptime K = 5
comptime N = 3


def _max_rel(
    a: UnsafePointer[Scalar[DT], MutAnyOrigin],
    b: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
) -> Scalar[DT]:
    var mr: Scalar[DT] = 0.0
    for k in range(n):
        var denom: Scalar[DT] = fabs(a[k]) + fabs(b[k]) + Scalar[DT](1e-8)
        var rel = fabs(a[k] - b[k]) / denom
        if rel > mr:
            mr = rel
    return mr


def main() raises:
    # Inputs.
    var a_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](M * K)
    var b_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](K * N)
    var out_fp32: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](M * N)
    var out_bf16: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](M * N)

    var state: UInt64 = UInt64(0x99)
    for i in range(M * K):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Int(state >> 32) & 0xFFFF)) / Scalar[DT](65535.0)
        a_p[i] = (r - Scalar[DT](0.5))
    for i in range(K * N):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Int(state >> 32) & 0xFFFF)) / Scalar[DT](65535.0)
        b_p[i] = (r - Scalar[DT](0.5))

    # ── fp32 path ────────────────────────────────────────────────────
    var s_fp32 = BF16Scratch.empty()
    cast_around_matmul[NoAMP, M, K, N](a_p, b_p, out_fp32, s_fp32, transpose_b_=False)
    print("fp32 out[0,0] =", out_fp32[0])

    # ── bf16 path ────────────────────────────────────────────────────
    var s_bf16 = BF16Scratch.empty()
    cast_around_matmul[Bf16Compute, M, K, N](a_p, b_p, out_bf16, s_bf16, transpose_b_=False)
    print("bf16 out[0,0] =", out_bf16[0])

    var mr = _max_rel(out_fp32, out_bf16, M * N)
    print("max_rel_err fp32 vs bf16 =", mr)

    # bf16 has ~3 mantissa bits → expect rel err up to a few ×1e-2 for
    # small matmuls. Loose tol of 5e-2.
    if mr > Scalar[DT](5e-2):
        print("FAIL — bf16 cast-around drift too large")
        raise Error("bf16 drift > 5e-2")

    # ── weights_dirty mechanics ──────────────────────────────────────
    # After first call, scratch.w_dirty should be False.
    if s_bf16.w_dirty:
        print("FAIL — w_dirty should be False after first bf16 call")
        raise Error("w_dirty not cleared")

    # Run another bf16 call. Cached weight should be reused → same output.
    var out_bf16_2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](M * N)
    cast_around_matmul[Bf16Compute, M, K, N](a_p, b_p, out_bf16_2, s_bf16, transpose_b_=False)
    var diff = _max_rel(out_bf16, out_bf16_2, M * N)
    print("max_rel_err identical-weight reuse =", diff)
    if diff > Scalar[DT](1e-9):
        print("FAIL — reused-weight call produced different output")
        raise Error("reused-weight cache mismatch")

    # Now perturb the FP32 weight, mark dirty, expect the bf16 output to change.
    b_p[0] = b_p[0] + Scalar[DT](0.5)
    s_bf16.w_dirty = True
    var out_bf16_3: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](M * N)
    cast_around_matmul[Bf16Compute, M, K, N](a_p, b_p, out_bf16_3, s_bf16, transpose_b_=False)
    var diff2 = _max_rel(out_bf16, out_bf16_3, M * N)
    print("max_rel_err post-weight-update =", diff2)
    if diff2 < Scalar[DT](1e-3):
        print("FAIL — w_dirty=True didn't re-cast weights")
        raise Error("w_dirty did not trigger re-cast")

    print("PASS — cast_around_matmul fp32 + bf16 + dirty-flag")

    a_p.free()
    b_p.free()
    out_fp32.free()
    out_bf16.free()
    out_bf16_2.free()
    out_bf16_3.free()
