"""Probe: does `linalg.matmul[target="gpu"]` accept bf16 TileTensor inputs?

Critical for Phase 3 — without this, the cast-around-matmul plan is dead
and we'd need a custom bf16 MMA kernel.
"""

from std.math import abs as fabs
from max.gpu.host import DeviceContext
from std.testing import assert_true
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul


comptime FP32 = DType.float32
comptime BF16 = DType.bfloat16


def test_bf16_matmul() raises:
    comptime M = 4
    comptime K = 8
    comptime N = 4

    var ctx = DeviceContext()

    # Build A (M x K) and B (K x N) in fp32 on host.
    var a_host = ctx.enqueue_create_host_buffer[FP32](M * K)
    var b_host = ctx.enqueue_create_host_buffer[FP32](K * N)
    var c_host_fp32 = ctx.enqueue_create_host_buffer[FP32](M * N)
    ctx.synchronize()
    for i in range(M * K):
        a_host.unsafe_ptr()[i] = Scalar[FP32](Float32(i) * 0.01)
    for i in range(K * N):
        b_host.unsafe_ptr()[i] = Scalar[FP32](Float32(i) * 0.02)

    # fp32 reference matmul (manual).
    for m in range(M):
        for n in range(N):
            var s: Scalar[FP32] = 0.0
            for k in range(K):
                s += a_host.unsafe_ptr()[m * K + k] * b_host.unsafe_ptr()[k * N + n]
            c_host_fp32.unsafe_ptr()[m * N + n] = s

    # Now do the GPU bf16 path: cast A, B to bf16 and call linalg.matmul.
    var a_bf16_h = ctx.enqueue_create_host_buffer[BF16](M * K)
    var b_bf16_h = ctx.enqueue_create_host_buffer[BF16](K * N)
    ctx.synchronize()
    for i in range(M * K):
        a_bf16_h.unsafe_ptr()[i] = a_host.unsafe_ptr()[i].cast[BF16]()
    for i in range(K * N):
        b_bf16_h.unsafe_ptr()[i] = b_host.unsafe_ptr()[i].cast[BF16]()

    var a_dev = ctx.enqueue_create_buffer[BF16](M * K)
    var b_dev = ctx.enqueue_create_buffer[BF16](K * N)
    var c_dev = ctx.enqueue_create_buffer[BF16](M * N)
    ctx.enqueue_copy(a_dev, a_bf16_h)
    ctx.enqueue_copy(b_dev, b_bf16_h)

    var a_tt = TileTensor(a_dev, row_major[M, K]())
    var b_tt = TileTensor(b_dev, row_major[K, N]())
    var c_tt = TileTensor(c_dev, row_major[M, N]())

    max_matmul[target="gpu"](c_tt, a_tt, b_tt, ctx)

    var c_back_h = ctx.enqueue_create_host_buffer[BF16](M * N)
    ctx.enqueue_copy(c_back_h, c_dev)
    ctx.synchronize()

    var max_rel: Scalar[FP32] = 0.0
    for i in range(M * N):
        var got_fp32 = c_back_h.unsafe_ptr()[i].cast[FP32]()
        var ref_fp32 = c_host_fp32.unsafe_ptr()[i]
        var diff = fabs(got_fp32 - ref_fp32)
        if fabs(ref_fp32) > 1e-6:
            var rel = diff / fabs(ref_fp32)
            if rel > max_rel:
                max_rel = rel

    print("bf16 linalg.matmul on Apple:")
    print("  M, K, N    = " + String(M) + ", " + String(K) + ", " + String(N))
    print("  max-rel-err vs fp32 ref = " + String(max_rel))

    # bf16 matmul: 7-bit mantissa, expect 1-2% rel error from accumulating
    # K terms. Be tolerant.
    assert_true(max_rel < 0.05,
        "bf16 matmul rel-err " + String(max_rel) + " > 5% — broken?")


def main() raises:
    print("=" * 60)
    print("nn Phase 3 probe — bf16 linalg.matmul on Apple")
    print("=" * 60)
    test_bf16_matmul()
    print("=" * 60)
    print("PROBE PASSED — bf16 matmul is usable on Apple")
    print("=" * 60)
