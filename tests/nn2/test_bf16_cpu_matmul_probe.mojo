"""Phase 8.3 probe: does `linalg.matmul[target="cpu"]` accept bf16 TileTensors?

Mirrors `test_bf16_matmul_probe.mojo` (GPU) on CPU. If this passes, the
cast-around-matmul path is viable on CPU and Path A of Phase 8.3 can ship.
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul


comptime FP32 = DType.float32
comptime BF16 = DType.bfloat16


def test_bf16_cpu_matmul() raises:
    comptime M = 4
    comptime K = 8
    comptime N = 4

    # fp32 host buffers (reference path).
    var a_fp32: UnsafePointer[Scalar[FP32], MutAnyOrigin] = alloc[Scalar[FP32]](M * K)
    var b_fp32: UnsafePointer[Scalar[FP32], MutAnyOrigin] = alloc[Scalar[FP32]](K * N)
    var c_ref:  UnsafePointer[Scalar[FP32], MutAnyOrigin] = alloc[Scalar[FP32]](M * N)
    for i in range(M * K):
        a_fp32[i] = Scalar[FP32](Float32(i) * 0.01)
    for i in range(K * N):
        b_fp32[i] = Scalar[FP32](Float32(i) * 0.02)

    # fp32 reference matmul (manual).
    for m in range(M):
        for n in range(N):
            var s: Scalar[FP32] = 0.0
            for k in range(K):
                s += a_fp32[m * K + k] * b_fp32[k * N + n]
            c_ref[m * N + n] = s

    # bf16 cast-around path.
    var a_bf16: UnsafePointer[Scalar[BF16], MutAnyOrigin] = alloc[Scalar[BF16]](M * K)
    var b_bf16: UnsafePointer[Scalar[BF16], MutAnyOrigin] = alloc[Scalar[BF16]](K * N)
    var c_bf16: UnsafePointer[Scalar[BF16], MutAnyOrigin] = alloc[Scalar[BF16]](M * N)
    for i in range(M * K):
        a_bf16[i] = a_fp32[i].cast[BF16]()
    for i in range(K * N):
        b_bf16[i] = b_fp32[i].cast[BF16]()

    var a_tt = TileTensor(a_bf16, row_major[M, K]())
    var b_tt = TileTensor(b_bf16, row_major[K, N]())
    var c_tt = TileTensor(c_bf16, row_major[M, N]())

    max_matmul[target="cpu"](c_tt, a_tt, b_tt, None)

    var max_rel: Scalar[FP32] = 0.0
    for i in range(M * N):
        var got_fp32 = c_bf16[i].cast[FP32]()
        var ref_fp32 = c_ref[i]
        var diff = fabs(got_fp32 - ref_fp32)
        if fabs(ref_fp32) > 1e-6:
            var rel = diff / fabs(ref_fp32)
            if rel > max_rel:
                max_rel = rel

    print("bf16 linalg.matmul on CPU:")
    print("  M, K, N    = " + String(M) + ", " + String(K) + ", " + String(N))
    print("  max-rel-err vs fp32 ref = " + String(max_rel))

    assert_true(max_rel < 0.05,
        "bf16 CPU matmul rel-err " + String(max_rel) + " > 5% — broken?")

    a_fp32.free()
    b_fp32.free()
    c_ref.free()
    a_bf16.free()
    b_bf16.free()
    c_bf16.free()


def main() raises:
    print("=" * 60)
    print("nn2 Phase 8.3 probe — bf16 linalg.matmul on CPU")
    print("=" * 60)
    test_bf16_cpu_matmul()
    print("=" * 60)
    print("PROBE PASSED — bf16 CPU matmul is usable, Path A unblocked")
    print("=" * 60)
