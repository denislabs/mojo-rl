"""Direct Apple Accelerate cblas_sgemm wrapper.

`linalg.matmul` does not expose `alpha`, `beta`, or `transpose_a`. This module
calls cblas_sgemm directly via Modular's already-loaded dylib symbols, so we
can do `dW += cache.T @ grad_out` in a single BLAS call (no transpose
materialization, no tmp buffer, no manual accumulation loop).

macOS / fp32 / row-major only. Intended for the autodiff backward path of
the fused matmul primitives. Non-macOS targets must keep the existing
`max_matmul` workaround.
"""

from std.sys import CompilationTarget
from linalg.matmul.cpu.apple_accelerate import (
    get_cblas_f32_function,
    _CBLASOrder,
    _CBLASTranspose,
)


@always_inline
def apple_sgemm_accum[
    transpose_a: Bool = False,
    transpose_b: Bool = False,
](
    M: Int,
    N: Int,
    K: Int,
    alpha: Float32,
    A: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int,
    B: UnsafePointer[Float32, ImmutAnyOrigin],
    ldb: Int,
    beta: Float32,
    C: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int,
) raises:
    """C = alpha * op(A) @ op(B) + beta * C  (row-major fp32).

    Direct cblas_sgemm. Required because `linalg.matmul` blocks transpose_a
    and lacks alpha/beta. Use beta=1.0 to accumulate into pre-existing C
    (e.g., `dW += cache.T @ grad_out` in the autodiff backward).

    op(A) is (M, K), op(B) is (K, N), C is (M, N).

    Args:
        M, N, K: matmul dimensions after any transposes.
        alpha, beta: scaling constants (typical: alpha=1, beta=0 or 1).
        A, B, C: pointers to row-major fp32 buffers.
        lda, ldb, ldc: leading dimensions = number of columns of the
            underlying (untransposed) buffer.
    """
    comptime assert (
        CompilationTarget.is_macos()
    ), "apple_sgemm_accum is macOS-only (Apple Accelerate)"

    var cblas_gemm = get_cblas_f32_function()
    var ta = _CBLASTranspose.TRANSPOSE if transpose_a else _CBLASTranspose.NO_TRANSPOSE
    var tb = _CBLASTranspose.TRANSPOSE if transpose_b else _CBLASTranspose.NO_TRANSPOSE

    cblas_gemm(
        _CBLASOrder.ROW_MAJOR,
        ta,
        tb,
        Int32(M),
        Int32(N),
        Int32(K),
        alpha,
        A,
        Int32(lda),
        B,
        Int32(ldb),
        beta,
        C,
        Int32(ldc),
    )
