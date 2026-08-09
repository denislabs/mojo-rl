"""Direct Apple Accelerate cblas_sgemm wrapper (PCN-local).

Vendored verbatim from `nn.autodiff.apple_cblas` during the nn
re-architecture (Phase B) so PCN's conv blocks drop their last
`mojo_rl.nn` dependency in the im2col GEMM path. Self-contained: depends
only on Modular's `linalg` Accelerate bindings, not on legacy nn.

`linalg.matmul` does not expose `alpha`, `beta`, or `transpose_a`. This
module calls cblas_sgemm directly via Modular's already-loaded dylib
symbols, so we can do `dW += cache.T @ grad_out` in a single BLAS call
(no transpose materialization, no tmp buffer, no manual accumulation loop).

macOS / fp32 / row-major only. Non-macOS targets keep the `max_matmul`
workaround.
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
    A: Pointer[Float32, ImmutAnyOrigin],
    lda: Int,
    B: Pointer[Float32, ImmutAnyOrigin],
    ldb: Int,
    beta: Float32,
    C: Pointer[Float32, MutAnyOrigin],
    ldc: Int,
) raises:
    """C = alpha * op(A) @ op(B) + beta * C  (row-major fp32).

    Direct cblas_sgemm. Required because `linalg.matmul` blocks transpose_a
    and lacks alpha/beta. Use beta=1.0 to accumulate into pre-existing C
    (e.g., `dW += cache.T @ grad_out` in the autodiff backward).

    op(A) is (M, K), op(B) is (K, N), C is (M, N).

    Args:
        M: Number of rows of the underlying (untransposed) buffer.
        N: Number of columns of the underlying (untransposed) buffer.
        K: Number of columns of the other operand.
        alpha: Scaling constant for operand A (typical: alpha=1).
        A: Pointer to row-major fp32 buffer for operand A.
        lda: Leading dimension of A = number of columns of the untransposed buffer.
        B: Pointer to row-major fp32 buffer for operand B.
        ldb: Leading dimension of B = number of columns of the untransposed buffer.
        beta: Scaling constant for output/accumulation (typical: beta=0 or 1).
        C: Pointer to row-major fp32 buffer for output/accumulation.
        ldc: Leading dimension of C = number of columns of the untransposed buffer.

    """
    comptime assert (
        CompilationTarget.is_macos()
    ), "apple_sgemm_accum is macOS-only (Apple Accelerate)"

    var cblas_gemm = get_cblas_f32_function()
    var ta = (
        _CBLASTranspose.TRANSPOSE if transpose_a else _CBLASTranspose.NO_TRANSPOSE
    )
    var tb = (
        _CBLASTranspose.TRANSPOSE if transpose_b else _CBLASTranspose.NO_TRANSPOSE
    )

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
