"""Parity test: naive CPU matmul vs `linalg.matmul[target="cpu"]`.

Sanity-checks that `linalg.matmul` accepts `target="cpu"` (delegates to the
vendor BLAS / Modular CPU GEMM path) and produces results matching the naive
triple-loop body used by `MatMul.eval` in
`mojo_rl/nn/autodiff/primitives/matmul.mojo`.

Run:
    pixi run mojo run -I . tests/nn/test_max_matmul_cpu.mojo
"""

from std.memory import alloc
from std.random import seed, random_float64
from std.testing import assert_true
from layout import Layout, LayoutTensor
from layout.tile_tensor import lt_to_tt
from linalg.matmul import matmul as max_matmul

from mojo_rl.nn.constants import dtype


def _naive[
    M: Int, N: Int, K: Int
](
    a: LayoutTensor[dtype, Layout.row_major(M, K), MutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(K, N), MutAnyOrigin],
    mut c: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
):
    for i in range(M):
        for j in range(N):
            var acc: c.element_type = 0
            for k in range(K):
                acc += a[i, k] * b[k, j]
            c[i, j] = acc


def _parity[M: Int, K: Int, N: Int](tol: Float64) raises:
    var label = "[" + String(M) + "x" + String(K) + "] @ [" + String(
        K
    ) + "x" + String(N) + "]"

    var a_buf: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
        Scalar[dtype]
    ](M * K)
    var b_buf: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
        Scalar[dtype]
    ](K * N)
    var c_n_buf: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
        Scalar[dtype]
    ](M * N)
    var c_b_buf: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
        Scalar[dtype]
    ](M * N)

    for i in range(M * K):
        a_buf[i] = Scalar[dtype](random_float64(-1.0, 1.0))
    for i in range(K * N):
        b_buf[i] = Scalar[dtype](random_float64(-1.0, 1.0))
    for i in range(M * N):
        c_n_buf[i] = 0
        c_b_buf[i] = 0

    var a = LayoutTensor[dtype, Layout.row_major(M, K), MutAnyOrigin](a_buf)
    var b = LayoutTensor[dtype, Layout.row_major(K, N), MutAnyOrigin](b_buf)
    var c_naive = LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin](
        c_n_buf
    )
    var c_blas = LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin](
        c_b_buf
    )

    _naive[M, N, K](a, b, c_naive)
    max_matmul[target="cpu"](lt_to_tt(c_blas), lt_to_tt(a), lt_to_tt(b), None)

    var max_abs: Float64 = 0.0
    var max_rel: Float64 = 0.0
    for i in range(M * N):
        var x = Float64(c_n_buf[i])
        var y = Float64(c_b_buf[i])
        var d = abs(x - y)
        if d > max_abs:
            max_abs = d
        var denom = max(abs(x), abs(y))
        if denom > 1e-6:
            var r = d / denom
            if r > max_rel:
                max_rel = r

    print(
        "  "
        + label
        + "  max|diff|="
        + String(max_abs)
        + "  max rel="
        + String(max_rel)
    )
    assert_true(
        max_rel < tol,
        "parity violation on " + label + ": max rel diff = " + String(max_rel),
    )

    a_buf.free()
    b_buf.free()
    c_n_buf.free()
    c_b_buf.free()


def main() raises:
    seed(7)
    print("linalg.matmul[target='cpu'] vs naive — parity")
    _parity[16, 32, 16](1e-4)
    _parity[33, 17, 41](1e-4)  # odd, non-power-of-two shapes
    _parity[64, 128, 64](1e-4)
    _parity[128, 64, 256](1e-4)
    _parity[256, 256, 256](1e-4)
    print("All parity tests PASSED.")
