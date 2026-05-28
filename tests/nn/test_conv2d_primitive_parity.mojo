"""Forward+backward parity checksum for Conv2DLayer (= AutoFused[Conv2D]).

Used to validate the packed-im2col + batched-matmul refactor in
mojo_rl/nn/autodiff/primitives/conv2d.mojo. Captures output / grad_input /
grad_W sums for several shapes — values should match baseline to ~1e-7 rel.
"""

from std.memory import alloc, memset
from std.random import seed, random_float64
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.model import Conv2DLayer, Model


@always_inline
def fill_random(p: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int):
    for i in range(n):
        p[i] = Scalar[dtype](random_float64(-1.0, 1.0))


def check_conv[M: Model, BATCH: Int](label: String) raises:
    comptime IN_DIM = M.IN_DIM
    comptime OUT_DIM = M.OUT_DIM
    comptime PARAM_SIZE = M.PARAM_SIZE
    comptime STATE_SIZE = M.STATE_SIZE
    comptime CACHE_SIZE = M.CACHE_SIZE

    print("─" * 70)
    print(label)
    print(
        "  IN_DIM=",
        IN_DIM,
        " OUT_DIM=",
        OUT_DIM,
        " PS=",
        PARAM_SIZE,
        " BATCH=",
        BATCH,
    )

    seed(20260521)

    var input_buf = alloc[Scalar[dtype]](BATCH * IN_DIM)
    var output_buf = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    var grad_in_buf = alloc[Scalar[dtype]](BATCH * IN_DIM)
    var grad_out_buf = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    var params_buf = alloc[Scalar[dtype]](PARAM_SIZE)
    var grads_buf = alloc[Scalar[dtype]](PARAM_SIZE)
    var state_buf = alloc[Scalar[dtype]](max(1, STATE_SIZE))
    var cache_buf = alloc[Scalar[dtype]](BATCH * max(1, CACHE_SIZE))

    fill_random(input_buf, BATCH * IN_DIM)
    fill_random(grad_out_buf, BATCH * OUT_DIM)
    memset(output_buf, 0, BATCH * OUT_DIM)
    memset(grad_in_buf, 0, BATCH * IN_DIM)
    memset(grads_buf, 0, PARAM_SIZE)
    memset(cache_buf, 0, BATCH * max(1, CACHE_SIZE))
    memset(state_buf, 0, max(1, STATE_SIZE))

    var params_lt = LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
    ](params_buf)
    var state_lt = LayoutTensor[
        dtype, Layout.row_major(STATE_SIZE), MutAnyOrigin
    ](state_buf)
    M.initialize_params[Kaiming[], dtype](params_lt)
    comptime if STATE_SIZE > 0:
        M.initialize_state[dtype](state_lt)

    var input_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
    ](input_buf)
    var output_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ](output_buf)
    var grad_in_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
    ](grad_in_buf)
    var grad_out_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ](grad_out_buf)
    var cache_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin
    ](cache_buf)
    var grads_lt = LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
    ](grads_buf)

    M.forward[BATCH, dtype](
        input_lt, output_lt, params_lt, state_lt, cache_lt
    )
    M.backward[BATCH, dtype](
        grad_out_lt, grad_in_lt, params_lt, state_lt, cache_lt, grads_lt
    )

    var out_sum: Float64 = 0.0
    var out_sumsq: Float64 = 0.0
    for i in range(BATCH * OUT_DIM):
        var x = Float64(output_buf[i])
        out_sum += x
        out_sumsq += x * x

    var gi_sum: Float64 = 0.0
    var gi_sumsq: Float64 = 0.0
    for i in range(BATCH * IN_DIM):
        var x = Float64(grad_in_buf[i])
        gi_sum += x
        gi_sumsq += x * x

    var gw_sum: Float64 = 0.0
    var gw_sumsq: Float64 = 0.0
    for i in range(PARAM_SIZE):
        var x = Float64(grads_buf[i])
        gw_sum += x
        gw_sumsq += x * x

    print("  output    sum=", String(out_sum), " sumsq=", String(out_sumsq))
    print("  grad_in   sum=", String(gi_sum), " sumsq=", String(gi_sumsq))
    print("  grad_W    sum=", String(gw_sum), " sumsq=", String(gw_sumsq))

    input_buf.free()
    output_buf.free()
    grad_in_buf.free()
    grad_out_buf.free()
    params_buf.free()
    grads_buf.free()
    state_buf.free()
    cache_buf.free()


def main() raises:
    print("=" * 70)
    print(" Conv2D primitive (via Conv2DLayer) parity checksum")
    print("=" * 70)
    print()

    check_conv[Conv2DLayer[2, 4, 3, 1, 1, 5, 5], 4](
        "Conv2DLayer[2,4,3x3,5x5 same]"
    )
    check_conv[Conv2DLayer[3, 8, 3, 1, 0, 3, 3], 4](
        "Conv2DLayer[3,8,3x3,3x3 valid]"
    )
    check_conv[Conv2DLayer[128, 128, 3, 1, 1, 3, 3], 8](
        "Conv2DLayer[128,128,3x3,3x3 same] (TTT-sized)"
    )
    check_conv[Conv2DLayer[128, 128, 3, 1, 0, 3, 3], 8](
        "Conv2DLayer[128,128,3x3,1x1 valid] (TTT L4-sized)"
    )
    check_conv[Conv2DLayer[128, 128, 3, 1, 1, 6, 7], 8](
        "Conv2DLayer[128,128,3x3,6x7 same] (C4 ResBlock Conv2 sized)"
    )
    check_conv[Conv2DLayer[128, 32, 1, 1, 0, 6, 7], 8](
        "Conv2DLayer[128,32,1x1,6x7 valid] (alt head shape)"
    )

    print("=" * 70)
    print(" Done.")
    print("=" * 70)
