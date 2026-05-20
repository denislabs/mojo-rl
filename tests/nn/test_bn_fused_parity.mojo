"""Forward+backward numeric parity for Conv2DBatchNormReLU and
LinearBatchNormReLU.

Runs each layer with a fixed seed / fixed input / fixed grad_output,
then prints checksums (sum of output, sum of grads, sum of grad_input).
Used to validate the upcoming BLAS-matmul refactor: capture checksums on
the current implementation, then re-run after the refactor and require
the printed values to match to ~1e-3 relative tolerance.

Run:
    pixi run mojo run -I . tests/nn/test_bn_fused_parity.mojo
"""

from std.memory import alloc, memset
from std.random import seed, random_float64
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.model import Conv2DBatchNormReLU, LinearBatchNormReLU


@always_inline
def fill_random(p: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int):
    for i in range(n):
        p[i] = Scalar[dtype](random_float64(-1.0, 1.0))


@always_inline
def fmt(x: Float64) -> String:
    return String(x)


def check_conv_bn[
    IN_C: Int, OUT_C: Int, K: Int, S: Int, P: Int, IH: Int, IW: Int, BATCH: Int
](label: String) raises:
    comptime Layer = Conv2DBatchNormReLU[IN_C, OUT_C, K, S, P, IH, IW]
    comptime IN_DIM = Layer.IN_DIM
    comptime OUT_DIM = Layer.OUT_DIM
    comptime PARAM_SIZE = Layer.PARAM_SIZE
    comptime STATE_SIZE = Layer.STATE_SIZE
    comptime CACHE_SIZE = Layer.CACHE_SIZE

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

    seed(20260520)

    var input_buf = alloc[Scalar[dtype]](BATCH * IN_DIM)
    var output_buf = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    var grad_in_buf = alloc[Scalar[dtype]](BATCH * IN_DIM)
    var grad_out_buf = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    var params_buf = alloc[Scalar[dtype]](PARAM_SIZE)
    var grads_buf = alloc[Scalar[dtype]](PARAM_SIZE)
    var state_buf = alloc[Scalar[dtype]](max(1, STATE_SIZE))
    var cache_buf = alloc[Scalar[dtype]](BATCH * CACHE_SIZE)

    fill_random(input_buf, BATCH * IN_DIM)
    fill_random(grad_out_buf, BATCH * OUT_DIM)
    memset(output_buf, 0, BATCH * OUT_DIM)
    memset(grad_in_buf, 0, BATCH * IN_DIM)
    memset(grads_buf, 0, PARAM_SIZE)
    memset(cache_buf, 0, BATCH * CACHE_SIZE)

    var params_lt = LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
    ](params_buf)
    var state_lt = LayoutTensor[
        dtype, Layout.row_major(STATE_SIZE), MutAnyOrigin
    ](state_buf)
    Layer.initialize_params[Kaiming[], dtype](params_lt)
    Layer.initialize_state[dtype](state_lt)

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

    Layer.forward[BATCH, dtype](
        input_lt, output_lt, params_lt, state_lt, cache_lt
    )
    Layer.backward[BATCH, dtype](
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

    print("  output    sum=", fmt(out_sum), " sumsq=", fmt(out_sumsq))
    print("  grad_in   sum=", fmt(gi_sum), " sumsq=", fmt(gi_sumsq))
    print("  grad_W    sum=", fmt(gw_sum), " sumsq=", fmt(gw_sumsq))

    input_buf.free()
    output_buf.free()
    grad_in_buf.free()
    grad_out_buf.free()
    params_buf.free()
    grads_buf.free()
    state_buf.free()
    cache_buf.free()


def check_linear_bn[
    IN_DIM: Int, OUT_DIM: Int, BATCH: Int
](label: String) raises:
    comptime Layer = LinearBatchNormReLU[IN_DIM, OUT_DIM]
    comptime PARAM_SIZE = Layer.PARAM_SIZE
    comptime STATE_SIZE = Layer.STATE_SIZE
    comptime CACHE_SIZE = Layer.CACHE_SIZE

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

    seed(20260520)

    var input_buf = alloc[Scalar[dtype]](BATCH * IN_DIM)
    var output_buf = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    var grad_in_buf = alloc[Scalar[dtype]](BATCH * IN_DIM)
    var grad_out_buf = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    var params_buf = alloc[Scalar[dtype]](PARAM_SIZE)
    var grads_buf = alloc[Scalar[dtype]](PARAM_SIZE)
    var state_buf = alloc[Scalar[dtype]](max(1, STATE_SIZE))
    var cache_buf = alloc[Scalar[dtype]](BATCH * CACHE_SIZE)

    fill_random(input_buf, BATCH * IN_DIM)
    fill_random(grad_out_buf, BATCH * OUT_DIM)
    memset(output_buf, 0, BATCH * OUT_DIM)
    memset(grad_in_buf, 0, BATCH * IN_DIM)
    memset(grads_buf, 0, PARAM_SIZE)
    memset(cache_buf, 0, BATCH * CACHE_SIZE)

    var params_lt = LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
    ](params_buf)
    var state_lt = LayoutTensor[
        dtype, Layout.row_major(STATE_SIZE), MutAnyOrigin
    ](state_buf)
    Layer.initialize_params[Kaiming[], dtype](params_lt)
    Layer.initialize_state[dtype](state_lt)

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

    Layer.forward[BATCH, dtype](
        input_lt, output_lt, params_lt, state_lt, cache_lt
    )
    Layer.backward[BATCH, dtype](
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

    print("  output    sum=", fmt(out_sum), " sumsq=", fmt(out_sumsq))
    print("  grad_in   sum=", fmt(gi_sum), " sumsq=", fmt(gi_sumsq))
    print("  grad_W    sum=", fmt(gw_sum), " sumsq=", fmt(gw_sumsq))

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
    print(" BN-fused layer parity checksum (capture / compare)")
    print("=" * 70)
    print()

    check_conv_bn[3, 8, 3, 1, 1, 5, 5, 4]("Conv2DBatchNormReLU[3,8,3x3,5x5]")
    check_conv_bn[3, 128, 3, 1, 1, 3, 3, 8](
        "Conv2DBatchNormReLU[3,128,3x3,3x3] (AZ TTT L1)"
    )
    check_conv_bn[128, 128, 3, 1, 1, 3, 3, 8](
        "Conv2DBatchNormReLU[128,128,3x3,3x3] (AZ TTT L2/L3)"
    )
    check_conv_bn[128, 128, 3, 1, 0, 3, 3, 8](
        "Conv2DBatchNormReLU[128,128,3x3,1x1] (AZ TTT L4 valid)"
    )

    check_linear_bn[16, 8, 4]("LinearBatchNormReLU[16,8]")
    check_linear_bn[128, 256, 8]("LinearBatchNormReLU[128,256] (AZ TTT L5)")
    check_linear_bn[256, 128, 8]("LinearBatchNormReLU[256,128] (AZ TTT L6)")

    print("=" * 70)
    print(" Done.")
    print("=" * 70)
