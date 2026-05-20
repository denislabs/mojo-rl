"""Time each step of Conv2DBatchNormReLU L4 backward to identify the
residual bottleneck after BLAS-matmul optimization. Uses BATCH=64,
shape (128, 128, 3x3 valid, 3x3 -> 1x1) — the AZ TTT CNN L4 config.

Run:
    pixi run mojo run -I . tests/nn/test_conv_l4_breakdown.mojo
"""

from std.memory import alloc, memset
from std.random import seed, random_float64
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor
from layout.tile_tensor import lt_to_tt
from linalg.matmul import matmul as max_matmul

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.model import Conv2DBatchNormReLU


def main() raises:
    seed(20260520)

    comptime IN_C = 128
    comptime OUT_C = 128
    comptime K = 3
    comptime S = 1
    comptime P = 0
    comptime IH = 3
    comptime IW = 3
    comptime BATCH = 64
    comptime ITERS = 30

    comptime Layer = Conv2DBatchNormReLU[IN_C, OUT_C, K, S, P, IH, IW]
    comptime IN_DIM = Layer.IN_DIM
    comptime OUT_DIM = Layer.OUT_DIM
    comptime PARAM_SIZE = Layer.PARAM_SIZE
    comptime STATE_SIZE = Layer.STATE_SIZE
    comptime CACHE_SIZE = Layer.CACHE_SIZE
    comptime col_size = Layer.col_size
    comptime spatial_out = Layer.spatial_out

    print(
        "L4: in=",
        IN_C,
        "x",
        IH,
        "x",
        IW,
        "  out=",
        OUT_C,
        "x1x1  col_size=",
        col_size,
        "  spatial_out=",
        spatial_out,
        "  BATCH=",
        BATCH,
    )

    # Allocate buffers and run forward once to populate cache + state.
    var input_buf = alloc[Scalar[dtype]](BATCH * IN_DIM)
    var output_buf = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    var grad_in_buf = alloc[Scalar[dtype]](BATCH * IN_DIM)
    var grad_out_buf = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    var params_buf = alloc[Scalar[dtype]](PARAM_SIZE)
    var grads_buf = alloc[Scalar[dtype]](PARAM_SIZE)
    var state_buf = alloc[Scalar[dtype]](max(1, STATE_SIZE))
    var cache_buf = alloc[Scalar[dtype]](BATCH * CACHE_SIZE)

    for i in range(BATCH * IN_DIM):
        input_buf[i] = Scalar[dtype](random_float64(-1.0, 1.0))
    for i in range(BATCH * OUT_DIM):
        grad_out_buf[i] = Scalar[dtype](random_float64(-1.0, 1.0))
    memset(grads_buf, 0, PARAM_SIZE)

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

    # Time the full backward
    Layer.backward[BATCH, dtype](
        grad_out_lt, grad_in_lt, params_lt, state_lt, cache_lt, grads_lt
    )  # warmup
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        memset(grads_buf, 0, PARAM_SIZE)
        Layer.backward[BATCH, dtype](
            grad_out_lt, grad_in_lt, params_lt, state_lt, cache_lt, grads_lt
        )
    var t1 = perf_counter_ns()
    print(
        "Full backward:                          ",
        String(Float64(t1 - t0) / 1e6 / Float64(ITERS))[byte=:8],
        "ms",
    )

    # Now time individual sub-operations.
    # 1) Per-batch scalar accumulate dW += dW_tmp at the shape used by L4.
    var dW_tmp_buf = alloc[Scalar[dtype]](OUT_C * col_size)
    for i in range(OUT_C * col_size):
        dW_tmp_buf[i] = Scalar[dtype](random_float64(-1.0, 1.0))
    memset(grads_buf, 0, PARAM_SIZE)
    var t2 = perf_counter_ns()
    for _ in range(ITERS):
        for _ in range(BATCH):
            for oc in range(OUT_C):
                for k in range(col_size):
                    grads_buf[oc * col_size + k] = (
                        grads_buf[oc * col_size + k] + dW_tmp_buf[oc * col_size + k]
                    )
    var t3 = perf_counter_ns()
    print(
        "Just per-batch scalar dW accumulate     ",
        String(Float64(t3 - t2) / 1e6 / Float64(ITERS))[byte=:8],
        "ms   (",
        BATCH * OUT_C * col_size,
        "scalar adds per iter)",
    )

    # 2) Per-batch BLAS matmul (the BLAS calls themselves)
    var go_buf = alloc[Scalar[dtype]](OUT_C * spatial_out)
    var col_buf = alloc[Scalar[dtype]](spatial_out * col_size)
    for i in range(OUT_C * spatial_out):
        go_buf[i] = Scalar[dtype](random_float64(-1.0, 1.0))
    for i in range(spatial_out * col_size):
        col_buf[i] = Scalar[dtype](random_float64(-1.0, 1.0))
    var go_lt = LayoutTensor[
        dtype, Layout.row_major(OUT_C, spatial_out), MutAnyOrigin
    ](go_buf)
    var col_lt = LayoutTensor[
        dtype, Layout.row_major(spatial_out, col_size), MutAnyOrigin
    ](col_buf)
    var dW_tmp_lt = LayoutTensor[
        dtype, Layout.row_major(OUT_C, col_size), MutAnyOrigin
    ](dW_tmp_buf)
    var t4 = perf_counter_ns()
    for _ in range(ITERS):
        for _ in range(BATCH):
            max_matmul[target="cpu"](
                lt_to_tt(dW_tmp_lt), lt_to_tt(go_lt), lt_to_tt(col_lt), None
            )
    var t5 = perf_counter_ns()
    print(
        "Per-batch BLAS matmul (dW_tmp = go@col):",
        String(Float64(t5 - t4) / 1e6 / Float64(ITERS))[byte=:8],
        "ms",
    )

    input_buf.free()
    output_buf.free()
    grad_in_buf.free()
    grad_out_buf.free()
    params_buf.free()
    grads_buf.free()
    state_buf.free()
    cache_buf.free()
    dW_tmp_buf.free()
    go_buf.free()
    col_buf.free()
