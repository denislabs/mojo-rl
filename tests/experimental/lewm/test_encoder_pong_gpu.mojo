"""LeWMEncoder GPU smoke test at Pong pixel scale (Phase 3 GPU port).

Validates the encoder's GPU path runs end-to-end on Apple Metal at
4-channel × 84×84 inputs. Mirrors `test_encoder_pong.mojo` but uses
`GPUNetworkState` + `ctx.enqueue_create_buffer` + `forward_gpu` /
`backward_gpu`.

If this fails on a specific kernel, that's the blocker for the full
GPU trainer port (see `docs/LEWM_PORT_PLAN.md`).

Run:
    pixi run -e apple mojo run -I . tests/experimental/lewm/test_encoder_pong_gpu.mojo
"""

from std.math import abs
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import NetworkState, GPUNetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier
from mojo_rl.experimental.lewm import LeWMEncoder
from layout import Layout, LayoutTensor


def main() raises:
    print("=== LeWMEncoder Pong-scale GPU smoke ===")
    comptime BATCH = 2
    comptime IN_CH = 4
    comptime IMG = 84
    comptime PATCH = 14
    comptime N_PATCHES = (IMG // PATCH) * (IMG // PATCH)   # 36
    comptime HIDDEN = 64
    comptime HEADS = 2
    comptime LAYERS = 2
    comptime EMB = 64
    comptime PROJ_H = 128

    comptime M = LeWMEncoder[
        IN_CH, IMG, IMG, PATCH, HIDDEN, HEADS, LAYERS, N_PATCHES,
        EMB, 2, PROJ_H,
    ]

    print(
        "  shapes: IN_DIM=", M.IN_DIM,
        " OUT_DIM=", M.OUT_DIM,
        " PARAM_SIZE=", M.PARAM_SIZE,
        " CACHE_SIZE=", M.CACHE_SIZE,
        " WORKSPACE/sample=", M.WORKSPACE_SIZE_PER_SAMPLE,
    )

    var ctx = DeviceContext()

    # ---------- Init on CPU, upload to GPU ----------
    var cpu = NetworkState[M, Adam[]]()
    cpu.initialize[Xavier[]]()
    var gpu = GPUNetworkState[M, Adam[]](ctx)
    gpu.upload_from(cpu, ctx)

    # ---------- Activation buffers on device ----------
    var input_buf = ctx.enqueue_create_buffer[dtype](BATCH * M.IN_DIM)
    var output_buf = ctx.enqueue_create_buffer[dtype](BATCH * M.OUT_DIM)
    var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * M.CACHE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[dtype](
        BATCH * M.WORKSPACE_SIZE_PER_SAMPLE if M.WORKSPACE_SIZE_PER_SAMPLE > 0 else 1
    )

    var grad_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * M.OUT_DIM)
    var grad_in_buf = ctx.enqueue_create_buffer[dtype](BATCH * M.IN_DIM)

    # ---------- Seed input on host, copy to device ----------
    var input_host = ctx.enqueue_create_host_buffer[dtype](BATCH * M.IN_DIM)
    for i in range(BATCH * M.IN_DIM):
        input_host[i] = Scalar[dtype](Float64((i * 7) % 256) / 255.0)
    ctx.enqueue_copy(input_buf, input_host)

    # Constant grad_output = 1 (so we backward "as if" L = sum(output)).
    var grad_out_host = ctx.enqueue_create_host_buffer[dtype](BATCH * M.OUT_DIM)
    for i in range(BATCH * M.OUT_DIM):
        grad_out_host[i] = Scalar[dtype](1.0)
    ctx.enqueue_copy(grad_out_buf, grad_out_host)

    # ---------- Views ----------
    var input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](input_buf)
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](output_buf)
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](cache_buf)
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](grad_out_buf)
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](grad_in_buf)

    var params = gpu.params_view()
    var state = gpu.model_state_view()
    var grads = gpu.grads_view()

    # ---------- Forward ----------
    print("\n--- Forward ---")
    M.forward_gpu[BATCH, dtype](
        ctx, output_t, input_t, params, state, cache_t, workspace_buf,
    )
    ctx.synchronize()
    print("  forward returned OK")

    # ---------- Inspect output (download) ----------
    var output_host = ctx.enqueue_create_host_buffer[dtype](BATCH * M.OUT_DIM)
    ctx.enqueue_copy(output_host, output_buf)
    ctx.synchronize()
    var nan_count = 0
    var max_abs_out: Float64 = 0.0
    for i in range(BATCH * M.OUT_DIM):
        var v = Float64(output_host[i])
        if v != v:
            nan_count += 1
        var av = abs(v)
        if av > max_abs_out:
            max_abs_out = av
    print(
        "  forward output: max|out|=", max_abs_out,
        " NaN=", nan_count, "/", BATCH * M.OUT_DIM,
    )
    if nan_count > 0:
        print("  [FAIL] forward produced NaN")
        return

    # ---------- Backward ----------
    print("\n--- Backward ---")
    gpu.zero_grads(ctx)
    M.backward_gpu[BATCH, dtype](
        ctx, grad_in_t, grad_out_t, params, state, cache_t, grads,
        workspace_buf,
    )
    ctx.synchronize()
    print("  backward returned OK")

    # ---------- Inspect param grads ----------
    var grads_host = ctx.enqueue_create_host_buffer[dtype](M.PARAM_SIZE)
    ctx.enqueue_copy(grads_host, gpu.grads_buf)
    ctx.synchronize()
    var max_abs_grad: Float64 = 0.0
    var nz_count = 0
    var nan_grad = 0
    for i in range(M.PARAM_SIZE):
        var v = Float64(grads_host[i])
        if v != v:
            nan_grad += 1
        var av = abs(v)
        if av > max_abs_grad:
            max_abs_grad = av
        if av > 1e-8:
            nz_count += 1
    print(
        "  backward grads: max|g|=", max_abs_grad,
        " nz=", nz_count, "/", M.PARAM_SIZE,
        " NaN=", nan_grad,
    )

    if (
        nan_count == 0
        and nan_grad == 0
        and max_abs_out > 1e-6
        and max_abs_grad > 1e-6
        and nz_count > M.PARAM_SIZE // 4
    ):
        print("\n  [PASS] LeWMEncoder GPU smoke")
    else:
        print("\n  [FAIL] LeWMEncoder GPU smoke")
