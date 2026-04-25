"""Diagnostic for CIFAR-10 training explosion.

Replicates the full CIFAR architecture (6 Conv2DBN+ReLU + 3 MaxPool + Flatten
+ LinearReLU[2048,128] + Linear[128,10]), but runs ONE training step at a
time and prints:
  - Max |params|, |grads| per major layer block AFTER backward
  - Max |output|, min/max logits after forward
  - Max |grad_out| after loss backward
  - Loss value
for a fixed random input + fixed one-hot targets. Then we apply an
optimizer step and repeat for a few steps to see where explosion occurs.

Run:
  pixi run -e apple mojo run -I . tests/nn/test_cifar_diagnostic.mojo
"""

from std.math import sqrt, log
from std.random import seed, random_float64
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, gpu_align
from mojo_rl.nn.model.conv2d_bn_relu import Conv2DBatchNormReLU
from mojo_rl.nn.model.pool_layer import MaxPoolLayer
from mojo_rl.nn.model.flatten_layer import FlattenLayer
from mojo_rl.nn.model.linear_act import LinearReLU
from mojo_rl.nn.model.linear import Linear
from mojo_rl.nn.model.sequential import Sequential
from mojo_rl.nn.loss.cross_entropy import CrossEntropyLoss
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.training.trainer import Trainer
from mojo_rl.nn.initializer.initializers import Kaiming
from mojo_rl.nn.datasets.cifar10 import CIFAR10


comptime BATCH = 128
comptime STEPS = 390  # ~1 full epoch of unique batches
comptime N_POOL = STEPS * BATCH  # 49920, just under 50000


comptime CNN = Sequential[
    Conv2DBatchNormReLU[3, 32, 3, 1, 1, 32, 32],
    Conv2DBatchNormReLU[32, 32, 3, 1, 1, 32, 32],
    MaxPoolLayer[32, 32, 32, 2],
    Conv2DBatchNormReLU[32, 64, 3, 1, 1, 16, 16],
    Conv2DBatchNormReLU[64, 64, 3, 1, 1, 16, 16],
    MaxPoolLayer[64, 16, 16, 2],
    Conv2DBatchNormReLU[64, 128, 3, 1, 1, 8, 8],
    Conv2DBatchNormReLU[128, 128, 3, 1, 1, 8, 8],
    MaxPoolLayer[128, 8, 8, 2],
    FlattenLayer[128 * 4 * 4],
    LinearReLU[128 * 4 * 4, 128],
    Linear[128, 10],
]


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _randn() -> Float64:
    from std.math import cos as stdlibcos
    var u1 = random_float64()
    var u2 = random_float64()
    if u1 < 1e-12:
        u1 = 1e-12
    return sqrt(-2.0 * log(u1)) * stdlibcos(6.283185307179586 * u2)


def _report_stats(name: String, buf: UnsafePointer[Scalar[dtype], ...], n: Int):
    var mx: Float64 = 0.0
    var mn: Float64 = 1e30
    var mx_signed: Float64 = -1e30
    var mn_signed: Float64 = 1e30
    var nan_ct: Int = 0
    var inf_ct: Int = 0
    for i in range(n):
        var v = Float64(buf[i])
        if v != v:
            nan_ct += 1
            continue
        if v > 1e30 or v < -1e30:
            inf_ct += 1
            continue
        if v > mx_signed:
            mx_signed = v
        if v < mn_signed:
            mn_signed = v
        var av = v if v >= 0.0 else -v
        if av > mx:
            mx = av
    print(
        "  "
        + name
        + ": max|.|="
        + String(mx)
        + " range=["
        + String(mn_signed)
        + ", "
        + String(mx_signed)
        + "] nan="
        + String(nan_ct)
        + " inf="
        + String(inf_ct)
    )


# ── Per-layer param block offsets (precomputed statically for breakdown) ──
# Layer order: L0..L11 of CNN Sequential.
# Layers with PARAM_SIZE > 0: L0..L1, L3..L4, L6..L7 (conv+BN),
# L10 (LinearReLU), L11 (Linear). L2/L5/L8 are MaxPool (0 params).
# L9 is Flatten (0 params).
comptime L0 = Conv2DBatchNormReLU[3, 32, 3, 1, 1, 32, 32]
comptime L1 = Conv2DBatchNormReLU[32, 32, 3, 1, 1, 32, 32]
comptime L3 = Conv2DBatchNormReLU[32, 64, 3, 1, 1, 16, 16]
comptime L4 = Conv2DBatchNormReLU[64, 64, 3, 1, 1, 16, 16]
comptime L6 = Conv2DBatchNormReLU[64, 128, 3, 1, 1, 8, 8]
comptime L7 = Conv2DBatchNormReLU[128, 128, 3, 1, 1, 8, 8]
comptime L10 = LinearReLU[128 * 4 * 4, 128]
comptime L11 = Linear[128, 10]


def _report_params_and_grads(
    p_host: UnsafePointer[Scalar[dtype], ...],
    g_host: UnsafePointer[Scalar[dtype], ...],
    s_host: UnsafePointer[Scalar[dtype], ...],
    tag: String,
):
    # Compute each layer's cumulative offset in the Sequential param buffer,
    # matching Sequential._param_offset[i]().
    var off_L0 = 0
    var off_L1 = gpu_align(L0.PARAM_SIZE)
    var off_L2 = off_L1 + gpu_align(L1.PARAM_SIZE)  # maxpool at L2
    var off_L3 = off_L2 + gpu_align(0)              # maxpool has PARAM_SIZE=0
    var off_L4 = off_L3 + gpu_align(L3.PARAM_SIZE)
    var off_L5 = off_L4 + gpu_align(L4.PARAM_SIZE)
    var off_L6 = off_L5 + gpu_align(0)
    var off_L7 = off_L6 + gpu_align(L6.PARAM_SIZE)
    var off_L8 = off_L7 + gpu_align(L7.PARAM_SIZE)
    var off_L9 = off_L8 + gpu_align(0)              # flatten
    var off_L10 = off_L9 + gpu_align(0)
    var off_L11 = off_L10 + gpu_align(L10.PARAM_SIZE)

    # State offsets (no alignment in Sequential._state_offset).
    # State sizes per layer (Conv2DBNReLU = 2*out_channels, others = 0).
    var sof_L0 = 0
    var sof_L1 = L0.STATE_SIZE
    var sof_L3 = sof_L1 + L1.STATE_SIZE  # L2 maxpool has STATE_SIZE=0
    var sof_L4 = sof_L3 + L3.STATE_SIZE
    var sof_L6 = sof_L4 + L4.STATE_SIZE  # L5 maxpool has STATE_SIZE=0
    var sof_L7 = sof_L6 + L6.STATE_SIZE

    print("  === params [" + tag + "] ===")
    _report_stats("  L0  W    ", p_host + off_L0 + L0.W_OFF, L0.CONV_W_SIZE)
    _report_stats("  L0  bias ", p_host + off_L0 + L0.BIAS_OFF, L0.out_channels)
    _report_stats("  L0  gamma", p_host + off_L0 + L0.GAMMA_OFF, L0.out_channels)
    _report_stats("  L0  beta ", p_host + off_L0 + L0.BETA_OFF, L0.out_channels)
    _report_stats("  L0  rmean", s_host + sof_L0 + L0.RMEAN_OFF, L0.out_channels)
    _report_stats("  L0  rvar ", s_host + sof_L0 + L0.RVAR_OFF, L0.out_channels)
    _report_stats("  L7  W    ", p_host + off_L7 + L7.W_OFF, L7.CONV_W_SIZE)
    _report_stats("  L7  gamma", p_host + off_L7 + L7.GAMMA_OFF, L7.out_channels)
    _report_stats("  L7  rvar ", s_host + sof_L7 + L7.RVAR_OFF, L7.out_channels)
    _report_stats("  L10 params", p_host + off_L10, L10.PARAM_SIZE)
    _report_stats("  L11 params", p_host + off_L11, L11.PARAM_SIZE)
    print("  === grads [" + tag + "] ===")
    _report_stats("  L0  dW   ", g_host + off_L0 + L0.W_OFF, L0.CONV_W_SIZE)
    _report_stats("  L0  dgamma", g_host + off_L0 + L0.GAMMA_OFF, L0.out_channels)
    _report_stats("  L0  dbeta", g_host + off_L0 + L0.BETA_OFF, L0.out_channels)
    # NOTE: rmean/rvar are state-only post-Phase-3, no longer have grads.
    _report_stats("  L7  dW   ", g_host + off_L7 + L7.W_OFF, L7.CONV_W_SIZE)
    _report_stats("  L7  dgamma", g_host + off_L7 + L7.GAMMA_OFF, L7.out_channels)
    _report_stats("  L10 grads", g_host + off_L10, L10.PARAM_SIZE)
    _report_stats("  L11 grads", g_host + off_L11, L11.PARAM_SIZE)


def main() raises:
    seed(42)

    var ctx = DeviceContext()
    comptime TRAINER = Trainer[CNN, Adam[LR=0.001], CrossEntropyLoss]
    var state = TRAINER.init_state_gpu[Kaiming[]](ctx)

    # ── fixed input + one-hot target on device ──
    comptime IN_DIM = CNN.IN_DIM
    comptime OUT_DIM = CNN.OUT_DIM
    comptime CACHE_SIZE = CNN.CACHE_SIZE
    comptime WS_SIZE = BATCH * CNN.WORKSPACE_SIZE_PER_SAMPLE

    print("CIFAR diagnostic: BATCH=" + String(BATCH)
          + " PARAM_SIZE=" + String(CNN.PARAM_SIZE)
          + " CACHE=" + String(CACHE_SIZE)
          + " WS/sample=" + String(CNN.WORKSPACE_SIZE_PER_SAMPLE))

    # Load real CIFAR-10 — use first N_POOL samples
    print("loading CIFAR-10...")
    var ds = CIFAR10()

    var input_host = ctx.enqueue_create_host_buffer[dtype](N_POOL * IN_DIM)
    for i in range(N_POOL * IN_DIM):
        input_host[i] = ds.train_images[i]
    var input_buf = ctx.enqueue_create_buffer[dtype](N_POOL * IN_DIM)
    ctx.enqueue_copy(input_buf, input_host)

    var target_host = ctx.enqueue_create_host_buffer[dtype](N_POOL * OUT_DIM)
    for i in range(N_POOL * OUT_DIM):
        target_host[i] = Scalar[dtype](0.0)
    for b in range(N_POOL):
        var label = Int(ds.train_labels[b])
        target_host[b * OUT_DIM + label] = Scalar[dtype](1.0)
    var target_buf = ctx.enqueue_create_buffer[dtype](N_POOL * OUT_DIM)
    ctx.enqueue_copy(target_buf, target_host)

    # Quick input stats sanity check
    var mx: Float64 = 0.0
    for i in range(BATCH * IN_DIM):
        var v = Float64(input_host[i])
        var av = v if v >= 0.0 else -v
        if av > mx:
            mx = av
    print("  first batch max|pixel|=" + String(mx))

    # ── per-step device buffers ──
    var output_buf = ctx.enqueue_create_buffer[dtype](BATCH * OUT_DIM)
    var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * CACHE_SIZE)
    var grad_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * OUT_DIM)
    var grad_in_buf = ctx.enqueue_create_buffer[dtype](BATCH * IN_DIM)
    var loss_buf = ctx.enqueue_create_buffer[dtype](1)
    var ws_buf = ctx.enqueue_create_buffer[dtype](WS_SIZE if WS_SIZE > 0 else 1)

    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ](output_buf.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin
    ](cache_buf.unsafe_ptr())
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ](grad_out_buf.unsafe_ptr())
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
    ](grad_in_buf.unsafe_ptr())
    var loss_t = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
        loss_buf.unsafe_ptr()
    )

    # ── Host buffers for downloading snapshots ──
    var out_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT_DIM)
    var go_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT_DIM)
    var params_dl = ctx.enqueue_create_host_buffer[dtype](CNN.PARAM_SIZE)
    var grads_dl = ctx.enqueue_create_host_buffer[dtype](CNN.PARAM_SIZE)
    comptime MS_SIZE = max(1, CNN.STATE_SIZE)
    var state_dl = ctx.enqueue_create_host_buffer[dtype](MS_SIZE)
    var loss_host = ctx.enqueue_create_host_buffer[dtype](1)

    # Snapshot initial params
    ctx.enqueue_copy(params_dl, state.params_buf)
    ctx.synchronize()
    print("\n── initial params snapshot ──")
    _report_stats("  full params", params_dl.unsafe_ptr(), CNN.PARAM_SIZE)

    for step in range(STEPS):
        # NO cycling — use unique batches
        var batch_idx = step
        var input_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
        ](input_buf.unsafe_ptr() + batch_idx * BATCH * IN_DIM)
        var target_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ](target_buf.unsafe_ptr() + batch_idx * BATCH * OUT_DIM)

        state.zero_grads(ctx)
        var params = state.params_view()
        var grads = state.grads_view()

        CNN.forward_gpu[BATCH](
            ctx, output_t, input_t, params, state.model_state_view(), cache_t, ws_buf
        )
        ctx.enqueue_copy(out_host, output_buf)

        # Compute loss value (device-side)
        CrossEntropyLoss.forward_gpu[BATCH, OUT_DIM](
            ctx, loss_t, output_t, target_t
        )
        ctx.enqueue_copy(loss_host, loss_buf)
        ctx.synchronize()

        # Summarize logits + loss in one line
        var max_logit: Float64 = 0.0
        for i in range(BATCH * OUT_DIM):
            var v = Float64(out_host.unsafe_ptr()[i])
            if v != v:
                max_logit = 1e30
                break
            var av = v if v >= 0.0 else -v
            if av > max_logit:
                max_logit = av
        var loss_f = Float64(loss_host[0])
        # Only print every 20 steps OR if logits explode
        var is_spike = (max_logit > 100.0) or (max_logit != max_logit)
        if (step + 1) % 20 == 0 or is_spike or step < 5:
            print("step " + String(step + 1)
                  + "  bi=" + String(batch_idx)
                  + "  max|logit|=" + String(max_logit)
                  + "  loss=" + String(loss_f))
            if is_spike:
                print("  !!! LOGIT SPIKE !!!")
                break

        # Loss backward
        CrossEntropyLoss.backward_gpu[BATCH, OUT_DIM](
            ctx, grad_out_t, output_t, target_t
        )

        # Model backward
        CNN.backward_gpu[BATCH](
            ctx, grad_in_t, grad_out_t, params, state.model_state_view(), cache_t, grads, ws_buf
        )

        # Snapshot grads + params + state
        ctx.enqueue_copy(params_dl, state.params_buf)
        ctx.enqueue_copy(grads_dl, state.grads_buf)
        ctx.enqueue_copy(state_dl, state.model_state_buf)
        ctx.synchronize()

        # Compact summary of L0 rvar, L0 dW, L11 grads, L11 params
        var off_L0 = 0
        var off_L11 = (
            gpu_align(L0.PARAM_SIZE)
            + gpu_align(L1.PARAM_SIZE)
            + gpu_align(0)  # maxpool
            + gpu_align(L3.PARAM_SIZE)
            + gpu_align(L4.PARAM_SIZE)
            + gpu_align(0)
            + gpu_align(L6.PARAM_SIZE)
            + gpu_align(L7.PARAM_SIZE)
            + gpu_align(0)
            + gpu_align(0)  # flatten
            + gpu_align(L10.PARAM_SIZE)
        )
        # rvar lives in model_state (state-relative offsets), L0 state offset = 0.
        var sof_L0 = 0
        var rvar_ptr = state_dl.unsafe_ptr() + sof_L0 + L0.RVAR_OFF
        var max_rvar: Float64 = 0.0
        for i in range(L0.out_channels):
            var v = Float64(rvar_ptr[i])
            var av = v if v >= 0.0 else -v
            if av > max_rvar:
                max_rvar = av

        var max_dW_L0: Float64 = 0.0
        var dW_L0 = grads_dl.unsafe_ptr() + off_L0 + L0.W_OFF
        for i in range(L0.CONV_W_SIZE):
            var v = Float64(dW_L0[i])
            var av = v if v >= 0.0 else -v
            if av > max_dW_L0:
                max_dW_L0 = av

        var max_L11_grad: Float64 = 0.0
        var L11_g = grads_dl.unsafe_ptr() + off_L11
        for i in range(L11.PARAM_SIZE):
            var v = Float64(L11_g[i])
            var av = v if v >= 0.0 else -v
            if av > max_L11_grad:
                max_L11_grad = av

        var max_L11_param: Float64 = 0.0
        var L11_p = params_dl.unsafe_ptr() + off_L11
        for i in range(L11.PARAM_SIZE):
            var v = Float64(L11_p[i])
            var av = v if v >= 0.0 else -v
            if av > max_L11_param:
                max_L11_param = av

        if (step + 1) % 10 == 0 or is_spike or step < 5:
            print("       rvar_L0=" + String(max_rvar)[byte=:10]
                  + "  dW_L0="  + String(max_dW_L0)[byte=:10]
                  + "  dL11=" + String(max_L11_grad)[byte=:10]
                  + "  pL11=" + String(max_L11_param)[byte=:10])

        # Optimizer step
        state.optimizer_step(ctx)
        ctx.synchronize()

    print("\n=== done ===")
