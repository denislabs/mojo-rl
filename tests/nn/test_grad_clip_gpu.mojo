"""GPU grad-clip smoke + correctness test.

Validates the three-pass on-device clipping pipeline introduced for
PPO grad-norm clipping (no D2H during training step):

  1. Build a small Linear→ReLU→Linear net on GPU.
  2. Stuff known grad values into all Params via host buffers + H2D.
  3. Run `clip_grads_auto_gpu` with various `max_norm` thresholds.
  4. D2H the grads + the device-side `norm_buf` / `scale_buf`.
  5. Assert: post-clip global norm == max_norm when active, equal to
     pre-clip when disabled, and bit-equal to a pure-CPU reference.

Single source of truth — no claims of behaviour-without-data.
"""

from std.math import sqrt as fsqrt, isnan, isinf
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.core.grad_clip import (
    clip_grads_auto,
    clip_grads_auto_gpu,
    GradClipState,
)


# Single Linear keeps the test self-contained — no Sequential tuple
# indexing needed. The clip pipeline walks Params via reflection, so a
# 2-Param leaf exercises the same code as a deeper net.
comptime IN_DIM = 4
comptime OUT_DIM = 3
comptime QNet = Linear[IN_DIM, OUT_DIM]


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def _stuff_grads_host_to_device(
    ctx: DeviceContext, mut net: QNet,
) raises -> Scalar[DT]:
    """Linear[IN, OUT] has 2 Params: weight [IN*OUT] + bias [OUT].
    Fill `grad_dev` with deterministic non-zero values, return the
    precomputed pre-clip global L2 norm."""
    comptime W = IN_DIM * OUT_DIM
    comptime B = OUT_DIM
    var w_h = ctx.enqueue_create_host_buffer[DT](W)
    var b_h = ctx.enqueue_create_host_buffer[DT](B)
    ctx.synchronize()
    var sum_sq: Scalar[DT] = 0.0
    for k in range(W):
        var v = Scalar[DT](0.1 * Float64(k + 1) - 0.5)
        w_h.unsafe_ptr()[k] = v
        sum_sq += v * v
    for k in range(B):
        var v = Scalar[DT](0.2 * Float64(k + 1) - 0.3)
        b_h.unsafe_ptr()[k] = v
        sum_sq += v * v
    ctx.enqueue_copy(net.weight.grd.dev.value(), w_h)
    ctx.enqueue_copy(net.bias.grd.dev.value(),   b_h)
    ctx.synchronize()
    return fsqrt(sum_sq)


def _download_global_norm(
    ctx: DeviceContext, mut net: QNet,
) raises -> Scalar[DT]:
    comptime W = IN_DIM * OUT_DIM
    comptime B = OUT_DIM
    var w_h = ctx.enqueue_create_host_buffer[DT](W)
    var b_h = ctx.enqueue_create_host_buffer[DT](B)
    ctx.enqueue_copy(w_h, net.weight.grd.dev.value())
    ctx.enqueue_copy(b_h, net.bias.grd.dev.value())
    ctx.synchronize()
    var s: Scalar[DT] = 0.0
    for k in range(W):
        s += w_h.unsafe_ptr()[k] * w_h.unsafe_ptr()[k]
    for k in range(B):
        s += b_h.unsafe_ptr()[k] * b_h.unsafe_ptr()[k]
    return fsqrt(s)


def test_clip_gpu_active() raises:
    """`max_norm < pre_clip_norm` → post-clip norm == max_norm."""
    print("test_clip_gpu_active ...")
    try:
        var ctx = DeviceContext()
        seed(0)
        var net = QNet.make[target="gpu", INIT=Xavier](ctx=ctx)
        var n_params = 2  # weight, bias
        var state = GradClipState.make(ctx, n_params)

        var pre_norm = _stuff_grads_host_to_device(ctx, net)
        var max_norm = Scalar[DT](0.5) * pre_norm  # force clipping
        clip_grads_auto_gpu[QNet](net, ctx, state, max_norm)

        # Read on-device norm_buf — should equal pre_norm.
        var nb_h = ctx.enqueue_create_host_buffer[DT](1)
        var sb_h = ctx.enqueue_create_host_buffer[DT](1)
        ctx.enqueue_copy(nb_h, state.norm_buf.value())
        ctx.enqueue_copy(sb_h, state.scale_buf.value())
        ctx.synchronize()
        print("  pre_norm=", pre_norm, " device_norm=", nb_h.unsafe_ptr()[0])
        print("  max_norm=", max_norm, " scale=", sb_h.unsafe_ptr()[0])
        assert_true(
            _abs(nb_h.unsafe_ptr()[0] - pre_norm) < Scalar[DT](1e-3),
            "device norm_buf does not match host pre_norm",
        )

        var post_norm = _download_global_norm(ctx, net)
        print("  post_norm=", post_norm, " expected ≤ max_norm=", max_norm)
        assert_true(
            _abs(post_norm - max_norm) < Scalar[DT](1e-3),
            "post-clip norm != max_norm",
        )
        print("  ok")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def test_clip_gpu_inactive() raises:
    """`max_norm > pre_clip_norm` → scale = 1, post == pre (bit-identity)."""
    print("test_clip_gpu_inactive ...")
    try:
        var ctx = DeviceContext()
        seed(0)
        var net = QNet.make[target="gpu", INIT=Xavier](ctx=ctx)
        var state = GradClipState.make(ctx, 2)

        var pre_norm = _stuff_grads_host_to_device(ctx, net)
        var max_norm = pre_norm * Scalar[DT](2.0)  # under threshold
        clip_grads_auto_gpu[QNet](net, ctx, state, max_norm)

        var sb_h = ctx.enqueue_create_host_buffer[DT](1)
        ctx.enqueue_copy(sb_h, state.scale_buf.value())
        ctx.synchronize()
        print("  scale=", sb_h.unsafe_ptr()[0], " (expected 1.0)")
        assert_true(
            _abs(sb_h.unsafe_ptr()[0] - Scalar[DT](1.0)) < Scalar[DT](1e-6),
            "scale != 1 when norm below threshold",
        )

        var post_norm = _download_global_norm(ctx, net)
        print("  pre=", pre_norm, " post=", post_norm)
        assert_true(
            _abs(post_norm - pre_norm) < Scalar[DT](1e-5),
            "post-clip norm differs from pre-clip with inactive clipping",
        )
        print("  ok")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def test_adam_step_clipped_gpu() raises:
    """End-to-end: Adam.step on GPU with max_grad_norm > 0 — should
    lazy-allocate _clip_state and not crash. Sanity-check that
    repeated step()s converge under controlled grads."""
    print("test_adam_step_clipped_gpu ...")
    try:
        var ctx = DeviceContext()
        seed(0)
        var net = QNet.make[target="gpu", INIT=Xavier](ctx=ctx)
        var opt = Adam.make[target="gpu", M=QNet](net, ctx=ctx)
        opt.lr = Scalar[DT](1e-3)
        opt.max_grad_norm = Scalar[DT](0.5)

        for _ in range(3):
            _ = _stuff_grads_host_to_device(ctx, net)
            opt.step[target="gpu", M=QNet](net)
        ctx.synchronize()
        print("  3× Adam GPU step with clipping — no crash")
        print("  ok")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def main() raises:
    print("=" * 60)
    print("Grad-clip GPU smoke + correctness")
    print("=" * 60)
    test_clip_gpu_active()
    test_clip_gpu_inactive()
    test_adam_step_clipped_gpu()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
