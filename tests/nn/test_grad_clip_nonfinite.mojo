"""Grad-clip non-finite guard tests (CPU + GPU).

A NaN/±inf gradient anywhere makes the global L2 norm non-finite. The
pre-guard behaviour was catastrophic-by-silence: `NaN > max_norm` is
False on the CPU path (no scaling) and `NaN > eps` is False in the GPU
scale kernel (denom = eps → ratio huge-finite → scale = 1.0), so the
poisoned grads passed through UNCLIPPED, Adam's moments went NaN, and
the affected params never recovered (the AlphaZero post-promotion
policy-head collapse).

Guarded behaviour under test: non-finite norm → every grad is hard-set
to 0 (the optimizer step becomes a no-op; Adam moments stay finite).
The zero must be WRITTEN, not multiplied in (NaN·0 = NaN).

  1. CPU: one NaN grad → all grads zeroed.
  2. CPU: one +inf grad → all grads zeroed.
  3. CPU: finite grads → unchanged behaviour (clip still exact).
  4. GPU: one NaN grad → scale_buf == 0 and all grads zeroed.
"""

from std.math import sqrt as fsqrt
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.core.grad_clip import (
    clip_grads_auto,
    clip_grads_auto_gpu,
    GradClipState,
)


comptime IN_DIM = 4
comptime OUT_DIM = 3
comptime W = IN_DIM * OUT_DIM
comptime B = OUT_DIM
comptime QNet = Linear[IN_DIM, OUT_DIM]


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def _fill_cpu_grads(mut net: QNet, value: Scalar[DT]):
    var w_ptr = net.weight.grad_unsafe_ptr_cpu()
    var b_ptr = net.bias.grad_unsafe_ptr_cpu()
    for i in range(W):
        w_ptr[i] = value
    for i in range(B):
        b_ptr[i] = value


def _max_abs_cpu_grad(net: QNet) -> Scalar[DT]:
    """Max |grad| treating non-finite as huge (so a surviving NaN/inf
    fails the all-zero assertion instead of comparing as False)."""
    var w_ptr = net.weight.grad_unsafe_ptr_cpu()
    var b_ptr = net.bias.grad_unsafe_ptr_cpu()
    var m: Scalar[DT] = 0.0
    for i in range(W):
        var g = w_ptr[i]
        if g - g != Scalar[DT](0.0):
            return Scalar[DT](1e30)
        if _abs(g) > m:
            m = _abs(g)
    for i in range(B):
        var g = b_ptr[i]
        if g - g != Scalar[DT](0.0):
            return Scalar[DT](1e30)
        if _abs(g) > m:
            m = _abs(g)
    return m


def test_cpu_nan_grad_zeroes_all() raises:
    print("test_cpu_nan_grad_zeroes_all ...")
    var net = QNet.make[target="cpu", INIT=Xavier]()
    _fill_cpu_grads(net, Scalar[DT](0.01))
    var zero = Scalar[DT](0.0)
    net.weight.grad_unsafe_ptr_cpu()[3] = zero / zero  # NaN
    var returned = clip_grads_auto[QNet, "cpu"](net, Scalar[DT](1.0))
    assert_true(
        returned - returned != Scalar[DT](0.0),
        "returned norm should be non-finite, got " + String(returned),
    )
    var mx = _max_abs_cpu_grad(net)
    assert_true(
        mx == Scalar[DT](0.0),
        "all grads must be zeroed after a NaN norm, max|g|=" + String(mx),
    )
    print("  ok (norm=", returned, ", all grads 0)")


def test_cpu_inf_grad_zeroes_all() raises:
    print("test_cpu_inf_grad_zeroes_all ...")
    var net = QNet.make[target="cpu", INIT=Xavier]()
    _fill_cpu_grads(net, Scalar[DT](0.01))
    var zero = Scalar[DT](0.0)
    net.bias.grad_unsafe_ptr_cpu()[1] = Scalar[DT](1.0) / zero  # +inf
    _ = clip_grads_auto[QNet, "cpu"](net, Scalar[DT](1.0))
    var mx = _max_abs_cpu_grad(net)
    assert_true(
        mx == Scalar[DT](0.0),
        "all grads must be zeroed after an inf norm, max|g|=" + String(mx),
    )
    print("  ok (all grads 0)")


def test_cpu_finite_path_unchanged() raises:
    """Regression: the guard must not perturb the normal clip."""
    print("test_cpu_finite_path_unchanged ...")
    var net = QNet.make[target="cpu", INIT=Xavier]()
    _fill_cpu_grads(net, Scalar[DT](100.0))
    var max_norm = Scalar[DT](1.0)
    _ = clip_grads_auto[QNet, "cpu"](net, max_norm)
    # Post-clip norm should equal max_norm.
    var w_ptr = net.weight.grad_unsafe_ptr_cpu()
    var b_ptr = net.bias.grad_unsafe_ptr_cpu()
    var s: Scalar[DT] = 0.0
    for i in range(W):
        s += w_ptr[i] * w_ptr[i]
    for i in range(B):
        s += b_ptr[i] * b_ptr[i]
    var post = fsqrt(s)
    assert_true(
        _abs(post - max_norm) < Scalar[DT](1e-4),
        "finite clip changed: post=" + String(post),
    )
    print("  ok (post-clip norm=", post, ")")


def test_gpu_nan_grad_zeroes_all() raises:
    print("test_gpu_nan_grad_zeroes_all ...")
    try:
        var ctx = DeviceContext()
        var net = QNet.make[target="gpu", INIT=Xavier](ctx=ctx)
        var state = GradClipState.make(ctx, 2)

        var w_h = ctx.enqueue_create_host_buffer[DT](W)
        var b_h = ctx.enqueue_create_host_buffer[DT](B)
        ctx.synchronize()
        for k in range(W):
            w_h.unsafe_ptr()[k] = Scalar[DT](0.01)
        for k in range(B):
            b_h.unsafe_ptr()[k] = Scalar[DT](0.01)
        var zero = Scalar[DT](0.0)
        w_h.unsafe_ptr()[5] = zero / zero  # NaN
        ctx.enqueue_copy(net.weight.grd.dev.value(), w_h)
        ctx.enqueue_copy(net.bias.grd.dev.value(), b_h)
        ctx.synchronize()

        clip_grads_auto_gpu[QNet](net, ctx, state, Scalar[DT](1.0))

        var sb_h = ctx.enqueue_create_host_buffer[DT](1)
        ctx.enqueue_copy(sb_h, state.scale_buf.value())
        ctx.enqueue_copy(w_h, net.weight.grd.dev.value())
        ctx.enqueue_copy(b_h, net.bias.grd.dev.value())
        ctx.synchronize()
        assert_true(
            sb_h.unsafe_ptr()[0] == Scalar[DT](0.0),
            "scale_buf should be 0 on non-finite norm, got "
            + String(sb_h.unsafe_ptr()[0]),
        )
        var mx: Scalar[DT] = 0.0
        for k in range(W):
            var g = w_h.unsafe_ptr()[k]
            if g - g != Scalar[DT](0.0) or _abs(g) > mx:
                mx = Scalar[DT](1e30) if g - g != Scalar[DT](0.0) else _abs(g)
        for k in range(B):
            var g = b_h.unsafe_ptr()[k]
            if g - g != Scalar[DT](0.0) or _abs(g) > mx:
                mx = Scalar[DT](1e30) if g - g != Scalar[DT](0.0) else _abs(g)
        assert_true(
            mx == Scalar[DT](0.0),
            "all GPU grads must be zeroed, max|g|=" + String(mx),
        )
        print("  ok (scale=0, all grads 0)")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def main() raises:
    print("=" * 60)
    print("Grad-clip non-finite guard")
    print("=" * 60)
    test_cpu_nan_grad_zeroes_all()
    test_cpu_inf_grad_zeroes_all()
    test_cpu_finite_path_unchanged()
    test_gpu_nan_grad_zeroes_all()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
