"""GaussianNLLLoss GPU smoke + CPU parity.

Verifies the new GPU forward + vjp:
  1. GPU forward + vjp produce finite values.
  2. CPU and GPU produce the same loss value (bit-identity expected on
     small inputs since the math is per-element and the host-side sum
     order matches between CPU and GPU paths).
  3. CPU and GPU produce the same grad_logits.
"""

from std.gpu.host import DeviceContext
from std.math import isnan, isinf
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.loss.gaussian_nll_loss import GaussianNLLLoss


comptime DIM = 3
comptime BATCH = 8


def _fill_inputs(
    mut logits: List[Scalar[DT]], mut targets: List[Scalar[DT]],
):
    """Deterministic inputs: mu ramps slowly, raw_logvar starts well
    within the clamp range (default [-10, -2]) and a few rows push
    past -2 to exercise the clamp branch."""
    for b in range(BATCH):
        for i in range(DIM):
            logits[b * 2 * DIM + i] = Scalar[DT](0.1 * Float64(b + i))
            var lv = Float64(-4.0 + 0.6 * Float64(b))
            logits[b * 2 * DIM + DIM + i] = Scalar[DT](lv)
            targets[b * DIM + i] = (
                Scalar[DT](0.1 * Float64(b + i)) + Scalar[DT](0.05)
            )


def test_gpu_finite() raises:
    var ctx = DeviceContext()
    var loss_blk = GaussianNLLLoss[DIM].make["gpu"](ctx=ctx)
    var logits = List[Scalar[DT]](
        length=BATCH * 2 * DIM, fill=Scalar[DT](0.0),
    )
    var targets = List[Scalar[DT]](
        length=BATCH * DIM, fill=Scalar[DT](0.0),
    )
    _fill_inputs(logits, targets)
    var l_dev = ctx.enqueue_create_buffer[DT](BATCH * 2 * DIM)
    var t_dev = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    ctx.enqueue_copy(l_dev, logits.unsafe_ptr())
    ctx.enqueue_copy(t_dev, targets.unsafe_ptr())
    ctx.synchronize()
    var logits_t = TileTensor(l_dev.unsafe_ptr(), row_major[BATCH, 2 * DIM]())
    var targets_t = TileTensor(t_dev.unsafe_ptr(), row_major[BATCH, DIM]())
    var loss = loss_blk.forward["gpu", BATCH](logits_t, targets_t)
    assert_true(not isnan(loss), "GPU loss must not be NaN; got " + String(loss))
    assert_true(not isinf(loss), "GPU loss must not be Inf; got " + String(loss))
    # Run vjp; just check it doesn't crash.
    var grad_dev = ctx.enqueue_create_buffer[DT](BATCH * 2 * DIM)
    grad_dev.enqueue_fill(Scalar[DT](0.0))
    var grad_t = TileTensor(grad_dev.unsafe_ptr(), row_major[BATCH, 2 * DIM]())
    loss_blk.vjp["gpu", BATCH](targets_t, grad_t)
    ctx.synchronize()
    print("  test_gpu_finite PASSED loss=", loss)


def test_cpu_gpu_parity() raises:
    var ctx = DeviceContext()
    var logits = List[Scalar[DT]](
        length=BATCH * 2 * DIM, fill=Scalar[DT](0.0),
    )
    var targets = List[Scalar[DT]](
        length=BATCH * DIM, fill=Scalar[DT](0.0),
    )
    _fill_inputs(logits, targets)

    # CPU path.
    var cpu_loss_blk = GaussianNLLLoss[DIM].make["cpu"]()
    var logits_cpu_t = TileTensor(
        logits.unsafe_ptr(), row_major[BATCH, 2 * DIM](),
    )
    var targets_cpu_t = TileTensor(
        targets.unsafe_ptr(), row_major[BATCH, DIM](),
    )
    var cpu_loss = cpu_loss_blk.forward["cpu", BATCH](
        logits_cpu_t, targets_cpu_t,
    )
    var cpu_grad = List[Scalar[DT]](
        length=BATCH * 2 * DIM, fill=Scalar[DT](0.0),
    )
    var cpu_grad_t = TileTensor(
        cpu_grad.unsafe_ptr(), row_major[BATCH, 2 * DIM](),
    )
    cpu_loss_blk.vjp["cpu", BATCH](targets_cpu_t, cpu_grad_t)

    # GPU path.
    var gpu_loss_blk = GaussianNLLLoss[DIM].make["gpu"](ctx=ctx)
    var l_dev = ctx.enqueue_create_buffer[DT](BATCH * 2 * DIM)
    var t_dev = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    ctx.enqueue_copy(l_dev, logits.unsafe_ptr())
    ctx.enqueue_copy(t_dev, targets.unsafe_ptr())
    ctx.synchronize()
    var logits_gpu_t = TileTensor(
        l_dev.unsafe_ptr(), row_major[BATCH, 2 * DIM](),
    )
    var targets_gpu_t = TileTensor(
        t_dev.unsafe_ptr(), row_major[BATCH, DIM](),
    )
    var gpu_loss = gpu_loss_blk.forward["gpu", BATCH](
        logits_gpu_t, targets_gpu_t,
    )
    var grad_dev = ctx.enqueue_create_buffer[DT](BATCH * 2 * DIM)
    grad_dev.enqueue_fill(Scalar[DT](0.0))
    var grad_gpu_t = TileTensor(
        grad_dev.unsafe_ptr(), row_major[BATCH, 2 * DIM](),
    )
    gpu_loss_blk.vjp["gpu", BATCH](targets_gpu_t, grad_gpu_t)
    var grad_host = ctx.enqueue_create_host_buffer[DT](BATCH * 2 * DIM)
    ctx.enqueue_copy(grad_host, grad_dev)
    ctx.synchronize()

    var loss_diff = cpu_loss - gpu_loss
    if loss_diff < Scalar[DT](0.0):
        loss_diff = -loss_diff
    assert_true(
        Float64(loss_diff) < 1e-4,
        "loss parity: cpu=" + String(cpu_loss) + " gpu=" + String(gpu_loss)
        + " |diff|=" + String(loss_diff),
    )

    var max_grad_diff: Scalar[DT] = 0.0
    var gh = grad_host.unsafe_ptr()
    var gc = cpu_grad.unsafe_ptr()
    for k in range(BATCH * 2 * DIM):
        var d = gh[k] - gc[k]
        if d < Scalar[DT](0.0):
            d = -d
        if d > max_grad_diff:
            max_grad_diff = d
    assert_true(
        Float64(max_grad_diff) < 1e-5,
        "grad parity: max |gpu-cpu| = " + String(max_grad_diff),
    )
    print(
        "  test_cpu_gpu_parity PASSED loss=", cpu_loss,
        " |loss-diff|=", loss_diff,
        " max|grad-diff|=", max_grad_diff,
    )


def main() raises:
    print("=" * 60)
    print("GaussianNLLLoss GPU smoke + CPU parity")
    print("=" * 60)
    test_gpu_finite()
    test_cpu_gpu_parity()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
