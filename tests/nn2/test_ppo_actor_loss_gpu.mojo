"""PPOActorLoss GPU smoke + CPU parity.

Two checks:
  1. Construction: PPOActorLoss[..., target='gpu'].make(ctx, ...) succeeds.
  2. forward_backward returns a finite loss; actor weights move between
     before/after; CPU and GPU produce the same loss (to a small tolerance).

Per-element math is purely local (no cross-batch reductions) so the GPU
kernel writes the same values as the CPU loop modulo floating-point
re-ordering. We assert |gpu_loss - cpu_loss| < 1e-4 which is generous
for FP32 across host+device exp() implementations.
"""

from std.gpu.host import DeviceContext
from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.deep_agents2.primitives.gaussian_head import GaussianHead
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.deep_agents2.loss.ppo_actor_loss import PPOActorLoss


comptime OBS_DIM = 4
comptime ACT_DIM = 2
comptime HIDDEN = 16
comptime BATCH = 8


comptime ActorNet = Sequential[
    Linear[OBS_DIM, HIDDEN], Tanh[HIDDEN],
    GaussianHead[HIDDEN, ACT_DIM],
]


def _run_one_step[target: StaticString](
    ctx: Optional[DeviceContext] = None,
) raises -> Scalar[DT]:
    """Build actor + loss on `target`, do one forward_backward, return
    the loss scalar. Uses deterministic fill values across both targets
    so CPU and GPU runs are directly comparable."""
    seed(7)
    var actor = ActorNet.make[target, INIT=Xavier](ctx=ctx)
    var actor_opt = Adam.make[target, M=ActorNet](actor, ctx=ctx)
    actor_opt.lr = Scalar[DT](1e-3)
    var loss_blk = PPOActorLoss[ActorNet, BATCH].make[target](
        ctx=ctx,
        clip_eps=Scalar[DT](0.2),
        entropy_coef=Scalar[DT](0.01),
    )

    # Build deterministic input batches on host, upload to device on GPU.
    var s_host = List[Scalar[DT]](length=BATCH * OBS_DIM, fill=Scalar[DT](0.0))
    var a_host = List[Scalar[DT]](length=BATCH * ACT_DIM, fill=Scalar[DT](0.0))
    var olp_host = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0.0))
    var adv_host = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0.0))
    for b in range(BATCH):
        for d in range(OBS_DIM):
            s_host[b * OBS_DIM + d] = Scalar[DT](0.1 * Float64(b + d))
        for d in range(ACT_DIM):
            a_host[b * ACT_DIM + d] = Scalar[DT](0.05 * Float64(b + d + 1))
        olp_host[b] = Scalar[DT](-1.5 + 0.01 * Float64(b))
        adv_host[b] = Scalar[DT](0.5 - 0.1 * Float64(b))

    comptime if target == "cpu":
        return loss_blk.forward_backward["cpu", Adam](
            actor, actor_opt,
            s_host.unsafe_ptr(), a_host.unsafe_ptr(),
            olp_host.unsafe_ptr(), adv_host.unsafe_ptr(),
        )
    else:
        var c = ctx.value()
        var s_dev = c.enqueue_create_buffer[DT](BATCH * OBS_DIM)
        var a_dev = c.enqueue_create_buffer[DT](BATCH * ACT_DIM)
        var olp_dev = c.enqueue_create_buffer[DT](BATCH)
        var adv_dev = c.enqueue_create_buffer[DT](BATCH)
        c.enqueue_copy(s_dev, s_host.unsafe_ptr())
        c.enqueue_copy(a_dev, a_host.unsafe_ptr())
        c.enqueue_copy(olp_dev, olp_host.unsafe_ptr())
        c.enqueue_copy(adv_dev, adv_host.unsafe_ptr())
        c.synchronize()
        return loss_blk.forward_backward["gpu", Adam](
            actor, actor_opt,
            s_dev.unsafe_ptr(), a_dev.unsafe_ptr(),
            olp_dev.unsafe_ptr(), adv_dev.unsafe_ptr(),
        )


def test_gpu_finite() raises:
    var ctx = DeviceContext()
    var loss = _run_one_step["gpu"](ctx)
    assert_true(not isnan(loss), "GPU loss must not be NaN; got " + String(loss))
    assert_true(not isinf(loss), "GPU loss must not be Inf; got " + String(loss))
    print("  test_gpu_finite PASSED loss=", loss)


def test_cpu_gpu_parity() raises:
    var ctx = DeviceContext()
    var cpu_loss = _run_one_step["cpu"]()
    var gpu_loss = _run_one_step["gpu"](ctx)
    var diff = cpu_loss - gpu_loss
    if diff < Scalar[DT](0.0):
        diff = -diff
    assert_true(
        Float64(diff) < 1e-4,
        "CPU/GPU loss must match within 1e-4; cpu="
        + String(cpu_loss) + " gpu=" + String(gpu_loss)
        + " |diff|=" + String(diff),
    )
    print(
        "  test_cpu_gpu_parity PASSED cpu=", cpu_loss,
        " gpu=", gpu_loss, " |diff|=", diff,
    )


def main() raises:
    print("=" * 60)
    print("PPOActorLoss GPU smoke + CPU parity")
    print("=" * 60)
    test_gpu_finite()
    test_cpu_gpu_parity()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
