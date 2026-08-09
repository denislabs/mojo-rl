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

from max.gpu.host import DeviceContext
from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import Tanh
from mojo_rl.deep_agents.primitives.gaussian_head import GaussianHead
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.core.initializer import Xavier
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.deep_agents.ppo.actor_loss import PPOActorLoss


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
    var actor_opt = Adam(lr=Scalar[DT](1e-3))
    actor_opt.adopt[target, M=ActorNet](actor, ctx)
    var loss_blk = PPOActorLoss[ActorNet, BATCH].make[target](
        ctx=ctx,
        clip_eps=Scalar[DT](0.2),
        entropy_coef=Scalar[DT](0.01),
    )

    # Build deterministic input batches as storage Tensors (host-fill;
    # upload to device on GPU).
    var s_t = Tensor.alloc(BATCH * OBS_DIM)
    var a_t = Tensor.alloc(BATCH * ACT_DIM)
    var olp_t = Tensor.alloc(BATCH)
    var adv_t = Tensor.alloc(BATCH)
    for b in range(BATCH):
        for d in range(OBS_DIM):
            s_t.data[b * OBS_DIM + d] = Scalar[DT](0.1 * Float64(b + d))
        for d in range(ACT_DIM):
            a_t.data[b * ACT_DIM + d] = Scalar[DT](0.05 * Float64(b + d + 1))
        olp_t.data[b] = Scalar[DT](-1.5 + 0.01 * Float64(b))
        adv_t.data[b] = Scalar[DT](0.5 - 0.1 * Float64(b))

    comptime if target == "cpu":
        return loss_blk.forward_backward["cpu"](
            actor, actor_opt, s_t, a_t, olp_t, adv_t
        )
    else:
        var c = ctx.value()
        s_t.upload(c)
        a_t.upload(c)
        olp_t.upload(c)
        adv_t.upload(c)
        return loss_blk.forward_backward["gpu"](
            actor, actor_opt, s_t, a_t, olp_t, adv_t,
            Scalar[DT](0.0), ctx,
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
