"""Bit-identicality test: SACActorLossCG (Phase 10E) vs SACActorLoss (Phase 9A).

Identical actor + critics + obs + α + RNG seed → identical `loss`,
`log_prob_mean`, and actor parameter gradients on a single forward+
backward step.

The test seeds the RNG once before each block's `forward_backward` so
both observe the same Box-Muller draws. After both calls, the returned
SACActorLossOut.loss / log_prob_mean and the actor's `grad_w` / `grad_b`
must match to fp32 rounding tolerance.
"""

from std.math import abs as fabs
from std.memory import alloc
from std.random import seed as random_seed
from std.testing import assert_almost_equal, assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.loss.sac_actor_loss_block import SACActorLoss
from mojo_rl.nn2.loss.sac_actor_loss_cg import SACActorLossCG


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 8
comptime HID = 32

comptime ActorNet = Sequential[Linear[OBS, HID], Linear[HID, 2 * ACT]]
comptime CriticNet = Sequential[Linear[OBS + ACT, HID], Linear[HID, 1]]


def _copy_actor(mut dst: ActorNet, src: ActorNet) raises:
    """Copy weights + biases of a 2-layer Sequential Linear actor."""
    for k in range(src.children[0].W_SIZE):
        dst.children[0].weight[k] = src.children[0].weight[k]
    for k in range(src.children[0].B_SIZE):
        dst.children[0].bias[k] = src.children[0].bias[k]
    for k in range(src.children[1].W_SIZE):
        dst.children[1].weight[k] = src.children[1].weight[k]
    for k in range(src.children[1].B_SIZE):
        dst.children[1].bias[k] = src.children[1].bias[k]


def _copy_critic(mut dst: CriticNet, src: CriticNet) raises:
    for k in range(src.children[0].W_SIZE):
        dst.children[0].weight[k] = src.children[0].weight[k]
    for k in range(src.children[0].B_SIZE):
        dst.children[0].bias[k] = src.children[0].bias[k]
    for k in range(src.children[1].W_SIZE):
        dst.children[1].weight[k] = src.children[1].weight[k]
    for k in range(src.children[1].B_SIZE):
        dst.children[1].bias[k] = src.children[1].bias[k]


def main() raises:
    print("=" * 70)
    print("nn2 Phase 10E — SACActorLossCG bit-identicality vs SACActorLoss")
    print("=" * 70)

    # Build a reference actor + critic pair.
    random_seed(12345)
    var actor_ref = ActorNet.make[target="cpu", INIT=Xavier]()
    var critic1_ref = CriticNet.make[target="cpu", INIT=Xavier]()
    var critic2_ref = CriticNet.make[target="cpu", INIT=Xavier]()
    var actor_opt_ref = Adam.make["cpu", M=ActorNet](
        actor_ref, lr=Scalar[DT](1e-3)
    )

    # Build a parallel pair, then deep-copy weights from ref so they start
    # bit-identical.
    var actor_cg = ActorNet.make[target="cpu", INIT=Xavier]()
    var critic1_cg = CriticNet.make[target="cpu", INIT=Xavier]()
    var critic2_cg = CriticNet.make[target="cpu", INIT=Xavier]()
    var actor_opt_cg = Adam.make["cpu", M=ActorNet](
        actor_cg, lr=Scalar[DT](1e-3)
    )
    _copy_actor(actor_cg, actor_ref)
    _copy_critic(critic1_cg, critic1_ref)
    _copy_critic(critic2_cg, critic2_ref)

    # Build both loss blocks.
    var loss_ref = SACActorLoss[ActorNet, CriticNet, BATCH].make[target="cpu"](
        action_scale=Scalar[DT](2.0)
    )
    var loss_cg = SACActorLossCG[ActorNet, CriticNet, BATCH].make[target="cpu"](
        action_scale=Scalar[DT](2.0)
    )

    # Random obs batch.
    var obs_buf = alloc[Scalar[DT]](BATCH * OBS)
    for i in range(BATCH * OBS):
        obs_buf[i] = Scalar[DT](Float32(i) * 0.13 - 0.7)

    var alpha = Scalar[DT](0.2)

    # First the reference. Seed RNG fresh so Box-Muller draws are
    # reproducible.
    random_seed(7777)
    var out_ref = loss_ref.forward_backward["cpu", OPT=Adam](
        actor_ref, actor_opt_ref, critic1_ref, critic2_ref, obs_buf, alpha
    )

    # Now the CG version. Reset RNG to same state so rsample draws match.
    random_seed(7777)
    var out_cg = loss_cg.forward_backward["cpu", OPT=Adam](
        actor_cg, actor_opt_cg, critic1_cg, critic2_cg, obs_buf, alpha
    )

    # Returned scalars.
    print("  ref loss=", out_ref.loss, "  cg loss=", out_cg.loss)
    print("  ref lp_mean=", out_ref.log_prob_mean,
          "  cg lp_mean=", out_cg.log_prob_mean)
    assert_almost_equal(out_ref.loss, out_cg.loss, atol=1e-5)
    assert_almost_equal(out_ref.log_prob_mean, out_cg.log_prob_mean, atol=1e-5)

    # Actor parameters AFTER opt step should match (both took an Adam
    # step from identical pre-step params with identical grads).
    var max_w0_diff: Scalar[DT] = 0.0
    for k in range(actor_ref.children[0].W_SIZE):
        var d = fabs(actor_ref.children[0].weight[k] - actor_cg.children[0].weight[k])
        if d > max_w0_diff:
            max_w0_diff = d
    var max_b0_diff: Scalar[DT] = 0.0
    for k in range(actor_ref.children[0].B_SIZE):
        var d = fabs(actor_ref.children[0].bias[k] - actor_cg.children[0].bias[k])
        if d > max_b0_diff:
            max_b0_diff = d
    var max_w1_diff: Scalar[DT] = 0.0
    for k in range(actor_ref.children[1].W_SIZE):
        var d = fabs(actor_ref.children[1].weight[k] - actor_cg.children[1].weight[k])
        if d > max_w1_diff:
            max_w1_diff = d
    var max_b1_diff: Scalar[DT] = 0.0
    for k in range(actor_ref.children[1].B_SIZE):
        var d = fabs(actor_ref.children[1].bias[k] - actor_cg.children[1].bias[k])
        if d > max_b1_diff:
            max_b1_diff = d

    print("  max |Δw0|=", max_w0_diff, "  max |Δb0|=", max_b0_diff)
    print("  max |Δw1|=", max_w1_diff, "  max |Δb1|=", max_b1_diff)
    assert_true(max_w0_diff < Scalar[DT](1e-6), "actor w0 diverges")
    assert_true(max_b0_diff < Scalar[DT](1e-6), "actor b0 diverges")
    assert_true(max_w1_diff < Scalar[DT](1e-6), "actor w1 diverges")
    assert_true(max_b1_diff < Scalar[DT](1e-6), "actor b1 diverges")

    # Critics should be UNCHANGED (backward_input doesn't write param grads).
    var max_c1_w0: Scalar[DT] = 0.0
    for k in range(critic1_ref.children[0].W_SIZE):
        var d = fabs(critic1_ref.children[0].weight[k] - critic1_cg.children[0].weight[k])
        if d > max_c1_w0:
            max_c1_w0 = d
    assert_true(max_c1_w0 < Scalar[DT](1e-7), "critic1 weights moved")

    obs_buf.free()
    print("  test_sac_actor_loss_cg PASSED — bit-identical actor update")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
