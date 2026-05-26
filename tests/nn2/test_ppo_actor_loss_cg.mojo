"""PPOActorLossCG FullGraph — parity vs hand-rolled bespoke pipeline.

Phase I.2.c. The hand-rolled pendulum_ppo_nn2 example pipeline does:
  actor.forward → bespoke PPOActorLoss.forward / vjp → actor.vjp → step

The PPOActorLossCG block collapses this to a single `forward_backward`
call. After one step, both actor instances (initialised identically)
must hold bit-identical parameters.

Smaller scale than the full Pendulum loop, but exercises the FullGraph
end-to-end: ExternalNode binding, multi-InputSlot, hetero-variadic Node,
seed_grad_inv_batch, and the actor's autodiff path.
"""

from std.math import abs as mojo_abs
from std.memory import alloc
from std.random import seed, random_float64
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.nn2.primitives.gaussian_head import GaussianHead
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.loss.ppo_actor_loss import PPOActorLoss
from mojo_rl.nn2.loss.ppo_actor_loss_cg import PPOActorLossCG


comptime OBS = 3
comptime ACT = 1
comptime HIDDEN = 32
comptime BATCH = 16
comptime AUX = ACT + 2

comptime ActorNet = Sequential[
    Linear[OBS, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN], Tanh[HIDDEN],
    GaussianHead[HIDDEN, ACT],
]


def _copy_params(
    mut src: ActorNet, mut dst: ActorNet
) raises:
    """Copy actor params src → dst. Mirrors `state_walker` round-trip."""
    # Easiest: forward both and re-init dst from src's checkpoint surface.
    # The simplest in-test method: write the same initial state by
    # seeding both actors with the same seed before construction.
    pass


def test_one_step_parity() raises:
    print("test_one_step_parity ...")

    # Build two actors with identical initial weights (same seed).
    seed(123)
    var actor_a = ActorNet.make[target="cpu", INIT=Xavier]()
    seed(123)
    var actor_b = ActorNet.make[target="cpu", INIT=Xavier]()

    var opt_a = Adam.make[target="cpu", M=ActorNet](actor_a)
    opt_a.lr = Scalar[DT](3e-4)
    var opt_b = Adam.make[target="cpu", M=ActorNet](actor_b)
    opt_b.lr = Scalar[DT](3e-4)

    var bespoke = PPOActorLoss[ACT].make["cpu"](
        clip_eps=Scalar[DT](0.2), entropy_coef=Scalar[DT](0.01)
    )
    var cg = PPOActorLossCG[ActorNet, BATCH].make["cpu"](
        clip_eps=Scalar[DT](0.2), entropy_coef=Scalar[DT](0.01)
    )

    # Build a minibatch.
    seed(7)
    var s: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OBS)
    var act: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * ACT)
    var olp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var adv: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var ao: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OBS)

    for b in range(BATCH):
        for d in range(OBS):
            s[b * OBS + d] = Scalar[DT](-1.0 + 2.0 * random_float64())
        for j in range(ACT):
            act[b * ACT + j] = Scalar[DT](-1.0 + 2.0 * random_float64())
        olp[b] = Scalar[DT](-1.0 + 2.0 * random_float64())
        adv[b] = Scalar[DT](-2.0 + 4.0 * random_float64())

    var s_t = TileTensor(s, row_major[BATCH, OBS]())
    var act_t = TileTensor(act, row_major[BATCH, ACT]())
    var olp_t = TileTensor(olp, row_major[BATCH]())
    var adv_t = TileTensor(adv, row_major[BATCH]())
    var ao_t = TileTensor(ao, row_major[BATCH, 2 * ACT]())
    var go_t = TileTensor(go, row_major[BATCH, 2 * ACT]())
    var gi_t = TileTensor(gi, row_major[BATCH, OBS]())

    # ── Bespoke pipeline ─────────────────────────────────────────────
    actor_a.forward["cpu", BATCH](s_t, output=ao_t)
    var loss_bespoke = bespoke.forward["cpu", BATCH](
        ao_t, act_t, olp_t, adv_t
    )
    bespoke.vjp["cpu", BATCH](ao_t, act_t, olp_t, adv_t, go_t)
    opt_a.zero_grad["cpu", M=ActorNet](actor_a)
    actor_a.vjp["cpu", BATCH](go_t, gi_t)
    opt_a.step["cpu", M=ActorNet](actor_a)

    # ── FullGraph pipeline (quaternary, post-I.2.5) ──────────────────
    var loss_cg = cg.forward_backward[target="cpu", OPT=Adam](
        actor_b, opt_b, s, act, olp, adv,
    )

    print("  bespoke loss =", loss_bespoke, "  cg loss =", loss_cg)
    var loss_diff = loss_bespoke - loss_cg
    if loss_diff < Scalar[DT](0.0):
        loss_diff = -loss_diff
    assert_true(loss_diff < Scalar[DT](1e-5),
                "PPOActorLossCG forward loss must match bespoke")

    # ── Compare resulting params via second forward; states should
    #    be identical so a fresh forward on the same s yields the same ao.
    var ao2_a: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var ao2_b: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var ao2a_t = TileTensor(ao2_a, row_major[BATCH, 2 * ACT]())
    var ao2b_t = TileTensor(ao2_b, row_major[BATCH, 2 * ACT]())
    actor_a.forward["cpu", BATCH](s_t, output=ao2a_t)
    actor_b.forward["cpu", BATCH](s_t, output=ao2b_t)
    var max_diff: Scalar[DT] = 0.0
    for k in range(BATCH * 2 * ACT):
        var d = ao2_a[k] - ao2_b[k]
        if d < Scalar[DT](0.0):
            d = -d
        if d > max_diff:
            max_diff = d
    print("  post-step actor_output max |Δ| =", max_diff)
    assert_true(max_diff < Scalar[DT](1e-5),
                "PPOActorLossCG step must produce identical actor params "
                "to bespoke pipeline")

    s.free(); act.free(); olp.free(); adv.free()
    ao.free(); go.free(); gi.free(); ao2_a.free(); ao2_b.free()
    print("  ok")


def main() raises:
    print("=" * 70)
    print("PPOActorLossCG one-step parity vs hand-rolled bespoke (Phase I.2.c)")
    print("=" * 70)
    test_one_step_parity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
