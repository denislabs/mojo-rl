"""PPOObjective primitive — parity test against bespoke PPOActorLoss.

Phase I.2.b + I.2.5. The bespoke `PPOActorLoss[ACT]` is the trusted
reference: identical math, unchanged since Phase 6. After I.2.5's
GraphNode N-ary refactor, `PPOObjective[ACT]` is quaternary
(actor_out, action, old_log_prob, advantage) — no more aux packing.

We verify:
  1. The same per-sample loss vector (bespoke sum vs PPOObjective sum).
  2. The same grad_actor_output (mean seed go = 1/BATCH per row).
  3. grad_action / grad_old_log_prob / grad_advantage all zero.
"""

from std.math import abs as mojo_abs
from std.memory import alloc
from std.random import seed, random_float64
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.loss.ppo_actor_loss import PPOActorLoss
from mojo_rl.nn2.primitives.ppo_objective import PPOObjective
from mojo_rl.nn2.initializer import Xavier


comptime ACT = 1
comptime BATCH = 64


def test_forward_parity() raises:
    print("test_forward_parity ...")
    seed(7)

    var ao: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var act: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * ACT)
    var olp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var adv: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var loss_per_b: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)

    for b in range(BATCH):
        for j in range(ACT):
            ao[b * 2 * ACT + j] = Scalar[DT](-1.0 + 2.0 * random_float64())
            ao[b * 2 * ACT + ACT + j] = Scalar[DT](-2.0 + 2.0 * random_float64())
            act[b * ACT + j] = Scalar[DT](-1.0 + 2.0 * random_float64())
        olp[b] = Scalar[DT](-1.0 + 2.0 * random_float64())
        adv[b] = Scalar[DT](-2.0 + 4.0 * random_float64())

    # Hetero-variadic workaround: pass all 4 inputs with IN0_DIM-shaped
    # Layout (2*ACT). PPOObjective.forward typed_views to recover real
    # shapes (ACT for action, 1 for olp/adv).
    var ao_t = TileTensor(ao, row_major[BATCH, 2 * ACT]())
    var act_t = TileTensor(act, row_major[BATCH, 2 * ACT]())
    var olp_t = TileTensor(olp, row_major[BATCH, 2 * ACT]())
    var adv_t = TileTensor(adv, row_major[BATCH, 2 * ACT]())
    var lpb_t = TileTensor(loss_per_b, row_major[BATCH, 1]())

    # Bespoke pipeline (used for parity oracle).
    var bespoke_act_t = TileTensor(act, row_major[BATCH, ACT]())
    var bespoke_olp_t = TileTensor(olp, row_major[BATCH]())
    var bespoke_adv_t = TileTensor(adv, row_major[BATCH]())
    var bespoke = PPOActorLoss[ACT].make["cpu"](
        clip_eps=Scalar[DT](0.2), entropy_coef=Scalar[DT](0.01)
    )
    var bespoke_mean = bespoke.forward["cpu", BATCH](
        ao_t, bespoke_act_t, bespoke_olp_t, bespoke_adv_t
    )
    var bespoke_sum = bespoke_mean * Scalar[DT](BATCH)

    var obj = PPOObjective[ACT].make[target="cpu", INIT=Xavier]()
    obj.clip_eps = Scalar[DT](0.2)
    obj.entropy_coef = Scalar[DT](0.01)
    obj.forward["cpu", BATCH](ao_t, act_t, olp_t, adv_t, output=lpb_t)
    var obj_sum: Scalar[DT] = 0.0
    for b in range(BATCH):
        obj_sum += loss_per_b[b]

    var diff = bespoke_sum - obj_sum
    if diff < Scalar[DT](0.0):
        diff = -diff
    print("  bespoke_sum =", bespoke_sum, " obj_sum =", obj_sum, " |Δ| =", diff)
    assert_true(diff < Scalar[DT](1e-4),
                "PPOObjective forward must match bespoke up to 1e-4")
    ao.free(); act.free(); olp.free(); adv.free(); loss_per_b.free()
    print("  ok")


def test_backward_parity() raises:
    print("test_backward_parity ...")
    seed(11)

    var ao: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var act: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * ACT)
    var olp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var adv: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var loss_per_b: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var grad_seed: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var grad_act: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * ACT)
    var grad_olp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var grad_adv: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)

    var gout_bespoke: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var gout_obj: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)

    for b in range(BATCH):
        for j in range(ACT):
            ao[b * 2 * ACT + j] = Scalar[DT](-1.0 + 2.0 * random_float64())
            ao[b * 2 * ACT + ACT + j] = Scalar[DT](-2.0 + 2.0 * random_float64())
            act[b * ACT + j] = Scalar[DT](-1.0 + 2.0 * random_float64())
        olp[b] = Scalar[DT](-1.0 + 2.0 * random_float64())
        adv[b] = Scalar[DT](-2.0 + 4.0 * random_float64())
        # Mean-seed: 1/BATCH per row.
        grad_seed[b] = Scalar[DT](1.0) / Scalar[DT](BATCH)

    # Hetero-variadic: pass all inputs with IN0_DIM layout for unification.
    var ao_t = TileTensor(ao, row_major[BATCH, 2 * ACT]())
    var act_t = TileTensor(act, row_major[BATCH, 2 * ACT]())
    var olp_t = TileTensor(olp, row_major[BATCH, 2 * ACT]())
    var adv_t = TileTensor(adv, row_major[BATCH, 2 * ACT]())
    var lpb_t = TileTensor(loss_per_b, row_major[BATCH, 1]())
    var gs_t = TileTensor(grad_seed, row_major[BATCH, 1]())
    var goo_t = TileTensor(gout_obj, row_major[BATCH, 2 * ACT]())
    var gact_t = TileTensor(grad_act, row_major[BATCH, 2 * ACT]())
    var golp_t = TileTensor(grad_olp, row_major[BATCH, 2 * ACT]())
    var gadv_t = TileTensor(grad_adv, row_major[BATCH, 2 * ACT]())

    # Bespoke (kernel bakes 1/BATCH inline).
    var bespoke_act_t = TileTensor(act, row_major[BATCH, ACT]())
    var bespoke_olp_t = TileTensor(olp, row_major[BATCH]())
    var bespoke_adv_t = TileTensor(adv, row_major[BATCH]())
    var gob_t = TileTensor(gout_bespoke, row_major[BATCH, 2 * ACT]())
    var bespoke = PPOActorLoss[ACT].make["cpu"](
        clip_eps=Scalar[DT](0.2), entropy_coef=Scalar[DT](0.01)
    )
    bespoke.vjp["cpu", BATCH](
        ao_t, bespoke_act_t, bespoke_olp_t, bespoke_adv_t, gob_t,
    )

    # Quaternary form (kernel takes go=1/BATCH per-row externally).
    var obj = PPOObjective[ACT].make[target="cpu", INIT=Xavier]()
    obj.clip_eps = Scalar[DT](0.2)
    obj.entropy_coef = Scalar[DT](0.01)
    obj.forward["cpu", BATCH](ao_t, act_t, olp_t, adv_t, output=lpb_t)
    obj.vjp["cpu", BATCH](gs_t, goo_t, gact_t, golp_t, gadv_t)

    var max_diff: Scalar[DT] = 0.0
    for k in range(BATCH * 2 * ACT):
        var d = gout_bespoke[k] - gout_obj[k]
        if d < Scalar[DT](0.0):
            d = -d
        if d > max_diff:
            max_diff = d
    print("  max |Δ grad_actor_output| =", max_diff)
    assert_true(max_diff < Scalar[DT](1e-6),
                "PPOObjective vjp must match bespoke to 1e-6")

    # grad_action / grad_olp / grad_adv must be zero.
    var nonzero_act: Scalar[DT] = 0.0
    for k in range(BATCH * ACT):
        var v = grad_act[k]
        if v < Scalar[DT](0.0): v = -v
        if v > nonzero_act: nonzero_act = v
    var nonzero_olp: Scalar[DT] = 0.0
    for k in range(BATCH):
        var v = grad_olp[k]
        if v < Scalar[DT](0.0): v = -v
        if v > nonzero_olp: nonzero_olp = v
    var nonzero_adv: Scalar[DT] = 0.0
    for k in range(BATCH):
        var v = grad_adv[k]
        if v < Scalar[DT](0.0): v = -v
        if v > nonzero_adv: nonzero_adv = v
    print(
        "  max |grad_action| =", nonzero_act,
        "  max |grad_olp| =", nonzero_olp,
        "  max |grad_adv| =", nonzero_adv,
    )
    assert_true(
        nonzero_act == Scalar[DT](0.0)
        and nonzero_olp == Scalar[DT](0.0)
        and nonzero_adv == Scalar[DT](0.0),
        "non-differentiable input grads must all be zero",
    )

    ao.free(); act.free(); olp.free(); adv.free()
    loss_per_b.free(); grad_seed.free()
    grad_act.free(); grad_olp.free(); grad_adv.free()
    gout_bespoke.free(); gout_obj.free()
    print("  ok")


def main() raises:
    print("=" * 70)
    print("PPOObjective parity vs bespoke PPOActorLoss (Phase I.2.5)")
    print("=" * 70)
    test_forward_parity()
    test_backward_parity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
