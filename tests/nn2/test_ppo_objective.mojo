"""PPOObjective primitive — parity test against bespoke PPOActorLoss.

Phase I.2.b. The bespoke `PPOActorLoss[ACT]` (mojo_rl/nn2/loss/ppo_actor_loss.mojo)
is the trusted reference: identical math, ~unchanged since Phase 6. We
verify that `PPOObjective[ACT]` (the FullGraph-compatible binary Module)
produces:
  1. The same per-sample loss vector. The bespoke `forward` returns
     `mean = (1/BATCH) Σ loss_b`, so we recover the sum from the bespoke
     side via `mean * BATCH` and compare to the sum of `PPOObjective`'s
     per-sample output.
  2. The same grad_actor_output. PPOObjective's vjp receives go = 1/BATCH
     per sample (the standard mean-seed); under that seed, the per-element
     grad matches the bespoke kernel exactly (which bakes 1/BATCH into
     the kernel directly).
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


def _rand_fill(
    p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, lo: Float64, hi: Float64
):
    for i in range(n):
        p[i] = Scalar[DT](lo + (hi - lo) * random_float64())


def test_forward_parity() raises:
    print("test_forward_parity ...")
    seed(7)

    var ao: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var act: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * ACT)
    var olp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var adv: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var aux: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * (ACT + 2))
    var loss_per_b: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)

    # mu ~ U[-1, 1]; log_std ~ U[-2, 0]; action ~ U[-1, 1].
    for b in range(BATCH):
        for j in range(ACT):
            ao[b * 2 * ACT + j] = Scalar[DT](-1.0 + 2.0 * random_float64())
            ao[b * 2 * ACT + ACT + j] = Scalar[DT](-2.0 + 2.0 * random_float64())
            act[b * ACT + j] = Scalar[DT](-1.0 + 2.0 * random_float64())
        olp[b] = Scalar[DT](-1.0 + 2.0 * random_float64())
        adv[b] = Scalar[DT](-2.0 + 4.0 * random_float64())
        # Pack aux = [action | old_log_prob | advantage].
        for j in range(ACT):
            aux[b * (ACT + 2) + j] = act[b * ACT + j]
        aux[b * (ACT + 2) + ACT] = olp[b]
        aux[b * (ACT + 2) + ACT + 1] = adv[b]

    var ao_t = TileTensor(ao, row_major[BATCH, 2 * ACT]())
    var act_t = TileTensor(act, row_major[BATCH, ACT]())
    var olp_t = TileTensor(olp, row_major[BATCH]())
    var adv_t = TileTensor(adv, row_major[BATCH]())
    # Hetero-variadic workaround: pass aux with IN0_DIM-shaped layout
    # type so the variadic pack unifies; PPOObjective recovers the real
    # ACT+2 shape via typed_view internally. See concat.mojo.
    var aux_t = TileTensor(aux, row_major[BATCH, 2 * ACT]())
    var lpb_t = TileTensor(loss_per_b, row_major[BATCH, 1]())

    var bespoke = PPOActorLoss[ACT].make["cpu"](
        clip_eps=Scalar[DT](0.2), entropy_coef=Scalar[DT](0.01)
    )
    var bespoke_mean = bespoke.forward["cpu", BATCH](
        ao_t, act_t, olp_t, adv_t
    )
    var bespoke_sum = bespoke_mean * Scalar[DT](BATCH)

    var obj = PPOObjective[ACT].make[target="cpu", INIT=Xavier]()
    obj.clip_eps = Scalar[DT](0.2)
    obj.entropy_coef = Scalar[DT](0.01)
    obj.forward["cpu", BATCH](ao_t, aux_t, output=lpb_t)
    var obj_sum: Scalar[DT] = 0.0
    for b in range(BATCH):
        obj_sum += loss_per_b[b]

    var diff = bespoke_sum - obj_sum
    if diff < Scalar[DT](0.0):
        diff = -diff
    print("  bespoke_sum =", bespoke_sum, " obj_sum =", obj_sum, " |Δ| =", diff)
    assert_true(diff < Scalar[DT](1e-4),
                "PPOObjective forward must match bespoke up to 1e-4")
    ao.free(); act.free(); olp.free(); adv.free(); aux.free(); loss_per_b.free()
    print("  ok")


def test_backward_parity() raises:
    print("test_backward_parity ...")
    seed(11)

    var ao: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var act: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * ACT)
    var olp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var adv: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var aux: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * (ACT + 2))
    var loss_per_b: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var grad_seed: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var grad_aux: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * (ACT + 2))

    var gout_bespoke: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var gout_obj: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)

    for b in range(BATCH):
        for j in range(ACT):
            ao[b * 2 * ACT + j] = Scalar[DT](-1.0 + 2.0 * random_float64())
            ao[b * 2 * ACT + ACT + j] = Scalar[DT](-2.0 + 2.0 * random_float64())
            act[b * ACT + j] = Scalar[DT](-1.0 + 2.0 * random_float64())
        olp[b] = Scalar[DT](-1.0 + 2.0 * random_float64())
        adv[b] = Scalar[DT](-2.0 + 4.0 * random_float64())
        for j in range(ACT):
            aux[b * (ACT + 2) + j] = act[b * ACT + j]
        aux[b * (ACT + 2) + ACT] = olp[b]
        aux[b * (ACT + 2) + ACT + 1] = adv[b]
        # Mean-seed: 1/BATCH per row (matches what ComputeGraph would
        # supply via seed_grad_inv_batch).
        grad_seed[b] = Scalar[DT](1.0) / Scalar[DT](BATCH)

    var ao_t = TileTensor(ao, row_major[BATCH, 2 * ACT]())
    var act_t = TileTensor(act, row_major[BATCH, ACT]())
    var olp_t = TileTensor(olp, row_major[BATCH]())
    var adv_t = TileTensor(adv, row_major[BATCH]())
    # Hetero-variadic workaround: aux passed with IN0_DIM-shaped Layout
    # so the variadic pack unifies; PPOObjective.forward typed_views to
    # recover the real ACT+2 shape.
    var aux_t = TileTensor(aux, row_major[BATCH, 2 * ACT]())
    var lpb_t = TileTensor(loss_per_b, row_major[BATCH, 1]())
    var gs_t = TileTensor(grad_seed, row_major[BATCH, 1]())
    # For vjp, grad_inputs[0] (grad_actor_output, IN0_DIM=2*ACT) is the
    # "leading" shape that the pack unifies to; grad_inputs[1] (grad_aux)
    # is real-shape [BATCH, ACT+2] but typed as [BATCH, 2*ACT].
    var gaux_t = TileTensor(grad_aux, row_major[BATCH, 2 * ACT]())
    var gob_t = TileTensor(gout_bespoke, row_major[BATCH, 2 * ACT]())
    var goo_t = TileTensor(gout_obj, row_major[BATCH, 2 * ACT]())

    # Bespoke (kernel bakes 1/BATCH inline).
    var bespoke = PPOActorLoss[ACT].make["cpu"](
        clip_eps=Scalar[DT](0.2), entropy_coef=Scalar[DT](0.01)
    )
    bespoke.vjp["cpu", BATCH](ao_t, act_t, olp_t, adv_t, gob_t)

    # New form (kernel takes go=1/BATCH per-row externally).
    var obj = PPOObjective[ACT].make[target="cpu", INIT=Xavier]()
    obj.clip_eps = Scalar[DT](0.2)
    obj.entropy_coef = Scalar[DT](0.01)
    obj.forward["cpu", BATCH](ao_t, aux_t, output=lpb_t)
    obj.vjp["cpu", BATCH](gs_t, goo_t, gaux_t)

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

    # grad_aux must be zero (action / olp / adv are non-differentiable).
    var aux_nonzero = Scalar[DT](0.0)
    for k in range(BATCH * (ACT + 2)):
        var v = grad_aux[k]
        if v < Scalar[DT](0.0):
            v = -v
        if v > aux_nonzero:
            aux_nonzero = v
    print("  max |grad_aux| =", aux_nonzero)
    assert_true(aux_nonzero == Scalar[DT](0.0),
                "grad_aux must be identically zero")

    ao.free(); act.free(); olp.free(); adv.free(); aux.free()
    loss_per_b.free(); grad_seed.free(); grad_aux.free()
    gout_bespoke.free(); gout_obj.free()
    print("  ok")


def main() raises:
    print("=" * 70)
    print("PPOObjective parity vs bespoke PPOActorLoss (Phase I.2.b)")
    print("=" * 70)
    test_forward_parity()
    test_backward_parity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
