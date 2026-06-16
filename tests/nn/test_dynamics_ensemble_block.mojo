"""DynamicsEnsembleBlock smoke + convergence test.

Phase I.1.b. Validates:

  1. **Construction**: `make[target="cpu", INIT=Kaiming]` builds N
     members + N optimisers + elite-indices list of length NUM_ELITES.
     predict_member doesn't crash.
  2. **Per-member training reduces NLL**: train one ensemble member for
     ~200 steps on a fixed synthetic regression (target distribution =
     constant mean / unit variance).  Expect NLL to drop substantially
     vs initial.
  3. **Elite re-ranking**: synthesise a holdout-loss vector with known
     ordering, call `update_elites`, verify `elite_indices` contains
     the lowest-loss members in some order.
"""

from std.memory import alloc
from std.random import seed, random_float64
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.elementwise import Elementwise
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.deep_agents.mbpo.dynamics_ensemble_block import (
    DynamicsEnsembleBlock,
)


# Pendulum-shaped dummy ensemble: OBS=3, ACT=1 → IN=4, PRED=4, OUT=8.
comptime OBS = 3
comptime ACT = 1
comptime IN_DIM = OBS + ACT
comptime PRED_DIM = 1 + OBS
comptime OUT_DIM = 2 * PRED_DIM
comptime HIDDEN = 32
comptime BATCH = 16
comptime N = 4
comptime NUM_ELITES = 3

comptime DynNet = Sequential[
    Linear[IN_DIM, HIDDEN], Elementwise[HIDDEN, SwishOp],
    Linear[HIDDEN, HIDDEN], Elementwise[HIDDEN, SwishOp],
    Linear[HIDDEN, OUT_DIM],
]

comptime Block = DynamicsEnsembleBlock[
    DynNet, N, NUM_ELITES, IN_DIM, OUT_DIM, BATCH
]


def test_construction_and_predict() raises:
    print("test_construction_and_predict ...")
    seed(42)
    var blk = Block.make[target="cpu", INIT=Kaiming]()

    assert_true(
        len(blk.members) == N,
        "members list should have length N",
    )
    assert_true(
        len(blk.opts) == N,
        "opts list should have length N",
    )
    assert_true(
        len(blk.elite_indices) == NUM_ELITES,
        "elite_indices initial length should be NUM_ELITES",
    )

    # Predict on a single random mini-batch from member 0.
    var in_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * IN_DIM
    )
    var mu_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * PRED_DIM
    )
    var lv_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * PRED_DIM
    )
    for i in range(BATCH * IN_DIM):
        in_p[i] = Scalar[DT](random_float64() * 2.0 - 1.0)

    var in_t = TileTensor(in_p, row_major[BATCH, IN_DIM]())
    var mu_t = TileTensor(mu_p, row_major[BATCH, PRED_DIM]())
    var lv_t = TileTensor(lv_p, row_major[BATCH, PRED_DIM]())
    blk.predict_member["cpu"](0, in_t, mu_t, lv_t)

    # Sanity: clamped logvars must lie in [-10, -2].
    for k in range(BATCH * PRED_DIM):
        var v = lv_p[k]
        assert_true(
            v >= Scalar[DT](-10.0) and v <= Scalar[DT](-2.0),
            "predicted logvar should be in [-10, -2]",
        )
    print("  ok (members=", N, ", elites=", NUM_ELITES, ", logvars in bounds)")


def test_member_training_reduces_loss() raises:
    """Train member 0 on a synthetic fixed-target dataset for ~200 steps.

    Target distribution: PRED_DIM iid samples with mean=0, variance=0.01.
    The optimal predictor is `µ = 0, logvar = log(0.01) ≈ −4.6`. Starting
    from random init the NLL is large and positive; after training it
    should drop substantially.

    This validates the full train_member_step pipeline (zero_grad,
    forward, NLL, vjp, member.vjp, opt.step) end-to-end."""
    print("test_member_training_reduces_loss ...")
    seed(2026)
    var blk = Block.make[target="cpu", INIT=Kaiming]()
    blk.set_lr(Scalar[DT](1e-3))

    var in_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * IN_DIM
    )
    var tgt_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * PRED_DIM
    )
    var in_t = TileTensor(in_p, row_major[BATCH, IN_DIM]())
    var tgt_t = TileTensor(tgt_p, row_major[BATCH, PRED_DIM]())

    # Generate one fixed mini-batch (no resampling — proves the optimiser
    # can drive NLL down on a fixed example).
    for i in range(BATCH * IN_DIM):
        in_p[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    for i in range(BATCH * PRED_DIM):
        # Small Gaussian-ish target.
        tgt_p[i] = Scalar[DT]((random_float64() - 0.5) * 0.2)

    var first = blk.eval_member_loss["cpu"](0, in_t, tgt_t)
    print("  initial NLL:", first)

    for _ in range(200):
        _ = blk.train_member_step["cpu"](0, in_t, tgt_t)

    var after = blk.eval_member_loss["cpu"](0, in_t, tgt_t)
    print("  after-200 NLL:", after, " (Δ=", first - after, ")")
    assert_true(
        after < first,
        "Training member 0 should reduce NLL",
    )
    # Modest threshold — 200 steps of fp32 SGD-Adam on a tiny ensemble
    # net with no obs-normalisation typically drops by ≥3 NLL units even
    # on a hard target. A bigger drop is fine; a smaller one signals an
    # opt step / grad-flow bug.
    assert_true(
        first - after > Scalar[DT](1.0),
        "NLL should drop by > 1.0 over 200 steps; got drop=" + String(first - after),
    )
    print("  ok (NLL reduction > 1.0)")


def test_update_elites() raises:
    print("test_update_elites ...")
    var blk = Block.make[target="cpu", INIT=Kaiming]()
    # Simulated holdout losses: member 2 best (lowest), then 0, 3, 1.
    var losses = [
        Scalar[DT](5.0),  # member 0
        Scalar[DT](9.0),  # member 1 (worst)
        Scalar[DT](1.0),  # member 2 (best)
        Scalar[DT](7.0),  # member 3
    ]
    blk.update_elites(losses)
    assert_true(
        len(blk.elite_indices) == NUM_ELITES,
        "elite_indices length should still be NUM_ELITES",
    )

    # Expected elite set: {2, 0, 3} in some order (NUM_ELITES=3, lowest 3).
    var saw_2 = False
    var saw_0 = False
    var saw_3 = False
    for k in range(NUM_ELITES):
        var idx = blk.elite_indices[k]
        if idx == 0:
            saw_0 = True
        if idx == 2:
            saw_2 = True
        if idx == 3:
            saw_3 = True
        assert_true(
            idx != 1,
            "member 1 has the highest loss and must NOT be in elites",
        )
    assert_true(saw_0 and saw_2 and saw_3,
                "elite set should contain members 0, 2, 3")
    print("  ok (elite_indices = lowest-loss 3 of 4)")


def main() raises:
    print("=" * 70)
    print("DynamicsEnsembleBlock validation (Phase I.1.b)")
    print("=" * 70)
    test_construction_and_predict()
    test_member_training_reduces_loss()
    test_update_elites()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
