"""GATE B — DynamicsEnsembleBlock isolation test (storage, CPU).

Builds a small N=3 probabilistic dynamics ensemble, trains each member with
GaussianNLL on a KNOWN deterministic map, and asserts:
  - the training NLL drops substantially,
  - `predict_member` mean tracks the target (MSE below threshold),
  - `update_elites` selects the lowest-holdout-loss members.

The map: target[b] = [reward, Δobs...] where reward and each Δobs is a fixed
linear function of the (obs, act) input. PRED_DIM = 1 + OBS.
"""

from std.math import sqrt as fsqrt
from std.random import random_float64, seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.initializer import Kaiming
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.activations import Swish
from mojo_rl.nn.storage.combinators.sequential import Sequential

from mojo_rl.deep_agents.mbpo.dynamics_ensemble_block import (
    DynamicsEnsembleBlock,
)


comptime OBS = 3
comptime ACT = 1
comptime IN_DIM = OBS + ACT            # 4
comptime PRED = 1 + OBS                # 4
comptime OUT_DIM = 2 * PRED            # 8
comptime H = 32
comptime BATCH = 16
comptime N = 3
comptime NUM_ELITES = 2

comptime DynNet = Sequential[
    Linear[IN_DIM, H], Swish[H],
    Linear[H, H], Swish[H],
    Linear[H, OUT_DIM],
]


def _fill_batch(
    mut in_t: Tensor, mut tgt_t: Tensor
):
    """Random (obs, act) inputs; target = deterministic linear map."""
    for b in range(BATCH):
        var ib = b * IN_DIM
        for c in range(IN_DIM):
            in_t.data[ib + c] = Scalar[DT](2.0 * random_float64() - 1.0)
        var tb = b * PRED
        # reward = 0.5*o0 - 0.3*o1 + 0.2*a
        tgt_t.data[tb + 0] = (
            Scalar[DT](0.5) * in_t.data[ib + 0]
            - Scalar[DT](0.3) * in_t.data[ib + 1]
            + Scalar[DT](0.2) * in_t.data[ib + 3]
        )
        # Δobs[d] = 0.4*o_d + 0.1*a
        for d in range(OBS):
            tgt_t.data[tb + 1 + d] = (
                Scalar[DT](0.4) * in_t.data[ib + d]
                + Scalar[DT](0.1) * in_t.data[ib + 3]
            )


def main() raises:
    seed(123)
    print("=== GATE B: DynamicsEnsembleBlock isolation (CPU) ===")

    var blk = DynamicsEnsembleBlock[
        DynNet, N, NUM_ELITES, IN_DIM, OUT_DIM, BATCH
    ].make["cpu", Kaiming]()
    blk.set_lr(Scalar[DT](1e-3))
    blk.set_weight_decay(Scalar[DT](5e-5))

    var in_t = Tensor.alloc(BATCH * IN_DIM)
    var tgt_t = Tensor.alloc(BATCH * PRED)

    # Initial loss on a fresh batch (member 0).
    _fill_batch(in_t, tgt_t)
    var loss0 = blk.eval_member_loss["cpu"](0, in_t, tgt_t)
    print("  member0 initial NLL:", loss0)

    # Train all members.
    var n_steps = 400
    var last_loss = Scalar[DT](0.0)
    for m in range(N):
        for step in range(n_steps):
            _fill_batch(in_t, tgt_t)
            var l = blk.train_member_step["cpu"](m, in_t, tgt_t)
            if m == 0:
                last_loss = l
    print("  member0 final NLL:", last_loss)
    assert_true(
        last_loss < loss0 - Scalar[DT](0.5),
        "NLL did not drop substantially",
    )

    # predict_member mean should track the target.
    _fill_batch(in_t, tgt_t)
    var mu_t = Tensor.alloc(BATCH * PRED)
    var lv_t = Tensor.alloc(BATCH * PRED)
    blk.predict_member["cpu"](0, in_t, mu_t, lv_t)
    var mse = Scalar[DT](0.0)
    for b in range(BATCH):
        for j in range(PRED):
            var d = mu_t.data[b * PRED + j] - tgt_t.data[b * PRED + j]
            mse += d * d
    mse /= Scalar[DT](BATCH * PRED)
    print("  predict_member mean MSE vs target:", mse)
    assert_true(mse < Scalar[DT](0.05), "predict mean MSE too high")

    # update_elites: feed known holdout losses, lowest NUM_ELITES selected.
    var hl = List[Scalar[DT]]()
    hl.append(Scalar[DT](3.0))   # member 0
    hl.append(Scalar[DT](1.0))   # member 1 (lowest)
    hl.append(Scalar[DT](2.0))   # member 2
    blk.update_elites(hl)
    print(
        "  elites:",
        blk.elite_indices[0],
        blk.elite_indices[1],
    )
    assert_true(len(blk.elite_indices) == NUM_ELITES, "wrong elite count")
    # Lowest two losses are members 1 (1.0) and 2 (2.0).
    var has1 = False
    var has2 = False
    for i in range(NUM_ELITES):
        if blk.elite_indices[i] == 1:
            has1 = True
        if blk.elite_indices[i] == 2:
            has2 = True
    assert_true(has1 and has2, "elite selection wrong")
    assert_true(blk.elite_indices[0] == 1, "best elite should be member 1")

    print("=== GATE B PASSED ===")
