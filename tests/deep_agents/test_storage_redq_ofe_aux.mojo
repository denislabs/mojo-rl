"""OFE aux-loss isolation gate (CPU, storage) — de-risks the OFE feature path.

Builds the OFE nets (state_branch / action_branch / predictor, small) + the
storage `OFEAuxLossStep`, feeds a batch where next-state is a KNOWN function of
(s, a), runs several aux steps, and asserts the aux MSE DECREASES substantially
— i.e. the feature extractor learns to predict next-state. This validates:

  * the OFE feature path (state_branch → concat → action_branch → predictor),
  * the auxiliary MSE loss + its gradient,
  * the variadic storage `Concat[PHI_S, ACT]` glue (a real consumer of the new
    variadic primitive),

all in isolation, BEFORE wiring the full REDQ-OFE trainer.

Run: pixi run mojo run -I . tests/deep_agents/test_storage_redq_ofe_aux.mojo
"""

from std.random import seed, random_float64
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Xavier
from mojo_rl.nn.optimizer.adam import Adam

from mojo_rl.deep_agents.redq_ofe.ofe_nets import (
    OFEStateBranch6, OFEActionBranch6, OFEPredictorHead,
    state_branch_out_dim, action_branch_out_dim,
)
from mojo_rl.deep_agents.redq_ofe.aux_loss_step import OFEAuxLossStep
from mojo_rl.deep_agents.training.trainer_block import TrainerState


comptime OBS = 4
comptime ACT = 2
comptime BATCH = 64
comptime PER_UNIT = 8   # small for the isolation test

comptime PHI_S_DIM = state_branch_out_dim(OBS, 6, PER_UNIT)        # OBS + 48
comptime PHI_SA_DIM = action_branch_out_dim(OBS, ACT, 6, PER_UNIT) # OBS+ACT+96

comptime SB = OFEStateBranch6[OBS, PER_UNIT]
comptime AB = OFEActionBranch6[PHI_S_DIM + ACT, PER_UNIT]
comptime PRED = OFEPredictorHead[PHI_SA_DIM, OBS]


def _fill_known_batch(mut state: TrainerState[OBS, ACT, BATCH]):
    """next_obs[b,d] = f(s,a): a fixed nonlinear-ish function of (s, a)."""
    for b in range(BATCH):
        for d in range(OBS):
            var sv = Scalar[DT](2.0 * random_float64() - 1.0)
            state.mb_s.data[b * OBS + d] = sv
        for j in range(ACT):
            var av = Scalar[DT](2.0 * random_float64() - 1.0)
            state.mb_a.data[b * ACT + j] = av
        # next_obs = 0.5*s + 0.3*(rolled action) + small bias  (deterministic
        # function of (s,a), so the OFE chain CAN fit it).
        for d in range(OBS):
            var s_d = state.mb_s.data[b * OBS + d]
            var a_d = state.mb_a.data[b * ACT + (d % ACT)]
            state.mb_sp.data[b * OBS + d] = (
                Scalar[DT](0.5) * s_d
                + Scalar[DT](0.3) * a_d
                + Scalar[DT](0.1)
            )


def main() raises:
    seed(7)
    print("=" * 60)
    print("OFE aux-loss isolation gate (storage, CPU)")
    print("=" * 60)

    var state_branch = SB.make["cpu", Xavier]()
    var action_branch = AB.make["cpu", Xavier]()
    var predictor = PRED.make["cpu", Xavier]()

    var sb_opt = Adam(lr=Scalar[DT](1e-3))
    var ab_opt = Adam(lr=Scalar[DT](1e-3))
    var pred_opt = Adam(lr=Scalar[DT](1e-3))

    var aux = OFEAuxLossStep[SB, AB, PRED, OBS, ACT, BATCH].make["cpu"]()

    var state = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    _fill_known_batch(state)

    print("PHI_S_DIM :", PHI_S_DIM)
    print("PHI_SA_DIM:", PHI_SA_DIM)

    var first_loss: Scalar[DT] = Scalar[DT](0.0)
    var last_loss: Scalar[DT] = Scalar[DT](0.0)
    comptime N_STEPS = 400
    for it in range(N_STEPS):
        var loss = aux.step["cpu"](
            state_branch, action_branch, predictor,
            sb_opt, ab_opt, pred_opt, state,
        )
        if it == 0:
            first_loss = loss
        last_loss = loss
        if it % 50 == 0:
            print("  aux step", it, " loss =", loss)

    print("-" * 60)
    print("first aux loss:", first_loss)
    print("last  aux loss:", last_loss)
    print("ratio (last/first):", last_loss / first_loss)
    print("-" * 60)

    assert_true(first_loss > Scalar[DT](0.0), "first aux loss > 0")
    assert_true(
        last_loss < first_loss * Scalar[DT](0.25),
        "aux MSE decreases substantially (< 25% of initial)",
    )
    print("OFE AUX-LOSS ISOLATION GATE OK")
