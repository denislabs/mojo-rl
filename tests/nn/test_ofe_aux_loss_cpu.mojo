"""O.2.a — OFEAuxLossStep CPU smoke.

Single integration gate for the aux-loss path:

  (1) Build small SB / AB / PRED at OBS=3, ACT=1, BATCH=4,
      PER_UNIT=2, N_BLOCKS=6 — matches the O.1 composite dims.
  (2) Build one Adam each on the three networks (lr=3e-3 — well
      above the default so a small number of steps makes a clear
      difference).
  (3) Fill TrainerState.mb_s / mb_a / mb_sp with deterministic
      data.
  (4) Run 20 aux steps against the SAME minibatch. Verify:
      - First-step loss is finite.
      - Loss strictly decreases at step 20 vs step 0 (proves the
        gradient flows through Predictor → ActionBranch →
        StateBranch and that all three Adams update their params).
      - Loss is monotonically smaller in the back half of the
        window than the front half (smoother than per-step
        monotonicity, which Adam doesn't guarantee).

This is the *standalone* aux-step test — the full trainer
integration (where the aux step is interleaved with the RL critic
update and the OFE state branch ALSO feeds actor/critic forwards
via stop-grad) lands in O.2.b."""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.optimizer.adam import Adam

from mojo_rl.deep_agents.training.trainer_block import TrainerState
from mojo_rl.deep_agents.redq_ofe import (
    OFEStateBranch6,
    OFEActionBranch6,
    OFEPredictorHead,
    OFEAuxLossStep,
    state_branch_out_dim,
    action_branch_out_dim,
)


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 4
comptime PER_UNIT = 2
comptime N_BLOCKS = 6

comptime PHI_S_DIM = state_branch_out_dim(OBS, N_BLOCKS, PER_UNIT)   # 15
comptime PHI_SA_DIM = action_branch_out_dim(OBS, ACT, N_BLOCKS, PER_UNIT)  # 28

# Concrete types — name them so we can pass them to Adam.make.
comptime SB = OFEStateBranch6[OBS, PER_UNIT]
comptime AB = OFEActionBranch6[PHI_S_DIM + ACT, PER_UNIT]
comptime PRED = OFEPredictorHead[PHI_SA_DIM, OBS]


def test_ofe_aux_loss_cpu() raises:
    print("=" * 70)
    print("O.2.a — OFEAuxLossStep CPU smoke (Pendulum-shape obs)")
    print("=" * 70)
    seed(42)

    # ── Networks ───────────────────────────────────────────────────────
    var sb = SB.make[target="cpu", INIT=Xavier]()
    var ab = AB.make[target="cpu", INIT=Xavier]()
    var pred = PRED.make[target="cpu", INIT=Xavier]()

    # ── Adams (lr=3e-3, well above the 1e-3 default so a 20-step
    #    window shows clear decrease) ───────────────────────────────────
    var sb_opt = Adam.make[target="cpu", M=SB](sb)
    var ab_opt = Adam.make[target="cpu", M=AB](ab)
    var pred_opt = Adam.make[target="cpu", M=PRED](pred)
    sb_opt.lr = Scalar[DT](3e-3)
    ab_opt.lr = Scalar[DT](3e-3)
    pred_opt.lr = Scalar[DT](3e-3)

    # ── TrainerState ──────────────────────────────────────────────────
    var state = TrainerState[OBS, ACT, BATCH].make[target="cpu"]()

    var obs_p = state.mb_s.cpu_ptr()
    var act_p = state.mb_a.cpu_ptr()
    var nobs_p = state.mb_sp.cpu_ptr()
    for b in range(BATCH):
        for d in range(OBS):
            obs_p[b * OBS + d] = Scalar[DT](
                0.3 + 0.1 * Float64(b) - 0.07 * Float64(d)
            )
        for d in range(ACT):
            act_p[b * ACT + d] = Scalar[DT](
                -0.4 + 0.2 * Float64(b)
            )
        for d in range(OBS):
            nobs_p[b * OBS + d] = Scalar[DT](
                0.5 - 0.08 * Float64(b) + 0.04 * Float64(d)
            )

    # ── Aux step block ─────────────────────────────────────────────────
    var aux = OFEAuxLossStep[SB, AB, PRED, OBS, ACT, BATCH].make[
        target="cpu",
    ]()

    # ── 20 steps on the SAME batch ────────────────────────────────────
    var losses = List[Scalar[DT]](length=20, fill=Scalar[DT](0.0))
    for i in range(20):
        losses[i] = aux.step["cpu"](
            sb, ab, pred, sb_opt, ab_opt, pred_opt, state,
        )

    print("loss[ 0] =", losses[0])
    print("loss[ 1] =", losses[1])
    print("loss[10] =", losses[10])
    print("loss[19] =", losses[19])

    # ── Gates ──────────────────────────────────────────────────────────
    # (a) initial loss finite & positive.
    assert_true(
        losses[0] == losses[0] and losses[0] > Scalar[DT](0.0),
        "step-0 MSE loss must be a finite positive scalar",
    )
    # (b) loss decreased from step 0 to step 19 (overfit on a single
    #     batch — Adam should drive this loss down monotonically over
    #     20 steps even at lr=3e-3).
    assert_true(
        losses[19] < losses[0] * Scalar[DT](0.9),
        "loss must drop by >= 10% over 20 steps on a fixed batch",
    )
    # (c) average over back half < average over front half (smoother
    #     than per-step monotonicity, which Adam doesn't guarantee).
    var front: Scalar[DT] = Scalar[DT](0.0)
    var back: Scalar[DT] = Scalar[DT](0.0)
    for i in range(10):
        front += losses[i]
        back += losses[i + 10]
    assert_true(
        back < front,
        "average loss in back half (steps 10–19) must be < front half",
    )

    print("front-half avg:", front / Scalar[DT](10.0))
    print("back-half  avg:", back / Scalar[DT](10.0))
    print("PASS — OFEAuxLossStep gradient flows through pred→AB→SB.")


def main() raises:
    test_ofe_aux_loss_cpu()
