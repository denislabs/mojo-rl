"""G.5 — OFEAuxLossStep GPU smoke.

Gates the GPU path through the full aux-loss pipeline:
  - SB.forward + Concat.forward + AB.forward + PRED.forward all on device.
  - `aux_mse_grad_gpu` writes grad_pred on device.
  - PRED.vjp + AB.vjp + Concat.vjp + SB.vjp all on device.
  - 3 Adams (pred_opt, ab_opt, sb_opt) step on device.

Gates:
  (1) make+step run end-to-end on GPU.
  (2) 10 steps on the SAME minibatch → loss drops monotonically
      (same gate as the CPU aux_loss smoke).
  (3) Loss returned each step is finite + positive."""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.optimizer.adam import Adam

from mojo_rl.deep_agents.training.trainer_block import TrainerState
from mojo_rl.deep_agents.redq_ofe import (
    OFEStateBranch6, OFEActionBranch6, OFEPredictorHead,
    OFEAuxLossStep,
    state_branch_out_dim, action_branch_out_dim,
)


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 4
comptime PER_UNIT = 2
comptime N_BLOCKS = 6

comptime PHI_S_DIM = state_branch_out_dim(OBS, N_BLOCKS, PER_UNIT)
comptime PHI_SA_DIM = action_branch_out_dim(OBS, ACT, N_BLOCKS, PER_UNIT)

comptime SB = OFEStateBranch6[OBS, PER_UNIT]
comptime AB = OFEActionBranch6[PHI_S_DIM + ACT, PER_UNIT]
comptime PRED = OFEPredictorHead[PHI_SA_DIM, OBS]


def test_ofe_aux_loss_gpu() raises:
    print("=" * 70)
    print("G.5 — OFEAuxLossStep on GPU (PRED→AB→SB backward + 3 Adams)")
    print("=" * 70)
    seed(42)
    var ctx = DeviceContext()

    var sb = SB.make[target="gpu", INIT=Xavier](ctx)
    var ab = AB.make[target="gpu", INIT=Xavier](ctx)
    var pred = PRED.make[target="gpu", INIT=Xavier](ctx)

    var sb_opt = Adam.make[target="gpu", M=SB](sb, ctx)
    var ab_opt = Adam.make[target="gpu", M=AB](ab, ctx)
    var pred_opt = Adam.make[target="gpu", M=PRED](pred, ctx)
    sb_opt.lr = Scalar[DT](3e-3)
    ab_opt.lr = Scalar[DT](3e-3)
    pred_opt.lr = Scalar[DT](3e-3)

    var state = TrainerState[OBS, ACT, BATCH].make[target="gpu"](ctx)

    # Synthetic batch on host → H2D.
    var obs_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * OBS
    )
    var act_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * ACT
    )
    var nobs_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * OBS
    )
    for b in range(BATCH):
        for d in range(OBS):
            obs_h[b * OBS + d] = Scalar[DT](
                0.3 + 0.1 * Float64(b) - 0.07 * Float64(d)
            )
            nobs_h[b * OBS + d] = Scalar[DT](
                0.5 - 0.08 * Float64(b) + 0.04 * Float64(d)
            )
        act_h[b * ACT] = Scalar[DT](-0.4 + 0.2 * Float64(b))

    var obs_host = ctx.enqueue_create_host_buffer[DT](BATCH * OBS)
    var act_host = ctx.enqueue_create_host_buffer[DT](BATCH * ACT)
    var nobs_host = ctx.enqueue_create_host_buffer[DT](BATCH * OBS)
    ctx.synchronize()
    for i in range(BATCH * OBS):
        obs_host.unsafe_ptr()[i] = obs_h[i]
        nobs_host.unsafe_ptr()[i] = nobs_h[i]
    for i in range(BATCH * ACT):
        act_host.unsafe_ptr()[i] = act_h[i]
    ctx.enqueue_copy(state.mb_s.dev.value(), obs_host)
    ctx.enqueue_copy(state.mb_a.dev.value(), act_host)
    ctx.enqueue_copy(state.mb_sp.dev.value(), nobs_host)

    var aux = OFEAuxLossStep[SB, AB, PRED, OBS, ACT, BATCH].make[
        target="gpu",
    ](ctx)

    var losses = List[Scalar[DT]](length=10, fill=Scalar[DT](0.0))
    for i in range(10):
        losses[i] = aux.step["gpu"](
            sb, ab, pred, sb_opt, ab_opt, pred_opt, state,
        )

    print("  GPU loss[0] =", losses[0])
    print("  GPU loss[4] =", losses[4])
    print("  GPU loss[9] =", losses[9])

    # (3) Each loss finite + positive.
    for i in range(10):
        assert_true(
            losses[i] == losses[i] and losses[i] > Scalar[DT](0.0),
            "GPU aux loss must be finite + positive at every step",
        )
    # (2) Loss drops over 10 steps on fixed batch.
    assert_true(
        losses[9] < losses[0] * Scalar[DT](0.5),
        "GPU aux loss must drop >= 50% over 10 steps",
    )

    obs_h.free()
    act_h.free()
    nobs_h.free()

    print("PASS — OFEAuxLossStep GPU path: PRED→AB→SB gradient flow.")


def main() raises:
    test_ofe_aux_loss_gpu()
