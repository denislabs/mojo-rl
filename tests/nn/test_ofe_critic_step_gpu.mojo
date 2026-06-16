"""G.3 — EnsembleCriticStepOFE GPU smoke.

Gates the GPU path in `EnsembleCriticStepOFE.step`:
  - GPU `concat_sa_gpu[PHI_S_DIM, ACT, BATCH]` (reused from
    off_policy_critic — width-agnostic).
  - Device `action_branch.forward` on the 6-block OFEActionBranch6.
  - Loop over N online critics through `CriticUpdateBlock.step["gpu"]`
    (GPU-capable since the SAC port).

Gates:
  (1) make + step run end-to-end on GPU.
  (2) Loss returned is finite.
  (3) Critic loss DECREASES over 5 fixed-`y` steps — proves the
      gradient flows back through `critic.vjp` and `opt.step` runs
      on device. Identical contract to the CPU critic test (just
      different precision tolerance)."""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.primitives.linear import Linear

from mojo_rl.deep_agents.training.trainer_block import TrainerState
from mojo_rl.deep_agents.redq.ensemble import CriticEnsemble
from mojo_rl.deep_agents.redq_ofe import (
    OFEStateBranch6, OFEActionBranch6,
    OFEFeatureStep, EnsembleCriticStepOFE,
    state_branch_out_dim, action_branch_out_dim,
)


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 4
comptime PER_UNIT = 2
comptime N_BLOCKS = 6
comptime N = 2

comptime PHI_S_DIM = state_branch_out_dim(OBS, N_BLOCKS, PER_UNIT)
comptime PHI_SA_DIM = action_branch_out_dim(OBS, ACT, N_BLOCKS, PER_UNIT)

comptime SB = OFEStateBranch6[OBS, PER_UNIT]
comptime AB = OFEActionBranch6[PHI_S_DIM + ACT, PER_UNIT]
comptime CRITIC = Sequential[Linear[PHI_SA_DIM, 1]]


def test_critic_step_gpu() raises:
    print("=" * 70)
    print("G.3 — EnsembleCriticStepOFE on GPU (concat_sa_gpu + critic.vjp)")
    print("=" * 70)
    seed(42)
    var ctx = DeviceContext()

    var sb = SB.make[target="gpu", INIT=Xavier](ctx)
    var ab = AB.make[target="gpu", INIT=Xavier](ctx)
    var ensemble = CriticEnsemble[CRITIC, N].make[
        target="gpu", INIT=Xavier,
    ](ctx)
    var feat = OFEFeatureStep[
        SB, OBS, ACT, BATCH,
    ].make[target="gpu"](ctx)
    var cstep = EnsembleCriticStepOFE[
        AB, CRITIC, N, BATCH, PHI_S_DIM, ACT,
    ].make[target="gpu"](ctx)

    var state = TrainerState[OBS, ACT, BATCH].make[target="gpu"](ctx)

    var obs_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * OBS
    )
    var act_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * ACT
    )
    var y_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH
    )
    for b in range(BATCH):
        for d in range(OBS):
            obs_h[b * OBS + d] = Scalar[DT](
                0.3 + 0.1 * Float64(b) - 0.05 * Float64(d)
            )
        act_h[b * ACT] = Scalar[DT](-0.2 + 0.15 * Float64(b))
        y_h[b] = Scalar[DT](0.5 - 0.1 * Float64(b))

    var obs_host = ctx.enqueue_create_host_buffer[DT](BATCH * OBS)
    var act_host = ctx.enqueue_create_host_buffer[DT](BATCH * ACT)
    var y_host = ctx.enqueue_create_host_buffer[DT](BATCH)
    ctx.synchronize()
    for i in range(BATCH * OBS):
        obs_host.unsafe_ptr()[i] = obs_h[i]
    for i in range(BATCH * ACT):
        act_host.unsafe_ptr()[i] = act_h[i]
    for b in range(BATCH):
        y_host.unsafe_ptr()[b] = y_h[b]
    ctx.enqueue_copy(state.mb_s.dev.value(), obs_host)
    ctx.enqueue_copy(state.mb_a.dev.value(), act_host)
    ctx.enqueue_copy(state.mb_y.dev.value(), y_host)

    # Feature step (φ(s) only — we don't need φ(s') for the critic).
    feat.step["gpu"](sb, state)
    var phi_s_p = feat.phi_s_ptr["gpu"]()

    # 5 critic steps on the fixed (φ(s), mb_a, mb_y).
    var losses = List[Scalar[DT]](length=5, fill=Scalar[DT](0.0))
    for i in range(5):
        losses[i] = cstep.step["gpu"](
            ab, ensemble, phi_s_p,
            state.mb_a.dev_ptr(), state.mb_y.dev_ptr(),
        )
        ctx.synchronize()
    print("  GPU critic loss[0] =", losses[0])
    print("  GPU critic loss[2] =", losses[2])
    print("  GPU critic loss[4] =", losses[4])

    # (1)(2) finite.
    assert_true(
        losses[0] == losses[0] and losses[0] > Scalar[DT](0.0),
        "GPU critic_loss[0] finite + positive",
    )
    # (3) decreases over 5 steps on fixed target. Looser than the
    # 10-step CPU gate (`test_ofe_critic_and_actor_cpu.mojo`) — 5
    # GPU steps at lr=1e-3 drop ~20% in practice.
    assert_true(
        losses[4] < losses[0] * Scalar[DT](0.85),
        "GPU critic loss must drop >= 15% over 5 steps on fixed y",
    )

    obs_h.free()
    act_h.free()
    y_h.free()

    print("PASS — EnsembleCriticStepOFE GPU path works.")


def main() raises:
    test_critic_step_gpu()
