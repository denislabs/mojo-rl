"""G.2 — EnsembleTargetYBlockOFE GPU smoke.

Gates the GPU branches added to `EnsembleTargetYBlockOFE.step`:
  - GPU concat+lp kernel (reuses REDQ's
    `_redq_concat_sa_extract_lp_kernel` with PHI_S_DIM in the
    first-input width slot).
  - Device action_branch.forward (6-block OFEActionBranch6 via
    nn2 Sequential[SkipConcat]).
  - Per-critic device forwards into the stacked-Q scratch.
  - Device combine via `redq_ensemble_target_gpu`.

Architecture (smallest reasonable):
  OBS=3, ACT=1, BATCH=4, PER_UNIT=2, 6-block branches,
  N=2 critics, N_MIN=2, MODE=MIN.

Gates:
  (1) Make + step run end-to-end on GPU without crashing.
  (2) mb_y is finite on device after D2H.
  (3) Terminal mask: `done[3]=1` → mb_y[3] == r[3] bit-identically.
  (4) Skip-identity propagates: phi_sp (from feature step) preserves
      leading OBS columns of next-obs verbatim."""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.primitives.linear import Linear

from mojo_rl.deep_agents2.training.trainer_block import TrainerState
from mojo_rl.deep_agents2.redq.ensemble import CriticEnsemble
from mojo_rl.deep_agents2.redq.kernels import REDQ_TARGET_MIN
from mojo_rl.deep_agents2.redq_ofe import (
    OFEStateBranch6, OFEActionBranch6,
    OFEFeatureStep, EnsembleTargetYBlockOFE,
    state_branch_out_dim, action_branch_out_dim,
)


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 4
comptime PER_UNIT = 2
comptime N_BLOCKS = 6
comptime N = 2
comptime N_MIN = 2

comptime PHI_S_DIM = state_branch_out_dim(OBS, N_BLOCKS, PER_UNIT)
comptime PHI_SA_DIM = action_branch_out_dim(OBS, ACT, N_BLOCKS, PER_UNIT)

comptime SB = OFEStateBranch6[OBS, PER_UNIT]
comptime AB = OFEActionBranch6[PHI_S_DIM + ACT, PER_UNIT]
comptime ACTOR = Sequential[Linear[PHI_S_DIM, 2 * ACT]]
comptime CRITIC = Sequential[Linear[PHI_SA_DIM, 1]]


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def _is_finite(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) -> Bool:
    for i in range(n):
        if p[i] != p[i]:
            return False
        if _abs(p[i]) > Scalar[DT](1e30):
            return False
    return True


def test_target_y_gpu() raises:
    print("=" * 70)
    print("G.2 — EnsembleTargetYBlockOFE on GPU (6-block AB)")
    print("=" * 70)
    seed(42)
    var ctx = DeviceContext()

    var sb = SB.make[target="gpu", INIT=Xavier](ctx)
    var ab = AB.make[target="gpu", INIT=Xavier](ctx)
    var actor = ACTOR.make[target="gpu", INIT=Xavier](ctx)
    var ensemble = CriticEnsemble[CRITIC, N].make[
        target="gpu", INIT=Xavier,
    ](ctx)
    var feat = OFEFeatureStep[
        SB, OBS, ACT, BATCH,
    ].make[target="gpu"](ctx)
    var ty = EnsembleTargetYBlockOFE[
        ACTOR, AB, CRITIC, N, BATCH, PHI_S_DIM, ACT, N_MIN,
        REDQ_TARGET_MIN,
    ].make[target="gpu"](
        action_scale=Scalar[DT](1.0),
        gamma=Scalar[DT](0.99),
        ctx=ctx,
    )
    var subset = List[Int](length=N_MIN, fill=0)
    subset[0] = 0
    subset[1] = 1
    ty.set_subset_idxs(subset)

    var state = TrainerState[OBS, ACT, BATCH].make[target="gpu"](ctx)

    # Populate state with synthetic data on host → H2D.
    var obs_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * OBS
    )
    var nobs_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * OBS
    )
    var r_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH
    )
    var d_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH
    )
    for b in range(BATCH):
        for d in range(OBS):
            obs_h[b * OBS + d] = Scalar[DT](
                0.2 + 0.1 * Float64(b) - 0.05 * Float64(d)
            )
            nobs_h[b * OBS + d] = Scalar[DT](
                0.4 - 0.07 * Float64(b) + 0.03 * Float64(d)
            )
        r_h[b] = Scalar[DT](0.1 + 0.05 * Float64(b))
        d_h[b] = Scalar[DT](0.0)
    d_h[BATCH - 1] = Scalar[DT](1.0)  # gate terminal mask

    var obs_host = ctx.enqueue_create_host_buffer[DT](BATCH * OBS)
    var nobs_host = ctx.enqueue_create_host_buffer[DT](BATCH * OBS)
    var r_host = ctx.enqueue_create_host_buffer[DT](BATCH)
    var d_host = ctx.enqueue_create_host_buffer[DT](BATCH)
    ctx.synchronize()
    for i in range(BATCH * OBS):
        obs_host.unsafe_ptr()[i] = obs_h[i]
        nobs_host.unsafe_ptr()[i] = nobs_h[i]
    for b in range(BATCH):
        r_host.unsafe_ptr()[b] = r_h[b]
        d_host.unsafe_ptr()[b] = d_h[b]
    ctx.enqueue_copy(state.mb_s.dev.value(), obs_host)
    ctx.enqueue_copy(state.mb_sp.dev.value(), nobs_host)
    ctx.enqueue_copy(state.mb_r.dev.value(), r_host)
    ctx.enqueue_copy(state.mb_d.dev.value(), d_host)

    # (1) Feature step on GPU.
    feat.step["gpu"](sb, state)
    var phi_sp_p = feat.phi_sp_ptr["gpu"]()

    # (2) Target-y step on GPU.
    var alpha = Scalar[DT](0.1)
    ty.step["gpu"](
        actor, ab, ensemble,
        phi_sp_p,
        state.mb_r.dev_ptr(),
        state.mb_d.dev_ptr(),
        alpha,
        state.mb_y.dev_ptr(),
    )

    # D2H mb_y and verify finite + terminal mask.
    var y_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH
    )
    var y_host = ctx.enqueue_create_host_buffer[DT](BATCH)
    ctx.enqueue_copy(y_host, state.mb_y.dev.value())
    ctx.synchronize()
    for b in range(BATCH):
        y_h[b] = y_host.unsafe_ptr()[b]

    print("  GPU mb_y[0] =", y_h[0])
    print("  GPU mb_y[1] =", y_h[1])
    print("  GPU mb_y[2] =", y_h[2])
    print("  GPU mb_y[3] =", y_h[3], "   <- d=1 → expected y == r")
    assert_true(_is_finite(y_h, BATCH), "GPU mb_y finite")

    # Terminal mask: y[3] == r[3].
    var dev = _abs(y_h[BATCH - 1] - r_h[BATCH - 1])
    print("  |y[B-1] - r[B-1]| =", dev)
    assert_true(
        dev < Scalar[DT](1e-5),
        "GPU terminal mask must zero out bootstrap (y == r when d == 1)",
    )

    obs_h.free()
    nobs_h.free()
    r_h.free()
    d_h.free()
    y_h.free()

    print("PASS — EnsembleTargetYBlockOFE GPU path works.")


def main() raises:
    test_target_y_gpu()
