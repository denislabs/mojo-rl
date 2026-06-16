"""G.1 — OFEFeatureStep GPU smoke.

Gates whether the OFEStateBranch6 composite (Sequential of 6 stacked
`SkipConcat[Sequential[Linear, LayerNorm, SiLU]]` DenseBlocks)
compiles + runs on GPU. The O.1 GPU smoke only tested a single
OFEDenseBlock — this is the first end-to-end gate of the deep
6-nested-generic stack on device.

Architecture:
  - OBS=3, ACT=1 (ACT only used for TrainerState param), BATCH=4
  - PER_UNIT=2, N_BLOCKS=6 → PHI_S_DIM = 3 + 12 = 15

Gates:
  (1) make[target="gpu"](ctx) compiles and constructs.
  (2) feat.step["gpu"](state_branch, state) runs without error.
  (3) phi_s / phi_sp pointers contain finite values on device.
  (4) Skip-identity propagates through 6 blocks: leading OBS
      columns of phi_s == raw obs (mb_s) bit-identically on device.

Apple Metal note: 6-deep nested generics can crash the Metal
compiler (`feedback_metal_nested_generics.md`). If this test fails
to compile on Apple, fall back to OFEStateBranch with fewer blocks
or skip GPU on Apple via the apple env."""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Xavier

from mojo_rl.deep_agents.training.trainer_block import TrainerState
from mojo_rl.deep_agents.redq_ofe import (
    OFEStateBranch6,
    OFEFeatureStep,
    state_branch_out_dim,
)


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 4
comptime PER_UNIT = 2
comptime N_BLOCKS = 6
comptime PHI_S_DIM = state_branch_out_dim(OBS, N_BLOCKS, PER_UNIT)
comptime SB = OFEStateBranch6[OBS, PER_UNIT]


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def _is_finite(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) -> Bool:
    for i in range(n):
        if p[i] != p[i]:
            return False
        if _abs(p[i]) > Scalar[DT](1e30):
            return False
    return True


def test_ofe_feature_step_gpu() raises:
    print("=" * 70)
    print("G.1 — OFEFeatureStep on GPU (6-block state branch)")
    print("=" * 70)
    seed(42)
    var ctx = DeviceContext()

    # State branch (6 stacked OFEDenseBlocks via Sequential) on GPU.
    var sb = SB.make[target="gpu", INIT=Xavier](ctx)
    var feat = OFEFeatureStep[
        SB, OBS, ACT, BATCH,
    ].make[target="gpu"](ctx)

    # TrainerState on GPU. mb_s / mb_sp live on device.
    var state = TrainerState[OBS, ACT, BATCH].make[target="gpu"](ctx)

    # Populate mb_s / mb_sp on host then H2D.
    var N_OBS_TOTAL = BATCH * OBS
    var obs_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        N_OBS_TOTAL
    )
    var nobs_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        N_OBS_TOTAL
    )
    for b in range(BATCH):
        for d in range(OBS):
            obs_h[b * OBS + d] = Scalar[DT](
                0.2 + 0.1 * Float64(b) - 0.05 * Float64(d)
            )
            nobs_h[b * OBS + d] = Scalar[DT](
                0.4 - 0.07 * Float64(b) + 0.03 * Float64(d)
            )

    var obs_host = ctx.enqueue_create_host_buffer[DT](N_OBS_TOTAL)
    var nobs_host = ctx.enqueue_create_host_buffer[DT](N_OBS_TOTAL)
    ctx.synchronize()
    for i in range(N_OBS_TOTAL):
        obs_host.unsafe_ptr()[i] = obs_h[i]
        nobs_host.unsafe_ptr()[i] = nobs_h[i]
    ctx.enqueue_copy(state.mb_s.dev.value(), obs_host)
    ctx.enqueue_copy(state.mb_sp.dev.value(), nobs_host)

    # (2) Run feature step on GPU.
    feat.step["gpu"](sb, state)

    # (3) D2H phi_s / phi_sp and check finite.
    var N_PHI = BATCH * PHI_S_DIM
    var phi_s_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        N_PHI
    )
    var phi_sp_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        N_PHI
    )
    var phi_s_host = ctx.enqueue_create_host_buffer[DT](N_PHI)
    var phi_sp_host = ctx.enqueue_create_host_buffer[DT](N_PHI)
    ctx.enqueue_copy(phi_s_host, feat.phi_s.dev.value())
    ctx.enqueue_copy(phi_sp_host, feat.phi_sp.dev.value())
    ctx.synchronize()
    for i in range(N_PHI):
        phi_s_h[i] = phi_s_host.unsafe_ptr()[i]
        phi_sp_h[i] = phi_sp_host.unsafe_ptr()[i]

    assert_true(
        _is_finite(phi_s_h, N_PHI),
        "phi_s output finite on device",
    )
    assert_true(
        _is_finite(phi_sp_h, N_PHI),
        "phi_sp output finite on device",
    )

    # (4) Skip-identity through 6 stacked SkipConcats: leading OBS
    # columns of phi_s == raw obs bit-identically.
    var max_skip: Scalar[DT] = Scalar[DT](0.0)
    for b in range(BATCH):
        for d in range(OBS):
            var diff = _abs(
                phi_s_h[b * PHI_S_DIM + d] - obs_h[b * OBS + d]
            )
            if diff > max_skip:
                max_skip = diff
    print("  GPU skip-identity max |phi_s[:, 0:OBS] - obs| =", max_skip)
    assert_true(
        max_skip == Scalar[DT](0),
        "GPU 6-block state branch must preserve obs in leading"
        " OBS columns bit-identically",
    )

    obs_h.free()
    nobs_h.free()
    phi_s_h.free()
    phi_sp_h.free()

    print("PASS — OFEFeatureStep works on GPU with 6-block state branch.")


def main() raises:
    test_ofe_feature_step_gpu()
