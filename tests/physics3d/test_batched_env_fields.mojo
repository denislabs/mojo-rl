"""GPU-batched fields env gate: Phyics3dBatchedEnvFields vs the legacy
slab pipeline, Walker2D (contacts), BIT-EXACT.

Reference = the legacy `Phyics3dEnv` GPU env re-assembled with the PGS-RK4
physics that the fields integrator implements: legacy `reset_kernel_gpu`,
`_pre_step_gpu`, `apply_actions_kernel_gpu`, FRAME_SKIP x [4 x
(rk4_stage_kernel[PGSSolver] + PGSSolver.solve_gpu) + rk4_combine_kernel],
`compute_cfrc_ext_gpu` + `compute_cvel_gpu`, `_extract_obs_rewards_dones_gpu`
— i.e. the exact hook arithmetic the facade reuses, around the exact legacy
physics the fields RK4 was gated bit-exact against
(test_rk4_contacts_fields). Same reset seed => obs / reward / done /
terminated / qpos must match BIT-EXACTLY every step, through a selective
reset and beyond.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_batched_env_fields.mojo
"""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.ptr import mptr
from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.phyics3d_batched_env_fields import Phyics3dBatchedEnvFields
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel
from mojo_rl.envs.walker2d.walker2d_config import Walker2dConfig
from mojo_rl.physics3d.integrator.rk4_integrator import RK4Integrator
from mojo_rl.physics3d.solver.pgs_solver import PGSSolver
from mojo_rl.physics3d.gpu import compute_cfrc_ext_gpu, compute_cvel_gpu
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    ws_solver_offset,
    rk4_extra_workspace_size,
    qpos_offset,
    metadata_offset,
    META_IDX_NUM_CONTACTS,
)

comptime NQ = Walker2dModel.NQ
comptime NV = Walker2dModel.NV
comptime NBODY = Walker2dModel.NBODY
comptime NJOINT = Walker2dModel.NJOINT
comptime NGEOM = Walker2dModel.NGEOM
comptime MC = Walker2dModel.MAX_CONTACTS
comptime CONE = Walker2dModel.CONE_TYPE
comptime OBS_DIM = Walker2dModel.OBS_DIM
comptime ACT_DIM = Walker2dModel.ACTION_DIM
comptime BATCH = 2
comptime N_STEPS = 20
comptime SS = state_size[NQ, NV, NBODY, MC, 0]()
comptime MS = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
comptime SOLVER_WS = PGSSolver.solver_workspace_size[NV, MC]()
comptime WS = (
    ws_solver_offset[NV, NBODY]() + SOLVER_WS
    + rk4_extra_workspace_size[NQ, NV]()
)
comptime FRAME_SKIP = Walker2dConfig.FRAME_SKIP
comptime MAX_STEPS = Walker2dConfig.MAX_STEPS
comptime RESET_SEED = UInt64(123)

comptime LegacyEnv = Phyics3dEnv[
    Walker2dModel, Walker2dConfig, DT, TERMINATE_ON_UNHEALTHY=True
]
# Pinned to PGS + serial + dense: this gate's legacy reference is the
# serial rk4_stage[PGS] pipeline, so the facade must match it exactly.
# The production default (newton + parallel + treewalk) is gated in
# test_batched_env_fields_production.mojo.
comptime FieldsEnv = Phyics3dBatchedEnvFields[
    Walker2dModel,
    Walker2dConfig,
    BATCH,
    TERMINATE_ON_UNHEALTHY=True,
    SOLVER="pgs",
    PARALLEL_GPU=False,
    CRBA_TREEWALK=False,
]


def _legacy_stage_kernel[
    B_: Int, STAGE: Int
](
    state: LayoutTensor[DT, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DT, Layout.row_major(1, MS), MutAnyOrigin],
    workspace: LayoutTensor[DT, Layout.row_major(B_, WS), MutAnyOrigin],
):
    RK4Integrator[SOLVER=PGSSolver].rk4_stage_kernel[
        DT, NQ, NV, NBODY, NJOINT, MC, SS, MS, B_, WS,
        NGEOM, SOLVER_WS, STAGE,
    ](state, model, workspace)


def _legacy_combine_kernel[
    B_: Int
](
    state: LayoutTensor[DT, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DT, Layout.row_major(1, MS), MutAnyOrigin],
    workspace: LayoutTensor[DT, Layout.row_major(B_, WS), MutAnyOrigin],
):
    RK4Integrator[SOLVER=PGSSolver].rk4_combine_kernel[
        DT, NQ, NV, NBODY, NJOINT, MC, SS, MS, B_, WS,
        SOLVER_WS,
    ](state, model, workspace)


def _legacy_pgs_kernel[
    B_: Int
](
    state: LayoutTensor[DT, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DT, Layout.row_major(1, MS), MutAnyOrigin],
    workspace: LayoutTensor[DT, Layout.row_major(B_, WS), MutAnyOrigin],
):
    PGSSolver.solve_gpu[
        DT, NQ, NV, NBODY, NJOINT, MC, SS, MS, NV, B_, WS, NGEOM,
        0, CONE, 0, 0,
    ](state, model, workspace)


def _action_val(t: Int, e: Int, j: Int) -> Scalar[DT]:
    return Scalar[DT]((t * 5 + e * 3 + j * 7) % 9 - 4) / 8.0


def main() raises:
    print("--- Batched fields env vs legacy slab pipeline, Walker2D ---")
    var ctx = DeviceContext()

    # ── facade env ────────────────────────────────────────────────────
    var env = FieldsEnv(ctx)
    env.reset_batch[BATCH](Optional(ctx), RESET_SEED)

    # ── legacy reference (slab + PGS-RK4 pipeline) ────────────────────
    var slab_t = TensorImpl[DT].alloc(BATCH * SS)
    slab_t.upload(ctx)
    var slab_buf = slab_t.dev.value()
    var model_t = TensorImpl[DT].alloc(MS)
    model_t.upload(ctx)
    var model_buf = model_t.dev.value()
    Walker2dModel.init_model_gpu(ctx, model_buf)
    var ws_t = TensorImpl[DT].alloc(BATCH * WS)
    ws_t.upload(ctx)

    var ref_actions = ctx.enqueue_create_buffer[DT](BATCH * ACT_DIM)
    var ref_obs = ctx.enqueue_create_buffer[DT](BATCH * OBS_DIM)
    var ref_rew = ctx.enqueue_create_buffer[DT](BATCH)
    var ref_done = ctx.enqueue_create_buffer[DT](BATCH)
    var ref_term = ctx.enqueue_create_buffer[DT](BATCH)
    ctx.enqueue_memset(ref_done, 0)

    LegacyEnv.reset_kernel_gpu[BATCH, SS](ctx, slab_buf, rng_seed=RESET_SEED)
    LegacyEnv.extract_obs_kernel_gpu[BATCH, SS, OBS_DIM](
        ctx, slab_buf, ref_obs
    )

    # host staging
    var h_act = ctx.enqueue_create_host_buffer[DT](BATCH * ACT_DIM)
    var h_obs_f = ctx.enqueue_create_host_buffer[DT](BATCH * OBS_DIM)
    var h_obs_l = ctx.enqueue_create_host_buffer[DT](BATCH * OBS_DIM)
    var h_scal_f = ctx.enqueue_create_host_buffer[DT](BATCH)
    var h_scal_l = ctx.enqueue_create_host_buffer[DT](BATCH)
    var h_done = ctx.enqueue_create_host_buffer[DT](BATCH)
    ctx.synchronize()

    # initial obs must match bit-exactly (same reset seed, same hooks)
    ctx.enqueue_copy(h_obs_f, env._obs)
    ctx.enqueue_copy(h_obs_l, ref_obs)
    ctx.synchronize()
    for i in range(BATCH * OBS_DIM):
        if h_obs_f[i] != h_obs_l[i]:
            raise Error("initial obs mismatch at " + String(i))
    print("  reset: obs BIT-EXACT")

    comptime O_QPOS = qpos_offset[NQ, NV]()
    comptime O_META = metadata_offset[NQ, NV, NBODY, MC]()
    comptime METADATA_SIZE_L = 4

    @parameter
    def run_reference_step() raises:
        LegacyEnv._pre_step_gpu[BATCH, SS](ctx, slab_buf)
        Walker2dModel.apply_actions_kernel_gpu[DT, BATCH, SS, ACT_DIM](
            ctx, slab_buf, ref_actions
        )
        for _ in range(FRAME_SKIP):
            comptime for stg in range(4):
                ctx.enqueue_function[_legacy_stage_kernel[BATCH, stg]](
                    slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
                    model_t.lt["gpu", Layout.row_major(1, MS)](),
                    ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
                    grid_dim=(BATCH,),
                    block_dim=(1,),
                )
                ctx.enqueue_function[_legacy_pgs_kernel[BATCH]](
                    slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
                    model_t.lt["gpu", Layout.row_major(1, MS)](),
                    ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
                    grid_dim=(BATCH,),
                    block_dim=(1, MC),
                )
            ctx.enqueue_function[_legacy_combine_kernel[BATCH]](
                slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
                model_t.lt["gpu", Layout.row_major(1, MS)](),
                ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
                grid_dim=(BATCH,),
                block_dim=(1,),
            )
        compute_cfrc_ext_gpu[DT, BATCH, SS, MS, NQ, NV, NBODY, MC, 0](
            ctx, slab_buf, model_buf
        )
        compute_cvel_gpu[DT, BATCH, SS, NQ, NV, NBODY, MC, 0](ctx, slab_buf)
        LegacyEnv._extract_obs_rewards_dones_gpu[
            BATCH, SS, MS, OBS_DIM, MAX_STEPS
        ](
            ctx, slab_buf, model_buf, ref_actions,
            ref_rew, ref_done, ref_term, ref_obs,
        )

    @parameter
    def compare_step(step: Int) raises -> Int:
        """Bit-compare obs/reward/done/terminated + qpos; return ncon."""
        ctx.enqueue_copy(h_obs_f, env._obs)
        ctx.enqueue_copy(h_obs_l, ref_obs)
        ctx.synchronize()
        var bad = 0
        for i in range(BATCH * OBS_DIM):
            if h_obs_f[i] != h_obs_l[i]:
                if bad < 3:
                    print(
                        "  obs diff @", i, ":", h_obs_f[i], "vs", h_obs_l[i]
                    )
                bad += 1
        ctx.enqueue_copy(h_scal_f, env._reward)
        ctx.enqueue_copy(h_scal_l, ref_rew)
        ctx.synchronize()
        for e in range(BATCH):
            if h_scal_f[e] != h_scal_l[e]:
                print("  reward diff e", e, ":", h_scal_f[e], h_scal_l[e])
                bad += 1
        ctx.enqueue_copy(h_scal_f, env._done)
        ctx.enqueue_copy(h_scal_l, ref_done)
        ctx.synchronize()
        for e in range(BATCH):
            if h_scal_f[e] != h_scal_l[e]:
                print("  done diff e", e, ":", h_scal_f[e], h_scal_l[e])
                bad += 1
        ctx.enqueue_copy(h_scal_f, env._terminated)
        ctx.enqueue_copy(h_scal_l, ref_term)
        ctx.synchronize()
        for e in range(BATCH):
            if h_scal_f[e] != h_scal_l[e]:
                print("  term diff e", e, ":", h_scal_f[e], h_scal_l[e])
                bad += 1
        env.d.qpos.download(ctx)
        env.d.meta.download(ctx)
        slab_t.download(ctx)
        var ncon = 0
        for e in range(BATCH):
            for i in range(NQ):
                if env.d.qpos.data[e * NQ + i] != slab_t.data[
                    e * SS + O_QPOS + i
                ]:
                    bad += 1
            ncon += Int(
                env.d.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS]
            )
        if bad != 0:
            raise Error(
                "step " + String(step) + ": " + String(bad) + " mismatches"
            )
        return ncon

    var total_ncon = 0
    for t in range(N_STEPS):
        for e in range(BATCH):
            for j in range(ACT_DIM):
                h_act[e * ACT_DIM + j] = _action_val(t, e, j)
        ctx.enqueue_copy(env._action, h_act)
        ctx.enqueue_copy(ref_actions, h_act)
        env.step_batch[BATCH](Optional(ctx), 0)
        run_reference_step()
        var ncon = compare_step(t)
        total_ncon += ncon
        if t % 5 == 4 or ncon > 0 and t < 3:
            print("  step", t, ": BIT-EXACT, contacts:", ncon)
    if total_ncon == 0:
        raise Error("no contacts over the run — gate is vacuous")
    print("  stepping: BIT-EXACT over", N_STEPS, "steps,",
          "total contacts:", total_ncon)

    # ── selective reset parity ────────────────────────────────────────
    # Force env 0 done on both sides; facade bumps its device counter
    # 42 -> 43, so hand the reference a counter at 43.
    h_done[0] = Scalar[DT](1.0)
    for e in range(1, BATCH):
        h_done[e] = Scalar[DT](0.0)
    ctx.enqueue_copy(env._done, h_done)
    ctx.enqueue_copy(ref_done, h_done)
    var ref_counter = ctx.enqueue_create_buffer[DType.uint64](1)
    ref_counter.enqueue_fill(UInt64(43))
    env.selective_reset_batch[BATCH](Optional(ctx), 0)
    LegacyEnv.selective_reset_kernel_gpu[BATCH, SS](
        ctx,
        slab_buf,
        ref_done,
        rng_seed=0,
        rng_counter_ptr=mptr(ref_counter.unsafe_ptr()),
    )
    LegacyEnv.extract_obs_kernel_gpu[BATCH, SS, OBS_DIM](
        ctx, slab_buf, ref_obs
    )
    ctx.enqueue_copy(h_obs_f, env._obs)
    ctx.enqueue_copy(h_obs_l, ref_obs)
    ctx.synchronize()
    for i in range(BATCH * OBS_DIM):
        if h_obs_f[i] != h_obs_l[i]:
            raise Error("post-selective-reset obs mismatch at " + String(i))
    print("  selective reset: obs BIT-EXACT (env 0 reset, env 1 live)")

    # two more steps to prove the reset state feeds the physics correctly
    for t in range(2):
        for e in range(BATCH):
            for j in range(ACT_DIM):
                h_act[e * ACT_DIM + j] = _action_val(100 + t, e, j)
        ctx.enqueue_copy(env._action, h_act)
        ctx.enqueue_copy(ref_actions, h_act)
        env.step_batch[BATCH](Optional(ctx), 0)
        run_reference_step()
        _ = compare_step(100 + t)
    print("  post-reset stepping: BIT-EXACT")

    print("test_batched_env_fields: ALL PASS")
