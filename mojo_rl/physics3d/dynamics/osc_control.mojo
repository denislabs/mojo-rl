"""ONE OSC_POSE CONTROL STEP OVER A BATCH — the stage a batched env calls.

    osc_control_step[target, DTYPE, D, BATCH](
        d, m, scratch, refs, state, work, ctrl, actions,
        nact, policy_step, ctx,
    )

`osc_pose_gpu.mojo` has the four per-lane primitives and `osc_pose.mojo` is a
one-lane host caller of them. Between those two there was a gap: the SEQUENCE —
refresh the dynamics, set the goal on a policy step only, run, ramp the gripper
— existed twice, once inside `OscPose.update`/`run` and once (not at all) on the
batched path. This is that sequence, written once, `target`-generic.

## ⚠⚠ WHY THE CADENCE IS AN ARGUMENT AND NOT A COUNTER

`policy_step` is passed in. robosuite's `Robot.control(action, policy_step)`
gates `set_goal` on that flag while running the torque law every substep, and
the two cadences are NOT interchangeable: `set_goal` adds the action's position
delta to the CURRENT end-effector pose, so calling it every substep turns a 5 cm
command into 5 cm twenty-five times over. The arm still tracks something smooth,
which is why this is worth stating — the failure looks like a tuning problem.

A counter inside this function would have to know the frame skip and would be
wrong the first time a caller stepped it twice for one action. The caller owns
its own loop and already knows.

## ⚠ THE REFRESH IS THE FIRST HALF OF THE INTEGRATOR'S PIPELINE, RUN AGAIN

`osc_refresh_dynamics` recomputes FK, body velocities, `subtree_com`, `cdof`,
CRBA and RNE at the CURRENT state, and the integrator then recomputes all of it.
robosuite pays exactly the same cost (`BaseController.update` calls
`sim.forward()`, then `mj_step` does it again), so a controller that skipped it
would not be the benchmark's. It is not an oversight and it is not free; see
that function's own note about the warm-start optimisation deliberately not
taken.

⚠ AND IT IS WHY A BATCHED ENV CANNOT JUST READ `d`'s FK PRODUCTS. Both
integrators run FK at the START of a substep, so at the top of the next one the
FK products describe the state BEFORE the previous integration. Reading them
without this refresh gives a controller acting on a state one substep stale —
visible as a small lag and nothing else.

## ⚠ WHAT IT WRITES: `ctrl`, NOT `qfrc`

OSC_POSE is a CONTROLLER, not an actuator. It maps a 7-word policy action onto
the model's `nact` actuator commands, and the ordinary actuation path then turns
those into forces. That is why the batched env runs this BEFORE
`apply_actions_kernel_gpu` rather than in place of it, and why it is not a
`custom_apply_actions_gpu` hook: that hook REPLACES actuation, which would mean
re-implementing every `<motor>` and `<position>` in the model.

## ⚠ SINGULARITY IS A FLAG PER LANE AND THIS DOES NOT RAISE

`osc_run_gpu` writes `OSC_IDX_SINGULAR` and leaves that lane's torques at zero.
Over a batch there is no single answer to raise about, so the flag stays in
`state` and `osc_singular_lanes` reads it back. A driver should treat a flagged
lane as one to reset — `OscPose.run` raises because one lane is the whole run.
"""

from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim

from mojo_rl.nn.core.tensor import TensorImpl

from ..fields import (
    Data, Model, DynamicsScratch, DimsLike, DYN1, DYN2, rl1, rl2,
)
from ..gpu.constants import (
    MODEL_BODY_SIZE, MODEL_JOINT_SIZE, MODEL_META_SIZE, MODEL_SITE_SIZE,
)
from .osc_pose_gpu import (
    OSC_ACTION_DIM, OSC_REF_WORDS, OSC_STATE_WORDS, OSC_WORK_WORDS,
    OSC_IDX_SINGULAR,
    osc_refresh_dynamics, osc_reset_gpu, osc_set_goal_gpu, osc_run_gpu,
    osc_gripper_gpu,
)


comptime OSC_TPB: Int = 64


@always_inline
def _osc_lane[
    DTYPE: DType, L_STATE: Layout, L_WORK: Layout, L_REFS: Layout,
    L_CTRL: Layout, L_ACT: Layout,
    L_QPOS: Layout, L_NV: Layout, L_B4: Layout, L_SX: Layout, L_B3: Layout,
    L_CDOF: Layout, L_M: Layout, L_JOINTS: Layout, L_BODIES: Layout,
    L_SITES: Layout, L_MMETA: Layout,
](
    state: LayoutTensor[DTYPE, L_STATE, MutAnyOrigin],
    work: LayoutTensor[DTYPE, L_WORK, MutAnyOrigin],
    refs: LayoutTensor[DTYPE, L_REFS, MutAnyOrigin],
    ctrl: LayoutTensor[DTYPE, L_CTRL, MutAnyOrigin],
    actions: LayoutTensor[DTYPE, L_ACT, MutAnyOrigin],
    qpos: LayoutTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, L_NV, MutAnyOrigin],
    xquat: LayoutTensor[DTYPE, L_B4, MutAnyOrigin],
    site_xpos: LayoutTensor[DTYPE, L_SX, MutAnyOrigin],
    subtree_com: LayoutTensor[DTYPE, L_B3, MutAnyOrigin],
    cdof: LayoutTensor[DTYPE, L_CDOF, MutAnyOrigin],
    mass: LayoutTensor[DTYPE, L_M, MutAnyOrigin],
    bias: LayoutTensor[DTYPE, L_NV, MutAnyOrigin],
    joints: LayoutTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    bodies: LayoutTensor[DTYPE, L_BODIES, MutAnyOrigin],
    sites: LayoutTensor[DTYPE, L_SITES, MutAnyOrigin],
    mmeta: LayoutTensor[DTYPE, L_MMETA, MutAnyOrigin],
    env: Int,
    nv: Int,
    policy_step: Int,
):
    """`Robot.control` for one lane: set_goal (gated), run, gripper ramp.

    ⚠ THE ORDER IS THE REFERENCE'S. `osc_gripper_gpu` writes the two finger
    controls AFTER `osc_run_gpu` has written the seven arm torques into the same
    `ctrl` row; run first would have the arm law overwrite the fingers.
    """
    if policy_step != 0:
        # ⚠⚠ THE SINGULAR FLAG IS PER CONTROL STEP, CLEARED HERE. `osc_run_gpu`
        # only ever SETS it and `osc_reset_gpu` was the only clear, so one
        # singular substep kept a lane reporting "singular, torques zero" for
        # the rest of the episode while its torques were in fact being
        # computed again. Cleared on the policy step, it answers "did any of
        # THIS step's substeps fail", which is what `osc_singular_lanes` says.
        state[env, OSC_IDX_SINGULAR] = Scalar[DTYPE](0)
        osc_set_goal_gpu[DTYPE](
            state, work, refs, actions, xquat, site_xpos, sites, env
        )
    osc_run_gpu[DTYPE](
        state, work, refs, ctrl, qpos, qvel, xquat, site_xpos, subtree_com,
        cdof, mass, bias, joints, bodies, sites, mmeta, env, nv,
    )
    osc_gripper_gpu[DTYPE, ACT_DIM=OSC_ACTION_DIM](
        state, refs, ctrl, actions, env
    )


def _osc_kernel[
    DTYPE: DType, L_STATE: Layout, L_WORK: Layout, L_REFS: Layout,
    L_CTRL: Layout, L_ACT: Layout,
    L_QPOS: Layout, L_NV: Layout, L_B4: Layout, L_SX: Layout, L_B3: Layout,
    L_CDOF: Layout, L_M: Layout, L_JOINTS: Layout, L_BODIES: Layout,
    L_SITES: Layout, L_MMETA: Layout, BATCH: Int,
](
    state: LayoutTensor[DTYPE, L_STATE, MutAnyOrigin],
    work: LayoutTensor[DTYPE, L_WORK, MutAnyOrigin],
    refs: LayoutTensor[DTYPE, L_REFS, MutAnyOrigin],
    ctrl: LayoutTensor[DTYPE, L_CTRL, MutAnyOrigin],
    actions: LayoutTensor[DTYPE, L_ACT, MutAnyOrigin],
    qpos: LayoutTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, L_NV, MutAnyOrigin],
    xquat: LayoutTensor[DTYPE, L_B4, MutAnyOrigin],
    site_xpos: LayoutTensor[DTYPE, L_SX, MutAnyOrigin],
    subtree_com: LayoutTensor[DTYPE, L_B3, MutAnyOrigin],
    cdof: LayoutTensor[DTYPE, L_CDOF, MutAnyOrigin],
    mass: LayoutTensor[DTYPE, L_M, MutAnyOrigin],
    bias: LayoutTensor[DTYPE, L_NV, MutAnyOrigin],
    joints: LayoutTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    bodies: LayoutTensor[DTYPE, L_BODIES, MutAnyOrigin],
    sites: LayoutTensor[DTYPE, L_SITES, MutAnyOrigin],
    mmeta: LayoutTensor[DTYPE, L_MMETA, MutAnyOrigin],
    nv: Int64,
    policy_step: Int64,
):
    # ⚠ `Int64`, NOT `Int`. A kernel scalar must be a FIXED-WIDTH type —
    # `Int`/`UInt` do not conform to `DevicePassable` and the failure is a wall
    # of instantiation notes ending in "use a fixed-width type", nowhere near
    # this line.
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _osc_lane[DTYPE](
        state, work, refs, ctrl, actions, qpos, qvel, xquat, site_xpos,
        subtree_com, cdof, mass, bias, joints, bodies, sites, mmeta,
        env, Int(nv), Int(policy_step),
    )


def _osc_reset_kernel[
    DTYPE: DType, L_STATE: Layout, L_WORK: Layout, L_REFS: Layout,
    L_QPOS: Layout, L_B4: Layout, L_SX: Layout, L_SITES: Layout, BATCH: Int,
](
    state: LayoutTensor[DTYPE, L_STATE, MutAnyOrigin],
    work: LayoutTensor[DTYPE, L_WORK, MutAnyOrigin],
    refs: LayoutTensor[DTYPE, L_REFS, MutAnyOrigin],
    qpos: LayoutTensor[DTYPE, L_QPOS, MutAnyOrigin],
    xquat: LayoutTensor[DTYPE, L_B4, MutAnyOrigin],
    site_xpos: LayoutTensor[DTYPE, L_SX, MutAnyOrigin],
    sites: LayoutTensor[DTYPE, L_SITES, MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    osc_reset_gpu[DTYPE](
        state, work, refs, qpos, xquat, site_xpos, sites, env
    )


def osc_reset_batch[
    target: StaticString, DTYPE: DType, D: DimsLike, BATCH: Int = 1,
    PARALLEL: Bool = False,
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    mut refs: TensorImpl[DTYPE],
    mut state: TensorImpl[DTYPE],
    mut work: TensorImpl[DTYPE],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`OscPose.reset` for every lane: anchor the goal and `q0` on the CURRENT
    pose.

    ⚠⚠ CALL IT AFTER THE EPISODE'S STATE IS IN PLACE AND BEFORE THE FIRST STEP.
    `q0` is the nullspace target for the whole episode and the goal is the pose
    the arm holds until an action moves it, so a reset taken at the wrong moment
    pulls the arm toward a configuration no episode ever had.

    ⚠ AND IT REFRESHES FIRST. `osc_reset_gpu` reads `site_xpos` and `xquat`, and
    after a batched reset writes `qpos` those describe the PREVIOUS episode
    until FK runs — the goal would be anchored on the pose the last episode
    ended in, which is a plausible-looking arm drifting somewhere.
    """
    osc_refresh_dynamics[target, DTYPE, D, BATCH, PARALLEL](d, m, scratch, ctx)
    var nb = d.dims.get_nbody()
    var ns = d.dims.get_nsite()

    comptime if target == "cpu":
        for e in range(BATCH):
            osc_reset_gpu[DTYPE](
                state.lt_dyn["cpu", DYN2](rl2(BATCH, OSC_STATE_WORDS)),
                work.lt_dyn["cpu", DYN2](rl2(BATCH, OSC_WORK_WORDS)),
                refs.lt_dyn["cpu", DYN1](rl1(OSC_REF_WORDS)),
                d.qpos.lt_dyn["cpu", DYN2](rl2(BATCH, d.dims.get_nq())),
                d.xquat.lt_dyn["cpu", DYN2](rl2(BATCH, nb * 4)),
                d.site_xpos.lt_dyn["cpu", DYN2](rl2(BATCH, ns * 3)),
                m.sites.lt_dyn["cpu", DYN2](rl2(ns, MODEL_SITE_SIZE)),
                e,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + OSC_TPB - 1) // OSC_TPB
        comptime L_STATE = Layout.row_major(BATCH, OSC_STATE_WORDS)
        comptime L_WORK = Layout.row_major(BATCH, OSC_WORK_WORDS)
        comptime L_REFS = Layout.row_major(OSC_REF_WORDS)
        comptime L_QPOS = Layout.row_major(BATCH, D.NQ)
        comptime L_B4 = Layout.row_major(BATCH, D.NBODY * 4)
        comptime L_SX = Layout.row_major(BATCH, D.NSITE * 3)
        comptime L_SITES = Layout.row_major(D.NSITE, MODEL_SITE_SIZE)
        c.enqueue_function[
            _osc_reset_kernel[
                DTYPE, L_STATE, L_WORK, L_REFS, L_QPOS, L_B4, L_SX, L_SITES,
                BATCH,
            ]
        ](
            state.lt["gpu", L_STATE](),
            work.lt["gpu", L_WORK](),
            refs.lt["gpu", L_REFS](),
            d.qpos.lt["gpu", L_QPOS](),
            d.xquat.lt["gpu", L_B4](),
            d.site_xpos.lt["gpu", L_SX](),
            m.sites.lt["gpu", L_SITES](),
            grid_dim=(BLOCKS,),
            block_dim=(OSC_TPB,),
        )


def osc_singular_lanes[DTYPE: DType](state: TensorImpl[DTYPE], batch: Int) -> Int:
    """How many lanes hit a singular operational-space inertia on the last step.

    ⚠ READ IT. `osc_run_gpu` leaves a flagged lane's torques at ZERO, which is a
    limp arm and not an error — a driver that never reads this reports a
    training curve for a batch in which some lanes were not being controlled.
    `OscPose.run` raises instead, because there one lane is the whole run.
    """
    var n = 0
    for e in range(batch):
        if state.data[e * OSC_STATE_WORDS + OSC_IDX_SINGULAR] != Scalar[DTYPE](0):
            n += 1
    return n


def osc_control_step[
    target: StaticString, DTYPE: DType, D: DimsLike, BATCH: Int = 1,
    PARALLEL: Bool = False,
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    mut refs: TensorImpl[DTYPE],
    mut state: TensorImpl[DTYPE],
    mut work: TensorImpl[DTYPE],
    mut ctrl: TensorImpl[DTYPE],
    mut actions: TensorImpl[DTYPE],
    nact: Int,
    policy_step: Bool,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Refresh the dynamics, then one control step for every lane.

    `refs` is `[OSC_REF_WORDS]` and SHARED; `state` `[BATCH, OSC_STATE_WORDS]`,
    `work` `[BATCH, OSC_WORK_WORDS]`, `ctrl` `[BATCH, nact]` and `actions`
    `[BATCH, OSC_ACTION_DIM]` are per lane.
    """
    osc_refresh_dynamics[target, DTYPE, D, BATCH, PARALLEL](d, m, scratch, ctx)

    var nv = d.dims.get_nv()
    var nb = d.dims.get_nbody()
    var ns = d.dims.get_nsite()
    var nj = d.dims.get_njoint()
    var ps = 1 if policy_step else 0

    comptime if target == "cpu":
        var st_v = state.lt_dyn["cpu", DYN2](rl2(BATCH, OSC_STATE_WORDS))
        var wk_v = work.lt_dyn["cpu", DYN2](rl2(BATCH, OSC_WORK_WORDS))
        var rf_v = refs.lt_dyn["cpu", DYN1](rl1(OSC_REF_WORDS))
        var ct_v = ctrl.lt_dyn["cpu", DYN2](rl2(BATCH, nact))
        var ac_v = actions.lt_dyn["cpu", DYN2](rl2(BATCH, OSC_ACTION_DIM))
        var qp_v = d.qpos.lt_dyn["cpu", DYN2](rl2(BATCH, d.dims.get_nq()))
        var qv_v = d.qvel.lt_dyn["cpu", DYN2](rl2(BATCH, nv))
        var xq_v = d.xquat.lt_dyn["cpu", DYN2](rl2(BATCH, nb * 4))
        var sx_v = d.site_xpos.lt_dyn["cpu", DYN2](rl2(BATCH, ns * 3))
        var sc_v = d.subtree_com.lt_dyn["cpu", DYN2](rl2(BATCH, nb * 3))
        var cd_v = scratch.cdof.lt_dyn["cpu", DYN2](rl2(BATCH, nv * 6))
        var mm_v = scratch.M.lt_dyn["cpu", DYN2](rl2(BATCH, nv * nv))
        var bi_v = scratch.bias.lt_dyn["cpu", DYN2](rl2(BATCH, nv))
        var jn_v = m.joints.lt_dyn["cpu", DYN2](rl2(nj, MODEL_JOINT_SIZE))
        var bd_v = m.bodies.lt_dyn["cpu", DYN2](rl2(nb, MODEL_BODY_SIZE))
        var si_v = m.sites.lt_dyn["cpu", DYN2](rl2(ns, MODEL_SITE_SIZE))
        var me_v = m.meta.lt_dyn["cpu", DYN1](rl1(MODEL_META_SIZE))
        for e in range(BATCH):
            _osc_lane[DTYPE](
                st_v, wk_v, rf_v, ct_v, ac_v, qp_v, qv_v, xq_v, sx_v, sc_v,
                cd_v, mm_v, bi_v, jn_v, bd_v, si_v, me_v, e, nv, ps,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + OSC_TPB - 1) // OSC_TPB
        comptime L_STATE = Layout.row_major(BATCH, OSC_STATE_WORDS)
        comptime L_WORK = Layout.row_major(BATCH, OSC_WORK_WORDS)
        comptime L_REFS = Layout.row_major(OSC_REF_WORDS)
        comptime L_ACT = Layout.row_major(BATCH, OSC_ACTION_DIM)
        comptime L_QPOS = Layout.row_major(BATCH, D.NQ)
        comptime L_NV = Layout.row_major(BATCH, D.NV)
        comptime L_B4 = Layout.row_major(BATCH, D.NBODY * 4)
        comptime L_B3 = Layout.row_major(BATCH, D.NBODY * 3)
        comptime L_SX = Layout.row_major(BATCH, D.NSITE * 3)
        comptime L_CDOF = Layout.row_major(BATCH, D.NV * 6)
        comptime L_M = Layout.row_major(BATCH, D.NV * D.NV)
        comptime L_JOINTS = Layout.row_major(D.NJOINT, MODEL_JOINT_SIZE)
        comptime L_BODIES = Layout.row_major(D.NBODY, MODEL_BODY_SIZE)
        comptime L_SITES = Layout.row_major(D.NSITE, MODEL_SITE_SIZE)
        comptime L_MMETA = Layout.row_major(MODEL_META_SIZE)
        comptime L_CTRL = Layout.row_major(BATCH, D.NACT)
        c.enqueue_function[
            _osc_kernel[
                DTYPE, L_STATE, L_WORK, L_REFS, L_CTRL, L_ACT, L_QPOS, L_NV,
                L_B4, L_SX, L_B3, L_CDOF, L_M, L_JOINTS, L_BODIES, L_SITES,
                L_MMETA, BATCH,
            ]
        ](
            state.lt["gpu", L_STATE](),
            work.lt["gpu", L_WORK](),
            refs.lt["gpu", L_REFS](),
            ctrl.lt["gpu", L_CTRL](),
            actions.lt["gpu", L_ACT](),
            d.qpos.lt["gpu", L_QPOS](),
            d.qvel.lt["gpu", L_NV](),
            d.xquat.lt["gpu", L_B4](),
            d.site_xpos.lt["gpu", L_SX](),
            d.subtree_com.lt["gpu", L_B3](),
            scratch.cdof.lt["gpu", L_CDOF](),
            scratch.M.lt["gpu", L_M](),
            scratch.bias.lt["gpu", L_NV](),
            m.joints.lt["gpu", L_JOINTS](),
            m.bodies.lt["gpu", L_BODIES](),
            m.sites.lt["gpu", L_SITES](),
            m.meta.lt["gpu", L_MMETA](),
            Int64(nv),
            Int64(ps),
            grid_dim=(BLOCKS,),
            block_dim=(OSC_TPB,),
        )
