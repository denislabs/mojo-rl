"""End-to-end ENV loop on the fields path: InvertedPendulum closed-loop
balancing (obs -> controller -> action -> qfrc -> physics -> obs), gated
bit-exact against the legacy GPU pipeline.

InvertedPendulum is the intended pilot env: contact-free by construction
(contype=0 on all geoms), joint damping=1 (exercises the implicit-damping
finalize), slide limits +-1, single motor (gear=100, ctrlrange +-3, dof 0),
default obs = qpos || qvel (OBS_DIM=4), FRAME_SKIP=2.

Per control step, BOTH paths: read obs -> same PD balancing controller ->
qfrc[0] = gear * clamp(u) -> 2 physics substeps. Fields path uses
EulerIntegratorFields + extract_obs_qpos_qvel_fields; legacy path uses the
Euler step_kernel + limits + finalize on the flat slab with obs read from
the slab. Obs bit-exactness per step implies the whole closed loop stays
locked (identical actions). Reward/termination (pole angle) also compared.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_ip_fields_env_loop.mojo
"""

from std.math import abs
from std.gpu import block_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.integrator.euler_integrator import EulerIntegrator
from mojo_rl.physics3d.integrator.euler_fields import EulerIntegratorFields
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.physics3d.constraints import detect_and_solve_limits_gpu
from mojo_rl.physics3d.fields import DataFields, ModelFields
from mojo_rl.physics3d.gpu.buffer_utils import copy_model_to_buffer
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    ws_m_inv_offset,
    qpos_offset,
    qvel_offset,
    qfrc_offset,
    model_metadata_offset,
    MODEL_META_IDX_TIMESTEP,
)
from mojo_rl.envs.phyics3d_obs_fields import extract_obs_qpos_qvel_fields
from mojo_rl.envs.inverted_pendulum.inverted_pendulum_xml import (
    InvertedPendulumModel,
)

comptime DTYPE = DType.float32
comptime IPM = InvertedPendulumModel
comptime NQ = IPM.NQ  # 2
comptime NV = IPM.NV  # 2
comptime NBODY = IPM.NBODY
comptime NJOINT = IPM.NJOINT
comptime NGEOM = IPM.NGEOM
comptime MC = IPM.MAX_CONTACTS
comptime NSITE = IPM.NSITE
comptime NEQ = IPM.MAX_EQUALITY
comptime NTEN = IPM.MAX_TENDON
comptime BATCH = 2
comptime OBS_DIM = NQ + NV  # obs_qpos_skip=0
comptime FRAME_SKIP = 2
comptime N_CTRL_STEPS = 60
comptime GEAR: Float64 = 100.0
comptime CTRL_MAX: Float64 = 3.0

comptime SS = state_size[NQ, NV, NBODY, MC, NSITE]()
comptime MS = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM, NEQ, NTEN, NSITE]()
comptime WS = ws_m_inv_offset[NV, NBODY]() + NV * NV


def _legacy_step_kernel[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS), MutAnyOrigin],
):
    EulerIntegrator[SOLVER=NewtonSolver].step_kernel[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, SS, MS, B_, WS
    ](state, model, workspace)


def _legacy_limits_kernel[
    B_: Int
](
    dt: Scalar[DTYPE],
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS), MutAnyOrigin],
):
    var env = Int(block_idx.x)
    if env >= B_:
        return
    detect_and_solve_limits_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, SS, MS, WS, B_, 50, NGEOM
    ](env, dt, state, model, workspace)


def _legacy_finalize_kernel[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS), MutAnyOrigin],
):
    EulerIntegrator[SOLVER=NewtonSolver].step_finalize_kernel[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, SS, MS, B_, WS
    ](state, model, workspace)


@always_inline
def _controller(obs: InlineArray[Float64, OBS_DIM]) -> Float64:
    """Deterministic PD balancing controller on [x, theta, xd, thd].

    theta > 0 tips the pole toward +x (hinge about +y), so the cart must be
    pushed toward +x to get back under it: POSITIVE angle gains. Small
    recentering terms on x/xd (force = gear * u = 100 * u)."""
    var u = (
        0.3 * obs[0] + 0.8 * obs[2] + 6.0 * obs[1] + 1.5 * obs[3]
    )
    if u > CTRL_MAX:
        u = CTRL_MAX
    elif u < -CTRL_MAX:
        u = -CTRL_MAX
    return u


def main() raises:
    print(
        "--- IP closed-loop env on fields vs legacy GPU, BATCH=", BATCH,
        " ctrl steps=", N_CTRL_STEPS, "x", FRAME_SKIP, "substeps ---",
    )
    var ctx = DeviceContext()

    # Model via the parser path, bridged to fields.
    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, IPM.CONE_TYPE, NTEN,
        NSITE,
    ]()
    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MC, NSITE]()
    IPM.setup_model_and_data(model, data)
    var hb = ctx.enqueue_create_host_buffer[DTYPE](MS)
    ctx.synchronize()
    copy_model_to_buffer(model, hb)
    var model_t = TensorImpl[DTYPE].alloc(MS)
    for i in range(MS):
        model_t.data[i] = hb[i]
    model_t.upload(ctx)
    var mf = ModelFields[DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)
    comptime O_META = model_metadata_offset[NBODY, NJOINT]()
    var dt = model_t.data[O_META + MODEL_META_IDX_TIMESTEP]

    # Reset: perturbed pole per env.
    comptime O_QPOS = qpos_offset[NQ, NV]()
    comptime O_QVEL = qvel_offset[NQ, NV]()
    comptime O_QFRC = qfrc_offset[NQ, NV]()
    var pole0 = List[Float64]()
    pole0.append(0.05)
    pole0.append(-0.12)

    var slab_t = TensorImpl[DTYPE].alloc(BATCH * SS)
    var d = DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    for e in range(BATCH):
        slab_t.data[e * SS + O_QPOS + 1] = Scalar[DTYPE](pole0[e])
        d.qpos.data[e * NQ + 1] = Scalar[DTYPE](pole0[e])
    slab_t.upload(ctx)
    d.upload_all(ctx)
    var ws_t = TensorImpl[DTYPE].alloc(BATCH * WS)
    ws_t.upload(ctx)

    var integ = EulerIntegratorFields[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTEN, NSITE, 0, 0,
        BATCH=BATCH,
    ]()
    integ.prepare_gpu(ctx)
    var obs_t = TensorImpl[DTYPE].alloc(BATCH * OBS_DIM)
    obs_t.upload(ctx)

    var total_reward = List[Float64](length=BATCH, fill=0.0)
    var max_angle = List[Float64](length=BATCH, fill=0.0)
    var bad_total = 0
    for step in range(N_CTRL_STEPS):
        # ── obs, both paths ──────────────────────────────────────────────
        extract_obs_qpos_qvel_fields[
            "gpu", DTYPE, NQ, NV, NBODY, MC, NSITE, 0, BATCH
        ](d, obs_t, ctx)
        obs_t.download(ctx)
        slab_t.download(ctx)
        for e in range(BATCH):
            # fields obs == legacy slab obs (bit-exact)?
            for i in range(NQ):
                if (
                    obs_t.data[e * OBS_DIM + i]
                    != slab_t.data[e * SS + O_QPOS + i]
                ):
                    bad_total += 1
            for i in range(NV):
                if (
                    obs_t.data[e * OBS_DIM + NQ + i]
                    != slab_t.data[e * SS + O_QVEL + i]
                ):
                    bad_total += 1

        # ── controller + action -> qfrc, both paths (same obs) ──────────
        for e in range(BATCH):
            var obs_arr = InlineArray[Float64, OBS_DIM](uninitialized=True)
            for i in range(OBS_DIM):
                obs_arr[i] = Float64(obs_t.data[e * OBS_DIM + i])
            var u = _controller(obs_arr)
            var f = Scalar[DTYPE](GEAR * u)
            d.qfrc.data[e * NV + 0] = f
            d.qfrc.data[e * NV + 1] = Scalar[DTYPE](0)
            slab_t.data[e * SS + O_QFRC + 0] = f
            slab_t.data[e * SS + O_QFRC + 1] = Scalar[DTYPE](0)

            # reward/termination (config formula: |pole| < 0.2)
            var angle = Float64(obs_t.data[e * OBS_DIM + 1])
            if angle > -0.2 and angle < 0.2:
                total_reward[e] += 1.0
            var a_abs = angle if angle > 0 else -angle
            if a_abs > max_angle[e]:
                max_angle[e] = a_abs
        d.qfrc.upload(ctx)
        slab_t.upload(ctx)

        # ── physics: FRAME_SKIP substeps, both paths ─────────────────────
        for _ in range(FRAME_SKIP):
            ctx.enqueue_function[_legacy_step_kernel[BATCH]](
                slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
                model_t.lt["gpu", Layout.row_major(1, MS)](),
                ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
                grid_dim=(BATCH,),
                block_dim=(1,),
            )
            ctx.enqueue_function[_legacy_limits_kernel[BATCH]](
                dt,
                slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
                model_t.lt["gpu", Layout.row_major(1, MS)](),
                ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
                grid_dim=(BATCH,),
                block_dim=(1,),
            )
            ctx.enqueue_function[_legacy_finalize_kernel[BATCH]](
                slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
                model_t.lt["gpu", Layout.row_major(1, MS)](),
                ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
                grid_dim=(BATCH,),
                block_dim=(1,),
            )
            integ.step["gpu"](d, mf, ctx)

    if bad_total != 0:
        raise Error("fields obs diverged from legacy slab obs")
    print(
        "  PASS:", N_CTRL_STEPS, "control steps closed-loop, obs BIT-EXACT"
        " fields vs legacy every step",
    )
    var final0 = Float64(obs_t.data[0 * OBS_DIM + 1])
    var final1 = Float64(obs_t.data[1 * OBS_DIM + 1])
    print(
        "  episode returns (|pole|<0.2 reward):", total_reward[0],
        total_reward[1], "/", N_CTRL_STEPS,
        " max|angle|:", max_angle[0], max_angle[1],
        " final angle:", final0, final1,
    )
    # Balance sanity: the pole must never fall (|angle| stays far from the
    # +-1.57 hinge range) and must end near upright.
    var f0 = final0 if final0 > 0 else -final0
    var f1 = final1 if final1 > 0 else -final1
    if max_angle[0] > 0.5 or max_angle[1] > 0.5 or f0 > 0.15 or f1 > 0.15:
        raise Error("controller failed to keep the pole up — dynamics look wrong")
    print("  PASS: both envs kept the pole up (bounded + upright at end)")

    # ── CPU fields variant of the same closed loop ───────────────────────
    var dc = DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    for e in range(BATCH):
        dc.qpos.data[e * NQ + 1] = Scalar[DTYPE](pole0[e])
    var integ_c = EulerIntegratorFields[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTEN, NSITE, 0, 0,
        BATCH=BATCH,
    ]()
    var obs_c = TensorImpl[DTYPE].alloc(BATCH * OBS_DIM)
    for _ in range(N_CTRL_STEPS):
        extract_obs_qpos_qvel_fields[
            "cpu", DTYPE, NQ, NV, NBODY, MC, NSITE, 0, BATCH
        ](dc, obs_c)
        for e in range(BATCH):
            var obs_arr = InlineArray[Float64, OBS_DIM](uninitialized=True)
            for i in range(OBS_DIM):
                obs_arr[i] = Float64(obs_c.data[e * OBS_DIM + i])
            var u = _controller(obs_arr)
            dc.qfrc.data[e * NV + 0] = Scalar[DTYPE](GEAR * u)
            dc.qfrc.data[e * NV + 1] = Scalar[DTYPE](0)
        for _ in range(FRAME_SKIP):
            integ_c.step["cpu"](dc, mf)
    var worst = Float64(0)
    d.qpos.download(ctx)
    for e in range(BATCH):
        for i in range(NQ):
            var err = abs(
                Float64(dc.qpos.data[e * NQ + i])
                - Float64(d.qpos.data[e * NQ + i])
            )
            if err > worst:
                worst = err
    print("  fields-CPU closed loop vs fields-GPU, final qpos worst err:", worst)
    if worst > 1e-2:
        raise Error("fields-CPU closed loop diverged from GPU")
    print("  PASS: fields-CPU closed loop within 1e-2 after", N_CTRL_STEPS, "steps")

    print("test_ip_fields_env_loop: ALL PASS")
