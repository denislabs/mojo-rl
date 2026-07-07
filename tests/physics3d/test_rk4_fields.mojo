"""P2 gate: stateful per-field RK4 integrator vs the legacy RK4 step,
FULL STEP, contact-free AND constraint-free comparison.

Legacy `RK4Integrator.step_gpu` launches 4 x (rk4_stage_kernel + Newton
solver) + rk4_combine_kernel. The solver launch is what applies contacts
and joint limits; the stage kernels themselves never touch limits. Here we
replicate exactly what step_gpu launches MINUS the 4 solver launches —
which equals unconstrained RK4 dynamics, exactly what
`RK4IntegratorFields.step` implements (it has no contacts/limits seam yet).
The legacy stage kernel also runs contact DETECTION (writes contact state
only, never qpos/qvel/qacc); poses are free-flight (rootz=2.0, no floor
contact) so detection finds nothing and the skipped-vs-run difference is
provably inert. Hinge angles are chosen strictly INSIDE the walker2d
ranges (thigh/leg in (-150,0)deg -> small NEGATIVE values; foot in
(-45,45)deg) and a host-side check verifies no limit is violated at the
end, proving the skipped limit handling is inactive too.

Walker2D, BATCH=3 (distinct qpos/qvel/qfrc per env), 3 CONSECUTIVE steps:
- fields-GPU vs legacy-GPU: qpos/qvel/qacc BIT-EXACT after every step.
- fields-CPU (same formula bodies) vs fields-GPU: 1e-3 after 3 steps.
- no joint-limit violations in the final qpos (limit inactivity proof).

Run: pixi run -e apple mojo run -I . tests/physics3d/test_rk4_fields.mojo
"""

from std.math import abs
from layout import Layout, LayoutTensor

from std.gpu.host import DeviceContext

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import DataFields, ModelFields
from mojo_rl.physics3d.integrator.rk4_integrator import RK4Integrator
from mojo_rl.physics3d.integrator.rk4_fields import RK4IntegratorFields
from mojo_rl.physics3d.solver.newton_solver import NewtonSolver
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    ws_solver_offset,
    rk4_extra_workspace_size,
    model_joint_offset,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
)
from mojo_rl.physics3d.joint_types import JNT_HINGE, JNT_SLIDE
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel

comptime DTYPE = DType.float32
comptime NQ = Walker2dModel.NQ
comptime NV = Walker2dModel.NV
comptime NBODY = Walker2dModel.NBODY
comptime NJOINT = Walker2dModel.NJOINT
comptime NGEOM = Walker2dModel.NGEOM
comptime MAX_CONTACTS = Walker2dModel.MAX_CONTACTS
comptime BATCH = 3
comptime N_STEPS = 3
comptime SS = state_size[NQ, NV, NBODY, MAX_CONTACTS, 0]()
comptime MS = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
# Same workspace formula as RK4Integrator.step_gpu: integrator temps +
# M_inv + Newton solver ws + RK4 extras (q0/v0/A[0..3]/C1/C2). The stage
# kernel computes M_inv (Newton NEEDS_M_INV) and addresses the rk4 regions
# past the solver ws, so ws must cover all of it.
comptime SOLVER_WS = NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()
comptime WS = (
    ws_solver_offset[NV, NBODY]() + SOLVER_WS
    + rk4_extra_workspace_size[NQ, NV]()
)


def _legacy_stage_kernel[
    B_: Int, STAGE: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS), MutAnyOrigin],
):
    RK4Integrator[SOLVER=NewtonSolver].rk4_stage_kernel[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, SS, MS, B_, WS,
        NGEOM, SOLVER_WS, STAGE,
    ](state, model, workspace)


def _legacy_combine_kernel[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS), MutAnyOrigin],
):
    RK4Integrator[SOLVER=NewtonSolver].rk4_combine_kernel[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, SS, MS, B_, WS,
        SOLVER_WS,
    ](state, model, workspace)


def _init_qpos(e: Int, i: Int) -> Scalar[DTYPE]:
    """Free-flight walker2d pose, strictly inside all joint ranges.

    qpos = [rootx, rootz, rooty, thigh, leg, foot, thigh_l, leg_l, foot_l];
    thigh/leg ranges are (-150, 0) deg, foot is (-45, 45) deg — all hinge
    values stay well inside with per-env variation <= 0.10 rad.
    """
    var ef = Scalar[DTYPE](e)
    if i == 0:  # rootx (unlimited slide)
        return Scalar[DTYPE](0.05) * ef - Scalar[DTYPE](0.05)
    elif i == 1:  # rootz: 2.0 -> torso ~2m up, no floor contact possible
        return Scalar[DTYPE](2.0)
    elif i == 2:  # rooty (unlimited hinge)
        return Scalar[DTYPE](0.04) * (ef - Scalar[DTYPE](1.0))
    elif i == 3:  # thigh
        return Scalar[DTYPE](-0.30) - Scalar[DTYPE](0.05) * ef
    elif i == 4:  # leg
        return Scalar[DTYPE](-0.50) + Scalar[DTYPE](0.03) * ef
    elif i == 5:  # foot
        return Scalar[DTYPE](-0.20) + Scalar[DTYPE](0.04) * ef
    elif i == 6:  # thigh_left
        return Scalar[DTYPE](-0.40) + Scalar[DTYPE](0.05) * ef
    elif i == 7:  # leg_left
        return Scalar[DTYPE](-0.35) - Scalar[DTYPE](0.04) * ef
    else:  # foot_left
        return Scalar[DTYPE](-0.15) - Scalar[DTYPE](0.03) * ef


def main() raises:
    print("--- RK4 full-step A/B: fields vs legacy, walker2d BATCH=", BATCH)
    var ctx = DeviceContext()

    # Model on device + bridge to fields.
    var model_t = TensorImpl[DTYPE].alloc(MS)
    model_t.upload(ctx)
    var mbuf = model_t.dev.value()
    Walker2dModel.init_model_gpu(ctx, mbuf)
    model_t.download(ctx)
    var mf = ModelFields[DTYPE, NV, NBODY, NJOINT, NGEOM]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)

    # Initial conditions: distinct per env, free-flight, inside all limits.
    comptime O_QPOS = qpos_offset[NQ, NV]()
    comptime O_QVEL = qvel_offset[NQ, NV]()
    comptime O_QACC = qacc_offset[NQ, NV]()
    comptime O_QFRC = qfrc_offset[NQ, NV]()

    var slab_t = TensorImpl[DTYPE].alloc(BATCH * SS)
    var d = DataFields[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, 0, BATCH]()
    var dc = DataFields[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, 0, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            var qp = _init_qpos(e, i)
            slab_t.data[e * SS + O_QPOS + i] = qp
            d.qpos.data[e * NQ + i] = qp
            dc.qpos.data[e * NQ + i] = qp
        for i in range(NV):
            var qv = Scalar[DTYPE]((e * 11 + i * 5) % 7 - 3) / 20.0
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 4.0
            slab_t.data[e * SS + O_QVEL + i] = qv
            slab_t.data[e * SS + O_QFRC + i] = qf
            d.qvel.data[e * NV + i] = qv
            d.qfrc.data[e * NV + i] = qf
            dc.qvel.data[e * NV + i] = qv
            dc.qfrc.data[e * NV + i] = qf
    slab_t.upload(ctx)
    d.upload_all(ctx)

    var ws_t = TensorImpl[DTYPE].alloc(BATCH * WS)
    ws_t.upload(ctx)

    var integ = RK4IntegratorFields[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        0, 0, 0, 0, 0, BATCH=BATCH,
    ]()
    integ.prepare_gpu(ctx)
    var integ_c = RK4IntegratorFields[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        0, 0, 0, 0, 0, BATCH=BATCH,
    ]()

    for step in range(N_STEPS):
        # Legacy: exactly step_gpu's launch sequence MINUS the 4 solver
        # launches (contact/limit-free — see module docstring).
        ctx.enqueue_function[_legacy_stage_kernel[BATCH, 0]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
            model_t.lt["gpu", Layout.row_major(1, MS)](),
            ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ctx.enqueue_function[_legacy_stage_kernel[BATCH, 1]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
            model_t.lt["gpu", Layout.row_major(1, MS)](),
            ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ctx.enqueue_function[_legacy_stage_kernel[BATCH, 2]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
            model_t.lt["gpu", Layout.row_major(1, MS)](),
            ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ctx.enqueue_function[_legacy_stage_kernel[BATCH, 3]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
            model_t.lt["gpu", Layout.row_major(1, MS)](),
            ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ctx.enqueue_function[_legacy_combine_kernel[BATCH]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
            model_t.lt["gpu", Layout.row_major(1, MS)](),
            ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        # Fields: stateful integrator step (GPU + CPU shadow).
        integ.step["gpu"](d, mf, ctx)
        integ_c.step["cpu"](dc, mf)

        # Compare fields-GPU vs legacy-GPU bit-exact.
        slab_t.download(ctx)
        d.qpos.download(ctx)
        d.qvel.download(ctx)
        d.qacc.download(ctx)
        var bad = 0
        for e in range(BATCH):
            for i in range(NQ):
                if d.qpos.data[e * NQ + i] != slab_t.data[e * SS + O_QPOS + i]:
                    bad += 1
            for i in range(NV):
                if d.qvel.data[e * NV + i] != slab_t.data[e * SS + O_QVEL + i]:
                    bad += 1
                if d.qacc.data[e * NV + i] != slab_t.data[e * SS + O_QACC + i]:
                    bad += 1
        if bad != 0:
            raise Error("step " + String(step) + ": fields-GPU != legacy-GPU")
        print(
            "  step", step,
            ": fields-GPU == legacy-GPU BIT-EXACT (qpos/qvel/qacc)",
        )

    # No joint-limit violations in the final pose (proves the skipped limit
    # handling was provably inactive on both sides).
    for j in range(NJOINT):
        var jo = model_joint_offset[NBODY](j)
        var jt = Int(model_t.data[jo + JOINT_IDX_TYPE])
        if jt != JNT_HINGE and jt != JNT_SLIDE:
            continue
        var rmin = model_t.data[jo + JOINT_IDX_RANGE_MIN]
        var rmax = model_t.data[jo + JOINT_IDX_RANGE_MAX]
        if not (rmin < rmax):
            continue  # unlimited joint
        var qadr = Int(model_t.data[jo + JOINT_IDX_QPOS_ADR])
        for e in range(BATCH):
            var qp = d.qpos.data[e * NQ + qadr]
            if qp <= rmin or qp >= rmax:
                raise Error(
                    "joint " + String(j) + " env " + String(e)
                    + " violates its range — pose selection broken"
                )
    print("  final qpos strictly inside all joint ranges (limits inactive)")

    # fields-CPU vs fields-GPU after N_STEPS.
    var worst = Float64(0)
    for i in range(BATCH * NQ):
        var err = abs(Float64(dc.qpos.data[i]) - Float64(d.qpos.data[i]))
        if err > worst:
            worst = err
    for i in range(BATCH * NV):
        var err = abs(Float64(dc.qvel.data[i]) - Float64(d.qvel.data[i]))
        if err > worst:
            worst = err
    print(
        "  fields-CPU vs fields-GPU after", N_STEPS, "steps, worst err:",
        worst,
    )
    if worst > 1e-3:
        raise Error("fields-CPU tolerance exceeded")
    print("  PASS: fields-CPU within 1e-3 after", N_STEPS, "steps")

    print("test_rk4_fields: ALL PASS")
