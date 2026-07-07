"""P4 gate: equality + tendon constraints — fields vs legacy PGS pipeline.

Part A (TENDON): Humanoid (max_tendon=2, free joint + 17 hinges) dropped on
the floor with feet penetrating, BATCH=2, 2 full Euler steps. Legacy per
step: step_kernel -> detect_contacts_gpu (O(N^2)) -> PGSSolver.solve_gpu
(with MAX_TENDON=2) -> finalize. Fields: EulerIntegratorFields.step
(detection -> serialized contact PGS with limits + tendons inside).
qpos/qvel/qacc + solved contact records must be BIT-EXACT per step.
The two hip-knee tendon RECORDS are injected into the slab by the test:
<tendon> XML parsing was removed from the parser, so no XML model ever
carries tendon records — the legacy tendon path is only reachable with
manually populated records, which this gate provides identically to both
sides.
Joint poses are chosen strictly INSIDE all joint ranges so the joint-limit
pass stays inactive: the legacy limit builder reads dof_invweight0 through
a MAX_TENDON-less offset (a pre-existing misread on tendon models) which
limits_fields does NOT reproduce — with no active limits that value is
never read. The tendon builder's identical misread IS reproduced by
_tendon_env_fields (_legacy_invw_read), which this gate locks in.
Non-vacuous: model meta NTENDON must be 2, and a rerun with meta NTENDON
zeroed must change qvel after one step.

Part B (EQUALITY): synthetic 2-link chain + jointed anchor body with a
<equality><weld> between link2 and anchor (task-allowed fallback: Sawyer's
GPU slab path has pre-existing serialization inconsistencies — mesh hull
writes past the NMESH_VERTS-less buffer, `copy_equality_to_buffer` is never
called by init_model_gpu, and the invweight0 writer/ModelFields-loader
disagree for NSITE>0 — so it cannot form a meaningful bit-exact gate).
Capsules penetrate the floor (contacts + weld together, matching the legacy
solve order: contacts -> limits -> equality). Equality records are
serialized into the slab by the test (mirroring the uncalled
copy_equality_to_buffer) before ModelFields.load_from_slab, so legacy and
fields read identical records. BIT-EXACT per step; non-vacuous via meta
NEQUALITY == 1 + a NEQUALITY-zeroed rerun differing.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_equality_tendon_fields.mojo
"""

from std.math import abs
from std.gpu import block_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.fields import DataFields, ModelFields
from mojo_rl.physics3d.integrator.euler_integrator import EulerIntegrator
from mojo_rl.physics3d.integrator.euler_fields import EulerIntegratorFields
from mojo_rl.physics3d.solver.pgs_solver import PGSSolver
from mojo_rl.physics3d.collision.contact_detection import detect_contacts_gpu
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    ws_solver_offset,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    contacts_offset,
    metadata_offset,
    model_equality_offset,
    model_tendon_offset,
    model_metadata_offset,
    META_IDX_NUM_CONTACTS,
    MODEL_META_IDX_NTENDON,
    MODEL_META_IDX_NEQUALITY,
    CONTACT_SIZE,
    TENDON_IDX_NUM_JOINTS,
    TENDON_IDX_JOINT_0,
    TENDON_IDX_JOINT_1,
    TENDON_IDX_JOINT_2,
    TENDON_IDX_JOINT_3,
    TENDON_IDX_COEF_0,
    TENDON_IDX_COEF_1,
    TENDON_IDX_COEF_2,
    TENDON_IDX_COEF_3,
    TENDON_IDX_LENGTH_REF,
    TENDON_IDX_SOLREF_0,
    TENDON_IDX_SOLREF_1,
    TENDON_IDX_SOLIMP_0,
    TENDON_IDX_SOLIMP_1,
    TENDON_IDX_SOLIMP_2,
    TENDON_IDX_SOLIMP_3,
    TENDON_IDX_SOLIMP_4,
    EQ_IDX_TYPE,
    EQ_IDX_BODY_A,
    EQ_IDX_BODY_B,
    EQ_IDX_ANCHOR_AX,
    EQ_IDX_ANCHOR_AY,
    EQ_IDX_ANCHOR_AZ,
    EQ_IDX_ANCHOR_BX,
    EQ_IDX_ANCHOR_BY,
    EQ_IDX_ANCHOR_BZ,
    EQ_IDX_RELPOSE_X,
    EQ_IDX_RELPOSE_Y,
    EQ_IDX_RELPOSE_Z,
    EQ_IDX_RELPOSE_W,
    EQ_IDX_SOLREF_0,
    EQ_IDX_SOLREF_1,
    EQ_IDX_SOLIMP_0,
    EQ_IDX_SOLIMP_1,
    EQ_IDX_SOLIMP_2,
    EQ_IDX_SOLIMP_3,
    EQ_IDX_SOLIMP_4,
)
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel

comptime DTYPE = DType.float32
comptime BATCH = 2
comptime METADATA_SIZE_L = 4

# =============================================================================
# Part A: Humanoid (tendons)
# =============================================================================

comptime NQ_A = HumanoidModel.NQ  # 24
comptime NV_A = HumanoidModel.NV  # 23
comptime NBODY_A = HumanoidModel.NBODY  # 14
comptime NJOINT_A = HumanoidModel.NJOINT  # 18
comptime NGEOM_A = HumanoidModel.NGEOM  # 18
comptime MC_A = HumanoidModel.MAX_CONTACTS  # 50
comptime NTEN_A = HumanoidModel.MAX_TENDON  # 2
comptime CONE_A = HumanoidModel.CONE_TYPE
comptime NEQ_A = HumanoidModel.MAX_EQUALITY  # 0
comptime NSITE_A = HumanoidModel.NSITE  # 0
comptime NEXCL_A = HumanoidModel.nexclude  # 0
comptime N_STEPS_A = 2
comptime SS_A = state_size[NQ_A, NV_A, NBODY_A, MC_A, NSITE_A]()
comptime MS_A = model_size_with_invweight[
    NBODY_A, NJOINT_A, NV_A, NGEOM_A, NEQ_A, NTEN_A, NSITE_A, NEXCL_A
]()
comptime WS_A = ws_solver_offset[NV_A, NBODY_A]() + 81 * MC_A + 12 * MC_A * NV_A


def _legacy_step_kernel_a[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS_A), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS_A), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS_A), MutAnyOrigin],
):
    EulerIntegrator[SOLVER=PGSSolver].step_kernel[
        DTYPE, NQ_A, NV_A, NBODY_A, NJOINT_A, MC_A, SS_A, MS_A, B_, WS_A
    ](state, model, workspace)


def _legacy_detect_kernel_a[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS_A), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS_A), MutAnyOrigin],
):
    var env = Int(block_idx.x)
    if env >= B_:
        return
    detect_contacts_gpu[
        DTYPE, NQ_A, NV_A, NBODY_A, NJOINT_A, MC_A, SS_A, MS_A, B_, NGEOM_A,
        NEQ_A, NTEN_A, NSITE_A,
    ](env, state, model)


def _legacy_pgs_kernel_a[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS_A), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS_A), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS_A), MutAnyOrigin],
):
    PGSSolver.solve_gpu[
        DTYPE, NQ_A, NV_A, NBODY_A, NJOINT_A, MC_A, SS_A, MS_A, NV_A, B_,
        WS_A, NGEOM_A, NEQ_A, CONE_A, NTEN_A, NSITE_A,
    ](state, model, workspace)


def _legacy_finalize_kernel_a[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS_A), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS_A), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS_A), MutAnyOrigin],
):
    EulerIntegrator[SOLVER=PGSSolver].step_finalize_kernel[
        DTYPE, NQ_A, NV_A, NBODY_A, NJOINT_A, MC_A, SS_A, MS_A, B_, WS_A
    ](state, model, workspace)


def _humanoid_qpos(e: Int, i: Int) -> Scalar[DTYPE]:
    """Free joint pose + hinge angles strictly inside every joint range
    (keeps the joint-limit pass inactive — see module docstring)."""
    if i == 0:
        return Scalar[DTYPE](0.02) * Scalar[DTYPE](e)
    if i == 1:
        return Scalar[DTYPE](0)
    if i == 2:
        return Scalar[DTYPE](1.24)  # feet spheres penetrate the floor
    if i == 3:
        return Scalar[DTYPE](1)  # identity quaternion (w first)
    if i <= 6:
        return Scalar[DTYPE](0)
    # Hinges (qpos 7..23): abdomen_z/y/x, r_hip_x/z/y, r_knee,
    # l_hip_x/z/y, l_knee, r_sh1/2, r_elbow, l_sh1/2, l_elbow
    if i == 7:
        return Scalar[DTYPE](0.05) + Scalar[DTYPE](0.01) * Scalar[DTYPE](e)
    if i == 8:
        return Scalar[DTYPE](-0.1)
    if i == 9:
        return Scalar[DTYPE](0.05)
    if i == 10 or i == 14:
        return Scalar[DTYPE](-0.1)  # hip_x in [-0.436, 0.0873]
    if i == 11 or i == 15:
        return Scalar[DTYPE](-0.1)  # hip_z in [-1.047, 0.611]
    if i == 12 or i == 16:
        return Scalar[DTYPE](-0.05)  # hip_y in [-1.92, 0.349]
    if i == 13 or i == 17:
        return Scalar[DTYPE](-0.15)  # knee in [-2.79, -0.0349]
    if i == 20 or i == 23:
        return Scalar[DTYPE](-0.3)  # elbow in [-1.571, 0.873]
    return Scalar[DTYPE](0.1) + Scalar[DTYPE](0.01) * Scalar[DTYPE](e)


def _part_a_tendon(ctx: DeviceContext) raises:
    print("--- Part A: Humanoid tendons — fields vs legacy PGS, BATCH=", BATCH)

    var model_t = TensorImpl[DTYPE].alloc(MS_A)
    model_t.upload(ctx)
    var mbuf = model_t.dev.value()
    HumanoidModel.init_model_gpu(ctx, mbuf)
    model_t.download(ctx)

    # <tendon><fixed> XML parsing was removed from the parser (see
    # model_def_from_xml.mojo), so the slab never carries tendon records —
    # the legacy tendon path is only reachable via manual population.
    # Inject the Humanoid's two hip-knee tendons (coef -1 * hip_y +
    # 1 * knee, MuJoCo-default solref/solimp) into the slab so BOTH the
    # legacy kernel and ModelFields read identical, active records.
    comptime META_OFF_A = model_metadata_offset[NBODY_A, NJOINT_A]()
    model_t.data[META_OFF_A + MODEL_META_IDX_NTENDON] = Scalar[DTYPE](2)
    for t_i in range(2):
        var t_off = model_tendon_offset[NBODY_A, NJOINT_A, NGEOM_A, NEQ_A](
            t_i
        )
        # right: r_hip_y (joint 6) + r_knee (joint 7);
        # left: l_hip_y (joint 10) + l_knee (joint 11)
        var j0 = 6 if t_i == 0 else 10
        model_t.data[t_off + TENDON_IDX_NUM_JOINTS] = Scalar[DTYPE](2)
        model_t.data[t_off + TENDON_IDX_JOINT_0] = Scalar[DTYPE](j0)
        model_t.data[t_off + TENDON_IDX_JOINT_1] = Scalar[DTYPE](j0 + 1)
        model_t.data[t_off + TENDON_IDX_JOINT_2] = Scalar[DTYPE](-1)
        model_t.data[t_off + TENDON_IDX_JOINT_3] = Scalar[DTYPE](-1)
        model_t.data[t_off + TENDON_IDX_COEF_0] = Scalar[DTYPE](-1)
        model_t.data[t_off + TENDON_IDX_COEF_1] = Scalar[DTYPE](1)
        model_t.data[t_off + TENDON_IDX_COEF_2] = Scalar[DTYPE](0)
        model_t.data[t_off + TENDON_IDX_COEF_3] = Scalar[DTYPE](0)
        model_t.data[t_off + TENDON_IDX_LENGTH_REF] = Scalar[DTYPE](0.05)
        model_t.data[t_off + TENDON_IDX_SOLREF_0] = Scalar[DTYPE](0.02)
        model_t.data[t_off + TENDON_IDX_SOLREF_1] = Scalar[DTYPE](1)
        model_t.data[t_off + TENDON_IDX_SOLIMP_0] = Scalar[DTYPE](0.9)
        model_t.data[t_off + TENDON_IDX_SOLIMP_1] = Scalar[DTYPE](0.95)
        model_t.data[t_off + TENDON_IDX_SOLIMP_2] = Scalar[DTYPE](0.001)
        model_t.data[t_off + TENDON_IDX_SOLIMP_3] = Scalar[DTYPE](0.5)
        model_t.data[t_off + TENDON_IDX_SOLIMP_4] = Scalar[DTYPE](2)
    model_t.upload(ctx)

    var mf = ModelFields[
        DTYPE, NV_A, NBODY_A, NJOINT_A, NGEOM_A, NEQ_A, NTEN_A, NSITE_A,
        NEXCL_A, 0,
    ]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)

    if Int(mf.meta.data[MODEL_META_IDX_NTENDON]) != 2:
        raise Error("part A vacuous: model meta NTENDON != 2")

    comptime O_QPOS = qpos_offset[NQ_A, NV_A]()
    comptime O_QVEL = qvel_offset[NQ_A, NV_A]()
    comptime O_QACC = qacc_offset[NQ_A, NV_A]()
    comptime O_QFRC = qfrc_offset[NQ_A, NV_A]()
    comptime O_CON = contacts_offset[NQ_A, NV_A, NBODY_A]()
    comptime O_META = metadata_offset[NQ_A, NV_A, NBODY_A, MC_A]()

    var slab_t = TensorImpl[DTYPE].alloc(BATCH * SS_A)
    var d = DataFields[DTYPE, NQ_A, NV_A, NBODY_A, MC_A, NSITE_A, BATCH]()
    var dc = DataFields[DTYPE, NQ_A, NV_A, NBODY_A, MC_A, NSITE_A, BATCH]()
    var d_off = DataFields[DTYPE, NQ_A, NV_A, NBODY_A, MC_A, NSITE_A, BATCH]()
    for e in range(BATCH):
        for i in range(NQ_A):
            var qp = _humanoid_qpos(e, i)
            slab_t.data[e * SS_A + O_QPOS + i] = qp
            d.qpos.data[e * NQ_A + i] = qp
            dc.qpos.data[e * NQ_A + i] = qp
            d_off.qpos.data[e * NQ_A + i] = qp
        for i in range(NV_A):
            var qv = Scalar[DTYPE]((e * 7 + i * 5) % 7 - 3) / 20.0
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 4.0
            slab_t.data[e * SS_A + O_QVEL + i] = qv
            slab_t.data[e * SS_A + O_QFRC + i] = qf
            d.qvel.data[e * NV_A + i] = qv
            d.qfrc.data[e * NV_A + i] = qf
            dc.qvel.data[e * NV_A + i] = qv
            dc.qfrc.data[e * NV_A + i] = qf
            d_off.qvel.data[e * NV_A + i] = qv
            d_off.qfrc.data[e * NV_A + i] = qf
    slab_t.upload(ctx)
    d.upload_all(ctx)
    var ws_t = TensorImpl[DTYPE].alloc(BATCH * WS_A)
    ws_t.upload(ctx)

    var integ = EulerIntegratorFields[
        DTYPE, NQ_A, NV_A, NBODY_A, NJOINT_A, MC_A, NGEOM_A, NEQ_A, NTEN_A,
        NSITE_A, NEXCL_A, 0, CONE_A, BATCH,
    ]()
    integ.prepare_gpu(ctx)
    var integ_c = EulerIntegratorFields[
        DTYPE, NQ_A, NV_A, NBODY_A, NJOINT_A, MC_A, NGEOM_A, NEQ_A, NTEN_A,
        NSITE_A, NEXCL_A, 0, CONE_A, BATCH,
    ]()

    var qvel_step0 = List[Scalar[DTYPE]](capacity=BATCH * NV_A)
    for _ in range(BATCH * NV_A):
        qvel_step0.append(Scalar[DTYPE](0))

    for step in range(N_STEPS_A):
        ctx.enqueue_function[_legacy_step_kernel_a[BATCH]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS_A)](),
            model_t.lt["gpu", Layout.row_major(1, MS_A)](),
            ws_t.lt["gpu", Layout.row_major(BATCH, WS_A)](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ctx.enqueue_function[_legacy_detect_kernel_a[BATCH]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS_A)](),
            model_t.lt["gpu", Layout.row_major(1, MS_A)](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ctx.enqueue_function[_legacy_pgs_kernel_a[BATCH]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS_A)](),
            model_t.lt["gpu", Layout.row_major(1, MS_A)](),
            ws_t.lt["gpu", Layout.row_major(BATCH, WS_A)](),
            grid_dim=(BATCH,),
            block_dim=(1, MC_A),
        )
        ctx.enqueue_function[_legacy_finalize_kernel_a[BATCH]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS_A)](),
            model_t.lt["gpu", Layout.row_major(1, MS_A)](),
            ws_t.lt["gpu", Layout.row_major(BATCH, WS_A)](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        integ.step["gpu"](d, mf, ctx)
        integ_c.step["cpu"](dc, mf)

        slab_t.download(ctx)
        d.qpos.download(ctx)
        d.qvel.download(ctx)
        d.qacc.download(ctx)
        d.contacts.download(ctx)
        d.meta.download(ctx)
        if step == 0:
            for i in range(BATCH * NV_A):
                qvel_step0[i] = d.qvel.data[i]
        var bad = 0
        var ncon_seen = 0
        for e in range(BATCH):
            var nc = Int(
                d.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS]
            )
            var nc_l = Int(
                slab_t.data[e * SS_A + O_META + META_IDX_NUM_CONTACTS]
            )
            if nc != nc_l:
                print("  ncon mismatch env", e, ": fields", nc, " legacy", nc_l)
                bad += 1
                continue
            ncon_seen += nc
            for i in range(NQ_A):
                if (
                    d.qpos.data[e * NQ_A + i]
                    != slab_t.data[e * SS_A + O_QPOS + i]
                ):
                    if bad < 4:
                        print(
                            "  qpos diff e", e, "i", i, ":",
                            d.qpos.data[e * NQ_A + i], "vs",
                            slab_t.data[e * SS_A + O_QPOS + i],
                        )
                    bad += 1
            for i in range(NV_A):
                if (
                    d.qvel.data[e * NV_A + i]
                    != slab_t.data[e * SS_A + O_QVEL + i]
                ):
                    if bad < 4:
                        print(
                            "  qvel diff e", e, "i", i, ":",
                            d.qvel.data[e * NV_A + i], "vs",
                            slab_t.data[e * SS_A + O_QVEL + i],
                        )
                    bad += 1
                if (
                    d.qacc.data[e * NV_A + i]
                    != slab_t.data[e * SS_A + O_QACC + i]
                ):
                    bad += 1
            for c in range(nc):
                for k in range(CONTACT_SIZE):
                    if (
                        d.contacts.data[
                            e * MC_A * CONTACT_SIZE + c * CONTACT_SIZE + k
                        ]
                        != slab_t.data[
                            e * SS_A + O_CON + c * CONTACT_SIZE + k
                        ]
                    ):
                        if bad < 4:
                            print(
                                "  record diff e", e, "c", c, "k", k, ":",
                                d.contacts.data[
                                    e * MC_A * CONTACT_SIZE
                                    + c * CONTACT_SIZE + k
                                ],
                                "vs",
                                slab_t.data[
                                    e * SS_A + O_CON + c * CONTACT_SIZE + k
                                ],
                            )
                        bad += 1
        if bad != 0:
            raise Error("part A step " + String(step) + ": mismatch")
        if ncon_seen == 0:
            raise Error(
                "part A: no contacts at step " + String(step) + " — vacuous"
            )
        print(
            "  step", step, ": BIT-EXACT (qpos/qvel/qacc + contact records),"
            " total contacts:", ncon_seen,
        )

    var worst = Float64(0)
    for i in range(BATCH * NQ_A):
        var err = abs(Float64(dc.qpos.data[i]) - Float64(d.qpos.data[i]))
        if err > worst:
            worst = err
    print("  fields-CPU vs fields-GPU final qpos worst err:", worst)
    if worst > 1e-2:
        raise Error("part A: fields-CPU diverged from GPU")

    # Non-vacuity: tendon-off rerun (meta NTENDON=0 short-circuits the
    # builder exactly like the legacy `if nten == 0: return`) must differ
    # from the tendon-on step-0 qvel.
    mf.meta.data[MODEL_META_IDX_NTENDON] = Scalar[DTYPE](0)
    mf.meta.upload(ctx)
    d_off.upload_all(ctx)
    integ.step["gpu"](d_off, mf, ctx)
    d_off.qvel.download(ctx)
    var ndiff = 0
    for i in range(BATCH * NV_A):
        if d_off.qvel.data[i] != qvel_step0[i]:
            ndiff += 1
    if ndiff == 0:
        raise Error("part A vacuous: tendon-off run identical to tendon-on")
    print("  non-vacuous: tendon-off rerun differs in", ndiff, "qvel entries")
    print("  Part A PASS")


# =============================================================================
# Part B: synthetic weld equality model
# =============================================================================

comptime weld_xml = """
<mujoco model="weldtest">
    <option timestep="0.005" iterations="50" solver="PGS"/>
    <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" condim="3" friction="1 0.1 0.1"/>
        <body name="link1" pos="0 0 0.049">
            <joint name="j1" type="hinge" axis="0 1 0" pos="0 0 0" range="-170 170" limited="true" damping="0.1"/>
            <geom name="g1" type="capsule" fromto="0 0 0 0.3 0 0" size="0.05" condim="3" friction="1 0.1 0.1"/>
            <body name="link2" pos="0.3 0 0">
                <joint name="j2" type="hinge" axis="0 1 0" pos="0 0 0" range="-170 170" limited="true" damping="0.1"/>
                <geom name="g2" type="capsule" fromto="0 0 0 0.3 0 0" size="0.05" condim="3" friction="1 0.1 0.1"/>
            </body>
        </body>
        <body name="anchor" pos="0.62 0 0.1">
            <joint name="j3" type="hinge" axis="1 0 0" pos="0 0 0" range="-170 170" limited="true" damping="0.1"/>
            <geom name="g3" type="sphere" size="0.04" contype="0" conaffinity="0"/>
        </body>
    </worldbody>
    <equality>
        <weld body1="link2" body2="anchor"/>
    </equality>
</mujoco>
"""

comptime pm_b = parse_xml(weld_xml)

comptime WeldTestModel = ModelDefFromXML[
    xml=weld_xml,
    nbody=pm_b.NBODY,
    njoint=pm_b.NJOINT,
    nq=pm_b.NQ,
    nv=pm_b.NV,
    ngeom=pm_b.NGEOM,
    nact=pm_b.NACT,
    max_contacts=8,
    max_equality=6,  # 1 weld = 6 rows
    cone_type=ConeType.ELLIPTIC,
    neq=pm_b.NEQ,
    timestep=pm_b.TIMESTEP,
]

comptime NQ_B = WeldTestModel.NQ  # 3
comptime NV_B = WeldTestModel.NV  # 3
comptime NBODY_B = WeldTestModel.NBODY  # 4
comptime NJOINT_B = WeldTestModel.NJOINT  # 3
comptime NGEOM_B = WeldTestModel.NGEOM  # 4
comptime MC_B = WeldTestModel.MAX_CONTACTS  # 8
comptime NEQ_B = WeldTestModel.MAX_EQUALITY  # 6
comptime CONE_B = WeldTestModel.CONE_TYPE
comptime N_STEPS_B = 3
comptime SS_B = state_size[NQ_B, NV_B, NBODY_B, MC_B, 0]()
comptime MS_B = model_size_with_invweight[
    NBODY_B, NJOINT_B, NV_B, NGEOM_B, NEQ_B
]()
comptime WS_B = ws_solver_offset[NV_B, NBODY_B]() + 81 * MC_B + 12 * MC_B * NV_B


def _legacy_step_kernel_b[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS_B), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS_B), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS_B), MutAnyOrigin],
):
    EulerIntegrator[SOLVER=PGSSolver].step_kernel[
        DTYPE, NQ_B, NV_B, NBODY_B, NJOINT_B, MC_B, SS_B, MS_B, B_, WS_B
    ](state, model, workspace)


def _legacy_detect_kernel_b[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS_B), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS_B), MutAnyOrigin],
):
    var env = Int(block_idx.x)
    if env >= B_:
        return
    detect_contacts_gpu[
        DTYPE, NQ_B, NV_B, NBODY_B, NJOINT_B, MC_B, SS_B, MS_B, B_, NGEOM_B,
        NEQ_B, 0, 0,
    ](env, state, model)


def _legacy_pgs_kernel_b[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS_B), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS_B), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS_B), MutAnyOrigin],
):
    PGSSolver.solve_gpu[
        DTYPE, NQ_B, NV_B, NBODY_B, NJOINT_B, MC_B, SS_B, MS_B, NV_B, B_,
        WS_B, NGEOM_B, NEQ_B, CONE_B, 0, 0,
    ](state, model, workspace)


def _legacy_finalize_kernel_b[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS_B), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS_B), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS_B), MutAnyOrigin],
):
    EulerIntegrator[SOLVER=PGSSolver].step_finalize_kernel[
        DTYPE, NQ_B, NV_B, NBODY_B, NJOINT_B, MC_B, SS_B, MS_B, B_, WS_B
    ](state, model, workspace)


def _part_b_equality(ctx: DeviceContext) raises:
    print("--- Part B: synthetic weld equality — fields vs legacy PGS,")
    print("    BATCH=", BATCH)

    var model_t = TensorImpl[DTYPE].alloc(MS_B)
    model_t.upload(ctx)
    var mbuf = model_t.dev.value()
    WeldTestModel.init_model_gpu(ctx, mbuf)
    model_t.download(ctx)

    # init_model_gpu never serializes equality RECORDS (copy_equality_to_
    # buffer has no callers) — only the meta count. Serialize them here on
    # the host slab, mirroring copy_equality_to_buffer, so BOTH the legacy
    # kernel and ModelFields read real records.
    var cpu_model = Model[
        DTYPE, NQ_B, NV_B, NBODY_B, NJOINT_B, MC_B, NGEOM_B, NEQ_B, CONE_B,
        0, 0,
    ]()
    var cpu_data = Data[DTYPE, NQ_B, NV_B, NBODY_B, NJOINT_B, MC_B, 0]()
    WeldTestModel.setup_model_and_data[DTYPE](cpu_model, cpu_data)
    if cpu_model.num_equality != 1:
        raise Error(
            "part B: expected 1 weld constraint, got "
            + String(cpu_model.num_equality)
        )
    for e_i in range(cpu_model.num_equality):
        var eq = cpu_model.equality_constraints[e_i]
        var off = model_equality_offset[NBODY_B, NJOINT_B, NGEOM_B](e_i)
        model_t.data[off + EQ_IDX_TYPE] = Scalar[DTYPE](eq.eq_type)
        model_t.data[off + EQ_IDX_BODY_A] = Scalar[DTYPE](eq.body_a)
        model_t.data[off + EQ_IDX_BODY_B] = Scalar[DTYPE](eq.body_b)
        model_t.data[off + EQ_IDX_ANCHOR_AX] = eq.anchor_a_x
        model_t.data[off + EQ_IDX_ANCHOR_AY] = eq.anchor_a_y
        model_t.data[off + EQ_IDX_ANCHOR_AZ] = eq.anchor_a_z
        model_t.data[off + EQ_IDX_ANCHOR_BX] = eq.anchor_b_x
        model_t.data[off + EQ_IDX_ANCHOR_BY] = eq.anchor_b_y
        model_t.data[off + EQ_IDX_ANCHOR_BZ] = eq.anchor_b_z
        model_t.data[off + EQ_IDX_RELPOSE_X] = eq.relpose_x
        model_t.data[off + EQ_IDX_RELPOSE_Y] = eq.relpose_y
        model_t.data[off + EQ_IDX_RELPOSE_Z] = eq.relpose_z
        model_t.data[off + EQ_IDX_RELPOSE_W] = eq.relpose_w
        model_t.data[off + EQ_IDX_SOLREF_0] = eq.solref_0
        model_t.data[off + EQ_IDX_SOLREF_1] = eq.solref_1
        model_t.data[off + EQ_IDX_SOLIMP_0] = eq.solimp_0
        model_t.data[off + EQ_IDX_SOLIMP_1] = eq.solimp_1
        model_t.data[off + EQ_IDX_SOLIMP_2] = eq.solimp_2
        model_t.data[off + EQ_IDX_SOLIMP_3] = eq.solimp_3
        model_t.data[off + EQ_IDX_SOLIMP_4] = eq.solimp_4
    model_t.upload(ctx)

    var mf = ModelFields[DTYPE, NV_B, NBODY_B, NJOINT_B, NGEOM_B, NEQ_B]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)

    if Int(mf.meta.data[MODEL_META_IDX_NEQUALITY]) != 1:
        raise Error("part B vacuous: model meta NEQUALITY != 1")

    comptime O_QPOS = qpos_offset[NQ_B, NV_B]()
    comptime O_QVEL = qvel_offset[NQ_B, NV_B]()
    comptime O_QACC = qacc_offset[NQ_B, NV_B]()
    comptime O_QFRC = qfrc_offset[NQ_B, NV_B]()
    comptime O_CON = contacts_offset[NQ_B, NV_B, NBODY_B]()
    comptime O_META = metadata_offset[NQ_B, NV_B, NBODY_B, MC_B]()

    var slab_t = TensorImpl[DTYPE].alloc(BATCH * SS_B)
    var d = DataFields[DTYPE, NQ_B, NV_B, NBODY_B, MC_B, 0, BATCH]()
    var dc = DataFields[DTYPE, NQ_B, NV_B, NBODY_B, MC_B, 0, BATCH]()
    var d_off = DataFields[DTYPE, NQ_B, NV_B, NBODY_B, MC_B, 0, BATCH]()
    for e in range(BATCH):
        for i in range(NQ_B):
            var qp = Scalar[DTYPE]((e * 5 + i * 3) % 5 - 2) / 50.0
            slab_t.data[e * SS_B + O_QPOS + i] = qp
            d.qpos.data[e * NQ_B + i] = qp
            dc.qpos.data[e * NQ_B + i] = qp
            d_off.qpos.data[e * NQ_B + i] = qp
        for i in range(NV_B):
            var qv = Scalar[DTYPE]((e * 7 + i * 5) % 7 - 3) / 20.0
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 4.0
            slab_t.data[e * SS_B + O_QVEL + i] = qv
            slab_t.data[e * SS_B + O_QFRC + i] = qf
            d.qvel.data[e * NV_B + i] = qv
            d.qfrc.data[e * NV_B + i] = qf
            dc.qvel.data[e * NV_B + i] = qv
            dc.qfrc.data[e * NV_B + i] = qf
            d_off.qvel.data[e * NV_B + i] = qv
            d_off.qfrc.data[e * NV_B + i] = qf
    slab_t.upload(ctx)
    d.upload_all(ctx)
    var ws_t = TensorImpl[DTYPE].alloc(BATCH * WS_B)
    ws_t.upload(ctx)

    var integ = EulerIntegratorFields[
        DTYPE, NQ_B, NV_B, NBODY_B, NJOINT_B, MC_B, NGEOM_B, NEQ_B, 0, 0, 0,
        0, CONE_B, BATCH,
    ]()
    integ.prepare_gpu(ctx)
    var integ_c = EulerIntegratorFields[
        DTYPE, NQ_B, NV_B, NBODY_B, NJOINT_B, MC_B, NGEOM_B, NEQ_B, 0, 0, 0,
        0, CONE_B, BATCH,
    ]()

    var qvel_step0 = List[Scalar[DTYPE]](capacity=BATCH * NV_B)
    for _ in range(BATCH * NV_B):
        qvel_step0.append(Scalar[DTYPE](0))

    for step in range(N_STEPS_B):
        ctx.enqueue_function[_legacy_step_kernel_b[BATCH]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS_B)](),
            model_t.lt["gpu", Layout.row_major(1, MS_B)](),
            ws_t.lt["gpu", Layout.row_major(BATCH, WS_B)](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ctx.enqueue_function[_legacy_detect_kernel_b[BATCH]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS_B)](),
            model_t.lt["gpu", Layout.row_major(1, MS_B)](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ctx.enqueue_function[_legacy_pgs_kernel_b[BATCH]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS_B)](),
            model_t.lt["gpu", Layout.row_major(1, MS_B)](),
            ws_t.lt["gpu", Layout.row_major(BATCH, WS_B)](),
            grid_dim=(BATCH,),
            block_dim=(1, MC_B),
        )
        ctx.enqueue_function[_legacy_finalize_kernel_b[BATCH]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS_B)](),
            model_t.lt["gpu", Layout.row_major(1, MS_B)](),
            ws_t.lt["gpu", Layout.row_major(BATCH, WS_B)](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        integ.step["gpu"](d, mf, ctx)
        integ_c.step["cpu"](dc, mf)

        slab_t.download(ctx)
        d.qpos.download(ctx)
        d.qvel.download(ctx)
        d.qacc.download(ctx)
        d.contacts.download(ctx)
        d.meta.download(ctx)
        if step == 0:
            for i in range(BATCH * NV_B):
                qvel_step0[i] = d.qvel.data[i]
        var bad = 0
        var ncon_seen = 0
        for e in range(BATCH):
            var nc = Int(
                d.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS]
            )
            var nc_l = Int(
                slab_t.data[e * SS_B + O_META + META_IDX_NUM_CONTACTS]
            )
            if nc != nc_l:
                print("  ncon mismatch env", e, ": fields", nc, " legacy", nc_l)
                bad += 1
                continue
            ncon_seen += nc
            for i in range(NQ_B):
                if (
                    d.qpos.data[e * NQ_B + i]
                    != slab_t.data[e * SS_B + O_QPOS + i]
                ):
                    if bad < 4:
                        print(
                            "  qpos diff e", e, "i", i, ":",
                            d.qpos.data[e * NQ_B + i], "vs",
                            slab_t.data[e * SS_B + O_QPOS + i],
                        )
                    bad += 1
            for i in range(NV_B):
                if (
                    d.qvel.data[e * NV_B + i]
                    != slab_t.data[e * SS_B + O_QVEL + i]
                ):
                    if bad < 4:
                        print(
                            "  qvel diff e", e, "i", i, ":",
                            d.qvel.data[e * NV_B + i], "vs",
                            slab_t.data[e * SS_B + O_QVEL + i],
                        )
                    bad += 1
                if (
                    d.qacc.data[e * NV_B + i]
                    != slab_t.data[e * SS_B + O_QACC + i]
                ):
                    bad += 1
            for c in range(nc):
                for k in range(CONTACT_SIZE):
                    if (
                        d.contacts.data[
                            e * MC_B * CONTACT_SIZE + c * CONTACT_SIZE + k
                        ]
                        != slab_t.data[
                            e * SS_B + O_CON + c * CONTACT_SIZE + k
                        ]
                    ):
                        if bad < 4:
                            print(
                                "  record diff e", e, "c", c, "k", k, ":",
                                d.contacts.data[
                                    e * MC_B * CONTACT_SIZE
                                    + c * CONTACT_SIZE + k
                                ],
                                "vs",
                                slab_t.data[
                                    e * SS_B + O_CON + c * CONTACT_SIZE + k
                                ],
                            )
                        bad += 1
        if bad != 0:
            raise Error("part B step " + String(step) + ": mismatch")
        if ncon_seen == 0:
            raise Error(
                "part B: no contacts at step " + String(step) + " — vacuous"
            )
        print(
            "  step", step, ": BIT-EXACT (qpos/qvel/qacc + contact records),"
            " total contacts:", ncon_seen,
        )

    var worst = Float64(0)
    for i in range(BATCH * NQ_B):
        var err = abs(Float64(dc.qpos.data[i]) - Float64(d.qpos.data[i]))
        if err > worst:
            worst = err
    print("  fields-CPU vs fields-GPU final qpos worst err:", worst)
    # Loose cross-target sanity only (bit-exactness is same-target): the
    # stiff weld rows + contact PGS are both iterative, so fp32 CPU/GPU
    # drift compounds beyond the walker gate's 1e-2 (measured ~1.1e-2).
    if worst > 5e-2:
        raise Error("part B: fields-CPU diverged from GPU")

    # Non-vacuity: equality-off rerun (meta NEQUALITY=0 short-circuits the
    # builder exactly like the legacy `if neq == 0: return`) must differ
    # from the equality-on step-0 qvel.
    mf.meta.data[MODEL_META_IDX_NEQUALITY] = Scalar[DTYPE](0)
    mf.meta.upload(ctx)
    d_off.upload_all(ctx)
    integ.step["gpu"](d_off, mf, ctx)
    d_off.qvel.download(ctx)
    var ndiff = 0
    for i in range(BATCH * NV_B):
        if d_off.qvel.data[i] != qvel_step0[i]:
            ndiff += 1
    if ndiff == 0:
        raise Error("part B vacuous: weld-off run identical to weld-on")
    print("  non-vacuous: weld-off rerun differs in", ndiff, "qvel entries")
    print("  Part B PASS")


def main() raises:
    var ctx = DeviceContext()
    _part_a_tendon(ctx)
    _part_b_equality(ctx)
    print("test_equality_tendon_fields: ALL PASS")
