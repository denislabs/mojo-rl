"""P4 gate: standalone Newton contact solve — fields vs legacy, BIT-EXACT.

Walker2D dropped onto the floor (rootz=1.10, feet penetrating), BATCH=2,
3 successive solves with qvel/qfrc perturbed between rounds. Env 1
additionally has one limited hinge pushed past its upper range so the
joint-limit rows activate (ELLIPTIC: the post-core `_limits_env_fields`
tail; PYRAMIDAL: the inline limit-edge rows of the Newton core); env 0
stays mid-range (limit rows inactive) — non-vacuity is asserted host-side
from the model ranges.

Per round, per side: smooth prep (legacy `step_kernel` vs the fields prep
chain of EulerIntegratorFields.step) -> contact detection -> Newton solve
(legacy `NewtonSolver.solve_gpu`, 2D launch, vs `solve_newton_fields`).
qacc_constrained AND the solved contact force records must be BIT-EXACT.
Both cone types are gated (cone type is a solver comptime param, not model
data — walker2d model, ELLIPTIC and PYRAMIDAL legs). A fields-CPU smoke
run checks the single-source CPU path stays close to GPU.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_newton_solve_fields.mojo
"""

from std.math import abs
from std.gpu import block_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import (
    DataFields,
    ModelFields,
    DynamicsScratch,
    ContactScratch,
)
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.joint_types import JNT_HINGE, JNT_SLIDE
from mojo_rl.physics3d.integrator.euler_integrator import EulerIntegrator
from mojo_rl.physics3d.integrator.euler_fields import (
    _armature_kernel,
    _fnet_passive_kernel,
    _qacc_writeback_kernel,
    _armature_env_fields,
    _fnet_passive_env_fields,
    _qacc_writeback_env_fields,
)
from mojo_rl.physics3d.kinematics.forward_kinematics_fields import (
    forward_kinematics_fields,
    compute_body_velocities_fields,
)
from mojo_rl.physics3d.dynamics.subtree_com_fields import (
    compute_subtree_com_fields,
)
from mojo_rl.physics3d.dynamics.cdof_fields import compute_cdof_fields
from mojo_rl.physics3d.dynamics.mass_matrix_fields import (
    compute_mass_matrix_fields,
)
from mojo_rl.physics3d.dynamics.ldl_fields import (
    ldl_factor_fields,
    ldl_solve_fields,
    compute_m_inv_fields,
)
from mojo_rl.physics3d.dynamics.rne_fields import (
    compute_bias_forces_rne_fields,
)
from mojo_rl.physics3d.collision.contact_detection_fields import (
    detect_contacts_fields,
)
from mojo_rl.physics3d.solver.newton_solver import NewtonSolver
from mojo_rl.physics3d.solver.newton_solve_fields import solve_newton_fields
from mojo_rl.physics3d.collision.contact_detection import detect_contacts_gpu
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    ws_solver_offset,
    ws_qacc_constrained_offset,
    qpos_offset,
    qvel_offset,
    qfrc_offset,
    contacts_offset,
    metadata_offset,
    META_IDX_NUM_CONTACTS,
    METADATA_SIZE,
    CONTACT_SIZE,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
)
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel

comptime DTYPE = DType.float32
comptime NQ = Walker2dModel.NQ
comptime NV = Walker2dModel.NV
comptime NBODY = Walker2dModel.NBODY
comptime NJOINT = Walker2dModel.NJOINT
comptime NGEOM = Walker2dModel.NGEOM
comptime MC = Walker2dModel.MAX_CONTACTS
comptime BATCH = 2
comptime N_ROUNDS = 3
comptime SS = state_size[NQ, NV, NBODY, MC, 0]()
comptime MS = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
# PGS-sized workspace (Newton uses a prefix: 35*MC + 6*MC*NV).
comptime WS = ws_solver_offset[NV, NBODY]() + 81 * MC + 12 * MC * NV

comptime O_QPOS = qpos_offset[NQ, NV]()
comptime O_QVEL = qvel_offset[NQ, NV]()
comptime O_QFRC = qfrc_offset[NQ, NV]()
comptime O_CON = contacts_offset[NQ, NV, NBODY]()
comptime O_META = metadata_offset[NQ, NV, NBODY, MC]()
comptime O_QC = ws_qacc_constrained_offset[NV, NBODY]()


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


def _legacy_detect_kernel[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS), MutAnyOrigin],
):
    var env = Int(block_idx.x)
    if env >= B_:
        return
    detect_contacts_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, SS, MS, B_, NGEOM
    ](env, state, model)


def _legacy_newton_kernel[
    B_: Int, CONE_T: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS), MutAnyOrigin],
):
    NewtonSolver.solve_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, SS, MS, NV, B_, WS, NGEOM,
        0, CONE_T, 0, 0,
    ](state, model, workspace)


def _fields_prep[
    target: StaticString
](
    mut d: DataFields[DTYPE, NQ, NV, NBODY, MC, 0, BATCH],
    mut mf: ModelFields[DTYPE, NV, NBODY, NJOINT, NGEOM, 0, 0, 0, 0, 0],
    mut scratch: DynamicsScratch[DTYPE, NV, NBODY, BATCH],
    ctx: Optional[DeviceContext],
) raises:
    """Smooth-dynamics prep + detection, mirroring EulerIntegratorFields.step
    up to the constraint seam (order verbatim)."""
    forward_kinematics_fields[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0,
        BATCH,
    ](d, mf, ctx)
    compute_body_velocities_fields[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0,
        BATCH,
    ](d, mf, ctx)
    compute_subtree_com_fields[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0,
        BATCH,
    ](d, mf, ctx)
    compute_cdof_fields[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0,
        BATCH,
    ](d, mf, scratch, ctx)
    compute_mass_matrix_fields[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0,
        BATCH,
    ](d, mf, scratch, ctx)

    comptime L_JOINT = Layout.row_major(NJOINT, MODEL_JOINT_SIZE)
    comptime L_M = Layout.row_major(BATCH, NV * NV)
    comptime L_NV = Layout.row_major(BATCH, NV)
    comptime L_QPOS = Layout.row_major(BATCH, NQ)

    comptime if target == "cpu":
        # CPU armature/fnet/qacc-writeback via the same env bodies the GPU
        # kernels wrap (mirrors EulerIntegratorFields.step's CPU branch).
        var joints_v = mf.joints.lt["cpu", L_JOINT]()
        var M_v = scratch.M.lt["cpu", L_M]()
        for e in range(BATCH):
            _armature_env_fields[DTYPE, NV, NJOINT, BATCH](e, joints_v, M_v)
        ldl_factor_fields[target, DTYPE, NV, NBODY, BATCH](scratch, ctx)
        compute_m_inv_fields[target, DTYPE, NV, NBODY, BATCH](scratch, ctx)
        compute_bias_forces_rne_fields[
            target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0,
            BATCH,
        ](d, mf, scratch, ctx)
        var qpos_v = d.qpos.lt["cpu", L_QPOS]()
        var qvel_v = d.qvel.lt["cpu", L_NV]()
        var qfrc_v = d.qfrc.lt["cpu", L_NV]()
        var bias_v = scratch.bias.lt["cpu", L_NV]()
        var fnet_v = scratch.fnet.lt["cpu", L_NV]()
        for e in range(BATCH):
            _fnet_passive_env_fields[DTYPE, NQ, NV, NJOINT, BATCH](
                e, qpos_v, qvel_v, qfrc_v, joints_v, bias_v, fnet_v
            )
        ldl_solve_fields[target, DTYPE, NV, NBODY, BATCH](scratch, ctx)
        var qacc_ws_v = scratch.qacc_ws.lt["cpu", L_NV]()
        var qacc_v = d.qacc.lt["cpu", L_NV]()
        var qacc_c_v = scratch.qacc_constrained.lt["cpu", L_NV]()
        for e in range(BATCH):
            _qacc_writeback_env_fields[DTYPE, NV, BATCH](
                e, qacc_ws_v, qacc_v, qacc_c_v
            )
    else:
        ctx.value().enqueue_function[
            _armature_kernel[DTYPE, NV, NJOINT, BATCH]
        ](
            mf.joints.lt["gpu", L_JOINT](),
            scratch.M.lt["gpu", L_M](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ldl_factor_fields[target, DTYPE, NV, NBODY, BATCH](scratch, ctx)
        compute_m_inv_fields[target, DTYPE, NV, NBODY, BATCH](scratch, ctx)
        compute_bias_forces_rne_fields[
            target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0,
            BATCH,
        ](d, mf, scratch, ctx)
        ctx.value().enqueue_function[
            _fnet_passive_kernel[DTYPE, NQ, NV, NJOINT, BATCH]
        ](
            d.qpos.lt["gpu", L_QPOS](),
            d.qvel.lt["gpu", L_NV](),
            d.qfrc.lt["gpu", L_NV](),
            mf.joints.lt["gpu", L_JOINT](),
            scratch.bias.lt["gpu", L_NV](),
            scratch.fnet.lt["gpu", L_NV](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ldl_solve_fields[target, DTYPE, NV, NBODY, BATCH](scratch, ctx)
        ctx.value().enqueue_function[
            _qacc_writeback_kernel[DTYPE, NV, BATCH]
        ](
            scratch.qacc_ws.lt["gpu", L_NV](),
            d.qacc.lt["gpu", L_NV](),
            scratch.qacc_constrained.lt["gpu", L_NV](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )

    detect_contacts_fields[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0,
        BATCH,
    ](d, mf, ctx)


def _find_limited_joint(
    model_data: List[Scalar[DTYPE]],
) -> Tuple[Int, Scalar[DTYPE]]:
    """First HINGE/SLIDE joint with a finite range: (qpos_adr, rmax)."""
    for j in range(NJOINT):
        var j_off = NBODY * MODEL_BODY_SIZE + j * MODEL_JOINT_SIZE
        var jtype = Int(model_data[j_off + JOINT_IDX_TYPE])
        if jtype != JNT_HINGE and jtype != JNT_SLIDE:
            continue
        var rmin = model_data[j_off + JOINT_IDX_RANGE_MIN]
        var rmax = model_data[j_off + JOINT_IDX_RANGE_MAX]
        if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
            continue
        var qpos_adr = Int(model_data[j_off + JOINT_IDX_QPOS_ADR])
        return (qpos_adr, rmax)
    return (-1, Scalar[DTYPE](0))


def _count_violated_limits(
    model_data: List[Scalar[DTYPE]],
    qpos: List[Scalar[DTYPE]],
    env: Int,
) -> Int:
    """Host-side count of active joint-limit rows for one env."""
    var count = 0
    for j in range(NJOINT):
        var j_off = NBODY * MODEL_BODY_SIZE + j * MODEL_JOINT_SIZE
        var jtype = Int(model_data[j_off + JOINT_IDX_TYPE])
        if jtype != JNT_HINGE and jtype != JNT_SLIDE:
            continue
        var rmin = model_data[j_off + JOINT_IDX_RANGE_MIN]
        var rmax = model_data[j_off + JOINT_IDX_RANGE_MAX]
        if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
            continue
        var qpos_adr = Int(model_data[j_off + JOINT_IDX_QPOS_ADR])
        var pos = qpos[env * NQ + qpos_adr]
        if pos - rmin < Scalar[DTYPE](0):
            count += 1
        if rmax - pos < Scalar[DTYPE](0):
            count += 1
    return count


def run_leg[CONE_T: Int](ctx: DeviceContext, leg: String) raises:
    print("--- Newton solve leg:", leg, "(BATCH=", BATCH, ")")

    # === Model ===
    var model_t = TensorImpl[DTYPE].alloc(MS)
    model_t.upload(ctx)
    var mbuf = model_t.dev.value()
    Walker2dModel.init_model_gpu(ctx, mbuf)
    model_t.download(ctx)
    var mf = ModelFields[DTYPE, NV, NBODY, NJOINT, NGEOM]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)

    # === State (walker on the floor; env 1 with one joint past its limit) ===
    var slab_t = TensorImpl[DTYPE].alloc(BATCH * SS)
    var d = DataFields[DTYPE, NQ, NV, NBODY, MC, 0, BATCH]()
    var lim = _find_limited_joint(model_t.data)
    var lim_qpos_adr = lim[0]
    var lim_rmax = lim[1]
    if lim_qpos_adr < 0:
        raise Error("no limited joint found — limit leg vacuous")
    for e in range(BATCH):
        for i in range(NQ):
            var qp = Scalar[DTYPE]((e * 5 + i * 3) % 5 - 2) / 40.0
            if i == 1:
                qp = 1.10  # feet penetrate the floor
            slab_t.data[e * SS + O_QPOS + i] = qp
            d.qpos.data[e * NQ + i] = qp
        # Pull every limited joint of this env into its range interior
        # (walker2d thigh/leg/foot ranges exclude small positive angles, so
        # the raw pattern would violate limits in BOTH envs); env 1 then
        # gets the chosen joint pushed past its upper limit.
        for j in range(NJOINT):
            var j_off = NBODY * MODEL_BODY_SIZE + j * MODEL_JOINT_SIZE
            var jtype = Int(model_t.data[j_off + JOINT_IDX_TYPE])
            if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                continue
            var rmin = model_t.data[j_off + JOINT_IDX_RANGE_MIN]
            var rmax = model_t.data[j_off + JOINT_IDX_RANGE_MAX]
            if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
                continue
            var qpos_adr = Int(model_t.data[j_off + JOINT_IDX_QPOS_ADR])
            # Clamp the pattern value into the range interior with a small
            # margin — keeps the near-straight standing pose (feet on floor)
            # while deactivating the limit rows.
            var qp_in = d.qpos.data[e * NQ + qpos_adr]
            if qp_in > rmax - Scalar[DTYPE](0.1):
                qp_in = rmax - Scalar[DTYPE](0.1)
            if qp_in < rmin + Scalar[DTYPE](0.1):
                qp_in = rmin + Scalar[DTYPE](0.1)
            if e == 1 and qpos_adr == lim_qpos_adr:
                qp_in = lim_rmax + Scalar[DTYPE](0.05)  # past upper limit
            slab_t.data[e * SS + O_QPOS + qpos_adr] = qp_in
            d.qpos.data[e * NQ + qpos_adr] = qp_in
        for i in range(NV):
            var qv = Scalar[DTYPE]((e * 7 + i * 5) % 7 - 3) / 20.0
            if i == 1:
                qv = -0.5  # falling
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 4.0
            slab_t.data[e * SS + O_QVEL + i] = qv
            slab_t.data[e * SS + O_QFRC + i] = qf
            d.qvel.data[e * NV + i] = qv
            d.qfrc.data[e * NV + i] = qf
    slab_t.upload(ctx)
    d.upload_all(ctx)
    var ws_t = TensorImpl[DTYPE].alloc(BATCH * WS)
    ws_t.upload(ctx)

    # Non-vacuity of the limit rows: env 1 must violate, env 0 must not.
    var qpos_host = List[Scalar[DTYPE]]()
    for e in range(BATCH):
        for i in range(NQ):
            qpos_host.append(d.qpos.data[e * NQ + i])
    var nlim0 = _count_violated_limits(model_t.data, qpos_host, 0)
    var nlim1 = _count_violated_limits(model_t.data, qpos_host, 1)
    if nlim0 != 0:
        raise Error("env 0 unexpectedly violates a joint limit")
    if nlim1 < 1:
        raise Error("env 1 has no violated joint limit — limit rows vacuous")
    print("  limit rows: env0", nlim0, " env1", nlim1, "(non-vacuous)")

    var scratch = DynamicsScratch[DTYPE, NV, NBODY, BATCH]()
    var cscratch = ContactScratch[DTYPE, NV, MC, BATCH]()
    scratch.upload_all(ctx)
    cscratch.upload_all(ctx)

    for rnd in range(N_ROUNDS):
        # --- Legacy: step (smooth prep) -> detect -> Newton solve ---
        ctx.enqueue_function[_legacy_step_kernel[BATCH]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
            model_t.lt["gpu", Layout.row_major(1, MS)](),
            ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ctx.enqueue_function[_legacy_detect_kernel[BATCH]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
            model_t.lt["gpu", Layout.row_major(1, MS)](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ctx.enqueue_function[_legacy_newton_kernel[BATCH, CONE_T]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
            model_t.lt["gpu", Layout.row_major(1, MS)](),
            ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
            grid_dim=(BATCH,),
            block_dim=(1, MC),
        )

        # --- Fields: prep chain -> detect -> Newton solve ---
        _fields_prep["gpu"](d, mf, scratch, ctx)
        solve_newton_fields[
            "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0,
            CONE_T, BATCH,
        ](d, mf, scratch, cscratch, ctx)

        # --- Compare BIT-EXACT ---
        slab_t.download(ctx)
        ws_t.download(ctx)
        scratch.qacc_constrained.download(ctx)
        d.contacts.download(ctx)
        d.meta.download(ctx)
        var bad = 0
        var ncon_seen = 0
        for e in range(BATCH):
            var nc = Int(
                d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
            )
            var nc_l = Int(
                slab_t.data[e * SS + O_META + META_IDX_NUM_CONTACTS]
            )
            if nc != nc_l:
                print("  ncon mismatch env", e, ": fields", nc, " legacy", nc_l)
                bad += 1
                continue
            ncon_seen += nc
            for i in range(NV):
                var qc_f = scratch.qacc_constrained.data[e * NV + i]
                var qc_l = ws_t.data[e * WS + O_QC + i]
                if qc_f != qc_l:
                    if bad < 6:
                        print(
                            "  qacc_constrained diff e", e, "i", i, ":",
                            qc_f, "vs", qc_l,
                        )
                    bad += 1
            for c in range(nc):
                for k in range(CONTACT_SIZE):
                    var rec_f = d.contacts.data[
                        e * MC * CONTACT_SIZE + c * CONTACT_SIZE + k
                    ]
                    var rec_l = slab_t.data[
                        e * SS + O_CON + c * CONTACT_SIZE + k
                    ]
                    if rec_f != rec_l:
                        if bad < 6:
                            print(
                                "  record diff e", e, "c", c, "k", k, ":",
                                rec_f, "vs", rec_l,
                            )
                        bad += 1
        if bad != 0:
            raise Error(
                leg + " round " + String(rnd) + ": Newton solve mismatch"
            )
        if ncon_seen == 0:
            raise Error(
                leg + " round " + String(rnd) + ": no contacts — vacuous"
            )
        print(
            "  round", rnd,
            ": BIT-EXACT (qacc_constrained + contact records),"
            " total contacts:", ncon_seen,
        )

        # Perturb qvel/qfrc for the next round (both sides identically);
        # qpos/contact warm-start records stay device-consistent (the slab
        # host copy was just downloaded).
        if rnd + 1 < N_ROUNDS:
            for e in range(BATCH):
                for i in range(NV):
                    var dv = Scalar[DTYPE](
                        (e * 3 + i * 7 + rnd * 11) % 13 - 6
                    ) / 50.0
                    var df = Scalar[DTYPE](
                        (e * 9 + i * 5 + rnd * 17) % 11 - 5
                    ) / 10.0
                    slab_t.data[e * SS + O_QVEL + i] += dv
                    slab_t.data[e * SS + O_QFRC + i] += df
                    d.qvel.data[e * NV + i] += dv
                    d.qfrc.data[e * NV + i] += df
            slab_t.upload(ctx)
            d.qvel.upload(ctx)
            d.qfrc.upload(ctx)
    print("  PASS:", leg, "leg bit-exact over", N_ROUNDS, "rounds")


def run_cpu_smoke(ctx: DeviceContext) raises:
    """Single-source CPU path smoke: fields-CPU Newton solve close to
    fields-GPU (iterative solver -> loose cross-target tolerance)."""
    print("--- Newton solve fields-CPU vs fields-GPU smoke (ELLIPTIC)")
    var model_t = TensorImpl[DTYPE].alloc(MS)
    model_t.upload(ctx)
    var mbuf = model_t.dev.value()
    Walker2dModel.init_model_gpu(ctx, mbuf)
    model_t.download(ctx)
    var mf = ModelFields[DTYPE, NV, NBODY, NJOINT, NGEOM]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)

    var d = DataFields[DTYPE, NQ, NV, NBODY, MC, 0, BATCH]()
    var dc = DataFields[DTYPE, NQ, NV, NBODY, MC, 0, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            var qp = Scalar[DTYPE]((e * 5 + i * 3) % 5 - 2) / 40.0
            if i == 1:
                qp = 1.10
            d.qpos.data[e * NQ + i] = qp
            dc.qpos.data[e * NQ + i] = qp
        for i in range(NV):
            var qv = Scalar[DTYPE]((e * 7 + i * 5) % 7 - 3) / 20.0
            if i == 1:
                qv = -0.5
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 4.0
            d.qvel.data[e * NV + i] = qv
            d.qfrc.data[e * NV + i] = qf
            dc.qvel.data[e * NV + i] = qv
            dc.qfrc.data[e * NV + i] = qf
    d.upload_all(ctx)

    var scratch = DynamicsScratch[DTYPE, NV, NBODY, BATCH]()
    var cscratch = ContactScratch[DTYPE, NV, MC, BATCH]()
    scratch.upload_all(ctx)
    cscratch.upload_all(ctx)
    var scratch_c = DynamicsScratch[DTYPE, NV, NBODY, BATCH]()
    var cscratch_c = ContactScratch[DTYPE, NV, MC, BATCH]()

    _fields_prep["gpu"](d, mf, scratch, ctx)
    solve_newton_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0,
        ConeType.ELLIPTIC, BATCH,
    ](d, mf, scratch, cscratch, ctx)
    _fields_prep["cpu"](dc, mf, scratch_c, None)
    solve_newton_fields[
        "cpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0,
        ConeType.ELLIPTIC, BATCH,
    ](dc, mf, scratch_c, cscratch_c, None)

    scratch.qacc_constrained.download(ctx)
    var worst = Float64(0)
    for i in range(BATCH * NV):
        var g = Float64(scratch.qacc_constrained.data[i])
        var c = Float64(scratch_c.qacc_constrained.data[i])
        var err = abs(g - c) / (1.0 + abs(g))
        if err > worst:
            worst = err
    print("  fields-CPU vs fields-GPU qacc_constrained worst rel err:", worst)
    if worst > 1e-2:
        raise Error("fields-CPU Newton solve diverged from GPU")
    print("  PASS: fields-CPU within 1e-2 (relative)")


def main() raises:
    var ctx = DeviceContext()
    run_leg[ConeType.ELLIPTIC](ctx, "ELLIPTIC")
    run_leg[ConeType.PYRAMIDAL](ctx, "PYRAMIDAL")
    run_cpu_smoke(ctx)
    print("test_newton_solve_fields: ALL PASS")
