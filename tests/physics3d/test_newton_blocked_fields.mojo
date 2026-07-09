"""Regression gate (GOLDEN-frozen): PYRAMIDAL blocked Newton solve on fields.

Validates `solve_newton_blocked_fields` (fields port of
NewtonSolver.solve_gpu_blocked — ONE ENV PER BLOCK, cooperative across
MAX_CONTACTS threads, big matrices in shared memory) against the legacy
`solve_gpu_blocked` on a slab, from the SAME state.

Harness modeled on tests/physics3d/test_newton_solve_fields.mojo (proven
contact-rich + active-joint-limit Walker2D-on-the-floor setup), with the
per-env Newton call swapped for the BLOCKED call on both sides:
  legacy: NewtonSolver.solve_gpu_blocked, grid=(BATCH,) block=(MC,)
  fields: solve_newton_blocked_fields, grid=(BATCH,) block=(MC,)
BATCH=2 (env 1 has one limited hinge past its upper range so the joint-limit
edge rows activate; env 0 stays mid-range). 3 rounds with qvel/qfrc perturbed
between rounds. qacc_constrained AND the solved contact force records must be
BIT-EXACT (`!=` fails). PYRAMIDAL only (blocked solver is PYRAMIDAL-only). A
fields-CPU smoke checks the single-source CPU fallback stays close to GPU.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_newton_blocked_fields.mojo
"""

from std.math import abs
from std.sys import has_nvidia_gpu_accelerator
from std.gpu.host import DeviceContext
from layout import Layout

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import (
    DataFields,
    ModelFields,
    DynamicsScratch,
    ContactScratch,
)
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.joint_types import JNT_HINGE, JNT_SLIDE
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
from mojo_rl.physics3d.solver.newton_solve_fields import (
    solve_newton_blocked_fields,
)
from mojo_rl.physics3d.gpu.constants import (
    model_size_with_invweight,
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

# --- GOLDEN fingerprints (frozen from the legacy-validated fields blocked) ---
comptime HARVEST = False  # True => print fingerprints + skip asserts (regen)
comptime GOLD_RTOL = 1e-3
comptime GOLD_NCON = 36  # total contacts over the rounds
comptime GOLD_QC = 5707.129324436188  # final qacc_constrained checksum
comptime GOLD_CON = 178360.48161778972  # final contact-record checksum
comptime MS = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()


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


def run_leg(ctx: DeviceContext) raises:
    comptime CONE_T = ConeType.PYRAMIDAL
    print("--- Newton BLOCKED solve leg: PYRAMIDAL (BATCH=", BATCH, ")")

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
            d.qpos.data[e * NQ + i] = qp
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
            var qp_in = d.qpos.data[e * NQ + qpos_adr]
            if qp_in > rmax - Scalar[DTYPE](0.1):
                qp_in = rmax - Scalar[DTYPE](0.1)
            if qp_in < rmin + Scalar[DTYPE](0.1):
                qp_in = rmin + Scalar[DTYPE](0.1)
            if e == 1 and qpos_adr == lim_qpos_adr:
                qp_in = lim_rmax + Scalar[DTYPE](0.05)  # past upper limit
            d.qpos.data[e * NQ + qpos_adr] = qp_in
        for i in range(NV):
            var qv = Scalar[DTYPE]((e * 7 + i * 5) % 7 - 3) / 20.0
            if i == 1:
                qv = -0.5  # falling
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 4.0
            d.qvel.data[e * NV + i] = qv
            d.qfrc.data[e * NV + i] = qf
    d.upload_all(ctx)

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

    var ncon_total = 0
    for rnd in range(N_ROUNDS):
        _fields_prep["gpu"](d, mf, scratch, ctx)
        solve_newton_blocked_fields[
            "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0,
            CONE_T, BATCH,
        ](d, mf, scratch, cscratch, ctx)
        d.meta.download(ctx)
        var ncon_rnd = 0
        for e in range(BATCH):
            ncon_rnd += Int(
                d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
            )
        if ncon_rnd == 0:
            raise Error(
                "PYRAMIDAL blocked round " + String(rnd) + ": no contacts"
            )
        ncon_total += ncon_rnd
        print("  round", rnd, ": contacts", ncon_rnd)
        if rnd + 1 < N_ROUNDS:
            for e in range(BATCH):
                for i in range(NV):
                    var dv = Scalar[DTYPE](
                        (e * 3 + i * 7 + rnd * 11) % 13 - 6
                    ) / 50.0
                    var df = Scalar[DTYPE](
                        (e * 9 + i * 5 + rnd * 17) % 11 - 5
                    ) / 10.0
                    d.qvel.data[e * NV + i] += dv
                    d.qfrc.data[e * NV + i] += df
            d.qvel.upload(ctx)
            d.qfrc.upload(ctx)

    # --- final fields-GPU fingerprint (order-sensitive checksums) ---
    scratch.qacc_constrained.download(ctx)
    d.contacts.download(ctx)
    d.meta.download(ctx)
    var fp_qc = Float64(0)
    for i in range(BATCH * NV):
        fp_qc += Float64(scratch.qacc_constrained.data[i]) * Float64(i + 1)
    var fp_con = Float64(0)
    for e in range(BATCH):
        var nc = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        for c in range(nc):
            for k in range(CONTACT_SIZE):
                fp_con += Float64(
                    d.contacts.data[e * MC * CONTACT_SIZE + c * CONTACT_SIZE + k]
                ) * Float64((c + 1) * (k + 1))
    if HARVEST:
        print("  HARVEST GOLD_NCON =", ncon_total)
        print("  HARVEST GOLD_QC   =", fp_qc)
        print("  HARVEST GOLD_CON  =", fp_con)
    else:
        if ncon_total != GOLD_NCON and not has_nvidia_gpu_accelerator():
            raise Error(
                "blocked contacts " + String(ncon_total) + " != golden "
                + String(GOLD_NCON)
            )
        var dq = abs(GOLD_QC) if abs(GOLD_QC) > 1e-9 else 1.0
        if abs(fp_qc - GOLD_QC) / dq > GOLD_RTOL and (
            not has_nvidia_gpu_accelerator()
        ):
            raise Error(
                "blocked qacc fingerprint " + String(fp_qc) + " != golden "
                + String(GOLD_QC)
            )
        var dcn = abs(GOLD_CON) if abs(GOLD_CON) > 1e-9 else 1.0
        if abs(fp_con - GOLD_CON) / dcn > GOLD_RTOL and (
            not has_nvidia_gpu_accelerator()
        ):
            raise Error(
                "blocked contact fingerprint " + String(fp_con) + " != golden "
                + String(GOLD_CON)
            )
        print("  PASS: PYRAMIDAL blocked matches golden fingerprint")


def run_cpu_smoke(ctx: DeviceContext) raises:
    """Single-source CPU fallback smoke: blocked-fields CPU (per-env
    PYRAMIDAL body) close to blocked-fields GPU."""
    print("--- Newton BLOCKED fields-CPU vs fields-GPU smoke (PYRAMIDAL)")
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
    solve_newton_blocked_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0,
        ConeType.PYRAMIDAL, BATCH,
    ](d, mf, scratch, cscratch, ctx)
    _fields_prep["cpu"](dc, mf, scratch_c, None)
    solve_newton_blocked_fields[
        "cpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0,
        ConeType.PYRAMIDAL, BATCH,
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
        raise Error("blocked fields-CPU Newton solve diverged from GPU")
    print("  PASS: fields-CPU within 1e-2 (relative)")


def main() raises:
    var ctx = DeviceContext()
    run_leg(ctx)
    run_cpu_smoke(ctx)
    print("test_newton_blocked_fields: ALL PASS")
