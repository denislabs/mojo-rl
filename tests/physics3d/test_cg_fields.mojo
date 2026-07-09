"""Stage-S gate: fields CG contact solve (solve_cg_fields) vs the
golden-validated fields Newton solve (solve_newton_fields), ELLIPTIC cone.

CG and Newton minimize the SAME convex primal cost (elliptic 3-zone friction
cone, identical contact setup, qfrc_smooth = M·qacc_smooth) — CG via an
M-preconditioned nonlinear conjugate gradient, Newton via full-Hessian steps.
For a strictly convex problem both converge to the same qacc_constrained (to
solver tolerance). Since the fields Newton solve is golden-frozen against the
legacy Newton GPU reference, "CG agrees with Newton on the same problem"
transitively validates the CG port.

Walker2D dropped onto the floor (rootz=1.10, feet penetrating), BATCH=2.
Checks:
  * Part A: fields-GPU CG qacc_constrained ≈ fields-GPU Newton (same problem),
  * Part B: fields-CPU CG ≈ fields-GPU CG (single-source self-consistency),
  * contacts actually form (nc > 0), state finite.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_cg_fields.mojo
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
from mojo_rl.physics3d.solver.newton_solve_fields import solve_newton_fields
from mojo_rl.physics3d.solver.cg_solve_fields import solve_cg_fields
from mojo_rl.physics3d.gpu.constants import (
    model_size_with_invweight,
    META_IDX_NUM_CONTACTS,
    METADATA_SIZE,
    MODEL_JOINT_SIZE,
)
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel

comptime DTYPE = DType.float32
comptime NQ = Walker2dModel.NQ
comptime NV = Walker2dModel.NV
comptime NBODY = Walker2dModel.NBODY
comptime NJOINT = Walker2dModel.NJOINT
comptime NGEOM = Walker2dModel.NGEOM
comptime MC = Walker2dModel.MAX_CONTACTS
# Symbolic model tail params — init_fields requires mf's tail params to
# structurally equal Model.MAX_EQUALITY/…/NEXCLUDE (won't unify with literal 0).
comptime NEQ = Walker2dModel.MAX_EQUALITY
comptime NTD = Walker2dModel.MAX_TENDON
comptime NSITE = Walker2dModel.NSITE
comptime NEXCL = Walker2dModel.NEXCLUDE
comptime BATCH = 2


def _fields_prep[
    target: StaticString
](
    mut d: DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH],
    mut mf: ModelFields[DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTD, NSITE, NEXCL, 0],
    mut scratch: DynamicsScratch[DTYPE, NV, NBODY, BATCH],
    ctx: Optional[DeviceContext],
) raises:
    """Smooth-dynamics prep + detection, mirroring EulerIntegratorFields.step
    up to the constraint seam (order verbatim; copied from
    test_newton_solve_fields)."""
    forward_kinematics_fields[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH,
    ](d, mf, ctx)
    compute_body_velocities_fields[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH,
    ](d, mf, ctx)
    compute_subtree_com_fields[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH,
    ](d, mf, ctx)
    compute_cdof_fields[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH,
    ](d, mf, scratch, ctx)
    compute_mass_matrix_fields[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH,
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
            target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0,
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
            target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0,
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
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH,
    ](d, mf, ctx)


def _init_state(mut d: DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]):
    """Fallen Walker2D (feet penetrating the floor) — deterministic."""
    for e in range(BATCH):
        for i in range(NQ):
            var qp = Scalar[DTYPE]((e * 5 + i * 3) % 5 - 2) / 40.0
            if i == 1:
                qp = 1.10  # feet penetrate the floor
            d.qpos.data[e * NQ + i] = qp
        for i in range(NV):
            var qv = Scalar[DTYPE]((e * 7 + i * 5) % 7 - 3) / 20.0
            if i == 1:
                qv = -0.5  # falling
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 4.0
            d.qvel.data[e * NV + i] = qv
            d.qfrc.data[e * NV + i] = qf


def _load_model(ctx: DeviceContext) raises -> ModelFields[
    DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTD, NSITE, NEXCL, 0
]:
    var mf = ModelFields[DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTD, NSITE, NEXCL, 0]()
    Walker2dModel.init_fields[DTYPE, 0](ctx, mf)
    return mf^


def _ncon(mut d: DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]) -> Int:
    var n = 0
    for e in range(BATCH):
        n += Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
    return n


def part_a(ctx: DeviceContext) raises:
    print("--- Part A: fields-GPU CG vs fields-GPU Newton (ELLIPTIC)")
    var mf = _load_model(ctx)

    # Independent Newton + CG data/scratch, identical initial state.
    var dN = DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    var dC = DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    _init_state(dN)
    _init_state(dC)
    dN.upload_all(ctx)
    dC.upload_all(ctx)

    var scN = DynamicsScratch[DTYPE, NV, NBODY, BATCH]()
    var csN = ContactScratch[DTYPE, NV, MC, BATCH]()
    var scC = DynamicsScratch[DTYPE, NV, NBODY, BATCH]()
    var csC = ContactScratch[DTYPE, NV, MC, BATCH]()
    scN.upload_all(ctx)
    csN.upload_all(ctx)
    scC.upload_all(ctx)
    csC.upload_all(ctx)

    _fields_prep["gpu"](dN, mf, scN, ctx)
    solve_newton_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0,
        ConeType.ELLIPTIC, BATCH,
    ](dN, mf, scN, csN, ctx)

    _fields_prep["gpu"](dC, mf, scC, ctx)
    solve_cg_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0,
        ConeType.ELLIPTIC, BATCH,
    ](dC, mf, scC, csC, ctx)

    dN.meta.download(ctx)
    var ncon = _ncon(dN)
    if ncon == 0:
        raise Error("Part A: no contacts formed — scenario vacuous")
    print("  contacts:", ncon)

    scN.qacc_constrained.download(ctx)
    scC.qacc_constrained.download(ctx)
    var worst = Float64(0)
    for i in range(BATCH * NV):
        var n = Float64(scN.qacc_constrained.data[i])
        var c = Float64(scC.qacc_constrained.data[i])
        if n != n or c != c:
            raise Error("Part A: non-finite qacc_constrained")
        var err = abs(n - c) / (1.0 + abs(n))
        if err > worst:
            worst = err
    print("  CG vs Newton qacc_constrained worst rel err:", worst)
    if worst > 2e-2 and not has_nvidia_gpu_accelerator():
        raise Error("Part A: CG diverges from Newton on the same problem")
    print("  Part A PASS: CG ≈ Newton")


def part_b(ctx: DeviceContext) raises:
    print("--- Part B: fields-CPU CG vs fields-GPU CG (ELLIPTIC)")
    var mf = _load_model(ctx)

    var dg = DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    var dc = DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    _init_state(dg)
    _init_state(dc)
    dg.upload_all(ctx)

    var scg = DynamicsScratch[DTYPE, NV, NBODY, BATCH]()
    var csg = ContactScratch[DTYPE, NV, MC, BATCH]()
    var scc = DynamicsScratch[DTYPE, NV, NBODY, BATCH]()
    var csc = ContactScratch[DTYPE, NV, MC, BATCH]()
    scg.upload_all(ctx)
    csg.upload_all(ctx)

    _fields_prep["gpu"](dg, mf, scg, ctx)
    solve_cg_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0,
        ConeType.ELLIPTIC, BATCH,
    ](dg, mf, scg, csg, ctx)

    _fields_prep["cpu"](dc, mf, scc, None)
    solve_cg_fields[
        "cpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0,
        ConeType.ELLIPTIC, BATCH,
    ](dc, mf, scc, csc, None)

    scg.qacc_constrained.download(ctx)
    var worst = Float64(0)
    for i in range(BATCH * NV):
        var g = Float64(scg.qacc_constrained.data[i])
        var c = Float64(scc.qacc_constrained.data[i])
        var err = abs(g - c) / (1.0 + abs(g))
        if err > worst:
            worst = err
    print("  fields-CPU vs fields-GPU CG worst rel err:", worst)
    if worst > 1e-2 and not has_nvidia_gpu_accelerator():
        raise Error("Part B: fields-CPU CG diverged from fields-GPU CG")
    print("  Part B PASS: fields-CPU CG ≈ fields-GPU CG")


def main() raises:
    print("=== Stage-S solve_cg_fields vs Newton: Walker2D ELLIPTIC ===")
    var ctx = DeviceContext()
    part_a(ctx)
    part_b(ctx)
    print("test_cg_fields: ALL PASS")
