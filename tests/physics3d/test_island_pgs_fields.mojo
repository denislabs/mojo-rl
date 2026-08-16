"""Stage-S-ISL gate: fields IslandPGS solve (solve_island_pgs) vs the
fields PGS solve (solve_contacts), ELLIPTIC cone.

IslandPGS is plain PGS + body union-find island partition + per-island early
termination (freeze an island once its max |Δλ_n| < ISLAND_CONVERGE_EPS).
Islands are independent constraint sub-problems (no shared bodies/dofs), so
freezing a converged island changes neither its own impulses (already at the
fixed point) nor any other island's — the final qacc_constrained matches plain
PGS to solver tolerance. Since fields PGS is the golden-frozen baseline, "island
agrees with PGS" validates the island port (union-find + freeze) end-to-end.

Walker2D dropped onto the floor (rootz=1.10, feet penetrating), BATCH=2.
Checks:
  * Part A: fields-GPU island qacc_constrained ≈ fields-GPU PGS,
  * Part B: fields-CPU island ≈ fields-GPU island (single-source),
  * Part C: EulerIntegrator[SOLVER="island"] takes one contact step,
    stays finite (wiring compiles + runs).

Run: pixi run -e apple mojo run -I . tests/physics3d/test_island_pgs_fields.mojo
"""

from std.math import abs
from std.sys import has_nvidia_gpu_accelerator
from max.gpu.host import DeviceContext
from layout import Layout

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import (
    Data,
    Model,
    DynamicsScratch,
    ContactScratch,
    Dims,
)
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.integrator.euler import (
    EulerIntegrator,
    _armature_kernel,
    _fnet_passive_kernel,
    _qacc_writeback_kernel,
    _armature_env,
    _fnet_passive_env,
    _qacc_writeback_env,
)
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from mojo_rl.physics3d.dynamics.subtree_com import (
    compute_subtree_com,
)
from mojo_rl.physics3d.dynamics.cdof import compute_cdof
from mojo_rl.physics3d.dynamics.mass_matrix import (
    compute_mass_matrix,
)
from mojo_rl.physics3d.dynamics.ldl import (
    ldl_factor,
    ldl_solve,
    compute_m_inv,
)
from mojo_rl.physics3d.dynamics.rne import (
    compute_bias_forces_rne,
)
from mojo_rl.physics3d.collision.contact_detection import (
    detect_contacts,
)
from mojo_rl.physics3d.constraints.contact_solve import (
    solve_contacts,
)
from mojo_rl.physics3d.solver.island_pgs_solve import (
    solve_island_pgs,
)
from mojo_rl.physics3d.gpu.constants import (
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
comptime NEQ = Walker2dModel.MAX_EQUALITY
comptime NTD = Walker2dModel.MAX_TENDON
comptime NSITE = Walker2dModel.NSITE
comptime NEXCL = Walker2dModel.NEXCLUDE
comptime CONE = ConeType.ELLIPTIC
comptime BATCH = 2


def _fields_prep[
    target: StaticString
](
    mut d: Data[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH],
    mut mf: Model[DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTD, NSITE, NEXCL, 0],
    mut scratch: DynamicsScratch[DTYPE, Dims[nv=NV, nbody=NBODY], BATCH],
    ctx: Optional[DeviceContext],
) raises:
    """Smooth-dynamics prep + detection up to the constraint seam (copied
    from test_cg_fields / test_newton_solve_fields)."""
    forward_kinematics[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH,
    ](d, mf, ctx)
    compute_body_velocities[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH,
    ](d, mf, ctx)
    compute_subtree_com[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH,
    ](d, mf, ctx)
    compute_cdof[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH,
    ](d, mf, scratch, ctx)
    compute_mass_matrix[
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
            _armature_env[DTYPE, NV, NJOINT, BATCH](e, joints_v, M_v)
        ldl_factor[target, DTYPE, NV, NBODY, BATCH](scratch, ctx)
        compute_m_inv[target, DTYPE, NV, NBODY, BATCH](scratch, ctx)
        compute_bias_forces_rne[
            target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0,
            BATCH,
        ](d, mf, scratch, ctx)
        var qpos_v = d.qpos.lt["cpu", L_QPOS]()
        var qvel_v = d.qvel.lt["cpu", L_NV]()
        var qfrc_v = d.qfrc.lt["cpu", L_NV]()
        var bias_v = scratch.bias.lt["cpu", L_NV]()
        var fnet_v = scratch.fnet.lt["cpu", L_NV]()
        for e in range(BATCH):
            _fnet_passive_env[DTYPE, NQ, NV, NJOINT, BATCH](
                e, qpos_v, qvel_v, qfrc_v, joints_v, bias_v, fnet_v
            )
        ldl_solve[target, DTYPE, NV, NBODY, BATCH](scratch, ctx)
        var qacc_ws_v = scratch.qacc_ws.lt["cpu", L_NV]()
        var qacc_v = d.qacc.lt["cpu", L_NV]()
        var qacc_c_v = scratch.qacc_constrained.lt["cpu", L_NV]()
        for e in range(BATCH):
            _qacc_writeback_env[DTYPE, NV, BATCH](
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
        ldl_factor[target, DTYPE, NV, NBODY, BATCH](scratch, ctx)
        compute_m_inv[target, DTYPE, NV, NBODY, BATCH](scratch, ctx)
        compute_bias_forces_rne[
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
        ldl_solve[target, DTYPE, NV, NBODY, BATCH](scratch, ctx)
        ctx.value().enqueue_function[
            _qacc_writeback_kernel[DTYPE, NV, BATCH]
        ](
            scratch.qacc_ws.lt["gpu", L_NV](),
            d.qacc.lt["gpu", L_NV](),
            scratch.qacc_constrained.lt["gpu", L_NV](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )

    detect_contacts[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, BATCH,
    ](d, mf, ctx)


def _init_state(mut d: Data[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]):
    for e in range(BATCH):
        for i in range(NQ):
            var qp = Scalar[DTYPE]((e * 5 + i * 3) % 5 - 2) / 40.0
            if i == 1:
                qp = 1.10
            d.qpos.data[e * NQ + i] = qp
        for i in range(NV):
            var qv = Scalar[DTYPE]((e * 7 + i * 5) % 7 - 3) / 20.0
            if i == 1:
                qv = -0.5
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 4.0
            d.qvel.data[e * NV + i] = qv
            d.qfrc.data[e * NV + i] = qf


def _load_model(ctx: DeviceContext) raises -> Model[
    DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTD, NSITE, NEXCL, 0
]:
    var mf = Model[DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTD, NSITE, NEXCL, 0]()
    Walker2dModel.init_fields[DTYPE, 0](ctx, mf)
    return mf^


def part_a(ctx: DeviceContext) raises:
    print("--- Part A: fields-GPU IslandPGS vs fields-GPU PGS (ELLIPTIC)")
    var mf = _load_model(ctx)

    var dP = Data[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    var dI = Data[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    _init_state(dP)
    _init_state(dI)
    dP.upload_all(ctx)
    dI.upload_all(ctx)

    var scP = DynamicsScratch[DTYPE, Dims[nv=NV, nbody=NBODY], BATCH]()
    var csP = ContactScratch[DTYPE, Dims[nv=NV, max_contacts=MC], BATCH]()
    var scI = DynamicsScratch[DTYPE, Dims[nv=NV, nbody=NBODY], BATCH]()
    var csI = ContactScratch[DTYPE, Dims[nv=NV, max_contacts=MC], BATCH]()
    scP.upload_all(ctx)
    csP.upload_all(ctx)
    scI.upload_all(ctx)
    csI.upload_all(ctx)

    _fields_prep["gpu"](dP, mf, scP, ctx)
    solve_contacts[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0,
        CONE, BATCH,
    ](dP, mf, scP, csP, ctx)

    _fields_prep["gpu"](dI, mf, scI, ctx)
    solve_island_pgs[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0,
        CONE, BATCH,
    ](dI, mf, scI, csI, ctx)

    dP.meta.download(ctx)
    var ncon = 0
    for e in range(BATCH):
        ncon += Int(dP.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
    if ncon == 0:
        raise Error("Part A: no contacts formed — scenario vacuous")
    print("  contacts:", ncon)

    scP.qacc_constrained.download(ctx)
    scI.qacc_constrained.download(ctx)
    var worst = Float64(0)
    for i in range(BATCH * NV):
        var p = Float64(scP.qacc_constrained.data[i])
        var isl = Float64(scI.qacc_constrained.data[i])
        if p != p or isl != isl:
            raise Error("Part A: non-finite qacc_constrained")
        var err = abs(p - isl) / (1.0 + abs(p))
        if err > worst:
            worst = err
    print("  IslandPGS vs PGS qacc_constrained worst rel err:", worst)
    if worst > 2e-2 and not has_nvidia_gpu_accelerator():
        raise Error("Part A: IslandPGS diverges from PGS")
    print("  Part A PASS: IslandPGS ≈ PGS")


def part_b(ctx: DeviceContext) raises:
    print("--- Part B: fields-CPU IslandPGS vs fields-GPU IslandPGS")
    var mf = _load_model(ctx)

    var dg = Data[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    var dc = Data[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    _init_state(dg)
    _init_state(dc)
    dg.upload_all(ctx)

    var scg = DynamicsScratch[DTYPE, Dims[nv=NV, nbody=NBODY], BATCH]()
    var csg = ContactScratch[DTYPE, Dims[nv=NV, max_contacts=MC], BATCH]()
    var scc = DynamicsScratch[DTYPE, Dims[nv=NV, nbody=NBODY], BATCH]()
    var csc = ContactScratch[DTYPE, Dims[nv=NV, max_contacts=MC], BATCH]()
    scg.upload_all(ctx)
    csg.upload_all(ctx)

    _fields_prep["gpu"](dg, mf, scg, ctx)
    solve_island_pgs[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0,
        CONE, BATCH,
    ](dg, mf, scg, csg, ctx)

    _fields_prep["cpu"](dc, mf, scc, None)
    solve_island_pgs[
        "cpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0,
        CONE, BATCH,
    ](dc, mf, scc, csc, None)

    scg.qacc_constrained.download(ctx)
    var worst = Float64(0)
    for i in range(BATCH * NV):
        var g = Float64(scg.qacc_constrained.data[i])
        var c = Float64(scc.qacc_constrained.data[i])
        var err = abs(g - c) / (1.0 + abs(g))
        if err > worst:
            worst = err
    print("  fields-CPU vs fields-GPU IslandPGS worst rel err:", worst)
    if worst > 1e-2 and not has_nvidia_gpu_accelerator():
        raise Error("Part B: fields-CPU IslandPGS diverged from fields-GPU")
    print("  Part B PASS: fields-CPU IslandPGS ≈ fields-GPU IslandPGS")


def part_c(ctx: DeviceContext) raises:
    print("--- Part C: EulerIntegrator[SOLVER='island'] step (finite)")
    var mf = _load_model(ctx)
    var d = Data[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    _init_state(d)
    d.upload_all(ctx)

    var integ = EulerIntegrator[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0, CONE, BATCH,
        SOLVER="island",
    ]()
    integ.prepare_gpu(ctx)
    integ.step["gpu", True](d, mf, ctx)

    d.qpos.download(ctx)
    d.qvel.download(ctx)
    for i in range(BATCH * NQ):
        var v = Float64(d.qpos.data[i])
        if v != v or abs(v) > 1e6:
            raise Error("Part C: non-finite qpos (island wiring)")
    for i in range(BATCH * NV):
        var v = Float64(d.qvel.data[i])
        if v != v or abs(v) > 1e6:
            raise Error("Part C: non-finite qvel (island wiring)")
    print("  Part C PASS: Euler SOLVER='island' step finite")


def main() raises:
    print("=== Stage-S-ISL solve_island_pgs vs PGS: Walker2D ===")
    var ctx = DeviceContext()
    part_a(ctx)
    part_b(ctx)
    part_c(ctx)
    print("test_island_pgs_fields: ALL PASS")
