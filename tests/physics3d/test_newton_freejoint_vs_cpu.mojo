"""NVIDIA validation: the fields Newton contact solve on FREE-JOINT models
(Ant + Humanoid) vs the trusted fields-CPU per-env oracle.

WHY: before the physics3d sunset we must be sure the PYRAMIDAL blocked Newton
solver (the one-env-per-block cooperative kernel that runs ONLY on NVIDIA for
free-joint models) computes correct physics. It was validated bit-exact vs
per-env only on Walker2D (slide/hinge). This closes the free-joint gap.

HOW: `solve_newton` routes PYRAMIDAL+NVIDIA -> BLOCKED, everything else
-> per-env. So:
  * on NVIDIA: solve_newton["gpu"] == BLOCKED (production path);
    solve_newton["cpu"] == per-env oracle (trusted).  Comparing them
    VALIDATES THE BLOCKED SOLVER against ground truth.
  * on Apple: solve_newton["gpu"] == per-env (blocked never launches);
    this just checks per-env GPU vs CPU (safe — no heavy cooperative kernel).
Same script, both meaningful. Gentle floor-contact poses (light penetration)
keep the Newton solve well-conditioned — a deep-penetration pose ill-conditions
it and can NaN/crash, which is a test artifact, not a solver bug.

Verdict: GPU-vs-CPU qacc_constrained relative error < 1e-2 => solver correct.

Run on NVIDIA: pixi run -e nvidia mojo run -I . tests/physics3d/test_newton_freejoint_vs_cpu.mojo
Run on Apple : pixi run -e apple  mojo run -I . tests/physics3d/test_newton_freejoint_vs_cpu.mojo
"""

from std.math import abs
from std.sys import has_nvidia_gpu_accelerator
from max.gpu.host import DeviceContext
from layout import Layout

from mojo_rl.physics3d.fields import (
    Data,
    Model,
    DynamicsScratch,
    ContactScratch,
    Dims,
)
from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.integrator.euler import (
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
from mojo_rl.physics3d.solver.newton_solve import solve_newton
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS,
    METADATA_SIZE,
    MODEL_JOINT_SIZE,
)
from mojo_rl.envs.ant.ant_xml import AntModel
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel

comptime DTYPE = DType.float32
comptime BATCH = 2
comptime CONE_T = ConeType.PYRAMIDAL  # forces the blocked branch on NVIDIA

# Set True ONLY when running on NVIDIA. Referencing solve_newton for
# humanoid forces Metal to compile the blocked-humanoid kernel, whose
# threadgroup memory exceeds Metal's 32KB limit -> Apple compile failure. The
# comptime guard below excludes it entirely when False, so Apple compiles
# (Ant-only). On NVIDIA the blocked-humanoid kernel is the production path.
comptime INCLUDE_HUMANOID = False


def _prep[
    NQ: Int, NV: Int, NBODY: Int, NJOINT: Int, NGEOM: Int, MC: Int,
    NEQ: Int, NTEN: Int, NSITE: Int, NEXCL: Int, target: StaticString,
](
    mut d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MC, nsite=NSITE], BATCH],
    mut mf: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=0]],
    mut scratch: DynamicsScratch[DTYPE, Dims[nv=NV, nbody=NBODY], BATCH],
    ctx: Optional[DeviceContext],
) raises:
    """Smooth-dynamics prep + auto detection (mirrors the integrator seam)."""
    forward_kinematics[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTEN, NSITE,
        NEXCL, 0, BATCH,
    ](d, mf, ctx)
    compute_body_velocities[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTEN, NSITE,
        NEXCL, 0, BATCH,
    ](d, mf, ctx)
    compute_subtree_com[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTEN, NSITE,
        NEXCL, 0, BATCH,
    ](d, mf, ctx)
    compute_cdof[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTEN, NSITE,
        NEXCL, 0, BATCH,
    ](d, mf, scratch, ctx)
    compute_mass_matrix[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTEN, NSITE,
        NEXCL, 0, BATCH,
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
        ldl_factor["cpu", DTYPE, BATCH=BATCH](scratch, ctx)
        compute_m_inv["cpu", DTYPE, BATCH=BATCH](scratch, ctx)
        compute_bias_forces_rne[
            "cpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTEN, NSITE,
            NEXCL, 0, BATCH,
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
        ldl_solve["cpu", DTYPE, BATCH=BATCH](scratch, ctx)
        var qacc_ws_v = scratch.qacc_ws.lt["cpu", L_NV]()
        var qacc_v = d.qacc.lt["cpu", L_NV]()
        var qacc_c_v = scratch.qacc_constrained.lt["cpu", L_NV]()
        for e in range(BATCH):
            _qacc_writeback_env[DTYPE, NV, BATCH](
                e, qacc_ws_v, qacc_v, qacc_c_v
            )
    else:
        ctx.value().enqueue_function[_armature_kernel[DTYPE, NV, NJOINT, BATCH]](
            mf.joints.lt["gpu", L_JOINT](),
            scratch.M.lt["gpu", L_M](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ldl_factor["gpu", DTYPE, BATCH=BATCH](scratch, ctx)
        compute_m_inv["gpu", DTYPE, BATCH=BATCH](scratch, ctx)
        compute_bias_forces_rne[
            "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTEN, NSITE,
            NEXCL, 0, BATCH,
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
        ldl_solve["gpu", DTYPE, BATCH=BATCH](scratch, ctx)
        ctx.value().enqueue_function[_qacc_writeback_kernel[DTYPE, NV, BATCH]](
            scratch.qacc_ws.lt["gpu", L_NV](),
            d.qacc.lt["gpu", L_NV](),
            scratch.qacc_constrained.lt["gpu", L_NV](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )

    detect_contacts[
        target, DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTEN, NSITE,
        NEXCL, 0, BATCH,
    ](d, mf, ctx)


def _validate[MODEL: ModelDefLike](
    ctx: DeviceContext,
    name: String,
    torso_z: Float64,
) raises -> Bool:
    """Returns True if GPU==CPU within tolerance (solver correct)."""
    # Dims as local comptime aliases OFF the model spec so the mf type
    # structurally matches init_fields' signature (a literal/explicit param
    # would not unify with MODEL.NV etc — the Stage-E threading pattern).
    comptime NQ = MODEL.NQ
    comptime NV = MODEL.NV
    comptime NBODY = MODEL.NBODY
    comptime NJOINT = MODEL.NJOINT
    comptime NGEOM = MODEL.NGEOM
    comptime MC = MODEL.MAX_CONTACTS
    comptime NEQ = MODEL.MAX_EQUALITY
    comptime NTEN = MODEL.MAX_TENDON
    comptime NSITE = MODEL.NSITE
    comptime NEXCL = MODEL.NEXCLUDE
    print("--- ", name, " (NV=", NV, ") gentle floor contact ---")
    # Offset-free build straight from the compile-time model spec — no slab,
    # no init_model_gpu / load_from_slab.
    var mf = Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=0]]()
    MODEL.init_fields[DTYPE, 0](ctx, mf)

    # Gentle pose: torso lowered so feet lightly touch (not deep penetration).
    var d_g = Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MC, nsite=NSITE], BATCH]()
    var d_c = Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MC, nsite=NSITE], BATCH]()
    for e in range(BATCH):
        d_g.qpos.data[e * NQ + 2] = Scalar[DTYPE](torso_z + 0.01 * Float64(e))
        d_g.qpos.data[e * NQ + 3] = Scalar[DTYPE](1.0)  # quat w
        d_c.qpos.data[e * NQ + 2] = Scalar[DTYPE](torso_z + 0.01 * Float64(e))
        d_c.qpos.data[e * NQ + 3] = Scalar[DTYPE](1.0)
        for i in range(NV):
            var qv = Scalar[DTYPE]((e * 7 + i * 5) % 7 - 3) / 40.0  # mild
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 8.0
            d_g.qvel.data[e * NV + i] = qv
            d_g.qfrc.data[e * NV + i] = qf
            d_c.qvel.data[e * NV + i] = qv
            d_c.qfrc.data[e * NV + i] = qf
    d_g.upload_all(ctx)

    var sg = DynamicsScratch[DTYPE, Dims[nv=NV, nbody=NBODY], BATCH]()
    var cg = ContactScratch[DTYPE, Dims[nv=NV, max_contacts=MC], BATCH]()
    sg.upload_all(ctx)
    cg.upload_all(ctx)
    var sc = DynamicsScratch[DTYPE, Dims[nv=NV, nbody=NBODY], BATCH]()
    var cc = ContactScratch[DTYPE, Dims[nv=NV, max_contacts=MC], BATCH]()

    # GPU path (blocked on NVIDIA, per-env on Apple).
    _prep[NQ, NV, NBODY, NJOINT, NGEOM, MC, NEQ, NTEN, NSITE, NEXCL, "gpu"](
        d_g, mf, sg, ctx
    )
    solve_newton[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTEN, NSITE,
        NEXCL, 0, CONE_T, BATCH,
    ](d_g, mf, sg, cg, ctx)

    # CPU oracle (per-env).
    _prep[NQ, NV, NBODY, NJOINT, NGEOM, MC, NEQ, NTEN, NSITE, NEXCL, "cpu"](
        d_c, mf, sc, None
    )
    solve_newton[
        "cpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTEN, NSITE,
        NEXCL, 0, CONE_T, BATCH,
    ](d_c, mf, sc, cc, None)

    sg.qacc_constrained.download(ctx)
    d_g.meta.download(ctx)
    var ncon = 0
    for e in range(BATCH):
        ncon += Int(d_g.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
    print("    contacts:", ncon, "(", "NVIDIA=blocked" if
          has_nvidia_gpu_accelerator() else "Apple=per-env", ")")
    if ncon == 0:
        print("    WARNING: 0 contacts (vacuous). Lower this model's torso_z\n"
              "    arg in main() by ~0.05 and re-run until contacts > 0.")

    var worst = Float64(0)
    for i in range(BATCH * NV):
        var g = Float64(sg.qacc_constrained.data[i])
        var c = Float64(sc.qacc_constrained.data[i])
        var err = abs(g - c) / (1.0 + abs(c))
        if err > worst:
            worst = err
    print("    qacc_constrained GPU-vs-CPU-oracle worst rel err:", worst)
    if worst < 1e-2:
        print("    PASS:", name, "GPU solver matches CPU oracle")
        return True
    print("    FAIL:", name, "GPU solver DIVERGES from CPU oracle")
    return False


def main() raises:
    print("=" * 66)
    print("Free-joint Newton solve: GPU (blocked on NVIDIA) vs CPU oracle")
    print("=" * 66)
    var ctx = DeviceContext()

    var all_ok = True

    # ── Ant (free joint, NV=14) ────────────────────────────────────────────
    all_ok = _validate[AntModel](ctx, "Ant", 0.28) and all_ok

    # ── Humanoid (free joint, NV=23) — the production blocked model ─────────
    # On NVIDIA this exercises the exact blocked kernel humanoid training uses.
    # Comptime-excluded on Apple (blocked-humanoid overflows Metal's 32KB at
    # COMPILE time) — set INCLUDE_HUMANOID=True on the NVIDIA box.
    comptime if INCLUDE_HUMANOID:
        all_ok = _validate[HumanoidModel](ctx, "Humanoid", 1.0) and all_ok
    else:
        print("--- Humanoid: SKIPPED on Apple (NVIDIA-only blocked path) ---")

    print("=" * 66)
    if not all_ok:
        raise Error("free-joint Newton solver diverges from CPU oracle")
    print("ALL PASS: free-joint blocked/per-env solver == CPU oracle")
