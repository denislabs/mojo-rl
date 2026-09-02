"""_mt parity gate: cooperative within-env (PARALLEL=True) fields kernels
vs the serial fields kernels — same inputs, GPU vs GPU, BIT-EXACT.

A. Walker2D (NQ=9, NBODY=8), BATCH=2: per-op chained parity for every
   ported op — FK -> body velocities -> subtree_com (serial, shared) ->
   cdof -> CRBA -> LDL factor -> M^-1 -> RNE. Each side runs its own chain
   (serial vs PARALLEL dispatchers); every output tensor compared bit-exact
   after each op.
B. Humanoid (NBODY=14, real multi-level tree): FK + body-velocity parity
   (the level-parallel schedules with actual depth).
C. Integrator-level: RK4Integrator[..., PARALLEL_GPU=True] vs
   [..., PARALLEL_GPU=False] on Walker2D WITH CONTACTS, 3 full steps,
   qpos/qvel/qacc + contact records bit-exact; contacts asserted > 0.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_fields_mt_parity.mojo
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import Data, Model, DynamicsScratch, Dims, DimsLike
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
    compute_m_inv,
)
from mojo_rl.physics3d.dynamics.rne import (
    compute_bias_forces_rne,
)
from mojo_rl.physics3d.integrator.rk4 import RK4Integrator
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS,
    CONTACT_SIZE,
    METADATA_SIZE,
)
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel
from mojo_rl.physics3d.model.model_dims import ModelDims

comptime DTYPE = DType.float32

# ── Walker2D dims ──────────────────────────────────────────────────────────
comptime NQ = Walker2dModel.NQ  # 9
comptime NV = Walker2dModel.NV  # 9
comptime NBODY = Walker2dModel.NBODY  # 8
comptime NJOINT = Walker2dModel.NJOINT  # 9
comptime NGEOM = Walker2dModel.NGEOM  # 8
comptime MC = Walker2dModel.MAX_CONTACTS  # 20
comptime NEQ = Walker2dModel.MAX_EQUALITY
comptime NTD = Walker2dModel.MAX_TENDON
comptime NSITE = Walker2dModel.NSITE
comptime NEXCL = Walker2dModel.NEXCLUDE
comptime MD = ModelDims[Walker2dModel]
comptime CONE = Walker2dModel.CONE_TYPE
comptime BATCH = 2

# ── Humanoid dims (deep tree: NBODY=14) ────────────────────────────────────
comptime H_NQ = HumanoidModel.NQ  # 24
comptime H_NV = HumanoidModel.NV  # 23
comptime H_NBODY = HumanoidModel.NBODY  # 14
comptime H_NJOINT = HumanoidModel.NJOINT  # 18
comptime H_NGEOM = HumanoidModel.NGEOM  # 18
comptime H_MC = HumanoidModel.MAX_CONTACTS  # 50
comptime H_NEQ = HumanoidModel.MAX_EQUALITY  # 0
comptime H_NTEN = HumanoidModel.MAX_TENDON  # 2
comptime H_NSITE = HumanoidModel.NSITE  # 0
comptime H_NEXCL = HumanoidModel.nexclude
comptime MD_2 = Dims[
    nq=H_NQ,
    nv=H_NV,
    nbody=H_NBODY,
    njoint=H_NJOINT,
    ngeom=H_NGEOM,
    nsite=H_NSITE,
    max_contacts=H_MC,
    nequality=H_NEQ,
    ntendon=H_NTEN,
    nexclude=H_NEXCL,
    nmesh_verts=0,
    npair=HumanoidModel.NPAIR,
    nact=HumanoidModel.NACT,
    nten=HumanoidModel.NTEN_F,
    nkey=HumanoidModel.NKEY,
]
comptime H_BATCH = 2



def _cmp(
    name: String, a: TensorImpl[DTYPE], b: TensorImpl[DTYPE], n: Int
) raises:
    """Bit-exact comparison of the first n host elements (download first)."""
    var bad = 0
    for i in range(n):
        if a.data[i] != b.data[i]:
            if bad < 3:
                print(
                    "  DIFF ", name, " i=", i, ": ", a.data[i], " vs ",
                    b.data[i],
                )
            bad += 1
    if bad != 0:
        raise Error(name + ": not bit-exact")
    print("  PASS:", name, "bit-exact (serial vs PARALLEL)")


def test_walker2d_per_op() raises:
    print("--- A. Walker2D per-op _mt parity, BATCH=", BATCH, "---")
    var ctx = DeviceContext()

    var mf = Model[DTYPE, MD]()
    Walker2dModel.init_fields[DTYPE](ctx, mf)

    # Pseudo-random qpos/qvel (standard harness pattern; rootz lifted).
    var ds = Data[DTYPE, MD, BATCH]()
    var dp = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            var qp = Scalar[DTYPE]((e * 5 + i * 3) % 5 - 2) / 40.0
            if i == 1:
                qp = 1.25
            ds.qpos.data[e * NQ + i] = qp
            dp.qpos.data[e * NQ + i] = qp
        for i in range(NV):
            var qv = Scalar[DTYPE]((e * 17 + i * 13) % 9 - 4) / 4.0
            ds.qvel.data[e * NV + i] = qv
            dp.qvel.data[e * NV + i] = qv
    ds.upload_all(ctx)
    dp.upload_all(ctx)

    var ss = DynamicsScratch[DTYPE, MD, BATCH]()
    var sp = DynamicsScratch[DTYPE, MD, BATCH]()
    ss.upload_all(ctx)
    sp.upload_all(ctx)

    # 1. FK
    forward_kinematics["gpu", DTYPE, BATCH=BATCH](ds, mf, ctx)
    forward_kinematics["gpu", DTYPE, BATCH=BATCH, PARALLEL=True](dp, mf, ctx)
    ds.xpos.download(ctx)
    ds.xquat.download(ctx)
    ds.xipos.download(ctx)
    dp.xpos.download(ctx)
    dp.xquat.download(ctx)
    dp.xipos.download(ctx)
    _cmp("walker2d FK xpos", ds.xpos, dp.xpos, BATCH * NBODY * 3)
    _cmp("walker2d FK xquat", ds.xquat, dp.xquat, BATCH * NBODY * 4)
    _cmp("walker2d FK xipos", ds.xipos, dp.xipos, BATCH * NBODY * 3)

    # 2. Body velocities
    compute_body_velocities["gpu", DTYPE, BATCH=BATCH](ds, mf, ctx)
    compute_body_velocities["gpu", DTYPE, BATCH=BATCH, PARALLEL=True](dp, mf, ctx)
    ds.xvel.download(ctx)
    ds.xangvel.download(ctx)
    dp.xvel.download(ctx)
    dp.xangvel.download(ctx)
    _cmp("walker2d bodyvel xvel", ds.xvel, dp.xvel, BATCH * NBODY * 3)
    _cmp("walker2d bodyvel xangvel", ds.xangvel, dp.xangvel, BATCH * NBODY * 3)

    # 3. subtree_com (no _mt variant — serial on both chains, sanity check)
    compute_subtree_com["gpu", DTYPE, BATCH=BATCH](ds, mf, ctx)
    compute_subtree_com["gpu", DTYPE, BATCH=BATCH](dp, mf, ctx)
    ds.subtree_com.download(ctx)
    dp.subtree_com.download(ctx)
    _cmp(
        "walker2d subtree_com (serial both)",
        ds.subtree_com, dp.subtree_com, BATCH * NBODY * 3,
    )

    # 4. cdof
    compute_cdof["gpu", DTYPE, BATCH=BATCH](ds, mf, ss, ctx)
    compute_cdof["gpu", DTYPE, BATCH=BATCH, PARALLEL=True](dp, mf, sp, ctx)
    ss.cdof.download(ctx)
    sp.cdof.download(ctx)
    _cmp("walker2d cdof", ss.cdof, sp.cdof, BATCH * NV * 6)

    # 5. CRBA mass matrix
    compute_mass_matrix["gpu", DTYPE, BATCH=BATCH](ds, mf, ss, ctx)
    compute_mass_matrix["gpu", DTYPE, BATCH=BATCH, PARALLEL=True](dp, mf, sp, ctx)
    ss.M.download(ctx)
    sp.M.download(ctx)
    _cmp("walker2d CRBA M", ss.M, sp.M, BATCH * NV * NV)

    # 6. LDL factor
    ldl_factor["gpu", DTYPE, BATCH=BATCH](mf, ss, ctx)
    ldl_factor["gpu", DTYPE, BATCH=BATCH, PARALLEL=True](mf, sp, ctx)
    ss.L.download(ctx)
    ss.D.download(ctx)
    sp.L.download(ctx)
    sp.D.download(ctx)
    _cmp("walker2d LDL L", ss.L, sp.L, BATCH * NV * NV)
    _cmp("walker2d LDL D", ss.D, sp.D, BATCH * NV)

    # 7. M^-1 from LDL
    compute_m_inv["gpu", DTYPE, BATCH=BATCH](mf, ss, ctx)
    compute_m_inv["gpu", DTYPE, BATCH=BATCH, PARALLEL=True](mf, 
        sp, ctx
    )
    ss.m_inv.download(ctx)
    sp.m_inv.download(ctx)
    _cmp("walker2d M^-1", ss.m_inv, sp.m_inv, BATCH * NV * NV)

    # 8. RNE bias forces
    compute_bias_forces_rne["gpu", DTYPE, BATCH=BATCH](ds, mf, ss, ctx)
    compute_bias_forces_rne["gpu", DTYPE, BATCH=BATCH, PARALLEL=True](dp, mf, sp, ctx)
    ss.bias.download(ctx)
    sp.bias.download(ctx)
    _cmp("walker2d RNE bias", ss.bias, sp.bias, BATCH * NV)


def _humanoid_qpos(e: Int, i: Int) -> Scalar[DTYPE]:
    """Free joint pose + hinge angles (same shape as the equality/tendon
    fields gate harness): identity quaternion, feet near the floor, varied
    hinge angles per env."""
    if i == 0:
        return Scalar[DTYPE](0.02) * Scalar[DTYPE](e)
    if i == 1:
        return Scalar[DTYPE](0)
    if i == 2:
        return Scalar[DTYPE](1.24)
    if i == 3:
        return Scalar[DTYPE](1)  # identity quaternion (w first)
    if i <= 6:
        return Scalar[DTYPE](0)
    if i == 7:
        return Scalar[DTYPE](0.05) + Scalar[DTYPE](0.01) * Scalar[DTYPE](e)
    if i == 8:
        return Scalar[DTYPE](-0.1)
    if i == 9:
        return Scalar[DTYPE](0.05)
    if i == 10 or i == 14:
        return Scalar[DTYPE](-0.1)
    if i == 11 or i == 15:
        return Scalar[DTYPE](-0.1)
    if i == 12 or i == 16:
        return Scalar[DTYPE](-0.05)
    if i == 13 or i == 17:
        return Scalar[DTYPE](-0.15)
    if i == 20 or i == 23:
        return Scalar[DTYPE](-0.3)
    return Scalar[DTYPE](0.1) + Scalar[DTYPE](0.01) * Scalar[DTYPE](e)


def test_humanoid_fk_bodyvel() raises:
    print("--- B. Humanoid FK/bodyvel _mt parity (NBODY=", H_NBODY, ") ---")
    var ctx = DeviceContext()

    var mf = Model[DTYPE, MD_2]()
    HumanoidModel.init_fields[DTYPE](ctx, mf)

    var ds = Data[DTYPE, MD_2, H_BATCH]()
    var dp = Data[DTYPE, MD_2, H_BATCH]()
    for e in range(H_BATCH):
        for i in range(H_NQ):
            var qp = _humanoid_qpos(e, i)
            ds.qpos.data[e * H_NQ + i] = qp
            dp.qpos.data[e * H_NQ + i] = qp
        for i in range(H_NV):
            var qv = Scalar[DTYPE]((e * 17 + i * 13) % 9 - 4) / 4.0
            ds.qvel.data[e * H_NV + i] = qv
            dp.qvel.data[e * H_NV + i] = qv
    ds.upload_all(ctx)
    dp.upload_all(ctx)

    forward_kinematics["gpu", DTYPE, BATCH=H_BATCH](ds, mf, ctx)
    forward_kinematics["gpu", DTYPE, BATCH=H_BATCH, PARALLEL=True](dp, mf, ctx)
    ds.xpos.download(ctx)
    ds.xquat.download(ctx)
    ds.xipos.download(ctx)
    dp.xpos.download(ctx)
    dp.xquat.download(ctx)
    dp.xipos.download(ctx)
    _cmp("humanoid FK xpos", ds.xpos, dp.xpos, H_BATCH * H_NBODY * 3)
    _cmp("humanoid FK xquat", ds.xquat, dp.xquat, H_BATCH * H_NBODY * 4)
    _cmp("humanoid FK xipos", ds.xipos, dp.xipos, H_BATCH * H_NBODY * 3)

    compute_body_velocities["gpu", DTYPE, BATCH=H_BATCH](ds, mf, ctx)
    compute_body_velocities["gpu", DTYPE, BATCH=H_BATCH, PARALLEL=True](dp, mf, ctx)
    ds.xvel.download(ctx)
    ds.xangvel.download(ctx)
    dp.xvel.download(ctx)
    dp.xangvel.download(ctx)
    _cmp("humanoid bodyvel xvel", ds.xvel, dp.xvel, H_BATCH * H_NBODY * 3)
    _cmp(
        "humanoid bodyvel xangvel",
        ds.xangvel, dp.xangvel, H_BATCH * H_NBODY * 3,
    )


def test_rk4_integrator_parallel() raises:
    comptime N_STEPS = 3
    print("--- C. RK4Integrator PARALLEL_GPU parity WITH CONTACTS ---")
    var ctx = DeviceContext()

    var mf = Model[DTYPE, MD]()
    Walker2dModel.init_fields[DTYPE](ctx, mf)

    var ds = Data[DTYPE, MD, BATCH]()
    var dp = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            var qp = Scalar[DTYPE]((e * 5 + i * 3) % 5 - 2) / 40.0
            if i == 1:
                qp = 1.10  # feet penetrating -> contacts
            ds.qpos.data[e * NQ + i] = qp
            dp.qpos.data[e * NQ + i] = qp
        for i in range(NV):
            var qv = Scalar[DTYPE]((e * 7 + i * 5) % 7 - 3) / 20.0
            if i == 1:
                qv = -0.5
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 4.0
            ds.qvel.data[e * NV + i] = qv
            ds.qfrc.data[e * NV + i] = qf
            dp.qvel.data[e * NV + i] = qv
            dp.qfrc.data[e * NV + i] = qf
    ds.upload_all(ctx)
    dp.upload_all(ctx)

    var integ_s = RK4Integrator[DTYPE, MD, CONE, BATCH=BATCH]()
    integ_s.prepare_gpu(ctx)
    var integ_p = RK4Integrator[DTYPE, MD, CONE, BATCH=BATCH, PARALLEL_GPU=True]()
    integ_p.prepare_gpu(ctx)

    for step in range(N_STEPS):
        integ_s.step["gpu"](ds, mf, ctx)
        integ_p.step["gpu"](dp, mf, ctx)

        ds.qpos.download(ctx)
        ds.qvel.download(ctx)
        ds.qacc.download(ctx)
        ds.contacts.download(ctx)
        ds.meta.download(ctx)
        dp.qpos.download(ctx)
        dp.qvel.download(ctx)
        dp.qacc.download(ctx)
        dp.contacts.download(ctx)
        dp.meta.download(ctx)

        var bad = 0
        var ncon_seen = 0
        for e in range(BATCH):
            var nc_s = Int(
                ds.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
            )
            var nc_p = Int(
                dp.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
            )
            if nc_s != nc_p:
                print(
                    "  ncon mismatch env", e, ": serial", nc_s, " parallel",
                    nc_p,
                )
                bad += 1
                continue
            ncon_seen += nc_s
            for i in range(NQ):
                if ds.qpos.data[e * NQ + i] != dp.qpos.data[e * NQ + i]:
                    if bad < 4:
                        print(
                            "  qpos diff e", e, "i", i, ":",
                            ds.qpos.data[e * NQ + i], "vs",
                            dp.qpos.data[e * NQ + i],
                        )
                    bad += 1
            for i in range(NV):
                if ds.qvel.data[e * NV + i] != dp.qvel.data[e * NV + i]:
                    bad += 1
                if ds.qacc.data[e * NV + i] != dp.qacc.data[e * NV + i]:
                    bad += 1
            for c in range(nc_s):
                for k in range(CONTACT_SIZE):
                    var idx = e * MC * CONTACT_SIZE + c * CONTACT_SIZE + k
                    if ds.contacts.data[idx] != dp.contacts.data[idx]:
                        bad += 1
        if bad != 0:
            raise Error(
                "step " + String(step) + ": RK4 PARALLEL_GPU parity mismatch"
            )
        if ncon_seen == 0:
            raise Error("no contacts at step " + String(step) + " — vacuous")
        print(
            "  PASS: step", step,
            ": BIT-EXACT (qpos/qvel/qacc + contact records), contacts:",
            ncon_seen,
        )


def main() raises:
    test_walker2d_per_op()
    test_humanoid_fk_bodyvel()
    test_rk4_integrator_parallel()
    print("test_fields_mt_parity: ALL PASS")
