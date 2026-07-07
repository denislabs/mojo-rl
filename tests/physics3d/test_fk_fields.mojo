"""P2 gate: per-field single-source FK vs the legacy flat-slab FK.

A. Walker2D (NQ=9, NBODY=8, NSITE=0), BATCH=3 with a DIFFERENT qpos config
   per env (exercises batch striding):
   1. fields-GPU vs legacy-GPU: BIT-EXACT (same arithmetic, same target).
   2. fields-CPU (same formula body via .lt["cpu"] views) vs fields-GPU and
      vs the legacy CPU forward_kinematics(Model, Data): 1e-4 tolerances
      (the existing cpu_vs_gpu budget).
B. Synthetic 2-body hinge model with NSITE=2 (covers the sites kernel
   variant + site_xpos), BATCH=2: fields-GPU vs legacy-GPU BIT-EXACT
   including site_xpos; fields-CPU vs fields-GPU tolerance.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_fk_fields.mojo
"""

from std.math import abs
from std.gpu import block_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.joint_types import JNT_HINGE
from mojo_rl.physics3d.fields import DataFields, ModelFields
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    forward_kinematics_gpu,
)
from mojo_rl.physics3d.kinematics.forward_kinematics_fields import (
    forward_kinematics_fields,
)
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    qpos_offset,
    xpos_offset,
    xquat_offset,
    xipos_offset,
    site_xpos_offset,
    model_body_offset,
    model_joint_offset,
    model_metadata_offset,
    model_site_offset,
    MODEL_META_IDX_NBODY,
    MODEL_META_IDX_NJOINT,
    BODY_IDX_PARENT,
    BODY_IDX_POS_X,
    BODY_IDX_POS_Y,
    BODY_IDX_POS_Z,
    BODY_IDX_QUAT_W,
    BODY_IDX_IPOS_Z,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_POS_Z,
    JOINT_IDX_AXIS_Y,
    SITE_IDX_BODY,
    SITE_IDX_POS_X,
    SITE_IDX_POS_Z,
)
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel

comptime DTYPE = DType.float32
comptime POS_TOL: Float64 = 1e-4
comptime QUAT_TOL: Float64 = 1e-4

# ── Walker2D dims ──────────────────────────────────────────────────────────
comptime NQ = Walker2dModel.NQ  # 9
comptime NV = Walker2dModel.NV  # 9
comptime NBODY = Walker2dModel.NBODY  # 8
comptime NJOINT = Walker2dModel.NJOINT  # 9
comptime NGEOM = Walker2dModel.NGEOM  # 8
comptime MAX_CONTACTS = Walker2dModel.MAX_CONTACTS  # 20
comptime BATCH = 3
comptime SS = state_size[NQ, NV, NBODY, MAX_CONTACTS, 0]()
comptime MS = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()

# ── Synthetic sites model dims ─────────────────────────────────────────────
comptime S_NQ = 1
comptime S_NV = 1
comptime S_NBODY = 2
comptime S_NJOINT = 1
comptime S_NSITE = 2
comptime S_MC = 1
comptime S_BATCH = 2
comptime S_SS = state_size[S_NQ, S_NV, S_NBODY, S_MC, S_NSITE]()
comptime S_MS = model_size_with_invweight[
    S_NBODY, S_NJOINT, S_NV, 0, 0, 0, S_NSITE
]()


def _legacy_fk_kernel[
    NQ_: Int,
    NV_: Int,
    NBODY_: Int,
    NJOINT_: Int,
    MC_: Int,
    SS_: Int,
    MS_: Int,
    B_: Int,
    NSITE_: Int,
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS_), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS_), MutAnyOrigin],
):
    var env = Int(block_idx.x)
    if env >= B_:
        return
    forward_kinematics_gpu[
        DTYPE, NQ_, NV_, NBODY_, NJOINT_, MC_, SS_, MS_, B_, 0, 0, 0, NSITE_
    ](env, state, model)


def _quat_err(
    ax: Float64, ay: Float64, az: Float64, aw: Float64,
    bx: Float64, by: Float64, bz: Float64, bw: Float64,
) -> Float64:
    var dp = abs(ax - bx) + abs(ay - by) + abs(az - bz) + abs(aw - bw)
    var dn = abs(ax + bx) + abs(ay + by) + abs(az + bz) + abs(aw + bw)
    return dp if dp < dn else dn


def test_walker2d() raises:
    print("--- A. Walker2D fields FK, BATCH=", BATCH, "---")
    var ctx = DeviceContext()

    # Model: init on device via the existing flattening, bridge to fields.
    var model_t = TensorImpl[DTYPE].alloc(MS)
    model_t.upload(ctx)
    var mbuf = model_t.dev.value()
    Walker2dModel.init_model_gpu(ctx, mbuf)
    model_t.download(ctx)
    var mf = ModelFields[DTYPE, NV, NBODY, NJOINT, NGEOM]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)

    # Three distinct qpos configs (from the legacy walker2d FK gate).
    var qcfg = List[List[Float64]]()
    var q1 = List[Float64](length=NQ, fill=0.0)
    q1[1] = 1.25
    qcfg.append(q1^)
    var q2 = List[Float64](length=NQ, fill=0.0)
    q2[1] = 1.25
    q2[3] = 0.5
    q2[4] = -0.8
    q2[5] = 0.3
    qcfg.append(q2^)
    var q3 = List[Float64](length=NQ, fill=0.0)
    q3[1] = 1.25
    q3[2] = 0.5
    q3[3] = 1.0
    q3[4] = -1.2
    q3[5] = 0.6
    q3[6] = -1.0
    q3[7] = 1.2
    q3[8] = -0.6
    qcfg.append(q3^)

    # Legacy GPU path (flat slab).
    var slab_t = TensorImpl[DTYPE].alloc(BATCH * SS)
    comptime O_QPOS = qpos_offset[NQ, NV]()
    for e in range(BATCH):
        for i in range(NQ):
            slab_t.data[e * SS + O_QPOS + i] = Scalar[DTYPE](qcfg[e][i])
    slab_t.upload(ctx)
    ctx.enqueue_function[
        _legacy_fk_kernel[NQ, NV, NBODY, NJOINT, MAX_CONTACTS, SS, MS, BATCH, 0]
    ](
        slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
        model_t.lt["gpu", Layout.row_major(1, MS)](),
        grid_dim=(BATCH,),
        block_dim=(1,),
    )
    slab_t.download(ctx)

    # Fields GPU path.
    var d = DataFields[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, 0, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            d.qpos.data[e * NQ + i] = Scalar[DTYPE](qcfg[e][i])
    d.upload_all(ctx)
    forward_kinematics_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        0, 0, 0, 0, 0, BATCH,
    ](d, mf, ctx)
    d.download_all(ctx)

    # 1. fields-GPU vs legacy-GPU: bit-exact.
    comptime O_XPOS = xpos_offset[NQ, NV, NBODY]()
    comptime O_XQUAT = xquat_offset[NQ, NV, NBODY]()
    comptime O_XIPOS = xipos_offset[NQ, NV, NBODY]()
    var bad = 0
    for e in range(BATCH):
        for j in range(NBODY * 3):
            if d.xpos.data[e * NBODY * 3 + j] != slab_t.data[e * SS + O_XPOS + j]:
                bad += 1
            if (
                d.xipos.data[e * NBODY * 3 + j]
                != slab_t.data[e * SS + O_XIPOS + j]
            ):
                bad += 1
        for j in range(NBODY * 4):
            if (
                d.xquat.data[e * NBODY * 4 + j]
                != slab_t.data[e * SS + O_XQUAT + j]
            ):
                bad += 1
    if bad != 0:
        raise Error("walker2d fields-GPU vs legacy-GPU: not bit-exact")
    print("  PASS: fields-GPU == legacy-GPU bit-exact (xpos/xquat/xipos)")

    # 2. fields-CPU (same body) vs fields-GPU + legacy-CPU.
    var dc = DataFields[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, 0, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            dc.qpos.data[e * NQ + i] = Scalar[DTYPE](qcfg[e][i])
    forward_kinematics_fields[
        "cpu", DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        0, 0, 0, 0, 0, BATCH,
    ](dc, mf)

    var worst_gpu = Float64(0)
    var worst_cpu = Float64(0)
    for e in range(BATCH):
        # legacy CPU reference for this env's config
        var model_cpu = Model[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
            Walker2dModel.MAX_EQUALITY, Walker2dModel.CONE_TYPE,
            Walker2dModel.MAX_TENDON, Walker2dModel.NSITE,
        ]()
        var data_cpu = Data[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, Walker2dModel.NSITE
        ]()
        Walker2dModel.setup_model_and_data[DTYPE](model_cpu, data_cpu)
        for i in range(NQ):
            data_cpu.qpos[i] = Scalar[DTYPE](qcfg[e][i])
        forward_kinematics(model_cpu, data_cpu)

        for b in range(NBODY):
            for k in range(3):
                var fc = Float64(dc.xpos.data[e * NBODY * 3 + b * 3 + k])
                var fg = Float64(d.xpos.data[e * NBODY * 3 + b * 3 + k])
                var lc = Float64(data_cpu.xpos[b * 3 + k])
                var e1 = abs(fc - fg)
                var e2 = abs(fc - lc)
                if e1 > worst_gpu:
                    worst_gpu = e1
                if e2 > worst_cpu:
                    worst_cpu = e2
            var qe_g = _quat_err(
                Float64(dc.xquat.data[e * NBODY * 4 + b * 4 + 0]),
                Float64(dc.xquat.data[e * NBODY * 4 + b * 4 + 1]),
                Float64(dc.xquat.data[e * NBODY * 4 + b * 4 + 2]),
                Float64(dc.xquat.data[e * NBODY * 4 + b * 4 + 3]),
                Float64(d.xquat.data[e * NBODY * 4 + b * 4 + 0]),
                Float64(d.xquat.data[e * NBODY * 4 + b * 4 + 1]),
                Float64(d.xquat.data[e * NBODY * 4 + b * 4 + 2]),
                Float64(d.xquat.data[e * NBODY * 4 + b * 4 + 3]),
            )
            var qe_c = _quat_err(
                Float64(dc.xquat.data[e * NBODY * 4 + b * 4 + 0]),
                Float64(dc.xquat.data[e * NBODY * 4 + b * 4 + 1]),
                Float64(dc.xquat.data[e * NBODY * 4 + b * 4 + 2]),
                Float64(dc.xquat.data[e * NBODY * 4 + b * 4 + 3]),
                Float64(data_cpu.xquat[b * 4 + 0]),
                Float64(data_cpu.xquat[b * 4 + 1]),
                Float64(data_cpu.xquat[b * 4 + 2]),
                Float64(data_cpu.xquat[b * 4 + 3]),
            )
            if qe_g > worst_gpu:
                worst_gpu = qe_g
            if qe_c > worst_cpu:
                worst_cpu = qe_c
    print(
        "  fields-CPU worst err: vs fields-GPU=", worst_gpu,
        " vs legacy-CPU=", worst_cpu,
    )
    if worst_gpu > QUAT_TOL or worst_cpu > QUAT_TOL:
        raise Error("walker2d fields-CPU tolerance exceeded")
    print("  PASS: fields-CPU within 1e-4 of fields-GPU and legacy-CPU")


def test_synthetic_sites() raises:
    print("--- B. Synthetic hinge model with NSITE=2, BATCH=", S_BATCH, "---")
    var ctx = DeviceContext()

    # Hand-built flat model slab: worldbody + 1 hinge body, 2 sites.
    var flat = List[Scalar[DTYPE]](length=S_MS, fill=Scalar[DTYPE](0))
    # body 0 (world): identity quat
    flat[model_body_offset(0) + BODY_IDX_QUAT_W] = 1.0
    # body 1: parent=0, pos=(0.1, 0.2, 1.0), identity quat, ipos=(0,0,-0.25)
    var b1 = model_body_offset(1)
    flat[b1 + BODY_IDX_PARENT] = 0.0
    flat[b1 + BODY_IDX_POS_X] = 0.1
    flat[b1 + BODY_IDX_POS_Y] = 0.2
    flat[b1 + BODY_IDX_POS_Z] = 1.0
    flat[b1 + BODY_IDX_QUAT_W] = 1.0
    flat[b1 + BODY_IDX_IPOS_Z] = -0.25
    # joint 0: hinge on body 1, anchor (0,0,0.5), axis (0,1,0), qpos_adr 0
    var j0 = model_joint_offset[S_NBODY](0)
    flat[j0 + JOINT_IDX_TYPE] = Scalar[DTYPE](JNT_HINGE)
    flat[j0 + JOINT_IDX_BODY_ID] = 1.0
    flat[j0 + JOINT_IDX_QPOS_ADR] = 0.0
    flat[j0 + JOINT_IDX_POS_Z] = 0.5
    flat[j0 + JOINT_IDX_AXIS_Y] = 1.0
    # metadata: nbody / njoint (legacy GPU FK reads njoint at runtime)
    var mo = model_metadata_offset[S_NBODY, S_NJOINT]()
    flat[mo + MODEL_META_IDX_NBODY] = Scalar[DTYPE](S_NBODY)
    flat[mo + MODEL_META_IDX_NJOINT] = Scalar[DTYPE](S_NJOINT)
    # sites: site0 on body1 at (0,0,-0.5); site1 on body0 at (1,0,0)
    var s0 = model_site_offset[S_NBODY, S_NJOINT, 0, 0, 0](0)
    flat[s0 + SITE_IDX_BODY] = 1.0
    flat[s0 + SITE_IDX_POS_Z] = -0.5
    var s1 = model_site_offset[S_NBODY, S_NJOINT, 0, 0, 0](1)
    flat[s1 + SITE_IDX_BODY] = 0.0
    flat[s1 + SITE_IDX_POS_X] = 1.0

    var model_t = TensorImpl[DTYPE].alloc(S_MS)
    for i in range(S_MS):
        model_t.data[i] = flat[i]
    model_t.upload(ctx)

    var mf = ModelFields[DTYPE, S_NV, S_NBODY, S_NJOINT, 0, 0, 0, S_NSITE]()
    mf.load_from_slab(flat)
    mf.upload_all(ctx)

    # qpos: env0 angle 0.7, env1 angle -1.2
    var angles = List[Float64]()
    angles.append(0.7)
    angles.append(-1.2)

    # Legacy GPU.
    var slab_t = TensorImpl[DTYPE].alloc(S_BATCH * S_SS)
    for e in range(S_BATCH):
        slab_t.data[e * S_SS + 0] = Scalar[DTYPE](angles[e])
    slab_t.upload(ctx)
    ctx.enqueue_function[
        _legacy_fk_kernel[
            S_NQ, S_NV, S_NBODY, S_NJOINT, S_MC, S_SS, S_MS, S_BATCH, S_NSITE
        ]
    ](
        slab_t.lt["gpu", Layout.row_major(S_BATCH, S_SS)](),
        model_t.lt["gpu", Layout.row_major(1, S_MS)](),
        grid_dim=(S_BATCH,),
        block_dim=(1,),
    )
    slab_t.download(ctx)

    # Fields GPU (sites kernel variant).
    var d = DataFields[DTYPE, S_NQ, S_NV, S_NBODY, S_MC, S_NSITE, S_BATCH]()
    for e in range(S_BATCH):
        d.qpos.data[e * S_NQ + 0] = Scalar[DTYPE](angles[e])
    d.upload_all(ctx)
    forward_kinematics_fields[
        "gpu", DTYPE, S_NQ, S_NV, S_NBODY, S_NJOINT, S_MC, 0,
        0, 0, S_NSITE, 0, 0, S_BATCH,
    ](d, mf, ctx)
    d.download_all(ctx)

    comptime O_XPOS = xpos_offset[S_NQ, S_NV, S_NBODY]()
    comptime O_XQUAT = xquat_offset[S_NQ, S_NV, S_NBODY]()
    comptime O_XIPOS = xipos_offset[S_NQ, S_NV, S_NBODY]()
    comptime O_SITEX = site_xpos_offset[S_NQ, S_NV, S_NBODY, S_MC]()
    var bad = 0
    for e in range(S_BATCH):
        for j in range(S_NBODY * 3):
            if (
                d.xpos.data[e * S_NBODY * 3 + j]
                != slab_t.data[e * S_SS + O_XPOS + j]
            ):
                bad += 1
            if (
                d.xipos.data[e * S_NBODY * 3 + j]
                != slab_t.data[e * S_SS + O_XIPOS + j]
            ):
                bad += 1
        for j in range(S_NBODY * 4):
            if (
                d.xquat.data[e * S_NBODY * 4 + j]
                != slab_t.data[e * S_SS + O_XQUAT + j]
            ):
                bad += 1
        for j in range(S_NSITE * 3):
            if (
                d.site_xpos.data[e * S_NSITE * 3 + j]
                != slab_t.data[e * S_SS + O_SITEX + j]
            ):
                bad += 1
    if bad != 0:
        raise Error("synthetic fields-GPU vs legacy-GPU: not bit-exact")
    print("  PASS: fields-GPU == legacy-GPU bit-exact (incl. site_xpos)")

    # Fields CPU (sites path) vs fields GPU.
    var dc = DataFields[DTYPE, S_NQ, S_NV, S_NBODY, S_MC, S_NSITE, S_BATCH]()
    for e in range(S_BATCH):
        dc.qpos.data[e * S_NQ + 0] = Scalar[DTYPE](angles[e])
    forward_kinematics_fields[
        "cpu", DTYPE, S_NQ, S_NV, S_NBODY, S_NJOINT, S_MC, 0,
        0, 0, S_NSITE, 0, 0, S_BATCH,
    ](dc, mf)
    var worst = Float64(0)
    for i in range(S_BATCH * S_NBODY * 3):
        var err = abs(Float64(dc.xpos.data[i]) - Float64(d.xpos.data[i]))
        if err > worst:
            worst = err
    for i in range(S_BATCH * S_NSITE * 3):
        var err = abs(
            Float64(dc.site_xpos.data[i]) - Float64(d.site_xpos.data[i])
        )
        if err > worst:
            worst = err
    print("  fields-CPU vs fields-GPU worst err:", worst)
    if worst > POS_TOL:
        raise Error("synthetic fields-CPU tolerance exceeded")
    print("  PASS: fields-CPU within 1e-4 (incl. site_xpos)")


def main() raises:
    test_walker2d()
    test_synthetic_sites()
    print("test_fk_fields: ALL PASS")
