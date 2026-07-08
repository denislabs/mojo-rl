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
from mojo_rl.physics3d.dynamics.jacobian import (
    compute_subtree_com_gpu,
    compute_cdof_gpu,
)
from mojo_rl.physics3d.dynamics.subtree_com_fields import (
    compute_subtree_com_fields,
)
from mojo_rl.physics3d.dynamics.cdof_fields import compute_cdof_fields
from mojo_rl.physics3d.dynamics.mass_matrix import (
    compute_mass_matrix_full_gpu,
    ldl_factor_gpu,
    ldl_solve_workspace_gpu,
)
from mojo_rl.physics3d.dynamics.mass_matrix_fields import (
    compute_mass_matrix_fields,
)
from mojo_rl.physics3d.dynamics.ldl_fields import (
    ldl_factor_fields,
    ldl_solve_fields,
)
from mojo_rl.physics3d.dynamics.bias_forces import compute_bias_forces_rne_gpu
from mojo_rl.physics3d.dynamics.rne_fields import (
    compute_bias_forces_rne_fields,
)
from mojo_rl.physics3d.fields import DynamicsScratch
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    qpos_offset,
    xpos_offset,
    xquat_offset,
    xipos_offset,
    site_xpos_offset,
    subtree_com_offset,
    integrator_workspace_size,
    ws_cdof_offset,
    ws_M_offset,
    ws_L_offset,
    ws_D_offset,
    ws_fnet_offset,
    ws_qacc_ws_offset,
    ws_bias_offset,
    qvel_offset,
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


def _legacy_stcom_kernel[
    NQ_: Int,
    NV_: Int,
    NBODY_: Int,
    NJOINT_: Int,
    MC_: Int,
    SS_: Int,
    MS_: Int,
    B_: Int,
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS_), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS_), MutAnyOrigin],
):
    var env = Int(block_idx.x)
    if env >= B_:
        return
    compute_subtree_com_gpu[
        DTYPE, NQ_, NV_, NBODY_, NJOINT_, MC_, SS_, MS_, B_
    ](env, state, model)


def _legacy_cdof_kernel[
    NQ_: Int,
    NV_: Int,
    NBODY_: Int,
    NJOINT_: Int,
    MC_: Int,
    SS_: Int,
    MS_: Int,
    B_: Int,
    WS_: Int,
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS_), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS_), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS_), MutAnyOrigin],
):
    var env = Int(block_idx.x)
    if env >= B_:
        return
    compute_cdof_gpu[
        DTYPE, NQ_, NV_, NBODY_, NJOINT_, MC_, SS_, MS_, B_, WS_
    ](env, state, model, workspace)


def _legacy_mm_kernel[
    NQ_: Int,
    NV_: Int,
    NBODY_: Int,
    NJOINT_: Int,
    MC_: Int,
    SS_: Int,
    MS_: Int,
    B_: Int,
    WS_: Int,
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS_), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS_), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS_), MutAnyOrigin],
):
    var env = Int(block_idx.x)
    if env >= B_:
        return
    compute_mass_matrix_full_gpu[
        DTYPE, NQ_, NV_, NBODY_, NJOINT_, MC_, SS_, MS_, B_, WS_
    ](env, state, model, workspace)


def _legacy_rne_kernel[
    NQ_: Int,
    NV_: Int,
    NBODY_: Int,
    NJOINT_: Int,
    MC_: Int,
    SS_: Int,
    MS_: Int,
    B_: Int,
    WS_: Int,
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS_), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS_), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS_), MutAnyOrigin],
):
    var env = Int(block_idx.x)
    if env >= B_:
        return
    compute_bias_forces_rne_gpu[
        DTYPE, NQ_, NV_, NBODY_, NJOINT_, MC_, SS_, MS_, B_, WS_
    ](env, state, model, workspace)


def _legacy_ldl_kernel[
    NV_: Int, NBODY_: Int, B_: Int, WS_: Int
](workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS_), MutAnyOrigin],):
    var env = Int(block_idx.x)
    if env >= B_:
        return
    ldl_factor_gpu[DTYPE, NV_, NBODY_, B_, WS_](env, workspace)
    ldl_solve_workspace_gpu[DTYPE, NV_, NBODY_, B_, WS_](env, workspace)


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
    # DIAGNOSTIC: don't abort on the GPU-vs-GPU mismatch — let part 2
    # (fields-CPU vs fields-GPU) run so we learn WHICH GPU path is wrong.
    var gpu_mismatch = False
    var bad = 0
    var worst = Float64(0)
    var w_e = -1
    var w_i = -1
    var w_f = String("")
    var w_fv = Float64(0)
    var w_lv = Float64(0)
    for e in range(BATCH):
        for j in range(NBODY * 3):
            var pv = Float64(d.xpos.data[e * NBODY * 3 + j])
            var plv = Float64(slab_t.data[e * SS + O_XPOS + j])
            if pv != plv:
                bad += 1
                var dd = pv - plv
                if dd < 0:
                    dd = -dd
                if dd > worst:
                    worst = dd
                    w_e = e
                    w_i = j
                    w_f = "xpos"
                    w_fv = pv
                    w_lv = plv
            var iv = Float64(d.xipos.data[e * NBODY * 3 + j])
            var ilv = Float64(slab_t.data[e * SS + O_XIPOS + j])
            if iv != ilv:
                bad += 1
                var dd = iv - ilv
                if dd < 0:
                    dd = -dd
                if dd > worst:
                    worst = dd
                    w_e = e
                    w_i = j
                    w_f = "xipos"
                    w_fv = iv
                    w_lv = ilv
        for j in range(NBODY * 4):
            var qv = Float64(d.xquat.data[e * NBODY * 4 + j])
            var qlv = Float64(slab_t.data[e * SS + O_XQUAT + j])
            if qv != qlv:
                bad += 1
                var dd = qv - qlv
                if dd < 0:
                    dd = -dd
                if dd > worst:
                    worst = dd
                    w_e = e
                    w_i = j
                    w_f = "xquat"
                    w_fv = qv
                    w_lv = qlv
    if bad != 0:
        print(
            "  MISMATCH: bad_elems=",
            bad,
            " worst|delta|=",
            worst,
            " field=",
            w_f,
            " env=",
            w_e,
            " flat_idx=",
            w_i,
        )
        print("    fields-GPU=", w_fv, " legacy-GPU=", w_lv)
        gpu_mismatch = True
    else:
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
    # DISCRIMINATOR: worst_gpu is |fields-CPU - fields-GPU|. Large => the fields
    # GPU kernel miscomputes on this device; ~0 => fields-GPU is correct and the
    # legacy-GPU reference is the one diverging.
    if gpu_mismatch:
        if worst_gpu > QUAT_TOL:
            print(
                "  => fields-GPU disagrees with fields-CPU: the FIELDS GPU path"
                " miscomputes on this device."
            )
        else:
            print(
                "  => fields-GPU MATCHES fields-CPU: fields-GPU is correct; the"
                " LEGACY-GPU reference is the one diverging on this device."
            )
    if worst_gpu > QUAT_TOL or worst_cpu > QUAT_TOL:
        raise Error("walker2d fields-CPU tolerance exceeded")
    if gpu_mismatch:
        raise Error(
            "walker2d fields-GPU != legacy-GPU (see MISMATCH + discriminator"
            " above)"
        )
    print("  PASS: fields-CPU within 1e-4 of fields-GPU and legacy-CPU")

    # 3. subtree_com chained on the FK products (legacy slab still holds FK
    #    results on device; d holds bit-exact-equal xipos on device).
    ctx.enqueue_function[
        _legacy_stcom_kernel[NQ, NV, NBODY, NJOINT, MAX_CONTACTS, SS, MS, BATCH]
    ](
        slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
        model_t.lt["gpu", Layout.row_major(1, MS)](),
        grid_dim=(BATCH,),
        block_dim=(1,),
    )
    slab_t.download(ctx)
    compute_subtree_com_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        0, 0, 0, 0, 0, BATCH,
    ](d, mf, ctx)
    d.subtree_com.download(ctx)
    comptime O_STCOM = subtree_com_offset[NQ, NV, NBODY, MAX_CONTACTS]()
    var bad_st = 0
    for e in range(BATCH):
        for j in range(NBODY * 3):
            if (
                d.subtree_com.data[e * NBODY * 3 + j]
                != slab_t.data[e * SS + O_STCOM + j]
            ):
                bad_st += 1
    if bad_st != 0:
        raise Error("walker2d subtree_com fields-GPU vs legacy-GPU mismatch")
    print("  PASS: subtree_com fields-GPU == legacy-GPU bit-exact")

    compute_subtree_com_fields[
        "cpu", DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        0, 0, 0, 0, 0, BATCH,
    ](dc, mf)
    var worst_st = Float64(0)
    for i in range(BATCH * NBODY * 3):
        var err = abs(
            Float64(dc.subtree_com.data[i]) - Float64(d.subtree_com.data[i])
        )
        if err > worst_st:
            worst_st = err
    print("  subtree_com fields-CPU vs fields-GPU worst err:", worst_st)
    if worst_st > POS_TOL:
        raise Error("walker2d subtree_com fields-CPU tolerance exceeded")
    print("  PASS: subtree_com fields-CPU within 1e-4")

    # 4. cdof chained on FK + subtree_com (first workspace-array port:
    #    output lives in DynamicsScratch.cdof, not a ws slab).
    comptime WS = integrator_workspace_size[NV, NBODY]()
    var ws_t = TensorImpl[DTYPE].alloc(BATCH * WS)
    ws_t.upload(ctx)
    ctx.enqueue_function[
        _legacy_cdof_kernel[
            NQ, NV, NBODY, NJOINT, MAX_CONTACTS, SS, MS, BATCH, WS
        ]
    ](
        slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
        model_t.lt["gpu", Layout.row_major(1, MS)](),
        ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
        grid_dim=(BATCH,),
        block_dim=(1,),
    )
    ws_t.download(ctx)

    var scratch = DynamicsScratch[DTYPE, NV, NBODY, BATCH]()
    scratch.upload_all(ctx)
    compute_cdof_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        0, 0, 0, 0, 0, BATCH,
    ](d, mf, scratch, ctx)
    scratch.cdof.download(ctx)
    comptime O_CDOF = ws_cdof_offset()
    var bad_cd = 0
    for e in range(BATCH):
        for j in range(NV * 6):
            if (
                scratch.cdof.data[e * NV * 6 + j]
                != ws_t.data[e * WS + O_CDOF + j]
            ):
                bad_cd += 1
    if bad_cd != 0:
        raise Error("walker2d cdof fields-GPU vs legacy-GPU mismatch")
    print("  PASS: cdof fields-GPU == legacy-GPU bit-exact")

    var scratch_c = DynamicsScratch[DTYPE, NV, NBODY, BATCH]()
    compute_cdof_fields[
        "cpu", DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        0, 0, 0, 0, 0, BATCH,
    ](dc, mf, scratch_c)
    var worst_cd = Float64(0)
    for i in range(BATCH * NV * 6):
        var err = abs(
            Float64(scratch_c.cdof.data[i]) - Float64(scratch.cdof.data[i])
        )
        if err > worst_cd:
            worst_cd = err
    print("  cdof fields-CPU vs fields-GPU worst err:", worst_cd)
    if worst_cd > POS_TOL:
        raise Error("walker2d cdof fields-CPU tolerance exceeded")
    print("  PASS: cdof fields-CPU within 1e-4")

    # 5. CRBA mass matrix chained on cdof (reads scratch.cdof -> scratch.M).
    ctx.enqueue_function[
        _legacy_mm_kernel[
            NQ, NV, NBODY, NJOINT, MAX_CONTACTS, SS, MS, BATCH, WS
        ]
    ](
        slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
        model_t.lt["gpu", Layout.row_major(1, MS)](),
        ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
        grid_dim=(BATCH,),
        block_dim=(1,),
    )
    ws_t.download(ctx)
    compute_mass_matrix_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        0, 0, 0, 0, 0, BATCH,
    ](d, mf, scratch, ctx)
    scratch.M.download(ctx)
    comptime O_M = ws_M_offset[NV, NBODY]()
    var bad_mm = 0
    for e in range(BATCH):
        for j in range(NV * NV):
            if scratch.M.data[e * NV * NV + j] != ws_t.data[e * WS + O_M + j]:
                bad_mm += 1
    if bad_mm != 0:
        raise Error("walker2d mass matrix fields-GPU vs legacy-GPU mismatch")
    print("  PASS: mass matrix fields-GPU == legacy-GPU bit-exact")

    compute_mass_matrix_fields[
        "cpu", DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        0, 0, 0, 0, 0, BATCH,
    ](dc, mf, scratch_c)
    var worst_mm = Float64(0)
    for i in range(BATCH * NV * NV):
        var err = abs(
            Float64(scratch_c.M.data[i]) - Float64(scratch.M.data[i])
        )
        if err > worst_mm:
            worst_mm = err
    print("  mass matrix fields-CPU vs fields-GPU worst err:", worst_mm)
    if worst_mm > 1e-3:
        raise Error("walker2d mass matrix fields-CPU tolerance exceeded")
    print("  PASS: mass matrix fields-CPU within 1e-3")

    # 6. LDL factor + solve chained on M (fnet = synthetic per-DOF forces).
    comptime O_FNET = ws_fnet_offset[NV, NBODY]()
    for e in range(BATCH):
        for i in range(NV):
            var f = Scalar[DTYPE]((e * 31 + i * 7) % 11 - 5) / 3.0
            ws_t.data[e * WS + O_FNET + i] = f
            scratch.fnet.data[e * NV + i] = f
            scratch_c.fnet.data[e * NV + i] = f
    ws_t.upload(ctx)  # host holds cdof+M from download; re-upload whole slab
    scratch.fnet.upload(ctx)
    ctx.enqueue_function[_legacy_ldl_kernel[NV, NBODY, BATCH, WS]](
        ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
        grid_dim=(BATCH,),
        block_dim=(1,),
    )
    ws_t.download(ctx)
    ldl_factor_fields["gpu", DTYPE, NV, NBODY, BATCH](scratch, ctx)
    ldl_solve_fields["gpu", DTYPE, NV, NBODY, BATCH](scratch, ctx)
    scratch.L.download(ctx)
    scratch.D.download(ctx)
    scratch.qacc_ws.download(ctx)
    comptime O_L = ws_L_offset[NV, NBODY]()
    comptime O_D = ws_D_offset[NV, NBODY]()
    comptime O_QW = ws_qacc_ws_offset[NV, NBODY]()
    var bad_ld = 0
    for e in range(BATCH):
        for j in range(NV * NV):
            if scratch.L.data[e * NV * NV + j] != ws_t.data[e * WS + O_L + j]:
                bad_ld += 1
        for j in range(NV):
            if scratch.D.data[e * NV + j] != ws_t.data[e * WS + O_D + j]:
                bad_ld += 1
            if (
                scratch.qacc_ws.data[e * NV + j]
                != ws_t.data[e * WS + O_QW + j]
            ):
                bad_ld += 1
    if bad_ld != 0:
        raise Error("walker2d LDL fields-GPU vs legacy-GPU mismatch")
    print("  PASS: LDL factor+solve fields-GPU == legacy-GPU bit-exact")

    ldl_factor_fields["cpu", DTYPE, NV, NBODY, BATCH](scratch_c)
    ldl_solve_fields["cpu", DTYPE, NV, NBODY, BATCH](scratch_c)
    var worst_ld = Float64(0)
    for i in range(BATCH * NV):
        var err = abs(
            Float64(scratch_c.qacc_ws.data[i])
            - Float64(scratch.qacc_ws.data[i])
        )
        if err > worst_ld:
            worst_ld = err
    print("  LDL qacc fields-CPU vs fields-GPU worst err:", worst_ld)
    if worst_ld > 1e-3:
        raise Error("walker2d LDL fields-CPU tolerance exceeded")
    print("  PASS: LDL qacc fields-CPU within 1e-3")

    # 7. RNE bias forces chained on FK products + cdof (synthetic qvel;
    #    qvel affects no earlier stage).
    comptime O_QVEL = qvel_offset[NQ, NV]()
    for e in range(BATCH):
        for i in range(NV):
            var v = Scalar[DTYPE]((e * 17 + i * 13) % 9 - 4) / 4.0
            slab_t.data[e * SS + O_QVEL + i] = v
            d.qvel.data[e * NV + i] = v
            dc.qvel.data[e * NV + i] = v
    slab_t.upload(ctx)
    d.qvel.upload(ctx)
    ctx.enqueue_function[
        _legacy_rne_kernel[
            NQ, NV, NBODY, NJOINT, MAX_CONTACTS, SS, MS, BATCH, WS
        ]
    ](
        slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
        model_t.lt["gpu", Layout.row_major(1, MS)](),
        ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
        grid_dim=(BATCH,),
        block_dim=(1,),
    )
    ws_t.download(ctx)
    compute_bias_forces_rne_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        0, 0, 0, 0, 0, BATCH,
    ](d, mf, scratch, ctx)
    scratch.bias.download(ctx)
    comptime O_BIAS = ws_bias_offset[NV, NBODY]()
    var bad_rne = 0
    for e in range(BATCH):
        for j in range(NV):
            if (
                scratch.bias.data[e * NV + j]
                != ws_t.data[e * WS + O_BIAS + j]
            ):
                bad_rne += 1
    if bad_rne != 0:
        raise Error("walker2d RNE bias fields-GPU vs legacy-GPU mismatch")
    print("  PASS: RNE bias fields-GPU == legacy-GPU bit-exact")

    compute_bias_forces_rne_fields[
        "cpu", DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        0, 0, 0, 0, 0, BATCH,
    ](dc, mf, scratch_c)
    var worst_rne = Float64(0)
    for i in range(BATCH * NV):
        var err = abs(
            Float64(scratch_c.bias.data[i]) - Float64(scratch.bias.data[i])
        )
        if err > worst_rne:
            worst_rne = err
    print("  RNE bias fields-CPU vs fields-GPU worst err:", worst_rne)
    if worst_rne > 1e-3:
        raise Error("walker2d RNE fields-CPU tolerance exceeded")
    print("  PASS: RNE bias fields-CPU within 1e-3")


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
