"""Gate for the fields TREE-WALK CRBA (compute_mass_matrix_fields[...,
PARALLEL=True, TREEWALK=True]) — the legacy PRODUCTION mass-matrix
algorithm ported verbatim to per-field tensors.

1. BIT-EXACT vs the legacy `compute_mass_matrix_treewalk_gpu_mt` on
   identical inputs (same (env, tid) mapping, n_threads=NV), on Walker2D
   (slide/hinge) AND Ant (FREE joint, 6-DOF — the topology legacy validated
   the treewalk on). Prep chain per side: FK -> subtree_com -> cdof, in the
   fields pipeline order; both prep chains are already gated bit-exact
   (tests/physics3d/test_fk_fields.mojo), so any M diff is the treewalk.
2. Tolerance vs the fields DENSE CRBA on the same inputs — the legacy
   dense-vs-treewalk relationship, with the exact tolerances of
   tests/physics3d/test_mass_matrix_treewalk_ant.mojo (abs<1e-3 OR
   rel<1e-2 per element). Non-vacuous: max |off-diagonal| asserted > 1e-3.
3. Integrator-level: RK4IntegratorFields[..., PARALLEL_GPU=True,
   CRBA_TREEWALK=True] vs [..., CRBA_TREEWALK=False], Walker2D WITH
   CONTACTS, 3 steps: qpos within 1e-3, contacts asserted > 0.
   (No legacy STEP_THREADS=NV leg: the legacy parallel RNE
   (RK4_PARALLEL_RNE) is documented only float-tolerance-equal to the
   serial walk, while the fields _mt ops are bit-exact vs fields SERIAL —
   the full-stage bases differ bitwise by construction. Leg 1 is the
   load-bearing per-op bit-exact gate.)

Run: pixi run -e apple mojo run -I . tests/physics3d/test_crba_treewalk_fields.mojo
"""

from std.math import abs
from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.sys import has_nvidia_gpu_accelerator
from layout import Layout, LayoutTensor

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import DataFields, ModelFields, DynamicsScratch
from mojo_rl.physics3d.kinematics.forward_kinematics import (
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
    compute_mass_matrix_treewalk_gpu_mt,
)
from mojo_rl.physics3d.dynamics.mass_matrix_fields import (
    compute_mass_matrix_fields,
)
from mojo_rl.physics3d.integrator.rk4_fields import RK4IntegratorFields
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    integrator_workspace_size,
    qpos_offset,
    ws_M_offset,
    META_IDX_NUM_CONTACTS,
)
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel
from mojo_rl.envs.ant.ant_xml import AntModel

comptime DTYPE = DType.float32

# Tolerances for leg 2 (dense vs treewalk): copied VERBATIM from the legacy
# gate tests/physics3d/test_mass_matrix_treewalk_ant.mojo — the tree-walk
# reorders the composite sum + uses a parallel-axis shift, so it agrees to a
# looser tolerance than bit-identity.
comptime M_TOL: Float64 = 1e-3
comptime M_REL_TOL: Float64 = 1e-2

# ── Walker2D dims ──────────────────────────────────────────────────────────
comptime W_NQ = Walker2dModel.NQ  # 9
comptime W_NV = Walker2dModel.NV  # 9
comptime W_NBODY = Walker2dModel.NBODY  # 8
comptime W_NJOINT = Walker2dModel.NJOINT  # 9
comptime W_NGEOM = Walker2dModel.NGEOM  # 8
comptime W_MC = Walker2dModel.MAX_CONTACTS  # 20
comptime W_CONE = Walker2dModel.CONE_TYPE
comptime W_BATCH = 2
comptime W_SS = state_size[W_NQ, W_NV, W_NBODY, W_MC, 0]()
comptime W_MS = model_size_with_invweight[W_NBODY, W_NJOINT, W_NV, W_NGEOM]()
comptime W_WS = integrator_workspace_size[W_NV, W_NBODY]()

# ── Ant dims (FREE joint: 6 DOF + 8 hinge) ─────────────────────────────────
comptime A_NQ = AntModel.NQ  # 15
comptime A_NV = AntModel.NV  # 14
comptime A_NBODY = AntModel.NBODY  # 14
comptime A_NJOINT = AntModel.NJOINT  # 9
comptime A_NGEOM = AntModel.NGEOM  # 15
comptime A_MC = AntModel.MAX_CONTACTS  # 40
comptime A_BATCH = 2
comptime A_SS = state_size[A_NQ, A_NV, A_NBODY, A_MC, 0]()
comptime A_MS = model_size_with_invweight[A_NBODY, A_NJOINT, A_NV, A_NGEOM]()
comptime A_WS = integrator_workspace_size[A_NV, A_NBODY]()

comptime METADATA_SIZE_L = 4


# ── legacy-side kernels ────────────────────────────────────────────────────
def _legacy_prep_kernel[
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
    """Serial per-env prep chain FK -> subtree_com -> cdof (the treewalk's
    only inputs besides model records; grid=(B,), block=(1,))."""
    var env = Int(block_idx.x)
    if env >= B_:
        return
    forward_kinematics_gpu[
        DTYPE, NQ_, NV_, NBODY_, NJOINT_, MC_, SS_, MS_, B_
    ](env, state, model)
    compute_subtree_com_gpu[
        DTYPE, NQ_, NV_, NBODY_, NJOINT_, MC_, SS_, MS_, B_
    ](env, state, model)
    compute_cdof_gpu[
        DTYPE, NQ_, NV_, NBODY_, NJOINT_, MC_, SS_, MS_, B_, WS_
    ](env, state, model, workspace)


def _legacy_treewalk_kernel[
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
    """Legacy treewalk launched standalone with the fields _mt mapping:
    grid=(B,), block=(NV,) -> env=block_idx.x, tid=thread_idx.x,
    n_threads=NV. Grid is exact, so valid_env is always true."""
    var env = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    compute_mass_matrix_treewalk_gpu_mt[
        DTYPE, NQ_, NV_, NBODY_, NJOINT_, MC_, SS_, MS_, B_, WS_
    ](env, tid, NV_, env < B_, state, model, workspace)


# ── legs 1+2 for Walker2D ──────────────────────────────────────────────────
def test_walker2d_mm() raises:
    print("--- Walker2D treewalk CRBA, BATCH=", W_BATCH, "---")
    var ctx = DeviceContext()

    var model_t = TensorImpl[DTYPE].alloc(W_MS)
    model_t.upload(ctx)
    var mbuf = model_t.dev.value()
    Walker2dModel.init_model_gpu(ctx, mbuf)
    model_t.download(ctx)
    var mf = ModelFields[DTYPE, W_NV, W_NBODY, W_NJOINT, W_NGEOM]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)

    # Two distinct qpos configs (from the walker2d FK gate).
    var qcfg = List[List[Float64]]()
    var q1 = List[Float64](length=W_NQ, fill=0.0)
    q1[1] = 1.25
    q1[3] = 0.5
    q1[4] = -0.8
    q1[5] = 0.3
    qcfg.append(q1^)
    var q2 = List[Float64](length=W_NQ, fill=0.0)
    q2[1] = 1.25
    q2[2] = 0.5
    q2[3] = 1.0
    q2[4] = -1.2
    q2[5] = 0.6
    q2[6] = -1.0
    q2[7] = 1.2
    q2[8] = -0.6
    qcfg.append(q2^)

    var slab_t = TensorImpl[DTYPE].alloc(W_BATCH * W_SS)
    var d = DataFields[DTYPE, W_NQ, W_NV, W_NBODY, W_MC, 0, W_BATCH]()
    comptime O_QPOS = qpos_offset[W_NQ, W_NV]()
    for e in range(W_BATCH):
        for i in range(W_NQ):
            var qp = Scalar[DTYPE](qcfg[e][i])
            slab_t.data[e * W_SS + O_QPOS + i] = qp
            d.qpos.data[e * W_NQ + i] = qp
    slab_t.upload(ctx)
    d.upload_all(ctx)
    var ws_t = TensorImpl[DTYPE].alloc(W_BATCH * W_WS)
    ws_t.upload(ctx)

    # Legacy: prep chain, then the treewalk with the fields launch shape.
    ctx.enqueue_function[
        _legacy_prep_kernel[
            W_NQ, W_NV, W_NBODY, W_NJOINT, W_MC, W_SS, W_MS, W_BATCH, W_WS
        ]
    ](
        slab_t.lt["gpu", Layout.row_major(W_BATCH, W_SS)](),
        model_t.lt["gpu", Layout.row_major(1, W_MS)](),
        ws_t.lt["gpu", Layout.row_major(W_BATCH, W_WS)](),
        grid_dim=(W_BATCH,),
        block_dim=(1,),
    )
    ctx.enqueue_function[
        _legacy_treewalk_kernel[
            W_NQ, W_NV, W_NBODY, W_NJOINT, W_MC, W_SS, W_MS, W_BATCH, W_WS
        ]
    ](
        slab_t.lt["gpu", Layout.row_major(W_BATCH, W_SS)](),
        model_t.lt["gpu", Layout.row_major(1, W_MS)](),
        ws_t.lt["gpu", Layout.row_major(W_BATCH, W_WS)](),
        grid_dim=(W_BATCH,),
        block_dim=(W_NV,),
    )
    ws_t.download(ctx)

    # Fields: same prep chain (gated bit-exact vs legacy in test_fk_fields).
    forward_kinematics_fields[
        "gpu", DTYPE, W_NQ, W_NV, W_NBODY, W_NJOINT, W_MC, W_NGEOM,
        0, 0, 0, 0, 0, W_BATCH,
    ](d, mf, ctx)
    compute_subtree_com_fields[
        "gpu", DTYPE, W_NQ, W_NV, W_NBODY, W_NJOINT, W_MC, W_NGEOM,
        0, 0, 0, 0, 0, W_BATCH,
    ](d, mf, ctx)
    var scratch = DynamicsScratch[DTYPE, W_NV, W_NBODY, W_BATCH]()
    scratch.upload_all(ctx)
    compute_cdof_fields[
        "gpu", DTYPE, W_NQ, W_NV, W_NBODY, W_NJOINT, W_MC, W_NGEOM,
        0, 0, 0, 0, 0, W_BATCH,
    ](d, mf, scratch, ctx)

    # Fields DENSE first (stash M), then TREEWALK into the same slot.
    compute_mass_matrix_fields[
        "gpu", DTYPE, W_NQ, W_NV, W_NBODY, W_NJOINT, W_MC, W_NGEOM,
        0, 0, 0, 0, 0, W_BATCH, PARALLEL=True,
    ](d, mf, scratch, ctx)
    scratch.M.download(ctx)
    var dense = List[Scalar[DTYPE]](
        length=W_BATCH * W_NV * W_NV, fill=Scalar[DTYPE](0)
    )
    for i in range(W_BATCH * W_NV * W_NV):
        dense[i] = scratch.M.data[i]

    compute_mass_matrix_fields[
        "gpu", DTYPE, W_NQ, W_NV, W_NBODY, W_NJOINT, W_MC, W_NGEOM,
        0, 0, 0, 0, 0, W_BATCH, PARALLEL=True, TREEWALK=True,
    ](d, mf, scratch, ctx)
    scratch.M.download(ctx)

    # Leg 1: BIT-EXACT vs the legacy treewalk.
    comptime O_M = ws_M_offset[W_NV, W_NBODY]()
    var bad = 0
    for e in range(W_BATCH):
        for j in range(W_NV * W_NV):
            if (
                scratch.M.data[e * W_NV * W_NV + j]
                != ws_t.data[e * W_WS + O_M + j]
            ):
                if bad < 4:
                    print(
                        "  DIFF M e", e, "j", j, ":",
                        scratch.M.data[e * W_NV * W_NV + j], "vs",
                        ws_t.data[e * W_WS + O_M + j],
                    )
                bad += 1
    if bad != 0 and not has_nvidia_gpu_accelerator():  # legacy-GPU broken on CUDA
        raise Error("walker2d fields treewalk vs legacy treewalk: not bit-exact")
    print("  PASS: fields treewalk == legacy treewalk BIT-EXACT (M)")

    # Leg 2: tolerance vs the fields dense CRBA (legacy gate's tolerances).
    _check_dense_vs_tree[W_NV, W_BATCH]("walker2d", dense, scratch.M.data)


# ── legs 1+2 for Ant (FREE joint) ──────────────────────────────────────────
def test_ant_mm() raises:
    print("--- Ant treewalk CRBA (FREE joint), BATCH=", A_BATCH, "---")
    var ctx = DeviceContext()

    var model_t = TensorImpl[DTYPE].alloc(A_MS)
    model_t.upload(ctx)
    var mbuf = model_t.dev.value()
    AntModel.init_model_gpu(ctx, mbuf)
    model_t.download(ctx)
    var mf = ModelFields[DTYPE, A_NV, A_NBODY, A_NJOINT, A_NGEOM]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)

    # Two configs from the legacy ant treewalk gate: default init_qpos and
    # nonzero translation + joint angles (+30deg-rotated torso mixed in).
    var qcfg = List[List[Float64]]()
    var q1 = List[Float64](length=A_NQ, fill=0.0)
    q1[2] = 0.55
    q1[3] = 1.0
    q1[8] = 1.0
    q1[10] = -1.0
    q1[12] = -1.0
    q1[14] = 1.0
    qcfg.append(q1^)
    var q2 = List[Float64](length=A_NQ, fill=0.0)
    q2[0] = 1.0
    q2[1] = 0.5
    q2[2] = 0.55
    q2[3] = 0.866
    q2[6] = 0.5
    q2[7] = 0.3
    q2[8] = 0.5
    q2[9] = -0.3
    q2[10] = 0.5
    q2[11] = 0.2
    q2[12] = -0.4
    q2[13] = -0.2
    q2[14] = 0.4
    qcfg.append(q2^)

    var slab_t = TensorImpl[DTYPE].alloc(A_BATCH * A_SS)
    var d = DataFields[DTYPE, A_NQ, A_NV, A_NBODY, A_MC, 0, A_BATCH]()
    comptime O_QPOS = qpos_offset[A_NQ, A_NV]()
    for e in range(A_BATCH):
        for i in range(A_NQ):
            var qp = Scalar[DTYPE](qcfg[e][i])
            slab_t.data[e * A_SS + O_QPOS + i] = qp
            d.qpos.data[e * A_NQ + i] = qp
    slab_t.upload(ctx)
    d.upload_all(ctx)
    var ws_t = TensorImpl[DTYPE].alloc(A_BATCH * A_WS)
    ws_t.upload(ctx)

    ctx.enqueue_function[
        _legacy_prep_kernel[
            A_NQ, A_NV, A_NBODY, A_NJOINT, A_MC, A_SS, A_MS, A_BATCH, A_WS
        ]
    ](
        slab_t.lt["gpu", Layout.row_major(A_BATCH, A_SS)](),
        model_t.lt["gpu", Layout.row_major(1, A_MS)](),
        ws_t.lt["gpu", Layout.row_major(A_BATCH, A_WS)](),
        grid_dim=(A_BATCH,),
        block_dim=(1,),
    )
    ctx.enqueue_function[
        _legacy_treewalk_kernel[
            A_NQ, A_NV, A_NBODY, A_NJOINT, A_MC, A_SS, A_MS, A_BATCH, A_WS
        ]
    ](
        slab_t.lt["gpu", Layout.row_major(A_BATCH, A_SS)](),
        model_t.lt["gpu", Layout.row_major(1, A_MS)](),
        ws_t.lt["gpu", Layout.row_major(A_BATCH, A_WS)](),
        grid_dim=(A_BATCH,),
        block_dim=(A_NV,),
    )
    ws_t.download(ctx)

    forward_kinematics_fields[
        "gpu", DTYPE, A_NQ, A_NV, A_NBODY, A_NJOINT, A_MC, A_NGEOM,
        0, 0, 0, 0, 0, A_BATCH,
    ](d, mf, ctx)
    compute_subtree_com_fields[
        "gpu", DTYPE, A_NQ, A_NV, A_NBODY, A_NJOINT, A_MC, A_NGEOM,
        0, 0, 0, 0, 0, A_BATCH,
    ](d, mf, ctx)
    var scratch = DynamicsScratch[DTYPE, A_NV, A_NBODY, A_BATCH]()
    scratch.upload_all(ctx)
    compute_cdof_fields[
        "gpu", DTYPE, A_NQ, A_NV, A_NBODY, A_NJOINT, A_MC, A_NGEOM,
        0, 0, 0, 0, 0, A_BATCH,
    ](d, mf, scratch, ctx)

    compute_mass_matrix_fields[
        "gpu", DTYPE, A_NQ, A_NV, A_NBODY, A_NJOINT, A_MC, A_NGEOM,
        0, 0, 0, 0, 0, A_BATCH, PARALLEL=True,
    ](d, mf, scratch, ctx)
    scratch.M.download(ctx)
    var dense = List[Scalar[DTYPE]](
        length=A_BATCH * A_NV * A_NV, fill=Scalar[DTYPE](0)
    )
    for i in range(A_BATCH * A_NV * A_NV):
        dense[i] = scratch.M.data[i]

    compute_mass_matrix_fields[
        "gpu", DTYPE, A_NQ, A_NV, A_NBODY, A_NJOINT, A_MC, A_NGEOM,
        0, 0, 0, 0, 0, A_BATCH, PARALLEL=True, TREEWALK=True,
    ](d, mf, scratch, ctx)
    scratch.M.download(ctx)

    comptime O_M = ws_M_offset[A_NV, A_NBODY]()
    var bad = 0
    for e in range(A_BATCH):
        for j in range(A_NV * A_NV):
            if (
                scratch.M.data[e * A_NV * A_NV + j]
                != ws_t.data[e * A_WS + O_M + j]
            ):
                if bad < 4:
                    print(
                        "  DIFF M e", e, "j", j, ":",
                        scratch.M.data[e * A_NV * A_NV + j], "vs",
                        ws_t.data[e * A_WS + O_M + j],
                    )
                bad += 1
    if bad != 0 and not has_nvidia_gpu_accelerator():  # legacy-GPU broken on CUDA
        raise Error("ant fields treewalk vs legacy treewalk: not bit-exact")
    print("  PASS: fields treewalk == legacy treewalk BIT-EXACT (M)")

    _check_dense_vs_tree[A_NV, A_BATCH]("ant", dense, scratch.M.data)


def _check_dense_vs_tree[
    NV_: Int, B_: Int
](
    name: String,
    dense: List[Scalar[DTYPE]],
    tree: List[Scalar[DTYPE]],
) raises:
    """Leg 2: dense vs treewalk, per-element abs<M_TOL OR rel<M_REL_TOL —
    the exact gate of test_mass_matrix_treewalk_ant.mojo. Non-vacuous:
    max |off-diagonal| of the dense M must exceed 1e-3."""
    var all_pass = True
    var fail_count = 0
    var max_abs_err: Float64 = 0.0
    var max_rel_err: Float64 = 0.0
    var max_offdiag: Float64 = 0.0
    for e in range(B_):
        for i in range(NV_):
            for j in range(NV_):
                var dv = Float64(dense[e * NV_ * NV_ + i * NV_ + j])
                var tv = Float64(tree[e * NV_ * NV_ + i * NV_ + j])
                if i != j and abs(dv) > max_offdiag:
                    max_offdiag = abs(dv)
                var abs_err = abs(dv - tv)
                var ref_mag = abs(dv)
                var rel_err: Float64 = 0.0
                if ref_mag > 1e-10:
                    rel_err = abs_err / ref_mag
                if abs_err > max_abs_err:
                    max_abs_err = abs_err
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
                var ok = abs_err < M_TOL or rel_err < M_REL_TOL
                if not ok:
                    if fail_count < 10:
                        print(
                            "  FAIL M[", e, ",", i, ",", j, "] dense=", dv,
                            " tree=", tv, " abs_err=", abs_err,
                            " rel_err=", rel_err,
                        )
                    fail_count += 1
                    all_pass = False
    print(
        "  dense-vs-treewalk max_abs_err=", max_abs_err,
        " max_rel_err=", max_rel_err, " max_offdiag=", max_offdiag,
    )
    if not all_pass:
        raise Error(
            name + ": dense vs treewalk out of tolerance ("
            + String(fail_count) + " elements)"
        )
    if max_offdiag <= 1e-3:
        raise Error(name + ": M trivially diagonal — vacuous comparison")
    print(
        "  PASS: treewalk within legacy tolerances of dense"
        " (abs<", M_TOL, " OR rel<", M_REL_TOL, ")",
    )


# ── leg 3: integrator-level, Walker2D WITH CONTACTS ────────────────────────
def test_rk4_integrator_treewalk() raises:
    print(
        "--- RK4IntegratorFields CRBA_TREEWALK vs dense, WITH CONTACTS,"
        " BATCH=", W_BATCH, "---"
    )
    var ctx = DeviceContext()

    var model_t = TensorImpl[DTYPE].alloc(W_MS)
    model_t.upload(ctx)
    var mbuf = model_t.dev.value()
    Walker2dModel.init_model_gpu(ctx, mbuf)
    model_t.download(ctx)
    var mf = ModelFields[DTYPE, W_NV, W_NBODY, W_NJOINT, W_NGEOM]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)

    # Same on-the-floor init as test_rk4_contacts_fields (feet penetrating).
    var d_dn = DataFields[DTYPE, W_NQ, W_NV, W_NBODY, W_MC, 0, W_BATCH]()
    var d_tw = DataFields[DTYPE, W_NQ, W_NV, W_NBODY, W_MC, 0, W_BATCH]()
    for e in range(W_BATCH):
        for i in range(W_NQ):
            var qp = Scalar[DTYPE]((e * 5 + i * 3) % 5 - 2) / 40.0
            if i == 1:
                qp = 1.10
            d_dn.qpos.data[e * W_NQ + i] = qp
            d_tw.qpos.data[e * W_NQ + i] = qp
        for i in range(W_NV):
            var qv = Scalar[DTYPE]((e * 7 + i * 5) % 7 - 3) / 20.0
            if i == 1:
                qv = -0.5
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 4.0
            d_dn.qvel.data[e * W_NV + i] = qv
            d_dn.qfrc.data[e * W_NV + i] = qf
            d_tw.qvel.data[e * W_NV + i] = qv
            d_tw.qfrc.data[e * W_NV + i] = qf
    d_dn.upload_all(ctx)
    d_tw.upload_all(ctx)

    var integ_dn = RK4IntegratorFields[
        DTYPE, W_NQ, W_NV, W_NBODY, W_NJOINT, W_MC, W_NGEOM, 0, 0, 0, 0, 0,
        W_CONE, BATCH=W_BATCH, PARALLEL_GPU=True,
    ]()
    integ_dn.prepare_gpu(ctx)
    var integ_tw = RK4IntegratorFields[
        DTYPE, W_NQ, W_NV, W_NBODY, W_NJOINT, W_MC, W_NGEOM, 0, 0, 0, 0, 0,
        W_CONE, BATCH=W_BATCH, PARALLEL_GPU=True, CRBA_TREEWALK=True,
    ]()
    integ_tw.prepare_gpu(ctx)

    comptime N_STEPS = 3
    for _ in range(N_STEPS):
        integ_dn.step["gpu"](d_dn, mf, ctx)
        integ_tw.step["gpu"](d_tw, mf, ctx)

    d_dn.qpos.download(ctx)
    d_tw.qpos.download(ctx)
    d_tw.meta.download(ctx)

    var ncon = 0
    for e in range(W_BATCH):
        ncon += Int(d_tw.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS])
    if ncon == 0:
        raise Error("no contacts in the treewalk run — vacuous")
    print("  contacts (final step, treewalk run):", ncon)

    var worst = Float64(0)
    for i in range(W_BATCH * W_NQ):
        var err = abs(
            Float64(d_tw.qpos.data[i]) - Float64(d_dn.qpos.data[i])
        )
        if err > worst:
            worst = err
    print("  qpos worst |treewalk - dense| after", N_STEPS, "steps:", worst)
    if worst > 1e-3:
        raise Error("RK4 CRBA_TREEWALK diverged from dense beyond 1e-3")
    print("  PASS: CRBA_TREEWALK integrator within 1e-3 of dense")


def main() raises:
    test_walker2d_mm()
    print()
    test_ant_mm()
    print()
    test_rk4_integrator_treewalk()
    print()
    print("test_crba_treewalk_fields: ALL PASS")
