"""Fields-native constraint inverse-weights (body_invweight0 / dof_invweight0).

Port of legacy `dynamics/mass_matrix.compute_body_invweight0` (MuJoCo
`mj_setConst`) onto the per-field tensors — a ONE-TIME CPU build step run by the
fields model build (`init_fields`), replacing the CPU-`Model` invweight0 path.

Given a `Data` already at the reference pose (`d.qpos` set), it runs the
fields FK → subtree_com → cdof → CRBA(+armature) → LDL → M^-1 pipeline, then:
  body_invweight0[2i]   = avg(A[0,0], A[1,1], A[2,2])   (translation)
  body_invweight0[2i+1] = avg(A[3,3], A[4,4], A[5,5])   (rotation)
where A = J M^-1 J^T for the 6xNV body-CoM Jacobian J, and
  dof_invweight0[d]     = (M^-1)[d,d].
These feed every constraint/solver inverse-weight, so this must be
float-tolerance-equal to the legacy routine (gated in test_invweight0_fields).

CPU-only (build-time); no GPU kernels needed.
"""

from mojo_rl.physics3d.fields import Data, Model, DynamicsScratch
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
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
)
from mojo_rl.physics3d.joint_types import JNT_FREE, JNT_BALL
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    BODY_IDX_PARENT,
    BODY_IDX_ROOTID,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_ARMATURE,
)


def compute_invweight0[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
    NTENDON: Int = 0,
    NSITE: Int = 0,
    NEXCLUDE: Int = 0,
    NMESH_VERTS: Int = 0,
](
    mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
    mut mf: Model[
        DTYPE, NV, NBODY, NJOINT, NGEOM, NEQUALITY, NTENDON, NSITE, NEXCLUDE,
        NMESH_VERTS,
    ],
    mut sc: DynamicsScratch[DTYPE, NV, NBODY, 1],
) raises:
    """Compute mf.body_invweight0 / mf.dof_invweight0 from `d` (at the reference
    pose) — see module docstring. Build-time, single-env (BATCH=1)."""
    # ── fields pipeline: FK -> subtree_com -> cdof -> CRBA -> +armature -> LDL -> M^-1
    forward_kinematics[
        "cpu", DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, NEQUALITY,
        NTENDON, NSITE, NEXCLUDE, NMESH_VERTS, 1,
    ](d, mf, None)
    compute_subtree_com[
        "cpu", DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, NEQUALITY,
        NTENDON, NSITE, NEXCLUDE, NMESH_VERTS, 1,
    ](d, mf, None)
    compute_cdof[
        "cpu", DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, NEQUALITY,
        NTENDON, NSITE, NEXCLUDE, NMESH_VERTS, 1,
    ](d, mf, sc, None)
    compute_mass_matrix[
        "cpu", DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, NEQUALITY,
        NTENDON, NSITE, NEXCLUDE, NMESH_VERTS, 1,
    ](d, mf, sc, None)

    # Add armature to the diagonal (matches legacy mass_matrix.mojo:447-457).
    for j in range(NJOINT):
        var jo = j * MODEL_JOINT_SIZE
        var jtype = Int(mf.joints.data[jo + JOINT_IDX_TYPE])
        var dof_adr = Int(mf.joints.data[jo + JOINT_IDX_DOF_ADR])
        var arm = mf.joints.data[jo + JOINT_IDX_ARMATURE]
        var ndof = 1
        if jtype == JNT_FREE:
            ndof = 6
        elif jtype == JNT_BALL:
            ndof = 3
        for dd in range(ndof):
            sc.M.data[(dof_adr + dd) * NV + (dof_adr + dd)] += arm

    ldl_factor["cpu", DTYPE, NV, NBODY, 1](sc)

    # ── dof->body map (matches legacy :461-476) ──────────────────────────────
    var dof_body = List[Int](length=NV, fill=0)
    for j in range(NJOINT):
        var jo = j * MODEL_JOINT_SIZE
        var jtype = Int(mf.joints.data[jo + JOINT_IDX_TYPE])
        var body = Int(mf.joints.data[jo + JOINT_IDX_BODY_ID])
        var dof_adr = Int(mf.joints.data[jo + JOINT_IDX_DOF_ADR])
        var ndof = 1
        if jtype == JNT_FREE:
            ndof = 6
        elif jtype == JNT_BALL:
            ndof = 3
        for dd in range(ndof):
            dof_body[dof_adr + dd] = body

    # World body: zero weights.
    mf.body_invweight0.data[0] = Scalar[DTYPE](0)
    mf.body_invweight0.data[1] = Scalar[DTYPE](0)

    # ── per-body invweight0 via A = J M^-1 J^T diagonal (legacy :482-586) ─────
    for i in range(NBODY):
        var ti_x = d.xipos.data[i * 3 + 0]
        var ti_y = d.xipos.data[i * 3 + 1]
        var ti_z = d.xipos.data[i * 3 + 2]

        var A_diag = InlineArray[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))
        for k in range(6):
            var J_row = List[Scalar[DTYPE]](length=NV, fill=Scalar[DTYPE](0))
            for dd in range(NV):
                var b = dof_body[dd]
                # Does DOF dd affect body i (dd's body == i or an ancestor)?
                var affects = b == i
                if not affects:
                    var current = i
                    while current > 0:
                        var par = Int(
                            mf.bodies.data[current * MODEL_BODY_SIZE + BODY_IDX_PARENT]
                        )
                        if par == b:
                            affects = True
                            break
                        current = par
                if not affects:
                    continue

                var ang_x = sc.cdof.data[dd * 6 + 0]
                var ang_y = sc.cdof.data[dd * 6 + 1]
                var ang_z = sc.cdof.data[dd * 6 + 2]
                var lin_x = sc.cdof.data[dd * 6 + 3]
                var lin_y = sc.cdof.data[dd * 6 + 4]
                var lin_z = sc.cdof.data[dd * 6 + 5]

                # Fields cdof.lin is referenced at subtree_com[rootid[b]]
                # (MuJoCo convention, cdof.mojo:108), so shift the CoM
                # Jacobian from THAT point — not the body CoM.
                var rb = Int(
                    mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_ROOTID]
                )
                var dx = ti_x - d.subtree_com.data[rb * 3 + 0]
                var dy = ti_y - d.subtree_com.data[rb * 3 + 1]
                var dz = ti_z - d.subtree_com.data[rb * 3 + 2]

                if k == 0:
                    J_row[dd] = lin_x + ang_y * dz - ang_z * dy
                elif k == 1:
                    J_row[dd] = lin_y + ang_z * dx - ang_x * dz
                elif k == 2:
                    J_row[dd] = lin_z + ang_x * dy - ang_y * dx
                elif k == 3:
                    J_row[dd] = ang_x
                elif k == 4:
                    J_row[dd] = ang_y
                else:
                    J_row[dd] = ang_z

            # A[k,k] = J_row . M^-1 . J_row via a direct LDL solve (matches
            # legacy compute_body_invweight0 arithmetic bit-for-bit).
            for q in range(NV):
                sc.fnet.data[q] = J_row[q]
            ldl_solve["cpu", DTYPE, NV, NBODY, 1](sc)
            var dot_val = Scalar[DTYPE](0)
            for q in range(NV):
                dot_val += J_row[q] * sc.qacc_ws.data[q]
            A_diag[k] = dot_val

        var tran = (A_diag[0] + A_diag[1] + A_diag[2]) / Scalar[DTYPE](3)
        var rot = (A_diag[3] + A_diag[4] + A_diag[5]) / Scalar[DTYPE](3)
        # Fallback: near-zero one uses the other (MuJoCo behavior).
        if tran < Scalar[DTYPE](1e-10) and rot > Scalar[DTYPE](1e-10):
            tran = rot
        elif rot < Scalar[DTYPE](1e-10) and tran > Scalar[DTYPE](1e-10):
            rot = tran
        mf.body_invweight0.data[2 * i] = tran
        mf.body_invweight0.data[2 * i + 1] = rot

    # ── dof_invweight0[d] = (M^-1)[d,d] via solve M x = e_d (legacy :588-601) ─
    for dd in range(NV):
        for q in range(NV):
            sc.fnet.data[q] = Scalar[DTYPE](1) if q == dd else Scalar[DTYPE](0)
        ldl_solve["cpu", DTYPE, NV, NBODY, 1](sc)
        mf.dof_invweight0.data[dd] = sc.qacc_ws.data[dd]
