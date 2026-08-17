"""One row of the contact Jacobian — moved out of `constraints/` (phase 2.0).

⚠ THIS LIVES IN `dynamics/` FOR A STRUCTURAL REASON, not a taxonomic one.
`dynamics/tendon.mojo` needs it, and while it sat in
`constraints/contact_solve.mojo` that need was the single import making
`dynamics` depend on `constraints`. Together with `constraints -> solver` (the
`qcqp`/`elliptic_layout` leaves, moved in the same step) it closed a cycle that
put {constraints, dynamics, solver} into one 22.5k-line strongly-connected
component — 136 of the sweep's 238 dim-carrying declarations with no
intermediate green state between them. That is what made §5.4's
package-at-a-time gating protocol unexecutable. See docs §11.2/§11.3.

The body is VERBATIM from `contact_solve.mojo` — a relocation, not a rewrite.
Its only dependencies were `gpu.constants` offsets and the joint-type
constants, which is why it moved cleanly.

⚠ ITS SIBLING `_angular_jacobian_row` DELIBERATELY DID NOT MOVE. Nothing in
`dynamics` imports it, so it creates no cycle, and moving code that is not in
the way would widen this step past what its gate covers.
"""

from layout import Layout, LayoutTensor

from ..joint_types import JNT_FREE, JNT_BALL
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_META_IDX_NJOINT,
    BODY_IDX_PARENT,
    BODY_IDX_ROOTID,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
)


@always_inline
def _contact_jacobian_row[
    DTYPE: DType,
    V_SIZE: Int,
    L_SUBTREE_COM: Layout,
    L_JOINTS: Layout,
    L_BODIES: Layout,
    L_MMETA: Layout,
    L_CDOF: Layout,
](
    env: Int,
    subtree_com: LayoutTensor[
        DTYPE, L_SUBTREE_COM, MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, L_JOINTS, MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, L_BODIES, MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, L_MMETA, MutAnyOrigin
    ],
    cdof: LayoutTensor[DTYPE, L_CDOF, MutAnyOrigin],
    contact_body_a: Int,
    contact_body_b: Int,
    contact_pos_x: Scalar[DTYPE],
    contact_pos_y: Scalar[DTYPE],
    contact_pos_z: Scalar[DTYPE],
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    mut J_row: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """One row of the contact Jacobian (verbatim from
    compute_contact_jacobian_row_gpu; the legacy body computed an unused
    xpos offset, dropped here).

    Bilateral: J_row[i] = J_a[i] - J_b[i] for body-body contacts.
    For ground contacts (body_b = 0, worldbody), only body_a contributes.
    """
    for i in range(V_SIZE):
        J_row[i] = 0

    var num_joints = Int(
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NJOINT])
    )

    for j_idx in range(num_joints):
        var jnt_type = Int(
            rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_TYPE])
        )
        var joint_body = Int(
            rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_BODY_ID])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_DOF_ADR])
        )

        # Check if this joint affects body_a
        var affects_a = False
        if contact_body_a == joint_body:
            affects_a = True
        else:
            var current = contact_body_a
            while current > 0:
                var current_parent = Int(
                    rebind[Scalar[DTYPE]](bodies[current, BODY_IDX_PARENT])
                )
                if current_parent == joint_body:
                    affects_a = True
                    break
                current = current_parent

        # Check if this joint affects body_b (only if body_b > 0, i.e. not ground)
        var affects_b = False
        if contact_body_b > 0:
            if contact_body_b == joint_body:
                affects_b = True
            else:
                var current_b = contact_body_b
                while current_b > 0:
                    var current_parent_b = Int(
                        rebind[Scalar[DTYPE]](
                            bodies[current_b, BODY_IDX_PARENT]
                        )
                    )
                    if current_parent_b == joint_body:
                        affects_b = True
                        break
                    current_b = current_parent_b

        if not affects_a and not affects_b:
            continue

        var num_dof = 1
        if jnt_type == JNT_FREE:
            num_dof = 6
        elif jnt_type == JNT_BALL:
            num_dof = 3

        # Reference = subtree_com[rootid] (must match cdof computation)
        var jb_rootid = Int(
            rebind[Scalar[DTYPE]](bodies[joint_body, BODY_IDX_ROOTID])
        )
        var b_x = rebind[Scalar[DTYPE]](
            subtree_com[env, jb_rootid * 3 + 0]
        )
        var b_y = rebind[Scalar[DTYPE]](
            subtree_com[env, jb_rootid * 3 + 1]
        )
        var b_z = rebind[Scalar[DTYPE]](
            subtree_com[env, jb_rootid * 3 + 2]
        )

        var rx = contact_pos_x - b_x
        var ry = contact_pos_y - b_y
        var rz = contact_pos_z - b_z

        for d in range(num_dof):
            var dof_idx = dof_adr + d

            var ang_x = cdof[env, dof_idx * 6 + 0]
            var ang_y = cdof[env, dof_idx * 6 + 1]
            var ang_z = cdof[env, dof_idx * 6 + 2]
            var lin_x = cdof[env, dof_idx * 6 + 3]
            var lin_y = cdof[env, dof_idx * 6 + 4]
            var lin_z = cdof[env, dof_idx * 6 + 5]

            # J_trans = cdof_lin + cdof_ang x r
            var cross_x = ang_y * rz - ang_z * ry
            var cross_y = ang_z * rx - ang_x * rz
            var cross_z = ang_x * ry - ang_y * rx

            var jt_x = lin_x + cross_x
            var jt_y = lin_y + cross_y
            var jt_z = lin_z + cross_z

            var val = jt_x * dir_x + jt_y * dir_y + jt_z * dir_z

            # Body A contributes positively, body B negatively
            if affects_a:
                J_row[dof_idx] += rebind[Scalar[DTYPE]](val)
            if affects_b:
                J_row[dof_idx] -= rebind[Scalar[DTYPE]](val)
