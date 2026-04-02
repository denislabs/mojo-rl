"""Constraint builder for physics3d solvers.

Extracts all constraint setup (contact normals, friction, joint limits,
equality constraints) into a single builder function. Solvers become pure
iterative algorithms consuming pre-built ConstraintData.

This consolidates code previously duplicated across PGS, CG, and Newton solvers.
"""

from std.math import sqrt, pow
from ..types import (
    Model,
    Data,
    EQ_CONNECT,
    EQ_WELD,
    _max_one,
    ConeType,
    TendonDef,
)
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..dynamics.jacobian import compute_contact_jacobian_row
from ..kinematics.quat_math import quat_mul, quat_conjugate, quat_rotate
from .constraint_data import (
    ConstraintRow,
    ConstraintData,
    CNSTR_NORMAL,
    CNSTR_FRICTION_T1,
    CNSTR_FRICTION_T2,
    CNSTR_LIMIT,
    CNSTR_FRICTION_TORSION,
    CNSTR_FRICTION_ROLL1,
    CNSTR_FRICTION_ROLL2,
    CNSTR_EQUALITY_CONNECT,
    CNSTR_EQUALITY_WELD,
    CNSTR_EQUALITY_TENDON,
)


def _compute_aref[
    DTYPE: DType,
](
    penetration: Scalar[DTYPE],
    si_dmin: Scalar[DTYPE],
    si_dmax: Scalar[DTYPE],
    si_width: Scalar[DTYPE],
    si_midpoint: Scalar[DTYPE],
    si_power: Scalar[DTYPE],
    K_spring: Scalar[DTYPE],
    B_damp: Scalar[DTYPE],
    v_n: Scalar[DTYPE],
    K: Scalar[DTYPE],
    diagApprox: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Compute acceleration-level constraint reference and inv_K.

    Returns (bias, inv_K_imp, imp) where:
    - bias = -aref (negated for solver: residual = a + bias + R*lambda)
    - inv_K_imp = 1/(K + R) where R = (1-imp)/imp * diagApprox (MuJoCo body_invweight0)
    - imp = impedance from solimp piecewise power formula

    MuJoCo formula: aref = -B*vel - K*imp*pos = K*imp*pen - B*v_n
    Our solver uses: residual = J*qacc + bias + R*lambda (= 0 at convergence)
    MuJoCo uses: R = (1-imp)/imp * diagApprox, D = imp/((1-imp)*diagApprox)
    So: bias = -aref = -K*imp*pen + B*v_n
    """
    # MuJoCo piecewise power impedance formula
    var imp: Scalar[DTYPE]
    if si_dmin == si_dmax or si_width <= Scalar[DTYPE](0):
        imp = Scalar[DTYPE](0.5) * (si_dmin + si_dmax)
    else:
        var x = penetration / si_width
        var y: Scalar[DTYPE]
        if x <= Scalar[DTYPE](0):
            y = Scalar[DTYPE](0)
        elif x >= Scalar[DTYPE](1):
            y = Scalar[DTYPE](1)
        elif si_power == Scalar[DTYPE](1):
            y = x
        elif x <= si_midpoint:
            var a = Scalar[DTYPE](1) / pow(
                si_midpoint, si_power - Scalar[DTYPE](1)
            )
            y = a * pow(x, si_power)
        else:
            var b = Scalar[DTYPE](1) / pow(
                Scalar[DTYPE](1) - si_midpoint, si_power - Scalar[DTYPE](1)
            )
            y = Scalar[DTYPE](1) - b * pow(Scalar[DTYPE](1) - x, si_power)
        imp = si_dmin + y * (si_dmax - si_dmin)
    # MuJoCo uses mjMINIMP ~1e-6 (only prevents division by zero)
    if imp < Scalar[DTYPE](1e-6):
        imp = Scalar[DTYPE](1e-6)
    # MuJoCo: aref = -B*vel - K*I*pos, where vel>0=separating, pos<0=penetrating
    # Our convention: v_n = J*qvel where positive = separating for ground contacts
    # (normal points from ground toward body, J = J_body, vel > 0 = moving up = separating)
    # bias = -aref = B*v_n + K*I*pen → bias = -K*imp*pen + B*v_n
    var bias = -K_spring * imp * penetration + B_damp * v_n
    # MuJoCo: R = (1-imp)/imp * diagApprox, inv_K_imp = 1/(K + R)
    var R = (Scalar[DTYPE](1.0) - imp) / imp * diagApprox
    var inv_K_imp = Scalar[DTYPE](1.0) / (K + R)
    return (bias, inv_K_imp, imp)


def _compute_angular_jacobian_row[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    V_SIZE: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
    NSITE: Int = 0,
](
    model: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
        NSITE,
    ],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
    cdof: List[Scalar[DTYPE]],
    contact_body_a: Int,
    contact_body_b: Int,
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    mut J_row: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Compute angular-only Jacobian row for torsional/rolling friction.

    Maps joint velocities to angular velocity along a given direction.
    Unlike the full contact Jacobian, this uses only the angular part of cdof
    (no cross product with contact position offset).

    J[dof] = cdof_ang[dof] . dir  (bilateral: body_a - body_b)
    """
    for i in range(NV):
        J_row[i] = Scalar[DTYPE](0)

    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr

        var affects_a = _joint_affects_body(model, j, contact_body_a)
        var affects_b = (contact_body_b > 0) and _joint_affects_body(
            model, j, contact_body_b
        )

        if not affects_a and not affects_b:
            continue

        var num_dof = 1
        if joint.jnt_type == JNT_FREE:
            num_dof = 6
        elif joint.jnt_type == JNT_BALL:
            num_dof = 3

        for d in range(num_dof):
            var dof_idx = dof_adr + d
            var ang_x = cdof[dof_idx * 6 + 0]
            var ang_y = cdof[dof_idx * 6 + 1]
            var ang_z = cdof[dof_idx * 6 + 2]

            var val = ang_x * dir_x + ang_y * dir_y + ang_z * dir_z

            if affects_a:
                J_row[dof_idx] = J_row[dof_idx] + val
            if affects_b:
                J_row[dof_idx] = J_row[dof_idx] - val


def _joint_affects_body[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
    NSITE: Int = 0,
](
    model: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
        NSITE,
    ],
    joint_idx: Int,
    body_idx: Int,
) -> Bool:
    """Check if a joint affects a body (body is the joint's body or a descendant).
    """
    var joint_body = model.joints[joint_idx].body_id
    if body_idx == joint_body:
        return True
    var current = body_idx
    while current > 0:
        if model.body_parent[current] == joint_body:
            return True
        current = model.body_parent[current]
    return False


def _compute_weld_jacobian_row[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    V_SIZE: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
    NSITE: Int = 0,
](
    model: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
        NSITE,
    ],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
    cdof: List[Scalar[DTYPE]],
    body_a: Int,
    body_b: Int,
    pos_a_x: Scalar[DTYPE],
    pos_a_y: Scalar[DTYPE],
    pos_a_z: Scalar[DTYPE],
    pos_b_x: Scalar[DTYPE],
    pos_b_y: Scalar[DTYPE],
    pos_b_z: Scalar[DTYPE],
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    mut J_row: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Compute weld/connect Jacobian: J = J_a(at pos_a) - J_b(at pos_b).

    Unlike compute_contact_jacobian_row which uses a single contact point,
    this computes each body's Jacobian at its OWN anchor position.
    MuJoCo: mj_jacDifPair(body_b, body_a, pos_b, pos_a, jac_b, jac_a, jacdif)
    """
    for i in range(V_SIZE):
        J_row[i] = Scalar[DTYPE](0)

    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr

        var affects_a = _joint_affects_body(model, j, body_a)
        var affects_b = (body_b > 0) and _joint_affects_body(model, j, body_b)

        if not affects_a and not affects_b:
            continue

        var num_dof = 1
        if joint.jnt_type == JNT_FREE:
            num_dof = 6
        elif joint.jnt_type == JNT_BALL:
            num_dof = 3

        # Reference point for cdof cross product
        var ref_body = joint.body_id
        var ref_x: Scalar[DTYPE]
        var ref_y: Scalar[DTYPE]
        var ref_z: Scalar[DTYPE]
        if data.has_subtree_com:
            var root = model.body_rootid[ref_body]
            ref_x = data.subtree_com[root * 3 + 0]
            ref_y = data.subtree_com[root * 3 + 1]
            ref_z = data.subtree_com[root * 3 + 2]
        else:
            ref_x = data.xipos[ref_body * 3 + 0]
            ref_y = data.xipos[ref_body * 3 + 1]
            ref_z = data.xipos[ref_body * 3 + 2]

        for d in range(num_dof):
            var dof_idx = dof_adr + d
            var ang_x = cdof[dof_idx * 6 + 0]
            var ang_y = cdof[dof_idx * 6 + 1]
            var ang_z = cdof[dof_idx * 6 + 2]
            var lin_x = cdof[dof_idx * 6 + 3]
            var lin_y = cdof[dof_idx * 6 + 4]
            var lin_z = cdof[dof_idx * 6 + 5]

            if affects_a:
                # Jacobian for body_a at pos_a
                var ra_x = pos_a_x - ref_x
                var ra_y = pos_a_y - ref_y
                var ra_z = pos_a_z - ref_z
                var cross_x = ang_y * ra_z - ang_z * ra_y
                var cross_y = ang_z * ra_x - ang_x * ra_z
                var cross_z = ang_x * ra_y - ang_y * ra_x
                var val = (lin_x + cross_x) * dir_x + (lin_y + cross_y) * dir_y + (lin_z + cross_z) * dir_z
                J_row[dof_idx] = J_row[dof_idx] + val

            if affects_b:
                # Jacobian for body_b at pos_b (separate point!)
                var rb_x = pos_b_x - ref_x
                var rb_y = pos_b_y - ref_y
                var rb_z = pos_b_z - ref_z
                var cross_x = ang_y * rb_z - ang_z * rb_y
                var cross_y = ang_z * rb_x - ang_x * rb_z
                var cross_z = ang_x * rb_y - ang_y * rb_x
                var val = (lin_x + cross_x) * dir_x + (lin_y + cross_y) * dir_y + (lin_z + cross_z) * dir_z
                J_row[dof_idx] = J_row[dof_idx] - val


def build_constraints[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    MAX_ROWS: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
    NSITE: Int = 0,
](
    model: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
        NSITE,
    ],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
    cdof: List[Scalar[DTYPE]],
    M_inv: List[Scalar[DTYPE]],
    dt: Scalar[DTYPE],
    mut constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
):
    """Build all constraints from contacts and joint limits.

    Populates constraints with:
    1. Contact normal rows (with Jacobian, K, aref bias, warm-start)
    2. Contact friction rows (t1, t2 per active normal, with Coulomb coupling)
    3. Joint limit rows (for active HINGE/SLIDE limits)

    After this call, solvers can iterate over constraints.rows without
    knowing anything about contacts, Jacobians, or impedance.

    Note: qvel is the CURRENT velocity (not predicted), used for damping
    in the acceleration-level aref computation.
    """
    var num_contacts = data.num_contacts
    var nc = num_contacts
    if nc > MAX_CONTACTS:
        nc = MAX_CONTACTS

    comptime MC = _max_one[MAX_CONTACTS]()

    # Read solref/solimp for contacts
    var sr_tc = model.solref_contact[0]
    var sr_dr = model.solref_contact[1]
    var si_dmin = model.solimp_contact[0]
    var si_dmax = model.solimp_contact[1]
    var si_width = model.solimp_contact[2]
    var si_midpoint = model.solimp_contact[3]
    var si_power = model.solimp_contact[4]
    if si_width < Scalar[DTYPE](1e-6):
        si_width = Scalar[DTYPE](1e-6)
    if si_dmax < Scalar[DTYPE](1e-4):
        si_dmax = Scalar[DTYPE](1e-4)
    # Acceleration-level spring/damper coefficients (dt-independent)
    # MuJoCo: K = 1/(dmax² * tc² * dr²), B = 2/(dmax * tc)
    var K_spring = Scalar[DTYPE](1.0) / (si_dmax * si_dmax * sr_tc * sr_tc * sr_dr * sr_dr)
    var B_damp = Scalar[DTYPE](2.0) / (si_dmax * sr_tc)
    var default_friction = Scalar[DTYPE](
        0.5
    )  # Fallback (contacts always have friction from geom specs)

    comptime V_SIZE = _max_one[NV]()
    comptime CDOF_SIZE = _max_one[NV * 6]()
    var row_idx = 0
    var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

    # =========================================================================
    # Phase 1 + 2: Contact constraints (branching on cone_type)
    # =========================================================================
    # Track which contacts are active (penetrating) for friction phase
    var contact_active = InlineArray[Int, MC](uninitialized=True)
    var contact_normal_row = InlineArray[Int, MC](uninitialized=True)
    for i in range(MC):
        contact_active[i] = 0
        contact_normal_row[i] = -1

    comptime if CONE_TYPE == ConeType.PYRAMIDAL:
        # =================================================================
        # PYRAMIDAL CONE: Build edge rows (all >= 0 constraints)
        # J_edge± = J_normal ± mu_k * J_tangent_k
        # No separate normal or friction rows — edges encode both.
        # =================================================================
        var J_n = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var J_t = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

        for c in range(nc):
            var contact = data.contacts[c]
            if contact.dist >= contact.includemargin:
                continue

            var condim = contact.condim
            var nx = contact.normal_x
            var ny = contact.normal_y
            var nz = contact.normal_z

            # Compute normal Jacobian
            compute_contact_jacobian_row(
                model,
                data,
                cdof,
                contact.body_a,
                contact.body_b,
                contact.pos_x,
                contact.pos_y,
                contact.pos_z,
                nx,
                ny,
                nz,
                J_n,
            )

            # Compute aref using normal Jacobian
            var k_n: Scalar[DTYPE] = 0
            var v_n: Scalar[DTYPE] = 0
            for i in range(NV):
                var mi_j_sum: Scalar[DTYPE] = 0
                for j_idx in range(NV):
                    mi_j_sum += M_inv[i * NV + j_idx] * J_n[j_idx]
                k_n += J_n[i] * mi_j_sum
                v_n += J_n[i] * data.qvel[i]
            if k_n < Scalar[DTYPE](1e-10):
                k_n = Scalar[DTYPE](1e-10)

            # Compute diagApprox from body_invweight0 (MuJoCo-style)
            var diag_n: Scalar[DTYPE] = 0
            if contact.body_a > 0 and contact.body_a < NBODY:
                diag_n += model.body_invweight0[contact.body_a * 2]
            if contact.body_b > 0 and contact.body_b < NBODY:
                diag_n += model.body_invweight0[contact.body_b * 2]
            if diag_n < Scalar[DTYPE](1e-10):
                diag_n = k_n

            # MuJoCo: aref uses (pos - includemargin), penetration = -(dist - margin)
            var penetration = -(contact.dist - contact.includemargin)
            var imp_result = _compute_aref[DTYPE](
                penetration,
                si_dmin,
                si_dmax,
                si_width,
                si_midpoint,
                si_power,
                K_spring,
                B_damp,
                v_n,
                k_n,
                diag_n,
            )
            var bias_n = imp_result[0]
            var inv_K_imp_n = imp_result[1]
            var imp_n_val = imp_result[2]

            # For condim=1 (frictionless), just add a normal row
            if condim <= 1:
                if row_idx >= MAX_ROWS:
                    break
                for i in range(NV):
                    constraints.J[row_idx * NV + i] = J_n[i]
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * J_n[j_idx]
                    constraints.MinvJT[row_idx * NV + i] = mi_j_sum
                constraints.rows[row_idx].K = k_n
                constraints.rows[row_idx].bias = bias_n
                constraints.rows[row_idx].inv_K_imp = inv_K_imp_n
                constraints.rows[row_idx].lo = Scalar[DTYPE](0)
                constraints.rows[row_idx].hi = Scalar[DTYPE](1e20)
                constraints.rows[row_idx].lambda_val = contact.force_n
                constraints.rows[row_idx].constraint_type = CNSTR_NORMAL
                constraints.rows[row_idx].friction_parent = -1
                constraints.rows[row_idx].friction_coef = Scalar[DTYPE](0)
                constraints.rows[row_idx].source_contact_idx = c
                constraints.rows[row_idx].source_dof = -1
                constraints.rows[row_idx].limit_sign = Scalar[DTYPE](0)
                contact_active[c] = 1
                row_idx += 1
                continue

            # Friction coefficient
            var mu_slide = contact.friction
            if mu_slide <= Scalar[DTYPE](0):
                mu_slide = default_friction

            # Tangent basis (MuJoCo mju_makeFrame with capsule axis hint)
            var hint_x = contact.frame_t1_x
            var hint_y = contact.frame_t1_y
            var hint_z = contact.frame_t1_z
            var hint_len_sq = (
                hint_x * hint_x + hint_y * hint_y + hint_z * hint_z
            )

            # If no hint (non-capsule), use MuJoCo mju_makeFrame default:
            # pick the axis with the smallest absolute dot product with normal.
            if hint_len_sq < Scalar[DTYPE](0.25):
                var abs_nx = abs(nx)
                var abs_ny = abs(ny)
                var abs_nz = abs(nz)
                if abs_nx <= abs_ny and abs_nx <= abs_nz:
                    hint_x = Scalar[DTYPE](1)
                    hint_y = Scalar[DTYPE](0)
                    hint_z = Scalar[DTYPE](0)
                elif abs_ny <= abs_nz:
                    hint_x = Scalar[DTYPE](0)
                    hint_y = Scalar[DTYPE](1)
                    hint_z = Scalar[DTYPE](0)
                else:
                    hint_x = Scalar[DTYPE](0)
                    hint_y = Scalar[DTYPE](0)
                    hint_z = Scalar[DTYPE](1)

            # Gram-Schmidt: orthogonalize hint against normal
            var dot_nh = nx * hint_x + ny * hint_y + nz * hint_z
            var t1_x = hint_x - dot_nh * nx
            var t1_y = hint_y - dot_nh * ny
            var t1_z = hint_z - dot_nh * nz
            var t1_mag = sqrt(t1_x * t1_x + t1_y * t1_y + t1_z * t1_z)
            if t1_mag > Scalar[DTYPE](1e-10):
                t1_x /= t1_mag
                t1_y /= t1_mag
                t1_z /= t1_mag

            # T2 = cross(normal, T1)
            var t2_x = ny * t1_z - nz * t1_y
            var t2_y = nz * t1_x - nx * t1_z
            var t2_z = nx * t1_y - ny * t1_x

            # Helper: build one pyramid edge row (J_edge = J_n + sign * mu * J_t)
            # Collect tangent directions and their mu values
            # condim=3: 2 tangent dirs → 4 edges
            # condim=4: 3 tangent dirs → 6 edges
            # condim=6: 5 tangent dirs → 10 edges
            var num_tangent_dirs = 0
            if condim >= 3:
                num_tangent_dirs = 2
            if condim >= 4:
                num_tangent_dirs = 3
            if condim >= 6:
                num_tangent_dirs = 5

            # Build edges for each tangent direction
            for td in range(num_tangent_dirs):
                if row_idx + 1 >= MAX_ROWS:
                    break

                var mu_td = mu_slide
                # Compute the tangent Jacobian for this direction
                if td == 0:
                    # Tangent 1 (slide)
                    compute_contact_jacobian_row(
                        model,
                        data,
                        cdof,
                        contact.body_a,
                        contact.body_b,
                        contact.pos_x,
                        contact.pos_y,
                        contact.pos_z,
                        t1_x,
                        t1_y,
                        t1_z,
                        J_t,
                    )
                elif td == 1:
                    # Tangent 2 (slide)
                    compute_contact_jacobian_row(
                        model,
                        data,
                        cdof,
                        contact.body_a,
                        contact.body_b,
                        contact.pos_x,
                        contact.pos_y,
                        contact.pos_z,
                        t2_x,
                        t2_y,
                        t2_z,
                        J_t,
                    )
                elif td == 2:
                    # Torsion (angular along normal)
                    mu_td = contact.friction_spin

                    _compute_angular_jacobian_row[
                        DTYPE,
                        NQ,
                        NV,
                        NBODY,
                        NJOINT,
                        MAX_CONTACTS,
                        V_SIZE,
                        NGEOM,
                        MAX_EQUALITY,
                        CONE_TYPE,
                        MAX_TENDON,
                        NSITE,
                    ](
                        model,
                        data,
                        cdof,
                        contact.body_a,
                        contact.body_b,
                        nx,
                        ny,
                        nz,
                        J_t,
                    )
                elif td == 3:
                    # Roll 1 (angular along tangent 1)
                    mu_td = contact.friction_roll

                    _compute_angular_jacobian_row[
                        DTYPE,
                        NQ,
                        NV,
                        NBODY,
                        NJOINT,
                        MAX_CONTACTS,
                        V_SIZE,
                        NGEOM,
                        MAX_EQUALITY,
                        CONE_TYPE,
                        MAX_TENDON,
                        NSITE,
                    ](
                        model,
                        data,
                        cdof,
                        contact.body_a,
                        contact.body_b,
                        t1_x,
                        t1_y,
                        t1_z,
                        J_t,
                    )
                elif td == 4:
                    # Roll 2 (angular along tangent 2)
                    mu_td = contact.friction_roll

                    _compute_angular_jacobian_row[
                        DTYPE,
                        NQ,
                        NV,
                        NBODY,
                        NJOINT,
                        MAX_CONTACTS,
                        V_SIZE,
                        NGEOM,
                        MAX_EQUALITY,
                        CONE_TYPE,
                        MAX_TENDON,
                        NSITE,
                    ](
                        model,
                        data,
                        cdof,
                        contact.body_a,
                        contact.body_b,
                        t2_x,
                        t2_y,
                        t2_z,
                        J_t,
                    )

                if mu_td <= Scalar[DTYPE](1e-12):
                    continue

                # Extract impedance from normal (shared by all edges)
                var imp_n = imp_n_val

                # MuJoCo pyramidal diagApprox (engine_core_constraint.c:1203-1205):
                #   dA = tran + fri^2 * (j < 2 ? tran : rot)
                # For condim=3, td=0,1 are sliding: dA = tran + mu^2*tran
                # For condim=4, td=2 is torsion: dA = tran + mu_spin^2*rot
                # For condim=6, td=3,4 are rolling: dA = tran + mu_roll^2*rot
                # We use tran for both (rot weight not stored separately)
                var diag_edge = diag_n + mu_td * mu_td * diag_n

                # MuJoCo pyramidal R adjustment (engine_core_constraint.c:1484-1493):
                #   R_initial = (1-imp)/imp * dA
                #   R[i+1] = R[i] / impratio  (impratio=1 → same)
                #   con->mu = friction[0] * sqrt(1/impratio)  (=friction[0] for impratio=1)
                #   Rpy = 2 * con->mu^2 * R_initial
                # For impratio=1: Rpy = 2 * mu^2 * (1-imp)/imp * dA
                var R_initial = (Scalar[DTYPE](1.0) - imp_n) / imp_n * diag_edge
                var R_edge = (
                    Scalar[DTYPE](2.0) * mu_slide * mu_slide * R_initial
                )

                # Build edge+ and edge- rows
                for sign_idx in range(2):
                    if row_idx >= MAX_ROWS:
                        break
                    var sign = Scalar[DTYPE](1.0) if sign_idx == 0 else Scalar[
                        DTYPE
                    ](-1.0)

                    # J_edge = J_n + sign * mu * J_t
                    var k_edge: Scalar[DTYPE] = 0
                    for i in range(NV):
                        var j_edge = J_n[i] + sign * mu_td * J_t[i]
                        constraints.J[row_idx * NV + i] = j_edge
                        var mi_j_sum: Scalar[DTYPE] = 0
                        for j_idx in range(NV):
                            var j_edge_j = (
                                J_n[j_idx] + sign * mu_td * J_t[j_idx]
                            )
                            mi_j_sum += M_inv[i * NV + j_idx] * j_edge_j
                        constraints.MinvJT[row_idx * NV + i] = mi_j_sum
                        k_edge += (J_n[i] + sign * mu_td * J_t[i]) * mi_j_sum
                    if k_edge < Scalar[DTYPE](1e-10):
                        k_edge = Scalar[DTYPE](1e-10)

                    # MuJoCo edge bias: uses edge Jacobian velocity directly
                    # efc_vel = J_edge * qvel, then:
                    # efc_aref = -B*efc_vel - K*imp*(pos - margin)
                    # bias = -efc_aref (our convention)
                    # pos = con->dist for both edges, margin = includemargin (0)
                    var v_edge: Scalar[DTYPE] = 0
                    for i in range(NV):
                        v_edge += constraints.J[row_idx * NV + i] * data.qvel[i]
                    # aref = -B*v_edge - K*imp*dist  (dist = -penetration, negative)
                    # bias = -aref = B*v_edge + K*imp*dist
                    # = B*v_edge - K*imp*penetration
                    var bias_edge = (
                        B_damp * v_edge - K_spring * imp_n * penetration
                    )

                    var inv_K_edge = Scalar[DTYPE](1.0) / (k_edge + R_edge)

                    constraints.rows[row_idx].K = k_edge
                    constraints.rows[row_idx].bias = bias_edge
                    constraints.rows[row_idx].inv_K_imp = inv_K_edge
                    constraints.rows[row_idx].lo = Scalar[DTYPE](0)
                    constraints.rows[row_idx].hi = Scalar[DTYPE](1e20)

                    # Warm-start: decompose stored contact forces back into edge lambdas.
                    # Writeback accumulates: force_n = Σ λ, force_tk = mu*(λ+ - λ-)
                    # Inversion: λ_td± = (force_n/num_tangent_dirs ± force_tk/mu) / 2
                    var prev_force_tk: Scalar[DTYPE] = 0
                    if td == 0:
                        prev_force_tk = contact.force_t1
                    elif td == 1:
                        prev_force_tk = contact.force_t2
                    elif td == 2:
                        prev_force_tk = contact.force_torsion
                    elif td == 3:
                        prev_force_tk = contact.force_roll1
                    elif td == 4:
                        prev_force_tk = contact.force_roll2
                    var sign_val = Scalar[DTYPE](
                        1.0
                    ) if sign_idx == 0 else Scalar[DTYPE](-1.0)
                    var ws_mu = mu_td if mu_td > Scalar[DTYPE](
                        1e-8
                    ) else Scalar[DTYPE](1e-8)
                    var ws_lam = (
                        contact.force_n / Scalar[DTYPE](num_tangent_dirs)
                        + sign_val * prev_force_tk / ws_mu
                    )
                    ws_lam = ws_lam / Scalar[DTYPE](2)
                    if ws_lam < Scalar[DTYPE](0):
                        ws_lam = Scalar[DTYPE](0)
                    constraints.rows[row_idx].lambda_val = ws_lam
                    constraints.rows[
                        row_idx
                    ].constraint_type = CNSTR_PYRAMID_EDGE
                    constraints.rows[row_idx].friction_parent = -1
                    constraints.rows[row_idx].friction_coef = mu_td
                    constraints.rows[row_idx].source_contact_idx = c
                    constraints.rows[row_idx].source_dof = td * 2 + sign_idx
                    constraints.rows[row_idx].limit_sign = Scalar[DTYPE](0)
                    row_idx += 1

            contact_active[c] = 1

        constraints.num_normals = row_idx
        constraints.num_friction = 0

    else:
        # =================================================================
        # ELLIPTIC CONE: Separate normal + friction rows (default)
        # =================================================================
        for c in range(nc):
            var contact = data.contacts[c]

            if contact.dist >= contact.includemargin:
                continue

            if row_idx >= MAX_ROWS:
                break

            # Compute normal Jacobian row
            compute_contact_jacobian_row(
                model,
                data,
                cdof,
                contact.body_a,
                contact.body_b,
                contact.pos_x,
                contact.pos_y,
                contact.pos_z,
                contact.normal_x,
                contact.normal_y,
                contact.normal_z,
                J_row,
            )

            # Store J and compute K = J @ M_inv @ J^T, MinvJT
            var k: Scalar[DTYPE] = 0
            var v_n: Scalar[DTYPE] = 0
            for i in range(NV):
                constraints.J[row_idx * NV + i] = J_row[i]
                var mi_j_sum: Scalar[DTYPE] = 0
                for j_idx in range(NV):
                    mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
                constraints.MinvJT[row_idx * NV + i] = mi_j_sum
                k += J_row[i] * mi_j_sum
                v_n += J_row[i] * data.qvel[i]

            if k < Scalar[DTYPE](1e-10):
                k = Scalar[DTYPE](1e-10)

            # Compute diagApprox from body_invweight0 (MuJoCo-style)
            var diag_n: Scalar[DTYPE] = 0
            if contact.body_a > 0 and contact.body_a < NBODY:
                diag_n += model.body_invweight0[contact.body_a * 2]
            if contact.body_b > 0 and contact.body_b < NBODY:
                diag_n += model.body_invweight0[contact.body_b * 2]
            if diag_n < Scalar[DTYPE](1e-10):
                diag_n = k  # Fallback to exact K if no invweight0

            # MuJoCo: aref uses (pos - includemargin), penetration = -(dist - margin)
            var penetration = -(contact.dist - contact.includemargin)
            var imp_result = _compute_aref[DTYPE](
                penetration,
                si_dmin,
                si_dmax,
                si_width,
                si_midpoint,
                si_power,
                K_spring,
                B_damp,
                v_n,
                k,
                diag_n,
            )

            var friction_coef = contact.friction
            if friction_coef <= Scalar[DTYPE](0):
                friction_coef = default_friction

            constraints.rows[row_idx].K = k
            constraints.rows[row_idx].bias = imp_result[0]
            constraints.rows[row_idx].inv_K_imp = imp_result[1]
            constraints.rows[row_idx].lo = Scalar[DTYPE](0)
            constraints.rows[row_idx].hi = Scalar[DTYPE](1e20)
            constraints.rows[row_idx].lambda_val = contact.force_n
            constraints.rows[row_idx].constraint_type = CNSTR_NORMAL
            constraints.rows[row_idx].friction_parent = -1
            constraints.rows[row_idx].friction_coef = friction_coef
            constraints.rows[row_idx].source_contact_idx = c
            constraints.rows[row_idx].source_dof = -1
            constraints.rows[row_idx].limit_sign = Scalar[DTYPE](0)
            constraints.rows[row_idx].diagApprox = diag_n

            contact_active[c] = 1
            contact_normal_row[c] = row_idx
            row_idx += 1

        constraints.num_normals = row_idx

    # =========================================================================
    # Phase 2: Contact friction constraints (condim-aware, elliptic only)
    # condim=1: no friction rows
    # condim=3: t1 + t2 (2 rows, slide friction)
    # condim=4: t1 + t2 + torsion (3 rows)
    # condim=6: t1 + t2 + torsion + roll1 + roll2 (5 rows)
    # Pyramidal cone encodes friction in edge rows (Phase 1), so skip.
    # =========================================================================
    var friction_start = row_idx

    # Skip friction rows for pyramidal cone (friction encoded in edge rows)
    var nc_friction = nc if CONE_TYPE != ConeType.PYRAMIDAL else 0

    for c in range(nc_friction):
        if contact_active[c] == 0:
            continue

        var contact = data.contacts[c]
        var normal_row = contact_normal_row[c]
        var condim = contact.condim

        # condim=1: frictionless, skip
        if condim <= 1:
            continue

        # Per-contact slide friction
        var friction_coef = contact.friction
        if friction_coef <= Scalar[DTYPE](0):
            friction_coef = default_friction
        var nx = contact.normal_x
        var ny = contact.normal_y
        var nz = contact.normal_z

        # Compute tangent basis (MuJoCo mju_makeFrame with capsule axis hint)
        var hint_x = contact.frame_t1_x
        var hint_y = contact.frame_t1_y
        var hint_z = contact.frame_t1_z
        var hint_len_sq = hint_x * hint_x + hint_y * hint_y + hint_z * hint_z

        # If no hint (non-capsule), use MuJoCo mju_makeFrame default:
        # pick the axis with the smallest absolute dot product with normal.
        if hint_len_sq < Scalar[DTYPE](0.25):
            var abs_nx = abs(nx)
            var abs_ny = abs(ny)
            var abs_nz = abs(nz)
            if abs_nx <= abs_ny and abs_nx <= abs_nz:
                hint_x = Scalar[DTYPE](1)
                hint_y = Scalar[DTYPE](0)
                hint_z = Scalar[DTYPE](0)
            elif abs_ny <= abs_nz:
                hint_x = Scalar[DTYPE](0)
                hint_y = Scalar[DTYPE](1)
                hint_z = Scalar[DTYPE](0)
            else:
                hint_x = Scalar[DTYPE](0)
                hint_y = Scalar[DTYPE](0)
                hint_z = Scalar[DTYPE](1)

        # Gram-Schmidt: orthogonalize hint against normal
        var dot_nh = nx * hint_x + ny * hint_y + nz * hint_z
        var t1_x = hint_x - dot_nh * nx
        var t1_y = hint_y - dot_nh * ny
        var t1_z = hint_z - dot_nh * nz
        var t1_mag = sqrt(t1_x * t1_x + t1_y * t1_y + t1_z * t1_z)
        if t1_mag < Scalar[DTYPE](1e-10):
            # Hint is parallel to normal (e.g. vertical capsule on ground floor).
            # Fall back using MuJoCo mju_makeFrame: pick least-aligned basis axis.
            var abs_nx = abs(nx)
            var abs_ny = abs(ny)
            var abs_nz = abs(nz)
            if abs_nx <= abs_ny and abs_nx <= abs_nz:
                hint_x = Scalar[DTYPE](1)
                hint_y = Scalar[DTYPE](0)
                hint_z = Scalar[DTYPE](0)
            elif abs_ny <= abs_nz:
                hint_x = Scalar[DTYPE](0)
                hint_y = Scalar[DTYPE](1)
                hint_z = Scalar[DTYPE](0)
            else:
                hint_x = Scalar[DTYPE](0)
                hint_y = Scalar[DTYPE](0)
                hint_z = Scalar[DTYPE](1)
            dot_nh = nx * hint_x + ny * hint_y + nz * hint_z
            t1_x = hint_x - dot_nh * nx
            t1_y = hint_y - dot_nh * ny
            t1_z = hint_z - dot_nh * nz
            t1_mag = sqrt(t1_x * t1_x + t1_y * t1_y + t1_z * t1_z)
        if t1_mag > Scalar[DTYPE](1e-10):
            t1_x = t1_x / t1_mag
            t1_y = t1_y / t1_mag
            t1_z = t1_z / t1_mag

        # T2 = cross(normal, T1)
        var t2_x = ny * t1_z - nz * t1_y
        var t2_y = nz * t1_x - nx * t1_z
        var t2_z = nx * t1_y - ny * t1_x

        # --- Tangent 1 (slide) ---
        if row_idx + 1 >= MAX_ROWS:
            break

        compute_contact_jacobian_row(
            model,
            data,
            cdof,
            contact.body_a,
            contact.body_b,
            contact.pos_x,
            contact.pos_y,
            contact.pos_z,
            t1_x,
            t1_y,
            t1_z,
            J_row,
        )

        var k1: Scalar[DTYPE] = 0
        var v_t1: Scalar[DTYPE] = 0
        for i in range(NV):
            constraints.J[row_idx * NV + i] = J_row[i]
            var mi_j_sum: Scalar[DTYPE] = 0
            for j_idx in range(NV):
                mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
            constraints.MinvJT[row_idx * NV + i] = mi_j_sum
            k1 += J_row[i] * mi_j_sum
            v_t1 += J_row[i] * data.qvel[i]
        if k1 < Scalar[DTYPE](1e-10):
            k1 = Scalar[DTYPE](1e-10)

        # Compute friction regularizer from parent normal's diagApprox
        # R_n = 1/inv_K_imp_n - K_n (exact regularizer from stored values)
        var R_n = (
            Scalar[DTYPE](1.0) / constraints.rows[normal_row].inv_K_imp
            - constraints.rows[normal_row].K
        )
        if R_n < Scalar[DTYPE](1e-12):
            R_n = Scalar[DTYPE](1e-12)

        var R_f1 = R_n / model.impratio
        var inv_K_imp_f1 = Scalar[DTYPE](1.0) / (k1 + R_f1)

        # Friction velocity damping bias (MuJoCo-style):
        # aref_friction = B_damp * v_tangential → bias = -aref
        var bias_f1 = B_damp * v_t1

        constraints.rows[row_idx].K = k1
        constraints.rows[row_idx].bias = bias_f1
        constraints.rows[row_idx].inv_K_imp = inv_K_imp_f1
        constraints.rows[row_idx].lo = Scalar[DTYPE](-1e20)
        constraints.rows[row_idx].hi = Scalar[DTYPE](1e20)
        constraints.rows[row_idx].lambda_val = contact.force_t1
        constraints.rows[row_idx].constraint_type = CNSTR_FRICTION_T1
        constraints.rows[row_idx].friction_parent = normal_row
        constraints.rows[row_idx].friction_coef = friction_coef
        constraints.rows[row_idx].source_contact_idx = c
        constraints.rows[row_idx].source_dof = -1
        constraints.rows[row_idx].limit_sign = Scalar[DTYPE](0)
        row_idx += 1

        # --- Tangent 2 (slide) ---
        compute_contact_jacobian_row(
            model,
            data,
            cdof,
            contact.body_a,
            contact.body_b,
            contact.pos_x,
            contact.pos_y,
            contact.pos_z,
            t2_x,
            t2_y,
            t2_z,
            J_row,
        )

        var k2: Scalar[DTYPE] = 0
        var v_t2: Scalar[DTYPE] = 0
        for i in range(NV):
            constraints.J[row_idx * NV + i] = J_row[i]
            var mi_j_sum: Scalar[DTYPE] = 0
            for j_idx in range(NV):
                mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
            constraints.MinvJT[row_idx * NV + i] = mi_j_sum
            k2 += J_row[i] * mi_j_sum
            v_t2 += J_row[i] * data.qvel[i]
        if k2 < Scalar[DTYPE](1e-10):
            k2 = Scalar[DTYPE](1e-10)

        var R_f2 = R_n / model.impratio
        var inv_K_imp_f2 = Scalar[DTYPE](1.0) / (k2 + R_f2)

        # Friction velocity damping bias
        var bias_f2 = B_damp * v_t2

        constraints.rows[row_idx].K = k2
        constraints.rows[row_idx].bias = bias_f2
        constraints.rows[row_idx].inv_K_imp = inv_K_imp_f2
        constraints.rows[row_idx].lo = Scalar[DTYPE](-1e20)
        constraints.rows[row_idx].hi = Scalar[DTYPE](1e20)
        constraints.rows[row_idx].lambda_val = contact.force_t2
        constraints.rows[row_idx].constraint_type = CNSTR_FRICTION_T2
        constraints.rows[row_idx].friction_parent = normal_row
        constraints.rows[row_idx].friction_coef = friction_coef
        constraints.rows[row_idx].source_contact_idx = c
        constraints.rows[row_idx].source_dof = -1
        constraints.rows[row_idx].limit_sign = Scalar[DTYPE](0)
        row_idx += 1

        # --- Torsional friction (condim >= 4) ---
        if condim >= 4 and row_idx < MAX_ROWS:
            _compute_angular_jacobian_row[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                V_SIZE,
                NGEOM,
                MAX_EQUALITY,
                CONE_TYPE,
                MAX_TENDON,
                NSITE,
            ](
                model,
                data,
                cdof,
                contact.body_a,
                contact.body_b,
                nx,
                ny,
                nz,
                J_row,
            )

            var k3: Scalar[DTYPE] = 0
            var v_t3: Scalar[DTYPE] = 0
            for i in range(NV):
                constraints.J[row_idx * NV + i] = J_row[i]
                var mi_j_sum: Scalar[DTYPE] = 0
                for j_idx in range(NV):
                    mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
                constraints.MinvJT[row_idx * NV + i] = mi_j_sum
                k3 += J_row[i] * mi_j_sum
                v_t3 += J_row[i] * data.qvel[i]
            if k3 < Scalar[DTYPE](1e-10):
                k3 = Scalar[DTYPE](1e-10)

            # Torsion regularizer: scale by mu_slide^2/mu_spin^2 (MuJoCo convention)
            var mu_spin = contact.friction_spin
            var R_f3 = R_n / model.impratio
            if mu_spin > Scalar[DTYPE](1e-12):
                R_f3 = (
                    R_f3 * friction_coef * friction_coef / (mu_spin * mu_spin)
                )
            var inv_K_imp_f3 = Scalar[DTYPE](1.0) / (k3 + R_f3)

            var bias_f3 = B_damp * v_t3
            constraints.rows[row_idx].K = k3
            constraints.rows[row_idx].bias = bias_f3
            constraints.rows[row_idx].inv_K_imp = inv_K_imp_f3
            constraints.rows[row_idx].lo = Scalar[DTYPE](-1e20)
            constraints.rows[row_idx].hi = Scalar[DTYPE](1e20)
            constraints.rows[row_idx].lambda_val = contact.force_torsion
            constraints.rows[row_idx].constraint_type = CNSTR_FRICTION_TORSION
            constraints.rows[row_idx].friction_parent = normal_row
            constraints.rows[row_idx].friction_coef = mu_spin
            constraints.rows[row_idx].source_contact_idx = c
            constraints.rows[row_idx].source_dof = -1
            constraints.rows[row_idx].limit_sign = Scalar[DTYPE](0)
            row_idx += 1

        # --- Rolling friction (condim >= 6) ---
        if condim >= 6:
            # Roll 1: angular velocity along tangent 1
            if row_idx < MAX_ROWS:
                _compute_angular_jacobian_row[
                    DTYPE,
                    NQ,
                    NV,
                    NBODY,
                    NJOINT,
                    MAX_CONTACTS,
                    V_SIZE,
                    NGEOM,
                    MAX_EQUALITY,
                    CONE_TYPE,
                    MAX_TENDON,
                    NSITE,
                ](
                    model,
                    data,
                    cdof,
                    contact.body_a,
                    contact.body_b,
                    t1_x,
                    t1_y,
                    t1_z,
                    J_row,
                )

                var k4: Scalar[DTYPE] = 0
                var v_t4: Scalar[DTYPE] = 0
                for i in range(NV):
                    constraints.J[row_idx * NV + i] = J_row[i]
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
                    constraints.MinvJT[row_idx * NV + i] = mi_j_sum
                    k4 += J_row[i] * mi_j_sum
                    v_t4 += J_row[i] * data.qvel[i]
                if k4 < Scalar[DTYPE](1e-10):
                    k4 = Scalar[DTYPE](1e-10)

                # Roll regularizer: scale by mu_slide^2/mu_roll^2
                var mu_roll1 = contact.friction_roll
                var R_f4 = R_n / model.impratio
                if mu_roll1 > Scalar[DTYPE](1e-12):
                    R_f4 = (
                        R_f4
                        * friction_coef
                        * friction_coef
                        / (mu_roll1 * mu_roll1)
                    )
                var inv_K_imp_f4 = Scalar[DTYPE](1.0) / (k4 + R_f4)

                var bias_f4 = B_damp * v_t4
                constraints.rows[row_idx].K = k4
                constraints.rows[row_idx].bias = bias_f4
                constraints.rows[row_idx].inv_K_imp = inv_K_imp_f4
                constraints.rows[row_idx].lo = Scalar[DTYPE](-1e20)
                constraints.rows[row_idx].hi = Scalar[DTYPE](1e20)
                constraints.rows[row_idx].lambda_val = contact.force_roll1
                constraints.rows[row_idx].constraint_type = CNSTR_FRICTION_ROLL1
                constraints.rows[row_idx].friction_parent = normal_row
                constraints.rows[row_idx].friction_coef = mu_roll1
                constraints.rows[row_idx].source_contact_idx = c
                constraints.rows[row_idx].source_dof = -1
                constraints.rows[row_idx].limit_sign = Scalar[DTYPE](0)
                row_idx += 1

            # Roll 2: angular velocity along tangent 2
            if row_idx < MAX_ROWS:
                _compute_angular_jacobian_row[
                    DTYPE,
                    NQ,
                    NV,
                    NBODY,
                    NJOINT,
                    MAX_CONTACTS,
                    V_SIZE,
                    NGEOM,
                    MAX_EQUALITY,
                    CONE_TYPE,
                    MAX_TENDON,
                    NSITE,
                ](
                    model,
                    data,
                    cdof,
                    contact.body_a,
                    contact.body_b,
                    t2_x,
                    t2_y,
                    t2_z,
                    J_row,
                )

                var k5: Scalar[DTYPE] = 0
                var v_t5: Scalar[DTYPE] = 0
                for i in range(NV):
                    constraints.J[row_idx * NV + i] = J_row[i]
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
                    constraints.MinvJT[row_idx * NV + i] = mi_j_sum
                    k5 += J_row[i] * mi_j_sum
                    v_t5 += J_row[i] * data.qvel[i]
                if k5 < Scalar[DTYPE](1e-10):
                    k5 = Scalar[DTYPE](1e-10)

                var mu_roll2 = contact.friction_roll
                var R_f5 = R_n / model.impratio
                if mu_roll2 > Scalar[DTYPE](1e-12):
                    R_f5 = (
                        R_f5
                        * friction_coef
                        * friction_coef
                        / (mu_roll2 * mu_roll2)
                    )
                var inv_K_imp_f5 = Scalar[DTYPE](1.0) / (k5 + R_f5)

                var bias_f5 = B_damp * v_t5
                constraints.rows[row_idx].K = k5
                constraints.rows[row_idx].bias = bias_f5
                constraints.rows[row_idx].inv_K_imp = inv_K_imp_f5
                constraints.rows[row_idx].lo = Scalar[DTYPE](-1e20)
                constraints.rows[row_idx].hi = Scalar[DTYPE](1e20)
                constraints.rows[row_idx].lambda_val = contact.force_roll2
                constraints.rows[row_idx].constraint_type = CNSTR_FRICTION_ROLL2
                constraints.rows[row_idx].friction_parent = normal_row
                constraints.rows[row_idx].friction_coef = mu_roll2
                constraints.rows[row_idx].source_contact_idx = c
                constraints.rows[row_idx].source_dof = -1
                constraints.rows[row_idx].limit_sign = Scalar[DTYPE](0)
                row_idx += 1

    constraints.num_friction = row_idx - friction_start

    # =========================================================================
    # Phase 3: Joint limit constraints
    # =========================================================================
    var limits_start = row_idx

    # Read model-level solref/solimp for limits (fallback for per-joint)
    var lr_tc_default = model.solref_limit[0]
    var lr_dr_default = model.solref_limit[1]
    var li_dmin_default = model.solimp_limit[0]
    var li_dmax_default = model.solimp_limit[1]
    var li_width_default = model.solimp_limit[2]
    var li_midpoint_default = model.solimp_limit[3]
    var li_power_default = model.solimp_limit[4]

    for j in range(model.num_joints):
        var joint = model.joints[j]
        if joint.jnt_type != JNT_HINGE and joint.jnt_type != JNT_SLIDE:
            continue
        var dof = joint.dof_adr
        var pos = data.qpos[joint.qpos_adr]
        var rmin = joint.range_min
        var rmax = joint.range_max
        if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
            continue

        # Per-joint solref/solimp: use if set (>0), else model-level default
        var lr_tc = model.joint_solref_limit[j * 2]
        var lr_dr = model.joint_solref_limit[j * 2 + 1]
        if lr_tc <= Scalar[DTYPE](0):
            lr_tc = lr_tc_default
        if lr_dr <= Scalar[DTYPE](0):
            lr_dr = lr_dr_default
        var li_dmin = model.joint_solimp_limit[j * 5]
        var li_dmax = model.joint_solimp_limit[j * 5 + 1]
        var li_width = model.joint_solimp_limit[j * 5 + 2]
        var li_midpoint = model.joint_solimp_limit[j * 5 + 3]
        var li_power = model.joint_solimp_limit[j * 5 + 4]
        # solimp dmin can legitimately be 0, so check if ALL three are 0
        # (unset) to fall back to defaults
        if li_dmax <= Scalar[DTYPE](0) and li_width <= Scalar[DTYPE](0):
            li_dmin = li_dmin_default
            li_dmax = li_dmax_default
            li_width = li_width_default
            li_midpoint = li_midpoint_default
            li_power = li_power_default
        if li_width < Scalar[DTYPE](1e-6):
            li_width = Scalar[DTYPE](1e-6)
        if li_dmax < Scalar[DTYPE](1e-4):
            li_dmax = Scalar[DTYPE](1e-4)

        # Acceleration-level spring/damper for limits
        # MuJoCo: K = 1/(dmax² * tc² * dr²), B = 2/(dmax * tc)
        var l_K_spring = Scalar[DTYPE](1.0) / (
            li_dmax * li_dmax * lr_tc * lr_tc * lr_dr * lr_dr
        )
        var l_B_damp = Scalar[DTYPE](2.0) / (li_dmax * lr_tc)

        # Compute diagApprox for limit: MuJoCo uses dof_invweight0[dof_adr]
        var diag_lim: Scalar[DTYPE] = 0
        if dof >= 0 and dof < NV:
            diag_lim = model.dof_invweight0[dof]

        # Lower limit: q >= range_min → constraint dist = q - range_min
        var dist_lo = pos - rmin
        if dist_lo < Scalar[DTYPE](0) and row_idx < MAX_ROWS:
            var sign = Scalar[DTYPE](1)  # J[dof] = +1
            var K_lim = M_inv[dof * NV + dof]
            if K_lim < Scalar[DTYPE](1e-10):
                K_lim = Scalar[DTYPE](1e-10)

            # Jacobian: J[dof] = sign, all others zero
            for i in range(NV):
                constraints.J[row_idx * NV + i] = Scalar[DTYPE](0)
                constraints.MinvJT[row_idx * NV + i] = (
                    M_inv[i * NV + dof] * sign
                )
            constraints.J[row_idx * NV + dof] = sign

            # Acceleration-level aref for limit
            var penetration = -dist_lo
            if penetration < Scalar[DTYPE](0):
                penetration = Scalar[DTYPE](0)
            # Use current velocity for damping (not qacc)
            var v_lim = sign * data.qvel[dof]
            var diag_lim_use = diag_lim
            if diag_lim_use < Scalar[DTYPE](1e-10):
                diag_lim_use = K_lim  # Fallback
            var imp_result = _compute_aref[DTYPE](
                penetration,
                li_dmin,
                li_dmax,
                li_width,
                li_midpoint,
                li_power,
                l_K_spring,
                l_B_damp,
                v_lim,
                K_lim,
                diag_lim_use,
            )

            constraints.rows[row_idx].K = K_lim
            constraints.rows[row_idx].bias = imp_result[0]
            constraints.rows[row_idx].inv_K_imp = imp_result[1]
            constraints.rows[row_idx].lo = Scalar[DTYPE](0)
            constraints.rows[row_idx].hi = Scalar[DTYPE](1e20)
            constraints.rows[row_idx].lambda_val = Scalar[DTYPE](0)
            constraints.rows[row_idx].constraint_type = CNSTR_LIMIT
            constraints.rows[row_idx].friction_parent = -1
            constraints.rows[row_idx].friction_coef = Scalar[DTYPE](0)
            constraints.rows[row_idx].source_contact_idx = -1
            constraints.rows[row_idx].source_dof = dof
            constraints.rows[row_idx].limit_sign = sign
            constraints.rows[row_idx].diagApprox = diag_lim_use
            row_idx += 1

        # Upper limit: q <= range_max → constraint dist = range_max - q
        var dist_hi = rmax - pos
        if dist_hi < Scalar[DTYPE](0) and row_idx < MAX_ROWS:
            var sign = Scalar[DTYPE](-1)  # J[dof] = -1
            var K_lim = M_inv[dof * NV + dof]
            if K_lim < Scalar[DTYPE](1e-10):
                K_lim = Scalar[DTYPE](1e-10)

            for i in range(NV):
                constraints.J[row_idx * NV + i] = Scalar[DTYPE](0)
                constraints.MinvJT[row_idx * NV + i] = (
                    M_inv[i * NV + dof] * sign
                )
            constraints.J[row_idx * NV + dof] = sign

            var penetration = -dist_hi
            if penetration < Scalar[DTYPE](0):
                penetration = Scalar[DTYPE](0)
            # Use current velocity for damping (not qacc)
            var v_lim = sign * data.qvel[dof]
            var diag_lim_use_hi = diag_lim
            if diag_lim_use_hi < Scalar[DTYPE](1e-10):
                diag_lim_use_hi = K_lim
            var imp_result = _compute_aref[DTYPE](
                penetration,
                li_dmin,
                li_dmax,
                li_width,
                li_midpoint,
                li_power,
                l_K_spring,
                l_B_damp,
                v_lim,
                K_lim,
                diag_lim_use_hi,
            )

            constraints.rows[row_idx].K = K_lim
            constraints.rows[row_idx].bias = imp_result[0]
            constraints.rows[row_idx].inv_K_imp = imp_result[1]
            constraints.rows[row_idx].lo = Scalar[DTYPE](0)
            constraints.rows[row_idx].hi = Scalar[DTYPE](1e20)
            constraints.rows[row_idx].lambda_val = Scalar[DTYPE](0)
            constraints.rows[row_idx].constraint_type = CNSTR_LIMIT
            constraints.rows[row_idx].friction_parent = -1
            constraints.rows[row_idx].friction_coef = Scalar[DTYPE](0)
            constraints.rows[row_idx].source_contact_idx = -1
            constraints.rows[row_idx].source_dof = dof
            constraints.rows[row_idx].limit_sign = sign
            constraints.rows[row_idx].diagApprox = diag_lim_use_hi
            row_idx += 1

    constraints.num_limits = row_idx - limits_start

    # =========================================================================
    # Phase 4: Equality constraints (connect + weld)
    # Connect: 3 position rows (bilateral)
    # Weld: 3 position + 3 orientation rows (bilateral)
    # All rows have lo=-1e20, hi=1e20 (bilateral — force can push or pull)
    # =========================================================================
    var eq_start = row_idx

    comptime if MAX_EQUALITY > 0:
        # Read solref/solimp for each equality constraint (per-constraint)
        for eq_idx in range(model.num_equality):
            var eq = model.equality_constraints[eq_idx]
            var body_a = eq.body_a
            var body_b = eq.body_b

            # Read per-constraint impedance params
            var eq_sr_tc = eq.solref_0
            var eq_sr_dr = eq.solref_1
            var eq_si_dmin = eq.solimp_0
            var eq_si_dmax = eq.solimp_1
            var eq_si_width = eq.solimp_2
            var eq_si_midpoint = eq.solimp_3
            var eq_si_power = eq.solimp_4
            if eq_si_width < Scalar[DTYPE](1e-6):
                eq_si_width = Scalar[DTYPE](1e-6)
            if eq_si_dmax < Scalar[DTYPE](1e-4):
                eq_si_dmax = Scalar[DTYPE](1e-4)
            var eq_K_spring = Scalar[DTYPE](1.0) / (
                eq_sr_tc * eq_sr_tc * eq_si_dmax * eq_si_dmax
            )
            var eq_B_damp = (
                Scalar[DTYPE](2.0) * eq_sr_dr / (eq_sr_tc * eq_si_dmax)
            )

            # Compute world anchor positions
            var world_a_x: Scalar[DTYPE]
            var world_a_y: Scalar[DTYPE]
            var world_a_z: Scalar[DTYPE]
            if body_a > 0:
                var rot_a = quat_rotate[DTYPE](
                    data.xquat[body_a * 4 + 0],
                    data.xquat[body_a * 4 + 1],
                    data.xquat[body_a * 4 + 2],
                    data.xquat[body_a * 4 + 3],
                    eq.anchor_a_x,
                    eq.anchor_a_y,
                    eq.anchor_a_z,
                )
                world_a_x = data.xpos[body_a * 3 + 0] + rot_a[0]
                world_a_y = data.xpos[body_a * 3 + 1] + rot_a[1]
                world_a_z = data.xpos[body_a * 3 + 2] + rot_a[2]
            else:
                world_a_x = eq.anchor_a_x
                world_a_y = eq.anchor_a_y
                world_a_z = eq.anchor_a_z

            var world_b_x: Scalar[DTYPE]
            var world_b_y: Scalar[DTYPE]
            var world_b_z: Scalar[DTYPE]
            if body_b > 0:
                var rot_b = quat_rotate[DTYPE](
                    data.xquat[body_b * 4 + 0],
                    data.xquat[body_b * 4 + 1],
                    data.xquat[body_b * 4 + 2],
                    data.xquat[body_b * 4 + 3],
                    eq.anchor_b_x,
                    eq.anchor_b_y,
                    eq.anchor_b_z,
                )
                world_b_x = data.xpos[body_b * 3 + 0] + rot_b[0]
                world_b_y = data.xpos[body_b * 3 + 1] + rot_b[1]
                world_b_z = data.xpos[body_b * 3 + 2] + rot_b[2]
            else:
                world_b_x = eq.anchor_b_x
                world_b_y = eq.anchor_b_y
                world_b_z = eq.anchor_b_z

            # Position error: world_a - world_b (should be zero)
            var pos_err_x = world_a_x - world_b_x
            var pos_err_y = world_a_y - world_b_y
            var pos_err_z = world_a_z - world_b_z

            # --- 3 position rows (connect + weld) ---
            var dirs = InlineArray[Scalar[DTYPE], 9](uninitialized=True)
            dirs[0] = 1
            dirs[1] = 0
            dirs[2] = 0  # x-axis
            dirs[3] = 0
            dirs[4] = 1
            dirs[5] = 0  # y-axis
            dirs[6] = 0
            dirs[7] = 0
            dirs[8] = 1  # z-axis

            var pos_errs = InlineArray[Scalar[DTYPE], 3](uninitialized=True)
            pos_errs[0] = pos_err_x
            pos_errs[1] = pos_err_y
            pos_errs[2] = pos_err_z

            for d in range(3):
                if row_idx >= MAX_ROWS:
                    break

                var dir_x = dirs[d * 3 + 0]
                var dir_y = dirs[d * 3 + 1]
                var dir_z = dirs[d * 3 + 2]

                # Compute Jacobian: J = J_a(at world_a) - J_b(at world_b)
                # Each body's Jacobian uses its OWN anchor point (MuJoCo convention)
                _compute_weld_jacobian_row(
                    model,
                    data,
                    cdof,
                    body_a,
                    body_b,
                    world_a_x,
                    world_a_y,
                    world_a_z,
                    world_b_x,
                    world_b_y,
                    world_b_z,
                    dir_x,
                    dir_y,
                    dir_z,
                    J_row,
                )

                # Compute K = J @ M_inv @ J^T and v_n = J @ qvel
                var k_eq: Scalar[DTYPE] = 0
                var v_eq: Scalar[DTYPE] = 0
                for i in range(NV):
                    constraints.J[row_idx * NV + i] = J_row[i]
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
                    constraints.MinvJT[row_idx * NV + i] = mi_j_sum
                    k_eq += J_row[i] * mi_j_sum
                    v_eq += J_row[i] * data.qvel[i]
                if k_eq < Scalar[DTYPE](1e-10):
                    k_eq = Scalar[DTYPE](1e-10)

                # Equality: signed error, not penetration
                var err_d = pos_errs[d]

                # Equality diagApprox: translation weights
                var diag_eq: Scalar[DTYPE] = 0
                if body_a > 0 and body_a < NBODY:
                    diag_eq += model.body_invweight0[body_a * 2]
                if body_b > 0 and body_b < NBODY:
                    diag_eq += model.body_invweight0[body_b * 2]
                if diag_eq < Scalar[DTYPE](1e-10):
                    diag_eq = k_eq

                # Impedance computed from absolute distance (position-dependent)
                var imp_result = _compute_aref[DTYPE](
                    abs(err_d),
                    eq_si_dmin,
                    eq_si_dmax,
                    eq_si_width,
                    eq_si_midpoint,
                    eq_si_power,
                    eq_K_spring,
                    eq_B_damp,
                    v_eq,
                    k_eq,
                    diag_eq,
                )

                # MuJoCo equality bias: bias = -aref = B*vel + K*I*pos
                # where pos is the SIGNED error (not abs). Contact formula
                # uses -K*I*pen because contact pos = -penetration, but
                # equality pos is signed directly.
                var imp_val = imp_result[2]
                var bias_eq = eq_K_spring * imp_val * err_d + eq_B_damp * v_eq

                constraints.rows[row_idx].K = k_eq
                constraints.rows[row_idx].bias = bias_eq
                constraints.rows[row_idx].inv_K_imp = imp_result[1]
                constraints.rows[row_idx].lo = Scalar[DTYPE](-1e20)
                constraints.rows[row_idx].hi = Scalar[DTYPE](1e20)
                constraints.rows[row_idx].lambda_val = Scalar[DTYPE](0)
                constraints.rows[
                    row_idx
                ].constraint_type = CNSTR_EQUALITY_CONNECT
                constraints.rows[row_idx].friction_parent = -1
                constraints.rows[row_idx].friction_coef = Scalar[DTYPE](0)
                constraints.rows[row_idx].source_contact_idx = -1
                constraints.rows[row_idx].source_dof = -1
                constraints.rows[row_idx].limit_sign = Scalar[DTYPE](0)
                constraints.rows[row_idx].diagApprox = diag_eq
                row_idx += 1

            # --- 3 orientation rows (weld only) ---
            if eq.eq_type == EQ_WELD:
                # Orientation error: 0.5 * imag(conj(quat_b) * quat_a * relpose)
                var qa_x = data.xquat[body_a * 4 + 0] if body_a > 0 else Scalar[
                    DTYPE
                ](0)
                var qa_y = data.xquat[body_a * 4 + 1] if body_a > 0 else Scalar[
                    DTYPE
                ](0)
                var qa_z = data.xquat[body_a * 4 + 2] if body_a > 0 else Scalar[
                    DTYPE
                ](0)
                var qa_w = data.xquat[body_a * 4 + 3] if body_a > 0 else Scalar[
                    DTYPE
                ](1)

                var qb_x = data.xquat[body_b * 4 + 0] if body_b > 0 else Scalar[
                    DTYPE
                ](0)
                var qb_y = data.xquat[body_b * 4 + 1] if body_b > 0 else Scalar[
                    DTYPE
                ](0)
                var qb_z = data.xquat[body_b * 4 + 2] if body_b > 0 else Scalar[
                    DTYPE
                ](0)
                var qb_w = data.xquat[body_b * 4 + 3] if body_b > 0 else Scalar[
                    DTYPE
                ](1)

                # conj(qb) * qa
                var qb_conj = quat_conjugate[DTYPE](qb_x, qb_y, qb_z, qb_w)
                var q_diff = quat_mul[DTYPE](
                    qb_conj[0],
                    qb_conj[1],
                    qb_conj[2],
                    qb_conj[3],
                    qa_x,
                    qa_y,
                    qa_z,
                    qa_w,
                )
                # q_diff * relpose
                var q_err = quat_mul[DTYPE](
                    q_diff[0],
                    q_diff[1],
                    q_diff[2],
                    q_diff[3],
                    eq.relpose_x,
                    eq.relpose_y,
                    eq.relpose_z,
                    eq.relpose_w,
                )
                # Orientation error = 0.5 * imaginary part
                var rot_err_x = Scalar[DTYPE](0.5) * q_err[0]
                var rot_err_y = Scalar[DTYPE](0.5) * q_err[1]
                var rot_err_z = Scalar[DTYPE](0.5) * q_err[2]

                var rot_errs = InlineArray[Scalar[DTYPE], 3](uninitialized=True)
                rot_errs[0] = rot_err_x
                rot_errs[1] = rot_err_y
                rot_errs[2] = rot_err_z

                for d in range(3):
                    if row_idx >= MAX_ROWS:
                        break

                    var dir_x = dirs[d * 3 + 0]
                    var dir_y = dirs[d * 3 + 1]
                    var dir_z = dirs[d * 3 + 2]

                    # Angular-only Jacobian (like torsional friction)
                    _compute_angular_jacobian_row(
                        model,
                        data,
                        cdof,
                        body_a,
                        body_b,
                        dir_x,
                        dir_y,
                        dir_z,
                        J_row,
                    )

                    var k_rot: Scalar[DTYPE] = 0
                    var v_rot: Scalar[DTYPE] = 0
                    for i in range(NV):
                        constraints.J[row_idx * NV + i] = J_row[i]
                        var mi_j_sum: Scalar[DTYPE] = 0
                        for j_idx in range(NV):
                            mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
                        constraints.MinvJT[row_idx * NV + i] = mi_j_sum
                        k_rot += J_row[i] * mi_j_sum
                        v_rot += J_row[i] * data.qvel[i]
                    if k_rot < Scalar[DTYPE](1e-10):
                        k_rot = Scalar[DTYPE](1e-10)

                    var err_rot = rot_errs[d]

                    # Weld rotation: use rotation weights
                    var diag_rot: Scalar[DTYPE] = 0
                    if body_a > 0 and body_a < NBODY:
                        diag_rot += model.body_invweight0[body_a * 2 + 1]
                    if body_b > 0 and body_b < NBODY:
                        diag_rot += model.body_invweight0[body_b * 2 + 1]
                    if diag_rot < Scalar[DTYPE](1e-10):
                        diag_rot = k_rot

                    # Impedance from absolute distance
                    var imp_rot = _compute_aref[DTYPE](
                        abs(err_rot),
                        eq_si_dmin,
                        eq_si_dmax,
                        eq_si_width,
                        eq_si_midpoint,
                        eq_si_power,
                        eq_K_spring,
                        eq_B_damp,
                        v_rot,
                        k_rot,
                        diag_rot,
                    )

                    # MuJoCo equality bias: bias = K*I*pos + B*vel (signed pos)
                    var imp_rot_val = imp_rot[2]
                    var bias_rot = eq_K_spring * imp_rot_val * err_rot + eq_B_damp * v_rot

                    constraints.rows[row_idx].K = k_rot
                    constraints.rows[row_idx].bias = bias_rot
                    constraints.rows[row_idx].inv_K_imp = imp_rot[1]
                    constraints.rows[row_idx].lo = Scalar[DTYPE](-1e20)
                    constraints.rows[row_idx].hi = Scalar[DTYPE](1e20)
                    constraints.rows[row_idx].lambda_val = Scalar[DTYPE](0)
                    constraints.rows[
                        row_idx
                    ].constraint_type = CNSTR_EQUALITY_WELD
                    constraints.rows[row_idx].friction_parent = -1
                    constraints.rows[row_idx].friction_coef = Scalar[DTYPE](0)
                    constraints.rows[row_idx].source_contact_idx = -1
                    constraints.rows[row_idx].source_dof = -1
                    constraints.rows[row_idx].limit_sign = Scalar[DTYPE](0)
                    constraints.rows[row_idx].diagApprox = diag_rot
                    row_idx += 1

    # =========================================================================
    # Phase 5: Fixed tendon constraints
    # Each tendon produces 1 bilateral row: ten_length - length_ref = 0
    # Jacobian is trivial: J[dof_adr_i] = coef_i
    # =========================================================================

    comptime if MAX_TENDON > 0:
        for t_idx in range(model.num_tendons):
            if row_idx >= MAX_ROWS:
                break

            var ten = model.tendons[t_idx]

            # Read per-tendon impedance params
            var ten_sr_tc = ten.solref_0
            var ten_sr_dr = ten.solref_1
            var ten_si_dmin = ten.solimp_0
            var ten_si_dmax = ten.solimp_1
            var ten_si_width = ten.solimp_2
            var ten_si_midpoint = ten.solimp_3
            var ten_si_power = ten.solimp_4
            if ten_si_width < Scalar[DTYPE](1e-6):
                ten_si_width = Scalar[DTYPE](1e-6)
            if ten_si_dmax < Scalar[DTYPE](1e-4):
                ten_si_dmax = Scalar[DTYPE](1e-4)
            # MuJoCo: K = 1/(dmax² * timeconst² * dampratio²)
            var ten_K_spring = Scalar[DTYPE](1.0) / (
                ten_si_dmax * ten_si_dmax * ten_sr_tc * ten_sr_tc * ten_sr_dr * ten_sr_dr
            )
            # MuJoCo: B = 2/(dmax * timeconst)
            var ten_B_damp = (
                Scalar[DTYPE](2.0) / (ten_si_dmax * ten_sr_tc)
            )

            # Compute tendon length: Σ coef_i * qpos[joint.qpos_adr]
            # and tendon velocity: Σ coef_i * qvel[joint.dof_adr]
            var ten_length: Scalar[DTYPE] = 0
            var ten_vel: Scalar[DTYPE] = 0

            # Build trivial Jacobian: J[dof_adr_i] = coef_i
            for i in range(NV):
                J_row[i] = Scalar[DTYPE](0)

            # Joint 0
            if ten.num_joints > 0 and ten.joint_idx_0 >= 0:
                var j = model.joints[ten.joint_idx_0]
                ten_length += ten.coef_0 * data.qpos[j.qpos_adr]
                ten_vel += ten.coef_0 * data.qvel[j.dof_adr]
                J_row[j.dof_adr] = ten.coef_0
            # Joint 1
            if ten.num_joints > 1 and ten.joint_idx_1 >= 0:
                var j = model.joints[ten.joint_idx_1]
                ten_length += ten.coef_1 * data.qpos[j.qpos_adr]
                ten_vel += ten.coef_1 * data.qvel[j.dof_adr]
                J_row[j.dof_adr] = ten.coef_1
            # Joint 2
            if ten.num_joints > 2 and ten.joint_idx_2 >= 0:
                var j = model.joints[ten.joint_idx_2]
                ten_length += ten.coef_2 * data.qpos[j.qpos_adr]
                ten_vel += ten.coef_2 * data.qvel[j.dof_adr]
                J_row[j.dof_adr] = ten.coef_2
            # Joint 3
            if ten.num_joints > 3 and ten.joint_idx_3 >= 0:
                var j = model.joints[ten.joint_idx_3]
                ten_length += ten.coef_3 * data.qpos[j.qpos_adr]
                ten_vel += ten.coef_3 * data.qvel[j.dof_adr]
                J_row[j.dof_adr] = ten.coef_3

            # Tendon error: ten_length - length_ref (signed)
            var ten_err = ten_length - ten.length_ref

            # Compute K = J @ M_inv @ J^T
            var k_ten: Scalar[DTYPE] = 0
            for i in range(NV):
                constraints.J[row_idx * NV + i] = J_row[i]
                var mi_j_sum: Scalar[DTYPE] = 0
                for j_idx in range(NV):
                    mi_j_sum += M_inv[i * NV + j_idx] * J_row[j_idx]
                constraints.MinvJT[row_idx * NV + i] = mi_j_sum
                k_ten += J_row[i] * mi_j_sum
            if k_ten < Scalar[DTYPE](1e-10):
                k_ten = Scalar[DTYPE](1e-10)

            # diagApprox: sum of dof_invweight0 for involved DOFs
            var diag_ten: Scalar[DTYPE] = 0
            if ten.num_joints > 0 and ten.joint_idx_0 >= 0:
                diag_ten += model.dof_invweight0[
                    model.joints[ten.joint_idx_0].dof_adr
                ]
            if ten.num_joints > 1 and ten.joint_idx_1 >= 0:
                diag_ten += model.dof_invweight0[
                    model.joints[ten.joint_idx_1].dof_adr
                ]
            if ten.num_joints > 2 and ten.joint_idx_2 >= 0:
                diag_ten += model.dof_invweight0[
                    model.joints[ten.joint_idx_2].dof_adr
                ]
            if ten.num_joints > 3 and ten.joint_idx_3 >= 0:
                diag_ten += model.dof_invweight0[
                    model.joints[ten.joint_idx_3].dof_adr
                ]
            if diag_ten < Scalar[DTYPE](1e-10):
                diag_ten = k_ten

            # Compute impedance from |error| (symmetric schedule)
            var imp_ten = _compute_aref[DTYPE](
                abs(ten_err),
                ten_si_dmin,
                ten_si_dmax,
                ten_si_width,
                ten_si_midpoint,
                ten_si_power,
                ten_K_spring,
                ten_B_damp,
                ten_vel,
                k_ten,
                diag_ten,
            )

            # Bilateral equality constraint bias (MuJoCo formula):
            #   aref = -B*vel - K*imp*pos  (pos = ten_err, signed)
            #   bias = -aref = B*vel + K*imp*pos
            var imp_val = imp_ten[2]
            var bias_ten = ten_B_damp * ten_vel + ten_K_spring * imp_val * ten_err

            constraints.rows[row_idx].K = k_ten
            constraints.rows[row_idx].bias = bias_ten
            constraints.rows[row_idx].inv_K_imp = imp_ten[1]
            constraints.rows[row_idx].lo = Scalar[DTYPE](-1e20)
            constraints.rows[row_idx].hi = Scalar[DTYPE](1e20)
            constraints.rows[row_idx].lambda_val = Scalar[DTYPE](0)
            constraints.rows[row_idx].constraint_type = CNSTR_EQUALITY_TENDON
            constraints.rows[row_idx].friction_parent = -1
            constraints.rows[row_idx].friction_coef = Scalar[DTYPE](0)
            constraints.rows[row_idx].source_contact_idx = -1
            constraints.rows[row_idx].source_dof = -1
            constraints.rows[row_idx].limit_sign = Scalar[DTYPE](0)
            constraints.rows[row_idx].diagApprox = diag_ten
            row_idx += 1

    constraints.num_equality = row_idx - eq_start
    constraints.num_rows = row_idx


def writeback_forces[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    MAX_ROWS: Int,
    NSITE: Int = 0,
](
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
):
    """Write solved constraint forces back to data.contacts for warm-starting.

    Loops over all constraint rows and writes force_n/t1/t2 based on constraint_type.
    """
    # Zero contact forces for pyramidal mode (multiple edges accumulate per contact)
    var has_pyramid = False
    for r in range(constraints.num_rows):
        if constraints.rows[r].constraint_type == CNSTR_PYRAMID_EDGE:
            has_pyramid = True
            break
    if has_pyramid:
        for c in range(data.num_contacts):
            data.contacts[c].force_n = Scalar[DTYPE](0)
            data.contacts[c].force_t1 = Scalar[DTYPE](0)
            data.contacts[c].force_t2 = Scalar[DTYPE](0)
            data.contacts[c].force_torsion = Scalar[DTYPE](0)
            data.contacts[c].force_roll1 = Scalar[DTYPE](0)
            data.contacts[c].force_roll2 = Scalar[DTYPE](0)

    for r in range(constraints.num_rows):
        var row = constraints.rows[r]
        var c = row.source_contact_idx
        if c < 0:
            continue

        if row.constraint_type == CNSTR_NORMAL:
            data.contacts[c].force_n = row.lambda_val
        elif row.constraint_type == CNSTR_FRICTION_T1:
            data.contacts[c].force_t1 = row.lambda_val
        elif row.constraint_type == CNSTR_FRICTION_T2:
            data.contacts[c].force_t2 = row.lambda_val
        elif row.constraint_type == CNSTR_FRICTION_TORSION:
            data.contacts[c].force_torsion = row.lambda_val
        elif row.constraint_type == CNSTR_FRICTION_ROLL1:
            data.contacts[c].force_roll1 = row.lambda_val
        elif row.constraint_type == CNSTR_FRICTION_ROLL2:
            data.contacts[c].force_roll2 = row.lambda_val
        elif row.constraint_type == CNSTR_PYRAMID_EDGE:
            # Decode: source_dof = td * 2 + sign_idx
            var td = row.source_dof // 2
            var sign_idx = row.source_dof % 2
            var sign = Scalar[DTYPE](1.0) if sign_idx == 0 else Scalar[DTYPE](
                -1.0
            )
            # All edges contribute to normal force
            data.contacts[c].force_n += row.lambda_val
            # Tangent force = mu * sign * lambda
            var tangent_force = row.friction_coef * sign * row.lambda_val
            if td == 0:
                data.contacts[c].force_t1 += tangent_force
            elif td == 1:
                data.contacts[c].force_t2 += tangent_force
            elif td == 2:
                data.contacts[c].force_torsion += tangent_force
            elif td == 3:
                data.contacts[c].force_roll1 += tangent_force
            elif td == 4:
                data.contacts[c].force_roll2 += tangent_force
