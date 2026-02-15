"""Constraint builder for physics3d solvers.

Extracts all constraint setup (contact normals, friction, joint limits,
equality constraints) into a single builder function. Solvers become pure
iterative algorithms consuming pre-built ConstraintData.

This consolidates code previously duplicated across PGS, CG, and Newton solvers.
"""

from math import sqrt
from ..types import Model, Data, EQ_CONNECT, EQ_WELD, _max_one, ConeType
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
)


fn _compute_aref[
    DTYPE: DType,
](
    penetration: Scalar[DTYPE],
    si_dmin: Scalar[DTYPE],
    si_dmax: Scalar[DTYPE],
    si_width: Scalar[DTYPE],
    K_spring: Scalar[DTYPE],
    B_damp: Scalar[DTYPE],
    v_n: Scalar[DTYPE],
    K: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE]]:
    """Compute acceleration-level constraint reference and inv_K.

    Returns (bias, inv_K) where:
    - bias = -aref = -(K_spring * imp * penetration) + (B_damp * imp * v_n)
    - inv_K = imp / K (MuJoCo regularizer: AR = K + R, R = (1-imp)/imp * K, so AR = K/imp)

    MuJoCo formula: aref = K*d*pen - B*d*v_n  (both terms scaled by impedance d)
    Both position and velocity terms push apart (positive aref for penetrating contact).
    Stored negated as bias so PGS formula -(a + bias) * inv_K works.
    """
    var x = penetration / si_width
    if x > Scalar[DTYPE](1.0):
        x = Scalar[DTYPE](1.0)
    var imp = si_dmin + (
        Scalar[DTYPE](3.0) * x * x - Scalar[DTYPE](2.0) * x * x * x
    ) * (si_dmax - si_dmin)
    # MuJoCo uses mjMINIMP ~1e-6 (only prevents division by zero)
    if imp < Scalar[DTYPE](1e-6):
        imp = Scalar[DTYPE](1e-6)
    # MuJoCo: aref = -B*vel - K*imp*pos, bias = -aref = B*vel + K*imp*pen
    # Only K term is scaled by imp, NOT B (see engine_core_constraint.c:2384)
    var bias = -K_spring * imp * penetration + B_damp * v_n
    # MuJoCo: AR[i,i] = K + (1-imp)/imp * K = K/imp, so inv = imp/K
    var inv_K = imp / K
    return (bias, inv_K)


fn _compute_angular_jacobian_row[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    V_SIZE: Int,
    CDOF_SIZE: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
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
    ],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    cdof: InlineArray[Scalar[DTYPE], CDOF_SIZE],
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
    for i in range(V_SIZE):
        J_row[i] = Scalar[DTYPE](0)

    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr

        var affects_a = _joint_affects_body(model, j, contact_body_a)
        var affects_b = (contact_body_b >= 0) and _joint_affects_body(
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


fn _joint_affects_body[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
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
    while current >= 0:
        if model.body_parent[current] == joint_body:
            return True
        current = model.body_parent[current]
    return False


fn build_constraints[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    MAX_ROWS: Int,
    V_SIZE: Int,
    M_SIZE: Int,
    CDOF_SIZE: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
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
    ],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    cdof: InlineArray[Scalar[DTYPE], CDOF_SIZE],
    M_inv: InlineArray[Scalar[DTYPE], M_SIZE],
    qvel: InlineArray[Scalar[DTYPE], V_SIZE],
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
    if si_width < Scalar[DTYPE](1e-6):
        si_width = Scalar[DTYPE](1e-6)
    if si_dmax < Scalar[DTYPE](1e-4):
        si_dmax = Scalar[DTYPE](1e-4)
    # Acceleration-level spring/damper coefficients (dt-independent)
    var K_spring = Scalar[DTYPE](1.0) / (
        si_dmax * si_dmax * sr_tc * sr_tc * sr_dr * sr_dr
    )
    var B_damp = Scalar[DTYPE](2.0) / (si_dmax * sr_tc)
    var default_friction = model.friction

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

    @parameter
    if CONE_TYPE == ConeType.PYRAMIDAL:
        # =================================================================
        # PYRAMIDAL CONE: Build edge rows (all >= 0 constraints)
        # J_edge± = J_normal ± mu_k * J_tangent_k
        # No separate normal or friction rows — edges encode both.
        # =================================================================
        var J_n = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var J_t = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

        for c in range(nc):
            var contact = data.contacts[c]
            if contact.dist >= Scalar[DTYPE](0):
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

            var penetration = -contact.dist
            var imp_result = _compute_aref[DTYPE](
                penetration,
                si_dmin,
                si_dmax,
                si_width,
                K_spring,
                B_damp,
                v_n,
                k_n,
            )
            var bias_n = imp_result[0]
            var inv_K_imp_n = imp_result[1]

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

            # Tangent basis
            var t1_x: Scalar[DTYPE]
            var t1_y: Scalar[DTYPE]
            var t1_z: Scalar[DTYPE]
            if abs(nx) < Scalar[DTYPE](0.9):
                t1_x = Scalar[DTYPE](0)
                t1_y = -nz
                t1_z = ny
            else:
                t1_x = nz
                t1_y = Scalar[DTYPE](0)
                t1_z = -nx
            var t1_mag = sqrt(t1_x * t1_x + t1_y * t1_y + t1_z * t1_z)
            if t1_mag > Scalar[DTYPE](1e-10):
                t1_x /= t1_mag
                t1_y /= t1_mag
                t1_z /= t1_mag
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
                        CDOF_SIZE,
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
                        CDOF_SIZE,
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
                        CDOF_SIZE,
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

                # Pyramidal regularizer: R_pyramid = 2 * mu^2 * R_normal
                var R_n = Scalar[DTYPE](1.0) / inv_K_imp_n - k_n  # R_normal
                var R_edge = Scalar[DTYPE](2.0) * mu_td * mu_td * R_n

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

                    # Edge impedance: inv_K = 1 / (k_edge + R_edge)
                    var inv_K_edge = Scalar[DTYPE](1.0) / (k_edge + R_edge)

                    constraints.rows[row_idx].K = k_edge
                    constraints.rows[row_idx].bias = bias_n
                    constraints.rows[row_idx].inv_K_imp = inv_K_edge
                    constraints.rows[row_idx].lo = Scalar[DTYPE](0)
                    constraints.rows[row_idx].hi = Scalar[DTYPE](1e20)
                    constraints.rows[row_idx].lambda_val = Scalar[DTYPE](0)
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

            if contact.dist >= Scalar[DTYPE](0):
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

            var penetration = -contact.dist
            var imp_result = _compute_aref[DTYPE](
                penetration,
                si_dmin,
                si_dmax,
                si_width,
                K_spring,
                B_damp,
                v_n,
                k,
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

        # Compute tangent basis
        var t1_x: Scalar[DTYPE]
        var t1_y: Scalar[DTYPE]
        var t1_z: Scalar[DTYPE]

        if abs(nx) < Scalar[DTYPE](0.9):
            t1_x = Scalar[DTYPE](0)
            t1_y = -nz
            t1_z = ny
        else:
            t1_x = nz
            t1_y = Scalar[DTYPE](0)
            t1_z = -nx

        var t1_mag = sqrt(t1_x * t1_x + t1_y * t1_y + t1_z * t1_z)
        if t1_mag > Scalar[DTYPE](1e-10):
            t1_x = t1_x / t1_mag
            t1_y = t1_y / t1_mag
            t1_z = t1_z / t1_mag

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

        # Compute friction regularizer from parent normal's impedance
        var imp_n = (
            constraints.rows[normal_row].inv_K_imp
            * constraints.rows[normal_row].K
        )
        var R_n = (
            (Scalar[DTYPE](1.0) - imp_n)
            / imp_n
            * constraints.rows[normal_row].K
        )
        var R_f1 = R_n / model.impratio
        var inv_K_imp_f1 = Scalar[DTYPE](1.0) / (k1 + R_f1)

        # Friction velocity damping bias (MuJoCo-style):
        # aref_friction = B_damp * imp * v_tangential → bias = -aref
        var bias_f1 = B_damp * imp_n * v_t1

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
        var bias_f2 = B_damp * imp_n * v_t2

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
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, CDOF_SIZE
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

            var bias_f3 = B_damp * imp_n * v_t3
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
                    CDOF_SIZE,
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

                var bias_f4 = B_damp * imp_n * v_t4
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
                    CDOF_SIZE,
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

                var bias_f5 = B_damp * imp_n * v_t5
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

    # Read solref/solimp for limits
    var lr_tc = model.solref_limit[0]
    var lr_dr = model.solref_limit[1]
    var li_dmin = model.solimp_limit[0]
    var li_dmax = model.solimp_limit[1]
    var li_width = model.solimp_limit[2]
    if li_width < Scalar[DTYPE](1e-6):
        li_width = Scalar[DTYPE](1e-6)
    if li_dmax < Scalar[DTYPE](1e-4):
        li_dmax = Scalar[DTYPE](1e-4)
    # Acceleration-level spring/damper for limits
    var l_K_spring = Scalar[DTYPE](1.0) / (
        li_dmax * li_dmax * lr_tc * lr_tc * lr_dr * lr_dr
    )
    var l_B_damp = Scalar[DTYPE](2.0) / (li_dmax * lr_tc)

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

        # Lower limit: q >= range_min → constraint dist = q - range_min
        var dist_lo = pos - rmin
        if dist_lo < Scalar[DTYPE](0.01) and row_idx < MAX_ROWS:
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
            var imp_result = _compute_aref[DTYPE](
                penetration,
                li_dmin,
                li_dmax,
                li_width,
                l_K_spring,
                l_B_damp,
                v_lim,
                K_lim,
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
            row_idx += 1

        # Upper limit: q <= range_max → constraint dist = range_max - q
        var dist_hi = rmax - pos
        if dist_hi < Scalar[DTYPE](0.01) and row_idx < MAX_ROWS:
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
            var imp_result = _compute_aref[DTYPE](
                penetration,
                li_dmin,
                li_dmax,
                li_width,
                l_K_spring,
                l_B_damp,
                v_lim,
                K_lim,
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
            row_idx += 1

    constraints.num_limits = row_idx - limits_start

    # =========================================================================
    # Phase 4: Equality constraints (connect + weld)
    # Connect: 3 position rows (bilateral)
    # Weld: 3 position + 3 orientation rows (bilateral)
    # All rows have lo=-1e20, hi=1e20 (bilateral — force can push or pull)
    # =========================================================================
    var eq_start = row_idx

    @parameter
    if MAX_EQUALITY > 0:
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
            if eq_si_width < Scalar[DTYPE](1e-6):
                eq_si_width = Scalar[DTYPE](1e-6)
            if eq_si_dmax < Scalar[DTYPE](1e-4):
                eq_si_dmax = Scalar[DTYPE](1e-4)
            var eq_K_spring = Scalar[DTYPE](1.0) / (
                eq_si_dmax
                * eq_si_dmax
                * eq_sr_tc
                * eq_sr_tc
                * eq_sr_dr
                * eq_sr_dr
            )
            var eq_B_damp = Scalar[DTYPE](2.0) / (eq_si_dmax * eq_sr_tc)

            # Compute world anchor positions
            var world_a_x: Scalar[DTYPE]
            var world_a_y: Scalar[DTYPE]
            var world_a_z: Scalar[DTYPE]
            if body_a >= 0:
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
            if body_b >= 0:
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

                # Compute Jacobian row: J = contact_jacobian(body_a, body_b, world_a, dir)
                compute_contact_jacobian_row(
                    model,
                    data,
                    cdof,
                    body_a,
                    body_b,
                    world_a_x,
                    world_a_y,
                    world_a_z,
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

                # Penetration = |pos_error[d]|, with sign handling for bilateral
                var err_d = pos_errs[d]
                var penetration = abs(err_d)

                var imp_result = _compute_aref[DTYPE](
                    penetration,
                    eq_si_dmin,
                    eq_si_dmax,
                    eq_si_width,
                    eq_K_spring,
                    eq_B_damp,
                    v_eq,
                    k_eq,
                )

                # For bilateral equality: bias sign depends on error direction
                # error > 0 → need negative force → bias stays as computed
                # error < 0 → need positive force → flip bias sign
                var bias_eq = imp_result[0]
                if err_d < Scalar[DTYPE](0):
                    bias_eq = -bias_eq

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
                row_idx += 1

            # --- 3 orientation rows (weld only) ---
            if eq.eq_type == EQ_WELD:
                # Orientation error: 0.5 * imag(conj(quat_b) * quat_a * relpose)
                var qa_x = data.xquat[
                    body_a * 4 + 0
                ] if body_a >= 0 else Scalar[DTYPE](0)
                var qa_y = data.xquat[
                    body_a * 4 + 1
                ] if body_a >= 0 else Scalar[DTYPE](0)
                var qa_z = data.xquat[
                    body_a * 4 + 2
                ] if body_a >= 0 else Scalar[DTYPE](0)
                var qa_w = data.xquat[
                    body_a * 4 + 3
                ] if body_a >= 0 else Scalar[DTYPE](1)

                var qb_x = data.xquat[
                    body_b * 4 + 0
                ] if body_b >= 0 else Scalar[DTYPE](0)
                var qb_y = data.xquat[
                    body_b * 4 + 1
                ] if body_b >= 0 else Scalar[DTYPE](0)
                var qb_z = data.xquat[
                    body_b * 4 + 2
                ] if body_b >= 0 else Scalar[DTYPE](0)
                var qb_w = data.xquat[
                    body_b * 4 + 3
                ] if body_b >= 0 else Scalar[DTYPE](1)

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
                    var pen_rot = abs(err_rot)

                    var imp_rot = _compute_aref[DTYPE](
                        pen_rot,
                        eq_si_dmin,
                        eq_si_dmax,
                        eq_si_width,
                        eq_K_spring,
                        eq_B_damp,
                        v_rot,
                        k_rot,
                    )

                    var bias_rot = imp_rot[0]
                    if err_rot < Scalar[DTYPE](0):
                        bias_rot = -bias_rot

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
                    row_idx += 1

    constraints.num_equality = row_idx - eq_start
    constraints.num_rows = row_idx


fn writeback_forces[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    MAX_ROWS: Int,
](
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
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
