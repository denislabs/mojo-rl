"""Shared infrastructure for MuJoCo-style primal solvers.

Primal solvers operate in qacc (acceleration) space, minimizing:
  cost = 0.5*(qacc - qacc_smooth)^T * M * (qacc - qacc_smooth)  [Gauss term]
       + sum_i penalty_i(J*qacc - aref)                         [constraint costs]

Forces are derived from qacc: force[i] = -D[i] * (J*qacc - aref)[i].
The Hessian is H = M + J^T*D_active*J (naturally PD).

Cone-aware constraint_update handles elliptic friction cones with 3-zone logic
(satisfied/quadratic/cone), matching MuJoCo's mj_constraintUpdate_impl exactly.
This eliminates the need for a separate PGS friction phase.

Reference: mujoco-main/src/engine/engine_solver.c (mj_solPrimal)
           mujoco-main/src/engine/engine_core_constraint.c (mj_constraintUpdate_impl)
"""

from math import sqrt, max
from ..types import _max_one
from ..constraints.constraint_data import (
    CNSTR_NORMAL,
    CNSTR_FRICTION_T1,
    CNSTR_FRICTION_T2,
    CNSTR_LIMIT,
    CNSTR_FRICTION_TORSION,
    CNSTR_FRICTION_ROLL1,
    CNSTR_FRICTION_ROLL2,
    CNSTR_PYRAMID_EDGE,
    CNSTR_EQUALITY_CONNECT,
    CNSTR_EQUALITY_WELD,
    ConstraintData,
)

# Constraint states (matching MuJoCo mjCnstrState)
comptime PRIMAL_SATISFIED: Int = 0  # Inequality constraint satisfied (jar >= 0)
comptime PRIMAL_QUADRATIC: Int = 1  # Active constraint (quadratic cost)
comptime PRIMAL_CONE: Int = 2  # On friction cone boundary (elliptic only)

# Primal solver parameters
comptime PRIMAL_MAX_LINESEARCH: Int = 20
comptime PRIMAL_MINVAL: Float64 = 1e-12


@always_inline
fn primal_D[DTYPE: DType](
    inv_K_imp: Scalar[DTYPE],
    K: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """Compute MuJoCo-matching primal D = 1/R from stored constraint values.

    Our constraint rows store:
      inv_K_imp = 1/(K + R)  (combined, for PGS dual solver)
      K = Delassus diagonal (or K_spring for normals)

    MuJoCo's primal solver uses:
      D = 1/R  where  R = 1/inv_K_imp - K

    For normal rows: inv_K_imp = imp/K_n, R = (1-imp)/imp * K_n
      → D = imp / ((1-imp) * K_n)
    For friction: inv_K_imp = 1/(K_f + R_f), R = R_f
      → D = 1/R_f
    """
    var R = Scalar[DTYPE](1) / inv_K_imp - K
    if R < Scalar[DTYPE](PRIMAL_MINVAL):
        R = Scalar[DTYPE](PRIMAL_MINVAL)
    return Scalar[DTYPE](1) / R


@always_inline
fn constraint_update[
    DTYPE: DType,
    MAX_ROWS: Int,
    NV: Int,
    R_SIZE: Int,
](
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    jar: InlineArray[Scalar[DTYPE], R_SIZE],
    mut force: InlineArray[Scalar[DTYPE], R_SIZE],
    mut state: InlineArray[Int, R_SIZE],
    mut cost: Scalar[DTYPE],
):
    """Compute constraint forces, state, and cost from jar = J*qacc - aref.

    Handles all constraint types including elliptic cone contacts with 3-zone
    logic matching MuJoCo's mj_constraintUpdate_impl:
    - Top zone (N >= 0, N >= mu*T): satisfied, force=0
    - Bottom zone (N + mu*T <= 0): full quadratic, force=-D*jar
    - Middle zone: cone projection, force from projected cone boundary

    For non-contact constraints:
    - Equality: always quadratic (bilateral)
    - Limit/Pyramid: inequality (jar >= 0 → satisfied, else quadratic)
    """
    cost = Scalar[DTYPE](0)

    var num_normals = constraints.num_normals
    var num_friction = constraints.num_friction
    var friction_start = num_normals
    var limits_start = num_normals + num_friction
    var equality_start = limits_start + constraints.num_limits

    # Initialize all rows
    for i in range(constraints.num_rows):
        force[i] = Scalar[DTYPE](0)
        state[i] = PRIMAL_SATISFIED

    # === Process contacts as groups (normal + friction children) ===
    var fric_idx = 0
    for n in range(num_normals):
        var D_n = primal_D(constraints.rows[n].inv_K_imp, constraints.rows[n].K)
        var N = jar[n]

        # Find friction children for this normal
        var group_size = 0
        while fric_idx + group_size < num_friction:
            if (
                constraints.rows[
                    friction_start + fric_idx + group_size
                ].friction_parent
                != n
            ):
                break
            group_size += 1

        if group_size == 0:
            # Frictionless contact — simple inequality
            if N >= Scalar[DTYPE](0):
                pass  # Already satisfied
            else:
                force[n] = -D_n * jar[n]
                state[n] = PRIMAL_QUADRATIC
                cost += Scalar[DTYPE](0.5) * D_n * jar[n] * jar[n]
        else:
            # Compute tangent magnitude T
            var T_sq: Scalar[DTYPE] = 0
            for g in range(group_size):
                var fr = friction_start + fric_idx + g
                T_sq += jar[fr] * jar[fr]
            var T = sqrt(T_sq)

            # Get mu from first friction child (all should have same mu for slide)
            var mu = constraints.rows[
                friction_start + fric_idx
            ].friction_coef

            # Three-zone logic (MuJoCo mj_constraintUpdate_impl)
            if N >= Scalar[DTYPE](0) and N * N >= mu * mu * T_sq:
                # Top zone: satisfied — everything stays zero
                pass

            elif (N + mu * T) <= Scalar[DTYPE](0):
                # Bottom zone: full quadratic for all rows
                force[n] = -D_n * jar[n]
                state[n] = PRIMAL_QUADRATIC
                cost += Scalar[DTYPE](0.5) * D_n * jar[n] * jar[n]
                for g in range(group_size):
                    var fr = friction_start + fric_idx + g
                    var D_f = primal_D(
                        constraints.rows[fr].inv_K_imp,
                        constraints.rows[fr].K,
                    )
                    force[fr] = -D_f * jar[fr]
                    state[fr] = PRIMAL_QUADRATIC
                    cost += Scalar[DTYPE](0.5) * D_f * jar[fr] * jar[fr]

            else:
                # Middle zone: cone projection
                var Dm = D_n / (mu * mu * (Scalar[DTYPE](1) + mu * mu))
                var s = N - mu * T
                force[n] = -Dm * s
                state[n] = PRIMAL_CONE
                var T_safe = max(T, Scalar[DTYPE](PRIMAL_MINVAL))
                for g in range(group_size):
                    var fr = friction_start + fric_idx + g
                    force[fr] = Dm * mu * s * jar[fr] / T_safe
                    state[fr] = PRIMAL_CONE
                cost += Scalar[DTYPE](0.5) * Dm * s * s

        fric_idx += group_size

    # === Process limit constraints (inequality) ===
    for r_off in range(constraints.num_limits):
        var r = limits_start + r_off
        var D_r = primal_D(
            constraints.rows[r].inv_K_imp, constraints.rows[r].K
        )
        if jar[r] >= Scalar[DTYPE](0):
            pass  # Satisfied
        else:
            force[r] = -D_r * jar[r]
            state[r] = PRIMAL_QUADRATIC
            cost += Scalar[DTYPE](0.5) * D_r * jar[r] * jar[r]

    # === Process equality constraints (bilateral, always quadratic) ===
    for r_off in range(constraints.num_equality):
        var r = equality_start + r_off
        var D_r = primal_D(
            constraints.rows[r].inv_K_imp, constraints.rows[r].K
        )
        force[r] = -D_r * jar[r]
        state[r] = PRIMAL_QUADRATIC
        cost += Scalar[DTYPE](0.5) * D_r * jar[r] * jar[r]


@always_inline
fn compute_jar[
    DTYPE: DType,
    MAX_ROWS: Int,
    NV: Int,
    V_SIZE: Int,
    R_SIZE: Int,
](
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    qacc: InlineArray[Scalar[DTYPE], V_SIZE],
    mut jar: InlineArray[Scalar[DTYPE], R_SIZE],
):
    """Compute jar[i] = J[i,:] . qacc - aref[i] for all constraints.

    aref = -bias (stored as bias in constraint rows), so jar = J*qacc + bias.
    """
    for r in range(constraints.num_rows):
        var val: Scalar[DTYPE] = 0
        for i in range(NV):
            val += constraints.J[r * NV + i] * qacc[i]
        jar[r] = val + constraints.rows[r].bias


@always_inline
fn compute_qfrc_constraint[
    DTYPE: DType,
    MAX_ROWS: Int,
    NV: Int,
    V_SIZE: Int,
    R_SIZE: Int,
](
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    force: InlineArray[Scalar[DTYPE], R_SIZE],
    mut qfrc: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Compute qfrc_constraint = J^T * force."""
    for i in range(NV):
        qfrc[i] = Scalar[DTYPE](0)
    for r in range(constraints.num_rows):
        if force[r] == Scalar[DTYPE](0):
            continue
        for i in range(NV):
            qfrc[i] += constraints.J[r * NV + i] * force[r]


@always_inline
fn compute_gauss_cost[
    DTYPE: DType,
    NV: Int,
    V_SIZE: Int,
](
    Ma: InlineArray[Scalar[DTYPE], V_SIZE],
    qfrc_smooth: InlineArray[Scalar[DTYPE], V_SIZE],
    qacc: InlineArray[Scalar[DTYPE], V_SIZE],
    qacc_smooth: InlineArray[Scalar[DTYPE], V_SIZE],
) -> Scalar[DTYPE]:
    """Compute Gauss cost = 0.5 * (Ma - qfrc_smooth) . (qacc - qacc_smooth)."""
    var cost_val: Scalar[DTYPE] = 0
    for i in range(NV):
        cost_val += (Ma[i] - qfrc_smooth[i]) * (qacc[i] - qacc_smooth[i])
    return Scalar[DTYPE](0.5) * cost_val


@always_inline
fn compute_total_cost[
    DTYPE: DType,
    MAX_ROWS: Int,
    NV: Int,
    V_SIZE: Int,
    R_SIZE: Int,
](
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    qacc: InlineArray[Scalar[DTYPE], V_SIZE],
    qacc_smooth: InlineArray[Scalar[DTYPE], V_SIZE],
    qfrc_smooth: InlineArray[Scalar[DTYPE], V_SIZE],
    M_hat: InlineArray[Scalar[DTYPE], _max_one[NV * NV]()],
) -> Scalar[DTYPE]:
    """Compute total primal cost at the given qacc.

    Total cost = Gauss cost + constraint cost.
    This evaluates the full cost function from scratch (no precomputation).
    Used by the linesearch for cone-aware cost evaluation.
    """
    comptime MR = _max_one[MAX_ROWS]()

    # Compute Ma = M * qacc
    var Ma = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        Ma[i] = Scalar[DTYPE](0)
        for j in range(NV):
            Ma[i] += M_hat[i * NV + j] * qacc[j]

    # Gauss cost
    var gauss = compute_gauss_cost[DTYPE, NV, V_SIZE](
        Ma, qfrc_smooth, qacc, qacc_smooth
    )

    # Compute jar = J*qacc + bias
    var jar = InlineArray[Scalar[DTYPE], R_SIZE](uninitialized=True)
    for i in range(R_SIZE):
        jar[i] = Scalar[DTYPE](0)
    compute_jar[DTYPE, MAX_ROWS, NV, V_SIZE, R_SIZE](constraints, qacc, jar)

    # Constraint cost via cone-aware update
    var force = InlineArray[Scalar[DTYPE], R_SIZE](uninitialized=True)
    var cstate = InlineArray[Int, R_SIZE](uninitialized=True)
    for i in range(R_SIZE):
        force[i] = Scalar[DTYPE](0)
        cstate[i] = PRIMAL_SATISFIED
    var cnstr_cost: Scalar[DTYPE] = 0
    constraint_update[DTYPE, MAX_ROWS, NV, R_SIZE](
        constraints, jar, force, cstate, cnstr_cost
    )

    return gauss + cnstr_cost


@always_inline
fn compute_primal_D_values[
    DTYPE: DType,
    MAX_ROWS: Int,
    NV: Int,
    V_SIZE: Int,
    R_SIZE: Int,
](
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    mut D_values: InlineArray[Scalar[DTYPE], R_SIZE],
):
    """Compute MuJoCo-matching D values using diagonal approximation.

    MuJoCo uses diagApprox = sum_j J[r,j]² / D_LDL[j] (where D_LDL is from
    the LDL factorization of M), NOT the exact Delassus diagonal J*M^{-1}*J^T.

    All rows in a cone contact group share the same diagApprox (the normal's),
    with friction R scaled by mu ratios (which for equal mu gives same D).

    Steps:
    1. LDL factorize M_hat → get D_LDL diagonal
    2. For each normal: diagApprox_n = sum_j J_n[j]² / D_LDL[j]
    3. Extract imp from the normal: imp = inv_K_imp * K (where K is exact)
    4. R_n = (1-imp)/imp * diagApprox_n, D_n = 1/R_n
    5. Friction rows get the same D (for equal mu)
    """
    comptime M_SIZE = _max_one[NV * NV]()
    var num_normals = constraints.num_normals
    var num_friction = constraints.num_friction
    var friction_start = num_normals
    var limits_start = num_normals + num_friction
    var equality_start = limits_start + constraints.num_limits

    # Initialize D_values to 0
    for i in range(R_SIZE):
        D_values[i] = Scalar[DTYPE](0)

    # LDL factorize M_hat → D_LDL diagonal
    var L_ldl = InlineArray[Scalar[DTYPE], M_SIZE](fill=Scalar[DTYPE](0))
    var D_ldl = InlineArray[Scalar[DTYPE], V_SIZE](fill=Scalar[DTYPE](0))

    for j in range(NV):
        var s: Scalar[DTYPE] = 0
        for k in range(j):
            s += L_ldl[j * NV + k] * L_ldl[j * NV + k] * D_ldl[k]
        D_ldl[j] = constraints.M_hat[j * NV + j] - s
        for i_row in range(j + 1, NV):
            var s2: Scalar[DTYPE] = 0
            for k in range(j):
                s2 += L_ldl[i_row * NV + k] * L_ldl[j * NV + k] * D_ldl[k]
            if abs(D_ldl[j]) > Scalar[DTYPE](PRIMAL_MINVAL):
                L_ldl[i_row * NV + j] = (
                    constraints.M_hat[i_row * NV + j] - s2
                ) / D_ldl[j]
            else:
                L_ldl[i_row * NV + j] = Scalar[DTYPE](0)

    # Compute diagApprox per row using MuJoCo convention:
    # all rows in a contact group use the normal row's diagApprox
    var fric_idx = 0
    for n in range(num_normals):
        # Compute diagApprox for this normal row
        var diag_n: Scalar[DTYPE] = 0
        for j in range(NV):
            if abs(D_ldl[j]) > Scalar[DTYPE](PRIMAL_MINVAL):
                diag_n += (
                    constraints.J[n * NV + j]
                    * constraints.J[n * NV + j]
                    / D_ldl[j]
                )

        # Extract imp from normal row: imp = inv_K_imp * K
        var imp = constraints.rows[n].inv_K_imp * constraints.rows[n].K
        if imp < Scalar[DTYPE](PRIMAL_MINVAL):
            imp = Scalar[DTYPE](PRIMAL_MINVAL)
        if imp > Scalar[DTYPE](1.0) - Scalar[DTYPE](PRIMAL_MINVAL):
            imp = Scalar[DTYPE](1.0) - Scalar[DTYPE](PRIMAL_MINVAL)

        # R = (1-imp)/imp * diagApprox, D = 1/R
        var R_n = (Scalar[DTYPE](1) - imp) / imp * diag_n
        if R_n < Scalar[DTYPE](PRIMAL_MINVAL):
            R_n = Scalar[DTYPE](PRIMAL_MINVAL)
        D_values[n] = Scalar[DTYPE](1) / R_n

        # Find friction children — they get the same D (for equal mu)
        var group_size = 0
        while fric_idx + group_size < num_friction:
            if (
                constraints.rows[
                    friction_start + fric_idx + group_size
                ].friction_parent
                != n
            ):
                break
            group_size += 1

        for g in range(group_size):
            var fr = friction_start + fric_idx + g
            # For equal mu, friction D = normal D (MuJoCo convention)
            # For different mu, R_f = R_n * mu_n²/mu_f² → D_f = D_n * mu_f²/mu_n²
            # Since our friction coef is per-direction mu:
            D_values[fr] = D_values[n]

        fric_idx += group_size

    # Limits: use their own diagApprox
    for r_off in range(constraints.num_limits):
        var r = limits_start + r_off
        var diag_r: Scalar[DTYPE] = 0
        for j in range(NV):
            if abs(D_ldl[j]) > Scalar[DTYPE](PRIMAL_MINVAL):
                diag_r += (
                    constraints.J[r * NV + j]
                    * constraints.J[r * NV + j]
                    / D_ldl[j]
                )

        var imp = constraints.rows[r].inv_K_imp * constraints.rows[r].K
        if imp < Scalar[DTYPE](PRIMAL_MINVAL):
            imp = Scalar[DTYPE](PRIMAL_MINVAL)
        if imp > Scalar[DTYPE](1.0) - Scalar[DTYPE](PRIMAL_MINVAL):
            imp = Scalar[DTYPE](1.0) - Scalar[DTYPE](PRIMAL_MINVAL)

        var R_r = (Scalar[DTYPE](1) - imp) / imp * diag_r
        if R_r < Scalar[DTYPE](PRIMAL_MINVAL):
            R_r = Scalar[DTYPE](PRIMAL_MINVAL)
        D_values[r] = Scalar[DTYPE](1) / R_r

    # Equality: use their own diagApprox
    for r_off in range(constraints.num_equality):
        var r = equality_start + r_off
        var diag_r: Scalar[DTYPE] = 0
        for j in range(NV):
            if abs(D_ldl[j]) > Scalar[DTYPE](PRIMAL_MINVAL):
                diag_r += (
                    constraints.J[r * NV + j]
                    * constraints.J[r * NV + j]
                    / D_ldl[j]
                )

        var imp = constraints.rows[r].inv_K_imp * constraints.rows[r].K
        if imp < Scalar[DTYPE](PRIMAL_MINVAL):
            imp = Scalar[DTYPE](PRIMAL_MINVAL)
        if imp > Scalar[DTYPE](1.0) - Scalar[DTYPE](PRIMAL_MINVAL):
            imp = Scalar[DTYPE](1.0) - Scalar[DTYPE](PRIMAL_MINVAL)

        var R_r = (Scalar[DTYPE](1) - imp) / imp * diag_r
        if R_r < Scalar[DTYPE](PRIMAL_MINVAL):
            R_r = Scalar[DTYPE](PRIMAL_MINVAL)
        D_values[r] = Scalar[DTYPE](1) / R_r


@always_inline
fn constraint_update_with_D[
    DTYPE: DType,
    MAX_ROWS: Int,
    NV: Int,
    R_SIZE: Int,
](
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    jar: InlineArray[Scalar[DTYPE], R_SIZE],
    D_values: InlineArray[Scalar[DTYPE], R_SIZE],
    mut force: InlineArray[Scalar[DTYPE], R_SIZE],
    mut state: InlineArray[Int, R_SIZE],
    mut cost: Scalar[DTYPE],
):
    """Cone-aware constraint_update using precomputed D values.

    Same 3-zone logic as constraint_update but uses externally-provided D values
    (computed via MuJoCo's diagonal approximation) instead of primal_D.
    """
    cost = Scalar[DTYPE](0)

    var num_normals = constraints.num_normals
    var num_friction = constraints.num_friction
    var friction_start = num_normals
    var limits_start = num_normals + num_friction
    var equality_start = limits_start + constraints.num_limits

    # Initialize all rows
    for i in range(constraints.num_rows):
        force[i] = Scalar[DTYPE](0)
        state[i] = PRIMAL_SATISFIED

    # Process contacts as groups
    var fric_idx = 0
    for n in range(num_normals):
        var D_n = D_values[n]
        var N = jar[n]

        var group_size = 0
        while fric_idx + group_size < num_friction:
            if (
                constraints.rows[
                    friction_start + fric_idx + group_size
                ].friction_parent
                != n
            ):
                break
            group_size += 1

        if group_size == 0:
            if N >= Scalar[DTYPE](0):
                pass
            else:
                force[n] = -D_n * jar[n]
                state[n] = PRIMAL_QUADRATIC
                cost += Scalar[DTYPE](0.5) * D_n * jar[n] * jar[n]
        else:
            var T_sq: Scalar[DTYPE] = 0
            for g in range(group_size):
                var fr = friction_start + fric_idx + g
                T_sq += jar[fr] * jar[fr]
            var T = sqrt(T_sq)
            var mu = constraints.rows[
                friction_start + fric_idx
            ].friction_coef

            if N >= Scalar[DTYPE](0) and N * N >= mu * mu * T_sq:
                pass
            elif (N + mu * T) <= Scalar[DTYPE](0):
                force[n] = -D_n * jar[n]
                state[n] = PRIMAL_QUADRATIC
                cost += Scalar[DTYPE](0.5) * D_n * jar[n] * jar[n]
                for g in range(group_size):
                    var fr = friction_start + fric_idx + g
                    var D_f = D_values[fr]
                    force[fr] = -D_f * jar[fr]
                    state[fr] = PRIMAL_QUADRATIC
                    cost += Scalar[DTYPE](0.5) * D_f * jar[fr] * jar[fr]
            else:
                var Dm = D_n / (mu * mu * (Scalar[DTYPE](1) + mu * mu))
                var s = N - mu * T
                force[n] = -Dm * s
                state[n] = PRIMAL_CONE
                var T_safe = max(T, Scalar[DTYPE](PRIMAL_MINVAL))
                for g in range(group_size):
                    var fr = friction_start + fric_idx + g
                    force[fr] = Dm * mu * s * jar[fr] / T_safe
                    state[fr] = PRIMAL_CONE
                cost += Scalar[DTYPE](0.5) * Dm * s * s

        fric_idx += group_size

    # Limits
    for r_off in range(constraints.num_limits):
        var r = limits_start + r_off
        var D_r = D_values[r]
        if jar[r] >= Scalar[DTYPE](0):
            pass
        else:
            force[r] = -D_r * jar[r]
            state[r] = PRIMAL_QUADRATIC
            cost += Scalar[DTYPE](0.5) * D_r * jar[r] * jar[r]

    # Equality
    for r_off in range(constraints.num_equality):
        var r = equality_start + r_off
        var D_r = D_values[r]
        force[r] = -D_r * jar[r]
        state[r] = PRIMAL_QUADRATIC
        cost += Scalar[DTYPE](0.5) * D_r * jar[r] * jar[r]


@always_inline
fn compute_total_cost_with_D[
    DTYPE: DType,
    MAX_ROWS: Int,
    NV: Int,
    V_SIZE: Int,
    R_SIZE: Int,
](
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    D_values: InlineArray[Scalar[DTYPE], R_SIZE],
    qacc: InlineArray[Scalar[DTYPE], V_SIZE],
    qacc_smooth: InlineArray[Scalar[DTYPE], V_SIZE],
    qfrc_smooth: InlineArray[Scalar[DTYPE], V_SIZE],
    M_hat: InlineArray[Scalar[DTYPE], _max_one[NV * NV]()],
) -> Scalar[DTYPE]:
    """Compute total primal cost using precomputed D values."""
    var Ma = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        Ma[i] = Scalar[DTYPE](0)
        for j in range(NV):
            Ma[i] += M_hat[i * NV + j] * qacc[j]

    var gauss = compute_gauss_cost[DTYPE, NV, V_SIZE](
        Ma, qfrc_smooth, qacc, qacc_smooth
    )

    var jar = InlineArray[Scalar[DTYPE], R_SIZE](uninitialized=True)
    for i in range(R_SIZE):
        jar[i] = Scalar[DTYPE](0)
    compute_jar[DTYPE, MAX_ROWS, NV, V_SIZE, R_SIZE](constraints, qacc, jar)

    var force = InlineArray[Scalar[DTYPE], R_SIZE](uninitialized=True)
    var cstate = InlineArray[Int, R_SIZE](uninitialized=True)
    for i in range(R_SIZE):
        force[i] = Scalar[DTYPE](0)
        cstate[i] = PRIMAL_SATISFIED
    var cnstr_cost: Scalar[DTYPE] = 0
    constraint_update_with_D[DTYPE, MAX_ROWS, NV, R_SIZE](
        constraints, jar, D_values, force, cstate, cnstr_cost
    )

    return gauss + cnstr_cost


@always_inline
fn primal_linesearch_with_D[
    DTYPE: DType,
    MAX_ROWS: Int,
    NV: Int,
    V_SIZE: Int,
    R_SIZE: Int,
](
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    D_values: InlineArray[Scalar[DTYPE], R_SIZE],
    qacc: InlineArray[Scalar[DTYPE], V_SIZE],
    qacc_smooth: InlineArray[Scalar[DTYPE], V_SIZE],
    qfrc_smooth: InlineArray[Scalar[DTYPE], V_SIZE],
    Ma: InlineArray[Scalar[DTYPE], V_SIZE],
    Mv: InlineArray[Scalar[DTYPE], V_SIZE],
    search: InlineArray[Scalar[DTYPE], V_SIZE],
    jar: InlineArray[Scalar[DTYPE], R_SIZE],
    force: InlineArray[Scalar[DTYPE], R_SIZE],
    tolerance: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """Linesearch using precomputed D values."""
    var snorm: Scalar[DTYPE] = 0
    for i in range(NV):
        snorm += search[i] * search[i]
    snorm = sqrt(snorm)
    if snorm < Scalar[DTYPE](PRIMAL_MINVAL):
        return Scalar[DTYPE](0)

    var d1: Scalar[DTYPE] = 0
    for i in range(NV):
        var grad_i = Ma[i] - qfrc_smooth[i]
        d1 += grad_i * search[i]
    for r in range(constraints.num_rows):
        if force[r] == Scalar[DTYPE](0):
            continue
        for i in range(NV):
            d1 -= constraints.J[r * NV + i] * force[r] * search[i]

    if d1 >= Scalar[DTYPE](0):
        return Scalar[DTYPE](0)

    var d2: Scalar[DTYPE] = 0
    for i in range(NV):
        d2 += Mv[i] * search[i]
    for r in range(constraints.num_rows):
        var Jv: Scalar[DTYPE] = 0
        for i in range(NV):
            Jv += constraints.J[r * NV + i] * search[i]
        d2 += D_values[r] * Jv * Jv

    if d2 < Scalar[DTYPE](PRIMAL_MINVAL):
        d2 = Scalar[DTYPE](PRIMAL_MINVAL)

    var alpha = -d1 / d2

    var cost0 = compute_total_cost_with_D[
        DTYPE, MAX_ROWS, NV, V_SIZE, R_SIZE
    ](
        constraints,
        D_values,
        qacc,
        qacc_smooth,
        qfrc_smooth,
        constraints.M_hat,
    )

    comptime ARMIJO_C: Float64 = 1e-4
    comptime BACKTRACK_BETA: Float64 = 0.5

    var qacc_trial = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for _ in range(PRIMAL_MAX_LINESEARCH):
        for i in range(NV):
            qacc_trial[i] = qacc[i] + alpha * search[i]
        var cost_trial = compute_total_cost_with_D[
            DTYPE, MAX_ROWS, NV, V_SIZE, R_SIZE
        ](
            constraints,
            D_values,
            qacc_trial,
            qacc_smooth,
            qfrc_smooth,
            constraints.M_hat,
        )
        if cost_trial <= cost0 + Scalar[DTYPE](ARMIJO_C) * alpha * d1:
            return alpha
        alpha *= Scalar[DTYPE](BACKTRACK_BETA)

    return alpha


@always_inline
fn primal_linesearch[
    DTYPE: DType,
    MAX_ROWS: Int,
    NV: Int,
    V_SIZE: Int,
    R_SIZE: Int,
](
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    qacc: InlineArray[Scalar[DTYPE], V_SIZE],
    qacc_smooth: InlineArray[Scalar[DTYPE], V_SIZE],
    qfrc_smooth: InlineArray[Scalar[DTYPE], V_SIZE],
    Ma: InlineArray[Scalar[DTYPE], V_SIZE],
    Mv: InlineArray[Scalar[DTYPE], V_SIZE],
    search: InlineArray[Scalar[DTYPE], V_SIZE],
    jar: InlineArray[Scalar[DTYPE], R_SIZE],
    force: InlineArray[Scalar[DTYPE], R_SIZE],
    tolerance: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """Cone-aware linesearch using cost evaluation + Armijo backtracking.

    Finds alpha minimizing cost(qacc + alpha*search) using:
    1. Initial Newton step from gradient and Hessian
    2. Armijo backtracking to ensure sufficient decrease

    Unlike the quadratic-polynomial linesearch, this correctly handles
    cone contacts where the cost function is piecewise non-quadratic.
    """
    # Compute search norm
    var snorm: Scalar[DTYPE] = 0
    for i in range(NV):
        snorm += search[i] * search[i]
    snorm = sqrt(snorm)
    if snorm < Scalar[DTYPE](PRIMAL_MINVAL):
        return Scalar[DTYPE](0)

    # Compute directional derivative at alpha=0
    # d1 = grad . search where grad = Ma - qfrc_smooth - J^T*force
    var d1: Scalar[DTYPE] = 0
    for i in range(NV):
        var grad_i = Ma[i] - qfrc_smooth[i]
        d1 += grad_i * search[i]
    # Subtract J^T*force contribution
    for r in range(constraints.num_rows):
        if force[r] == Scalar[DTYPE](0):
            continue
        for i in range(NV):
            d1 -= constraints.J[r * NV + i] * force[r] * search[i]

    # If not a descent direction, return 0
    if d1 >= Scalar[DTYPE](0):
        return Scalar[DTYPE](0)

    # Second derivative (Gauss + quadratic approximation of active constraints)
    # d2 = search^T * H * search where H ≈ M + J^T*D*J (for active rows)
    var d2: Scalar[DTYPE] = 0
    # Gauss part: search^T * M * search
    for i in range(NV):
        d2 += Mv[i] * search[i]

    # Constraint part: approximate with D * (J*search)^2 for active rows
    for r in range(constraints.num_rows):
        var Jv: Scalar[DTYPE] = 0
        for i in range(NV):
            Jv += constraints.J[r * NV + i] * search[i]
        var D_r = primal_D(
            constraints.rows[r].inv_K_imp, constraints.rows[r].K
        )
        # Use D_r for quadratic rows, Dm for cone rows
        d2 += D_r * Jv * Jv

    if d2 < Scalar[DTYPE](PRIMAL_MINVAL):
        d2 = Scalar[DTYPE](PRIMAL_MINVAL)

    # Initial Newton step
    var alpha = -d1 / d2

    # Current cost at alpha=0
    var cost0 = compute_total_cost[DTYPE, MAX_ROWS, NV, V_SIZE, R_SIZE](
        constraints,
        qacc,
        qacc_smooth,
        qfrc_smooth,
        constraints.M_hat,
    )

    # Armijo backtracking
    comptime ARMIJO_C: Float64 = 1e-4
    comptime BACKTRACK_BETA: Float64 = 0.5

    var qacc_trial = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for _ in range(PRIMAL_MAX_LINESEARCH):
        # Evaluate cost at trial alpha
        for i in range(NV):
            qacc_trial[i] = qacc[i] + alpha * search[i]

        var cost_trial = compute_total_cost[
            DTYPE, MAX_ROWS, NV, V_SIZE, R_SIZE
        ](
            constraints,
            qacc_trial,
            qacc_smooth,
            qfrc_smooth,
            constraints.M_hat,
        )

        # Armijo condition: f(alpha) <= f(0) + c * alpha * d1
        if cost_trial <= cost0 + Scalar[DTYPE](ARMIJO_C) * alpha * d1:
            return alpha

        # Backtrack
        alpha *= Scalar[DTYPE](BACKTRACK_BETA)

    return alpha
