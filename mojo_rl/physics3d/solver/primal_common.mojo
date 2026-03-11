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

from std.math import sqrt, max
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
fn primal_D[
    DTYPE: DType
](inv_K_imp: Scalar[DTYPE], K: Scalar[DTYPE],) -> Scalar[DTYPE]:
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
    jar: List[Scalar[DTYPE]],
    mut force: List[Scalar[DTYPE]],
    mut state: List[Int],
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
            var mu = constraints.rows[friction_start + fric_idx].friction_coef

            # Three-zone logic (MuJoCo mj_constraintUpdate_impl)
            if N >= Scalar[DTYPE](0) and N * N >= mu * mu * T_sq:
                # Top zone: satisfied — everything stays zero
                pass

            elif (mu * N + T) <= Scalar[DTYPE](0):
                # Bottom zone (polar cone in U-space): full quadratic for all rows
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
                # MuJoCo U-space: U=(con->mu*jar_n, friction*jar_f)
                # con->mu = mu_s (per-direction friction, for impratio=1)
                # Dm = D_n/(1 + mu^2) in jar-space
                var mu_sq_combined = mu * mu
                var Dm = D_n / (Scalar[DTYPE](1) + mu_sq_combined)
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
        var D_r = primal_D(constraints.rows[r].inv_K_imp, constraints.rows[r].K)
        if jar[r] >= Scalar[DTYPE](0):
            pass  # Satisfied
        else:
            force[r] = -D_r * jar[r]
            state[r] = PRIMAL_QUADRATIC
            cost += Scalar[DTYPE](0.5) * D_r * jar[r] * jar[r]

    # === Process equality constraints (bilateral, always quadratic) ===
    for r_off in range(constraints.num_equality):
        var r = equality_start + r_off
        var D_r = primal_D(constraints.rows[r].inv_K_imp, constraints.rows[r].K)
        force[r] = -D_r * jar[r]
        state[r] = PRIMAL_QUADRATIC
        cost += Scalar[DTYPE](0.5) * D_r * jar[r] * jar[r]


@always_inline
fn compute_jar[
    DTYPE: DType,
    MAX_ROWS: Int,
    NV: Int,
](
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    qacc: List[Scalar[DTYPE]],
    mut jar: List[Scalar[DTYPE]],
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
    force: List[Scalar[DTYPE]],
    mut qfrc: List[Scalar[DTYPE]],
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
    Ma: List[Scalar[DTYPE]],
    qfrc_smooth: List[Scalar[DTYPE]],
    qacc: List[Scalar[DTYPE]],
    qacc_smooth: List[Scalar[DTYPE]],
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
    qacc: List[Scalar[DTYPE]],
    qacc_smooth: List[Scalar[DTYPE]],
    qfrc_smooth: List[Scalar[DTYPE]],
    M_hat: List[Scalar[DTYPE]],
) -> Scalar[DTYPE]:
    """Compute total primal cost at the given qacc.

    Total cost = Gauss cost + constraint cost.
    This evaluates the full cost function from scratch (no precomputation).
    Used by the linesearch for cone-aware cost evaluation.
    """
    comptime MR = _max_one[MAX_ROWS]()

    # Compute Ma = M * qacc
    var Ma = List[Scalar[DTYPE]](capacity=V_SIZE)
    for i in range(V_SIZE):
        Ma.append(Scalar[DTYPE](0))
        for j in range(NV):
            Ma[i] += M_hat[i * NV + j] * qacc[j]

    # Gauss cost
    var gauss = compute_gauss_cost[DTYPE, NV, V_SIZE](
        Ma, qfrc_smooth, qacc, qacc_smooth
    )

    # Compute jar = J*qacc + bias
    var jar = List[Scalar[DTYPE]](capacity=R_SIZE)
    for _ in range(R_SIZE):
        jar.append(Scalar[DTYPE](0))
    compute_jar[DTYPE, MAX_ROWS, NV](constraints, qacc, jar)

    # Constraint cost via cone-aware update
    var force = List[Scalar[DTYPE]](capacity=R_SIZE)
    for _ in range(R_SIZE):
        force.append(Scalar[DTYPE](0))
    var cstate = List[Int](capacity=R_SIZE)
    for _ in range(R_SIZE):
        cstate.append(PRIMAL_SATISFIED)
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
    jar: List[Scalar[DTYPE]],
    D_values: List[Scalar[DTYPE]],
    mut force: List[Scalar[DTYPE]],
    mut state: List[Int],
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
            var mu = constraints.rows[friction_start + fric_idx].friction_coef

            if N >= Scalar[DTYPE](0) and N * N >= mu * mu * T_sq:
                pass
            elif (mu * N + T) <= Scalar[DTYPE](0):
                # Bottom zone (polar cone in U-space): full quadratic
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
                # Middle zone: cone projection
                # Dm = D_n / (1 + mu^2) in jar-space (no group_size factor)
                var mu_sq_combined = mu * mu
                var Dm = D_n / (Scalar[DTYPE](1) + mu_sq_combined)
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


@no_inline
fn compute_total_cost_with_D[
    DTYPE: DType,
    MAX_ROWS: Int,
    NV: Int,
    V_SIZE: Int,
    R_SIZE: Int,
](
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    D_values: List[Scalar[DTYPE]],
    qacc: List[Scalar[DTYPE]],
    qacc_smooth: List[Scalar[DTYPE]],
    qfrc_smooth: List[Scalar[DTYPE]],
    M_hat: List[Scalar[DTYPE]],
) -> Scalar[DTYPE]:
    """Compute total primal cost using precomputed D values."""
    var Ma = List[Scalar[DTYPE]](capacity=V_SIZE)
    for i in range(V_SIZE):
        Ma.append(Scalar[DTYPE](0))
        for j in range(NV):
            Ma[i] += M_hat[i * NV + j] * qacc[j]

    var gauss = compute_gauss_cost[DTYPE, NV, V_SIZE](
        Ma, qfrc_smooth, qacc, qacc_smooth
    )

    var jar = List[Scalar[DTYPE]](capacity=R_SIZE)
    for _ in range(R_SIZE):
        jar.append(Scalar[DTYPE](0))
    compute_jar[DTYPE, MAX_ROWS, NV](constraints, qacc, jar)

    var force = List[Scalar[DTYPE]](capacity=R_SIZE)
    for _ in range(R_SIZE):
        force.append(Scalar[DTYPE](0))
    var cstate = List[Int](capacity=R_SIZE)
    for _ in range(R_SIZE):
        cstate.append(PRIMAL_SATISFIED)
    var cnstr_cost: Scalar[DTYPE] = 0
    constraint_update_with_D[DTYPE, MAX_ROWS, NV, R_SIZE](
        constraints, jar, D_values, force, cstate, cnstr_cost
    )

    return gauss + cnstr_cost


@no_inline
fn primal_linesearch_with_D[
    DTYPE: DType,
    MAX_ROWS: Int,
    NV: Int,
    V_SIZE: Int,
    R_SIZE: Int,
](
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    D_values: List[Scalar[DTYPE]],
    qacc: List[Scalar[DTYPE]],
    qacc_smooth: List[Scalar[DTYPE]],
    qfrc_smooth: List[Scalar[DTYPE]],
    Ma: List[Scalar[DTYPE]],
    Mv: List[Scalar[DTYPE]],
    search: List[Scalar[DTYPE]],
    jar: List[Scalar[DTYPE]],
    force: List[Scalar[DTYPE]],
    tolerance: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """Newton-based linesearch matching MuJoCo's PrimalSearch.

    Uses analytical 1st and 2nd derivatives of cost(alpha) to do Newton
    root-finding on dcost/dalpha = 0. This handles zone transitions correctly
    by recomputing zones at each alpha evaluation.

    Three phases:
    1. Initial Newton step from alpha=0
    2. One-sided Newton pursuit (follow derivative sign)
    3. Bracketed Newton refinement when sign change detected

    Reference: engine_solver.c PrimalSearch + PrimalEval
    """
    var snorm: Scalar[DTYPE] = 0
    for i in range(NV):
        snorm += search[i] * search[i]
    snorm = sqrt(snorm)
    if snorm < Scalar[DTYPE](PRIMAL_MINVAL):
        return Scalar[DTYPE](0)

    var scale = Scalar[DTYPE](1.0)
    var Mdiag_sum: Scalar[DTYPE] = 0
    for i in range(NV):
        Mdiag_sum += constraints.M_hat[i * NV + i]
    if Mdiag_sum > Scalar[DTYPE](PRIMAL_MINVAL):
        scale = Scalar[DTYPE](1.0) / Mdiag_sum
    var gtol = tolerance * snorm / scale

    # Pre-compute Jv = J * search for all rows
    var Jv = InlineArray[Scalar[DTYPE], R_SIZE](uninitialized=True)
    for r in range(constraints.num_rows):
        Jv[r] = Scalar[DTYPE](0)
        for i in range(NV):
            Jv[r] += constraints.J[r * NV + i] * search[i]
    for r in range(constraints.num_rows, R_SIZE):
        Jv[r] = Scalar[DTYPE](0)

    # Pre-compute Gauss coefficients: cost_gauss(alpha) = 0.5*a*alpha^2 + b*alpha + c
    # where a = search . M . search, b = (Ma - qfrc_smooth) . search
    var gauss_a: Scalar[DTYPE] = 0  # search . M . search
    var gauss_b: Scalar[DTYPE] = 0  # (Ma - qfrc_smooth) . search
    for i in range(NV):
        gauss_a += Mv[i] * search[i]
        gauss_b += (Ma[i] - qfrc_smooth[i]) * search[i]

    # === PrimalEval: compute d1 (derivative) and d2 (curvature) at given alpha ===
    # Returns (d1, d2, cost)
    # Inline to avoid closure capture issues

    # Evaluate at alpha=0 (p0)
    var p0_d1: Scalar[DTYPE] = gauss_b  # Gauss derivative at alpha=0
    var p0_d2: Scalar[DTYPE] = gauss_a  # Gauss curvature

    var num_normals = constraints.num_normals
    var num_friction = constraints.num_friction
    var friction_start = num_normals

    # Add constraint contributions at alpha=0
    var fric_idx_init = 0
    for n_init in range(num_normals):
        var group_sz_init = 0
        while fric_idx_init + group_sz_init < num_friction:
            if (
                constraints.rows[
                    friction_start + fric_idx_init + group_sz_init
                ].friction_parent
                != n_init
            ):
                break
            group_sz_init += 1

        var N0 = jar[n_init]
        var mu_init = constraints.rows[
            friction_start + fric_idx_init
        ].friction_coef if group_sz_init > 0 else Scalar[DTYPE](0)

        if group_sz_init == 0:
            # Frictionless normal: d/dalpha of 0.5*D*jar^2 = D*jar*Jv, d2 = D*Jv^2
            if N0 < Scalar[DTYPE](0):
                p0_d1 += D_values[n_init] * jar[n_init] * Jv[n_init]
                p0_d2 += D_values[n_init] * Jv[n_init] * Jv[n_init]
        else:
            var T0_sq: Scalar[DTYPE] = 0
            for g_init in range(group_sz_init):
                var fr_init = friction_start + fric_idx_init + g_init
                T0_sq += jar[fr_init] * jar[fr_init]
            var T0 = sqrt(T0_sq)

            if N0 >= Scalar[DTYPE](0) and N0 * N0 >= mu_init * mu_init * T0_sq:
                pass  # TOP zone: no contribution
            elif (mu_init * N0 + T0) <= Scalar[DTYPE](0):
                # BOTTOM zone: per-row quadratic
                p0_d1 += D_values[n_init] * jar[n_init] * Jv[n_init]
                p0_d2 += D_values[n_init] * Jv[n_init] * Jv[n_init]
                for g_init in range(group_sz_init):
                    var fr_init = friction_start + fric_idx_init + g_init
                    p0_d1 += D_values[fr_init] * jar[fr_init] * Jv[fr_init]
                    p0_d2 += D_values[fr_init] * Jv[fr_init] * Jv[fr_init]
            else:
                # CONE zone: cone projection derivative
                var mu_sq_c = mu_init * mu_init
                var Dm = D_values[n_init] / (Scalar[DTYPE](1) + mu_sq_c)
                var T0_safe = max(T0, Scalar[DTYPE](PRIMAL_MINVAL))
                var s_val = N0 - mu_init * T0
                # ds/dalpha = Jv_n - mu * dT/dalpha
                # dT/dalpha = sum(jar_fj * Jv_fj) / T
                var dTda: Scalar[DTYPE] = 0
                for g_init in range(group_sz_init):
                    var fr_init = friction_start + fric_idx_init + g_init
                    dTda += jar[fr_init] * Jv[fr_init]
                dTda /= T0_safe
                var dsda = Jv[n_init] - mu_init * dTda
                # d(0.5*Dm*s^2)/dalpha = Dm*s*dsda
                p0_d1 += Dm * s_val * dsda
                # d2: Dm*(dsda^2 + s*d2sda2)
                # d2s/dalpha2 = -mu * (sum(Jv_fj^2) - dTda^2) / T
                var Jv_f_sq: Scalar[DTYPE] = 0
                for g_init in range(group_sz_init):
                    var fr_init = friction_start + fric_idx_init + g_init
                    Jv_f_sq += Jv[fr_init] * Jv[fr_init]
                var d2sda2 = -mu_init * (Jv_f_sq - dTda * dTda) / T0_safe
                p0_d2 += Dm * (dsda * dsda + s_val * d2sda2)

        fric_idx_init += group_sz_init

    # Limits at alpha=0
    var limits_start = num_normals + num_friction
    for l_init in range(constraints.num_limits):
        var r_init = limits_start + l_init
        if jar[r_init] < Scalar[DTYPE](0):
            p0_d1 += D_values[r_init] * jar[r_init] * Jv[r_init]
            p0_d2 += D_values[r_init] * Jv[r_init] * Jv[r_init]

    # Equality at alpha=0
    var eq_start = limits_start + constraints.num_limits
    for e_init in range(constraints.num_equality):
        var r_init = eq_start + e_init
        p0_d1 += D_values[r_init] * jar[r_init] * Jv[r_init]
        p0_d2 += D_values[r_init] * Jv[r_init] * Jv[r_init]

    # Ensure d2 > 0
    if p0_d2 < Scalar[DTYPE](PRIMAL_MINVAL):
        p0_d2 = Scalar[DTYPE](PRIMAL_MINVAL)

    # Check d1 direction
    if p0_d1 >= Scalar[DTYPE](0):
        return Scalar[DTYPE](0)

    # === Phase 1: Initial Newton step ===
    var alpha1 = -p0_d1 / p0_d2

    # Evaluate at alpha1: recompute jar, zones, derivatives
    var jar_a = InlineArray[Scalar[DTYPE], R_SIZE](uninitialized=True)
    for r in range(R_SIZE):
        jar_a[r] = Scalar[DTYPE](0)

    # Function to evaluate derivatives at a given alpha (inline)
    # Returns (d1, d2) for the given alpha value
    var eval_alpha = alpha1
    for r in range(constraints.num_rows):
        jar_a[r] = jar[r] + eval_alpha * Jv[r]

    var d1_eval: Scalar[DTYPE] = gauss_a * eval_alpha + gauss_b
    var d2_eval: Scalar[DTYPE] = gauss_a

    var fric_idx_ev = 0
    for n_ev in range(num_normals):
        var group_sz_ev = 0
        while fric_idx_ev + group_sz_ev < num_friction:
            if (
                constraints.rows[
                    friction_start + fric_idx_ev + group_sz_ev
                ].friction_parent
                != n_ev
            ):
                break
            group_sz_ev += 1

        var N_ev = jar_a[n_ev]
        var mu_ev = constraints.rows[
            friction_start + fric_idx_ev
        ].friction_coef if group_sz_ev > 0 else Scalar[DTYPE](0)

        if group_sz_ev == 0:
            if N_ev < Scalar[DTYPE](0):
                d1_eval += D_values[n_ev] * jar_a[n_ev] * Jv[n_ev]
                d2_eval += D_values[n_ev] * Jv[n_ev] * Jv[n_ev]
        else:
            var T_ev_sq: Scalar[DTYPE] = 0
            for g_ev in range(group_sz_ev):
                var fr_ev = friction_start + fric_idx_ev + g_ev
                T_ev_sq += jar_a[fr_ev] * jar_a[fr_ev]
            var T_ev = sqrt(T_ev_sq)

            if (
                N_ev >= Scalar[DTYPE](0)
                and N_ev * N_ev >= mu_ev * mu_ev * T_ev_sq
            ):
                pass
            elif (mu_ev * N_ev + T_ev) <= Scalar[DTYPE](0):
                d1_eval += D_values[n_ev] * jar_a[n_ev] * Jv[n_ev]
                d2_eval += D_values[n_ev] * Jv[n_ev] * Jv[n_ev]
                for g_ev in range(group_sz_ev):
                    var fr_ev = friction_start + fric_idx_ev + g_ev
                    d1_eval += D_values[fr_ev] * jar_a[fr_ev] * Jv[fr_ev]
                    d2_eval += D_values[fr_ev] * Jv[fr_ev] * Jv[fr_ev]
            else:
                var mu_sq_ev = mu_ev * mu_ev
                var Dm_ev = D_values[n_ev] / (Scalar[DTYPE](1) + mu_sq_ev)
                var T_ev_safe = max(T_ev, Scalar[DTYPE](PRIMAL_MINVAL))
                var s_ev = N_ev - mu_ev * T_ev
                var dTda_ev: Scalar[DTYPE] = 0
                for g_ev in range(group_sz_ev):
                    var fr_ev = friction_start + fric_idx_ev + g_ev
                    dTda_ev += jar_a[fr_ev] * Jv[fr_ev]
                dTda_ev /= T_ev_safe
                var dsda_ev = Jv[n_ev] - mu_ev * dTda_ev
                d1_eval += Dm_ev * s_ev * dsda_ev
                var Jv_f_sq_ev: Scalar[DTYPE] = 0
                for g_ev in range(group_sz_ev):
                    var fr_ev = friction_start + fric_idx_ev + g_ev
                    Jv_f_sq_ev += Jv[fr_ev] * Jv[fr_ev]
                var d2sda2_ev = (
                    -mu_ev * (Jv_f_sq_ev - dTda_ev * dTda_ev) / T_ev_safe
                )
                d2_eval += Dm_ev * (dsda_ev * dsda_ev + s_ev * d2sda2_ev)

        fric_idx_ev += group_sz_ev

    for l_ev in range(constraints.num_limits):
        var r_ev = limits_start + l_ev
        if jar_a[r_ev] < Scalar[DTYPE](0):
            d1_eval += D_values[r_ev] * jar_a[r_ev] * Jv[r_ev]
            d2_eval += D_values[r_ev] * Jv[r_ev] * Jv[r_ev]

    for e_ev in range(constraints.num_equality):
        var r_ev = eq_start + e_ev
        d1_eval += D_values[r_ev] * jar_a[r_ev] * Jv[r_ev]
        d2_eval += D_values[r_ev] * Jv[r_ev] * Jv[r_ev]

    if d2_eval < Scalar[DTYPE](PRIMAL_MINVAL):
        d2_eval = Scalar[DTYPE](PRIMAL_MINVAL)

    # Check convergence at alpha1
    if d1_eval * d1_eval < gtol * gtol:
        return alpha1

    # Save p1 state
    var p1_alpha = alpha1
    var p1_d1 = d1_eval
    var p1_d2 = d2_eval

    # Determine direction for one-sided search
    var dir_sign: Scalar[DTYPE] = -1 if p1_d1 > Scalar[DTYPE](0) else Scalar[
        DTYPE
    ](1)

    # === Phase 2: One-sided Newton pursuit ===
    # Follow the derivative until sign changes (bracket found)
    var p2_alpha = Scalar[DTYPE](0)
    var p2_d1 = p0_d1
    var p2_d2 = p0_d2
    var bracket_found = False

    for _ in range(PRIMAL_MAX_LINESEARCH):
        p2_alpha = p1_alpha
        p2_d1 = p1_d1
        p2_d2 = p1_d2

        # Newton step from p1
        if p1_d2 > Scalar[DTYPE](PRIMAL_MINVAL):
            p1_alpha = p1_alpha - p1_d1 / p1_d2
        else:
            p1_alpha = p1_alpha + dir_sign
        eval_alpha = p1_alpha

        # Re-evaluate at new alpha
        for r in range(constraints.num_rows):
            jar_a[r] = jar[r] + eval_alpha * Jv[r]

        d1_eval = gauss_a * eval_alpha + gauss_b
        d2_eval = gauss_a

        var fric_idx_p = 0
        for n_p in range(num_normals):
            var group_sz_p = 0
            while fric_idx_p + group_sz_p < num_friction:
                if (
                    constraints.rows[
                        friction_start + fric_idx_p + group_sz_p
                    ].friction_parent
                    != n_p
                ):
                    break
                group_sz_p += 1

            var N_p = jar_a[n_p]
            var mu_p = constraints.rows[
                friction_start + fric_idx_p
            ].friction_coef if group_sz_p > 0 else Scalar[DTYPE](0)

            if group_sz_p == 0:
                if N_p < Scalar[DTYPE](0):
                    d1_eval += D_values[n_p] * jar_a[n_p] * Jv[n_p]
                    d2_eval += D_values[n_p] * Jv[n_p] * Jv[n_p]
            else:
                var T_p_sq: Scalar[DTYPE] = 0
                for g_p in range(group_sz_p):
                    var fr_p = friction_start + fric_idx_p + g_p
                    T_p_sq += jar_a[fr_p] * jar_a[fr_p]
                var T_p = sqrt(T_p_sq)

                if (
                    N_p >= Scalar[DTYPE](0)
                    and N_p * N_p >= mu_p * mu_p * T_p_sq
                ):
                    pass
                elif (mu_p * N_p + T_p) <= Scalar[DTYPE](0):
                    d1_eval += D_values[n_p] * jar_a[n_p] * Jv[n_p]
                    d2_eval += D_values[n_p] * Jv[n_p] * Jv[n_p]
                    for g_p in range(group_sz_p):
                        var fr_p = friction_start + fric_idx_p + g_p
                        d1_eval += D_values[fr_p] * jar_a[fr_p] * Jv[fr_p]
                        d2_eval += D_values[fr_p] * Jv[fr_p] * Jv[fr_p]
                else:
                    var mu_sq_p = mu_p * mu_p
                    var Dm_p = D_values[n_p] / (Scalar[DTYPE](1) + mu_sq_p)
                    var T_p_safe = max(T_p, Scalar[DTYPE](PRIMAL_MINVAL))
                    var s_p = N_p - mu_p * T_p
                    var dTda_p: Scalar[DTYPE] = 0
                    for g_p in range(group_sz_p):
                        var fr_p = friction_start + fric_idx_p + g_p
                        dTda_p += jar_a[fr_p] * Jv[fr_p]
                    dTda_p /= T_p_safe
                    var dsda_p = Jv[n_p] - mu_p * dTda_p
                    d1_eval += Dm_p * s_p * dsda_p
                    var Jv_f_sq_p: Scalar[DTYPE] = 0
                    for g_p in range(group_sz_p):
                        var fr_p = friction_start + fric_idx_p + g_p
                        Jv_f_sq_p += Jv[fr_p] * Jv[fr_p]
                    var d2sda2_p = (
                        -mu_p * (Jv_f_sq_p - dTda_p * dTda_p) / T_p_safe
                    )
                    d2_eval += Dm_p * (dsda_p * dsda_p + s_p * d2sda2_p)

            fric_idx_p += group_sz_p

        for l_p in range(constraints.num_limits):
            var r_p = limits_start + l_p
            if jar_a[r_p] < Scalar[DTYPE](0):
                d1_eval += D_values[r_p] * jar_a[r_p] * Jv[r_p]
                d2_eval += D_values[r_p] * Jv[r_p] * Jv[r_p]

        for e_p in range(constraints.num_equality):
            var r_p = eq_start + e_p
            d1_eval += D_values[r_p] * jar_a[r_p] * Jv[r_p]
            d2_eval += D_values[r_p] * Jv[r_p] * Jv[r_p]

        if d2_eval < Scalar[DTYPE](PRIMAL_MINVAL):
            d2_eval = Scalar[DTYPE](PRIMAL_MINVAL)

        p1_d1 = d1_eval
        p1_d2 = d2_eval

        # Check convergence
        if p1_d1 * p1_d1 < gtol * gtol:
            return p1_alpha

        # Check if sign changed (bracket found)
        if p1_d1 * dir_sign > Scalar[DTYPE](0):
            bracket_found = True
            break

    if not bracket_found:
        return p1_alpha

    # === Phase 3: Bracketed Newton refinement ===
    # p1 and p2 bracket a root (opposite derivative signs)
    for _ in range(PRIMAL_MAX_LINESEARCH):
        # Try midpoint
        var mid_alpha = (p1_alpha + p2_alpha) * Scalar[DTYPE](0.5)
        eval_alpha = mid_alpha

        for r in range(constraints.num_rows):
            jar_a[r] = jar[r] + eval_alpha * Jv[r]

        d1_eval = gauss_a * eval_alpha + gauss_b
        d2_eval = gauss_a

        var fric_idx_m = 0
        for n_m in range(num_normals):
            var group_sz_m = 0
            while fric_idx_m + group_sz_m < num_friction:
                if (
                    constraints.rows[
                        friction_start + fric_idx_m + group_sz_m
                    ].friction_parent
                    != n_m
                ):
                    break
                group_sz_m += 1

            var N_m = jar_a[n_m]
            var mu_m = constraints.rows[
                friction_start + fric_idx_m
            ].friction_coef if group_sz_m > 0 else Scalar[DTYPE](0)

            if group_sz_m == 0:
                if N_m < Scalar[DTYPE](0):
                    d1_eval += D_values[n_m] * jar_a[n_m] * Jv[n_m]
                    d2_eval += D_values[n_m] * Jv[n_m] * Jv[n_m]
            else:
                var T_m_sq: Scalar[DTYPE] = 0
                for g_m in range(group_sz_m):
                    var fr_m = friction_start + fric_idx_m + g_m
                    T_m_sq += jar_a[fr_m] * jar_a[fr_m]
                var T_m = sqrt(T_m_sq)

                if (
                    N_m >= Scalar[DTYPE](0)
                    and N_m * N_m >= mu_m * mu_m * T_m_sq
                ):
                    pass
                elif (mu_m * N_m + T_m) <= Scalar[DTYPE](0):
                    d1_eval += D_values[n_m] * jar_a[n_m] * Jv[n_m]
                    d2_eval += D_values[n_m] * Jv[n_m] * Jv[n_m]
                    for g_m in range(group_sz_m):
                        var fr_m = friction_start + fric_idx_m + g_m
                        d1_eval += D_values[fr_m] * jar_a[fr_m] * Jv[fr_m]
                        d2_eval += D_values[fr_m] * Jv[fr_m] * Jv[fr_m]
                else:
                    var mu_sq_m = mu_m * mu_m
                    var Dm_m = D_values[n_m] / (Scalar[DTYPE](1) + mu_sq_m)
                    var T_m_safe = max(T_m, Scalar[DTYPE](PRIMAL_MINVAL))
                    var s_m = N_m - mu_m * T_m
                    var dTda_m: Scalar[DTYPE] = 0
                    for g_m in range(group_sz_m):
                        var fr_m = friction_start + fric_idx_m + g_m
                        dTda_m += jar_a[fr_m] * Jv[fr_m]
                    dTda_m /= T_m_safe
                    var dsda_m = Jv[n_m] - mu_m * dTda_m
                    d1_eval += Dm_m * s_m * dsda_m
                    var Jv_f_sq_m: Scalar[DTYPE] = 0
                    for g_m in range(group_sz_m):
                        var fr_m = friction_start + fric_idx_m + g_m
                        Jv_f_sq_m += Jv[fr_m] * Jv[fr_m]
                    var d2sda2_m = (
                        -mu_m * (Jv_f_sq_m - dTda_m * dTda_m) / T_m_safe
                    )
                    d2_eval += Dm_m * (dsda_m * dsda_m + s_m * d2sda2_m)

            fric_idx_m += group_sz_m

        for l_m in range(constraints.num_limits):
            var r_m = limits_start + l_m
            if jar_a[r_m] < Scalar[DTYPE](0):
                d1_eval += D_values[r_m] * jar_a[r_m] * Jv[r_m]
                d2_eval += D_values[r_m] * Jv[r_m] * Jv[r_m]

        for e_m in range(constraints.num_equality):
            var r_m = eq_start + e_m
            d1_eval += D_values[r_m] * jar_a[r_m] * Jv[r_m]
            d2_eval += D_values[r_m] * Jv[r_m] * Jv[r_m]

        if d2_eval < Scalar[DTYPE](PRIMAL_MINVAL):
            d2_eval = Scalar[DTYPE](PRIMAL_MINVAL)

        var mid_d1 = d1_eval

        # Check convergence
        if mid_d1 * mid_d1 < gtol * gtol:
            return mid_alpha

        # Update bracket: replace the endpoint with same sign as midpoint
        if mid_d1 * p1_d1 > Scalar[DTYPE](0):
            p1_alpha = mid_alpha
            p1_d1 = mid_d1
            p1_d2 = d2_eval
        else:
            p2_alpha = mid_alpha
            p2_d1 = mid_d1
            p2_d2 = d2_eval

        # Check if bracket is tiny
        if (p1_alpha - p2_alpha) * (p1_alpha - p2_alpha) < Scalar[DTYPE](
            PRIMAL_MINVAL
        ):
            break

    # Return whichever has smaller |d1|
    if p1_d1 * p1_d1 < p2_d1 * p2_d1:
        return p1_alpha
    return p2_alpha


@always_inline
fn primal_linesearch[
    DTYPE: DType,
    MAX_ROWS: Int,
    NV: Int,
    V_SIZE: Int,
    R_SIZE: Int,
](
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    qacc: List[Scalar[DTYPE]],
    qacc_smooth: List[Scalar[DTYPE]],
    qfrc_smooth: List[Scalar[DTYPE]],
    Ma: List[Scalar[DTYPE]],
    Mv: List[Scalar[DTYPE]],
    search: List[Scalar[DTYPE]],
    jar: List[Scalar[DTYPE]],
    force: List[Scalar[DTYPE]],
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
        var D_r = primal_D(constraints.rows[r].inv_K_imp, constraints.rows[r].K)
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

    var qacc_trial = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        qacc_trial.append(Scalar[DTYPE](0))
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
