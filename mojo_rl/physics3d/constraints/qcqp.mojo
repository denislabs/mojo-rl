"""QCQP (Quadratically Constrained Quadratic Program) solvers for friction cone projection.

Two families of solvers:

1. Simple cone projection (qcqp2/3/5): Project friction forces onto elliptic cone.
   Used by pyramidal cone mode and as fallback.

2. MuJoCo-matching QCQP (mj_qcqp2/3/5): Solve the full QP:
     min  0.5*x'*A*x + x'*b  s.t.  sum(xi/di)^2 <= r^2
   These take the AR Hessian submatrix and adjusted bias, matching
   MuJoCo's engine_util_solve.c (mju_QCQP2/3/QCQP).

Reference: MuJoCo source (engine_util_solve.c lines 991-1212)
"""

from std.math import sqrt
from ..types import _max_one


@always_inline
def qcqp2[
    DTYPE: DType
](
    mut f1: Scalar[DTYPE],
    mut f2: Scalar[DTYPE],
    mu: Scalar[DTYPE],
    fn_val: Scalar[DTYPE],
):
    """Project 2D friction force onto circular Coulomb cone.

    This is the standard radial projection: if ||(f1,f2)|| > mu*fn,
    scale down to the cone boundary.

    Args:
        f1: Tangent force 1 (modified in-place).
        f2: Tangent force 2 (modified in-place).
        mu: Friction coefficient.
        fn_val: Normal force magnitude.
    """
    var max_friction = mu * fn_val
    if max_friction <= Scalar[DTYPE](0):
        f1 = Scalar[DTYPE](0)
        f2 = Scalar[DTYPE](0)
        return

    var t_mag = sqrt(f1 * f1 + f2 * f2)
    if t_mag > max_friction:
        var scale = max_friction / t_mag
        f1 = f1 * scale
        f2 = f2 * scale


@always_inline
def qcqp3[
    DTYPE: DType
](
    mut f1: Scalar[DTYPE],
    mut f2: Scalar[DTYPE],
    mut f3: Scalar[DTYPE],
    mu1: Scalar[DTYPE],
    mu2: Scalar[DTYPE],
    mu3: Scalar[DTYPE],
    fn_val: Scalar[DTYPE],
):
    """Project 3D friction force onto elliptic cone (condim=4).

    Uses Newton's method on the dual variable to find the projection.
    The cone constraint is: sum_i (f_i / (mu_i * fn))^2 <= 1

    Args:
        f1: Friction force 1 (modified in-place).
        f2: Friction force 2 (modified in-place).
        f3: Friction force 3 (modified in-place).
        mu1: Friction coefficient 1.
        mu2: Friction coefficient 2.
        mu3: Friction coefficient 3.
        fn_val: Normal force magnitude.
    """
    if fn_val <= Scalar[DTYPE](0):
        f1 = Scalar[DTYPE](0)
        f2 = Scalar[DTYPE](0)
        f3 = Scalar[DTYPE](0)
        return

    # Scale factors (radius per dimension)
    var d1 = mu1 * fn_val
    var d2 = mu2 * fn_val
    var d3 = mu3 * fn_val

    if d1 < Scalar[DTYPE](1e-10):
        d1 = Scalar[DTYPE](1e-10)
    if d2 < Scalar[DTYPE](1e-10):
        d2 = Scalar[DTYPE](1e-10)
    if d3 < Scalar[DTYPE](1e-10):
        d3 = Scalar[DTYPE](1e-10)

    # Check if already inside cone
    var s1 = f1 / d1
    var s2 = f2 / d2
    var s3 = f3 / d3
    var r2 = s1 * s1 + s2 * s2 + s3 * s3

    if r2 <= Scalar[DTYPE](1.0):
        return  # Already inside cone

    # Newton iteration on dual variable lambda
    # Minimize ||f - f_unc||^2 s.t. ||f/d||^2 <= 1
    # Solution: f_i = f_unc_i * d_i^2 / (d_i^2 + lambda)
    # Initial guess: use the constraint violation magnitude
    var lam = sqrt(r2) - Scalar[DTYPE](1.0)

    for _ in range(20):
        var v1 = f1 * d1 * d1 / (d1 * d1 + lam)
        var v2 = f2 * d2 * d2 / (d2 * d2 + lam)
        var v3 = f3 * d3 * d3 / (d3 * d3 + lam)

        var g1 = v1 / d1
        var g2 = v2 / d2
        var g3 = v3 / d3
        var val = g1 * g1 + g2 * g2 + g3 * g3 - Scalar[DTYPE](1.0)

        if abs(val) < Scalar[DTYPE](1e-8):
            f1 = v1 / (d1 * d1) * d1 * d1  # = v1
            f2 = v2
            f3 = v3
            return

        # Derivative of constraint w.r.t. lambda
        var dv1 = -f1 * d1 * d1 / ((d1 * d1 + lam) * (d1 * d1 + lam))
        var dv2 = -f2 * d2 * d2 / ((d2 * d2 + lam) * (d2 * d2 + lam))
        var dv3 = -f3 * d3 * d3 / ((d3 * d3 + lam) * (d3 * d3 + lam))
        var deriv = Scalar[DTYPE](2.0) * (
            g1 * dv1 / d1 + g2 * dv2 / d2 + g3 * dv3 / d3
        )

        if abs(deriv) < Scalar[DTYPE](1e-12):
            break

        lam = lam - val / deriv
        if lam < Scalar[DTYPE](0):
            lam = Scalar[DTYPE](0)

    # Final projection with converged lambda
    f1 = f1 * d1 * d1 / (d1 * d1 + lam)
    f2 = f2 * d2 * d2 / (d2 * d2 + lam)
    f3 = f3 * d3 * d3 / (d3 * d3 + lam)


@always_inline
def qcqp5[
    DTYPE: DType
](
    mut f1: Scalar[DTYPE],
    mut f2: Scalar[DTYPE],
    mut f3: Scalar[DTYPE],
    mut f4: Scalar[DTYPE],
    mut f5: Scalar[DTYPE],
    mu1: Scalar[DTYPE],
    mu2: Scalar[DTYPE],
    mu3: Scalar[DTYPE],
    mu4: Scalar[DTYPE],
    mu5: Scalar[DTYPE],
    fn_val: Scalar[DTYPE],
):
    """Project 5D friction force onto elliptic cone (condim=6).

    Uses Newton's method on the dual variable.
    The cone constraint is: sum_i (f_i / (mu_i * fn))^2 <= 1

    Args:
        f1: Friction force 1 (modified in-place).
        f2: Friction force 2 (modified in-place).
        f3: Friction force 3 (modified in-place).
        f4: Friction force 4 (modified in-place).
        f5: Friction force 5 (modified in-place).
        mu1: Friction coefficient 1.
        mu2: Friction coefficient 2.
        mu3: Friction coefficient 3.
        mu4: Friction coefficient 4.
        mu5: Friction coefficient 5.
        fn_val: Normal force magnitude.
    """
    if fn_val <= Scalar[DTYPE](0):
        f1 = Scalar[DTYPE](0)
        f2 = Scalar[DTYPE](0)
        f3 = Scalar[DTYPE](0)
        f4 = Scalar[DTYPE](0)
        f5 = Scalar[DTYPE](0)
        return

    # Scale factors
    var d1 = mu1 * fn_val
    var d2 = mu2 * fn_val
    var d3 = mu3 * fn_val
    var d4 = mu4 * fn_val
    var d5 = mu5 * fn_val

    if d1 < Scalar[DTYPE](1e-10):
        d1 = Scalar[DTYPE](1e-10)
    if d2 < Scalar[DTYPE](1e-10):
        d2 = Scalar[DTYPE](1e-10)
    if d3 < Scalar[DTYPE](1e-10):
        d3 = Scalar[DTYPE](1e-10)
    if d4 < Scalar[DTYPE](1e-10):
        d4 = Scalar[DTYPE](1e-10)
    if d5 < Scalar[DTYPE](1e-10):
        d5 = Scalar[DTYPE](1e-10)

    # Check if already inside cone
    var s1 = f1 / d1
    var s2 = f2 / d2
    var s3 = f3 / d3
    var s4 = f4 / d4
    var s5 = f5 / d5
    var r2 = s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4 + s5 * s5

    if r2 <= Scalar[DTYPE](1.0):
        return

    # Newton iteration on dual variable
    var lam = sqrt(r2) - Scalar[DTYPE](1.0)

    for _ in range(20):
        var dd1 = d1 * d1 + lam
        var dd2 = d2 * d2 + lam
        var dd3 = d3 * d3 + lam
        var dd4 = d4 * d4 + lam
        var dd5 = d5 * d5 + lam

        var v1 = f1 * d1 * d1 / dd1
        var v2 = f2 * d2 * d2 / dd2
        var v3 = f3 * d3 * d3 / dd3
        var v4 = f4 * d4 * d4 / dd4
        var v5 = f5 * d5 * d5 / dd5

        var g1 = v1 / d1
        var g2 = v2 / d2
        var g3 = v3 / d3
        var g4 = v4 / d4
        var g5 = v5 / d5
        var val = (
            g1 * g1 + g2 * g2 + g3 * g3 + g4 * g4 + g5 * g5 - Scalar[DTYPE](1.0)
        )

        if abs(val) < Scalar[DTYPE](1e-8):
            f1 = v1
            f2 = v2
            f3 = v3
            f4 = v4
            f5 = v5
            return

        var dv1 = -f1 * d1 * d1 / (dd1 * dd1)
        var dv2 = -f2 * d2 * d2 / (dd2 * dd2)
        var dv3 = -f3 * d3 * d3 / (dd3 * dd3)
        var dv4 = -f4 * d4 * d4 / (dd4 * dd4)
        var dv5 = -f5 * d5 * d5 / (dd5 * dd5)
        var deriv = Scalar[DTYPE](2.0) * (
            g1 * dv1 / d1
            + g2 * dv2 / d2
            + g3 * dv3 / d3
            + g4 * dv4 / d4
            + g5 * dv5 / d5
        )

        if abs(deriv) < Scalar[DTYPE](1e-12):
            break

        lam = lam - val / deriv
        if lam < Scalar[DTYPE](0):
            lam = Scalar[DTYPE](0)

    # Final projection
    f1 = f1 * d1 * d1 / (d1 * d1 + lam)
    f2 = f2 * d2 * d2 / (d2 * d2 + lam)
    f3 = f3 * d3 * d3 / (d3 * d3 + lam)
    f4 = f4 * d4 * d4 / (d4 * d4 + lam)
    f5 = f5 * d5 * d5 / (d5 * d5 + lam)


# =============================================================================
# MuJoCo-matching QCQP solvers
# Solve: min 0.5*x'*A*x + x'*b  s.t.  sum(xi/di)^2 <= r^2
# Reference: engine_util_solve.c mju_QCQP2/3/QCQP
# =============================================================================


@always_inline
def mj_qcqp2[
    DTYPE: DType
](
    mut res0: Scalar[DTYPE],
    mut res1: Scalar[DTYPE],
    A: InlineArray[Scalar[DTYPE], 4],  # 2x2 row-major
    b: InlineArray[Scalar[DTYPE], 2],
    d: InlineArray[Scalar[DTYPE], 2],  # scaling (mu per direction)
    r: Scalar[DTYPE],  # constraint radius (normal force)
) -> Bool:
    """Solve 2D QCQP matching MuJoCo's mju_QCQP2.

    Returns True if constrained (lambda > 0).
    """
    # Scale A, b so constraint becomes x'*x <= r*r
    var b1 = b[0] * d[0]
    var b2 = b[1] * d[1]
    var A11 = A[0] * d[0] * d[0]
    var A22 = A[3] * d[1] * d[1]
    var A12 = A[1] * d[0] * d[1]

    # Newton iteration
    var la: Scalar[DTYPE] = 0
    var v1: Scalar[DTYPE] = 0
    var v2: Scalar[DTYPE] = 0

    for _ in range(20):
        # det(A+la)
        var det = (A11 + la) * (A22 + la) - A12 * A12

        # Check SPD
        if det < Scalar[DTYPE](1e-10):
            res0 = 0
            res1 = 0
            return False

        # P = inv(A+la)
        var detinv = Scalar[DTYPE](1.0) / det
        var P11 = (A22 + la) * detinv
        var P22 = (A11 + la) * detinv
        var P12 = -A12 * detinv

        # v = -P*b
        v1 = -P11 * b1 - P12 * b2
        v2 = -P12 * b1 - P22 * b2

        # val = v'*v - r*r
        var val = v1 * v1 + v2 * v2 - r * r

        # Check convergence or unconstrained
        if val < Scalar[DTYPE](1e-10):
            break

        # deriv = -2 * v' * P * v
        var deriv = Scalar[DTYPE](-2.0) * (
            P11 * v1 * v1 + Scalar[DTYPE](2.0) * P12 * v1 * v2 + P22 * v2 * v2
        )

        # Update
        var delta = -val / deriv
        if delta < Scalar[DTYPE](1e-10):
            break
        la += delta

    # Undo scaling
    res0 = v1 * d[0]
    res1 = v2 * d[1]
    return la != Scalar[DTYPE](0)


@always_inline
def mj_qcqp3[
    DTYPE: DType
](
    mut res0: Scalar[DTYPE],
    mut res1: Scalar[DTYPE],
    mut res2: Scalar[DTYPE],
    A: InlineArray[Scalar[DTYPE], 9],  # 3x3 row-major
    b: InlineArray[Scalar[DTYPE], 3],
    d: InlineArray[Scalar[DTYPE], 3],
    r: Scalar[DTYPE],
) -> Bool:
    """Solve 3D QCQP matching MuJoCo's mju_QCQP3.

    Returns True if constrained (lambda > 0).
    """
    # Scale A, b
    var b1 = b[0] * d[0]
    var b2 = b[1] * d[1]
    var b3 = b[2] * d[2]
    var A11 = A[0] * d[0] * d[0]
    var A22 = A[4] * d[1] * d[1]
    var A33 = A[8] * d[2] * d[2]
    var A12 = A[1] * d[0] * d[1]
    var A13 = A[2] * d[0] * d[2]
    var A23 = A[5] * d[1] * d[2]

    var la: Scalar[DTYPE] = 0
    var v1: Scalar[DTYPE] = 0
    var v2: Scalar[DTYPE] = 0
    var v3: Scalar[DTYPE] = 0

    for _ in range(20):
        # Cofactors (unscaled P)
        var P11 = (A22 + la) * (A33 + la) - A23 * A23
        var P22 = (A11 + la) * (A33 + la) - A13 * A13
        var P33 = (A11 + la) * (A22 + la) - A12 * A12
        var P12 = A13 * A23 - A12 * (A33 + la)
        var P13 = A12 * A23 - A13 * (A22 + la)
        var P23 = A12 * A13 - A23 * (A11 + la)

        # det(A+la)
        var det = (A11 + la) * P11 + A12 * P12 + A13 * P13

        if det < Scalar[DTYPE](1e-10):
            res0 = 0
            res1 = 0
            res2 = 0
            return False

        var detinv = Scalar[DTYPE](1.0) / det
        P11 *= detinv
        P22 *= detinv
        P33 *= detinv
        P12 *= detinv
        P13 *= detinv
        P23 *= detinv

        # v = -P*b
        v1 = -P11 * b1 - P12 * b2 - P13 * b3
        v2 = -P12 * b1 - P22 * b2 - P23 * b3
        v3 = -P13 * b1 - P23 * b2 - P33 * b3

        var val = v1 * v1 + v2 * v2 + v3 * v3 - r * r

        if val < Scalar[DTYPE](1e-10):
            break

        var deriv = Scalar[DTYPE](-2.0) * (
            P11 * v1 * v1 + P22 * v2 * v2 + P33 * v3 * v3
        ) - Scalar[DTYPE](4.0) * (P12 * v1 * v2 + P13 * v1 * v3 + P23 * v2 * v3)

        var delta = -val / deriv
        if delta < Scalar[DTYPE](1e-10):
            break
        la += delta

    res0 = v1 * d[0]
    res1 = v2 * d[1]
    res2 = v3 * d[2]
    return la != Scalar[DTYPE](0)


@always_inline
def mj_qcqp5[
    DTYPE: DType
](
    mut res: InlineArray[Scalar[DTYPE], 5],
    A: InlineArray[Scalar[DTYPE], 25],  # 5x5 row-major
    b: InlineArray[Scalar[DTYPE], 5],
    d: InlineArray[Scalar[DTYPE], 5],
    r: Scalar[DTYPE],
) -> Bool:
    """Solve 5D QCQP matching MuJoCo's mju_QCQP (n=5) via Cholesky.

    Returns True if constrained (lambda > 0).
    """
    # Scale A, b
    var As = InlineArray[Scalar[DTYPE], 25](fill=Scalar[DTYPE](0))
    var bs = InlineArray[Scalar[DTYPE], 5](fill=Scalar[DTYPE](0))
    for i in range(5):
        bs[i] = b[i] * d[i]
        for j in range(5):
            As[i * 5 + j] = A[i * 5 + j] * d[i] * d[j]

    var la: Scalar[DTYPE] = 0

    for _ in range(20):
        # Make Ala = As + la*I
        var Ala = InlineArray[Scalar[DTYPE], 25](fill=Scalar[DTYPE](0))
        for i in range(5):
            for j in range(5):
                Ala[i * 5 + j] = As[i * 5 + j]
            Ala[i * 5 + i] += la

        # Cholesky factorize (in-place, lower triangular)
        var L = InlineArray[Scalar[DTYPE], 25](fill=Scalar[DTYPE](0))
        var rank_ok = True
        for i in range(5):
            for j in range(i + 1):
                var s: Scalar[DTYPE] = 0
                for k in range(j):
                    s += L[i * 5 + k] * L[j * 5 + k]
                if i == j:
                    var diag = Ala[i * 5 + i] - s
                    if diag < Scalar[DTYPE](1e-10):
                        rank_ok = False
                        break
                    L[i * 5 + j] = sqrt(diag)
                else:
                    L[i * 5 + j] = (Ala[i * 5 + j] - s) / L[j * 5 + j]
            if not rank_ok:
                break

        if not rank_ok:
            for i in range(5):
                res[i] = 0
            return False

        # Solve L*y = -bs (forward substitution)
        var y = InlineArray[Scalar[DTYPE], 5](fill=Scalar[DTYPE](0))
        for i in range(5):
            var s: Scalar[DTYPE] = 0
            for j in range(i):
                s += L[i * 5 + j] * y[j]
            y[i] = (-bs[i] - s) / L[i * 5 + i]

        # Solve L'*x = y (back substitution) -> res = -(A+la)^-1 * bs
        for i_rev in range(5):
            var i = 4 - i_rev
            var s: Scalar[DTYPE] = 0
            for j in range(i + 1, 5):
                s += L[j * 5 + i] * res[j]
            res[i] = (y[i] - s) / L[i * 5 + i]

        # val = res'*res - r*r
        var val: Scalar[DTYPE] = 0
        for i in range(5):
            val += res[i] * res[i]
        val -= r * r

        if val < Scalar[DTYPE](1e-10):
            break

        # Solve L*L'*tmp = res for deriv
        var tmp_y = InlineArray[Scalar[DTYPE], 5](fill=Scalar[DTYPE](0))
        for i in range(5):
            var s: Scalar[DTYPE] = 0
            for j in range(i):
                s += L[i * 5 + j] * tmp_y[j]
            tmp_y[i] = (res[i] - s) / L[i * 5 + i]
        var tmp = InlineArray[Scalar[DTYPE], 5](fill=Scalar[DTYPE](0))
        for i_rev in range(5):
            var i = 4 - i_rev
            var s: Scalar[DTYPE] = 0
            for j in range(i + 1, 5):
                s += L[j * 5 + i] * tmp[j]
            tmp[i] = (tmp_y[i] - s) / L[i * 5 + i]

        var deriv: Scalar[DTYPE] = 0
        for i in range(5):
            deriv += res[i] * tmp[i]
        deriv *= Scalar[DTYPE](-2.0)

        var delta = -val / deriv
        if delta < Scalar[DTYPE](1e-10):
            break
        la += delta

    # Undo scaling
    for i in range(5):
        res[i] = res[i] * d[i]
    return la != Scalar[DTYPE](0)


@always_inline
def cost_change[
    DTYPE: DType, MAX_DIM: Int, AR_SIZE: Int
](
    force: InlineArray[Scalar[DTYPE], MAX_DIM],
    oldforce: InlineArray[Scalar[DTYPE], MAX_DIM],
    AR: InlineArray[Scalar[DTYPE], AR_SIZE],
    res: InlineArray[Scalar[DTYPE], MAX_DIM],
    dim: Int,
) -> Scalar[DTYPE]:
    """Compute cost change from MuJoCo's costChange function.

    Returns the change value. Positive change means the update increased cost
    and should be reverted.

    change = 0.5*delta'*AR*delta + delta'*res
    """
    var change: Scalar[DTYPE] = 0
    if dim == 1:
        var delta = force[0] - oldforce[0]
        change = Scalar[DTYPE](0.5) * delta * delta * AR[0] + delta * res[0]
    else:
        for i in range(dim):
            var delta_i = force[i] - oldforce[i]
            change += delta_i * res[i]
            for j in range(dim):
                var delta_j = force[j] - oldforce[j]
                change += (
                    Scalar[DTYPE](0.5) * delta_i * AR[i * dim + j] * delta_j
                )
    return change
