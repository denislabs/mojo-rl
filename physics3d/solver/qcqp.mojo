"""QCQP (Quadratically Constrained Quadratic Program) solvers for friction cone projection.

Implements elliptic friction cone projection for condim 3/4/6:
- qcqp2: 2D projection (condim=3, tangent t1+t2)
- qcqp3: 3D projection (condim=4, t1+t2+torsion)
- qcqp5: 5D projection (condim=6, t1+t2+torsion+roll1+roll2)

Algorithm: Newton on dual variable for n-D elliptic cone projection.
Given unconstrained friction forces f and friction limits mu*fn per dim,
project f onto the elliptic cone ||f_i / mu_i|| <= fn.

Reference: MuJoCo source (engine_solver.c, mj_solQCQP)
"""

from math import sqrt


@always_inline
fn qcqp2[DTYPE: DType](
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
fn qcqp3[DTYPE: DType](
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
        f1-f3: Friction forces (modified in-place).
        mu1-mu3: Per-dimension friction coefficients.
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
fn qcqp5[DTYPE: DType](
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
        f1-f5: Friction forces (modified in-place).
        mu1-mu5: Per-dimension friction coefficients.
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
        var val = g1 * g1 + g2 * g2 + g3 * g3 + g4 * g4 + g5 * g5 - Scalar[DTYPE](1.0)

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
            g1 * dv1 / d1 + g2 * dv2 / d2 + g3 * dv3 / d3
            + g4 * dv4 / d4 + g5 * dv5 / d5
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
