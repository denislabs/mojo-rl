"""The v1 ablations: what the cocycle loss actually does. GATES ONLY.

Nothing here is on an execution path. These are the arms the design document
argues against, implemented so the argument can be measured rather than
asserted (docs/SHEAF_WORLD_MODELS_V2.md §1.2, rows B / C / C').

  B   translations per edge — a single global frame. Can LOCATE the seam, cannot
      represent it, and fails by PARITY.
  C   free GL(d) morphisms + a cocycle loss. Gradient descent "resolves" the
      contradiction by crushing the frustrated dimension: det H -> 0.
  C'  orthogonal morphisms + the same cocycle loss. INERT.

C' is the sharp one, and it is exact algebra rather than an empirical trend.
`L = ||H - I||_F^2 = 4 - 2 tr H`. On the `det = -1` component of O(2) every `H`
is a reflection, so `tr H = 0` and `L = 4` identically. Moving along ANY tangent
direction of the product manifold keeps `H` a reflection, so the loss cannot
change and its tangent gradient is **exactly zero**. `cocycle_tangent_norm`
measures precisely that: a raw gradient that is large, and a tangent projection
at the floating-point floor. The orientable case is the control — there the
tangent gradient is not zero, which is what makes the measurement non-trivial.
"""

from std.math import abs, sqrt

from .so_d import SqMat, expm_skew
from .procrustes import PairBatch, procrustes_o_d, mean_squared_residual


def holonomy_product[
    D: Int, dtype: DType = DType.float64
](rs: List[SqMat[D, dtype]]) -> SqMat[D, dtype]:
    """`R_{n-1} ... R_0` — the ring holonomy based at cell 0."""
    var h = SqMat[D, dtype].identity()
    for i in range(len(rs)):
        h = rs[i] * h
    return h^


def cocycle_grad[
    D: Int, dtype: DType = DType.float64
](rs: List[SqMat[D, dtype]], i: Int) -> SqMat[D, dtype]:
    """`d/dR_i` of `||H - I||_F^2`, with `H = R_{n-1} ... R_0`.

    `d/dR_i tr(G^T left R_i right) = left^T G right^T`.
    """
    var n = len(rs)
    var h = holonomy_product[D, dtype](rs)
    var g = (h - SqMat[D, dtype].identity()).scaled(Scalar[dtype](2))
    var left = SqMat[D, dtype].identity()
    for k in range(i + 1, n):
        left = rs[k] * left
    var right = SqMat[D, dtype].identity()
    for k in range(0, i):
        right = rs[k] * right
    return (left.transpose() * g) * right.transpose()


def cocycle_tangent_norm[
    D: Int, dtype: DType = DType.float64
](rs: List[SqMat[D, dtype]]) -> List[Float64]:
    """Returns `[raw gradient norm, tangent-projected norm]` of the cocycle loss.

    The tangent projection at `R` is the skew part of `R^T grad`, i.e. the
    component of the gradient that a Riemannian step could actually follow.
    On the `det = -1` component this is exactly zero — the loss is constant
    there — which is the whole reason the cocycle term cannot help.
    """
    var raw = Float64(0)
    var tan = Float64(0)
    for i in range(len(rs)):
        var g = cocycle_grad[D, dtype](rs, i)
        raw += Float64(g.frobenius_norm()) ** 2
        var a = rs[i].transpose() * g
        var skew = (a - a.transpose()).scaled(Scalar[dtype](0.5))
        tan += Float64(skew.frobenius_norm()) ** 2
    var out = List[Float64]()
    out.append(sqrt(raw))
    out.append(sqrt(tan))
    return out^


def min_singular_value[dtype: DType = DType.float64](m: SqMat[2, dtype]) -> Float64:
    """Smallest singular value of a 2x2 — the "crushed dimension" detector."""
    var a = Float64(m[0, 0])
    var b = Float64(m[0, 1])
    var c = Float64(m[1, 0])
    var d = Float64(m[1, 1])
    var s00 = a * a + c * c
    var s01 = a * b + c * d
    var s11 = b * b + d * d
    var tr = s00 + s11
    var det = s00 * s11 - s01 * s01
    var disc = tr * tr / 4.0 - det
    if disc < 0:
        disc = 0
    var lo = tr / 2.0 - sqrt(disc)
    if lo < 0:
        lo = 0
    return sqrt(lo)


def fit_free_with_cocycle[
    dtype: DType = DType.float64
](
    batches: List[PairBatch[2, dtype]],
    lam: Float64,
    steps: Int = 3000,
    clip: Float64 = 5.0,
) raises -> List[SqMat[2, dtype]]:
    """Ablation C: unconstrained GL(2) morphisms, fit loss + `lam` * cocycle.

    Initialised at the local Procrustes solution, exactly as the numpy
    prototype does, so any drift away from it is caused by the cocycle term and
    not by a bad starting point.
    """
    var n = len(batches)
    var rs = List[SqMat[2, dtype]]()
    for e in range(n):
        rs.append(procrustes_o_d[2, dtype](batches[e]))
    var lr = 0.05 / (1.0 + lam)

    for _ in range(steps):
        var grads = List[SqMat[2, dtype]]()
        for e in range(n):
            var g = SqMat[2, dtype]()
            var cnt = batches[e].count()
            if cnt > 0:
                for k in range(cnt):
                    var err = List[Scalar[dtype]](length=2, fill=0)
                    for i in range(2):
                        var pred = Scalar[dtype](0)
                        for j in range(2):
                            pred += rs[e][i, j] * batches[e].xs[k * 2 + j]
                        err[i] = pred - batches[e].ys[k * 2 + i]
                    for i in range(2):
                        for j in range(2):
                            g[i, j] = g[i, j] + Scalar[dtype](
                                2.0 / Float64(cnt)
                            ) * err[i] * batches[e].xs[k * 2 + j]
            grads.append(g^)
        for e in range(n):
            var cg = cocycle_grad[2, dtype](rs, e)
            grads[e] = grads[e] + cg.scaled(Scalar[dtype](lam))
        for e in range(n):
            var gn = Float64(grads[e].frobenius_norm())
            var g = grads[e].copy()
            if gn > clip:
                g = g.scaled(Scalar[dtype](clip / gn))
            rs[e] = rs[e] - g.scaled(Scalar[dtype](lr))
    return rs^


def fit_orthogonal_with_cocycle[
    dtype: DType = DType.float64
](
    batches: List[PairBatch[2, dtype]], lam: Float64, steps: Int = 3000
) raises -> List[SqMat[2, dtype]]:
    """Ablation C': the same loss, but the morphisms are held in O(2).

    Riemannian: project the gradient onto so(2) and step with `exp`. The
    cocycle part of that projection is identically zero on the `det = -1`
    component, so this arm should be indistinguishable from plain Procrustes.
    """
    var n = len(batches)
    var rs = List[SqMat[2, dtype]]()
    for e in range(n):
        rs.append(procrustes_o_d[2, dtype](batches[e]))
    var lr = 0.05 / (1.0 + lam)

    for _ in range(steps):
        var grads = List[SqMat[2, dtype]]()
        for e in range(n):
            var g = SqMat[2, dtype]()
            var cnt = batches[e].count()
            if cnt > 0:
                for k in range(cnt):
                    var err = List[Scalar[dtype]](length=2, fill=0)
                    for i in range(2):
                        var pred = Scalar[dtype](0)
                        for j in range(2):
                            pred += rs[e][i, j] * batches[e].xs[k * 2 + j]
                        err[i] = pred - batches[e].ys[k * 2 + i]
                    for i in range(2):
                        for j in range(2):
                            g[i, j] = g[i, j] + Scalar[dtype](
                                2.0 / Float64(cnt)
                            ) * err[i] * batches[e].xs[k * 2 + j]
            grads.append(g^)
        for e in range(n):
            grads[e] = grads[e] + cocycle_grad[2, dtype](rs, e).scaled(
                Scalar[dtype](lam)
            )
        for e in range(n):
            var a = rs[e].transpose() * grads[e]
            var skew = (a - a.transpose()).scaled(Scalar[dtype](0.5))
            rs[e] = rs[e] * expm_skew[2, dtype](
                skew.scaled(Scalar[dtype](-lr))
            )
    return rs^


def fit_translations[
    dtype: DType = DType.float64
](batches: List[PairBatch[2, dtype]]) -> List[Scalar[dtype]]:
    """Ablation B: a single global frame, transitions are per-edge translations."""
    var n = len(batches)
    var out = List[Scalar[dtype]](length=n * 2, fill=0)
    for e in range(n):
        var cnt = batches[e].count()
        if cnt == 0:
            continue
        for k in range(cnt):
            for c in range(2):
                out[e * 2 + c] += (
                    batches[e].ys[k * 2 + c] - batches[e].xs[k * 2 + c]
                )
        for c in range(2):
            out[e * 2 + c] /= Scalar[dtype](cnt)
    return out^
