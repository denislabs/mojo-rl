"""Per-edge O(D) fit and its residual — the sheaf-free half of the observables.

`argmin_{R in O(D)} sum_k ||R x_k - y_k||^2` is the orthogonal Procrustes
problem. The textbook answer is `R = U V^T` from the SVD of `M = sum_k y_k x_k^T`,
which is what the numpy oracle (`tools/swm/dump_mobius_reference.py`) computes.

**There is no generic SVD in this repo**, so this file takes the other exact
route: `U V^T` is precisely the orthogonal factor of the polar decomposition
`M = (U V^T) (V S V^T)`, and that factor is reachable by a Newton iteration
using only matmul and inverse, both of which `so_d.mojo` already has.

That difference is a feature, not a workaround. The Phase 1 gate feeds this
code the *same* observation pairs the oracle saw and demands the *same*
transports back. numpy gets there by SVD, this gets there by Newton — an
independent implementation, so a shared bug cannot make the gate pass. A
transcription of the numpy source would have been blind.

The polar factor inherits `sign(det M)`, so this yields O(D) and not SO(D) —
required, since the reflection is the entire signal being measured.
"""

from std.collections import InlineArray
from std.math import abs, sqrt

from .so_d import SqMat


struct PairBatch[D: Int, dtype: DType = DType.float64](Copyable, Movable):
    """Observed transitions on one edge: `y ~ R x + noise`, row-major flat."""

    var xs: List[Scalar[Self.dtype]]
    var ys: List[Scalar[Self.dtype]]

    def __init__(out self):
        self.xs = List[Scalar[Self.dtype]]()
        self.ys = List[Scalar[Self.dtype]]()

    def __init__(out self, *, copy: Self):
        self.xs = copy.xs.copy()
        self.ys = copy.ys.copy()

    def __init__(out self, *, deinit move: Self):
        self.xs = move.xs^
        self.ys = move.ys^

    def count(self) -> Int:
        return len(self.xs) // Self.D

    def push(
        mut self,
        x: InlineArray[Scalar[Self.dtype], Self.D],
        y: InlineArray[Scalar[Self.dtype], Self.D],
    ):
        for i in range(Self.D):
            self.xs.append(x[i])
            self.ys.append(y[i])

    def x_at(self, k: Int) -> InlineArray[Scalar[Self.dtype], Self.D]:
        var v = InlineArray[Scalar[Self.dtype], Self.D](fill=0)
        for i in range(Self.D):
            v[i] = self.xs[k * Self.D + i]
        return v^

    def y_at(self, k: Int) -> InlineArray[Scalar[Self.dtype], Self.D]:
        var v = InlineArray[Scalar[Self.dtype], Self.D](fill=0)
        for i in range(Self.D):
            v[i] = self.ys[k * Self.D + i]
        return v^


def cross_covariance[
    D: Int, dtype: DType = DType.float64
](batch: PairBatch[D, dtype]) -> SqMat[D, dtype]:
    """`M = sum_k y_k x_k^T`."""
    var m = SqMat[D, dtype]()
    var n = batch.count()
    for k in range(n):
        for i in range(D):
            var yi = batch.ys[k * D + i]
            if yi == 0:
                continue
            for j in range(D):
                m[i, j] = m[i, j] + yi * batch.xs[k * D + j]
    return m^


def polar_orthogonal_factor[
    D: Int, dtype: DType = DType.float64
](m: SqMat[D, dtype], tol: Float64 = 1e-15, max_iter: Int = 100) raises -> SqMat[
    D, dtype
]:
    """Orthogonal polar factor of `m` by scaled Newton: `R <- (gR + R^-T/g)/2`.

    Quadratically convergent; the Frobenius-norm scaling `g` is what keeps the
    first few steps from crawling when `m` is badly scaled. Raises if `m` is
    singular (the polar factor is not unique there).

    NOTE on the scaling: it is a convergence accelerator, not a correctness
    property — unscaled Newton reaches the same fixed point, just later. G4
    cannot distinguish the two (a `g = 1` mutant passes it, measured), and that
    is expected rather than a hole in the gate: on the well-conditioned 2x2
    problems of Phase 1, `max_iter = 100` is ample either way. The scaling earns
    its place in Phase 3, where learned encoders make `m` badly conditioned.
    """
    var r = m.copy()
    for _ in range(max_iter):
        var r_inv_t = r.inverse_transpose()
        # Higham's scaling: g = (||R^-T||_F / ||R||_F)^(1/2)
        var nr = Float64(r.frobenius_norm())
        var ni = Float64(r_inv_t.frobenius_norm())
        var g = Scalar[dtype](1)
        if nr > 0 and ni > 0:
            g = Scalar[dtype](sqrt(ni / nr))
        var nxt = (r.scaled(g) + r_inv_t.scaled(Scalar[dtype](1) / g)).scaled(
            Scalar[dtype](0.5)
        )
        var delta = Float64(nxt.max_abs_diff(r))
        r = nxt^
        if delta <= tol:
            break
    return r^


def procrustes_o_d[
    D: Int, dtype: DType = DType.float64
](batch: PairBatch[D, dtype]) raises -> SqMat[D, dtype]:
    """`argmin_{R in O(D)} sum ||R x - y||^2`, with the reflection preserved."""
    if batch.count() == 0:
        raise Error("procrustes_o_d: empty batch")
    return polar_orthogonal_factor[D, dtype](cross_covariance[D, dtype](batch))


def mean_squared_residual[
    D: Int, dtype: DType = DType.float64
](batch: PairBatch[D, dtype], r: SqMat[D, dtype]) -> Scalar[dtype]:
    """`mean_k ||R x_k - y_k||^2` — the PRE-CONSENSUS per-edge residual.

    Measured on observed pairs, never at the optimum of an inference: on a
    frustrated cycle the inference redistributes the disagreement and destroys
    the very premise ("residuals nominal") the classification rule reads
    (docs/SHEAF_WORLD_MODELS_V2.md §4.4).
    """
    var n = batch.count()
    if n == 0:
        return 0
    var total = Scalar[dtype](0)
    for k in range(n):
        for i in range(D):
            var pred = Scalar[dtype](0)
            for j in range(D):
                pred += r[i, j] * batch.xs[k * D + j]
            var e = pred - batch.ys[k * D + i]
            total += e * e
    return total / Scalar[dtype](n)
