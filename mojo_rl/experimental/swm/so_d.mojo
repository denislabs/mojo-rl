"""Small dense D x D algebra for the SWM frame channel (CPU, comptime D).

This is deliberately NOT `mojo_rl.nn` machinery: the frame channel manipulates
many tiny `D x D` orthogonal matrices (one per graph edge), not a few large
weight slabs. `D` is a compile-time parameter, so an `InlineArray` backing is
safe here — the runtime-bounds caveat that makes fixed-capacity arrays lose to
the heap does not apply, and nothing in this file ever runs on a GPU (the
Metal wide-per-thread `InlineArray` miscompute is out of scope by construction).

Row-major: `m[r, c]` with `r` the row. Default dtype is float64, because the
Phase 1 gates compare against a numpy oracle at ~1e-13 and float32 would swamp
the agreement.

What lives here:
  - `SqMat[D]`      dense square matrix: matmul, transpose, det, inverse
  - `skew_from_vector`  the D(D-1)/2 free parameters of so(D) -> skew matrix
  - `cayley`        (I - S)(I + S)^-1, lands in SO(D)
  - `expm_skew`     matrix exponential of a skew matrix, lands in SO(D)
  - `householder`   a fixed reflection, det = -1 (the other component of O(D))

Neither `cayley` nor `expm_skew` can reach the `det = -1` component: that is
the whole reason the transport carries a separate discrete orientation bit
(see docs/SHEAF_WORLD_MODELS_V2.md §4.2).
"""

from std.collections import InlineArray
from std.math import sqrt, abs, ceil, log2


struct SqMat[D: Int, dtype: DType = DType.float64](
    ImplicitlyCopyable, Movable
):
    """Dense `D x D` matrix, row-major, stack-allocated."""

    comptime SIZE: Int = Self.D * Self.D
    comptime SKEW_PARAMS: Int = Self.D * (Self.D - 1) // 2

    var data: InlineArray[Scalar[Self.dtype], Self.SIZE]

    def __init__(out self):
        """All zeros."""
        self.data = InlineArray[Scalar[Self.dtype], Self.SIZE](fill=0)

    def __init__(out self, *, copy: Self):
        self.data = copy.data.copy()

    def __init__(out self, *, deinit move: Self):
        self.data = move.data^

    @staticmethod
    def identity() -> Self:
        var m = Self()
        for i in range(Self.D):
            m.data[i * Self.D + i] = 1
        return m^

    # -- element access -------------------------------------------------------

    def __getitem__(self, r: Int, c: Int) -> Scalar[Self.dtype]:
        return self.data[r * Self.D + c]

    def __setitem__(mut self, r: Int, c: Int, v: Scalar[Self.dtype]):
        self.data[r * Self.D + c] = v

    # -- arithmetic -----------------------------------------------------------

    def __mul__(self, other: Self) -> Self:
        """Matrix product `self @ other`."""
        var out = Self()
        for i in range(Self.D):
            for k in range(Self.D):
                var a = self.data[i * Self.D + k]
                if a == 0:
                    continue
                for j in range(Self.D):
                    out.data[i * Self.D + j] += a * other.data[k * Self.D + j]
        return out^

    def __add__(self, other: Self) -> Self:
        var out = Self()
        for i in range(Self.SIZE):
            out.data[i] = self.data[i] + other.data[i]
        return out^

    def __sub__(self, other: Self) -> Self:
        var out = Self()
        for i in range(Self.SIZE):
            out.data[i] = self.data[i] - other.data[i]
        return out^

    def scaled(self, s: Scalar[Self.dtype]) -> Self:
        var out = Self()
        for i in range(Self.SIZE):
            out.data[i] = self.data[i] * s
        return out^

    def transpose(self) -> Self:
        var out = Self()
        for i in range(Self.D):
            for j in range(Self.D):
                out.data[j * Self.D + i] = self.data[i * Self.D + j]
        return out^

    def matvec(
        self, v: InlineArray[Scalar[Self.dtype], Self.D]
    ) -> InlineArray[Scalar[Self.dtype], Self.D]:
        var out = InlineArray[Scalar[Self.dtype], Self.D](fill=0)
        for i in range(Self.D):
            var s = Scalar[Self.dtype](0)
            for j in range(Self.D):
                s += self.data[i * Self.D + j] * v[j]
            out[i] = s
        return out^

    # -- scalar summaries -----------------------------------------------------

    def trace(self) -> Scalar[Self.dtype]:
        var s = Scalar[Self.dtype](0)
        for i in range(Self.D):
            s += self.data[i * Self.D + i]
        return s

    def frobenius_norm(self) -> Scalar[Self.dtype]:
        var s = Scalar[Self.dtype](0)
        for i in range(Self.SIZE):
            s += self.data[i] * self.data[i]
        return sqrt(s)

    def dist_to_identity(self) -> Scalar[Self.dtype]:
        """`||self - I||_F` — the continuous part of a holonomy reading."""
        return (self - Self.identity()).frobenius_norm()

    def max_abs_diff(self, other: Self) -> Scalar[Self.dtype]:
        var worst = Scalar[Self.dtype](0)
        for i in range(Self.SIZE):
            var d = abs(self.data[i] - other.data[i])
            if d > worst:
                worst = d
        return worst

    def orthogonality_error(self) -> Scalar[Self.dtype]:
        """`max |R^T R - I|` — zero exactly when the matrix is in O(D)."""
        return (self.transpose() * self).max_abs_diff(Self.identity())

    def is_orthogonal(self, tol: Scalar[Self.dtype] = 1e-10) -> Bool:
        return self.orthogonality_error() <= tol

    # -- LU-based determinant and inverse -------------------------------------

    def det(self) -> Scalar[Self.dtype]:
        """Determinant by Gaussian elimination with partial pivoting.

        The SIGN is the observable that matters here (`det H in {+1, -1}` is the
        Z/2 invariant); the magnitude is only ever used as a sanity check.
        """
        var a = self.data.copy()
        var sign = Scalar[Self.dtype](1)
        for col in range(Self.D):
            var piv = col
            var best = abs(a[col * Self.D + col])
            for r in range(col + 1, Self.D):
                var v = abs(a[r * Self.D + col])
                if v > best:
                    best = v
                    piv = r
            if best == 0:
                return 0
            if piv != col:
                for j in range(Self.D):
                    var t = a[col * Self.D + j]
                    a[col * Self.D + j] = a[piv * Self.D + j]
                    a[piv * Self.D + j] = t
                sign = -sign
            var d = a[col * Self.D + col]
            for r in range(col + 1, Self.D):
                var f = a[r * Self.D + col] / d
                if f == 0:
                    continue
                for j in range(col, Self.D):
                    a[r * Self.D + j] -= f * a[col * Self.D + j]
        var out = sign
        for i in range(Self.D):
            out *= a[i * Self.D + i]
        return out

    def inverse(self) raises -> Self:
        """Gauss-Jordan inverse. Raises on a singular matrix."""
        var a = self.data.copy()
        var inv = Self.identity()
        for col in range(Self.D):
            var piv = col
            var best = abs(a[col * Self.D + col])
            for r in range(col + 1, Self.D):
                var v = abs(a[r * Self.D + col])
                if v > best:
                    best = v
                    piv = r
            if best == 0:
                raise Error("SqMat.inverse: singular matrix")
            if piv != col:
                for j in range(Self.D):
                    var t = a[col * Self.D + j]
                    a[col * Self.D + j] = a[piv * Self.D + j]
                    a[piv * Self.D + j] = t
                    var u = inv.data[col * Self.D + j]
                    inv.data[col * Self.D + j] = inv.data[piv * Self.D + j]
                    inv.data[piv * Self.D + j] = u
            var d = a[col * Self.D + col]
            for j in range(Self.D):
                a[col * Self.D + j] /= d
                inv.data[col * Self.D + j] /= d
            for r in range(Self.D):
                if r == col:
                    continue
                var f = a[r * Self.D + col]
                if f == 0:
                    continue
                for j in range(Self.D):
                    a[r * Self.D + j] -= f * a[col * Self.D + j]
                    inv.data[r * Self.D + j] -= f * inv.data[col * Self.D + j]
        return inv^

    def inverse_transpose(self) raises -> Self:
        return self.inverse().transpose()


# =============================================================================
# so(D) -> SO(D)
# =============================================================================


def skew_from_vector[
    D: Int, dtype: DType = DType.float64
](v: Span[Scalar[dtype], _]) raises -> SqMat[D, dtype]:
    """Pack the `D(D-1)/2` free parameters of so(D) into a skew matrix.

    Entry `k` fills the strictly-lower-triangular slot `(i, j)`, `i > j`, in
    row-major order, and its negation goes to `(j, i)`.

    The argument is a `Span` and not a sized `InlineArray` on purpose. A
    signature of `InlineArray[..., D * (D - 1) // 2]` type-checks when `D` is
    itself a parameter, but FAILS to fold when a caller passes a literal
    (`skew_from_vector[2]` -> "types parameters include unfolded expression at
    parser time"). Since Phase 3 generates these coefficients at runtime from
    `W_a a + W_l l + W_c c` anyway, a span is both the working and the natural
    spelling. Length is checked here rather than by the type.
    """
    if len(v) != SqMat[D, dtype].SKEW_PARAMS:
        raise Error(
            "skew_from_vector: expected "
            + String(SqMat[D, dtype].SKEW_PARAMS)
            + " coefficients for D="
            + String(D)
            + ", got "
            + String(len(v))
        )
    var s = SqMat[D, dtype]()
    var k = 0
    for i in range(D):
        for j in range(i):
            s[i, j] = v[k]
            s[j, i] = -v[k]
            k += 1
    return s^


def cayley[
    D: Int, dtype: DType = DType.float64
](s: SqMat[D, dtype]) raises -> SqMat[D, dtype]:
    """`(I - S)(I + S)^-1` for skew `S`. Always lands in SO(D).

    `I + S` is never singular for skew `S` (its eigenvalues are `1 + i*theta`),
    so the inverse here cannot raise for a genuinely skew argument.
    """
    var eye = SqMat[D, dtype].identity()
    return (eye - s) * (eye + s).inverse()


def expm_skew[
    D: Int, dtype: DType = DType.float64
](s: SqMat[D, dtype]) -> SqMat[D, dtype]:
    """`exp(S)` for skew `S` by scaling-and-squaring with a Taylor series.

    Also lands in SO(D). Used for the Riemannian transport update
    `R <- R exp(-lr * skew(R^T grad))`, which must stay on the manifold rather
    than drift off it and be re-projected.
    """
    comptime TAYLOR_TERMS = 18
    var nrm = Float64(s.frobenius_norm())
    var squarings = 0
    if nrm > 0.5:
        squarings = Int(ceil(log2(nrm / 0.5)))
    var scale = Scalar[dtype](1.0 / Float64(1 << squarings))
    var a = s.scaled(scale)

    var out = SqMat[D, dtype].identity()
    var term = SqMat[D, dtype].identity()
    for k in range(1, TAYLOR_TERMS + 1):
        term = (term * a).scaled(Scalar[dtype](1.0 / Float64(k)))
        out = out + term
    for _ in range(squarings):
        out = out * out
    return out^


def householder[
    D: Int, dtype: DType = DType.float64
](v: Span[Scalar[dtype], _]) raises -> SqMat[D, dtype]:
    """`I - 2 v v^T / (v^T v)`, a reflection: orthogonal with `det = -1`.

    This is the `Q` of §4.2 — the fixed generator of the other component of
    O(D). The per-edge orientation bit selects between `R` and `Q R`.
    """
    if len(v) != D:
        raise Error(
            "householder: expected D=" + String(D) + " components, got "
            + String(len(v))
        )
    var vv = Scalar[dtype](0)
    for i in range(D):
        vv += v[i] * v[i]
    var m = SqMat[D, dtype].identity()
    if vv == 0:
        return m^
    var f = Scalar[dtype](2) / vv
    for i in range(D):
        for j in range(D):
            m[i, j] = m[i, j] - f * v[i] * v[j]
    return m^


def fixed_subspace_dim[
    D: Int, dtype: DType = DType.float64
](h: SqMat[D, dtype], tol: Float64 = 1e-8) -> Int:
    """`dim ker(H - I)`: how many latent directions admit a global frame.

    This is the reading the design doc asks for in dimensions above two
    (§2). In 2D a reflection fixes a line, so `det H = -1` implies a
    one-dimensional fixed subspace and the two readings agree. In 3D they come
    apart: `H = -I` has `det = -1` and NO fixed vector at all. So `det H` is the
    Z/2 class and `dim ker(H - I)` is the finer statement of WHICH directions
    survive; a method that only ever reads the determinant is under-reporting
    above 2D.

    Rank by Gaussian elimination with partial pivoting; the dimension is
    `D - rank`.
    """
    var a = (h - SqMat[D, dtype].identity()).data.copy()
    var scale = Float64(0)
    for i in range(D * D):
        var v = abs(Float64(a[i]))
        if v > scale:
            scale = v
    if scale < tol:
        return D
    var rank = 0
    var row = 0
    for col in range(D):
        var piv = -1
        var best = tol * scale
        for r in range(row, D):
            var v = abs(Float64(a[r * D + col]))
            if v > best:
                best = v
                piv = r
        if piv < 0:
            continue
        if piv != row:
            for j in range(D):
                var t = a[row * D + j]
                a[row * D + j] = a[piv * D + j]
                a[piv * D + j] = t
        var d = a[row * D + col]
        for r in range(row + 1, D):
            var f = a[r * D + col] / d
            if f == 0:
                continue
            for j in range(col, D):
                a[r * D + j] -= f * a[row * D + j]
        rank += 1
        row += 1
        if row == D:
            break
    return D - rank
