"""Learned orthogonal transports, indexed by (action, place) — never by latents.

`det H` is EXACTLY the product of the per-edge orientation bits around a cycle:
Cayley and `exp` both land in SO(D), so no continuous mechanism — not the angle,
not the Riemannian gradient, not the confidence weight — contributes anything to
the Z/2 class. All of the risk in the whole method is concentrated in bit
selection, which is why this file spends more care on the bit than on the angle.

## The bit needs hysteresis, and a naive argmin will not do

The design doc selects the bit by a greedy per-edge `argmin` over residuals. The
numpy prototype gets away with that because its SVD decides in closed form over
160 pairs at once. Online, with a residual that is noisy and an encoder that is
still moving, a bare argmin chatters. Worse, the obvious cheap version is
outright wrong: if you train ONE rotation `R` under the current bit and compare
`R` against `QR`, the loser is not a fair candidate — `R` has been fitted to the
wrong component, so `QR` is `Q` times a bad rotation rather than the best
reflection.

So both branches are carried and trained: `R_plus` in SO(D), and `Q R_minus`
with `R_minus` in SO(D) trained on the same transitions. Each is the best of its
component. Selection compares like with like, and switches only on a margin
after a minimum number of observations.

## The rule that is enforced by the signature

`transport_for` takes an action and a place. It does NOT take a latent, and it
never will: conditioning `R` on the coordinates of the fibres it relates lets a
network manufacture the matrix that maps `u_p` to `u_q`, the glue loss becomes
identically zero, and the holonomy stops measuring anything (v2 §4.2). Keeping
`u` out of the signature makes that failure a compile error rather than a code
review.

Indexing by a table of one-hot (action, place) IS the generator `rho(a, l, c)`
of the design doc with one-hot inputs and no parameter sharing. Sharing (and the
Cayley parametrisation that goes with a continuous place code) is what a scaled
version needs; on E1's twelve oracle places it would only add a fitting problem
between the measurement and the observable.
"""

from std.math import abs, sqrt

from .so_d import SqMat, expm_skew, householder


struct TransportTable[
    D: Int, N_ACTIONS: Int, N_PLACES: Int, dtype: DType = DType.float64
](Copyable, Movable):
    """One O(D) transport per (action, place), both components carried."""

    comptime N_ENTRIES: Int = Self.N_ACTIONS * Self.N_PLACES

    var r_plus: List[SqMat[Self.D, Self.dtype]]
    """Best fit within SO(D)."""
    var r_minus: List[SqMat[Self.D, Self.dtype]]
    """Core of the reflected branch; the candidate transport is `Q r_minus`."""
    var q: SqMat[Self.D, Self.dtype]
    """The fixed Householder reflection generating the other component."""

    var sigma: List[Int8]
    var res_plus: List[Float64]
    """EWMA of the squared residual of the SO(D) branch."""
    var res_minus: List[Float64]
    var seen: List[Int]
    var flips: List[Int]
    """How many times each entry's bit changed — the stability metric."""

    var ewma_decay: Float64
    var flip_margin: Float64
    """Relative margin the challenger must beat the incumbent by."""
    var min_observations: Int

    def __init__(
        out self,
        ewma_decay: Float64 = 0.99,
        flip_margin: Float64 = 0.25,
        min_observations: Int = 64,
    ) raises:
        self.r_plus = List[SqMat[Self.D, Self.dtype]]()
        self.r_minus = List[SqMat[Self.D, Self.dtype]]()
        for _ in range(Self.N_ENTRIES):
            self.r_plus.append(SqMat[Self.D, Self.dtype].identity())
            self.r_minus.append(SqMat[Self.D, Self.dtype].identity())
        var v = List[Scalar[Self.dtype]](length=Self.D, fill=0)
        v[0] = 0
        v[Self.D - 1] = 1
        self.q = householder[Self.D, Self.dtype](Span(v))
        self.sigma = List[Int8](length=Self.N_ENTRIES, fill=1)
        self.res_plus = List[Float64](length=Self.N_ENTRIES, fill=0)
        self.res_minus = List[Float64](length=Self.N_ENTRIES, fill=0)
        self.seen = List[Int](length=Self.N_ENTRIES, fill=0)
        self.flips = List[Int](length=Self.N_ENTRIES, fill=0)
        self.ewma_decay = ewma_decay
        self.flip_margin = flip_margin
        self.min_observations = min_observations

    def __init__(out self, *, copy: Self):
        self.r_plus = copy.r_plus.copy()
        self.r_minus = copy.r_minus.copy()
        self.q = copy.q.copy()
        self.sigma = copy.sigma.copy()
        self.res_plus = copy.res_plus.copy()
        self.res_minus = copy.res_minus.copy()
        self.seen = copy.seen.copy()
        self.flips = copy.flips.copy()
        self.ewma_decay = copy.ewma_decay
        self.flip_margin = copy.flip_margin
        self.min_observations = copy.min_observations

    def __init__(out self, *, deinit move: Self):
        self.r_plus = move.r_plus^
        self.r_minus = move.r_minus^
        self.q = move.q^
        self.sigma = move.sigma^
        self.res_plus = move.res_plus^
        self.res_minus = move.res_minus^
        self.seen = move.seen^
        self.flips = move.flips^
        self.ewma_decay = move.ewma_decay
        self.flip_margin = move.flip_margin
        self.min_observations = move.min_observations

    def index(self, action: Int, place: Int) -> Int:
        return action * Self.N_PLACES + place

    def candidate(self, entry: Int, branch: Int) -> SqMat[Self.D, Self.dtype]:
        """`branch = 0` -> the SO(D) fit; `branch = 1` -> the reflected fit."""
        if branch == 0:
            return self.r_plus[entry].copy()
        return self.q * self.r_minus[entry]

    def transport_for(
        self, action: Int, place: Int
    ) -> SqMat[Self.D, Self.dtype]:
        """The selected transport. Takes NO latent, by design (see the header)."""
        var e = self.index(action, place)
        return self.candidate(e, 0 if self.sigma[e] > 0 else 1)

    def transport_at(self, entry: Int) -> SqMat[Self.D, Self.dtype]:
        return self.candidate(entry, 0 if self.sigma[entry] > 0 else 1)

    # -- learning -------------------------------------------------------------

    def observe(
        mut self,
        action: Int,
        place: Int,
        u_src: List[Scalar[Self.dtype]],
        u_dst: List[Scalar[Self.dtype]],
        lr: Float64,
        allow_flip: Bool,
    ) raises -> List[Scalar[Self.dtype]]:
        """Train BOTH branches on one transition; return `dL/du_src` of the
        selected branch.

        Riemannian update per branch: `grad = 2 eps u_src^T`, projected onto
        so(D) and applied as `R <- R exp(-lr * skew(R^T grad))`, so the matrix
        stays exactly on the manifold instead of drifting off and being
        re-projected. The target is the encoder's own `u_dst` (a PRE-CONSENSUS
        anchor) and carries no gradient here: a consensus target would train the
        transports to reduce the harmonic charge, which is cocycle minimisation
        by the back door (v2 §4.5).
        """
        var e = self.index(action, place)

        var res = List[Float64](length=2, fill=0)
        for branch in range(2):
            var r = self.candidate(e, branch)
            var eps = List[Scalar[Self.dtype]](length=Self.D, fill=0)
            var sq = Float64(0)
            for i in range(Self.D):
                var pred = Scalar[Self.dtype](0)
                for j in range(Self.D):
                    pred += r[i, j] * u_src[j]
                eps[i] = pred - u_dst[i]
                sq += Float64(eps[i] * eps[i])
            res[branch] = sq

            # grad wrt the candidate matrix, then onto so(D).
            var grad = SqMat[Self.D, Self.dtype]()
            for i in range(Self.D):
                for j in range(Self.D):
                    grad[i, j] = Scalar[Self.dtype](2) * eps[i] * u_src[j]
            # The reflected branch's free parameter is r_minus, and
            # d(Q r_minus)/d(r_minus) = Q^T applied on the left.
            var core = self.r_plus[e].copy() if branch == 0 else self.r_minus[e].copy()
            var g_core = grad.copy() if branch == 0 else self.q.transpose() * grad
            var a = core.transpose() * g_core
            var skew = (a - a.transpose()).scaled(Scalar[Self.dtype](0.5))
            var upd = core * expm_skew[Self.D, Self.dtype](
                skew.scaled(Scalar[Self.dtype](-lr))
            )
            if branch == 0:
                self.r_plus[e] = upd^
            else:
                self.r_minus[e] = upd^

        var d = self.ewma_decay
        self.res_plus[e] = d * self.res_plus[e] + (1.0 - d) * res[0]
        self.res_minus[e] = d * self.res_minus[e] + (1.0 - d) * res[1]
        self.seen[e] += 1

        if allow_flip:
            self.maybe_flip(e)

        # dL/du_src for the SELECTED branch: 2 R^T eps.
        var sel = self.transport_at(e)
        var d_src = List[Scalar[Self.dtype]](length=Self.D, fill=0)
        for i in range(Self.D):
            var pred = Scalar[Self.dtype](0)
            for j in range(Self.D):
                pred += sel[i, j] * u_src[j]
            var ei = pred - u_dst[i]
            for j in range(Self.D):
                d_src[j] += Scalar[Self.dtype](2) * sel[i, j] * ei
        return d_src^

    def maybe_flip(mut self, entry: Int):
        """Switch component only on a MARGIN and after enough observations.

        A bare argmin chatters: the two residuals are noisy and the encoder is
        still moving, so the sign of their difference flips on measurement noise
        long before it means anything.
        """
        if self.seen[entry] < self.min_observations:
            return
        var incumbent = (
            self.res_plus[entry] if self.sigma[entry] > 0 else self.res_minus[entry]
        )
        var challenger = (
            self.res_minus[entry] if self.sigma[entry] > 0 else self.res_plus[entry]
        )
        if challenger < incumbent * (1.0 - self.flip_margin):
            self.sigma[entry] = -self.sigma[entry]
            self.flips[entry] += 1

    def total_flips(self) -> Int:
        var n = 0
        for i in range(Self.N_ENTRIES):
            n += self.flips[i]
        return n

    def n_reflected(self) -> Int:
        var n = 0
        for i in range(Self.N_ENTRIES):
            if self.sigma[i] < 0:
                n += 1
        return n
