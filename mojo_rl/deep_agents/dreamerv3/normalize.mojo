"""PercentileNormalize — DreamerV3 running return/value/advantage normalizer.

Ports `references/dreamerv3-main/embodied/jax/utils.py:Normalize`. Three
impls used by DreamerV3:

  * `perc`    — retnorm `{rate:0.01, perclo:5, perchi:95, limit:1.0,
                debias:False}`. EMA of the 5th/95th percentiles; stats
                return `(offset=lo, scale=max(limit, hi-lo))`.
  * `none`    — valnorm / advnorm → always `(0.0, 1.0)` (identity).
  * `meanstd` — EMA of mean and mean-of-squares; included for parity even
                though the DreamerV3 default config doesn't use it.

`debias` (default True in the reference class, but `False` in retnorm's
config) divides the read state by `max(rate, corr)` where `corr` is an
EMA of 1.0 — a bias-correction for the cold-start EMA.

Pure statistics on detached inputs — no gradient. Percentile uses linear
interpolation matching `jnp.percentile` (method='linear').
"""

from std.math import sqrt

from mojo_rl.nn.constants import DT


@always_inline
def _percentile_linear(
    sorted_buf: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, q: Scalar[DT]
) -> Scalar[DT]:
    """`jnp.percentile(x, q)` with method='linear' over a pre-sorted buffer.
    virtual index = (q/100)·(n-1); linear interp between neighbours."""
    if n == 1:
        return sorted_buf[0]
    var idx = (q / Scalar[DT](100.0)) * Scalar[DT](n - 1)
    var lo = Int(idx)
    if lo >= n - 1:
        return sorted_buf[n - 1]
    var frac = idx - Scalar[DT](lo)
    return sorted_buf[lo] + frac * (sorted_buf[lo + 1] - sorted_buf[lo])


@always_inline
def _insertion_sort(buf: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int):
    for i in range(1, n):
        var key = buf[i]
        var j = i - 1
        while j >= 0 and buf[j] > key:
            buf[j + 1] = buf[j]
            j -= 1
        buf[j + 1] = key


struct PercentileNormalize(Movable & ImplicitlyDeletable):
    # impl: "none" | "perc" | "meanstd"
    var impl: String
    var rate: Scalar[DT]
    var perclo: Scalar[DT]
    var perchi: Scalar[DT]
    var limit: Scalar[DT]
    var debias: Bool

    # EMA state.
    var lo: Scalar[DT]
    var hi: Scalar[DT]
    var mean: Scalar[DT]
    var sqrs: Scalar[DT]
    var corr: Scalar[DT]

    def __init__(out self):
        self.impl = String("none")
        self.rate = Scalar[DT](0.01)
        self.perclo = Scalar[DT](5.0)
        self.perchi = Scalar[DT](95.0)
        self.limit = Scalar[DT](1e-8)
        self.debias = True
        self.lo = Scalar[DT](0.0)
        self.hi = Scalar[DT](0.0)
        self.mean = Scalar[DT](0.0)
        self.sqrs = Scalar[DT](0.0)
        self.corr = Scalar[DT](0.0)

    @staticmethod
    def make(
        impl: String,
        rate: Scalar[DT] = Scalar[DT](0.01),
        perclo: Scalar[DT] = Scalar[DT](5.0),
        perchi: Scalar[DT] = Scalar[DT](95.0),
        limit: Scalar[DT] = Scalar[DT](1e-8),
        debias: Bool = True,
    ) -> Self:
        var p = Self()
        p.impl = impl
        p.rate = rate
        p.perclo = perclo
        p.perchi = perchi
        p.limit = limit
        p.debias = debias
        return p^

    def update(
        mut self, x: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int
    ) raises:
        """EMA-update the running statistics from a flat sample `x[0:n]`."""
        var keep = Scalar[DT](1.0) - self.rate
        if self.impl == "none":
            return
        if self.impl == "perc":
            # Sort a scratch copy for percentile reads.
            var tmp = List[Scalar[DT]](length=n, fill=Scalar[DT](0.0))
            var tp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                tmp.unsafe_ptr()
            )
            for i in range(n):
                tp[i] = x[i]
            _insertion_sort(tp, n)
            var plo = _percentile_linear(tp, n, self.perclo)
            var phi = _percentile_linear(tp, n, self.perchi)
            self.lo = keep * self.lo + self.rate * plo
            self.hi = keep * self.hi + self.rate * phi
        elif self.impl == "meanstd":
            var s: Scalar[DT] = 0.0
            var s2: Scalar[DT] = 0.0
            for i in range(n):
                s += x[i]
                s2 += x[i] * x[i]
            self.mean = keep * self.mean + self.rate * (s / Scalar[DT](n))
            self.sqrs = keep * self.sqrs + self.rate * (s2 / Scalar[DT](n))
        else:
            raise Error("PercentileNormalize: unknown impl '" + self.impl + "'")
        if self.debias and self.impl != "none":
            self.corr = keep * self.corr + self.rate * Scalar[DT](1.0)

    def stats(self) raises -> Tuple[Scalar[DT], Scalar[DT]]:
        """Return `(offset, scale)`."""
        var corr = Scalar[DT](1.0)
        if self.debias and self.impl != "none":
            var c = self.corr
            corr = Scalar[DT](1.0) / (self.rate if self.rate > c else c)
        if self.impl == "none":
            return Tuple(Scalar[DT](0.0), Scalar[DT](1.0))
        elif self.impl == "perc":
            var lo = self.lo * corr
            var hi = self.hi * corr
            var span = hi - lo
            return Tuple(lo, self.limit if self.limit > span else span)
        elif self.impl == "meanstd":
            var mean = self.mean * corr
            var v = self.sqrs * corr - mean * mean
            var std = sqrt(v if v > Scalar[DT](0.0) else Scalar[DT](0.0))
            return Tuple(mean, self.limit if self.limit > std else std)
        else:
            raise Error("PercentileNormalize: unknown impl '" + self.impl + "'")
