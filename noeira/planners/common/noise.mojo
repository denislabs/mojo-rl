"""Host-side noise sampling helpers used by planners.

Phase 0 ships only CPU helpers that mirror the convention already used by
`tdmpc2/mppi.mojo` (Box-Muller via `std.random.random_float64`). GPU kernel
noise samplers (Philox-backed) live alongside their planner kernels and are
not duplicated here — see `muzero/gpu_mcts.mojo` for the Dirichlet pattern.

Helpers:
  - gaussian_sample()           — standard normal via Box-Muller
  - uniform_sample(lo, hi)      — uniform in [lo, hi)
  - gumbel_sample()             — standard Gumbel via inverse CDF
  - GaussianRng                 — reproducible host RNG with explicit seed
"""

from std.math import sqrt, log, cos
from std.random import random_float64, seed as set_seed


comptime _TWO_PI: Float64 = 6.283185307179586
comptime _LOG_EPS: Float64 = 1e-10


@always_inline
def gaussian_sample() -> Float64:
    """Standard normal sample via Box-Muller.

    Matches `tdmpc2/mppi._gaussian_sample` so host-side planner behavior is
    bitwise identical when this helper replaces the inline version in Phase 2.
    """
    var u1 = random_float64()
    var u2 = random_float64()
    if u1 < _LOG_EPS:
        u1 = _LOG_EPS
    return sqrt(-2.0 * log(u1)) * cos(_TWO_PI * u2)


@always_inline
def uniform_sample(lo: Float64, hi: Float64) -> Float64:
    """Uniform sample in [lo, hi)."""
    return lo + (hi - lo) * random_float64()


@always_inline
def gumbel_sample() -> Float64:
    """Standard Gumbel(0, 1) via inverse CDF: -log(-log(U)).

    Used by EZv2's Gumbel-Top-k root candidate selection. Numerical guard
    avoids log(0) when U is extremely close to 0 or 1.
    """
    var u = random_float64()
    if u < _LOG_EPS:
        u = _LOG_EPS
    if u > 1.0 - _LOG_EPS:
        u = 1.0 - _LOG_EPS
    return -log(-log(u))


@fieldwise_init
struct GaussianRng(Copyable, Movable):
    """Wrapper around the host RNG with an explicit seed entry point.

    Phase 0 keeps the implementation a thin reseed-on-construct shim so tests
    that need bit-reproducible Gaussian draws can write:

        var rng = GaussianRng(seed=42)
        var z = rng.sample()

    A future revision will swap in a stateful Philox stream so multiple Rngs
    can coexist without seed collisions; the API is forward-compatible.
    """

    var _seed: Int

    @staticmethod
    def with_seed(s: Int) -> Self:
        return Self(_seed=s)

    def sample(self) -> Float64:
        set_seed(self._seed)
        return gaussian_sample()
