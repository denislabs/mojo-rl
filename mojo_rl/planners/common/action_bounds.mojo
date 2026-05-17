"""Action bounds helpers — clipping, tanh squashing, range rescaling.

Used by CEM/MPPI/Sampled-MCTS to keep candidate actions inside the env's
action space. Phase 0 provides Float64 host helpers; GPU vectorized variants
arrive when the planners are migrated (Phase 1/2).
"""

from std.math import tanh as math_tanh


# ─── Scalar helpers ───────────────────────────────────────────────────────


def clip(x: Float64, lo: Float64, hi: Float64) -> Float64:
    """Clamp x to [lo, hi]. Pure function — useful in tight inner loops."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def tanh_squash(x: Float64) -> Float64:
    """Squash an unbounded real to (-1, 1) via tanh. Used by stochastic
    actors (SAC, TD-MPC2) before scaling to env bounds."""
    return math_tanh(x)


def scale_to_range(
    x_in_unit: Float64, lo: Float64, hi: Float64
) -> Float64:
    """Map x in [-1, 1] to [lo, hi] linearly.

    Inverse of `(x - lo) / (hi - lo) * 2 - 1`.
    No assertion that x is in [-1, 1] — caller's responsibility.
    """
    var half_range = (hi - lo) * 0.5
    var mid = (hi + lo) * 0.5
    return mid + x_in_unit * half_range


# ─── List helpers ─────────────────────────────────────────────────────────


def clip_inplace(mut buf: List[Float64], lo: Float64, hi: Float64):
    """Clip every element of buf to [lo, hi] in place."""
    for i in range(len(buf)):
        buf[i] = clip(buf[i], lo, hi)
