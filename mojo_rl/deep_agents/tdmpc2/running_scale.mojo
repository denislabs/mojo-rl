"""RunningScale — EMA of the (95th − 5th) percentile of Q-values.

Ports `references/tdmpc2-main/tdmpc2/common/scale.py:RunningScale`. The
policy loss divides Q by this running scale so the entropy term and the
Q term stay commensurate as Q magnitudes drift during training.

  update(x): range = pctl_95(x) − pctl_5(x) (linear-interpolated, audit #14);
             value ← lerp(value, range, tau)
  scaling:   q / value

CPU only for P1/P2 (the trainer downloads the t=0 Q vector and updates
host-side; reference does the same percentile on a small [batch] vector).
"""

from std.math import floor

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor


struct RunningScale(Movable & ImplicitlyDeletable):
    var value: Scalar[DT]
    var tau: Scalar[DT]

    def __init__(out self, tau: Scalar[DT] = Scalar[DT](0.01)):
        self.value = Scalar[DT](1.0)
        self.tau = tau

    @always_inline
    def _interp_pctl(
        self,
        sorted: List[Scalar[DT]],
        n: Int,
        pct: Scalar[DT],
    ) -> Scalar[DT]:
        # position = pct·(n−1)/100 ; linear interp between floor/ceil ranks.
        var pos = pct * Scalar[DT](n - 1) / Scalar[DT](100.0)
        var fl = floor(pos)
        var fi = Int(fl)
        var ci = fi + 1
        if ci > n - 1:
            ci = n - 1
        var w_ceil = pos - fl
        var w_floor = Scalar[DT](1.0) - w_ceil
        return sorted[fi] * w_floor + sorted[ci] * w_ceil

    def update_from(
        mut self,
        ref x: Tensor,
        n: Int,
    ):
        """Update the running scale from the first `n` Q-values in `x` (e.g.
        the t=0 Q estimates over the batch, downloaded host-side)."""
        if n <= 1:
            return
        var s = List[Scalar[DT]](length=n, fill=Scalar[DT](0.0))
        for i in range(n):
            s[i] = x.data[i]
        # insertion sort (n = batch size, small).
        for i in range(1, n):
            var key = s[i]
            var j = i - 1
            while j >= 0 and s[j] > key:
                s[j + 1] = s[j]
                j -= 1
            s[j + 1] = key
        var p5 = self._interp_pctl(s, n, Scalar[DT](5.0))
        var p95 = self._interp_pctl(s, n, Scalar[DT](95.0))
        var rng = p95 - p5
        # lerp(value, rng, tau)
        self.value = self.value + self.tau * (rng - self.value)

    @always_inline
    def inv(self) -> Scalar[DT]:
        """1 / value — the multiplier applied to Q in the policy loss."""
        return Scalar[DT](1.0) / self.value
