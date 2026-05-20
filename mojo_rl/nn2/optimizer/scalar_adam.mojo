"""ScalarAdam — single-scalar Adam optimizer.

Phase 9B. Bespoke (NOT `Optimizer`-conforming, since the `Optimizer` trait
walks Module-tree params via `for_each_param`). Used for SAC's auto-tuned
entropy temperature `log_alpha`: a single trainable scalar with its own
moments and bias-correction.

Hyperparameters mirror Adam (β₁=0.9, β₂=0.999, ε=1e-8). Bias correction
is computed by iterating β^t — fine for `t` up to a few hundred thousand.
"""

from std.math import sqrt as fsqrt

from ..constants import DT


@fieldwise_init
struct ScalarAdam(Movable & ImplicitlyDestructible):
    var value: Scalar[DT]
    var m: Scalar[DT]
    var v: Scalar[DT]
    var t: Int
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]

    @staticmethod
    def new(initial: Scalar[DT], lr: Scalar[DT]) -> Self:
        return Self(
            value=initial, m=0.0, v=0.0, t=0,
            lr=lr, beta1=0.9, beta2=0.999, eps=1e-8,
        )

    def step(mut self, grad: Scalar[DT]):
        self.t += 1
        var one: Scalar[DT] = 1.0
        self.m = self.beta1 * self.m + (one - self.beta1) * grad
        self.v = self.beta2 * self.v + (one - self.beta2) * grad * grad
        var b1t = self.beta1
        var b2t = self.beta2
        for _ in range(self.t - 1):
            b1t *= self.beta1
            b2t *= self.beta2
        var m_hat = self.m / (one - b1t)
        var v_hat = self.v / (one - b2t)
        self.value = self.value - self.lr * m_hat / (fsqrt(v_hat) + self.eps)
