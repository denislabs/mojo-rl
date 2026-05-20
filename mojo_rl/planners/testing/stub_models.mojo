"""Stub world models for isolated planner tests.

Each stub is tiny, deterministic, and closed-form-checkable so that planner
correctness can be asserted without loading a trained agent or buffer. Phase 1
will wrap these in `RolloutCallback` / `Representation` / `Dynamics` /
`Prediction` adapters; Phase 0 just ships the raw oracles.

Available stubs:
  - IdentityDynamics       z' = z + a
  - GoalReachReward        r(z) = -‖z - goal‖²
  - LinearQuadratic1D      z' = A z + B a, r = -(Q z² + R a²)  (has closed-form LQR)
  - TwoArmBandit           2-action Bernoulli; expected reward per arm
  - KnownValueTree         branching B depth D; ground-truth max / negamax root value
"""

from std.math import sqrt


# ─── Pairable dynamics + reward ───────────────────────────────────────────


struct IdentityDynamics(Copyable, Movable):
    """IdentityDynamics: z' = z + a — pure pass-through, no internal state.

    Paired with `GoalReachReward`: the one-step-optimal action from state z
    is exactly `goal - z` since reward is maximized when z' = goal.
    """

    @staticmethod
    def step(z: List[Float64], a: List[Float64]) raises -> List[Float64]:
        if len(z) != len(a):
            raise Error("IdentityDynamics: dim mismatch z vs a")
        var out = List[Float64](capacity=len(z))
        for i in range(len(z)):
            out.append(z[i] + a[i])
        return out^


@fieldwise_init
struct GoalReachReward(Copyable, Movable):
    """GoalReachReward: r(z) = -‖z - goal‖². Strictly concave in z; argmax is z = goal.
    """

    var goal: List[Float64]

    def reward(self, z: List[Float64]) raises -> Float64:
        if len(z) != len(self.goal):
            raise Error("GoalReachReward: dim mismatch")
        var s: Float64 = 0.0
        for i in range(len(z)):
            var d = z[i] - self.goal[i]
            s += d * d
        return -s

    def distance(self, z: List[Float64]) raises -> Float64:
        """Euclidean distance to goal — convenience for tolerance asserts."""
        var s: Float64 = 0.0
        if len(z) != len(self.goal):
            raise Error("GoalReachReward: dim mismatch")
        for i in range(len(z)):
            var d = z[i] - self.goal[i]
            s += d * d
        return sqrt(s)


# ─── Scalar linear-quadratic system ───────────────────────────────────────


@fieldwise_init
struct LinearQuadratic1D(Copyable, Movable):
    """Scalar LQ: z' = A·z + B·a, r = -(Q·z² + R·a²).

    Has a closed-form LQR optimal policy, so MPPI / iLQR can be checked
    against the analytic answer. For an infinite-horizon problem with these
    dynamics, the optimal feedback is a* = -K·z where K = (B·P·A) / (R + B²·P)
    and P solves the scalar Riccati equation P = Q + A²·P - (A·B·P)² / (R + B²·P).
    """

    var A: Float64
    var B: Float64
    var Q: Float64
    var R: Float64

    def step(self, z: Float64, a: Float64) -> Float64:
        return self.A * z + self.B * a

    def reward(self, z: Float64, a: Float64) -> Float64:
        return -(self.Q * z * z + self.R * a * a)

    def lqr_gain_infinite_horizon(self) -> Float64:
        """Solve the scalar discrete-time algebraic Riccati equation by
        fixed-point iteration. Returns gain K such that a* = -K·z.

        Converges in <50 iterations for any stable (Q,R > 0) problem.
        """
        var P: Float64 = self.Q
        for _ in range(200):
            var denom = self.R + self.B * self.B * P
            var P_next = (
                self.Q
                + self.A * self.A * P
                - (self.A * self.B * P) * (self.A * self.B * P) / denom
            )
            if abs(P_next - P) < 1e-12:
                P = P_next
                break
            P = P_next
        var denom = self.R + self.B * self.B * P
        return (self.B * P * self.A) / denom


# ─── Discrete bandit ──────────────────────────────────────────────────────


@fieldwise_init
struct TwoArmBandit(Copyable, Movable):
    """Two-armed Bernoulli bandit with known mean rewards.

    Useful as the simplest possible MCTS / PUCT smoke test: with a uniform
    prior and N simulations, the visit count for the better arm should
    dominate by roughly O(N) and not just O(sqrt N)."""

    var p_left: Float64
    var p_right: Float64

    def expected_reward(self, action: Int) raises -> Float64:
        if action == 0:
            return self.p_left
        if action == 1:
            return self.p_right
        raise Error("TwoArmBandit: action must be 0 or 1")

    def best_action(self) -> Int:
        return 0 if self.p_left >= self.p_right else 1


# ─── Fixed-shape value tree ───────────────────────────────────────────────


@fieldwise_init
struct KnownValueTree(Copyable, Movable):
    """Branching `branching`, depth `depth` tree with leaf values stored in
    row-major order (leftmost leaf first).

    `branching ** depth` must equal len(leaf_values). Provides:
      - max_value: best leaf (ground truth for SinglePlayer + deterministic
        dynamics — full enumeration upper bound on MCTS root value)
      - negamax_value: value at root under zero-sum alternating play
        (ground truth for SelfPlay MCTS)
    """

    var branching: Int
    var depth: Int
    var leaf_values: List[Float64]

    def num_leaves(self) -> Int:
        var n: Int = 1
        for _ in range(self.depth):
            n *= self.branching
        return n

    def max_value(self) raises -> Float64:
        if len(self.leaf_values) == 0:
            raise Error("KnownValueTree: empty leaf_values")
        var best = self.leaf_values[0]
        for i in range(1, len(self.leaf_values)):
            if self.leaf_values[i] > best:
                best = self.leaf_values[i]
        return best

    def negamax_value(self) raises -> Float64:
        """Iterative bottom-up negamax: parent_value = -max(child_values).
        After `depth` collapses the single remaining value is the root."""
        if len(self.leaf_values) != self.num_leaves():
            raise Error("KnownValueTree: leaf_values size mismatch num_leaves")
        var current = self.leaf_values.copy()
        var current_n = self.num_leaves()
        for _level in range(self.depth):
            var next_n = current_n // self.branching
            var next_vals = List[Float64](capacity=next_n)
            for parent in range(next_n):
                var base = parent * self.branching
                var best = current[base]
                for c in range(1, self.branching):
                    if current[base + c] > best:
                        best = current[base + c]
                next_vals.append(-best)
            current = next_vals^
            current_n = next_n
        return current[0]
