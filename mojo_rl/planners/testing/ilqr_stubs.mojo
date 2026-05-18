"""Stub callbacks for isolated iLQR planner tests.

Two callbacks here, both implementing ``RolloutJacobianCallbackCPU``:

* ``LinearQuadratic1DILQRCallback`` — wraps the existing scalar
  ``LinearQuadratic1D`` stub. Analytic Jacobians are trivial
  (constants); the analytic LQR finite-horizon Riccati lives here too
  as ``finite_horizon_first_gain(T)`` so the oracle test can compare.

* ``Pendulum2DILQRCallback`` — simple 2-D pendulum with semi-implicit
  Euler dynamics + quadratic stabilizing cost (no swing-up — start
  near upright, stabilize). Hand-coded Jacobians of the dynamics +
  diagonal cost Hessian. Nonlinear in dynamics (``cos(θ)`` in ``A``)
  so iLQR has to iterate and the line search / LM logic gets
  exercised.

These live in the testing namespace so production code never depends
on them.
"""

from std.math import sin, cos

from mojo_rl.planners.trajectory import RolloutJacobianCallbackCPU


# =============================================================================
# LinearQuadratic1DILQRCallback
# =============================================================================


@fieldwise_init
struct LinearQuadratic1DILQRCallback(
    Copyable, Movable, ImplicitlyDestructible, RolloutJacobianCallbackCPU
):
    """1-D LQ ``RolloutJacobianCallbackCPU`` adapter.

    Cost ``l(z, u) = Q·z² + R·u²``; terminal ``Φ(z) = Q_T·z²``. Both
    are minimized (iLQR-cost convention) — the planner does not see
    rewards. Note the factor-of-2 in Hessians: ``l_zz = 2Q``,
    ``l_uu = 2R``, ``∂Φ/∂z² = 2Q_T``.
    """

    comptime LATENT_DIM = 1
    comptime ACTION_DIM = 1

    var A: Float64
    var B: Float64
    var Q: Float64
    var R: Float64
    var Q_T: Float64

    def step_cpu(
        mut self,
        z: List[Float64],
        u: List[Float64],
        mut z_next_out: List[Float64],
    ) raises -> Float64:
        z_next_out[0] = self.A * z[0] + self.B * u[0]
        return self.Q * z[0] * z[0] + self.R * u[0] * u[0]

    def linearize_cpu(
        mut self,
        z: List[Float64],
        u: List[Float64],
        mut A_out: List[Float64],
        mut B_out: List[Float64],
        mut l_z_out: List[Float64],
        mut l_u_out: List[Float64],
        mut l_zz_out: List[Float64],
        mut l_uu_out: List[Float64],
        mut l_zu_out: List[Float64],
    ) raises:
        A_out[0] = self.A
        B_out[0] = self.B
        l_z_out[0] = 2.0 * self.Q * z[0]
        l_u_out[0] = 2.0 * self.R * u[0]
        l_zz_out[0] = 2.0 * self.Q
        l_uu_out[0] = 2.0 * self.R
        l_zu_out[0] = 0.0

    def terminal_cpu(
        mut self,
        z: List[Float64],
        mut V_z_out: List[Float64],
        mut V_zz_out: List[Float64],
    ) raises -> Float64:
        V_z_out[0] = 2.0 * self.Q_T * z[0]
        V_zz_out[0] = 2.0 * self.Q_T
        return self.Q_T * z[0] * z[0]

    def finite_horizon_first_gain(self, T: Int) -> Float64:
        """Return ``K_0`` such that LQR's optimal first action is
        ``u_0 = -K_0 · z_0``. Backward Riccati from ``P[T] = Q_T`` to
        ``P[1]``, then ``K_0 = (B·P[1]·A) / (R + B²·P[1])``.
        """
        var P = self.Q_T
        for _ in range(T - 1):
            var denom = self.R + self.B * self.B * P
            P = (
                self.Q
                + self.A * self.A * P
                - (self.A * self.B * P) * (self.A * self.B * P) / denom
            )
        var denom = self.R + self.B * self.B * P
        return (self.B * P * self.A) / denom


# =============================================================================
# Pendulum2DILQRCallback
# =============================================================================


@fieldwise_init
struct Pendulum2DILQRCallback(
    Copyable, Movable, ImplicitlyDestructible, RolloutJacobianCallbackCPU
):
    """2-D pendulum stabilizer with semi-implicit Euler dynamics.

    State ``z = [θ, θ̇]``; action ``u = [τ]``. Stabilize around
    ``θ = 0`` (already upright in this convention).

    Discrete dynamics::

        θ̇' = θ̇ + dt · (-g/L · sin(θ) + τ / (m·L²))
        θ'  = θ + dt · θ̇'

    Quadratic stabilizing cost::

        l(z, u)    = (θ - θ_target)² + w_v·θ̇² + w_u·τ²
        Φ_term(z)  = w_θ_term·(θ - θ_target)² + w_v_term·θ̇²

    Nonlinear via ``sin(θ)`` / ``cos(θ)`` in the Jacobian — iLQR has
    to iterate from initial ``U = 0``. Diagonal cost Hessian so
    ``l_zu = 0``.
    """

    comptime LATENT_DIM = 2
    comptime ACTION_DIM = 1

    var dt: Float64
    var g: Float64
    var L: Float64
    var m: Float64
    var w_v: Float64
    var w_u: Float64
    var w_th_term: Float64
    var w_v_term: Float64
    var theta_target: Float64

    def step_cpu(
        mut self,
        z: List[Float64],
        u: List[Float64],
        mut z_next_out: List[Float64],
    ) raises -> Float64:
        var theta = z[0]
        var thetad = z[1]
        var tau = u[0]
        var thetad_new = thetad + self.dt * (
            -self.g / self.L * sin(theta) + tau / (self.m * self.L * self.L)
        )
        var theta_new = theta + self.dt * thetad_new
        z_next_out[0] = theta_new
        z_next_out[1] = thetad_new
        var dth = theta - self.theta_target
        return dth * dth + self.w_v * thetad * thetad + self.w_u * tau * tau

    def linearize_cpu(
        mut self,
        z: List[Float64],
        u: List[Float64],
        mut A_out: List[Float64],
        mut B_out: List[Float64],
        mut l_z_out: List[Float64],
        mut l_u_out: List[Float64],
        mut l_zz_out: List[Float64],
        mut l_uu_out: List[Float64],
        mut l_zu_out: List[Float64],
    ) raises:
        var theta = z[0]
        var thetad = z[1]
        var tau = u[0]
        var inv_mL2 = 1.0 / (self.m * self.L * self.L)
        var gL = self.g / self.L
        var ddot_dtheta = -gL * cos(theta)
        # Row-major (2,2) — A_out[i*2 + j] = A[i, j]
        # A[0,0] = ∂θ'/∂θ = 1 + dt * (dt * ddot_dtheta)
        A_out[0] = 1.0 + self.dt * self.dt * ddot_dtheta
        A_out[1] = self.dt
        A_out[2] = self.dt * ddot_dtheta
        A_out[3] = 1.0
        # B (2,1)
        B_out[0] = self.dt * self.dt * inv_mL2
        B_out[1] = self.dt * inv_mL2
        # Cost gradient
        l_z_out[0] = 2.0 * (theta - self.theta_target)
        l_z_out[1] = 2.0 * self.w_v * thetad
        l_u_out[0] = 2.0 * self.w_u * tau
        # Cost Hessian (diagonal)
        l_zz_out[0] = 2.0
        l_zz_out[1] = 0.0
        l_zz_out[2] = 0.0
        l_zz_out[3] = 2.0 * self.w_v
        l_uu_out[0] = 2.0 * self.w_u
        l_zu_out[0] = 0.0
        l_zu_out[1] = 0.0

    def terminal_cpu(
        mut self,
        z: List[Float64],
        mut V_z_out: List[Float64],
        mut V_zz_out: List[Float64],
    ) raises -> Float64:
        var dth = z[0] - self.theta_target
        var thetad = z[1]
        V_z_out[0] = 2.0 * self.w_th_term * dth
        V_z_out[1] = 2.0 * self.w_v_term * thetad
        V_zz_out[0] = 2.0 * self.w_th_term
        V_zz_out[1] = 0.0
        V_zz_out[2] = 0.0
        V_zz_out[3] = 2.0 * self.w_v_term
        return (
            self.w_th_term * dth * dth + self.w_v_term * thetad * thetad
        )
