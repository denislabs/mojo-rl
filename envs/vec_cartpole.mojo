"""Vectorized CartPole environment using SIMD operations.

Runs multiple CartPole environments in parallel using SIMD vectors for state
storage and vectorized physics calculations. This provides significant speedup
for data collection in algorithms like PPO and A2C.

Physics based on OpenAI Gym / Gymnasium CartPole-v1:
https://gymnasium.farama.org/environments/classic_control/cart_pole/

Performance notes:
- Uses native SIMD methods (.eq(), .lt(), .gt(), etc.) for element-wise comparisons
- Uses @always_inline for zero function call overhead
- Uses @parameter for compile-time loop unrolling
- Pre-computes SIMD constant vectors to avoid runtime splatting

Example:
    ```mojo
    from envs import VecCartPoleEnv

    fn main() raises:
        var env = VecCartPoleEnv[8]()  # 8 parallel environments
        var obs = env.reset_vec()

        # Random actions for all 8 environments
        var actions = SIMD[DType.int32, 8](0, 1, 0, 1, 1, 0, 1, 0)
        var result = env.step_vec(actions)

        print("Rewards:", result.rewards)
        print("Dones:", result.dones)
    ```
"""

from math import cos, sin
from random import random_float64
from core import VecStepResult, random_simd_centered


struct VecCartPoleEnv[num_envs: Int = 8]:
    """Vectorized CartPole environment running num_envs instances in parallel.

    State for each environment: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Actions: 0 (push left), 1 (push right)

    Episode terminates when:
    - Pole angle > +/-12 degrees (+/-0.2095 rad)
    - Cart position > +/-2.4
    - Episode length > 500 steps

    Auto-reset: When an environment is done, it automatically resets.

    Parameters:
        num_envs: Number of parallel environments (default 8, works with AVX).
    """

    # Physical constants (same as Gymnasium)
    var gravity: Float64
    var masscart: Float64
    var masspole: Float64
    var total_mass: Float64
    var length: Float64  # Half the pole's length
    var polemass_length: Float64
    var force_mag: Float64
    var tau: Float64  # Time step (seconds)

    # Thresholds for episode termination
    var theta_threshold_radians: Float64
    var x_threshold: Float64

    # Vectorized state (Structure of Arrays layout)
    var x: SIMD[DType.float64, Self.num_envs]  # Cart positions
    var x_dot: SIMD[DType.float64, Self.num_envs]  # Cart velocities
    var theta: SIMD[DType.float64, Self.num_envs]  # Pole angles
    var theta_dot: SIMD[DType.float64, Self.num_envs]  # Pole angular velocities

    # Episode tracking
    var steps: SIMD[DType.int32, Self.num_envs]
    var max_steps: Int

    # Pre-computed SIMD constants for efficiency
    var pos_force_vec: SIMD[DType.float64, Self.num_envs]
    var neg_force_vec: SIMD[DType.float64, Self.num_envs]
    var one_vec_i32: SIMD[DType.int32, Self.num_envs]
    var one_vec_f64: SIMD[DType.float64, Self.num_envs]
    var zero_vec_f64: SIMD[DType.float64, Self.num_envs]
    var max_steps_vec: SIMD[DType.int32, Self.num_envs]
    var neg_x_threshold_vec: SIMD[DType.float64, Self.num_envs]
    var pos_x_threshold_vec: SIMD[DType.float64, Self.num_envs]
    var neg_theta_threshold_vec: SIMD[DType.float64, Self.num_envs]
    var pos_theta_threshold_vec: SIMD[DType.float64, Self.num_envs]

    fn __init__(out self, max_steps: Int = 500):
        """Initialize VecCartPoleEnv with default physics parameters.

        Args:
            max_steps: Maximum steps per episode before truncation.
        """
        # Physics constants from Gymnasium
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # Half pole length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # 50 Hz updates

        # Termination thresholds
        self.theta_threshold_radians = 12.0 * 3.141592653589793 / 180.0  # 12 degrees
        self.x_threshold = 2.4

        # Initialize state to zeros (will be reset before use)
        self.x = SIMD[DType.float64, Self.num_envs]()
        self.x_dot = SIMD[DType.float64, Self.num_envs]()
        self.theta = SIMD[DType.float64, Self.num_envs]()
        self.theta_dot = SIMD[DType.float64, Self.num_envs]()

        # Episode tracking
        self.steps = SIMD[DType.int32, Self.num_envs]()
        self.max_steps = max_steps

        # Pre-compute SIMD constant vectors (avoid any runtime splatting)
        self.pos_force_vec = SIMD[DType.float64, Self.num_envs](self.force_mag)
        self.neg_force_vec = SIMD[DType.float64, Self.num_envs](-self.force_mag)
        self.one_vec_i32 = SIMD[DType.int32, Self.num_envs](1)
        self.one_vec_f64 = SIMD[DType.float64, Self.num_envs](1.0)
        self.zero_vec_f64 = SIMD[DType.float64, Self.num_envs](0.0)
        self.max_steps_vec = SIMD[DType.int32, Self.num_envs](Int32(max_steps))
        self.neg_x_threshold_vec = SIMD[DType.float64, Self.num_envs](
            -self.x_threshold
        )
        self.pos_x_threshold_vec = SIMD[DType.float64, Self.num_envs](self.x_threshold)
        self.neg_theta_threshold_vec = SIMD[DType.float64, Self.num_envs](
            -self.theta_threshold_radians
        )
        self.pos_theta_threshold_vec = SIMD[DType.float64, Self.num_envs](
            self.theta_threshold_radians
        )

    fn reset_vec(mut self) -> List[SIMD[DType.float64, 4]]:
        """Reset all environments to random initial states.

        Returns:
            List of observations, one per environment.
        """
        # Random initial state in [-0.05, 0.05] for each component
        self.x = random_simd_centered[Self.num_envs](0.05)
        self.x_dot = random_simd_centered[Self.num_envs](0.05)
        self.theta = random_simd_centered[Self.num_envs](0.05)
        self.theta_dot = random_simd_centered[Self.num_envs](0.05)

        self.steps = SIMD[DType.int32, Self.num_envs]()

        return self._get_all_obs()

    fn step_vec(
        mut self, actions: SIMD[DType.int32, Self.num_envs]
    ) -> VecStepResult[Self.num_envs]:
        """Step all environments in parallel.

        Args:
            actions: SIMD vector of actions (0=left, 1=right) for each environment.

        Returns:
            VecStepResult containing observations, rewards, and done flags.
        """
        # Determine force direction using native SIMD .eq() method
        # action == 1 -> +force_mag, action == 0 -> -force_mag
        var action_mask = actions.eq(self.one_vec_i32)
        var force = action_mask.select(self.pos_force_vec, self.neg_force_vec)

        # Physics calculations (vectorized - cos/sin work on SIMD)
        var costheta = cos(self.theta)
        var sintheta = sin(self.theta)

        # Equations of motion (derived from Lagrangian mechanics)
        var temp = (
            force + self.polemass_length * self.theta_dot * self.theta_dot * sintheta
        ) / self.total_mass

        var thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )

        var xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler integration (all SIMD operations)
        self.x = self.x + self.tau * self.x_dot
        self.x_dot = self.x_dot + self.tau * xacc
        self.theta = self.theta + self.tau * self.theta_dot
        self.theta_dot = self.theta_dot + self.tau * thetaacc

        self.steps = self.steps + self.one_vec_i32

        # Check termination conditions using native SIMD comparison methods
        var x_too_low = self.x.lt(self.neg_x_threshold_vec)
        var x_too_high = self.x.gt(self.pos_x_threshold_vec)
        var x_out_of_bounds = x_too_low | x_too_high  # Native bitwise OR

        var theta_too_low = self.theta.lt(self.neg_theta_threshold_vec)
        var theta_too_high = self.theta.gt(self.pos_theta_threshold_vec)
        var theta_out_of_bounds = theta_too_low | theta_too_high

        var terminated = x_out_of_bounds | theta_out_of_bounds
        var truncated = self.steps.ge(self.max_steps_vec)
        var done = terminated | truncated

        # Reward: +1 for every step the pole stays upright (0 if terminated)
        var rewards = terminated.select(self.zero_vec_f64, self.one_vec_f64)

        # Get observations before auto-reset
        var obs = self._get_all_obs()

        # Auto-reset environments that are done
        self._reset_where(done)

        return VecStepResult[Self.num_envs](obs^, rewards, done)

    fn step_vec_raw(
        mut self, actions: SIMD[DType.int32, Self.num_envs]
    ) -> Tuple[SIMD[DType.float64, Self.num_envs], SIMD[DType.bool, Self.num_envs]]:
        """Step all environments without observation collection (for benchmarking).

        This version skips observation list creation, returning only rewards and dones.
        Use for maximum throughput when observations can be accessed directly via
        state fields (self.x, self.x_dot, self.theta, self.theta_dot).

        Args:
            actions: SIMD vector of actions (0=left, 1=right) for each environment.

        Returns:
            Tuple of (rewards, dones) as SIMD vectors.
        """
        # Force direction using native .eq()
        var action_mask = actions.eq(self.one_vec_i32)
        var force = action_mask.select(self.pos_force_vec, self.neg_force_vec)

        # Physics calculations (vectorized)
        var costheta = cos(self.theta)
        var sintheta = sin(self.theta)

        var temp = (
            force + self.polemass_length * self.theta_dot * self.theta_dot * sintheta
        ) / self.total_mass

        var thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )

        var xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler integration
        self.x = self.x + self.tau * self.x_dot
        self.x_dot = self.x_dot + self.tau * xacc
        self.theta = self.theta + self.tau * self.theta_dot
        self.theta_dot = self.theta_dot + self.tau * thetaacc

        self.steps = self.steps + self.one_vec_i32

        # Check termination conditions using native SIMD methods
        var x_out_of_bounds = self.x.lt(self.neg_x_threshold_vec) | self.x.gt(
            self.pos_x_threshold_vec
        )
        var theta_out_of_bounds = self.theta.lt(
            self.neg_theta_threshold_vec
        ) | self.theta.gt(self.pos_theta_threshold_vec)

        var terminated = x_out_of_bounds | theta_out_of_bounds
        var done = terminated | self.steps.ge(self.max_steps_vec)

        var rewards = terminated.select(self.zero_vec_f64, self.one_vec_f64)

        # Auto-reset
        self._reset_where(done)

        return (rewards, done)

    @always_inline
    fn _reset_where(mut self, mask: SIMD[DType.bool, Self.num_envs]):
        """Reset specific environments based on a boolean mask.

        Args:
            mask: Boolean mask indicating which environments to reset.
        """
        # CRITICAL OPTIMIZATION: Only generate random numbers if at least one
        # environment needs reset. This avoids the massive overhead of generating
        # 32 random numbers every step when no reset is needed.
        if not mask.reduce_or():
            return

        # Generate new random initial states
        var new_x = random_simd_centered[Self.num_envs](0.05)
        var new_x_dot = random_simd_centered[Self.num_envs](0.05)
        var new_theta = random_simd_centered[Self.num_envs](0.05)
        var new_theta_dot = random_simd_centered[Self.num_envs](0.05)
        var zero_steps = SIMD[DType.int32, Self.num_envs]()

        # Apply only where mask is True using SIMD.select
        self.x = mask.select(new_x, self.x)
        self.x_dot = mask.select(new_x_dot, self.x_dot)
        self.theta = mask.select(new_theta, self.theta)
        self.theta_dot = mask.select(new_theta_dot, self.theta_dot)
        self.steps = mask.select(zero_steps, self.steps)

    @always_inline
    fn _get_all_obs(self) -> List[SIMD[DType.float64, 4]]:
        """Collect observations from all environments.

        Returns:
            List of 4D observation vectors, one per environment.
        """
        var obs = List[SIMD[DType.float64, 4]](capacity=Self.num_envs)

        @parameter
        for i in range(Self.num_envs):
            var o = SIMD[DType.float64, 4]()
            o[0] = self.x[i]
            o[1] = self.x_dot[i]
            o[2] = self.theta[i]
            o[3] = self.theta_dot[i]
            obs.append(o)
        return obs^

    @always_inline
    fn get_num_envs(self) -> Int:
        """Return the number of parallel environments."""
        return Self.num_envs

    @always_inline
    fn get_obs_dim(self) -> Int:
        """Return the observation dimension (4 for CartPole)."""
        return 4

    @always_inline
    fn get_num_actions(self) -> Int:
        """Return the number of discrete actions (2 for CartPole)."""
        return 2

    @always_inline
    fn get_max_steps(self) -> Int:
        """Return the maximum steps per episode."""
        return self.max_steps


# ============================================================================
# Convenience type aliases
# ============================================================================


comptime VecCartPole8 = VecCartPoleEnv[8]
"""Vectorized CartPole with 8 parallel environments (AVX compatible)."""

comptime VecCartPole16 = VecCartPoleEnv[16]
"""Vectorized CartPole with 16 parallel environments (AVX-512 compatible)."""
