"""GPU CartPole Environment.

Implements the GPUEnv trait for CartPole-v1 physics.
All methods are static and GPU-safe (no heap allocation).

State: [x, x_dot, theta, theta_dot]
Obs: Same as state (4D)
Actions: 0 (left), 1 (right)

Physics matches Gymnasium CartPole-v1.
"""

from math import cos, sin

from deep_rl.gpu import gpu_random_range, xorshift32
from core import GPUEnvDims

# =============================================================================
# Physics Constants
# =============================================================================

# These are module-level constants that get inlined at compile time
comptime GRAVITY: Float64 = 9.8
comptime CART_MASS: Float64 = 1.0
comptime POLE_MASS: Float64 = 0.1
comptime TOTAL_MASS: Float64 = CART_MASS + POLE_MASS
comptime POLE_HALF_LENGTH: Float64 = 0.5
comptime POLE_MASS_LENGTH: Float64 = POLE_MASS * POLE_HALF_LENGTH
comptime FORCE_MAG: Float64 = 10.0
comptime TAU: Float64 = 0.02  # Time step

# Termination thresholds
comptime X_THRESHOLD: Float64 = 2.4
comptime THETA_THRESHOLD: Float64 = 0.2095  # ~12 degrees

# Initial state randomization range
comptime INIT_RANGE: Float64 = 0.05


# =============================================================================
# GPUCartPole
# =============================================================================


struct GPUCartPole(GPUEnvDims):
    """GPU-compatible CartPole environment implementing GPUEnv trait.

    State representation: [x, x_dot, theta, theta_dot]
    - x: Cart position
    - x_dot: Cart velocity
    - theta: Pole angle (radians, 0 = upright)
    - theta_dot: Pole angular velocity

    Actions:
    - 0: Push cart left (-FORCE_MAG)
    - 1: Push cart right (+FORCE_MAG)

    Reward: +1 for each step the pole stays upright
    Done: When pole angle > 12° or cart position > 2.4
    """

    # Use alias inside struct (required for trait conformance)
    comptime OBS_DIM = 4
    comptime NUM_ACTIONS = 2
    comptime STATE_SIZE = 4

    @staticmethod
    fn step[
        dtype: DType
    ](
        mut state: InlineArray[Scalar[dtype], Self.STATE_SIZE],
        action: Int,
        rng: Scalar[DType.uint32],
    ) -> Tuple[Scalar[dtype], Bool, Scalar[DType.uint32]]:
        """Execute one CartPole physics step.

        Uses Euler integration matching Gymnasium CartPole-v1.

        Returns: (reward, done, rng) - rng is unchanged as step doesn't need randomness
        """
        # Unpack state
        var x = state[0]
        var x_dot = state[1]
        var theta = state[2]
        var theta_dot = state[3]

        # Cast physics constants to dtype
        var force_mag = Scalar[dtype](FORCE_MAG)
        var gravity = Scalar[dtype](GRAVITY)
        var pole_mass = Scalar[dtype](POLE_MASS)
        var total_mass = Scalar[dtype](TOTAL_MASS)
        var pole_half_length = Scalar[dtype](POLE_HALF_LENGTH)
        var pole_mass_length = Scalar[dtype](POLE_MASS_LENGTH)
        var tau = Scalar[dtype](TAU)
        var x_threshold = Scalar[dtype](X_THRESHOLD)
        var theta_threshold = Scalar[dtype](THETA_THRESHOLD)

        # Compute force
        var force = force_mag if action == 1 else -force_mag

        # Physics calculations
        var cos_theta = cos(theta)
        var sin_theta = sin(theta)

        var temp = (
            force + pole_mass_length * theta_dot * theta_dot * sin_theta
        ) / total_mass
        var theta_acc = (gravity * sin_theta - cos_theta * temp) / (
            pole_half_length
            * (
                Scalar[dtype](4.0 / 3.0)
                - pole_mass * cos_theta * cos_theta / total_mass
            )
        )
        var x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass

        # Euler integration
        x = x + tau * x_dot
        x_dot = x_dot + tau * x_acc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * theta_acc

        # Update state
        state[0] = x
        state[1] = x_dot
        state[2] = theta
        state[3] = theta_dot

        # Check termination
        var done = (
            (x < -x_threshold)
            or (x > x_threshold)
            or (theta < -theta_threshold)
            or (theta > theta_threshold)
        )

        # Reward: +1 for staying alive
        var reward = Scalar[dtype](1.0) if not done else Scalar[dtype](0.0)

        return (reward, done, rng)

    @staticmethod
    fn reset[
        dtype: DType
    ](
        mut state: InlineArray[Scalar[dtype], Self.STATE_SIZE],
        rng: Scalar[DType.uint32],
    ) -> Scalar[DType.uint32]:
        """Reset CartPole to initial state with small random perturbation.

        Returns: new RNG state
        """
        var low = Scalar[dtype](-INIT_RANGE)
        var high = Scalar[dtype](INIT_RANGE)

        var current_rng = rng

        var r0 = gpu_random_range[dtype](current_rng, low, high)
        state[0] = r0[0]  # x
        current_rng = r0[1]

        var r1 = gpu_random_range[dtype](current_rng, low, high)
        state[1] = r1[0]  # x_dot
        current_rng = r1[1]

        var r2 = gpu_random_range[dtype](current_rng, low, high)
        state[2] = r2[0]  # theta
        current_rng = r2[1]

        var r3 = gpu_random_range[dtype](current_rng, low, high)
        state[3] = r3[0]  # theta_dot
        current_rng = r3[1]

        return current_rng

    @staticmethod
    fn get_obs[
        dtype: DType
    ](
        state: InlineArray[Scalar[dtype], Self.STATE_SIZE],
    ) -> InlineArray[
        Scalar[dtype], Self.STATE_SIZE
    ]:
        """Get observation from state (identity for CartPole)."""
        var obs = InlineArray[Scalar[dtype], 4](fill=Scalar[dtype](0))
        obs[0] = state[0]
        obs[1] = state[1]
        obs[2] = state[2]
        obs[3] = state[3]
        return obs
