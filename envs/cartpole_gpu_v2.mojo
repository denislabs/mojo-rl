"""GPU CartPole Environment v2 - Module-level functions for kernel inlining.

This design provides:
1. Compile-time constants as module-level aliases
2. Inline functions that get inlined into kernels
3. No trait-based polymorphism (avoids compile-time explosion)

Usage: Import this module and use the constants/functions directly in kernels.
"""

from math import cos, sin

from deep_rl.gpu import gpu_random_range, xorshift32


# =============================================================================
# Compile-time Constants
# =============================================================================

comptime CARTPOLE_OBS_DIM: Int = 4
comptime CARTPOLE_NUM_ACTIONS: Int = 2
comptime CARTPOLE_STATE_SIZE: Int = 4

# Physics constants
comptime GRAVITY: Float64 = 9.8
comptime CART_MASS: Float64 = 1.0
comptime POLE_MASS: Float64 = 0.1
comptime TOTAL_MASS: Float64 = CART_MASS + POLE_MASS
comptime POLE_HALF_LENGTH: Float64 = 0.5
comptime POLE_MASS_LENGTH: Float64 = POLE_MASS * POLE_HALF_LENGTH
comptime FORCE_MAG: Float64 = 10.0
comptime TAU: Float64 = 0.02

comptime X_THRESHOLD: Float64 = 2.4
comptime THETA_THRESHOLD: Float64 = 0.2095
comptime INIT_RANGE: Float64 = 0.05


# =============================================================================
# Inline Environment Functions
# =============================================================================


@always_inline
fn cartpole_step[
    dtype: DType
](
    mut state: InlineArray[Scalar[dtype], CARTPOLE_STATE_SIZE],
    action: Int,
    rng: Scalar[DType.uint32],
) -> Tuple[Scalar[dtype], Bool, Scalar[DType.uint32]]:
    """Execute one CartPole physics step."""
    var x = state[0]
    var x_dot = state[1]
    var theta = state[2]
    var theta_dot = state[3]

    var force_mag = Scalar[dtype](FORCE_MAG)
    var gravity = Scalar[dtype](GRAVITY)
    var total_mass = Scalar[dtype](TOTAL_MASS)
    var pole_half_length = Scalar[dtype](POLE_HALF_LENGTH)
    var pole_mass_length = Scalar[dtype](POLE_MASS_LENGTH)
    var pole_mass = Scalar[dtype](POLE_MASS)
    var tau = Scalar[dtype](TAU)
    var x_threshold = Scalar[dtype](X_THRESHOLD)
    var theta_threshold = Scalar[dtype](THETA_THRESHOLD)

    var force = force_mag if action == 1 else -force_mag

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

    x = x + tau * x_dot
    x_dot = x_dot + tau * x_acc
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * theta_acc

    state[0] = x
    state[1] = x_dot
    state[2] = theta
    state[3] = theta_dot

    var done = (
        (x < -x_threshold)
        or (x > x_threshold)
        or (theta < -theta_threshold)
        or (theta > theta_threshold)
    )

    var reward = Scalar[dtype](1.0) if not done else Scalar[dtype](0.0)

    return (reward, done, rng)


@always_inline
fn cartpole_reset[
    dtype: DType
](
    mut state: InlineArray[Scalar[dtype], CARTPOLE_STATE_SIZE],
    rng: Scalar[DType.uint32],
) -> Scalar[DType.uint32]:
    """Reset CartPole to initial state."""
    var low = Scalar[dtype](-INIT_RANGE)
    var high = Scalar[dtype](INIT_RANGE)

    var current_rng = rng

    var r0 = gpu_random_range[dtype](current_rng, low, high)
    state[0] = r0[0]
    current_rng = r0[1]

    var r1 = gpu_random_range[dtype](current_rng, low, high)
    state[1] = r1[0]
    current_rng = r1[1]

    var r2 = gpu_random_range[dtype](current_rng, low, high)
    state[2] = r2[0]
    current_rng = r2[1]

    var r3 = gpu_random_range[dtype](current_rng, low, high)
    state[3] = r3[0]
    current_rng = r3[1]

    return current_rng


@always_inline
fn cartpole_get_obs[
    dtype: DType
](
    state: InlineArray[Scalar[dtype], CARTPOLE_STATE_SIZE],
) -> InlineArray[
    Scalar[dtype], CARTPOLE_OBS_DIM
]:
    """Get observation from state (identity for CartPole)."""
    var obs = InlineArray[Scalar[dtype], CARTPOLE_OBS_DIM](
        fill=Scalar[dtype](0)
    )
    obs[0] = state[0]
    obs[1] = state[1]
    obs[2] = state[2]
    obs[3] = state[3]
    return obs
