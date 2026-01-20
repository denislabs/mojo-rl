"""Pendulum V2 environment constants.

Physics constants matched to Gymnasium Pendulum-v1:
https://gymnasium.farama.org/environments/classic_control/pendulum/

State layout for GPU-batched simulation with flat [BATCH, STATE_SIZE] buffers.
"""

from math import pi


struct PendulumLayout:
    """Compile-time layout calculator for Pendulum state buffers.

    Layout per environment (8 floats total):
        - Observation: [cos(theta), sin(theta), theta_dot] = 3 floats (at offset 0!)
        - Physics: [theta] = 1 float (theta_dot is in observation)
        - Metadata: [step_count, done, total_reward, last_torque] = 4 floats

    IMPORTANT: Observations must be stored at offset 0 for compatibility with
    the generic _extract_obs_from_state_continuous_kernel used by PPO agents.
    """

    # =========================================================================
    # Component Sizes
    # =========================================================================

    comptime OBS_SIZE: Int = 3  # cos(theta), sin(theta), theta_dot
    comptime PHYSICS_SIZE: Int = 1  # theta only (theta_dot is in obs)
    comptime METADATA_SIZE: Int = 4  # step, done, total_reward, last_torque

    # =========================================================================
    # Offsets within Environment State
    # =========================================================================

    # Observation at the start (REQUIRED for GPU training!)
    comptime OBS_OFFSET: Int = 0

    # Observation field indices (relative to OBS_OFFSET)
    comptime OBS_COS_THETA: Int = 0
    comptime OBS_SIN_THETA: Int = 1
    comptime OBS_THETA_DOT: Int = 2

    # Physics state follows observation
    comptime PHYSICS_OFFSET: Int = Self.OBS_SIZE

    # Physics field indices (relative to PHYSICS_OFFSET)
    # Note: theta_dot is stored in observation, not here
    comptime THETA: Int = 0  # Raw angle for physics computations

    # Absolute offset for theta
    comptime THETA_ABS: Int = Self.PHYSICS_OFFSET + Self.THETA  # = 3

    # Metadata follows physics
    comptime METADATA_OFFSET: Int = Self.OBS_SIZE + Self.PHYSICS_SIZE

    # Metadata field indices (relative to METADATA_OFFSET)
    comptime META_STEP_COUNT: Int = 0
    comptime META_DONE: Int = 1
    comptime META_TOTAL_REWARD: Int = 2
    comptime META_LAST_TORQUE: Int = 3

    # =========================================================================
    # Total Sizes
    # =========================================================================

    comptime STATE_SIZE: Int = Self.OBS_SIZE + Self.PHYSICS_SIZE + Self.METADATA_SIZE  # 8
    comptime OBS_DIM: Int = 3  # cos(theta), sin(theta), theta_dot
    comptime ACTION_DIM: Int = 1  # torque

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    @always_inline
    fn physics_offset(field: Int) -> Int:
        """Get offset for a physics field."""
        return Self.PHYSICS_OFFSET + field

    @staticmethod
    @always_inline
    fn metadata_offset(field: Int) -> Int:
        """Get offset for a metadata field."""
        return Self.METADATA_OFFSET + field


struct PConstants:
    """Pendulum physics and environment constants.

    Physics matched to Gymnasium Pendulum-v1.
    """

    # =========================================================================
    # Physics Constants (from Gymnasium)
    # =========================================================================

    comptime MAX_SPEED: Float64 = 8.0  # Maximum angular velocity (rad/s)
    comptime MAX_TORQUE: Float64 = 2.0  # Maximum torque
    comptime DT: Float64 = 0.05  # Time step (20 Hz)
    comptime G: Float64 = 10.0  # Gravity (m/s^2)
    comptime M: Float64 = 1.0  # Mass (kg)
    comptime L: Float64 = 1.0  # Length (m)

    # =========================================================================
    # Episode Constants
    # =========================================================================

    comptime MAX_STEPS: Int = 200  # Episode length (fixed, no early termination)

    # =========================================================================
    # State Layout (from PendulumLayout)
    # =========================================================================

    comptime Layout = PendulumLayout

    comptime STATE_SIZE: Int = Self.Layout.STATE_SIZE  # 8
    comptime OBS_DIM: Int = Self.Layout.OBS_DIM  # 3
    comptime ACTION_DIM: Int = Self.Layout.ACTION_DIM  # 1

    # Offsets
    comptime OBS_OFFSET: Int = Self.Layout.OBS_OFFSET  # 0
    comptime PHYSICS_OFFSET: Int = Self.Layout.PHYSICS_OFFSET  # 3
    comptime METADATA_OFFSET: Int = Self.Layout.METADATA_OFFSET  # 4

    # Observation field indices (at offset 0)
    comptime OBS_COS_THETA: Int = Self.Layout.OBS_COS_THETA  # 0
    comptime OBS_SIN_THETA: Int = Self.Layout.OBS_SIN_THETA  # 1
    comptime OBS_THETA_DOT: Int = Self.Layout.OBS_THETA_DOT  # 2

    # Physics field indices (theta at absolute offset 3)
    comptime THETA_ABS: Int = Self.Layout.THETA_ABS  # 3

    # Metadata field indices (relative to METADATA_OFFSET)
    comptime META_STEP_COUNT: Int = Self.Layout.META_STEP_COUNT
    comptime META_DONE: Int = Self.Layout.META_DONE
    comptime META_TOTAL_REWARD: Int = Self.Layout.META_TOTAL_REWARD
    comptime META_LAST_TORQUE: Int = Self.Layout.META_LAST_TORQUE

    # =========================================================================
    # Rendering Constants
    # =========================================================================

    comptime WINDOW_W: Int = 500
    comptime WINDOW_H: Int = 500
    comptime FPS: Int = 20  # Matches 1/DT

    # =========================================================================
    # Reward Constants
    # =========================================================================

    # Reward = -(theta^2 + 0.1*theta_dot^2 + 0.001*torque^2)
    comptime THETA_COST: Float64 = 1.0
    comptime THETA_DOT_COST: Float64 = 0.1
    comptime TORQUE_COST: Float64 = 0.001
