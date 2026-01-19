"""CarRacingLayout - Compile-time state layout for GPU-batched car simulation.

This module provides compile-time computation of buffer sizes and offsets
for CarRacing state stored in a flat [BATCH, STATE_SIZE] layout.

CarRacing state differs from impulse-based physics:
- No separate rigid bodies - hull position is the only dynamic state
- Wheel positions are derived from hull position + angle
- Controls (steering, gas, brake) are stored in state for GPU access
- Track tiles are stored separately (shared across all envs)

Layout per environment:
    [observation | hull_state | wheel_states | controls | metadata]

Example:
    ```mojo
    from physics_gpu.car.layout import CarRacingLayout

    comptime Layout = CarRacingLayout[OBS_DIM=13, METADATA_SIZE=5]

    # Access computed sizes at compile time
    comptime state_size = Layout.STATE_SIZE
    comptime hull_offset = Layout.HULL_OFFSET
    ```
"""

from .constants import (
    HULL_STATE_SIZE,
    WHEEL_STATE_SIZE,
    NUM_WHEELS,
    CONTROL_SIZE,
)


struct CarRacingLayout[
    OBS_DIM: Int = 13,
    METADATA_SIZE: Int = 5,
]:
    """Compile-time layout calculator for CarRacing state buffers.

    Parameters:
        OBS_DIM: Observation dimension (13 for state-based CarRacing).
            [x, y, angle, vx, vy, omega, wheel_angles[4], wheel_omegas[3]]
        METADATA_SIZE: Size of metadata (step_count, total_reward, done, etc.).

    Layout per environment (39 floats total with defaults):
        - Observation: [0, OBS_DIM) = 13 floats
        - Hull state: [OBS_DIM, OBS_DIM + 6) = 6 floats [x, y, angle, vx, vy, omega]
        - Wheel states: [next, next + 12) = 12 floats (4 wheels x 3 floats each)
        - Controls: [next, next + 3) = 3 floats [steering, gas, brake]
        - Metadata: [next, next + 5) = 5 floats
    """

    # =========================================================================
    # Component Sizes
    # =========================================================================

    # Hull state: [x, y, angle, vx, vy, omega]
    comptime HULL_SIZE: Int = HULL_STATE_SIZE  # 6

    # Wheel states: 4 wheels x [omega, joint_angle, phase]
    comptime WHEELS_SIZE: Int = NUM_WHEELS * WHEEL_STATE_SIZE  # 4 * 3 = 12

    # Controls: [steering, gas, brake]
    comptime CONTROLS_SIZE: Int = CONTROL_SIZE  # 3

    # =========================================================================
    # Offsets within Environment State (cumulative)
    # =========================================================================

    # Observation at the start
    comptime OBS_OFFSET: Int = 0

    # Hull state follows observation
    comptime HULL_OFFSET: Int = Self.OBS_DIM

    # Wheel states follow hull
    comptime WHEELS_OFFSET: Int = Self.HULL_OFFSET + Self.HULL_SIZE

    # Controls follow wheel states
    comptime CONTROLS_OFFSET: Int = Self.WHEELS_OFFSET + Self.WHEELS_SIZE

    # Metadata at the end
    comptime METADATA_OFFSET: Int = Self.CONTROLS_OFFSET + Self.CONTROLS_SIZE

    # =========================================================================
    # Total State Size
    # =========================================================================

    # Total size per environment
    # Default: 13 + 6 + 12 + 3 + 5 = 39 floats
    comptime STATE_SIZE: Int = Self.METADATA_OFFSET + Self.METADATA_SIZE

    # =========================================================================
    # Utility Methods - Hull Access
    # =========================================================================

    @staticmethod
    @always_inline
    fn hull_field_offset(field: Int) -> Int:
        """Compute offset for a hull field.

        Args:
            field: Field index (HULL_X, HULL_Y, etc. from constants).

        Returns:
            Offset to the hull field.
        """
        return Self.HULL_OFFSET + field

    # =========================================================================
    # Utility Methods - Wheel Access
    # =========================================================================

    @staticmethod
    @always_inline
    fn wheel_offset(wheel: Int) -> Int:
        """Compute offset for a wheel's state.

        Args:
            wheel: Wheel index (0=FL, 1=FR, 2=RL, 3=RR).

        Returns:
            Offset to wheel's first field (omega).
        """
        return Self.WHEELS_OFFSET + wheel * WHEEL_STATE_SIZE

    @staticmethod
    @always_inline
    fn wheel_field_offset(wheel: Int, field: Int) -> Int:
        """Compute offset for a specific wheel field.

        Args:
            wheel: Wheel index.
            field: Field index (WHEEL_OMEGA, WHEEL_JOINT_ANGLE, WHEEL_PHASE).

        Returns:
            Offset to the specific field.
        """
        return Self.WHEELS_OFFSET + wheel * WHEEL_STATE_SIZE + field

    # =========================================================================
    # Utility Methods - Control Access
    # =========================================================================

    @staticmethod
    @always_inline
    fn control_offset(ctrl: Int) -> Int:
        """Compute offset for a control input.

        Args:
            ctrl: Control index (CTRL_STEERING, CTRL_GAS, CTRL_BRAKE).

        Returns:
            Offset to the control value.
        """
        return Self.CONTROLS_OFFSET + ctrl

    # =========================================================================
    # Utility Methods - Metadata Access
    # =========================================================================

    @staticmethod
    @always_inline
    fn metadata_offset(field: Int) -> Int:
        """Compute offset for a metadata field.

        Args:
            field: Metadata field index.

        Returns:
            Offset to the metadata field.
        """
        return Self.METADATA_OFFSET + field


# =============================================================================
# Metadata Field Indices
# =============================================================================

comptime META_STEP_COUNT: Int = 0
comptime META_TOTAL_REWARD: Int = 1
comptime META_DONE: Int = 2
comptime META_TRUNCATED: Int = 3
comptime META_TILES_VISITED: Int = 4
