"""CarRacingLayout - Compile-time state layout for GPU-batched car simulation.

This module provides compile-time computation of buffer sizes and offsets
for CarRacing state stored in a flat [BATCH, STATE_SIZE] layout.

CarRacing state differs from impulse-based physics:
- No separate rigid bodies - hull position is the only dynamic state
- Wheel positions are derived from hull position + angle
- Controls (steering, gas, brake) are stored in state for GPU access
- Track tiles and visited flags are embedded per-environment

Layout per environment (3040 floats):
    [observation | hull_state | wheel_states | controls | metadata | track_tiles | visited_flags]

    Offset    Size     Description
    ------    ----     -----------
    0         13       OBS (observation vector)
    13        6        HULL_STATE (x, y, angle, vx, vy, omega)
    19        12       WHEELS (4 wheels × 3 fields each)
    31        3        CONTROLS (steering, gas, brake)
    34        6        METADATA (step_count, total_reward, done, truncated, tiles_visited, num_tiles)
    40        2700     TRACK_TILES (300 tiles × 9 floats each)
    2740      300      VISITED_FLAGS (300 boolean flags as floats)
    ------    ----
    Total:    3040     floats per environment

Example:
    ```mojo
    from physics_gpu.car.layout import CarRacingLayout

    comptime Layout = CarRacingLayout[OBS_DIM=13, METADATA_SIZE=6]

    # Access computed sizes at compile time
    comptime state_size = Layout.STATE_SIZE
    comptime hull_offset = Layout.HULL_OFFSET
    comptime track_offset = Layout.TRACK_OFFSET
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
    METADATA_SIZE: Int = 6,
    MAX_TILES: Int = 300,
    TILE_DATA_SIZE: Int = 9,
]:
    """Compile-time layout calculator for CarRacing state buffers.

    Parameters:
        OBS_DIM: Observation dimension (13 for state-based CarRacing).
            [x, y, angle, vx, vy, omega, wheel_angles[4], wheel_omegas[3]]
        METADATA_SIZE: Size of metadata (step_count, total_reward, done, truncated, tiles_visited, num_tiles).
        MAX_TILES: Maximum track tiles per environment.
        TILE_DATA_SIZE: Size of each tile (9 floats: 4 vertices × 2 coords + friction).

    Layout per environment (3040 floats total with defaults):
        - Observation: [0, OBS_DIM) = 13 floats
        - Hull state: [OBS_DIM, OBS_DIM + 6) = 6 floats [x, y, angle, vx, vy, omega]
        - Wheel states: [next, next + 12) = 12 floats (4 wheels x 3 floats each)
        - Controls: [next, next + 3) = 3 floats [steering, gas, brake]
        - Metadata: [next, next + 6) = 6 floats
        - Track tiles: [next, next + 2700) = 300 tiles × 9 floats
        - Visited flags: [next, next + 300) = 300 boolean flags
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

    # Track tiles: MAX_TILES tiles × TILE_DATA_SIZE floats each
    comptime TRACK_SIZE: Int = Self.MAX_TILES * Self.TILE_DATA_SIZE  # 300 * 9 = 2700

    # Visited flags: one per tile
    comptime VISITED_SIZE: Int = Self.MAX_TILES  # 300

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

    # Metadata follows controls
    comptime METADATA_OFFSET: Int = Self.CONTROLS_OFFSET + Self.CONTROLS_SIZE

    # Track tiles follow metadata (embedded per-environment)
    comptime TRACK_OFFSET: Int = Self.METADATA_OFFSET + Self.METADATA_SIZE

    # Visited flags follow track tiles
    comptime VISITED_OFFSET: Int = Self.TRACK_OFFSET + Self.TRACK_SIZE

    # =========================================================================
    # Total State Size
    # =========================================================================

    # Total size per environment
    # Default: 13 + 6 + 12 + 3 + 6 + 2700 + 300 = 3040 floats
    comptime STATE_SIZE: Int = Self.VISITED_OFFSET + Self.VISITED_SIZE

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

    # =========================================================================
    # Utility Methods - Track Tile Access
    # =========================================================================

    @staticmethod
    @always_inline
    fn track_tile_offset(tile_idx: Int) -> Int:
        """Compute offset for a track tile's data.

        Args:
            tile_idx: Tile index (0 to MAX_TILES-1).

        Returns:
            Offset to the tile's first field (v0x).
        """
        return Self.TRACK_OFFSET + tile_idx * Self.TILE_DATA_SIZE

    @staticmethod
    @always_inline
    fn track_tile_field_offset(tile_idx: Int, field: Int) -> Int:
        """Compute offset for a specific field of a track tile.

        Args:
            tile_idx: Tile index.
            field: Field index within tile (0-8: v0x, v0y, v1x, v1y, v2x, v2y, v3x, v3y, friction).

        Returns:
            Offset to the specific field.
        """
        return Self.TRACK_OFFSET + tile_idx * Self.TILE_DATA_SIZE + field

    # =========================================================================
    # Utility Methods - Visited Flags Access
    # =========================================================================

    @staticmethod
    @always_inline
    fn visited_flag_offset(tile_idx: Int) -> Int:
        """Compute offset for a visited flag.

        Args:
            tile_idx: Tile index.

        Returns:
            Offset to the visited flag for this tile.
        """
        return Self.VISITED_OFFSET + tile_idx


# =============================================================================
# Metadata Field Indices
# =============================================================================

comptime META_STEP_COUNT: Int = 0
comptime META_TOTAL_REWARD: Int = 1
comptime META_DONE: Int = 2
comptime META_TRUNCATED: Int = 3
comptime META_TILES_VISITED: Int = 4
comptime META_NUM_TILES: Int = 5
