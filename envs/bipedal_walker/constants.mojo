struct BipedalWalkerConstants[DTYPE: DType]:
    comptime FPS: Int = 50
    comptime SCALE: Scalar[Self.DTYPE] = 30.0

    comptime MOTORS_TORQUE: Scalar[Self.DTYPE] = 80.0
    comptime SPEED_HIP: Scalar[Self.DTYPE] = 4.0
    comptime SPEED_KNEE: Scalar[Self.DTYPE] = 6.0

    comptime LIDAR_RANGE: Scalar[
        Self.DTYPE
    ] = 160.0 / Self.SCALE  # ~5.33 meters
    comptime NUM_LIDAR: Int = 10

    comptime INITIAL_RANDOM: Scalar[Self.DTYPE] = 5.0

    # Hull polygon vertices (scaled to Box2D units)
    comptime HULL_VERTEX_COUNT: Int = 5

    # Leg dimensions
    comptime LEG_DOWN: Scalar[Self.DTYPE] = -8.0 / Self.SCALE
    comptime LEG_W: Scalar[Self.DTYPE] = 8.0 / Self.SCALE
    comptime LEG_H: Scalar[Self.DTYPE] = 34.0 / Self.SCALE

    # Terrain
    comptime TERRAIN_STEP: Scalar[Self.DTYPE] = 14.0 / Self.SCALE
    comptime TERRAIN_LENGTH: Int = 200
    comptime TERRAIN_HEIGHT: Scalar[
        Self.DTYPE
    ] = 400.0 / Self.SCALE / 4.0  # ~3.33
    comptime TERRAIN_GRASS: Int = 10
    comptime TERRAIN_STARTPAD: Int = 20
    comptime FRICTION: Scalar[Self.DTYPE] = 2.5

    # Viewport
    comptime VIEWPORT_W: Int = 600
    comptime VIEWPORT_H: Int = 400

    # Terrain types for hardcore mode
    comptime TERRAIN_GRASS_TYPE: Int = 0
    comptime TERRAIN_STUMP: Int = 1
    comptime TERRAIN_STAIRS: Int = 2
    comptime TERRAIN_PIT: Int = 3
