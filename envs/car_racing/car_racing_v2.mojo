"""CarRacing V2 GPU environment using the physics_gpu/car/ module.

This implementation uses the modular car physics components:
- CarRacingLayout for compile-time layout computation
- CarDynamics for slip-based tire friction physics
- CarPhysicsKernel for fused GPU physics step
- TileCollision for friction zone lookup

The flat state layout is compatible with GPU batch operations.
All physics data is packed per-environment for efficient GPU access.

Reference: gymnasium/envs/box2d/car_racing.py
"""

from math import sqrt, cos, sin, pi
from layout import Layout, LayoutTensor
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer

from core import BoxContinuousActionEnv, Action
from render import (
    RendererBase,
    RotatingCamera,
    Transform2D,
    SDL_Color,
    SDL_Point,
    Vec2 as RenderVec2,
    car_red,
    black,
    white,
    rgb,
)

from .state import CarRacingV2State
from .action import CarRacingV2Action
from .constants import CRConstants
from .track import TrackTileV2, TrackGenerator

from physics_gpu import dtype, TPB
from physics_gpu.car import (
    CarRacingLayout,
    CarDynamics,
    CarPhysicsKernel,
    TileCollision,
    WheelFriction,
)
from physics_gpu.car.constants import (
    HULL_X,
    HULL_Y,
    HULL_ANGLE,
    HULL_VX,
    HULL_VY,
    HULL_OMEGA,
    HULL_STATE_SIZE,
    WHEEL_OMEGA,
    WHEEL_JOINT_ANGLE,
    WHEEL_PHASE,
    WHEEL_STATE_SIZE,
    CTRL_STEERING,
    CTRL_GAS,
    CTRL_BRAKE,
    NUM_WHEELS,
    TILE_DATA_SIZE,
    MAX_TRACK_TILES,
    WHEEL_POS_FL_X,
    WHEEL_POS_FL_Y,
    WHEEL_POS_FR_X,
    WHEEL_POS_FR_Y,
    WHEEL_POS_RL_X,
    WHEEL_POS_RL_Y,
    WHEEL_POS_RR_X,
    WHEEL_POS_RR_Y,
)
from physics_gpu.car.layout import (
    META_STEP_COUNT,
    META_TOTAL_REWARD,
    META_DONE,
    META_TRUNCATED,
    META_TILES_VISITED,
)


# =============================================================================
# CarRacingV2 Environment
# =============================================================================


struct CarRacingV2(BoxContinuousActionEnv, Copyable, Movable):
    """CarRacing environment with GPU-accelerated physics.

    This environment uses the physics_gpu/car/ module for slip-based
    tire friction physics with GPU acceleration.

    Features:
    - Top-down racing with procedural track
    - 4-wheel friction model with slip physics
    - Continuous action space: [steering, gas, brake]
    - State-based observation (13D) for RL algorithms

    CPU Mode:
    - Uses CarDynamics.step_single_env() for physics
    - Track stored in List[TrackTileV2]

    GPU Mode (batch):
    - Uses CarPhysicsKernel.step_gpu() for batched physics
    - Track stored in GPU buffer (shared across envs)
    """

    # Required trait aliases
    comptime dtype = DType.float64  # For Env trait
    comptime Layout = CRConstants.Layout
    comptime STATE_SIZE: Int = CRConstants.STATE_SIZE
    comptime OBS_DIM: Int = CRConstants.OBS_DIM
    comptime ACTION_DIM: Int = CRConstants.ACTION_DIM
    comptime StateType = CarRacingV2State[Self.dtype]
    comptime ActionType = CarRacingV2Action[Self.dtype]

    # Track generator (Float64 for precision)
    var track: TrackGenerator[DType.float64]

    # Physics state buffer (using physics_gpu dtype = float32)
    var state_buffer: List[Scalar[dtype]]

    # Track tiles buffer (for TileCollision)
    var tiles_buffer: List[Scalar[dtype]]

    # Episode state
    var step_count: Int
    var total_reward: Float64
    var done: Bool
    var truncated: Bool
    var tiles_visited: Int

    # Configuration
    var max_steps: Int
    var lap_complete_percent: Float64
    var domain_randomize: Bool

    # Cached state
    var cached_state: CarRacingV2State[DType.float64]

    # =========================================================================
    # Initialization
    # =========================================================================

    fn __init__(
        out self,
        max_steps: Int = CRConstants.MAX_STEPS,
        lap_complete_percent: Float64 = CRConstants.LAP_COMPLETE_PERCENT,
        domain_randomize: Bool = False,
    ):
        """Initialize the CarRacing V2 environment.

        Args:
            max_steps: Maximum steps per episode (0 = unlimited).
            lap_complete_percent: Fraction of tiles to complete lap.
            domain_randomize: Randomize colors each reset.
        """
        # Initialize track
        self.track = TrackGenerator[DType.float64]()

        # Initialize state buffer
        self.state_buffer = List[Scalar[dtype]](capacity=Self.STATE_SIZE)
        for _ in range(Self.STATE_SIZE):
            self.state_buffer.append(Scalar[dtype](0.0))

        # Initialize tiles buffer
        var tiles_size = MAX_TRACK_TILES * TILE_DATA_SIZE
        self.tiles_buffer = List[Scalar[dtype]](capacity=tiles_size)
        for _ in range(tiles_size):
            self.tiles_buffer.append(Scalar[dtype](0.0))

        # Episode state
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.truncated = False
        self.tiles_visited = 0

        # Configuration
        self.max_steps = max_steps
        self.lap_complete_percent = lap_complete_percent
        self.domain_randomize = domain_randomize

        # Cached state
        self.cached_state = CarRacingV2State[DType.float64]()

    fn __copyinit__(out self, other: Self):
        self.track = TrackGenerator[DType.float64]()
        self.track.track = other.track.track.copy()
        self.track.track_length = other.track.track_length
        self.state_buffer = other.state_buffer.copy()
        self.tiles_buffer = other.tiles_buffer.copy()
        self.step_count = other.step_count
        self.total_reward = other.total_reward
        self.done = other.done
        self.truncated = other.truncated
        self.tiles_visited = other.tiles_visited
        self.max_steps = other.max_steps
        self.lap_complete_percent = other.lap_complete_percent
        self.domain_randomize = other.domain_randomize
        self.cached_state = other.cached_state

    fn __moveinit__(out self, deinit other: Self):
        self.track = other.track^
        self.state_buffer = other.state_buffer^
        self.tiles_buffer = other.tiles_buffer^
        self.step_count = other.step_count
        self.total_reward = other.total_reward
        self.done = other.done
        self.truncated = other.truncated
        self.tiles_visited = other.tiles_visited
        self.max_steps = other.max_steps
        self.lap_complete_percent = other.lap_complete_percent
        self.domain_randomize = other.domain_randomize
        self.cached_state = other.cached_state^

    # =========================================================================
    # Env Trait Methods
    # =========================================================================

    fn reset(mut self) -> Self.StateType:
        """Reset the environment and return initial state."""
        # Generate track (procedural random track)
        var seed = UInt64(self.step_count + 42)  # Different seed each reset
        self.track.generate_random_track(seed)

        # Copy track to tiles buffer
        self._update_tiles_buffer()

        # Reset episode state
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.truncated = False
        self.tiles_visited = 0

        # Get start position
        var start = self.track.get_start_position()
        var start_x = Float64(start[0])
        var start_y = Float64(start[1])
        var start_angle = Float64(start[2])

        # Initialize state buffer
        self._init_state_buffer(start_x, start_y, start_angle)

        # Update observation
        self._update_observation()

        # Cache and return state
        self._update_cached_state()
        return self.cached_state

    fn step(
        mut self, action: Self.ActionType
    ) -> Tuple[Self.StateType, Float64, Bool]:
        """Take action and return (next_state, reward, done)."""
        # Clamp and remap action inputs
        var steering = clamp(Float64(action.steering), -1.0, 1.0)
        var gas = clamp((Float64(action.gas) + 1.0) * 0.5, 0.0, 1.0)
        var brake = clamp((Float64(action.brake) + 1.0) * 0.5, 0.0, 1.0)

        # Store controls in state buffer
        self._set_control(steering, gas, brake)

        # Step physics
        self._step_physics()

        # Check tile visits and compute reward
        var step_reward = self._check_tiles_and_reward()

        # Increment step count
        self.step_count += 1

        # Check termination conditions
        self._check_termination(step_reward)

        # Update observation
        self._update_observation()

        # Cache and return state
        self._update_cached_state()
        return (self.cached_state, step_reward, self.done)

    fn get_state(self) -> Self.StateType:
        """Get current state."""
        return self.cached_state

    fn render(mut self, mut renderer: RendererBase):
        """Render the environment with rotating camera following the car."""
        # Get car state from physics buffer
        var state = self._get_state_tensor()
        var hull_x = Float64(rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_X]))
        var hull_y = Float64(rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_Y]))
        var hull_angle = Float64(rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_ANGLE]))
        var hull_vx = Float64(rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_VX]))
        var hull_vy = Float64(rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_VY]))

        # Background color (grass green)
        var bg_color = SDL_Color(102, 204, 102, 255)
        if not renderer.begin_frame_with_color(bg_color):
            return

        # Create rotating camera - follows car with rotation
        # Camera zoom interpolates during first second
        var t = Float64(self.step_count) * CRConstants.DT
        var zoom = CRConstants.ZOOM * CRConstants.SCALE * min(t, 1.0) + 0.1 * CRConstants.SCALE * max(1.0 - t, 0.0)

        # Screen center for camera (car appears in lower portion)
        var screen_center_x = Float64(CRConstants.WINDOW_W) / 2.0
        var screen_center_y = Float64(CRConstants.WINDOW_H) * 3.0 / 4.0

        var camera = renderer.make_rotating_camera_offset(
            hull_x, hull_y,
            -hull_angle,  # Negative to rotate view opposite to car
            zoom,
            screen_center_x,
            screen_center_y,
        )

        # Draw grass patches
        self._draw_grass(renderer, camera)

        # Draw track
        self._draw_track(renderer, camera)

        # Draw car
        self._draw_car(renderer, camera, hull_x, hull_y, hull_angle)

        # Draw HUD
        self._draw_info(renderer, hull_vx, hull_vy)

        renderer.flip()

    fn _draw_grass(self, mut renderer: RendererBase, camera: RotatingCamera):
        """Draw grass patches in a checkerboard pattern."""
        var grass_clr = SDL_Color(68, 160, 68, 255)  # Darker grass for pattern

        for gx in range(-10, 10, 2):
            for gy in range(-10, 10, 2):
                var world_x = Float64(gx) * CRConstants.GRASS_DIM
                var world_y = Float64(gy) * CRConstants.GRASS_DIM

                # Skip if too far from camera
                var dist = sqrt(
                    (world_x - camera.x) ** 2 + (world_y - camera.y) ** 2
                )
                if dist > CRConstants.PLAYFIELD:
                    continue

                # Grass quad corners
                var vertices = List[RenderVec2]()
                vertices.append(RenderVec2(world_x, world_y))
                vertices.append(RenderVec2(world_x + CRConstants.GRASS_DIM, world_y))
                vertices.append(RenderVec2(world_x + CRConstants.GRASS_DIM, world_y + CRConstants.GRASS_DIM))
                vertices.append(RenderVec2(world_x, world_y + CRConstants.GRASS_DIM))

                renderer.draw_polygon_rotating(vertices, camera, grass_clr, filled=True)

    fn _draw_track(self, mut renderer: RendererBase, camera: RotatingCamera):
        """Draw track tiles."""
        var road_clr = SDL_Color(102, 102, 102, 255)  # Gray road

        for i in range(self.track.track_length):
            var tile = self.track.track[i]

            # Check if tile is visible (rough culling)
            var dist = sqrt(
                (Float64(tile.center_x) - camera.x) ** 2
                + (Float64(tile.center_y) - camera.y) ** 2
            )
            if dist > 500.0:
                continue

            # Tile color with slight variation
            var c = UInt8(Int(0.01 * Float64(i % 3) * 255.0))
            var tile_color = SDL_Color(
                UInt8(min(Int(road_clr.r) + Int(c), 255)),
                UInt8(min(Int(road_clr.g) + Int(c), 255)),
                UInt8(min(Int(road_clr.b) + Int(c), 255)),
                255,
            )

            # Green tint for visited tiles
            if tile.visited:
                tile_color = SDL_Color(
                    UInt8(max(Int(tile_color.r) - 30, 0)),
                    UInt8(min(Int(tile_color.g) + 30, 255)),
                    UInt8(max(Int(tile_color.b) - 30, 0)),
                    255,
                )

            # Create polygon vertices
            var vertices = List[RenderVec2]()
            vertices.append(RenderVec2(Float64(tile.v0_x), Float64(tile.v0_y)))
            vertices.append(RenderVec2(Float64(tile.v1_x), Float64(tile.v1_y)))
            vertices.append(RenderVec2(Float64(tile.v2_x), Float64(tile.v2_y)))
            vertices.append(RenderVec2(Float64(tile.v3_x), Float64(tile.v3_y)))

            renderer.draw_polygon_rotating(vertices, camera, tile_color, filled=True)

    fn _draw_car(
        self,
        mut renderer: RendererBase,
        camera: RotatingCamera,
        hull_x: Float64,
        hull_y: Float64,
        hull_angle: Float64,
    ):
        """Draw the car (hull and wheels)."""
        var state = self._get_state_tensor()

        # Hull transform
        var hull_transform = Transform2D(hull_x, hull_y, hull_angle)

        # Hull color
        var hull_color = car_red()
        var car_size = CRConstants.SIZE

        # Hull polygon 1 (front spoiler)
        var hull1 = List[RenderVec2]()
        hull1.append(RenderVec2(-60.0 * car_size, 130.0 * car_size))
        hull1.append(RenderVec2(60.0 * car_size, 130.0 * car_size))
        hull1.append(RenderVec2(60.0 * car_size, 110.0 * car_size))
        hull1.append(RenderVec2(-60.0 * car_size, 110.0 * car_size))
        renderer.draw_transformed_polygon_rotating(hull1, hull_transform, camera, hull_color, filled=True)

        # Hull polygon 2 (cabin)
        var hull2 = List[RenderVec2]()
        hull2.append(RenderVec2(-15.0 * car_size, 120.0 * car_size))
        hull2.append(RenderVec2(15.0 * car_size, 120.0 * car_size))
        hull2.append(RenderVec2(20.0 * car_size, 20.0 * car_size))
        hull2.append(RenderVec2(-20.0 * car_size, 20.0 * car_size))
        renderer.draw_transformed_polygon_rotating(hull2, hull_transform, camera, hull_color, filled=True)

        # Hull polygon 3 (body)
        var hull3 = List[RenderVec2]()
        hull3.append(RenderVec2(25.0 * car_size, 20.0 * car_size))
        hull3.append(RenderVec2(50.0 * car_size, -10.0 * car_size))
        hull3.append(RenderVec2(50.0 * car_size, -40.0 * car_size))
        hull3.append(RenderVec2(20.0 * car_size, -90.0 * car_size))
        hull3.append(RenderVec2(-20.0 * car_size, -90.0 * car_size))
        hull3.append(RenderVec2(-50.0 * car_size, -40.0 * car_size))
        hull3.append(RenderVec2(-50.0 * car_size, -10.0 * car_size))
        hull3.append(RenderVec2(-25.0 * car_size, 20.0 * car_size))
        renderer.draw_transformed_polygon_rotating(hull3, hull_transform, camera, hull_color, filled=True)

        # Hull polygon 4 (rear spoiler)
        var hull4 = List[RenderVec2]()
        hull4.append(RenderVec2(-50.0 * car_size, -120.0 * car_size))
        hull4.append(RenderVec2(50.0 * car_size, -120.0 * car_size))
        hull4.append(RenderVec2(50.0 * car_size, -90.0 * car_size))
        hull4.append(RenderVec2(-50.0 * car_size, -90.0 * car_size))
        renderer.draw_transformed_polygon_rotating(hull4, hull_transform, camera, hull_color, filled=True)

        # Draw wheels
        var wheel_color = black()
        var hw = 14.0 * car_size  # Wheel width
        var hr = 27.0 * car_size  # Wheel radius

        # Wheel positions (local coords) - FL, FR, RL, RR
        var wheel_pos_x = InlineArray[Float64, 4](WHEEL_POS_FL_X, WHEEL_POS_FR_X, WHEEL_POS_RL_X, WHEEL_POS_RR_X)
        var wheel_pos_y = InlineArray[Float64, 4](WHEEL_POS_FL_Y, WHEEL_POS_FR_Y, WHEEL_POS_RL_Y, WHEEL_POS_RR_Y)

        for i in range(4):
            # Get wheel joint angle from state
            var wheel_off = Self.Layout.WHEELS_OFFSET + i * WHEEL_STATE_SIZE
            var joint_angle = Float64(rebind[Scalar[dtype]](state[0, wheel_off + WHEEL_JOINT_ANGLE]))

            # World position of wheel
            var local_x = wheel_pos_x[i]
            var local_y = wheel_pos_y[i]
            var c = cos(hull_angle)
            var s = sin(hull_angle)
            var world_x = hull_x + local_x * c - local_y * s
            var world_y = hull_y + local_x * s + local_y * c

            # Wheel angle (hull angle + joint angle for front wheels)
            var wheel_angle = hull_angle
            if i < 2:  # Front wheels have steering
                wheel_angle += joint_angle

            var wheel_transform = Transform2D(world_x, world_y, wheel_angle)

            # Draw wheel
            var wheel_verts = List[RenderVec2]()
            wheel_verts.append(RenderVec2(-hw, hr))
            wheel_verts.append(RenderVec2(hw, hr))
            wheel_verts.append(RenderVec2(hw, -hr))
            wheel_verts.append(RenderVec2(-hw, -hr))
            renderer.draw_transformed_polygon_rotating(wheel_verts, wheel_transform, camera, wheel_color, filled=True)

    fn _draw_info(self, mut renderer: RendererBase, vx: Float64, vy: Float64):
        """Draw HUD info panel."""
        var W = Float64(CRConstants.WINDOW_W)
        var H = Float64(CRConstants.WINDOW_H)
        var s = W / 40.0  # Scale unit
        var h = H / 40.0  # Height unit

        # Draw black background panel
        var bg_color = SDL_Color(0, 0, 0, 255)
        renderer.draw_rect(0, Int(H - 5 * h), Int(W), Int(5 * h), bg_color, 0)

        # Calculate speed
        var speed = sqrt(vx * vx + vy * vy)

        # Draw speed indicator (white bar)
        if speed > 0.0001:
            var speed_height = 0.02 * speed
            self._draw_vertical_indicator(renderer, 5, speed_height, white(), s, h, Int(H))

        # Get wheel omegas from physics buffer
        var state = self._get_state_tensor()
        for i in range(4):
            var wheel_off = Self.Layout.WHEELS_OFFSET + i * WHEEL_STATE_SIZE
            var omega = Float64(rebind[Scalar[dtype]](state[0, wheel_off + WHEEL_OMEGA]))

            var abs_omega = omega if omega >= 0.0 else -omega
            if abs_omega > 0.0001:
                # Front wheels in blue, rear in purple
                var color = SDL_Color(0, 0, 255, 255) if i < 2 else SDL_Color(128, 0, 255, 255)
                self._draw_vertical_indicator(renderer, 7 + i, 0.01 * omega, color, s, h, Int(H))

        # Draw progress indicator (green)
        var progress = Float64(self.tiles_visited) / Float64(max(self.track.track_length, 1))
        self._draw_vertical_indicator(renderer, 15, progress * 3.0, SDL_Color(0, 255, 0, 255), s, h, Int(H))

    fn _draw_vertical_indicator(
        self,
        mut renderer: RendererBase,
        x_pos: Int,
        height: Float64,
        color: SDL_Color,
        s: Float64,
        h: Float64,
        screen_h: Int,
    ):
        """Draw a vertical indicator bar."""
        var x = Int(Float64(x_pos) * s)
        var bar_h = min(height * h, 4.0 * h)
        var abs_bar_h = bar_h if bar_h >= 0.0 else -bar_h

        if bar_h >= 0.0:
            var y = screen_h - Int(5 * h) - Int(bar_h)
            renderer.draw_rect(x, y, Int(s), Int(abs_bar_h), color, 0)
        else:
            var y = screen_h - Int(5 * h)
            renderer.draw_rect(x, y, Int(s), Int(abs_bar_h), color, 0)

    fn close(mut self):
        """Clean up resources."""
        pass

    # =========================================================================
    # BoxContinuousActionEnv Trait Methods
    # =========================================================================

    fn get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return observation as list."""
        return self.cached_state.to_list_typed[Self.dtype]()

    fn reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset and return initial observation."""
        var state = self.reset()
        return state.to_list_typed[Self.dtype]()

    fn obs_dim(self) -> Int:
        """Observation dimension: 13."""
        return Self.OBS_DIM

    fn action_dim(self) -> Int:
        """Action dimension: 3."""
        return Self.ACTION_DIM

    fn action_low(self) -> Scalar[Self.dtype]:
        """Action lower bound: -1.0."""
        return -1.0

    fn action_high(self) -> Scalar[Self.dtype]:
        """Action upper bound: 1.0."""
        return 1.0

    fn step_continuous(
        mut self, action: Scalar[Self.dtype]
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Step with single action (applies to steering only)."""
        var act = CarRacingV2Action[Self.dtype](action, 0.0, 0.0)
        var result = self.step(act)
        return (result[0].to_list_typed[Self.dtype](), result[1], result[2])

    fn step_continuous_vec(
        mut self, action: List[Scalar[Self.dtype]]
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Step with action vector."""
        var act = CarRacingV2Action[Self.dtype].from_list(action)
        var result = self.step(act)
        return (result[0].to_list_typed[Self.dtype](), result[1], result[2])

    fn step_continuous_vec_f64(
        mut self, action: List[Float64]
    ) -> Tuple[List[Float64], Float64, Bool]:
        """Step with action vector (Float64 convenience method)."""
        var typed_action = List[Scalar[Self.dtype]]()
        for i in range(len(action)):
            typed_action.append(Scalar[Self.dtype](action[i]))
        var result = self.step_continuous_vec(typed_action)
        var obs_f64 = List[Float64]()
        for i in range(len(result[0])):
            obs_f64.append(Float64(result[0][i]))
        return (obs_f64^, Float64(result[1]), result[2])

    # =========================================================================
    # Internal State Buffer Access
    # =========================================================================

    fn _get_state_tensor(
        self,
    ) -> LayoutTensor[
        dtype,
        Layout.row_major(1, Self.STATE_SIZE),
        MutAnyOrigin,
    ]:
        """Get state tensor view (BATCH=1 for CPU mode)."""
        return LayoutTensor[
            dtype,
            Layout.row_major(1, Self.STATE_SIZE),
            MutAnyOrigin,
        ](self.state_buffer.unsafe_ptr())

    fn _get_tiles_tensor(
        self,
    ) -> LayoutTensor[
        dtype,
        Layout.row_major(MAX_TRACK_TILES, TILE_DATA_SIZE),
        MutAnyOrigin,
    ]:
        """Get tiles tensor view."""
        return LayoutTensor[
            dtype,
            Layout.row_major(MAX_TRACK_TILES, TILE_DATA_SIZE),
            MutAnyOrigin,
        ](self.tiles_buffer.unsafe_ptr())

    fn _init_state_buffer(
        mut self,
        x: Float64,
        y: Float64,
        angle: Float64,
    ):
        """Initialize state buffer with starting position."""
        var state = self._get_state_tensor()

        # Clear buffer
        for i in range(Self.STATE_SIZE):
            state[0, i] = Scalar[dtype](0.0)

        # Set hull state
        state[0, Self.Layout.HULL_OFFSET + HULL_X] = Scalar[dtype](x)
        state[0, Self.Layout.HULL_OFFSET + HULL_Y] = Scalar[dtype](y)
        state[0, Self.Layout.HULL_OFFSET + HULL_ANGLE] = Scalar[dtype](angle)

    fn _set_control(
        mut self,
        steering: Float64,
        gas: Float64,
        brake: Float64,
    ):
        """Set control inputs in state buffer."""
        var state = self._get_state_tensor()
        state[0, Self.Layout.CONTROLS_OFFSET + CTRL_STEERING] = Scalar[dtype](steering)
        state[0, Self.Layout.CONTROLS_OFFSET + CTRL_GAS] = Scalar[dtype](gas)
        state[0, Self.Layout.CONTROLS_OFFSET + CTRL_BRAKE] = Scalar[dtype](brake)

    fn _update_tiles_buffer(mut self):
        """Copy track tiles to tiles buffer."""
        var tiles = self._get_tiles_tensor()
        self.track.to_buffer[MAX_TRACK_TILES](tiles)

    # =========================================================================
    # Physics Step
    # =========================================================================

    fn _step_physics(mut self):
        """Step physics using CarDynamics."""
        var state = self._get_state_tensor()
        var tiles = self._get_tiles_tensor()
        var dt = Scalar[dtype](CRConstants.DT)

        CarDynamics.step_with_obs[
            1,  # BATCH = 1 for CPU mode
            Self.STATE_SIZE,
            Self.Layout.OBS_OFFSET,
            Self.Layout.OBS_DIM,
            Self.Layout.HULL_OFFSET,
            Self.Layout.WHEELS_OFFSET,
            Self.Layout.CONTROLS_OFFSET,
            MAX_TRACK_TILES,
        ](0, state, tiles, self.track.track_length, dt)

    # =========================================================================
    # Reward and Termination
    # =========================================================================

    fn _check_tiles_and_reward(mut self) -> Float64:
        """Check tile visits and compute reward."""
        var state = self._get_state_tensor()
        var tiles = self._get_tiles_tensor()

        # Get hull position (use rebind to extract Scalar from LayoutTensor)
        var hull_x = rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_X])
        var hull_y = rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_Y])

        # Check which tile we're on
        var tile_idx = TileCollision.check_tile_visited[MAX_TRACK_TILES](
            hull_x, hull_y, tiles, self.track.track_length
        )

        # Reward for visiting new tiles
        var step_reward: Float64 = -0.1  # Time penalty

        if tile_idx >= 0:
            if self.track.mark_tile_visited(tile_idx):
                # New tile visited!
                self.tiles_visited += 1
                step_reward += 1000.0 / Float64(self.track.track_length)

        self.total_reward += step_reward
        return step_reward

    fn _check_termination(mut self, mut step_reward: Float64):
        """Check termination conditions."""
        var state = self._get_state_tensor()

        # Get hull position (rebind to Scalar, then cast to Float64)
        var hull_x = Float64(rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_X]))
        var hull_y = Float64(rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_Y]))

        var playfield = Float64(CRConstants.PLAYFIELD)

        # Check if off playfield (use explicit comparison)
        var abs_x = hull_x if hull_x >= 0.0 else -hull_x
        var abs_y = hull_y if hull_y >= 0.0 else -hull_y
        if abs_x > playfield or abs_y > playfield:
            self.done = True
            step_reward = -100.0

        # Check lap completion
        var progress = Float64(self.tiles_visited) / Float64(max(
            self.track.track_length, 1
        ))
        if progress >= self.lap_complete_percent:
            self.done = True

        # Check max steps (truncation)
        if self.max_steps > 0 and self.step_count >= self.max_steps:
            self.done = True
            self.truncated = True

    # =========================================================================
    # Observation Update
    # =========================================================================

    fn _update_observation(mut self):
        """Update observation in state buffer (called after physics step)."""
        # CarDynamics.step_with_obs already updates observations
        pass

    fn _update_cached_state(mut self):
        """Update cached state from buffer."""
        var state = self._get_state_tensor()
        var obs_off = Self.Layout.OBS_OFFSET

        # Extract values using rebind and convert to Float64
        self.cached_state.x = Float64(rebind[Scalar[dtype]](state[0, obs_off + 0]))
        self.cached_state.y = Float64(rebind[Scalar[dtype]](state[0, obs_off + 1]))
        self.cached_state.angle = Float64(rebind[Scalar[dtype]](state[0, obs_off + 2]))
        self.cached_state.vx = Float64(rebind[Scalar[dtype]](state[0, obs_off + 3]))
        self.cached_state.vy = Float64(rebind[Scalar[dtype]](state[0, obs_off + 4]))
        self.cached_state.angular_velocity = Float64(rebind[Scalar[dtype]](state[0, obs_off + 5]))
        self.cached_state.wheel_angle_fl = Float64(rebind[Scalar[dtype]](state[0, obs_off + 6]))
        self.cached_state.wheel_angle_fr = Float64(rebind[Scalar[dtype]](state[0, obs_off + 7]))
        self.cached_state.wheel_omega_fl = Float64(rebind[Scalar[dtype]](state[0, obs_off + 8]))
        self.cached_state.wheel_omega_fr = Float64(rebind[Scalar[dtype]](state[0, obs_off + 9]))
        self.cached_state.wheel_omega_rl = Float64(rebind[Scalar[dtype]](state[0, obs_off + 10]))
        self.cached_state.wheel_omega_rr = Float64(rebind[Scalar[dtype]](state[0, obs_off + 11]))
        self.cached_state.speed = Float64(rebind[Scalar[dtype]](state[0, obs_off + 12]))

    # =========================================================================
    # GPU Batch Operations (Static Methods)
    # =========================================================================

    @staticmethod
    fn reset_kernel_gpu[BATCH: Int](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        tiles_buf: DeviceBuffer[dtype],
        num_tiles: Int,
    ) raises:
        """Reset all environments (GPU batch).

        Args:
            ctx: GPU device context.
            states_buf: State buffer [BATCH * STATE_SIZE].
            tiles_buf: Track tiles buffer (shared).
            num_tiles: Number of track tiles.
        """
        # TODO: Implement GPU reset kernel
        pass

    @staticmethod
    fn step_kernel_gpu[BATCH: Int](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        tiles_buf: DeviceBuffer[dtype],
        num_tiles: Int,
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
    ) raises:
        """Step all environments (GPU batch).

        Args:
            ctx: GPU device context.
            states_buf: State buffer [BATCH * STATE_SIZE].
            actions_buf: Action buffer [BATCH * 3].
            tiles_buf: Track tiles buffer (shared).
            num_tiles: Number of track tiles.
            rewards_buf: Output rewards [BATCH].
            dones_buf: Output done flags [BATCH].
        """
        # Step 1: Copy actions to state controls
        # (TODO: Add action copy kernel)

        # Step 2: Run physics
        CarPhysicsKernel.step_gpu[
            BATCH,
            CRConstants.STATE_SIZE,
            CRConstants.OBS_OFFSET,
            CRConstants.OBS_DIM,
            CRConstants.HULL_OFFSET,
            CRConstants.WHEELS_OFFSET,
            CRConstants.CONTROLS_OFFSET,
            MAX_TRACK_TILES,
        ](ctx, states_buf, tiles_buf, num_tiles, Scalar[dtype](CRConstants.DT))

        # Step 3: Compute rewards and dones
        # (TODO: Add reward/done kernel)


# =============================================================================
# Helper Functions
# =============================================================================


fn clamp(x: Float64, low: Float64, high: Float64) -> Float64:
    """Clamp value to range."""
    if x < low:
        return low
    if x > high:
        return high
    return x
