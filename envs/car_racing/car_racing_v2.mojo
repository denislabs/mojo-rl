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
from random.philox import Random as PhiloxRandom

from core import BoxContinuousActionEnv, GPUContinuousEnv, Action
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


struct CarRacingV2[DTYPE: DType](
    BoxContinuousActionEnv, Copyable, GPUContinuousEnv, Movable
):
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
    comptime dtype = Self.DTYPE  # For Env trait
    comptime Layout = CRConstants.Layout
    comptime STATE_SIZE: Int = CRConstants.STATE_SIZE
    comptime OBS_DIM: Int = CRConstants.OBS_DIM
    comptime ACTION_DIM: Int = CRConstants.ACTION_DIM
    comptime StateType = CarRacingV2State[Self.DTYPE]
    comptime ActionType = CarRacingV2Action[Self.DTYPE]

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
    var cached_state: CarRacingV2State[Self.dtype]

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
        self.cached_state = CarRacingV2State[Self.dtype]()

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
        self.track.generate_random_track(seed, verbose=False)

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
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
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
        return (self.cached_state, Scalar[Self.dtype](step_reward), self.done)

    fn get_state(self) -> Self.StateType:
        """Get current state."""
        return self.cached_state

    fn render(mut self, mut renderer: RendererBase):
        """Render the environment with rotating camera following the car."""
        # Get car state from physics buffer
        var state = self._get_state_tensor()
        var hull_x = Float64(
            rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_X])
        )
        var hull_y = Float64(
            rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_Y])
        )
        var hull_angle = Float64(
            rebind[Scalar[dtype]](
                state[0, Self.Layout.HULL_OFFSET + HULL_ANGLE]
            )
        )
        var hull_vx = Float64(
            rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_VX])
        )
        var hull_vy = Float64(
            rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_VY])
        )

        # Background color (grass green)
        var bg_color = SDL_Color(102, 204, 102, 255)
        if not renderer.begin_frame_with_color(bg_color):
            return

        # Create rotating camera - follows car with rotation
        # Camera zoom interpolates during first second
        var t = Float64(self.step_count) * CRConstants.DT
        var zoom = CRConstants.ZOOM * CRConstants.SCALE * min(
            t, 1.0
        ) + 0.1 * CRConstants.SCALE * max(1.0 - t, 0.0)

        # Screen center for camera (car appears in lower portion)
        var screen_center_x = Float64(CRConstants.WINDOW_W) / 2.0
        var screen_center_y = Float64(CRConstants.WINDOW_H) * 3.0 / 4.0

        var camera = renderer.make_rotating_camera_offset(
            hull_x,
            hull_y,
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
                vertices.append(
                    RenderVec2(world_x + CRConstants.GRASS_DIM, world_y)
                )
                vertices.append(
                    RenderVec2(
                        world_x + CRConstants.GRASS_DIM,
                        world_y + CRConstants.GRASS_DIM,
                    )
                )
                vertices.append(
                    RenderVec2(world_x, world_y + CRConstants.GRASS_DIM)
                )

                renderer.draw_polygon_rotating(
                    vertices, camera, grass_clr, filled=True
                )

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

            renderer.draw_polygon_rotating(
                vertices, camera, tile_color, filled=True
            )

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
        renderer.draw_transformed_polygon_rotating(
            hull1, hull_transform, camera, hull_color, filled=True
        )

        # Hull polygon 2 (cabin)
        var hull2 = List[RenderVec2]()
        hull2.append(RenderVec2(-15.0 * car_size, 120.0 * car_size))
        hull2.append(RenderVec2(15.0 * car_size, 120.0 * car_size))
        hull2.append(RenderVec2(20.0 * car_size, 20.0 * car_size))
        hull2.append(RenderVec2(-20.0 * car_size, 20.0 * car_size))
        renderer.draw_transformed_polygon_rotating(
            hull2, hull_transform, camera, hull_color, filled=True
        )

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
        renderer.draw_transformed_polygon_rotating(
            hull3, hull_transform, camera, hull_color, filled=True
        )

        # Hull polygon 4 (rear spoiler)
        var hull4 = List[RenderVec2]()
        hull4.append(RenderVec2(-50.0 * car_size, -120.0 * car_size))
        hull4.append(RenderVec2(50.0 * car_size, -120.0 * car_size))
        hull4.append(RenderVec2(50.0 * car_size, -90.0 * car_size))
        hull4.append(RenderVec2(-50.0 * car_size, -90.0 * car_size))
        renderer.draw_transformed_polygon_rotating(
            hull4, hull_transform, camera, hull_color, filled=True
        )

        # Draw wheels
        var wheel_color = black()
        var hw = 14.0 * car_size  # Wheel width
        var hr = 27.0 * car_size  # Wheel radius

        # Wheel positions (local coords) - FL, FR, RL, RR
        var wheel_pos_x = InlineArray[Float64, 4](
            WHEEL_POS_FL_X, WHEEL_POS_FR_X, WHEEL_POS_RL_X, WHEEL_POS_RR_X
        )
        var wheel_pos_y = InlineArray[Float64, 4](
            WHEEL_POS_FL_Y, WHEEL_POS_FR_Y, WHEEL_POS_RL_Y, WHEEL_POS_RR_Y
        )

        for i in range(4):
            # Get wheel joint angle from state
            var wheel_off = Self.Layout.WHEELS_OFFSET + i * WHEEL_STATE_SIZE
            var joint_angle = Float64(
                rebind[Scalar[dtype]](state[0, wheel_off + WHEEL_JOINT_ANGLE])
            )

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
            renderer.draw_transformed_polygon_rotating(
                wheel_verts, wheel_transform, camera, wheel_color, filled=True
            )

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
            self._draw_vertical_indicator(
                renderer, 5, speed_height, white(), s, h, Int(H)
            )

        # Get wheel omegas from physics buffer
        var state = self._get_state_tensor()
        for i in range(4):
            var wheel_off = Self.Layout.WHEELS_OFFSET + i * WHEEL_STATE_SIZE
            var omega = Float64(
                rebind[Scalar[dtype]](state[0, wheel_off + WHEEL_OMEGA])
            )

            var abs_omega = omega if omega >= 0.0 else -omega
            if abs_omega > 0.0001:
                # Front wheels in blue, rear in purple
                var color = SDL_Color(0, 0, 255, 255) if i < 2 else SDL_Color(
                    128, 0, 255, 255
                )
                self._draw_vertical_indicator(
                    renderer, 7 + i, 0.01 * omega, color, s, h, Int(H)
                )

        # Draw progress indicator (green)
        var progress = Float64(self.tiles_visited) / Float64(
            max(self.track.track_length, 1)
        )
        self._draw_vertical_indicator(
            renderer,
            15,
            progress * 3.0,
            SDL_Color(0, 255, 0, 255),
            s,
            h,
            Int(H),
        )

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
        """Draw a vertical indicator bar inside the HUD panel."""
        var x = Int(Float64(x_pos) * s)
        var bar_h = min(height * h, 4.0 * h)
        var abs_bar_h = bar_h if bar_h >= 0.0 else -bar_h

        # HUD panel occupies bottom 5*h pixels
        # Baseline is in the middle of the HUD panel
        var baseline = screen_h - Int(2.5 * h)  # Middle of HUD

        if bar_h >= 0.0:
            # Positive: draw upward from baseline
            var y = baseline - Int(abs_bar_h)
            renderer.draw_rect(x, y, Int(s), Int(abs_bar_h) + 1, color, 0)
        else:
            # Negative: draw downward from baseline
            renderer.draw_rect(
                x, baseline, Int(s), Int(abs_bar_h) + 1, color, 0
            )

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
        return (
            result[0].to_list_typed[Self.dtype](),
            Scalar[Self.dtype](result[1]),
            result[2],
        )

    fn step_continuous_vec[
        DTYPE_VEC: DType
    ](mut self, action: List[Scalar[DTYPE_VEC]]) -> Tuple[
        List[Scalar[DTYPE_VEC]], Scalar[DTYPE_VEC], Bool
    ]:
        """Step with action vector (DTYPE convenience method)."""
        var typed_action = List[Scalar[Self.dtype]]()
        for i in range(len(action)):
            typed_action.append(Scalar[Self.dtype](action[i]))
        var act = CarRacingV2Action[Self.dtype].from_list(typed_action)
        var result = self.step(act)
        return (
            result[0].to_list_typed[DTYPE_VEC](),
            Scalar[DTYPE_VEC](result[1]),
            result[2],
        )

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
        state[0, Self.Layout.CONTROLS_OFFSET + CTRL_STEERING] = Scalar[dtype](
            steering
        )
        state[0, Self.Layout.CONTROLS_OFFSET + CTRL_GAS] = Scalar[dtype](gas)
        state[0, Self.Layout.CONTROLS_OFFSET + CTRL_BRAKE] = Scalar[dtype](
            brake
        )

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
        var hull_x = rebind[Scalar[dtype]](
            state[0, Self.Layout.HULL_OFFSET + HULL_X]
        )
        var hull_y = rebind[Scalar[dtype]](
            state[0, Self.Layout.HULL_OFFSET + HULL_Y]
        )

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
        var hull_x = Float64(
            rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_X])
        )
        var hull_y = Float64(
            rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_Y])
        )

        var playfield = Float64(CRConstants.PLAYFIELD)

        # Check if off playfield (use explicit comparison)
        var abs_x = hull_x if hull_x >= 0.0 else -hull_x
        var abs_y = hull_y if hull_y >= 0.0 else -hull_y
        if abs_x > playfield or abs_y > playfield:
            self.done = True
            step_reward = -100.0

        # Check lap completion
        var progress = Float64(self.tiles_visited) / Float64(
            max(self.track.track_length, 1)
        )
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
        # Copy hull state to observation buffer
        var state = self._get_state_tensor()

        # Hull state (6 values)
        state[0, Self.Layout.OBS_OFFSET + 0] = state[
            0, Self.Layout.HULL_OFFSET + HULL_X
        ]
        state[0, Self.Layout.OBS_OFFSET + 1] = state[
            0, Self.Layout.HULL_OFFSET + HULL_Y
        ]
        state[0, Self.Layout.OBS_OFFSET + 2] = state[
            0, Self.Layout.HULL_OFFSET + HULL_ANGLE
        ]
        state[0, Self.Layout.OBS_OFFSET + 3] = state[
            0, Self.Layout.HULL_OFFSET + HULL_VX
        ]
        state[0, Self.Layout.OBS_OFFSET + 4] = state[
            0, Self.Layout.HULL_OFFSET + HULL_VY
        ]
        state[0, Self.Layout.OBS_OFFSET + 5] = state[
            0, Self.Layout.HULL_OFFSET + HULL_OMEGA
        ]

        # Front wheel angles (2 values)
        var fl_off = Self.Layout.WHEELS_OFFSET + 0 * WHEEL_STATE_SIZE
        var fr_off = Self.Layout.WHEELS_OFFSET + 1 * WHEEL_STATE_SIZE
        state[0, Self.Layout.OBS_OFFSET + 6] = state[
            0, fl_off + WHEEL_JOINT_ANGLE
        ]
        state[0, Self.Layout.OBS_OFFSET + 7] = state[
            0, fr_off + WHEEL_JOINT_ANGLE
        ]

        # Wheel angular velocities (4 values)
        for wheel in range(NUM_WHEELS):
            var wheel_off = Self.Layout.WHEELS_OFFSET + wheel * WHEEL_STATE_SIZE
            state[0, Self.Layout.OBS_OFFSET + 8 + wheel] = state[
                0, wheel_off + WHEEL_OMEGA
            ]

        # Speed indicator (computed from vx, vy)
        var vx = rebind[Scalar[dtype]](
            state[0, Self.Layout.HULL_OFFSET + HULL_VX]
        )
        var vy = rebind[Scalar[dtype]](
            state[0, Self.Layout.HULL_OFFSET + HULL_VY]
        )
        var speed = sqrt(vx * vx + vy * vy)
        state[0, Self.Layout.OBS_OFFSET + 12] = speed

    fn _update_cached_state(mut self):
        """Update cached state from buffer."""
        var state = self._get_state_tensor()
        var obs_off = Self.Layout.OBS_OFFSET

        # Extract values using rebind and convert to Float64
        self.cached_state.x = rebind[Scalar[Self.dtype]](state[0, obs_off + 0])
        self.cached_state.y = rebind[Scalar[Self.dtype]](state[0, obs_off + 1])
        self.cached_state.angle = rebind[Scalar[Self.dtype]](
            state[0, obs_off + 2]
        )
        self.cached_state.vx = rebind[Scalar[Self.dtype]](state[0, obs_off + 3])
        self.cached_state.vy = rebind[Scalar[Self.dtype]](state[0, obs_off + 4])
        self.cached_state.angular_velocity = rebind[Scalar[Self.dtype]](
            state[0, obs_off + 5]
        )
        self.cached_state.wheel_angle_fl = rebind[Scalar[Self.dtype]](
            state[0, obs_off + 6]
        )
        self.cached_state.wheel_angle_fr = rebind[Scalar[Self.dtype]](
            state[0, obs_off + 7]
        )
        self.cached_state.wheel_omega_fl = rebind[Scalar[Self.dtype]](
            state[0, obs_off + 8]
        )

        self.cached_state.wheel_omega_fr = rebind[Scalar[Self.dtype]](
            state[0, obs_off + 9]
        )

        self.cached_state.wheel_omega_rl = rebind[Scalar[Self.dtype]](
            state[0, obs_off + 10]
        )

        self.cached_state.wheel_omega_rr = rebind[Scalar[Self.dtype]](
            state[0, obs_off + 11]
        )

        self.cached_state.speed = rebind[Scalar[Self.dtype]](
            state[0, obs_off + 12]
        )

    # =========================================================================
    # GPU Batch Operations (Static Methods) - GPUContinuousEnv Trait
    # =========================================================================

    @staticmethod
    fn step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        actions: DeviceBuffer[dtype],
        mut rewards: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
        mut obs: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Perform one environment step with continuous actions (GPUContinuousEnv trait).

        NOTE: This trait-compliant method requires a pre-initialized track in a
        separate tiles buffer. For full functionality, use step_kernel_gpu_with_track.
        This method uses a simple circular track stored as a static buffer.

        Args:
            ctx: GPU device context.
            states: State buffer [BATCH_SIZE * STATE_SIZE].
            actions: Continuous actions buffer [BATCH_SIZE * ACTION_DIM].
            rewards: Rewards buffer (output) [BATCH_SIZE].
            dones: Done flags buffer (output) [BATCH_SIZE].
            obs: Observations buffer (output) [BATCH_SIZE * OBS_DIM].
            rng_seed: Optional random seed for physics.
        """
        # Create a simple circular track buffer for trait compliance
        # In production, use step_kernel_gpu_with_track for proper track support
        var tiles_buf = ctx.enqueue_create_buffer[dtype](
            MAX_TRACK_TILES * TILE_DATA_SIZE
        )

        # Initialize simple circular track
        CarRacingV2[Self.dtype]._init_circular_track_gpu(ctx, tiles_buf)

        # Use the full implementation with track
        var obs_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OBS_DIM)
        CarRacingV2[Self.dtype].step_kernel_gpu_with_track[BATCH_SIZE](
            ctx, states, actions, tiles_buf, 100, rewards, dones, obs_buf
        )

        # Copy observations to output buffer
        CarRacingV2[Self.dtype]._copy_obs_gpu[BATCH_SIZE, OBS_DIM](
            ctx, states, obs
        )

    @staticmethod
    fn reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states: DeviceBuffer[dtype]) raises:
        """Reset all environments to random initial values (GPUContinuousEnv trait).

        Initializes cars at start position of a simple circular track.
        For procedural tracks, use reset_kernel_gpu_with_track.

        Args:
            ctx: GPU device context.
            states: State buffer [BATCH_SIZE * STATE_SIZE].
        """
        var states_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn reset_wrapper(
            states: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            CarRacingV2[Self.dtype]._reset_env_gpu[BATCH_SIZE, STATE_SIZE](
                states, env, env * 12345
            )

        ctx.enqueue_function[reset_wrapper, reset_wrapper](
            states_tensor,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
        rng_seed: UInt32,
    ) raises:
        """Reset only done environments (GPUContinuousEnv trait).

        Args:
            ctx: GPU device context.
            states: State buffer [BATCH_SIZE * STATE_SIZE].
            dones: Done flags buffer [BATCH_SIZE].
            rng_seed: Random seed for initialization.
        """
        var states_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states.unsafe_ptr())

        var dones_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn selective_reset_wrapper(
            states: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
            ],
            dones: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            seed: Scalar[dtype],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            # Only reset if done
            if rebind[Scalar[dtype]](dones[env]) > Scalar[dtype](0.5):
                CarRacingV2[Self.dtype]._reset_env_gpu[BATCH_SIZE, STATE_SIZE](
                    states, env, Int(seed) + env
                )
                dones[env] = Scalar[dtype](0.0)

        ctx.enqueue_function[selective_reset_wrapper, selective_reset_wrapper](
            states_tensor,
            dones_tensor,
            Scalar[dtype](rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # Extended GPU Methods (CarRacing-specific with track buffer)
    # =========================================================================

    @staticmethod
    fn step_kernel_gpu_with_track[
        BATCH: Int
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        tiles_buf: DeviceBuffer[dtype],
        num_tiles: Int,
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        mut obs_buf: DeviceBuffer[dtype],
    ) raises:
        """Step all environments with shared track data (CarRacing-specific).

        This is the primary GPU step method that provides full functionality
        with proper track tile collision and reward computation.

        Args:
            ctx: GPU device context.
            states_buf: State buffer [BATCH * STATE_SIZE].
            actions_buf: Action buffer [BATCH * 3].
            tiles_buf: Track tiles buffer (shared).
            num_tiles: Number of track tiles.
            rewards_buf: Output rewards [BATCH].
            dones_buf: Output done flags [BATCH].
            obs_buf: Output observations [BATCH * OBS_DIM].
        """
        # Create tensor views
        var states = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, CRConstants.STATE_SIZE),
            MutAnyOrigin,
        ](states_buf.unsafe_ptr())

        var actions = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, CRConstants.ACTION_DIM),
            MutAnyOrigin,
        ](actions_buf.unsafe_ptr())

        var tiles = LayoutTensor[
            dtype,
            Layout.row_major(MAX_TRACK_TILES, TILE_DATA_SIZE),
            MutAnyOrigin,
        ](tiles_buf.unsafe_ptr())

        var rewards = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())

        var dones = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
            dones_buf.unsafe_ptr()
        )

        var obs = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRConstants.OBS_DIM), MutAnyOrigin
        ](obs_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        # Fused step kernel: actions -> physics -> reward -> done -> obs
        @always_inline
        fn step_wrapper(
            states: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, CRConstants.STATE_SIZE),
                MutAnyOrigin,
            ],
            actions: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, CRConstants.ACTION_DIM),
                MutAnyOrigin,
            ],
            tiles: LayoutTensor[
                dtype,
                Layout.row_major(MAX_TRACK_TILES, TILE_DATA_SIZE),
                MutAnyOrigin,
            ],
            rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            obs: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, CRConstants.OBS_DIM),
                MutAnyOrigin,
            ],
            num_tiles: Int,
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH:
                return
            CarRacingV2[Self.dtype]._step_env_gpu[BATCH](
                env, states, actions, tiles, num_tiles, rewards, dones, obs
            )

        ctx.enqueue_function[step_wrapper, step_wrapper](
            states,
            actions,
            tiles,
            rewards,
            dones,
            obs,
            num_tiles,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn reset_kernel_gpu_with_track[
        BATCH: Int
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        tiles_buf: DeviceBuffer[dtype],
        num_tiles: Int,
    ) raises:
        """Reset all environments with shared track data (CarRacing-specific).

        Args:
            ctx: GPU device context.
            states_buf: State buffer [BATCH * STATE_SIZE].
            tiles_buf: Track tiles buffer (shared).
            num_tiles: Number of track tiles.
        """
        var states = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, CRConstants.STATE_SIZE),
            MutAnyOrigin,
        ](states_buf.unsafe_ptr())

        var tiles = LayoutTensor[
            dtype,
            Layout.row_major(MAX_TRACK_TILES, TILE_DATA_SIZE),
            MutAnyOrigin,
        ](tiles_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn reset_wrapper(
            states: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, CRConstants.STATE_SIZE),
                MutAnyOrigin,
            ],
            tiles: LayoutTensor[
                dtype,
                Layout.row_major(MAX_TRACK_TILES, TILE_DATA_SIZE),
                MutAnyOrigin,
            ],
            num_tiles: Int,
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH:
                return
            CarRacingV2[Self.dtype]._reset_env_with_track_gpu[BATCH](
                states, tiles, num_tiles, env, env * 12345
            )

        ctx.enqueue_function[reset_wrapper, reset_wrapper](
            states,
            tiles,
            num_tiles,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # GPU Helper Methods (Private)
    # =========================================================================

    @always_inline
    @staticmethod
    fn _reset_env_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        seed: Int,
    ):
        """Reset a single environment to initial state (GPU-compatible).

        Initializes car at start of a simple circular track.
        """
        # Simple circular track start position
        var start_x = Scalar[dtype](
            CRConstants.TRACK_RAD
        )  # Start at right side of circle
        var start_y = Scalar[dtype](0.0)
        var start_angle = Scalar[dtype](
            pi / 2.0
        )  # Facing up (tangent to circle)

        # Add small random perturbation
        var rng = PhiloxRandom(seed=seed + env, offset=0)
        var rand_vals = rng.step_uniform()
        var rand_angle: Scalar[dtype] = (
            rand_vals[0] - Scalar[dtype](0.5)
        ) * Scalar[dtype](0.1)

        # Clear entire state
        for i in range(STATE_SIZE):
            states[env, i] = Scalar[dtype](0.0)

        # Set hull state
        states[env, CRConstants.HULL_OFFSET + HULL_X] = start_x
        states[env, CRConstants.HULL_OFFSET + HULL_Y] = start_y
        states[env, CRConstants.HULL_OFFSET + HULL_ANGLE] = (
            start_angle + rand_angle
        )
        states[env, CRConstants.HULL_OFFSET + HULL_VX] = Scalar[dtype](0.0)
        states[env, CRConstants.HULL_OFFSET + HULL_VY] = Scalar[dtype](0.0)
        states[env, CRConstants.HULL_OFFSET + HULL_OMEGA] = Scalar[dtype](0.0)

        # Initialize wheel states (all zeros - angles and omegas)
        for wheel in range(NUM_WHEELS):
            var wheel_off = CRConstants.WHEELS_OFFSET + wheel * WHEEL_STATE_SIZE
            states[env, wheel_off + WHEEL_OMEGA] = Scalar[dtype](0.0)
            states[env, wheel_off + WHEEL_JOINT_ANGLE] = Scalar[dtype](0.0)
            states[env, wheel_off + WHEEL_PHASE] = Scalar[dtype](0.0)

        # Initialize controls to zero
        states[env, CRConstants.CONTROLS_OFFSET + CTRL_STEERING] = Scalar[
            dtype
        ](0.0)
        states[env, CRConstants.CONTROLS_OFFSET + CTRL_GAS] = Scalar[dtype](0.0)
        states[env, CRConstants.CONTROLS_OFFSET + CTRL_BRAKE] = Scalar[dtype](
            0.0
        )

        # Initialize metadata
        states[env, CRConstants.METADATA_OFFSET + META_STEP_COUNT] = Scalar[
            dtype
        ](0.0)
        states[env, CRConstants.METADATA_OFFSET + META_TOTAL_REWARD] = Scalar[
            dtype
        ](0.0)
        states[env, CRConstants.METADATA_OFFSET + META_DONE] = Scalar[dtype](
            0.0
        )
        states[env, CRConstants.METADATA_OFFSET + META_TRUNCATED] = Scalar[
            dtype
        ](0.0)
        states[env, CRConstants.METADATA_OFFSET + META_TILES_VISITED] = Scalar[
            dtype
        ](0.0)

        # Initialize observation
        states[env, CRConstants.OBS_OFFSET + 0] = start_x
        states[env, CRConstants.OBS_OFFSET + 1] = start_y
        states[env, CRConstants.OBS_OFFSET + 2] = start_angle + rand_angle
        for i in range(3, CRConstants.OBS_DIM):
            states[env, CRConstants.OBS_OFFSET + i] = Scalar[dtype](0.0)

    @always_inline
    @staticmethod
    fn _reset_env_with_track_gpu[
        BATCH: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, CRConstants.STATE_SIZE), MutAnyOrigin
        ],
        tiles: LayoutTensor[
            dtype,
            Layout.row_major(MAX_TRACK_TILES, TILE_DATA_SIZE),
            MutAnyOrigin,
        ],
        num_tiles: Int,
        env: Int,
        seed: Int,
    ):
        """Reset environment using track data to find start position."""
        # Get start position from first tile (approximate center)
        var tile0_v0x = rebind[Scalar[dtype]](tiles[0, 0])  # TILE_V0_X
        var tile0_v0y = rebind[Scalar[dtype]](tiles[0, 1])  # TILE_V0_Y
        var tile0_v3x = rebind[Scalar[dtype]](tiles[0, 6])  # TILE_V3_X
        var tile0_v3y = rebind[Scalar[dtype]](tiles[0, 7])  # TILE_V3_Y

        var start_x = (tile0_v0x + tile0_v3x) / Scalar[dtype](2.0)
        var start_y = (tile0_v0y + tile0_v3y) / Scalar[dtype](2.0)

        # Compute angle from tile direction (v0 to v1)
        var tile0_v1x = rebind[Scalar[dtype]](tiles[0, 2])  # TILE_V1_X
        var tile0_v1y = rebind[Scalar[dtype]](tiles[0, 3])  # TILE_V1_Y
        var dx = tile0_v1x - tile0_v0x
        var dy = tile0_v1y - tile0_v0y
        var track_dir = Scalar[dtype](0.0)
        if dx != Scalar[dtype](0.0) or dy != Scalar[dtype](0.0):
            # atan2 approximation for GPU
            var abs_dx = dx if dx >= Scalar[dtype](0.0) else -dx
            var abs_dy = dy if dy >= Scalar[dtype](0.0) else -dy
            if abs_dx > abs_dy:
                track_dir = dy / dx
                if dx < Scalar[dtype](0.0):
                    track_dir = track_dir + Scalar[dtype](pi)
            else:
                track_dir = Scalar[dtype](pi / 2.0) - dx / dy
                if dy < Scalar[dtype](0.0):
                    track_dir = track_dir + Scalar[dtype](pi)

        var start_angle = track_dir

        # Clear and initialize state
        for i in range(CRConstants.STATE_SIZE):
            states[env, i] = Scalar[dtype](0.0)

        # Set hull state
        states[env, CRConstants.HULL_OFFSET + HULL_X] = start_x
        states[env, CRConstants.HULL_OFFSET + HULL_Y] = start_y
        states[env, CRConstants.HULL_OFFSET + HULL_ANGLE] = start_angle

        # Initialize observation
        states[env, CRConstants.OBS_OFFSET + 0] = start_x
        states[env, CRConstants.OBS_OFFSET + 1] = start_y
        states[env, CRConstants.OBS_OFFSET + 2] = start_angle

    @always_inline
    @staticmethod
    fn _step_env_gpu[
        BATCH: Int,
    ](
        env: Int,
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, CRConstants.STATE_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH, CRConstants.ACTION_DIM), MutAnyOrigin
        ],
        tiles: LayoutTensor[
            dtype,
            Layout.row_major(MAX_TRACK_TILES, TILE_DATA_SIZE),
            MutAnyOrigin,
        ],
        num_tiles: Int,
        rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH, CRConstants.OBS_DIM), MutAnyOrigin
        ],
    ):
        """Execute one step for a single environment (GPU kernel body)."""
        # Check if already done
        var was_done = rebind[Scalar[dtype]](
            states[env, CRConstants.METADATA_OFFSET + META_DONE]
        ) > Scalar[dtype](0.5)
        if was_done:
            rewards[env] = Scalar[dtype](0.0)
            dones[env] = Scalar[dtype](1.0)
            return

        # Step 1: Copy actions to controls (clamp and remap)
        var steering = rebind[Scalar[dtype]](actions[env, 0])
        var gas_raw = rebind[Scalar[dtype]](actions[env, 1])
        var brake_raw = rebind[Scalar[dtype]](actions[env, 2])

        # Clamp steering to [-1, 1]
        if steering > Scalar[dtype](1.0):
            steering = Scalar[dtype](1.0)
        elif steering < Scalar[dtype](-1.0):
            steering = Scalar[dtype](-1.0)

        # Remap gas and brake from [-1, 1] to [0, 1]
        var gas = (gas_raw + Scalar[dtype](1.0)) * Scalar[dtype](0.5)
        var brake = (brake_raw + Scalar[dtype](1.0)) * Scalar[dtype](0.5)
        if gas > Scalar[dtype](1.0):
            gas = Scalar[dtype](1.0)
        elif gas < Scalar[dtype](0.0):
            gas = Scalar[dtype](0.0)
        if brake > Scalar[dtype](1.0):
            brake = Scalar[dtype](1.0)
        elif brake < Scalar[dtype](0.0):
            brake = Scalar[dtype](0.0)

        states[env, CRConstants.CONTROLS_OFFSET + CTRL_STEERING] = steering
        states[env, CRConstants.CONTROLS_OFFSET + CTRL_GAS] = gas
        states[env, CRConstants.CONTROLS_OFFSET + CTRL_BRAKE] = brake

        # Step 2: Run physics (using CarDynamics)
        var dt = Scalar[dtype](CRConstants.DT)
        CarDynamics.step_with_obs[
            BATCH,
            CRConstants.STATE_SIZE,
            CRConstants.OBS_OFFSET,
            CRConstants.OBS_DIM,
            CRConstants.HULL_OFFSET,
            CRConstants.WHEELS_OFFSET,
            CRConstants.CONTROLS_OFFSET,
            MAX_TRACK_TILES,
        ](env, states, tiles, num_tiles, dt)

        # Step 3: Compute reward
        var step_reward = Scalar[dtype](-0.1)  # Time penalty

        # Check tile visits for reward
        var hull_x = rebind[Scalar[dtype]](
            states[env, CRConstants.HULL_OFFSET + HULL_X]
        )
        var hull_y = rebind[Scalar[dtype]](
            states[env, CRConstants.HULL_OFFSET + HULL_Y]
        )
        var tile_idx = TileCollision.check_tile_visited[MAX_TRACK_TILES](
            hull_x, hull_y, tiles, num_tiles
        )

        # Simple tile visit reward (in full implementation, track visited tiles)
        var tiles_visited = rebind[Scalar[dtype]](
            states[env, CRConstants.METADATA_OFFSET + META_TILES_VISITED]
        )
        if tile_idx >= 0:
            # Reward for being on track (simplified - full version tracks unique visits)
            var tile_idx_f = Scalar[dtype](tile_idx)
            if tile_idx_f > tiles_visited:
                step_reward = step_reward + Scalar[dtype](1000.0) / Scalar[
                    dtype
                ](num_tiles)
                states[
                    env, CRConstants.METADATA_OFFSET + META_TILES_VISITED
                ] = tile_idx_f

        # Step 4: Check termination
        var done = Scalar[dtype](0.0)
        var playfield = Scalar[dtype](CRConstants.PLAYFIELD)

        # Off playfield check
        var abs_x = hull_x if hull_x >= Scalar[dtype](0.0) else -hull_x
        var abs_y = hull_y if hull_y >= Scalar[dtype](0.0) else -hull_y
        if abs_x > playfield or abs_y > playfield:
            done = Scalar[dtype](1.0)
            step_reward = Scalar[dtype](-100.0)

        # Lap completion check
        var num_tiles_f = Scalar[dtype](num_tiles)
        var progress = tiles_visited / num_tiles_f
        if progress >= Scalar[dtype](CRConstants.LAP_COMPLETE_PERCENT):
            done = Scalar[dtype](1.0)

        # Max steps check
        var step_count = rebind[Scalar[dtype]](
            states[env, CRConstants.METADATA_OFFSET + META_STEP_COUNT]
        )
        step_count = step_count + Scalar[dtype](1.0)
        states[env, CRConstants.METADATA_OFFSET + META_STEP_COUNT] = step_count

        if step_count >= Scalar[dtype](CRConstants.MAX_STEPS):
            done = Scalar[dtype](1.0)
            states[env, CRConstants.METADATA_OFFSET + META_TRUNCATED] = Scalar[
                dtype
            ](1.0)

        # Update metadata
        var total_reward = rebind[Scalar[dtype]](
            states[env, CRConstants.METADATA_OFFSET + META_TOTAL_REWARD]
        )
        total_reward = total_reward + step_reward
        states[
            env, CRConstants.METADATA_OFFSET + META_TOTAL_REWARD
        ] = total_reward
        states[env, CRConstants.METADATA_OFFSET + META_DONE] = done

        # Step 5: Write outputs
        rewards[env] = step_reward
        dones[env] = done

        # Copy observation to output buffer
        for i in range(CRConstants.OBS_DIM):
            obs[env, i] = states[env, CRConstants.OBS_OFFSET + i]

    @staticmethod
    fn _init_circular_track_gpu(
        ctx: DeviceContext,
        mut tiles_buf: DeviceBuffer[dtype],
    ) raises:
        """Initialize a simple circular track in GPU buffer."""
        var tiles = LayoutTensor[
            dtype,
            Layout.row_major(MAX_TRACK_TILES, TILE_DATA_SIZE),
            MutAnyOrigin,
        ](tiles_buf.unsafe_ptr())

        @always_inline
        fn init_track_wrapper(
            tiles: LayoutTensor[
                dtype,
                Layout.row_major(MAX_TRACK_TILES, TILE_DATA_SIZE),
                MutAnyOrigin,
            ],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            if tid > 0:
                return

            # Generate simple circular track (100 tiles)
            var num_tiles = 100
            var rad = Scalar[dtype](CRConstants.TRACK_RAD)
            var width = Scalar[dtype](CRConstants.TRACK_WIDTH)
            var two_pi = Scalar[dtype](2.0 * pi)

            for i in range(num_tiles):
                var alpha1 = (
                    two_pi * Scalar[dtype](i) / Scalar[dtype](num_tiles)
                )
                var alpha2 = (
                    two_pi * Scalar[dtype](i + 1) / Scalar[dtype](num_tiles)
                )

                var x1 = rad * cos(alpha1)
                var y1 = rad * sin(alpha1)
                var x2 = rad * cos(alpha2)
                var y2 = rad * sin(alpha2)

                # Inner edge
                tiles[i, 0] = x1 - width * cos(alpha1)  # v0_x
                tiles[i, 1] = y1 - width * sin(alpha1)  # v0_y
                tiles[i, 2] = x2 - width * cos(alpha2)  # v1_x
                tiles[i, 3] = y2 - width * sin(alpha2)  # v1_y

                # Outer edge
                tiles[i, 4] = x2 + width * cos(alpha2)  # v2_x
                tiles[i, 5] = y2 + width * sin(alpha2)  # v2_y
                tiles[i, 6] = x1 + width * cos(alpha1)  # v3_x
                tiles[i, 7] = y1 + width * sin(alpha1)  # v3_y

                # Friction (road)
                tiles[i, 8] = Scalar[dtype](1.0)

        ctx.enqueue_function[init_track_wrapper, init_track_wrapper](
            tiles,
            grid_dim=(1,),
            block_dim=(1,),
        )

    @staticmethod
    fn _copy_obs_gpu[
        BATCH_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states: DeviceBuffer[dtype],
        mut obs: DeviceBuffer[dtype],
    ) raises:
        """Copy observations from state buffer to output obs buffer."""
        var states_tensor = LayoutTensor[
            dtype,
            Layout.row_major(BATCH_SIZE, CRConstants.STATE_SIZE),
            MutAnyOrigin,
        ](states.unsafe_ptr())

        var obs_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ](obs.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn copy_obs_wrapper(
            states: LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, CRConstants.STATE_SIZE),
                MutAnyOrigin,
            ],
            obs: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            for i in range(OBS_DIM):
                obs[env, i] = states[env, CRConstants.OBS_OFFSET + i]

        ctx.enqueue_function[copy_obs_wrapper, copy_obs_wrapper](
            states_tensor,
            obs_tensor,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )


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
