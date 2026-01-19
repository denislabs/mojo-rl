"""
LunarLander: Native Mojo Implementation

A pure Mojo implementation of the LunarLander environment using
a custom 2D physics engine. Matches Gymnasium's LunarLander-v3.

Features:
- Both discrete (4 actions) and continuous (2D throttle) action spaces
- Wind and turbulence effects
- SDL2 rendering
- Contact callbacks for leg ground detection
"""

from math import sin, cos, sqrt, pi, tanh
from random import random_float64

from physics.vec2 import Vec2, vec2
from physics.shape import PolygonShape, CircleShape, EdgeShape
from physics.body import Body, BODY_STATIC, BODY_DYNAMIC
from physics.fixture import (
    Filter,
    CATEGORY_GROUND,
    CATEGORY_LANDER,
    CATEGORY_LEG,
    CATEGORY_PARTICLE,
)
from physics.world import World
from physics.joint import RevoluteJoint

from core import (
    State,
    Action,
    BoxDiscreteActionEnv,
    BoxContinuousActionEnv,
    BoxSpace,
    DiscreteSpace,
)
from render import (
    RendererBase,
    SDL_Color,
    SDL_Point,
    Camera,
    Vec2 as RenderVec2,
    Transform2D,
    # Colors
    space_black,
    moon_gray,
    dark_gray,
    white,
    yellow,
    red,
    lander_gray,
    contact_green,
    inactive_gray,
    flame_color,
    rgb,
    darken,
    # Shapes
    make_lander_body,
    make_leg_box,
    scale_vertices,
)
from .state import LunarLanderState
from .particle import Particle
from .action import LunarLanderAction

# ===== Constants =====

comptime VIEWPORT_W: Int = 600
comptime VIEWPORT_H: Int = 400
comptime FPS: Int = 60

# ===== Environment =====


struct LunarLanderEnv[DTYPE: DType](BoxDiscreteActionEnv, Copyable, Movable):
    """Native Mojo LunarLander environment.

    Implements BoxDiscreteActionEnv for function approximation methods:
    - Continuous 8D observation: [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]
    - Discrete 4 actions: 0=nop, 1=left, 2=main, 3=right

    Can be used with tile coding, linear function approximation, DQN, PPO, etc.
    """

    # Type aliases for trait conformance
    comptime dtype = Self.DTYPE
    comptime StateType = LunarLanderState[Self.dtype]
    comptime ActionType = LunarLanderAction

    comptime SCALE: Scalar[Self.dtype] = 30.0

    comptime MAIN_ENGINE_POWER: Scalar[Self.dtype] = 13.0
    comptime SIDE_ENGINE_POWER: Scalar[Self.dtype] = 0.6

    comptime INITIAL_RANDOM: Scalar[Self.dtype] = 1000.0

    # Lander polygon vertices (in SCALE units)
    comptime LANDER_POLY_COUNT: Int = 6

    comptime LEG_AWAY: Scalar[Self.dtype] = 20.0
    comptime LEG_DOWN: Scalar[Self.dtype] = 18.0
    comptime LEG_W: Scalar[Self.dtype] = 2.0
    comptime LEG_H: Scalar[Self.dtype] = 8.0
    comptime LEG_SPRING_TORQUE: Scalar[Self.dtype] = 40.0

    comptime SIDE_ENGINE_HEIGHT: Scalar[Self.dtype] = 14.0
    comptime SIDE_ENGINE_AWAY: Scalar[Self.dtype] = 12.0
    comptime MAIN_ENGINE_Y_LOCATION: Scalar[Self.dtype] = 4.0

    # Physics world
    var world: World[Self.dtype]

    # Body indices
    var moon_idx: Int
    var lander_idx: Int
    var left_leg_idx: Int
    var right_leg_idx: Int

    # Fixture indices for ground contact detection
    var lander_fixture_idx: Int
    var left_leg_fixture_idx: Int
    var right_leg_fixture_idx: Int
    var terrain_fixture_start: Int
    var terrain_fixture_count: Int

    # Joint indices
    var left_joint_idx: Int
    var right_joint_idx: Int

    # Particles (visual flame effects)
    var flame_particles: List[Particle[Self.dtype]]

    # Game state
    var game_over: Bool
    var prev_shaping: Scalar[Self.dtype]
    var initialized: Bool

    # Terrain
    var helipad_x1: Scalar[Self.dtype]
    var helipad_x2: Scalar[Self.dtype]
    var helipad_y: Scalar[Self.dtype]
    var terrain_x: List[Scalar[Self.dtype]]
    var terrain_y: List[Scalar[Self.dtype]]

    # Configuration
    var continuous: Bool
    var gravity: Scalar[Self.dtype]
    var enable_wind: Bool
    var wind_power: Scalar[Self.dtype]
    var turbulence_power: Scalar[Self.dtype]

    # Wind state
    var wind_idx: Int
    var torque_idx: Int

    fn __init__(
        out self,
        continuous: Bool = False,
        gravity: Scalar[Self.dtype] = -10.0,
        enable_wind: Bool = False,
        wind_power: Scalar[Self.dtype] = 15.0,
        turbulence_power: Scalar[Self.dtype] = 1.5,
    ) raises:
        """Create LunarLander environment."""
        self.world = World[Self.dtype](Vec2[Self.dtype](0.0, gravity))
        self.moon_idx = -1
        self.lander_idx = -1
        self.left_leg_idx = -1
        self.right_leg_idx = -1
        self.lander_fixture_idx = -1
        self.left_leg_fixture_idx = -1
        self.right_leg_fixture_idx = -1
        self.terrain_fixture_start = -1
        self.terrain_fixture_count = 0
        self.left_joint_idx = -1
        self.right_joint_idx = -1
        self.flame_particles = List[Particle[Self.dtype]]()
        self.game_over = False
        self.prev_shaping = 0.0
        self.initialized = False
        self.helipad_x1 = 0.0
        self.helipad_x2 = 0.0
        self.helipad_y = 0.0
        self.terrain_x = List[Scalar[Self.dtype]]()
        self.terrain_y = List[Scalar[Self.dtype]]()
        self.continuous = continuous
        self.gravity = gravity
        self.enable_wind = enable_wind
        self.wind_power = wind_power
        self.turbulence_power = turbulence_power
        self.wind_idx = 0
        self.torque_idx = 0

    fn reset(mut self) -> Self.StateType:
        """Reset the environment and return initial state.

        Implements Env trait.
        """
        return self._reset_internal()

    fn reset(mut self, seed: Int) -> LunarLanderState[Self.dtype]:
        """Reset environment to initial state with optional seed."""
        # Note: seed is currently ignored (use random.seed() externally)
        return self._reset_internal()

    fn _reset_internal(mut self) -> LunarLanderState[Self.dtype]:
        """Internal reset implementation."""
        # Recreate world
        self.world = World[Self.dtype](Vec2[Self.dtype](0.0, self.gravity))
        self.flame_particles.clear()
        self.game_over = False
        self.prev_shaping = 0.0

        var W = Scalar[Self.dtype](VIEWPORT_W) / Self.SCALE
        var H = Scalar[Self.dtype](VIEWPORT_H) / Self.SCALE

        # Generate terrain
        comptime CHUNKS: Int = 11
        self.terrain_x.clear()
        self.terrain_y.clear()

        # Generate random heights
        var heights = List[Scalar[Self.dtype]]()
        for _ in range(CHUNKS + 1):
            heights.append(Scalar[Self.dtype](random_float64()) * H / 2.0)

        # Create chunk x positions
        for i in range(CHUNKS):
            self.terrain_x.append(
                W / Scalar[Self.dtype](CHUNKS - 1) * Scalar[Self.dtype](i)
            )

        # Set helipad flat area
        self.helipad_x1 = self.terrain_x[CHUNKS // 2 - 1]
        self.helipad_x2 = self.terrain_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4.0

        heights[CHUNKS // 2 - 2] = self.helipad_y
        heights[CHUNKS // 2 - 1] = self.helipad_y
        heights[CHUNKS // 2] = self.helipad_y
        heights[CHUNKS // 2 + 1] = self.helipad_y
        heights[CHUNKS // 2 + 2] = self.helipad_y

        # Smooth terrain (centered smoothing like Gymnasium)
        for i in range(CHUNKS):
            # Use centered smoothing: (i-1, i, i+1) with clamping at boundaries
            var idx_prev = max(i - 1, 0)
            var idx_next = min(i + 1, CHUNKS)
            self.terrain_y.append(
                Scalar[Self.dtype](0.33)
                * (heights[idx_prev] + heights[i] + heights[idx_next])
            )

        # Create moon (static body for terrain)
        self.moon_idx = self.world.create_body(
            BODY_STATIC, Vec2[Self.dtype].zero()
        )
        self.terrain_fixture_start = len(self.world.fixtures)

        # Create edge fixtures for terrain
        for i in range(CHUNKS - 1):
            var p1 = Vec2[Self.dtype](self.terrain_x[i], self.terrain_y[i])
            var p2 = Vec2[Self.dtype](
                self.terrain_x[i + 1], self.terrain_y[i + 1]
            )
            var edge = EdgeShape[Self.dtype](p1, p2)
            _ = self.world.create_edge_fixture(
                self.moon_idx,
                edge^,
                friction=0.1,
                filter=Filter.ground(),
            )

        self.terrain_fixture_count = (
            len(self.world.fixtures) - self.terrain_fixture_start
        )

        # Create lander
        var initial_x = W / 2.0
        var initial_y = H
        self.lander_idx = self.world.create_body(
            BODY_DYNAMIC, Vec2[Self.dtype](initial_x, initial_y)
        )

        # Lander polygon shape (vertices from Gymnasium)
        var lander_verts = List[Vec2[Self.dtype]]()
        lander_verts.append(
            Vec2[Self.dtype](-14.0 / Self.SCALE, 17.0 / Self.SCALE)
        )
        lander_verts.append(
            Vec2[Self.dtype](-17.0 / Self.SCALE, 0.0 / Self.SCALE)
        )
        lander_verts.append(
            Vec2[Self.dtype](-17.0 / Self.SCALE, -10.0 / Self.SCALE)
        )
        lander_verts.append(
            Vec2[Self.dtype](17.0 / Self.SCALE, -10.0 / Self.SCALE)
        )
        lander_verts.append(
            Vec2[Self.dtype](17.0 / Self.SCALE, 0.0 / Self.SCALE)
        )
        lander_verts.append(
            Vec2[Self.dtype](14.0 / Self.SCALE, 17.0 / Self.SCALE)
        )

        var lander_poly = PolygonShape[Self.dtype](lander_verts^)
        self.lander_fixture_idx = self.world.create_polygon_fixture(
            self.lander_idx,
            lander_poly^,
            density=5.0,
            friction=0.1,
            restitution=0.0,
            filter=Filter.lander(),
        )

        # Apply random initial force
        var fx = (
            Scalar[Self.dtype](random_float64() * 2.0 - 1.0)
            * Self.INITIAL_RANDOM
        )
        var fy = (
            Scalar[Self.dtype](random_float64() * 2.0 - 1.0)
            * Self.INITIAL_RANDOM
        )
        self.world.bodies[self.lander_idx].apply_force_to_center(
            Vec2[Self.dtype](fx, fy)
        )

        # Create legs
        self._create_legs(initial_x, initial_y)

        # Initialize wind
        if self.enable_wind:
            self.wind_idx = Int((random_float64() * 2.0 - 1.0) * 9999.0)
            self.torque_idx = Int((random_float64() * 2.0 - 1.0) * 9999.0)

        self.initialized = True

        # Step physics once to integrate the random initial force
        # This ensures the initial observation includes the velocity from the force,
        # which matches GPU behavior and prevents the large reward spike in step 0
        self.world.step(Scalar[Self.dtype](1.0) / Scalar[Self.dtype](FPS), 6, 2)

        # Compute initial shaping so first step reward is delta, not absolute
        var init_state = self._get_state()
        self.prev_shaping = (
            Scalar[Self.dtype](-100.0)
            * sqrt(init_state.x * init_state.x + init_state.y * init_state.y)
            - Scalar[Self.dtype](100.0)
            * sqrt(
                init_state.vx * init_state.vx + init_state.vy * init_state.vy
            )
            - Scalar[Self.dtype](100.0) * abs(init_state.angle)
            + Scalar[Self.dtype](10.0) * init_state.left_leg_contact
            + Scalar[Self.dtype](10.0) * init_state.right_leg_contact
        )

        return init_state^

    fn _create_legs(
        mut self, initial_x: Scalar[Self.dtype], initial_y: Scalar[Self.dtype]
    ):
        """Create lander legs and joints."""
        for side in range(2):
            var i = -1 if side == 0 else 1

            var leg_x = (
                initial_x - Scalar[Self.dtype](i) * Self.LEG_AWAY / Self.SCALE
            )
            var leg_angle = Scalar[Self.dtype](i) * Scalar[Self.dtype](0.05)

            var leg_idx = self.world.create_body(
                BODY_DYNAMIC, Vec2[Self.dtype](leg_x, initial_y), leg_angle
            )

            # Leg is a box
            var leg_poly = PolygonShape[Self.dtype].from_box(
                Self.LEG_W / Self.SCALE, Self.LEG_H / Self.SCALE
            )
            var leg_fixture = self.world.create_polygon_fixture(
                leg_idx,
                leg_poly^,
                density=5.0,
                friction=0.1,
                restitution=0.0,
                filter=Filter.leg(),
            )

            if side == 0:
                self.left_leg_idx = leg_idx
                self.left_leg_fixture_idx = leg_fixture
            else:
                self.right_leg_idx = leg_idx
                self.right_leg_fixture_idx = leg_fixture

            # Create revolute joint
            var local_anchor_a = Vec2[Self.dtype].zero()
            var local_anchor_b = Vec2[Self.dtype](
                Scalar[Self.dtype](i) * Self.LEG_AWAY / Self.SCALE,
                Self.LEG_DOWN / Self.SCALE,
            )

            var lower_angle: Scalar[Self.dtype]
            var upper_angle: Scalar[Self.dtype]
            if i == -1:
                lower_angle = 0.9 - 0.5
                upper_angle = 0.9
            else:
                lower_angle = -0.9
                upper_angle = -0.9 + 0.5

            var joint_idx = self.world.create_revolute_joint(
                self.lander_idx,
                leg_idx,
                local_anchor_a,
                local_anchor_b,
                enable_motor=True,
                motor_speed=Scalar[Self.dtype](0.3) * Scalar[Self.dtype](i),
                max_motor_torque=Self.LEG_SPRING_TORQUE,
                enable_limit=True,
                lower_angle=lower_angle,
                upper_angle=upper_angle,
            )

            if side == 0:
                self.left_joint_idx = joint_idx
            else:
                self.right_joint_idx = joint_idx

    fn step_discrete(
        mut self, action: Int
    ) -> Tuple[LunarLanderState[Self.dtype], Scalar[Self.dtype], Bool]:
        """Step with discrete action (0=nop, 1=left, 2=main, 3=right)."""
        return self._step_internal(action, Vec2[Self.dtype].zero())

    fn step_continuous(
        mut self, action: Vec2[Self.dtype]
    ) -> Tuple[LunarLanderState[Self.dtype], Scalar[Self.dtype], Bool]:
        """Step with continuous action (main_throttle, lateral_throttle)."""
        return self._step_internal(-1, action)

    fn _step_internal(
        mut self, discrete_action: Int, continuous_action: Vec2[Self.dtype]
    ) -> Tuple[LunarLanderState[Self.dtype], Scalar[Self.dtype], Bool]:
        """Internal step function."""
        # Apply wind
        if self.enable_wind:
            var left_contact = self._is_leg_contacting(
                self.left_leg_fixture_idx
            )
            var right_contact = self._is_leg_contacting(
                self.right_leg_fixture_idx
            )

            if not (left_contact or right_contact):
                # Wind force
                var wind_mag = (
                    Scalar[Self.dtype](
                        tanh(
                            sin(0.02 * Scalar[Self.dtype](self.wind_idx))
                            + sin(pi * 0.01 * Scalar[Self.dtype](self.wind_idx))
                        )
                    )
                    * self.wind_power
                )
                self.wind_idx += 1
                self.world.bodies[self.lander_idx].apply_force_to_center(
                    Vec2[Self.dtype](wind_mag, 0.0)
                )

                # Turbulence torque
                var torque_mag = (
                    Scalar[Self.dtype](
                        tanh(
                            sin(0.02 * Scalar[Self.dtype](self.torque_idx))
                            + sin(
                                pi * 0.01 * Scalar[Self.dtype](self.torque_idx)
                            )
                        )
                    )
                    * self.turbulence_power
                )
                self.torque_idx += 1
                self.world.bodies[self.lander_idx].apply_torque(torque_mag)

        # Get lander state
        var lander = self.world.bodies[self.lander_idx].copy()
        var tip = Vec2[Self.dtype](sin(lander.angle), cos(lander.angle))
        var side = Vec2[Self.dtype](-tip.y, tip.x)

        # Random dispersion
        var dispersion_x = (
            Scalar[Self.dtype](random_float64() * 2.0 - 1.0) / Self.SCALE
        )
        var dispersion_y = (
            Scalar[Self.dtype](random_float64() * 2.0 - 1.0) / Self.SCALE
        )

        # Apply engine forces
        var m_power: Scalar[Self.dtype] = 0.0
        var s_power: Scalar[Self.dtype] = 0.0

        if self.continuous:
            # Continuous action: [main, lateral]
            var main_throttle = continuous_action.x
            var lateral_throttle = continuous_action.y

            if main_throttle > Scalar[Self.dtype](0.0):
                m_power = (
                    clamp(
                        main_throttle,
                        Scalar[Self.dtype](0.0),
                        Scalar[Self.dtype](1.0),
                    )
                    + Scalar[Self.dtype](1.0)
                ) * Scalar[Self.dtype](0.5)

            if abs(lateral_throttle) > Scalar[Self.dtype](0.5):
                var direction = sign(lateral_throttle)
                s_power = clamp(
                    abs(lateral_throttle),
                    Scalar[Self.dtype](0.5),
                    Scalar[Self.dtype](1.0),
                )
                self._apply_side_engine(
                    lander,
                    tip,
                    side,
                    dispersion_x,
                    dispersion_y,
                    direction,
                    s_power,
                )
        else:
            # Discrete action
            if discrete_action == 2:  # Main engine
                m_power = 1.0
            elif discrete_action == 1:  # Left engine
                s_power = 1.0
                self._apply_side_engine(
                    lander,
                    tip,
                    side,
                    dispersion_x,
                    dispersion_y,
                    Scalar[Self.dtype](-1.0),
                    s_power,
                )
            elif discrete_action == 3:  # Right engine
                s_power = 1.0
                self._apply_side_engine(
                    lander,
                    tip,
                    side,
                    dispersion_x,
                    dispersion_y,
                    Scalar[Self.dtype](1.0),
                    s_power,
                )

        # Apply main engine
        if m_power > Scalar[Self.dtype](0.0):
            var ox = (
                tip.x
                * (
                    Self.MAIN_ENGINE_Y_LOCATION / Self.SCALE
                    + Scalar[Self.dtype](2.0) * dispersion_x
                )
                + side.x * dispersion_y
            )
            var oy = (
                -tip.y
                * (
                    Self.MAIN_ENGINE_Y_LOCATION / Self.SCALE
                    + Scalar[Self.dtype](2.0) * dispersion_x
                )
                - side.y * dispersion_y
            )
            var impulse_pos = Vec2[Self.dtype](
                lander.position.x + ox, lander.position.y + oy
            )
            var impulse = Vec2[Self.dtype](
                -ox * Self.MAIN_ENGINE_POWER * m_power,
                -oy * Self.MAIN_ENGINE_POWER * m_power,
            )
            self.world.bodies[self.lander_idx].apply_linear_impulse(
                impulse, impulse_pos
            )

            # Spawn main engine flame particles
            self._spawn_main_engine_particles(lander.position, tip, m_power)

        # Spawn side engine particles if firing
        if s_power > Scalar[Self.dtype](0.0):
            var direction = Scalar[Self.dtype](
                -1.0
            ) if discrete_action == 1 else Scalar[Self.dtype](1.0)
            if self.continuous:
                direction = sign(continuous_action.y)
            self._spawn_side_engine_particles(
                lander.position, tip, side, direction, s_power
            )

        # Update existing particles
        self._update_particles()

        # Step physics
        self.world.step(Scalar[Self.dtype](1.0) / Scalar[Self.dtype](FPS), 6, 2)

        # Check leg contacts (used internally for state, variables checked for debug)
        _ = self._is_leg_contacting(self.left_leg_fixture_idx)
        _ = self._is_leg_contacting(self.right_leg_fixture_idx)

        # Check lander contact (crash)
        var lander_contact = self._is_lander_contacting()
        if lander_contact:
            self.game_over = True

        # Get state
        var state = self._get_state()

        # Compute reward
        var reward = self._compute_reward(state, m_power, s_power)

        # Check termination
        var terminated = False
        if self.game_over or abs(state.x) >= Scalar[Self.dtype](1.0):
            terminated = True
            reward = -100.0
        if not self.world.bodies[self.lander_idx].awake:
            terminated = True
            reward = 100.0

        return (state^, reward, terminated)

    fn _apply_side_engine(
        mut self,
        lander: Body[Self.dtype],
        tip: Vec2[Self.dtype],
        side: Vec2[Self.dtype],
        dispersion_x: Scalar[Self.dtype],
        dispersion_y: Scalar[Self.dtype],
        direction: Scalar[Self.dtype],
        s_power: Scalar[Self.dtype],
    ):
        """Apply side engine force."""
        var ox = tip.x * dispersion_x + side.x * (
            Scalar[Self.dtype](3.0) * dispersion_y
            + direction * Self.SIDE_ENGINE_AWAY / Self.SCALE
        )
        var oy = -tip.y * dispersion_x - side.y * (
            Scalar[Self.dtype](3.0) * dispersion_y
            + direction * Self.SIDE_ENGINE_AWAY / Self.SCALE
        )
        var impulse_pos = Vec2[Self.dtype](
            lander.position.x
            + ox
            - tip.x * Scalar[Self.dtype](17.0) / Self.SCALE,
            lander.position.y
            + oy
            + tip.y * Self.SIDE_ENGINE_HEIGHT / Self.SCALE,
        )
        var impulse = Vec2[Self.dtype](
            -ox * Self.SIDE_ENGINE_POWER * s_power,
            -oy * Self.SIDE_ENGINE_POWER * s_power,
        )
        self.world.bodies[self.lander_idx].apply_linear_impulse(
            impulse, impulse_pos
        )

    fn _spawn_main_engine_particles(
        mut self,
        pos: Vec2[Self.dtype],
        tip: Vec2[Self.dtype],
        power: Scalar[Self.dtype],
    ):
        """Spawn flame particles from main engine."""
        # Spawn 2-4 particles per frame when engine is on
        var num_particles = 2 + Int(random_float64() * 3.0)
        for _ in range(num_particles):
            # Position below the lander (opposite of tip direction)
            var offset_x = Scalar[Self.dtype](random_float64() - 0.5) * Scalar[
                Self.dtype
            ](0.3)
            var px = pos.x - tip.x * Scalar[Self.dtype](0.5) + offset_x
            var py = pos.y - tip.y * Scalar[Self.dtype](0.5)  # Below lander

            # Velocity DOWNWARD (opposite of thrust direction = -tip)
            var spread = Scalar[Self.dtype](random_float64() - 0.5) * Scalar[
                Self.dtype
            ](2.0)
            var vx = -tip.x * Scalar[Self.dtype](3.0) * power + spread
            var vy = -tip.y * Scalar[Self.dtype](3.0) * power + Scalar[
                Self.dtype
            ](
                random_float64() - 0.5
            )  # Fixed: -tip.y

            # Short lifetime
            var ttl = Scalar[Self.dtype](0.1 + random_float64() * 0.2)

            self.flame_particles.append(
                Particle[Self.dtype](px, py, vx, vy, ttl)
            )

    fn _spawn_side_engine_particles(
        mut self,
        pos: Vec2[Self.dtype],
        tip: Vec2[Self.dtype],
        side: Vec2[Self.dtype],
        direction: Scalar[Self.dtype],
        power: Scalar[Self.dtype],
    ):
        """Spawn flame particles from side engine.

        direction: -1 for left engine, +1 for right engine
        Exhaust goes OUTWARD from the engine (opposite to thrust on lander)
        """
        # Spawn 1-2 particles per frame when engine is on
        var num_particles = 1 + Int(random_float64() * 2.0)
        for _ in range(num_particles):
            # Position at side of lander where engine is
            var px = pos.x - side.x * direction * Scalar[Self.dtype](0.6)
            var py = pos.y - side.y * direction * Scalar[Self.dtype](0.6)

            # Velocity: exhaust goes outward from the engine
            # Left engine (dir=-1) pushes lander right, so exhaust goes LEFT (-side * -(-1) = -side)
            # Right engine (dir=+1) pushes lander left, so exhaust goes RIGHT (-side * -(+1) = +side... wait)
            # Actually: left engine exhaust goes left (in side direction), right engine exhaust goes right (in -side direction)
            # So velocity = -side * direction
            var vx = -side.x * direction * Scalar[Self.dtype](
                2.0
            ) * power + Scalar[Self.dtype](random_float64() - 0.5)
            var vy = -side.y * direction * Scalar[Self.dtype](
                2.0
            ) * power + Scalar[Self.dtype](random_float64() - 0.5)

            # Short lifetime
            var ttl = Scalar[Self.dtype](0.08 + random_float64() * 0.15)

            self.flame_particles.append(
                Particle[Self.dtype](px, py, vx, vy, ttl)
            )

    fn _update_particles(mut self):
        """Update particle positions and remove dead particles."""
        var dt = Scalar[Self.dtype](1.0) / Scalar[Self.dtype](FPS)
        var new_particles = List[Particle[Self.dtype]]()

        for i in range(len(self.flame_particles)):
            var p = self.flame_particles[i]
            # Update position
            var new_x = p.x + p.vx * dt
            var new_y = p.y + p.vy * dt
            # Apply gravity to particles
            var new_vy = p.vy + self.gravity * dt * Scalar[Self.dtype](0.3)
            var new_ttl = p.ttl - dt

            if new_ttl > Scalar[Self.dtype](0.0):
                new_particles.append(
                    Particle[Self.dtype](new_x, new_y, p.vx, new_vy, new_ttl)
                )

        self.flame_particles = new_particles^

    fn _get_state(self) -> LunarLanderState[Self.dtype]:
        """Get current observation state."""
        var lander = self.world.bodies[self.lander_idx].copy()
        var pos = lander.position
        var vel = lander.linear_velocity

        var W = Scalar[Self.dtype](VIEWPORT_W) / Self.SCALE
        var H = Scalar[Self.dtype](VIEWPORT_H) / Self.SCALE

        var state = LunarLanderState[Self.dtype]()
        state.x = (pos.x - W / Scalar[Self.dtype](2.0)) / (
            W / Scalar[Self.dtype](2.0)
        )
        state.y = (pos.y - (self.helipad_y + Self.LEG_DOWN / Self.SCALE)) / (
            H / Scalar[Self.dtype](2.0)
        )
        state.vx = (
            vel.x * (W / Scalar[Self.dtype](2.0)) / Scalar[Self.dtype](FPS)
        )
        state.vy = (
            vel.y * (H / Scalar[Self.dtype](2.0)) / Scalar[Self.dtype](FPS)
        )
        state.angle = lander.angle
        state.angular_velocity = (
            Scalar[Self.dtype](20.0)
            * lander.angular_velocity
            / Scalar[Self.dtype](FPS)
        )
        state.left_leg_contact = Scalar[Self.dtype](
            1.0
        ) if self._is_leg_contacting(self.left_leg_fixture_idx) else Scalar[
            Self.dtype
        ](
            0.0
        )
        state.right_leg_contact = Scalar[Self.dtype](
            1.0
        ) if self._is_leg_contacting(self.right_leg_fixture_idx) else Scalar[
            Self.dtype
        ](
            0.0
        )

        return state^

    fn _compute_reward(
        mut self,
        state: LunarLanderState[Self.dtype],
        m_power: Scalar[Self.dtype],
        s_power: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype]:
        """Compute reward based on state."""
        var shaping = (
            Scalar[Self.dtype](-100.0)
            * sqrt(state.x * state.x + state.y * state.y)
            - Scalar[Self.dtype](100.0)
            * sqrt(state.vx * state.vx + state.vy * state.vy)
            - Scalar[Self.dtype](100.0) * abs(state.angle)
            + Scalar[Self.dtype](10.0) * state.left_leg_contact
            + Scalar[Self.dtype](10.0) * state.right_leg_contact
        )

        var reward: Scalar[Self.dtype] = 0.0
        if self.initialized:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power * Scalar[Self.dtype](0.30)
        reward -= s_power * Scalar[Self.dtype](0.03)

        return reward

    fn _is_leg_contacting(self, leg_fixture_idx: Int) -> Bool:
        """Check if leg is in contact with ground."""
        var contacts = self.world.get_fixture_contacts(leg_fixture_idx)
        for i in range(len(contacts)):
            var fix_idx = contacts[i]
            # Check if contacting terrain
            if (
                fix_idx >= self.terrain_fixture_start
                and fix_idx
                < self.terrain_fixture_start + self.terrain_fixture_count
            ):
                return True
        return False

    fn _is_lander_contacting(self) -> Bool:
        """Check if lander body is in contact with ground."""
        var contacts = self.world.get_fixture_contacts(self.lander_fixture_idx)
        for i in range(len(contacts)):
            var fix_idx = contacts[i]
            if (
                fix_idx >= self.terrain_fixture_start
                and fix_idx
                < self.terrain_fixture_start + self.terrain_fixture_count
            ):
                return True
        return False

    fn render(mut self, mut renderer: RendererBase):
        """Render the environment using the provided renderer.

        The renderer should be initialized before calling this method.
        Call renderer.init_display() before first use if needed.
        """

        # Begin frame with space background
        if not renderer.begin_frame_with_color(space_black()):
            return

        # Create camera - centered at viewport center, with physics scale
        # Camera Y at helipad level + some offset to see terrain and sky
        var W = Float64(VIEWPORT_W) / Float64(Self.SCALE)
        var H = Float64(VIEWPORT_H) / Float64(Self.SCALE)
        var camera = Camera(
            W / 2.0,  # Center X in world units
            H / 2.0,  # Center Y in world units
            Float64(Self.SCALE),  # Zoom = physics scale
            VIEWPORT_W,
            VIEWPORT_H,
            flip_y=True,  # Y increases upward in physics
        )

        # Draw terrain (filled)
        self._draw_terrain(camera, renderer)

        # Draw helipad
        self._draw_helipad(camera, renderer)

        # Draw helipad flags
        self._draw_flags(camera, renderer)

        # Draw flame particles (behind lander)
        self._draw_particles(camera, renderer)

        # Draw legs (before lander so lander draws on top)
        self._draw_legs(camera, renderer)

        # Draw lander
        self._draw_lander(camera, renderer)

        renderer.flip()

    fn _draw_terrain(self, camera: Camera, mut renderer: RendererBase):
        """Draw terrain as filled polygons using world coordinates."""
        var terrain_color = moon_gray()
        var terrain_dark = dark_gray()

        # Draw each terrain segment as a filled quad (from terrain line to bottom)
        for i in range(len(self.terrain_x) - 1):
            # Create polygon vertices in world coordinates
            var vertices = List[RenderVec2]()
            vertices.append(
                RenderVec2(
                    Float64(self.terrain_x[i]), Float64(self.terrain_y[i])
                )
            )
            vertices.append(
                RenderVec2(
                    Float64(self.terrain_x[i + 1]),
                    Float64(self.terrain_y[i + 1]),
                )
            )
            vertices.append(
                RenderVec2(Float64(self.terrain_x[i + 1]), 0.0)
            )  # Bottom
            vertices.append(RenderVec2(Float64(self.terrain_x[i]), 0.0))

            renderer.draw_polygon_world(
                vertices, camera, terrain_color, filled=True
            )

            # Draw terrain outline for contrast
            renderer.draw_line_world(
                RenderVec2(
                    Float64(self.terrain_x[i]), Float64(self.terrain_y[i])
                ),
                RenderVec2(
                    Float64(self.terrain_x[i + 1]),
                    Float64(self.terrain_y[i + 1]),
                ),
                camera,
                terrain_dark,
                2,
            )

    fn _draw_helipad(self, camera: Camera, mut renderer: RendererBase):
        """Draw the helipad landing zone using world coordinates."""
        var helipad_color = darken(moon_gray(), 0.8)

        # Helipad is a thick horizontal bar (in world units)
        var bar_height = 4.0 / Float64(Self.SCALE)  # 4 pixels in world units
        renderer.draw_rect_world(
            RenderVec2(
                Float64(self.helipad_x1 + self.helipad_x2) / 2.0,
                Float64(self.helipad_y) + bar_height / 2.0,
            ),
            Float64(self.helipad_x2 - self.helipad_x1),
            bar_height,
            camera,
            helipad_color,
            centered=True,
        )

    fn _draw_flags(self, camera: Camera, mut renderer: RendererBase):
        """Draw helipad flags with poles using world coordinates."""
        var white_color = white()
        var yellow_color = yellow()
        var red_color = red()

        # Flag dimensions in world units
        var pole_height = 50.0 / Float64(Self.SCALE)
        var flag_width = 25.0 / Float64(Self.SCALE)
        var flag_height = 20.0 / Float64(Self.SCALE)

        for flag_idx in range(2):
            var x_pos = Float64(self.helipad_x1) if flag_idx == 0 else Float64(
                self.helipad_x2
            )
            var ground_y = Float64(self.helipad_y)
            var pole_top_y = ground_y + pole_height

            # Flag pole (white vertical line)
            renderer.draw_line_world(
                RenderVec2(x_pos, ground_y),
                RenderVec2(x_pos, pole_top_y),
                camera,
                white_color,
                2,
            )

            # Flag as a filled triangle
            var flag_color = yellow_color if flag_idx == 0 else red_color
            var flag_verts = List[RenderVec2]()
            flag_verts.append(RenderVec2(x_pos, pole_top_y))
            flag_verts.append(
                RenderVec2(x_pos + flag_width, pole_top_y - flag_height / 2.0)
            )
            flag_verts.append(RenderVec2(x_pos, pole_top_y - flag_height))
            renderer.draw_polygon_world(
                flag_verts, camera, flag_color, filled=True
            )

    fn _draw_lander(self, camera: Camera, mut renderer: RendererBase):
        """Draw lander body as filled polygon using Transform2D."""
        if self.lander_idx < 0:
            return

        var lander = self.world.bodies[self.lander_idx].copy()
        var pos = lander.position
        var angle = lander.angle

        # Use shape factory for lander body, scale from pixels to world units
        var lander_verts_raw = make_lander_body()
        var lander_verts = scale_vertices(
            lander_verts_raw^, 1.0 / Float64(Self.SCALE)
        )

        # Create transform for lander position and rotation
        var transform = Transform2D(
            Float64(pos.x), Float64(pos.y), Float64(angle)
        )

        # Draw filled lander body (grayish-white like the original)
        var lander_fill = rgb(230, 230, 230)
        var lander_outline = rgb(100, 100, 100)
        renderer.draw_transformed_polygon(
            lander_verts, transform, camera, lander_fill, filled=True
        )
        renderer.draw_transformed_polygon(
            lander_verts, transform, camera, lander_outline, filled=False
        )

    fn _draw_legs(self, camera: Camera, mut renderer: RendererBase):
        """Draw lander legs as filled polygons using Transform2D."""
        # Check leg ground contact for color
        var left_contact = self._is_leg_contacting(self.left_leg_fixture_idx)
        var right_contact = self._is_leg_contacting(self.right_leg_fixture_idx)

        for leg_idx in range(2):
            var idx = self.left_leg_idx if leg_idx == 0 else self.right_leg_idx
            if idx < 0:
                continue

            var leg = self.world.bodies[idx].copy()
            var pos = leg.position
            var angle = leg.angle

            # Color changes when leg touches ground (green = contact)
            var is_touching = left_contact if leg_idx == 0 else right_contact
            var leg_fill = contact_green() if is_touching else inactive_gray()
            var leg_outline = darken(leg_fill, 0.6)

            # Leg box vertices using shape factory (in world units)
            var leg_verts = make_leg_box(
                Float64(Self.LEG_W) * 2.0 / Float64(Self.SCALE),
                Float64(Self.LEG_H) * 2.0 / Float64(Self.SCALE),
            )

            # Create transform for leg position and rotation
            var transform = Transform2D(
                Float64(pos.x), Float64(pos.y), Float64(angle)
            )

            # Draw filled leg
            renderer.draw_transformed_polygon(
                leg_verts, transform, camera, leg_fill, filled=True
            )
            renderer.draw_transformed_polygon(
                leg_verts, transform, camera, leg_outline, filled=False
            )

    fn _draw_particles(self, camera: Camera, mut renderer: RendererBase):
        """Draw flame particles using world coordinates and flame_color."""
        for i in range(len(self.flame_particles)):
            var p = self.flame_particles[i]

            # Color based on remaining lifetime using flame_color utility
            var life_ratio = Float64(p.ttl) / 0.3  # Normalize to 0-1
            var particle_color = flame_color(life_ratio)

            # Radius in world units (2-4 pixels converted)
            var radius_world = (2.0 + life_ratio * 2.0) / Float64(Self.SCALE)

            # Draw using Camera world coordinates
            renderer.draw_circle_world(
                RenderVec2(Float64(p.x), Float64(p.y)),
                radius_world,
                camera,
                particle_color,
                filled=True,
            )

    fn close(self):
        """Close the environment.

        Note: Rendering is now decoupled. Call renderer.close() separately
        when using a RendererBase for visualization.
        """
        pass

    # ===== Trait Methods (BoxDiscreteActionEnv) =====

    fn step(
        mut self, action: LunarLanderAction
    ) -> Tuple[LunarLanderState[Self.dtype], Scalar[Self.dtype], Bool]:
        """Take an action and return (next_state, reward, done).

        Implements Env trait.
        """
        return self.step_discrete(action.action_idx)

    fn get_state(self) -> LunarLanderState[Self.dtype]:
        """Return current state representation.

        Implements Env trait.
        """
        return self._get_state()

    fn get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return current continuous observation as a flexible list.

        Implements ContinuousStateEnv trait.
        """
        return self._get_state().to_list()

    fn reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset environment and return initial continuous observation.

        Implements ContinuousStateEnv trait.
        """
        var state = self.reset()
        return state.to_list()

    fn obs_dim(self) -> Int:
        """Return the dimension of the observation vector.

        Implements ContinuousStateEnv trait.
        LunarLander has 8D observations: [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]
        """
        return 8

    fn action_from_index(self, action_idx: Int) -> Self.ActionType:
        """Create an action from an integer index.

        Implements DiscreteActionEnv trait.
        """
        return LunarLanderAction(action_idx=action_idx)

    fn num_actions(self) -> Int:
        """Return the number of discrete actions available.

        Implements DiscreteActionEnv trait.
        LunarLander has 4 actions: 0=nop, 1=left, 2=main, 3=right
        """
        return 4

    fn step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Take discrete action and return (continuous_obs, reward, done).

        Implements BoxDiscreteActionEnv trait.
        Convenience method for function approximation algorithms.
        """
        var result = self.step_discrete(action)
        var obs = result[0].to_list()
        return (obs^, result[1], result[2])


# ===== Helper functions =====


fn clamp[
    dtype: DType
](x: Scalar[dtype], low: Scalar[dtype], high: Scalar[dtype]) -> Scalar[dtype]:
    """Clamp value to range."""
    if x < low:
        return low
    if x > high:
        return high
    return x


fn abs[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Absolute value."""
    return x if x >= Scalar[dtype](0.0) else -x


fn sign[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Sign of value."""
    if x > Scalar[dtype](0.0):
        return Scalar[dtype](1.0)
    elif x < Scalar[dtype](0.0):
        return Scalar[dtype](-1.0)
    return Scalar[dtype](0.0)


fn min(a: Int, b: Int) -> Int:
    """Minimum of two integers."""
    return a if a < b else b


fn max(a: Int, b: Int) -> Int:
    """Maximum of two integers."""
    return a if a > b else b
