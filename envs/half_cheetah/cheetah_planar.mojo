"""Planar HalfCheetah Environment.

A 2D planar version of the MuJoCo HalfCheetah environment.
The cheetah consists of 7 bodies (torso, 2x thigh, 2x shin, 2x foot) connected by 6 hinge joints.

Key difference from Walker2d: HalfCheetah has no healthy reward and does NOT terminate on falling.
It only rewards forward velocity and penalizes control effort.

Observation space (17D):
    [0]: z position of torso (height)
    [1]: angle of torso
    [2]: angle of back thigh
    [3]: angle of back shin
    [4]: angle of back foot
    [5]: angle of front thigh
    [6]: angle of front shin
    [7]: angle of front foot
    [8]: velocity of x
    [9]: velocity of z (vertical)
    [10]: angular velocity of torso
    [11]: angular velocity of back thigh
    [12]: angular velocity of back shin
    [13]: angular velocity of back foot
    [14]: angular velocity of front thigh
    [15]: angular velocity of front shin
    [16]: angular velocity of front foot

Action space (6D):
    [0]: torque applied at back thigh (hip)
    [1]: torque applied at back shin (knee)
    [2]: torque applied at back foot (ankle)
    [3]: torque applied at front thigh (hip)
    [4]: torque applied at front shin (knee)
    [5]: torque applied at front foot (ankle)

Reward:
    reward = forward_velocity - ctrl_cost
    (No healthy reward, no termination on falling)
"""

from math import sqrt, cos, sin, pi
from random import random_float64
from layout import Layout, LayoutTensor
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from random.philox import Random as PhiloxRandom

from core import (
    GPUContinuousEnv,
    BoxContinuousActionEnv,
)

from render import (
    RendererBase,
    SDL_Color,
    Vec2,
    Camera,
    # Colors
    sky_blue,
    black,
    light_gray,
    dark_gray,
    rgb,
    ground_brown,
    hull_purple,
    orange,
)

from physics2d import (
    dtype,
    TPB,
    BODY_STATE_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    IDX_INV_MASS,
    IDX_INV_INERTIA,
    IDX_SHAPE,
    JOINT_DATA_SIZE,
    JOINT_TYPE,
    JOINT_BODY_A,
    JOINT_BODY_B,
    JOINT_ANCHOR_AX,
    JOINT_ANCHOR_AY,
    JOINT_ANCHOR_BX,
    JOINT_ANCHOR_BY,
    JOINT_REF_ANGLE,
    JOINT_LOWER_LIMIT,
    JOINT_UPPER_LIMIT,
    JOINT_MAX_MOTOR_TORQUE,
    JOINT_MOTOR_SPEED,
    JOINT_FLAGS,
    JOINT_REVOLUTE,
    JOINT_FLAG_LIMIT_ENABLED,
    JOINT_FLAG_MOTOR_ENABLED,
    HalfCheetahLayout,
)

from .state import HalfCheetahPlanarState
from .action import HalfCheetahPlanarAction
from .constants import HCConstants


# =============================================================================
# HalfCheetahPlanar Environment with Trait Conformance
# =============================================================================


struct HalfCheetahPlanar[DTYPE: DType = DType.float64](
    BoxContinuousActionEnv,
    Copyable,
    GPUContinuousEnv,
    Movable,
):
    """Planar HalfCheetah locomotion environment.

    Unlike Hopper and Walker2d, HalfCheetah does NOT terminate on falling.
    Episodes only end after MAX_STEPS.

    Implements BoxContinuousActionEnv and GPUContinuousEnv traits for
    compatibility with deep RL algorithms and GPU-accelerated training.
    """

    # =========================================================================
    # Trait Type Compile-Time Constants
    # =========================================================================

    # Required for GPUContinuousEnv trait
    comptime STATE_SIZE: Int = HCConstants.STATE_SIZE_VAL
    comptime OBS_DIM: Int = HCConstants.OBS_DIM_VAL
    comptime ACTION_DIM: Int = HCConstants.ACTION_DIM_VAL

    # Required for BoxContinuousActionEnv trait
    comptime dtype = Self.DTYPE
    comptime StateType = HalfCheetahPlanarState[Self.dtype]
    comptime ActionType = HalfCheetahPlanarAction[Self.dtype]

    # =========================================================================
    # Instance Variables
    # =========================================================================

    var bodies: List[Float64]
    var joint_angles: List[Float64]
    var joint_velocities: List[Float64]
    var step_count: Int
    var done: Bool
    var total_reward: Float64
    var prev_x: Float64

    # Cached state for trait interface
    var cached_state: HalfCheetahPlanarState[Self.dtype]

    # =========================================================================
    # Initialization
    # =========================================================================

    fn __init__(out self, seed: UInt64 = 42):
        """Initialize the HalfCheetah environment."""
        self.bodies = List[Float64]()
        self.joint_angles = List[Float64]()
        self.joint_velocities = List[Float64]()
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0
        self.prev_x = 0.0
        self.cached_state = HalfCheetahPlanarState[Self.dtype]()

        for _ in range(HCConstants.NUM_BODIES * BODY_STATE_SIZE):
            self.bodies.append(0.0)

        for _ in range(HCConstants.MAX_JOINTS):
            self.joint_angles.append(0.0)
            self.joint_velocities.append(0.0)

        self._reset_state()

    fn __copyinit__(out self, read other: Self):
        """Copy constructor."""
        self.bodies = List[Float64](other.bodies)
        self.joint_angles = List[Float64](other.joint_angles)
        self.joint_velocities = List[Float64](other.joint_velocities)
        self.step_count = other.step_count
        self.done = other.done
        self.total_reward = other.total_reward
        self.prev_x = other.prev_x
        self.cached_state = other.cached_state

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.bodies = other.bodies^
        self.joint_angles = other.joint_angles^
        self.joint_velocities = other.joint_velocities^
        self.step_count = other.step_count
        self.done = other.done
        self.total_reward = other.total_reward
        self.prev_x = other.prev_x
        self.cached_state = other.cached_state

    # =========================================================================
    # Internal State Management
    # =========================================================================

    fn _reset_state(mut self):
        """Reset to initial position."""
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0

        var torso_y = HCConstants.INIT_HEIGHT

        # Torso (horizontal)
        self._init_body(
            HCConstants.BODY_TORSO,
            0.0,
            torso_y,
            0.0,
            0.0,
            0.0,
            0.0,
            HCConstants.TORSO_MASS,
        )

        # Back leg (behind torso)
        var back_hip_x = -HCConstants.TORSO_LENGTH / 2
        var back_hip_y = torso_y

        self._init_body(
            HCConstants.BODY_BTHIGH,
            back_hip_x,
            back_hip_y - HCConstants.BTHIGH_LENGTH / 2,
            0.0,
            0.0,
            0.0,
            0.0,
            HCConstants.BTHIGH_MASS,
        )

        self._init_body(
            HCConstants.BODY_BSHIN,
            back_hip_x,
            back_hip_y
            - HCConstants.BTHIGH_LENGTH
            - HCConstants.BSHIN_LENGTH / 2,
            0.0,
            0.0,
            0.0,
            0.0,
            HCConstants.BSHIN_MASS,
        )

        self._init_body(
            HCConstants.BODY_BFOOT,
            back_hip_x,
            back_hip_y
            - HCConstants.BTHIGH_LENGTH
            - HCConstants.BSHIN_LENGTH
            - HCConstants.BFOOT_LENGTH / 2,
            0.0,
            0.0,
            0.0,
            0.0,
            HCConstants.BFOOT_MASS,
        )

        # Front leg (in front of torso)
        var front_hip_x = HCConstants.TORSO_LENGTH / 2
        var front_hip_y = torso_y

        self._init_body(
            HCConstants.BODY_FTHIGH,
            front_hip_x,
            front_hip_y - HCConstants.FTHIGH_LENGTH / 2,
            0.0,
            0.0,
            0.0,
            0.0,
            HCConstants.FTHIGH_MASS,
        )

        self._init_body(
            HCConstants.BODY_FSHIN,
            front_hip_x,
            front_hip_y
            - HCConstants.FTHIGH_LENGTH
            - HCConstants.FSHIN_LENGTH / 2,
            0.0,
            0.0,
            0.0,
            0.0,
            HCConstants.FSHIN_MASS,
        )

        self._init_body(
            HCConstants.BODY_FFOOT,
            front_hip_x,
            front_hip_y
            - HCConstants.FTHIGH_LENGTH
            - HCConstants.FSHIN_LENGTH
            - HCConstants.FFOOT_LENGTH / 2,
            0.0,
            0.0,
            0.0,
            0.0,
            HCConstants.FFOOT_MASS,
        )

        for i in range(HCConstants.MAX_JOINTS):
            self.joint_angles[i] = 0.0
            self.joint_velocities[i] = 0.0

        self.prev_x = 0.0
        self._update_cached_state()

    fn _init_body(
        mut self,
        body_idx: Int,
        x: Float64,
        y: Float64,
        angle: Float64,
        vx: Float64,
        vy: Float64,
        omega: Float64,
        mass: Float64,
    ):
        """Initialize a body's state."""
        var base = body_idx * BODY_STATE_SIZE
        self.bodies[base + IDX_X] = x
        self.bodies[base + IDX_Y] = y
        self.bodies[base + IDX_ANGLE] = angle
        self.bodies[base + IDX_VX] = vx
        self.bodies[base + IDX_VY] = vy
        self.bodies[base + IDX_OMEGA] = omega
        self.bodies[base + IDX_INV_MASS] = 1.0 / mass if mass > 0.0 else 0.0
        self.bodies[base + IDX_INV_INERTIA] = (
            1.0 / (mass * 0.1) if mass > 0.0 else 0.0
        )

    fn _update_cached_state(mut self):
        """Update cached observation state."""
        var torso_base = HCConstants.BODY_TORSO * BODY_STATE_SIZE

        self.cached_state.torso_z = Scalar[Self.dtype](
            self.bodies[torso_base + IDX_Y]
        )
        self.cached_state.torso_angle = Scalar[Self.dtype](
            self.bodies[torso_base + IDX_ANGLE]
        )
        self.cached_state.vel_x = Scalar[Self.dtype](
            self.bodies[torso_base + IDX_VX]
        )
        self.cached_state.vel_z = Scalar[Self.dtype](
            self.bodies[torso_base + IDX_VY]
        )
        self.cached_state.torso_omega = Scalar[Self.dtype](
            self.bodies[torso_base + IDX_OMEGA]
        )

        # Joint angles
        self.cached_state.bthigh_angle = Scalar[Self.dtype](
            self.joint_angles[HCConstants.JOINT_BTHIGH]
        )
        self.cached_state.bshin_angle = Scalar[Self.dtype](
            self.joint_angles[HCConstants.JOINT_BSHIN]
        )
        self.cached_state.bfoot_angle = Scalar[Self.dtype](
            self.joint_angles[HCConstants.JOINT_BFOOT]
        )
        self.cached_state.fthigh_angle = Scalar[Self.dtype](
            self.joint_angles[HCConstants.JOINT_FTHIGH]
        )
        self.cached_state.fshin_angle = Scalar[Self.dtype](
            self.joint_angles[HCConstants.JOINT_FSHIN]
        )
        self.cached_state.ffoot_angle = Scalar[Self.dtype](
            self.joint_angles[HCConstants.JOINT_FFOOT]
        )

        # Joint velocities
        self.cached_state.bthigh_omega = Scalar[Self.dtype](
            self.joint_velocities[HCConstants.JOINT_BTHIGH]
        )
        self.cached_state.bshin_omega = Scalar[Self.dtype](
            self.joint_velocities[HCConstants.JOINT_BSHIN]
        )
        self.cached_state.bfoot_omega = Scalar[Self.dtype](
            self.joint_velocities[HCConstants.JOINT_BFOOT]
        )
        self.cached_state.fthigh_omega = Scalar[Self.dtype](
            self.joint_velocities[HCConstants.JOINT_FTHIGH]
        )
        self.cached_state.fshin_omega = Scalar[Self.dtype](
            self.joint_velocities[HCConstants.JOINT_FSHIN]
        )
        self.cached_state.ffoot_omega = Scalar[Self.dtype](
            self.joint_velocities[HCConstants.JOINT_FFOOT]
        )

    # =========================================================================
    # BoxContinuousActionEnv Trait Methods
    # =========================================================================

    fn reset(mut self) -> Self.StateType:
        """Reset environment and return initial state."""
        self._reset_state()

        for i in range(HCConstants.MAX_JOINTS):
            self.joint_angles[i] = (random_float64() - 0.5) * 0.01
            self.joint_velocities[i] = (random_float64() - 0.5) * 0.01

        self._update_cached_state()
        return self.get_state()

    fn step(
        mut self, action: Self.ActionType
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        """Take an action and return (next_state, reward, done)."""
        var result = self._step_internal(action)
        return (self.get_state(), result[0], result[1])

    fn _step_internal(
        mut self, action: HalfCheetahPlanarAction[Self.dtype]
    ) -> Tuple[Scalar[Self.dtype], Bool]:
        """Internal step implementation."""
        var torques = List[Float64]()
        torques.append(
            Float64(
                max(
                    min(action.bthigh, Scalar[Self.dtype](1.0)),
                    Scalar[Self.dtype](-1.0),
                )
            )
            * HCConstants.GEAR_RATIO
        )
        torques.append(
            Float64(
                max(
                    min(action.bshin, Scalar[Self.dtype](1.0)),
                    Scalar[Self.dtype](-1.0),
                )
            )
            * HCConstants.GEAR_RATIO
        )
        torques.append(
            Float64(
                max(
                    min(action.bfoot, Scalar[Self.dtype](1.0)),
                    Scalar[Self.dtype](-1.0),
                )
            )
            * HCConstants.GEAR_RATIO
        )
        torques.append(
            Float64(
                max(
                    min(action.fthigh, Scalar[Self.dtype](1.0)),
                    Scalar[Self.dtype](-1.0),
                )
            )
            * HCConstants.GEAR_RATIO
        )
        torques.append(
            Float64(
                max(
                    min(action.fshin, Scalar[Self.dtype](1.0)),
                    Scalar[Self.dtype](-1.0),
                )
            )
            * HCConstants.GEAR_RATIO
        )
        torques.append(
            Float64(
                max(
                    min(action.ffoot, Scalar[Self.dtype](1.0)),
                    Scalar[Self.dtype](-1.0),
                )
            )
            * HCConstants.GEAR_RATIO
        )

        var torso_base = HCConstants.BODY_TORSO * BODY_STATE_SIZE
        var x_before = self.bodies[torso_base + IDX_X]

        for _ in range(HCConstants.FRAME_SKIP):
            self._physics_step(torques)

        var x_after = self.bodies[torso_base + IDX_X]

        # Forward velocity reward
        var forward_velocity = (x_after - x_before) / (
            HCConstants.DT * Float64(HCConstants.FRAME_SKIP)
        )
        var forward_reward = (
            HCConstants.FORWARD_REWARD_WEIGHT * forward_velocity
        )

        # Control cost
        var ctrl_cost = (
            Float64(action.bthigh * action.bthigh)
            + Float64(action.bshin * action.bshin)
            + Float64(action.bfoot * action.bfoot)
            + Float64(action.fthigh * action.fthigh)
            + Float64(action.fshin * action.fshin)
            + Float64(action.ffoot * action.ffoot)
        )
        ctrl_cost *= HCConstants.CTRL_COST_WEIGHT

        # Total reward (no healthy reward for HalfCheetah)
        var reward = forward_reward - ctrl_cost

        self.step_count += 1
        self.total_reward += reward

        # Only terminate on max steps (no falling termination)
        self.done = self.step_count >= HCConstants.MAX_STEPS

        self._update_cached_state()
        return (Scalar[Self.dtype](reward), self.done)

    fn get_state(self) -> Self.StateType:
        """Return current state representation."""
        return self.cached_state

    fn get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return current observation as a list (ContinuousStateEnv trait)."""
        return self.cached_state.to_list()

    fn reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset and return initial observation (ContinuousStateEnv trait)."""
        var state = self.reset()
        return state.to_list()

    fn obs_dim(self) -> Int:
        """Return observation dimension (ContinuousStateEnv trait)."""
        return HCConstants.OBS_DIM_VAL

    fn action_dim(self) -> Int:
        """Return action dimension (ContinuousActionEnv trait)."""
        return HCConstants.ACTION_DIM_VAL

    fn action_low(self) -> Scalar[Self.dtype]:
        """Return minimum action value (ContinuousActionEnv trait)."""
        return Scalar[Self.dtype](-1.0)

    fn action_high(self) -> Scalar[Self.dtype]:
        """Return maximum action value (ContinuousActionEnv trait)."""
        return Scalar[Self.dtype](1.0)

    fn step_continuous(
        mut self, action: Scalar[Self.dtype]
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Step with single scalar action (applied to all joints)."""
        var act = HalfCheetahPlanarAction[Self.dtype](
            action, action, action, action, action, action
        )
        var result = self._step_internal(act)
        return (self.get_obs_list(), result[0], result[1])

    fn step_continuous_vec[
        DTYPE_VEC: DType
    ](mut self, action: List[Scalar[DTYPE_VEC]]) -> Tuple[
        List[Scalar[DTYPE_VEC]], Scalar[DTYPE_VEC], Bool
    ]:
        """Step with vector action (BoxContinuousActionEnv trait)."""
        var bthigh = Scalar[Self.dtype](action[0]) if len(
            action
        ) > 0 else Scalar[Self.dtype](0)
        var bshin = Scalar[Self.dtype](action[1]) if len(
            action
        ) > 1 else Scalar[Self.dtype](0)
        var bfoot = Scalar[Self.dtype](action[2]) if len(
            action
        ) > 2 else Scalar[Self.dtype](0)
        var fthigh = Scalar[Self.dtype](action[3]) if len(
            action
        ) > 3 else Scalar[Self.dtype](0)
        var fshin = Scalar[Self.dtype](action[4]) if len(
            action
        ) > 4 else Scalar[Self.dtype](0)
        var ffoot = Scalar[Self.dtype](action[5]) if len(
            action
        ) > 5 else Scalar[Self.dtype](0)

        var act = HalfCheetahPlanarAction[Self.dtype](
            bthigh, bshin, bfoot, fthigh, fshin, ffoot
        )
        var result = self._step_internal(act)

        var obs = self.cached_state.to_list_typed[DTYPE_VEC]()
        return (obs^, Scalar[DTYPE_VEC](result[0]), result[1])

    # =========================================================================
    # Legacy API (backward compatibility)
    # =========================================================================

    fn get_observation(self) -> List[Float64]:
        """Get current observation (17D) - legacy API."""
        var obs = List[Float64]()

        var torso_base = HCConstants.BODY_TORSO * BODY_STATE_SIZE
        var torso_y = self.bodies[torso_base + IDX_Y]
        var torso_angle = self.bodies[torso_base + IDX_ANGLE]
        var torso_vx = self.bodies[torso_base + IDX_VX]
        var torso_vy = self.bodies[torso_base + IDX_VY]
        var torso_omega = self.bodies[torso_base + IDX_OMEGA]

        # Position observations
        obs.append(torso_y)
        obs.append(torso_angle)

        # Joint angles
        obs.append(self.joint_angles[HCConstants.JOINT_BTHIGH])
        obs.append(self.joint_angles[HCConstants.JOINT_BSHIN])
        obs.append(self.joint_angles[HCConstants.JOINT_BFOOT])
        obs.append(self.joint_angles[HCConstants.JOINT_FTHIGH])
        obs.append(self.joint_angles[HCConstants.JOINT_FSHIN])
        obs.append(self.joint_angles[HCConstants.JOINT_FFOOT])

        # Velocities
        obs.append(torso_vx)
        obs.append(torso_vy)
        obs.append(torso_omega)

        # Joint velocities
        obs.append(self.joint_velocities[HCConstants.JOINT_BTHIGH])
        obs.append(self.joint_velocities[HCConstants.JOINT_BSHIN])
        obs.append(self.joint_velocities[HCConstants.JOINT_BFOOT])
        obs.append(self.joint_velocities[HCConstants.JOINT_FTHIGH])
        obs.append(self.joint_velocities[HCConstants.JOINT_FSHIN])
        obs.append(self.joint_velocities[HCConstants.JOINT_FFOOT])

        return obs^

    fn is_done(self) -> Bool:
        """Check if episode is done."""
        return self.done

    fn get_total_reward(self) -> Float64:
        """Get total accumulated reward."""
        return self.total_reward

    # =========================================================================
    # Rendering Methods
    # =========================================================================

    fn render(mut self, mut renderer: RendererBase):
        """Render the current state using SDL2.

        Draws the HalfCheetah as a side-view articulated body with:
        - Horizontal torso
        - Back leg (thigh, shin, foot)
        - Front leg (thigh, shin, foot)
        - Ground line
        - Info text overlay

        Args:
            renderer: External renderer to use for drawing.
        """
        if not renderer.begin_frame():
            return

        # Colors for the cheetah body
        var torso_color = rgb(139, 90, 43)  # Brown for torso
        var back_leg_color = rgb(180, 120, 60)  # Lighter brown for back leg
        var front_leg_color = rgb(160, 100, 50)  # Medium brown for front leg
        var joint_color = rgb(80, 80, 80)  # Dark gray for joints
        var ground_color = ground_brown()
        var sky_color = sky_blue()

        # Clear screen with sky color
        renderer.clear_with_color(sky_color)

        # Get torso position for camera tracking
        var torso_base = HCConstants.BODY_TORSO * BODY_STATE_SIZE
        var torso_x = self.bodies[torso_base + IDX_X]
        var torso_y = self.bodies[torso_base + IDX_Y]
        var torso_angle = self.bodies[torso_base + IDX_ANGLE]

        # Create camera that follows the torso
        # Zoom: pixels per world unit, Y-flip for physics coords
        var zoom = 200.0
        var camera = renderer.make_camera_at(
            torso_x,  # Follow torso X
            torso_y * 0.5 + 0.3,  # Slight offset to show ground
            zoom,
            True,  # flip_y for physics coordinates
        )

        # Draw ground line
        renderer.draw_ground_line(0.0, camera, ground_color, 3)

        # Draw ground fill (rectangle below ground line)
        var ground_fill_color = rgb(100, 80, 60)
        var bounds = camera.get_viewport_bounds()
        var min_corner = bounds[0]
        var max_corner = bounds[1]
        renderer.draw_rect_world(
            Vec2((min_corner.x + max_corner.x) / 2.0, -0.5),
            max_corner.x - min_corner.x + 2.0,
            1.0,
            camera,
            ground_fill_color,
            True,  # centered
            0,  # filled
        )

        # Draw torso (horizontal capsule)
        var torso_start = Vec2(
            torso_x - HCConstants.TORSO_LENGTH / 2 * cos(torso_angle),
            torso_y - HCConstants.TORSO_LENGTH / 2 * sin(torso_angle),
        )
        var torso_end = Vec2(
            torso_x + HCConstants.TORSO_LENGTH / 2 * cos(torso_angle),
            torso_y + HCConstants.TORSO_LENGTH / 2 * sin(torso_angle),
        )
        renderer.draw_line_world(torso_start, torso_end, camera, torso_color, 12)
        # Draw torso caps
        renderer.draw_circle_world(
            torso_start, HCConstants.TORSO_RADIUS * 1.5, camera, torso_color, True
        )
        renderer.draw_circle_world(
            torso_end, HCConstants.TORSO_RADIUS * 1.5, camera, torso_color, True
        )

        # Draw back leg (thigh, shin, foot)
        var back_hip_x = torso_x - HCConstants.TORSO_LENGTH / 2
        var back_hip_y = torso_y

        # Back thigh
        var bthigh_angle = self.joint_angles[HCConstants.JOINT_BTHIGH]
        var bthigh_end = Vec2(
            back_hip_x + sin(bthigh_angle) * HCConstants.BTHIGH_LENGTH,
            back_hip_y - cos(bthigh_angle) * HCConstants.BTHIGH_LENGTH,
        )
        renderer.draw_line_world(
            Vec2(back_hip_x, back_hip_y), bthigh_end, camera, back_leg_color, 8
        )

        # Back shin
        var bshin_angle = self.joint_angles[HCConstants.JOINT_BSHIN]
        var total_bshin_angle = bthigh_angle + bshin_angle
        var bshin_end = Vec2(
            bthigh_end.x + sin(total_bshin_angle) * HCConstants.BSHIN_LENGTH,
            bthigh_end.y - cos(total_bshin_angle) * HCConstants.BSHIN_LENGTH,
        )
        renderer.draw_line_world(bthigh_end, bshin_end, camera, back_leg_color, 6)

        # Back foot
        var bfoot_angle = self.joint_angles[HCConstants.JOINT_BFOOT]
        var total_bfoot_angle = total_bshin_angle + bfoot_angle
        var bfoot_end = Vec2(
            bshin_end.x + sin(total_bfoot_angle) * HCConstants.BFOOT_LENGTH,
            bshin_end.y - cos(total_bfoot_angle) * HCConstants.BFOOT_LENGTH,
        )
        renderer.draw_line_world(bshin_end, bfoot_end, camera, back_leg_color, 5)

        # Draw front leg (thigh, shin, foot)
        var front_hip_x = torso_x + HCConstants.TORSO_LENGTH / 2
        var front_hip_y = torso_y

        # Front thigh
        var fthigh_angle = self.joint_angles[HCConstants.JOINT_FTHIGH]
        var fthigh_end = Vec2(
            front_hip_x + sin(fthigh_angle) * HCConstants.FTHIGH_LENGTH,
            front_hip_y - cos(fthigh_angle) * HCConstants.FTHIGH_LENGTH,
        )
        renderer.draw_line_world(
            Vec2(front_hip_x, front_hip_y), fthigh_end, camera, front_leg_color, 8
        )

        # Front shin
        var fshin_angle = self.joint_angles[HCConstants.JOINT_FSHIN]
        var total_fshin_angle = fthigh_angle + fshin_angle
        var fshin_end = Vec2(
            fthigh_end.x + sin(total_fshin_angle) * HCConstants.FSHIN_LENGTH,
            fthigh_end.y - cos(total_fshin_angle) * HCConstants.FSHIN_LENGTH,
        )
        renderer.draw_line_world(fthigh_end, fshin_end, camera, front_leg_color, 6)

        # Front foot
        var ffoot_angle = self.joint_angles[HCConstants.JOINT_FFOOT]
        var total_ffoot_angle = total_fshin_angle + ffoot_angle
        var ffoot_end = Vec2(
            fshin_end.x + sin(total_ffoot_angle) * HCConstants.FFOOT_LENGTH,
            fshin_end.y - cos(total_ffoot_angle) * HCConstants.FFOOT_LENGTH,
        )
        renderer.draw_line_world(fshin_end, ffoot_end, camera, front_leg_color, 5)

        # Draw joints as small circles
        var joint_radius = 0.03
        # Back hip
        renderer.draw_circle_world(
            Vec2(back_hip_x, back_hip_y), joint_radius, camera, joint_color, True
        )
        # Back knee
        renderer.draw_circle_world(bthigh_end, joint_radius, camera, joint_color, True)
        # Back ankle
        renderer.draw_circle_world(bshin_end, joint_radius, camera, joint_color, True)
        # Front hip
        renderer.draw_circle_world(
            Vec2(front_hip_x, front_hip_y), joint_radius, camera, joint_color, True
        )
        # Front knee
        renderer.draw_circle_world(fthigh_end, joint_radius, camera, joint_color, True)
        # Front ankle
        renderer.draw_circle_world(fshin_end, joint_radius, camera, joint_color, True)

        # Draw foot endpoints (for ground contact visualization)
        var foot_marker_color = rgb(255, 100, 100)  # Red-ish
        renderer.draw_circle_world(
            bfoot_end, HCConstants.BFOOT_RADIUS, camera, foot_marker_color, True
        )
        renderer.draw_circle_world(
            ffoot_end, HCConstants.FFOOT_RADIUS, camera, foot_marker_color, True
        )

        # Get velocity for display
        var vel_x = self.bodies[torso_base + IDX_VX]

        # Draw info text
        var info_lines = List[String]()
        info_lines.append("HalfCheetah")
        info_lines.append("Step: " + String(self.step_count))
        info_lines.append("Reward: " + String(Int(self.total_reward)))
        info_lines.append("X: " + String(torso_x)[:7])
        info_lines.append("Vel: " + String(vel_x)[:6])
        renderer.draw_info_box(info_lines)

        # Update display
        renderer.flip()

    fn close(mut self):
        """Clean up resources.

        Since the renderer is external, this is a no-op.
        The caller is responsible for closing the renderer.
        """
        pass

    # =========================================================================
    # Physics Implementation
    # =========================================================================

    fn _physics_step(mut self, torques: List[Float64]):
        """Perform one physics step."""
        var dt = HCConstants.DT

        for i in range(HCConstants.MAX_JOINTS):
            var torque = torques[i] if i < len(torques) else 0.0
            self.joint_velocities[i] += torque * dt * 0.01

        for i in range(HCConstants.MAX_JOINTS):
            self.joint_angles[i] += self.joint_velocities[i] * dt
            self._clamp_joint(i)

        var torso_base = HCConstants.BODY_TORSO * BODY_STATE_SIZE
        self.bodies[torso_base + IDX_VY] += HCConstants.GRAVITY_Y * dt

        self._update_body_positions()
        self._handle_ground_collision()

    fn _clamp_joint(mut self, joint_idx: Int):
        """Clamp joint angle to limits."""
        var low: Float64
        var high: Float64

        if joint_idx == HCConstants.JOINT_BTHIGH:
            low = HCConstants.BTHIGH_LIMIT_LOW
            high = HCConstants.BTHIGH_LIMIT_HIGH
        elif joint_idx == HCConstants.JOINT_BSHIN:
            low = HCConstants.BSHIN_LIMIT_LOW
            high = HCConstants.BSHIN_LIMIT_HIGH
        elif joint_idx == HCConstants.JOINT_BFOOT:
            low = HCConstants.BFOOT_LIMIT_LOW
            high = HCConstants.BFOOT_LIMIT_HIGH
        elif joint_idx == HCConstants.JOINT_FTHIGH:
            low = HCConstants.FTHIGH_LIMIT_LOW
            high = HCConstants.FTHIGH_LIMIT_HIGH
        elif joint_idx == HCConstants.JOINT_FSHIN:
            low = HCConstants.FSHIN_LIMIT_LOW
            high = HCConstants.FSHIN_LIMIT_HIGH
        else:  # FFOOT
            low = HCConstants.FFOOT_LIMIT_LOW
            high = HCConstants.FFOOT_LIMIT_HIGH

        if self.joint_angles[joint_idx] < low:
            self.joint_angles[joint_idx] = low
            self.joint_velocities[joint_idx] = 0.0
        elif self.joint_angles[joint_idx] > high:
            self.joint_angles[joint_idx] = high
            self.joint_velocities[joint_idx] = 0.0

    fn _update_body_positions(mut self):
        """Update body positions via forward kinematics."""
        var torso_base = HCConstants.BODY_TORSO * BODY_STATE_SIZE
        var torso_x = self.bodies[torso_base + IDX_X]
        var torso_y = self.bodies[torso_base + IDX_Y]

        torso_x += self.bodies[torso_base + IDX_VX] * HCConstants.DT
        torso_y += self.bodies[torso_base + IDX_VY] * HCConstants.DT
        self.bodies[torso_base + IDX_X] = torso_x
        self.bodies[torso_base + IDX_Y] = torso_y

        # Update back leg
        var back_hip_x = torso_x - HCConstants.TORSO_LENGTH / 2
        var back_hip_y = torso_y

        self._update_leg_positions(
            back_hip_x,
            back_hip_y,
            HCConstants.JOINT_BTHIGH,
            HCConstants.JOINT_BSHIN,
            HCConstants.JOINT_BFOOT,
            HCConstants.BODY_BTHIGH,
            HCConstants.BODY_BSHIN,
            HCConstants.BODY_BFOOT,
            HCConstants.BTHIGH_LENGTH,
            HCConstants.BSHIN_LENGTH,
            HCConstants.BFOOT_LENGTH,
        )

        # Update front leg
        var front_hip_x = torso_x + HCConstants.TORSO_LENGTH / 2
        var front_hip_y = torso_y

        self._update_leg_positions(
            front_hip_x,
            front_hip_y,
            HCConstants.JOINT_FTHIGH,
            HCConstants.JOINT_FSHIN,
            HCConstants.JOINT_FFOOT,
            HCConstants.BODY_FTHIGH,
            HCConstants.BODY_FSHIN,
            HCConstants.BODY_FFOOT,
            HCConstants.FTHIGH_LENGTH,
            HCConstants.FSHIN_LENGTH,
            HCConstants.FFOOT_LENGTH,
        )

    fn _update_leg_positions(
        mut self,
        hip_x: Float64,
        hip_y: Float64,
        thigh_joint: Int,
        shin_joint: Int,
        foot_joint: Int,
        thigh_body: Int,
        shin_body: Int,
        foot_body: Int,
        thigh_length: Float64,
        shin_length: Float64,
        foot_length: Float64,
    ):
        """Update positions for one leg."""
        var thigh_angle = self.joint_angles[thigh_joint]

        var thigh_base = thigh_body * BODY_STATE_SIZE
        self.bodies[thigh_base + IDX_X] = (
            hip_x + sin(thigh_angle) * thigh_length / 2
        )
        self.bodies[thigh_base + IDX_Y] = (
            hip_y - cos(thigh_angle) * thigh_length / 2
        )
        self.bodies[thigh_base + IDX_ANGLE] = thigh_angle

        var knee_x = hip_x + sin(thigh_angle) * thigh_length
        var knee_y = hip_y - cos(thigh_angle) * thigh_length
        var shin_angle = self.joint_angles[shin_joint]
        var total_shin_angle = thigh_angle + shin_angle

        var shin_base = shin_body * BODY_STATE_SIZE
        self.bodies[shin_base + IDX_X] = (
            knee_x + sin(total_shin_angle) * shin_length / 2
        )
        self.bodies[shin_base + IDX_Y] = (
            knee_y - cos(total_shin_angle) * shin_length / 2
        )
        self.bodies[shin_base + IDX_ANGLE] = total_shin_angle

        var ankle_x = knee_x + sin(total_shin_angle) * shin_length
        var ankle_y = knee_y - cos(total_shin_angle) * shin_length
        var foot_angle = self.joint_angles[foot_joint]
        var total_foot_angle = total_shin_angle + foot_angle

        var foot_base = foot_body * BODY_STATE_SIZE
        self.bodies[foot_base + IDX_X] = (
            ankle_x + sin(total_foot_angle) * foot_length / 2
        )
        self.bodies[foot_base + IDX_Y] = (
            ankle_y - cos(total_foot_angle) * foot_length / 2
        )
        self.bodies[foot_base + IDX_ANGLE] = total_foot_angle

    fn _handle_ground_collision(mut self):
        """Handle ground collision for both feet."""
        var ground_y = 0.0
        var torso_base = HCConstants.BODY_TORSO * BODY_STATE_SIZE

        # Check back foot
        var bfoot_base = HCConstants.BODY_BFOOT * BODY_STATE_SIZE
        var bfoot_y = self.bodies[bfoot_base + IDX_Y]
        if bfoot_y - HCConstants.BFOOT_RADIUS < ground_y:
            var penetration = ground_y - (bfoot_y - HCConstants.BFOOT_RADIUS)
            self.bodies[torso_base + IDX_Y] += penetration
            if self.bodies[torso_base + IDX_VY] < 0:
                self.bodies[torso_base + IDX_VY] = 0.0

        # Check front foot
        var ffoot_base = HCConstants.BODY_FFOOT * BODY_STATE_SIZE
        var ffoot_y = self.bodies[ffoot_base + IDX_Y]
        if ffoot_y - HCConstants.FFOOT_RADIUS < ground_y:
            var penetration = ground_y - (ffoot_y - HCConstants.FFOOT_RADIUS)
            self.bodies[torso_base + IDX_Y] += penetration
            if self.bodies[torso_base + IDX_VY] < 0:
                self.bodies[torso_base + IDX_VY] = 0.0

    # =========================================================================
    # GPUContinuousEnv Trait - GPU Kernel Methods
    # =========================================================================

    @staticmethod
    fn step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        mut obs_buf: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """GPU step kernel for batched continuous actions."""
        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var actions = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var rewards = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var obs = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ](obs_buf.unsafe_ptr())

        @always_inline
        fn step_wrapper(
            states: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
            ],
            actions: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
            ],
            rewards: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            dones: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            obs: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
            ],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            HalfCheetahPlanar[dtype]._step_env_gpu[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, ACTION_DIM
            ](states, actions, rewards, dones, obs, i)

        ctx.enqueue_function[step_wrapper, step_wrapper](
            states,
            actions,
            rewards,
            dones,
            obs,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """GPU reset kernel."""
        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())

        @always_inline
        fn reset_wrapper(
            states: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
            ],
            seed: Scalar[dtype],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            var combined_seed = Int(seed) * 2654435761 + (i + 1) * 12345
            HalfCheetahPlanar[dtype]._reset_env_gpu[BATCH_SIZE, STATE_SIZE](
                states, i, combined_seed
            )

        ctx.enqueue_function[reset_wrapper, reset_wrapper](
            states,
            Scalar[dtype](rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        rng_seed: UInt64,
    ) raises:
        """GPU selective reset kernel - resets only done environments."""
        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var dones = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())

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
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            var done_val = dones[i]
            if done_val > Scalar[dtype](0.5):
                var combined_seed = Int(seed) * 2654435761 + (i + 1) * 12345
                HalfCheetahPlanar[dtype]._reset_env_gpu[BATCH_SIZE, STATE_SIZE](
                    states, i, combined_seed
                )
                dones[i] = Scalar[dtype](0.0)

        ctx.enqueue_function[selective_reset_wrapper, selective_reset_wrapper](
            states,
            dones,
            Scalar[dtype](rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # GPU Helper Functions
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
        """Reset a single environment (GPU version)."""
        var rng = PhiloxRandom(seed=seed, offset=0)
        var rand_vals = rng.step_uniform()

        var torso_y = Scalar[dtype](HCConstants.INIT_HEIGHT)

        # Initialize bodies at observation offset (used for obs extraction)
        var obs_off = HCConstants.OBS_OFFSET

        # Torso state
        var torso_off = HCConstants.BODIES_OFFSET
        states[env, torso_off + IDX_X] = Scalar[dtype](0.0)
        states[env, torso_off + IDX_Y] = torso_y
        states[env, torso_off + IDX_ANGLE] = Scalar[dtype](0.0)
        states[env, torso_off + IDX_VX] = Scalar[dtype](0.0)
        states[env, torso_off + IDX_VY] = Scalar[dtype](0.0)
        states[env, torso_off + IDX_OMEGA] = Scalar[dtype](0.0)
        states[env, torso_off + IDX_INV_MASS] = Scalar[dtype](
            1.0 / HCConstants.TORSO_MASS
        )
        states[env, torso_off + IDX_INV_INERTIA] = Scalar[dtype](
            1.0 / (HCConstants.TORSO_MASS * 0.1)
        )

        # Initialize joint angles and velocities with small random noise
        var joints_off = HCConstants.JOINTS_OFFSET
        for j in range(6):
            var joint_off = joints_off + j * JOINT_DATA_SIZE
            var rand_angle = (rand_vals[j % 4] - Scalar[dtype](0.5)) * Scalar[
                dtype
            ](0.01)
            states[env, joint_off + JOINT_REF_ANGLE] = rand_angle

        # Initialize metadata
        var meta_off = HCConstants.METADATA_OFFSET
        states[env, meta_off + HCConstants.META_STEP_COUNT] = Scalar[dtype](0)
        states[env, meta_off + HCConstants.META_PREV_X] = Scalar[dtype](0.0)
        states[env, meta_off + HCConstants.META_DONE] = Scalar[dtype](0)
        states[env, meta_off + HCConstants.META_TOTAL_REWARD] = Scalar[dtype](
            0.0
        )

        # Write initial observation
        states[env, obs_off + 0] = torso_y  # torso_z
        states[env, obs_off + 1] = Scalar[dtype](0.0)  # torso_angle
        for j in range(6):
            states[env, obs_off + 2 + j] = Scalar[dtype](0.0)  # joint angles
        states[env, obs_off + 8] = Scalar[dtype](0.0)  # vel_x
        states[env, obs_off + 9] = Scalar[dtype](0.0)  # vel_z
        states[env, obs_off + 10] = Scalar[dtype](0.0)  # torso_omega
        for j in range(6):
            states[env, obs_off + 11 + j] = Scalar[dtype](
                0.0
            )  # joint velocities

    @always_inline
    @staticmethod
    fn _step_env_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        ACTION_DIM: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ],
        rewards: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        env: Int,
    ):
        """Step a single environment (GPU version)."""
        var obs_off = HCConstants.OBS_OFFSET
        var meta_off = HCConstants.METADATA_OFFSET
        var torso_off = HCConstants.BODIES_OFFSET

        # Get previous x position
        var x_before = rebind[Scalar[dtype]](states[env, torso_off + IDX_X])

        # Apply actions with clamping
        var torques = InlineArray[Scalar[dtype], 6](fill=Scalar[dtype](0.0))
        for j in range(6):
            var a = rebind[Scalar[dtype]](actions[env, j])
            if a > Scalar[dtype](1.0):
                a = Scalar[dtype](1.0)
            elif a < Scalar[dtype](-1.0):
                a = Scalar[dtype](-1.0)
            torques[j] = a * Scalar[dtype](HCConstants.GEAR_RATIO)

        # Physics step (simplified for GPU)
        var dt = Scalar[dtype](HCConstants.DT)
        var gravity = Scalar[dtype](HCConstants.GRAVITY_Y)

        for _ in range(HCConstants.FRAME_SKIP):
            # Update velocities from torques
            for j in range(6):
                var joint_vel_off = obs_off + 11 + j
                var vel = rebind[Scalar[dtype]](states[env, joint_vel_off])
                vel = vel + torques[j] * dt * Scalar[dtype](0.01)
                states[env, joint_vel_off] = vel

            # Update joint angles
            for j in range(6):
                var joint_angle_off = obs_off + 2 + j
                var joint_vel_off = obs_off + 11 + j
                var angle = rebind[Scalar[dtype]](states[env, joint_angle_off])
                var vel = rebind[Scalar[dtype]](states[env, joint_vel_off])
                angle = angle + vel * dt
                # Clamp joint angles (simplified)
                if angle < Scalar[dtype](-1.5):
                    angle = Scalar[dtype](-1.5)
                    states[env, joint_vel_off] = Scalar[dtype](0.0)
                elif angle > Scalar[dtype](1.5):
                    angle = Scalar[dtype](1.5)
                    states[env, joint_vel_off] = Scalar[dtype](0.0)
                states[env, joint_angle_off] = angle

            # Apply gravity
            var vy = rebind[Scalar[dtype]](states[env, torso_off + IDX_VY])
            vy = vy + gravity * dt
            states[env, torso_off + IDX_VY] = vy

            # Update torso position
            var x = rebind[Scalar[dtype]](states[env, torso_off + IDX_X])
            var y = rebind[Scalar[dtype]](states[env, torso_off + IDX_Y])
            var vx = rebind[Scalar[dtype]](states[env, torso_off + IDX_VX])
            x = x + vx * dt
            y = y + vy * dt
            states[env, torso_off + IDX_X] = x
            states[env, torso_off + IDX_Y] = y

            # Ground collision
            if y < Scalar[dtype](HCConstants.INIT_HEIGHT * 0.3):
                y = Scalar[dtype](HCConstants.INIT_HEIGHT * 0.3)
                states[env, torso_off + IDX_Y] = y
                if vy < Scalar[dtype](0.0):
                    states[env, torso_off + IDX_VY] = Scalar[dtype](0.0)

        # Compute reward
        var x_after = rebind[Scalar[dtype]](states[env, torso_off + IDX_X])
        var forward_velocity = (x_after - x_before) / (
            dt * Scalar[dtype](HCConstants.FRAME_SKIP)
        )
        var forward_reward = (
            Scalar[dtype](HCConstants.FORWARD_REWARD_WEIGHT) * forward_velocity
        )

        var ctrl_cost = Scalar[dtype](0.0)
        for j in range(6):
            var a = rebind[Scalar[dtype]](actions[env, j])
            ctrl_cost = ctrl_cost + a * a
        ctrl_cost = ctrl_cost * Scalar[dtype](HCConstants.CTRL_COST_WEIGHT)

        var reward = forward_reward - ctrl_cost
        rewards[env] = reward

        # Update step count and check done
        var step_count = rebind[Scalar[dtype]](
            states[env, meta_off + HCConstants.META_STEP_COUNT]
        )
        step_count = step_count + Scalar[dtype](1.0)
        states[env, meta_off + HCConstants.META_STEP_COUNT] = step_count

        var done = Scalar[dtype](0.0)
        if step_count >= Scalar[dtype](HCConstants.MAX_STEPS):
            done = Scalar[dtype](1.0)
        dones[env] = done

        # Update observation buffer
        var torso_y = rebind[Scalar[dtype]](states[env, torso_off + IDX_Y])
        var torso_angle = rebind[Scalar[dtype]](
            states[env, torso_off + IDX_ANGLE]
        )
        var torso_vx = rebind[Scalar[dtype]](states[env, torso_off + IDX_VX])
        var torso_vy = rebind[Scalar[dtype]](states[env, torso_off + IDX_VY])
        var torso_omega = rebind[Scalar[dtype]](
            states[env, torso_off + IDX_OMEGA]
        )

        obs[env, 0] = torso_y
        obs[env, 1] = torso_angle
        for j in range(6):
            obs[env, 2 + j] = rebind[Scalar[dtype]](
                states[env, obs_off + 2 + j]
            )
        obs[env, 8] = torso_vx
        obs[env, 9] = torso_vy
        obs[env, 10] = torso_omega
        for j in range(6):
            obs[env, 11 + j] = rebind[Scalar[dtype]](
                states[env, obs_off + 11 + j]
            )

        # Update cached observation in state
        for j in range(17):
            states[env, obs_off + j] = obs[env, j]
