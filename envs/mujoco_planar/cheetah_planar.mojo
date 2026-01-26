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

from physics2d import (
    dtype,
    BODY_STATE_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    IDX_INV_MASS,
    IDX_INV_INERTIA,
)

from physics2d.articulated import (
    HALFCHEETAH_NUM_BODIES,
    HALFCHEETAH_NUM_JOINTS,
)


struct HalfCheetahPlanarConstants:
    """Constants for planar HalfCheetah environment."""

    # ==========================================================================
    # Physics Parameters
    # ==========================================================================

    comptime DT: Float64 = 0.01
    comptime FRAME_SKIP: Int = 5
    comptime GRAVITY: Float64 = -9.81

    # ==========================================================================
    # Body Geometry (from MuJoCo HalfCheetah XML)
    # ==========================================================================

    # Torso (longer horizontal body)
    comptime TORSO_LENGTH: Float64 = 1.0
    comptime TORSO_RADIUS: Float64 = 0.046
    comptime TORSO_MASS: Float64 = 6.25

    # Back thigh
    comptime BTHIGH_LENGTH: Float64 = 0.145
    comptime BTHIGH_RADIUS: Float64 = 0.046
    comptime BTHIGH_MASS: Float64 = 1.0

    # Back shin
    comptime BSHIN_LENGTH: Float64 = 0.15
    comptime BSHIN_RADIUS: Float64 = 0.046
    comptime BSHIN_MASS: Float64 = 1.0

    # Back foot
    comptime BFOOT_LENGTH: Float64 = 0.094
    comptime BFOOT_RADIUS: Float64 = 0.046
    comptime BFOOT_MASS: Float64 = 1.0

    # Front thigh
    comptime FTHIGH_LENGTH: Float64 = 0.133
    comptime FTHIGH_RADIUS: Float64 = 0.046
    comptime FTHIGH_MASS: Float64 = 1.0

    # Front shin
    comptime FSHIN_LENGTH: Float64 = 0.106
    comptime FSHIN_RADIUS: Float64 = 0.046
    comptime FSHIN_MASS: Float64 = 1.0

    # Front foot
    comptime FFOOT_LENGTH: Float64 = 0.07
    comptime FFOOT_RADIUS: Float64 = 0.046
    comptime FFOOT_MASS: Float64 = 1.0

    # Initial height
    comptime INIT_HEIGHT: Float64 = 0.7

    # ==========================================================================
    # Body Indices
    # ==========================================================================

    comptime BODY_TORSO: Int = 0
    comptime BODY_BTHIGH: Int = 1  # Back thigh
    comptime BODY_BSHIN: Int = 2  # Back shin
    comptime BODY_BFOOT: Int = 3  # Back foot
    comptime BODY_FTHIGH: Int = 4  # Front thigh
    comptime BODY_FSHIN: Int = 5  # Front shin
    comptime BODY_FFOOT: Int = 6  # Front foot

    # ==========================================================================
    # Joint Indices
    # ==========================================================================

    comptime JOINT_BTHIGH: Int = 0  # Back hip
    comptime JOINT_BSHIN: Int = 1  # Back knee
    comptime JOINT_BFOOT: Int = 2  # Back ankle
    comptime JOINT_FTHIGH: Int = 3  # Front hip
    comptime JOINT_FSHIN: Int = 4  # Front knee
    comptime JOINT_FFOOT: Int = 5  # Front ankle

    # ==========================================================================
    # Joint Limits (from MuJoCo XML)
    # ==========================================================================

    comptime BTHIGH_LIMIT_LOW: Float64 = -0.52
    comptime BTHIGH_LIMIT_HIGH: Float64 = 1.05

    comptime BSHIN_LIMIT_LOW: Float64 = -0.785
    comptime BSHIN_LIMIT_HIGH: Float64 = 0.785

    comptime BFOOT_LIMIT_LOW: Float64 = -0.4
    comptime BFOOT_LIMIT_HIGH: Float64 = 0.785

    comptime FTHIGH_LIMIT_LOW: Float64 = -1.0
    comptime FTHIGH_LIMIT_HIGH: Float64 = 0.7

    comptime FSHIN_LIMIT_LOW: Float64 = -1.2
    comptime FSHIN_LIMIT_HIGH: Float64 = 0.87

    comptime FFOOT_LIMIT_LOW: Float64 = -0.5
    comptime FFOOT_LIMIT_HIGH: Float64 = 0.5

    # ==========================================================================
    # Motor Parameters
    # ==========================================================================

    comptime MAX_TORQUE: Float64 = 120.0
    comptime GEAR_RATIO: Float64 = 120.0

    # ==========================================================================
    # Observation and Action Dimensions
    # ==========================================================================

    comptime OBS_DIM: Int = 17
    comptime ACTION_DIM: Int = 6

    # ==========================================================================
    # Reward Parameters
    # ==========================================================================

    comptime CTRL_COST_WEIGHT: Float64 = 0.1
    comptime FORWARD_REWARD_WEIGHT: Float64 = 1.0

    # ==========================================================================
    # Episode Parameters (no early termination)
    # ==========================================================================

    comptime MAX_STEPS: Int = 1000

    # ==========================================================================
    # Counts
    # ==========================================================================

    comptime NUM_BODIES: Int = 7
    comptime NUM_JOINTS: Int = 6


struct HalfCheetahPlanar:
    """Planar HalfCheetah locomotion environment.

    Unlike Hopper and Walker2d, HalfCheetah does NOT terminate on falling.
    Episodes only end after MAX_STEPS.
    """

    var bodies: List[Float64]
    var joint_angles: List[Float64]
    var joint_velocities: List[Float64]
    var step_count: Int
    var done: Bool
    var total_reward: Float64
    var prev_x: Float64

    fn __init__(out self):
        """Initialize the HalfCheetah environment."""
        self.bodies = List[Float64]()
        self.joint_angles = List[Float64]()
        self.joint_velocities = List[Float64]()
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0
        self.prev_x = 0.0

        for _ in range(HalfCheetahPlanarConstants.NUM_BODIES * BODY_STATE_SIZE):
            self.bodies.append(0.0)

        for _ in range(HalfCheetahPlanarConstants.NUM_JOINTS):
            self.joint_angles.append(0.0)
            self.joint_velocities.append(0.0)

        self._reset_state()

    fn _reset_state(mut self):
        """Reset to initial position."""
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0

        var torso_y = HalfCheetahPlanarConstants.INIT_HEIGHT

        # Torso (horizontal)
        self._init_body(
            HalfCheetahPlanarConstants.BODY_TORSO,
            0.0,
            torso_y,
            0.0,
            0.0,
            0.0,
            0.0,
            HalfCheetahPlanarConstants.TORSO_MASS,
        )

        # Back leg (behind torso)
        var back_hip_x = -HalfCheetahPlanarConstants.TORSO_LENGTH / 2
        var back_hip_y = torso_y

        self._init_body(
            HalfCheetahPlanarConstants.BODY_BTHIGH,
            back_hip_x,
            back_hip_y - HalfCheetahPlanarConstants.BTHIGH_LENGTH / 2,
            0.0,
            0.0,
            0.0,
            0.0,
            HalfCheetahPlanarConstants.BTHIGH_MASS,
        )

        self._init_body(
            HalfCheetahPlanarConstants.BODY_BSHIN,
            back_hip_x,
            back_hip_y
            - HalfCheetahPlanarConstants.BTHIGH_LENGTH
            - HalfCheetahPlanarConstants.BSHIN_LENGTH / 2,
            0.0,
            0.0,
            0.0,
            0.0,
            HalfCheetahPlanarConstants.BSHIN_MASS,
        )

        self._init_body(
            HalfCheetahPlanarConstants.BODY_BFOOT,
            back_hip_x,
            back_hip_y
            - HalfCheetahPlanarConstants.BTHIGH_LENGTH
            - HalfCheetahPlanarConstants.BSHIN_LENGTH
            - HalfCheetahPlanarConstants.BFOOT_LENGTH / 2,
            0.0,
            0.0,
            0.0,
            0.0,
            HalfCheetahPlanarConstants.BFOOT_MASS,
        )

        # Front leg (in front of torso)
        var front_hip_x = HalfCheetahPlanarConstants.TORSO_LENGTH / 2
        var front_hip_y = torso_y

        self._init_body(
            HalfCheetahPlanarConstants.BODY_FTHIGH,
            front_hip_x,
            front_hip_y - HalfCheetahPlanarConstants.FTHIGH_LENGTH / 2,
            0.0,
            0.0,
            0.0,
            0.0,
            HalfCheetahPlanarConstants.FTHIGH_MASS,
        )

        self._init_body(
            HalfCheetahPlanarConstants.BODY_FSHIN,
            front_hip_x,
            front_hip_y
            - HalfCheetahPlanarConstants.FTHIGH_LENGTH
            - HalfCheetahPlanarConstants.FSHIN_LENGTH / 2,
            0.0,
            0.0,
            0.0,
            0.0,
            HalfCheetahPlanarConstants.FSHIN_MASS,
        )

        self._init_body(
            HalfCheetahPlanarConstants.BODY_FFOOT,
            front_hip_x,
            front_hip_y
            - HalfCheetahPlanarConstants.FTHIGH_LENGTH
            - HalfCheetahPlanarConstants.FSHIN_LENGTH
            - HalfCheetahPlanarConstants.FFOOT_LENGTH / 2,
            0.0,
            0.0,
            0.0,
            0.0,
            HalfCheetahPlanarConstants.FFOOT_MASS,
        )

        for i in range(HalfCheetahPlanarConstants.NUM_JOINTS):
            self.joint_angles[i] = 0.0
            self.joint_velocities[i] = 0.0

        self.prev_x = 0.0

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

    fn reset(mut self) -> List[Float64]:
        """Reset environment and return initial observation."""
        self._reset_state()

        for i in range(HalfCheetahPlanarConstants.NUM_JOINTS):
            self.joint_angles[i] = (random_float64() - 0.5) * 0.01
            self.joint_velocities[i] = (random_float64() - 0.5) * 0.01

        return self.get_observation()

    fn get_observation(self) -> List[Float64]:
        """Get current observation (17D)."""
        var obs = List[Float64]()

        var torso_base = HalfCheetahPlanarConstants.BODY_TORSO * BODY_STATE_SIZE
        var torso_y = self.bodies[torso_base + IDX_Y]
        var torso_angle = self.bodies[torso_base + IDX_ANGLE]
        var torso_vx = self.bodies[torso_base + IDX_VX]
        var torso_vy = self.bodies[torso_base + IDX_VY]
        var torso_omega = self.bodies[torso_base + IDX_OMEGA]

        # Position observations
        obs.append(torso_y)  # [0]
        obs.append(torso_angle)  # [1]

        # Back leg joint angles
        obs.append(
            self.joint_angles[HalfCheetahPlanarConstants.JOINT_BTHIGH]
        )  # [2]
        obs.append(
            self.joint_angles[HalfCheetahPlanarConstants.JOINT_BSHIN]
        )  # [3]
        obs.append(
            self.joint_angles[HalfCheetahPlanarConstants.JOINT_BFOOT]
        )  # [4]

        # Front leg joint angles
        obs.append(
            self.joint_angles[HalfCheetahPlanarConstants.JOINT_FTHIGH]
        )  # [5]
        obs.append(
            self.joint_angles[HalfCheetahPlanarConstants.JOINT_FSHIN]
        )  # [6]
        obs.append(
            self.joint_angles[HalfCheetahPlanarConstants.JOINT_FFOOT]
        )  # [7]

        # Velocities
        obs.append(torso_vx)  # [8]
        obs.append(torso_vy)  # [9]
        obs.append(torso_omega)  # [10]

        # Back leg joint velocities
        obs.append(
            self.joint_velocities[HalfCheetahPlanarConstants.JOINT_BTHIGH]
        )  # [11]
        obs.append(
            self.joint_velocities[HalfCheetahPlanarConstants.JOINT_BSHIN]
        )  # [12]
        obs.append(
            self.joint_velocities[HalfCheetahPlanarConstants.JOINT_BFOOT]
        )  # [13]

        # Front leg joint velocities
        obs.append(
            self.joint_velocities[HalfCheetahPlanarConstants.JOINT_FTHIGH]
        )  # [14]
        obs.append(
            self.joint_velocities[HalfCheetahPlanarConstants.JOINT_FSHIN]
        )  # [15]
        obs.append(
            self.joint_velocities[HalfCheetahPlanarConstants.JOINT_FFOOT]
        )  # [16]

        return obs^

    fn step(
        mut self,
        action: List[Float64],
    ) -> Tuple[List[Float64], Float64, Bool]:
        """Take a step in the environment.

        Note: HalfCheetah does NOT terminate on falling, only on max steps.
        """
        var torques = List[Float64]()
        for i in range(HalfCheetahPlanarConstants.ACTION_DIM):
            var a = action[i] if i < len(action) else 0.0
            if a > 1.0:
                a = 1.0
            elif a < -1.0:
                a = -1.0
            torques.append(a * HalfCheetahPlanarConstants.GEAR_RATIO)

        var torso_base = HalfCheetahPlanarConstants.BODY_TORSO * BODY_STATE_SIZE
        var x_before = self.bodies[torso_base + IDX_X]

        for _ in range(HalfCheetahPlanarConstants.FRAME_SKIP):
            self._physics_step(torques)

        var x_after = self.bodies[torso_base + IDX_X]

        # Forward velocity reward
        var forward_velocity = (x_after - x_before) / (
            HalfCheetahPlanarConstants.DT
            * Float64(HalfCheetahPlanarConstants.FRAME_SKIP)
        )
        var forward_reward = (
            HalfCheetahPlanarConstants.FORWARD_REWARD_WEIGHT * forward_velocity
        )

        # Control cost
        var ctrl_cost = 0.0
        for i in range(len(action)):
            var a = action[i] if action[i] >= -1.0 and action[i] <= 1.0 else (
                1.0 if action[i] > 0 else -1.0
            )
            ctrl_cost += a * a
        ctrl_cost *= HalfCheetahPlanarConstants.CTRL_COST_WEIGHT

        # Total reward (no healthy reward for HalfCheetah)
        var reward = forward_reward - ctrl_cost

        self.step_count += 1
        self.total_reward += reward

        # Only terminate on max steps (no falling termination)
        self.done = self.step_count >= HalfCheetahPlanarConstants.MAX_STEPS

        return (self.get_observation(), reward, self.done)

    fn _physics_step(mut self, torques: List[Float64]):
        """Perform one physics step."""
        var dt = HalfCheetahPlanarConstants.DT

        for i in range(HalfCheetahPlanarConstants.NUM_JOINTS):
            var torque = torques[i] if i < len(torques) else 0.0
            self.joint_velocities[i] += torque * dt * 0.01

        for i in range(HalfCheetahPlanarConstants.NUM_JOINTS):
            self.joint_angles[i] += self.joint_velocities[i] * dt
            self._clamp_joint(i)

        var torso_base = HalfCheetahPlanarConstants.BODY_TORSO * BODY_STATE_SIZE
        self.bodies[torso_base + IDX_VY] += (
            HalfCheetahPlanarConstants.GRAVITY * dt
        )

        self._update_body_positions()
        self._handle_ground_collision()

    fn _clamp_joint(mut self, joint_idx: Int):
        """Clamp joint angle to limits."""
        var low: Float64
        var high: Float64

        if joint_idx == HalfCheetahPlanarConstants.JOINT_BTHIGH:
            low = HalfCheetahPlanarConstants.BTHIGH_LIMIT_LOW
            high = HalfCheetahPlanarConstants.BTHIGH_LIMIT_HIGH
        elif joint_idx == HalfCheetahPlanarConstants.JOINT_BSHIN:
            low = HalfCheetahPlanarConstants.BSHIN_LIMIT_LOW
            high = HalfCheetahPlanarConstants.BSHIN_LIMIT_HIGH
        elif joint_idx == HalfCheetahPlanarConstants.JOINT_BFOOT:
            low = HalfCheetahPlanarConstants.BFOOT_LIMIT_LOW
            high = HalfCheetahPlanarConstants.BFOOT_LIMIT_HIGH
        elif joint_idx == HalfCheetahPlanarConstants.JOINT_FTHIGH:
            low = HalfCheetahPlanarConstants.FTHIGH_LIMIT_LOW
            high = HalfCheetahPlanarConstants.FTHIGH_LIMIT_HIGH
        elif joint_idx == HalfCheetahPlanarConstants.JOINT_FSHIN:
            low = HalfCheetahPlanarConstants.FSHIN_LIMIT_LOW
            high = HalfCheetahPlanarConstants.FSHIN_LIMIT_HIGH
        else:  # FFOOT
            low = HalfCheetahPlanarConstants.FFOOT_LIMIT_LOW
            high = HalfCheetahPlanarConstants.FFOOT_LIMIT_HIGH

        if self.joint_angles[joint_idx] < low:
            self.joint_angles[joint_idx] = low
            self.joint_velocities[joint_idx] = 0.0
        elif self.joint_angles[joint_idx] > high:
            self.joint_angles[joint_idx] = high
            self.joint_velocities[joint_idx] = 0.0

    fn _update_body_positions(mut self):
        """Update body positions via forward kinematics."""
        var torso_base = HalfCheetahPlanarConstants.BODY_TORSO * BODY_STATE_SIZE
        var torso_x = self.bodies[torso_base + IDX_X]
        var torso_y = self.bodies[torso_base + IDX_Y]

        torso_x += (
            self.bodies[torso_base + IDX_VX] * HalfCheetahPlanarConstants.DT
        )
        torso_y += (
            self.bodies[torso_base + IDX_VY] * HalfCheetahPlanarConstants.DT
        )
        self.bodies[torso_base + IDX_X] = torso_x
        self.bodies[torso_base + IDX_Y] = torso_y

        # Update back leg
        var back_hip_x = torso_x - HalfCheetahPlanarConstants.TORSO_LENGTH / 2
        var back_hip_y = torso_y

        self._update_leg_positions(
            back_hip_x,
            back_hip_y,
            HalfCheetahPlanarConstants.JOINT_BTHIGH,
            HalfCheetahPlanarConstants.JOINT_BSHIN,
            HalfCheetahPlanarConstants.JOINT_BFOOT,
            HalfCheetahPlanarConstants.BODY_BTHIGH,
            HalfCheetahPlanarConstants.BODY_BSHIN,
            HalfCheetahPlanarConstants.BODY_BFOOT,
            HalfCheetahPlanarConstants.BTHIGH_LENGTH,
            HalfCheetahPlanarConstants.BSHIN_LENGTH,
            HalfCheetahPlanarConstants.BFOOT_LENGTH,
        )

        # Update front leg
        var front_hip_x = torso_x + HalfCheetahPlanarConstants.TORSO_LENGTH / 2
        var front_hip_y = torso_y

        self._update_leg_positions(
            front_hip_x,
            front_hip_y,
            HalfCheetahPlanarConstants.JOINT_FTHIGH,
            HalfCheetahPlanarConstants.JOINT_FSHIN,
            HalfCheetahPlanarConstants.JOINT_FFOOT,
            HalfCheetahPlanarConstants.BODY_FTHIGH,
            HalfCheetahPlanarConstants.BODY_FSHIN,
            HalfCheetahPlanarConstants.BODY_FFOOT,
            HalfCheetahPlanarConstants.FTHIGH_LENGTH,
            HalfCheetahPlanarConstants.FSHIN_LENGTH,
            HalfCheetahPlanarConstants.FFOOT_LENGTH,
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
        var torso_base = HalfCheetahPlanarConstants.BODY_TORSO * BODY_STATE_SIZE

        # Check back foot
        var bfoot_base = HalfCheetahPlanarConstants.BODY_BFOOT * BODY_STATE_SIZE
        var bfoot_y = self.bodies[bfoot_base + IDX_Y]
        if bfoot_y - HalfCheetahPlanarConstants.BFOOT_RADIUS < ground_y:
            var penetration = ground_y - (
                bfoot_y - HalfCheetahPlanarConstants.BFOOT_RADIUS
            )
            self.bodies[torso_base + IDX_Y] += penetration
            if self.bodies[torso_base + IDX_VY] < 0:
                self.bodies[torso_base + IDX_VY] = 0.0

        # Check front foot
        var ffoot_base = HalfCheetahPlanarConstants.BODY_FFOOT * BODY_STATE_SIZE
        var ffoot_y = self.bodies[ffoot_base + IDX_Y]
        if ffoot_y - HalfCheetahPlanarConstants.FFOOT_RADIUS < ground_y:
            var penetration = ground_y - (
                ffoot_y - HalfCheetahPlanarConstants.FFOOT_RADIUS
            )
            self.bodies[torso_base + IDX_Y] += penetration
            if self.bodies[torso_base + IDX_VY] < 0:
                self.bodies[torso_base + IDX_VY] = 0.0

    fn is_done(self) -> Bool:
        """Check if episode is done."""
        return self.done

    fn get_total_reward(self) -> Float64:
        """Get total accumulated reward."""
        return self.total_reward
