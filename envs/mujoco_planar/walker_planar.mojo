"""Planar Walker2d Environment.

A 2D planar version of the MuJoCo Walker2d environment.
The walker consists of 7 bodies (torso, 2x thigh, 2x leg, 2x foot) connected by 6 hinge joints.

Observation space (17D):
    [0]: z position of torso (height)
    [1]: angle of torso
    [2]: angle of right thigh joint
    [3]: angle of right leg joint
    [4]: angle of right foot joint
    [5]: angle of left thigh joint
    [6]: angle of left leg joint
    [7]: angle of left foot joint
    [8]: velocity of x
    [9]: velocity of z (vertical)
    [10]: angular velocity of torso
    [11]: angular velocity of right thigh
    [12]: angular velocity of right leg
    [13]: angular velocity of right foot
    [14]: angular velocity of left thigh
    [15]: angular velocity of left leg
    [16]: angular velocity of left foot

Action space (6D):
    [0]: torque applied at right thigh (hip)
    [1]: torque applied at right leg (knee)
    [2]: torque applied at right foot (ankle)
    [3]: torque applied at left thigh (hip)
    [4]: torque applied at left leg (knee)
    [5]: torque applied at left foot (ankle)
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
    WALKER_NUM_BODIES,
    WALKER_NUM_JOINTS,
)


struct Walker2dPlanarConstants:
    """Constants for planar Walker2d environment."""

    # ==========================================================================
    # Physics Parameters
    # ==========================================================================

    comptime DT: Float64 = 0.008
    comptime FRAME_SKIP: Int = 4
    comptime GRAVITY: Float64 = -9.81

    # ==========================================================================
    # Body Geometry (from MuJoCo Walker2d XML)
    # ==========================================================================

    # Torso dimensions
    comptime TORSO_LENGTH: Float64 = 0.4
    comptime TORSO_RADIUS: Float64 = 0.05
    comptime TORSO_MASS: Float64 = 3.6651914

    # Thigh dimensions (same for both legs)
    comptime THIGH_LENGTH: Float64 = 0.45
    comptime THIGH_RADIUS: Float64 = 0.05
    comptime THIGH_MASS: Float64 = 4.0578904

    # Leg dimensions
    comptime LEG_LENGTH: Float64 = 0.5
    comptime LEG_RADIUS: Float64 = 0.04
    comptime LEG_MASS: Float64 = 2.7813568

    # Foot dimensions
    comptime FOOT_LENGTH: Float64 = 0.2
    comptime FOOT_RADIUS: Float64 = 0.06
    comptime FOOT_MASS: Float64 = 3.1667254

    # Initial height
    comptime INIT_HEIGHT: Float64 = 1.25

    # ==========================================================================
    # Body Indices
    # ==========================================================================

    comptime BODY_TORSO: Int = 0
    comptime BODY_THIGH_R: Int = 1
    comptime BODY_LEG_R: Int = 2
    comptime BODY_FOOT_R: Int = 3
    comptime BODY_THIGH_L: Int = 4
    comptime BODY_LEG_L: Int = 5
    comptime BODY_FOOT_L: Int = 6

    # ==========================================================================
    # Joint Indices
    # ==========================================================================

    comptime JOINT_HIP_R: Int = 0
    comptime JOINT_KNEE_R: Int = 1
    comptime JOINT_ANKLE_R: Int = 2
    comptime JOINT_HIP_L: Int = 3
    comptime JOINT_KNEE_L: Int = 4
    comptime JOINT_ANKLE_L: Int = 5

    # ==========================================================================
    # Joint Limits
    # ==========================================================================

    comptime HIP_LIMIT_LOW: Float64 = -1.0
    comptime HIP_LIMIT_HIGH: Float64 = 0.7

    comptime KNEE_LIMIT_LOW: Float64 = -1.2
    comptime KNEE_LIMIT_HIGH: Float64 = 0.0

    comptime ANKLE_LIMIT_LOW: Float64 = -0.5
    comptime ANKLE_LIMIT_HIGH: Float64 = 0.5

    # ==========================================================================
    # Motor Parameters
    # ==========================================================================

    comptime MAX_TORQUE: Float64 = 100.0
    comptime GEAR_RATIO: Float64 = 100.0

    # ==========================================================================
    # Observation and Action Dimensions
    # ==========================================================================

    comptime OBS_DIM: Int = 17
    comptime ACTION_DIM: Int = 6

    # ==========================================================================
    # Reward Parameters
    # ==========================================================================

    comptime HEALTHY_REWARD: Float64 = 1.0
    comptime CTRL_COST_WEIGHT: Float64 = 0.001
    comptime FORWARD_REWARD_WEIGHT: Float64 = 1.0

    # ==========================================================================
    # Termination Conditions
    # ==========================================================================

    comptime MIN_Z: Float64 = 0.8
    comptime MAX_Z: Float64 = 2.0
    comptime MAX_ANGLE: Float64 = 1.0

    # ==========================================================================
    # Counts
    # ==========================================================================

    comptime NUM_BODIES: Int = 7
    comptime NUM_JOINTS: Int = 6


struct Walker2dPlanar:
    """Planar Walker2d locomotion environment."""

    var bodies: List[Float64]
    var joint_angles: List[Float64]
    var joint_velocities: List[Float64]
    var step_count: Int
    var done: Bool
    var total_reward: Float64
    var prev_x: Float64

    fn __init__(out self):
        """Initialize the Walker2d environment."""
        self.bodies = List[Float64]()
        self.joint_angles = List[Float64]()
        self.joint_velocities = List[Float64]()
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0
        self.prev_x = 0.0

        for _ in range(Walker2dPlanarConstants.NUM_BODIES * BODY_STATE_SIZE):
            self.bodies.append(0.0)

        for _ in range(Walker2dPlanarConstants.NUM_JOINTS):
            self.joint_angles.append(0.0)
            self.joint_velocities.append(0.0)

        self._reset_state()

    fn _reset_state(mut self):
        """Reset to initial standing position."""
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0

        var torso_y = Walker2dPlanarConstants.INIT_HEIGHT
        self._init_body(
            Walker2dPlanarConstants.BODY_TORSO,
            0.0,
            torso_y,
            0.0,
            0.0,
            0.0,
            0.0,
            Walker2dPlanarConstants.TORSO_MASS,
        )

        # Right leg
        var thigh_y = (
            torso_y
            - Walker2dPlanarConstants.TORSO_LENGTH / 2
            - Walker2dPlanarConstants.THIGH_LENGTH / 2
        )
        self._init_body(
            Walker2dPlanarConstants.BODY_THIGH_R,
            0.1,
            thigh_y,
            0.0,
            0.0,
            0.0,
            0.0,
            Walker2dPlanarConstants.THIGH_MASS,
        )

        var leg_y = (
            thigh_y
            - Walker2dPlanarConstants.THIGH_LENGTH / 2
            - Walker2dPlanarConstants.LEG_LENGTH / 2
        )
        self._init_body(
            Walker2dPlanarConstants.BODY_LEG_R,
            0.1,
            leg_y,
            0.0,
            0.0,
            0.0,
            0.0,
            Walker2dPlanarConstants.LEG_MASS,
        )

        var foot_y = (
            leg_y
            - Walker2dPlanarConstants.LEG_LENGTH / 2
            - Walker2dPlanarConstants.FOOT_LENGTH / 2
        )
        self._init_body(
            Walker2dPlanarConstants.BODY_FOOT_R,
            0.1,
            foot_y,
            0.0,
            0.0,
            0.0,
            0.0,
            Walker2dPlanarConstants.FOOT_MASS,
        )

        # Left leg
        self._init_body(
            Walker2dPlanarConstants.BODY_THIGH_L,
            -0.1,
            thigh_y,
            0.0,
            0.0,
            0.0,
            0.0,
            Walker2dPlanarConstants.THIGH_MASS,
        )

        self._init_body(
            Walker2dPlanarConstants.BODY_LEG_L,
            -0.1,
            leg_y,
            0.0,
            0.0,
            0.0,
            0.0,
            Walker2dPlanarConstants.LEG_MASS,
        )

        self._init_body(
            Walker2dPlanarConstants.BODY_FOOT_L,
            -0.1,
            foot_y,
            0.0,
            0.0,
            0.0,
            0.0,
            Walker2dPlanarConstants.FOOT_MASS,
        )

        for i in range(Walker2dPlanarConstants.NUM_JOINTS):
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

        for i in range(Walker2dPlanarConstants.NUM_JOINTS):
            self.joint_angles[i] = (random_float64() - 0.5) * 0.01
            self.joint_velocities[i] = (random_float64() - 0.5) * 0.01

        return self.get_observation()

    fn get_observation(self) -> List[Float64]:
        """Get current observation (17D)."""
        var obs = List[Float64]()

        var torso_base = Walker2dPlanarConstants.BODY_TORSO * BODY_STATE_SIZE
        var torso_y = self.bodies[torso_base + IDX_Y]
        var torso_angle = self.bodies[torso_base + IDX_ANGLE]
        var torso_vx = self.bodies[torso_base + IDX_VX]
        var torso_vy = self.bodies[torso_base + IDX_VY]
        var torso_omega = self.bodies[torso_base + IDX_OMEGA]

        # Position observations
        obs.append(torso_y)  # [0]
        obs.append(torso_angle)  # [1]

        # Right leg joint angles
        obs.append(
            self.joint_angles[Walker2dPlanarConstants.JOINT_HIP_R]
        )  # [2]
        obs.append(
            self.joint_angles[Walker2dPlanarConstants.JOINT_KNEE_R]
        )  # [3]
        obs.append(
            self.joint_angles[Walker2dPlanarConstants.JOINT_ANKLE_R]
        )  # [4]

        # Left leg joint angles
        obs.append(
            self.joint_angles[Walker2dPlanarConstants.JOINT_HIP_L]
        )  # [5]
        obs.append(
            self.joint_angles[Walker2dPlanarConstants.JOINT_KNEE_L]
        )  # [6]
        obs.append(
            self.joint_angles[Walker2dPlanarConstants.JOINT_ANKLE_L]
        )  # [7]

        # Velocities
        obs.append(torso_vx)  # [8]
        obs.append(torso_vy)  # [9]
        obs.append(torso_omega)  # [10]

        # Right leg joint velocities
        obs.append(
            self.joint_velocities[Walker2dPlanarConstants.JOINT_HIP_R]
        )  # [11]
        obs.append(
            self.joint_velocities[Walker2dPlanarConstants.JOINT_KNEE_R]
        )  # [12]
        obs.append(
            self.joint_velocities[Walker2dPlanarConstants.JOINT_ANKLE_R]
        )  # [13]

        # Left leg joint velocities
        obs.append(
            self.joint_velocities[Walker2dPlanarConstants.JOINT_HIP_L]
        )  # [14]
        obs.append(
            self.joint_velocities[Walker2dPlanarConstants.JOINT_KNEE_L]
        )  # [15]
        obs.append(
            self.joint_velocities[Walker2dPlanarConstants.JOINT_ANKLE_L]
        )  # [16]

        return obs^

    fn step(
        mut self,
        action: List[Float64],
    ) -> Tuple[List[Float64], Float64, Bool]:
        """Take a step in the environment."""
        var torques = List[Float64]()
        for i in range(Walker2dPlanarConstants.ACTION_DIM):
            var a = action[i] if i < len(action) else 0.0
            if a > 1.0:
                a = 1.0
            elif a < -1.0:
                a = -1.0
            torques.append(a * Walker2dPlanarConstants.GEAR_RATIO)

        var torso_base = Walker2dPlanarConstants.BODY_TORSO * BODY_STATE_SIZE
        var x_before = self.bodies[torso_base + IDX_X]

        for _ in range(Walker2dPlanarConstants.FRAME_SKIP):
            self._physics_step(torques)

        var x_after = self.bodies[torso_base + IDX_X]
        var torso_y = self.bodies[torso_base + IDX_Y]
        var torso_angle = self.bodies[torso_base + IDX_ANGLE]

        var forward_velocity = (x_after - x_before) / (
            Walker2dPlanarConstants.DT
            * Float64(Walker2dPlanarConstants.FRAME_SKIP)
        )
        var forward_reward = (
            Walker2dPlanarConstants.FORWARD_REWARD_WEIGHT * forward_velocity
        )

        var ctrl_cost = 0.0
        for i in range(len(action)):
            var a = action[i] if action[i] >= -1.0 and action[i] <= 1.0 else (
                1.0 if action[i] > 0 else -1.0
            )
            ctrl_cost += a * a
        ctrl_cost *= Walker2dPlanarConstants.CTRL_COST_WEIGHT

        var is_healthy = (
            torso_y >= Walker2dPlanarConstants.MIN_Z
            and torso_y <= Walker2dPlanarConstants.MAX_Z
            and abs(torso_angle) <= Walker2dPlanarConstants.MAX_ANGLE
        )
        var healthy_reward = (
            Walker2dPlanarConstants.HEALTHY_REWARD if is_healthy else 0.0
        )

        var reward = forward_reward + healthy_reward - ctrl_cost

        self.done = not is_healthy
        self.step_count += 1
        self.total_reward += reward

        return (self.get_observation(), reward, self.done)

    fn _physics_step(mut self, torques: List[Float64]):
        """Perform one physics step."""
        var dt = Walker2dPlanarConstants.DT

        for i in range(Walker2dPlanarConstants.NUM_JOINTS):
            var torque = torques[i] if i < len(torques) else 0.0
            self.joint_velocities[i] += torque * dt * 0.01

        for i in range(Walker2dPlanarConstants.NUM_JOINTS):
            self.joint_angles[i] += self.joint_velocities[i] * dt
            self._clamp_joint(i)

        var torso_base = Walker2dPlanarConstants.BODY_TORSO * BODY_STATE_SIZE
        self.bodies[torso_base + IDX_VY] += Walker2dPlanarConstants.GRAVITY * dt

        self._update_body_positions()
        self._handle_ground_collision()

    fn _clamp_joint(mut self, joint_idx: Int):
        """Clamp joint angle to limits."""
        var is_hip = (
            joint_idx == Walker2dPlanarConstants.JOINT_HIP_R
            or joint_idx == Walker2dPlanarConstants.JOINT_HIP_L
        )
        var is_knee = (
            joint_idx == Walker2dPlanarConstants.JOINT_KNEE_R
            or joint_idx == Walker2dPlanarConstants.JOINT_KNEE_L
        )

        var low: Float64
        var high: Float64

        if is_hip:
            low = Walker2dPlanarConstants.HIP_LIMIT_LOW
            high = Walker2dPlanarConstants.HIP_LIMIT_HIGH
        elif is_knee:
            low = Walker2dPlanarConstants.KNEE_LIMIT_LOW
            high = Walker2dPlanarConstants.KNEE_LIMIT_HIGH
        else:
            low = Walker2dPlanarConstants.ANKLE_LIMIT_LOW
            high = Walker2dPlanarConstants.ANKLE_LIMIT_HIGH

        if self.joint_angles[joint_idx] < low:
            self.joint_angles[joint_idx] = low
            self.joint_velocities[joint_idx] = 0.0
        elif self.joint_angles[joint_idx] > high:
            self.joint_angles[joint_idx] = high
            self.joint_velocities[joint_idx] = 0.0

    fn _update_body_positions(mut self):
        """Update body positions via forward kinematics."""
        var torso_base = Walker2dPlanarConstants.BODY_TORSO * BODY_STATE_SIZE
        var torso_x = self.bodies[torso_base + IDX_X]
        var torso_y = self.bodies[torso_base + IDX_Y]

        torso_x += self.bodies[torso_base + IDX_VX] * Walker2dPlanarConstants.DT
        torso_y += self.bodies[torso_base + IDX_VY] * Walker2dPlanarConstants.DT
        self.bodies[torso_base + IDX_X] = torso_x
        self.bodies[torso_base + IDX_Y] = torso_y

        # Update right leg
        self._update_leg_positions(
            torso_x + 0.1,
            torso_y,
            Walker2dPlanarConstants.JOINT_HIP_R,
            Walker2dPlanarConstants.JOINT_KNEE_R,
            Walker2dPlanarConstants.JOINT_ANKLE_R,
            Walker2dPlanarConstants.BODY_THIGH_R,
            Walker2dPlanarConstants.BODY_LEG_R,
            Walker2dPlanarConstants.BODY_FOOT_R,
        )

        # Update left leg
        self._update_leg_positions(
            torso_x - 0.1,
            torso_y,
            Walker2dPlanarConstants.JOINT_HIP_L,
            Walker2dPlanarConstants.JOINT_KNEE_L,
            Walker2dPlanarConstants.JOINT_ANKLE_L,
            Walker2dPlanarConstants.BODY_THIGH_L,
            Walker2dPlanarConstants.BODY_LEG_L,
            Walker2dPlanarConstants.BODY_FOOT_L,
        )

    fn _update_leg_positions(
        mut self,
        hip_x: Float64,
        hip_y: Float64,
        hip_joint: Int,
        knee_joint: Int,
        ankle_joint: Int,
        thigh_body: Int,
        leg_body: Int,
        foot_body: Int,
    ):
        """Update positions for one leg."""
        var joint_y = hip_y - Walker2dPlanarConstants.TORSO_LENGTH / 2
        var hip_angle = self.joint_angles[hip_joint]

        var thigh_base = thigh_body * BODY_STATE_SIZE
        self.bodies[thigh_base + IDX_X] = (
            hip_x + sin(hip_angle) * Walker2dPlanarConstants.THIGH_LENGTH / 2
        )
        self.bodies[thigh_base + IDX_Y] = (
            joint_y - cos(hip_angle) * Walker2dPlanarConstants.THIGH_LENGTH / 2
        )
        self.bodies[thigh_base + IDX_ANGLE] = hip_angle

        var knee_x = (
            hip_x + sin(hip_angle) * Walker2dPlanarConstants.THIGH_LENGTH
        )
        var knee_y = (
            joint_y - cos(hip_angle) * Walker2dPlanarConstants.THIGH_LENGTH
        )
        var knee_angle = self.joint_angles[knee_joint]
        var total_angle = hip_angle + knee_angle

        var leg_base = leg_body * BODY_STATE_SIZE
        self.bodies[leg_base + IDX_X] = (
            knee_x + sin(total_angle) * Walker2dPlanarConstants.LEG_LENGTH / 2
        )
        self.bodies[leg_base + IDX_Y] = (
            knee_y - cos(total_angle) * Walker2dPlanarConstants.LEG_LENGTH / 2
        )
        self.bodies[leg_base + IDX_ANGLE] = total_angle

        var ankle_x = (
            knee_x + sin(total_angle) * Walker2dPlanarConstants.LEG_LENGTH
        )
        var ankle_y = (
            knee_y - cos(total_angle) * Walker2dPlanarConstants.LEG_LENGTH
        )
        var ankle_angle = self.joint_angles[ankle_joint]
        var foot_angle = total_angle + ankle_angle

        var foot_base = foot_body * BODY_STATE_SIZE
        self.bodies[foot_base + IDX_X] = (
            ankle_x + sin(foot_angle) * Walker2dPlanarConstants.FOOT_LENGTH / 2
        )
        self.bodies[foot_base + IDX_Y] = (
            ankle_y - cos(foot_angle) * Walker2dPlanarConstants.FOOT_LENGTH / 2
        )
        self.bodies[foot_base + IDX_ANGLE] = foot_angle

    fn _handle_ground_collision(mut self):
        """Handle ground collision for both feet."""
        var ground_y = 0.0
        var torso_base = Walker2dPlanarConstants.BODY_TORSO * BODY_STATE_SIZE

        # Check right foot
        var foot_r_base = Walker2dPlanarConstants.BODY_FOOT_R * BODY_STATE_SIZE
        var foot_r_y = self.bodies[foot_r_base + IDX_Y]
        if foot_r_y - Walker2dPlanarConstants.FOOT_RADIUS < ground_y:
            var penetration = ground_y - (
                foot_r_y - Walker2dPlanarConstants.FOOT_RADIUS
            )
            self.bodies[torso_base + IDX_Y] += penetration
            if self.bodies[torso_base + IDX_VY] < 0:
                self.bodies[torso_base + IDX_VY] = 0.0

        # Check left foot
        var foot_l_base = Walker2dPlanarConstants.BODY_FOOT_L * BODY_STATE_SIZE
        var foot_l_y = self.bodies[foot_l_base + IDX_Y]
        if foot_l_y - Walker2dPlanarConstants.FOOT_RADIUS < ground_y:
            var penetration = ground_y - (
                foot_l_y - Walker2dPlanarConstants.FOOT_RADIUS
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
