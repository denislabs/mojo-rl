"""Planar Hopper Environment.

A 2D planar version of the MuJoCo Hopper environment.
The hopper consists of 4 bodies (torso, thigh, leg, foot) connected by 3 hinge joints.

Observation space (11D):
    [0]: z position of torso (height)
    [1]: angle of torso
    [2]: angle of thigh joint
    [3]: angle of leg joint
    [4]: angle of foot joint
    [5]: velocity of x
    [6]: velocity of z (vertical)
    [7]: angular velocity of torso
    [8]: angular velocity of thigh joint
    [9]: angular velocity of leg joint
    [10]: angular velocity of foot joint

Action space (3D):
    [0]: torque applied at thigh (hip)
    [1]: torque applied at leg (knee)
    [2]: torque applied at foot (ankle)

Reward:
    forward_reward = forward_velocity
    healthy_reward = 1.0 (if healthy)
    ctrl_cost = 0.001 * sum(action^2)
    reward = forward_reward + healthy_reward - ctrl_cost

Done:
    - z position of torso < 0.7 (fallen)
    - abs(angle) > 0.2 (tilted too much)
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
    HOPPER_NUM_BODIES,
    HOPPER_NUM_JOINTS,
)


struct HopperPlanarConstants:
    """Constants for planar Hopper environment."""

    # ==========================================================================
    # Physics Parameters (matching MuJoCo)
    # ==========================================================================

    comptime DT: Float64 = 0.008  # MuJoCo default timestep
    comptime FRAME_SKIP: Int = 4  # Steps per action
    comptime GRAVITY: Float64 = -9.81

    # ==========================================================================
    # Body Geometry (from MuJoCo Hopper XML)
    # ==========================================================================

    # Torso: capsule with length 0.4, radius 0.05
    comptime TORSO_LENGTH: Float64 = 0.4
    comptime TORSO_RADIUS: Float64 = 0.05
    comptime TORSO_MASS: Float64 = 3.53429174

    # Thigh: capsule with length 0.45, radius 0.05
    comptime THIGH_LENGTH: Float64 = 0.45
    comptime THIGH_RADIUS: Float64 = 0.05
    comptime THIGH_MASS: Float64 = 3.92699082

    # Leg: capsule with length 0.5, radius 0.04
    comptime LEG_LENGTH: Float64 = 0.5
    comptime LEG_RADIUS: Float64 = 0.04
    comptime LEG_MASS: Float64 = 2.71433605

    # Foot: capsule with length 0.39, radius 0.06
    comptime FOOT_LENGTH: Float64 = 0.39
    comptime FOOT_RADIUS: Float64 = 0.06
    comptime FOOT_MASS: Float64 = 5.0893801

    # Initial height (bottom of foot at ground)
    comptime INIT_HEIGHT: Float64 = 1.25

    # ==========================================================================
    # Body Indices
    # ==========================================================================

    comptime BODY_TORSO: Int = 0
    comptime BODY_THIGH: Int = 1
    comptime BODY_LEG: Int = 2
    comptime BODY_FOOT: Int = 3

    # ==========================================================================
    # Joint Indices
    # ==========================================================================

    comptime JOINT_HIP: Int = 0  # thigh-torso
    comptime JOINT_KNEE: Int = 1  # leg-thigh
    comptime JOINT_ANKLE: Int = 2  # foot-leg

    # ==========================================================================
    # Joint Limits (radians)
    # ==========================================================================

    comptime HIP_LIMIT_LOW: Float64 = -2.61799  # -150 degrees
    comptime HIP_LIMIT_HIGH: Float64 = 0.0

    comptime KNEE_LIMIT_LOW: Float64 = -2.61799  # -150 degrees
    comptime KNEE_LIMIT_HIGH: Float64 = 0.0

    comptime ANKLE_LIMIT_LOW: Float64 = -0.785398  # -45 degrees
    comptime ANKLE_LIMIT_HIGH: Float64 = 0.785398  # 45 degrees

    # ==========================================================================
    # Motor Parameters
    # ==========================================================================

    comptime MAX_TORQUE: Float64 = 200.0
    comptime GEAR_RATIO: Float64 = 200.0  # gear ratio from action to torque

    # ==========================================================================
    # Observation and Action Dimensions
    # ==========================================================================

    comptime OBS_DIM: Int = 11
    comptime ACTION_DIM: Int = 3

    # ==========================================================================
    # Reward Parameters
    # ==========================================================================

    comptime HEALTHY_REWARD: Float64 = 1.0
    comptime CTRL_COST_WEIGHT: Float64 = 0.001
    comptime FORWARD_REWARD_WEIGHT: Float64 = 1.0

    # ==========================================================================
    # Termination Conditions
    # ==========================================================================

    comptime MIN_Z: Float64 = 0.7  # Minimum height before termination
    comptime MAX_ANGLE: Float64 = 0.2  # Maximum torso angle before termination

    # ==========================================================================
    # Counts
    # ==========================================================================

    comptime NUM_BODIES: Int = 4
    comptime NUM_JOINTS: Int = 3


struct HopperPlanar:
    """Planar Hopper locomotion environment.

    A 2D version of the MuJoCo Hopper using physics2d infrastructure.
    """

    # State arrays
    var bodies: List[
        Float64
    ]  # Body states: [x, y, angle, vx, vy, omega, inv_mass, inv_inertia]
    var joint_angles: List[Float64]  # Joint angles
    var joint_velocities: List[Float64]  # Joint angular velocities

    # Environment state
    var step_count: Int
    var done: Bool
    var total_reward: Float64
    var prev_x: Float64  # For computing forward velocity

    fn __init__(out self):
        """Initialize the Hopper environment."""
        self.bodies = List[Float64]()
        self.joint_angles = List[Float64]()
        self.joint_velocities = List[Float64]()
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0
        self.prev_x = 0.0

        # Allocate body state array
        for _ in range(HopperPlanarConstants.NUM_BODIES * BODY_STATE_SIZE):
            self.bodies.append(0.0)

        # Allocate joint arrays
        for _ in range(HopperPlanarConstants.NUM_JOINTS):
            self.joint_angles.append(0.0)
            self.joint_velocities.append(0.0)

        # Initialize to reset state
        self._reset_state()

    fn _reset_state(mut self):
        """Reset the hopper to initial standing position."""
        # Reset counters
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0

        # Initialize bodies from top to bottom
        # Torso at initial height
        var torso_y = HopperPlanarConstants.INIT_HEIGHT
        self._init_body(
            HopperPlanarConstants.BODY_TORSO,
            0.0,
            torso_y,
            0.0,  # x, y, angle
            0.0,
            0.0,
            0.0,  # vx, vy, omega
            HopperPlanarConstants.TORSO_MASS,
        )

        # Thigh below torso
        var thigh_y = (
            torso_y
            - HopperPlanarConstants.TORSO_LENGTH / 2
            - HopperPlanarConstants.THIGH_LENGTH / 2
        )
        self._init_body(
            HopperPlanarConstants.BODY_THIGH,
            0.0,
            thigh_y,
            0.0,
            0.0,
            0.0,
            0.0,
            HopperPlanarConstants.THIGH_MASS,
        )

        # Leg below thigh
        var leg_y = (
            thigh_y
            - HopperPlanarConstants.THIGH_LENGTH / 2
            - HopperPlanarConstants.LEG_LENGTH / 2
        )
        self._init_body(
            HopperPlanarConstants.BODY_LEG,
            0.0,
            leg_y,
            0.0,
            0.0,
            0.0,
            0.0,
            HopperPlanarConstants.LEG_MASS,
        )

        # Foot below leg
        var foot_y = (
            leg_y
            - HopperPlanarConstants.LEG_LENGTH / 2
            - HopperPlanarConstants.FOOT_LENGTH / 2
        )
        self._init_body(
            HopperPlanarConstants.BODY_FOOT,
            0.0,
            foot_y,
            0.0,
            0.0,
            0.0,
            0.0,
            HopperPlanarConstants.FOOT_MASS,
        )

        # Reset joint angles and velocities
        for i in range(HopperPlanarConstants.NUM_JOINTS):
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

        # Approximate inertia for capsule
        var inertia = mass * 0.1  # Simplified
        self.bodies[base + IDX_INV_INERTIA] = (
            1.0 / inertia if inertia > 0.0 else 0.0
        )

    fn reset(mut self) -> List[Float64]:
        """Reset the environment and return initial observation.

        Returns:
            Initial observation (11D).
        """
        self._reset_state()

        # Add small random noise to initial state
        for i in range(HopperPlanarConstants.NUM_JOINTS):
            self.joint_angles[i] = (random_float64() - 0.5) * 0.01
            self.joint_velocities[i] = (random_float64() - 0.5) * 0.01

        return self.get_observation()

    fn get_observation(self) -> List[Float64]:
        """Get the current observation.

        Returns:
            Observation vector (11D).
        """
        var obs = List[Float64]()

        # Get torso state
        var torso_base = HopperPlanarConstants.BODY_TORSO * BODY_STATE_SIZE
        var torso_y = self.bodies[torso_base + IDX_Y]
        var torso_angle = self.bodies[torso_base + IDX_ANGLE]
        var torso_vx = self.bodies[torso_base + IDX_VX]
        var torso_vy = self.bodies[torso_base + IDX_VY]
        var torso_omega = self.bodies[torso_base + IDX_OMEGA]

        # Position observations
        obs.append(torso_y)  # [0] z position (height)
        obs.append(torso_angle)  # [1] torso angle

        # Joint angles
        obs.append(
            self.joint_angles[HopperPlanarConstants.JOINT_HIP]
        )  # [2] hip
        obs.append(
            self.joint_angles[HopperPlanarConstants.JOINT_KNEE]
        )  # [3] knee
        obs.append(
            self.joint_angles[HopperPlanarConstants.JOINT_ANKLE]
        )  # [4] ankle

        # Velocities
        obs.append(torso_vx)  # [5] x velocity
        obs.append(torso_vy)  # [6] z velocity (vertical)
        obs.append(torso_omega)  # [7] angular velocity

        # Joint velocities
        obs.append(
            self.joint_velocities[HopperPlanarConstants.JOINT_HIP]
        )  # [8]
        obs.append(
            self.joint_velocities[HopperPlanarConstants.JOINT_KNEE]
        )  # [9]
        obs.append(
            self.joint_velocities[HopperPlanarConstants.JOINT_ANKLE]
        )  # [10]

        return obs^

    fn step(
        mut self,
        action: List[Float64],
    ) -> Tuple[List[Float64], Float64, Bool]:
        """Take a step in the environment.

        Args:
            action: Action vector (3D) - torques for hip, knee, ankle.

        Returns:
            Tuple of (observation, reward, done).
        """
        # Clamp and scale actions
        var torques = List[Float64]()
        for i in range(HopperPlanarConstants.ACTION_DIM):
            var a = action[i] if i < len(action) else 0.0
            # Clamp to [-1, 1]
            if a > 1.0:
                a = 1.0
            elif a < -1.0:
                a = -1.0
            # Scale by gear ratio
            torques.append(a * HopperPlanarConstants.GEAR_RATIO)

        # Get current x position for reward calculation
        var torso_base = HopperPlanarConstants.BODY_TORSO * BODY_STATE_SIZE
        var x_before = self.bodies[torso_base + IDX_X]

        # Apply physics for frame_skip steps
        for _ in range(HopperPlanarConstants.FRAME_SKIP):
            self._physics_step(torques)

        # Get new state
        var x_after = self.bodies[torso_base + IDX_X]
        var torso_y = self.bodies[torso_base + IDX_Y]
        var torso_angle = self.bodies[torso_base + IDX_ANGLE]

        # Compute reward
        var forward_velocity = (x_after - x_before) / (
            HopperPlanarConstants.DT * Float64(HopperPlanarConstants.FRAME_SKIP)
        )
        var forward_reward = (
            HopperPlanarConstants.FORWARD_REWARD_WEIGHT * forward_velocity
        )

        # Control cost
        var ctrl_cost = 0.0
        for i in range(len(action)):
            var a = action[i] if action[i] >= -1.0 and action[i] <= 1.0 else (
                1.0 if action[i] > 0 else -1.0
            )
            ctrl_cost += a * a
        ctrl_cost *= HopperPlanarConstants.CTRL_COST_WEIGHT

        # Check if healthy
        var is_healthy = (
            torso_y >= HopperPlanarConstants.MIN_Z
            and abs(torso_angle) <= HopperPlanarConstants.MAX_ANGLE
        )
        var healthy_reward = (
            HopperPlanarConstants.HEALTHY_REWARD if is_healthy else 0.0
        )

        # Total reward
        var reward = forward_reward + healthy_reward - ctrl_cost

        # Check termination
        self.done = not is_healthy
        self.step_count += 1
        self.total_reward += reward

        return (self.get_observation(), reward, self.done)

    fn _physics_step(mut self, torques: List[Float64]):
        """Perform one physics simulation step.

        This is a simplified physics model for the planar hopper.
        In a full implementation, this would use the physics2d infrastructure.
        """
        var dt = HopperPlanarConstants.DT

        # Apply torques to joint velocities
        for i in range(HopperPlanarConstants.NUM_JOINTS):
            var torque = torques[i] if i < len(torques) else 0.0
            # Simplified: torque directly affects joint angular velocity
            # In reality, this would go through the constraint solver
            self.joint_velocities[i] += torque * dt * 0.01

        # Update joint angles
        for i in range(HopperPlanarConstants.NUM_JOINTS):
            self.joint_angles[i] += self.joint_velocities[i] * dt

            # Clamp joint angles to limits
            if i == HopperPlanarConstants.JOINT_HIP:
                if self.joint_angles[i] < HopperPlanarConstants.HIP_LIMIT_LOW:
                    self.joint_angles[i] = HopperPlanarConstants.HIP_LIMIT_LOW
                    self.joint_velocities[i] = 0.0
                elif (
                    self.joint_angles[i] > HopperPlanarConstants.HIP_LIMIT_HIGH
                ):
                    self.joint_angles[i] = HopperPlanarConstants.HIP_LIMIT_HIGH
                    self.joint_velocities[i] = 0.0
            elif i == HopperPlanarConstants.JOINT_KNEE:
                if self.joint_angles[i] < HopperPlanarConstants.KNEE_LIMIT_LOW:
                    self.joint_angles[i] = HopperPlanarConstants.KNEE_LIMIT_LOW
                    self.joint_velocities[i] = 0.0
                elif (
                    self.joint_angles[i] > HopperPlanarConstants.KNEE_LIMIT_HIGH
                ):
                    self.joint_angles[i] = HopperPlanarConstants.KNEE_LIMIT_HIGH
                    self.joint_velocities[i] = 0.0
            elif i == HopperPlanarConstants.JOINT_ANKLE:
                if self.joint_angles[i] < HopperPlanarConstants.ANKLE_LIMIT_LOW:
                    self.joint_angles[i] = HopperPlanarConstants.ANKLE_LIMIT_LOW
                    self.joint_velocities[i] = 0.0
                elif (
                    self.joint_angles[i]
                    > HopperPlanarConstants.ANKLE_LIMIT_HIGH
                ):
                    self.joint_angles[
                        i
                    ] = HopperPlanarConstants.ANKLE_LIMIT_HIGH
                    self.joint_velocities[i] = 0.0

        # Apply gravity to torso
        var torso_base = HopperPlanarConstants.BODY_TORSO * BODY_STATE_SIZE
        self.bodies[torso_base + IDX_VY] += HopperPlanarConstants.GRAVITY * dt

        # Simple forward kinematics to update body positions
        self._update_body_positions()

        # Simple ground collision
        self._handle_ground_collision()

    fn _update_body_positions(mut self):
        """Update body positions based on joint angles (forward kinematics)."""
        # Get torso position
        var torso_base = HopperPlanarConstants.BODY_TORSO * BODY_STATE_SIZE
        var torso_x = self.bodies[torso_base + IDX_X]
        var torso_y = self.bodies[torso_base + IDX_Y]
        var torso_angle = self.bodies[torso_base + IDX_ANGLE]

        # Integrate torso position
        torso_x += self.bodies[torso_base + IDX_VX] * HopperPlanarConstants.DT
        torso_y += self.bodies[torso_base + IDX_VY] * HopperPlanarConstants.DT
        self.bodies[torso_base + IDX_X] = torso_x
        self.bodies[torso_base + IDX_Y] = torso_y

        # Update thigh position based on hip joint
        var hip_angle = self.joint_angles[HopperPlanarConstants.JOINT_HIP]
        var thigh_base = HopperPlanarConstants.BODY_THIGH * BODY_STATE_SIZE
        var thigh_length = HopperPlanarConstants.THIGH_LENGTH

        # Thigh hangs from torso bottom
        var joint_x = torso_x
        var joint_y = torso_y - HopperPlanarConstants.TORSO_LENGTH / 2

        self.bodies[thigh_base + IDX_X] = (
            joint_x + sin(hip_angle) * thigh_length / 2
        )
        self.bodies[thigh_base + IDX_Y] = (
            joint_y - cos(hip_angle) * thigh_length / 2
        )
        self.bodies[thigh_base + IDX_ANGLE] = hip_angle

        # Update leg position based on knee joint
        var knee_angle = self.joint_angles[HopperPlanarConstants.JOINT_KNEE]
        var leg_base = HopperPlanarConstants.BODY_LEG * BODY_STATE_SIZE
        var leg_length = HopperPlanarConstants.LEG_LENGTH

        # Leg hangs from thigh bottom
        var knee_x = (
            self.bodies[thigh_base + IDX_X] + sin(hip_angle) * thigh_length / 2
        )
        var knee_y = (
            self.bodies[thigh_base + IDX_Y] - cos(hip_angle) * thigh_length / 2
        )

        var total_angle = hip_angle + knee_angle
        self.bodies[leg_base + IDX_X] = (
            knee_x + sin(total_angle) * leg_length / 2
        )
        self.bodies[leg_base + IDX_Y] = (
            knee_y - cos(total_angle) * leg_length / 2
        )
        self.bodies[leg_base + IDX_ANGLE] = total_angle

        # Update foot position based on ankle joint
        var ankle_angle = self.joint_angles[HopperPlanarConstants.JOINT_ANKLE]
        var foot_base = HopperPlanarConstants.BODY_FOOT * BODY_STATE_SIZE
        var foot_length = HopperPlanarConstants.FOOT_LENGTH

        # Foot hangs from leg bottom
        var ankle_x = (
            self.bodies[leg_base + IDX_X] + sin(total_angle) * leg_length / 2
        )
        var ankle_y = (
            self.bodies[leg_base + IDX_Y] - cos(total_angle) * leg_length / 2
        )

        var foot_angle = total_angle + ankle_angle
        self.bodies[foot_base + IDX_X] = (
            ankle_x + sin(foot_angle) * foot_length / 2
        )
        self.bodies[foot_base + IDX_Y] = (
            ankle_y - cos(foot_angle) * foot_length / 2
        )
        self.bodies[foot_base + IDX_ANGLE] = foot_angle

    fn _handle_ground_collision(mut self):
        """Handle collision with ground plane at y=0."""
        var ground_y = 0.0

        # Check foot collision
        var foot_base = HopperPlanarConstants.BODY_FOOT * BODY_STATE_SIZE
        var foot_y = self.bodies[foot_base + IDX_Y]
        var foot_radius = HopperPlanarConstants.FOOT_RADIUS

        if foot_y - foot_radius < ground_y:
            # Foot touching ground - stop vertical motion
            var penetration = ground_y - (foot_y - foot_radius)

            # Push torso up by penetration
            var torso_base = HopperPlanarConstants.BODY_TORSO * BODY_STATE_SIZE
            self.bodies[torso_base + IDX_Y] += penetration

            # Stop downward velocity
            if self.bodies[torso_base + IDX_VY] < 0:
                self.bodies[torso_base + IDX_VY] = 0.0

            # Add forward velocity from foot contact (simplified friction)
            var foot_vx = self.bodies[foot_base + IDX_VX]
            self.bodies[torso_base + IDX_VX] += foot_vx * 0.1

    fn is_done(self) -> Bool:
        """Check if episode is done."""
        return self.done

    fn get_total_reward(self) -> Float64:
        """Get total accumulated reward."""
        return self.total_reward
