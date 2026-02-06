"""Bipedal Walker Environment for RL (Phase 10a).

A minimal bipedal locomotion environment using the physics engine:
- 1 Torso (sphere) + 2 Legs (vertical capsules)
- 2 actuated hinge joints (left hip, right hip)
- Ground contact with friction

Physical Configuration:
              Torso (body 0)
             mass=1.0, sphere
               radius=0.2
                  *
                 /|
    Left Hip   / |   Right Hip
   (joint 0)  /  |  (joint 1)
             |   |
             |   |
     Left Leg    Right Leg
     (body 1)    (body 2)
    mass=0.3     mass=0.3
    capsule      capsule
    r=0.04       r=0.04
    hl=0.10      hl=0.10

  ============================== Ground (z=0)
"""

from math import sqrt, sin, cos, atan2

from ..types import Model, Data
from ..integrator import ImpulseIntegrator
from ..joints import get_joint_angle, get_joint_angular_velocity


comptime PI: Float64 = 3.14159265358979323846


struct WalkerEnv[DTYPE: DType = DType.float64]:
    """Bipedal walker environment for reinforcement learning.

    Physical Configuration:
        - Body 0 (Torso): Sphere (mass=1.0, radius=0.20)
        - Body 1 (Left Leg): Vertical capsule (mass=0.3, radius=0.04, half_length=0.10)
        - Body 2 (Right Leg): Vertical capsule (mass=0.3, radius=0.04, half_length=0.10)
        - Joint 0 (Left Hip): Torso -> Left Leg, Y-axis rotation
        - Joint 1 (Right Hip): Torso -> Right Leg, Y-axis rotation

    Observation Space (12 dimensions):
        [0]  Torso height (z position)
        [1]  Torso x velocity
        [2]  Torso z velocity
        [3]  Torso pitch angle (rotation around Y-axis)
        [4]  Torso pitch angular velocity
        [5]  Left hip angle
        [6]  Left hip angular velocity
        [7]  Left leg ground contact (1.0 if contact, 0.0 otherwise)
        [8]  Right hip angle
        [9]  Right hip angular velocity
        [10] Right leg ground contact (1.0 if contact, 0.0 otherwise)
        [11] Torso roll angle (rotation around X-axis)

    Action Space (2 dimensions):
        [0] Left hip torque, normalized to [-1, 1], scaled by torque_limit
        [1] Right hip torque, normalized to [-1, 1], scaled by torque_limit

    Reward:
        reward = forward_velocity + alive_bonus - control_cost - height_penalty
        where:
        - forward_velocity = torso x velocity (encourages forward motion)
        - alive_bonus = 1.0 (if not terminated)
        - control_cost = 0.005 * (left_torque^2 + right_torque^2)
        - height_penalty = 0.5 * (torso_height - 0.45)^2

    Termination:
        Episode ends when:
        - Torso height < min_height (fallen)
        - |Torso pitch| > max_pitch (tipped forward/backward)
        - |Torso roll| > max_roll (tipped sideways)
        - Episode length > max_steps

    Parameters:
        DTYPE: Data type for physics (default float64).
    """

    # Physics
    var model: Model[Self.DTYPE, 3, 15, 2]
    var data: Data[Self.DTYPE, 3, 15, 2]

    # Environment parameters
    var torque_limit: Scalar[Self.DTYPE]
    var min_height: Scalar[Self.DTYPE]
    var max_pitch: Scalar[Self.DTYPE]
    var max_roll: Scalar[Self.DTYPE]
    var max_steps: Int
    var current_step: Int

    # Configuration
    var torso_mass: Scalar[Self.DTYPE]
    var torso_radius: Scalar[Self.DTYPE]
    var leg_mass: Scalar[Self.DTYPE]
    var leg_radius: Scalar[Self.DTYPE]
    var leg_half_length: Scalar[Self.DTYPE]  # Half-length of leg capsules
    var hip_offset_x: Scalar[Self.DTYPE]     # Horizontal offset from torso center to hip
    var hip_offset_z: Scalar[Self.DTYPE]     # Vertical offset from torso center to hip
    var hip_height: Scalar[Self.DTYPE]       # Height from leg center to hip (top of capsule)

    fn __init__(
        out self,
        torque_limit: Scalar[Self.DTYPE] = 15.0,
        min_height: Scalar[Self.DTYPE] = 0.15,
        max_pitch: Scalar[Self.DTYPE] = 1.0,  # ~57 degrees
        max_roll: Scalar[Self.DTYPE] = 0.5,   # ~29 degrees
        max_steps: Int = 1000,
        timestep: Scalar[Self.DTYPE] = 0.01,
        friction: Scalar[Self.DTYPE] = 0.8,
    ):
        """Initialize the walker environment.

        Args:
            torque_limit: Maximum hip torque in N*m (default 15.0).
            min_height: Minimum torso height before termination (default 0.15).
            max_pitch: Maximum torso pitch (radians) before termination (default 1.0).
            max_roll: Maximum torso roll (radians) before termination (default 0.5).
            max_steps: Maximum episode length (default 1000).
            timestep: Physics timestep in seconds (default 0.01).
            friction: Ground friction coefficient (default 0.8).
        """
        self.torque_limit = torque_limit
        self.min_height = min_height
        self.max_pitch = max_pitch
        self.max_roll = max_roll
        self.max_steps = max_steps
        self.current_step = 0

        # Body configuration
        self.torso_mass = Scalar[Self.DTYPE](1.0)
        self.torso_radius = Scalar[Self.DTYPE](0.20)
        self.leg_mass = Scalar[Self.DTYPE](0.3)
        self.leg_radius = Scalar[Self.DTYPE](0.04)       # Thin leg capsule
        self.leg_half_length = Scalar[Self.DTYPE](0.10)  # Capsule half-length
        self.hip_offset_x = Scalar[Self.DTYPE](0.10)     # 10cm from center
        self.hip_offset_z = Scalar[Self.DTYPE](0.20)     # 20cm below torso center
        # Hip is at top of leg capsule: half_length + radius above center
        self.hip_height = self.leg_half_length + self.leg_radius

        # Initialize physics model
        self.model = Model[Self.DTYPE, 3, 15, 2](
            gravity_z=Scalar[Self.DTYPE](-9.81),
            timestep=timestep,
            ground_z=Scalar[Self.DTYPE](0.0),
            friction=friction,
            restitution=Scalar[Self.DTYPE](0.0),
        )

        # Configure bodies
        # Torso: sphere
        self.model.set_body(0, mass=self.torso_mass, radius=self.torso_radius)
        # Left Leg: vertical capsule
        self.model.set_body_capsule(
            1,
            mass=self.leg_mass,
            radius=self.leg_radius,
            half_length=self.leg_half_length,
        )
        # Right Leg: vertical capsule
        self.model.set_body_capsule(
            2,
            mass=self.leg_mass,
            radius=self.leg_radius,
            half_length=self.leg_half_length,
        )

        # Add left hip joint: Torso (body 0) -> Left Leg (body 1)
        _ = self.model.add_hinge_joint(
            parent=0,  # Torso
            child=1,   # Left Leg
            anchor_parent=(
                -self.hip_offset_x,  # Left side of torso
                Scalar[Self.DTYPE](0.0),
                -self.hip_offset_z,  # Below torso center
            ),
            anchor_child=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                self.hip_height,  # Top of leg capsule (above center)
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),  # Y-axis rotation (sagittal plane)
                Scalar[Self.DTYPE](0.0),
            ),
        )

        # Add right hip joint: Torso (body 0) -> Right Leg (body 2)
        _ = self.model.add_hinge_joint(
            parent=0,  # Torso
            child=2,   # Right Leg
            anchor_parent=(
                self.hip_offset_x,  # Right side of torso
                Scalar[Self.DTYPE](0.0),
                -self.hip_offset_z,  # Below torso center
            ),
            anchor_child=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                self.hip_height,  # Top of leg capsule (above center)
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),  # Y-axis rotation (sagittal plane)
                Scalar[Self.DTYPE](0.0),
            ),
        )

        # Set torque limits
        self.model.joints[0].torque_limit = torque_limit
        self.model.joints[1].torque_limit = torque_limit

        # Initialize data
        self.data = Data[Self.DTYPE, 3, 15, 2]()
        self._reset_state()

    fn _reset_state(mut self):
        """Reset bodies to initial standing position."""
        # Calculate initial positions
        # Leg capsule center: positioned so lowest point touches ground
        # Lowest point at z = center_z - half_length - radius = 0
        # So center_z = half_length + radius
        var leg_z = self.leg_half_length + self.leg_radius

        # Torso above legs, connected by hips
        # Hip is at top of leg capsule: leg_z + hip_height
        # Then add hip_offset_z to get to torso center
        var torso_z = leg_z + self.hip_height + self.hip_offset_z

        # Set torso position (body 0) - centered at x=0
        self.data.set_body_position(0, 0.0, 0.0, torso_z)

        # Set left leg position (body 1) - offset in x
        self.data.set_body_position(1, -self.hip_offset_x, 0.0, leg_z)

        # Set right leg position (body 2) - offset in x
        self.data.set_body_position(2, self.hip_offset_x, 0.0, leg_z)

        # Reset velocities
        for i in range(3):
            self.data.set_body_velocity(i, 0.0, 0.0, 0.0)
            self.data.set_body_angular_velocity(i, 0.0, 0.0, 0.0)

        # Reset quaternions to identity
        for i in range(3):
            self.data.quaternions[i * 4 + 0] = 0.0
            self.data.quaternions[i * 4 + 1] = 0.0
            self.data.quaternions[i * 4 + 2] = 0.0
            self.data.quaternions[i * 4 + 3] = 1.0

        # Reset joint torques
        self.model.joints[0].target_torque = Scalar[Self.DTYPE](0.0)
        self.model.joints[1].target_torque = Scalar[Self.DTYPE](0.0)

        # Reset contact count
        self.data.num_contacts = 0

    fn reset(mut self) -> InlineArray[Scalar[Self.DTYPE], 12]:
        """Reset the environment to initial state.

        Returns:
            Initial observation (12 dimensions).
        """
        self._reset_state()
        self.current_step = 0
        return self.get_observation()

    fn step(
        mut self, action: InlineArray[Scalar[Self.DTYPE], 2]
    ) -> Tuple[
        InlineArray[Scalar[Self.DTYPE], 12],  # observation
        Scalar[Self.DTYPE],                    # reward
        Bool,                                   # terminated
        Bool,                                   # truncated
    ]:
        """Take one environment step.

        Args:
            action: Hip torques in range [-1, 1], will be scaled by torque_limit.
                [0] = left hip torque
                [1] = right hip torque

        Returns:
            Tuple of (observation, reward, terminated, truncated).
        """
        # Clamp and scale actions
        var left_action = action[0]
        var right_action = action[1]

        if left_action > 1.0:
            left_action = Scalar[Self.DTYPE](1.0)
        elif left_action < -1.0:
            left_action = Scalar[Self.DTYPE](-1.0)

        if right_action > 1.0:
            right_action = Scalar[Self.DTYPE](1.0)
        elif right_action < -1.0:
            right_action = Scalar[Self.DTYPE](-1.0)

        var left_torque = left_action * self.torque_limit
        var right_torque = right_action * self.torque_limit

        self.model.joints[0].target_torque = left_torque
        self.model.joints[1].target_torque = right_torque

        # Physics step
        ImpulseIntegrator.step(self.model, self.data)

        self.current_step += 1

        # Get observation
        var obs = self.get_observation()

        # Check termination
        var terminated = self._is_terminated(obs)
        var truncated = self.current_step >= self.max_steps

        # Compute reward
        var reward = self._compute_reward(obs, left_torque, right_torque, terminated)

        return (obs^, reward, terminated, truncated)

    fn get_observation(self) -> InlineArray[Scalar[Self.DTYPE], 12]:
        """Get current observation vector.

        Returns:
            12-dimensional observation:
            [0]  Torso height (z)
            [1]  Torso x velocity
            [2]  Torso z velocity
            [3]  Torso pitch angle
            [4]  Torso pitch angular velocity
            [5]  Left hip angle
            [6]  Left hip angular velocity
            [7]  Left leg ground contact
            [8]  Right hip angle
            [9]  Right hip angular velocity
            [10] Right leg ground contact
            [11] Torso roll angle
        """
        var obs = InlineArray[Scalar[Self.DTYPE], 12](uninitialized=True)

        # Torso position and velocity (body 0)
        var torso_pos = self.data.get_body_position(0)
        var torso_vel = self.data.get_body_velocity(0)

        obs[0] = torso_pos[2]  # Height (z)
        obs[1] = torso_vel[0]  # x velocity
        obs[2] = torso_vel[2]  # z velocity

        # Torso orientation from quaternion
        var qx = self.data.quaternions[0]
        var qy = self.data.quaternions[1]
        var qz = self.data.quaternions[2]
        var qw = self.data.quaternions[3]

        # Extract pitch (rotation around Y-axis)
        var sin_pitch = Scalar[Self.DTYPE](2.0) * (qw * qy - qz * qx)
        if sin_pitch > 1.0:
            sin_pitch = Scalar[Self.DTYPE](1.0)
        elif sin_pitch < -1.0:
            sin_pitch = Scalar[Self.DTYPE](-1.0)
        from math import asin
        obs[3] = asin(sin_pitch)

        # Torso pitch angular velocity (Y component)
        var torso_ang_vel = self.data.get_body_angular_velocity(0)
        obs[4] = torso_ang_vel[1]

        # Left hip joint (joint 0)
        obs[5] = get_joint_angle(self.model, self.data, 0)
        obs[6] = get_joint_angular_velocity(self.model, self.data, 0)

        # Left foot contact (body 1)
        var left_contact = False
        for c in range(self.data.num_contacts):
            var contact = self.data.contacts[c]
            if contact.body_a == 1 or contact.body_b == 1:
                left_contact = True
                break
        obs[7] = Scalar[Self.DTYPE](1.0) if left_contact else Scalar[Self.DTYPE](0.0)

        # Right hip joint (joint 1)
        obs[8] = get_joint_angle(self.model, self.data, 1)
        obs[9] = get_joint_angular_velocity(self.model, self.data, 1)

        # Right foot contact (body 2)
        var right_contact = False
        for c in range(self.data.num_contacts):
            var contact = self.data.contacts[c]
            if contact.body_a == 2 or contact.body_b == 2:
                right_contact = True
                break
        obs[10] = Scalar[Self.DTYPE](1.0) if right_contact else Scalar[Self.DTYPE](0.0)

        # Torso roll (rotation around X-axis)
        # Using quaternion: roll = atan2(2*(qw*qx + qy*qz), 1 - 2*(qx^2 + qy^2))
        var sin_roll_cos_pitch = Scalar[Self.DTYPE](2.0) * (qw * qx + qy * qz)
        var cos_roll_cos_pitch = Scalar[Self.DTYPE](1.0) - Scalar[Self.DTYPE](2.0) * (qx * qx + qy * qy)
        obs[11] = atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

        return obs^

    fn _is_terminated(self, obs: InlineArray[Scalar[Self.DTYPE], 12]) -> Bool:
        """Check if episode should terminate.

        Args:
            obs: Current observation.

        Returns:
            True if terminated (fallen or tipped).
        """
        # Check height
        var height = obs[0]
        if height < self.min_height:
            return True

        # Check pitch
        var pitch = obs[3]
        if pitch > self.max_pitch or pitch < -self.max_pitch:
            return True

        # Check roll
        var roll = obs[11]
        if roll > self.max_roll or roll < -self.max_roll:
            return True

        return False

    fn _compute_reward(
        self,
        obs: InlineArray[Scalar[Self.DTYPE], 12],
        left_torque: Scalar[Self.DTYPE],
        right_torque: Scalar[Self.DTYPE],
        terminated: Bool,
    ) -> Scalar[Self.DTYPE]:
        """Compute reward for current state.

        Args:
            obs: Current observation.
            left_torque: Applied left hip torque.
            right_torque: Applied right hip torque.
            terminated: Whether episode terminated.

        Returns:
            Scalar reward value.
        """
        # Forward velocity reward
        var forward_vel = obs[1]

        # Alive bonus (only if not terminated)
        var alive_bonus: Scalar[Self.DTYPE] = 0.0
        if not terminated:
            alive_bonus = Scalar[Self.DTYPE](1.0)

        # Control cost
        var control_cost = Scalar[Self.DTYPE](0.005) * (
            left_torque * left_torque + right_torque * right_torque
        )

        # Height penalty (penalize deviation from target height ~0.45)
        var target_height = Scalar[Self.DTYPE](0.45)
        var height_diff = obs[0] - target_height
        var height_penalty = Scalar[Self.DTYPE](0.5) * height_diff * height_diff

        return forward_vel + alive_bonus - control_cost - height_penalty

    fn get_torso_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get torso (body 0) position for visualization."""
        return self.data.get_body_position(0)

    fn get_left_leg_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get left leg (body 1) position for visualization."""
        return self.data.get_body_position(1)

    fn get_right_leg_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get right leg (body 2) position for visualization."""
        return self.data.get_body_position(2)

    fn get_left_hip_anchor_world(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get left hip joint anchor point in world coordinates for visualization."""
        var torso_pos = self.data.get_body_position(0)
        # For now, assume no rotation for simplicity
        return (
            torso_pos[0] - self.hip_offset_x,
            torso_pos[1],
            torso_pos[2] - self.hip_offset_z,
        )

    fn get_right_hip_anchor_world(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get right hip joint anchor point in world coordinates for visualization."""
        var torso_pos = self.data.get_body_position(0)
        # For now, assume no rotation for simplicity
        return (
            torso_pos[0] + self.hip_offset_x,
            torso_pos[1],
            torso_pos[2] - self.hip_offset_z,
        )
