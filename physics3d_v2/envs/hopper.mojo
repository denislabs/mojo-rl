"""Simple 2-body Hopper Environment for RL.

A minimal locomotion environment using the physics engine:
- 2 spheres: Torso (larger) + Foot (smaller)
- 1 actuated hinge joint (hip)
- Ground contact with friction

This is designed to be a stepping stone toward more complex walkers.
"""

from math import sqrt, sin, cos, atan2

from ..types import Model, Data
from ..integrator import ImpulseIntegrator
from ..joints import get_joint_angle, get_joint_angular_velocity


comptime PI: Float64 = 3.14159265358979323846


struct HopperEnv[DTYPE: DType = DType.float64]:
    """Simple 2-body hopper environment for reinforcement learning.

    Physical Configuration:
        - Body 0 (Torso): Larger sphere (mass=1.0, radius=0.15)
        - Body 1 (Foot): Smaller sphere (mass=0.5, radius=0.1)
        - Joint 0 (Hip): Hinge connecting torso to foot, Y-axis rotation

    Observation Space (8 dimensions):
        [0] Torso height (z position)
        [1] Torso x velocity
        [2] Torso z velocity
        [3] Torso pitch angle (rotation around Y-axis, in radians)
        [4] Torso pitch angular velocity
        [5] Hip angle (relative angle between torso and foot)
        [6] Hip angular velocity
        [7] Foot ground contact (1.0 if contact, 0.0 otherwise)

    Action Space (1 dimension):
        [0] Hip torque, normalized to [-1, 1], scaled by torque_limit

    Reward:
        reward = forward_velocity + alive_bonus - control_cost
        where:
        - forward_velocity = torso x velocity (encourages forward motion)
        - alive_bonus = 1.0 (if not terminated)
        - control_cost = 0.01 * torque^2 (penalizes large actions)

    Termination:
        Episode ends when:
        - Torso height < min_height (fallen)
        - |Torso pitch| > max_pitch (tipped over)
        - Episode length > max_steps

    Parameters:
        DTYPE: Data type for physics (default float64).
    """

    # Physics
    var model: Model[Self.DTYPE, 2, 10, 1]
    var data: Data[Self.DTYPE, 2, 10, 1]

    # Environment parameters
    var torque_limit: Scalar[Self.DTYPE]
    var min_height: Scalar[Self.DTYPE]
    var max_pitch: Scalar[Self.DTYPE]
    var max_steps: Int
    var current_step: Int

    # Configuration
    var torso_mass: Scalar[Self.DTYPE]
    var torso_radius: Scalar[Self.DTYPE]
    var foot_mass: Scalar[Self.DTYPE]
    var foot_radius: Scalar[Self.DTYPE]
    var hip_height: Scalar[Self.DTYPE]  # Height of hip joint above foot center

    fn __init__(
        out self,
        torque_limit: Scalar[Self.DTYPE] = 10.0,
        min_height: Scalar[Self.DTYPE] = 0.15,
        max_pitch: Scalar[Self.DTYPE] = 1.0,  # ~57 degrees
        max_steps: Int = 1000,
        timestep: Scalar[Self.DTYPE] = 0.01,
        friction: Scalar[Self.DTYPE] = 0.8,
    ):
        """Initialize the hopper environment.

        Args:
            torque_limit: Maximum hip torque in N·m (default 10.0).
            min_height: Minimum torso height before termination (default 0.15).
            max_pitch: Maximum torso pitch (radians) before termination (default 1.0).
            max_steps: Maximum episode length (default 1000).
            timestep: Physics timestep in seconds (default 0.01).
            friction: Ground friction coefficient (default 0.8).
        """
        self.torque_limit = torque_limit
        self.min_height = min_height
        self.max_pitch = max_pitch
        self.max_steps = max_steps
        self.current_step = 0

        # Body configuration
        self.torso_mass = Scalar[Self.DTYPE](1.0)
        self.torso_radius = Scalar[Self.DTYPE](0.15)
        self.foot_mass = Scalar[Self.DTYPE](0.5)
        self.foot_radius = Scalar[Self.DTYPE](0.1)
        self.hip_height = Scalar[Self.DTYPE](0.2)  # Distance from foot center to hip

        # Initialize physics model
        self.model = Model[Self.DTYPE, 2, 10, 1](
            gravity_z=Scalar[Self.DTYPE](-9.81),
            timestep=timestep,
            ground_z=Scalar[Self.DTYPE](0.0),
            friction=friction,
            restitution=Scalar[Self.DTYPE](0.0),
        )

        # Configure bodies
        self.model.set_body(0, mass=self.torso_mass, radius=self.torso_radius)
        self.model.set_body(1, mass=self.foot_mass, radius=self.foot_radius)

        # Add hip joint: Torso (body 0) -> Foot (body 1)
        # Anchor at bottom of torso / top of leg segment
        _ = self.model.add_hinge_joint(
            parent=0,  # Torso
            child=1,  # Foot
            anchor_parent=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                -self.torso_radius,  # Bottom of torso
            ),
            anchor_child=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                self.hip_height,  # Above foot center
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),  # Y-axis rotation (sagittal plane)
                Scalar[Self.DTYPE](0.0),
            ),
        )

        # Set torque limit
        self.model.joints[0].torque_limit = torque_limit

        # Initialize data
        self.data = Data[Self.DTYPE, 2, 10, 1]()
        self._reset_state()

    fn _reset_state(mut self):
        """Reset bodies to initial standing position."""
        # Foot just above ground (radius height)
        var foot_z = self.foot_radius
        self.data.set_body_position(1, 0.0, 0.0, foot_z)

        # Torso above foot, connected by hip
        var torso_z = foot_z + self.hip_height + self.torso_radius
        self.data.set_body_position(0, 0.0, 0.0, torso_z)

        # Reset velocities
        self.data.set_body_velocity(0, 0.0, 0.0, 0.0)
        self.data.set_body_velocity(1, 0.0, 0.0, 0.0)
        self.data.set_body_angular_velocity(0, 0.0, 0.0, 0.0)
        self.data.set_body_angular_velocity(1, 0.0, 0.0, 0.0)

        # Reset quaternions to identity
        for i in range(2):
            self.data.quaternions[i * 4 + 0] = 0.0
            self.data.quaternions[i * 4 + 1] = 0.0
            self.data.quaternions[i * 4 + 2] = 0.0
            self.data.quaternions[i * 4 + 3] = 1.0

        # Reset joint torque
        self.model.joints[0].target_torque = Scalar[Self.DTYPE](0.0)

        # Reset contact count
        self.data.num_contacts = 0

    fn reset(mut self) -> InlineArray[Scalar[Self.DTYPE], 8]:
        """Reset the environment to initial state.

        Returns:
            Initial observation (8 dimensions).
        """
        self._reset_state()
        self.current_step = 0
        return self.get_observation()

    fn step(
        mut self, action: Scalar[Self.DTYPE]
    ) -> Tuple[
        InlineArray[Scalar[Self.DTYPE], 8],  # observation
        Scalar[Self.DTYPE],  # reward
        Bool,  # terminated
        Bool,  # truncated
    ]:
        """Take one environment step.

        Args:
            action: Hip torque in range [-1, 1], will be scaled by torque_limit.

        Returns:
            Tuple of (observation, reward, terminated, truncated).
        """
        # Clamp and scale action
        var clamped_action = action
        if clamped_action > 1.0:
            clamped_action = Scalar[Self.DTYPE](1.0)
        elif clamped_action < -1.0:
            clamped_action = Scalar[Self.DTYPE](-1.0)

        var torque = clamped_action * self.torque_limit
        self.model.joints[0].target_torque = torque

        # Physics step
        ImpulseIntegrator.step(self.model, self.data)

        self.current_step += 1

        # Get observation
        var obs = self.get_observation()

        # Check termination
        var terminated = self._is_terminated(obs)
        var truncated = self.current_step >= self.max_steps

        # Compute reward
        var reward = self._compute_reward(obs, torque, terminated)

        return (obs^, reward, terminated, truncated)

    fn get_observation(self) -> InlineArray[Scalar[Self.DTYPE], 8]:
        """Get current observation vector.

        Returns:
            8-dimensional observation:
            [0] Torso height (z)
            [1] Torso x velocity
            [2] Torso z velocity
            [3] Torso pitch angle
            [4] Torso pitch angular velocity
            [5] Hip angle
            [6] Hip angular velocity
            [7] Foot ground contact
        """
        var obs = InlineArray[Scalar[Self.DTYPE], 8](uninitialized=True)

        # Torso position and velocity
        var torso_pos = self.data.get_body_position(0)
        var torso_vel = self.data.get_body_velocity(0)

        obs[0] = torso_pos[2]  # Height (z)
        obs[1] = torso_vel[0]  # x velocity
        obs[2] = torso_vel[2]  # z velocity

        # Torso pitch (rotation around Y-axis)
        # Extract from quaternion: pitch = atan2(2*(qw*qy - qz*qx), 1 - 2*(qx² + qy²))
        var qx = self.data.quaternions[0]
        var qy = self.data.quaternions[1]
        var qz = self.data.quaternions[2]
        var qw = self.data.quaternions[3]

        var sin_pitch = Scalar[Self.DTYPE](2.0) * (qw * qy - qz * qx)
        # Clamp to avoid numerical issues with asin
        if sin_pitch > 1.0:
            sin_pitch = Scalar[Self.DTYPE](1.0)
        elif sin_pitch < -1.0:
            sin_pitch = Scalar[Self.DTYPE](-1.0)

        # Use asin for pitch (simpler than full euler extraction)
        from math import asin

        obs[3] = asin(sin_pitch)

        # Torso pitch angular velocity (Y component)
        var torso_ang_vel = self.data.get_body_angular_velocity(0)
        obs[4] = torso_ang_vel[1]

        # Hip joint angle and velocity
        obs[5] = get_joint_angle(self.model, self.data, 0)
        obs[6] = get_joint_angular_velocity(self.model, self.data, 0)

        # Foot ground contact
        # Check if foot (body 1) has any contacts
        var has_contact = False
        for c in range(self.data.num_contacts):
            var contact = self.data.contacts[c]
            if contact.body_a == 1 or contact.body_b == 1:
                has_contact = True
                break

        obs[7] = Scalar[Self.DTYPE](1.0) if has_contact else Scalar[Self.DTYPE](0.0)

        return obs^

    fn _is_terminated(self, obs: InlineArray[Scalar[Self.DTYPE], 8]) -> Bool:
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

        return False

    fn _compute_reward(
        self,
        obs: InlineArray[Scalar[Self.DTYPE], 8],
        torque: Scalar[Self.DTYPE],
        terminated: Bool,
    ) -> Scalar[Self.DTYPE]:
        """Compute reward for current state.

        Args:
            obs: Current observation.
            torque: Applied torque.
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
        var control_cost = Scalar[Self.DTYPE](0.01) * torque * torque

        return forward_vel + alive_bonus - control_cost

    fn get_torso_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get torso (body 0) position for visualization."""
        return self.data.get_body_position(0)

    fn get_foot_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get foot (body 1) position for visualization."""
        return self.data.get_body_position(1)

    fn get_hip_anchor_world(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get hip joint anchor point in world coordinates for visualization."""
        # Hip is at bottom of torso
        var torso_pos = self.data.get_body_position(0)
        # For now, assume no rotation for simplicity
        return (
            torso_pos[0],
            torso_pos[1],
            torso_pos[2] - self.torso_radius,
        )
