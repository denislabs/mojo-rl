"""4-Body Hopper Environment for RL (MuJoCo-like).

A realistic locomotion environment using the physics engine:
- Torso: Capsule (vertical) with root joints constraining to X-Z plane
- Thigh: Capsule (vertical)
- Leg: Capsule (vertical)
- Foot: Capsule (horizontal, rotated 90° around Y-axis)
- 3 root joints (rootx slide, rootz slide, rooty hinge)
- 3 actuated hinge joints (hip, knee, ankle)
- Ground contact with friction

This matches the MuJoCo Hopper structure:
- Root slide joints constrain torso to X-Z plane (no Y motion)
- Root hinge joint allows pitch rotation around Y-axis
- The torso has 3 DOF: X translation, Z translation, Y rotation
"""

from math import sqrt, sin, cos, atan2

from ..types import Model, Data
from ..integrator import ImpulseIntegrator, PGSIntegrator
from ..joints import (
    get_joint_angle,
    get_joint_angular_velocity,
    get_slide_joint_position,
    get_slide_joint_velocity,
)


comptime PI: Float64 = 3.14159265358979323846


struct HopperEnv[DTYPE: DType = DType.float64]:
    """4-body hopper environment for reinforcement learning.

    Physical Configuration (matching MuJoCo Hopper):
        - Body 0 (Torso): Capsule (mass=1.0, radius=0.05, half_length=0.2)
        - Body 1 (Thigh): Capsule (mass=0.5, radius=0.05, half_length=0.225)
        - Body 2 (Leg): Capsule (mass=0.3, radius=0.04, half_length=0.25)
        - Body 3 (Foot): Horizontal capsule (mass=0.2, radius=0.06, half_length=0.195)

    Root Joints (constraining torso to X-Z plane):
        - Slide Joint 0 (RootX): World -> Torso, X-axis translation
        - Slide Joint 1 (RootZ): World -> Torso, Z-axis translation
        - Hinge Joint 0 (RootY): World -> Torso, Y-axis rotation (pitch)

    Body Joints (actuated):
        - Hinge Joint 1 (Hip): Torso -> Thigh, Y-axis rotation
        - Hinge Joint 2 (Knee): Thigh -> Leg, Y-axis rotation
        - Hinge Joint 3 (Ankle): Leg -> Foot, Y-axis rotation

    Observation Space (11 dimensions, matching MuJoCo):
        [0] Torso height (z position)
        [1] Torso pitch angle (rotation around Y-axis)
        [2] Hip joint angle (thigh angle)
        [3] Knee joint angle (leg angle)
        [4] Ankle joint angle (foot angle)
        [5] Torso x velocity
        [6] Torso z velocity
        [7] Torso pitch angular velocity
        [8] Hip angular velocity
        [9] Knee angular velocity
        [10] Ankle angular velocity

    Action Space (3 dimensions):
        [0] Hip torque, normalized to [-1, 1], scaled by torque_limit
        [1] Knee torque
        [2] Ankle torque

    Reward:
        reward = forward_velocity + alive_bonus - control_cost
        where:
        - forward_velocity = torso x velocity (encourages forward motion)
        - alive_bonus = 1.0 (if not terminated)
        - control_cost = 0.001 * sum(torque^2) (penalizes large actions)

    Termination:
        Episode ends when:
        - Torso height < min_height (fallen)
        - |Torso pitch| > max_pitch (tipped over)
        - Episode length > max_steps

    Parameters:
        DTYPE: Data type for physics (default float64).
    """

    # Physics: 4 bodies, 20 max contacts, 4 hinge joints, 2 slide joints
    # Hinge: RootY (0), Hip (1), Knee (2), Ankle (3)
    # Slide: RootX (0), RootZ (1)
    var model: Model[Self.DTYPE, 4, 20, 4, 2]
    var data: Data[Self.DTYPE, 4, 20, 4, 2]

    # Environment parameters
    var torque_limit: Scalar[Self.DTYPE]
    var min_height: Scalar[Self.DTYPE]
    var max_pitch: Scalar[Self.DTYPE]
    var max_steps: Int
    var current_step: Int

    # Body dimensions (matching MuJoCo Hopper)
    var torso_mass: Scalar[Self.DTYPE]
    var torso_radius: Scalar[Self.DTYPE]
    var torso_half_length: Scalar[Self.DTYPE]

    var thigh_mass: Scalar[Self.DTYPE]
    var thigh_radius: Scalar[Self.DTYPE]
    var thigh_half_length: Scalar[Self.DTYPE]

    var leg_mass: Scalar[Self.DTYPE]
    var leg_radius: Scalar[Self.DTYPE]
    var leg_half_length: Scalar[Self.DTYPE]

    var foot_mass: Scalar[Self.DTYPE]
    var foot_radius: Scalar[Self.DTYPE]
    var foot_half_length: Scalar[Self.DTYPE]

    fn __init__(
        out self,
        torque_limit: Scalar[Self.DTYPE] = 200.0,  # MuJoCo uses gear=200
        min_height: Scalar[Self.DTYPE] = 0.7,
        max_pitch: Scalar[Self.DTYPE] = 1.0,  # ~57 degrees
        max_steps: Int = 1000,
        timestep: Scalar[Self.DTYPE] = 0.002,  # MuJoCo uses 0.002
        friction: Scalar[Self.DTYPE] = 0.9,
    ):
        """Initialize the 4-body hopper environment.

        Args:
            torque_limit: Maximum joint torque in N·m (default 200.0, matching MuJoCo gear).
            min_height: Minimum torso height before termination (default 0.7).
            max_pitch: Maximum torso pitch (radians) before termination (default 1.0).
            max_steps: Maximum episode length (default 1000).
            timestep: Physics timestep in seconds (default 0.002).
            friction: Ground friction coefficient (default 0.9).
        """
        self.torque_limit = torque_limit
        self.min_height = min_height
        self.max_pitch = max_pitch
        self.max_steps = max_steps
        self.current_step = 0

        # Body configuration (matching MuJoCo Hopper dimensions)
        self.torso_mass = Scalar[Self.DTYPE](1.0)
        self.torso_radius = Scalar[Self.DTYPE](0.05)
        self.torso_half_length = Scalar[Self.DTYPE](0.2)

        self.thigh_mass = Scalar[Self.DTYPE](0.5)
        self.thigh_radius = Scalar[Self.DTYPE](0.05)
        self.thigh_half_length = Scalar[Self.DTYPE](0.225)

        self.leg_mass = Scalar[Self.DTYPE](0.3)
        self.leg_radius = Scalar[Self.DTYPE](0.04)
        self.leg_half_length = Scalar[Self.DTYPE](0.25)

        self.foot_mass = Scalar[Self.DTYPE](0.2)
        self.foot_radius = Scalar[Self.DTYPE](0.06)
        self.foot_half_length = Scalar[Self.DTYPE](0.195)

        # Initialize physics model (4 hinge joints, 2 slide joints)
        self.model = Model[Self.DTYPE, 4, 20, 4, 2](
            gravity_z=Scalar[Self.DTYPE](-9.81),
            timestep=timestep,
            ground_z=Scalar[Self.DTYPE](0.0),
            friction=friction,
            restitution=Scalar[Self.DTYPE](0.0),
        )

        # Configure bodies
        # Body 0: Torso (vertical capsule)
        self.model.set_body_capsule(
            0,
            mass=self.torso_mass,
            radius=self.torso_radius,
            half_length=self.torso_half_length,
        )

        # Body 1: Thigh (vertical capsule)
        self.model.set_body_capsule(
            1,
            mass=self.thigh_mass,
            radius=self.thigh_radius,
            half_length=self.thigh_half_length,
        )

        # Body 2: Leg (vertical capsule)
        self.model.set_body_capsule(
            2,
            mass=self.leg_mass,
            radius=self.leg_radius,
            half_length=self.leg_half_length,
        )

        # Body 3: Foot (horizontal capsule - rotated 90° around Y-axis)
        self.model.set_body_capsule(
            3,
            mass=self.foot_mass,
            radius=self.foot_radius,
            half_length=self.foot_half_length,
        )

        # Calculate initial torso height for anchor points
        var foot_z = self.foot_radius
        var leg_z = foot_z + self.leg_radius + self.leg_half_length
        var thigh_z = leg_z + self.leg_half_length + self.thigh_half_length
        var torso_z = thigh_z + self.thigh_half_length + self.torso_half_length

        # Add root joints (constrain torso to X-Z plane)
        # Slide Joint 0: RootX (World -> Torso, X-axis translation)
        _ = self.model.add_slide_joint(
            parent=-1,  # World anchor
            child=0,    # Torso
            anchor_parent=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                torso_z,  # Initial torso height
            ),
            anchor_child=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
            ),
            axis=(
                Scalar[Self.DTYPE](1.0),  # X-axis translation
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
            ),
        )

        # Slide Joint 1: RootZ (World -> Torso, Z-axis translation)
        _ = self.model.add_slide_joint(
            parent=-1,  # World anchor
            child=0,    # Torso
            anchor_parent=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
            ),
            anchor_child=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),  # Z-axis translation
            ),
        )

        # Hinge Joint 0: RootY (World -> Torso, Y-axis rotation/pitch)
        _ = self.model.add_hinge_joint(
            parent=-1,  # World anchor
            child=0,    # Torso
            anchor_parent=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                torso_z,  # Initial torso height
            ),
            anchor_child=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),  # Y-axis rotation
                Scalar[Self.DTYPE](0.0),
            ),
        )

        # Hinge Joint 1: Hip (Torso -> Thigh)
        _ = self.model.add_hinge_joint(
            parent=0,  # Torso
            child=1,   # Thigh
            anchor_parent=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                -self.torso_half_length,  # Bottom of torso
            ),
            anchor_child=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                self.thigh_half_length,  # Top of thigh
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),  # Y-axis rotation
                Scalar[Self.DTYPE](0.0),
            ),
        )

        # Hinge Joint 2: Knee (Thigh -> Leg)
        _ = self.model.add_hinge_joint(
            parent=1,  # Thigh
            child=2,   # Leg
            anchor_parent=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                -self.thigh_half_length,  # Bottom of thigh
            ),
            anchor_child=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                self.leg_half_length,  # Top of leg
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),  # Y-axis rotation
                Scalar[Self.DTYPE](0.0),
            ),
        )

        # Hinge Joint 3: Ankle (Leg -> Foot)
        _ = self.model.add_hinge_joint(
            parent=2,  # Leg
            child=3,   # Foot
            anchor_parent=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                -self.leg_half_length,  # Bottom of leg
            ),
            anchor_child=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](0.0),  # Center of foot
            ),
            axis=(
                Scalar[Self.DTYPE](0.0),
                Scalar[Self.DTYPE](1.0),  # Y-axis rotation
                Scalar[Self.DTYPE](0.0),
            ),
        )

        # Set torque limits for actuated joints (Hip=1, Knee=2, Ankle=3)
        # RootY (joint 0) is not actuated
        self.model.joints[1].torque_limit = torque_limit
        self.model.joints[2].torque_limit = torque_limit
        self.model.joints[3].torque_limit = torque_limit

        # Initialize data
        self.data = Data[Self.DTYPE, 4, 20, 4, 2]()
        self._reset_state()

    fn _reset_state(mut self):
        """Reset bodies to initial standing position."""
        # Calculate positions from ground up (matching MuJoCo)
        # Foot is horizontal (rotated 90° around Y), so its height is just its radius
        var foot_z = self.foot_radius  # ~0.06

        # Leg center: above foot, accounting for leg half-length + leg radius
        var leg_z = foot_z + self.leg_radius + self.leg_half_length  # ~0.35

        # Thigh center: above leg
        var thigh_z = leg_z + self.leg_half_length + self.thigh_half_length  # ~0.825

        # Torso center: above thigh
        var torso_z = thigh_z + self.thigh_half_length + self.torso_half_length  # ~1.25

        # Set positions
        self.data.set_body_position(0, 0.0, 0.0, torso_z)  # Torso
        self.data.set_body_position(1, 0.0, 0.0, thigh_z)  # Thigh
        self.data.set_body_position(2, 0.0, 0.0, leg_z)    # Leg
        self.data.set_body_position(3, 0.0, 0.0, foot_z)   # Foot

        # Reset velocities
        for i in range(4):
            self.data.set_body_velocity(i, 0.0, 0.0, 0.0)
            self.data.set_body_angular_velocity(i, 0.0, 0.0, 0.0)

        # Reset quaternions to identity for vertical bodies (torso, thigh, leg)
        for i in range(3):
            self.data.quaternions[i * 4 + 0] = 0.0  # qx
            self.data.quaternions[i * 4 + 1] = 0.0  # qy
            self.data.quaternions[i * 4 + 2] = 0.0  # qz
            self.data.quaternions[i * 4 + 3] = 1.0  # qw

        # Foot quaternion: 90° rotation around Y-axis (horizontal capsule)
        # Quaternion for 90° Y rotation: (sin(45°), 0, 0, cos(45°)) in (x,y,z,w) format
        # Actually for Y-axis rotation: qx=0, qy=sin(θ/2), qz=0, qw=cos(θ/2)
        # θ = 90° = π/2, so θ/2 = π/4 = 45°
        # sin(45°) ≈ 0.70710678, cos(45°) ≈ 0.70710678
        self.data.quaternions[3 * 4 + 0] = 0.0           # qx
        self.data.quaternions[3 * 4 + 1] = 0.70710678    # qy (sin(π/4))
        self.data.quaternions[3 * 4 + 2] = 0.0           # qz
        self.data.quaternions[3 * 4 + 3] = 0.70710678    # qw (cos(π/4))

        # Reset joint torques (4 hinge joints: RootY, Hip, Knee, Ankle)
        for j in range(4):
            self.model.joints[j].target_torque = Scalar[Self.DTYPE](0.0)

        # Reset contact count
        self.data.num_contacts = 0

    fn reset(mut self) -> InlineArray[Scalar[Self.DTYPE], 11]:
        """Reset the environment to initial state.

        Returns:
            Initial observation (11 dimensions).
        """
        self._reset_state()
        self.current_step = 0
        return self.get_observation()

    fn step(
        mut self,
        action_hip: Scalar[Self.DTYPE],
        action_knee: Scalar[Self.DTYPE],
        action_ankle: Scalar[Self.DTYPE],
    ) -> Tuple[
        InlineArray[Scalar[Self.DTYPE], 11],  # observation
        Scalar[Self.DTYPE],  # reward
        Bool,  # terminated
        Bool,  # truncated
    ]:
        """Take one environment step.

        Args:
            action_hip: Hip torque in range [-1, 1], scaled by torque_limit.
            action_knee: Knee torque in range [-1, 1].
            action_ankle: Ankle torque in range [-1, 1].

        Returns:
            Tuple of (observation, reward, terminated, truncated).
        """
        # Clamp and scale actions
        var hip_torque = self._clamp_action(action_hip) * self.torque_limit
        var knee_torque = self._clamp_action(action_knee) * self.torque_limit
        var ankle_torque = self._clamp_action(action_ankle) * self.torque_limit

        # Joint indices: RootY=0, Hip=1, Knee=2, Ankle=3
        self.model.joints[1].target_torque = hip_torque
        self.model.joints[2].target_torque = knee_torque
        self.model.joints[3].target_torque = ankle_torque

        # Physics step (use PGS integrator for slide joint support)
        PGSIntegrator.step(self.model, self.data)

        self.current_step += 1

        # Get observation
        var obs = self.get_observation()

        # Check termination
        var terminated = self._is_terminated(obs)
        var truncated = self.current_step >= self.max_steps

        # Compute reward
        var reward = self._compute_reward(obs, hip_torque, knee_torque, ankle_torque, terminated)

        return (obs^, reward, terminated, truncated)

    fn _clamp_action(self, action: Scalar[Self.DTYPE]) -> Scalar[Self.DTYPE]:
        """Clamp action to [-1, 1]."""
        if action > 1.0:
            return Scalar[Self.DTYPE](1.0)
        elif action < -1.0:
            return Scalar[Self.DTYPE](-1.0)
        return action

    fn get_observation(self) -> InlineArray[Scalar[Self.DTYPE], 11]:
        """Get current observation vector.

        Returns:
            11-dimensional observation (matching MuJoCo Hopper):
            [0] Torso height (z)
            [1] Torso pitch angle
            [2] Hip joint angle
            [3] Knee joint angle
            [4] Ankle joint angle
            [5] Torso x velocity
            [6] Torso z velocity
            [7] Torso pitch angular velocity
            [8] Hip angular velocity
            [9] Knee angular velocity
            [10] Ankle angular velocity
        """
        var obs = InlineArray[Scalar[Self.DTYPE], 11](uninitialized=True)

        # Torso position and velocity
        var torso_pos = self.data.get_body_position(0)
        var torso_vel = self.data.get_body_velocity(0)

        obs[0] = torso_pos[2]  # Height (z)

        # Torso pitch (rotation around Y-axis)
        var qx = self.data.quaternions[0]
        var qy = self.data.quaternions[1]
        var qz = self.data.quaternions[2]
        var qw = self.data.quaternions[3]

        var sin_pitch = Scalar[Self.DTYPE](2.0) * (qw * qy - qz * qx)
        if sin_pitch > 1.0:
            sin_pitch = Scalar[Self.DTYPE](1.0)
        elif sin_pitch < -1.0:
            sin_pitch = Scalar[Self.DTYPE](-1.0)

        from math import asin
        obs[1] = asin(sin_pitch)

        # Joint angles (Hip=1, Knee=2, Ankle=3)
        obs[2] = get_joint_angle(self.model, self.data, 1)  # Hip
        obs[3] = get_joint_angle(self.model, self.data, 2)  # Knee
        obs[4] = get_joint_angle(self.model, self.data, 3)  # Ankle

        # Velocities
        obs[5] = torso_vel[0]  # x velocity
        obs[6] = torso_vel[2]  # z velocity

        # Torso pitch angular velocity (Y component)
        var torso_ang_vel = self.data.get_body_angular_velocity(0)
        obs[7] = torso_ang_vel[1]

        # Joint angular velocities (Hip=1, Knee=2, Ankle=3)
        obs[8] = get_joint_angular_velocity(self.model, self.data, 1)   # Hip
        obs[9] = get_joint_angular_velocity(self.model, self.data, 2)   # Knee
        obs[10] = get_joint_angular_velocity(self.model, self.data, 3)  # Ankle

        return obs^

    fn _is_terminated(self, obs: InlineArray[Scalar[Self.DTYPE], 11]) -> Bool:
        """Check if episode should terminate."""
        # Check height
        var height = obs[0]
        if height < self.min_height:
            return True

        # Check pitch
        var pitch = obs[1]
        if pitch > self.max_pitch or pitch < -self.max_pitch:
            return True

        return False

    fn _compute_reward(
        self,
        obs: InlineArray[Scalar[Self.DTYPE], 11],
        hip_torque: Scalar[Self.DTYPE],
        knee_torque: Scalar[Self.DTYPE],
        ankle_torque: Scalar[Self.DTYPE],
        terminated: Bool,
    ) -> Scalar[Self.DTYPE]:
        """Compute reward for current state."""
        # Forward velocity reward
        var forward_vel = obs[5]

        # Alive bonus (only if not terminated)
        var alive_bonus: Scalar[Self.DTYPE] = 0.0
        if not terminated:
            alive_bonus = Scalar[Self.DTYPE](1.0)

        # Control cost (sum of squared torques)
        var control_cost = Scalar[Self.DTYPE](0.001) * (
            hip_torque * hip_torque +
            knee_torque * knee_torque +
            ankle_torque * ankle_torque
        )

        return forward_vel + alive_bonus - control_cost

    fn get_torso_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get torso (body 0) position for visualization."""
        return self.data.get_body_position(0)

    fn get_thigh_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get thigh (body 1) position for visualization."""
        return self.data.get_body_position(1)

    fn get_leg_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get leg (body 2) position for visualization."""
        return self.data.get_body_position(2)

    fn get_foot_position(
        self,
    ) -> Tuple[Scalar[Self.DTYPE], Scalar[Self.DTYPE], Scalar[Self.DTYPE]]:
        """Get foot (body 3) position for visualization."""
        return self.data.get_body_position(3)
