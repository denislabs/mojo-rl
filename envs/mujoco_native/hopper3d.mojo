"""3D Hopper Environment using physics3d.

A full 3D implementation of the MuJoCo Hopper environment.
The hopper consists of 4 bodies (torso, thigh, leg, foot) connected by
3 hinge joints (hip, knee, ankle).

Observation (11D):
    - z position of torso (height)
    - pitch angle of torso
    - thigh joint angle
    - leg joint angle
    - foot joint angle
    - x velocity of torso
    - z velocity of torso
    - pitch angular velocity
    - thigh joint angular velocity
    - leg joint angular velocity
    - foot joint angular velocity

Action (3D):
    - hip torque
    - knee torque
    - ankle torque

Reward:
    reward = forward_velocity + healthy_bonus - control_cost
"""

from math import sqrt, cos, sin, pi
from math3d import Vec3, Quat, Mat3
from physics3d import (
    Body3D,
    Contact3D,
    Hinge3D,
    PDController,
    sphere_plane_collision,
    capsule_plane_collision,
)
from .base import MuJoCoEnvBase3D, MuJoCoConstants, Body3DState, JointState


struct Hopper3DConstants:
    """Constants for Hopper3D environment."""

    # Dimensions
    alias NUM_BODIES: Int = 4
    alias NUM_JOINTS: Int = 3
    alias OBS_DIM: Int = 11
    alias ACTION_DIM: Int = 3

    # Body indices
    alias TORSO: Int = 0
    alias THIGH: Int = 1
    alias LEG: Int = 2
    alias FOOT: Int = 3

    # Joint indices
    alias HIP: Int = 0
    alias KNEE: Int = 1
    alias ANKLE: Int = 2

    # Body dimensions (lengths)
    alias TORSO_LENGTH: Float64 = 0.4
    alias THIGH_LENGTH: Float64 = 0.45
    alias LEG_LENGTH: Float64 = 0.5
    alias FOOT_LENGTH: Float64 = 0.39

    # Body radius (for collision)
    alias BODY_RADIUS: Float64 = 0.05

    # Body masses
    alias TORSO_MASS: Float64 = 3.6651914291880923
    alias THIGH_MASS: Float64 = 4.057890510886845
    alias LEG_MASS: Float64 = 2.7813566959781637
    alias FOOT_MASS: Float64 = 5.315574695889897

    # Joint limits (radians)
    alias HIP_LOWER: Float64 = -2.61799  # -150 degrees
    alias HIP_UPPER: Float64 = 0.0
    alias KNEE_LOWER: Float64 = -2.61799  # -150 degrees
    alias KNEE_UPPER: Float64 = 0.0
    alias ANKLE_LOWER: Float64 = -0.785398  # -45 degrees
    alias ANKLE_UPPER: Float64 = 0.785398  # 45 degrees

    # Action scaling
    alias ACTION_SCALE: Float64 = 200.0

    # Reward parameters
    alias FORWARD_REWARD_WEIGHT: Float64 = 1.0
    alias CTRL_COST_WEIGHT: Float64 = 0.001
    alias HEALTHY_REWARD: Float64 = 1.0

    # Termination conditions
    alias MIN_Z: Float64 = 0.7  # Minimum torso height
    alias MAX_Z: Float64 = 2.0  # Maximum torso height
    alias MAX_ANGLE: Float64 = 0.2  # Maximum torso pitch angle

    # Initial state
    alias INITIAL_HEIGHT: Float64 = 1.25
    alias INITIAL_NOISE: Float64 = 0.005


struct Hopper3D:
    """3D Hopper environment with full physics simulation.

    The hopper starts in a standing position and must learn to hop forward.
    The agent receives a reward for forward velocity and staying "healthy"
    (not falling over).
    """

    var base: MuJoCoEnvBase3D
    var step_count: Int
    var max_steps: Int
    var prev_x: Float64

    fn __init__(out self, max_steps: Int = 1000):
        self.base = MuJoCoEnvBase3D(
            Hopper3DConstants.NUM_BODIES,
            Hopper3DConstants.NUM_JOINTS,
        )
        self.step_count = 0
        self.max_steps = max_steps
        self.prev_x = 0.0
        self._setup_bodies()
        self._setup_joints()

    fn _setup_bodies(mut self):
        """Initialize body states with proper masses and positions."""
        # Torso at top
        self.base.bodies[Hopper3DConstants.TORSO] = Body3DState.create(
            Vec3(0, 0, Hopper3DConstants.INITIAL_HEIGHT),
            Quat.identity(),
            Hopper3DConstants.TORSO_MASS,
        )

        # Thigh below torso
        var thigh_z = (
            Hopper3DConstants.INITIAL_HEIGHT
            - Hopper3DConstants.TORSO_LENGTH / 2
            - Hopper3DConstants.THIGH_LENGTH / 2
        )
        self.base.bodies[Hopper3DConstants.THIGH] = Body3DState.create(
            Vec3(0, 0, thigh_z), Quat.identity(), Hopper3DConstants.THIGH_MASS
        )

        # Leg below thigh
        var leg_z = thigh_z - Hopper3DConstants.THIGH_LENGTH / 2 - Hopper3DConstants.LEG_LENGTH / 2
        self.base.bodies[Hopper3DConstants.LEG] = Body3DState.create(
            Vec3(0, 0, leg_z), Quat.identity(), Hopper3DConstants.LEG_MASS
        )

        # Foot at bottom
        var foot_z = leg_z - Hopper3DConstants.LEG_LENGTH / 2 - Hopper3DConstants.FOOT_LENGTH / 2
        self.base.bodies[Hopper3DConstants.FOOT] = Body3DState.create(
            Vec3(0, 0, foot_z), Quat.identity(), Hopper3DConstants.FOOT_MASS
        )

    fn _setup_joints(mut self):
        """Initialize joint states with proper limits."""
        self.base.joints[Hopper3DConstants.HIP] = JointState.with_limits(
            Hopper3DConstants.HIP_LOWER, Hopper3DConstants.HIP_UPPER
        )
        self.base.joints[Hopper3DConstants.KNEE] = JointState.with_limits(
            Hopper3DConstants.KNEE_LOWER, Hopper3DConstants.KNEE_UPPER
        )
        self.base.joints[Hopper3DConstants.ANKLE] = JointState.with_limits(
            Hopper3DConstants.ANKLE_LOWER, Hopper3DConstants.ANKLE_UPPER
        )

    fn reset(mut self) -> List[Float64]:
        """Reset environment to initial state with small random perturbations."""
        self.step_count = 0
        self.base.time = 0.0

        # Reset bodies
        self._setup_bodies()

        # Add small random noise to joint angles
        # In a real implementation, this would use a random generator
        for i in range(Hopper3DConstants.NUM_JOINTS):
            self.base.joints[i].angle = 0.0
            self.base.joints[i].velocity = 0.0
            self.base.joints[i].torque = 0.0

        # Update body positions based on joint angles
        self._forward_kinematics()

        self.prev_x = self.base.bodies[Hopper3DConstants.TORSO].position.x
        return self.get_observation()

    fn step(mut self, action: List[Float64]) -> Tuple[List[Float64], Float64, Bool]:
        """Take one environment step.

        Args:
            action: List of 3 torques [hip, knee, ankle]

        Returns:
            (observation, reward, done)
        """
        # Clamp actions and scale
        var scaled_action = List[Float64]()
        for i in range(min(len(action), Hopper3DConstants.ACTION_DIM)):
            var a = action[i]
            # Clamp to [-1, 1]
            if a < -1.0:
                a = -1.0
            elif a > 1.0:
                a = 1.0
            scaled_action.append(a * Hopper3DConstants.ACTION_SCALE)

        # Pad with zeros if needed
        while len(scaled_action) < Hopper3DConstants.ACTION_DIM:
            scaled_action.append(0.0)

        # Apply actions as joint torques
        for i in range(Hopper3DConstants.NUM_JOINTS):
            self.base.apply_joint_torque(i, scaled_action[i])

        # Simulate physics for frame_skip steps
        for _ in range(self.base.frame_skip):
            self._physics_step()

        self.step_count += 1

        # Compute reward
        var x_now = self.base.bodies[Hopper3DConstants.TORSO].position.x
        var forward_velocity = (x_now - self.prev_x) / (
            self.base.timestep * Float64(self.base.frame_skip)
        )
        self.prev_x = x_now

        var forward_reward = Hopper3DConstants.FORWARD_REWARD_WEIGHT * forward_velocity

        var healthy = self.base.compute_healthy_reward(
            Hopper3DConstants.TORSO,
            Hopper3DConstants.MIN_Z,
            Hopper3DConstants.MAX_Z,
            Hopper3DConstants.MAX_ANGLE,
        )
        var healthy_reward = Hopper3DConstants.HEALTHY_REWARD * healthy

        var ctrl_cost = self.base.compute_control_cost(
            action, Hopper3DConstants.CTRL_COST_WEIGHT
        )

        var reward = forward_reward + healthy_reward - ctrl_cost

        # Check termination
        var terminated = self.base.is_terminated(
            Hopper3DConstants.TORSO,
            Hopper3DConstants.MIN_Z,
            Hopper3DConstants.MAX_Z,
            Hopper3DConstants.MAX_ANGLE,
        )
        var truncated = self.step_count >= self.max_steps
        var done = terminated or truncated

        return (self.get_observation(), reward, done)

    fn _physics_step(mut self):
        """Perform one physics simulation step."""
        var dt = self.base.timestep
        var gravity = MuJoCoConstants.GRAVITY

        # 1. Integrate joint dynamics
        for i in range(Hopper3DConstants.NUM_JOINTS):
            # Simple Euler integration for joints
            var torque = self.base.joints[i].torque
            # Approximate inertia based on connected body masses
            var inertia = 0.1  # Simplified
            var accel = torque / inertia

            self.base.joints[i].velocity += accel * dt
            self.base.joints[i].angle += self.base.joints[i].velocity * dt
            self.base.joints[i].clamp_angle()

        # 2. Update body positions via forward kinematics
        self._forward_kinematics()

        # 3. Apply gravity to all bodies
        for i in range(Hopper3DConstants.NUM_BODIES):
            self.base.bodies[i].linear_velocity.z += gravity * dt

        # 4. Detect ground contacts
        self.base.detect_ground_contacts()

        # 5. Resolve contacts (simple position correction)
        self._resolve_contacts()

        # 6. Integrate body positions
        for i in range(Hopper3DConstants.NUM_BODIES):
            self.base.bodies[i].position.x += self.base.bodies[i].linear_velocity.x * dt
            self.base.bodies[i].position.y += self.base.bodies[i].linear_velocity.y * dt
            self.base.bodies[i].position.z += self.base.bodies[i].linear_velocity.z * dt

        # Update time
        self.base.time += dt

    fn _forward_kinematics(mut self):
        """Update body positions based on joint angles.

        Uses forward kinematics to compute body positions from joint angles,
        keeping the kinematic chain consistent.
        """
        # Torso position is the reference (root of the chain)
        var torso = self.base.bodies[Hopper3DConstants.TORSO]

        # Hip joint connects torso to thigh
        var hip_angle = self.base.joints[Hopper3DConstants.HIP].angle
        var hip_pos = Vec3(
            torso.position.x, torso.position.y, torso.position.z - Hopper3DConstants.TORSO_LENGTH / 2
        )

        # Thigh position (rotated by hip angle in xz plane)
        var thigh_offset_x = -Hopper3DConstants.THIGH_LENGTH / 2 * sin(hip_angle)
        var thigh_offset_z = -Hopper3DConstants.THIGH_LENGTH / 2 * cos(hip_angle)
        self.base.bodies[Hopper3DConstants.THIGH].position = Vec3(
            hip_pos.x + thigh_offset_x, hip_pos.y, hip_pos.z + thigh_offset_z
        )

        # Knee joint at end of thigh
        var knee_pos = Vec3(
            hip_pos.x + 2 * thigh_offset_x,
            hip_pos.y,
            hip_pos.z + 2 * thigh_offset_z,
        )

        # Leg position (rotated by hip + knee angles)
        var knee_angle = self.base.joints[Hopper3DConstants.KNEE].angle
        var total_angle = hip_angle + knee_angle
        var leg_offset_x = -Hopper3DConstants.LEG_LENGTH / 2 * sin(total_angle)
        var leg_offset_z = -Hopper3DConstants.LEG_LENGTH / 2 * cos(total_angle)
        self.base.bodies[Hopper3DConstants.LEG].position = Vec3(
            knee_pos.x + leg_offset_x, knee_pos.y, knee_pos.z + leg_offset_z
        )

        # Ankle joint at end of leg
        var ankle_pos = Vec3(
            knee_pos.x + 2 * leg_offset_x,
            knee_pos.y,
            knee_pos.z + 2 * leg_offset_z,
        )

        # Foot position (rotated by hip + knee + ankle angles)
        var ankle_angle = self.base.joints[Hopper3DConstants.ANKLE].angle
        var foot_total_angle = total_angle + ankle_angle
        var foot_offset_x = -Hopper3DConstants.FOOT_LENGTH / 2 * sin(foot_total_angle)
        var foot_offset_z = -Hopper3DConstants.FOOT_LENGTH / 2 * cos(foot_total_angle)
        self.base.bodies[Hopper3DConstants.FOOT].position = Vec3(
            ankle_pos.x + foot_offset_x,
            ankle_pos.y,
            ankle_pos.z + foot_offset_z,
        )

        # Update orientations based on angles
        self.base.bodies[Hopper3DConstants.THIGH].orientation = Quat.from_axis_angle(
            Vec3(0, 1, 0), hip_angle
        )
        self.base.bodies[Hopper3DConstants.LEG].orientation = Quat.from_axis_angle(
            Vec3(0, 1, 0), total_angle
        )
        self.base.bodies[Hopper3DConstants.FOOT].orientation = Quat.from_axis_angle(
            Vec3(0, 1, 0), foot_total_angle
        )

    fn _resolve_contacts(mut self):
        """Resolve ground contacts with simple position correction."""
        var ground_z = 0.0

        for i in range(len(self.base.contacts)):
            var contact = self.base.contacts[i]
            var body_idx = contact.body_index

            if contact.penetration > 0:
                # Position correction
                self.base.bodies[body_idx].position.z += contact.penetration

                # Velocity correction (stop downward velocity)
                if self.base.bodies[body_idx].linear_velocity.z < 0:
                    # Apply restitution (mostly inelastic)
                    self.base.bodies[body_idx].linear_velocity.z *= -0.1

                # Apply friction
                var friction = MuJoCoConstants.GROUND_FRICTION
                self.base.bodies[body_idx].linear_velocity.x *= (1.0 - friction * 0.1)
                self.base.bodies[body_idx].linear_velocity.y *= (1.0 - friction * 0.1)

        # Propagate contact forces up the kinematic chain
        self._propagate_contact_to_torso()

    fn _propagate_contact_to_torso(mut self):
        """Propagate foot contact effects to torso position."""
        # If foot is in contact, the whole chain should maintain structure
        if self.base.has_ground_contact(Hopper3DConstants.FOOT):
            # Re-run forward kinematics to maintain chain structure
            # after foot position has been corrected
            var foot_pos = self.base.bodies[Hopper3DConstants.FOOT].position

            # Work backwards to update torso
            var ankle_angle = self.base.joints[Hopper3DConstants.ANKLE].angle
            var knee_angle = self.base.joints[Hopper3DConstants.KNEE].angle
            var hip_angle = self.base.joints[Hopper3DConstants.HIP].angle

            var foot_total_angle = hip_angle + knee_angle + ankle_angle

            # Ankle position from foot
            var ankle_pos = Vec3(
                foot_pos.x + Hopper3DConstants.FOOT_LENGTH / 2 * sin(foot_total_angle),
                foot_pos.y,
                foot_pos.z + Hopper3DConstants.FOOT_LENGTH / 2 * cos(foot_total_angle),
            )

            # Knee position from ankle
            var total_angle = hip_angle + knee_angle
            var knee_pos = Vec3(
                ankle_pos.x + Hopper3DConstants.LEG_LENGTH / 2 * sin(total_angle),
                ankle_pos.y,
                ankle_pos.z + Hopper3DConstants.LEG_LENGTH / 2 * cos(total_angle),
            )

            # Hip position from knee
            var hip_pos = Vec3(
                knee_pos.x + Hopper3DConstants.THIGH_LENGTH / 2 * sin(hip_angle),
                knee_pos.y,
                knee_pos.z + Hopper3DConstants.THIGH_LENGTH / 2 * cos(hip_angle),
            )

            # Update torso position
            self.base.bodies[Hopper3DConstants.TORSO].position = Vec3(
                hip_pos.x,
                hip_pos.y,
                hip_pos.z + Hopper3DConstants.TORSO_LENGTH / 2,
            )

    fn get_observation(self) -> List[Float64]:
        """Build observation vector (11D).

        Returns:
            [z, pitch, hip_angle, knee_angle, ankle_angle,
             vx, vz, pitch_vel, hip_vel, knee_vel, ankle_vel]
        """
        var obs = List[Float64]()

        # Position (1D) - z height of torso
        obs.append(self.base.bodies[Hopper3DConstants.TORSO].position.z)

        # Orientation (1D) - pitch angle of torso
        var pitch = self.base._get_pitch(
            self.base.bodies[Hopper3DConstants.TORSO].orientation
        )
        obs.append(pitch)

        # Joint angles (3D)
        obs.append(self.base.joints[Hopper3DConstants.HIP].angle)
        obs.append(self.base.joints[Hopper3DConstants.KNEE].angle)
        obs.append(self.base.joints[Hopper3DConstants.ANKLE].angle)

        # Velocities (2D) - x and z velocity of torso
        obs.append(self.base.bodies[Hopper3DConstants.TORSO].linear_velocity.x)
        obs.append(self.base.bodies[Hopper3DConstants.TORSO].linear_velocity.z)

        # Angular velocity (1D) - pitch angular velocity
        obs.append(self.base.bodies[Hopper3DConstants.TORSO].angular_velocity.y)

        # Joint velocities (3D)
        obs.append(self.base.joints[Hopper3DConstants.HIP].velocity)
        obs.append(self.base.joints[Hopper3DConstants.KNEE].velocity)
        obs.append(self.base.joints[Hopper3DConstants.ANKLE].velocity)

        return obs^

    fn render(self):
        """Render the environment (placeholder for visualization)."""
        # Would use render3d module here
        pass

    fn close(self):
        """Clean up environment resources."""
        pass
