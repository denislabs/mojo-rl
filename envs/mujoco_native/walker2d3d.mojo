"""3D Walker2d Environment using physics3d.

A full 3D implementation of the MuJoCo Walker2d environment.
The walker consists of 7 bodies (torso + 2 legs with thigh, leg, foot each)
connected by 6 hinge joints.

Observation (17D):
    - z position of torso (height)
    - pitch angle of torso
    - 6 joint angles (right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle)
    - x velocity of torso
    - z velocity of torso
    - pitch angular velocity
    - 6 joint angular velocities

Action (6D):
    - right hip torque
    - right knee torque
    - right ankle torque
    - left hip torque
    - left knee torque
    - left ankle torque

Reward:
    reward = forward_velocity + healthy_bonus - control_cost
"""

from math import sqrt, cos, sin, pi
from math3d import Vec3, Quat, Mat3
from physics3d import Body3D, Contact3D, Hinge3D, PDController
from .base import MuJoCoEnvBase3D, MuJoCoConstants, Body3DState, JointState


struct Walker2d3DConstants:
    """Constants for Walker2d3D environment."""

    # Dimensions
    alias NUM_BODIES: Int = 7
    alias NUM_JOINTS: Int = 6
    alias OBS_DIM: Int = 17
    alias ACTION_DIM: Int = 6

    # Body indices
    alias TORSO: Int = 0
    alias RIGHT_THIGH: Int = 1
    alias RIGHT_LEG: Int = 2
    alias RIGHT_FOOT: Int = 3
    alias LEFT_THIGH: Int = 4
    alias LEFT_LEG: Int = 5
    alias LEFT_FOOT: Int = 6

    # Joint indices
    alias RIGHT_HIP: Int = 0
    alias RIGHT_KNEE: Int = 1
    alias RIGHT_ANKLE: Int = 2
    alias LEFT_HIP: Int = 3
    alias LEFT_KNEE: Int = 4
    alias LEFT_ANKLE: Int = 5

    # Body dimensions
    alias TORSO_LENGTH: Float64 = 0.4
    alias THIGH_LENGTH: Float64 = 0.45
    alias LEG_LENGTH: Float64 = 0.5
    alias FOOT_LENGTH: Float64 = 0.2

    # Body masses
    alias TORSO_MASS: Float64 = 3.53
    alias THIGH_MASS: Float64 = 4.06
    alias LEG_MASS: Float64 = 2.78
    alias FOOT_MASS: Float64 = 1.81

    # Joint limits (radians)
    alias HIP_LOWER: Float64 = -1.0
    alias HIP_UPPER: Float64 = 0.7
    alias KNEE_LOWER: Float64 = -1.2
    alias KNEE_UPPER: Float64 = 0.0
    alias ANKLE_LOWER: Float64 = -0.5
    alias ANKLE_UPPER: Float64 = 0.5

    # Action scaling
    alias ACTION_SCALE: Float64 = 100.0

    # Reward parameters
    alias FORWARD_REWARD_WEIGHT: Float64 = 1.0
    alias CTRL_COST_WEIGHT: Float64 = 0.001
    alias HEALTHY_REWARD: Float64 = 1.0

    # Termination conditions
    alias MIN_Z: Float64 = 0.8
    alias MAX_Z: Float64 = 2.0
    alias MAX_ANGLE: Float64 = 1.0

    # Initial state
    alias INITIAL_HEIGHT: Float64 = 1.25


struct Walker2d3D:
    """3D Walker2d environment with full physics simulation."""

    var base: MuJoCoEnvBase3D
    var step_count: Int
    var max_steps: Int
    var prev_x: Float64

    fn __init__(out self, max_steps: Int = 1000):
        self.base = MuJoCoEnvBase3D(
            Walker2d3DConstants.NUM_BODIES,
            Walker2d3DConstants.NUM_JOINTS,
        )
        self.step_count = 0
        self.max_steps = max_steps
        self.prev_x = 0.0
        self._setup_bodies()
        self._setup_joints()

    fn _setup_bodies(mut self):
        """Initialize body states."""
        # Torso
        self.base.bodies[Walker2d3DConstants.TORSO] = Body3DState.create(
            Vec3(0, 0, Walker2d3DConstants.INITIAL_HEIGHT),
            Quat.identity(),
            Walker2d3DConstants.TORSO_MASS,
        )

        # Right leg
        var thigh_z = Walker2d3DConstants.INITIAL_HEIGHT - Walker2d3DConstants.TORSO_LENGTH / 2 - Walker2d3DConstants.THIGH_LENGTH / 2
        self.base.bodies[Walker2d3DConstants.RIGHT_THIGH] = Body3DState.create(
            Vec3(0, -0.1, thigh_z), Quat.identity(), Walker2d3DConstants.THIGH_MASS
        )

        var leg_z = thigh_z - Walker2d3DConstants.THIGH_LENGTH / 2 - Walker2d3DConstants.LEG_LENGTH / 2
        self.base.bodies[Walker2d3DConstants.RIGHT_LEG] = Body3DState.create(
            Vec3(0, -0.1, leg_z), Quat.identity(), Walker2d3DConstants.LEG_MASS
        )

        var foot_z = leg_z - Walker2d3DConstants.LEG_LENGTH / 2 - 0.05
        self.base.bodies[Walker2d3DConstants.RIGHT_FOOT] = Body3DState.create(
            Vec3(0.1, -0.1, foot_z), Quat.identity(), Walker2d3DConstants.FOOT_MASS
        )

        # Left leg (mirrored)
        self.base.bodies[Walker2d3DConstants.LEFT_THIGH] = Body3DState.create(
            Vec3(0, 0.1, thigh_z), Quat.identity(), Walker2d3DConstants.THIGH_MASS
        )

        self.base.bodies[Walker2d3DConstants.LEFT_LEG] = Body3DState.create(
            Vec3(0, 0.1, leg_z), Quat.identity(), Walker2d3DConstants.LEG_MASS
        )

        self.base.bodies[Walker2d3DConstants.LEFT_FOOT] = Body3DState.create(
            Vec3(0.1, 0.1, foot_z), Quat.identity(), Walker2d3DConstants.FOOT_MASS
        )

    fn _setup_joints(mut self):
        """Initialize joint states."""
        # Right leg joints
        self.base.joints[Walker2d3DConstants.RIGHT_HIP] = JointState.with_limits(
            Walker2d3DConstants.HIP_LOWER, Walker2d3DConstants.HIP_UPPER
        )
        self.base.joints[Walker2d3DConstants.RIGHT_KNEE] = JointState.with_limits(
            Walker2d3DConstants.KNEE_LOWER, Walker2d3DConstants.KNEE_UPPER
        )
        self.base.joints[Walker2d3DConstants.RIGHT_ANKLE] = JointState.with_limits(
            Walker2d3DConstants.ANKLE_LOWER, Walker2d3DConstants.ANKLE_UPPER
        )

        # Left leg joints
        self.base.joints[Walker2d3DConstants.LEFT_HIP] = JointState.with_limits(
            Walker2d3DConstants.HIP_LOWER, Walker2d3DConstants.HIP_UPPER
        )
        self.base.joints[Walker2d3DConstants.LEFT_KNEE] = JointState.with_limits(
            Walker2d3DConstants.KNEE_LOWER, Walker2d3DConstants.KNEE_UPPER
        )
        self.base.joints[Walker2d3DConstants.LEFT_ANKLE] = JointState.with_limits(
            Walker2d3DConstants.ANKLE_LOWER, Walker2d3DConstants.ANKLE_UPPER
        )

    fn reset(mut self) -> List[Float64]:
        """Reset environment to initial state."""
        self.step_count = 0
        self.base.time = 0.0
        self._setup_bodies()
        self._setup_joints()
        self._forward_kinematics()
        self.prev_x = self.base.bodies[Walker2d3DConstants.TORSO].position.x
        return self.get_observation()

    fn step(mut self, action: List[Float64]) -> Tuple[List[Float64], Float64, Bool]:
        """Take one environment step."""
        # Scale and clamp actions
        var scaled_action = List[Float64]()
        for i in range(min(len(action), Walker2d3DConstants.ACTION_DIM)):
            var a = action[i]
            if a < -1.0:
                a = -1.0
            elif a > 1.0:
                a = 1.0
            scaled_action.append(a * Walker2d3DConstants.ACTION_SCALE)

        while len(scaled_action) < Walker2d3DConstants.ACTION_DIM:
            scaled_action.append(0.0)

        # Apply torques
        for i in range(Walker2d3DConstants.NUM_JOINTS):
            self.base.apply_joint_torque(i, scaled_action[i])

        # Simulate
        for _ in range(self.base.frame_skip):
            self._physics_step()

        self.step_count += 1

        # Compute reward
        var x_now = self.base.bodies[Walker2d3DConstants.TORSO].position.x
        var forward_velocity = (x_now - self.prev_x) / (
            self.base.timestep * Float64(self.base.frame_skip)
        )
        self.prev_x = x_now

        var forward_reward = Walker2d3DConstants.FORWARD_REWARD_WEIGHT * forward_velocity
        var healthy = self.base.compute_healthy_reward(
            Walker2d3DConstants.TORSO,
            Walker2d3DConstants.MIN_Z,
            Walker2d3DConstants.MAX_Z,
            Walker2d3DConstants.MAX_ANGLE,
        )
        var healthy_reward = Walker2d3DConstants.HEALTHY_REWARD * healthy
        var ctrl_cost = self.base.compute_control_cost(action, Walker2d3DConstants.CTRL_COST_WEIGHT)

        var reward = forward_reward + healthy_reward - ctrl_cost

        var terminated = self.base.is_terminated(
            Walker2d3DConstants.TORSO,
            Walker2d3DConstants.MIN_Z,
            Walker2d3DConstants.MAX_Z,
            Walker2d3DConstants.MAX_ANGLE,
        )
        var done = terminated or (self.step_count >= self.max_steps)

        return (self.get_observation(), reward, done)

    fn _physics_step(mut self):
        """Perform one physics step."""
        var dt = self.base.timestep
        var gravity = MuJoCoConstants.GRAVITY

        # Integrate joints
        for i in range(Walker2d3DConstants.NUM_JOINTS):
            var torque = self.base.joints[i].torque
            var inertia = 0.1
            var accel = torque / inertia
            self.base.joints[i].velocity += accel * dt
            self.base.joints[i].angle += self.base.joints[i].velocity * dt
            self.base.joints[i].clamp_angle()

        self._forward_kinematics()

        # Apply gravity
        for i in range(Walker2d3DConstants.NUM_BODIES):
            self.base.bodies[i].linear_velocity.z += gravity * dt

        self.base.detect_ground_contacts()
        self._resolve_contacts()

        # Integrate positions
        for i in range(Walker2d3DConstants.NUM_BODIES):
            self.base.bodies[i].position.x += self.base.bodies[i].linear_velocity.x * dt
            self.base.bodies[i].position.y += self.base.bodies[i].linear_velocity.y * dt
            self.base.bodies[i].position.z += self.base.bodies[i].linear_velocity.z * dt

        self.base.time += dt

    fn _forward_kinematics(mut self):
        """Update body positions based on joint angles."""
        var torso = self.base.bodies[Walker2d3DConstants.TORSO]

        # Right leg chain
        var r_hip_angle = self.base.joints[Walker2d3DConstants.RIGHT_HIP].angle
        var r_knee_angle = self.base.joints[Walker2d3DConstants.RIGHT_KNEE].angle
        var r_ankle_angle = self.base.joints[Walker2d3DConstants.RIGHT_ANKLE].angle

        var hip_pos = Vec3(torso.position.x, torso.position.y - 0.1, torso.position.z - Walker2d3DConstants.TORSO_LENGTH / 2)

        var r_thigh_offset_x = -Walker2d3DConstants.THIGH_LENGTH / 2 * sin(r_hip_angle)
        var r_thigh_offset_z = -Walker2d3DConstants.THIGH_LENGTH / 2 * cos(r_hip_angle)
        self.base.bodies[Walker2d3DConstants.RIGHT_THIGH].position = Vec3(
            hip_pos.x + r_thigh_offset_x, hip_pos.y, hip_pos.z + r_thigh_offset_z
        )

        var r_knee_pos = Vec3(hip_pos.x + 2 * r_thigh_offset_x, hip_pos.y, hip_pos.z + 2 * r_thigh_offset_z)
        var r_total_angle = r_hip_angle + r_knee_angle
        var r_leg_offset_x = -Walker2d3DConstants.LEG_LENGTH / 2 * sin(r_total_angle)
        var r_leg_offset_z = -Walker2d3DConstants.LEG_LENGTH / 2 * cos(r_total_angle)
        self.base.bodies[Walker2d3DConstants.RIGHT_LEG].position = Vec3(
            r_knee_pos.x + r_leg_offset_x, r_knee_pos.y, r_knee_pos.z + r_leg_offset_z
        )

        var r_ankle_pos = Vec3(r_knee_pos.x + 2 * r_leg_offset_x, r_knee_pos.y, r_knee_pos.z + 2 * r_leg_offset_z)
        var r_foot_angle = r_total_angle + r_ankle_angle
        self.base.bodies[Walker2d3DConstants.RIGHT_FOOT].position = Vec3(
            r_ankle_pos.x + Walker2d3DConstants.FOOT_LENGTH / 2 * cos(r_foot_angle),
            r_ankle_pos.y,
            r_ankle_pos.z - 0.05,
        )

        # Left leg chain (mirrored)
        var l_hip_angle = self.base.joints[Walker2d3DConstants.LEFT_HIP].angle
        var l_knee_angle = self.base.joints[Walker2d3DConstants.LEFT_KNEE].angle
        var l_ankle_angle = self.base.joints[Walker2d3DConstants.LEFT_ANKLE].angle

        var l_hip_pos = Vec3(torso.position.x, torso.position.y + 0.1, torso.position.z - Walker2d3DConstants.TORSO_LENGTH / 2)

        var l_thigh_offset_x = -Walker2d3DConstants.THIGH_LENGTH / 2 * sin(l_hip_angle)
        var l_thigh_offset_z = -Walker2d3DConstants.THIGH_LENGTH / 2 * cos(l_hip_angle)
        self.base.bodies[Walker2d3DConstants.LEFT_THIGH].position = Vec3(
            l_hip_pos.x + l_thigh_offset_x, l_hip_pos.y, l_hip_pos.z + l_thigh_offset_z
        )

        var l_knee_pos = Vec3(l_hip_pos.x + 2 * l_thigh_offset_x, l_hip_pos.y, l_hip_pos.z + 2 * l_thigh_offset_z)
        var l_total_angle = l_hip_angle + l_knee_angle
        var l_leg_offset_x = -Walker2d3DConstants.LEG_LENGTH / 2 * sin(l_total_angle)
        var l_leg_offset_z = -Walker2d3DConstants.LEG_LENGTH / 2 * cos(l_total_angle)
        self.base.bodies[Walker2d3DConstants.LEFT_LEG].position = Vec3(
            l_knee_pos.x + l_leg_offset_x, l_knee_pos.y, l_knee_pos.z + l_leg_offset_z
        )

        var l_ankle_pos = Vec3(l_knee_pos.x + 2 * l_leg_offset_x, l_knee_pos.y, l_knee_pos.z + 2 * l_leg_offset_z)
        var l_foot_angle = l_total_angle + l_ankle_angle
        self.base.bodies[Walker2d3DConstants.LEFT_FOOT].position = Vec3(
            l_ankle_pos.x + Walker2d3DConstants.FOOT_LENGTH / 2 * cos(l_foot_angle),
            l_ankle_pos.y,
            l_ankle_pos.z - 0.05,
        )

    fn _resolve_contacts(mut self):
        """Resolve ground contacts."""
        for i in range(len(self.base.contacts)):
            var contact = self.base.contacts[i]
            var body_idx = contact.body_index

            if contact.penetration > 0:
                self.base.bodies[body_idx].position.z += contact.penetration
                if self.base.bodies[body_idx].linear_velocity.z < 0:
                    self.base.bodies[body_idx].linear_velocity.z *= -0.1
                var friction = MuJoCoConstants.GROUND_FRICTION
                self.base.bodies[body_idx].linear_velocity.x *= (1.0 - friction * 0.1)
                self.base.bodies[body_idx].linear_velocity.y *= (1.0 - friction * 0.1)

    fn get_observation(self) -> List[Float64]:
        """Build observation vector (17D)."""
        var obs = List[Float64]()

        # Height
        obs.append(self.base.bodies[Walker2d3DConstants.TORSO].position.z)

        # Pitch
        obs.append(self.base._get_pitch(self.base.bodies[Walker2d3DConstants.TORSO].orientation))

        # Joint angles (6)
        for i in range(Walker2d3DConstants.NUM_JOINTS):
            obs.append(self.base.joints[i].angle)

        # Velocities
        obs.append(self.base.bodies[Walker2d3DConstants.TORSO].linear_velocity.x)
        obs.append(self.base.bodies[Walker2d3DConstants.TORSO].linear_velocity.z)
        obs.append(self.base.bodies[Walker2d3DConstants.TORSO].angular_velocity.y)

        # Joint velocities (6)
        for i in range(Walker2d3DConstants.NUM_JOINTS):
            obs.append(self.base.joints[i].velocity)

        return obs^
