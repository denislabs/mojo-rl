"""3D HalfCheetah Environment using physics3d.

A full 3D implementation of the MuJoCo HalfCheetah environment.
The cheetah consists of 7 bodies (torso + front/back legs with thigh and shin)
connected by 6 hinge joints.

Observation (17D):
    - z position of torso (height)
    - pitch angle of torso
    - 6 joint angles
    - x velocity of torso
    - z velocity of torso
    - pitch angular velocity
    - 6 joint angular velocities

Action (6D):
    - back thigh torque
    - back shin torque
    - back foot torque
    - front thigh torque
    - front shin torque
    - front foot torque

Reward:
    reward = forward_velocity - control_cost
    (No healthy reward, no early termination)
"""

from math import sqrt, cos, sin, pi
from math3d import Vec3, Quat, Mat3
from physics3d import Body3D, Contact3D, Hinge3D, PDController
from .base import MuJoCoEnvBase3D, MuJoCoConstants, Body3DState, JointState


struct HalfCheetah3DConstants:
    """Constants for HalfCheetah3D environment."""

    # Dimensions
    alias NUM_BODIES: Int = 7
    alias NUM_JOINTS: Int = 6
    alias OBS_DIM: Int = 17
    alias ACTION_DIM: Int = 6

    # Body indices
    alias TORSO: Int = 0
    alias BACK_THIGH: Int = 1
    alias BACK_SHIN: Int = 2
    alias BACK_FOOT: Int = 3
    alias FRONT_THIGH: Int = 4
    alias FRONT_SHIN: Int = 5
    alias FRONT_FOOT: Int = 6

    # Joint indices
    alias BACK_HIP: Int = 0
    alias BACK_KNEE: Int = 1
    alias BACK_ANKLE: Int = 2
    alias FRONT_HIP: Int = 3
    alias FRONT_KNEE: Int = 4
    alias FRONT_ANKLE: Int = 5

    # Body dimensions
    alias TORSO_LENGTH: Float64 = 1.0
    alias THIGH_LENGTH: Float64 = 0.145
    alias SHIN_LENGTH: Float64 = 0.15
    alias FOOT_LENGTH: Float64 = 0.094

    # Body masses
    alias TORSO_MASS: Float64 = 6.25
    alias THIGH_MASS: Float64 = 1.0
    alias SHIN_MASS: Float64 = 1.0
    alias FOOT_MASS: Float64 = 1.0

    # Joint limits
    alias BACK_HIP_LOWER: Float64 = -0.52
    alias BACK_HIP_UPPER: Float64 = 1.05
    alias BACK_KNEE_LOWER: Float64 = -0.785
    alias BACK_KNEE_UPPER: Float64 = 0.785
    alias BACK_ANKLE_LOWER: Float64 = -0.4
    alias BACK_ANKLE_UPPER: Float64 = 0.785
    alias FRONT_HIP_LOWER: Float64 = -1.0
    alias FRONT_HIP_UPPER: Float64 = 0.7
    alias FRONT_KNEE_LOWER: Float64 = -1.2
    alias FRONT_KNEE_UPPER: Float64 = 0.87
    alias FRONT_ANKLE_LOWER: Float64 = -0.5
    alias FRONT_ANKLE_UPPER: Float64 = 0.5

    # Action scaling
    alias ACTION_SCALE: Float64 = 120.0

    # Reward parameters
    alias FORWARD_REWARD_WEIGHT: Float64 = 1.0
    alias CTRL_COST_WEIGHT: Float64 = 0.1

    # Initial state
    alias INITIAL_HEIGHT: Float64 = 0.7


struct HalfCheetah3D:
    """3D HalfCheetah environment with full physics simulation.

    Unlike Hopper and Walker, HalfCheetah has NO early termination
    and NO healthy reward - only forward velocity minus control cost.
    """

    var base: MuJoCoEnvBase3D
    var step_count: Int
    var max_steps: Int
    var prev_x: Float64

    fn __init__(out self, max_steps: Int = 1000):
        self.base = MuJoCoEnvBase3D(
            HalfCheetah3DConstants.NUM_BODIES,
            HalfCheetah3DConstants.NUM_JOINTS,
        )
        self.step_count = 0
        self.max_steps = max_steps
        self.prev_x = 0.0
        self._setup_bodies()
        self._setup_joints()

    fn _setup_bodies(mut self):
        """Initialize body states."""
        # Torso (horizontal orientation)
        self.base.bodies[HalfCheetah3DConstants.TORSO] = Body3DState.create(
            Vec3(0, 0, HalfCheetah3DConstants.INITIAL_HEIGHT),
            Quat.identity(),
            HalfCheetah3DConstants.TORSO_MASS,
        )

        # Back leg (at rear of torso)
        var back_hip_x = -HalfCheetah3DConstants.TORSO_LENGTH / 2
        var thigh_z = HalfCheetah3DConstants.INITIAL_HEIGHT - HalfCheetah3DConstants.THIGH_LENGTH / 2

        self.base.bodies[HalfCheetah3DConstants.BACK_THIGH] = Body3DState.create(
            Vec3(back_hip_x, 0, thigh_z), Quat.identity(), HalfCheetah3DConstants.THIGH_MASS
        )

        var shin_z = thigh_z - HalfCheetah3DConstants.THIGH_LENGTH / 2 - HalfCheetah3DConstants.SHIN_LENGTH / 2
        self.base.bodies[HalfCheetah3DConstants.BACK_SHIN] = Body3DState.create(
            Vec3(back_hip_x, 0, shin_z), Quat.identity(), HalfCheetah3DConstants.SHIN_MASS
        )

        var foot_z = shin_z - HalfCheetah3DConstants.SHIN_LENGTH / 2 - 0.02
        self.base.bodies[HalfCheetah3DConstants.BACK_FOOT] = Body3DState.create(
            Vec3(back_hip_x, 0, foot_z), Quat.identity(), HalfCheetah3DConstants.FOOT_MASS
        )

        # Front leg (at front of torso)
        var front_hip_x = HalfCheetah3DConstants.TORSO_LENGTH / 2

        self.base.bodies[HalfCheetah3DConstants.FRONT_THIGH] = Body3DState.create(
            Vec3(front_hip_x, 0, thigh_z), Quat.identity(), HalfCheetah3DConstants.THIGH_MASS
        )

        self.base.bodies[HalfCheetah3DConstants.FRONT_SHIN] = Body3DState.create(
            Vec3(front_hip_x, 0, shin_z), Quat.identity(), HalfCheetah3DConstants.SHIN_MASS
        )

        self.base.bodies[HalfCheetah3DConstants.FRONT_FOOT] = Body3DState.create(
            Vec3(front_hip_x, 0, foot_z), Quat.identity(), HalfCheetah3DConstants.FOOT_MASS
        )

    fn _setup_joints(mut self):
        """Initialize joint states."""
        self.base.joints[HalfCheetah3DConstants.BACK_HIP] = JointState.with_limits(
            HalfCheetah3DConstants.BACK_HIP_LOWER, HalfCheetah3DConstants.BACK_HIP_UPPER
        )
        self.base.joints[HalfCheetah3DConstants.BACK_KNEE] = JointState.with_limits(
            HalfCheetah3DConstants.BACK_KNEE_LOWER, HalfCheetah3DConstants.BACK_KNEE_UPPER
        )
        self.base.joints[HalfCheetah3DConstants.BACK_ANKLE] = JointState.with_limits(
            HalfCheetah3DConstants.BACK_ANKLE_LOWER, HalfCheetah3DConstants.BACK_ANKLE_UPPER
        )
        self.base.joints[HalfCheetah3DConstants.FRONT_HIP] = JointState.with_limits(
            HalfCheetah3DConstants.FRONT_HIP_LOWER, HalfCheetah3DConstants.FRONT_HIP_UPPER
        )
        self.base.joints[HalfCheetah3DConstants.FRONT_KNEE] = JointState.with_limits(
            HalfCheetah3DConstants.FRONT_KNEE_LOWER, HalfCheetah3DConstants.FRONT_KNEE_UPPER
        )
        self.base.joints[HalfCheetah3DConstants.FRONT_ANKLE] = JointState.with_limits(
            HalfCheetah3DConstants.FRONT_ANKLE_LOWER, HalfCheetah3DConstants.FRONT_ANKLE_UPPER
        )

    fn reset(mut self) -> List[Float64]:
        """Reset environment to initial state."""
        self.step_count = 0
        self.base.time = 0.0
        self._setup_bodies()
        self._setup_joints()
        self._forward_kinematics()
        self.prev_x = self.base.bodies[HalfCheetah3DConstants.TORSO].position.x
        return self.get_observation()

    fn step(mut self, action: List[Float64]) -> Tuple[List[Float64], Float64, Bool]:
        """Take one environment step."""
        # Scale and clamp actions
        var scaled_action = List[Float64]()
        for i in range(min(len(action), HalfCheetah3DConstants.ACTION_DIM)):
            var a = action[i]
            if a < -1.0:
                a = -1.0
            elif a > 1.0:
                a = 1.0
            scaled_action.append(a * HalfCheetah3DConstants.ACTION_SCALE)

        while len(scaled_action) < HalfCheetah3DConstants.ACTION_DIM:
            scaled_action.append(0.0)

        # Apply torques
        for i in range(HalfCheetah3DConstants.NUM_JOINTS):
            self.base.apply_joint_torque(i, scaled_action[i])

        # Simulate
        for _ in range(self.base.frame_skip):
            self._physics_step()

        self.step_count += 1

        # Compute reward (NO healthy reward for HalfCheetah!)
        var x_now = self.base.bodies[HalfCheetah3DConstants.TORSO].position.x
        var forward_velocity = (x_now - self.prev_x) / (
            self.base.timestep * Float64(self.base.frame_skip)
        )
        self.prev_x = x_now

        var forward_reward = HalfCheetah3DConstants.FORWARD_REWARD_WEIGHT * forward_velocity
        var ctrl_cost = self.base.compute_control_cost(action, HalfCheetah3DConstants.CTRL_COST_WEIGHT)

        var reward = forward_reward - ctrl_cost

        # NO early termination for HalfCheetah
        var done = self.step_count >= self.max_steps

        return (self.get_observation(), reward, done)

    fn _physics_step(mut self):
        """Perform one physics step."""
        var dt = self.base.timestep
        var gravity = MuJoCoConstants.GRAVITY

        # Integrate joints
        for i in range(HalfCheetah3DConstants.NUM_JOINTS):
            var torque = self.base.joints[i].torque
            var inertia = 0.05
            var accel = torque / inertia
            self.base.joints[i].velocity += accel * dt
            self.base.joints[i].angle += self.base.joints[i].velocity * dt
            self.base.joints[i].clamp_angle()

        self._forward_kinematics()

        # Apply gravity
        for i in range(HalfCheetah3DConstants.NUM_BODIES):
            self.base.bodies[i].linear_velocity.z += gravity * dt

        self.base.detect_ground_contacts()
        self._resolve_contacts()

        # Integrate positions
        for i in range(HalfCheetah3DConstants.NUM_BODIES):
            self.base.bodies[i].position.x += self.base.bodies[i].linear_velocity.x * dt
            self.base.bodies[i].position.y += self.base.bodies[i].linear_velocity.y * dt
            self.base.bodies[i].position.z += self.base.bodies[i].linear_velocity.z * dt

        self.base.time += dt

    fn _forward_kinematics(mut self):
        """Update body positions based on joint angles."""
        var torso = self.base.bodies[HalfCheetah3DConstants.TORSO]

        # Back leg chain
        var b_hip_angle = self.base.joints[HalfCheetah3DConstants.BACK_HIP].angle
        var b_knee_angle = self.base.joints[HalfCheetah3DConstants.BACK_KNEE].angle
        var b_ankle_angle = self.base.joints[HalfCheetah3DConstants.BACK_ANKLE].angle

        var back_hip_pos = Vec3(
            torso.position.x - HalfCheetah3DConstants.TORSO_LENGTH / 2,
            torso.position.y,
            torso.position.z,
        )

        var b_thigh_offset_x = -HalfCheetah3DConstants.THIGH_LENGTH / 2 * sin(b_hip_angle)
        var b_thigh_offset_z = -HalfCheetah3DConstants.THIGH_LENGTH / 2 * cos(b_hip_angle)
        self.base.bodies[HalfCheetah3DConstants.BACK_THIGH].position = Vec3(
            back_hip_pos.x + b_thigh_offset_x, back_hip_pos.y, back_hip_pos.z + b_thigh_offset_z
        )

        var b_knee_pos = Vec3(back_hip_pos.x + 2 * b_thigh_offset_x, back_hip_pos.y, back_hip_pos.z + 2 * b_thigh_offset_z)
        var b_total_angle = b_hip_angle + b_knee_angle
        var b_shin_offset_x = -HalfCheetah3DConstants.SHIN_LENGTH / 2 * sin(b_total_angle)
        var b_shin_offset_z = -HalfCheetah3DConstants.SHIN_LENGTH / 2 * cos(b_total_angle)
        self.base.bodies[HalfCheetah3DConstants.BACK_SHIN].position = Vec3(
            b_knee_pos.x + b_shin_offset_x, b_knee_pos.y, b_knee_pos.z + b_shin_offset_z
        )

        var b_ankle_pos = Vec3(b_knee_pos.x + 2 * b_shin_offset_x, b_knee_pos.y, b_knee_pos.z + 2 * b_shin_offset_z)
        var b_foot_angle = b_total_angle + b_ankle_angle
        self.base.bodies[HalfCheetah3DConstants.BACK_FOOT].position = Vec3(
            b_ankle_pos.x + HalfCheetah3DConstants.FOOT_LENGTH / 2 * cos(b_foot_angle),
            b_ankle_pos.y,
            b_ankle_pos.z - 0.02,
        )

        # Front leg chain
        var f_hip_angle = self.base.joints[HalfCheetah3DConstants.FRONT_HIP].angle
        var f_knee_angle = self.base.joints[HalfCheetah3DConstants.FRONT_KNEE].angle
        var f_ankle_angle = self.base.joints[HalfCheetah3DConstants.FRONT_ANKLE].angle

        var front_hip_pos = Vec3(
            torso.position.x + HalfCheetah3DConstants.TORSO_LENGTH / 2,
            torso.position.y,
            torso.position.z,
        )

        var f_thigh_offset_x = -HalfCheetah3DConstants.THIGH_LENGTH / 2 * sin(f_hip_angle)
        var f_thigh_offset_z = -HalfCheetah3DConstants.THIGH_LENGTH / 2 * cos(f_hip_angle)
        self.base.bodies[HalfCheetah3DConstants.FRONT_THIGH].position = Vec3(
            front_hip_pos.x + f_thigh_offset_x, front_hip_pos.y, front_hip_pos.z + f_thigh_offset_z
        )

        var f_knee_pos = Vec3(front_hip_pos.x + 2 * f_thigh_offset_x, front_hip_pos.y, front_hip_pos.z + 2 * f_thigh_offset_z)
        var f_total_angle = f_hip_angle + f_knee_angle
        var f_shin_offset_x = -HalfCheetah3DConstants.SHIN_LENGTH / 2 * sin(f_total_angle)
        var f_shin_offset_z = -HalfCheetah3DConstants.SHIN_LENGTH / 2 * cos(f_total_angle)
        self.base.bodies[HalfCheetah3DConstants.FRONT_SHIN].position = Vec3(
            f_knee_pos.x + f_shin_offset_x, f_knee_pos.y, f_knee_pos.z + f_shin_offset_z
        )

        var f_ankle_pos = Vec3(f_knee_pos.x + 2 * f_shin_offset_x, f_knee_pos.y, f_knee_pos.z + 2 * f_shin_offset_z)
        var f_foot_angle = f_total_angle + f_ankle_angle
        self.base.bodies[HalfCheetah3DConstants.FRONT_FOOT].position = Vec3(
            f_ankle_pos.x + HalfCheetah3DConstants.FOOT_LENGTH / 2 * cos(f_foot_angle),
            f_ankle_pos.y,
            f_ankle_pos.z - 0.02,
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
        obs.append(self.base.bodies[HalfCheetah3DConstants.TORSO].position.z)

        # Pitch
        obs.append(self.base._get_pitch(self.base.bodies[HalfCheetah3DConstants.TORSO].orientation))

        # Joint angles (6)
        for i in range(HalfCheetah3DConstants.NUM_JOINTS):
            obs.append(self.base.joints[i].angle)

        # Velocities
        obs.append(self.base.bodies[HalfCheetah3DConstants.TORSO].linear_velocity.x)
        obs.append(self.base.bodies[HalfCheetah3DConstants.TORSO].linear_velocity.z)
        obs.append(self.base.bodies[HalfCheetah3DConstants.TORSO].angular_velocity.y)

        # Joint velocities (6)
        for i in range(HalfCheetah3DConstants.NUM_JOINTS):
            obs.append(self.base.joints[i].velocity)

        return obs^
