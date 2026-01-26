"""Base classes and utilities for MuJoCo-style 3D environments."""

from math import sqrt, cos, sin, pi
from math3d import Vec3, Quat, Mat3, Mat4
from physics3d import (
    PhysicsState3D,
    Body3D,
    Contact3D,
    Hinge3D,
    Ball3D,
    PDController,
)


struct MuJoCoConstants:
    """Common constants for MuJoCo-style environments."""

    # Physics
    alias GRAVITY: Float64 = -9.81
    alias TIMESTEP: Float64 = 0.002  # 500 Hz physics
    alias FRAME_SKIP: Int = 5  # 100 Hz control

    # Solver
    alias VELOCITY_ITERATIONS: Int = 6
    alias POSITION_ITERATIONS: Int = 2

    # Rendering
    alias WINDOW_WIDTH: Int = 800
    alias WINDOW_HEIGHT: Int = 600

    # Contact
    alias GROUND_FRICTION: Float64 = 0.9
    alias CONTACT_STIFFNESS: Float64 = 10000.0
    alias CONTACT_DAMPING: Float64 = 500.0


@fieldwise_init
struct Body3DState(ImplicitlyCopyable, Movable):
    """State of a single 3D body for easy access."""

    var position: Vec3
    var orientation: Quat
    var linear_velocity: Vec3
    var angular_velocity: Vec3
    var mass: Float64
    var inv_mass: Float64

    @staticmethod
    fn default() -> Self:
        """Create a default body state."""
        return Self(
            Vec3.zero(),
            Quat.identity(),
            Vec3.zero(),
            Vec3.zero(),
            1.0,
            1.0,
        )

    @staticmethod
    fn create(pos: Vec3, orient: Quat, mass: Float64) -> Self:
        """Create a body state with position, orientation, and mass."""
        var inv_mass = 1.0 / mass if mass > 0 else 0.0
        return Self(pos, orient, Vec3.zero(), Vec3.zero(), mass, inv_mass)


@fieldwise_init
struct JointState(ImplicitlyCopyable, Movable):
    """State of a single joint."""

    var angle: Float64  # Current angle (radians)
    var velocity: Float64  # Angular velocity (rad/s)
    var torque: Float64  # Applied torque
    var lower_limit: Float64
    var upper_limit: Float64

    @staticmethod
    fn default() -> Self:
        """Create a default joint state."""
        return Self(0.0, 0.0, 0.0, -pi, pi)

    @staticmethod
    fn with_limits(lower: Float64, upper: Float64) -> Self:
        """Create a joint state with specific limits."""
        return Self(0.0, 0.0, 0.0, lower, upper)

    fn clamp_angle(mut self):
        """Clamp angle to joint limits."""
        if self.angle < self.lower_limit:
            self.angle = self.lower_limit
            self.velocity = max(0.0, self.velocity)
        elif self.angle > self.upper_limit:
            self.angle = self.upper_limit
            self.velocity = min(0.0, self.velocity)


@fieldwise_init
struct ContactInfo(ImplicitlyCopyable, Movable):
    """Information about a ground contact."""

    var body_index: Int
    var contact_point: Vec3
    var normal: Vec3
    var penetration: Float64
    var normal_impulse: Float64
    var friction_impulse: Vec3

    @staticmethod
    fn default() -> Self:
        """Create a default contact info."""
        return Self(-1, Vec3.zero(), Vec3(0, 0, 1), 0.0, 0.0, Vec3.zero())


struct MuJoCoEnvBase3D:
    """Base class for MuJoCo-style 3D environments.

    This provides common functionality for all locomotion environments:
    - Body state management
    - Joint state management
    - Ground collision detection
    - Reward computation helpers
    - Observation building
    """

    var bodies: List[Body3DState]
    var joints: List[JointState]
    var contacts: List[ContactInfo]
    var timestep: Float64
    var frame_skip: Int
    var time: Float64

    fn __init__(out self, num_bodies: Int, num_joints: Int):
        self.bodies = List[Body3DState]()
        for _ in range(num_bodies):
            self.bodies.append(Body3DState.default())

        self.joints = List[JointState]()
        for _ in range(num_joints):
            self.joints.append(JointState.default())

        self.contacts = List[ContactInfo]()
        self.timestep = MuJoCoConstants.TIMESTEP
        self.frame_skip = MuJoCoConstants.FRAME_SKIP
        self.time = 0.0

    fn get_body_position(self, index: Int) -> Vec3:
        """Get position of body at index."""
        return self.bodies[index].position

    fn get_body_velocity(self, index: Int) -> Vec3:
        """Get linear velocity of body at index."""
        return self.bodies[index].linear_velocity

    fn get_body_angular_velocity(self, index: Int) -> Vec3:
        """Get angular velocity of body at index."""
        return self.bodies[index].angular_velocity

    fn set_body_position(mut self, index: Int, pos: Vec3):
        """Set position of body at index."""
        self.bodies[index].position = pos

    fn set_body_velocity(mut self, index: Int, vel: Vec3):
        """Set linear velocity of body at index."""
        self.bodies[index].linear_velocity = vel

    fn get_joint_angle(self, index: Int) -> Float64:
        """Get angle of joint at index."""
        return self.joints[index].angle

    fn get_joint_velocity(self, index: Int) -> Float64:
        """Get velocity of joint at index."""
        return self.joints[index].velocity

    fn set_joint_angle(mut self, index: Int, angle: Float64):
        """Set angle of joint at index."""
        self.joints[index].angle = angle

    fn set_joint_velocity(mut self, index: Int, velocity: Float64):
        """Set velocity of joint at index."""
        self.joints[index].velocity = velocity

    fn apply_joint_torque(mut self, index: Int, torque: Float64):
        """Apply torque to joint at index."""
        self.joints[index].torque = torque

    fn compute_forward_velocity(self, torso_index: Int) -> Float64:
        """Compute forward (x-direction) velocity of torso."""
        return self.bodies[torso_index].linear_velocity.x

    fn compute_healthy_reward(
        self,
        torso_index: Int,
        min_z: Float64,
        max_z: Float64,
        max_angle: Float64,
    ) -> Float64:
        """Compute healthy reward based on torso height and angle.

        Returns 1.0 if healthy, 0.0 otherwise.
        """
        var z = self.bodies[torso_index].position.z
        if z < min_z or z > max_z:
            return 0.0

        # Check torso angle (pitch)
        var orient = self.bodies[torso_index].orientation
        # Extract pitch angle from quaternion
        var pitch = self._get_pitch(orient)
        if abs(pitch) > max_angle:
            return 0.0

        return 1.0

    fn _get_pitch(self, q: Quat) -> Float64:
        """Extract pitch angle from quaternion."""
        # Pitch = asin(2(qw*qy - qz*qx))
        var sinp = 2.0 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1.0:
            return pi / 2.0 if sinp > 0 else -pi / 2.0

        # Use approximation for small angles
        return sinp  # asin(x) ≈ x for small x

    fn compute_control_cost(self, actions: List[Float64], weight: Float64) -> Float64:
        """Compute control cost as weighted sum of squared actions."""
        var cost = 0.0
        for i in range(len(actions)):
            cost += actions[i] * actions[i]
        return weight * cost

    fn compute_contact_cost(self, weight: Float64) -> Float64:
        """Compute contact cost from contact forces."""
        var cost = 0.0
        for i in range(len(self.contacts)):
            var impulse = self.contacts[i].normal_impulse
            cost += impulse * impulse
        return weight * cost

    fn is_terminated(
        self,
        torso_index: Int,
        min_z: Float64,
        max_z: Float64,
        max_angle: Float64,
    ) -> Bool:
        """Check if episode should terminate."""
        var z = self.bodies[torso_index].position.z
        if z < min_z or z > max_z:
            return True

        var orient = self.bodies[torso_index].orientation
        var pitch = self._get_pitch(orient)
        if abs(pitch) > max_angle:
            return True

        return False

    fn detect_ground_contacts(mut self, ground_z: Float64 = 0.0):
        """Detect contacts between bodies and ground plane."""
        self.contacts = List[ContactInfo]()

        for i in range(len(self.bodies)):
            var body = self.bodies[i]
            # Simple sphere-plane collision (approximate body as sphere)
            var radius = 0.05  # Default body radius
            var bottom_z = body.position.z - radius

            if bottom_z < ground_z:
                var contact = ContactInfo(
                    i,
                    Vec3(body.position.x, body.position.y, ground_z),
                    Vec3(0, 0, 1),
                    ground_z - bottom_z,
                    0.0,
                    Vec3.zero(),
                )
                self.contacts.append(contact^)

    fn has_ground_contact(self, body_index: Int) -> Bool:
        """Check if specific body has ground contact."""
        for i in range(len(self.contacts)):
            if self.contacts[i].body_index == body_index:
                return True
        return False

    fn total_ground_contacts(self) -> Int:
        """Return total number of ground contacts."""
        return len(self.contacts)
