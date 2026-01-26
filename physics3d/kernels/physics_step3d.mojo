"""3D Physics World and Simulation Step.

Orchestrates the physics simulation pipeline:
1. Integrate velocities with forces and gravity
2. Detect collisions
3. Solve velocity constraints (contacts and joints)
4. Integrate positions
5. Solve position constraints
"""

from math3d import Vec3, Quat
from ..constants import (
    BODY_STATE_SIZE_3D,
    IDX_PX, IDX_PY, IDX_PZ,
    IDX_QW, IDX_QX, IDX_QY, IDX_QZ,
    IDX_SHAPE_3D,
    IDX_BODY_TYPE,
    BODY_DYNAMIC,
    SHAPE_SPHERE,
    SHAPE_CAPSULE,
    DEFAULT_GRAVITY_Z_3D,
    DEFAULT_DT_3D,
    DEFAULT_VELOCITY_ITERATIONS_3D,
    DEFAULT_POSITION_ITERATIONS_3D,
    DEFAULT_FRICTION_3D,
)
from ..collision import (
    Contact3D,
    ContactManifold,
    SpherePlaneCollision,
    CapsulePlaneCollision,
)
from ..integrators import (
    SemiImplicitEuler3D,
    integrate_velocities_3d,
    integrate_positions_3d,
)
from ..solvers import (
    ContactSolver3D,
    solve_contact_velocity,
    solve_contact_position,
)


struct PhysicsWorld3D:
    """3D Physics world for MuJoCo-style environments.

    Manages rigid body simulation with contacts and joints.
    """

    var num_bodies: Int
    var gravity: Vec3
    var dt: Float64
    var velocity_iterations: Int
    var position_iterations: Int
    var friction: Float64
    var ground_height: Float64

    # Body shape information for collision detection
    var shape_types: List[Int]
    var shape_radii: List[Float64]
    var shape_half_heights: List[Float64]  # For capsules
    var shape_axes: List[Int]  # Capsule axis (0=X, 1=Y, 2=Z)

    fn __init__(
        out self,
        num_bodies: Int,
        gravity: Vec3 = Vec3(0.0, 0.0, DEFAULT_GRAVITY_Z_3D),
        dt: Float64 = DEFAULT_DT_3D,
        velocity_iterations: Int = DEFAULT_VELOCITY_ITERATIONS_3D,
        position_iterations: Int = DEFAULT_POSITION_ITERATIONS_3D,
        friction: Float64 = DEFAULT_FRICTION_3D,
        ground_height: Float64 = 0.0,
    ):
        """Initialize physics world.

        Args:
            num_bodies: Number of bodies in the world.
            gravity: Gravity acceleration vector.
            dt: Time step.
            velocity_iterations: Constraint solver velocity iterations.
            position_iterations: Constraint solver position iterations.
            friction: Ground friction coefficient.
            ground_height: Z-coordinate of ground plane.
        """
        self.num_bodies = num_bodies
        self.gravity = gravity
        self.dt = dt
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations
        self.friction = friction
        self.ground_height = ground_height

        # Initialize shape arrays with defaults
        self.shape_types = List[Int]()
        self.shape_radii = List[Float64]()
        self.shape_half_heights = List[Float64]()
        self.shape_axes = List[Int]()

        for _ in range(num_bodies):
            self.shape_types.append(SHAPE_SPHERE)
            self.shape_radii.append(0.05)  # Default radius
            self.shape_half_heights.append(0.0)
            self.shape_axes.append(2)  # Z-axis default

    fn set_sphere_shape(mut self, body_idx: Int, radius: Float64):
        """Set body shape to sphere.

        Args:
            body_idx: Body index.
            radius: Sphere radius.
        """
        self.shape_types[body_idx] = SHAPE_SPHERE
        self.shape_radii[body_idx] = radius
        self.shape_half_heights[body_idx] = 0.0

    fn set_capsule_shape(
        mut self,
        body_idx: Int,
        radius: Float64,
        half_height: Float64,
        axis: Int = 2,
    ):
        """Set body shape to capsule.

        Args:
            body_idx: Body index.
            radius: Capsule radius.
            half_height: Half-height of cylindrical part.
            axis: Local axis (0=X, 1=Y, 2=Z).
        """
        self.shape_types[body_idx] = SHAPE_CAPSULE
        self.shape_radii[body_idx] = radius
        self.shape_half_heights[body_idx] = half_height
        self.shape_axes[body_idx] = axis

    fn detect_ground_contacts(
        self,
        state: List[Float64],
    ) -> List[Contact3D]:
        """Detect collisions between bodies and ground plane.

        Args:
            state: Physics state array.

        Returns:
            List of contact points.
        """
        var contacts = List[Contact3D]()

        for i in range(self.num_bodies):
            var base = i * BODY_STATE_SIZE_3D

            # Check if body is dynamic
            var body_type = Int(state[base + IDX_BODY_TYPE])
            if body_type != BODY_DYNAMIC:
                continue

            var pos = Vec3(
                state[base + IDX_PX],
                state[base + IDX_PY],
                state[base + IDX_PZ],
            )

            var shape_type = self.shape_types[i]
            var radius = self.shape_radii[i]

            if shape_type == SHAPE_SPHERE:
                var contact = SpherePlaneCollision.detect_ground(
                    pos, radius, self.ground_height, i
                )
                if contact.is_valid():
                    contacts.append(contact^)

            elif shape_type == SHAPE_CAPSULE:
                var q = Quat(
                    state[base + IDX_QW],
                    state[base + IDX_QX],
                    state[base + IDX_QY],
                    state[base + IDX_QZ],
                )
                var half_height = self.shape_half_heights[i]
                var axis = self.shape_axes[i]

                var manifold = CapsulePlaneCollision.detect_ground(
                    pos, q, radius, half_height, axis, self.ground_height, i
                )

                for j in range(len(manifold.contacts)):
                    # Copy the contact from manifold
                    var c = Contact3D(
                        body_a=manifold.contacts[j].body_a,
                        body_b=manifold.contacts[j].body_b,
                        point=manifold.contacts[j].point,
                        normal=manifold.contacts[j].normal,
                        depth=manifold.contacts[j].depth,
                    )
                    contacts.append(c^)

        return contacts^

    fn step(self, mut state: List[Float64]):
        """Perform one physics simulation step.

        Pipeline:
        1. Integrate velocities (with gravity)
        2. Detect collisions
        3. Solve velocity constraints
        4. Integrate positions
        5. Solve position constraints

        Args:
            state: Physics state array (modified in place).
        """
        # Step 1: Integrate velocities
        for i in range(self.num_bodies):
            integrate_velocities_3d(state, i, self.gravity, self.dt)

        # Step 2: Detect collisions
        var contacts = self.detect_ground_contacts(state)

        # Step 3: Solve velocity constraints
        for _ in range(self.velocity_iterations):
            for i in range(len(contacts)):
                solve_contact_velocity(state, contacts[i], self.friction, 0.0)

        # Step 4: Integrate positions
        for i in range(self.num_bodies):
            integrate_positions_3d(state, i, self.dt)

        # Step 5: Solve position constraints
        for _ in range(self.position_iterations):
            for i in range(len(contacts)):
                solve_contact_position(state, contacts[i])

    fn step_with_joints(
        self,
        mut state: List[Float64],
        joint_body_a: List[Int],
        joint_body_b: List[Int],
        joint_anchor_a: List[Vec3],
        joint_anchor_b: List[Vec3],
        joint_axis: List[Vec3],
    ):
        """Perform physics step with joint constraints.

        Args:
            state: Physics state array.
            joint_body_a: List of first body indices for each joint.
            joint_body_b: List of second body indices for each joint.
            joint_anchor_a: Local anchors on body A.
            joint_anchor_b: Local anchors on body B.
            joint_axis: Joint axes (for hinge joints).
        """
        from ..solvers import solve_hinge_velocity, solve_hinge_position

        # Step 1: Integrate velocities
        for i in range(self.num_bodies):
            integrate_velocities_3d(state, i, self.gravity, self.dt)

        # Step 2: Detect collisions
        var contacts = self.detect_ground_contacts(state)

        # Step 3: Solve velocity constraints
        var num_joints = len(joint_body_a)

        for _ in range(self.velocity_iterations):
            # Contact constraints
            for i in range(len(contacts)):
                solve_contact_velocity(state, contacts[i], self.friction, 0.0)

            # Joint constraints
            for j in range(num_joints):
                solve_hinge_velocity(
                    state,
                    joint_body_a[j],
                    joint_body_b[j],
                    joint_anchor_a[j],
                    joint_anchor_b[j],
                    joint_axis[j],
                )

        # Step 4: Integrate positions
        for i in range(self.num_bodies):
            integrate_positions_3d(state, i, self.dt)

        # Step 5: Solve position constraints
        for _ in range(self.position_iterations):
            # Contact constraints
            for i in range(len(contacts)):
                solve_contact_position(state, contacts[i])

            # Joint constraints
            for j in range(num_joints):
                solve_hinge_position(
                    state,
                    joint_body_a[j],
                    joint_body_b[j],
                    joint_anchor_a[j],
                    joint_anchor_b[j],
                )

    fn apply_motor_torques(
        self,
        mut state: List[Float64],
        joint_body_a: List[Int],
        joint_body_b: List[Int],
        joint_axis: List[Vec3],
        torques: List[Float64],
    ):
        """Apply motor torques to joints.

        Args:
            state: Physics state array.
            joint_body_a: First body indices.
            joint_body_b: Second body indices.
            joint_axis: Joint axes in world space.
            torques: Motor torques to apply.
        """
        from ..constants import IDX_TX, IDX_TY, IDX_TZ

        var num_joints = len(joint_body_a)

        for j in range(num_joints):
            var body_a = joint_body_a[j]
            var body_b = joint_body_b[j]
            var axis = joint_axis[j]
            var torque = torques[j]

            # Torque vector
            var tau = axis * torque

            # Apply equal and opposite torques
            if body_a >= 0:
                var base_a = body_a * BODY_STATE_SIZE_3D
                state[base_a + IDX_TX] += tau.x
                state[base_a + IDX_TY] += tau.y
                state[base_a + IDX_TZ] += tau.z

            if body_b >= 0:
                var base_b = body_b * BODY_STATE_SIZE_3D
                state[base_b + IDX_TX] -= tau.x
                state[base_b + IDX_TY] -= tau.y
                state[base_b + IDX_TZ] -= tau.z

    fn get_body_position(self, state: List[Float64], body_idx: Int) -> Vec3:
        """Get body center position.

        Args:
            state: Physics state array.
            body_idx: Body index.

        Returns:
            Position vector.
        """
        var base = body_idx * BODY_STATE_SIZE_3D
        return Vec3(
            state[base + IDX_PX],
            state[base + IDX_PY],
            state[base + IDX_PZ],
        )

    fn get_body_orientation(self, state: List[Float64], body_idx: Int) -> Quat:
        """Get body orientation.

        Args:
            state: Physics state array.
            body_idx: Body index.

        Returns:
            Orientation quaternion.
        """
        var base = body_idx * BODY_STATE_SIZE_3D
        return Quat(
            state[base + IDX_QW],
            state[base + IDX_QX],
            state[base + IDX_QY],
            state[base + IDX_QZ],
        )

    fn get_body_velocity(self, state: List[Float64], body_idx: Int) -> Vec3:
        """Get body linear velocity.

        Args:
            state: Physics state array.
            body_idx: Body index.

        Returns:
            Velocity vector.
        """
        var base = body_idx * BODY_STATE_SIZE_3D
        return Vec3(
            state[base + IDX_VX],
            state[base + IDX_VY],
            state[base + IDX_VZ],
        )

    fn get_body_angular_velocity(self, state: List[Float64], body_idx: Int) -> Vec3:
        """Get body angular velocity.

        Args:
            state: Physics state array.
            body_idx: Body index.

        Returns:
            Angular velocity vector.
        """
        var base = body_idx * BODY_STATE_SIZE_3D
        return Vec3(
            state[base + IDX_WX],
            state[base + IDX_WY],
            state[base + IDX_WZ],
        )
