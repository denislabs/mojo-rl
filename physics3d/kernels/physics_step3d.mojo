"""3D Physics World and Simulation Step.

Orchestrates the physics simulation pipeline:
1. Integrate velocities with forces and gravity
2. Detect collisions
3. Solve velocity constraints (contacts and joints)
4. Integrate positions
5. Solve position constraints

GPU support follows the physics2d fused kernel pattern, combining all
physics operations in ONE kernel launch for maximum performance.
"""

from math import sqrt
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer

from math3d import Vec3, Quat
from ..constants import (
    dtype,
    TPB,
    BODY_STATE_SIZE_3D,
    CONTACT_DATA_SIZE_3D,
    IDX_PX,
    IDX_PY,
    IDX_PZ,
    IDX_QW,
    IDX_QX,
    IDX_QY,
    IDX_QZ,
    IDX_VX,
    IDX_VY,
    IDX_VZ,
    IDX_WX,
    IDX_WY,
    IDX_WZ,
    IDX_FX,
    IDX_FY,
    IDX_FZ,
    IDX_TX,
    IDX_TY,
    IDX_TZ,
    IDX_INV_MASS,
    IDX_IXX,
    IDX_IYY,
    IDX_IZZ,
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
    CONTACT_BODY_A_3D,
    CONTACT_BODY_B_3D,
    CONTACT_POINT_X,
    CONTACT_POINT_Y,
    CONTACT_POINT_Z,
    CONTACT_NORMAL_X,
    CONTACT_NORMAL_Y,
    CONTACT_NORMAL_Z,
    CONTACT_DEPTH_3D,
    CONTACT_IMPULSE_N,
    CONTACT_IMPULSE_T1,
    CONTACT_IMPULSE_T2,
    CONTACT_TANGENT1_X,
    CONTACT_TANGENT1_Y,
    CONTACT_TANGENT1_Z,
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
from ..integrators.euler3d import SemiImplicitEuler3DGPU
from ..collision.sphere_plane import SpherePlaneCollisionGPU
from ..solvers.impulse3d import ImpulseSolver3DGPU


struct PhysicsWorld3D[DTYPE: DType]:
    """3D Physics world for MuJoCo-style environments.

    Manages rigid body simulation with contacts and joints.
    """

    var num_bodies: Int
    var gravity: Vec3[Self.DTYPE]
    var dt: Scalar[Self.DTYPE]
    var velocity_iterations: Int
    var position_iterations: Int
    var friction: Scalar[Self.DTYPE]
    var ground_height: Scalar[Self.DTYPE]

    # Body shape information for collision detection
    var shape_types: List[Int]
    var shape_radii: List[Scalar[Self.DTYPE]]
    var shape_half_heights: List[Scalar[Self.DTYPE]]  # For capsules
    var shape_axes: List[Int]  # Capsule axis (0=X, 1=Y, 2=Z)

    fn __init__(
        out self,
        num_bodies: Int,
        gravity: Vec3[Self.DTYPE] = Vec3[Self.DTYPE](
            0.0, 0.0, Scalar[Self.DTYPE](DEFAULT_GRAVITY_Z_3D)
        ),
        dt: Scalar[Self.DTYPE] = Scalar[Self.DTYPE](DEFAULT_DT_3D),
        velocity_iterations: Int = DEFAULT_VELOCITY_ITERATIONS_3D,
        position_iterations: Int = DEFAULT_POSITION_ITERATIONS_3D,
        friction: Scalar[Self.DTYPE] = Scalar[Self.DTYPE](DEFAULT_FRICTION_3D),
        ground_height: Scalar[Self.DTYPE] = Scalar[Self.DTYPE](0.0),
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
        self.shape_radii = List[Scalar[Self.DTYPE]]()
        self.shape_half_heights = List[Scalar[Self.DTYPE]]()
        self.shape_axes = List[Int]()

        for _ in range(num_bodies):
            self.shape_types.append(SHAPE_SPHERE)
            self.shape_radii.append(0.05)  # Default radius
            self.shape_half_heights.append(0.0)
            self.shape_axes.append(2)  # Z-axis default

    fn set_sphere_shape(mut self, body_idx: Int, radius: Scalar[Self.DTYPE]):
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
        radius: Scalar[Self.DTYPE],
        half_height: Scalar[Self.DTYPE],
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
        state: List[Scalar[Self.DTYPE]],
    ) -> List[Contact3D[Self.DTYPE]]:
        """Detect collisions between bodies and ground plane.

        Args:
            state: Physics state array.

        Returns:
            List of contact points.
        """
        var contacts = List[Contact3D[Self.DTYPE]]()

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

    fn step(self, mut state: List[Scalar[Self.DTYPE]]):
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
                solve_contact_velocity[Self.DTYPE](
                    state, contacts[i], self.friction, 0.0
                )

        # Step 4: Integrate positions
        for i in range(self.num_bodies):
            integrate_positions_3d(state, i, self.dt)

        # Step 5: Solve position constraints
        for _ in range(self.position_iterations):
            for i in range(len(contacts)):
                solve_contact_position[Self.DTYPE](state, contacts[i])

    fn step_with_joints(
        self,
        mut state: List[Scalar[Self.DTYPE]],
        joint_body_a: List[Int],
        joint_body_b: List[Int],
        joint_anchor_a: List[Vec3[Self.DTYPE]],
        joint_anchor_b: List[Vec3[Self.DTYPE]],
        joint_axis: List[Vec3[Self.DTYPE]],
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
                solve_contact_position[Self.DTYPE](state, contacts[i])

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
        mut state: List[Scalar[Self.DTYPE]],
        joint_body_a: List[Int],
        joint_body_b: List[Int],
        joint_axis: List[Vec3[Self.DTYPE]],
        torques: List[Scalar[Self.DTYPE]],
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

    fn get_body_position(
        self, state: List[Scalar[Self.DTYPE]], body_idx: Int
    ) -> Vec3[Self.DTYPE]:
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

    fn get_body_orientation(
        self, state: List[Scalar[Self.DTYPE]], body_idx: Int
    ) -> Quat[Self.DTYPE]:
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

    fn get_body_velocity(
        self, state: List[Scalar[Self.DTYPE]], body_idx: Int
    ) -> Vec3[Self.DTYPE]:
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

    fn get_body_angular_velocity(
        self, state: List[Scalar[Self.DTYPE]], body_idx: Int
    ) -> Vec3[Self.DTYPE]:
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


# =============================================================================
# GPU Implementation - Fused Physics Step Kernel
# =============================================================================


struct Physics3DStepKernel:
    """Fused physics step kernel for 3D - runs ENTIRE physics step in ONE kernel.

    This kernel performs:
    1. Velocity integration (forces + gravity → velocities)
    2. Collision detection (bodies vs ground → contacts)
    3. Velocity constraint solving (all iterations)
    4. Position integration (velocities → positions)
    5. Position constraint solving (all iterations)

    ALL IN ONE GPU KERNEL LAUNCH.

    Performance benefit: Reduces kernel launches from 9+ to 1.

    """

    # =========================================================================
    # Fused Physics Step Kernel - Single Environment Logic
    # =========================================================================

    @always_inline
    @staticmethod
    fn _step_single_env[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        CONTACTS_OFFSET: Int,
        CONTACT_COUNT_OFFSET: Int,
        VEL_ITERATIONS: Int,
        POS_ITERATIONS: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        shape_types: LayoutTensor[
            dtype,
            Layout.row_major(NUM_BODIES),
            MutAnyOrigin,
        ],
        shape_radii: LayoutTensor[
            dtype,
            Layout.row_major(NUM_BODIES),
            MutAnyOrigin,
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE_3D),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH),
            MutAnyOrigin,
        ],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        gravity_z: Scalar[dtype],
        dt: Scalar[dtype],
        ground_height: Scalar[dtype],
        friction: Scalar[dtype],
        restitution: Scalar[dtype],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ):
        """Run complete physics step for a single environment.

        This is the core fused logic that can be called from fused kernels
        or standalone kernels.
        """
        # =====================================================================
        # Step 1: Integrate velocities (v' = v + a*dt)
        # =====================================================================
        SemiImplicitEuler3DGPU.integrate_velocities_single_env[
            BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET
        ](env, state, gravity_x, gravity_y, gravity_z, dt)

        # =====================================================================
        # Step 2: Collision detection (spheres vs ground)
        # =====================================================================
        # Reset contact count for this environment
        contact_counts[env] = Scalar[dtype](0)

        # Detect sphere-ground collisions
        SpherePlaneCollisionGPU.detect_single_env_with_separate_contacts[
            BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
        ](
            env,
            state,
            shape_types,
            shape_radii,
            contacts,
            contact_counts,
            ground_height,
        )

        # Get contact count after detection
        var n_contacts = Int(contact_counts[env])

        # =====================================================================
        # Step 3: Velocity constraints (all iterations)
        # =====================================================================
        for _ in range(VEL_ITERATIONS):
            ImpulseSolver3DGPU.solve_velocity_single_env[
                BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
            ](env, state, contacts, n_contacts, friction, restitution)

        # =====================================================================
        # Step 4: Integrate positions (x' = x + v'*dt)
        # =====================================================================
        SemiImplicitEuler3DGPU.integrate_positions_single_env[
            BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET
        ](env, state, dt)

        # =====================================================================
        # Step 5: Position constraints (all iterations)
        # =====================================================================
        for _ in range(POS_ITERATIONS):
            ImpulseSolver3DGPU.solve_position_single_env[
                BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
            ](env, state, contacts, n_contacts, baumgarte, slop)

    # =========================================================================
    # GPU Kernel Entry Point
    # =========================================================================

    @always_inline
    @staticmethod
    fn _step_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        CONTACTS_OFFSET: Int,
        CONTACT_COUNT_OFFSET: Int,
        VEL_ITERATIONS: Int,
        POS_ITERATIONS: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        shape_types: LayoutTensor[
            dtype,
            Layout.row_major(NUM_BODIES),
            MutAnyOrigin,
        ],
        shape_radii: LayoutTensor[
            dtype,
            Layout.row_major(NUM_BODIES),
            MutAnyOrigin,
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE_3D),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH),
            MutAnyOrigin,
        ],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        gravity_z: Scalar[dtype],
        dt: Scalar[dtype],
        ground_height: Scalar[dtype],
        friction: Scalar[dtype],
        restitution: Scalar[dtype],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ):
        """GPU kernel that runs the ENTIRE physics step in one kernel."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        Physics3DStepKernel._step_single_env[
            BATCH,
            NUM_BODIES,
            MAX_CONTACTS,
            STATE_SIZE,
            BODIES_OFFSET,
            CONTACTS_OFFSET,
            CONTACT_COUNT_OFFSET,
            VEL_ITERATIONS,
            POS_ITERATIONS,
        ](
            env,
            state,
            shape_types,
            shape_radii,
            contacts,
            contact_counts,
            gravity_x,
            gravity_y,
            gravity_z,
            dt,
            ground_height,
            friction,
            restitution,
            baumgarte,
            slop,
        )

    # =========================================================================
    # Public GPU API
    # =========================================================================

    @staticmethod
    fn step_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        CONTACTS_OFFSET: Int,
        CONTACT_COUNT_OFFSET: Int,
        VEL_ITERATIONS: Int = 10,
        POS_ITERATIONS: Int = 5,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        shape_types_buf: DeviceBuffer[dtype],
        shape_radii_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        gravity_z: Scalar[dtype],
        dt: Scalar[dtype],
        ground_height: Scalar[dtype],
        friction: Scalar[dtype],
        restitution: Scalar[dtype],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ) raises:
        """Run the ENTIRE physics step in ONE kernel launch.

        This fuses:
        - Velocity integration
        - Collision detection (sphere-ground)
        - Velocity constraints (all iterations)
        - Position integration
        - Position constraints (all iterations)

        Args:
            ctx: GPU device context.
            state_buf: State buffer [BATCH * STATE_SIZE].
            shape_types_buf: Shape types [NUM_BODIES].
            shape_radii_buf: Shape radii [NUM_BODIES].
            contacts_buf: Contact buffer [BATCH * MAX_CONTACTS * CONTACT_DATA_SIZE_3D].
            contact_counts_buf: Contact counts [BATCH].
            gravity_x, gravity_y, gravity_z: Gravity components.
            dt: Time step.
            ground_height: Z-coordinate of ground plane.
            friction: Coulomb friction coefficient.
            restitution: Bounce coefficient.
            baumgarte: Position correction factor.
            slop: Penetration allowance.
        """
        var state = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ](state_buf.unsafe_ptr())
        var shape_types = LayoutTensor[
            dtype,
            Layout.row_major(NUM_BODIES),
            MutAnyOrigin,
        ](shape_types_buf.unsafe_ptr())
        var shape_radii = LayoutTensor[
            dtype,
            Layout.row_major(NUM_BODIES),
            MutAnyOrigin,
        ](shape_radii_buf.unsafe_ptr())
        var contacts = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE_3D),
            MutAnyOrigin,
        ](contacts_buf.unsafe_ptr())
        var contact_counts = LayoutTensor[
            dtype,
            Layout.row_major(BATCH),
            MutAnyOrigin,
        ](contact_counts_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[
                dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            shape_types: LayoutTensor[
                dtype, Layout.row_major(NUM_BODIES), MutAnyOrigin
            ],
            shape_radii: LayoutTensor[
                dtype, Layout.row_major(NUM_BODIES), MutAnyOrigin
            ],
            contacts: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE_3D),
                MutAnyOrigin,
            ],
            contact_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ],
            gravity_x: Scalar[dtype],
            gravity_y: Scalar[dtype],
            gravity_z: Scalar[dtype],
            dt: Scalar[dtype],
            ground_height: Scalar[dtype],
            friction: Scalar[dtype],
            restitution: Scalar[dtype],
            baumgarte: Scalar[dtype],
            slop: Scalar[dtype],
        ):
            Physics3DStepKernel._step_kernel[
                BATCH,
                NUM_BODIES,
                MAX_CONTACTS,
                STATE_SIZE,
                BODIES_OFFSET,
                CONTACTS_OFFSET,
                CONTACT_COUNT_OFFSET,
                VEL_ITERATIONS,
                POS_ITERATIONS,
            ](
                state,
                shape_types,
                shape_radii,
                contacts,
                contact_counts,
                gravity_x,
                gravity_y,
                gravity_z,
                dt,
                ground_height,
                friction,
                restitution,
                baumgarte,
                slop,
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            shape_types,
            shape_radii,
            contacts,
            contact_counts,
            gravity_x,
            gravity_y,
            gravity_z,
            dt,
            ground_height,
            friction,
            restitution,
            baumgarte,
            slop,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn step_gpu_with_layout[
        L: PhysicsLayout3D,
        VEL_ITERATIONS: Int = 10,
        POS_ITERATIONS: Int = 5,
        BATCH: Int = 1024,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        shape_types_buf: DeviceBuffer[dtype],
        shape_radii_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        gravity_z: Scalar[dtype],
        dt: Scalar[dtype],
        ground_height: Scalar[dtype],
        friction: Scalar[dtype],
        restitution: Scalar[dtype],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ) raises:
        """Run physics step using a PhysicsLayout3D for automatic offset computation.

        This is a convenience wrapper that extracts layout parameters automatically.

        Args:
            ctx: GPU device context.
            state_buf: State buffer.
            shape_types_buf: Shape types.
            shape_radii_buf: Shape radii.
            contacts_buf: Contact buffer.
            contact_counts_buf: Contact counts.
            gravity_x, gravity_y, gravity_z: Gravity components.
            dt: Time step.
            ground_height: Ground plane height.
            friction, restitution, baumgarte, slop: Physics parameters.
        """
        Physics3DStepKernel.step_gpu[
            BATCH,
            L.NUM_BODIES,
            L.MAX_CONTACTS,
            L.STATE_SIZE,
            L.BODIES_OFFSET,
            L.CONTACTS_OFFSET,
            L.CONTACT_COUNT_OFFSET,
            VEL_ITERATIONS,
            POS_ITERATIONS,
        ](
            ctx,
            state_buf,
            shape_types_buf,
            shape_radii_buf,
            contacts_buf,
            contact_counts_buf,
            gravity_x,
            gravity_y,
            gravity_z,
            dt,
            ground_height,
            friction,
            restitution,
            baumgarte,
            slop,
        )


# Import layout for convenience method
from ..layout import PhysicsLayout3D
