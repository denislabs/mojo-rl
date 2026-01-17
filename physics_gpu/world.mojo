"""PhysicsWorld - Central orchestrator for GPU-accelerated physics simulation.

The PhysicsWorld coordinates:
1. Velocity integration (apply forces and gravity)
2. Collision detection (generate contact manifolds)
3. Constraint solving (resolve contacts with impulses)
4. Position integration (update positions with new velocities)

This follows Box2D's simulation loop order for stable, energy-conserving physics.
"""

from layout import LayoutTensor, Layout
from gpu.host import DeviceContext, DeviceBuffer

from .constants import (
    dtype,
    TPB,
    BODY_STATE_SIZE,
    SHAPE_MAX_SIZE,
    CONTACT_DATA_SIZE,
    IDX_FX,
    IDX_FY,
    IDX_TAU,
    DEFAULT_GRAVITY_X,
    DEFAULT_GRAVITY_Y,
    DEFAULT_DT,
)
from .integrators.euler import SemiImplicitEuler
from .collision.flat_terrain import FlatTerrainCollision
from .solvers.impulse import ImpulseSolver


struct PhysicsWorld[
    BATCH: Int,
    NUM_BODIES: Int,
    NUM_SHAPES: Int,
    MAX_CONTACTS_PER_ENV: Int = 16,
]:
    """GPU-accelerated physics world for batched RL environments.

    Template Parameters:
        BATCH: Number of parallel environments.
        NUM_BODIES: Number of bodies per environment.
        NUM_SHAPES: Total number of shape definitions.
        MAX_CONTACTS_PER_ENV: Maximum contacts per environment.

    The world maintains all state in flat arrays suitable for GPU execution.
    Both CPU and GPU execution paths use the same data layout for consistency.
    """

    # CPU state storage using List (handles memory management)
    var bodies_list: List[Scalar[dtype]]
    var shapes_list: List[Scalar[dtype]]
    var forces_list: List[Scalar[dtype]]
    var contacts_list: List[Scalar[dtype]]
    var contact_counts_list: List[Scalar[dtype]]

    # Physics parameters
    var gravity_x: Scalar[dtype]
    var gravity_y: Scalar[dtype]
    var dt: Scalar[dtype]

    # Components
    var integrator: SemiImplicitEuler
    var collision: FlatTerrainCollision
    var solver: ImpulseSolver

    # =========================================================================
    # Initialization
    # =========================================================================

    fn __init__(
        out self,
        ground_y: Float64 = 0.0,
        gravity_x: Float64 = DEFAULT_GRAVITY_X,
        gravity_y: Float64 = DEFAULT_GRAVITY_Y,
        dt: Float64 = DEFAULT_DT,
        friction: Float64 = 0.3,
        restitution: Float64 = 0.0,
    ):
        """Initialize the physics world.

        Args:
            ground_y: Y coordinate of flat ground.
            gravity_x: Gravity X component.
            gravity_y: Gravity Y component.
            dt: Physics timestep.
            friction: Contact friction coefficient.
            restitution: Contact restitution (bounciness).
        """
        # Calculate sizes
        comptime BODIES_SIZE = Self.BATCH * Self.NUM_BODIES * BODY_STATE_SIZE
        comptime SHAPES_SIZE = Self.NUM_SHAPES * SHAPE_MAX_SIZE
        comptime FORCES_SIZE = Self.BATCH * Self.NUM_BODIES * 3
        comptime CONTACTS_SIZE = Self.BATCH * Self.MAX_CONTACTS_PER_ENV * CONTACT_DATA_SIZE
        comptime COUNTS_SIZE = Self.BATCH

        # Initialize lists with zeros
        self.bodies_list = List[Scalar[dtype]](capacity=BODIES_SIZE)
        self.shapes_list = List[Scalar[dtype]](capacity=SHAPES_SIZE)
        self.forces_list = List[Scalar[dtype]](capacity=FORCES_SIZE)
        self.contacts_list = List[Scalar[dtype]](capacity=CONTACTS_SIZE)
        self.contact_counts_list = List[Scalar[dtype]](capacity=COUNTS_SIZE)

        # Fill with zeros
        for _ in range(BODIES_SIZE):
            self.bodies_list.append(Scalar[dtype](0))
        for _ in range(SHAPES_SIZE):
            self.shapes_list.append(Scalar[dtype](0))
        for _ in range(FORCES_SIZE):
            self.forces_list.append(Scalar[dtype](0))
        for _ in range(CONTACTS_SIZE):
            self.contacts_list.append(Scalar[dtype](0))
        for _ in range(COUNTS_SIZE):
            self.contact_counts_list.append(Scalar[dtype](0))

        # Store parameters
        self.gravity_x = Scalar[dtype](gravity_x)
        self.gravity_y = Scalar[dtype](gravity_y)
        self.dt = Scalar[dtype](dt)

        # Initialize components
        self.integrator = SemiImplicitEuler()
        self.collision = FlatTerrainCollision(ground_y)
        self.solver = ImpulseSolver(friction, restitution)

    # =========================================================================
    # Tensor Views (zero-copy)
    # =========================================================================

    @always_inline
    fn get_bodies_tensor(
        mut self,
    ) -> LayoutTensor[
        dtype,
        Layout.row_major(Self.BATCH, Self.NUM_BODIES, BODY_STATE_SIZE),
        MutAnyOrigin,
    ]:
        """Get tensor view of body state."""
        return LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ](self.bodies_list.unsafe_ptr())

    @always_inline
    fn get_shapes_tensor(
        mut self,
    ) -> LayoutTensor[
        dtype, Layout.row_major(Self.NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
    ]:
        """Get tensor view of shapes."""
        return LayoutTensor[
            dtype,
            Layout.row_major(Self.NUM_SHAPES, SHAPE_MAX_SIZE),
            MutAnyOrigin,
        ](self.shapes_list.unsafe_ptr())

    @always_inline
    fn get_forces_tensor(
        mut self,
    ) -> LayoutTensor[
        dtype, Layout.row_major(Self.BATCH, Self.NUM_BODIES, 3), MutAnyOrigin
    ]:
        """Get tensor view of forces."""
        return LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.NUM_BODIES, 3), MutAnyOrigin
        ](self.forces_list.unsafe_ptr())

    @always_inline
    fn get_contacts_tensor(
        mut self,
    ) -> LayoutTensor[
        dtype,
        Layout.row_major(Self.BATCH, Self.MAX_CONTACTS_PER_ENV, CONTACT_DATA_SIZE),
        MutAnyOrigin,
    ]:
        """Get tensor view of contacts."""
        return LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.MAX_CONTACTS_PER_ENV, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ](self.contacts_list.unsafe_ptr())

    @always_inline
    fn get_contact_counts_tensor(
        mut self,
    ) -> LayoutTensor[dtype, Layout.row_major(Self.BATCH), MutAnyOrigin]:
        """Get tensor view of contact counts."""
        return LayoutTensor[dtype, Layout.row_major(Self.BATCH), MutAnyOrigin](
            self.contact_counts_list.unsafe_ptr()
        )

    # =========================================================================
    # CPU Physics Step
    # =========================================================================

    fn step(mut self):
        """Execute one physics step on CPU.

        This follows Box2D's simulation order:
        1. Integrate velocities (apply forces and gravity)
        2. Detect collisions (generate contacts)
        3. Solve velocity constraints (apply impulses)
        4. Integrate positions (update using new velocities)
        5. Solve position constraints (resolve penetration)
        6. Clear forces
        """
        # Get tensor views
        var bodies = self.get_bodies_tensor()
        var shapes = self.get_shapes_tensor()
        var forces = self.get_forces_tensor()
        var contacts = self.get_contacts_tensor()
        var contact_counts = self.get_contact_counts_tensor()

        # 1. Integrate velocities
        self.integrator.integrate_velocities[Self.BATCH, Self.NUM_BODIES](
            bodies, forces, self.gravity_x, self.gravity_y, self.dt
        )

        # 2. Detect collisions
        self.collision.detect[Self.BATCH, Self.NUM_BODIES, Self.NUM_SHAPES, Self.MAX_CONTACTS_PER_ENV](
            bodies, shapes, contacts, contact_counts
        )

        # 3. Solve velocity constraints (multiple iterations)
        for _ in range(ImpulseSolver.VELOCITY_ITERATIONS):
            self.solver.solve_velocity[Self.BATCH, Self.NUM_BODIES, Self.MAX_CONTACTS_PER_ENV](
                bodies, contacts, contact_counts
            )

        # 4. Integrate positions
        self.integrator.integrate_positions[Self.BATCH, Self.NUM_BODIES](bodies, self.dt)

        # 5. Solve position constraints (multiple iterations)
        for _ in range(ImpulseSolver.POSITION_ITERATIONS):
            self.solver.solve_position[Self.BATCH, Self.NUM_BODIES, Self.MAX_CONTACTS_PER_ENV](
                bodies, contacts, contact_counts
            )

        # 6. Clear forces for next step
        self._clear_forces()

    fn _clear_forces(mut self):
        """Clear accumulated forces after physics step."""
        var bodies = self.get_bodies_tensor()
        for env in range(Self.BATCH):
            for body in range(Self.NUM_BODIES):
                bodies[env, body, IDX_FX] = Scalar[dtype](0)
                bodies[env, body, IDX_FY] = Scalar[dtype](0)
                bodies[env, body, IDX_TAU] = Scalar[dtype](0)

    # =========================================================================
    # GPU Physics Step
    # =========================================================================

    fn step_gpu(mut self, ctx: DeviceContext) raises:
        """Execute one physics step on GPU.

        This mirrors the CPU step but uses GPU kernels.
        Data is transferred to GPU, processed, and transferred back.
        """
        # Calculate buffer sizes
        comptime BODIES_SIZE = Self.BATCH * Self.NUM_BODIES * BODY_STATE_SIZE
        comptime SHAPES_SIZE = Self.NUM_SHAPES * SHAPE_MAX_SIZE
        comptime FORCES_SIZE = Self.BATCH * Self.NUM_BODIES * 3
        comptime CONTACTS_SIZE = Self.BATCH * Self.MAX_CONTACTS_PER_ENV * CONTACT_DATA_SIZE
        comptime COUNTS_SIZE = Self.BATCH

        # Allocate GPU buffers
        var bodies_buf = ctx.enqueue_create_buffer[dtype](BODIES_SIZE)
        var shapes_buf = ctx.enqueue_create_buffer[dtype](SHAPES_SIZE)
        var forces_buf = ctx.enqueue_create_buffer[dtype](FORCES_SIZE)
        var contacts_buf = ctx.enqueue_create_buffer[dtype](CONTACTS_SIZE)
        var contact_counts_buf = ctx.enqueue_create_buffer[dtype](COUNTS_SIZE)

        # Copy to GPU
        ctx.enqueue_copy(bodies_buf, self.bodies_list.unsafe_ptr())
        ctx.enqueue_copy(shapes_buf, self.shapes_list.unsafe_ptr())
        ctx.enqueue_copy(forces_buf, self.forces_list.unsafe_ptr())

        # 1. Integrate velocities
        SemiImplicitEuler.integrate_velocities_gpu[Self.BATCH, Self.NUM_BODIES](
            ctx, bodies_buf, forces_buf, self.gravity_x, self.gravity_y, self.dt
        )

        # 2. Detect collisions
        FlatTerrainCollision.detect_gpu[
            Self.BATCH, Self.NUM_BODIES, Self.NUM_SHAPES, Self.MAX_CONTACTS_PER_ENV
        ](
            ctx,
            bodies_buf,
            shapes_buf,
            contacts_buf,
            contact_counts_buf,
            self.collision.ground_y,
        )

        # 3. Solve velocity constraints (multiple iterations)
        for _ in range(ImpulseSolver.VELOCITY_ITERATIONS):
            ImpulseSolver.solve_velocity_gpu[
                Self.BATCH, Self.NUM_BODIES, Self.MAX_CONTACTS_PER_ENV
            ](
                ctx,
                bodies_buf,
                contacts_buf,
                contact_counts_buf,
                self.solver.friction,
                self.solver.restitution,
            )

        # 4. Integrate positions
        SemiImplicitEuler.integrate_positions_gpu[Self.BATCH, Self.NUM_BODIES](
            ctx, bodies_buf, self.dt
        )

        # 5. Solve position constraints (multiple iterations)
        for _ in range(ImpulseSolver.POSITION_ITERATIONS):
            ImpulseSolver.solve_position_gpu[
                Self.BATCH, Self.NUM_BODIES, Self.MAX_CONTACTS_PER_ENV
            ](
                ctx,
                bodies_buf,
                contacts_buf,
                contact_counts_buf,
                self.solver.baumgarte,
                self.solver.slop,
            )

        # Copy back to CPU
        ctx.enqueue_copy(self.bodies_list.unsafe_ptr(), bodies_buf)
        ctx.enqueue_copy(self.contacts_list.unsafe_ptr(), contacts_buf)
        ctx.enqueue_copy(self.contact_counts_list.unsafe_ptr(), contact_counts_buf)

        # Synchronize
        ctx.synchronize()

        # Clear forces on CPU (simpler than another kernel)
        self._clear_forces()

    # =========================================================================
    # Helper Methods
    # =========================================================================

    fn apply_force(mut self, env: Int, body: Int, fx: Float64, fy: Float64):
        """Apply a force to a body (accumulated until step)."""
        var forces = self.get_forces_tensor()
        forces[env, body, 0] = forces[env, body, 0] + Scalar[dtype](fx)
        forces[env, body, 1] = forces[env, body, 1] + Scalar[dtype](fy)

    fn apply_torque(mut self, env: Int, body: Int, torque: Float64):
        """Apply a torque to a body (accumulated until step)."""
        var forces = self.get_forces_tensor()
        forces[env, body, 2] = forces[env, body, 2] + Scalar[dtype](torque)

    fn set_body_position(mut self, env: Int, body: Int, x: Float64, y: Float64):
        """Set body position."""
        var bodies = self.get_bodies_tensor()
        bodies[env, body, 0] = Scalar[dtype](x)
        bodies[env, body, 1] = Scalar[dtype](y)

    fn set_body_angle(mut self, env: Int, body: Int, angle: Float64):
        """Set body angle."""
        var bodies = self.get_bodies_tensor()
        bodies[env, body, 2] = Scalar[dtype](angle)

    fn set_body_velocity(
        mut self, env: Int, body: Int, vx: Float64, vy: Float64, omega: Float64
    ):
        """Set body linear and angular velocity."""
        var bodies = self.get_bodies_tensor()
        bodies[env, body, 3] = Scalar[dtype](vx)
        bodies[env, body, 4] = Scalar[dtype](vy)
        bodies[env, body, 5] = Scalar[dtype](omega)

    fn get_body_x(mut self, env: Int, body: Int) -> Scalar[dtype]:
        """Get body x position."""
        var bodies = self.get_bodies_tensor()
        return rebind[Scalar[dtype]](bodies[env, body, 0])

    fn get_body_y(mut self, env: Int, body: Int) -> Scalar[dtype]:
        """Get body y position."""
        var bodies = self.get_bodies_tensor()
        return rebind[Scalar[dtype]](bodies[env, body, 1])

    fn get_body_angle(mut self, env: Int, body: Int) -> Scalar[dtype]:
        """Get body angle."""
        var bodies = self.get_bodies_tensor()
        return rebind[Scalar[dtype]](bodies[env, body, 2])

    fn get_body_vx(mut self, env: Int, body: Int) -> Scalar[dtype]:
        """Get body x velocity."""
        var bodies = self.get_bodies_tensor()
        return rebind[Scalar[dtype]](bodies[env, body, 3])

    fn get_body_vy(mut self, env: Int, body: Int) -> Scalar[dtype]:
        """Get body y velocity."""
        var bodies = self.get_bodies_tensor()
        return rebind[Scalar[dtype]](bodies[env, body, 4])

    fn get_body_omega(mut self, env: Int, body: Int) -> Scalar[dtype]:
        """Get body angular velocity."""
        var bodies = self.get_bodies_tensor()
        return rebind[Scalar[dtype]](bodies[env, body, 5])

    fn set_body_mass(
        mut self, env: Int, body: Int, mass: Float64, inertia: Float64
    ):
        """Set body mass properties. Use mass=0 for static bodies."""
        var bodies = self.get_bodies_tensor()
        bodies[env, body, 9] = Scalar[dtype](mass)
        if mass > 0:
            bodies[env, body, 10] = Scalar[dtype](1.0 / mass)
            if inertia > 0:
                bodies[env, body, 11] = Scalar[dtype](1.0 / inertia)
            else:
                bodies[env, body, 11] = Scalar[dtype](0)
        else:
            bodies[env, body, 10] = Scalar[dtype](0)
            bodies[env, body, 11] = Scalar[dtype](0)

    fn set_body_shape(mut self, env: Int, body: Int, shape_idx: Int):
        """Set body shape reference."""
        var bodies = self.get_bodies_tensor()
        bodies[env, body, 12] = Scalar[dtype](shape_idx)

    fn define_polygon_shape(
        mut self,
        shape_idx: Int,
        vertices_x: List[Float64],
        vertices_y: List[Float64],
    ):
        """Define a polygon shape. Vertices should be in CCW order."""
        var shapes = self.get_shapes_tensor()
        var n_verts = min(len(vertices_x), 8)

        shapes[shape_idx, 0] = Scalar[dtype](0)  # SHAPE_POLYGON
        shapes[shape_idx, 1] = Scalar[dtype](n_verts)

        for i in range(n_verts):
            shapes[shape_idx, 2 + i * 2] = Scalar[dtype](vertices_x[i])
            shapes[shape_idx, 3 + i * 2] = Scalar[dtype](vertices_y[i])

    fn define_circle_shape(
        mut self,
        shape_idx: Int,
        radius: Float64,
        center_x: Float64 = 0.0,
        center_y: Float64 = 0.0,
    ):
        """Define a circle shape."""
        var shapes = self.get_shapes_tensor()
        shapes[shape_idx, 0] = Scalar[dtype](1)  # SHAPE_CIRCLE
        shapes[shape_idx, 1] = Scalar[dtype](radius)
        shapes[shape_idx, 2] = Scalar[dtype](center_x)
        shapes[shape_idx, 3] = Scalar[dtype](center_y)

    fn get_contact_count(mut self, env: Int) -> Int:
        """Get number of active contacts for an environment."""
        var contact_counts = self.get_contact_counts_tensor()
        return Int(contact_counts[env])

    fn reset_env(mut self, env: Int):
        """Reset a single environment to initial state."""
        var bodies = self.get_bodies_tensor()
        var forces = self.get_forces_tensor()
        var contacts = self.get_contacts_tensor()
        var contact_counts = self.get_contact_counts_tensor()

        # Clear body velocities and forces
        for body in range(Self.NUM_BODIES):
            bodies[env, body, 3] = Scalar[dtype](0)  # vx
            bodies[env, body, 4] = Scalar[dtype](0)  # vy
            bodies[env, body, 5] = Scalar[dtype](0)  # omega
            forces[env, body, 0] = Scalar[dtype](0)
            forces[env, body, 1] = Scalar[dtype](0)
            forces[env, body, 2] = Scalar[dtype](0)

        # Clear contacts
        for c in range(Self.MAX_CONTACTS_PER_ENV):
            for i in range(CONTACT_DATA_SIZE):
                contacts[env, c, i] = Scalar[dtype](0)
        contact_counts[env] = Scalar[dtype](0)
