"""PhysicsKernelStrided - Unified physics step orchestrator for strided 2D layouts.

This module provides a single unified function that executes a complete physics
simulation step on GPU. It coordinates all physics components:

1. Velocity integration (gravity + forces)
2. Collision detection (edge terrain or flat terrain)
3. Velocity constraint solving (contacts + joints)
4. Position integration
5. Position constraint solving

This replaces 20+ lines of manual physics component calls with a single function.

Example:
    ```mojo
    from physics_gpu.kernel_strided import PhysicsKernelStrided
    from physics_gpu.layout_strided import LunarLanderLayoutStrided

    alias Layout = LunarLanderLayoutStrided

    # One call instead of many!
    PhysicsKernelStrided.step_gpu[BATCH, Layout](
        ctx,
        states_buf,
        shapes_buf,
        contacts_buf,
        contact_counts_buf,
        config,
        collision_type=CollisionType.EDGE_TERRAIN,
    )
    ```
"""

from layout import LayoutTensor, Layout
from gpu.host import DeviceContext, DeviceBuffer

from .constants import (
    dtype,
    TPB,
    BODY_STATE_SIZE,
    SHAPE_MAX_SIZE,
    CONTACT_DATA_SIZE,
    JOINT_DATA_SIZE,
    DEFAULT_FRICTION,
    DEFAULT_RESTITUTION,
    DEFAULT_BAUMGARTE,
    DEFAULT_SLOP,
    DEFAULT_VELOCITY_ITERATIONS,
    DEFAULT_POSITION_ITERATIONS,
)
from .layout_strided import PhysicsLayoutStrided
from .integrators.euler import SemiImplicitEuler
from .collision.edge_terrain import EdgeTerrainCollision
from .collision.flat_terrain import FlatTerrainCollision
from .solvers.impulse import ImpulseSolver
from .joints.revolute import RevoluteJointSolver


struct PhysicsConfigStrided(Copyable, Movable):
    """Configuration for strided physics simulation.

    Contains all parameters needed for a physics step.
    """

    var gravity_x: Float64
    var gravity_y: Float64
    var dt: Float64
    var friction: Float64
    var restitution: Float64
    var baumgarte: Float64
    var slop: Float64
    var velocity_iterations: Int
    var position_iterations: Int

    fn __init__(
        out self,
        gravity_x: Float64 = 0.0,
        gravity_y: Float64 = -10.0,
        dt: Float64 = 0.02,
        friction: Float64 = DEFAULT_FRICTION,
        restitution: Float64 = DEFAULT_RESTITUTION,
        baumgarte: Float64 = DEFAULT_BAUMGARTE,
        slop: Float64 = DEFAULT_SLOP,
        velocity_iterations: Int = DEFAULT_VELOCITY_ITERATIONS,
        position_iterations: Int = DEFAULT_POSITION_ITERATIONS,
    ):
        self.gravity_x = gravity_x
        self.gravity_y = gravity_y
        self.dt = dt
        self.friction = friction
        self.restitution = restitution
        self.baumgarte = baumgarte
        self.slop = slop
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations

    fn __copyinit__(out self, existing: Self):
        self.gravity_x = existing.gravity_x
        self.gravity_y = existing.gravity_y
        self.dt = existing.dt
        self.friction = existing.friction
        self.restitution = existing.restitution
        self.baumgarte = existing.baumgarte
        self.slop = existing.slop
        self.velocity_iterations = existing.velocity_iterations
        self.position_iterations = existing.position_iterations

    fn __moveinit__(out self, deinit existing: Self):
        self.gravity_x = existing.gravity_x
        self.gravity_y = existing.gravity_y
        self.dt = existing.dt
        self.friction = existing.friction
        self.restitution = existing.restitution
        self.baumgarte = existing.baumgarte
        self.slop = existing.slop
        self.velocity_iterations = existing.velocity_iterations
        self.position_iterations = existing.position_iterations


struct CollisionType:
    """Enum for collision detection type."""

    comptime NONE: Int = 0  # No collision detection
    comptime FLAT_TERRAIN: Int = 1  # Simple flat ground at fixed Y
    comptime EDGE_TERRAIN: Int = 2  # Edge-based terrain (stored in state)


struct PhysicsKernelStrided:
    """Unified physics step orchestrator for strided 2D layouts.

    Executes a complete physics simulation step:
    1. Integrate velocities (apply gravity + forces)
    2. Detect collisions
    3. Solve velocity constraints (contacts + joints) - multiple iterations
    4. Integrate positions
    5. Solve position constraints - multiple iterations

    All operations use the strided 2D [BATCH, STATE_SIZE] layout.
    """

    # =========================================================================
    # GPU Physics Step - Edge Terrain
    # =========================================================================

    @staticmethod
    fn step_gpu_edge_terrain[
        BATCH: Int,
        LayoutType: PhysicsLayoutStrided,
        NUM_SHAPES: Int = 3,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        shapes_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
        config: PhysicsConfigStrided,
    ) raises:
        """Execute full physics step on GPU with edge terrain collision.

        This is the main entry point for physics-based environments with
        complex terrain (like LunarLander).

        Args:
            ctx: GPU device context.
            states_buf: State buffer [BATCH, STATE_SIZE].
            shapes_buf: Shape definitions [NUM_SHAPES, SHAPE_MAX_SIZE].
            contacts_buf: Contact workspace [BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE].
            contact_counts_buf: Contact counts [BATCH].
            config: Physics configuration.
        """
        # Compile-time layout constants
        comptime STATE_SIZE = LayoutType.STATE_SIZE
        comptime NUM_BODIES = LayoutType.NUM_BODIES
        comptime MAX_CONTACTS = LayoutType.MAX_CONTACTS
        comptime MAX_JOINTS = LayoutType.MAX_JOINTS
        comptime MAX_TERRAIN_EDGES = LayoutType.MAX_TERRAIN_EDGES
        comptime BODIES_OFFSET = LayoutType.BODIES_OFFSET
        comptime FORCES_OFFSET = LayoutType.FORCES_OFFSET
        comptime JOINTS_OFFSET = LayoutType.JOINTS_OFFSET
        comptime JOINT_COUNT_OFFSET = LayoutType.JOINT_COUNT_OFFSET
        comptime EDGES_OFFSET = LayoutType.EDGES_OFFSET
        comptime EDGE_COUNT_OFFSET = LayoutType.EDGE_COUNT_OFFSET

        # Scalar parameters
        var gravity_x = Scalar[dtype](config.gravity_x)
        var gravity_y = Scalar[dtype](config.gravity_y)
        var dt = Scalar[dtype](config.dt)
        var friction = Scalar[dtype](config.friction)
        var restitution = Scalar[dtype](config.restitution)
        var baumgarte = Scalar[dtype](config.baumgarte)
        var slop = Scalar[dtype](config.slop)

        # 1. Integrate velocities (apply gravity + forces)
        SemiImplicitEuler.integrate_velocities_gpu_strided[
            BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET
        ](ctx, states_buf, gravity_x, gravity_y, dt)

        # 2. Detect collisions against edge terrain
        EdgeTerrainCollision.detect_gpu_strided[
            BATCH,
            NUM_BODIES,
            NUM_SHAPES,
            MAX_CONTACTS,
            MAX_TERRAIN_EDGES,
            STATE_SIZE,
            BODIES_OFFSET,
            EDGES_OFFSET,
            EDGE_COUNT_OFFSET,
        ](ctx, states_buf, shapes_buf, contacts_buf, contact_counts_buf)

        # 3. Solve velocity constraints (contacts + joints)
        # Note: We need to create a joint counts buffer from state
        # For now, we use the existing joint count stored in state
        for _ in range(config.velocity_iterations):
            # Solve contact velocity constraints
            ImpulseSolver.solve_velocity_gpu_strided[
                BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
            ](ctx, states_buf, contacts_buf, contact_counts_buf, friction, restitution)

            # Solve joint velocity constraints
            # Note: Joint counts would need to be extracted from state
            # RevoluteJointSolver.solve_velocity_gpu_strided[...](...)

        # 4. Integrate positions
        SemiImplicitEuler.integrate_positions_gpu_strided[
            BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET
        ](ctx, states_buf, dt)

        # 5. Solve position constraints
        for _ in range(config.position_iterations):
            ImpulseSolver.solve_position_gpu_strided[
                BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
            ](ctx, states_buf, contacts_buf, contact_counts_buf, baumgarte, slop)

        ctx.synchronize()

    # =========================================================================
    # GPU Physics Step - Flat Terrain
    # =========================================================================

    @staticmethod
    fn step_gpu_flat_terrain[
        BATCH: Int,
        LayoutType: PhysicsLayoutStrided,
        NUM_SHAPES: Int = 3,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        shapes_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
        config: PhysicsConfigStrided,
        ground_y: Float64,
    ) raises:
        """Execute full physics step on GPU with flat terrain collision.

        Simpler collision detection for environments with flat ground.

        Args:
            ctx: GPU device context.
            states_buf: State buffer [BATCH, STATE_SIZE].
            shapes_buf: Shape definitions [NUM_SHAPES, SHAPE_MAX_SIZE].
            contacts_buf: Contact workspace [BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE].
            contact_counts_buf: Contact counts [BATCH].
            config: Physics configuration.
            ground_y: Y coordinate of flat ground.
        """
        # Compile-time layout constants
        comptime STATE_SIZE = LayoutType.STATE_SIZE
        comptime NUM_BODIES = LayoutType.NUM_BODIES
        comptime MAX_CONTACTS = LayoutType.MAX_CONTACTS
        comptime BODIES_OFFSET = LayoutType.BODIES_OFFSET
        comptime FORCES_OFFSET = LayoutType.FORCES_OFFSET

        # Scalar parameters
        var gravity_x = Scalar[dtype](config.gravity_x)
        var gravity_y = Scalar[dtype](config.gravity_y)
        var dt = Scalar[dtype](config.dt)
        var friction = Scalar[dtype](config.friction)
        var restitution = Scalar[dtype](config.restitution)
        var baumgarte = Scalar[dtype](config.baumgarte)
        var slop = Scalar[dtype](config.slop)
        var ground_y_scalar = Scalar[dtype](ground_y)

        # 1. Integrate velocities
        SemiImplicitEuler.integrate_velocities_gpu_strided[
            BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET
        ](ctx, states_buf, gravity_x, gravity_y, dt)

        # 2. Detect collisions against flat terrain
        FlatTerrainCollision.detect_gpu_strided[
            BATCH, NUM_BODIES, NUM_SHAPES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
        ](ctx, states_buf, shapes_buf, contacts_buf, contact_counts_buf, ground_y_scalar)

        # 3. Solve velocity constraints
        for _ in range(config.velocity_iterations):
            ImpulseSolver.solve_velocity_gpu_strided[
                BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
            ](ctx, states_buf, contacts_buf, contact_counts_buf, friction, restitution)

        # 4. Integrate positions
        SemiImplicitEuler.integrate_positions_gpu_strided[
            BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET
        ](ctx, states_buf, dt)

        # 5. Solve position constraints
        for _ in range(config.position_iterations):
            ImpulseSolver.solve_position_gpu_strided[
                BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
            ](ctx, states_buf, contacts_buf, contact_counts_buf, baumgarte, slop)

        ctx.synchronize()

    # =========================================================================
    # GPU Physics Step - No Collision
    # =========================================================================

    @staticmethod
    fn step_gpu_no_collision[
        BATCH: Int,
        LayoutType: PhysicsLayoutStrided,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        config: PhysicsConfigStrided,
    ) raises:
        """Execute physics step on GPU without collision detection.

        Useful for environments like Acrobot where there's no ground contact.

        Args:
            ctx: GPU device context.
            states_buf: State buffer [BATCH, STATE_SIZE].
            config: Physics configuration.
        """
        # Compile-time layout constants
        comptime STATE_SIZE = LayoutType.STATE_SIZE
        comptime NUM_BODIES = LayoutType.NUM_BODIES
        comptime BODIES_OFFSET = LayoutType.BODIES_OFFSET
        comptime FORCES_OFFSET = LayoutType.FORCES_OFFSET

        # Scalar parameters
        var gravity_x = Scalar[dtype](config.gravity_x)
        var gravity_y = Scalar[dtype](config.gravity_y)
        var dt = Scalar[dtype](config.dt)

        # 1. Integrate velocities
        SemiImplicitEuler.integrate_velocities_gpu_strided[
            BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET
        ](ctx, states_buf, gravity_x, gravity_y, dt)

        # 2. Integrate positions
        SemiImplicitEuler.integrate_positions_gpu_strided[
            BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET
        ](ctx, states_buf, dt)

        ctx.synchronize()

    # =========================================================================
    # GPU Physics Step - With Joint Constraints
    # =========================================================================

    @staticmethod
    fn step_gpu_with_joints[
        BATCH: Int,
        LayoutType: PhysicsLayoutStrided,
        NUM_SHAPES: Int = 3,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        shapes_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
        joint_counts_buf: DeviceBuffer[dtype],
        config: PhysicsConfigStrided,
    ) raises:
        """Execute full physics step on GPU with edge terrain and joint constraints.

        This is the most complete physics step, handling:
        - Gravity and forces
        - Edge terrain collision
        - Contact constraints
        - Revolute joint constraints

        Args:
            ctx: GPU device context.
            states_buf: State buffer [BATCH, STATE_SIZE].
            shapes_buf: Shape definitions [NUM_SHAPES, SHAPE_MAX_SIZE].
            contacts_buf: Contact workspace [BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE].
            contact_counts_buf: Contact counts [BATCH].
            joint_counts_buf: Joint counts per environment [BATCH].
            config: Physics configuration.
        """
        # Compile-time layout constants
        comptime STATE_SIZE = LayoutType.STATE_SIZE
        comptime NUM_BODIES = LayoutType.NUM_BODIES
        comptime MAX_CONTACTS = LayoutType.MAX_CONTACTS
        comptime MAX_JOINTS = LayoutType.MAX_JOINTS
        comptime MAX_TERRAIN_EDGES = LayoutType.MAX_TERRAIN_EDGES
        comptime BODIES_OFFSET = LayoutType.BODIES_OFFSET
        comptime FORCES_OFFSET = LayoutType.FORCES_OFFSET
        comptime JOINTS_OFFSET = LayoutType.JOINTS_OFFSET
        comptime EDGES_OFFSET = LayoutType.EDGES_OFFSET
        comptime EDGE_COUNT_OFFSET = LayoutType.EDGE_COUNT_OFFSET

        # Scalar parameters
        var gravity_x = Scalar[dtype](config.gravity_x)
        var gravity_y = Scalar[dtype](config.gravity_y)
        var dt = Scalar[dtype](config.dt)
        var friction = Scalar[dtype](config.friction)
        var restitution = Scalar[dtype](config.restitution)
        var baumgarte = Scalar[dtype](config.baumgarte)
        var slop = Scalar[dtype](config.slop)

        # 1. Integrate velocities (apply gravity + forces)
        SemiImplicitEuler.integrate_velocities_gpu_strided[
            BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET
        ](ctx, states_buf, gravity_x, gravity_y, dt)

        # 2. Detect collisions against edge terrain
        EdgeTerrainCollision.detect_gpu_strided[
            BATCH,
            NUM_BODIES,
            NUM_SHAPES,
            MAX_CONTACTS,
            MAX_TERRAIN_EDGES,
            STATE_SIZE,
            BODIES_OFFSET,
            EDGES_OFFSET,
            EDGE_COUNT_OFFSET,
        ](ctx, states_buf, shapes_buf, contacts_buf, contact_counts_buf)

        # 3. Solve velocity constraints (contacts + joints)
        for _ in range(config.velocity_iterations):
            # Solve contact velocity constraints
            ImpulseSolver.solve_velocity_gpu_strided[
                BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
            ](ctx, states_buf, contacts_buf, contact_counts_buf, friction, restitution)

            # Solve joint velocity constraints
            RevoluteJointSolver.solve_velocity_gpu_strided[
                BATCH, NUM_BODIES, MAX_JOINTS, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
            ](ctx, states_buf, joint_counts_buf, dt)

        # 4. Integrate positions
        SemiImplicitEuler.integrate_positions_gpu_strided[
            BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET
        ](ctx, states_buf, dt)

        # 5. Solve position constraints (contacts + joints)
        for _ in range(config.position_iterations):
            ImpulseSolver.solve_position_gpu_strided[
                BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
            ](ctx, states_buf, contacts_buf, contact_counts_buf, baumgarte, slop)

            RevoluteJointSolver.solve_position_gpu_strided[
                BATCH, NUM_BODIES, MAX_JOINTS, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
            ](ctx, states_buf, joint_counts_buf, baumgarte, slop)

        ctx.synchronize()

    # =========================================================================
    # CPU Physics Step (for testing/debugging)
    # =========================================================================

    @staticmethod
    fn step_cpu[
        BATCH: Int,
        LayoutType: PhysicsLayoutStrided,
    ](
        mut states: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, LayoutType.STATE_SIZE),
            MutAnyOrigin,
        ],
        shapes: LayoutTensor[
            dtype,
            Layout.row_major(LayoutType.NUM_SHAPES, SHAPE_MAX_SIZE),
            MutAnyOrigin,
        ],
        mut contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, LayoutType.MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        mut contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
        config: PhysicsConfigStrided,
        ground_y: Float64,
    ):
        """Execute physics step on CPU for testing.

        Args:
            states: State tensor [BATCH, STATE_SIZE].
            shapes: Shape tensor [NUM_SHAPES, SHAPE_MAX_SIZE].
            contacts: Contact workspace [BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE].
            contact_counts: Contact counts [BATCH].
            config: Physics configuration.
            ground_y: Y coordinate of flat ground.
        """
        # CPU implementation for testing
        # Note: This is a simplified version without full collision detection
        # Use for unit testing the layout and basic physics

        var gravity_x = Scalar[dtype](config.gravity_x)
        var gravity_y = Scalar[dtype](config.gravity_y)
        var dt = Scalar[dtype](config.dt)

        # For each environment
        for env in range(BATCH):
            # For each body
            @parameter
            for body in range(LayoutType.NUM_BODIES):
                var off = LayoutType.body_offset(body)
                var force_off = LayoutType.force_offset(body)

                # Get mass properties
                var inv_mass = rebind[Scalar[dtype]](states[env, off + 6])  # IDX_INV_MASS

                if inv_mass > Scalar[dtype](0):
                    # Get forces
                    var fx = rebind[Scalar[dtype]](states[env, force_off + 0])
                    var fy = rebind[Scalar[dtype]](states[env, force_off + 1])

                    # Apply gravity and forces to velocity
                    var vx = rebind[Scalar[dtype]](states[env, off + 3])  # IDX_VX
                    var vy = rebind[Scalar[dtype]](states[env, off + 4])  # IDX_VY

                    vx = vx + (gravity_x + fx * inv_mass) * dt
                    vy = vy + (gravity_y + fy * inv_mass) * dt

                    states[env, off + 3] = vx
                    states[env, off + 4] = vy

                    # Integrate position
                    var x = rebind[Scalar[dtype]](states[env, off + 0])  # IDX_X
                    var y = rebind[Scalar[dtype]](states[env, off + 1])  # IDX_Y

                    x = x + vx * dt
                    y = y + vy * dt

                    states[env, off + 0] = x
                    states[env, off + 1] = y
