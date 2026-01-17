"""PhysicsKernel - Stateless physics computation orchestrator.

This module provides stateless static methods for physics simulation,
similar to how deep_rl Model trait provides forward/backward methods.

The kernel composes:
- Integrator: Velocity and position updates
- CollisionSystem: Contact detection
- ConstraintSolver: Impulse-based constraint resolution

All state is passed via LayoutTensor views or DeviceBuffers - no heap
allocation in kernel code, making it suitable for GPU execution.

Example:
    ```mojo
    from physics_gpu.kernel import PhysicsKernel, PhysicsConfig

    # CPU step - pass LayoutTensor views
    PhysicsKernel.step[BATCH, NUM_BODIES, NUM_SHAPES, MAX_CONTACTS](
        bodies, shapes, forces, contacts, counts, config
    )

    # GPU step - pass DeviceBuffers
    PhysicsKernel.step_gpu[BATCH, NUM_BODIES, NUM_SHAPES, MAX_CONTACTS](
        ctx, bodies_buf, shapes_buf, forces_buf, contacts_buf, counts_buf, config
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
    IDX_FX,
    IDX_FY,
    IDX_TAU,
    DEFAULT_GRAVITY_X,
    DEFAULT_GRAVITY_Y,
    DEFAULT_DT,
    DEFAULT_FRICTION,
    DEFAULT_RESTITUTION,
    DEFAULT_BAUMGARTE,
    DEFAULT_SLOP,
)
from .layout import PhysicsLayout
from .integrators.euler import SemiImplicitEuler
from .collision.flat_terrain import FlatTerrainCollision
from .solvers.impulse import ImpulseSolver


@fieldwise_init
struct PhysicsConfig(Copyable, Movable):
    """Runtime configuration for physics simulation.

    This struct holds physics parameters that can vary at runtime.
    Passed to kernel methods rather than stored as state.
    """

    var gravity_x: Scalar[dtype]
    var gravity_y: Scalar[dtype]
    var dt: Scalar[dtype]
    var ground_y: Scalar[dtype]
    var friction: Scalar[dtype]
    var restitution: Scalar[dtype]
    var baumgarte: Scalar[dtype]
    var slop: Scalar[dtype]
    var velocity_iterations: Int
    var position_iterations: Int

    fn __init__(
        out self,
        gravity_x: Float64 = DEFAULT_GRAVITY_X,
        gravity_y: Float64 = DEFAULT_GRAVITY_Y,
        dt: Float64 = DEFAULT_DT,
        ground_y: Float64 = 0.0,
        friction: Float64 = DEFAULT_FRICTION,
        restitution: Float64 = DEFAULT_RESTITUTION,
        baumgarte: Float64 = DEFAULT_BAUMGARTE,
        slop: Float64 = DEFAULT_SLOP,
        velocity_iterations: Int = 6,
        position_iterations: Int = 2,
    ):
        """Initialize physics configuration with defaults."""
        self.gravity_x = Scalar[dtype](gravity_x)
        self.gravity_y = Scalar[dtype](gravity_y)
        self.dt = Scalar[dtype](dt)
        self.ground_y = Scalar[dtype](ground_y)
        self.friction = Scalar[dtype](friction)
        self.restitution = Scalar[dtype](restitution)
        self.baumgarte = Scalar[dtype](baumgarte)
        self.slop = Scalar[dtype](slop)
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations


struct PhysicsKernel:
    """Stateless physics computation orchestrator.

    This struct provides static methods for physics simulation.
    It has no instance state - all state is passed via parameters.
    This design enables both CPU and GPU execution with identical logic.

    The physics step follows Box2D's simulation order:
    1. Integrate velocities (apply forces and gravity)
    2. Detect collisions (generate contacts)
    3. Solve velocity constraints (apply impulses)
    4. Integrate positions (update using new velocities)
    5. Solve position constraints (resolve penetration)
    6. Clear forces
    """

    # Stateless - use static methods only

    # =========================================================================
    # CPU Physics Step
    # =========================================================================

    @staticmethod
    fn step[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS: Int,
    ](
        mut bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        shapes: LayoutTensor[
            dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
        ],
        mut forces: LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES, 3), MutAnyOrigin
        ],
        mut contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        mut contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
        config: PhysicsConfig,
    ):
        """Execute one physics step on CPU.

        Args:
            bodies: Body state tensor [BATCH, NUM_BODIES, BODY_STATE_SIZE].
            shapes: Shape definitions [NUM_SHAPES, SHAPE_MAX_SIZE].
            forces: Accumulated forces [BATCH, NUM_BODIES, 3].
            contacts: Contact manifold [BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE].
            contact_counts: Active contacts per env [BATCH].
            config: Physics configuration.
        """
        # Create temporary component instances (stateless, just for method dispatch)
        var integrator = SemiImplicitEuler()
        var collision = FlatTerrainCollision(Float64(config.ground_y))
        var solver = ImpulseSolver(
            Float64(config.friction), Float64(config.restitution)
        )

        # 1. Integrate velocities
        integrator.integrate_velocities[BATCH, NUM_BODIES](
            bodies, forces, config.gravity_x, config.gravity_y, config.dt
        )

        # 2. Detect collisions
        collision.detect[BATCH, NUM_BODIES, NUM_SHAPES, MAX_CONTACTS](
            bodies, shapes, contacts, contact_counts
        )

        # 3. Solve velocity constraints
        for _ in range(config.velocity_iterations):
            solver.solve_velocity[BATCH, NUM_BODIES, MAX_CONTACTS](
                bodies, contacts, contact_counts
            )

        # 4. Integrate positions
        integrator.integrate_positions[BATCH, NUM_BODIES](bodies, config.dt)

        # 5. Solve position constraints
        for _ in range(config.position_iterations):
            solver.solve_position[BATCH, NUM_BODIES, MAX_CONTACTS](
                bodies, contacts, contact_counts
            )

        # 6. Clear forces
        PhysicsKernel._clear_forces[BATCH, NUM_BODIES](forces)

    @staticmethod
    fn _clear_forces[
        BATCH: Int,
        NUM_BODIES: Int,
    ](
        mut forces: LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES, 3), MutAnyOrigin
        ],
    ):
        """Clear accumulated forces after physics step."""
        for env in range(BATCH):
            for body in range(NUM_BODIES):
                forces[env, body, 0] = Scalar[dtype](0)
                forces[env, body, 1] = Scalar[dtype](0)
                forces[env, body, 2] = Scalar[dtype](0)

    # =========================================================================
    # GPU Physics Step
    # =========================================================================

    @staticmethod
    fn step_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS: Int,
    ](
        ctx: DeviceContext,
        mut bodies_buf: DeviceBuffer[dtype],
        shapes_buf: DeviceBuffer[dtype],
        forces_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
        config: PhysicsConfig,
    ) raises:
        """Execute one physics step on GPU.

        This mirrors the CPU step but uses GPU kernels.

        Args:
            ctx: GPU device context.
            bodies_buf: Body state buffer [BATCH * NUM_BODIES * BODY_STATE_SIZE].
            shapes_buf: Shape definitions buffer [NUM_SHAPES * SHAPE_MAX_SIZE].
            forces_buf: Forces buffer [BATCH * NUM_BODIES * 3].
            contacts_buf: Contacts buffer [BATCH * MAX_CONTACTS * CONTACT_DATA_SIZE].
            contact_counts_buf: Contact counts buffer [BATCH].
            config: Physics configuration.
        """
        # 1. Integrate velocities
        SemiImplicitEuler.integrate_velocities_gpu[BATCH, NUM_BODIES](
            ctx,
            bodies_buf,
            forces_buf,
            config.gravity_x,
            config.gravity_y,
            config.dt,
        )

        # 2. Detect collisions
        FlatTerrainCollision.detect_gpu[
            BATCH, NUM_BODIES, NUM_SHAPES, MAX_CONTACTS
        ](
            ctx,
            bodies_buf,
            shapes_buf,
            contacts_buf,
            contact_counts_buf,
            config.ground_y,
        )

        # 3. Solve velocity constraints
        for _ in range(config.velocity_iterations):
            ImpulseSolver.solve_velocity_gpu[BATCH, NUM_BODIES, MAX_CONTACTS](
                ctx,
                bodies_buf,
                contacts_buf,
                contact_counts_buf,
                config.friction,
                config.restitution,
            )

        # 4. Integrate positions
        SemiImplicitEuler.integrate_positions_gpu[BATCH, NUM_BODIES](
            ctx, bodies_buf, config.dt
        )

        # 5. Solve position constraints
        for _ in range(config.position_iterations):
            ImpulseSolver.solve_position_gpu[BATCH, NUM_BODIES, MAX_CONTACTS](
                ctx,
                bodies_buf,
                contacts_buf,
                contact_counts_buf,
                config.baumgarte,
                config.slop,
            )

        # Note: Force clearing handled by PhysicsState after copy back

    # =========================================================================
    # Individual Operations (for custom simulation loops)
    # =========================================================================

    @staticmethod
    fn integrate_velocities[
        BATCH: Int,
        NUM_BODIES: Int,
    ](
        mut bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        forces: LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES, 3), MutAnyOrigin
        ],
        config: PhysicsConfig,
    ):
        """CPU: Integrate velocities only."""
        var integrator = SemiImplicitEuler()
        integrator.integrate_velocities[BATCH, NUM_BODIES](
            bodies, forces, config.gravity_x, config.gravity_y, config.dt
        )

    @staticmethod
    fn integrate_positions[
        BATCH: Int,
        NUM_BODIES: Int,
    ](
        mut bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        config: PhysicsConfig,
    ):
        """CPU: Integrate positions only."""
        var integrator = SemiImplicitEuler()
        integrator.integrate_positions[BATCH, NUM_BODIES](bodies, config.dt)

    @staticmethod
    fn detect_collisions[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS: Int,
    ](
        bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        shapes: LayoutTensor[
            dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
        ],
        mut contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        mut contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
        config: PhysicsConfig,
    ):
        """CPU: Detect collisions only."""
        var collision = FlatTerrainCollision(Float64(config.ground_y))
        collision.detect[BATCH, NUM_BODIES, NUM_SHAPES, MAX_CONTACTS](
            bodies, shapes, contacts, contact_counts
        )

    @staticmethod
    fn solve_velocity_constraints[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
    ](
        mut bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
        config: PhysicsConfig,
    ):
        """CPU: Solve velocity constraints only."""
        var solver = ImpulseSolver(
            Float64(config.friction), Float64(config.restitution)
        )
        for _ in range(config.velocity_iterations):
            solver.solve_velocity[BATCH, NUM_BODIES, MAX_CONTACTS](
                bodies, contacts, contact_counts
            )

    @staticmethod
    fn solve_position_constraints[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
    ](
        mut bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
        config: PhysicsConfig,
    ):
        """CPU: Solve position constraints only."""
        var solver = ImpulseSolver(
            Float64(config.friction), Float64(config.restitution)
        )
        for _ in range(config.position_iterations):
            solver.solve_position[BATCH, NUM_BODIES, MAX_CONTACTS](
                bodies, contacts, contact_counts
            )
