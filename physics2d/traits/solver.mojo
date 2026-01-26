"""ConstraintSolver trait for physics simulation.

ConstraintSolvers resolve contact constraints by applying impulses
to prevent interpenetration and enforce friction.
"""

from ..constants import dtype, BODY_STATE_SIZE, CONTACT_DATA_SIZE
from layout import LayoutTensor, Layout
from gpu.host import DeviceContext, DeviceBuffer


trait ConstraintSolver(Movable & ImplicitlyCopyable):
    """Trait for constraint solving algorithms.

    Constraint solvers resolve contacts by:
    1. Solving velocity constraints (apply impulses to correct velocities)
    2. Solving position constraints (push bodies apart to resolve penetration)

    Different implementations can use different algorithms:
    - ImpulseSolver: Simple sequential impulse method
    - ParallelImpulseSolver: Parallel constraint solving (for many contacts)

    Compile-time constants:
    - VELOCITY_ITERATIONS: Number of velocity solver iterations
    - POSITION_ITERATIONS: Number of position solver iterations

    Layout:
    - bodies: [BATCH, NUM_BODIES, BODY_STATE_SIZE]
    - contacts: [BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE]
    - contact_counts: [BATCH]
    """

    comptime VELOCITY_ITERATIONS: Int
    comptime POSITION_ITERATIONS: Int

    # =========================================================================
    # CPU Methods
    # =========================================================================

    fn solve_velocity[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
    ](
        self,
        mut bodies: LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE), MutAnyOrigin
        ],
        mut contacts: LayoutTensor[
            dtype, Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE), MutAnyOrigin
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
    ):
        """Solve velocity constraints for one iteration.

        Applies impulses to correct relative velocities at contact points.
        Should be called VELOCITY_ITERATIONS times per physics step.

        Args:
            bodies: Body state tensor [BATCH, NUM_BODIES, BODY_STATE_SIZE].
            contacts: Contact tensor [BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE].
            contact_counts: Number of contacts per env [BATCH].
        """
        ...

    fn solve_position[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
    ](
        self,
        mut bodies: LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE), MutAnyOrigin
        ],
        contacts: LayoutTensor[
            dtype, Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE), MutAnyOrigin
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
    ):
        """Solve position constraints for one iteration.

        Pushes bodies apart to resolve interpenetration.
        Should be called POSITION_ITERATIONS times per physics step.

        Args:
            bodies: Body state tensor [BATCH, NUM_BODIES, BODY_STATE_SIZE].
            contacts: Contact tensor [BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE].
            contact_counts: Number of contacts per env [BATCH].
        """
        ...

    # =========================================================================
    # GPU Methods
    # =========================================================================
    # Note: GPU methods are not defined in the trait because different
    # solvers may require different parameters (e.g., friction, restitution).
    # Each implementation provides its own solve_velocity_gpu and
    # solve_position_gpu methods. The PhysicsWorld orchestrator calls these
    # methods directly on the concrete implementation.
