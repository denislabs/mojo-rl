"""ConstraintSolver3D trait for 3D physics constraint solving.

Defines the interface for constraint solvers that handle
contacts and joints.
"""

from layout import LayoutTensor, Layout
from ..constants import dtype


trait ConstraintSolver3D:
    """Interface for 3D physics constraint solvers.

    Constraint solvers handle both contact constraints (from collision)
    and joint constraints (from articulated bodies).
    """

    @staticmethod
    fn solve_velocity_constraints[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        CONTACTS_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        contact_count: Int,
        joint_count: Int,
        dt: Scalar[dtype],
    ):
        """Solve velocity constraints for contacts and joints.

        Iteratively applies impulses to satisfy velocity constraints:
        - Contact: relative velocity at contact point ≥ 0 (non-penetration)
        - Joint: relative velocity at anchor matches joint motion
        """
        ...

    @staticmethod
    fn solve_position_constraints[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        CONTACTS_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        contact_count: Int,
        joint_count: Int,
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ):
        """Solve position constraints to correct drift.

        Uses Baumgarte stabilization to push bodies out of penetration
        and fix joint position errors.
        """
        ...

    @staticmethod
    fn solve[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        CONTACTS_OFFSET: Int,
        JOINTS_OFFSET: Int,
        VEL_ITERATIONS: Int,
        POS_ITERATIONS: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        contact_count: Int,
        joint_count: Int,
        dt: Scalar[dtype],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ):
        """Full constraint solving with multiple iterations.

        Runs velocity iterations, then position iterations.
        """
        ...
