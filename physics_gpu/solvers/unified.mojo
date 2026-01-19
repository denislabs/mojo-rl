"""Unified constraint solver - combines contact + joint constraints in single kernels.

This module provides optimized constraint solving by:
1. Fusing contact and joint velocity constraints into ONE kernel
2. Fusing contact and joint position constraints into ONE kernel
3. Running all iterations INSIDE the kernel (no Python loop overhead)

Performance benefit: Reduces kernel launches from 16 to 2 for constraint solving.
- Before: 6 vel_iters × 2 kernels + 2 pos_iters × 2 kernels = 16 launches
- After:  1 velocity kernel + 1 position kernel = 2 launches

The design is fully generic via compile-time parameters, working with any:
- Number of environments (BATCH)
- Number of bodies (NUM_BODIES)
- Number of contacts (MAX_CONTACTS)
- Number of joints (MAX_JOINTS)
- State layout (STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET)
- Number of iterations (VEL_ITERATIONS, POS_ITERATIONS)

MODULAR DESIGN: This solver reuses the single-environment methods from:
- ImpulseSolver.solve_velocity_single_env / solve_position_single_env
- RevoluteJointSolver.solve_velocity_single_env / solve_position_single_env

This ensures:
- No code duplication - constraint logic lives in ONE place
- Bug fixes apply everywhere automatically
- Easy to add new constraint types

Example usage:
    ```mojo
    # Instead of:
    for _ in range(6):
        ImpulseSolver.solve_velocity_gpu[...]()
        RevoluteJointSolver.solve_velocity_gpu[...]()

    # Use:
    UnifiedConstraintSolver.solve_velocity_gpu[..., VEL_ITERATIONS=6](...)
    ```
"""

from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer

from ..constants import (
    dtype,
    TPB,
    CONTACT_DATA_SIZE,
)
from .impulse import ImpulseSolver
from ..joints.revolute import RevoluteJointSolver


struct UnifiedConstraintSolver:
    """Fused constraint solver that combines contact + joint solving.

    This solver runs multiple iterations inside a single kernel launch,
    avoiding the overhead of launching separate kernels for each iteration.

    The actual constraint solving logic is delegated to:
    - ImpulseSolver.solve_velocity_single_env / solve_position_single_env
    - RevoluteJointSolver.solve_velocity_single_env / solve_position_single_env
    """

    # =========================================================================
    # Fused Velocity Kernel
    # =========================================================================

    @always_inline
    @staticmethod
    fn _solve_velocity_all_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
        VEL_ITERATIONS: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
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
        joint_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
        friction: Scalar[dtype],
        restitution: Scalar[dtype],
        dt: Scalar[dtype],
    ):
        """GPU kernel that runs ALL velocity iterations in one kernel."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        var contact_count = Int(contact_counts[env])
        var joint_count = Int(joint_counts[env])

        # Run all iterations inside the kernel
        for _ in range(VEL_ITERATIONS):
            # Contact velocity constraints (reuse ImpulseSolver logic)
            ImpulseSolver.solve_velocity_single_env[
                BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
            ](env, state, contacts, contact_count, friction, restitution)

            # Joint velocity constraints (reuse RevoluteJointSolver logic)
            RevoluteJointSolver.solve_velocity_single_env[
                BATCH,
                NUM_BODIES,
                MAX_JOINTS,
                STATE_SIZE,
                BODIES_OFFSET,
                JOINTS_OFFSET,
            ](env, state, joint_count, dt)

    # =========================================================================
    # Fused Position Kernel
    # =========================================================================

    @always_inline
    @staticmethod
    fn _solve_position_all_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
        POS_ITERATIONS: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
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
        joint_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ):
        """GPU kernel that runs ALL position iterations in one kernel."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        var contact_count = Int(contact_counts[env])
        var joint_count = Int(joint_counts[env])

        # Run all iterations inside the kernel
        for _ in range(POS_ITERATIONS):
            # Contact position constraints (reuse ImpulseSolver logic)
            ImpulseSolver.solve_position_single_env[
                BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
            ](env, state, contacts, contact_count, baumgarte, slop)

            # Joint position constraints (reuse RevoluteJointSolver logic)
            RevoluteJointSolver.solve_position_single_env[
                BATCH,
                NUM_BODIES,
                MAX_JOINTS,
                STATE_SIZE,
                BODIES_OFFSET,
                JOINTS_OFFSET,
            ](env, state, joint_count, baumgarte, slop)

    # =========================================================================
    # Public GPU API
    # =========================================================================

    @staticmethod
    fn solve_velocity_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
        VEL_ITERATIONS: Int,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        contact_counts_buf: DeviceBuffer[dtype],
        joint_counts_buf: DeviceBuffer[dtype],
        friction: Scalar[dtype],
        restitution: Scalar[dtype],
        dt: Scalar[dtype],
    ) raises:
        """Solve ALL velocity iterations in ONE kernel launch.

        Args:
            ctx: GPU device context.
            state_buf: State buffer [BATCH, STATE_SIZE].
            contacts_buf: Contact workspace [BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE].
            contact_counts_buf: Contact counts [BATCH].
            joint_counts_buf: Joint counts [BATCH].
            friction: Coulomb friction coefficient.
            restitution: Bounce coefficient.
            dt: Time step (for spring joints).
        """
        var state = LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ](state_buf.unsafe_ptr())
        var contacts = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ](contacts_buf.unsafe_ptr())
        var contact_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](contact_counts_buf.unsafe_ptr())
        var joint_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](joint_counts_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[
                dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            contacts: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
                MutAnyOrigin,
            ],
            contact_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ],
            joint_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ],
            friction: Scalar[dtype],
            restitution: Scalar[dtype],
            dt: Scalar[dtype],
        ):
            UnifiedConstraintSolver._solve_velocity_all_kernel[
                BATCH,
                NUM_BODIES,
                MAX_CONTACTS,
                MAX_JOINTS,
                STATE_SIZE,
                BODIES_OFFSET,
                JOINTS_OFFSET,
                VEL_ITERATIONS,
            ](state, contacts, contact_counts, joint_counts, friction, restitution, dt)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            contacts,
            contact_counts,
            joint_counts,
            friction,
            restitution,
            dt,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn solve_position_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
        POS_ITERATIONS: Int,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        contacts_buf: DeviceBuffer[dtype],
        contact_counts_buf: DeviceBuffer[dtype],
        joint_counts_buf: DeviceBuffer[dtype],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ) raises:
        """Solve ALL position iterations in ONE kernel launch.

        Args:
            ctx: GPU device context.
            state_buf: State buffer [BATCH, STATE_SIZE].
            contacts_buf: Contact workspace [BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE].
            contact_counts_buf: Contact counts [BATCH].
            joint_counts_buf: Joint counts [BATCH].
            baumgarte: Position correction factor.
            slop: Penetration allowance.
        """
        var state = LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ](state_buf.unsafe_ptr())
        var contacts = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ](contacts_buf.unsafe_ptr())
        var contact_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](contact_counts_buf.unsafe_ptr())
        var joint_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](joint_counts_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[
                dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            contacts: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
                MutAnyOrigin,
            ],
            contact_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ],
            joint_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ],
            baumgarte: Scalar[dtype],
            slop: Scalar[dtype],
        ):
            UnifiedConstraintSolver._solve_position_all_kernel[
                BATCH,
                NUM_BODIES,
                MAX_CONTACTS,
                MAX_JOINTS,
                STATE_SIZE,
                BODIES_OFFSET,
                JOINTS_OFFSET,
                POS_ITERATIONS,
            ](state, contacts, contact_counts, joint_counts, baumgarte, slop)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            contacts,
            contact_counts,
            joint_counts,
            baumgarte,
            slop,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
