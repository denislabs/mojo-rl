"""Fused physics step kernel - combines all physics operations in ONE kernel.

This module provides a fully fused physics step that runs:
1. Velocity integration (gravity + forces)
2. Collision detection
3. Velocity constraints (contact + joint, all iterations)
4. Position integration
5. Position constraints (contact + joint, all iterations)

ALL IN ONE KERNEL LAUNCH.

Performance benefit: Reduces kernel launches from 9+ to 1 for the physics core.
- Before: integrate_vel + collision + 6*(vel_const + joint_vel) + integrate_pos + 2*(pos_const + joint_pos)
- After:  1 unified physics step kernel

MODULAR DESIGN: This kernel reuses the single-environment methods from:
- SemiImplicitEuler.integrate_velocities_single_env / integrate_positions_single_env
- EdgeTerrainCollision.detect_single_env
- ImpulseSolver.solve_velocity_single_env / solve_position_single_env
- RevoluteJointSolver.solve_velocity_single_env / solve_position_single_env

Example usage:
    ```mojo
    # Instead of multiple kernel launches:
    SemiImplicitEuler.integrate_velocities_gpu[...]()
    EdgeTerrainCollision.detect_gpu[...]()
    for _ in range(6):
        ImpulseSolver.solve_velocity_gpu[...]()
        RevoluteJointSolver.solve_velocity_gpu[...]()
    SemiImplicitEuler.integrate_positions_gpu[...]()
    for _ in range(2):
        ImpulseSolver.solve_position_gpu[...]()
        RevoluteJointSolver.solve_position_gpu[...]()

    # Use:
    PhysicsStepKernel.step_gpu[..., VEL_ITERATIONS=6, POS_ITERATIONS=2](...)
    ```
"""

from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer

from ..constants import (
    dtype,
    TPB,
    CONTACT_DATA_SIZE,
    SHAPE_MAX_SIZE,
)
from ..integrators.euler import SemiImplicitEuler
from ..collision.edge_terrain import EdgeTerrainCollision
from ..solvers.impulse import ImpulseSolver
from ..joints.revolute import RevoluteJointSolver


struct PhysicsStepKernel:
    """Fused physics step kernel that runs the entire physics simulation.

    This kernel performs:
    1. Velocity integration (forces + gravity → velocities)
    2. Collision detection (bodies vs terrain → contacts)
    3. Velocity constraint solving (all iterations)
    4. Position integration (velocities → positions)
    5. Position constraint solving (all iterations)

    All in a single GPU kernel launch per environment.
    """

    # =========================================================================
    # Fused Physics Step Kernel
    # =========================================================================

    @always_inline
    @staticmethod
    fn _step_kernel[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS: Int,
        MAX_JOINTS: Int,
        MAX_EDGES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        FORCES_OFFSET: Int,
        JOINTS_OFFSET: Int,
        EDGES_OFFSET: Int,
        VEL_ITERATIONS: Int,
        POS_ITERATIONS: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        shapes: LayoutTensor[
            dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
        ],
        edge_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        joint_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
        friction: Scalar[dtype],
        restitution: Scalar[dtype],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ):
        """GPU kernel that runs the ENTIRE physics step in one kernel."""
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        var n_edges = Int(edge_counts[env])
        var n_joints = Int(joint_counts[env])

        # =====================================================================
        # Step 1: Integrate velocities (v' = v + a*dt)
        # =====================================================================
        SemiImplicitEuler.integrate_velocities_single_env[
            BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET
        ](env, state, gravity_x, gravity_y, dt)

        # =====================================================================
        # Step 2: Collision detection
        # =====================================================================
        EdgeTerrainCollision.detect_single_env[
            BATCH,
            NUM_BODIES,
            NUM_SHAPES,
            MAX_CONTACTS,
            MAX_EDGES,
            STATE_SIZE,
            BODIES_OFFSET,
            EDGES_OFFSET,
        ](env, state, shapes, n_edges, contacts, contact_counts)

        # Get contact count after detection
        var n_contacts = Int(contact_counts[env])

        # =====================================================================
        # Step 3: Velocity constraints (all iterations)
        # =====================================================================
        for _ in range(VEL_ITERATIONS):
            # Contact velocity constraints
            ImpulseSolver.solve_velocity_single_env[
                BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
            ](env, state, contacts, n_contacts, friction, restitution)

            # Joint velocity constraints
            RevoluteJointSolver.solve_velocity_single_env[
                BATCH,
                NUM_BODIES,
                MAX_JOINTS,
                STATE_SIZE,
                BODIES_OFFSET,
                JOINTS_OFFSET,
            ](env, state, n_joints, dt)

        # =====================================================================
        # Step 4: Integrate positions (x' = x + v'*dt)
        # =====================================================================
        SemiImplicitEuler.integrate_positions_single_env[
            BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET
        ](env, state, dt)

        # =====================================================================
        # Step 5: Position constraints (all iterations)
        # =====================================================================
        for _ in range(POS_ITERATIONS):
            # Contact position constraints
            ImpulseSolver.solve_position_single_env[
                BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
            ](env, state, contacts, n_contacts, baumgarte, slop)

            # Joint position constraints
            RevoluteJointSolver.solve_position_single_env[
                BATCH,
                NUM_BODIES,
                MAX_JOINTS,
                STATE_SIZE,
                BODIES_OFFSET,
                JOINTS_OFFSET,
            ](env, state, n_joints, baumgarte, slop)

    # =========================================================================
    # Public GPU API
    # =========================================================================

    @staticmethod
    fn step_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS: Int,
        MAX_JOINTS: Int,
        MAX_EDGES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        FORCES_OFFSET: Int,
        JOINTS_OFFSET: Int,
        EDGES_OFFSET: Int,
        VEL_ITERATIONS: Int,
        POS_ITERATIONS: Int,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        shapes_buf: DeviceBuffer[dtype],
        edge_counts_buf: DeviceBuffer[dtype],
        joint_counts_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
        friction: Scalar[dtype],
        restitution: Scalar[dtype],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ) raises:
        """Run the ENTIRE physics step in ONE kernel launch.

        This fuses:
        - Velocity integration
        - Collision detection
        - Velocity constraints (all iterations)
        - Position integration
        - Position constraints (all iterations)

        Args:
            ctx: GPU device context.
            state_buf: State buffer [BATCH, STATE_SIZE].
            shapes_buf: Shape definitions [NUM_SHAPES, SHAPE_MAX_SIZE].
            edge_counts_buf: Terrain edge counts [BATCH].
            joint_counts_buf: Joint counts [BATCH].
            contacts_buf: Contact workspace [BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE].
            contact_counts_buf: Contact counts [BATCH].
            gravity_x: Gravity X component.
            gravity_y: Gravity Y component.
            dt: Time step.
            friction: Coulomb friction coefficient.
            restitution: Bounce coefficient.
            baumgarte: Position correction factor.
            slop: Penetration allowance.
        """
        var state = LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ](state_buf.unsafe_ptr())
        var shapes = LayoutTensor[
            dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
        ](shapes_buf.unsafe_ptr())
        var edge_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](edge_counts_buf.unsafe_ptr())
        var joint_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](joint_counts_buf.unsafe_ptr())
        var contacts = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ](contacts_buf.unsafe_ptr())
        var contact_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](contact_counts_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[
                dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            shapes: LayoutTensor[
                dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
            ],
            edge_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ],
            joint_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ],
            contacts: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
                MutAnyOrigin,
            ],
            contact_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ],
            gravity_x: Scalar[dtype],
            gravity_y: Scalar[dtype],
            dt: Scalar[dtype],
            friction: Scalar[dtype],
            restitution: Scalar[dtype],
            baumgarte: Scalar[dtype],
            slop: Scalar[dtype],
        ):
            PhysicsStepKernel._step_kernel[
                BATCH,
                NUM_BODIES,
                NUM_SHAPES,
                MAX_CONTACTS,
                MAX_JOINTS,
                MAX_EDGES,
                STATE_SIZE,
                BODIES_OFFSET,
                FORCES_OFFSET,
                JOINTS_OFFSET,
                EDGES_OFFSET,
                VEL_ITERATIONS,
                POS_ITERATIONS,
            ](
                state,
                shapes,
                edge_counts,
                joint_counts,
                contacts,
                contact_counts,
                gravity_x,
                gravity_y,
                dt,
                friction,
                restitution,
                baumgarte,
                slop,
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            shapes,
            edge_counts,
            joint_counts,
            contacts,
            contact_counts,
            gravity_x,
            gravity_y,
            dt,
            friction,
            restitution,
            baumgarte,
            slop,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )


struct PhysicsStepKernelParallel:
    """Physics step kernel with parallel collision detection.

    Uses body×edge parallel collision detection instead of sequential.
    Constraint solving uses sparse contacts with validity flags.

    Trade-off: More parallelism in collision detection, but iterates over
    more contact slots (NUM_BODIES * MAX_EDGES * MAX_CONTACTS_PER_BODY_EDGE)
    in constraint solving.
    """

    @always_inline
    @staticmethod
    fn _step_kernel_parallel[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS_PER_BODY_EDGE: Int,
        MAX_JOINTS: Int,
        MAX_EDGES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        FORCES_OFFSET: Int,
        JOINTS_OFFSET: Int,
        EDGES_OFFSET: Int,
        VEL_ITERATIONS: Int,
        POS_ITERATIONS: Int,
    ](
        state: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, STATE_SIZE),
            MutAnyOrigin,
        ],
        shapes: LayoutTensor[
            dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
        ],
        edge_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        joint_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES * MAX_EDGES * MAX_CONTACTS_PER_BODY_EDGE, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        contact_flags: LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES * MAX_EDGES * MAX_CONTACTS_PER_BODY_EDGE), MutAnyOrigin
        ],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
        friction: Scalar[dtype],
        restitution: Scalar[dtype],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ):
        """GPU kernel for physics step with sparse contacts (constraint solving only).

        Note: This kernel handles everything EXCEPT collision detection.
        Collision is done in a separate parallel kernel.
        """
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        var n_joints = Int(joint_counts[env])
        comptime TOTAL_CONTACT_SLOTS = NUM_BODIES * MAX_EDGES * MAX_CONTACTS_PER_BODY_EDGE

        # Step 1: Integrate velocities
        SemiImplicitEuler.integrate_velocities_single_env[
            BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET
        ](env, state, gravity_x, gravity_y, dt)

        # Step 2: Collision detection is done EXTERNALLY in parallel kernel
        # (contacts and contact_flags already populated)

        # Step 3: Velocity constraints with sparse contacts
        for _ in range(VEL_ITERATIONS):
            ImpulseSolver.solve_velocity_single_env_sparse[
                BATCH, NUM_BODIES, TOTAL_CONTACT_SLOTS, STATE_SIZE, BODIES_OFFSET
            ](env, state, contacts, contact_flags, friction, restitution)

            RevoluteJointSolver.solve_velocity_single_env[
                BATCH, NUM_BODIES, MAX_JOINTS, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET,
            ](env, state, n_joints, dt)

        # Step 4: Integrate positions
        SemiImplicitEuler.integrate_positions_single_env[
            BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET
        ](env, state, dt)

        # Step 5: Position constraints with sparse contacts
        for _ in range(POS_ITERATIONS):
            ImpulseSolver.solve_position_single_env_sparse[
                BATCH, NUM_BODIES, TOTAL_CONTACT_SLOTS, STATE_SIZE, BODIES_OFFSET
            ](env, state, contacts, contact_flags, baumgarte, slop)

            RevoluteJointSolver.solve_position_single_env[
                BATCH, NUM_BODIES, MAX_JOINTS, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET,
            ](env, state, n_joints, baumgarte, slop)

    @staticmethod
    fn step_parallel_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS_PER_BODY_EDGE: Int,
        MAX_JOINTS: Int,
        MAX_EDGES: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        FORCES_OFFSET: Int,
        JOINTS_OFFSET: Int,
        EDGES_OFFSET: Int,
        VEL_ITERATIONS: Int,
        POS_ITERATIONS: Int,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
        shapes_buf: DeviceBuffer[dtype],
        edge_counts_buf: DeviceBuffer[dtype],
        joint_counts_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_flags_buf: DeviceBuffer[dtype],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
        friction: Scalar[dtype],
        restitution: Scalar[dtype],
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ) raises:
        """Run physics step with PARALLEL collision detection.

        This is a two-kernel approach:
        1. Parallel collision detection (body×edge, separate launch)
        2. Fused integration + constraints (this kernel)

        Call EdgeTerrainCollision.detect_body_edge_gpu() BEFORE this kernel.

        Args:
            ctx: GPU device context.
            state_buf: State buffer [BATCH, STATE_SIZE].
            shapes_buf: Shape definitions.
            edge_counts_buf: Terrain edge counts [BATCH].
            joint_counts_buf: Joint counts [BATCH].
            contacts_buf: Sparse contact buffer from parallel collision.
            contact_flags_buf: Contact validity flags from parallel collision.
            gravity_x, gravity_y: Gravity components.
            dt: Time step.
            friction, restitution: Contact parameters.
            baumgarte, slop: Position correction parameters.
        """
        comptime TOTAL_CONTACT_SLOTS = NUM_BODIES * MAX_EDGES * MAX_CONTACTS_PER_BODY_EDGE

        var state = LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ](state_buf.unsafe_ptr())
        var shapes = LayoutTensor[
            dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
        ](shapes_buf.unsafe_ptr())
        var edge_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](edge_counts_buf.unsafe_ptr())
        var joint_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](joint_counts_buf.unsafe_ptr())
        var contacts = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, TOTAL_CONTACT_SLOTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ](contacts_buf.unsafe_ptr())
        var contact_flags = LayoutTensor[
            dtype, Layout.row_major(BATCH, TOTAL_CONTACT_SLOTS), MutAnyOrigin
        ](contact_flags_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
            shapes: LayoutTensor[dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin],
            edge_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            joint_counts: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            contacts: LayoutTensor[dtype, Layout.row_major(BATCH, TOTAL_CONTACT_SLOTS, CONTACT_DATA_SIZE), MutAnyOrigin],
            contact_flags: LayoutTensor[dtype, Layout.row_major(BATCH, TOTAL_CONTACT_SLOTS), MutAnyOrigin],
            gravity_x: Scalar[dtype],
            gravity_y: Scalar[dtype],
            dt: Scalar[dtype],
            friction: Scalar[dtype],
            restitution: Scalar[dtype],
            baumgarte: Scalar[dtype],
            slop: Scalar[dtype],
        ):
            PhysicsStepKernelParallel._step_kernel_parallel[
                BATCH, NUM_BODIES, NUM_SHAPES, MAX_CONTACTS_PER_BODY_EDGE,
                MAX_JOINTS, MAX_EDGES, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET,
                JOINTS_OFFSET, EDGES_OFFSET, VEL_ITERATIONS, POS_ITERATIONS,
            ](
                state, shapes, edge_counts, joint_counts, contacts, contact_flags,
                gravity_x, gravity_y, dt, friction, restitution, baumgarte, slop,
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state, shapes, edge_counts, joint_counts, contacts, contact_flags,
            gravity_x, gravity_y, dt, friction, restitution, baumgarte, slop,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )
