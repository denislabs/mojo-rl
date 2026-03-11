"""Integrator trait for physics simulation pipelines.

Integrators advance the physics simulation forward in time,
orchestrating collision detection, constraint solving, and position updates.

Both CPU and GPU execution paths are supported through the trait interface.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from ..types import Model, Data, ConeType


trait Integrator(Movable & ImplicitlyCopyable):
    """Trait for Generalized Coordinates integrators.

    Integrators implement a complete physics step pipeline for joint-space dynamics:
    1. Forward kinematics (qpos -> xpos, xquat)
    2. Collision detection
    3. Compute dynamics (mass matrix, bias forces)
    4. Solve contact forces in joint space
    5. Integrate (qvel, qpos)

    Both CPU (step/simulate) and GPU (step_gpu/simulate_gpu) methods are provided.
    """

    # =========================================================================
    # CPU Methods
    # =========================================================================

    @staticmethod
    fn step[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
        NM: Int = 0,
        SPARSE: Bool = False,
    ](
        model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
        verbose: Bool = False,
    ):
        """Perform one physics simulation step on CPU.

        Args:
            model: Static model configuration.
            data: Mutable simulation state (will be modified).
            verbose: Whether to print debug information.
        """
        ...

    @staticmethod
    fn simulate[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
        NM: Int = 0,
        SPARSE: Bool = False,
    ](
        model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
        num_steps: Int,
    ):
        """Run simulation for multiple steps on CPU.

        Args:
            model: Static model configuration.
            data: Mutable simulation state (will be modified).
            num_steps: Number of simulation steps to run.
        """
        ...

    # =========================================================================
    # GPU Methods
    # =========================================================================

    @staticmethod
    fn step_gpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        BATCH: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
        NM: Int = 0,
        SPARSE: Bool = False,
        STEP_THREADS: Int = 1,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        mut workspace_buf: DeviceBuffer[DTYPE],
    ) raises:
        """Perform one physics simulation step on GPU.

        Runs physics for all BATCH environments in parallel.
        When STEP_THREADS > 1, uses a multi-threaded step kernel.

        Args:
            ctx: GPU device context.
            state_buf: Device buffer containing joint-space state for all environments.
                Layout: [BATCH, STATE_SIZE] where STATE_SIZE = NQ + NV + NV + NBODY*7.
            model_buf: Device buffer containing per-body model data.
            workspace_buf: Device buffer for solver workspace (M_inv, J_n, A).
                Layout: [BATCH, WS_SIZE].
        """
        ...

    @staticmethod
    fn simulate_gpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        BATCH: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
        NM: Int = 0,
        SPARSE: Bool = False,
        STEP_THREADS: Int = 1,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        mut workspace_buf: DeviceBuffer[DTYPE],
        num_steps: Int,
    ) raises:
        """Run simulation for multiple steps on GPU.

        Args:
            ctx: GPU device context.
            state_buf: Device buffer containing joint-space state for all environments.
            model_buf: Device buffer containing per-body model data.
            workspace_buf: Device buffer for solver workspace (M_inv, J_n, A).
            num_steps: Number of simulation steps to run.
        """
        ...
