"""Integrator trait for physics simulation pipelines.

Integrators advance the physics simulation forward in time,
orchestrating collision detection, constraint solving, and position updates.

Both CPU and GPU execution paths are supported through the trait interface.
"""

from gpu.host import DeviceContext, DeviceBuffer

from ..types import Model, Data


trait Integrator(Movable & ImplicitlyCopyable):
    """Trait for physics integrators.

    Integrators implement a complete physics step pipeline:
    1. Collision detection
    2. Apply forces (gravity)
    3. Solve constraints
    4. Integrate positions
    5. Position correction

    Different integrators use different constraint solvers:
    - ImpulseIntegrator: Split Impulse method (Bullet/Box2D style)
    - PGSIntegrator: Projected Gauss-Seidel (MuJoCo style)

    Both CPU (step/simulate) and GPU (step_gpu/simulate_gpu) methods are provided.
    """

    # =========================================================================
    # CPU Methods
    # =========================================================================

    @staticmethod
    fn step[
        DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
    ](
        model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS],
        mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS],
    ):
        """Perform one physics simulation step on CPU.

        Args:
            model: Static model configuration.
            data: Mutable simulation state (will be modified).
        """
        ...

    @staticmethod
    fn simulate[
        DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
    ](
        model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS],
        mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS],
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
        DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, BATCH: Int
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        dt: Scalar[DTYPE],
        gravity_z: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
        restitution: Scalar[DTYPE],
        friction: Scalar[DTYPE],
    ) raises:
        """Perform one physics simulation step on GPU.

        Runs physics for all BATCH environments in parallel.

        Args:
            ctx: GPU device context.
            state_buf: Device buffer containing state for all environments.
                Layout: [BATCH, STATE_SIZE] where STATE_SIZE depends on
                NUM_BODIES and MAX_CONTACTS.
            model_buf: Device buffer containing per-body model data
                (masses, radii, inertias).
            dt: Timestep.
            gravity_z: Z-component of gravity.
            ground_z: Ground plane height.
            restitution: Coefficient of restitution.
            friction: Friction coefficient.
        """
        ...

    @staticmethod
    fn simulate_gpu[
        DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, BATCH: Int
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        num_steps: Int,
        dt: Scalar[DTYPE],
        gravity_z: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
        restitution: Scalar[DTYPE],
        friction: Scalar[DTYPE],
    ) raises:
        """Run simulation for multiple steps on GPU.

        Args:
            ctx: GPU device context.
            state_buf: Device buffer containing state for all environments.
            model_buf: Device buffer containing per-body model data.
            num_steps: Number of simulation steps to run.
            dt: Timestep.
            gravity_z: Z-component of gravity.
            ground_z: Ground plane height.
            restitution: Coefficient of restitution.
            friction: Friction coefficient.
        """
        ...

    ...
