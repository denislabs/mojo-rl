"""Integrator trait for physics simulation pipelines.

Integrators advance the physics simulation forward in time,
orchestrating collision detection, constraint solving, and position updates.
"""

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
    """

    @staticmethod
    fn step[
        DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
    ](
        model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS],
        mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS],
    ):
        """Perform one physics simulation step.

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
        """Run simulation for multiple steps.

        Args:
            model: Static model configuration.
            data: Mutable simulation state (will be modified).
            num_steps: Number of simulation steps to run.
        """
        ...

    ...
