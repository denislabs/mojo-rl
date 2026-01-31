"""Impulse-based physics integrator (Bullet/Box2D style).

Uses Split Impulse approach for stable stacking:
- Velocity constraints handle collision response
- Position constraints directly correct penetration
- Resting contact detection prevents drift

Reference: Erin Catto's GDC presentations on constraint solving.
"""

from ..types import Model, Data
from ..traits import Integrator
from ..collision import CollisionDetector
from ..solver import (
    solve_velocity_constraints,
    solve_position_constraints,
    solve_resting_contacts,
)


struct ImpulseIntegrator(Integrator):
    """Impulse-based physics integrator.

    Uses Split Impulse method (similar to Bullet Physics / Box2D):
    - Velocity solver: Only handles velocity constraints (stopping/bouncing)
    - Position solver: Uses pseudo-velocities that don't affect real velocities

    This separation prevents position correction from adding energy to the system,
    which is critical for stable stacking.
    """

    @staticmethod
    fn step[
        DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
    ](
        model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS],
        mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS],
    ):
        """Perform one physics simulation step.

        Pipeline:
        1. Collision detection (pre-step)
        2. Apply gravity to velocities
        3. Solve velocity constraints (collision response)
        4. Handle resting contacts (gravity cancellation)
        5. Integrate positions
        6. Collision detection (post-step)
        7. Solve position constraints (penetration correction)

        Args:
            model: Static model configuration.
            data: Mutable simulation state.
        """
        var dt = model.timestep

        # 1. Collision detection (pre-step)
        CollisionDetector.detect_all_contacts(model, data)

        # 2. Apply gravity to velocities
        for i in range(NUM_BODIES):
            data.velocities[i * 3 + 2] += dt * model.gravity_z

        # 3. Solve velocity constraints (collision response with restitution)
        solve_velocity_constraints(model, data, iterations=30)

        # 4. Handle resting contacts
        # For bodies at rest on support, clamp small downward velocities
        solve_resting_contacts(model, data)

        # 5. Integrate positions
        for i in range(NUM_BODIES * 3):
            data.positions[i] += dt * data.velocities[i]

        # 6. Collision detection (post-step)
        CollisionDetector.detect_all_contacts(model, data)

        # 7. Solve position constraints (direct penetration correction)
        # Use aggressive correction with many iterations for stable stacking
        for _ in range(15):
            solve_position_constraints(
                model,
                data,
                baumgarte=Scalar[DTYPE](1.0),
                slop=Scalar[DTYPE](0.00001),
            )
            # Re-detect to get updated penetration depths
            CollisionDetector.detect_all_contacts(model, data)

        # Final resting contact handling after position correction
        solve_resting_contacts(model, data)

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
        for _ in range(num_steps):
            Self.step(model, data)
