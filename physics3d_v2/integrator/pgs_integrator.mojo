"""Projected Gauss-Seidel physics integrator (MuJoCo-style).

Uses a constraint-based approach following MuJoCo's philosophy:
1. Collision detection → Generate contact constraints
2. Apply gravity (external forces)
3. Solve constraints using PGS (Projected Gauss-Seidel)
4. Integrate positions
5. Position correction for residual penetration

The key insight from MuJoCo is that contacts are constraints, not impulses.
Constraint forces are computed to satisfy:
- Normal force >= 0 (no pulling)
- Penetration resolved via spring-damper reference acceleration

Reference: MuJoCo Warp solver.py
"""

from ..types import Model, Data
from ..traits import Integrator
from ..collision import CollisionDetector
from ..solver import solve_constraints_pgs, correct_positions


struct PGSIntegrator(Integrator):
    """Projected Gauss-Seidel physics integrator.

    Uses MuJoCo-style constraint formulation:
    - Contacts are soft constraints with spring-damper dynamics
    - solref = [timeconst, dampratio] controls stiffness
    - Reference acceleration: aref = -k * pos - b * vel
    - D (effective mass) includes impedance regularization
    - Constraint forces are clamped to >= 0 (unilateral)
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
        1. Collision detection
        2. Apply gravity to velocities
        3. Solve contact constraints (PGS solver)
        4. Integrate positions
        5. Collision detection (post-step)
        6. Position correction (Baumgarte)

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

        # 3. Solve contact constraints using PGS
        # This modifies velocities to satisfy contact constraints
        solve_constraints_pgs(model, data, dt, iterations=30)

        # 4. Integrate positions
        for i in range(NUM_BODIES * 3):
            data.positions[i] += dt * data.velocities[i]

        # 5. Collision detection (post-step)
        CollisionDetector.detect_all_contacts(model, data)

        # 6. Position correction for residual penetration
        for _ in range(10):
            correct_positions(
                model,
                data,
                baumgarte=Scalar[DTYPE](0.9),
                slop=Scalar[DTYPE](0.0001),
            )
            CollisionDetector.detect_all_contacts(model, data)

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
