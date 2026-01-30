"""Physics3D v2 multi-body simulation step (MuJoCo-style constraint solver).

This version uses a constraint-based approach following MuJoCo's philosophy:

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

from .types import MultiBodyModel, MultiBodyData
from .multi_body_collision import detect_all_contacts
from .constraint_solver import solve_constraints_pgs, correct_positions


fn step_multi_body_v2[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](
    model: MultiBodyModel[DTYPE, NUM_BODIES, MAX_CONTACTS],
    mut data: MultiBodyData[DTYPE, NUM_BODIES, MAX_CONTACTS],
):
    """Complete physics step using constraint-based solver.

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
    detect_all_contacts(model, data)

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
    detect_all_contacts(model, data)

    # 6. Position correction for residual penetration
    for _ in range(10):
        correct_positions(
            model, data,
            baumgarte=Scalar[DTYPE](0.9),
            slop=Scalar[DTYPE](0.0001)
        )
        detect_all_contacts(model, data)


fn simulate_multi_body_v2[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](
    model: MultiBodyModel[DTYPE, NUM_BODIES, MAX_CONTACTS],
    mut data: MultiBodyData[DTYPE, NUM_BODIES, MAX_CONTACTS],
    num_steps: Int,
):
    """Run multi-body simulation for multiple steps.

    Args:
        model: Static model configuration.
        data: Mutable state (will be modified).
        num_steps: Number of simulation steps to run.
    """
    for _ in range(num_steps):
        step_multi_body_v2(model, data)
