"""Physics3D v2 multi-body simulation step.

Phase 3: Complete physics step for multi-body systems.
Orchestrates collision detection, dynamics, integration, and constraint solving.

Uses Split Impulse approach for stable stacking:
- Velocity constraints handle collision response
- Position constraints directly correct penetration
- Resting contact detection prevents drift
"""

from .types import MultiBodyModel, MultiBodyData
from .multi_body_collision import detect_all_contacts
from .multi_body_solver import (
    solve_velocity_constraints,
    solve_position_constraints,
    solve_resting_contacts,
)


fn step_multi_body[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](
    model: MultiBodyModel[DTYPE, NUM_BODIES, MAX_CONTACTS],
    mut data: MultiBodyData[DTYPE, NUM_BODIES, MAX_CONTACTS],
):
    """Complete physics step for multi-body system.

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
    detect_all_contacts(model, data)

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
    detect_all_contacts(model, data)

    # 7. Solve position constraints (direct penetration correction)
    # Use aggressive correction with many iterations for stable stacking
    for _ in range(15):
        solve_position_constraints(
            model, data, baumgarte=Scalar[DTYPE](1.0), slop=Scalar[DTYPE](0.00001)
        )
        # Re-detect to get updated penetration depths
        detect_all_contacts(model, data)

    # Final resting contact handling after position correction
    solve_resting_contacts(model, data)


fn simulate_multi_body[
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
        step_multi_body(model, data)
