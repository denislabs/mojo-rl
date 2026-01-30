"""Physics3D v2 - Minimal physics engine rebuild.

A MuJoCo-inspired physics engine with Model/Data separation.
- Model: Static simulation configuration
- Data: Mutable simulation state

Phase 1: Single free-falling body
Phase 2: Ground contact (sphere-plane)

Example usage:
    from physics3d_v2 import Model, Data, Body, Geom, step

    # Create a 1kg sphere
    var body = Body.create_sphere(mass=1.0)
    var geom = Geom.sphere(radius=0.1)
    var model = Model.create(body, geom, timestep=0.01)

    # Initialize state at height 10m
    var data = Data()
    data.set_position(0, 0, 10)

    # Simulate
    for i in range(100):
        step(model, data)
        print("z =", data.get_z())
"""

from .constants import dtype, TILE, TPB, DEFAULT_GRAVITY_Z, DEFAULT_TIMESTEP
from .constants import GEOM_PLANE, GEOM_SPHERE
from .types import Body, Geom, Contact, Model, Data
from .kinematics import update_kinematics
from .dynamics import compute_acceleration
from .integrator import integrate
from .collision import detect_sphere_plane
from .solver import solve_contact

# Note: render module is imported separately to avoid SDL2 dependency
# for non-rendering use cases:
#   from physics3d_v2.render import Physics3DRenderer


fn step(model: Model, mut data: Data):
    """One complete simulation step (Phase 2 version with collision).

    Pipeline following MuJoCo-style sequential constraint solving:
    1. Update world-frame positions from qpos (kinematics)
    2. Collision detection (pre-step)
    3. Solve contact constraints (apply velocity corrections)
    4. Compute accelerations (gravity + applied forces, but zero out if in contact)
    5. Integrate velocities and positions
    6. Update kinematics again
    7. Collision detection (post-step)
    8. Solve position errors (Baumgarte correction)
    """
    # 1. Update world-frame positions from qpos
    update_kinematics(data)

    # 2. Pre-step collision detection
    detect_sphere_plane(model, data)

    # 3. If in contact, handle resting case properly
    if data.contact.active:
        # Solve velocity constraint (impulse-based)
        solve_contact(model, data)

    # 4. Compute accelerations (gravity + applied forces)
    # If in contact and not moving away, cancel out gravity
    compute_acceleration(model, data)

    # If in contact with ground and not already moving up, clamp downward acceleration
    if data.contact.active and data.qvel[2] <= Scalar[dtype](0):
        # Cancel gravity when resting on ground
        if data.qacc[2] < Scalar[dtype](0):
            data.qacc[2] = Scalar[dtype](0)

    # 5. Integrate velocities and positions
    integrate(model, data)

    # 6. Post-integration kinematics
    update_kinematics(data)

    # 7. Post-step collision detection
    detect_sphere_plane(model, data)

    # 8. Position correction if penetrating
    if data.contact.active:
        solve_contact(model, data)


fn step_no_collision(model: Model, mut data: Data):
    """One simulation step without collision (Phase 1 version).

    Pipeline:
    1. Update world-frame positions from qpos (kinematics)
    2. Compute accelerations (gravity + applied forces)
    3. Integrate velocities and positions

    Use this for free-fall tests or when collision is not needed.
    """
    # 1. Update world-frame positions from qpos
    update_kinematics(data)

    # 2. Compute accelerations (gravity + applied forces)
    compute_acceleration(model, data)

    # 3. Integrate velocities and positions
    integrate(model, data)


fn simulate(model: Model, mut data: Data, num_steps: Int):
    """Run simulation for multiple steps.

    Args:
        model: Static model configuration
        data: Mutable state (will be modified)
        num_steps: Number of simulation steps to run
    """
    for _ in range(num_steps):
        step(model, data)


fn simulate_no_collision(model: Model, mut data: Data, num_steps: Int):
    """Run simulation for multiple steps without collision.

    Args:
        model: Static model configuration
        data: Mutable state (will be modified)
        num_steps: Number of simulation steps to run
    """
    for _ in range(num_steps):
        step_no_collision(model, data)
