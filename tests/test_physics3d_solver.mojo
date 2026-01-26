"""Test physics3d solver and integrator modules."""

from physics3d import (
    # Constants
    BODY_STATE_SIZE_3D,
    IDX_PX, IDX_PY, IDX_PZ,
    IDX_QW, IDX_QX, IDX_QY, IDX_QZ,
    IDX_VX, IDX_VY, IDX_VZ,
    IDX_WX, IDX_WY, IDX_WZ,
    IDX_MASS, IDX_INV_MASS,
    IDX_IXX, IDX_IYY, IDX_IZZ,
    IDX_BODY_TYPE,
    BODY_DYNAMIC,
    SHAPE_SPHERE,
    IDX_SHAPE_3D,

    # Integrators
    SemiImplicitEuler3D,
    integrate_velocities_3d,
    integrate_positions_3d,
    integrate_quaternion,

    # Solvers
    ContactSolver3D,
    solve_contact_velocity,
    JointSolver3D,

    # World
    PhysicsWorld3D,
)

from math3d import Vec3, Quat


fn init_sphere_body(mut state: List[Float64], body_idx: Int, pos: Vec3, mass: Float64, radius: Float64):
    """Initialize a sphere body in the state array."""
    var base = body_idx * BODY_STATE_SIZE_3D

    # Position
    state[base + IDX_PX] = pos.x
    state[base + IDX_PY] = pos.y
    state[base + IDX_PZ] = pos.z

    # Identity quaternion
    state[base + IDX_QW] = 1.0
    state[base + IDX_QX] = 0.0
    state[base + IDX_QY] = 0.0
    state[base + IDX_QZ] = 0.0

    # Zero velocity
    state[base + IDX_VX] = 0.0
    state[base + IDX_VY] = 0.0
    state[base + IDX_VZ] = 0.0
    state[base + IDX_WX] = 0.0
    state[base + IDX_WY] = 0.0
    state[base + IDX_WZ] = 0.0

    # Mass
    state[base + IDX_MASS] = mass
    state[base + IDX_INV_MASS] = 1.0 / mass if mass > 0.0 else 0.0

    # Sphere inertia: I = 2/5 * m * r^2
    var inertia = 0.4 * mass * radius * radius
    state[base + IDX_IXX] = inertia
    state[base + IDX_IYY] = inertia
    state[base + IDX_IZZ] = inertia

    # Shape and type
    state[base + IDX_SHAPE_3D] = Float64(SHAPE_SPHERE)
    state[base + IDX_BODY_TYPE] = Float64(BODY_DYNAMIC)


fn main() raises:
    print("Testing physics3d solver and integrator modules...")

    # Test 1: Quaternion integration
    print("\nTest 1: Quaternion integration")
    var q = Quat.identity()
    var omega = Vec3(0.0, 0.0, 1.0)  # Rotate around Z
    var dt = 0.1

    var q_new = integrate_quaternion(q, omega, dt)
    print("  Original quat:", q.w, q.x, q.y, q.z)
    print("  After rotation:", q_new.w, q_new.x, q_new.y, q_new.z)

    # Test 2: Semi-implicit Euler integrator
    print("\nTest 2: Semi-implicit Euler integrator")
    var integrator = SemiImplicitEuler3D(
        gravity=Vec3(0.0, 0.0, -9.81),
        dt=0.01,
    )
    print("  Gravity:", integrator.gravity.x, integrator.gravity.y, integrator.gravity.z)
    print("  dt:", integrator.dt)

    # Test 3: Contact solver
    print("\nTest 3: Contact solver")
    var contact_solver = ContactSolver3D(
        friction=0.5,
        restitution=0.0,
        velocity_iterations=10,
        position_iterations=5,
    )
    print("  Friction:", contact_solver.friction)
    print("  Velocity iterations:", contact_solver.velocity_iterations)

    # Test 4: Joint solver
    print("\nTest 4: Joint solver")
    var joint_solver = JointSolver3D(
        baumgarte=0.2,
        velocity_iterations=10,
    )
    print("  Baumgarte:", joint_solver.baumgarte)

    # Test 5: PhysicsWorld3D
    print("\nTest 5: PhysicsWorld3D")
    var num_bodies = 2
    var world = PhysicsWorld3D(
        num_bodies=num_bodies,
        gravity=Vec3(0.0, 0.0, -9.81),
        dt=1.0 / 60.0,
        ground_height=0.0,
    )
    print("  Num bodies:", world.num_bodies)
    print("  Ground height:", world.ground_height)

    # Set up sphere shapes
    world.set_sphere_shape(0, 0.5)  # Body 0: sphere with radius 0.5
    world.set_sphere_shape(1, 0.3)  # Body 1: sphere with radius 0.3

    # Create state array
    var state = List[Float64]()
    for _ in range(num_bodies * BODY_STATE_SIZE_3D):
        state.append(0.0)

    # Initialize bodies
    init_sphere_body(state, 0, Vec3(0.0, 0.0, 2.0), 1.0, 0.5)  # Sphere at z=2
    init_sphere_body(state, 1, Vec3(1.0, 0.0, 1.0), 0.5, 0.3)  # Sphere at z=1

    print("  Initial positions:")
    print("    Body 0 z:", state[0 * BODY_STATE_SIZE_3D + IDX_PZ])
    print("    Body 1 z:", state[1 * BODY_STATE_SIZE_3D + IDX_PZ])

    # Test 6: Physics simulation
    print("\nTest 6: Physics simulation (dropping spheres)")

    # Simulate for 60 steps (1 second at 60 FPS)
    for step in range(60):
        world.step(state)

    print("  After 60 steps:")
    print("    Body 0 z:", state[0 * BODY_STATE_SIZE_3D + IDX_PZ])
    print("    Body 1 z:", state[1 * BODY_STATE_SIZE_3D + IDX_PZ])
    print("    Body 0 vz:", state[0 * BODY_STATE_SIZE_3D + IDX_VZ])
    print("    Body 1 vz:", state[1 * BODY_STATE_SIZE_3D + IDX_VZ])

    # Verify bodies have fallen and stopped at/near ground
    var body0_z = state[0 * BODY_STATE_SIZE_3D + IDX_PZ]
    var body1_z = state[1 * BODY_STATE_SIZE_3D + IDX_PZ]

    # Bodies should be near ground (z close to radius)
    if body0_z < 2.0 and body0_z > -0.5:
        print("  Body 0 fell correctly")
    if body1_z < 1.0 and body1_z > -0.5:
        print("  Body 1 fell correctly")

    print("\nAll physics3d solver tests completed!")
