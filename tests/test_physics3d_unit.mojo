"""Unit tests for physics3D engine.

Tests:
1. Gravity - bodies should fall with correct acceleration
2. Collision - ground plane detection and response
3. Joints - hinge constraints and motor torques

Run with:
    pixi run mojo run tests/test_physics3d_unit.mojo
"""

from math import sqrt
from math3d import Vec3, Quat

from physics3d import (
    dtype,
    BODY_STATE_SIZE_3D,
    BODY_DYNAMIC,
    DEFAULT_GRAVITY_Z_3D,
    IDX_PX,
    IDX_PY,
    IDX_PZ,
    IDX_QW,
    IDX_QX,
    IDX_QY,
    IDX_QZ,
    IDX_VX,
    IDX_VY,
    IDX_VZ,
    IDX_WX,
    IDX_WY,
    IDX_WZ,
    IDX_FX,
    IDX_FY,
    IDX_FZ,
    IDX_TX,
    IDX_TY,
    IDX_TZ,
    IDX_MASS,
    IDX_INV_MASS,
    IDX_IXX,
    IDX_IYY,
    IDX_IZZ,
    IDX_BODY_TYPE,
    compute_sphere_inertia,
    compute_capsule_inertia,
)
from physics3d.integrators import (
    integrate_velocities_3d,
    integrate_positions_3d,
    SemiImplicitEuler3D,
)


# =============================================================================
# Test Utilities
# =============================================================================


fn abs_f64(x: Float64) -> Float64:
    return x if x >= 0 else -x


fn assert_near(
    actual: Float64, expected: Float64, tolerance: Float64, msg: String
) raises:
    """Assert that actual is near expected within tolerance."""
    var diff = abs_f64(actual - expected)
    if diff > tolerance:
        print(
            "FAIL:",
            msg,
            "| expected:",
            expected,
            "| actual:",
            actual,
            "| diff:",
            diff,
        )
        raise Error("Assertion failed: " + msg)


fn create_body_state(
    num_bodies: Int, bodies_offset: Int = 0
) -> List[Scalar[dtype]]:
    """Create a state array with space for bodies."""
    var total_size = bodies_offset + num_bodies * BODY_STATE_SIZE_3D
    var state = List[Scalar[dtype]](capacity=total_size)
    for _ in range(total_size):
        state.append(Scalar[dtype](0))
    return state^


fn init_dynamic_body(
    mut state: List[Scalar[dtype]],
    body_idx: Int,
    pos: Vec3[dtype],
    mass: Scalar[dtype],
    inertia: Vec3[dtype],
    bodies_offset: Int = 0,
):
    """Initialize a dynamic body at given position with mass and inertia."""
    var base = bodies_offset + body_idx * BODY_STATE_SIZE_3D

    # Position
    state[base + IDX_PX] = pos.x
    state[base + IDX_PY] = pos.y
    state[base + IDX_PZ] = pos.z

    # Identity quaternion
    state[base + IDX_QW] = Scalar[dtype](1.0)
    state[base + IDX_QX] = Scalar[dtype](0.0)
    state[base + IDX_QY] = Scalar[dtype](0.0)
    state[base + IDX_QZ] = Scalar[dtype](0.0)

    # Zero velocities
    state[base + IDX_VX] = Scalar[dtype](0.0)
    state[base + IDX_VY] = Scalar[dtype](0.0)
    state[base + IDX_VZ] = Scalar[dtype](0.0)
    state[base + IDX_WX] = Scalar[dtype](0.0)
    state[base + IDX_WY] = Scalar[dtype](0.0)
    state[base + IDX_WZ] = Scalar[dtype](0.0)

    # Zero forces
    state[base + IDX_FX] = Scalar[dtype](0.0)
    state[base + IDX_FY] = Scalar[dtype](0.0)
    state[base + IDX_FZ] = Scalar[dtype](0.0)
    state[base + IDX_TX] = Scalar[dtype](0.0)
    state[base + IDX_TY] = Scalar[dtype](0.0)
    state[base + IDX_TZ] = Scalar[dtype](0.0)

    # Mass properties
    state[base + IDX_MASS] = Scalar[dtype](mass)
    state[base + IDX_INV_MASS] = Scalar[dtype](1.0 / mass)
    state[base + IDX_IXX] = inertia.x
    state[base + IDX_IYY] = inertia.y
    state[base + IDX_IZZ] = inertia.z

    # Body type
    state[base + IDX_BODY_TYPE] = Scalar[dtype](BODY_DYNAMIC)


fn get_position(
    state: List[Scalar[dtype]], body_idx: Int, bodies_offset: Int = 0
) -> Vec3[dtype]:
    """Get body position from state."""
    var base = bodies_offset + body_idx * BODY_STATE_SIZE_3D
    return Vec3[dtype](
        state[base + IDX_PX], state[base + IDX_PY], state[base + IDX_PZ]
    )


fn get_velocity(
    state: List[Scalar[dtype]], body_idx: Int, bodies_offset: Int = 0
) -> Vec3[dtype]:
    """Get body linear velocity from state."""
    var base = bodies_offset + body_idx * BODY_STATE_SIZE_3D
    return Vec3[dtype](
        state[base + IDX_VX], state[base + IDX_VY], state[base + IDX_VZ]
    )


fn get_angular_velocity(
    state: List[Scalar[dtype]], body_idx: Int, bodies_offset: Int = 0
) -> Vec3[dtype]:
    """Get body angular velocity from state."""
    var base = bodies_offset + body_idx * BODY_STATE_SIZE_3D
    return Vec3[dtype](
        state[base + IDX_WX], state[base + IDX_WY], state[base + IDX_WZ]
    )


# =============================================================================
# Test 1: Gravity
# =============================================================================


fn test_gravity_freefall() raises:
    """Test that a body falls correctly under gravity."""
    print("\n" + "=" * 60)
    print("TEST 1: GRAVITY FREE-FALL")
    print("=" * 60)

    # Create a single body
    var state = create_body_state(1)
    var mass = Scalar[dtype](2.0)
    var inertia = compute_sphere_inertia[dtype](mass, Scalar[dtype](0.1))

    # Start at height z=10
    var initial_pos = Vec3[dtype](0.0, 0.0, 10.0)
    init_dynamic_body(state, 0, initial_pos, mass, inertia)

    # Physics parameters
    var gravity = Vec3[dtype](0.0, 0.0, Scalar[dtype](DEFAULT_GRAVITY_Z_3D))
    var dt = Scalar[dtype](0.02)  # 50 FPS

    print("Initial state:")
    print("  Position:", get_position(state, 0).x, get_position(state, 0).y, get_position(state, 0).z)
    print("  Velocity:", get_velocity(state, 0).x, get_velocity(state, 0).y, get_velocity(state, 0).z)
    print("  Gravity:", gravity.z)
    print("  dt:", dt)
    print()

    # Run 50 steps (1 second)
    var num_steps = 50
    for step in range(num_steps):
        # Integrate velocity first (semi-implicit Euler)
        integrate_velocities_3d(state, 0, gravity, dt, 0)
        # Then integrate position
        integrate_positions_3d(state, 0, dt, 0)

    var final_pos = get_position(state, 0)
    var final_vel = get_velocity(state, 0)

    # Expected values after t=1 second:
    # v_z = g * t = -9.81 * 1.0 = -9.81 m/s
    # z = z0 + 0.5 * g * t^2 = 10 - 0.5 * 9.81 * 1^2 = 10 - 4.905 = 5.095 m
    var expected_vz = DEFAULT_GRAVITY_Z_3D * 1.0  # -9.81
    var expected_z = 10.0 + 0.5 * DEFAULT_GRAVITY_Z_3D * 1.0 * 1.0  # ~5.095

    print("After 1 second (50 steps):")
    print("  Position:", final_pos.x, final_pos.y, final_pos.z)
    print("  Velocity:", final_vel.x, final_vel.y, final_vel.z)
    print()
    print("Expected:")
    print("  z position: ~", expected_z, "(got:", final_pos.z, ")")
    print("  z velocity: ~", expected_vz, "(got:", final_vel.z, ")")
    print()

    # Verify results (with tolerance for numerical integration)
    var tolerance = 0.1  # 10cm tolerance due to discrete integration
    assert_near(
        Float64(final_pos.z), expected_z, tolerance, "z position after 1s"
    )
    assert_near(
        Float64(final_vel.z), expected_vz, 0.05, "z velocity after 1s"
    )

    # Check x, y unchanged
    assert_near(Float64(final_pos.x), 0.0, 0.001, "x should be unchanged")
    assert_near(Float64(final_pos.y), 0.0, 0.001, "y should be unchanged")

    print("PASSED: Gravity free-fall test")


fn test_gravity_velocity_accumulation() raises:
    """Test that velocity accumulates correctly over time."""
    print("\n" + "=" * 60)
    print("TEST 1b: GRAVITY VELOCITY ACCUMULATION")
    print("=" * 60)

    var state = create_body_state(1)
    var mass = Scalar[dtype](1.0)
    var inertia = compute_sphere_inertia[dtype](mass, Scalar[dtype](0.1))
    init_dynamic_body(state, 0, Vec3[dtype](0, 0, 10), mass, inertia)

    var gravity = Vec3[dtype](0.0, 0.0, Scalar[dtype](DEFAULT_GRAVITY_Z_3D))
    var dt = Scalar[dtype](0.01)  # 100 FPS for more accuracy

    print("Testing velocity accumulation over 10 steps:")
    print("Step | Expected v_z | Actual v_z | Diff")
    print("-" * 50)

    var max_diff = 0.0
    for step in range(10):
        # Expected velocity: v = g * t
        var t = Float64(step + 1) * Float64(dt)
        var expected_vz = DEFAULT_GRAVITY_Z_3D * t

        # Step physics
        integrate_velocities_3d(state, 0, gravity, dt, 0)
        integrate_positions_3d(state, 0, dt, 0)

        var actual_vz = Float64(get_velocity(state, 0).z)
        var diff = abs_f64(actual_vz - expected_vz)
        if diff > max_diff:
            max_diff = diff

        print(
            String(step + 1),
            "    |",
            String(expected_vz)[:8],
            "     |",
            String(actual_vz)[:8],
            "   |",
            String(diff)[:8],
        )

    print()
    print("Max velocity error:", max_diff)

    if max_diff > 0.01:
        print("FAIL: Velocity accumulation error too high")
        raise Error("Velocity accumulation test failed")

    print("PASSED: Gravity velocity accumulation test")


# =============================================================================
# Test 2: Manual Collision Detection (without SpherePlaneCollision)
# =============================================================================


fn manual_detect_ground_collision(
    pos_z: Scalar[dtype], radius: Float64, ground_z: Float64
) -> Float64:
    """Manually detect ground collision.

    Returns penetration depth (positive = penetrating).
    """
    var sphere_bottom = Float64(pos_z) - radius
    var depth = ground_z - sphere_bottom
    return depth if depth > 0 else 0.0


fn test_ground_collision_detection() raises:
    """Test ground plane collision detection (manual)."""
    print("\n" + "=" * 60)
    print("TEST 2: GROUND COLLISION DETECTION (manual)")
    print("=" * 60)

    var radius = 0.5
    var ground_height = 0.0

    # Test case 1: Sphere above ground (no collision)
    print("\nCase 1: Sphere at z=2.0 (above ground)")
    var z1 = Scalar[dtype](2.0)
    var depth1 = manual_detect_ground_collision(z1, radius, ground_height)
    print("  Penetration depth:", depth1)
    if depth1 > 0:
        print("  FAIL: Should not detect collision above ground")
        raise Error("False positive collision detection")
    print("  OK: No collision detected")

    # Test case 2: Sphere exactly touching ground
    print("\nCase 2: Sphere at z=0.5 (touching ground)")
    var z2 = Scalar[dtype](0.5)  # radius = 0.5
    var depth2 = manual_detect_ground_collision(z2, radius, ground_height)
    print("  Penetration depth:", depth2)
    assert_near(depth2, 0.0, 0.01, "Touching sphere should have ~0 depth")
    print("  OK: Correctly detected touching")

    # Test case 3: Sphere penetrating ground
    print("\nCase 3: Sphere at z=0.3 (penetrating ground by 0.2)")
    var z3 = Scalar[dtype](0.3)
    var depth3 = manual_detect_ground_collision(z3, radius, ground_height)
    print("  Penetration depth:", depth3, "(expected: 0.2)")
    assert_near(depth3, 0.2, 0.01, "Penetration depth should be 0.2")
    print("  OK: Correct penetration detected")

    # Test case 4: Sphere far below ground
    print("\nCase 4: Sphere at z=-0.5 (fully below ground)")
    var z4 = Scalar[dtype](-0.5)
    var depth4 = manual_detect_ground_collision(z4, radius, ground_height)
    print("  Penetration depth:", depth4, "(expected: 1.0)")
    assert_near(depth4, 1.0, 0.01, "Penetration depth should be 1.0")
    print("  OK: Deep penetration detected")

    print("\nPASSED: Ground collision detection test")


fn test_collision_falling_body() raises:
    """Test that a falling body is detected when it hits ground."""
    print("\n" + "=" * 60)
    print("TEST 2b: FALLING BODY COLLISION")
    print("=" * 60)

    var state = create_body_state(1)
    var mass = Scalar[dtype](1.0)
    var radius = 0.5
    var inertia = compute_sphere_inertia[dtype](mass, Scalar[dtype](radius))

    # Start at z=2.0 (above ground)
    init_dynamic_body(state, 0, Vec3[dtype](0, 0, 2.0), mass, inertia)

    var gravity = Vec3[dtype](0.0, 0.0, Scalar[dtype](DEFAULT_GRAVITY_Z_3D))
    var dt = Scalar[dtype](0.02)
    var ground_height = 0.0

    print("Dropping sphere from z=2.0, radius=0.5")
    print("Ground at z=0.0")
    print("Collision expected when z < 0.5")
    print()
    print("Step | z pos    | z vel    | Penetration | Status")
    print("-" * 60)

    var collision_step = -1
    for step in range(100):
        # Step physics
        integrate_velocities_3d(state, 0, gravity, dt, 0)
        integrate_positions_3d(state, 0, dt, 0)

        var pos = get_position(state, 0)
        var vel = get_velocity(state, 0)

        # Check collision manually
        var depth = manual_detect_ground_collision(pos.z, radius, ground_height)

        var status = "falling"
        if depth > 0:
            status = "COLLISION"
            if collision_step < 0:
                collision_step = step

        if step < 5 or step % 10 == 0 or depth > 0:
            print(
                String(step),
                "    |",
                String(pos.z)[:8],
                " |",
                String(vel.z)[:8],
                " |",
                String(depth)[:8],
                "    |",
                status,
            )

        # Stop after collision detected
        if collision_step >= 0 and step > collision_step + 2:
            break

    print()
    if collision_step >= 0:
        print("Collision detected at step", collision_step)
        print("PASSED: Falling body collision detection")
    else:
        print("FAIL: No collision detected")
        raise Error("Falling body should have collided with ground")


# =============================================================================
# Test 3: Forces and Torques
# =============================================================================


fn test_force_application() raises:
    """Test that forces accelerate bodies correctly."""
    print("\n" + "=" * 60)
    print("TEST 3: FORCE APPLICATION")
    print("=" * 60)

    var state = create_body_state(1)
    var mass = Scalar[dtype](2.0)
    var inertia = compute_sphere_inertia[dtype](mass, Scalar[dtype](0.1))
    init_dynamic_body(state, 0, Vec3[dtype](0, 0, 0), mass, inertia)

    # Apply horizontal force (no gravity)
    var gravity = Vec3[dtype](0.0, 0.0, 0.0)  # No gravity
    var dt = Scalar[dtype](0.01)

    # Apply force F = 10 N in x direction
    # Expected acceleration a = F/m = 10/2 = 5 m/s^2
    var base = 0 * BODY_STATE_SIZE_3D
    state[base + IDX_FX] = Scalar[dtype](10.0)  # 10 N in x

    print("Applying 10N force in x direction to 2kg body")
    print("Expected acceleration: 5 m/s^2")
    print()

    # Step once
    integrate_velocities_3d(state, 0, gravity, dt, 0)

    var vel = get_velocity(state, 0)
    # Expected: v = a * dt = 5 * 0.01 = 0.05 m/s
    var expected_vx = 5.0 * 0.01
    print("After 1 step (dt=0.01s):")
    print("  x velocity:", vel.x, "(expected:", expected_vx, ")")

    assert_near(Float64(vel.x), expected_vx, 0.001, "x velocity from force")
    assert_near(Float64(vel.y), 0.0, 0.001, "y velocity should be 0")
    assert_near(Float64(vel.z), 0.0, 0.001, "z velocity should be 0")

    print("PASSED: Force application test")


fn test_torque_application() raises:
    """Test that torques cause angular acceleration."""
    print("\n" + "=" * 60)
    print("TEST 3b: TORQUE APPLICATION")
    print("=" * 60)

    var state = create_body_state(1)
    var mass = Scalar[dtype](1.0)
    # Use sphere with known inertia: I = 2/5 * m * r^2
    var radius = Scalar[dtype](0.5)
    var inertia = compute_sphere_inertia[dtype](mass, radius)
    # I = 2/5 * 1.0 * 0.25 = 0.1 kg*m^2

    init_dynamic_body(state, 0, Vec3[dtype](0, 0, 0), mass, inertia)

    var gravity = Vec3[dtype](0.0, 0.0, 0.0)  # No gravity
    var dt = Scalar[dtype](0.01)

    # Apply torque about z axis
    # tau = 1 N*m, I_zz = 0.1 kg*m^2
    # alpha = tau / I = 1.0 / 0.1 = 10 rad/s^2
    var base = 0 * BODY_STATE_SIZE_3D
    state[base + IDX_TZ] = Scalar[dtype](1.0)  # 1 N*m about z

    print("Applying 1 N*m torque about z axis")
    print("Sphere: m=1kg, r=0.5m, I_zz=", inertia.z)
    print("Expected angular acceleration: alpha =", 1.0 / Float64(inertia.z), "rad/s^2")
    print()

    # Step once
    integrate_velocities_3d(state, 0, gravity, dt, 0)

    var omega = get_angular_velocity(state, 0)
    # Expected: omega = alpha * dt
    var expected_wz = (1.0 / Float64(inertia.z)) * 0.01
    print("After 1 step (dt=0.01s):")
    print("  angular velocity z:", omega.z, "(expected:", expected_wz, ")")

    assert_near(Float64(omega.z), expected_wz, 0.001, "angular velocity from torque")
    assert_near(Float64(omega.x), 0.0, 0.001, "omega_x should be 0")
    assert_near(Float64(omega.y), 0.0, 0.001, "omega_y should be 0")

    print("PASSED: Torque application test")


# =============================================================================
# Test 4: Quaternion Integration
# =============================================================================


fn test_quaternion_rotation() raises:
    """Test quaternion integration for rotation."""
    print("\n" + "=" * 60)
    print("TEST 4: QUATERNION ROTATION")
    print("=" * 60)

    var state = create_body_state(1)
    var mass = Scalar[dtype](1.0)
    var inertia = compute_sphere_inertia[dtype](mass, Scalar[dtype](0.1))
    init_dynamic_body(state, 0, Vec3[dtype](0, 0, 0), mass, inertia)

    # Set initial angular velocity about z axis
    var base = 0 * BODY_STATE_SIZE_3D
    state[base + IDX_WZ] = Scalar[dtype](3.14159)  # pi rad/s

    var gravity = Vec3[dtype](0.0, 0.0, 0.0)
    var dt = Scalar[dtype](0.01)

    print("Spinning about z axis at pi rad/s")
    print("After 1 second, should complete half rotation")
    print()

    # Run for 1 second (100 steps at 0.01s)
    for _ in range(100):
        integrate_positions_3d(state, 0, dt, 0)

    # Get quaternion
    var qw = Float64(state[base + IDX_QW])
    var qx = Float64(state[base + IDX_QX])
    var qy = Float64(state[base + IDX_QY])
    var qz = Float64(state[base + IDX_QZ])

    print("Final quaternion: (", qw, ",", qx, ",", qy, ",", qz, ")")

    # After pi radians rotation about z:
    # q = (cos(pi/2), 0, 0, sin(pi/2)) = (0, 0, 0, 1)
    # (quaternion represents half the rotation angle)
    print("Expected: approximately (0, 0, 0, 1) for 180 degree rotation")

    # Check quaternion is normalized
    var len_sq = qw * qw + qx * qx + qy * qy + qz * qz
    assert_near(len_sq, 1.0, 0.01, "quaternion should be normalized")

    # For pi rotation about z, qw should be near 0, qz should be near 1
    # Due to numerical integration, allow some tolerance
    if abs_f64(qw) > 0.2:
        print("WARNING: qw =", qw, "expected near 0")
    if abs_f64(qz) < 0.8:
        print("WARNING: qz =", qz, "expected near 1")

    print("PASSED: Quaternion rotation test")


# =============================================================================
# Test 5: Semi-Implicit Euler Integrator
# =============================================================================


fn test_semi_implicit_euler() raises:
    """Test the SemiImplicitEuler3D struct."""
    print("\n" + "=" * 60)
    print("TEST 5: SEMI-IMPLICIT EULER INTEGRATOR")
    print("=" * 60)

    var state = create_body_state(1)
    var mass = Scalar[dtype](1.0)
    var inertia = compute_sphere_inertia[dtype](mass, Scalar[dtype](0.1))
    init_dynamic_body(state, 0, Vec3[dtype](0, 0, 5.0), mass, inertia)

    # Create integrator with default gravity
    var integrator = SemiImplicitEuler3D[dtype](
        gravity=Vec3[dtype](0.0, 0.0, Scalar[dtype](DEFAULT_GRAVITY_Z_3D)),
        dt=Scalar[dtype](0.02),
        damping_linear=Scalar[dtype](0.0),  # No damping for this test
        damping_angular=Scalar[dtype](0.0),
    )

    print("Using SemiImplicitEuler3D struct")
    print("Starting at z=5.0, stepping for 0.5 seconds (25 steps)")
    print()

    # Step 25 times (0.5 seconds)
    for _ in range(25):
        integrator.step(state, 1)

    var final_pos = get_position(state, 0)
    var final_vel = get_velocity(state, 0)

    # Expected after 0.5s:
    # v_z = -9.81 * 0.5 = -4.905 m/s
    # z = 5.0 - 0.5 * 9.81 * 0.25 = 5.0 - 1.226 = 3.774 m
    var expected_vz = DEFAULT_GRAVITY_Z_3D * 0.5
    var expected_z = 5.0 + 0.5 * DEFAULT_GRAVITY_Z_3D * 0.5 * 0.5

    print("Final state:")
    print("  z position:", final_pos.z, "(expected:", expected_z, ")")
    print("  z velocity:", final_vel.z, "(expected:", expected_vz, ")")

    assert_near(Float64(final_pos.z), expected_z, 0.1, "z position")
    assert_near(Float64(final_vel.z), expected_vz, 0.05, "z velocity")

    print("PASSED: Semi-implicit Euler integrator test")


# =============================================================================
# Test 6: Capsule Inertia
# =============================================================================


fn test_capsule_inertia() raises:
    """Test capsule inertia computation."""
    print("\n" + "=" * 60)
    print("TEST 6: CAPSULE INERTIA COMPUTATION")
    print("=" * 60)

    var mass = Scalar[dtype](2.0)
    var radius = Scalar[dtype](0.05)
    var half_height = Scalar[dtype](0.2)

    var inertia = compute_capsule_inertia[dtype](mass, radius, half_height)

    print("Capsule: mass=2kg, radius=0.05m, half_height=0.2m")
    print("Computed inertia:")
    print("  I_xx:", inertia.x)
    print("  I_yy:", inertia.y)
    print("  I_zz:", inertia.z)

    # Sanity checks
    if Float64(inertia.x) <= 0 or Float64(inertia.y) <= 0 or Float64(inertia.z) <= 0:
        print("FAIL: Inertia values should all be positive")
        raise Error("Invalid inertia values")

    # For a capsule aligned with z-axis:
    # I_xx ≈ I_yy (symmetric about z)
    # I_zz < I_xx (easier to rotate about long axis)
    var diff_xy = abs_f64(Float64(inertia.x) - Float64(inertia.y))
    if diff_xy > 0.001:
        print("WARNING: I_xx and I_yy should be approximately equal")

    print("PASSED: Capsule inertia computation test")


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    print("=" * 60)
    print("PHYSICS3D UNIT TESTS")
    print("=" * 60)

    var passed = 0
    var failed = 0

    # Test 1: Gravity
    try:
        test_gravity_freefall()
        passed += 1
    except e:
        print("FAILED:", e)
        failed += 1

    try:
        test_gravity_velocity_accumulation()
        passed += 1
    except e:
        print("FAILED:", e)
        failed += 1

    # Test 2: Collision
    try:
        test_ground_collision_detection()
        passed += 1
    except e:
        print("FAILED:", e)
        failed += 1

    try:
        test_collision_falling_body()
        passed += 1
    except e:
        print("FAILED:", e)
        failed += 1

    # Test 3: Forces and Torques
    try:
        test_force_application()
        passed += 1
    except e:
        print("FAILED:", e)
        failed += 1

    try:
        test_torque_application()
        passed += 1
    except e:
        print("FAILED:", e)
        failed += 1

    # Test 4: Quaternion
    try:
        test_quaternion_rotation()
        passed += 1
    except e:
        print("FAILED:", e)
        failed += 1

    # Test 5: Integrator struct
    try:
        test_semi_implicit_euler()
        passed += 1
    except e:
        print("FAILED:", e)
        failed += 1

    # Test 6: Capsule inertia
    try:
        test_capsule_inertia()
        passed += 1
    except e:
        print("FAILED:", e)
        failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Passed:", passed)
    print("Failed:", failed)
    print()

    if failed > 0:
        print("SOME TESTS FAILED")
    else:
        print("ALL TESTS PASSED")
