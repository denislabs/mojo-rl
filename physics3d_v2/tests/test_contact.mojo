"""Phase 2 Validation: Ground contact test.

Tests sphere-plane collision detection and impulse-based contact resolution.

Test 1 (Ball drop): Drop sphere onto ground, verify it stops at correct height.
Test 2 (Ball at rest): Start sphere touching ground, verify no drift.
Test 3 (Ball bounce): Drop with restitution, verify bounce height.

Run with:
    pixi run mojo run physics3d_v2/tests/test_contact.mojo
"""

from physics3d_v2.types import Body, Geom, Model, Data
from physics3d_v2 import step


fn abs_val(x: Float64) -> Float64:
    """Absolute value."""
    if x < 0:
        return -x
    return x


fn max_val(a: Float64, b: Float64) -> Float64:
    """Maximum of two values."""
    if a > b:
        return a
    return b


fn test_ball_drop() -> Bool:
    """Test 1: Ball drop onto ground.

    Drop a sphere from h=1m with radius=0.1m.
    Expected: Ball stops at z=0.1m (radius above ground).
    Acceptance: Final z within 1mm of expected, velocity < 0.01 m/s.
    """
    print("=" * 60)
    print("Phase 2 Test 1: Ball Drop")
    print("=" * 60)

    # Setup
    var radius: Float64 = 0.1
    var initial_z: Float64 = 1.0
    var expected_final_z: Float64 = radius  # Should rest at z = radius
    var dt: Float64 = 0.001  # Small timestep for accuracy
    var max_time: Float64 = 2.0  # Should settle well before this
    var max_steps = Int(max_time / dt)

    # Create model
    var body = Body.create_sphere(mass=1.0)
    var geom = Geom.sphere(radius)
    var model = Model.create(
        body, geom, timestep=dt, gravity_z=-9.81, ground_z=0.0, restitution=0.0
    )

    # Initialize data
    var data = Data[DType.float64]()
    data.set_position(0, 0, initial_z)

    print("\nSetup:")
    print("  Radius:", radius, "m")
    print("  Initial z:", initial_z, "m")
    print("  Expected final z:", expected_final_z, "m")
    print("  Timestep:", dt, "s")

    # Simulate until settled or timeout
    var settled = False
    var settle_count = 0
    var settle_threshold = 100  # Steps with low velocity to consider settled

    for i in range(max_steps):
        step(model, data)

        var vz = data.get_vz()

        # Check if settled (low velocity near ground)
        if abs_val(vz) < 0.001 and data.contact.active:
            settle_count += 1
            if settle_count >= settle_threshold:
                settled = True
                print("\n  Settled after", Float64(i) * dt, "seconds")
                break
        else:
            settle_count = 0

    # Get final state
    var final_z = data.get_z()
    var final_vz = data.get_vz()
    var z_error = abs_val(final_z - expected_final_z)

    print("\nResults:")
    print("  Final z:", final_z, "m")
    print("  Final vz:", final_vz, "m/s")
    print("  Z error:", z_error * 1000, "mm")

    # Pass criteria
    var z_ok = z_error < 0.001  # Within 1mm
    var v_ok = abs_val(final_vz) < 0.01  # Velocity < 0.01 m/s
    var passed = settled and z_ok and v_ok

    print()
    if passed:
        print("PASSED: Ball stopped at correct height")
    else:
        if not settled:
            print("FAILED: Ball did not settle")
        if not z_ok:
            print("FAILED: Z error exceeds 1mm")
        if not v_ok:
            print("FAILED: Final velocity exceeds 0.01 m/s")

    print("=" * 60)
    return passed


fn test_ball_at_rest() -> Bool:
    """Test 2: Ball at rest on ground.

    Start sphere touching ground at z=0.1m (radius), zero velocity.
    Expected: Stays at z=0.1m for 1000 steps.
    Acceptance: Max drift < 1mm, no oscillation.
    """
    print("\n")
    print("=" * 60)
    print("Phase 2 Test 2: Ball at Rest")
    print("=" * 60)

    # Setup
    var radius: Float64 = 0.1
    var initial_z: Float64 = radius  # Exactly touching ground
    var dt: Float64 = 0.01
    var num_steps = 1000

    # Create model
    var body = Body.create_sphere(mass=1.0)
    var geom = Geom.sphere(radius)
    var model = Model.create(
        body, geom, timestep=dt, gravity_z=-9.81, ground_z=0.0, restitution=0.0
    )

    # Initialize data at rest position
    var data = Data[DType.float64]()
    data.set_position(0, 0, initial_z)

    print("\nSetup:")
    print("  Radius:", radius, "m")
    print("  Initial z:", initial_z, "m (touching ground)")
    print("  Steps:", num_steps)

    # Track min/max z for drift detection
    var min_z = initial_z
    var max_z = initial_z
    var max_vz: Float64 = 0.0

    for _ in range(num_steps):
        step(model, data)
        var z = data.get_z()
        var vz = data.get_vz()

        if z < min_z:
            min_z = z
        if z > max_z:
            max_z = z
        if abs_val(vz) > max_vz:
            max_vz = abs_val(vz)

    # Compute drift
    var drift = max_z - min_z
    var final_z = data.get_z()
    var position_error = abs_val(final_z - initial_z)

    print("\nResults:")
    print("  Final z:", final_z, "m")
    print("  Min z:", min_z, "m")
    print("  Max z:", max_z, "m")
    print("  Drift (max-min):", drift * 1000, "mm")
    print("  Position error:", position_error * 1000, "mm")
    print("  Max |vz|:", max_vz, "m/s")

    # Pass criteria
    var drift_ok = drift < 0.001  # Drift < 1mm
    var pos_ok = position_error < 0.002  # Final position within 2mm
    var passed = drift_ok and pos_ok

    print()
    if passed:
        print("PASSED: Ball stayed at rest with minimal drift")
    else:
        if not drift_ok:
            print("FAILED: Drift exceeds 1mm")
        if not pos_ok:
            print("FAILED: Position error exceeds 2mm")

    print("=" * 60)
    return passed


fn test_ball_bounce() -> Bool:
    """Test 3: Ball bounce with restitution.

    Drop sphere from h=1m with restitution e=0.5.
    Expected: Bounces to h ≈ e² × h = 0.25m.
    Acceptance: Bounce height within 10% of expected.
    """
    print("\n")
    print("=" * 60)
    print("Phase 2 Test 3: Ball Bounce")
    print("=" * 60)

    # Setup
    var radius: Float64 = 0.1
    var initial_z: Float64 = 1.0
    var restitution: Float64 = 0.5
    # Expected bounce height: e² × (h - r) + r
    # Kinetic energy at impact ∝ (h-r), after bounce ∝ e²(h-r)
    var expected_bounce_height = (
        restitution * restitution * (initial_z - radius) + radius
    )
    var dt: Float64 = 0.001
    var max_time: Float64 = 3.0
    var max_steps = Int(max_time / dt)

    # Create model with restitution
    var body = Body.create_sphere(mass=1.0)
    var geom = Geom.sphere(radius)
    var model = Model.create(
        body,
        geom,
        timestep=dt,
        gravity_z=-9.81,
        ground_z=0.0,
        restitution=restitution,
    )

    # Initialize data
    var data = Data[DType.float64]()
    data.set_position(0, 0, initial_z)

    print("\nSetup:")
    print("  Radius:", radius, "m")
    print("  Initial z:", initial_z, "m")
    print("  Restitution:", restitution)
    print("  Expected bounce height:", expected_bounce_height, "m")

    # Simulate and track max height after first contact
    var first_contact = False
    var after_bounce = False
    var max_height_after_bounce: Float64 = 0.0
    var prev_vz: Float64 = 0.0

    for i in range(max_steps):
        step(model, data)

        var z = data.get_z()
        var vz = data.get_vz()

        # Detect first contact
        if not first_contact and data.contact.active:
            first_contact = True
            print("  First contact at t =", Float64(i) * dt, "s")

        # After contact, detect when ball starts going up (bounce)
        if first_contact and not after_bounce and vz > 0.01:
            after_bounce = True
            print("  Bounce detected at t =", Float64(i) * dt, "s")

        # Track max height after bounce
        if after_bounce:
            if z > max_height_after_bounce:
                max_height_after_bounce = z

            # Stop when ball comes back down and velocity is downward again
            if prev_vz > 0 and vz < 0:
                print("  Apex reached at t =", Float64(i) * dt, "s")
                break

        prev_vz = vz

    print("\nResults:")
    print("  Max height after bounce:", max_height_after_bounce, "m")
    print("  Expected:", expected_bounce_height, "m")

    var height_error = abs_val(max_height_after_bounce - expected_bounce_height)
    var relative_error = height_error / expected_bounce_height * 100

    print("  Error:", relative_error, "%")

    # Pass criteria: within 10% of expected
    var passed = relative_error < 10.0

    print()
    if passed:
        print("PASSED: Bounce height within 10% of expected")
    else:
        print("FAILED: Bounce height error exceeds 10%")

    print("=" * 60)
    return passed


fn main():
    """Run all Phase 2 tests."""
    print("\n")
    print("=" * 60)
    print("         PHYSICS3D v2 - Phase 2 Validation Suite           ")
    print("=" * 60)
    print()

    var all_passed = True

    # Test 1: Ball drop
    if not test_ball_drop():
        all_passed = False

    # Test 2: Ball at rest
    if not test_ball_at_rest():
        all_passed = False

    # Test 3: Ball bounce
    if not test_ball_bounce():
        all_passed = False

    print("\n")
    print("=" * 60)
    if all_passed:
        print("              ALL PHASE 2 TESTS PASSED                     ")
    else:
        print("              SOME PHASE 2 TESTS FAILED                    ")
    print("=" * 60)
    print()
