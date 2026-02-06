"""Phase 1 Validation: Free fall test.

Tests a single free-falling body against analytical solution.
Analytical solution: z(t) = z0 + v0*t + 0.5*g*t^2, vz(t) = v0 + g*t

Expected behavior:
- Drop from h=10m with v0=0
- At t=1s: z ≈ 5.095m, vz ≈ -9.81 m/s
- Acceptance: < 1% error vs analytical

Run with:
    pixi run mojo run physics3d/tests/test_freefall.mojo
"""

from physics3d.types import Model, Data
from physics3d.integrator import ImpulseIntegrator

# Use 1 body for single-body free fall test
comptime NUM_BODIES: Int = 1
comptime MAX_CONTACTS: Int = 5
comptime DTYPE = DType.float64


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


fn analytical_z(z0: Float64, v0: Float64, g: Float64, t: Float64) -> Float64:
    """Analytical solution for z position under constant gravity."""
    return z0 + v0 * t + 0.5 * g * t * t


fn analytical_vz(v0: Float64, g: Float64, t: Float64) -> Float64:
    """Analytical solution for z velocity under constant gravity."""
    return v0 + g * t


fn test_freefall() -> Bool:
    """Test free fall from 10m height."""
    print("=" * 60)
    print("Phase 1 Validation: Free Fall Test")
    print("=" * 60)

    # Setup
    var z0: Float64 = 10.0
    var v0: Float64 = 0.0
    var g: Float64 = -9.81
    var dt: Float64 = 0.01
    var total_time: Float64 = 1.0
    var num_steps = Int(total_time / dt)

    # Create model - single sphere high above ground (no contact)
    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=g,
        timestep=dt,
        ground_z=-100.0,  # Ground far below to avoid contact
        restitution=0.0,
    )
    model.set_body(0, mass=1.0, radius=0.1)

    # Create data
    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    data.set_body_position(0, 0.0, 0.0, z0)
    data.set_body_velocity(0, 0.0, 0.0, v0)

    print("\nInitial state:")
    print("  z0 =", z0, "m")
    print("  v0 =", v0, "m/s")
    print("  g =", g, "m/s^2")
    print("  dt =", dt, "s")
    print("  Total time =", total_time, "s")
    print("  Steps =", num_steps)

    # Test points
    var test_times = List[Float64]()
    test_times.append(0.0)
    test_times.append(0.5)
    test_times.append(1.0)

    var max_z_error: Float64 = 0.0
    var max_vz_error: Float64 = 0.0

    print("\nSimulation results:")
    print("-" * 60)

    var current_step = 0
    for i in range(len(test_times)):
        var t = test_times[i]
        var target_step = Int(t / dt)

        # Advance simulation to target step
        while current_step < target_step:
            ImpulseIntegrator.step(model, data)
            current_step += 1

        # Get simulated values
        var sim_z = Float64(data.get_body_z(0))
        var sim_vz = Float64(data.get_body_vz(0))

        # Get analytical values
        var expected_z = analytical_z(z0, v0, g, t)
        var expected_vz = analytical_vz(v0, g, t)

        # Compute errors
        var z_error = abs_val(sim_z - expected_z)
        var vz_error = abs_val(sim_vz - expected_vz)

        # Relative errors (avoid division by zero)
        var z_rel_error: Float64 = 0.0
        if abs_val(expected_z) > 1e-10:
            z_rel_error = z_error / abs_val(expected_z) * 100.0

        var vz_rel_error: Float64 = 0.0
        if abs_val(expected_vz) > 1e-10:
            vz_rel_error = vz_error / abs_val(expected_vz) * 100.0

        max_z_error = max_val(max_z_error, z_rel_error)
        max_vz_error = max_val(max_vz_error, vz_rel_error)

        print("  t =", t, "s:")
        print(
            "    z  = ",
            sim_z,
            " (expected:",
            expected_z,
            ", error:",
            z_rel_error,
            "%)",
        )
        print(
            "    vz = ",
            sim_vz,
            " (expected:",
            expected_vz,
            ", error:",
            vz_rel_error,
            "%)",
        )

    print("-" * 60)
    print("\nMax relative errors:")
    print("  z error:  ", max_z_error, "%")
    print("  vz error: ", max_vz_error, "%")

    # Pass criteria: < 1% error
    var passed = max_z_error < 1.0 and max_vz_error < 1.0

    print()
    if passed:
        print("PASSED: Free fall test within 1% tolerance")
    else:
        print("FAILED: Errors exceed 1% tolerance")

    print("=" * 60)
    return passed


fn test_freefall_convergence() -> Bool:
    """Test that smaller timesteps give better accuracy (convergence)."""
    print("\n")
    print("=" * 60)
    print("Phase 1 Validation: Convergence Test")
    print("=" * 60)

    var z0: Float64 = 10.0
    var g: Float64 = -9.81
    var total_time: Float64 = 1.0

    var timesteps = List[Float64]()
    timesteps.append(0.1)
    timesteps.append(0.01)
    timesteps.append(0.001)

    var errors = List[Float64]()

    print("\nTesting convergence with different timesteps:")
    print("-" * 60)

    for i in range(len(timesteps)):
        var dt = timesteps[i]
        var num_steps = Int(total_time / dt)

        var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
            gravity_z=g,
            timestep=dt,
            ground_z=-100.0,  # Ground far below
            restitution=0.0,
        )
        model.set_body(0, mass=1.0, radius=0.1)

        var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
        data.set_body_position(0, 0.0, 0.0, z0)

        for _ in range(num_steps):
            ImpulseIntegrator.step(model, data)

        var sim_z = Float64(data.get_body_z(0))
        var expected_z = analytical_z(z0, 0.0, g, total_time)
        var error = abs_val(sim_z - expected_z)
        errors.append(error)

        print("  dt =", dt, "s: z =", sim_z, ", error =", error, "m")

    print("-" * 60)

    # Check that errors decrease with smaller timesteps
    var converging = errors[1] < errors[0] and errors[2] < errors[1]

    # Check approximate O(dt) convergence for semi-implicit Euler
    # Error ratio should be roughly dt1/dt2 for first-order methods
    var ratio1 = errors[0] / errors[1]
    var ratio2 = errors[1] / errors[2]

    print("\nConvergence ratios:")
    print("  Error(0.1)/Error(0.01)   =", ratio1, "(expected ~10 for O(dt))")
    print("  Error(0.01)/Error(0.001) =", ratio2, "(expected ~10 for O(dt))")

    # Allow some tolerance on the convergence rate
    var good_convergence = ratio1 > 5.0 and ratio2 > 5.0

    print()
    if converging and good_convergence:
        print("PASSED: Errors decrease appropriately with smaller timesteps")
    else:
        print("FAILED: Convergence not as expected")

    print("=" * 60)
    return converging and good_convergence


fn test_ball_drop() -> Bool:
    """Test ball drop onto ground (Phase 2 equivalent).

    Drop a sphere from h=1m with radius=0.1m.
    Expected: Ball stops at z=0.1m (radius above ground).
    """
    print("\n")
    print("=" * 60)
    print("Phase 2 Validation: Ball Drop onto Ground")
    print("=" * 60)

    # Setup
    var radius: Float64 = 0.1
    var initial_z: Float64 = 1.0
    var expected_final_z: Float64 = radius  # Should rest at z = radius
    var dt: Float64 = 0.001  # Small timestep for accuracy
    var max_time: Float64 = 2.0
    var max_steps = Int(max_time / dt)

    # Create model with ground at z=0
    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=dt,
        ground_z=0.0,
        restitution=0.0,  # Inelastic
    )
    model.set_body(0, mass=1.0, radius=radius)

    # Initialize data
    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    data.set_body_position(0, 0.0, 0.0, initial_z)

    print("\nSetup:")
    print("  Radius:", radius, "m")
    print("  Initial z:", initial_z, "m")
    print("  Expected final z:", expected_final_z, "m")
    print("  Timestep:", dt, "s")

    # Simulate until settled or timeout
    var settled = False
    var settle_count = 0
    var settle_threshold = 100

    for i in range(max_steps):
        ImpulseIntegrator.step(model, data)

        var vz = Float64(data.get_body_vz(0))

        # Check if settled (low velocity and has ground contact)
        if abs_val(vz) < 0.001 and data.num_contacts > 0:
            settle_count += 1
            if settle_count >= settle_threshold:
                settled = True
                print("\n  Settled after", Float64(i) * dt, "seconds")
                break
        else:
            settle_count = 0

    # Get final state
    var final_z = Float64(data.get_body_z(0))
    var final_vz = Float64(data.get_body_vz(0))
    var z_error = abs_val(final_z - expected_final_z)

    print("\nResults:")
    print("  Final z:", final_z, "m")
    print("  Final vz:", final_vz, "m/s")
    print("  Z error:", z_error * 1000, "mm")

    # Pass criteria
    var z_ok = z_error < 0.002  # Within 2mm
    var v_ok = abs_val(final_vz) < 0.01
    var passed = settled and z_ok and v_ok

    print()
    if passed:
        print("PASSED: Ball stopped at correct height")
    else:
        if not settled:
            print("FAILED: Ball did not settle")
        if not z_ok:
            print("FAILED: Z error exceeds 2mm")
        if not v_ok:
            print("FAILED: Final velocity exceeds 0.01 m/s")

    print("=" * 60)
    return passed


fn test_ball_bounce() -> Bool:
    """Test ball bounce with restitution.

    Drop sphere from h=1m with restitution e=0.5.
    Expected: Bounces to h ~ e^2 * h = 0.25m.
    """
    print("\n")
    print("=" * 60)
    print("Phase 2 Validation: Ball Bounce")
    print("=" * 60)

    # Setup
    var radius: Float64 = 0.1
    var initial_z: Float64 = 1.0
    var restitution: Float64 = 0.5
    # Expected bounce height: e^2 * (h - r) + r
    var expected_bounce_height = (
        restitution * restitution * (initial_z - radius) + radius
    )
    var dt: Float64 = 0.001
    var max_time: Float64 = 3.0
    var max_steps = Int(max_time / dt)

    # Create model with restitution
    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=dt,
        ground_z=0.0,
        restitution=restitution,
    )
    model.set_body(0, mass=1.0, radius=radius)

    # Initialize data
    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    data.set_body_position(0, 0.0, 0.0, initial_z)

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
        ImpulseIntegrator.step(model, data)

        var z = Float64(data.get_body_z(0))
        var vz = Float64(data.get_body_vz(0))

        # Detect first contact
        if not first_contact and data.num_contacts > 0:
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

            # Stop when ball comes back down
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
    """Run all Phase 1 & 2 tests."""
    print("\n")
    print("=" * 60)
    print("    PHYSICS3D v2 - Free Fall & Contact Validation Suite    ")
    print("=" * 60)
    print()

    var all_passed = True

    # Phase 1: Free fall tests
    if not test_freefall():
        all_passed = False

    if not test_freefall_convergence():
        all_passed = False

    # Phase 2: Contact tests
    if not test_ball_drop():
        all_passed = False

    if not test_ball_bounce():
        all_passed = False

    print("\n")
    print("=" * 60)
    if all_passed:
        print("              ALL TESTS PASSED                             ")
    else:
        print("              SOME TESTS FAILED                            ")
    print("=" * 60)
    print()
