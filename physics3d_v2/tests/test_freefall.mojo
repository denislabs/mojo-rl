"""Phase 1 Validation: Free fall test.

Tests a single free-falling body against analytical solution.
Analytical solution: z(t) = z0 + v0*t + 0.5*g*t^2, vz(t) = v0 + g*t

Expected behavior:
- Drop from h=10m with v0=0
- At t=1s: z ≈ 5.095m, vz ≈ -9.81 m/s
- Acceptance: < 1% error vs analytical

Run with:
    pixi run mojo run physics3d_v2/tests/test_freefall.mojo
"""

from physics3d_v2.types import Body, Geom, Model, Data
from physics3d_v2 import step_no_collision


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

    # Create model and data
    var body = Body.create_sphere(mass=1.0)
    var geom = Geom.sphere(0.1)
    var model = Model.create(body, geom, timestep=dt, gravity_z=g)

    var data = Data[DType.float64]()
    data.set_position(0, 0, z0)
    data.set_velocity(0, 0, v0)

    print("\nInitial state:")
    print("  z0 =", z0, "m")
    print("  v0 =", v0, "m/s")
    print("  g =", g, "m/s²")
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
            step_no_collision(model, data)
            current_step += 1

        # Get simulated values
        var sim_z = data.get_z()
        var sim_vz = data.get_vz()

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

        var body = Body.create_sphere(mass=1.0)
        var geom = Geom.sphere(0.1)
        var model = Model.create(body, geom, timestep=dt, gravity_z=g)

        var data = Data[DType.float64]()
        data.set_position(0, 0, z0)

        for _ in range(num_steps):
            step_no_collision(model, data)

        var sim_z = data.get_z()
        var expected_z = analytical_z(z0, 0.0, g, total_time)
        var error = abs_val(sim_z - expected_z)
        errors.append(error)

        print("  dt =", dt, "s: z =", sim_z, ", error =", error, "m")

    print("-" * 60)

    # Check that errors decrease with smaller timesteps
    var converging = errors[1] < errors[0] and errors[2] < errors[1]

    # Check approximate O(dt²) convergence for semi-implicit Euler
    # Error ratio should be roughly (dt1/dt2)² for second-order methods
    # Semi-implicit Euler is first-order, so ratio should be ~dt1/dt2
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


fn main():
    """Run all Phase 1 tests."""
    print("\n")
    print("=" * 60)
    print("         PHYSICS3D v2 - Phase 1 Validation Suite           ")
    print("=" * 60)
    print()

    var all_passed = True

    # Test 1: Basic free fall
    if not test_freefall():
        all_passed = False

    # Test 2: Convergence
    if not test_freefall_convergence():
        all_passed = False

    print("\n")
    print("=" * 60)
    if all_passed:
        print("              ALL PHASE 1 TESTS PASSED                     ")
    else:
        print("              SOME PHASE 1 TESTS FAILED                    ")
    print("=" * 60)
    print()
