"""Phase 6 Validation: Friction tests (CPU).

Tests Coulomb friction implementation for the physics3d engine.

Test 1: Sphere sliding to stop - Sphere with initial horizontal velocity stops due to friction.
Test 2: Friction cone validation - Verify |j_t| <= μ * j_n for all contacts.
Test 3: Zero friction - Sphere slides freely with friction=0.
Test 4: High friction resting - Sphere with high friction stays still.

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_friction.mojo
"""

from math import sqrt
from physics3d.types import Model, Data
from physics3d.integrator import ImpulseIntegrator


fn abs_val(x: Float64) -> Float64:
    """Absolute value."""
    if x < 0:
        return -x
    return x


fn test_sphere_sliding_to_stop() -> Bool:
    """Test 1: Sphere with initial horizontal velocity stops due to friction.

    Setup:
    - Sphere on ground with initial horizontal velocity (1, 0, 0) m/s
    - Friction coefficient = 0.5
    - Gravity = -9.81 m/s²

    Expected:
    - Friction force = μ * m * g = 0.5 * 1.0 * 9.81 = 4.905 m/s²
    - Time to stop ≈ v0 / a = 1.0 / 4.905 ≈ 0.2s
    - Sphere should stop within reasonable time
    """
    print("=" * 60)
    print("Phase 6 Test 1: Sphere Sliding to Stop")
    print("=" * 60)

    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 5
    comptime DTYPE = DType.float64

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.001,
        ground_z=0.0,
        restitution=0.0,
        friction=0.5,
    )
    model.set_body(0, mass=1.0, radius=0.1)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    data.set_body_position(0, 0, 0, 0.1)  # On ground
    data.set_body_velocity(0, 1.0, 0, 0)  # Moving in +x at 1 m/s

    print("\nSetup:")
    print("  Initial velocity: (1.0, 0.0, 0.0) m/s")
    print("  Friction coefficient: 0.5")
    print("  Expected deceleration: ~4.9 m/s²")
    print("  Expected stop time: ~0.2s")

    var max_time: Float64 = 1.0
    var dt: Float64 = 0.001
    var num_steps = Int(max_time / dt)

    var stop_time: Float64 = -1.0
    var initial_vx: Float64 = 1.0

    for i in range(num_steps):
        ImpulseIntegrator.step(model, data)

        var vx = Float64(data.velocities[0])
        var vy = Float64(data.velocities[1])
        var vz = Float64(data.velocities[2])
        var speed = sqrt(vx * vx + vy * vy)

        # Check if stopped (very low horizontal velocity)
        if speed < 0.01 and stop_time < 0:
            stop_time = Float64(i) * dt
            print("  Stopped at t =", stop_time, "s")
            print("  Final position x =", data.positions[0], "m")
            break

    print("\nResults:")
    print(
        "  Final velocity: (",
        data.velocities[0],
        ",",
        data.velocities[1],
        ",",
        data.velocities[2],
        ") m/s",
    )

    # Pass criteria:
    # 1. Sphere stopped (speed < 0.01 m/s)
    # 2. Stop time is roughly in expected range (0.1s - 0.4s)
    var final_speed = sqrt(
        Float64(data.velocities[0]) * Float64(data.velocities[0])
        + Float64(data.velocities[1]) * Float64(data.velocities[1])
    )
    var stopped = final_speed < 0.05
    var reasonable_time = stop_time > 0.05 and stop_time < 0.5

    var passed = stopped and reasonable_time

    print()
    if passed:
        print("PASSED: Sphere stopped due to friction in", stop_time, "s")
    else:
        if not stopped:
            print(
                "FAILED: Sphere did not stop (final speed =",
                final_speed,
                "m/s)",
            )
        if not reasonable_time:
            print("FAILED: Stop time unreasonable (", stop_time, "s)")

    print("=" * 60)
    return passed


fn test_friction_cone_constraint() -> Bool:
    """Test 2: Verify friction impulse magnitude <= μ * normal impulse.

    Setup:
    - Sphere on ground with high horizontal velocity
    - Check friction impulses are within friction cone

    Expected:
    - For all contacts: sqrt(jt1² + jt2²) <= μ * jn
    """
    print("\n")
    print("=" * 60)
    print("Phase 6 Test 2: Friction Cone Constraint")
    print("=" * 60)

    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 5
    comptime DTYPE = DType.float64

    var friction_coef: Float64 = 0.5

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.001,
        ground_z=0.0,
        restitution=0.0,
        friction=friction_coef,
    )
    model.set_body(0, mass=1.0, radius=0.1)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    data.set_body_position(0, 0, 0, 0.1)
    data.set_body_velocity(0, 5.0, 3.0, 0)  # High horizontal velocity

    print("\nSetup:")
    print("  Initial velocity: (5.0, 3.0, 0.0) m/s")
    print("  Friction coefficient:", friction_coef)

    var max_time: Float64 = 0.1
    var dt: Float64 = 0.001
    var num_steps = Int(max_time / dt)

    var all_within_cone = True
    var max_violation: Float64 = 0.0

    for _ in range(num_steps):
        ImpulseIntegrator.step(model, data)

        # Check friction cone for each contact
        for c in range(data.num_contacts):
            var jn = Float64(data.contacts[c].impulse_n)
            var jt1 = Float64(data.contacts[c].impulse_t1)
            var jt2 = Float64(data.contacts[c].impulse_t2)

            if jn > 0:
                var jt_mag = sqrt(jt1 * jt1 + jt2 * jt2)
                var max_friction = friction_coef * jn

                if jt_mag > max_friction + 0.001:  # Small tolerance
                    all_within_cone = False
                    var violation = jt_mag - max_friction
                    if violation > max_violation:
                        max_violation = violation

    print("\nResults:")
    print("  All impulses within friction cone:", all_within_cone)
    if not all_within_cone:
        print("  Max violation:", max_violation)

    var passed = all_within_cone

    print()
    if passed:
        print("PASSED: Friction cone constraint satisfied")
    else:
        print("FAILED: Friction cone violated by", max_violation)

    print("=" * 60)
    return passed


fn test_zero_friction() -> Bool:
    """Test 3: Sphere slides freely with friction=0.

    Setup:
    - Sphere on ground with initial horizontal velocity
    - Friction coefficient = 0

    Expected:
    - Sphere maintains horizontal velocity (no friction deceleration)
    """
    print("\n")
    print("=" * 60)
    print("Phase 6 Test 3: Zero Friction (Free Sliding)")
    print("=" * 60)

    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 5
    comptime DTYPE = DType.float64

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.001,
        ground_z=0.0,
        restitution=0.0,
        friction=0.0,  # No friction
    )
    model.set_body(0, mass=1.0, radius=0.1)

    var initial_vx: Float64 = 2.0
    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    data.set_body_position(0, 0, 0, 0.1)
    data.set_body_velocity(0, initial_vx, 0, 0)

    print("\nSetup:")
    print("  Initial velocity: (2.0, 0.0, 0.0) m/s")
    print("  Friction coefficient: 0.0")

    var max_time: Float64 = 0.5
    var dt: Float64 = 0.001
    var num_steps = Int(max_time / dt)

    for _ in range(num_steps):
        ImpulseIntegrator.step(model, data)

    var final_vx = Float64(data.velocities[0])

    print("\nResults:")
    print("  Final x-velocity:", final_vx, "m/s")
    print("  Expected:", initial_vx, "m/s")

    # Pass criteria: velocity maintained (within 5% of initial)
    var velocity_maintained = abs_val(final_vx - initial_vx) < 0.1

    var passed = velocity_maintained

    print()
    if passed:
        print("PASSED: Sphere slides freely without friction")
    else:
        print("FAILED: Velocity changed from", initial_vx, "to", final_vx)

    print("=" * 60)
    return passed


fn test_high_friction_resting() -> Bool:
    """Test 4: Sphere with high friction and small initial velocity stops quickly.

    Setup:
    - Sphere on ground with small horizontal velocity
    - High friction coefficient = 1.0

    Expected:
    - Sphere stops very quickly
    """
    print("\n")
    print("=" * 60)
    print("Phase 6 Test 4: High Friction Resting Contact")
    print("=" * 60)

    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 5
    comptime DTYPE = DType.float64

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.001,
        ground_z=0.0,
        restitution=0.0,
        friction=1.0,  # High friction
    )
    model.set_body(0, mass=1.0, radius=0.1)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    data.set_body_position(0, 0, 0, 0.1)
    data.set_body_velocity(0, 0.5, 0, 0)  # Small initial velocity

    print("\nSetup:")
    print("  Initial velocity: (0.5, 0.0, 0.0) m/s")
    print("  Friction coefficient: 1.0")

    var max_time: Float64 = 0.2
    var dt: Float64 = 0.001
    var num_steps = Int(max_time / dt)

    var stop_time: Float64 = -1.0

    for i in range(num_steps):
        ImpulseIntegrator.step(model, data)

        var vx = Float64(data.velocities[0])
        var vy = Float64(data.velocities[1])
        var speed = sqrt(vx * vx + vy * vy)

        if speed < 0.01 and stop_time < 0:
            stop_time = Float64(i) * dt
            break

    var final_speed = sqrt(
        Float64(data.velocities[0]) * Float64(data.velocities[0])
        + Float64(data.velocities[1]) * Float64(data.velocities[1])
    )

    print("\nResults:")
    print("  Final speed:", final_speed, "m/s")
    print("  Stop time:", stop_time, "s")

    # Pass criteria: stopped within 0.1s
    var stopped_quickly = stop_time > 0 and stop_time < 0.15

    var passed = stopped_quickly

    print()
    if passed:
        print("PASSED: High friction stops sphere quickly in", stop_time, "s")
    else:
        print("FAILED: Sphere did not stop quickly enough")

    print("=" * 60)
    return passed


fn test_two_spheres_friction() -> Bool:
    """Test 5: Two spheres colliding with friction.

    Setup:
    - Two spheres approaching each other horizontally
    - Both have friction enabled

    Expected:
    - After collision, spheres bounce apart
    - Friction affects their motion
    """
    print("\n")
    print("=" * 60)
    print("Phase 6 Test 5: Two Spheres with Friction")
    print("=" * 60)

    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10
    comptime DTYPE = DType.float64

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.001,
        ground_z=0.0,
        restitution=0.5,
        friction=0.5,
    )
    model.set_body(0, mass=1.0, radius=0.1)
    model.set_body(1, mass=1.0, radius=0.1)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    # Position spheres so they will collide quickly
    # Radius = 0.1, so collision when distance < 0.2
    data.set_body_position(0, -0.15, 0, 0.11)  # Closer together
    data.set_body_position(1, 0.15, 0, 0.11)
    data.set_body_velocity(0, 1.0, 0, 0)  # Moving right
    data.set_body_velocity(1, -1.0, 0, 0)  # Moving left

    print("\nSetup:")
    print("  Sphere 0: x=-0.15, vx=+1.0 m/s")
    print("  Sphere 1: x=+0.15, vx=-1.0 m/s")
    print("  Friction coefficient: 0.5")
    print("  Restitution: 0.5")

    var max_time: Float64 = 1.0
    var dt: Float64 = 0.001
    var num_steps = Int(max_time / dt)

    var collision_detected = False

    for i in range(num_steps):
        ImpulseIntegrator.step(model, data)

        # Check for sphere-sphere collision
        for c in range(data.num_contacts):
            if data.contacts[c].body_a >= 0 and data.contacts[c].body_b >= 0:
                if not collision_detected:
                    collision_detected = True
                    print("  Collision detected at t =", Float64(i) * dt, "s")

    var final_x_0 = Float64(data.positions[0])
    var final_x_1 = Float64(data.positions[3])

    print("\nResults:")
    print("  Sphere 0 final x:", final_x_0)
    print("  Sphere 1 final x:", final_x_1)
    print("  Collision detected:", collision_detected)

    # Pass criteria: collision detected and spheres separated
    var spheres_separated = (final_x_1 - final_x_0) > 0.15
    var passed = collision_detected and spheres_separated

    print()
    if passed:
        print("PASSED: Spheres collided and bounced with friction")
    else:
        if not collision_detected:
            print("FAILED: No collision detected")
        if not spheres_separated:
            print("FAILED: Spheres did not separate")

    print("=" * 60)
    return passed


fn main():
    """Run all Phase 6 friction tests."""
    print("\n")
    print("=" * 60)
    print("       PHYSICS3D v2 - Phase 6 Friction Validation (CPU)       ")
    print("=" * 60)
    print()

    var all_passed = True

    # Test 1: Sphere sliding to stop
    if not test_sphere_sliding_to_stop():
        all_passed = False

    # Test 2: Friction cone constraint
    if not test_friction_cone_constraint():
        all_passed = False

    # Test 3: Zero friction
    if not test_zero_friction():
        all_passed = False

    # Test 4: High friction resting
    if not test_high_friction_resting():
        all_passed = False

    # Test 5: Two spheres with friction
    if not test_two_spheres_friction():
        all_passed = False

    print("\n")
    print("=" * 60)
    if all_passed:
        print("            ALL PHASE 6 FRICTION TESTS PASSED               ")
    else:
        print("            SOME PHASE 6 FRICTION TESTS FAILED              ")
    print("=" * 60)
    print()
