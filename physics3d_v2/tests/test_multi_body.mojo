"""Phase 3 Validation: Multi-body tests.

Tests multiple bodies with sphere-plane and sphere-sphere collisions.

Test 1 (Two spheres collision): Two spheres approaching each other bounce apart.
Test 2 (Sphere stack): 3 spheres stacked should settle on ground.
Test 3 (Sphere fall on sphere): Sphere falling onto stationary sphere.

Run with:
    pixi run mojo run physics3d_v2/tests/test_multi_body.mojo
"""

from physics3d_v2.types import MultiBodyModel, MultiBodyData
from physics3d_v2.multi_body_step import step_multi_body


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


fn test_two_spheres_collide() -> Bool:
    """Test 1: Two spheres approaching each other should bounce.

    Setup:
    - Sphere 0: at x=-0.5, moving right (+x) at 1 m/s
    - Sphere 1: at x=+0.5, moving left (-x) at 1 m/s
    - Both at z=0.1 (just above ground), radius=0.1

    Expected:
    - They collide near x=0
    - After collision, spheres bounce apart (reverse velocities)
    """
    print("=" * 60)
    print("Phase 3 Test 1: Two Spheres Collision")
    print("=" * 60)

    # Setup: 2 bodies, max 10 contacts
    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10
    comptime DTYPE = DType.float64

    var model = MultiBodyModel[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81, timestep=0.001, ground_z=0.0, restitution=0.8
    )
    model.set_body(0, mass=1.0, radius=0.1)
    model.set_body(1, mass=1.0, radius=0.1)

    var data = MultiBodyData[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    data.set_body_position(0, -0.5, 0, 0.11)  # Slightly above ground
    data.set_body_position(1, 0.5, 0, 0.11)
    data.set_body_velocity(0, 1.0, 0, 0)  # Moving right
    data.set_body_velocity(1, -1.0, 0, 0)  # Moving left

    print("\nSetup:")
    print("  Sphere 0: x=-0.5, vx=+1.0 m/s")
    print("  Sphere 1: x=+0.5, vx=-1.0 m/s")
    print("  Restitution:", 0.8)

    var collision_detected = False
    var max_time: Float64 = 1.0
    var dt: Float64 = 0.001
    var num_steps = Int(max_time / dt)

    # Track whether bouncing occurred (velocities reversed after collision)
    var bounce_detected = False
    var min_x_0_after_collision: Float64 = 0.0  # Track if sphere 0 moved left
    var max_x_1_after_collision: Float64 = 0.0  # Track if sphere 1 moved right

    for i in range(num_steps):
        step_multi_body(model, data)

        # Check if collision happened (contact count > 0 for sphere-sphere)
        if data.num_contacts > 0:
            for c in range(data.num_contacts):
                if (
                    data.contacts[c].body_a >= 0
                    and data.contacts[c].body_b >= 0
                ):
                    if not collision_detected:
                        collision_detected = True
                        var t = Float64(i) * dt
                        print("  Collision detected at t =", t, "s")
                        print("  Sphere 0 x =", data.positions[0 * 3 + 0])
                        print("  Sphere 1 x =", data.positions[1 * 3 + 0])
                        min_x_0_after_collision = data.positions[0 * 3 + 0]
                        max_x_1_after_collision = data.positions[1 * 3 + 0]

        # After collision, track if spheres moved apart (bounce)
        if collision_detected:
            var x0 = data.positions[0 * 3 + 0]
            var x1 = data.positions[1 * 3 + 0]
            if x0 < min_x_0_after_collision:
                min_x_0_after_collision = x0
            if x1 > max_x_1_after_collision:
                max_x_1_after_collision = x1

            # Check if bounce occurred (sphere 0 moved left, sphere 1 moved right)
            if (
                data.velocities[0 * 3 + 0] < -0.1
                or data.velocities[1 * 3 + 0] > 0.1
            ):
                bounce_detected = True

    # Get final positions
    var final_x_0 = data.positions[0 * 3 + 0]
    var final_x_1 = data.positions[1 * 3 + 0]

    print("\nResults:")
    print(
        "  Sphere 0: final x =",
        final_x_0,
        ", min x after collision =",
        min_x_0_after_collision,
    )
    print(
        "  Sphere 1: final x =",
        final_x_1,
        ", max x after collision =",
        max_x_1_after_collision,
    )

    # Pass criteria:
    # 1. Collision was detected
    # 2. Spheres bounced apart (either velocity reversed briefly, or positions moved apart)
    var spheres_separated = (final_x_1 - final_x_0) > 0.15 or bounce_detected
    var passed = collision_detected and spheres_separated

    print()
    if passed:
        print("PASSED: Spheres collided and bounced apart")
    else:
        if not collision_detected:
            print("FAILED: No collision detected")
        if not spheres_separated:
            print("FAILED: Spheres did not separate after collision")

    print("=" * 60)
    return passed


fn test_sphere_stack() -> Bool:
    """Test 2: Stack of 3 spheres should settle on ground.

    Setup:
    - Sphere 0: z=0.1 (on ground)
    - Sphere 1: z=0.3 (on sphere 0)
    - Sphere 2: z=0.5 (on sphere 1)
    - All radius=0.1

    Expected:
    - All settle without sinking through ground or each other
    - Max drift < 2mm
    """
    print("\n")
    print("=" * 60)
    print("Phase 3 Test 2: Sphere Stack")
    print("=" * 60)

    comptime NUM_BODIES = 3
    comptime MAX_CONTACTS = 10
    comptime DTYPE = DType.float64

    var radius: Float64 = 0.1
    # Initial positions for stacked spheres touching each other
    var z0: Float64 = radius  # On ground
    var z1: Float64 = 3.0 * radius  # On sphere 0
    var z2: Float64 = 5.0 * radius  # On sphere 1

    var model = MultiBodyModel[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81, timestep=0.001, ground_z=0.0, restitution=0.0
    )
    model.set_body(0, mass=1.0, radius=radius)
    model.set_body(1, mass=1.0, radius=radius)
    model.set_body(2, mass=1.0, radius=radius)

    var data = MultiBodyData[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    data.set_body_position(0, 0, 0, z0)
    data.set_body_position(1, 0, 0, z1)
    data.set_body_position(2, 0, 0, z2)

    print("\nSetup:")
    print("  Sphere 0: z =", z0, "m (on ground)")
    print("  Sphere 1: z =", z1, "m (on sphere 0)")
    print("  Sphere 2: z =", z2, "m (on sphere 1)")

    # Simulate for 1 second
    var max_time: Float64 = 1.0
    var dt: Float64 = 0.001
    var num_steps = Int(max_time / dt)

    # Track min/max z for each sphere
    var min_z = List[Float64]()
    var max_z = List[Float64]()
    min_z.append(z0)
    min_z.append(z1)
    min_z.append(z2)
    max_z.append(z0)
    max_z.append(z1)
    max_z.append(z2)

    for _ in range(num_steps):
        step_multi_body(model, data)

        for b in range(NUM_BODIES):
            var z = data.get_body_z(b)
            if z < min_z[b]:
                min_z[b] = z
            if z > max_z[b]:
                max_z[b] = z

    print("\nResults:")
    var max_drift: Float64 = 0.0
    for b in range(NUM_BODIES):
        var final_z = data.get_body_z(b)
        var drift = max_z[b] - min_z[b]
        max_drift = max_val(max_drift, drift)
        print(
            "  Sphere",
            b,
            ": final z =",
            final_z,
            "m, drift =",
            drift * 1000,
            "mm",
        )

    print("  Max drift:", max_drift * 1000, "mm")

    # Pass criteria:
    # 1. Bottom sphere didn't sink below ground (z0 >= radius - tolerance)
    # 2. Max drift < 2mm
    var no_sink = data.get_body_z(0) >= radius - 0.01
    var low_drift = max_drift < 0.002

    var passed = no_sink and low_drift

    print()
    if passed:
        print("PASSED: Spheres stacked and settled")
    else:
        if not no_sink:
            print("FAILED: Bottom sphere sank through ground")
        if not low_drift:
            print("FAILED: Drift exceeds 2mm")

    print("=" * 60)
    return passed


fn test_sphere_fall_on_sphere() -> Bool:
    """Test 3: Sphere falling onto stationary sphere.

    Setup:
    - Sphere 0: stationary at z=0.1 (on ground)
    - Sphere 1: falling from z=1.0

    Expected:
    - Collision between spheres
    - Both spheres separate after collision
    """
    print("\n")
    print("=" * 60)
    print("Phase 3 Test 3: Sphere Fall on Sphere")
    print("=" * 60)

    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10
    comptime DTYPE = DType.float64

    var radius: Float64 = 0.1

    var model = MultiBodyModel[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81, timestep=0.001, ground_z=0.0, restitution=0.5
    )
    model.set_body(0, mass=1.0, radius=radius)
    model.set_body(1, mass=1.0, radius=radius)

    var data = MultiBodyData[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    data.set_body_position(0, 0, 0, radius)  # On ground
    data.set_body_position(1, 0, 0, 1.0)  # Falling from 1m

    print("\nSetup:")
    print("  Sphere 0: stationary at z =", radius, "m")
    print("  Sphere 1: falling from z = 1.0 m")
    print("  Restitution:", 0.5)

    var max_time: Float64 = 2.0
    var dt: Float64 = 0.001
    var num_steps = Int(max_time / dt)

    var sphere_collision_detected = False
    var sphere1_bounced = False
    var max_z_after_collision: Float64 = 0.0

    for i in range(num_steps):
        step_multi_body(model, data)

        # Check for sphere-sphere collision
        for c in range(data.num_contacts):
            if data.contacts[c].body_a >= 0 and data.contacts[c].body_b >= 0:
                if not sphere_collision_detected:
                    sphere_collision_detected = True
                    var t = Float64(i) * dt
                    print("  Sphere-sphere collision at t =", t, "s")

        # After collision, track max height of sphere 1
        if sphere_collision_detected:
            var z1 = data.get_body_z(1)
            if z1 > max_z_after_collision:
                max_z_after_collision = z1
            # Check if sphere 1 is moving up (bounced)
            if data.velocities[1 * 3 + 2] > 0.1:
                sphere1_bounced = True

    print("\nResults:")
    print("  Sphere 0 final z:", data.get_body_z(0), "m")
    print("  Sphere 1 final z:", data.get_body_z(1), "m")
    print("  Max z of sphere 1 after collision:", max_z_after_collision, "m")

    # Pass criteria:
    # 1. Sphere-sphere collision detected
    # 2. Sphere 1 bounced up after collision
    var passed = sphere_collision_detected and sphere1_bounced

    print()
    if passed:
        print("PASSED: Sphere fell and bounced off stationary sphere")
    else:
        if not sphere_collision_detected:
            print("FAILED: No sphere-sphere collision detected")
        if not sphere1_bounced:
            print("FAILED: Falling sphere did not bounce")

    print("=" * 60)
    return passed


fn main():
    """Run all Phase 3 tests."""
    print("\n")
    print("=" * 60)
    print("         PHYSICS3D v2 - Phase 3 Validation Suite           ")
    print("=" * 60)
    print()

    var all_passed = True

    # Test 1: Two spheres collision
    if not test_two_spheres_collide():
        all_passed = False

    # Test 2: Sphere stack
    if not test_sphere_stack():
        all_passed = False

    # Test 3: Sphere fall on sphere
    if not test_sphere_fall_on_sphere():
        all_passed = False

    print("\n")
    print("=" * 60)
    if all_passed:
        print("              ALL PHASE 3 TESTS PASSED                     ")
    else:
        print("              SOME PHASE 3 TESTS FAILED                    ")
    print("=" * 60)
    print()
