"""Phase 3 Validation: Multi-body tests (MuJoCo-style constraint solver).

Tests the constraint-based solver (v2) for multi-body physics.

Run with:
    pixi run mojo run physics3d_v2/tests/test_multi_body_v2.mojo
"""

from physics3d_v2.types import MultiBodyModel, MultiBodyData
from physics3d_v2.multi_body_step_v2 import step_multi_body_v2


fn abs_val(x: Float64) -> Float64:
    if x < 0:
        return -x
    return x


fn max_val(a: Float64, b: Float64) -> Float64:
    if a > b:
        return a
    return b


fn test_two_spheres_collide() -> Bool:
    """Test 1: Two spheres approaching each other should bounce."""
    print("=" * 60)
    print("Phase 3 Test 1: Two Spheres Collision (Constraint Solver)")
    print("=" * 60)

    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10
    comptime DTYPE = DType.float64

    var model = MultiBodyModel[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81, timestep=0.001, ground_z=0.0, restitution=0.8
    )
    model.set_body(0, mass=1.0, radius=0.1)
    model.set_body(1, mass=1.0, radius=0.1)

    var data = MultiBodyData[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    data.set_body_position(0, -0.5, 0, 0.11)
    data.set_body_position(1, 0.5, 0, 0.11)
    data.set_body_velocity(0, 1.0, 0, 0)
    data.set_body_velocity(1, -1.0, 0, 0)

    print("\nSetup:")
    print("  Sphere 0: x=-0.5, vx=+1.0 m/s")
    print("  Sphere 1: x=+0.5, vx=-1.0 m/s")

    var collision_detected = False
    var bounce_detected = False
    var max_time: Float64 = 1.0
    var dt: Float64 = 0.001
    var num_steps = Int(max_time / dt)

    for i in range(num_steps):
        step_multi_body_v2(model, data)

        if data.num_contacts > 0:
            for c in range(data.num_contacts):
                if data.contacts[c].body_a >= 0 and data.contacts[c].body_b >= 0:
                    if not collision_detected:
                        collision_detected = True
                        var t = Float64(i) * dt
                        print("  Collision at t =", t, "s")

        if collision_detected:
            if data.velocities[0 * 3 + 0] < -0.1 or data.velocities[1 * 3 + 0] > 0.1:
                bounce_detected = True

    var final_x_0 = data.positions[0 * 3 + 0]
    var final_x_1 = data.positions[1 * 3 + 0]
    var separation = final_x_1 - final_x_0

    print("\nResults:")
    print("  Sphere 0 final x:", final_x_0)
    print("  Sphere 1 final x:", final_x_1)
    print("  Separation:", separation, "m")

    var passed = collision_detected and (separation > 0.15 or bounce_detected)
    print()
    if passed:
        print("PASSED")
    else:
        print("FAILED")
    print("=" * 60)
    return passed


fn test_sphere_stack() -> Bool:
    """Test 2: Stack of 3 spheres should settle stably."""
    print("\n")
    print("=" * 60)
    print("Phase 3 Test 2: Sphere Stack (Constraint Solver)")
    print("=" * 60)

    comptime NUM_BODIES = 3
    comptime MAX_CONTACTS = 10
    comptime DTYPE = DType.float64

    var radius: Float64 = 0.1
    var z0: Float64 = radius
    var z1: Float64 = 3.0 * radius
    var z2: Float64 = 5.0 * radius

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
    print("  Sphere 0: z =", z0, "m")
    print("  Sphere 1: z =", z1, "m")
    print("  Sphere 2: z =", z2, "m")

    var max_time: Float64 = 1.0
    var dt: Float64 = 0.001
    var num_steps = Int(max_time / dt)

    var min_z = List[Float64]()
    var max_z = List[Float64]()
    min_z.append(z0)
    min_z.append(z1)
    min_z.append(z2)
    max_z.append(z0)
    max_z.append(z1)
    max_z.append(z2)

    for _ in range(num_steps):
        step_multi_body_v2(model, data)

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
        print("  Sphere", b, ": final z =", final_z, "m, drift =", drift * 1000, "mm")

    print("  Max drift:", max_drift * 1000, "mm")

    var no_sink = data.get_body_z(0) >= radius - 0.01
    var low_drift = max_drift < 0.002
    var passed = no_sink and low_drift

    print()
    if passed:
        print("PASSED")
    else:
        if not no_sink:
            print("FAILED: Bottom sphere sank")
        if not low_drift:
            print("FAILED: Drift exceeds 2mm")
    print("=" * 60)
    return passed


fn test_sphere_fall_on_sphere() -> Bool:
    """Test 3: Sphere falling onto stationary sphere should bounce."""
    print("\n")
    print("=" * 60)
    print("Phase 3 Test 3: Sphere Fall on Sphere (Constraint Solver)")
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
    data.set_body_position(0, 0, 0, radius)
    data.set_body_position(1, 0, 0, 1.0)

    print("\nSetup:")
    print("  Sphere 0: at z =", radius, "m (on ground)")
    print("  Sphere 1: falling from z = 1.0 m")

    var max_time: Float64 = 2.0
    var dt: Float64 = 0.001
    var num_steps = Int(max_time / dt)

    var collision_detected = False
    var bounce_detected = False
    var max_z_after_collision: Float64 = 0.0

    for i in range(num_steps):
        step_multi_body_v2(model, data)

        for c in range(data.num_contacts):
            if data.contacts[c].body_a >= 0 and data.contacts[c].body_b >= 0:
                if not collision_detected:
                    collision_detected = True
                    var t = Float64(i) * dt
                    print("  Collision at t =", t, "s")

        if collision_detected:
            var z1 = data.get_body_z(1)
            if z1 > max_z_after_collision:
                max_z_after_collision = z1
            if data.velocities[1 * 3 + 2] > 0.1:
                bounce_detected = True

    print("\nResults:")
    print("  Sphere 0 final z:", data.get_body_z(0), "m")
    print("  Sphere 1 final z:", data.get_body_z(1), "m")
    print("  Max z of sphere 1 after collision:", max_z_after_collision, "m")

    var passed = collision_detected and bounce_detected

    print()
    if passed:
        print("PASSED")
    else:
        if not collision_detected:
            print("FAILED: No collision")
        if not bounce_detected:
            print("FAILED: No bounce")
    print("=" * 60)
    return passed


fn main():
    print("\n")
    print("=" * 60)
    print("   PHYSICS3D v2 - Constraint Solver Tests (MuJoCo-style)   ")
    print("=" * 60)
    print()

    var all_passed = True

    if not test_two_spheres_collide():
        all_passed = False

    if not test_sphere_stack():
        all_passed = False

    if not test_sphere_fall_on_sphere():
        all_passed = False

    print("\n")
    print("=" * 60)
    if all_passed:
        print("              ALL TESTS PASSED                            ")
    else:
        print("              SOME TESTS FAILED                           ")
    print("=" * 60)
    print()
