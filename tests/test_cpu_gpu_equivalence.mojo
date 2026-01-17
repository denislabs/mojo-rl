"""Test CPU-GPU equivalence for the physics engine.

This test verifies that the CPU and GPU physics implementations produce
identical results. This is critical for RL training where we need:
1. GPU speed for training thousands of environments
2. CPU for debugging and visualization
3. Identical physics so agents trained on GPU work on CPU

The test simulates a simple scenario (falling box hitting ground)
and compares CPU vs GPU results at each step.
"""

from builtin.math import abs
from gpu.host import DeviceContext

from physics_gpu.constants import dtype
from physics_gpu.world import PhysicsWorld


fn test_falling_box() raises:
    """Test a box falling under gravity and hitting ground.

    This tests:
    - Velocity integration (gravity)
    - Position integration
    - Ground collision detection
    - Contact resolution
    """
    print("=" * 60)
    print("Test: Falling Box (CPU vs GPU)")
    print("=" * 60)

    # Parameters
    comptime BATCH: Int = 4  # Small batch for testing
    comptime NUM_BODIES: Int = 1
    comptime NUM_SHAPES: Int = 1
    comptime NUM_STEPS: Int = 100

    # Create CPU world
    var cpu_world = PhysicsWorld[BATCH, NUM_BODIES, NUM_SHAPES](
        ground_y=0.0,
        gravity_y=-10.0,
        dt=0.02,
    )

    # Create GPU world with same parameters
    var gpu_world = PhysicsWorld[BATCH, NUM_BODIES, NUM_SHAPES](
        ground_y=0.0,
        gravity_y=-10.0,
        dt=0.02,
    )

    # Define a 1x1 box shape
    var vertices_x = List[Float64]()
    var vertices_y = List[Float64]()
    vertices_x.append(-0.5)
    vertices_y.append(-0.5)
    vertices_x.append(0.5)
    vertices_y.append(-0.5)
    vertices_x.append(0.5)
    vertices_y.append(0.5)
    vertices_x.append(-0.5)
    vertices_y.append(0.5)

    cpu_world.define_polygon_shape(0, vertices_x, vertices_y)
    gpu_world.define_polygon_shape(0, vertices_x, vertices_y)

    # Setup bodies in each environment
    for env in range(BATCH):
        # Box starts at different heights for each environment
        var start_y = 2.0 + Float64(env) * 0.5

        cpu_world.set_body_position(env, 0, 0.0, start_y)
        cpu_world.set_body_mass(env, 0, 1.0, 0.1)
        cpu_world.set_body_shape(env, 0, 0)

        gpu_world.set_body_position(env, 0, 0.0, start_y)
        gpu_world.set_body_mass(env, 0, 1.0, 0.1)
        gpu_world.set_body_shape(env, 0, 0)

    # Get GPU context
    var ctx = DeviceContext()

    # Track maximum errors
    var max_pos_error = Scalar[dtype](0)
    var max_vel_error = Scalar[dtype](0)
    var max_angle_error = Scalar[dtype](0)

    # Simulate and compare
    print("Step | Max Pos Err  | Max Vel Err  | Max Angle Err")
    print("-" * 60)

    for step in range(NUM_STEPS):
        # Run CPU step
        cpu_world.step()

        # Run GPU step
        gpu_world.step_gpu(ctx)

        # Compare results
        var step_pos_error = Scalar[dtype](0)
        var step_vel_error = Scalar[dtype](0)
        var step_angle_error = Scalar[dtype](0)

        for env in range(BATCH):
            var cpu_x = cpu_world.get_body_x(env, 0)
            var cpu_y = cpu_world.get_body_y(env, 0)
            var gpu_x = gpu_world.get_body_x(env, 0)
            var gpu_y = gpu_world.get_body_y(env, 0)
            var cpu_angle = cpu_world.get_body_angle(env, 0)
            var gpu_angle = gpu_world.get_body_angle(env, 0)
            var cpu_vx = cpu_world.get_body_vx(env, 0)
            var cpu_vy = cpu_world.get_body_vy(env, 0)
            var cpu_omega = cpu_world.get_body_omega(env, 0)
            var gpu_vx = gpu_world.get_body_vx(env, 0)
            var gpu_vy = gpu_world.get_body_vy(env, 0)
            var gpu_omega = gpu_world.get_body_omega(env, 0)

            var pos_err = abs(cpu_x - gpu_x) + abs(cpu_y - gpu_y)
            var vel_err = (
                abs(cpu_vx - gpu_vx)
                + abs(cpu_vy - gpu_vy)
                + abs(cpu_omega - gpu_omega)
            )
            var angle_err = abs(cpu_angle - gpu_angle)

            if pos_err > step_pos_error:
                step_pos_error = pos_err
            if vel_err > step_vel_error:
                step_vel_error = vel_err
            if angle_err > step_angle_error:
                step_angle_error = angle_err

        if step_pos_error > max_pos_error:
            max_pos_error = step_pos_error
        if step_vel_error > max_vel_error:
            max_vel_error = step_vel_error
        if step_angle_error > max_angle_error:
            max_angle_error = step_angle_error

        # Print every 10 steps
        if step % 10 == 0:
            print(
                step,
                "   |",
                step_pos_error,
                "|",
                step_vel_error,
                "|",
                step_angle_error,
            )

    print("-" * 60)
    print("Final Maximum Errors:")
    print("  Position:", max_pos_error)
    print("  Velocity:", max_vel_error)
    print("  Angle:   ", max_angle_error)

    # Check if errors are within tolerance
    var tolerance = Scalar[dtype](1e-5)
    if (
        max_pos_error < tolerance
        and max_vel_error < tolerance
        and max_angle_error < tolerance
    ):
        print("\n✓ TEST PASSED: CPU and GPU results match within tolerance")
    else:
        print("\n✗ TEST FAILED: CPU and GPU results differ beyond tolerance")


fn test_multiple_bodies() raises:
    """Test multiple bodies falling and colliding with ground."""
    print("\n" + "=" * 60)
    print("Test: Multiple Bodies (CPU vs GPU)")
    print("=" * 60)

    comptime BATCH: Int = 8
    comptime NUM_BODIES: Int = 3
    comptime NUM_SHAPES: Int = 2
    comptime NUM_STEPS: Int = 50

    var cpu_world = PhysicsWorld[BATCH, NUM_BODIES, NUM_SHAPES](
        ground_y=0.0,
        gravity_y=-10.0,
    )

    var gpu_world = PhysicsWorld[BATCH, NUM_BODIES, NUM_SHAPES](
        ground_y=0.0,
        gravity_y=-10.0,
    )

    # Define shapes
    var box_x = List[Float64]()
    var box_y = List[Float64]()
    box_x.append(-0.5)
    box_y.append(-0.5)
    box_x.append(0.5)
    box_y.append(-0.5)
    box_x.append(0.5)
    box_y.append(0.5)
    box_x.append(-0.5)
    box_y.append(0.5)

    cpu_world.define_polygon_shape(0, box_x, box_y)
    gpu_world.define_polygon_shape(0, box_x, box_y)

    cpu_world.define_circle_shape(1, radius=0.3)
    gpu_world.define_circle_shape(1, radius=0.3)

    # Setup bodies
    for env in range(BATCH):
        # Body 0: box at left
        cpu_world.set_body_position(env, 0, -2.0, 3.0)
        cpu_world.set_body_mass(env, 0, 1.0, 0.1)
        cpu_world.set_body_shape(env, 0, 0)
        gpu_world.set_body_position(env, 0, -2.0, 3.0)
        gpu_world.set_body_mass(env, 0, 1.0, 0.1)
        gpu_world.set_body_shape(env, 0, 0)

        # Body 1: circle in middle
        cpu_world.set_body_position(env, 1, 0.0, 2.5)
        cpu_world.set_body_mass(env, 1, 0.5, 0.05)
        cpu_world.set_body_shape(env, 1, 1)
        gpu_world.set_body_position(env, 1, 0.0, 2.5)
        gpu_world.set_body_mass(env, 1, 0.5, 0.05)
        gpu_world.set_body_shape(env, 1, 1)

        # Body 2: box at right with initial velocity
        cpu_world.set_body_position(env, 2, 2.0, 2.0)
        cpu_world.set_body_velocity(env, 2, -1.0, 0.0, 0.5)
        cpu_world.set_body_mass(env, 2, 0.8, 0.08)
        cpu_world.set_body_shape(env, 2, 0)
        gpu_world.set_body_position(env, 2, 2.0, 2.0)
        gpu_world.set_body_velocity(env, 2, -1.0, 0.0, 0.5)
        gpu_world.set_body_mass(env, 2, 0.8, 0.08)
        gpu_world.set_body_shape(env, 2, 0)

    var ctx = DeviceContext()

    var total_error = Scalar[dtype](0)

    for step in range(NUM_STEPS):
        cpu_world.step()
        gpu_world.step_gpu(ctx)

        for env in range(BATCH):
            for body in range(NUM_BODIES):
                var cpu_x = cpu_world.get_body_x(env, body)
                var cpu_y = cpu_world.get_body_y(env, body)
                var gpu_x = gpu_world.get_body_x(env, body)
                var gpu_y = gpu_world.get_body_y(env, body)
                total_error = total_error + abs(cpu_x - gpu_x)
                total_error = total_error + abs(cpu_y - gpu_y)

    var avg_error = total_error / Scalar[dtype](
        NUM_STEPS * BATCH * NUM_BODIES * 2
    )
    print("Average position error:", avg_error)

    if avg_error < Scalar[dtype](1e-5):
        print("✓ TEST PASSED")
    else:
        print("✗ TEST FAILED")


fn test_contact_detection() raises:
    """Test that collision detection produces same contacts on CPU and GPU."""
    print("\n" + "=" * 60)
    print("Test: Contact Detection (CPU vs GPU)")
    print("=" * 60)

    comptime BATCH: Int = 16
    comptime NUM_BODIES: Int = 2
    comptime NUM_SHAPES: Int = 1

    var cpu_world = PhysicsWorld[BATCH, NUM_BODIES, NUM_SHAPES](ground_y=0.0)
    var gpu_world = PhysicsWorld[BATCH, NUM_BODIES, NUM_SHAPES](ground_y=0.0)

    # Box shape
    var box_x = List[Float64]()
    var box_y = List[Float64]()
    box_x.append(-0.5)
    box_y.append(-0.5)
    box_x.append(0.5)
    box_y.append(-0.5)
    box_x.append(0.5)
    box_y.append(0.5)
    box_x.append(-0.5)
    box_y.append(0.5)

    cpu_world.define_polygon_shape(0, box_x, box_y)
    gpu_world.define_polygon_shape(0, box_x, box_y)

    # Place bodies at various heights, some touching ground
    for env in range(BATCH):
        # Body 0 at varying heights
        var y0 = Float64(env) * 0.1 + 0.2  # 0.2 to 1.7
        cpu_world.set_body_position(env, 0, 0.0, y0)
        cpu_world.set_body_mass(env, 0, 1.0, 0.1)
        cpu_world.set_body_shape(env, 0, 0)
        gpu_world.set_body_position(env, 0, 0.0, y0)
        gpu_world.set_body_mass(env, 0, 1.0, 0.1)
        gpu_world.set_body_shape(env, 0, 0)

        # Body 1 higher up
        cpu_world.set_body_position(env, 1, 2.0, 5.0)
        cpu_world.set_body_mass(env, 1, 1.0, 0.1)
        cpu_world.set_body_shape(env, 1, 0)
        gpu_world.set_body_position(env, 1, 2.0, 5.0)
        gpu_world.set_body_mass(env, 1, 1.0, 0.1)
        gpu_world.set_body_shape(env, 1, 0)

    var ctx = DeviceContext()

    # Run one step to detect contacts
    cpu_world.step()
    gpu_world.step_gpu(ctx)

    # Compare contact counts
    var contact_match = True
    for env in range(BATCH):
        var cpu_contacts = cpu_world.get_contact_count(env)
        var gpu_contacts = gpu_world.get_contact_count(env)
        if cpu_contacts != gpu_contacts:
            print(
                "Env",
                env,
                ": CPU contacts=",
                cpu_contacts,
                ", GPU contacts=",
                gpu_contacts,
            )
            contact_match = False

    if contact_match:
        print("✓ TEST PASSED: Contact counts match")
    else:
        print("✗ TEST FAILED: Contact counts differ")


fn main() raises:
    """Run all CPU-GPU equivalence tests."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║      PHYSICS ENGINE CPU-GPU EQUIVALENCE TESTS              ║")
    print("╚════════════════════════════════════════════════════════════╝")

    test_falling_box()
    test_multiple_bodies()
    test_contact_detection()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
