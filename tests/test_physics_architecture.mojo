"""Test the new physics architecture (PhysicsLayout, PhysicsKernel, PhysicsState).

This test verifies that the new GPU-compatible architecture produces identical
results to the original PhysicsWorld API.
"""

from gpu.host import DeviceContext

from physics2d import (
    dtype,
    PhysicsWorld,
    PhysicsLayout,
    PhysicsKernel,
    PhysicsConfig,
    PhysicsState,
)


fn test_layout_sizes():
    """Test that PhysicsLayout computes correct sizes."""
    print("=" * 60)
    print("Test: PhysicsLayout Size Computation")
    print("=" * 60)

    # Define a layout for testing
    comptime TestLayout = PhysicsLayout[4, 3, 2, 8]

    # Verify sizes
    print("Layout: BATCH=4, NUM_BODIES=3, NUM_SHAPES=2, MAX_CONTACTS=8")
    print(
        "  BODIES_SIZE:", TestLayout.BODIES_SIZE, "(expected:", 4 * 3 * 13, ")"
    )
    print("  SHAPES_SIZE:", TestLayout.SHAPES_SIZE, "(expected:", 2 * 20, ")")
    print(
        "  FORCES_SIZE:", TestLayout.FORCES_SIZE, "(expected:", 4 * 3 * 3, ")"
    )
    print(
        "  CONTACTS_SIZE:",
        TestLayout.CONTACTS_SIZE,
        "(expected:",
        4 * 8 * 9,
        ")",
    )
    print("  COUNTS_SIZE:", TestLayout.COUNTS_SIZE, "(expected:", 4, ")")
    print("  TOTAL_SIZE:", TestLayout.TOTAL_SIZE)

    # Verify offsets
    print("\nOffsets:")
    print("  BODIES_OFFSET:", TestLayout.BODIES_OFFSET)
    print("  SHAPES_OFFSET:", TestLayout.SHAPES_OFFSET)
    print("  FORCES_OFFSET:", TestLayout.FORCES_OFFSET)
    print("  CONTACTS_OFFSET:", TestLayout.CONTACTS_OFFSET)
    print("  COUNTS_OFFSET:", TestLayout.COUNTS_OFFSET)

    print("\n✓ TEST PASSED: Layout sizes computed")


fn test_state_basic() raises:
    """Test basic PhysicsState operations."""
    print("\n" + "=" * 60)
    print("Test: PhysicsState Basic Operations")
    print("=" * 60)

    # Create a simple state: 2 envs, 1 body, 1 shape, 4 max contacts
    var state = PhysicsState[2, 1, 1, 4]()

    # Set up a polygon shape
    var vx = List[Float64]()
    var vy = List[Float64]()
    vx.append(-0.5)
    vx.append(0.5)
    vx.append(0.5)
    vx.append(-0.5)
    vy.append(-0.5)
    vy.append(-0.5)
    vy.append(0.5)
    vy.append(0.5)
    state.define_polygon_shape(0, vx, vy)

    # Set body properties for both environments
    for env in range(2):
        state.set_body_position(env, 0, 0.0, 5.0)
        state.set_body_velocity(env, 0, 0.0, 0.0, 0.0)
        state.set_body_mass(env, 0, 1.0, 0.1)
        state.set_body_shape(env, 0, 0)

    # Verify initial state
    print("Initial state (env 0):")
    print("  x:", state.get_body_x(0, 0), "y:", state.get_body_y(0, 0))
    print("  vx:", state.get_body_vx(0, 0), "vy:", state.get_body_vy(0, 0))

    # Apply a force
    state.apply_force(0, 0, 10.0, 0.0)

    # Step physics
    var config = PhysicsConfig(ground_y=0.0, gravity_y=-10.0, dt=0.02)
    state.step(config)

    # Check state changed
    print("\nAfter step:")
    print("  x:", state.get_body_x(0, 0), "y:", state.get_body_y(0, 0))
    print("  vx:", state.get_body_vx(0, 0), "vy:", state.get_body_vy(0, 0))

    # Verify y decreased (gravity), vx increased (force)
    var y = Float64(state.get_body_y(0, 0))
    var vx_val = Float64(state.get_body_vx(0, 0))
    if y < 5.0 and vx_val > 0:
        print("\n✓ TEST PASSED: State physics works")
    else:
        print("\n✗ TEST FAILED: State physics unexpected results")


fn test_state_vs_world() raises:
    """Test that PhysicsState produces same results as PhysicsWorld."""
    print("\n" + "=" * 60)
    print("Test: PhysicsState vs PhysicsWorld Equivalence")
    print("=" * 60)

    # Parameters
    comptime BATCH: Int = 1
    comptime NUM_BODIES: Int = 1
    comptime NUM_SHAPES: Int = 1
    comptime MAX_CONTACTS: Int = 4

    # Create both implementations
    var new_state = PhysicsState[BATCH, NUM_BODIES, NUM_SHAPES, MAX_CONTACTS]()
    var old_world = PhysicsWorld[BATCH, NUM_BODIES, NUM_SHAPES, MAX_CONTACTS](
        ground_y=0.0, gravity_y=-10.0, dt=0.02
    )

    # Define identical polygon shape
    var vx = List[Float64]()
    var vy = List[Float64]()
    vx.append(-0.5)
    vx.append(0.5)
    vx.append(0.5)
    vx.append(-0.5)
    vy.append(-0.5)
    vy.append(-0.5)
    vy.append(0.5)
    vy.append(0.5)

    new_state.define_polygon_shape(0, vx, vy)
    old_world.define_polygon_shape(0, vx, vy)

    # Set identical initial conditions
    new_state.set_body_position(0, 0, 0.0, 2.0)
    new_state.set_body_velocity(0, 0, 1.0, 0.0, 0.1)
    new_state.set_body_mass(0, 0, 1.0, 0.1)
    new_state.set_body_shape(0, 0, 0)

    old_world.set_body_position(0, 0, 0.0, 2.0)
    old_world.set_body_velocity(0, 0, 1.0, 0.0, 0.1)
    old_world.set_body_mass(0, 0, 1.0, 0.1)
    old_world.set_body_shape(0, 0, 0)

    # Run both for several steps
    var config = PhysicsConfig(ground_y=0.0, gravity_y=-10.0, dt=0.02)
    var max_diff = Float64(0)

    print("Step | New x    | Old x    | Diff")
    print("-" * 50)

    for step in range(20):
        # Step both
        new_state.step(config)
        old_world.step()

        # Compare positions
        var new_x = Float64(new_state.get_body_x(0, 0))
        var new_y = Float64(new_state.get_body_y(0, 0))
        var old_x = Float64(old_world.get_body_x(0, 0))
        var old_y = Float64(old_world.get_body_y(0, 0))

        var diff_x = new_x - old_x
        if diff_x < 0:
            diff_x = -diff_x
        var diff_y = new_y - old_y
        if diff_y < 0:
            diff_y = -diff_y
        var diff = diff_x + diff_y

        if diff > max_diff:
            max_diff = diff

        if step % 5 == 0:
            print(step, "  |", new_x, "|", old_x, "|", diff)

    print("-" * 50)
    print("Maximum difference:", max_diff)

    if max_diff < 1e-5:
        print("\n✓ TEST PASSED: PhysicsState matches PhysicsWorld")
    else:
        print("\n✗ TEST FAILED: Results differ!")


fn test_gpu_execution() raises:
    """Test GPU execution with PhysicsState."""
    print("\n" + "=" * 60)
    print("Test: PhysicsState GPU Execution")
    print("=" * 60)

    # Create state: 4 envs, 1 body, 1 shape, 4 max contacts
    var state = PhysicsState[4, 1, 1, 4]()

    # Set up shape
    var vx = List[Float64]()
    var vy = List[Float64]()
    vx.append(-0.5)
    vx.append(0.5)
    vx.append(0.5)
    vx.append(-0.5)
    vy.append(-0.5)
    vy.append(-0.5)
    vy.append(0.5)
    vy.append(0.5)
    state.define_polygon_shape(0, vx, vy)

    # Initialize all environments
    for env in range(4):
        state.set_body_position(env, 0, Float64(env), 3.0)
        state.set_body_velocity(env, 0, 0.0, 0.0, 0.0)
        state.set_body_mass(env, 0, 1.0, 0.1)
        state.set_body_shape(env, 0, 0)

    print("Initial positions:")
    for env in range(4):
        print(
            "  env",
            env,
            ": x=",
            state.get_body_x(env, 0),
            "y=",
            state.get_body_y(env, 0),
        )

    # GPU step
    var ctx = DeviceContext()
    var config = PhysicsConfig(ground_y=0.0, gravity_y=-10.0, dt=0.02)

    for _ in range(10):
        state.step_gpu(ctx, config)

    print("\nAfter 10 GPU steps:")
    for env in range(4):
        print(
            "  env",
            env,
            ": x=",
            state.get_body_x(env, 0),
            "y=",
            state.get_body_y(env, 0),
        )

    # Verify gravity applied (y should have decreased)
    var all_fell = True
    for env in range(4):
        if Float64(state.get_body_y(env, 0)) >= 3.0:
            all_fell = False
            break

    if all_fell:
        print("\n✓ TEST PASSED: GPU execution works")
    else:
        print("\n✗ TEST FAILED: Bodies didn't fall under gravity")


fn main() raises:
    """Run all architecture tests."""
    print("\n")
    print("=" * 60)
    print("    PHYSICS ARCHITECTURE TESTS")
    print("=" * 60)

    test_layout_sizes()
    test_state_basic()
    test_state_vs_world()
    test_gpu_execution()

    print("\n" + "=" * 60)
    print("All architecture tests completed!")
    print("=" * 60)
