"""Phase 6 Validation: Friction tests (GPU).

Tests Coulomb friction implementation on GPU and validates CPU/GPU parity.

Test 1: Sphere sliding to stop (GPU)
Test 2: Zero friction - free sliding (GPU)
Test 3: CPU vs GPU friction comparison

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/test_friction_gpu.mojo
"""

from math import sqrt
from gpu.host import DeviceContext

from physics3d.types import Model, Data
from physics3d.integrator import ImpulseIntegrator
from physics3d.gpu import (
    init_state_host_buffer,
    init_model_host_buffer,
    set_body_position,
    set_body_velocity,
    get_body_position,
    get_body_velocity,
    get_body_z,
    get_body_vz,
    compute_state_size,
)


fn abs_val(x: Float64) -> Float64:
    """Return absolute value."""
    if x < 0:
        return -x
    return x


fn test_sphere_sliding_to_stop_gpu() raises -> Bool:
    """Test 1: Sphere with initial horizontal velocity stops due to friction on GPU.

    Setup:
    - Sphere on ground with initial horizontal velocity (1, 0, 0) m/s
    - Friction coefficient = 0.5
    - Gravity = -9.81 m/s²

    Expected:
    - Sphere should stop within ~0.2s
    """
    print("=" * 60)
    print("GPU Test 1: Sphere Sliding to Stop")
    print("=" * 60)

    comptime DTYPE = DType.float32
    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 4
    comptime BATCH = 1

    var ctx = DeviceContext()

    # Physics parameters
    var dt = Scalar[DTYPE](0.001)
    var gravity_z = Scalar[DTYPE](-9.81)
    var ground_z = Scalar[DTYPE](0.0)
    var restitution = Scalar[DTYPE](0.0)
    var friction = Scalar[DTYPE](0.5)

    # Create model
    var model = init_model_host_buffer[DTYPE, NUM_BODIES, MAX_CONTACTS](ctx)
    var mass: Float32 = 1.0
    var radius: Float32 = 0.1
    var inertia = 0.4 * mass * radius * radius
    model[0] = mass
    model[1] = 1.0 / mass
    model[2] = radius
    model[3] = inertia
    model[4] = inertia
    model[5] = inertia
    model[6] = 1.0 / inertia
    model[7] = 1.0 / inertia
    model[8] = 1.0 / inertia

    # Initialize state
    var host_state = init_state_host_buffer[
        DTYPE, NUM_BODIES, MAX_CONTACTS, BATCH
    ](ctx)
    set_body_position[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0, x=0.0, y=0.0, z=0.1
    )
    set_body_velocity[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0, vx=1.0, vy=0.0, vz=0.0
    )

    # Copy to device
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](NUM_BODIES * 9)
    ctx.enqueue_copy(state_buf, host_state)
    ctx.enqueue_copy(model_buf, model)
    ctx.synchronize()

    print("\nSetup:")
    print("  Initial velocity: (1.0, 0.0, 0.0) m/s")
    print("  Friction coefficient: 0.5")

    # Simulate 500 steps (0.5 seconds)
    var num_steps = 500
    ImpulseIntegrator.simulate_gpu[DTYPE, NUM_BODIES, MAX_CONTACTS, 0, BATCH](
        ctx,
        state_buf,
        model_buf,
        num_steps,
        dt,
        gravity_z,
        ground_z,
        restitution,
        friction,
    )
    ctx.synchronize()

    # Copy back
    ctx.enqueue_copy(host_state, state_buf)
    ctx.synchronize()

    # Get results
    var vel = get_body_velocity[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0
    )
    var final_vx = Float64(vel[0])
    var final_vy = Float64(vel[1])
    var final_speed = sqrt(final_vx * final_vx + final_vy * final_vy)

    print("\nResults:")
    print("  Final velocity: (", vel[0], ",", vel[1], ",", vel[2], ") m/s")
    print("  Final horizontal speed:", final_speed, "m/s")

    # Pass criteria: sphere stopped (speed < 0.1 m/s)
    var passed = final_speed < 0.1

    print()
    if passed:
        print("PASSED: Sphere stopped due to friction")
    else:
        print("FAILED: Sphere did not stop")

    print("=" * 60)
    return passed


fn test_zero_friction_gpu() raises -> Bool:
    """Test 2: Sphere slides freely with friction=0 on GPU.

    Setup:
    - Sphere on ground with initial horizontal velocity (2, 0, 0) m/s
    - Friction coefficient = 0

    Expected:
    - Sphere maintains horizontal velocity
    """
    print("\n")
    print("=" * 60)
    print("GPU Test 2: Zero Friction (Free Sliding)")
    print("=" * 60)

    comptime DTYPE = DType.float32
    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 4
    comptime BATCH = 1

    var ctx = DeviceContext()

    # Physics parameters
    var dt = Scalar[DTYPE](0.001)
    var gravity_z = Scalar[DTYPE](-9.81)
    var ground_z = Scalar[DTYPE](0.0)
    var restitution = Scalar[DTYPE](0.0)
    var friction = Scalar[DTYPE](0.0)  # No friction

    # Create model
    var model = init_model_host_buffer[DTYPE, NUM_BODIES, MAX_CONTACTS](ctx)
    var mass: Float32 = 1.0
    var radius: Float32 = 0.1
    var inertia = 0.4 * mass * radius * radius
    model[0] = mass
    model[1] = 1.0 / mass
    model[2] = radius
    model[3] = inertia
    model[4] = inertia
    model[5] = inertia
    model[6] = 1.0 / inertia
    model[7] = 1.0 / inertia
    model[8] = 1.0 / inertia

    var initial_vx: Float32 = 2.0

    # Initialize state
    var host_state = init_state_host_buffer[
        DTYPE, NUM_BODIES, MAX_CONTACTS, BATCH
    ](ctx)
    set_body_position[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0, x=0.0, y=0.0, z=0.1
    )
    set_body_velocity[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0, vx=initial_vx, vy=0.0, vz=0.0
    )

    # Copy to device
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](NUM_BODIES * 9)
    ctx.enqueue_copy(state_buf, host_state)
    ctx.enqueue_copy(model_buf, model)
    ctx.synchronize()

    print("\nSetup:")
    print("  Initial velocity: (2.0, 0.0, 0.0) m/s")
    print("  Friction coefficient: 0.0")

    # Simulate 500 steps (0.5 seconds)
    var num_steps = 500
    ImpulseIntegrator.simulate_gpu[DTYPE, NUM_BODIES, MAX_CONTACTS, 0, BATCH](
        ctx,
        state_buf,
        model_buf,
        num_steps,
        dt,
        gravity_z,
        ground_z,
        restitution,
        friction,
    )
    ctx.synchronize()

    # Copy back
    ctx.enqueue_copy(host_state, state_buf)
    ctx.synchronize()

    # Get results
    var vel = get_body_velocity[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0
    )
    var final_vx = Float64(vel[0])

    print("\nResults:")
    print("  Final x-velocity:", final_vx, "m/s")
    print("  Expected:", initial_vx, "m/s")

    # Pass criteria: velocity maintained (within 5%)
    var velocity_maintained = abs_val(final_vx - Float64(initial_vx)) < 0.1

    var passed = velocity_maintained

    print()
    if passed:
        print("PASSED: Sphere slides freely without friction")
    else:
        print("FAILED: Velocity changed from", initial_vx, "to", final_vx)

    print("=" * 60)
    return passed


fn test_cpu_gpu_friction_comparison() raises -> Bool:
    """Test 3: Compare CPU and GPU friction behavior.

    Setup:
    - Sphere on ground with initial horizontal velocity (1, 0, 0) m/s
    - Friction coefficient = 0.5
    - Compare final velocities between CPU and GPU

    Expected:
    - Both should stop at similar times with similar final states
    """
    print("\n")
    print("=" * 60)
    print("GPU Test 3: CPU vs GPU Friction Comparison")
    print("=" * 60)

    comptime DTYPE_CPU = DType.float64
    comptime DTYPE_GPU = DType.float32
    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 4
    comptime BATCH = 1

    var initial_vx: Float64 = 1.0
    var friction_coef: Float64 = 0.5
    var num_steps = 500
    var dt: Float64 = 0.001

    # --- CPU Simulation ---
    var model_cpu = Model[DTYPE_CPU, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=dt,
        ground_z=0.0,
        restitution=0.0,
        friction=friction_coef,
    )
    model_cpu.set_body(0, mass=1.0, radius=0.1)

    var data_cpu = Data[DTYPE_CPU, NUM_BODIES, MAX_CONTACTS]()
    data_cpu.set_body_position(0, 0.0, 0.0, 0.1)
    data_cpu.set_body_velocity(0, initial_vx, 0.0, 0.0)

    ImpulseIntegrator.simulate(model_cpu, data_cpu, num_steps)

    var cpu_vel = data_cpu.get_body_velocity(0)
    var cpu_pos = data_cpu.get_body_position(0)

    # --- GPU Simulation ---
    var ctx = DeviceContext()

    var mass: Float32 = 1.0
    var radius: Float32 = 0.1
    var inertia = 0.4 * mass * radius * radius

    var model_gpu = init_model_host_buffer[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
        ctx
    )
    model_gpu[0] = mass
    model_gpu[1] = 1.0 / mass
    model_gpu[2] = radius
    model_gpu[3] = inertia
    model_gpu[4] = inertia
    model_gpu[5] = inertia
    model_gpu[6] = 1.0 / inertia
    model_gpu[7] = 1.0 / inertia
    model_gpu[8] = 1.0 / inertia

    var host_state = init_state_host_buffer[
        DTYPE_GPU, NUM_BODIES, MAX_CONTACTS, BATCH
    ](ctx)
    set_body_position[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0, x=0.0, y=0.0, z=0.1
    )
    set_body_velocity[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0, vx=Float32(initial_vx), vy=0.0, vz=0.0
    )

    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var state_buf = ctx.enqueue_create_buffer[DTYPE_GPU](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE_GPU](NUM_BODIES * 9)
    ctx.enqueue_copy(state_buf, host_state)
    ctx.enqueue_copy(model_buf, model_gpu)
    ctx.synchronize()

    ImpulseIntegrator.simulate_gpu[
        DTYPE_GPU, NUM_BODIES, MAX_CONTACTS, 0, BATCH
    ](
        ctx,
        state_buf,
        model_buf,
        num_steps,
        Scalar[DTYPE_GPU](dt),
        Scalar[DTYPE_GPU](-9.81),
        Scalar[DTYPE_GPU](0.0),
        Scalar[DTYPE_GPU](0.0),
        Scalar[DTYPE_GPU](friction_coef),
    )
    ctx.synchronize()

    ctx.enqueue_copy(host_state, state_buf)
    ctx.synchronize()

    var gpu_vel = get_body_velocity[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0
    )
    var gpu_pos = get_body_position[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0
    )

    # Compare
    var cpu_speed = sqrt(
        Float64(cpu_vel[0]) * Float64(cpu_vel[0])
        + Float64(cpu_vel[1]) * Float64(cpu_vel[1])
    )
    var gpu_speed = sqrt(
        Float64(gpu_vel[0]) * Float64(gpu_vel[0])
        + Float64(gpu_vel[1]) * Float64(gpu_vel[1])
    )

    print("\nSetup:")
    print("  Initial velocity: (1.0, 0.0, 0.0) m/s")
    print("  Friction coefficient:", friction_coef)
    print("  Steps:", num_steps, "at dt=", dt)

    print("\nResults after", num_steps, "steps:")
    print(
        "  CPU: pos=(",
        cpu_pos[0],
        ",",
        cpu_pos[1],
        ",",
        cpu_pos[2],
        "), speed=",
        cpu_speed,
    )
    print(
        "  GPU: pos=(",
        gpu_pos[0],
        ",",
        gpu_pos[1],
        ",",
        gpu_pos[2],
        "), speed=",
        gpu_speed,
    )

    # Both should have stopped (speed < 0.1 m/s)
    var cpu_stopped = cpu_speed < 0.1
    var gpu_stopped = gpu_speed < 0.1

    # Position difference should be small (allowing for float32 vs float64 differences)
    var pos_diff = abs_val(Float64(cpu_pos[0]) - Float64(gpu_pos[0]))

    print("  CPU stopped:", cpu_stopped)
    print("  GPU stopped:", gpu_stopped)
    print("  Position x difference:", pos_diff, "m")

    # Allow larger tolerance for position due to different precision
    var passed = cpu_stopped and gpu_stopped and pos_diff < 0.05

    print()
    if passed:
        print("PASSED: CPU and GPU friction behavior matches")
    else:
        if not cpu_stopped:
            print("FAILED: CPU did not stop")
        if not gpu_stopped:
            print("FAILED: GPU did not stop")
        if pos_diff >= 0.05:
            print("FAILED: Position difference too large")

    print("=" * 60)
    return passed


fn main() raises:
    """Run all GPU friction tests."""
    print()
    print("=" * 60)
    print("    Physics3D v2 GPU Friction Tests (Phase 6)")
    print("=" * 60)
    print()

    var passed_count = 0
    var total_count = 0

    # GPU-only friction tests
    total_count += 1
    if test_sphere_sliding_to_stop_gpu():
        passed_count += 1

    total_count += 1
    if test_zero_friction_gpu():
        passed_count += 1

    total_count += 1
    if test_cpu_gpu_friction_comparison():
        passed_count += 1

    # Summary
    print("\n")
    print("=" * 60)
    print("                    SUMMARY")
    print("=" * 60)
    print("Passed:", passed_count, "/", total_count, "tests")

    if passed_count == total_count:
        print("ALL GPU FRICTION TESTS PASSED")
    else:
        print("SOME GPU FRICTION TESTS FAILED")

    print("=" * 60)
    print()
