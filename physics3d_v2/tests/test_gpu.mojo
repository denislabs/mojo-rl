"""Test Physics3D v2 GPU implementation.

Validates that the GPU physics kernel produces correct results for
multi-body scenarios and matches CPU results:

1. Free fall (single body) - GPU only
2. Ball drop and contact (single body vs ground) - GPU only
3. Multi-body collision (sphere-sphere) - GPU only
4. Batched simulation (multiple environments) - GPU only
5. CPU vs GPU comparison: Free fall (Impulse)
6. CPU vs GPU comparison: Ball drop (Impulse)
7. CPU vs GPU comparison: Free fall (PGS)
8. CPU vs GPU comparison: Ball drop (PGS)
9. CPU vs GPU comparison: Two spheres collision (Impulse)
"""

from gpu.host import DeviceContext

from physics3d_v2.types import Model, Data
from physics3d_v2.integrator import ImpulseIntegrator, PGSIntegrator
from physics3d_v2.gpu import (
    # Buffer utilities
    init_state_host_buffer,
    init_model_host_buffer,
    set_body_position,
    set_body_velocity,
    get_body_position,
    get_body_velocity,
    get_body_z,
    get_body_vz,
    get_num_contacts,
    # Constants
    compute_state_size,
)


fn abs_val(x: Float64) -> Float64:
    """Return absolute value."""
    if x < 0:
        return -x
    return x


fn max_val(a: Float64, b: Float64) -> Float64:
    """Maximum of two values."""
    if a > b:
        return a
    return b


# =============================================================================
# GPU-Only Tests
# =============================================================================


fn test_free_fall_gpu() raises -> Bool:
    """Test free fall matches analytical solution on GPU."""
    print("=" * 60)
    print("GPU Test 1: Free Fall (Impulse Solver)")
    print("=" * 60)

    comptime DTYPE = DType.float32
    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 4
    comptime BATCH = 1

    var ctx = DeviceContext()

    # Physics parameters
    var dt = Scalar[DTYPE](0.01)
    var gravity_z = Scalar[DTYPE](-9.81)
    var ground_z = Scalar[DTYPE](0.0)
    var restitution = Scalar[DTYPE](0.0)

    # Initial height (high above ground so no contact)
    var initial_z: Float32 = 10.0

    # Create model with a single sphere
    var model = init_model_host_buffer[DTYPE, NUM_BODIES, MAX_CONTACTS](ctx)
    # Set body 0: mass=1.0, radius=0.1
    var mass: Float32 = 1.0
    var radius: Float32 = 0.1
    var inertia = 0.4 * mass * radius * radius
    model[0] = mass  # mass
    model[1] = 1.0 / mass  # inv_mass
    model[2] = radius  # radius
    model[3] = inertia  # ixx
    model[4] = inertia  # iyy
    model[5] = inertia  # izz
    model[6] = 1.0 / inertia  # inv_ixx
    model[7] = 1.0 / inertia  # inv_iyy
    model[8] = 1.0 / inertia  # inv_izz

    # Initialize state buffer
    var host_state = init_state_host_buffer[
        DTYPE, NUM_BODIES, MAX_CONTACTS, BATCH
    ](ctx)
    set_body_position[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0, x=0.0, y=0.0, z=initial_z
    )

    # Copy to device
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](NUM_BODIES * 9)
    ctx.enqueue_copy(state_buf, host_state)
    ctx.enqueue_copy(model_buf, model)
    ctx.synchronize()

    # Simulate 100 steps (1 second)
    var num_steps = 100
    ImpulseIntegrator.simulate_gpu[DTYPE, NUM_BODIES, MAX_CONTACTS, 0, BATCH](
        ctx,
        state_buf,
        model_buf,
        num_steps,
        dt,
        gravity_z,
        ground_z,
        restitution,
        Scalar[DTYPE](0.5),
    )
    ctx.synchronize()

    # Copy back to host
    ctx.enqueue_copy(host_state, state_buf)
    ctx.synchronize()

    # Get results
    var final_z = get_body_z[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0
    )
    var final_vz = get_body_vz[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0
    )

    # Analytical solution after t=1s:
    # z(t) = z0 + v0*t + 0.5*g*t^2 = 10 + 0 + 0.5*(-9.81)*1^2 = 5.095
    # vz(t) = v0 + g*t = 0 + (-9.81)*1 = -9.81
    var expected_z: Float32 = 5.095
    var expected_vz: Float32 = -9.81

    var z_error = abs_val(Float64(final_z) - Float64(expected_z))
    var vz_error = abs_val(Float64(final_vz) - Float64(expected_vz))

    print("After", num_steps, "steps (dt=0.01, t=1s):")
    print(
        "  z:  ", final_z, " (expected:", expected_z, ", error:", z_error, ")"
    )
    print(
        "  vz: ",
        final_vz,
        " (expected:",
        expected_vz,
        ", error:",
        vz_error,
        ")",
    )

    # Check tolerances
    var z_tolerance = 0.05  # ~1% error
    var vz_tolerance = 0.1

    var passed = z_error < z_tolerance and vz_error < vz_tolerance

    if passed:
        print("PASSED: Free fall within tolerance")
    else:
        print("FAILED: Error exceeds tolerance")

    print()
    return passed


fn test_ball_drop_gpu() raises -> Bool:
    """Test ball drop stops at correct height on GPU."""
    print("=" * 60)
    print("GPU Test 2: Ball Drop (Contact) - PGS Solver")
    print("=" * 60)

    comptime DTYPE = DType.float32
    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 4
    comptime BATCH = 1

    var ctx = DeviceContext()

    # Physics parameters
    var dt = Scalar[DTYPE](0.01)
    var gravity_z = Scalar[DTYPE](-9.81)
    var ground_z = Scalar[DTYPE](0.0)
    var restitution = Scalar[DTYPE](0.0)  # Inelastic

    var mass: Float32 = 1.0
    var radius: Float32 = 0.1
    var inertia = 0.4 * mass * radius * radius

    # Create model
    var model = init_model_host_buffer[DTYPE, NUM_BODIES, MAX_CONTACTS](ctx)
    model[0] = mass
    model[1] = 1.0 / mass
    model[2] = radius
    model[3] = inertia
    model[4] = inertia
    model[5] = inertia
    model[6] = 1.0 / inertia
    model[7] = 1.0 / inertia
    model[8] = 1.0 / inertia

    # Drop from 1m
    var initial_z: Float32 = 1.0

    # Initialize state buffer
    var host_state = init_state_host_buffer[
        DTYPE, NUM_BODIES, MAX_CONTACTS, BATCH
    ](ctx)
    set_body_position[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0, x=0.0, y=0.0, z=initial_z
    )

    # Copy to device
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](NUM_BODIES * 9)
    ctx.enqueue_copy(state_buf, host_state)
    ctx.enqueue_copy(model_buf, model)
    ctx.synchronize()

    # Simulate enough steps for ball to fall and settle
    var num_steps = 200
    PGSIntegrator.simulate_gpu[DTYPE, NUM_BODIES, MAX_CONTACTS, 0, BATCH](
        ctx,
        state_buf,
        model_buf,
        num_steps,
        dt,
        gravity_z,
        ground_z,
        restitution,
        Scalar[DTYPE](0.5),
    )
    ctx.synchronize()

    # Copy back to host
    ctx.enqueue_copy(host_state, state_buf)
    ctx.synchronize()

    # Get results
    var final_z = get_body_z[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0
    )
    var final_vz = get_body_vz[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0
    )
    var num_contacts = get_num_contacts[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0
    )

    # Expected: ball rests at z = radius = 0.1
    var expected_z: Float32 = 0.1
    var z_error = abs_val(Float64(final_z) - Float64(expected_z))

    print("After", num_steps, "steps:")
    print("  z:            ", final_z, " (expected:", expected_z, ")")
    print("  vz:           ", final_vz, " (expected: ~0)")
    print("  num_contacts: ", num_contacts)
    print("  z error:      ", z_error, "m")

    var z_tolerance = 0.01  # 1cm
    var vz_tolerance = 0.5

    var passed = (
        z_error < z_tolerance and abs_val(Float64(final_vz)) < vz_tolerance
    )

    if passed:
        print("PASSED: Ball settled at correct height")
    else:
        print("FAILED: Ball did not settle correctly")

    print()
    return passed


fn test_two_spheres_collision_gpu() raises -> Bool:
    """Test two spheres colliding on GPU."""
    print("=" * 60)
    print("GPU Test 3: Two Spheres Collision - Impulse Solver")
    print("=" * 60)

    comptime DTYPE = DType.float32
    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 8
    comptime BATCH = 1

    var ctx = DeviceContext()

    # Physics parameters
    var dt = Scalar[DTYPE](0.01)
    var gravity_z = Scalar[DTYPE](0.0)  # No gravity for this test
    var ground_z = Scalar[DTYPE](-10.0)  # Ground far below
    var restitution = Scalar[DTYPE](1.0)  # Elastic collision

    var mass: Float32 = 1.0
    var radius: Float32 = 0.5
    var inertia = 0.4 * mass * radius * radius

    # Create model for 2 identical spheres
    var model = init_model_host_buffer[DTYPE, NUM_BODIES, MAX_CONTACTS](ctx)
    for i in range(NUM_BODIES):
        var offset = i * 9
        model[offset + 0] = mass
        model[offset + 1] = 1.0 / mass
        model[offset + 2] = radius
        model[offset + 3] = inertia
        model[offset + 4] = inertia
        model[offset + 5] = inertia
        model[offset + 6] = 1.0 / inertia
        model[offset + 7] = 1.0 / inertia
        model[offset + 8] = 1.0 / inertia

    # Initialize state: two spheres approaching each other
    # Sphere 0: at x=-1, moving right (vx=+2)
    # Sphere 1: at x=+1, moving left (vx=-2)
    var host_state = init_state_host_buffer[
        DTYPE, NUM_BODIES, MAX_CONTACTS, BATCH
    ](ctx)
    set_body_position[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0, x=-1.0, y=0.0, z=1.0
    )
    set_body_velocity[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0, vx=2.0, vy=0.0, vz=0.0
    )
    set_body_position[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=1, x=1.0, y=0.0, z=1.0
    )
    set_body_velocity[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=1, vx=-2.0, vy=0.0, vz=0.0
    )

    # Copy to device
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](NUM_BODIES * 9)
    ctx.enqueue_copy(state_buf, host_state)
    ctx.enqueue_copy(model_buf, model)
    ctx.synchronize()

    print("Initial state:")
    var pos0 = get_body_position[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0
    )
    var vel0 = get_body_velocity[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0
    )
    var pos1 = get_body_position[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=1
    )
    var vel1 = get_body_velocity[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=1
    )
    print(
        "  Sphere 0: pos=(",
        pos0[0],
        ",",
        pos0[1],
        ",",
        pos0[2],
        "), vel=(",
        vel0[0],
        ",",
        vel0[1],
        ",",
        vel0[2],
        ")",
    )
    print(
        "  Sphere 1: pos=(",
        pos1[0],
        ",",
        pos1[1],
        ",",
        pos1[2],
        "), vel=(",
        vel1[0],
        ",",
        vel1[1],
        ",",
        vel1[2],
        ")",
    )

    # Simulate for enough time for collision to happen
    var num_steps = 100
    ImpulseIntegrator.simulate_gpu[DTYPE, NUM_BODIES, MAX_CONTACTS, 0, BATCH](
        ctx,
        state_buf,
        model_buf,
        num_steps,
        dt,
        gravity_z,
        ground_z,
        restitution,
        Scalar[DTYPE](0.5),
    )
    ctx.synchronize()

    # Copy back to host
    ctx.enqueue_copy(host_state, state_buf)
    ctx.synchronize()

    print("\nAfter", num_steps, "steps:")
    pos0 = get_body_position[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0
    )
    vel0 = get_body_velocity[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0
    )
    pos1 = get_body_position[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=1
    )
    vel1 = get_body_velocity[DTYPE, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=1
    )
    print(
        "  Sphere 0: pos=(",
        pos0[0],
        ",",
        pos0[1],
        ",",
        pos0[2],
        "), vel=(",
        vel0[0],
        ",",
        vel0[1],
        ",",
        vel0[2],
        ")",
    )
    print(
        "  Sphere 1: pos=(",
        pos1[0],
        ",",
        pos1[1],
        ",",
        pos1[2],
        "), vel=(",
        vel1[0],
        ",",
        vel1[1],
        ",",
        vel1[2],
        ")",
    )

    # After elastic collision, velocities should be swapped
    # Sphere 0 should be moving left (vx < 0)
    # Sphere 1 should be moving right (vx > 0)
    var passed = vel0[0] < 0 and vel1[0] > 0

    if passed:
        print("PASSED: Spheres bounced apart after collision")
    else:
        print("FAILED: Unexpected velocity after collision")

    print()
    return passed


fn test_batched_simulation_gpu() raises -> Bool:
    """Test multiple environments in parallel."""
    print("=" * 60)
    print("GPU Test 4: Batched Simulation (256 environments)")
    print("=" * 60)

    comptime DTYPE = DType.float32
    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 4
    comptime BATCH = 256  # Simulate 256 environments in parallel

    var ctx = DeviceContext()

    # Physics parameters
    var dt = Scalar[DTYPE](0.01)
    var gravity_z = Scalar[DTYPE](-9.81)
    var ground_z = Scalar[DTYPE](0.0)
    var restitution = Scalar[DTYPE](0.5)  # Bouncy for variety

    var mass: Float32 = 1.0
    var radius: Float32 = 0.1
    var inertia = 0.4 * mass * radius * radius

    # Create model (same for all environments)
    var model = init_model_host_buffer[DTYPE, NUM_BODIES, MAX_CONTACTS](ctx)
    model[0] = mass
    model[1] = 1.0 / mass
    model[2] = radius
    model[3] = inertia
    model[4] = inertia
    model[5] = inertia
    model[6] = 1.0 / inertia
    model[7] = 1.0 / inertia
    model[8] = 1.0 / inertia

    # Initialize state buffer with different starting heights
    var host_state = init_state_host_buffer[
        DTYPE, NUM_BODIES, MAX_CONTACTS, BATCH
    ](ctx)

    for i in range(BATCH):
        var height: Float32 = 0.5 + Float32(i) * 0.01  # 0.5 to ~3m
        set_body_position[DTYPE, NUM_BODIES, MAX_CONTACTS](
            host_state, env=i, body=0, x=0.0, y=0.0, z=height
        )

    # Copy to device
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](NUM_BODIES * 9)
    ctx.enqueue_copy(state_buf, host_state)
    ctx.enqueue_copy(model_buf, model)
    ctx.synchronize()

    # Simulate
    var num_steps = 100
    ImpulseIntegrator.simulate_gpu[DTYPE, NUM_BODIES, MAX_CONTACTS, 0, BATCH](
        ctx,
        state_buf,
        model_buf,
        num_steps,
        dt,
        gravity_z,
        ground_z,
        restitution,
        Scalar[DTYPE](0.5),
    )
    ctx.synchronize()

    # Copy back to host
    ctx.enqueue_copy(host_state, state_buf)
    ctx.synchronize()

    # Check a few sample environments
    print("Sample results after", num_steps, "steps:")
    var sample_indices = List[Int]()
    sample_indices.append(0)
    sample_indices.append(64)
    sample_indices.append(128)
    sample_indices.append(255)
    for idx in range(len(sample_indices)):
        var i = sample_indices[idx]
        var z = get_body_z[DTYPE, NUM_BODIES, MAX_CONTACTS](
            host_state, env=i, body=0
        )
        var vz = get_body_vz[DTYPE, NUM_BODIES, MAX_CONTACTS](
            host_state, env=i, body=0
        )
        print("  Env", i, ": z =", z, ", vz =", vz)

    # Verify all balls are above ground
    var all_valid = True
    var min_z: Float32 = 1000.0
    for i in range(BATCH):
        var z = get_body_z[DTYPE, NUM_BODIES, MAX_CONTACTS](
            host_state, env=i, body=0
        )
        if z < min_z:
            min_z = z
        if z < ground_z + radius - 0.01:
            print("ERROR: Env", i, "penetrating ground: z =", z)
            all_valid = False

    print("  Min z across all envs:", min_z)

    if all_valid:
        print(
            "PASSED: All", BATCH, "environments valid (no ground penetration)"
        )
    else:
        print("FAILED: Some environments have errors")

    print()
    return all_valid


# =============================================================================
# CPU vs GPU Comparison Tests
# =============================================================================


fn test_cpu_gpu_comparison_freefall_impulse() raises -> Bool:
    """Compare CPU and GPU free fall results using Impulse solver."""
    print("=" * 60)
    print("Comparison Test 5: CPU vs GPU Free Fall (Impulse)")
    print("=" * 60)

    comptime DTYPE_CPU = DType.float64
    comptime DTYPE_GPU = DType.float32
    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 4
    comptime BATCH = 1

    var initial_z: Float64 = 5.0
    var num_steps = 50
    var dt: Float64 = 0.01

    # --- CPU Simulation ---
    var model_cpu = Model[DTYPE_CPU, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81, timestep=dt, ground_z=-100.0, restitution=0.0
    )
    model_cpu.set_body(0, mass=1.0, radius=0.1)

    var data_cpu = Data[DTYPE_CPU, NUM_BODIES, MAX_CONTACTS]()
    data_cpu.set_body_position(0, 0.0, 0.0, initial_z)

    ImpulseIntegrator.simulate(model_cpu, data_cpu, num_steps)

    var cpu_z = Float64(data_cpu.get_body_z(0))
    var cpu_vz = Float64(data_cpu.get_body_vz(0))

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
        host_state, env=0, body=0, x=0.0, y=0.0, z=Float32(initial_z)
    )

    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var state_buf = ctx.enqueue_create_buffer[DTYPE_GPU](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE_GPU](NUM_BODIES * 9)
    ctx.enqueue_copy(state_buf, host_state)
    ctx.enqueue_copy(model_buf, model_gpu)
    ctx.synchronize()

    ImpulseIntegrator.simulate_gpu[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS, 0, BATCH](
        ctx,
        state_buf,
        model_buf,
        num_steps,
        Scalar[DTYPE_GPU](dt),
        Scalar[DTYPE_GPU](-9.81),
        Scalar[DTYPE_GPU](-100.0),
        Scalar[DTYPE_GPU](0.0),
        Scalar[DTYPE_GPU](0.5),
    )
    ctx.synchronize()

    ctx.enqueue_copy(host_state, state_buf)
    ctx.synchronize()

    var gpu_z = Float64(
        get_body_z[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
            host_state, env=0, body=0
        )
    )
    var gpu_vz = Float64(
        get_body_vz[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
            host_state, env=0, body=0
        )
    )

    # Compare
    var z_diff = abs_val(cpu_z - gpu_z)
    var vz_diff = abs_val(cpu_vz - gpu_vz)

    print("After", num_steps, "steps (dt=0.01):")
    print("  CPU:  z =", cpu_z, ", vz =", cpu_vz)
    print("  GPU:  z =", gpu_z, ", vz =", gpu_vz)
    print("  Diff: z =", z_diff, ", vz =", vz_diff)

    # Allow some tolerance for float32 vs float64 precision
    var z_tolerance = 0.01
    var vz_tolerance = 0.05

    var passed = z_diff < z_tolerance and vz_diff < vz_tolerance

    if passed:
        print("PASSED: CPU and GPU results match within tolerance")
    else:
        print("FAILED: CPU and GPU results differ significantly")

    print()
    return passed


fn test_cpu_gpu_comparison_balldrop_impulse() raises -> Bool:
    """Compare CPU and GPU ball drop results using Impulse solver."""
    print("=" * 60)
    print("Comparison Test 6: CPU vs GPU Ball Drop (Impulse)")
    print("=" * 60)

    comptime DTYPE_CPU = DType.float64
    comptime DTYPE_GPU = DType.float32
    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 4
    comptime BATCH = 1

    var initial_z: Float64 = 1.0
    var radius: Float64 = 0.1
    var num_steps = 200
    var dt: Float64 = 0.01

    # --- CPU Simulation ---
    var model_cpu = Model[DTYPE_CPU, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81, timestep=dt, ground_z=0.0, restitution=0.0
    )
    model_cpu.set_body(0, mass=1.0, radius=radius)

    var data_cpu = Data[DTYPE_CPU, NUM_BODIES, MAX_CONTACTS]()
    data_cpu.set_body_position(0, 0.0, 0.0, initial_z)

    ImpulseIntegrator.simulate(model_cpu, data_cpu, num_steps)

    var cpu_z = Float64(data_cpu.get_body_z(0))
    var cpu_vz = Float64(data_cpu.get_body_vz(0))

    # --- GPU Simulation ---
    var ctx = DeviceContext()

    var mass_f32: Float32 = 1.0
    var radius_f32: Float32 = 0.1
    var inertia = 0.4 * mass_f32 * radius_f32 * radius_f32

    var model_gpu = init_model_host_buffer[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
        ctx
    )
    model_gpu[0] = mass_f32
    model_gpu[1] = 1.0 / mass_f32
    model_gpu[2] = radius_f32
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
        host_state, env=0, body=0, x=0.0, y=0.0, z=Float32(initial_z)
    )

    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var state_buf = ctx.enqueue_create_buffer[DTYPE_GPU](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE_GPU](NUM_BODIES * 9)
    ctx.enqueue_copy(state_buf, host_state)
    ctx.enqueue_copy(model_buf, model_gpu)
    ctx.synchronize()

    ImpulseIntegrator.simulate_gpu[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS, 0, BATCH](
        ctx,
        state_buf,
        model_buf,
        num_steps,
        Scalar[DTYPE_GPU](dt),
        Scalar[DTYPE_GPU](-9.81),
        Scalar[DTYPE_GPU](0.0),
        Scalar[DTYPE_GPU](0.0),
        Scalar[DTYPE_GPU](0.5),
    )
    ctx.synchronize()

    ctx.enqueue_copy(host_state, state_buf)
    ctx.synchronize()

    var gpu_z = Float64(
        get_body_z[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
            host_state, env=0, body=0
        )
    )
    var gpu_vz = Float64(
        get_body_vz[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
            host_state, env=0, body=0
        )
    )

    # Compare
    var z_diff = abs_val(cpu_z - gpu_z)
    var vz_diff = abs_val(cpu_vz - gpu_vz)

    print("After", num_steps, "steps (dt=0.01):")
    print("  CPU:  z =", cpu_z, ", vz =", cpu_vz)
    print("  GPU:  z =", gpu_z, ", vz =", gpu_vz)
    print("  Diff: z =", z_diff, ", vz =", vz_diff)
    print("  Expected final z ~", radius)

    # Allow larger tolerance for contact settling
    var z_tolerance = 0.02
    var vz_tolerance = 0.5

    var passed = (
        z_diff < z_tolerance
        and abs_val(cpu_z - radius) < z_tolerance
        and abs_val(gpu_z - Float64(radius_f32)) < z_tolerance
    )

    if passed:
        print("PASSED: CPU and GPU both settled correctly")
    else:
        print("FAILED: CPU and GPU results differ or did not settle")

    print()
    return passed


fn test_cpu_gpu_comparison_freefall_pgs() raises -> Bool:
    """Compare CPU and GPU free fall results using PGS solver."""
    print("=" * 60)
    print("Comparison Test 7: CPU vs GPU Free Fall (PGS)")
    print("=" * 60)

    comptime DTYPE_CPU = DType.float64
    comptime DTYPE_GPU = DType.float32
    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 4
    comptime BATCH = 1

    var initial_z: Float64 = 5.0
    var num_steps = 50
    var dt: Float64 = 0.01

    # --- CPU Simulation ---
    var model_cpu = Model[DTYPE_CPU, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81, timestep=dt, ground_z=-100.0, restitution=0.0
    )
    model_cpu.set_body(0, mass=1.0, radius=0.1)

    var data_cpu = Data[DTYPE_CPU, NUM_BODIES, MAX_CONTACTS]()
    data_cpu.set_body_position(0, 0.0, 0.0, initial_z)

    PGSIntegrator.simulate(model_cpu, data_cpu, num_steps)

    var cpu_z = Float64(data_cpu.get_body_z(0))
    var cpu_vz = Float64(data_cpu.get_body_vz(0))

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
        host_state, env=0, body=0, x=0.0, y=0.0, z=Float32(initial_z)
    )

    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var state_buf = ctx.enqueue_create_buffer[DTYPE_GPU](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE_GPU](NUM_BODIES * 9)
    ctx.enqueue_copy(state_buf, host_state)
    ctx.enqueue_copy(model_buf, model_gpu)
    ctx.synchronize()

    PGSIntegrator.simulate_gpu[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS, 0, BATCH](
        ctx,
        state_buf,
        model_buf,
        num_steps,
        Scalar[DTYPE_GPU](dt),
        Scalar[DTYPE_GPU](-9.81),
        Scalar[DTYPE_GPU](-100.0),
        Scalar[DTYPE_GPU](0.0),
        Scalar[DTYPE_GPU](0.5),
    )
    ctx.synchronize()

    ctx.enqueue_copy(host_state, state_buf)
    ctx.synchronize()

    var gpu_z = Float64(
        get_body_z[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
            host_state, env=0, body=0
        )
    )
    var gpu_vz = Float64(
        get_body_vz[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
            host_state, env=0, body=0
        )
    )

    # Compare
    var z_diff = abs_val(cpu_z - gpu_z)
    var vz_diff = abs_val(cpu_vz - gpu_vz)

    print("After", num_steps, "steps (dt=0.01):")
    print("  CPU:  z =", cpu_z, ", vz =", cpu_vz)
    print("  GPU:  z =", gpu_z, ", vz =", gpu_vz)
    print("  Diff: z =", z_diff, ", vz =", vz_diff)

    var z_tolerance = 0.01
    var vz_tolerance = 0.05

    var passed = z_diff < z_tolerance and vz_diff < vz_tolerance

    if passed:
        print("PASSED: CPU and GPU results match within tolerance")
    else:
        print("FAILED: CPU and GPU results differ significantly")

    print()
    return passed


fn test_cpu_gpu_comparison_balldrop_pgs() raises -> Bool:
    """Compare CPU and GPU ball drop results using PGS solver."""
    print("=" * 60)
    print("Comparison Test 8: CPU vs GPU Ball Drop (PGS)")
    print("=" * 60)

    comptime DTYPE_CPU = DType.float64
    comptime DTYPE_GPU = DType.float32
    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 4
    comptime BATCH = 1

    var initial_z: Float64 = 1.0
    var radius: Float64 = 0.1
    var num_steps = 200
    var dt: Float64 = 0.01

    # --- CPU Simulation ---
    var model_cpu = Model[DTYPE_CPU, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81, timestep=dt, ground_z=0.0, restitution=0.0
    )
    model_cpu.set_body(0, mass=1.0, radius=radius)

    var data_cpu = Data[DTYPE_CPU, NUM_BODIES, MAX_CONTACTS]()
    data_cpu.set_body_position(0, 0.0, 0.0, initial_z)

    PGSIntegrator.simulate(model_cpu, data_cpu, num_steps)

    var cpu_z = Float64(data_cpu.get_body_z(0))
    var cpu_vz = Float64(data_cpu.get_body_vz(0))

    # --- GPU Simulation ---
    var ctx = DeviceContext()

    var mass_f32: Float32 = 1.0
    var radius_f32: Float32 = 0.1
    var inertia = 0.4 * mass_f32 * radius_f32 * radius_f32

    var model_gpu = init_model_host_buffer[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
        ctx
    )
    model_gpu[0] = mass_f32
    model_gpu[1] = 1.0 / mass_f32
    model_gpu[2] = radius_f32
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
        host_state, env=0, body=0, x=0.0, y=0.0, z=Float32(initial_z)
    )

    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var state_buf = ctx.enqueue_create_buffer[DTYPE_GPU](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE_GPU](NUM_BODIES * 9)
    ctx.enqueue_copy(state_buf, host_state)
    ctx.enqueue_copy(model_buf, model_gpu)
    ctx.synchronize()

    PGSIntegrator.simulate_gpu[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS, 0, BATCH](
        ctx,
        state_buf,
        model_buf,
        num_steps,
        Scalar[DTYPE_GPU](dt),
        Scalar[DTYPE_GPU](-9.81),
        Scalar[DTYPE_GPU](0.0),
        Scalar[DTYPE_GPU](0.0),
        Scalar[DTYPE_GPU](0.5),
    )
    ctx.synchronize()

    ctx.enqueue_copy(host_state, state_buf)
    ctx.synchronize()

    var gpu_z = Float64(
        get_body_z[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
            host_state, env=0, body=0
        )
    )
    var gpu_vz = Float64(
        get_body_vz[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
            host_state, env=0, body=0
        )
    )

    # Compare
    var z_diff = abs_val(cpu_z - gpu_z)
    var vz_diff = abs_val(cpu_vz - gpu_vz)

    print("After", num_steps, "steps (dt=0.01):")
    print("  CPU:  z =", cpu_z, ", vz =", cpu_vz)
    print("  GPU:  z =", gpu_z, ", vz =", gpu_vz)
    print("  Diff: z =", z_diff, ", vz =", vz_diff)
    print("  Expected final z ~", radius)

    var z_tolerance = 0.02
    var vz_tolerance = 0.5

    var passed = (
        z_diff < z_tolerance
        and abs_val(cpu_z - radius) < z_tolerance
        and abs_val(gpu_z - Float64(radius_f32)) < z_tolerance
    )

    if passed:
        print("PASSED: CPU and GPU both settled correctly")
    else:
        print("FAILED: CPU and GPU results differ or did not settle")

    print()
    return passed


fn test_cpu_gpu_comparison_two_spheres_impulse() raises -> Bool:
    """Compare CPU and GPU two-sphere collision using Impulse solver."""
    print("=" * 60)
    print("Comparison Test 9: CPU vs GPU Two Spheres (Impulse)")
    print("=" * 60)

    comptime DTYPE_CPU = DType.float64
    comptime DTYPE_GPU = DType.float32
    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 8
    comptime BATCH = 1

    var num_steps = 100
    var dt: Float64 = 0.01

    # --- CPU Simulation ---
    var model_cpu = Model[DTYPE_CPU, NUM_BODIES, MAX_CONTACTS](
        gravity_z=0.0,  # No gravity
        timestep=dt,
        ground_z=-10.0,  # Far below
        restitution=1.0,  # Elastic
    )
    model_cpu.set_body(0, mass=1.0, radius=0.5)
    model_cpu.set_body(1, mass=1.0, radius=0.5)

    var data_cpu = Data[DTYPE_CPU, NUM_BODIES, MAX_CONTACTS]()
    data_cpu.set_body_position(0, -1.0, 0.0, 1.0)
    data_cpu.set_body_velocity(0, 2.0, 0.0, 0.0)
    data_cpu.set_body_position(1, 1.0, 0.0, 1.0)
    data_cpu.set_body_velocity(1, -2.0, 0.0, 0.0)

    ImpulseIntegrator.simulate(model_cpu, data_cpu, num_steps)

    var cpu_pos0 = data_cpu.get_body_position(0)
    var cpu_vel0 = data_cpu.get_body_velocity(0)
    var cpu_pos1 = data_cpu.get_body_position(1)
    var cpu_vel1 = data_cpu.get_body_velocity(1)

    # --- GPU Simulation ---
    var ctx = DeviceContext()

    var mass: Float32 = 1.0
    var radius: Float32 = 0.5
    var inertia = 0.4 * mass * radius * radius

    var model_gpu = init_model_host_buffer[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
        ctx
    )
    for i in range(NUM_BODIES):
        var offset = i * 9
        model_gpu[offset + 0] = mass
        model_gpu[offset + 1] = 1.0 / mass
        model_gpu[offset + 2] = radius
        model_gpu[offset + 3] = inertia
        model_gpu[offset + 4] = inertia
        model_gpu[offset + 5] = inertia
        model_gpu[offset + 6] = 1.0 / inertia
        model_gpu[offset + 7] = 1.0 / inertia
        model_gpu[offset + 8] = 1.0 / inertia

    var host_state = init_state_host_buffer[
        DTYPE_GPU, NUM_BODIES, MAX_CONTACTS, BATCH
    ](ctx)
    set_body_position[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0, x=-1.0, y=0.0, z=1.0
    )
    set_body_velocity[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0, vx=2.0, vy=0.0, vz=0.0
    )
    set_body_position[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=1, x=1.0, y=0.0, z=1.0
    )
    set_body_velocity[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=1, vx=-2.0, vy=0.0, vz=0.0
    )

    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var state_buf = ctx.enqueue_create_buffer[DTYPE_GPU](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE_GPU](NUM_BODIES * 9)
    ctx.enqueue_copy(state_buf, host_state)
    ctx.enqueue_copy(model_buf, model_gpu)
    ctx.synchronize()

    ImpulseIntegrator.simulate_gpu[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS, 0, BATCH](
        ctx,
        state_buf,
        model_buf,
        num_steps,
        Scalar[DTYPE_GPU](dt),
        Scalar[DTYPE_GPU](0.0),
        Scalar[DTYPE_GPU](-10.0),
        Scalar[DTYPE_GPU](1.0),
        Scalar[DTYPE_GPU](0.5),
    )
    ctx.synchronize()

    ctx.enqueue_copy(host_state, state_buf)
    ctx.synchronize()

    var gpu_pos0 = get_body_position[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0
    )
    var gpu_vel0 = get_body_velocity[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=0
    )
    var gpu_pos1 = get_body_position[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=1
    )
    var gpu_vel1 = get_body_velocity[DTYPE_GPU, NUM_BODIES, MAX_CONTACTS](
        host_state, env=0, body=1
    )

    print("After", num_steps, "steps (dt=0.01):")
    print(
        "  CPU sphere 0: pos=(",
        cpu_pos0[0],
        ",",
        cpu_pos0[1],
        ",",
        cpu_pos0[2],
        "), vel=(",
        cpu_vel0[0],
        ",",
        cpu_vel0[1],
        ",",
        cpu_vel0[2],
        ")",
    )
    print(
        "  GPU sphere 0: pos=(",
        gpu_pos0[0],
        ",",
        gpu_pos0[1],
        ",",
        gpu_pos0[2],
        "), vel=(",
        gpu_vel0[0],
        ",",
        gpu_vel0[1],
        ",",
        gpu_vel0[2],
        ")",
    )
    print(
        "  CPU sphere 1: pos=(",
        cpu_pos1[0],
        ",",
        cpu_pos1[1],
        ",",
        cpu_pos1[2],
        "), vel=(",
        cpu_vel1[0],
        ",",
        cpu_vel1[1],
        ",",
        cpu_vel1[2],
        ")",
    )
    print(
        "  GPU sphere 1: pos=(",
        gpu_pos1[0],
        ",",
        gpu_pos1[1],
        ",",
        gpu_pos1[2],
        "), vel=(",
        gpu_vel1[0],
        ",",
        gpu_vel1[1],
        ",",
        gpu_vel1[2],
        ")",
    )

    # Check qualitative behavior: both should have bounced
    # Sphere 0 should be moving left (vx < 0), sphere 1 should be moving right (vx > 0)
    var cpu_bounced = cpu_vel0[0] < 0 and cpu_vel1[0] > 0
    var gpu_bounced = gpu_vel0[0] < 0 and gpu_vel1[0] > 0

    # Position comparison (relaxed due to different solver parameters)
    var pos_diff_0 = abs_val(Float64(cpu_pos0[0]) - Float64(gpu_pos0[0]))
    var pos_diff_1 = abs_val(Float64(cpu_pos1[0]) - Float64(gpu_pos1[0]))

    print("  Position diff sphere 0 x:", pos_diff_0)
    print("  Position diff sphere 1 x:", pos_diff_1)
    print("  CPU bounced:", cpu_bounced, ", GPU bounced:", gpu_bounced)

    # Allow larger tolerance for collision dynamics
    var pos_tolerance = 0.5  # Position can differ due to solver differences

    var passed = (
        cpu_bounced
        and gpu_bounced
        and pos_diff_0 < pos_tolerance
        and pos_diff_1 < pos_tolerance
    )

    if passed:
        print("PASSED: Both CPU and GPU show collision and bounce")
    else:
        if not cpu_bounced:
            print("FAILED: CPU did not show bounce")
        if not gpu_bounced:
            print("FAILED: GPU did not show bounce")
        if pos_diff_0 >= pos_tolerance or pos_diff_1 >= pos_tolerance:
            print("FAILED: Position difference too large")

    print()
    return passed


fn main() raises:
    """Run all GPU tests."""
    print()
    print("=" * 60)
    print("       Physics3D v2 GPU Tests (Multi-Body)")
    print("=" * 60)
    print()

    var passed_count = 0
    var total_count = 0

    # GPU-only tests
    total_count += 1
    if test_free_fall_gpu():
        passed_count += 1

    total_count += 1
    if test_ball_drop_gpu():
        passed_count += 1

    total_count += 1
    if test_two_spheres_collision_gpu():
        passed_count += 1

    total_count += 1
    if test_batched_simulation_gpu():
        passed_count += 1

    # CPU vs GPU comparison tests
    print()
    print("=" * 60)
    print("       CPU vs GPU Comparison Tests")
    print("=" * 60)
    print()

    total_count += 1
    if test_cpu_gpu_comparison_freefall_impulse():
        passed_count += 1

    total_count += 1
    if test_cpu_gpu_comparison_balldrop_impulse():
        passed_count += 1

    total_count += 1
    if test_cpu_gpu_comparison_freefall_pgs():
        passed_count += 1

    total_count += 1
    if test_cpu_gpu_comparison_balldrop_pgs():
        passed_count += 1

    total_count += 1
    if test_cpu_gpu_comparison_two_spheres_impulse():
        passed_count += 1

    # Summary
    print("=" * 60)
    print("                    SUMMARY")
    print("=" * 60)
    print("Passed:", passed_count, "/", total_count, "tests")

    if passed_count == total_count:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")

    print("=" * 60)
    print()
