"""Test Physics3D v2 GPU implementation.

Validates that the GPU physics kernel produces the same results as
the CPU implementation for Phase 1-2 scenarios:
1. Free fall (Phase 1)
2. Ball drop and contact (Phase 2)
"""

from gpu.host import DeviceContext

from physics3d_v2.gpu import (
    STATE_SIZE,
    Physics3DV2Kernel,
    init_state_host_buffer,
    set_position,
    get_z,
    get_vz,
    is_contact_active,
)
from physics3d_v2.gpu.constants import GEOM_SPHERE


fn abs_val(x: Float64) -> Float64:
    """Return absolute value."""
    if x < 0:
        return -x
    return x


fn test_free_fall_gpu() raises:
    """Test free fall matches analytical solution on GPU."""
    print("=" * 60)
    print("GPU Test 1: Free Fall")
    print("=" * 60)

    comptime DTYPE = DType.float32
    comptime BATCH = 1

    var ctx = DeviceContext()

    # Physics parameters
    var dt = Scalar[DTYPE](0.01)
    var gravity_z = Scalar[DTYPE](-9.81)
    var mass = Scalar[DTYPE](1.0)
    var radius = Scalar[DTYPE](0.1)
    # Sphere inertia: I = 2/5 * m * r^2
    var inertia = 0.4 * mass * radius * radius
    var ixx = inertia
    var iyy = inertia
    var izz = inertia
    var ground_z = Scalar[DTYPE](0.0)
    var restitution = Scalar[DTYPE](0.0)
    var baumgarte = Scalar[DTYPE](0.2)
    var slop = Scalar[DTYPE](0.001)

    # Initial height (above ground so no contact)
    var initial_z = Scalar[DTYPE](10.0)

    # Initialize state buffer
    var host_buf = init_state_host_buffer[DTYPE, BATCH](ctx)
    set_position(host_buf, 0, Scalar[DTYPE](0), Scalar[DTYPE](0), initial_z)

    # Copy to device
    var device_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    ctx.enqueue_copy(device_buf, host_buf)
    ctx.synchronize()

    # Simulate 100 steps (1 second)
    var num_steps = 100

    for _ in range(num_steps):
        Physics3DV2Kernel.step_gpu[DTYPE, BATCH](
            ctx,
            device_buf,
            dt,
            gravity_z,
            mass,
            ixx,
            iyy,
            izz,
            GEOM_SPHERE,
            radius,
            ground_z,
            restitution,
            baumgarte,
            slop,
        )

    ctx.synchronize()

    # Copy back to host
    ctx.enqueue_copy(host_buf, device_buf)
    ctx.synchronize()

    # Get results
    var final_z = get_z(host_buf, 0)
    var final_vz = get_vz(host_buf, 0)

    # Analytical solution after t=1s:
    # z(t) = z0 + v0*t + 0.5*g*t^2 = 10 + 0 + 0.5*(-9.81)*1^2 = 5.095
    # vz(t) = v0 + g*t = 0 + (-9.81)*1 = -9.81
    var expected_z = Scalar[DTYPE](5.095)
    var expected_vz = Scalar[DTYPE](-9.81)

    var z_error = abs_val(Float64(final_z) - Float64(expected_z))
    var vz_error = abs_val(Float64(final_vz) - Float64(expected_vz))

    print("After", num_steps, "steps (dt=0.01, t=1s):")
    print("  z:  ", final_z, " (expected:", expected_z, ", error:", z_error, ")")
    print("  vz: ", final_vz, " (expected:", expected_vz, ", error:", vz_error, ")")

    # Check tolerances
    var z_tolerance = 0.05  # ~1% error
    var vz_tolerance = 0.1

    if z_error < z_tolerance and vz_error < vz_tolerance:
        print("PASSED: Free fall within tolerance")
    else:
        print("FAILED: Error exceeds tolerance")

    print()


fn test_ball_drop_gpu() raises:
    """Test ball drop stops at correct height on GPU."""
    print("=" * 60)
    print("GPU Test 2: Ball Drop (Contact)")
    print("=" * 60)

    comptime DTYPE = DType.float32
    comptime BATCH = 1

    var ctx = DeviceContext()

    # Physics parameters
    var dt = Scalar[DTYPE](0.01)
    var gravity_z = Scalar[DTYPE](-9.81)
    var mass = Scalar[DTYPE](1.0)
    var radius = Scalar[DTYPE](0.1)
    var inertia = 0.4 * mass * radius * radius
    var ixx = inertia
    var iyy = inertia
    var izz = inertia
    var ground_z = Scalar[DTYPE](0.0)
    var restitution = Scalar[DTYPE](0.0)  # Inelastic
    var baumgarte = Scalar[DTYPE](0.2)
    var slop = Scalar[DTYPE](0.001)

    # Drop from 1m
    var initial_z = Scalar[DTYPE](1.0)

    # Initialize state buffer
    var host_buf = init_state_host_buffer[DTYPE, BATCH](ctx)
    set_position(host_buf, 0, Scalar[DTYPE](0), Scalar[DTYPE](0), initial_z)

    # Copy to device
    var device_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    ctx.enqueue_copy(device_buf, host_buf)
    ctx.synchronize()

    # Simulate enough steps for ball to fall and settle
    var num_steps = 200

    for _ in range(num_steps):
        Physics3DV2Kernel.step_gpu[DTYPE, BATCH](
            ctx,
            device_buf,
            dt,
            gravity_z,
            mass,
            ixx,
            iyy,
            izz,
            GEOM_SPHERE,
            radius,
            ground_z,
            restitution,
            baumgarte,
            slop,
        )

    ctx.synchronize()

    # Copy back to host
    ctx.enqueue_copy(host_buf, device_buf)
    ctx.synchronize()

    # Get results
    var final_z = get_z(host_buf, 0)
    var final_vz = get_vz(host_buf, 0)
    var contact_active = is_contact_active(host_buf, 0)

    # Expected: ball rests at z = radius = 0.1
    var expected_z = Scalar[DTYPE](0.1)
    var z_error = abs_val(Float64(final_z) - Float64(expected_z))

    print("After", num_steps, "steps:")
    print("  z:            ", final_z, " (expected:", expected_z, ")")
    print("  vz:           ", final_vz, " (expected: ~0)")
    print("  contact:      ", contact_active)
    print("  z error:      ", z_error, "m")

    var z_tolerance = 0.002  # 2mm
    var vz_tolerance = 0.1

    if z_error < z_tolerance and abs_val(Float64(final_vz)) < vz_tolerance:
        print("PASSED: Ball settled at correct height")
    else:
        print("FAILED: Ball did not settle correctly")

    print()


fn test_batched_simulation() raises:
    """Test multiple environments in parallel."""
    print("=" * 60)
    print("GPU Test 3: Batched Simulation (multiple envs)")
    print("=" * 60)

    comptime DTYPE = DType.float32
    comptime BATCH = 256  # Simulate 256 environments in parallel

    var ctx = DeviceContext()

    # Physics parameters
    var dt = Scalar[DTYPE](0.01)
    var gravity_z = Scalar[DTYPE](-9.81)
    var mass = Scalar[DTYPE](1.0)
    var radius = Scalar[DTYPE](0.1)
    var inertia = 0.4 * mass * radius * radius
    var ixx = inertia
    var iyy = inertia
    var izz = inertia
    var ground_z = Scalar[DTYPE](0.0)
    var restitution = Scalar[DTYPE](0.5)  # Bouncy for variety
    var baumgarte = Scalar[DTYPE](0.2)
    var slop = Scalar[DTYPE](0.001)

    # Initialize state buffer with different starting heights
    var host_buf = init_state_host_buffer[DTYPE, BATCH](ctx)

    for i in range(BATCH):
        var height = Scalar[DTYPE](0.5 + Float32(i) * 0.01)  # 0.5 to ~3m
        set_position(host_buf, i, Scalar[DTYPE](0), Scalar[DTYPE](0), height)

    # Copy to device
    var device_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    ctx.enqueue_copy(device_buf, host_buf)
    ctx.synchronize()

    # Simulate
    var num_steps = 100

    for _ in range(num_steps):
        Physics3DV2Kernel.step_gpu[DTYPE, BATCH](
            ctx,
            device_buf,
            dt,
            gravity_z,
            mass,
            ixx,
            iyy,
            izz,
            GEOM_SPHERE,
            radius,
            ground_z,
            restitution,
            baumgarte,
            slop,
        )

    ctx.synchronize()

    # Copy back to host
    ctx.enqueue_copy(host_buf, device_buf)
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
        var z = get_z(host_buf, i)
        var vz = get_vz(host_buf, i)
        print("  Env", i, ": z =", z, ", vz =", vz)

    # Verify all balls are above ground
    var all_valid = True
    for i in range(BATCH):
        var z = get_z(host_buf, i)
        if z < ground_z + radius - Scalar[DTYPE](0.01):
            print("ERROR: Env", i, "penetrating ground: z =", z)
            all_valid = False

    if all_valid:
        print("PASSED: All", BATCH, "environments valid (no ground penetration)")
    else:
        print("FAILED: Some environments have errors")

    print()


fn main() raises:
    """Run all GPU tests."""
    print()
    print("Physics3D v2 GPU Tests")
    print("=" * 60)
    print()

    test_free_fall_gpu()
    test_ball_drop_gpu()
    test_batched_simulation()

    print("All GPU tests complete.")
    print()
