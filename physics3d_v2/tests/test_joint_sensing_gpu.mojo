"""Test joint sensing with GPU-simulated data (Phase 7, Step 7.2).

The sensing functions are CPU-only, but this test verifies they work correctly
with state that was simulated on GPU and copied back.

Tests:
1. Angle sensing after GPU pendulum swing
2. Angular velocity sensing after GPU simulation
3. CPU vs GPU sensing parity
"""

from math import sqrt, sin, cos
from testing import assert_true

from gpu.host import DeviceContext

from physics3d_v2 import Model, Data, ImpulseIntegrator
from physics3d_v2.joints import get_joint_angle, get_joint_angular_velocity
from physics3d_v2.gpu.constants import (
    compute_state_size,
    body_offset,
    joint_offset,
    metadata_offset,
    BODY_IDX_PX,
    BODY_IDX_PY,
    BODY_IDX_PZ,
    BODY_IDX_QX,
    BODY_IDX_QY,
    BODY_IDX_QZ,
    BODY_IDX_QW,
    BODY_IDX_VX,
    BODY_IDX_VY,
    BODY_IDX_VZ,
    BODY_IDX_WX,
    BODY_IDX_WY,
    BODY_IDX_WZ,
    JOINT_IDX_PARENT,
    JOINT_IDX_ANCHOR_PX,
    JOINT_IDX_ANCHOR_PY,
    JOINT_IDX_ANCHOR_PZ,
    JOINT_IDX_ANCHOR_CX,
    JOINT_IDX_ANCHOR_CY,
    JOINT_IDX_ANCHOR_CZ,
    JOINT_IDX_AXIS_X,
    JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z,
    JOINT_IDX_TARGET_TORQUE,
    JOINT_IDX_TORQUE_LIMIT,
    META_IDX_NUM_JOINTS,
    MODEL_BODY_SIZE,
    MODEL_IDX_MASS,
    MODEL_IDX_INV_MASS,
    MODEL_IDX_RADIUS,
    MODEL_IDX_IXX,
    MODEL_IDX_IYY,
    MODEL_IDX_IZZ,
    MODEL_IDX_INV_IXX,
    MODEL_IDX_INV_IYY,
    MODEL_IDX_INV_IZZ,
)


comptime DTYPE = DType.float64
comptime PI: Float64 = 3.14159265358979323846


fn abs64(x: Scalar[DTYPE]) -> Scalar[DTYPE]:
    if x < 0:
        return -x
    return x


fn abs32(x: Float32) -> Float32:
    if x < 0:
        return -x
    return x


fn copy_gpu_state_to_cpu[
    NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int
](
    state_host: List[Float32],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
):
    """Copy GPU state buffer to CPU Data struct."""
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()

    for i in range(NUM_BODIES):
        var b_off = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](i)

        # Position
        data.positions[i * 3 + 0] = Float64(state_host[b_off + BODY_IDX_PX])
        data.positions[i * 3 + 1] = Float64(state_host[b_off + BODY_IDX_PY])
        data.positions[i * 3 + 2] = Float64(state_host[b_off + BODY_IDX_PZ])

        # Quaternion
        data.quaternions[i * 4 + 0] = Float64(state_host[b_off + BODY_IDX_QX])
        data.quaternions[i * 4 + 1] = Float64(state_host[b_off + BODY_IDX_QY])
        data.quaternions[i * 4 + 2] = Float64(state_host[b_off + BODY_IDX_QZ])
        data.quaternions[i * 4 + 3] = Float64(state_host[b_off + BODY_IDX_QW])

        # Velocities
        data.velocities[i * 3 + 0] = Float64(state_host[b_off + BODY_IDX_VX])
        data.velocities[i * 3 + 1] = Float64(state_host[b_off + BODY_IDX_VY])
        data.velocities[i * 3 + 2] = Float64(state_host[b_off + BODY_IDX_VZ])

        # Angular velocities
        data.angular_velocities[i * 3 + 0] = Float64(state_host[b_off + BODY_IDX_WX])
        data.angular_velocities[i * 3 + 1] = Float64(state_host[b_off + BODY_IDX_WY])
        data.angular_velocities[i * 3 + 2] = Float64(state_host[b_off + BODY_IDX_WZ])


fn test_angle_sensing_after_gpu_simulation() raises:
    """Test that angle sensing works correctly after GPU simulation."""
    print("Test 1: Angle sensing after GPU simulation...")

    var ctx = DeviceContext()

    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 5
    comptime MAX_JOINTS = 1
    comptime BATCH = 1
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()

    # Create CPU model for sensing
    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](
        gravity_z=-9.81,
        timestep=0.005,
        ground_z=-10.0,
    )
    model.set_body(0, mass=1.0, radius=0.1)
    _ = model.add_hinge_joint(
        parent=-1,
        child=0,
        anchor_parent=(0.0, 0.0, 1.0),
        anchor_child=(0.0, 0.0, 1.0),
        axis=(0.0, 1.0, 0.0),
    )

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()

    # Create GPU state with initial 30 degree angle
    var state_host = List[Float32](capacity=STATE_SIZE)
    for _ in range(STATE_SIZE):
        state_host.append(0.0)

    var theta0: Float32 = 30.0 * Float32(PI) / 180.0
    var half_theta = theta0 / 2.0

    var b0 = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    state_host[b0 + BODY_IDX_PX] = sin(theta0)
    state_host[b0 + BODY_IDX_PY] = 0.0
    state_host[b0 + BODY_IDX_PZ] = 1.0 - 1.0 * cos(theta0)

    state_host[b0 + BODY_IDX_QX] = 0.0
    state_host[b0 + BODY_IDX_QY] = sin(half_theta)
    state_host[b0 + BODY_IDX_QZ] = 0.0
    state_host[b0 + BODY_IDX_QW] = cos(half_theta)

    var j0 = joint_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    state_host[j0 + JOINT_IDX_PARENT] = -1.0
    state_host[j0 + JOINT_IDX_ANCHOR_PZ] = 1.0
    state_host[j0 + JOINT_IDX_ANCHOR_CZ] = 1.0
    state_host[j0 + JOINT_IDX_AXIS_Y] = 1.0
    state_host[j0 + JOINT_IDX_TORQUE_LIMIT] = 100.0

    var m_off = metadata_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()
    state_host[m_off + META_IDX_NUM_JOINTS] = 1.0

    # Model buffer
    var model_host = List[Float32](capacity=NUM_BODIES * MODEL_BODY_SIZE)
    for _ in range(NUM_BODIES * MODEL_BODY_SIZE):
        model_host.append(0.0)

    var mass: Float32 = 1.0
    var radius: Float32 = 0.1
    var inertia = 0.4 * mass * radius * radius
    model_host[MODEL_IDX_MASS] = mass
    model_host[MODEL_IDX_INV_MASS] = 1.0 / mass
    model_host[MODEL_IDX_RADIUS] = radius
    model_host[MODEL_IDX_IXX] = inertia
    model_host[MODEL_IDX_IYY] = inertia
    model_host[MODEL_IDX_IZZ] = inertia
    model_host[MODEL_IDX_INV_IXX] = 1.0 / inertia
    model_host[MODEL_IDX_INV_IYY] = 1.0 / inertia
    model_host[MODEL_IDX_INV_IZZ] = 1.0 / inertia

    # Copy to GPU
    var state_buf = ctx.enqueue_create_buffer[DType.float32](STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DType.float32](NUM_BODIES * MODEL_BODY_SIZE)
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())
    ctx.synchronize()

    # Check initial angle
    copy_gpu_state_to_cpu[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](state_host, data)
    var initial_angle = get_joint_angle(model, data, 0)
    print("  Initial angle:", initial_angle * 180.0 / PI, "deg (expected ~30)")

    # Simulate on GPU for 0.5 seconds (100 steps at 0.005s)
    for _ in range(100):
        ImpulseIntegrator.step_gpu[DType.float32, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, BATCH](
            ctx, state_buf, model_buf,
            dt=Float32(0.005), gravity_z=Float32(-9.81), ground_z=Float32(-10.0),
            restitution=Float32(0.0), friction=Float32(0.0),
        )
    ctx.synchronize()

    # Copy back and sense
    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    copy_gpu_state_to_cpu[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](state_host, data)
    var mid_angle = get_joint_angle(model, data, 0)
    print("  Angle at t=0.5s:", mid_angle * 180.0 / PI, "deg")

    # Simulate more
    for _ in range(100):
        ImpulseIntegrator.step_gpu[DType.float32, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, BATCH](
            ctx, state_buf, model_buf,
            dt=Float32(0.005), gravity_z=Float32(-9.81), ground_z=Float32(-10.0),
            restitution=Float32(0.0), friction=Float32(0.0),
        )
    ctx.synchronize()

    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    copy_gpu_state_to_cpu[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](state_host, data)
    var final_angle = get_joint_angle(model, data, 0)
    print("  Angle at t=1.0s:", final_angle * 180.0 / PI, "deg")

    # Angles should change during swing
    assert_true(
        abs64(final_angle - initial_angle) > 0.1,
        "Angle should change during GPU simulation",
    )

    print("  PASSED: Angle sensing works after GPU simulation")


fn test_angular_velocity_sensing_after_gpu() raises:
    """Test angular velocity sensing after GPU simulation."""
    print("\nTest 2: Angular velocity sensing after GPU simulation...")

    var ctx = DeviceContext()

    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 5
    comptime MAX_JOINTS = 1
    comptime BATCH = 1
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()

    # CPU model for sensing
    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](
        gravity_z=-9.81,
        timestep=0.005,
        ground_z=-10.0,
    )
    model.set_body(0, mass=1.0, radius=0.1)
    _ = model.add_hinge_joint(
        parent=-1,
        child=0,
        anchor_parent=(0.0, 0.0, 1.0),
        anchor_child=(0.0, 0.0, 1.0),
        axis=(0.0, 1.0, 0.0),
    )

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()

    # GPU state with initial angle and no velocity
    var state_host = List[Float32](capacity=STATE_SIZE)
    for _ in range(STATE_SIZE):
        state_host.append(0.0)

    var theta0: Float32 = 45.0 * Float32(PI) / 180.0
    var half_theta = theta0 / 2.0

    var b0 = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    state_host[b0 + BODY_IDX_PX] = sin(theta0)
    state_host[b0 + BODY_IDX_PZ] = 1.0 - cos(theta0)
    state_host[b0 + BODY_IDX_QY] = sin(half_theta)
    state_host[b0 + BODY_IDX_QW] = cos(half_theta)

    var j0 = joint_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    state_host[j0 + JOINT_IDX_PARENT] = -1.0
    state_host[j0 + JOINT_IDX_ANCHOR_PZ] = 1.0
    state_host[j0 + JOINT_IDX_ANCHOR_CZ] = 1.0
    state_host[j0 + JOINT_IDX_AXIS_Y] = 1.0
    state_host[j0 + JOINT_IDX_TORQUE_LIMIT] = 100.0

    var m_off = metadata_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()
    state_host[m_off + META_IDX_NUM_JOINTS] = 1.0

    var model_host = List[Float32](capacity=NUM_BODIES * MODEL_BODY_SIZE)
    for _ in range(NUM_BODIES * MODEL_BODY_SIZE):
        model_host.append(0.0)

    var mass: Float32 = 1.0
    var radius: Float32 = 0.1
    var inertia = 0.4 * mass * radius * radius
    model_host[MODEL_IDX_MASS] = mass
    model_host[MODEL_IDX_INV_MASS] = 1.0 / mass
    model_host[MODEL_IDX_RADIUS] = radius
    model_host[MODEL_IDX_IXX] = inertia
    model_host[MODEL_IDX_IYY] = inertia
    model_host[MODEL_IDX_IZZ] = inertia
    model_host[MODEL_IDX_INV_IXX] = 1.0 / inertia
    model_host[MODEL_IDX_INV_IYY] = 1.0 / inertia
    model_host[MODEL_IDX_INV_IZZ] = 1.0 / inertia

    var state_buf = ctx.enqueue_create_buffer[DType.float32](STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DType.float32](NUM_BODIES * MODEL_BODY_SIZE)
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())
    ctx.synchronize()

    # Check initial angular velocity (should be ~0)
    copy_gpu_state_to_cpu[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](state_host, data)
    var initial_omega = get_joint_angular_velocity(model, data, 0)
    print("  Initial angular velocity:", initial_omega, "rad/s")

    # Simulate - pendulum should accelerate
    for _ in range(50):
        ImpulseIntegrator.step_gpu[DType.float32, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, BATCH](
            ctx, state_buf, model_buf,
            dt=Float32(0.005), gravity_z=Float32(-9.81), ground_z=Float32(-10.0),
            restitution=Float32(0.0), friction=Float32(0.0),
        )
    ctx.synchronize()

    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    copy_gpu_state_to_cpu[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](state_host, data)
    var final_omega = get_joint_angular_velocity(model, data, 0)
    print("  Angular velocity at t=0.25s:", final_omega, "rad/s")

    # Pendulum should have non-zero angular velocity
    assert_true(
        abs64(final_omega) > 0.5,
        "Pendulum should have angular velocity after swing",
    )

    print("  PASSED: Angular velocity sensing works after GPU simulation")


fn test_cpu_gpu_sensing_parity() raises:
    """Test that sensing gives same results for CPU and GPU simulated states."""
    print("\nTest 3: CPU vs GPU sensing parity...")

    var ctx = DeviceContext()

    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 5
    comptime MAX_JOINTS = 1
    comptime BATCH = 1
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()

    # === CPU simulation ===
    var model_cpu = Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](
        gravity_z=-9.81,
        timestep=0.005,
        ground_z=-10.0,
    )
    model_cpu.set_body(0, mass=1.0, radius=0.1)
    _ = model_cpu.add_hinge_joint(
        parent=-1,
        child=0,
        anchor_parent=(0.0, 0.0, 1.0),
        anchor_child=(0.0, 0.0, 1.0),
        axis=(0.0, 1.0, 0.0),
    )

    var data_cpu = Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()
    var theta0 = Scalar[DTYPE](30.0 * PI / 180.0)
    data_cpu.set_body_position(0, sin(theta0), 0.0, 1.0 - cos(theta0))
    var half_theta = theta0 / 2.0
    data_cpu.quaternions[0] = 0.0
    data_cpu.quaternions[1] = sin(half_theta)
    data_cpu.quaternions[2] = 0.0
    data_cpu.quaternions[3] = cos(half_theta)

    # === GPU simulation ===
    var model_gpu = Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](
        gravity_z=-9.81,
        timestep=0.005,
        ground_z=-10.0,
    )
    model_gpu.set_body(0, mass=1.0, radius=0.1)
    _ = model_gpu.add_hinge_joint(
        parent=-1,
        child=0,
        anchor_parent=(0.0, 0.0, 1.0),
        anchor_child=(0.0, 0.0, 1.0),
        axis=(0.0, 1.0, 0.0),
    )

    var data_gpu = Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()

    var state_host = List[Float32](capacity=STATE_SIZE)
    for _ in range(STATE_SIZE):
        state_host.append(0.0)

    var theta0_f32: Float32 = 30.0 * Float32(PI) / 180.0
    var half_theta_f32 = theta0_f32 / 2.0

    var b0 = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    state_host[b0 + BODY_IDX_PX] = sin(theta0_f32)
    state_host[b0 + BODY_IDX_PZ] = 1.0 - cos(theta0_f32)
    state_host[b0 + BODY_IDX_QY] = sin(half_theta_f32)
    state_host[b0 + BODY_IDX_QW] = cos(half_theta_f32)

    var j0 = joint_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    state_host[j0 + JOINT_IDX_PARENT] = -1.0
    state_host[j0 + JOINT_IDX_ANCHOR_PZ] = 1.0
    state_host[j0 + JOINT_IDX_ANCHOR_CZ] = 1.0
    state_host[j0 + JOINT_IDX_AXIS_Y] = 1.0
    state_host[j0 + JOINT_IDX_TORQUE_LIMIT] = 100.0

    var m_off = metadata_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()
    state_host[m_off + META_IDX_NUM_JOINTS] = 1.0

    var model_host = List[Float32](capacity=NUM_BODIES * MODEL_BODY_SIZE)
    for _ in range(NUM_BODIES * MODEL_BODY_SIZE):
        model_host.append(0.0)

    var mass: Float32 = 1.0
    var radius: Float32 = 0.1
    var inertia = 0.4 * mass * radius * radius
    model_host[MODEL_IDX_MASS] = mass
    model_host[MODEL_IDX_INV_MASS] = 1.0 / mass
    model_host[MODEL_IDX_RADIUS] = radius
    model_host[MODEL_IDX_IXX] = inertia
    model_host[MODEL_IDX_IYY] = inertia
    model_host[MODEL_IDX_IZZ] = inertia
    model_host[MODEL_IDX_INV_IXX] = 1.0 / inertia
    model_host[MODEL_IDX_INV_IYY] = 1.0 / inertia
    model_host[MODEL_IDX_INV_IZZ] = 1.0 / inertia

    var state_buf = ctx.enqueue_create_buffer[DType.float32](STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DType.float32](NUM_BODIES * MODEL_BODY_SIZE)
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())
    ctx.synchronize()

    # Run both for 100 steps
    for _ in range(100):
        ImpulseIntegrator.step(model_cpu, data_cpu)

    for _ in range(100):
        ImpulseIntegrator.step_gpu[DType.float32, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, BATCH](
            ctx, state_buf, model_buf,
            dt=Float32(0.005), gravity_z=Float32(-9.81), ground_z=Float32(-10.0),
            restitution=Float32(0.0), friction=Float32(0.0),
        )
    ctx.synchronize()

    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    copy_gpu_state_to_cpu[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](state_host, data_gpu)

    # Compare sensed values
    var angle_cpu = get_joint_angle(model_cpu, data_cpu, 0)
    var angle_gpu = get_joint_angle(model_gpu, data_gpu, 0)

    var omega_cpu = get_joint_angular_velocity(model_cpu, data_cpu, 0)
    var omega_gpu = get_joint_angular_velocity(model_gpu, data_gpu, 0)

    print("  CPU angle:", angle_cpu * 180.0 / PI, "deg")
    print("  GPU angle:", angle_gpu * 180.0 / PI, "deg")
    print("  Angle diff:", abs64(angle_cpu - angle_gpu) * 180.0 / PI, "deg")

    print("  CPU omega:", omega_cpu, "rad/s")
    print("  GPU omega:", omega_gpu, "rad/s")
    print("  Omega diff:", abs64(omega_cpu - omega_gpu), "rad/s")

    # Allow some tolerance due to float32 vs float64
    assert_true(
        abs64(angle_cpu - angle_gpu) < 0.2,  # ~11 degrees tolerance
        "Angle should be similar for CPU and GPU",
    )

    assert_true(
        abs64(omega_cpu - omega_gpu) < 1.0,
        "Angular velocity should be similar for CPU and GPU",
    )

    print("  PASSED: CPU vs GPU sensing parity")


fn main() raises:
    print("=" * 60)
    print("Joint Sensing GPU Tests (Phase 7, Step 7.2)")
    print("=" * 60)

    test_angle_sensing_after_gpu_simulation()
    test_angular_velocity_sensing_after_gpu()
    test_cpu_gpu_sensing_parity()

    print("\n" + "=" * 60)
    print("All GPU sensing tests passed!")
    print("=" * 60)
