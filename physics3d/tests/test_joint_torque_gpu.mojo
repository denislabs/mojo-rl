"""Test joint torque actuation on GPU (Phase 7, Step 7.1).

Tests:
1. GPU torque application causes angular velocity change
2. GPU vs CPU parity for torque application
3. Batched torque simulation
"""

from math import sqrt
from testing import assert_true

from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from physics3d import Model, Data, ImpulseIntegrator
from physics3d.gpu.constants import (
    compute_state_size,
    body_offset,
    joint_offset,
    metadata_offset,
    BODY_STATE_SIZE,
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
    JOINT_STATE_SIZE,
    JOINT_IDX_PARENT,
    JOINT_IDX_CHILD,
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
    META_IDX_NUM_CONTACTS,
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


fn abs32(x: Float32) -> Float32:
    if x < 0:
        return -x
    return x


fn test_gpu_torque_causes_angular_velocity() raises:
    """Test that GPU torque application causes angular velocity change."""
    print("Test 1: GPU torque causes angular velocity...")

    var ctx = DeviceContext()

    # Configuration
    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 5
    comptime MAX_JOINTS = 1
    comptime BATCH = 1
    comptime STATE_SIZE = compute_state_size[
        NUM_BODIES, MAX_CONTACTS, MAX_JOINTS
    ]()

    # Create state buffer
    var state_host = List[Float32](capacity=STATE_SIZE)
    for _ in range(STATE_SIZE):
        state_host.append(0.0)

    # Set body 0 position at (0, 0, 1)
    var b0 = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    state_host[b0 + BODY_IDX_PX] = 0.0
    state_host[b0 + BODY_IDX_PY] = 0.0
    state_host[b0 + BODY_IDX_PZ] = 1.0

    # Set identity quaternion
    state_host[b0 + BODY_IDX_QX] = 0.0
    state_host[b0 + BODY_IDX_QY] = 0.0
    state_host[b0 + BODY_IDX_QZ] = 0.0
    state_host[b0 + BODY_IDX_QW] = 1.0

    # Set joint 0: world -> body 0
    var j0 = joint_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    state_host[j0 + JOINT_IDX_PARENT] = -1.0  # World anchor
    state_host[j0 + JOINT_IDX_CHILD] = 0.0
    state_host[j0 + JOINT_IDX_ANCHOR_PX] = 0.0
    state_host[j0 + JOINT_IDX_ANCHOR_PY] = 0.0
    state_host[j0 + JOINT_IDX_ANCHOR_PZ] = 1.0
    state_host[j0 + JOINT_IDX_ANCHOR_CX] = 0.0
    state_host[j0 + JOINT_IDX_ANCHOR_CY] = 0.0
    state_host[j0 + JOINT_IDX_ANCHOR_CZ] = 0.0
    state_host[j0 + JOINT_IDX_AXIS_X] = 0.0
    state_host[j0 + JOINT_IDX_AXIS_Y] = 1.0  # Y-axis
    state_host[j0 + JOINT_IDX_AXIS_Z] = 0.0
    state_host[j0 + JOINT_IDX_TARGET_TORQUE] = 5.0  # 5 N·m
    state_host[j0 + JOINT_IDX_TORQUE_LIMIT] = 100.0

    # Set metadata
    var m_off = metadata_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()
    state_host[m_off + META_IDX_NUM_CONTACTS] = 0.0
    state_host[m_off + META_IDX_NUM_JOINTS] = 1.0

    # Create model buffer
    var model_host = List[Float32](capacity=NUM_BODIES * MODEL_BODY_SIZE)
    for _ in range(NUM_BODIES * MODEL_BODY_SIZE):
        model_host.append(0.0)

    # Set body 0 properties
    var mass: Float32 = 1.0
    var radius: Float32 = 0.1
    var inertia = 0.4 * mass * radius * radius
    var inv_inertia = 1.0 / inertia

    model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_MASS] = mass
    model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_INV_MASS] = 1.0 / mass
    model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_RADIUS] = radius
    model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_IXX] = inertia
    model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_IYY] = inertia
    model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_IZZ] = inertia
    model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_INV_IXX] = inv_inertia
    model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_INV_IYY] = inv_inertia
    model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_INV_IZZ] = inv_inertia

    # Copy to GPU
    var state_buf = ctx.enqueue_create_buffer[DType.float32](STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DType.float32](
        NUM_BODIES * MODEL_BODY_SIZE
    )
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())
    ctx.synchronize()

    # Record initial angular velocity
    var initial_wy = state_host[b0 + BODY_IDX_WY]
    print("  Initial wy:", initial_wy)

    # Run one physics step on GPU
    ImpulseIntegrator.step_gpu[
        DType.float32, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, BATCH
    ](
        ctx,
        state_buf,
        model_buf,
        dt=Float32(0.001),
        gravity_z=Float32(0.0),  # No gravity
        ground_z=Float32(-10.0),
        restitution=Float32(0.0),
        friction=Float32(0.0),
    )
    ctx.synchronize()

    # Copy back
    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    var final_wy = state_host[b0 + BODY_IDX_WY]
    print("  Final wy:", final_wy)
    print("  Delta wy:", final_wy - initial_wy)

    # With I = 0.004, inv_I = 250, tau = 5, dt = 0.001
    # delta_w = tau * inv_I * dt = 5 * 250 * 0.001 = 1.25 rad/s
    assert_true(
        abs32(final_wy - initial_wy) > 0.5,
        "GPU torque should cause significant angular velocity change",
    )

    print("  PASSED: GPU torque causes angular velocity")


fn test_gpu_cpu_torque_parity() raises:
    """Test that GPU and CPU give same results for torque application."""
    print("\nTest 2: GPU vs CPU torque parity...")

    var ctx = DeviceContext()

    # Configuration
    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 5
    comptime MAX_JOINTS = 1
    comptime BATCH = 1
    comptime STATE_SIZE = compute_state_size[
        NUM_BODIES, MAX_CONTACTS, MAX_JOINTS
    ]()
    comptime DTYPE_GPU = DType.float32
    comptime DTYPE_CPU = DType.float64

    # === CPU Setup ===
    var model_cpu = Model[DTYPE_CPU, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](
        gravity_z=0.0,
        timestep=0.001,
        ground_z=-10.0,
    )
    model_cpu.set_body(0, mass=1.0, radius=0.1)
    _ = model_cpu.add_hinge_joint(
        parent=-1,
        child=0,
        anchor_parent=(0.0, 0.0, 1.0),
        anchor_child=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
    )
    model_cpu.joints[0].set_torque(5.0)

    var data_cpu = Data[DTYPE_CPU, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()
    data_cpu.set_body_position(0, 0.0, 0.0, 1.0)

    # === GPU Setup ===
    var state_host = List[Float32](capacity=STATE_SIZE)
    for _ in range(STATE_SIZE):
        state_host.append(0.0)

    var b0 = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    state_host[b0 + BODY_IDX_PZ] = 1.0
    state_host[b0 + BODY_IDX_QW] = 1.0

    var j0 = joint_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    state_host[j0 + JOINT_IDX_PARENT] = -1.0
    state_host[j0 + JOINT_IDX_CHILD] = 0.0
    state_host[j0 + JOINT_IDX_ANCHOR_PZ] = 1.0
    state_host[j0 + JOINT_IDX_AXIS_Y] = 1.0
    state_host[j0 + JOINT_IDX_TARGET_TORQUE] = 5.0
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
    var model_buf = ctx.enqueue_create_buffer[DType.float32](
        NUM_BODIES * MODEL_BODY_SIZE
    )
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())
    ctx.synchronize()

    # Run 100 steps on both
    for _ in range(100):
        ImpulseIntegrator.step(model_cpu, data_cpu)

    for _ in range(100):
        ImpulseIntegrator.step_gpu[
            DType.float32, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, BATCH
        ](
            ctx,
            state_buf,
            model_buf,
            dt=Float32(0.001),
            gravity_z=Float32(0.0),
            ground_z=Float32(-10.0),
            restitution=Float32(0.0),
            friction=Float32(0.0),
        )
    ctx.synchronize()

    # Copy back GPU results
    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    # Compare positions
    var cpu_pos = data_cpu.get_body_position(0)
    var gpu_x = state_host[b0 + BODY_IDX_PX]
    var gpu_y = state_host[b0 + BODY_IDX_PY]
    var gpu_z = state_host[b0 + BODY_IDX_PZ]

    print("  CPU position:", cpu_pos[0], cpu_pos[1], cpu_pos[2])
    print("  GPU position:", gpu_x, gpu_y, gpu_z)

    var diff_x = abs32(Float32(cpu_pos[0]) - gpu_x)
    var diff_y = abs32(Float32(cpu_pos[1]) - gpu_y)
    var diff_z = abs32(Float32(cpu_pos[2]) - gpu_z)
    var total_diff = diff_x + diff_y + diff_z

    print("  Position difference:", total_diff)

    # Allow some tolerance due to float32 vs float64
    assert_true(
        total_diff < 0.1,
        "GPU and CPU positions should be similar",
    )

    print("  PASSED: GPU vs CPU torque parity")


fn test_gpu_batched_torque() raises:
    """Test batched torque simulation on GPU."""
    print("\nTest 3: Batched GPU torque simulation...")

    var ctx = DeviceContext()

    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 5
    comptime MAX_JOINTS = 1
    comptime BATCH = 16
    comptime STATE_SIZE = compute_state_size[
        NUM_BODIES, MAX_CONTACTS, MAX_JOINTS
    ]()

    # Create state buffer for all environments
    var state_host = List[Float32](capacity=BATCH * STATE_SIZE)
    for _ in range(BATCH * STATE_SIZE):
        state_host.append(0.0)

    # Initialize each environment with different torques
    for env in range(BATCH):
        var base = env * STATE_SIZE
        var b0 = base + body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
        var j0 = base + joint_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
        var m_off = (
            base + metadata_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()
        )

        # Body at (0, 0, 1)
        state_host[b0 + BODY_IDX_PZ] = 1.0
        state_host[b0 + BODY_IDX_QW] = 1.0

        # Joint with varying torque
        state_host[j0 + JOINT_IDX_PARENT] = -1.0
        state_host[j0 + JOINT_IDX_CHILD] = 0.0
        state_host[j0 + JOINT_IDX_ANCHOR_PZ] = 1.0
        state_host[j0 + JOINT_IDX_AXIS_Y] = 1.0
        state_host[j0 + JOINT_IDX_TARGET_TORQUE] = (
            Float32(env) * 0.5
        )  # 0, 0.5, 1.0, ...
        state_host[j0 + JOINT_IDX_TORQUE_LIMIT] = 100.0

        state_host[m_off + META_IDX_NUM_JOINTS] = 1.0

    # Create model buffer (shared across all envs)
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

    var state_buf = ctx.enqueue_create_buffer[DType.float32](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DType.float32](
        NUM_BODIES * MODEL_BODY_SIZE
    )
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())
    ctx.synchronize()

    # Run 50 steps
    for _ in range(50):
        ImpulseIntegrator.step_gpu[
            DType.float32, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, BATCH
        ](
            ctx,
            state_buf,
            model_buf,
            dt=Float32(0.01),
            gravity_z=Float32(0.0),
            ground_z=Float32(-10.0),
            restitution=Float32(0.0),
            friction=Float32(0.0),
        )
    ctx.synchronize()

    # Copy back
    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    # Check that higher torque environments rotated more
    print("  Angular velocities by environment:")
    var prev_wy: Float32 = -1000.0
    var all_increasing = True

    for env in range(BATCH):
        var base = env * STATE_SIZE
        var b0 = base + body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
        var wy = state_host[b0 + BODY_IDX_WY]

        if env < 5:
            print(
                "    Env", env, "(torque =", Float32(env) * 0.5, "): wy =", wy
            )

        if env > 0 and wy < prev_wy - 0.01:
            all_increasing = False

        prev_wy = wy

    print("  ...")
    print(
        "  All environments show increasing angular velocity:", all_increasing
    )

    assert_true(
        all_increasing,
        "Higher torque should result in higher angular velocity",
    )

    print("  PASSED: Batched GPU torque simulation")


fn main() raises:
    print("=" * 60)
    print("Joint Torque GPU Tests (Phase 7, Step 7.1)")
    print("=" * 60)

    test_gpu_torque_causes_angular_velocity()
    test_gpu_cpu_torque_parity()
    test_gpu_batched_torque()

    print("\n" + "=" * 60)
    print("All GPU torque tests passed!")
    print("=" * 60)
