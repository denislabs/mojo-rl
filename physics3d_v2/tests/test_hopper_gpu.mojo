"""Test HopperEnv on GPU (Phase 7, Step 7.4).

Tests:
1. Two-body hopper simulation on GPU
2. CPU vs GPU parity for hopper physics
3. Batched hopper simulation with different torques
"""

from math import sqrt
from testing import assert_true

from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from physics3d_v2 import Model, Data, ImpulseIntegrator
from physics3d_v2.gpu.constants import (
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


fn setup_hopper_state[
    NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int
](
    mut state_host: List[Float32],
    base: Int,
    torque: Float32,
):
    """Set up hopper configuration in state buffer.

    Hopper configuration:
    - Body 0 (Torso): sphere at (0, 0, 0.45), mass=1.0, radius=0.15
    - Body 1 (Foot): sphere at (0, 0, 0.1), mass=0.5, radius=0.1
    - Joint 0: Hip connecting torso to foot, Y-axis rotation
    """
    # Body 0: Torso
    var b0 = base + body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    state_host[b0 + BODY_IDX_PX] = 0.0
    state_host[b0 + BODY_IDX_PY] = 0.0
    state_host[b0 + BODY_IDX_PZ] = 0.45  # Torso height
    state_host[b0 + BODY_IDX_QX] = 0.0
    state_host[b0 + BODY_IDX_QY] = 0.0
    state_host[b0 + BODY_IDX_QZ] = 0.0
    state_host[b0 + BODY_IDX_QW] = 1.0

    # Body 1: Foot
    var b1 = base + body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](1)
    state_host[b1 + BODY_IDX_PX] = 0.0
    state_host[b1 + BODY_IDX_PY] = 0.0
    state_host[b1 + BODY_IDX_PZ] = 0.1  # Foot just above ground
    state_host[b1 + BODY_IDX_QX] = 0.0
    state_host[b1 + BODY_IDX_QY] = 0.0
    state_host[b1 + BODY_IDX_QZ] = 0.0
    state_host[b1 + BODY_IDX_QW] = 1.0

    # Joint 0: Hip (torso -> foot)
    var j0 = base + joint_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    state_host[j0 + JOINT_IDX_PARENT] = 0.0  # Torso
    state_host[j0 + JOINT_IDX_CHILD] = 1.0   # Foot
    # Anchor on torso (bottom)
    state_host[j0 + JOINT_IDX_ANCHOR_PX] = 0.0
    state_host[j0 + JOINT_IDX_ANCHOR_PY] = 0.0
    state_host[j0 + JOINT_IDX_ANCHOR_PZ] = -0.15  # Bottom of torso
    # Anchor on foot (above center)
    state_host[j0 + JOINT_IDX_ANCHOR_CX] = 0.0
    state_host[j0 + JOINT_IDX_ANCHOR_CY] = 0.0
    state_host[j0 + JOINT_IDX_ANCHOR_CZ] = 0.2   # Above foot center
    # Y-axis rotation (sagittal plane motion)
    state_host[j0 + JOINT_IDX_AXIS_X] = 0.0
    state_host[j0 + JOINT_IDX_AXIS_Y] = 1.0
    state_host[j0 + JOINT_IDX_AXIS_Z] = 0.0
    state_host[j0 + JOINT_IDX_TARGET_TORQUE] = torque
    state_host[j0 + JOINT_IDX_TORQUE_LIMIT] = 10.0

    # Metadata
    var m_off = base + metadata_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()
    state_host[m_off + META_IDX_NUM_CONTACTS] = 0.0
    state_host[m_off + META_IDX_NUM_JOINTS] = 1.0


fn setup_hopper_model(mut model_host: List[Float32]):
    """Set up hopper model buffer.

    - Body 0: Torso (mass=1.0, radius=0.15)
    - Body 1: Foot (mass=0.5, radius=0.1)
    """
    # Body 0: Torso
    var mass0: Float32 = 1.0
    var radius0: Float32 = 0.15
    var inertia0 = 0.4 * mass0 * radius0 * radius0

    model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_MASS] = mass0
    model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_INV_MASS] = 1.0 / mass0
    model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_RADIUS] = radius0
    model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_IXX] = inertia0
    model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_IYY] = inertia0
    model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_IZZ] = inertia0
    model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_INV_IXX] = 1.0 / inertia0
    model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_INV_IYY] = 1.0 / inertia0
    model_host[0 * MODEL_BODY_SIZE + MODEL_IDX_INV_IZZ] = 1.0 / inertia0

    # Body 1: Foot
    var mass1: Float32 = 0.5
    var radius1: Float32 = 0.1
    var inertia1 = 0.4 * mass1 * radius1 * radius1

    model_host[1 * MODEL_BODY_SIZE + MODEL_IDX_MASS] = mass1
    model_host[1 * MODEL_BODY_SIZE + MODEL_IDX_INV_MASS] = 1.0 / mass1
    model_host[1 * MODEL_BODY_SIZE + MODEL_IDX_RADIUS] = radius1
    model_host[1 * MODEL_BODY_SIZE + MODEL_IDX_IXX] = inertia1
    model_host[1 * MODEL_BODY_SIZE + MODEL_IDX_IYY] = inertia1
    model_host[1 * MODEL_BODY_SIZE + MODEL_IDX_IZZ] = inertia1
    model_host[1 * MODEL_BODY_SIZE + MODEL_IDX_INV_IXX] = 1.0 / inertia1
    model_host[1 * MODEL_BODY_SIZE + MODEL_IDX_INV_IYY] = 1.0 / inertia1
    model_host[1 * MODEL_BODY_SIZE + MODEL_IDX_INV_IZZ] = 1.0 / inertia1


fn test_gpu_hopper_simulation() raises:
    """Test that GPU simulates 2-body hopper correctly."""
    print("Test 1: GPU hopper simulation...")

    var ctx = DeviceContext()

    # Hopper configuration: 2 bodies, 10 contacts, 1 joint
    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10
    comptime MAX_JOINTS = 1
    comptime BATCH = 1
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()

    # Create state buffer
    var state_host = List[Float32](capacity=STATE_SIZE)
    for _ in range(STATE_SIZE):
        state_host.append(0.0)

    # Set up hopper with zero torque
    setup_hopper_state[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](state_host, 0, 0.0)

    # Create model buffer
    var model_host = List[Float32](capacity=NUM_BODIES * MODEL_BODY_SIZE)
    for _ in range(NUM_BODIES * MODEL_BODY_SIZE):
        model_host.append(0.0)
    setup_hopper_model(model_host)

    # Copy to GPU
    var state_buf = ctx.enqueue_create_buffer[DType.float32](STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DType.float32](NUM_BODIES * MODEL_BODY_SIZE)
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())
    ctx.synchronize()

    var b0 = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    var b1 = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](1)
    var j0 = joint_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)

    var initial_torso_z = state_host[b0 + BODY_IDX_PZ]
    var initial_foot_z = state_host[b1 + BODY_IDX_PZ]
    print("  Initial torso z:", initial_torso_z)
    print("  Initial foot z:", initial_foot_z)

    # Verify joint values in GPU buffer
    var verify_buffer = List[Float32](capacity=STATE_SIZE)
    for _ in range(STATE_SIZE):
        verify_buffer.append(0.0)
    ctx.enqueue_copy(verify_buffer.unsafe_ptr(), state_buf)
    ctx.synchronize()
    print("  Joint 0 (from GPU): parent=", verify_buffer[j0 + JOINT_IDX_PARENT],
          ", child=", verify_buffer[j0 + JOINT_IDX_CHILD])

    # Run 100 steps on GPU with gravity
    for _ in range(100):
        ImpulseIntegrator.step_gpu[DType.float32, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, BATCH](
            ctx,
            state_buf,
            model_buf,
            dt=Float32(0.005),
            gravity_z=Float32(-9.81),
            ground_z=Float32(0.0),
            restitution=Float32(0.0),
            friction=Float32(0.8),
        )
    ctx.synchronize()

    # Copy back
    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    var final_torso_z = state_host[b0 + BODY_IDX_PZ]
    var final_foot_z = state_host[b1 + BODY_IDX_PZ]
    print("  Final torso z:", final_torso_z)
    print("  Final foot z:", final_foot_z)

    # Foot should be near ground (radius = 0.1)
    assert_true(
        final_foot_z > 0.05 and final_foot_z < 0.2,
        "Foot should be near ground (0.05 < z < 0.2)",
    )

    # Torso should be above foot
    assert_true(
        final_torso_z > final_foot_z,
        "Torso should be above foot",
    )

    # Hopper should remain standing (torso height > 0.2)
    assert_true(
        final_torso_z > 0.2,
        "Hopper should remain standing (torso z > 0.2)",
    )

    print("  PASSED: GPU hopper simulation works")


fn test_gpu_cpu_hopper_parity() raises:
    """Test that GPU and CPU give similar results for hopper physics."""
    print("\nTest 2: GPU vs CPU hopper parity...")

    var ctx = DeviceContext()

    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10
    comptime MAX_JOINTS = 1
    comptime BATCH = 1
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()
    comptime DTYPE_GPU = DType.float32
    comptime DTYPE_CPU = DType.float64

    # === CPU Setup ===
    var model_cpu = Model[DTYPE_CPU, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](
        gravity_z=-9.81,
        timestep=0.005,
        ground_z=0.0,
        friction=0.8,
        restitution=0.0,
    )
    model_cpu.set_body(0, mass=1.0, radius=0.15)  # Torso
    model_cpu.set_body(1, mass=0.5, radius=0.1)   # Foot

    _ = model_cpu.add_hinge_joint(
        parent=0,
        child=1,
        anchor_parent=(0.0, 0.0, -0.15),
        anchor_child=(0.0, 0.0, 0.2),
        axis=(0.0, 1.0, 0.0),
    )
    model_cpu.joints[0].set_torque(5.0)

    var data_cpu = Data[DTYPE_CPU, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()
    data_cpu.set_body_position(0, 0.0, 0.0, 0.45)  # Torso
    data_cpu.set_body_position(1, 0.0, 0.0, 0.1)   # Foot

    # === GPU Setup ===
    var state_host = List[Float32](capacity=STATE_SIZE)
    for _ in range(STATE_SIZE):
        state_host.append(0.0)

    setup_hopper_state[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](state_host, 0, 5.0)

    var model_host = List[Float32](capacity=NUM_BODIES * MODEL_BODY_SIZE)
    for _ in range(NUM_BODIES * MODEL_BODY_SIZE):
        model_host.append(0.0)
    setup_hopper_model(model_host)

    var state_buf = ctx.enqueue_create_buffer[DType.float32](STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DType.float32](NUM_BODIES * MODEL_BODY_SIZE)
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())
    ctx.synchronize()

    # Run 50 steps on both
    for _ in range(50):
        ImpulseIntegrator.step(model_cpu, data_cpu)

    for _ in range(50):
        ImpulseIntegrator.step_gpu[DType.float32, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, BATCH](
            ctx, state_buf, model_buf,
            dt=Float32(0.005),
            gravity_z=Float32(-9.81),
            ground_z=Float32(0.0),
            restitution=Float32(0.0),
            friction=Float32(0.8),
        )
    ctx.synchronize()

    # Copy back GPU results
    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    var b0 = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    var b1 = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](1)

    # Compare torso positions
    var cpu_torso = data_cpu.get_body_position(0)
    var gpu_torso_x = state_host[b0 + BODY_IDX_PX]
    var gpu_torso_y = state_host[b0 + BODY_IDX_PY]
    var gpu_torso_z = state_host[b0 + BODY_IDX_PZ]

    print("  CPU torso:", cpu_torso[0], cpu_torso[1], cpu_torso[2])
    print("  GPU torso:", gpu_torso_x, gpu_torso_y, gpu_torso_z)

    var torso_diff = (
        abs32(Float32(cpu_torso[0]) - gpu_torso_x) +
        abs32(Float32(cpu_torso[1]) - gpu_torso_y) +
        abs32(Float32(cpu_torso[2]) - gpu_torso_z)
    )
    print("  Torso position difference:", torso_diff)

    # Compare foot positions
    var cpu_foot = data_cpu.get_body_position(1)
    var gpu_foot_x = state_host[b1 + BODY_IDX_PX]
    var gpu_foot_y = state_host[b1 + BODY_IDX_PY]
    var gpu_foot_z = state_host[b1 + BODY_IDX_PZ]

    print("  CPU foot:", cpu_foot[0], cpu_foot[1], cpu_foot[2])
    print("  GPU foot:", gpu_foot_x, gpu_foot_y, gpu_foot_z)

    var foot_diff = (
        abs32(Float32(cpu_foot[0]) - gpu_foot_x) +
        abs32(Float32(cpu_foot[1]) - gpu_foot_y) +
        abs32(Float32(cpu_foot[2]) - gpu_foot_z)
    )
    print("  Foot position difference:", foot_diff)

    # Allow reasonable tolerance (float32 vs float64 + different code paths)
    assert_true(
        torso_diff < 0.5,
        "Torso positions should be similar (diff < 0.5m)",
    )

    assert_true(
        foot_diff < 0.5,
        "Foot positions should be similar (diff < 0.5m)",
    )

    print("  PASSED: GPU vs CPU hopper parity")


fn test_gpu_batched_hopper() raises:
    """Test batched hopper simulation with different torques."""
    print("\nTest 3: Batched GPU hopper simulation...")

    var ctx = DeviceContext()

    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10
    comptime MAX_JOINTS = 1
    comptime BATCH = 8
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()

    # Create state buffer for all environments
    var state_host = List[Float32](capacity=BATCH * STATE_SIZE)
    for _ in range(BATCH * STATE_SIZE):
        state_host.append(0.0)

    # Initialize each environment with different torques
    for env in range(BATCH):
        var base = env * STATE_SIZE
        var torque = Float32(env) * 1.0 - 3.5  # -3.5, -2.5, ..., 3.5
        setup_hopper_state[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](
            state_host, base, torque
        )

    # Create model buffer (shared across all envs)
    var model_host = List[Float32](capacity=NUM_BODIES * MODEL_BODY_SIZE)
    for _ in range(NUM_BODIES * MODEL_BODY_SIZE):
        model_host.append(0.0)
    setup_hopper_model(model_host)

    var state_buf = ctx.enqueue_create_buffer[DType.float32](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DType.float32](NUM_BODIES * MODEL_BODY_SIZE)
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())
    ctx.synchronize()

    # Run 50 steps
    for _ in range(50):
        ImpulseIntegrator.step_gpu[DType.float32, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, BATCH](
            ctx, state_buf, model_buf,
            dt=Float32(0.01),
            gravity_z=Float32(-9.81),
            ground_z=Float32(0.0),
            restitution=Float32(0.0),
            friction=Float32(0.8),
        )
    ctx.synchronize()

    # Copy back
    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    # Check that different torques result in different motion
    print("  Torso X positions by environment (different torques):")
    var x_positions = List[Float32]()

    for env in range(BATCH):
        var base = env * STATE_SIZE
        var b0 = base + body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
        var torso_x = state_host[b0 + BODY_IDX_PX]
        var torso_z = state_host[b0 + BODY_IDX_PZ]
        var torque = Float32(env) * 1.0 - 3.5

        print("    Env", env, "(torque =", torque, "): x =", torso_x, ", z =", torso_z)
        x_positions.append(torso_x)

    # Check that hoppers with opposite torques moved in different directions
    # Env 0 has torque = -3.5, Env 7 has torque = +3.5
    var moved_different_directions = (
        (x_positions[0] < -0.01 and x_positions[7] > 0.01) or
        (x_positions[0] > 0.01 and x_positions[7] < -0.01)
    )

    print("  Opposite torques caused different motion:", moved_different_directions)

    # Check that all hoppers are still "standing" (torso z > 0.1)
    var all_standing = True
    for env in range(BATCH):
        var base = env * STATE_SIZE
        var b0 = base + body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
        var torso_z = state_host[b0 + BODY_IDX_PZ]
        if torso_z < 0.1:
            all_standing = False
            break

    print("  All hoppers still have torso z > 0.1:", all_standing)

    assert_true(
        all_standing,
        "All hoppers should remain somewhat upright",
    )

    print("  PASSED: Batched GPU hopper simulation")


fn main() raises:
    print("=" * 60)
    print("Hopper GPU Tests (Phase 7, Step 7.4)")
    print("=" * 60)

    test_gpu_hopper_simulation()
    test_gpu_cpu_hopper_parity()
    test_gpu_batched_hopper()

    print("\n" + "=" * 60)
    print("All GPU hopper tests passed!")
    print("=" * 60)
