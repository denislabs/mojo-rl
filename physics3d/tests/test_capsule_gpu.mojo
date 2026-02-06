"""Phase 8 Test: Capsule GPU Parity.

Tests that GPU capsule collision produces same results as CPU:
1. Capsule-plane collision (GPU vs CPU)
2. Capsule-sphere collision (GPU vs CPU)
3. Batched capsule simulation
"""

from math import sqrt, sin, cos, pi
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import LayoutTensor, Layout
from physics3d import Model, Data, ImpulseIntegrator
from physics3d.gpu.constants import (
    GEOM_SPHERE,
    GEOM_CAPSULE,
    compute_state_size,
    body_offset,
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
    MODEL_IDX_GEOM_TYPE,
    MODEL_IDX_HALF_LENGTH,
    META_IDX_NUM_CONTACTS,
)
from physics3d.gpu.buffer_utils import (
    create_model_host_buffer,
    copy_data_to_host_buffer,
    copy_host_buffer_to_data,
)


fn test_capsule_plane_gpu_parity() raises -> Bool:
    """Test GPU capsule-plane collision matches CPU."""
    print("Test: Capsule-plane GPU parity")

    comptime DTYPE = DType.float32  # GPU uses float32
    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 5
    comptime MAX_JOINTS = 0
    comptime BATCH = 1
    comptime STATE_SIZE = compute_state_size[
        NUM_BODIES, MAX_CONTACTS, MAX_JOINTS
    ]()

    var ctx = DeviceContext()

    # Create model
    var model_cpu = Model[DType.float64, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.001,
        ground_z=0.0,
        restitution=0.0,
        friction=0.5,
    )
    model_cpu.set_body_capsule(0, mass=1.0, radius=0.1, half_length=0.2)

    # Create CPU data
    var data_cpu = Data[DType.float64, NUM_BODIES, MAX_CONTACTS]()
    data_cpu.set_body_position(0, 0.0, 0.0, 1.0)

    # Create GPU model buffer
    var model_host = ctx.enqueue_create_host_buffer[DTYPE](
        NUM_BODIES * MODEL_BODY_SIZE
    )
    for i in range(NUM_BODIES):
        var base = i * MODEL_BODY_SIZE
        model_host[base + MODEL_IDX_MASS] = Float32(model_cpu.masses[i])
        model_host[base + MODEL_IDX_INV_MASS] = Float32(model_cpu.inv_masses[i])
        model_host[base + MODEL_IDX_RADIUS] = Float32(model_cpu.radii[i])
        model_host[base + MODEL_IDX_IXX] = Float32(
            model_cpu.inertias[i * 3 + 0]
        )
        model_host[base + MODEL_IDX_IYY] = Float32(
            model_cpu.inertias[i * 3 + 1]
        )
        model_host[base + MODEL_IDX_IZZ] = Float32(
            model_cpu.inertias[i * 3 + 2]
        )
        model_host[base + MODEL_IDX_INV_IXX] = Float32(
            model_cpu.inv_inertias[i * 3 + 0]
        )
        model_host[base + MODEL_IDX_INV_IYY] = Float32(
            model_cpu.inv_inertias[i * 3 + 1]
        )
        model_host[base + MODEL_IDX_INV_IZZ] = Float32(
            model_cpu.inv_inertias[i * 3 + 2]
        )
        model_host[base + MODEL_IDX_GEOM_TYPE] = Float32(
            model_cpu.geom_types[i]
        )
        model_host[base + MODEL_IDX_HALF_LENGTH] = Float32(
            model_cpu.half_lengths[i]
        )

    var model_buf = ctx.enqueue_create_buffer[DTYPE](
        NUM_BODIES * MODEL_BODY_SIZE
    )
    ctx.enqueue_copy(model_buf, model_host)

    # Create GPU state buffer
    var state_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * STATE_SIZE)
    var b_off = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    state_host[b_off + BODY_IDX_PX] = Float32(0.0)
    state_host[b_off + BODY_IDX_PY] = Float32(0.0)
    state_host[b_off + BODY_IDX_PZ] = Float32(1.0)
    state_host[b_off + BODY_IDX_QX] = Float32(0.0)
    state_host[b_off + BODY_IDX_QY] = Float32(0.0)
    state_host[b_off + BODY_IDX_QZ] = Float32(0.0)
    state_host[b_off + BODY_IDX_QW] = Float32(1.0)
    state_host[b_off + BODY_IDX_VX] = Float32(0.0)
    state_host[b_off + BODY_IDX_VY] = Float32(0.0)
    state_host[b_off + BODY_IDX_VZ] = Float32(0.0)

    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    ctx.enqueue_copy(state_buf, state_host)

    # Simulate CPU
    for _ in range(200):
        ImpulseIntegrator.step(model_cpu, data_cpu)
    var cpu_z = data_cpu.get_body_z(0)

    # Simulate GPU (call step_gpu in a loop)
    for _ in range(200):
        ImpulseIntegrator.step_gpu[
            DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, BATCH
        ](
            ctx,
            state_buf,
            model_buf,
            dt=Float32(0.001),
            gravity_z=Float32(-9.81),
            ground_z=Float32(0.0),
            restitution=Float32(0.0),
            friction=Float32(0.5),
        )

    # Read back GPU result
    ctx.enqueue_copy(state_host, state_buf)
    ctx.synchronize()
    var gpu_z = state_host[b_off + BODY_IDX_PZ]

    var diff = Float64(gpu_z) - Float64(cpu_z)
    if diff < 0:
        diff = -diff
    print("  CPU z:", cpu_z)
    print("  GPU z:", gpu_z)
    print("  Difference:", diff)

    # Allow 2cm tolerance for float32 vs float64 difference
    var passed = diff < 0.02
    if passed:
        print("  PASSED")
    else:
        print("  FAILED")
    return passed


fn test_capsule_sphere_gpu_parity() raises -> Bool:
    """Test GPU capsule-sphere collision matches CPU."""
    print("Test: Capsule-sphere GPU parity")

    comptime DTYPE = DType.float32
    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10
    comptime MAX_JOINTS = 0
    comptime BATCH = 1
    comptime STATE_SIZE = compute_state_size[
        NUM_BODIES, MAX_CONTACTS, MAX_JOINTS
    ]()

    var ctx = DeviceContext()

    # Create model with capsule and sphere
    var model_cpu = Model[DType.float64, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.001,
        ground_z=0.0,
        restitution=0.0,
        friction=0.5,
    )
    # Capsule resting on ground (horizontal)
    model_cpu.set_body_capsule(0, mass=10.0, radius=0.1, half_length=0.2)
    model_cpu.set_body(1, mass=1.0, radius=0.05)

    # Create CPU data
    var data_cpu = Data[DType.float64, NUM_BODIES, MAX_CONTACTS]()
    # Horizontal capsule at rest height
    data_cpu.set_body_position(0, 0.0, 0.0, 0.1)
    var angle = pi / 2.0
    var half_angle = angle / 2.0
    data_cpu.quaternions[0 * 4 + 0] = Float64(0.0)
    data_cpu.quaternions[0 * 4 + 1] = sin(half_angle)
    data_cpu.quaternions[0 * 4 + 2] = Float64(0.0)
    data_cpu.quaternions[0 * 4 + 3] = cos(half_angle)
    # Sphere above capsule
    data_cpu.set_body_position(1, 0.0, 0.0, 0.5)

    # Create GPU buffers
    var model_host = ctx.enqueue_create_host_buffer[DTYPE](
        NUM_BODIES * MODEL_BODY_SIZE
    )
    for i in range(NUM_BODIES):
        var base = i * MODEL_BODY_SIZE
        model_host[base + MODEL_IDX_MASS] = Float32(model_cpu.masses[i])
        model_host[base + MODEL_IDX_INV_MASS] = Float32(model_cpu.inv_masses[i])
        model_host[base + MODEL_IDX_RADIUS] = Float32(model_cpu.radii[i])
        model_host[base + MODEL_IDX_IXX] = Float32(
            model_cpu.inertias[i * 3 + 0]
        )
        model_host[base + MODEL_IDX_IYY] = Float32(
            model_cpu.inertias[i * 3 + 1]
        )
        model_host[base + MODEL_IDX_IZZ] = Float32(
            model_cpu.inertias[i * 3 + 2]
        )
        model_host[base + MODEL_IDX_INV_IXX] = Float32(
            model_cpu.inv_inertias[i * 3 + 0]
        )
        model_host[base + MODEL_IDX_INV_IYY] = Float32(
            model_cpu.inv_inertias[i * 3 + 1]
        )
        model_host[base + MODEL_IDX_INV_IZZ] = Float32(
            model_cpu.inv_inertias[i * 3 + 2]
        )
        model_host[base + MODEL_IDX_GEOM_TYPE] = Float32(
            model_cpu.geom_types[i]
        )
        model_host[base + MODEL_IDX_HALF_LENGTH] = Float32(
            model_cpu.half_lengths[i]
        )

    var model_buf = ctx.enqueue_create_buffer[DTYPE](
        NUM_BODIES * MODEL_BODY_SIZE
    )
    ctx.enqueue_copy(model_buf, model_host)

    # Create state buffer
    var state_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * STATE_SIZE)

    # Initialize capsule (body 0)
    var b0_off = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    state_host[b0_off + BODY_IDX_PX] = Float32(0.0)
    state_host[b0_off + BODY_IDX_PY] = Float32(0.0)
    state_host[b0_off + BODY_IDX_PZ] = Float32(0.1)
    state_host[b0_off + BODY_IDX_QX] = Float32(0.0)
    state_host[b0_off + BODY_IDX_QY] = Float32(sin(half_angle))
    state_host[b0_off + BODY_IDX_QZ] = Float32(0.0)
    state_host[b0_off + BODY_IDX_QW] = Float32(cos(half_angle))

    # Initialize sphere (body 1)
    var b1_off = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](1)
    state_host[b1_off + BODY_IDX_PX] = Float32(0.0)
    state_host[b1_off + BODY_IDX_PY] = Float32(0.0)
    state_host[b1_off + BODY_IDX_PZ] = Float32(0.5)
    state_host[b1_off + BODY_IDX_QX] = Float32(0.0)
    state_host[b1_off + BODY_IDX_QY] = Float32(0.0)
    state_host[b1_off + BODY_IDX_QZ] = Float32(0.0)
    state_host[b1_off + BODY_IDX_QW] = Float32(1.0)

    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    ctx.enqueue_copy(state_buf, state_host)

    # Simulate CPU
    for _ in range(300):
        ImpulseIntegrator.step(model_cpu, data_cpu)
    var cpu_z0 = data_cpu.positions[0 * 3 + 2]
    var cpu_z1 = data_cpu.positions[1 * 3 + 2]

    # Simulate GPU
    for _ in range(300):
        ImpulseIntegrator.step_gpu[
            DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, BATCH
        ](
            ctx,
            state_buf,
            model_buf,
            dt=Float32(0.001),
            gravity_z=Float32(-9.81),
            ground_z=Float32(0.0),
            restitution=Float32(0.0),
            friction=Float32(0.5),
        )

    # Read back GPU result
    ctx.enqueue_copy(state_host, state_buf)
    ctx.synchronize()
    var gpu_z0 = state_host[b0_off + BODY_IDX_PZ]
    var gpu_z1 = state_host[b1_off + BODY_IDX_PZ]

    var diff0 = Float64(gpu_z0) - Float64(cpu_z0)
    if diff0 < 0:
        diff0 = -diff0
    var diff1 = Float64(gpu_z1) - Float64(cpu_z1)
    if diff1 < 0:
        diff1 = -diff1

    print("  Capsule - CPU z:", cpu_z0, ", GPU z:", gpu_z0, ", diff:", diff0)
    print("  Sphere  - CPU z:", cpu_z1, ", GPU z:", gpu_z1, ", diff:", diff1)

    # Allow 3cm tolerance
    var passed = (diff0 < 0.03) and (diff1 < 0.03)
    if passed:
        print("  PASSED")
    else:
        print("  FAILED")
    return passed


fn test_batched_capsules() raises -> Bool:
    """Test batched capsule simulation on GPU."""
    print("Test: Batched capsule simulation")

    comptime DTYPE = DType.float32
    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 5
    comptime MAX_JOINTS = 0
    comptime BATCH = 16
    comptime STATE_SIZE = compute_state_size[
        NUM_BODIES, MAX_CONTACTS, MAX_JOINTS
    ]()

    var ctx = DeviceContext()

    # Create model buffer (same for all envs)
    var model_host = ctx.enqueue_create_host_buffer[DTYPE](
        NUM_BODIES * MODEL_BODY_SIZE
    )
    var mass: Float32 = 1.0
    var radius: Float32 = 0.1
    var half_len: Float32 = 0.2
    var inertia = Float32(0.4) * mass * radius * radius

    model_host[MODEL_IDX_MASS] = mass
    model_host[MODEL_IDX_INV_MASS] = Float32(1.0) / mass
    model_host[MODEL_IDX_RADIUS] = radius
    model_host[MODEL_IDX_IXX] = inertia
    model_host[MODEL_IDX_IYY] = inertia
    model_host[MODEL_IDX_IZZ] = inertia
    model_host[MODEL_IDX_INV_IXX] = Float32(1.0) / inertia
    model_host[MODEL_IDX_INV_IYY] = Float32(1.0) / inertia
    model_host[MODEL_IDX_INV_IZZ] = Float32(1.0) / inertia
    model_host[MODEL_IDX_GEOM_TYPE] = Float32(GEOM_CAPSULE)
    model_host[MODEL_IDX_HALF_LENGTH] = half_len

    var model_buf = ctx.enqueue_create_buffer[DTYPE](
        NUM_BODIES * MODEL_BODY_SIZE
    )
    ctx.enqueue_copy(model_buf, model_host)

    # Create state buffer with different initial heights (small spread to allow settling)
    var state_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * STATE_SIZE)
    for env in range(BATCH):
        var env_base = env * STATE_SIZE
        var b_off = env_base + body_offset[
            NUM_BODIES, MAX_CONTACTS, MAX_JOINTS
        ](0)
        var init_z = Float32(
            0.5 + Float64(env) * 0.02
        )  # 0.5, 0.52, 0.54, ... (small spread)

        state_host[b_off + BODY_IDX_PX] = Float32(0.0)
        state_host[b_off + BODY_IDX_PY] = Float32(0.0)
        state_host[b_off + BODY_IDX_PZ] = init_z
        state_host[b_off + BODY_IDX_QX] = Float32(0.0)
        state_host[b_off + BODY_IDX_QY] = Float32(0.0)
        state_host[b_off + BODY_IDX_QZ] = Float32(0.0)
        state_host[b_off + BODY_IDX_QW] = Float32(1.0)

    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    ctx.enqueue_copy(state_buf, state_host)

    # Simulate
    for _ in range(500):
        ImpulseIntegrator.step_gpu[
            DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, BATCH
        ](
            ctx,
            state_buf,
            model_buf,
            dt=Float32(0.001),
            gravity_z=Float32(-9.81),
            ground_z=Float32(0.0),
            restitution=Float32(0.0),
            friction=Float32(0.5),
        )

    # Read back results
    ctx.enqueue_copy(state_host, state_buf)
    ctx.synchronize()

    # Expected rest height for vertical capsule
    var expected_z = half_len + radius  # 0.3

    var all_correct = True
    print("  Final heights (expected ~", expected_z, "):")
    for env in range(BATCH):
        var env_base = env * STATE_SIZE
        var b_off = env_base + body_offset[
            NUM_BODIES, MAX_CONTACTS, MAX_JOINTS
        ](0)
        var z = state_host[b_off + BODY_IDX_PZ]
        var error = z - expected_z
        if error < 0:
            error = -error

        if env < 4 or env >= BATCH - 2:
            print("    Env", env, ": z =", z, ", error =", error)
        elif env == 4:
            print("    ...")

        if error > 0.02:
            all_correct = False

    if all_correct:
        print("  PASSED (all environments within tolerance)")
    else:
        print("  FAILED (some environments outside tolerance)")
    return all_correct


fn main() raises:
    print("=" * 60)
    print("Phase 8: Capsule GPU Parity Tests")
    print("=" * 60)

    var passed = 0
    var total = 3

    if test_capsule_plane_gpu_parity():
        passed += 1
    print()

    if test_capsule_sphere_gpu_parity():
        passed += 1
    print()

    if test_batched_capsules():
        passed += 1
    print()

    print("=" * 60)
    print("Results:", passed, "/", total, "tests passed")
    if passed == total:
        print("All GPU parity tests PASSED!")
    else:
        print("Some tests FAILED")
    print("=" * 60)
