"""Phase 9 Test: Box GPU Parity.

Tests that GPU box collision produces same results as CPU:
1. Box-plane collision (GPU vs CPU)
2. Box-sphere collision (GPU vs CPU)
3. Box-box collision (GPU vs CPU)
"""

from math import sqrt, sin, cos, pi
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import LayoutTensor, Layout
from physics3d_v2 import Model, Data, ImpulseIntegrator
from physics3d_v2.gpu.constants import (
    GEOM_SPHERE,
    GEOM_BOX,
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
    MODEL_IDX_HALF_X,
    MODEL_IDX_HALF_Y,
    MODEL_IDX_HALF_Z,
    META_IDX_NUM_CONTACTS,
)
from physics3d_v2.gpu.buffer_utils import (
    create_model_host_buffer,
    copy_data_to_host_buffer,
    copy_host_buffer_to_data,
)


fn test_box_plane_gpu_parity() raises -> Bool:
    """Test GPU box-plane collision matches CPU."""
    print("Test: Box-plane GPU parity")

    comptime DTYPE = DType.float32  # GPU uses float32
    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 5
    comptime MAX_JOINTS = 0
    comptime BATCH = 1
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()

    var ctx = DeviceContext()

    # Create model
    var model_cpu = Model[DType.float64, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.001,
        ground_z=0.0,
        restitution=0.0,
        friction=0.5,
    )
    model_cpu.set_body_box(0, mass=1.0, half_x=0.1, half_y=0.1, half_z=0.2)

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
        model_host[base + MODEL_IDX_IXX] = Float32(model_cpu.inertias[i * 3 + 0])
        model_host[base + MODEL_IDX_IYY] = Float32(model_cpu.inertias[i * 3 + 1])
        model_host[base + MODEL_IDX_IZZ] = Float32(model_cpu.inertias[i * 3 + 2])
        model_host[base + MODEL_IDX_INV_IXX] = Float32(model_cpu.inv_inertias[i * 3 + 0])
        model_host[base + MODEL_IDX_INV_IYY] = Float32(model_cpu.inv_inertias[i * 3 + 1])
        model_host[base + MODEL_IDX_INV_IZZ] = Float32(model_cpu.inv_inertias[i * 3 + 2])
        model_host[base + MODEL_IDX_GEOM_TYPE] = Float32(model_cpu.geom_types[i])
        model_host[base + MODEL_IDX_HALF_LENGTH] = Float32(model_cpu.half_lengths[i])
        model_host[base + MODEL_IDX_HALF_X] = Float32(model_cpu.half_x[i])
        model_host[base + MODEL_IDX_HALF_Y] = Float32(model_cpu.half_y[i])
        model_host[base + MODEL_IDX_HALF_Z] = Float32(model_cpu.half_z[i])

    var model_buf = ctx.enqueue_create_buffer[DTYPE](NUM_BODIES * MODEL_BODY_SIZE)
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

    # Simulate GPU
    for _ in range(200):
        ImpulseIntegrator.step_gpu[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, BATCH](
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

    # Allow 3cm tolerance for float32 vs float64 difference
    var passed = diff < 0.03
    if passed:
        print("  PASSED")
    else:
        print("  FAILED")
    return passed


fn test_box_sphere_gpu_parity() raises -> Bool:
    """Test GPU box-sphere collision matches CPU."""
    print("Test: Box-sphere GPU parity")

    comptime DTYPE = DType.float32
    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10
    comptime MAX_JOINTS = 0
    comptime BATCH = 1
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()

    var ctx = DeviceContext()

    # Create model with box and sphere
    var model_cpu = Model[DType.float64, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.001,
        ground_z=0.0,
        restitution=0.0,
        friction=0.5,
    )
    # Box at rest (heavy)
    model_cpu.set_body_box(0, mass=100.0, half_x=0.3, half_y=0.3, half_z=0.1)
    # Sphere falling
    model_cpu.set_body(1, mass=1.0, radius=0.1)

    # Create CPU data
    var data_cpu = Data[DType.float64, NUM_BODIES, MAX_CONTACTS]()
    data_cpu.set_body_position(0, 0.0, 0.0, 0.1)  # Box at rest on ground
    data_cpu.set_body_position(1, 0.0, 0.0, 0.8)  # Sphere above

    # Create GPU buffers
    var model_host = ctx.enqueue_create_host_buffer[DTYPE](NUM_BODIES * MODEL_BODY_SIZE)
    for i in range(NUM_BODIES):
        var base = i * MODEL_BODY_SIZE
        model_host[base + MODEL_IDX_MASS] = Float32(model_cpu.masses[i])
        model_host[base + MODEL_IDX_INV_MASS] = Float32(model_cpu.inv_masses[i])
        model_host[base + MODEL_IDX_RADIUS] = Float32(model_cpu.radii[i])
        model_host[base + MODEL_IDX_IXX] = Float32(model_cpu.inertias[i * 3 + 0])
        model_host[base + MODEL_IDX_IYY] = Float32(model_cpu.inertias[i * 3 + 1])
        model_host[base + MODEL_IDX_IZZ] = Float32(model_cpu.inertias[i * 3 + 2])
        model_host[base + MODEL_IDX_INV_IXX] = Float32(model_cpu.inv_inertias[i * 3 + 0])
        model_host[base + MODEL_IDX_INV_IYY] = Float32(model_cpu.inv_inertias[i * 3 + 1])
        model_host[base + MODEL_IDX_INV_IZZ] = Float32(model_cpu.inv_inertias[i * 3 + 2])
        model_host[base + MODEL_IDX_GEOM_TYPE] = Float32(model_cpu.geom_types[i])
        model_host[base + MODEL_IDX_HALF_LENGTH] = Float32(model_cpu.half_lengths[i])
        model_host[base + MODEL_IDX_HALF_X] = Float32(model_cpu.half_x[i])
        model_host[base + MODEL_IDX_HALF_Y] = Float32(model_cpu.half_y[i])
        model_host[base + MODEL_IDX_HALF_Z] = Float32(model_cpu.half_z[i])

    var model_buf = ctx.enqueue_create_buffer[DTYPE](NUM_BODIES * MODEL_BODY_SIZE)
    ctx.enqueue_copy(model_buf, model_host)

    # Create state buffer
    var state_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * STATE_SIZE)

    # Initialize box (body 0)
    var b0_off = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    state_host[b0_off + BODY_IDX_PX] = Float32(0.0)
    state_host[b0_off + BODY_IDX_PY] = Float32(0.0)
    state_host[b0_off + BODY_IDX_PZ] = Float32(0.1)
    state_host[b0_off + BODY_IDX_QX] = Float32(0.0)
    state_host[b0_off + BODY_IDX_QY] = Float32(0.0)
    state_host[b0_off + BODY_IDX_QZ] = Float32(0.0)
    state_host[b0_off + BODY_IDX_QW] = Float32(1.0)

    # Initialize sphere (body 1)
    var b1_off = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](1)
    state_host[b1_off + BODY_IDX_PX] = Float32(0.0)
    state_host[b1_off + BODY_IDX_PY] = Float32(0.0)
    state_host[b1_off + BODY_IDX_PZ] = Float32(0.8)
    state_host[b1_off + BODY_IDX_QX] = Float32(0.0)
    state_host[b1_off + BODY_IDX_QY] = Float32(0.0)
    state_host[b1_off + BODY_IDX_QZ] = Float32(0.0)
    state_host[b1_off + BODY_IDX_QW] = Float32(1.0)

    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    ctx.enqueue_copy(state_buf, state_host)

    # Simulate CPU
    for _ in range(300):
        ImpulseIntegrator.step(model_cpu, data_cpu)
    var cpu_z1 = data_cpu.positions[1 * 3 + 2]

    # Simulate GPU
    for _ in range(300):
        ImpulseIntegrator.step_gpu[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, BATCH](
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
    var gpu_z1 = state_host[b1_off + BODY_IDX_PZ]

    var diff = Float64(gpu_z1) - Float64(cpu_z1)
    if diff < 0:
        diff = -diff

    print("  Sphere - CPU z:", cpu_z1, ", GPU z:", gpu_z1, ", diff:", diff)

    # Allow 3cm tolerance
    var passed = diff < 0.03
    if passed:
        print("  PASSED")
    else:
        print("  FAILED")
    return passed


fn test_box_box_gpu_parity() raises -> Bool:
    """Test GPU box-box collision matches CPU."""
    print("Test: Box-box GPU parity")

    comptime DTYPE = DType.float32
    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10
    comptime MAX_JOINTS = 0
    comptime BATCH = 1
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()

    var ctx = DeviceContext()

    # Create model with two boxes
    var model_cpu = Model[DType.float64, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.001,
        ground_z=0.0,
        restitution=0.0,
        friction=0.5,
    )
    # Large box at rest (heavy)
    model_cpu.set_body_box(0, mass=100.0, half_x=0.3, half_y=0.3, half_z=0.1)
    # Small box falling
    model_cpu.set_body_box(1, mass=1.0, half_x=0.1, half_y=0.1, half_z=0.1)

    # Create CPU data
    var data_cpu = Data[DType.float64, NUM_BODIES, MAX_CONTACTS]()
    data_cpu.set_body_position(0, 0.0, 0.0, 0.1)  # Big box at rest on ground
    data_cpu.set_body_position(1, 0.0, 0.0, 0.8)  # Small box above

    # Create GPU buffers
    var model_host = ctx.enqueue_create_host_buffer[DTYPE](NUM_BODIES * MODEL_BODY_SIZE)
    for i in range(NUM_BODIES):
        var base = i * MODEL_BODY_SIZE
        model_host[base + MODEL_IDX_MASS] = Float32(model_cpu.masses[i])
        model_host[base + MODEL_IDX_INV_MASS] = Float32(model_cpu.inv_masses[i])
        model_host[base + MODEL_IDX_RADIUS] = Float32(model_cpu.radii[i])
        model_host[base + MODEL_IDX_IXX] = Float32(model_cpu.inertias[i * 3 + 0])
        model_host[base + MODEL_IDX_IYY] = Float32(model_cpu.inertias[i * 3 + 1])
        model_host[base + MODEL_IDX_IZZ] = Float32(model_cpu.inertias[i * 3 + 2])
        model_host[base + MODEL_IDX_INV_IXX] = Float32(model_cpu.inv_inertias[i * 3 + 0])
        model_host[base + MODEL_IDX_INV_IYY] = Float32(model_cpu.inv_inertias[i * 3 + 1])
        model_host[base + MODEL_IDX_INV_IZZ] = Float32(model_cpu.inv_inertias[i * 3 + 2])
        model_host[base + MODEL_IDX_GEOM_TYPE] = Float32(model_cpu.geom_types[i])
        model_host[base + MODEL_IDX_HALF_LENGTH] = Float32(model_cpu.half_lengths[i])
        model_host[base + MODEL_IDX_HALF_X] = Float32(model_cpu.half_x[i])
        model_host[base + MODEL_IDX_HALF_Y] = Float32(model_cpu.half_y[i])
        model_host[base + MODEL_IDX_HALF_Z] = Float32(model_cpu.half_z[i])

    var model_buf = ctx.enqueue_create_buffer[DTYPE](NUM_BODIES * MODEL_BODY_SIZE)
    ctx.enqueue_copy(model_buf, model_host)

    # Create state buffer
    var state_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * STATE_SIZE)

    # Initialize big box (body 0)
    var b0_off = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    state_host[b0_off + BODY_IDX_PX] = Float32(0.0)
    state_host[b0_off + BODY_IDX_PY] = Float32(0.0)
    state_host[b0_off + BODY_IDX_PZ] = Float32(0.1)
    state_host[b0_off + BODY_IDX_QX] = Float32(0.0)
    state_host[b0_off + BODY_IDX_QY] = Float32(0.0)
    state_host[b0_off + BODY_IDX_QZ] = Float32(0.0)
    state_host[b0_off + BODY_IDX_QW] = Float32(1.0)

    # Initialize small box (body 1)
    var b1_off = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](1)
    state_host[b1_off + BODY_IDX_PX] = Float32(0.0)
    state_host[b1_off + BODY_IDX_PY] = Float32(0.0)
    state_host[b1_off + BODY_IDX_PZ] = Float32(0.8)
    state_host[b1_off + BODY_IDX_QX] = Float32(0.0)
    state_host[b1_off + BODY_IDX_QY] = Float32(0.0)
    state_host[b1_off + BODY_IDX_QZ] = Float32(0.0)
    state_host[b1_off + BODY_IDX_QW] = Float32(1.0)

    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    ctx.enqueue_copy(state_buf, state_host)

    # Simulate CPU
    for _ in range(300):
        ImpulseIntegrator.step(model_cpu, data_cpu)
    var cpu_z1 = data_cpu.positions[1 * 3 + 2]

    # Simulate GPU
    for _ in range(300):
        ImpulseIntegrator.step_gpu[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, BATCH](
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
    var gpu_z1 = state_host[b1_off + BODY_IDX_PZ]

    var diff = Float64(gpu_z1) - Float64(cpu_z1)
    if diff < 0:
        diff = -diff

    print("  Small box - CPU z:", cpu_z1, ", GPU z:", gpu_z1, ", diff:", diff)

    # Allow 3cm tolerance
    var passed = diff < 0.03
    if passed:
        print("  PASSED")
    else:
        print("  FAILED")
    return passed


fn main() raises:
    print("=" * 60)
    print("Phase 9: Box GPU Parity Tests")
    print("=" * 60)

    var passed = 0
    var total = 3

    if test_box_plane_gpu_parity():
        passed += 1
    print()

    if test_box_sphere_gpu_parity():
        passed += 1
    print()

    if test_box_box_gpu_parity():
        passed += 1
    print()

    print("=" * 60)
    print("Results:", passed, "/", total, "tests passed")
    if passed == total:
        print("All box GPU parity tests PASSED!")
    else:
        print("Some tests FAILED")
    print("=" * 60)
