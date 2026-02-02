"""GPU Slide Joint Test.

Tests the GPU slide joint solver by running a simulation on GPU.

Run with:
    cd mojo-rl
    pixi run -e apple mojo run physics3d_v2/tests/test_slide_joint_gpu.mojo
"""

from math import sqrt, sin, cos
from gpu.host import DeviceContext, DeviceBuffer
from physics3d_v2.integrator import PGSIntegrator
from physics3d_v2.gpu.constants import (
    compute_state_size,
    body_offset,
    slide_joint_offset,
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
    SLIDE_JOINT_STATE_SIZE,
    SLIDE_IDX_PARENT,
    SLIDE_IDX_CHILD,
    SLIDE_IDX_ANCHOR_PX,
    SLIDE_IDX_ANCHOR_PY,
    SLIDE_IDX_ANCHOR_PZ,
    SLIDE_IDX_ANCHOR_CX,
    SLIDE_IDX_ANCHOR_CY,
    SLIDE_IDX_ANCHOR_CZ,
    SLIDE_IDX_AXIS_X,
    SLIDE_IDX_AXIS_Y,
    SLIDE_IDX_AXIS_Z,
    SLIDE_IDX_TARGET_FORCE,
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

# Configuration
comptime NUM_BODIES: Int = 1
comptime MAX_CONTACTS: Int = 5
comptime MAX_JOINTS: Int = 0  # No hinge joints
comptime MAX_SLIDE_JOINTS: Int = 1
comptime BATCH: Int = 1
comptime DTYPE = DType.float32


fn abs_val(x: Float32) -> Float32:
    if x < 0:
        return -x
    return x


fn max_val(a: Float32, b: Float32) -> Float32:
    if a > b:
        return a
    return b


fn main() raises:
    print("=" * 60)
    print("    GPU Slide Joint Test")
    print("=" * 60)
    print()

    # Physics parameters
    var mass: Float32 = 1.0
    var radius: Float32 = 0.1
    var initial_x: Float32 = 1.0
    var anchor_z: Float32 = 0.5
    var dt: Float32 = 0.001
    var gravity_z: Float32 = -9.81  # Gravity perpendicular to slide axis
    var ground_z: Float32 = -10.0
    var restitution: Float32 = 0.0
    var friction: Float32 = 0.0

    # Compute state size
    comptime STATE_SIZE = compute_state_size[
        NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS
    ]()
    print("State size:", STATE_SIZE, "floats")

    # Create GPU context
    var ctx = DeviceContext()
    print("GPU device initialized")

    # Allocate buffers
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](NUM_BODIES * MODEL_BODY_SIZE)

    # Initialize state on host
    var state_host = List[Scalar[DTYPE]](capacity=STATE_SIZE)
    for _ in range(STATE_SIZE):
        state_host.append(Scalar[DTYPE](0))

    # Set initial body state
    var b_off = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS](0)
    state_host[b_off + BODY_IDX_PX] = Scalar[DTYPE](initial_x)
    state_host[b_off + BODY_IDX_PY] = Scalar[DTYPE](0)
    state_host[b_off + BODY_IDX_PZ] = Scalar[DTYPE](anchor_z)

    # Set identity quaternion
    state_host[b_off + BODY_IDX_QX] = Scalar[DTYPE](0)
    state_host[b_off + BODY_IDX_QY] = Scalar[DTYPE](0)
    state_host[b_off + BODY_IDX_QZ] = Scalar[DTYPE](0)
    state_host[b_off + BODY_IDX_QW] = Scalar[DTYPE](1)

    # Set slide joint state
    var sj_off = slide_joint_offset[
        NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS
    ](0)
    state_host[sj_off + SLIDE_IDX_PARENT] = Scalar[DTYPE](-1)  # World anchor
    state_host[sj_off + SLIDE_IDX_CHILD] = Scalar[DTYPE](0)
    state_host[sj_off + SLIDE_IDX_ANCHOR_PX] = Scalar[DTYPE](0)
    state_host[sj_off + SLIDE_IDX_ANCHOR_PY] = Scalar[DTYPE](0)
    state_host[sj_off + SLIDE_IDX_ANCHOR_PZ] = Scalar[DTYPE](anchor_z)
    state_host[sj_off + SLIDE_IDX_ANCHOR_CX] = Scalar[DTYPE](0)
    state_host[sj_off + SLIDE_IDX_ANCHOR_CY] = Scalar[DTYPE](0)
    state_host[sj_off + SLIDE_IDX_ANCHOR_CZ] = Scalar[DTYPE](0)
    state_host[sj_off + SLIDE_IDX_AXIS_X] = Scalar[DTYPE](1)  # X-axis
    state_host[sj_off + SLIDE_IDX_AXIS_Y] = Scalar[DTYPE](0)
    state_host[sj_off + SLIDE_IDX_AXIS_Z] = Scalar[DTYPE](0)
    state_host[sj_off + SLIDE_IDX_TARGET_FORCE] = Scalar[DTYPE](0)

    # Set metadata
    var m_off = metadata_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS]()
    state_host[m_off + META_IDX_NUM_CONTACTS] = Scalar[DTYPE](0)
    state_host[m_off + META_IDX_NUM_JOINTS] = Scalar[DTYPE](0)

    # Initialize model on host
    var model_host = List[Scalar[DTYPE]](capacity=NUM_BODIES * MODEL_BODY_SIZE)
    for _ in range(NUM_BODIES * MODEL_BODY_SIZE):
        model_host.append(Scalar[DTYPE](0))

    # Set body properties
    model_host[MODEL_IDX_MASS] = Scalar[DTYPE](mass)
    model_host[MODEL_IDX_INV_MASS] = Scalar[DTYPE](1.0 / mass)
    model_host[MODEL_IDX_RADIUS] = Scalar[DTYPE](radius)
    var inertia = Float32(0.4) * mass * radius * radius
    var inv_inertia = Float32(1.0) / inertia
    model_host[MODEL_IDX_IXX] = Scalar[DTYPE](inertia)
    model_host[MODEL_IDX_IYY] = Scalar[DTYPE](inertia)
    model_host[MODEL_IDX_IZZ] = Scalar[DTYPE](inertia)
    model_host[MODEL_IDX_INV_IXX] = Scalar[DTYPE](inv_inertia)
    model_host[MODEL_IDX_INV_IYY] = Scalar[DTYPE](inv_inertia)
    model_host[MODEL_IDX_INV_IZZ] = Scalar[DTYPE](inv_inertia)

    # Copy to GPU
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())
    ctx.synchronize()

    print("Initial state copied to GPU")
    print("Initial position: (", initial_x, ", 0,", anchor_z, ")")
    print("Slide axis: X (1, 0, 0)")
    print("Gravity: (0, 0,", gravity_z, ") - perpendicular to slide")
    print()

    # Run simulation
    var num_steps = 1000
    var max_y_drift: Float32 = 0.0
    var max_z_drift: Float32 = 0.0

    print("Running", num_steps, "steps on GPU...")

    for step in range(num_steps):
        PGSIntegrator.step_gpu[
            DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS, BATCH
        ](
            ctx,
            state_buf,
            model_buf,
            Scalar[DTYPE](dt),
            Scalar[DTYPE](gravity_z),
            Scalar[DTYPE](ground_z),
            Scalar[DTYPE](restitution),
            Scalar[DTYPE](friction),
        )

        # Read state every 200 steps
        if (step + 1) % 200 == 0:
            ctx.synchronize()
            ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
            ctx.synchronize()

            var px = Float32(state_host[b_off + BODY_IDX_PX])
            var py = Float32(state_host[b_off + BODY_IDX_PY])
            var pz = Float32(state_host[b_off + BODY_IDX_PZ])

            var y_drift = abs_val(py)
            var z_drift = abs_val(pz - anchor_z)
            max_y_drift = max_val(max_y_drift, y_drift)
            max_z_drift = max_val(max_z_drift, z_drift)

            var t = Float32(step + 1) * dt
            print(
                "  Step",
                step + 1,
                ": t =",
                t,
                "s, pos = (",
                px,
                ",",
                py,
                ",",
                pz,
                ")",
            )

    ctx.synchronize()

    # Read final state
    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    var final_px = Float32(state_host[b_off + BODY_IDX_PX])
    var final_py = Float32(state_host[b_off + BODY_IDX_PY])
    var final_pz = Float32(state_host[b_off + BODY_IDX_PZ])

    # Final drift
    var y_drift = abs_val(final_py)
    var z_drift = abs_val(final_pz - anchor_z)
    max_y_drift = max_val(max_y_drift, y_drift)
    max_z_drift = max_val(max_z_drift, z_drift)

    print()
    print("=" * 60)
    print("Results:")
    print("  Final position: (", final_px, ",", final_py, ",", final_pz, ")")
    print("  Max Y drift:", max_y_drift * 1000.0, "mm")
    print("  Max Z drift:", max_z_drift * 1000.0, "mm")
    print()

    # Check that perpendicular motion is constrained
    var passed = max_y_drift < 0.01 and max_z_drift < 0.01

    if passed:
        print("PASSED: GPU slide joint constraint maintained")
        print("  - Y drift within 10mm")
        print("  - Z drift within 10mm")
    else:
        print("FAILED: GPU slide joint constraint violated")
        if max_y_drift >= 0.01:
            print("  - Y drift too large:", max_y_drift * 1000.0, "mm")
        if max_z_drift >= 0.01:
            print("  - Z drift too large:", max_z_drift * 1000.0, "mm")

    print("=" * 60)
