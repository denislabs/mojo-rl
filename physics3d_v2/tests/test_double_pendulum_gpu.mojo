"""GPU Double Pendulum Test.

Tests the GPU joint solver with a two-link chain (double pendulum).

Setup:
- Joint 0: World -> Body 0 (first link)
- Joint 1: Body 0 -> Body 1 (second link)

Run with:
    cd mojo-rl
    pixi run -e apple mojo run physics3d_v2/tests/test_double_pendulum_gpu.mojo
"""

from math import sqrt, sin, cos
from gpu.host import DeviceContext, DeviceBuffer
from physics3d_v2.integrator import ImpulseIntegrator
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
comptime NUM_BODIES: Int = 2
comptime MAX_CONTACTS: Int = 10
comptime MAX_JOINTS: Int = 2
comptime BATCH: Int = 1
comptime DTYPE = DType.float32
comptime PI: Float32 = 3.14159265358979323846


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
    print("    GPU Double Pendulum Test (Two-Link Chain)")
    print("=" * 60)
    print()

    # Physics parameters
    var L1: Float32 = 1.0  # Length of first link
    var L2: Float32 = 1.0  # Length of second link
    var mass: Float32 = 1.0
    var radius: Float32 = 0.05
    var initial_angle_deg: Float32 = 30.0
    var initial_angle = initial_angle_deg * PI / 180.0
    var dt: Float32 = 0.001
    var gravity_z: Float32 = -9.81
    var ground_z: Float32 = -10.0
    var restitution: Float32 = 0.0
    var pivot_z = L1  # Pivot height

    # Compute state size
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()
    print("State size:", STATE_SIZE, "floats")
    print("Bodies:", NUM_BODIES)
    print("Joints:", MAX_JOINTS)

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

    # Compute initial body positions
    var body0_x = L1 * sin(initial_angle)
    var body0_z = pivot_z - L1 * cos(initial_angle)
    var body1_x = body0_x + L2 * sin(initial_angle)
    var body1_z = body0_z - L2 * cos(initial_angle)
    var half_angle = initial_angle / 2.0

    # Set body 0 state
    var b0_off = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    state_host[b0_off + BODY_IDX_PX] = Scalar[DTYPE](body0_x)
    state_host[b0_off + BODY_IDX_PY] = Scalar[DTYPE](0)
    state_host[b0_off + BODY_IDX_PZ] = Scalar[DTYPE](body0_z)
    state_host[b0_off + BODY_IDX_QX] = Scalar[DTYPE](0)
    state_host[b0_off + BODY_IDX_QY] = Scalar[DTYPE](-sin(half_angle))
    state_host[b0_off + BODY_IDX_QZ] = Scalar[DTYPE](0)
    state_host[b0_off + BODY_IDX_QW] = Scalar[DTYPE](cos(half_angle))

    # Set body 1 state
    var b1_off = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](1)
    state_host[b1_off + BODY_IDX_PX] = Scalar[DTYPE](body1_x)
    state_host[b1_off + BODY_IDX_PY] = Scalar[DTYPE](0)
    state_host[b1_off + BODY_IDX_PZ] = Scalar[DTYPE](body1_z)
    state_host[b1_off + BODY_IDX_QX] = Scalar[DTYPE](0)
    state_host[b1_off + BODY_IDX_QY] = Scalar[DTYPE](-sin(half_angle))
    state_host[b1_off + BODY_IDX_QZ] = Scalar[DTYPE](0)
    state_host[b1_off + BODY_IDX_QW] = Scalar[DTYPE](cos(half_angle))

    # Set joint 0 state: World -> Body 0
    var j0_off = joint_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    state_host[j0_off + JOINT_IDX_PARENT] = Scalar[DTYPE](-1)  # World anchor
    state_host[j0_off + JOINT_IDX_CHILD] = Scalar[DTYPE](0)
    state_host[j0_off + JOINT_IDX_ANCHOR_PX] = Scalar[DTYPE](0)  # Pivot at (0, 0, pivot_z)
    state_host[j0_off + JOINT_IDX_ANCHOR_PY] = Scalar[DTYPE](0)
    state_host[j0_off + JOINT_IDX_ANCHOR_PZ] = Scalar[DTYPE](pivot_z)
    state_host[j0_off + JOINT_IDX_ANCHOR_CX] = Scalar[DTYPE](0)  # L1 above body 0
    state_host[j0_off + JOINT_IDX_ANCHOR_CY] = Scalar[DTYPE](0)
    state_host[j0_off + JOINT_IDX_ANCHOR_CZ] = Scalar[DTYPE](L1)
    state_host[j0_off + JOINT_IDX_AXIS_X] = Scalar[DTYPE](0)
    state_host[j0_off + JOINT_IDX_AXIS_Y] = Scalar[DTYPE](1)
    state_host[j0_off + JOINT_IDX_AXIS_Z] = Scalar[DTYPE](0)

    # Set joint 1 state: Body 0 -> Body 1
    var j1_off = joint_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](1)
    state_host[j1_off + JOINT_IDX_PARENT] = Scalar[DTYPE](0)  # Body 0
    state_host[j1_off + JOINT_IDX_CHILD] = Scalar[DTYPE](1)
    state_host[j1_off + JOINT_IDX_ANCHOR_PX] = Scalar[DTYPE](0)  # At body 0's position
    state_host[j1_off + JOINT_IDX_ANCHOR_PY] = Scalar[DTYPE](0)
    state_host[j1_off + JOINT_IDX_ANCHOR_PZ] = Scalar[DTYPE](0)
    state_host[j1_off + JOINT_IDX_ANCHOR_CX] = Scalar[DTYPE](0)  # L2 above body 1
    state_host[j1_off + JOINT_IDX_ANCHOR_CY] = Scalar[DTYPE](0)
    state_host[j1_off + JOINT_IDX_ANCHOR_CZ] = Scalar[DTYPE](L2)
    state_host[j1_off + JOINT_IDX_AXIS_X] = Scalar[DTYPE](0)
    state_host[j1_off + JOINT_IDX_AXIS_Y] = Scalar[DTYPE](1)
    state_host[j1_off + JOINT_IDX_AXIS_Z] = Scalar[DTYPE](0)

    # Set metadata
    var m_off = metadata_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()
    state_host[m_off + META_IDX_NUM_CONTACTS] = Scalar[DTYPE](0)
    state_host[m_off + META_IDX_NUM_JOINTS] = Scalar[DTYPE](2)

    # Initialize model on host
    var model_host = List[Scalar[DTYPE]](capacity=NUM_BODIES * MODEL_BODY_SIZE)
    for _ in range(NUM_BODIES * MODEL_BODY_SIZE):
        model_host.append(Scalar[DTYPE](0))

    # Set body properties (same for both bodies)
    var inertia = Float32(0.4) * mass * radius * radius
    var inv_inertia = Float32(1.0) / inertia

    for body_idx in range(NUM_BODIES):
        var m_body_off = body_idx * MODEL_BODY_SIZE
        model_host[m_body_off + MODEL_IDX_MASS] = Scalar[DTYPE](mass)
        model_host[m_body_off + MODEL_IDX_INV_MASS] = Scalar[DTYPE](1.0 / mass)
        model_host[m_body_off + MODEL_IDX_RADIUS] = Scalar[DTYPE](radius)
        model_host[m_body_off + MODEL_IDX_IXX] = Scalar[DTYPE](inertia)
        model_host[m_body_off + MODEL_IDX_IYY] = Scalar[DTYPE](inertia)
        model_host[m_body_off + MODEL_IDX_IZZ] = Scalar[DTYPE](inertia)
        model_host[m_body_off + MODEL_IDX_INV_IXX] = Scalar[DTYPE](inv_inertia)
        model_host[m_body_off + MODEL_IDX_INV_IYY] = Scalar[DTYPE](inv_inertia)
        model_host[m_body_off + MODEL_IDX_INV_IZZ] = Scalar[DTYPE](inv_inertia)

    # Copy to GPU
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())
    ctx.synchronize()

    print("\nInitial state copied to GPU")
    print("  Body 0 position: (", body0_x, ", 0,", body0_z, ")")
    print("  Body 1 position: (", body1_x, ", 0,", body1_z, ")")
    print()

    # Debug: Print buffer layout
    print("Buffer layout:")
    print("  Body 0 offset:", b0_off)
    print("  Body 1 offset:", b1_off)
    print("  Joint 0 offset:", j0_off)
    print("  Joint 1 offset:", j1_off)
    print("  Metadata offset:", m_off)
    print()

    # Run simulation
    var num_steps = 2000
    var max_L1_error: Float32 = 0.0
    var max_L2_error: Float32 = 0.0

    print("Running", num_steps, "steps on GPU...")

    for step in range(num_steps):
        ImpulseIntegrator.step_gpu[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, BATCH](
            ctx,
            state_buf,
            model_buf,
            Scalar[DTYPE](dt),
            Scalar[DTYPE](gravity_z),
            Scalar[DTYPE](ground_z),
            Scalar[DTYPE](restitution),
            Scalar[DTYPE](0.0),  # friction
        )

        # Check constraint every 500 steps
        if (step + 1) % 500 == 0:
            ctx.synchronize()
            ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
            ctx.synchronize()

            # Get body positions
            var p0_x = Float32(state_host[b0_off + BODY_IDX_PX])
            var p0_y = Float32(state_host[b0_off + BODY_IDX_PY])
            var p0_z = Float32(state_host[b0_off + BODY_IDX_PZ])
            var p1_x = Float32(state_host[b1_off + BODY_IDX_PX])
            var p1_y = Float32(state_host[b1_off + BODY_IDX_PY])
            var p1_z = Float32(state_host[b1_off + BODY_IDX_PZ])

            # Distance from pivot to body 0
            var dx0 = p0_x
            var dy0 = p0_y
            var dz0 = p0_z - pivot_z
            var dist_0 = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)
            var L1_error = abs_val(dist_0 - L1) * 1000.0  # mm
            max_L1_error = max_val(max_L1_error, L1_error)

            # Distance from body 0 to body 1
            var dx1 = p1_x - p0_x
            var dy1 = p1_y - p0_y
            var dz1 = p1_z - p0_z
            var dist_1 = sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1)
            var L2_error = abs_val(dist_1 - L2) * 1000.0  # mm
            max_L2_error = max_val(max_L2_error, L2_error)

            var t = Float32(step + 1) * dt
            print(
                "  Step", step + 1, ": t =", t,
                "s, L1_err =", L1_error,
                "mm, L2_err =", L2_error, "mm"
            )

    ctx.synchronize()

    # Read final state
    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    var final_p0_x = Float32(state_host[b0_off + BODY_IDX_PX])
    var final_p0_y = Float32(state_host[b0_off + BODY_IDX_PY])
    var final_p0_z = Float32(state_host[b0_off + BODY_IDX_PZ])
    var final_p1_x = Float32(state_host[b1_off + BODY_IDX_PX])
    var final_p1_y = Float32(state_host[b1_off + BODY_IDX_PY])
    var final_p1_z = Float32(state_host[b1_off + BODY_IDX_PZ])

    print()
    print("=" * 60)
    print("Results:")
    print("  Final body 0 position: (", final_p0_x, ",", final_p0_y, ",", final_p0_z, ")")
    print("  Final body 1 position: (", final_p1_x, ",", final_p1_y, ",", final_p1_z, ")")
    print()
    print("  Max L1 constraint error:", max_L1_error, "mm")
    print("  Max L2 constraint error:", max_L2_error, "mm")
    print()

    # Check both constraints (20mm tolerance for GPU with float32)
    var passed = max_L1_error < 20.0 and max_L2_error < 20.0

    if passed:
        print("PASSED: GPU double pendulum constraints maintained")
    else:
        print("FAILED: GPU double pendulum constraint error too large")
        if max_L1_error >= 20.0:
            print("  - L1 error exceeds 20mm")
        if max_L2_error >= 20.0:
            print("  - L2 error exceeds 20mm")

    print("=" * 60)
