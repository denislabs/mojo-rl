"""GPU Pendulum Test.

Tests the GPU joint solver by running a simple pendulum simulation on GPU.

Run with:
    cd mojo-rl
    pixi run -e apple mojo run physics3d/tests/test_pendulum_gpu.mojo
"""

from math import sqrt, sin, cos
from gpu.host import DeviceContext, DeviceBuffer
from physics3d.integrator import ImpulseIntegrator
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
comptime NUM_BODIES: Int = 1
comptime MAX_CONTACTS: Int = 5
comptime MAX_JOINTS: Int = 1
comptime BATCH: Int = 1
comptime DTYPE = DType.float32
comptime PI: Float32 = 3.14159265358979323846


fn main() raises:
    print("=" * 60)
    print("    GPU Pendulum Test (Hinge Joint)")
    print("=" * 60)
    print()

    # Physics parameters
    var L: Float32 = 1.0  # Pendulum length
    var mass: Float32 = 1.0
    var radius: Float32 = 0.1
    var initial_angle_deg: Float32 = 30.0
    var initial_angle = initial_angle_deg * PI / 180.0
    var dt: Float32 = 0.001
    var gravity_z: Float32 = -9.81
    var ground_z: Float32 = -10.0
    var restitution: Float32 = 0.0

    # Compute state size
    comptime STATE_SIZE = compute_state_size[
        NUM_BODIES, MAX_CONTACTS, MAX_JOINTS
    ]()
    print("State size:", STATE_SIZE, "floats")

    # Create GPU context
    var ctx = DeviceContext()
    print("GPU device initialized")

    # Allocate buffers
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](
        NUM_BODIES * MODEL_BODY_SIZE
    )

    # Initialize state on host
    var state_host = List[Scalar[DTYPE]](capacity=STATE_SIZE)
    for _ in range(STATE_SIZE):
        state_host.append(Scalar[DTYPE](0))

    # Set initial body state
    var b_off = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    var bob_x = L * sin(initial_angle)
    var bob_z = L - L * cos(initial_angle)

    state_host[b_off + BODY_IDX_PX] = Scalar[DTYPE](bob_x)
    state_host[b_off + BODY_IDX_PY] = Scalar[DTYPE](0)
    state_host[b_off + BODY_IDX_PZ] = Scalar[DTYPE](bob_z)

    # Set initial quaternion (rotation by -initial_angle around Y)
    var half_angle = initial_angle / 2.0
    state_host[b_off + BODY_IDX_QX] = Scalar[DTYPE](0)
    state_host[b_off + BODY_IDX_QY] = Scalar[DTYPE](-sin(half_angle))
    state_host[b_off + BODY_IDX_QZ] = Scalar[DTYPE](0)
    state_host[b_off + BODY_IDX_QW] = Scalar[DTYPE](cos(half_angle))

    # Set joint state
    var j_off = joint_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    state_host[j_off + JOINT_IDX_PARENT] = Scalar[DTYPE](-1)  # World anchor
    state_host[j_off + JOINT_IDX_CHILD] = Scalar[DTYPE](0)
    state_host[j_off + JOINT_IDX_ANCHOR_PX] = Scalar[DTYPE](0)
    state_host[j_off + JOINT_IDX_ANCHOR_PY] = Scalar[DTYPE](0)
    state_host[j_off + JOINT_IDX_ANCHOR_PZ] = Scalar[DTYPE](L)
    state_host[j_off + JOINT_IDX_ANCHOR_CX] = Scalar[DTYPE](0)
    state_host[j_off + JOINT_IDX_ANCHOR_CY] = Scalar[DTYPE](0)
    state_host[j_off + JOINT_IDX_ANCHOR_CZ] = Scalar[DTYPE](L)
    state_host[j_off + JOINT_IDX_AXIS_X] = Scalar[DTYPE](0)
    state_host[j_off + JOINT_IDX_AXIS_Y] = Scalar[DTYPE](1)
    state_host[j_off + JOINT_IDX_AXIS_Z] = Scalar[DTYPE](0)

    # Set metadata
    var m_off = metadata_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()
    state_host[m_off + META_IDX_NUM_CONTACTS] = Scalar[DTYPE](0)
    state_host[m_off + META_IDX_NUM_JOINTS] = Scalar[DTYPE](1)

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
    print("Initial bob position: (", bob_x, ", 0,", bob_z, ")")
    print()

    # Run simulation
    var num_steps = 1000
    print("Running", num_steps, "steps on GPU...")

    for step in range(num_steps):
        ImpulseIntegrator.step_gpu[
            DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, BATCH
        ](
            ctx,
            state_buf,
            model_buf,
            Scalar[DTYPE](dt),
            Scalar[DTYPE](gravity_z),
            Scalar[DTYPE](ground_z),
            Scalar[DTYPE](restitution),
            Scalar[DTYPE](0.0),  # friction
        )

        # Print progress every 200 steps
        if (step + 1) % 200 == 0:
            ctx.synchronize()
            ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
            ctx.synchronize()

            var px = Float32(state_host[b_off + BODY_IDX_PX])
            var py = Float32(state_host[b_off + BODY_IDX_PY])
            var pz = Float32(state_host[b_off + BODY_IDX_PZ])

            # Distance from bob to pivot
            var pivot_z: Float32 = L
            var dx = px - 0.0
            var dy = py - 0.0
            var dz = pz - pivot_z
            var dist = sqrt(dx * dx + dy * dy + dz * dz)

            var t = Float32(step + 1) * dt
            print(
                "  Step",
                step + 1,
                ": t =",
                t,
                "s, pos = (",
                px,
                ",",
                pz,
                "), dist =",
                dist,
                "m",
            )

    ctx.synchronize()

    # Read final state
    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    var final_px = Float32(state_host[b_off + BODY_IDX_PX])
    var final_py = Float32(state_host[b_off + BODY_IDX_PY])
    var final_pz = Float32(state_host[b_off + BODY_IDX_PZ])

    # Check constraint
    var pivot_z: Float32 = L
    var dx = final_px - 0.0
    var dy = final_py - 0.0
    var dz = final_pz - pivot_z
    var final_dist = sqrt(dx * dx + dy * dy + dz * dz)
    var length_error = abs(final_dist - L) * 1000.0  # mm

    print()
    print("=" * 60)
    print("Results:")
    print("  Final position: (", final_px, ",", final_py, ",", final_pz, ")")
    print("  Distance to pivot:", final_dist, "m")
    print("  Expected length:", L, "m")
    print("  Length error:", length_error, "mm")
    print()

    if length_error < 10.0:
        print(
            "PASSED: GPU pendulum constraint maintained (error <",
            length_error,
            "mm)",
        )
    else:
        print(
            "FAILED: GPU pendulum constraint error too large (",
            length_error,
            "mm)",
        )

    print("=" * 60)


fn abs(x: Float32) -> Float32:
    if x < 0:
        return -x
    return x
