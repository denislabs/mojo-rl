"""Debug test: Read joint parent/child values on GPU.

Verifies that joint indices are correctly stored and read from GPU buffers.
"""

from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor

from physics3d_v2.gpu.constants import (
    compute_state_size,
    body_offset,
    joint_offset,
    metadata_offset,
    BODY_STATE_SIZE,
    BODY_IDX_PX,
    BODY_IDX_PY,
    BODY_IDX_PZ,
    BODY_IDX_QW,
    JOINT_STATE_SIZE,
    JOINT_IDX_PARENT,
    JOINT_IDX_CHILD,
    JOINT_IDX_ANCHOR_PX,
    JOINT_IDX_ANCHOR_PY,
    JOINT_IDX_ANCHOR_PZ,
    JOINT_IDX_AXIS_Y,
    META_IDX_NUM_JOINTS,
)


comptime NUM_BODIES = 2
comptime MAX_CONTACTS = 5
comptime MAX_JOINTS = 1
comptime BATCH = 1
comptime DTYPE = DType.float32
comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()


fn run_debug_kernel(
    ctx: DeviceContext,
    mut state_buf: DeviceBuffer[DTYPE],
    mut debug_buf: DeviceBuffer[DTYPE],
) raises:
    """Run kernel to read joint values and write to debug buffer."""

    var state = LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ](state_buf.unsafe_ptr())

    var debug = LayoutTensor[
        DTYPE, Layout.row_major(20), MutAnyOrigin
    ](debug_buf.unsafe_ptr())

    @always_inline
    fn kernel_wrapper(
        state: LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        debug: LayoutTensor[DTYPE, Layout.row_major(20), MutAnyOrigin],
    ):
        var env = 0

        # Read joint state
        var j_off = joint_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)

        var parent_f = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_PARENT])
        var child_f = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_CHILD])
        var anchor_pz = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_ANCHOR_PZ])
        var axis_y = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_AXIS_Y])

        # Write raw values to debug buffer
        debug[0] = Scalar[DTYPE](j_off)  # Joint offset
        debug[1] = parent_f              # Raw parent value
        debug[2] = child_f               # Raw child value
        debug[3] = anchor_pz             # Anchor Z (should be 1.0 for pendulum)
        debug[4] = axis_y                # Axis Y (should be 1.0)

        # Convert parent to int using sequential > method
        var body_a: Int = -1
        if parent_f > Scalar[DTYPE](-0.5):
            body_a = 0
        if parent_f > Scalar[DTYPE](0.5):
            body_a = 1
        if parent_f > Scalar[DTYPE](1.5):
            body_a = 2
        debug[5] = Scalar[DTYPE](body_a)

        # Convert child to int using sequential > method
        var body_b: Int = 0
        if child_f > Scalar[DTYPE](0.5):
            body_b = 1
        if child_f > Scalar[DTYPE](1.5):
            body_b = 2
        debug[6] = Scalar[DTYPE](body_b)

        # Also read metadata
        var m_off = metadata_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()
        var num_joints = rebind[Scalar[DTYPE]](state[env, m_off + META_IDX_NUM_JOINTS])
        debug[7] = Scalar[DTYPE](m_off)
        debug[8] = num_joints

    ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
        state,
        debug,
        grid_dim=(1,),
        block_dim=(1,),
    )


fn main() raises:
    print("=" * 60)
    print("GPU Joint Read Debug Test")
    print("=" * 60)

    var ctx = DeviceContext()

    print("Configuration:")
    print("  NUM_BODIES:", NUM_BODIES)
    print("  MAX_CONTACTS:", MAX_CONTACTS)
    print("  MAX_JOINTS:", MAX_JOINTS)
    print("  STATE_SIZE:", STATE_SIZE)
    print("  JOINT_STATE_SIZE:", JOINT_STATE_SIZE)

    var j_off = joint_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](0)
    print("  Joint offset(0):", j_off)

    # Create state buffer
    var state_host = List[Float32](capacity=STATE_SIZE)
    for _ in range(STATE_SIZE):
        state_host.append(0.0)

    # Set up pendulum-like joint (parent=-1, child=0)
    state_host[j_off + JOINT_IDX_PARENT] = -1.0  # World anchor
    state_host[j_off + JOINT_IDX_CHILD] = 0.0
    state_host[j_off + JOINT_IDX_ANCHOR_PZ] = 1.0  # Pivot at z=1
    state_host[j_off + JOINT_IDX_AXIS_Y] = 1.0    # Y-axis rotation

    # Set metadata
    var m_off = metadata_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()
    state_host[m_off + META_IDX_NUM_JOINTS] = 1.0

    print("\nHost values written:")
    print("  state_host[", j_off + JOINT_IDX_PARENT, "] =", state_host[j_off + JOINT_IDX_PARENT], "(parent)")
    print("  state_host[", j_off + JOINT_IDX_CHILD, "] =", state_host[j_off + JOINT_IDX_CHILD], "(child)")
    print("  state_host[", j_off + JOINT_IDX_ANCHOR_PZ, "] =", state_host[j_off + JOINT_IDX_ANCHOR_PZ], "(anchor_pz)")
    print("  state_host[", j_off + JOINT_IDX_AXIS_Y, "] =", state_host[j_off + JOINT_IDX_AXIS_Y], "(axis_y)")
    print("  state_host[", m_off + META_IDX_NUM_JOINTS, "] =", state_host[m_off + META_IDX_NUM_JOINTS], "(num_joints)")

    # Create debug buffer
    var debug_host = List[Float32](capacity=20)
    for _ in range(20):
        debug_host.append(-999.0)

    # Copy to GPU
    var state_buf = ctx.enqueue_create_buffer[DTYPE](STATE_SIZE)
    var debug_buf = ctx.enqueue_create_buffer[DTYPE](20)
    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.enqueue_copy(debug_buf, debug_host.unsafe_ptr())
    ctx.synchronize()

    # Run kernel
    run_debug_kernel(ctx, state_buf, debug_buf)
    ctx.synchronize()

    # Copy back
    ctx.enqueue_copy(debug_host.unsafe_ptr(), debug_buf)
    ctx.synchronize()

    print("\nGPU kernel results:")
    print("  debug[0] (j_off):", debug_host[0])
    print("  debug[1] (parent_f):", debug_host[1])
    print("  debug[2] (child_f):", debug_host[2])
    print("  debug[3] (anchor_pz):", debug_host[3])
    print("  debug[4] (axis_y):", debug_host[4])
    print("  debug[5] (body_a after conversion):", debug_host[5])
    print("  debug[6] (body_b after conversion):", debug_host[6])
    print("  debug[7] (m_off):", debug_host[7])
    print("  debug[8] (num_joints):", debug_host[8])

    # Verify
    var all_correct = True
    if debug_host[1] != -1.0:
        print("\nERROR: parent_f should be -1.0, got", debug_host[1])
        all_correct = False
    if debug_host[2] != 0.0:
        print("\nERROR: child_f should be 0.0, got", debug_host[2])
        all_correct = False
    if debug_host[5] != -1.0:
        print("\nERROR: body_a should be -1, got", debug_host[5])
        all_correct = False
    if debug_host[6] != 0.0:
        print("\nERROR: body_b should be 0, got", debug_host[6])
        all_correct = False

    if all_correct:
        print("\n✓ All values read correctly!")
    else:
        print("\n✗ Some values are incorrect!")

    # Now test with hopper configuration (parent=0, child=1)
    print("\n" + "=" * 60)
    print("Testing hopper configuration (parent=0, child=1)")
    print("=" * 60)

    state_host[j_off + JOINT_IDX_PARENT] = 0.0  # Torso
    state_host[j_off + JOINT_IDX_CHILD] = 1.0   # Foot

    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.synchronize()

    run_debug_kernel(ctx, state_buf, debug_buf)
    ctx.synchronize()

    ctx.enqueue_copy(debug_host.unsafe_ptr(), debug_buf)
    ctx.synchronize()

    print("\nGPU kernel results for hopper:")
    print("  debug[1] (parent_f):", debug_host[1])
    print("  debug[2] (child_f):", debug_host[2])
    print("  debug[5] (body_a after conversion):", debug_host[5])
    print("  debug[6] (body_b after conversion):", debug_host[6])

    all_correct = True
    if debug_host[1] != 0.0:
        print("\nERROR: parent_f should be 0.0, got", debug_host[1])
        all_correct = False
    if debug_host[2] != 1.0:
        print("\nERROR: child_f should be 1.0, got", debug_host[2])
        all_correct = False
    if debug_host[5] != 0.0:
        print("\nERROR: body_a should be 0, got", debug_host[5])
        all_correct = False
    if debug_host[6] != 1.0:
        print("\nERROR: body_b should be 1, got", debug_host[6])
        all_correct = False

    if all_correct:
        print("\n✓ Hopper values read correctly!")
    else:
        print("\n✗ Hopper values are incorrect!")

    print("\n" + "=" * 60)
