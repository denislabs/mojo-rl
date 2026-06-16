"""Smoke + regression gate for PhysicsStateOwned CPU tensor-view accessors.

Constructs a PhysicsStateOwned, writes a deterministic ramp through one view,
and reads it back through the whole-buffer and offset views to confirm they
alias the right regions of the underlying `List`s. Prints a checksum.

This is the gate for migrating the accessors off the UnsafeAnyOrigin hatch:
`mojo precompile` doesn't instantiate the generic struct, so only an actual
run instantiates the accessors. The checksum must be bit-identical pre/post.
"""

from mojo_rl.physics2d import PhysicsStateOwned, dtype


def main() raises:
    comptime NUM_BODIES = 2
    comptime NUM_SHAPES = 2
    comptime MAX_CONTACTS = 4
    comptime MAX_JOINTS = 2
    comptime BODY_STATE_SIZE = 13
    comptime BODIES_OFFSET = 8
    comptime FORCES_OFFSET = BODIES_OFFSET + NUM_BODIES * BODY_STATE_SIZE  # 34
    comptime JOINTS_OFFSET = FORCES_OFFSET + NUM_BODIES * 3  # 40
    comptime JOINT_COUNT_OFFSET = 74
    comptime EDGES_OFFSET = 80
    comptime EDGE_COUNT_OFFSET = 200
    comptime STATE_SIZE = 256

    var st = PhysicsStateOwned[
        NUM_BODIES,
        NUM_SHAPES,
        MAX_CONTACTS,
        MAX_JOINTS,
        STATE_SIZE,
        BODIES_OFFSET,
        FORCES_OFFSET,
        JOINTS_OFFSET,
        JOINT_COUNT_OFFSET,
        EDGES_OFFSET,
        EDGE_COUNT_OFFSET,
    ]()

    # Write a deterministic ramp across the whole state buffer.
    var state = st.get_state_tensor()
    for i in range(STATE_SIZE):
        state[0, i] = Scalar[dtype](Float64((i * 5 + 1) % 91) / 91.0)

    # Write known values into shapes / contacts / contact_counts.
    var shapes = st.get_shapes_tensor()
    for s in range(NUM_SHAPES):
        for k in range(2):
            shapes[s, k] = Scalar[dtype](Float64(s + 1) + Float64(k) * 0.25)
    var contacts = st.get_contacts_tensor()
    contacts[0, 0, 0] = Scalar[dtype](7.5)
    var contact_counts = st.get_contact_counts_tensor()
    contact_counts[0] = Scalar[dtype](3.0)

    # Read back through the offset views; they must alias the ramp at the
    # corresponding state offsets.
    var bodies = st.get_bodies_tensor()
    var forces = st.get_forces_tensor()
    var joints = st.get_joints_tensor()
    var joint_counts = st.get_joint_counts_tensor()

    var checksum = Float64(0.0)
    for b in range(NUM_BODIES):
        for f in range(BODY_STATE_SIZE):
            checksum += Float64(
                rebind[Scalar[dtype]](bodies[0, b, f])
            ) * Float64((b + f) % 7 + 1)
    for b in range(NUM_BODIES):
        for f in range(3):
            checksum += Float64(rebind[Scalar[dtype]](forces[0, b, f]))
    for j in range(MAX_JOINTS):
        for f in range(2):
            checksum += Float64(rebind[Scalar[dtype]](joints[0, j, f]))
    checksum += Float64(rebind[Scalar[dtype]](joint_counts[0]))
    checksum += Float64(rebind[Scalar[dtype]](shapes[0, 0])) + Float64(
        rebind[Scalar[dtype]](shapes[1, 1])
    )
    checksum += Float64(rebind[Scalar[dtype]](contacts[0, 0, 0])) + Float64(
        rebind[Scalar[dtype]](contact_counts[0])
    )

    # Cross-check: bodies[0,0,0] must equal state[0, BODIES_OFFSET].
    var alias_ok = (
        rebind[Scalar[dtype]](bodies[0, 0, 0])
        == rebind[Scalar[dtype]](state[0, BODIES_OFFSET])
        and rebind[Scalar[dtype]](joint_counts[0])
        == rebind[Scalar[dtype]](state[0, JOINT_COUNT_OFFSET])
    )

    print("state_accessors_checksum =", checksum)
    print("alias_ok =", alias_ok)
    if not alias_ok:
        print("FAIL: offset views do not alias state")
        return
    print("PhysicsStateOwned accessors smoke: OK")
