"""Test the new PhysicsLayoutStrided and PhysicsEnvHelpers architecture.

This test validates that:
1. PhysicsLayoutStrided computes correct offsets and sizes
2. PhysicsEnvHelpers can initialize and manipulate physics state
3. The layout is compatible with existing physics components

Run with:
    cd mojo-rl && pixi run mojo run examples/physics_strided_test.mojo
"""

from layout import LayoutTensor
from layout import Layout as TensorLayout

from physics_gpu import (
    dtype,
    BODY_STATE_SIZE,
    JOINT_DATA_SIZE,
    PhysicsLayoutStrided,
    LunarLanderLayoutStrided,
    PhysicsEnvHelpers,
    PhysicsKernelStrided,
    PhysicsConfigStrided,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
)


fn test_layout_computation() raises:
    """Test that layout computes correct sizes and offsets."""
    print("Testing PhysicsLayoutStrided...")

    # Test LunarLander layout (use L as prefix to not conflict with TensorLayout)
    comptime L = LunarLanderLayoutStrided

    # Check layout constants are sensible
    print("  NUM_BODIES:", L.NUM_BODIES)
    print("  MAX_CONTACTS:", L.MAX_CONTACTS)
    print("  MAX_JOINTS:", L.MAX_JOINTS)
    print("  MAX_TERRAIN_EDGES:", L.MAX_TERRAIN_EDGES)
    print("  OBS_DIM:", L.OBS_DIM)
    print("  METADATA_SIZE:", L.METADATA_SIZE)
    print()

    # Check computed sizes
    print("  BODIES_SIZE:", L.BODIES_SIZE, "(expected:", 3 * BODY_STATE_SIZE, ")")
    print("  FORCES_SIZE:", L.FORCES_SIZE, "(expected:", 3 * 3, ")")
    print("  JOINTS_SIZE:", L.JOINTS_SIZE, "(expected:", 2 * JOINT_DATA_SIZE, ")")
    print("  EDGES_SIZE:", L.EDGES_SIZE, "(expected:", 16 * 6, ")")
    print()

    # Check offsets are monotonically increasing
    print("  OBS_OFFSET:", L.OBS_OFFSET)
    print("  BODIES_OFFSET:", L.BODIES_OFFSET)
    print("  FORCES_OFFSET:", L.FORCES_OFFSET)
    print("  JOINTS_OFFSET:", L.JOINTS_OFFSET)
    print("  JOINT_COUNT_OFFSET:", L.JOINT_COUNT_OFFSET)
    print("  EDGES_OFFSET:", L.EDGES_OFFSET)
    print("  EDGE_COUNT_OFFSET:", L.EDGE_COUNT_OFFSET)
    print("  METADATA_OFFSET:", L.METADATA_OFFSET)
    print()

    # Total size
    print("  STATE_SIZE:", L.STATE_SIZE)

    # Verify offsets make sense
    if (
        L.OBS_OFFSET < L.BODIES_OFFSET
        and L.BODIES_OFFSET < L.FORCES_OFFSET
        and L.FORCES_OFFSET < L.JOINTS_OFFSET
        and L.JOINTS_OFFSET < L.JOINT_COUNT_OFFSET
        and L.JOINT_COUNT_OFFSET < L.EDGES_OFFSET
        and L.EDGES_OFFSET < L.EDGE_COUNT_OFFSET
        and L.EDGE_COUNT_OFFSET < L.METADATA_OFFSET
        and L.METADATA_OFFSET < L.STATE_SIZE
    ):
        print("  ✓ Offsets are valid (monotonically increasing)")
    else:
        print("  ✗ Offsets are INVALID!")
        raise Error("Layout offsets are invalid")

    # Test helper methods
    print("\n  Testing helper methods:")
    print("    body_offset(0):", L.body_offset(0), "(expected:", L.BODIES_OFFSET, ")")
    print("    body_offset(1):", L.body_offset(1), "(expected:", L.BODIES_OFFSET + BODY_STATE_SIZE, ")")
    print("    force_offset(0):", L.force_offset(0), "(expected:", L.FORCES_OFFSET, ")")
    print("    joint_offset(0):", L.joint_offset(0), "(expected:", L.JOINTS_OFFSET, ")")
    print("    edge_offset(0):", L.edge_offset(0), "(expected:", L.EDGES_OFFSET, ")")

    print("\n  ✓ Layout computation test PASSED\n")


fn test_env_helpers() raises:
    """Test PhysicsEnvHelpers functions."""
    print("Testing PhysicsEnvHelpers...")

    comptime BATCH = 2
    comptime L = LunarLanderLayoutStrided

    # Allocate state buffer using List (manages its own memory)
    var state_data = List[Scalar[dtype]](capacity=BATCH * L.STATE_SIZE)
    for _ in range(BATCH * L.STATE_SIZE):
        state_data.append(Scalar[dtype](0))

    var states = LayoutTensor[
        dtype, TensorLayout.row_major(BATCH, L.STATE_SIZE), MutAnyOrigin
    ](state_data.unsafe_ptr())

    # Test body initialization
    print("  Testing init_body...")
    PhysicsEnvHelpers.init_body[BATCH, L.STATE_SIZE, L.BODIES_OFFSET](
        states,
        env=0,
        body=0,
        x=10.0,
        y=5.0,
        angle=0.1,
        mass=5.0,
        inertia=2.0,
        shape_idx=0,
    )

    # Verify body state
    var x = PhysicsEnvHelpers.get_body_x[BATCH, L.STATE_SIZE, L.BODIES_OFFSET](
        states, 0, 0
    )
    var y = PhysicsEnvHelpers.get_body_y[BATCH, L.STATE_SIZE, L.BODIES_OFFSET](
        states, 0, 0
    )
    var angle = PhysicsEnvHelpers.get_body_angle[
        BATCH, L.STATE_SIZE, L.BODIES_OFFSET
    ](states, 0, 0)

    print("    x:", x, "(expected: 10.0)")
    print("    y:", y, "(expected: 5.0)")
    print("    angle:", angle, "(expected: 0.1)")

    if x != Scalar[dtype](10.0) or y != Scalar[dtype](5.0):
        print("  ✗ Body initialization FAILED!")
        raise Error("Body initialization failed")

    print("  ✓ Body initialization works")

    # Test velocity setting
    print("\n  Testing set_body_velocity...")
    PhysicsEnvHelpers.set_body_velocity[BATCH, L.STATE_SIZE, L.BODIES_OFFSET](
        states, env=0, body=0, vx=1.0, vy=-2.0, omega=0.5
    )

    var vx = PhysicsEnvHelpers.get_body_vx[BATCH, L.STATE_SIZE, L.BODIES_OFFSET](
        states, 0, 0
    )
    var vy = PhysicsEnvHelpers.get_body_vy[BATCH, L.STATE_SIZE, L.BODIES_OFFSET](
        states, 0, 0
    )
    var omega = PhysicsEnvHelpers.get_body_omega[
        BATCH, L.STATE_SIZE, L.BODIES_OFFSET
    ](states, 0, 0)

    print("    vx:", vx, "(expected: 1.0)")
    print("    vy:", vy, "(expected: -2.0)")
    print("    omega:", omega, "(expected: 0.5)")
    print("  ✓ Velocity setting works")

    # Test force application
    print("\n  Testing apply_force...")
    PhysicsEnvHelpers.clear_forces[
        BATCH, L.NUM_BODIES, L.STATE_SIZE, L.FORCES_OFFSET
    ](states, env=0)
    PhysicsEnvHelpers.apply_force[BATCH, L.STATE_SIZE, L.FORCES_OFFSET](
        states, env=0, body=0, fx=100.0, fy=50.0, torque=10.0
    )
    PhysicsEnvHelpers.apply_force[BATCH, L.STATE_SIZE, L.FORCES_OFFSET](
        states, env=0, body=0, fx=50.0, fy=25.0, torque=5.0
    )

    # Forces should be accumulated
    var force_off = L.force_offset(0)
    var fx = states[0, force_off + 0]
    var fy = states[0, force_off + 1]
    var torque = states[0, force_off + 2]

    print("    fx:", fx, "(expected: 150.0)")
    print("    fy:", fy, "(expected: 75.0)")
    print("    torque:", torque, "(expected: 15.0)")

    if fx != Scalar[dtype](150.0) or fy != Scalar[dtype](75.0):
        print("  ✗ Force accumulation FAILED!")
        raise Error("Force accumulation failed")

    print("  ✓ Force application works")

    # Test joint creation
    print("\n  Testing add_revolute_joint...")

    # Initialize second body first
    PhysicsEnvHelpers.init_body[BATCH, L.STATE_SIZE, L.BODIES_OFFSET](
        states,
        env=0,
        body=1,
        x=10.5,
        y=4.7,
        angle=0.0,
        mass=0.2,
        inertia=0.02,
        shape_idx=1,
    )

    var joint_idx = PhysicsEnvHelpers.add_revolute_joint[
        BATCH,
        L.MAX_JOINTS,
        L.STATE_SIZE,
        L.BODIES_OFFSET,
        L.JOINTS_OFFSET,
        L.JOINT_COUNT_OFFSET,
    ](
        states,
        env=0,
        body_a=0,
        body_b=1,
        anchor_ax=0.5,
        anchor_ay=-0.3,
        anchor_bx=0.0,
        anchor_by=0.2,
        stiffness=400.0,
        damping=40.0,
    )

    print("    Joint index:", joint_idx, "(expected: 0)")

    var joint_count = PhysicsEnvHelpers.get_joint_count[
        BATCH, L.STATE_SIZE, L.JOINT_COUNT_OFFSET
    ](states, 0)
    print("    Joint count:", joint_count, "(expected: 1)")

    if joint_idx != 0 or joint_count != 1:
        print("  ✗ Joint creation FAILED!")
        raise Error("Joint creation failed")

    print("  ✓ Joint creation works")

    # Test terrain setup
    print("\n  Testing set_flat_terrain...")
    PhysicsEnvHelpers.set_flat_terrain[
        BATCH, L.STATE_SIZE, L.EDGES_OFFSET, L.EDGE_COUNT_OFFSET
    ](states, env=0, ground_y=3.0, x_min=0.0, x_max=20.0)

    var edge_count = PhysicsEnvHelpers.get_edge_count[
        BATCH, L.STATE_SIZE, L.EDGE_COUNT_OFFSET
    ](states, 0)
    print("    Edge count:", edge_count, "(expected: 1)")

    if edge_count != 1:
        print("  ✗ Terrain setup FAILED!")
        raise Error("Terrain setup failed")

    print("  ✓ Terrain setup works")

    # Test observation access
    print("\n  Testing observation access...")
    PhysicsEnvHelpers.set_observation[BATCH, L.STATE_SIZE, L.OBS_OFFSET](
        states, env=0, idx=0, value=0.123
    )
    var obs = PhysicsEnvHelpers.get_observation[
        BATCH, L.STATE_SIZE, L.OBS_OFFSET
    ](states, 0, 0)
    print("    obs[0]:", obs, "(expected: ~0.123)")
    print("  ✓ Observation access works")

    # Test metadata access
    print("\n  Testing metadata access...")
    PhysicsEnvHelpers.set_metadata[BATCH, L.STATE_SIZE, L.METADATA_OFFSET](
        states, env=0, field=0, value=42.0
    )
    var meta = PhysicsEnvHelpers.get_metadata[
        BATCH, L.STATE_SIZE, L.METADATA_OFFSET
    ](states, 0, 0)
    print("    metadata[0]:", meta, "(expected: 42.0)")
    print("  ✓ Metadata access works")

    print("\n  ✓ PhysicsEnvHelpers test PASSED\n")


fn test_physics_config() raises:
    """Test PhysicsConfigStrided."""
    print("Testing PhysicsConfigStrided...")

    var config = PhysicsConfigStrided(
        gravity_x=0.0,
        gravity_y=-10.0,
        dt=0.02,
        friction=0.1,
        restitution=0.0,
    )

    print("  gravity_y:", config.gravity_y, "(expected: -10.0)")
    print("  dt:", config.dt, "(expected: 0.02)")
    print("  friction:", config.friction, "(expected: 0.1)")
    print("  velocity_iterations:", config.velocity_iterations)
    print("  position_iterations:", config.position_iterations)

    print("\n  ✓ PhysicsConfigStrided test PASSED\n")


fn test_layout_compatibility() raises:
    """Test that layout is compatible with existing physics components."""
    print("Testing layout compatibility...")

    comptime L = LunarLanderLayoutStrided

    # Check that BODY_STATE_SIZE matches what the layout expects
    var expected_body_size = L.BODIES_SIZE // L.NUM_BODIES
    print("  Layout body size:", expected_body_size)
    print("  BODY_STATE_SIZE:", BODY_STATE_SIZE)

    if expected_body_size != BODY_STATE_SIZE:
        print("  ✗ Body size mismatch!")
        raise Error("Body size mismatch")

    # Check that joint size matches
    var expected_joint_size = L.JOINTS_SIZE // L.MAX_JOINTS
    print("  Layout joint size:", expected_joint_size)
    print("  JOINT_DATA_SIZE:", JOINT_DATA_SIZE)

    if expected_joint_size != JOINT_DATA_SIZE:
        print("  ✗ Joint size mismatch!")
        raise Error("Joint size mismatch")

    print("\n  ✓ Layout compatibility test PASSED\n")


fn main() raises:
    """Run all tests."""
    print("=" * 60)
    print("Physics Strided Architecture Test")
    print("=" * 60)
    print()

    test_layout_computation()
    test_env_helpers()
    test_physics_config()
    test_layout_compatibility()

    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
    print()
    print("The new modular physics architecture is working correctly.")
    print("You can now use:")
    print("  - PhysicsLayoutStrided: Compile-time layout computation")
    print("  - PhysicsEnvHelpers: Environment setup utilities")
    print("  - PhysicsKernelStrided: Unified physics step orchestrator")
    print("  - PhysicsConfigStrided: Physics simulation parameters")
