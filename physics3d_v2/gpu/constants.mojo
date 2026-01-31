"""Physics3D v2 GPU constants - Multi-body flat buffer layout.

GPU kernels require flat buffer layouts. This file defines:
1. Per-body state layout (positions, velocities, etc.)
2. Per-contact layout (body indices, normal, depth, etc.)
3. Per-joint layout (body indices, anchors, axis, impulses)
4. Total buffer size computation

Buffer layout for multi-body:
  [BATCH, NUM_BODIES * BODY_STATE_SIZE + MAX_CONTACTS * CONTACT_STATE_SIZE + MAX_JOINTS * JOINT_STATE_SIZE + METADATA_SIZE]

Where:
  - BODY_STATE_SIZE = 22 floats per body
  - CONTACT_STATE_SIZE = 12 floats per contact
  - JOINT_STATE_SIZE = 16 floats per joint
  - METADATA_SIZE = 4 floats (num_contacts, num_joints, padding)
"""

# =============================================================================
# GPU Configuration
# =============================================================================

comptime TPB: Int = 256  # Threads per block (optimal for most GPUs)
comptime TILE: Int = 16  # Tile size for 2D operations


comptime MODEL_BODY_SIZE: Int = 9
comptime MODEL_IDX_MASS: Int = 0
comptime MODEL_IDX_INV_MASS: Int = 1
comptime MODEL_IDX_RADIUS: Int = 2
comptime MODEL_IDX_IXX: Int = 3
comptime MODEL_IDX_IYY: Int = 4
comptime MODEL_IDX_IZZ: Int = 5
comptime MODEL_IDX_INV_IXX: Int = 6
comptime MODEL_IDX_INV_IYY: Int = 7
comptime MODEL_IDX_INV_IZZ: Int = 8
# =============================================================================
# Per-Body State Layout (22 floats per body)
# =============================================================================

# Position (3 floats)
comptime BODY_POS_OFFSET: Int = 0
comptime BODY_IDX_PX: Int = 0
comptime BODY_IDX_PY: Int = 1
comptime BODY_IDX_PZ: Int = 2

# Quaternion (4 floats) [qx, qy, qz, qw]
comptime BODY_QUAT_OFFSET: Int = 3
comptime BODY_IDX_QX: Int = 3
comptime BODY_IDX_QY: Int = 4
comptime BODY_IDX_QZ: Int = 5
comptime BODY_IDX_QW: Int = 6

# Linear velocity (3 floats)
comptime BODY_VEL_OFFSET: Int = 7
comptime BODY_IDX_VX: Int = 7
comptime BODY_IDX_VY: Int = 8
comptime BODY_IDX_VZ: Int = 9

# Angular velocity (3 floats)
comptime BODY_ANGVEL_OFFSET: Int = 10
comptime BODY_IDX_WX: Int = 10
comptime BODY_IDX_WY: Int = 11
comptime BODY_IDX_WZ: Int = 12

# Linear acceleration (3 floats)
comptime BODY_ACC_OFFSET: Int = 13
comptime BODY_IDX_AX: Int = 13
comptime BODY_IDX_AY: Int = 14
comptime BODY_IDX_AZ: Int = 15

# Angular acceleration (3 floats)
comptime BODY_ANGACC_OFFSET: Int = 16
comptime BODY_IDX_ALPHA_X: Int = 16
comptime BODY_IDX_ALPHA_Y: Int = 17
comptime BODY_IDX_ALPHA_Z: Int = 18

# Applied forces (3 floats) - for external forces
comptime BODY_FORCE_OFFSET: Int = 19
comptime BODY_IDX_FX: Int = 19
comptime BODY_IDX_FY: Int = 20
comptime BODY_IDX_FZ: Int = 21

# Total body state size
comptime BODY_STATE_SIZE: Int = 22


# =============================================================================
# Per-Contact State Layout (12 floats per contact)
# =============================================================================

# Body indices (2 floats - stored as floats for GPU compatibility)
comptime CONTACT_IDX_BODY_A: Int = 0  # First body index
comptime CONTACT_IDX_BODY_B: Int = 1  # Second body index (-1 for ground)

# Contact position (3 floats)
comptime CONTACT_IDX_POS_X: Int = 2
comptime CONTACT_IDX_POS_Y: Int = 3
comptime CONTACT_IDX_POS_Z: Int = 4

# Contact normal (3 floats) - points from A to B
comptime CONTACT_IDX_NX: Int = 5
comptime CONTACT_IDX_NY: Int = 6
comptime CONTACT_IDX_NZ: Int = 7

# Signed distance (1 float) - negative = penetration
comptime CONTACT_IDX_DIST: Int = 8

# Impulses for warm starting (3 floats)
comptime CONTACT_IDX_IMPULSE_N: Int = 9  # Normal impulse
comptime CONTACT_IDX_IMPULSE_T1: Int = 10  # Tangent impulse 1
comptime CONTACT_IDX_IMPULSE_T2: Int = 11  # Tangent impulse 2

# Total contact state size
comptime CONTACT_STATE_SIZE: Int = 12


# =============================================================================
# Per-Joint State Layout (16 floats per joint)
# =============================================================================

# Body indices (2 floats - stored as floats for GPU compatibility)
comptime JOINT_IDX_PARENT: Int = 0  # Parent body index (-1 for world)
comptime JOINT_IDX_CHILD: Int = 1  # Child body index

# Anchor point on parent (3 floats) - local frame or world if parent=-1
comptime JOINT_IDX_ANCHOR_PX: Int = 2
comptime JOINT_IDX_ANCHOR_PY: Int = 3
comptime JOINT_IDX_ANCHOR_PZ: Int = 4

# Anchor point on child (3 floats) - local frame
comptime JOINT_IDX_ANCHOR_CX: Int = 5
comptime JOINT_IDX_ANCHOR_CY: Int = 6
comptime JOINT_IDX_ANCHOR_CZ: Int = 7

# Hinge axis (3 floats) - in parent's local frame or world if parent=-1
comptime JOINT_IDX_AXIS_X: Int = 8
comptime JOINT_IDX_AXIS_Y: Int = 9
comptime JOINT_IDX_AXIS_Z: Int = 10

# Accumulated impulses for warm starting (5 floats)
comptime JOINT_IDX_IMPULSE_LX: Int = 11  # Linear impulse X
comptime JOINT_IDX_IMPULSE_LY: Int = 12  # Linear impulse Y
comptime JOINT_IDX_IMPULSE_LZ: Int = 13  # Linear impulse Z
comptime JOINT_IDX_IMPULSE_AX: Int = 14  # Angular impulse 1
comptime JOINT_IDX_IMPULSE_AY: Int = 15  # Angular impulse 2

# Total joint state size
comptime JOINT_STATE_SIZE: Int = 16


# =============================================================================
# Metadata Layout (4 floats)
# =============================================================================

comptime META_IDX_NUM_CONTACTS: Int = 0  # Current number of active contacts
comptime META_IDX_NUM_JOINTS: Int = 1  # Current number of active joints
comptime META_IDX_PADDING_2: Int = 2
comptime META_IDX_PADDING_3: Int = 3

comptime METADATA_SIZE: Int = 4


# =============================================================================
# Helper Functions for Computing Offsets
# =============================================================================


fn compute_state_size[NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int = 0]() -> Int:
    """Compute total state buffer size per environment.

    Args:
        NUM_BODIES: Number of bodies in the simulation.
        MAX_CONTACTS: Maximum number of contacts.
        MAX_JOINTS: Maximum number of joints (default 0).

    Returns:
        Total buffer size in number of scalars.
    """
    return (
        NUM_BODIES * BODY_STATE_SIZE
        + MAX_CONTACTS * CONTACT_STATE_SIZE
        + MAX_JOINTS * JOINT_STATE_SIZE
        + METADATA_SIZE
    )


fn body_offset[NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int = 0](body_idx: Int) -> Int:
    """Get offset to start of body state within environment state."""
    return body_idx * BODY_STATE_SIZE


fn contact_offset[NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int = 0](contact_idx: Int) -> Int:
    """Get offset to start of contact state within environment state."""
    return NUM_BODIES * BODY_STATE_SIZE + contact_idx * CONTACT_STATE_SIZE


fn joint_offset[NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int = 0](joint_idx: Int) -> Int:
    """Get offset to start of joint state within environment state."""
    return (
        NUM_BODIES * BODY_STATE_SIZE
        + MAX_CONTACTS * CONTACT_STATE_SIZE
        + joint_idx * JOINT_STATE_SIZE
    )


fn metadata_offset[NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int = 0]() -> Int:
    """Get offset to metadata within environment state."""
    return (
        NUM_BODIES * BODY_STATE_SIZE
        + MAX_CONTACTS * CONTACT_STATE_SIZE
        + MAX_JOINTS * JOINT_STATE_SIZE
    )


# =============================================================================
# Geometry Types (same as CPU)
# =============================================================================

comptime GEOM_PLANE: Int = 0
comptime GEOM_SPHERE: Int = 1


# =============================================================================
# Physics Defaults
# =============================================================================

comptime DEFAULT_GRAVITY_Z: Float32 = -9.81
comptime DEFAULT_TIMESTEP: Float32 = 0.01
comptime DEFAULT_RESTITUTION: Float32 = 0.0
comptime DEFAULT_BAUMGARTE: Float32 = 0.2
comptime DEFAULT_SLOP: Float32 = 0.001


# =============================================================================
# Legacy Single-Body Layout (for backward compatibility)
# =============================================================================

# Old STATE_SIZE for single body (36 floats)
# Keeping for reference, but new code should use multi-body layout
comptime LEGACY_STATE_SIZE: Int = 36

# Old field indices (single body)
comptime IDX_X: Int = 0
comptime IDX_Y: Int = 1
comptime IDX_Z: Int = 2
comptime IDX_QX: Int = 3
comptime IDX_QY: Int = 4
comptime IDX_QZ: Int = 5
comptime IDX_QW: Int = 6
comptime IDX_VX: Int = 7
comptime IDX_VY: Int = 8
comptime IDX_VZ: Int = 9
comptime IDX_WX: Int = 10
comptime IDX_WY: Int = 11
comptime IDX_WZ: Int = 12
comptime IDX_AX: Int = 13
comptime IDX_AY: Int = 14
comptime IDX_AZ: Int = 15
comptime IDX_ALPHA_X: Int = 16
comptime IDX_ALPHA_Y: Int = 17
comptime IDX_ALPHA_Z: Int = 18
comptime IDX_FX: Int = 19
comptime IDX_FY: Int = 20
comptime IDX_FZ: Int = 21
comptime IDX_TAU_X: Int = 22
comptime IDX_TAU_Y: Int = 23
comptime IDX_TAU_Z: Int = 24
comptime IDX_XPOS_X: Int = 25
comptime IDX_XPOS_Y: Int = 26
comptime IDX_XPOS_Z: Int = 27
comptime IDX_CONTACT_ACTIVE: Int = 28
comptime IDX_CONTACT_DEPTH: Int = 29
comptime IDX_CONTACT_NX: Int = 30
comptime IDX_CONTACT_NY: Int = 31
comptime IDX_CONTACT_NZ: Int = 32
comptime IDX_CONTACT_PX: Int = 33
comptime IDX_CONTACT_PY: Int = 34
comptime IDX_CONTACT_PZ: Int = 35
