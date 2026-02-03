"""GPU buffer layout constants for Generalized Coordinates engine.

GPU kernels require flat buffer layouts. This file defines the state buffer
layout for the GC engine, which differs from the Cartesian engine:
- Primary state is qpos/qvel (joint space) not body positions/velocities
- Body positions (xpos, xquat) are computed values, stored for collision detection

Buffer layout per environment:
  [qpos: NQ | qvel: NV | qacc: NV | qfrc: NV |
   xpos: NBODY*3 | xquat: NBODY*4 | xvel: NBODY*3 | xangvel: NBODY*3 |
   contacts: MAX_CONTACTS*12 | metadata: 4]

Model buffer (static, same for all environments):
  Per body: [mass, inv_mass, inertia(3), inv_inertia(3), pos(3), quat(4),
             parent, geom_type, radius, half_length, half_x, half_y, half_z]
  Per joint: [jnt_type, body_id, qpos_adr, dof_adr, pos(3), axis(3), tau_limit]
  Metadata: [NBODY, NJOINT, gravity(4), timestep, ground_z, friction]
"""

# =============================================================================
# GPU Configuration (same as Cartesian engine)
# =============================================================================

comptime TPB: Int = 256  # Threads per block
comptime TILE: Int = 16  # Tile size for 2D operations


# =============================================================================
# State Buffer Layout - Joint Space (qpos, qvel, qacc, qfrc)
# =============================================================================

# These are computed as offsets based on NQ and NV parameters
# For a system with NQ total qpos and NV total qvel:
#
#   qpos: [0, NQ)
#   qvel: [NQ, NQ + NV)
#   qacc: [NQ + NV, NQ + 2*NV)
#   qfrc: [NQ + 2*NV, NQ + 3*NV)

fn gc_qpos_offset[NQ: Int, NV: Int]() -> Int:
    """Offset to qpos array (always 0)."""
    return 0


fn gc_qvel_offset[NQ: Int, NV: Int]() -> Int:
    """Offset to qvel array."""
    return NQ


fn gc_qacc_offset[NQ: Int, NV: Int]() -> Int:
    """Offset to qacc array."""
    return NQ + NV


fn gc_qfrc_offset[NQ: Int, NV: Int]() -> Int:
    """Offset to qfrc array."""
    return NQ + 2 * NV


# =============================================================================
# State Buffer Layout - World Space (xpos, xquat, xvel, xangvel)
# =============================================================================

fn gc_xpos_offset[NQ: Int, NV: Int, NBODY: Int]() -> Int:
    """Offset to xpos array (body world positions)."""
    return NQ + 3 * NV


fn gc_xquat_offset[NQ: Int, NV: Int, NBODY: Int]() -> Int:
    """Offset to xquat array (body world orientations)."""
    return NQ + 3 * NV + NBODY * 3


fn gc_xvel_offset[NQ: Int, NV: Int, NBODY: Int]() -> Int:
    """Offset to xvel array (body world linear velocities)."""
    return NQ + 3 * NV + NBODY * 3 + NBODY * 4


fn gc_xangvel_offset[NQ: Int, NV: Int, NBODY: Int]() -> Int:
    """Offset to xangvel array (body world angular velocities)."""
    return NQ + 3 * NV + NBODY * 3 + NBODY * 4 + NBODY * 3


# =============================================================================
# State Buffer Layout - Contacts
# =============================================================================

# Contact layout (same as Cartesian engine: 12 floats per contact)
comptime GC_CONTACT_SIZE: Int = 12

comptime GC_CONTACT_IDX_BODY_A: Int = 0
comptime GC_CONTACT_IDX_BODY_B: Int = 1
comptime GC_CONTACT_IDX_POS_X: Int = 2
comptime GC_CONTACT_IDX_POS_Y: Int = 3
comptime GC_CONTACT_IDX_POS_Z: Int = 4
comptime GC_CONTACT_IDX_NX: Int = 5
comptime GC_CONTACT_IDX_NY: Int = 6
comptime GC_CONTACT_IDX_NZ: Int = 7
comptime GC_CONTACT_IDX_DIST: Int = 8
comptime GC_CONTACT_IDX_IMPULSE_N: Int = 9
comptime GC_CONTACT_IDX_IMPULSE_T1: Int = 10
comptime GC_CONTACT_IDX_IMPULSE_T2: Int = 11


fn gc_contacts_offset[NQ: Int, NV: Int, NBODY: Int]() -> Int:
    """Offset to contacts array."""
    return NQ + 3 * NV + NBODY * 3 + NBODY * 4 + NBODY * 3 + NBODY * 3


fn gc_contact_offset[NQ: Int, NV: Int, NBODY: Int](contact_idx: Int) -> Int:
    """Offset to a specific contact."""
    return gc_contacts_offset[NQ, NV, NBODY]() + contact_idx * GC_CONTACT_SIZE


# =============================================================================
# State Buffer Layout - Metadata
# =============================================================================

comptime GC_METADATA_SIZE: Int = 4

comptime GC_META_IDX_NUM_CONTACTS: Int = 0
comptime GC_META_IDX_PADDING_1: Int = 1
comptime GC_META_IDX_PADDING_2: Int = 2
comptime GC_META_IDX_PADDING_3: Int = 3


fn gc_metadata_offset[NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int]() -> Int:
    """Offset to metadata."""
    return gc_contacts_offset[NQ, NV, NBODY]() + MAX_CONTACTS * GC_CONTACT_SIZE


# =============================================================================
# Total State Size Computation
# =============================================================================


fn gc_state_size[NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int]() -> Int:
    """Compute total state buffer size per environment.

    Returns:
        Total size in number of scalars.
    """
    return (
        NQ  # qpos
        + 3 * NV  # qvel + qacc + qfrc
        + NBODY * 3  # xpos
        + NBODY * 4  # xquat
        + NBODY * 3  # xvel
        + NBODY * 3  # xangvel
        + MAX_CONTACTS * GC_CONTACT_SIZE
        + GC_METADATA_SIZE
    )


# =============================================================================
# Model Buffer Layout - Per Body
# =============================================================================

comptime GC_MODEL_BODY_SIZE: Int = 22

comptime GC_BODY_IDX_MASS: Int = 0
comptime GC_BODY_IDX_INV_MASS: Int = 1
comptime GC_BODY_IDX_IXX: Int = 2
comptime GC_BODY_IDX_IYY: Int = 3
comptime GC_BODY_IDX_IZZ: Int = 4
comptime GC_BODY_IDX_INV_IXX: Int = 5
comptime GC_BODY_IDX_INV_IYY: Int = 6
comptime GC_BODY_IDX_INV_IZZ: Int = 7
comptime GC_BODY_IDX_POS_X: Int = 8  # Local position in parent frame
comptime GC_BODY_IDX_POS_Y: Int = 9
comptime GC_BODY_IDX_POS_Z: Int = 10
comptime GC_BODY_IDX_QUAT_X: Int = 11  # Local orientation in parent frame
comptime GC_BODY_IDX_QUAT_Y: Int = 12
comptime GC_BODY_IDX_QUAT_Z: Int = 13
comptime GC_BODY_IDX_QUAT_W: Int = 14
comptime GC_BODY_IDX_PARENT: Int = 15  # Parent body index (-1 for world)
comptime GC_BODY_IDX_GEOM_TYPE: Int = 16
comptime GC_BODY_IDX_RADIUS: Int = 17
comptime GC_BODY_IDX_HALF_LENGTH: Int = 18
comptime GC_BODY_IDX_HALF_X: Int = 19
comptime GC_BODY_IDX_HALF_Y: Int = 20
comptime GC_BODY_IDX_HALF_Z: Int = 21


fn gc_model_body_offset(body_idx: Int) -> Int:
    """Offset to a specific body in model buffer."""
    return body_idx * GC_MODEL_BODY_SIZE


# =============================================================================
# Model Buffer Layout - Per Joint
# =============================================================================

comptime GC_MODEL_JOINT_SIZE: Int = 11

comptime GC_JOINT_IDX_TYPE: Int = 0  # JNT_FREE, JNT_BALL, JNT_SLIDE, JNT_HINGE
comptime GC_JOINT_IDX_BODY_ID: Int = 1
comptime GC_JOINT_IDX_QPOS_ADR: Int = 2
comptime GC_JOINT_IDX_DOF_ADR: Int = 3
comptime GC_JOINT_IDX_POS_X: Int = 4
comptime GC_JOINT_IDX_POS_Y: Int = 5
comptime GC_JOINT_IDX_POS_Z: Int = 6
comptime GC_JOINT_IDX_AXIS_X: Int = 7
comptime GC_JOINT_IDX_AXIS_Y: Int = 8
comptime GC_JOINT_IDX_AXIS_Z: Int = 9
comptime GC_JOINT_IDX_TAU_LIMIT: Int = 10


fn gc_model_joint_offset[NBODY: Int](joint_idx: Int) -> Int:
    """Offset to a specific joint in model buffer."""
    return NBODY * GC_MODEL_BODY_SIZE + joint_idx * GC_MODEL_JOINT_SIZE


# =============================================================================
# Model Buffer Layout - Global Metadata
# =============================================================================

comptime GC_MODEL_META_SIZE: Int = 8

comptime GC_MODEL_META_IDX_NBODY: Int = 0
comptime GC_MODEL_META_IDX_NJOINT: Int = 1
comptime GC_MODEL_META_IDX_GRAVITY_X: Int = 2
comptime GC_MODEL_META_IDX_GRAVITY_Y: Int = 3
comptime GC_MODEL_META_IDX_GRAVITY_Z: Int = 4
comptime GC_MODEL_META_IDX_TIMESTEP: Int = 5
comptime GC_MODEL_META_IDX_GROUND_Z: Int = 6
comptime GC_MODEL_META_IDX_FRICTION: Int = 7


fn gc_model_metadata_offset[NBODY: Int, NJOINT: Int]() -> Int:
    """Offset to model metadata."""
    return NBODY * GC_MODEL_BODY_SIZE + NJOINT * GC_MODEL_JOINT_SIZE


fn gc_model_size[NBODY: Int, NJOINT: Int]() -> Int:
    """Total model buffer size."""
    return (
        NBODY * GC_MODEL_BODY_SIZE
        + NJOINT * GC_MODEL_JOINT_SIZE
        + GC_MODEL_META_SIZE
    )


# =============================================================================
# Geometry Types (same as Cartesian engine)
# =============================================================================

comptime GC_GEOM_PLANE: Int = 0
comptime GC_GEOM_SPHERE: Int = 1
comptime GC_GEOM_CAPSULE: Int = 2
comptime GC_GEOM_BOX: Int = 3


# =============================================================================
# Joint Types (same as joint_types.mojo but duplicated for GPU code)
# =============================================================================

comptime GC_JNT_FREE: Int = 0
comptime GC_JNT_BALL: Int = 1
comptime GC_JNT_SLIDE: Int = 2
comptime GC_JNT_HINGE: Int = 3
