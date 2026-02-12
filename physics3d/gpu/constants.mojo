"""Physics3D GPU constants - Flat buffer layout for GPU kernels.

Primary state is qpos/qvel (joint space). Body positions (xpos, xquat)
are computed via forward kinematics and stored for collision detection.

State buffer layout per environment:
  [qpos: NQ | qvel: NV | qacc: NV | qfrc: NV |
   xpos: NBODY*3 | xquat: NBODY*4 | xvel: NBODY*3 | xangvel: NBODY*3 |
   contacts: MAX_CONTACTS*CONTACT_SIZE | metadata: METADATA_SIZE]

Model buffer (static, same for all environments):
  Per body (MODEL_BODY_SIZE=16): [mass, inv_mass, inertia(3), inv_inertia(3),
    pos(3), quat(4), parent]
  Per joint (MODEL_JOINT_SIZE=18): [type, body_id, qpos_adr, dof_adr,
    pos(3), axis(3), tau_limit, range_min/max, armature, damping, stiffness, springref, frictionloss]
  Metadata (MODEL_META_SIZE=18): [NBODY, NJOINT, gravity(3), timestep, ground_z, friction,
    solref_contact(2), solimp_contact(3), solref_limit(2), solimp_limit(3)]
  Curriculum (MODEL_CURRICULUM_SIZE=8): [up to 8 curriculum parameters]
  Per geom (MODEL_GEOM_SIZE=17): [type, body, pos(3), quat(4), radius, half_length,
    half_x/y/z, friction, contype, conaffinity]
"""

# =============================================================================
# GPU Configuration
# =============================================================================

comptime TPB: Int = 256  # Threads per block (optimal for most GPUs)
comptime TILE: Int = 8  # Tile size for 2D operations


# =============================================================================
# Physics Defaults
# =============================================================================

comptime DEFAULT_GRAVITY_Z: Float32 = -9.81
comptime DEFAULT_TIMESTEP: Float32 = 0.01
comptime DEFAULT_RESTITUTION: Float32 = 0.0
comptime MAX_POS_CORRECTION_VEL: Float32 = 10.0  # Legacy, unused after accel-level migration


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


fn qpos_offset[NQ: Int, NV: Int]() -> Int:
    """Offset to qpos array (always 0)."""
    return 0


fn qvel_offset[NQ: Int, NV: Int]() -> Int:
    """Offset to qvel array."""
    return NQ


fn qacc_offset[NQ: Int, NV: Int]() -> Int:
    """Offset to qacc array."""
    return NQ + NV


fn qfrc_offset[NQ: Int, NV: Int]() -> Int:
    """Offset to qfrc array."""
    return NQ + 2 * NV


# =============================================================================
# State Buffer Layout - World Space (xpos, xquat, xvel, xangvel)
# =============================================================================


fn xpos_offset[NQ: Int, NV: Int, NBODY: Int]() -> Int:
    """Offset to xpos array (body world positions)."""
    return NQ + 3 * NV


fn xquat_offset[NQ: Int, NV: Int, NBODY: Int]() -> Int:
    """Offset to xquat array (body world orientations)."""
    return NQ + 3 * NV + NBODY * 3


fn xvel_offset[NQ: Int, NV: Int, NBODY: Int]() -> Int:
    """Offset to xvel array (body world linear velocities)."""
    return NQ + 3 * NV + NBODY * 3 + NBODY * 4


fn xangvel_offset[NQ: Int, NV: Int, NBODY: Int]() -> Int:
    """Offset to xangvel array (body world angular velocities)."""
    return NQ + 3 * NV + NBODY * 3 + NBODY * 4 + NBODY * 3


# =============================================================================
# State Buffer Layout - Contacts
# =============================================================================

# Contact layout (same as Cartesian engine: 12 floats per contact)
comptime CONTACT_SIZE: Int = 13

comptime CONTACT_IDX_BODY_A: Int = 0
comptime CONTACT_IDX_BODY_B: Int = 1
comptime CONTACT_IDX_POS_X: Int = 2
comptime CONTACT_IDX_POS_Y: Int = 3
comptime CONTACT_IDX_POS_Z: Int = 4
comptime CONTACT_IDX_NX: Int = 5
comptime CONTACT_IDX_NY: Int = 6
comptime CONTACT_IDX_NZ: Int = 7
comptime CONTACT_IDX_DIST: Int = 8
comptime CONTACT_IDX_FORCE_N: Int = 9
comptime CONTACT_IDX_FORCE_T1: Int = 10
comptime CONTACT_IDX_FORCE_T2: Int = 11
comptime CONTACT_IDX_FRICTION: Int = 12


fn contacts_offset[NQ: Int, NV: Int, NBODY: Int]() -> Int:
    """Offset to contacts array."""
    return NQ + 3 * NV + NBODY * 3 + NBODY * 4 + NBODY * 3 + NBODY * 3


fn contact_offset[NQ: Int, NV: Int, NBODY: Int](contact_idx: Int) -> Int:
    """Offset to a specific contact."""
    return contacts_offset[NQ, NV, NBODY]() + contact_idx * CONTACT_SIZE


# =============================================================================
# State Buffer Layout - Metadata
# =============================================================================

comptime METADATA_SIZE: Int = 4

comptime META_IDX_NUM_CONTACTS: Int = 0
comptime META_IDX_STEP_COUNT: Int = 1  # Episode step counter for truncation
comptime META_IDX_PREV_X: Int = 2  # Previous x position for velocity computation
comptime META_IDX_PADDING_3: Int = 3


fn metadata_offset[NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int]() -> Int:
    """Offset to metadata."""
    return contacts_offset[NQ, NV, NBODY]() + MAX_CONTACTS * CONTACT_SIZE


# =============================================================================
# Total State Size Computation
# =============================================================================


fn state_size[NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int]() -> Int:
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
        + MAX_CONTACTS * CONTACT_SIZE
        + METADATA_SIZE
    )


# =============================================================================
# Model Buffer Layout - Per Body
# =============================================================================

comptime MODEL_BODY_SIZE: Int = 16

comptime BODY_IDX_MASS: Int = 0
comptime BODY_IDX_INV_MASS: Int = 1
comptime BODY_IDX_IXX: Int = 2
comptime BODY_IDX_IYY: Int = 3
comptime BODY_IDX_IZZ: Int = 4
comptime BODY_IDX_INV_IXX: Int = 5
comptime BODY_IDX_INV_IYY: Int = 6
comptime BODY_IDX_INV_IZZ: Int = 7
comptime BODY_IDX_POS_X: Int = 8  # Local position in parent frame
comptime BODY_IDX_POS_Y: Int = 9
comptime BODY_IDX_POS_Z: Int = 10
comptime BODY_IDX_QUAT_X: Int = 11  # Local orientation in parent frame
comptime BODY_IDX_QUAT_Y: Int = 12
comptime BODY_IDX_QUAT_Z: Int = 13
comptime BODY_IDX_QUAT_W: Int = 14
comptime BODY_IDX_PARENT: Int = 15  # Parent body index (-1 for world)


fn model_body_offset(body_idx: Int) -> Int:
    """Offset to a specific body in model buffer."""
    return body_idx * MODEL_BODY_SIZE


# =============================================================================
# Model Buffer Layout - Per Joint
# =============================================================================

comptime MODEL_JOINT_SIZE: Int = 18  # Extended to include range limits + armature + damping + stiffness + springref + frictionloss

comptime JOINT_IDX_TYPE: Int = 0  # JNT_FREE, JNT_BALL, JNT_SLIDE, JNT_HINGE
comptime JOINT_IDX_BODY_ID: Int = 1
comptime JOINT_IDX_QPOS_ADR: Int = 2
comptime JOINT_IDX_DOF_ADR: Int = 3
comptime JOINT_IDX_POS_X: Int = 4
comptime JOINT_IDX_POS_Y: Int = 5
comptime JOINT_IDX_POS_Z: Int = 6
comptime JOINT_IDX_AXIS_X: Int = 7
comptime JOINT_IDX_AXIS_Y: Int = 8
comptime JOINT_IDX_AXIS_Z: Int = 9
comptime JOINT_IDX_TAU_LIMIT: Int = 10
comptime JOINT_IDX_RANGE_MIN: Int = 11  # Minimum position (radians for hinge, meters for slide)
comptime JOINT_IDX_RANGE_MAX: Int = 12  # Maximum position
comptime JOINT_IDX_ARMATURE: Int = 13  # Rotor inertia (added to M diagonal)
comptime JOINT_IDX_DAMPING: Int = 14  # Passive joint damping
comptime JOINT_IDX_STIFFNESS: Int = 15  # Passive joint stiffness (spring)
comptime JOINT_IDX_SPRINGREF: Int = 16  # Spring reference position (rest position)
comptime JOINT_IDX_FRICTIONLOSS: Int = 17  # Dry friction loss (Coulomb friction)


fn model_joint_offset[NBODY: Int](joint_idx: Int) -> Int:
    """Offset to a specific joint in model buffer."""
    return NBODY * MODEL_BODY_SIZE + joint_idx * MODEL_JOINT_SIZE


# =============================================================================
# Model Buffer Layout - Global Metadata
# =============================================================================

comptime MODEL_META_SIZE: Int = 18

comptime MODEL_META_IDX_NBODY: Int = 0
comptime MODEL_META_IDX_NJOINT: Int = 1
comptime MODEL_META_IDX_GRAVITY_X: Int = 2
comptime MODEL_META_IDX_GRAVITY_Y: Int = 3
comptime MODEL_META_IDX_GRAVITY_Z: Int = 4
comptime MODEL_META_IDX_TIMESTEP: Int = 5
comptime MODEL_META_IDX_GROUND_Z: Int = 6
comptime MODEL_META_IDX_FRICTION: Int = 7
# solref/solimp contact parameters (MuJoCo impedance model)
comptime MODEL_META_IDX_SOLREF_CONTACT_0: Int = 8   # timeconst
comptime MODEL_META_IDX_SOLREF_CONTACT_1: Int = 9   # dampratio
comptime MODEL_META_IDX_SOLIMP_CONTACT_0: Int = 10  # dmin
comptime MODEL_META_IDX_SOLIMP_CONTACT_1: Int = 11  # dmax
comptime MODEL_META_IDX_SOLIMP_CONTACT_2: Int = 12  # width
# solref/solimp limit parameters (MuJoCo impedance model)
comptime MODEL_META_IDX_SOLREF_LIMIT_0: Int = 13    # timeconst
comptime MODEL_META_IDX_SOLREF_LIMIT_1: Int = 14    # dampratio
comptime MODEL_META_IDX_SOLIMP_LIMIT_0: Int = 15    # dmin
comptime MODEL_META_IDX_SOLIMP_LIMIT_1: Int = 16    # dmax
comptime MODEL_META_IDX_SOLIMP_LIMIT_2: Int = 17    # width


fn model_metadata_offset[NBODY: Int, NJOINT: Int]() -> Int:
    """Offset to model metadata."""
    return NBODY * MODEL_BODY_SIZE + NJOINT * MODEL_JOINT_SIZE


# =============================================================================
# Model Buffer Layout - Unified Geoms (body-attached + static)
# =============================================================================

comptime MODEL_GEOM_SIZE: Int = 17  # Per unified geom

comptime GEOM_IDX_TYPE: Int = 0
comptime GEOM_IDX_BODY: Int = 1  # Body index (-1 for static)
comptime GEOM_IDX_POS_X: Int = 2
comptime GEOM_IDX_POS_Y: Int = 3
comptime GEOM_IDX_POS_Z: Int = 4
comptime GEOM_IDX_QUAT_X: Int = 5
comptime GEOM_IDX_QUAT_Y: Int = 6
comptime GEOM_IDX_QUAT_Z: Int = 7
comptime GEOM_IDX_QUAT_W: Int = 8
comptime GEOM_IDX_RADIUS: Int = 9
comptime GEOM_IDX_HALF_LENGTH: Int = 10
comptime GEOM_IDX_HALF_X: Int = 11
comptime GEOM_IDX_HALF_Y: Int = 12
comptime GEOM_IDX_HALF_Z: Int = 13
comptime GEOM_IDX_FRICTION: Int = 14
comptime GEOM_IDX_CONTYPE: Int = 15
comptime GEOM_IDX_CONAFFINITY: Int = 16


fn model_geom_offset[NBODY: Int, NJOINT: Int](geom_idx: Int) -> Int:
    """Offset to a specific unified geom in model buffer.

    Geoms are stored AFTER metadata+curriculum to avoid shifting metadata offsets.
    Layout: [bodies | joints | metadata | curriculum | geoms]
    """
    return NBODY * MODEL_BODY_SIZE + NJOINT * MODEL_JOINT_SIZE + MODEL_META_SIZE + MODEL_CURRICULUM_SIZE + geom_idx * MODEL_GEOM_SIZE


# =============================================================================
# Model Buffer Layout - Curriculum Parameters
# =============================================================================

# Fixed-size curriculum section (environments use what they need)
comptime MODEL_CURRICULUM_SIZE: Int = 8  # Up to 8 curriculum parameters

# Common curriculum parameter indices (environments can define their own)
comptime CURRICULUM_IDX_MIN_HEIGHT: Int = 0
comptime CURRICULUM_IDX_MAX_PITCH: Int = 1
comptime CURRICULUM_IDX_PARAM_2: Int = 2
comptime CURRICULUM_IDX_PARAM_3: Int = 3
comptime CURRICULUM_IDX_PARAM_4: Int = 4
comptime CURRICULUM_IDX_PARAM_5: Int = 5
comptime CURRICULUM_IDX_PARAM_6: Int = 6
comptime CURRICULUM_IDX_PARAM_7: Int = 7


fn model_curriculum_offset[NBODY: Int, NJOINT: Int]() -> Int:
    """Offset to curriculum parameters in model buffer."""
    return model_metadata_offset[NBODY, NJOINT]() + MODEL_META_SIZE


fn model_size[NBODY: Int, NJOINT: Int, NGEOM: Int = 0]() -> Int:
    """Total model buffer size.

    Layout: [bodies | joints | metadata | curriculum | geoms]
    """
    return NBODY * MODEL_BODY_SIZE + NJOINT * MODEL_JOINT_SIZE + MODEL_META_SIZE + MODEL_CURRICULUM_SIZE + NGEOM * MODEL_GEOM_SIZE


# =============================================================================
# Workspace Buffer Layout (per-environment scratch space for GPU kernels)
# =============================================================================
# Moves all integrator temporaries and solver arrays from InlineArrays
# (register pressure) to device memory.
#
# Layout per environment:
#   [integrator_temps | M_inv: NV*NV | solver workspace: SOLVER.solver_workspace_size()]
#
# Integrator temps section:
#   [cdof: NV*6 | crb: NBODY*10 | M: NV*NV | L: NV*NV | D: NV |
#    bias: NV | f_net: NV | qacc_ws: NV | qacc_constrained: NV]


fn integrator_workspace_size[NV: Int, NBODY: Int]() -> Int:
    """Total integrator temporaries size per environment."""
    return NV * 6 + NBODY * 10 + 2 * NV * NV + 5 * NV


fn ws_cdof_offset() -> Int:
    """Offset to cdof (NV*6) in workspace buffer."""
    return 0


fn ws_crb_offset[NV: Int]() -> Int:
    """Offset to crb (NBODY*10) in workspace buffer."""
    return NV * 6


fn ws_M_offset[NV: Int, NBODY: Int]() -> Int:
    """Offset to mass matrix M (NV*NV) in workspace buffer."""
    return NV * 6 + NBODY * 10


fn ws_L_offset[NV: Int, NBODY: Int]() -> Int:
    """Offset to LDL factor L (NV*NV) in workspace buffer."""
    return NV * 6 + NBODY * 10 + NV * NV


fn ws_D_offset[NV: Int, NBODY: Int]() -> Int:
    """Offset to LDL factor D (NV) in workspace buffer."""
    return NV * 6 + NBODY * 10 + 2 * NV * NV


fn ws_bias_offset[NV: Int, NBODY: Int]() -> Int:
    """Offset to bias forces (NV) in workspace buffer."""
    return NV * 6 + NBODY * 10 + 2 * NV * NV + NV


fn ws_fnet_offset[NV: Int, NBODY: Int]() -> Int:
    """Offset to f_net (NV) in workspace buffer."""
    return NV * 6 + NBODY * 10 + 2 * NV * NV + 2 * NV


fn ws_qacc_ws_offset[NV: Int, NBODY: Int]() -> Int:
    """Offset to qacc workspace (NV) in workspace buffer."""
    return NV * 6 + NBODY * 10 + 2 * NV * NV + 3 * NV


fn ws_qacc_constrained_offset[NV: Int, NBODY: Int]() -> Int:
    """Offset to qacc_constrained (NV) in workspace buffer.

    This slot stores the acceleration vector that the constraint solver
    modifies in-place (acceleration-level solving).
    """
    return NV * 6 + NBODY * 10 + 2 * NV * NV + 4 * NV


fn ws_m_inv_offset[NV: Int, NBODY: Int]() -> Int:
    """Offset to M_inv (NV*NV) in workspace buffer (after integrator temps)."""
    return integrator_workspace_size[NV, NBODY]()


fn ws_solver_offset[NV: Int, NBODY: Int]() -> Int:
    """Offset to solver workspace (after integrator temps + M_inv)."""
    return integrator_workspace_size[NV, NBODY]() + NV * NV
