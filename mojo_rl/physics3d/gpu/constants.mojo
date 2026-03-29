"""Physics3D GPU constants - Flat buffer layout for GPU kernels.

Primary state is qpos/qvel (joint space). Body positions (xpos, xquat, xipos)
are computed via forward kinematics and stored for collision detection.

State buffer layout per environment:
  [qpos: NQ | qvel: NV | qacc: NV | qfrc: NV |
   xpos: NBODY*3 | xquat: NBODY*4 | xipos: NBODY*3 |
   xvel: NBODY*3 | xangvel: NBODY*3 |
   contacts: MAX_CONTACTS*CONTACT_SIZE | metadata: METADATA_SIZE |
   site_xpos: NSITE*3 |
   cfrc_ext: NBODY*6 | cvel: NBODY*6 | cinert: NBODY*10 | qfrc_actuator: NV]

Model buffer (static, same for all environments):
  Per body (MODEL_BODY_SIZE=25): [mass, inv_mass, inertia(3), inv_inertia(3),
    pos(3), quat(4), parent, ipos(3), iquat(4), rootid, weldid]
  Per joint (MODEL_JOINT_SIZE=26): [type, body_id, qpos_adr, dof_adr,
    pos(3), axis(3), tau_limit, range_min/max, armature, damping, stiffness, springref, frictionloss,
    solref_limit(2), solimp_limit(5), qpos0]
  Metadata (MODEL_META_SIZE=26): [NBODY, NJOINT, gravity(3), timestep, _reserved(2),
    solref_contact(2), solimp_contact(5), solref_limit(2), solimp_limit(5), impratio, nequality, ntendon]
  Curriculum (MODEL_CURRICULUM_SIZE=8): [up to 8 curriculum parameters]
  Per geom (MODEL_GEOM_SIZE=29): [type, body, pos(3), quat(4), radius, half_length,
    half_x/y/z, friction, contype, conaffinity, condim, friction_spin, friction_roll,
    rbound, solref(2), solimp(5), margin]
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


def qpos_offset[NQ: Int, NV: Int]() -> Int:
    """Offset to qpos array (always 0)."""
    return 0


def qvel_offset[NQ: Int, NV: Int]() -> Int:
    """Offset to qvel array."""
    return NQ


def qacc_offset[NQ: Int, NV: Int]() -> Int:
    """Offset to qacc array."""
    return NQ + NV


def qfrc_offset[NQ: Int, NV: Int]() -> Int:
    """Offset to qfrc array."""
    return NQ + 2 * NV


# =============================================================================
# State Buffer Layout - World Space (xpos, xquat, xvel, xangvel)
# =============================================================================


def xpos_offset[NQ: Int, NV: Int, NBODY: Int]() -> Int:
    """Offset to xpos array (body world positions)."""
    return NQ + 3 * NV


def xquat_offset[NQ: Int, NV: Int, NBODY: Int]() -> Int:
    """Offset to xquat array (body world orientations)."""
    return NQ + 3 * NV + NBODY * 3


def xipos_offset[NQ: Int, NV: Int, NBODY: Int]() -> Int:
    """Offset to xipos array (body CoM world positions)."""
    return NQ + 3 * NV + NBODY * 3 + NBODY * 4


def xvel_offset[NQ: Int, NV: Int, NBODY: Int]() -> Int:
    """Offset to xvel array (body world linear velocities)."""
    return NQ + 3 * NV + NBODY * 3 + NBODY * 4 + NBODY * 3


def xangvel_offset[NQ: Int, NV: Int, NBODY: Int]() -> Int:
    """Offset to xangvel array (body world angular velocities)."""
    return NQ + 3 * NV + NBODY * 3 + NBODY * 4 + NBODY * 3 + NBODY * 3


# =============================================================================
# State Buffer Layout - Contacts
# =============================================================================

# Contact layout (same as Cartesian engine: 12 floats per contact)
comptime CONTACT_SIZE: Int = 22

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
comptime CONTACT_IDX_FRICTION_SPIN: Int = 13
comptime CONTACT_IDX_FRICTION_ROLL: Int = 14
comptime CONTACT_IDX_CONDIM: Int = 15
comptime CONTACT_IDX_FORCE_TORSION: Int = 16
comptime CONTACT_IDX_FORCE_ROLL1: Int = 17
comptime CONTACT_IDX_FORCE_ROLL2: Int = 18
comptime CONTACT_IDX_FRAME_T1_X: Int = 19  # T1 hint for tangent frame (capsule axis)
comptime CONTACT_IDX_FRAME_T1_Y: Int = 20
comptime CONTACT_IDX_FRAME_T1_Z: Int = 21


def contacts_offset[NQ: Int, NV: Int, NBODY: Int]() -> Int:
    """Offset to contacts array."""
    return (
        NQ + 3 * NV + NBODY * 3 + NBODY * 4 + NBODY * 3 + NBODY * 3 + NBODY * 3
    )


def contact_offset[NQ: Int, NV: Int, NBODY: Int](contact_idx: Int) -> Int:
    """Offset to a specific contact."""
    return contacts_offset[NQ, NV, NBODY]() + contact_idx * CONTACT_SIZE


# =============================================================================
# State Buffer Layout - Metadata
# =============================================================================

comptime METADATA_SIZE: Int = 4

comptime META_IDX_NUM_CONTACTS: Int = 0
comptime META_IDX_STEP_COUNT: Int = 1  # Episode step counter for truncation
comptime META_IDX_PREV_X: Int = 2  # Previous x position for velocity computation
comptime META_IDX_PREV_COM_X: Int = 3  # Reserved for prev CoM x (unused with cvel approach)


def metadata_offset[NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int]() -> Int:
    """Offset to metadata."""
    return contacts_offset[NQ, NV, NBODY]() + MAX_CONTACTS * CONTACT_SIZE


# =============================================================================
# Total State Size Computation
# =============================================================================


def site_xpos_offset[NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int]() -> Int:
    """Offset to site_xpos array (site world positions).

    Placed after metadata at end of state buffer.
    """
    return metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]() + METADATA_SIZE


def cfrc_ext_offset[
    NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int = 0
]() -> Int:
    """Offset to cfrc_ext array (external contact forces per body).

    Layout: [torque_x, torque_y, torque_z, force_x, force_y, force_z] per body.
    Placed after site_xpos.
    """
    return site_xpos_offset[NQ, NV, NBODY, MAX_CONTACTS]() + NSITE * 3


def cvel_offset[
    NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int = 0
]() -> Int:
    """Offset to cvel array (body CoM spatial velocities).

    Layout: [omega_x, omega_y, omega_z, v_x, v_y, v_z] per body.
    Placed after cfrc_ext.
    """
    return cfrc_ext_offset[NQ, NV, NBODY, MAX_CONTACTS, NSITE]() + NBODY * 6


def cinert_offset[
    NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int = 0
]() -> Int:
    """Offset to cinert array (composite rigid body inertia).

    Layout: [m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz] per body.
    Placed after cvel.
    """
    return cvel_offset[NQ, NV, NBODY, MAX_CONTACTS, NSITE]() + NBODY * 6


def subtree_com_offset[
    NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int = 0
]() -> Int:
    """Offset to subtree_com array (subtree center of mass).

    Layout: [x, y, z] per body. Placed after cinert.
    """
    return cinert_offset[NQ, NV, NBODY, MAX_CONTACTS, NSITE]() + NBODY * 10


def qfrc_actuator_offset[
    NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int = 0
]() -> Int:
    """Offset to qfrc_actuator array (actuator force per DOF).

    Captures gear * clamped_force before constraint solving.
    Placed after subtree_com.
    """
    return subtree_com_offset[NQ, NV, NBODY, MAX_CONTACTS, NSITE]() + NBODY * 3


def state_size[
    NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int = 0
]() -> Int:
    """Compute total state buffer size per environment.

    Returns:
        Total size in number of scalars.
    """
    return (
        NQ  # qpos
        + 3 * NV  # qvel + qacc + qfrc
        + NBODY * 3  # xpos
        + NBODY * 4  # xquat
        + NBODY * 3  # xipos (CoM world positions)
        + NBODY * 3  # xvel
        + NBODY * 3  # xangvel
        + MAX_CONTACTS * CONTACT_SIZE
        + METADATA_SIZE
        + NSITE * 3  # site_xpos (site world positions)
        + NBODY * 6  # cfrc_ext
        + NBODY * 6  # cvel
        + NBODY * 10  # cinert
        + NBODY * 3  # subtree_com
        + NV  # qfrc_actuator
    )


# =============================================================================
# Model Buffer Layout - Per Body
# =============================================================================

comptime MODEL_BODY_SIZE: Int = 25

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
comptime BODY_IDX_IPOS_X: Int = 16  # CoM offset from body origin (body frame)
comptime BODY_IDX_IPOS_Y: Int = 17
comptime BODY_IDX_IPOS_Z: Int = 18
comptime BODY_IDX_IQUAT_X: Int = 19  # Inertia frame quaternion (body frame)
comptime BODY_IDX_IQUAT_Y: Int = 20
comptime BODY_IDX_IQUAT_Z: Int = 21
comptime BODY_IDX_IQUAT_W: Int = 22
comptime BODY_IDX_ROOTID: Int = 23  # Root body index (child of worldbody)
comptime BODY_IDX_WELDID: Int = 24  # Weld body index (MuJoCo body_weldid)


def model_body_offset(body_idx: Int) -> Int:
    """Offset to a specific body in model buffer."""
    return body_idx * MODEL_BODY_SIZE


# =============================================================================
# Model Buffer Layout - Per Joint
# =============================================================================

comptime MODEL_JOINT_SIZE: Int = 26  # +7 for per-joint solref/solimp limits (5 params) + qpos0

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
comptime JOINT_IDX_SOLREF_LIMIT_0: Int = 18  # Per-joint limit solref timeconst
comptime JOINT_IDX_SOLREF_LIMIT_1: Int = 19  # Per-joint limit solref dampratio
comptime JOINT_IDX_SOLIMP_LIMIT_0: Int = 20  # Per-joint limit solimp dmin
comptime JOINT_IDX_SOLIMP_LIMIT_1: Int = 21  # Per-joint limit solimp dmax
comptime JOINT_IDX_SOLIMP_LIMIT_2: Int = 22  # Per-joint limit solimp width
comptime JOINT_IDX_SOLIMP_LIMIT_3: Int = 23  # Per-joint limit solimp midpoint
comptime JOINT_IDX_SOLIMP_LIMIT_4: Int = 24  # Per-joint limit solimp power
comptime JOINT_IDX_QPOS0: Int = 25  # Joint reference position (MuJoCo qpos0 / ref)


def model_joint_offset[NBODY: Int](joint_idx: Int) -> Int:
    """Offset to a specific joint in model buffer."""
    return NBODY * MODEL_BODY_SIZE + joint_idx * MODEL_JOINT_SIZE


# =============================================================================
# Model Buffer Layout - Global Metadata
# =============================================================================

comptime MODEL_META_SIZE: Int = 26

comptime MODEL_META_IDX_NBODY: Int = 0
comptime MODEL_META_IDX_NJOINT: Int = 1
comptime MODEL_META_IDX_GRAVITY_X: Int = 2
comptime MODEL_META_IDX_GRAVITY_Y: Int = 3
comptime MODEL_META_IDX_GRAVITY_Z: Int = 4
comptime MODEL_META_IDX_TIMESTEP: Int = 5
# Fluid dynamics parameters (MuJoCo option.density / option.viscosity)
# These occupy the previously-reserved slots 6 and 7.
comptime MODEL_META_IDX_DENSITY: Int = 6  # Fluid mass density (kg/m³), 0 = disabled
comptime MODEL_META_IDX_VISCOSITY: Int = 7  # Fluid dynamic viscosity (Pa·s), 0 = disabled
# solref/solimp contact parameters (MuJoCo impedance model)
comptime MODEL_META_IDX_SOLREF_CONTACT_0: Int = 8  # timeconst
comptime MODEL_META_IDX_SOLREF_CONTACT_1: Int = 9  # dampratio
comptime MODEL_META_IDX_SOLIMP_CONTACT_0: Int = 10  # dmin
comptime MODEL_META_IDX_SOLIMP_CONTACT_1: Int = 11  # dmax
comptime MODEL_META_IDX_SOLIMP_CONTACT_2: Int = 12  # width
comptime MODEL_META_IDX_SOLIMP_CONTACT_3: Int = 13  # midpoint
comptime MODEL_META_IDX_SOLIMP_CONTACT_4: Int = 14  # power
# solref/solimp limit parameters (MuJoCo impedance model)
comptime MODEL_META_IDX_SOLREF_LIMIT_0: Int = 15  # timeconst
comptime MODEL_META_IDX_SOLREF_LIMIT_1: Int = 16  # dampratio
comptime MODEL_META_IDX_SOLIMP_LIMIT_0: Int = 17  # dmin
comptime MODEL_META_IDX_SOLIMP_LIMIT_1: Int = 18  # dmax
comptime MODEL_META_IDX_SOLIMP_LIMIT_2: Int = 19  # width
comptime MODEL_META_IDX_SOLIMP_LIMIT_3: Int = 20  # midpoint
comptime MODEL_META_IDX_SOLIMP_LIMIT_4: Int = 21  # power
# Friction cone model
comptime MODEL_META_IDX_IMPRATIO: Int = 22  # MuJoCo impratio
# Equality constraints
comptime MODEL_META_IDX_NEQUALITY: Int = 23  # Number of equality constraints
# Fixed tendons
comptime MODEL_META_IDX_NTENDON: Int = 24  # Number of fixed tendons
comptime MODEL_META_IDX_NEXCLUDE: Int = 25  # Number of contact exclude pairs


def model_metadata_offset[NBODY: Int, NJOINT: Int]() -> Int:
    """Offset to model metadata."""
    return NBODY * MODEL_BODY_SIZE + NJOINT * MODEL_JOINT_SIZE


# =============================================================================
# Model Buffer Layout - Unified Geoms (body-attached + static)
# =============================================================================

comptime MODEL_GEOM_SIZE: Int = 30  # Per unified geom (+7 for solref/solimp(5) +1 for margin +1 mesh_id)

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
comptime GEOM_IDX_CONDIM: Int = 17
comptime GEOM_IDX_FRICTION_SPIN: Int = 18
comptime GEOM_IDX_FRICTION_ROLL: Int = 19
comptime GEOM_IDX_RBOUND: Int = 20
comptime GEOM_IDX_SOLREF_0: Int = 21  # Per-geom solref timeconst
comptime GEOM_IDX_SOLREF_1: Int = 22  # Per-geom solref dampratio
comptime GEOM_IDX_SOLIMP_0: Int = 23  # Per-geom solimp dmin
comptime GEOM_IDX_SOLIMP_1: Int = 24  # Per-geom solimp dmax
comptime GEOM_IDX_SOLIMP_2: Int = 25  # Per-geom solimp width
comptime GEOM_IDX_SOLIMP_3: Int = 26  # Per-geom solimp midpoint
comptime GEOM_IDX_SOLIMP_4: Int = 27  # Per-geom solimp power
comptime GEOM_IDX_MARGIN: Int = 28  # Per-geom contact margin
comptime GEOM_IDX_MESH_ID: Int = 29  # Mesh hull index (-1 if not mesh)


def model_geom_offset[NBODY: Int, NJOINT: Int](geom_idx: Int) -> Int:
    """Offset to a specific unified geom in model buffer.

    Geoms are stored AFTER metadata+curriculum to avoid shifting metadata offsets.
    Layout: [bodies | joints | metadata | curriculum | geoms | equality]
    """
    return (
        NBODY * MODEL_BODY_SIZE
        + NJOINT * MODEL_JOINT_SIZE
        + MODEL_META_SIZE
        + MODEL_CURRICULUM_SIZE
        + geom_idx * MODEL_GEOM_SIZE
    )


# =============================================================================
# Model Buffer Layout - Equality Constraints
# =============================================================================

comptime MODEL_EQ_SIZE: Int = 20  # Per equality constraint

comptime EQ_IDX_TYPE: Int = 0  # EQ_CONNECT=0 or EQ_WELD=1
comptime EQ_IDX_BODY_A: Int = 1
comptime EQ_IDX_BODY_B: Int = 2  # -1 for world
comptime EQ_IDX_ANCHOR_AX: Int = 3
comptime EQ_IDX_ANCHOR_AY: Int = 4
comptime EQ_IDX_ANCHOR_AZ: Int = 5
comptime EQ_IDX_ANCHOR_BX: Int = 6
comptime EQ_IDX_ANCHOR_BY: Int = 7
comptime EQ_IDX_ANCHOR_BZ: Int = 8
comptime EQ_IDX_RELPOSE_X: Int = 9
comptime EQ_IDX_RELPOSE_Y: Int = 10
comptime EQ_IDX_RELPOSE_Z: Int = 11
comptime EQ_IDX_RELPOSE_W: Int = 12
comptime EQ_IDX_SOLREF_0: Int = 13
comptime EQ_IDX_SOLREF_1: Int = 14
comptime EQ_IDX_SOLIMP_0: Int = 15
comptime EQ_IDX_SOLIMP_1: Int = 16
comptime EQ_IDX_SOLIMP_2: Int = 17
comptime EQ_IDX_SOLIMP_3: Int = 18  # solimp midpoint
comptime EQ_IDX_SOLIMP_4: Int = 19  # solimp power


def model_equality_offset[
    NBODY: Int, NJOINT: Int, NGEOM: Int
](eq_idx: Int) -> Int:
    """Offset to a specific equality constraint in model buffer.

    Equality stored AFTER geoms.
    Layout: [bodies | joints | metadata | curriculum | geoms | equality]
    """
    return (
        NBODY * MODEL_BODY_SIZE
        + NJOINT * MODEL_JOINT_SIZE
        + MODEL_META_SIZE
        + MODEL_CURRICULUM_SIZE
        + NGEOM * MODEL_GEOM_SIZE
        + eq_idx * MODEL_EQ_SIZE
    )


# =============================================================================
# Model Buffer Layout - Fixed Tendons
# =============================================================================

comptime MODEL_TENDON_SIZE: Int = 17  # Per fixed tendon

comptime TENDON_IDX_NUM_JOINTS: Int = 0
comptime TENDON_IDX_JOINT_0: Int = 1
comptime TENDON_IDX_JOINT_1: Int = 2
comptime TENDON_IDX_JOINT_2: Int = 3
comptime TENDON_IDX_JOINT_3: Int = 4
comptime TENDON_IDX_COEF_0: Int = 5
comptime TENDON_IDX_COEF_1: Int = 6
comptime TENDON_IDX_COEF_2: Int = 7
comptime TENDON_IDX_COEF_3: Int = 8
comptime TENDON_IDX_LENGTH_REF: Int = 9
comptime TENDON_IDX_SOLREF_0: Int = 10
comptime TENDON_IDX_SOLREF_1: Int = 11
comptime TENDON_IDX_SOLIMP_0: Int = 12
comptime TENDON_IDX_SOLIMP_1: Int = 13
comptime TENDON_IDX_SOLIMP_2: Int = 14
comptime TENDON_IDX_SOLIMP_3: Int = 15  # solimp midpoint
comptime TENDON_IDX_SOLIMP_4: Int = 16  # solimp power


def model_tendon_offset[
    NBODY: Int, NJOINT: Int, NGEOM: Int, NEQUALITY: Int = 0
](tendon_idx: Int) -> Int:
    """Offset to a specific tendon in model buffer.

    Tendons are stored AFTER equality constraints.
    Layout: [bodies | joints | metadata | curriculum | geoms | equality | tendons]
    """
    return (
        NBODY * MODEL_BODY_SIZE
        + NJOINT * MODEL_JOINT_SIZE
        + MODEL_META_SIZE
        + MODEL_CURRICULUM_SIZE
        + NGEOM * MODEL_GEOM_SIZE
        + NEQUALITY * MODEL_EQ_SIZE
        + tendon_idx * MODEL_TENDON_SIZE
    )


# =============================================================================
# Model Buffer Layout - Curriculum Parameters
# =============================================================================

# Fixed-size curriculum section (environments use what they need)
comptime MODEL_CURRICULUM_SIZE: Int = 8  # Up to 8 curriculum parameters

# Generic curriculum parameter indices (environments define their own semantics)
comptime CURRICULUM_IDX_PARAM_0: Int = 0
comptime CURRICULUM_IDX_PARAM_1: Int = 1
comptime CURRICULUM_IDX_PARAM_2: Int = 2
comptime CURRICULUM_IDX_PARAM_3: Int = 3
comptime CURRICULUM_IDX_PARAM_4: Int = 4
comptime CURRICULUM_IDX_PARAM_5: Int = 5
comptime CURRICULUM_IDX_PARAM_6: Int = 6
comptime CURRICULUM_IDX_PARAM_7: Int = 7


def model_curriculum_offset[NBODY: Int, NJOINT: Int]() -> Int:
    """Offset to curriculum parameters in model buffer."""
    return model_metadata_offset[NBODY, NJOINT]() + MODEL_META_SIZE


# =============================================================================
# Model Buffer Layout - Sites
# =============================================================================

# Site layout: [body_idx, pos_x, pos_y, pos_z]
comptime MODEL_SITE_SIZE: Int = 4  # Per site: body + pos(3)

comptime SITE_IDX_BODY: Int = 0  # Body index the site is attached to
comptime SITE_IDX_POS_X: Int = 1  # Local position in body frame
comptime SITE_IDX_POS_Y: Int = 2
comptime SITE_IDX_POS_Z: Int = 3


def model_site_offset[
    NBODY: Int,
    NJOINT: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
    NTENDON: Int = 0,
](site_idx: Int) -> Int:
    """Offset to a specific site in model buffer.

    Sites are stored AFTER tendons.
    Layout: [bodies | joints | metadata | curriculum | geoms | equality | tendons | sites]
    """
    return (
        NBODY * MODEL_BODY_SIZE
        + NJOINT * MODEL_JOINT_SIZE
        + MODEL_META_SIZE
        + MODEL_CURRICULUM_SIZE
        + NGEOM * MODEL_GEOM_SIZE
        + NEQUALITY * MODEL_EQ_SIZE
        + NTENDON * MODEL_TENDON_SIZE
        + site_idx * MODEL_SITE_SIZE
    )


def model_size[
    NBODY: Int,
    NJOINT: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
    NTENDON: Int = 0,
    NSITE: Int = 0,
]() -> Int:
    """Total model buffer size (without invweight0 arrays).

    Layout: [bodies | joints | metadata | curriculum | geoms | equality | tendons | sites]
    """
    return (
        NBODY * MODEL_BODY_SIZE
        + NJOINT * MODEL_JOINT_SIZE
        + MODEL_META_SIZE
        + MODEL_CURRICULUM_SIZE
        + NGEOM * MODEL_GEOM_SIZE
        + NEQUALITY * MODEL_EQ_SIZE
        + NTENDON * MODEL_TENDON_SIZE
        + NSITE * MODEL_SITE_SIZE
    )


def model_body_invweight0_offset[
    NBODY: Int,
    NJOINT: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
    NTENDON: Int = 0,
    NSITE: Int = 0,
]() -> Int:
    """Offset to body_invweight0[NBODY*2] in model buffer.

    Appended after geoms/equality/tendons/sites section.
    """
    return model_size[NBODY, NJOINT, NGEOM, NEQUALITY, NTENDON, NSITE]()


def model_dof_invweight0_offset[
    NBODY: Int,
    NJOINT: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
    NTENDON: Int = 0,
    NSITE: Int = 0,
]() -> Int:
    """Offset to dof_invweight0[NV] in model buffer.

    Appended after body_invweight0[NBODY*2].
    """
    return (
        model_body_invweight0_offset[
            NBODY, NJOINT, NGEOM, NEQUALITY, NTENDON, NSITE
        ]()
        + NBODY * 2
    )


def model_size_with_invweight[
    NBODY: Int,
    NJOINT: Int,
    NV: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
    NTENDON: Int = 0,
    NSITE: Int = 0,
    NEXCLUDE: Int = 0,
    NMESH_VERTS: Int = 0,
]() -> Int:
    """Total model buffer size including invweight0, exclude pairs, and mesh hulls.

    Layout: [bodies | joints | metadata | curriculum | geoms | equality | tendons | sites |
             body_invweight0(NBODY*2) | dof_invweight0(NV) | excludes(NEXCLUDE*2) |
             mesh_meta(MAX_GPU_MESHES*2) | mesh_verts(NMESH_VERTS*3)]
    """
    return (
        model_dof_invweight0_offset[
            NBODY, NJOINT, NGEOM, NEQUALITY, NTENDON, NSITE
        ]()
        + NV
        + NEXCLUDE * 2
        + MAX_GPU_MESHES * MODEL_MESH_META_SIZE
        + NMESH_VERTS * 3
    )


# Exclude pair section: stored as [body1_0, body2_0, body1_1, body2_1, ...]
# after dof_invweight0
comptime MODEL_EXCLUDE_PAIR_SIZE: Int = 2  # body1, body2


def model_exclude_offset[
    NBODY: Int,
    NJOINT: Int,
    NV: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
    NTENDON: Int = 0,
    NSITE: Int = 0,
]() -> Int:
    """Offset to exclude pairs section in model buffer."""
    return (
        model_dof_invweight0_offset[
            NBODY, NJOINT, NGEOM, NEQUALITY, NTENDON, NSITE
        ]()
        + NV
    )


# =============================================================================
# Model Buffer Layout - Mesh Collision Hull Data
# =============================================================================

# Mesh hull vertices stored AFTER exclude pairs in the model buffer.
# Layout: [mesh_meta(NMESH*2)] [mesh_verts(total_verts*3)]
# mesh_meta: [vertadr, vertnum] per mesh
# mesh_verts: flattened [x0,y0,z0, x1,y1,z1, ...] in local frame
comptime MAX_HULL_VERTS_PER_MESH: Int = 256
comptime MAX_GPU_MESHES: Int = 16
comptime MODEL_MESH_META_SIZE: Int = 2  # vertadr, vertnum per mesh


def model_mesh_meta_offset[
    NBODY: Int,
    NJOINT: Int,
    NV: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
    NTENDON: Int = 0,
    NSITE: Int = 0,
    NEXCLUDE: Int = 0,
]() -> Int:
    """Offset to mesh metadata [vertadr, vertnum] * MAX_GPU_MESHES."""
    return (
        model_exclude_offset[
            NBODY, NJOINT, NV, NGEOM, NEQUALITY, NTENDON, NSITE
        ]()
        + NEXCLUDE * 2
    )


def model_mesh_vert_offset[
    NBODY: Int,
    NJOINT: Int,
    NV: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
    NTENDON: Int = 0,
    NSITE: Int = 0,
    NEXCLUDE: Int = 0,
]() -> Int:
    """Offset to mesh hull vertex data."""
    return (
        model_mesh_meta_offset[
            NBODY, NJOINT, NV, NGEOM, NEQUALITY, NTENDON, NSITE, NEXCLUDE
        ]()
        + MAX_GPU_MESHES * MODEL_MESH_META_SIZE
    )


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


def integrator_workspace_size[NV: Int, NBODY: Int]() -> Int:
    """Total integrator temporaries size per environment."""
    return NV * 6 + NBODY * 10 + 2 * NV * NV + 5 * NV


def ws_cdof_offset() -> Int:
    """Offset to cdof (NV*6) in workspace buffer."""
    return 0


def ws_crb_offset[NV: Int]() -> Int:
    """Offset to crb (NBODY*10) in workspace buffer."""
    return NV * 6


def ws_M_offset[NV: Int, NBODY: Int]() -> Int:
    """Offset to mass matrix M (NV*NV) in workspace buffer."""
    return NV * 6 + NBODY * 10


def ws_L_offset[NV: Int, NBODY: Int]() -> Int:
    """Offset to LDL factor L (NV*NV) in workspace buffer."""
    return NV * 6 + NBODY * 10 + NV * NV


def ws_D_offset[NV: Int, NBODY: Int]() -> Int:
    """Offset to LDL factor D (NV) in workspace buffer."""
    return NV * 6 + NBODY * 10 + 2 * NV * NV


def ws_bias_offset[NV: Int, NBODY: Int]() -> Int:
    """Offset to bias forces (NV) in workspace buffer."""
    return NV * 6 + NBODY * 10 + 2 * NV * NV + NV


def ws_fnet_offset[NV: Int, NBODY: Int]() -> Int:
    """Offset to f_net (NV) in workspace buffer."""
    return NV * 6 + NBODY * 10 + 2 * NV * NV + 2 * NV


def ws_qacc_ws_offset[NV: Int, NBODY: Int]() -> Int:
    """Offset to qacc workspace (NV) in workspace buffer."""
    return NV * 6 + NBODY * 10 + 2 * NV * NV + 3 * NV


def ws_qacc_constrained_offset[NV: Int, NBODY: Int]() -> Int:
    """Offset to qacc_constrained (NV) in workspace buffer.

    This slot stores the acceleration vector that the constraint solver
    modifies in-place (acceleration-level solving).
    """
    return NV * 6 + NBODY * 10 + 2 * NV * NV + 4 * NV


def ws_m_inv_offset[NV: Int, NBODY: Int]() -> Int:
    """Offset to M_inv (NV*NV) in workspace buffer (after integrator temps)."""
    return integrator_workspace_size[NV, NBODY]()


def ws_solver_offset[NV: Int, NBODY: Int]() -> Int:
    """Offset to solver workspace (after integrator temps + M_inv)."""
    return integrator_workspace_size[NV, NBODY]() + NV * NV


# =============================================================================
# Implicit Integrator Extra Workspace
# =============================================================================
# Additional workspace for the full implicit integrator's RNE velocity
# derivative computation. Placed AFTER solver workspace so existing offsets
# are unchanged.
#
# Layout within implicit extra section:
#   [qDeriv: NV*NV | cdof_origin: NV*6 | cvel_origin: NBODY*6 |
#    cinert: NBODY*10 | cdof_dot: NV*6 |
#    Dcvel: NBODY*6*NV | Dcdofdot: NV*6*NV |
#    Dcacc: NBODY*6*NV | Dcfrcbody: NBODY*6*NV]


def implicit_extra_workspace_size[NV: Int, NBODY: Int]() -> Int:
    """Total implicit-extra workspace size per environment."""
    return (
        NV * NV  # qDeriv
        + NV * 6  # cdof_origin
        + NBODY * 6  # cvel_origin
        + NBODY * 10  # cinert
        + NV * 6  # cdof_dot
        + NBODY * 6 * NV  # Dcvel
        + NV * 6 * NV  # Dcdofdot
        + NBODY * 6 * NV  # Dcacc
        + NBODY * 6 * NV  # Dcfrcbody
    )


def ws_implicit_qderiv_offset(base: Int) -> Int:
    """Offset to qDeriv (NV*NV) within implicit extra workspace."""
    return base


def ws_implicit_cdof_origin_offset[NV: Int](base: Int) -> Int:
    """Offset to cdof_sc (NV*6) within implicit extra workspace (subtree-COM).
    """
    return base + NV * NV


def ws_implicit_cvel_origin_offset[NV: Int](base: Int) -> Int:
    """Offset to cvel_sc (NBODY*6) within implicit extra workspace (subtree-COM).
    """
    return base + NV * NV + NV * 6


def ws_implicit_cinert_offset[NV: Int, NBODY: Int](base: Int) -> Int:
    """Offset to cinert (NBODY*10) within implicit extra workspace."""
    return base + NV * NV + NV * 6 + NBODY * 6


def ws_implicit_cdof_dot_offset[NV: Int, NBODY: Int](base: Int) -> Int:
    """Offset to cdof_dot (NV*6) within implicit extra workspace."""
    return base + NV * NV + NV * 6 + NBODY * 6 + NBODY * 10


def ws_implicit_dcvel_offset[NV: Int, NBODY: Int](base: Int) -> Int:
    """Offset to Dcvel (NBODY*6*NV) within implicit extra workspace."""
    return base + NV * NV + NV * 6 + NBODY * 6 + NBODY * 10 + NV * 6


def ws_implicit_dcdofdot_offset[NV: Int, NBODY: Int](base: Int) -> Int:
    """Offset to Dcdofdot (NV*6*NV) within implicit extra workspace."""
    return (
        base
        + NV * NV
        + NV * 6
        + NBODY * 6
        + NBODY * 10
        + NV * 6
        + NBODY * 6 * NV
    )


def ws_implicit_dcacc_offset[NV: Int, NBODY: Int](base: Int) -> Int:
    """Offset to Dcacc (NBODY*6*NV) within implicit extra workspace."""
    return (
        base
        + NV * NV
        + NV * 6
        + NBODY * 6
        + NBODY * 10
        + NV * 6
        + NBODY * 6 * NV
        + NV * 6 * NV
    )


def ws_implicit_dcfrcbody_offset[NV: Int, NBODY: Int](base: Int) -> Int:
    """Offset to Dcfrcbody (NBODY*6*NV) within implicit extra workspace."""
    return (
        base
        + NV * NV
        + NV * 6
        + NBODY * 6
        + NBODY * 10
        + NV * 6
        + NBODY * 6 * NV
        + NV * 6 * NV
        + NBODY * 6 * NV
    )


# =============================================================================
# RK4 Integrator Extra Workspace
# =============================================================================
# Additional workspace for the RK4 integrator's 4-stage pipeline.
# Placed AFTER solver workspace so existing offsets are unchanged.
#
# Layout within RK4 extra section:
#   [q0: NQ | v0: NV | A0: NV | A1: NV | A2: NV | A3: NV | C1: NV | C2: NV]
#
# Total: NQ + 7*NV


def rk4_extra_workspace_size[NQ: Int, NV: Int]() -> Int:
    """Total RK4-extra workspace size per environment."""
    return NQ + 7 * NV


def ws_rk4_q0_offset[NV: Int, NBODY: Int](solver_ws_size: Int) -> Int:
    """Offset to saved initial qpos (NQ) in RK4 workspace.

    Placed after integrator_temps + M_inv + solver_ws.
    """
    return ws_solver_offset[NV, NBODY]() + solver_ws_size


def ws_rk4_v0_offset[
    NV: Int, NBODY: Int, NQ: Int
](solver_ws_size: Int,) -> Int:
    """Offset to saved initial qvel (NV) in RK4 workspace."""
    return ws_rk4_q0_offset[NV, NBODY](solver_ws_size) + NQ


def ws_rk4_A_offset[
    NV: Int, NBODY: Int, NQ: Int
](solver_ws_size: Int, stage: Int) -> Int:
    """Offset to A[stage] (NV) in RK4 workspace. stage in [0,3]."""
    return ws_rk4_v0_offset[NV, NBODY, NQ](solver_ws_size) + NV + stage * NV


def ws_rk4_C1_offset[
    NV: Int, NBODY: Int, NQ: Int
](solver_ws_size: Int,) -> Int:
    """Offset to C1 velocity intermediate (NV) in RK4 workspace."""
    return ws_rk4_v0_offset[NV, NBODY, NQ](solver_ws_size) + NV + 4 * NV


def ws_rk4_C2_offset[
    NV: Int, NBODY: Int, NQ: Int
](solver_ws_size: Int,) -> Int:
    """Offset to C2 velocity intermediate (NV) in RK4 workspace."""
    return ws_rk4_C1_offset[NV, NBODY, NQ](solver_ws_size) + NV
