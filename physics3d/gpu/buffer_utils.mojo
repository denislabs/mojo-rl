"""Physics3D GPU Buffer Utilities - GC state buffer management.

Provides functions to:
1. Create GC state and model buffers
2. Copy from CPU Model/Data to GPU buffers
3. Copy from GPU buffers back to CPU Data
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from .constants import (
    state_size,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    xpos_offset,
    xquat_offset,
    xvel_offset,
    xangvel_offset,
    contacts_offset,
    metadata_offset,
    CONTACT_SIZE,
    METADATA_SIZE,
    model_size,
    model_body_offset,
    model_joint_offset,
    model_metadata_offset,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_INV_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_INV_IXX,
    BODY_IDX_INV_IYY,
    BODY_IDX_INV_IZZ,
    BODY_IDX_POS_X,
    BODY_IDX_POS_Y,
    BODY_IDX_POS_Z,
    BODY_IDX_QUAT_X,
    BODY_IDX_QUAT_Y,
    BODY_IDX_QUAT_Z,
    BODY_IDX_QUAT_W,
    BODY_IDX_PARENT,
    BODY_IDX_IPOS_X,
    BODY_IDX_IPOS_Y,
    BODY_IDX_IPOS_Z,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_Y,
    BODY_IDX_IQUAT_Z,
    BODY_IDX_IQUAT_W,
    xipos_offset,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_POS_X,
    JOINT_IDX_POS_Y,
    JOINT_IDX_POS_Z,
    JOINT_IDX_AXIS_X,
    JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z,
    JOINT_IDX_TAU_LIMIT,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_DAMPING,
    JOINT_IDX_STIFFNESS,
    JOINT_IDX_SPRINGREF,
    JOINT_IDX_FRICTIONLOSS,
    JOINT_IDX_SOLREF_LIMIT_0,
    JOINT_IDX_SOLREF_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_0,
    JOINT_IDX_SOLIMP_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_2,
    JOINT_IDX_SOLIMP_LIMIT_3,
    JOINT_IDX_SOLIMP_LIMIT_4,
    JOINT_IDX_QPOS0,
    MODEL_META_IDX_NBODY,
    MODEL_META_IDX_NJOINT,
    MODEL_META_IDX_GRAVITY_X,
    MODEL_META_IDX_GRAVITY_Y,
    MODEL_META_IDX_GRAVITY_Z,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
    MODEL_META_IDX_SOLIMP_CONTACT_3,
    MODEL_META_IDX_SOLIMP_CONTACT_4,
    MODEL_META_IDX_SOLREF_LIMIT_0,
    MODEL_META_IDX_SOLREF_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_0,
    MODEL_META_IDX_SOLIMP_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_2,
    MODEL_META_IDX_SOLIMP_LIMIT_3,
    MODEL_META_IDX_SOLIMP_LIMIT_4,
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
    GEOM_IDX_FRICTION,
    GEOM_IDX_CONTYPE,
    GEOM_IDX_CONAFFINITY,
    GEOM_IDX_CONDIM,
    GEOM_IDX_FRICTION_SPIN,
    GEOM_IDX_FRICTION_ROLL,
    GEOM_IDX_RBOUND,
    GEOM_IDX_SOLREF_0,
    GEOM_IDX_SOLREF_1,
    GEOM_IDX_SOLIMP_0,
    GEOM_IDX_SOLIMP_1,
    GEOM_IDX_SOLIMP_2,
    GEOM_IDX_SOLIMP_3,
    GEOM_IDX_SOLIMP_4,
    GEOM_IDX_MARGIN,
    MODEL_META_IDX_IMPRATIO,
    MODEL_META_IDX_NEQUALITY,
    MODEL_META_IDX_NTENDON,
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_VISCOSITY,
    model_geom_offset,
    MODEL_EQ_SIZE,
    EQ_IDX_TYPE,
    EQ_IDX_BODY_A,
    EQ_IDX_BODY_B,
    EQ_IDX_ANCHOR_AX,
    EQ_IDX_ANCHOR_AY,
    EQ_IDX_ANCHOR_AZ,
    EQ_IDX_ANCHOR_BX,
    EQ_IDX_ANCHOR_BY,
    EQ_IDX_ANCHOR_BZ,
    EQ_IDX_RELPOSE_X,
    EQ_IDX_RELPOSE_Y,
    EQ_IDX_RELPOSE_Z,
    EQ_IDX_RELPOSE_W,
    EQ_IDX_SOLREF_0,
    EQ_IDX_SOLREF_1,
    EQ_IDX_SOLIMP_0,
    EQ_IDX_SOLIMP_1,
    EQ_IDX_SOLIMP_2,
    EQ_IDX_SOLIMP_3,
    EQ_IDX_SOLIMP_4,
    model_equality_offset,
    model_body_invweight0_offset,
    model_dof_invweight0_offset,
    MODEL_TENDON_SIZE,
    TENDON_IDX_NUM_JOINTS,
    TENDON_IDX_JOINT_0,
    TENDON_IDX_JOINT_1,
    TENDON_IDX_JOINT_2,
    TENDON_IDX_JOINT_3,
    TENDON_IDX_COEF_0,
    TENDON_IDX_COEF_1,
    TENDON_IDX_COEF_2,
    TENDON_IDX_COEF_3,
    TENDON_IDX_LENGTH_REF,
    TENDON_IDX_SOLREF_0,
    TENDON_IDX_SOLREF_1,
    TENDON_IDX_SOLIMP_0,
    TENDON_IDX_SOLIMP_1,
    TENDON_IDX_SOLIMP_2,
    TENDON_IDX_SOLIMP_3,
    TENDON_IDX_SOLIMP_4,
    model_tendon_offset,
)
from ..types import Model, Data, ConeType

# =============================================================================
# Host Buffer Creation
# =============================================================================


fn create_state_buffer[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    MAX_CONTACTS: Int,
    BATCH: Int,
](ctx: DeviceContext) raises -> HostBuffer[DTYPE]:
    """Allocate host buffer for GC state.

    Parameters:
        DTYPE: Data type (float32 or float64).
        NQ: Total qpos dimension.
        NV: Total qvel dimension.
        NBODY: Number of bodies.
        MAX_CONTACTS: Maximum contacts.
        BATCH: Number of environments.

    Args:
        ctx: Device context.

    Returns:
        Pointer to allocated buffer.
    """
    comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
    var total_size = STATE_SIZE * BATCH
    var buffer = ctx.enqueue_create_host_buffer[DTYPE](total_size)

    # Initialize to zero
    for i in range(total_size):
        buffer[i] = Scalar[DTYPE](0)

    return buffer


fn create_model_buffer[
    DTYPE: DType,
    NBODY: Int,
    NJOINT: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
](ctx: DeviceContext) raises -> HostBuffer[DTYPE]:
    """Allocate host buffer for GC model.

    The model buffer contains static configuration shared by all environments.

    Returns:
        Pointer to allocated buffer.
    """
    comptime MODEL_SIZE = model_size[NBODY, NJOINT, NGEOM, NEQUALITY]()
    var buffer = ctx.enqueue_create_host_buffer[DTYPE](MODEL_SIZE)

    # Initialize to zero
    for i in range(MODEL_SIZE):
        buffer[i] = Scalar[DTYPE](0)

    return buffer


# =============================================================================
# Copy Model to Buffer
# =============================================================================


fn copy_model_to_buffer[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
](
    model: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
    ],
    buffer: HostBuffer[DTYPE],
):
    """Copy Model data to a flat buffer for GPU.

    Args:
        model: Source model.
        buffer: Destination buffer (must be at least model_size bytes).
    """
    # Copy body data
    for body in range(NBODY):
        var offset = model_body_offset(body)
        buffer[offset + BODY_IDX_MASS] = model.body_mass[body]
        buffer[offset + BODY_IDX_INV_MASS] = model.body_inv_mass[body]
        buffer[offset + BODY_IDX_IXX] = model.body_inertia[body * 3 + 0]
        buffer[offset + BODY_IDX_IYY] = model.body_inertia[body * 3 + 1]
        buffer[offset + BODY_IDX_IZZ] = model.body_inertia[body * 3 + 2]
        buffer[offset + BODY_IDX_INV_IXX] = model.body_inv_inertia[body * 3 + 0]
        buffer[offset + BODY_IDX_INV_IYY] = model.body_inv_inertia[body * 3 + 1]
        buffer[offset + BODY_IDX_INV_IZZ] = model.body_inv_inertia[body * 3 + 2]
        buffer[offset + BODY_IDX_POS_X] = model.body_pos[body * 3 + 0]
        buffer[offset + BODY_IDX_POS_Y] = model.body_pos[body * 3 + 1]
        buffer[offset + BODY_IDX_POS_Z] = model.body_pos[body * 3 + 2]
        buffer[offset + BODY_IDX_QUAT_X] = model.body_quat[body * 4 + 0]
        buffer[offset + BODY_IDX_QUAT_Y] = model.body_quat[body * 4 + 1]
        buffer[offset + BODY_IDX_QUAT_Z] = model.body_quat[body * 4 + 2]
        buffer[offset + BODY_IDX_QUAT_W] = model.body_quat[body * 4 + 3]
        buffer[offset + BODY_IDX_PARENT] = Scalar[DTYPE](
            model.body_parent[body]
        )
        buffer[offset + BODY_IDX_IPOS_X] = model.body_ipos[body * 3 + 0]
        buffer[offset + BODY_IDX_IPOS_Y] = model.body_ipos[body * 3 + 1]
        buffer[offset + BODY_IDX_IPOS_Z] = model.body_ipos[body * 3 + 2]
        buffer[offset + BODY_IDX_IQUAT_X] = model.body_iquat[body * 4 + 0]
        buffer[offset + BODY_IDX_IQUAT_Y] = model.body_iquat[body * 4 + 1]
        buffer[offset + BODY_IDX_IQUAT_Z] = model.body_iquat[body * 4 + 2]
        buffer[offset + BODY_IDX_IQUAT_W] = model.body_iquat[body * 4 + 3]

    # Copy joint data
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var offset = model_joint_offset[NBODY](j)
        buffer[offset + JOINT_IDX_TYPE] = Scalar[DTYPE](joint.jnt_type)
        buffer[offset + JOINT_IDX_BODY_ID] = Scalar[DTYPE](joint.body_id)
        buffer[offset + JOINT_IDX_QPOS_ADR] = Scalar[DTYPE](joint.qpos_adr)
        buffer[offset + JOINT_IDX_DOF_ADR] = Scalar[DTYPE](joint.dof_adr)
        buffer[offset + JOINT_IDX_POS_X] = joint.pos_x
        buffer[offset + JOINT_IDX_POS_Y] = joint.pos_y
        buffer[offset + JOINT_IDX_POS_Z] = joint.pos_z
        buffer[offset + JOINT_IDX_AXIS_X] = joint.axis_x
        buffer[offset + JOINT_IDX_AXIS_Y] = joint.axis_y
        buffer[offset + JOINT_IDX_AXIS_Z] = joint.axis_z
        buffer[offset + JOINT_IDX_TAU_LIMIT] = joint.tau_limit
        buffer[offset + JOINT_IDX_RANGE_MIN] = joint.range_min
        buffer[offset + JOINT_IDX_RANGE_MAX] = joint.range_max
        buffer[offset + JOINT_IDX_ARMATURE] = joint.armature
        buffer[offset + JOINT_IDX_DAMPING] = joint.damping
        buffer[offset + JOINT_IDX_STIFFNESS] = joint.stiffness
        buffer[offset + JOINT_IDX_SPRINGREF] = joint.springref
        buffer[offset + JOINT_IDX_FRICTIONLOSS] = joint.frictionloss
        buffer[offset + JOINT_IDX_SOLREF_LIMIT_0] = model.joint_solref_limit[
            j * 2 + 0
        ]
        buffer[offset + JOINT_IDX_SOLREF_LIMIT_1] = model.joint_solref_limit[
            j * 2 + 1
        ]
        buffer[offset + JOINT_IDX_SOLIMP_LIMIT_0] = model.joint_solimp_limit[
            j * 5 + 0
        ]
        buffer[offset + JOINT_IDX_SOLIMP_LIMIT_1] = model.joint_solimp_limit[
            j * 5 + 1
        ]
        buffer[offset + JOINT_IDX_SOLIMP_LIMIT_2] = model.joint_solimp_limit[
            j * 5 + 2
        ]
        buffer[offset + JOINT_IDX_SOLIMP_LIMIT_3] = model.joint_solimp_limit[
            j * 5 + 3
        ]
        buffer[offset + JOINT_IDX_SOLIMP_LIMIT_4] = model.joint_solimp_limit[
            j * 5 + 4
        ]
        buffer[offset + JOINT_IDX_QPOS0] = model.qpos0[joint.qpos_adr]

    # Copy metadata
    var meta_offset = model_metadata_offset[NBODY, NJOINT]()
    buffer[meta_offset + MODEL_META_IDX_NBODY] = Scalar[DTYPE](NBODY)
    buffer[meta_offset + MODEL_META_IDX_NJOINT] = Scalar[DTYPE](
        model.num_joints
    )
    buffer[meta_offset + MODEL_META_IDX_GRAVITY_X] = model.gravity[0]
    buffer[meta_offset + MODEL_META_IDX_GRAVITY_Y] = model.gravity[1]
    buffer[meta_offset + MODEL_META_IDX_GRAVITY_Z] = model.gravity[2]
    buffer[meta_offset + MODEL_META_IDX_TIMESTEP] = model.timestep
    # Fluid dynamics options (density=0 / viscosity=0 disables fluid forces)
    buffer[meta_offset + MODEL_META_IDX_DENSITY] = model.opt_density
    buffer[meta_offset + MODEL_META_IDX_VISCOSITY] = model.opt_viscosity
    # solref/solimp contact
    buffer[
        meta_offset + MODEL_META_IDX_SOLREF_CONTACT_0
    ] = model.solref_contact[0]
    buffer[
        meta_offset + MODEL_META_IDX_SOLREF_CONTACT_1
    ] = model.solref_contact[1]
    buffer[
        meta_offset + MODEL_META_IDX_SOLIMP_CONTACT_0
    ] = model.solimp_contact[0]
    buffer[
        meta_offset + MODEL_META_IDX_SOLIMP_CONTACT_1
    ] = model.solimp_contact[1]
    buffer[
        meta_offset + MODEL_META_IDX_SOLIMP_CONTACT_2
    ] = model.solimp_contact[2]
    buffer[
        meta_offset + MODEL_META_IDX_SOLIMP_CONTACT_3
    ] = model.solimp_contact[3]
    buffer[
        meta_offset + MODEL_META_IDX_SOLIMP_CONTACT_4
    ] = model.solimp_contact[4]
    # solref/solimp limit
    buffer[meta_offset + MODEL_META_IDX_SOLREF_LIMIT_0] = model.solref_limit[0]
    buffer[meta_offset + MODEL_META_IDX_SOLREF_LIMIT_1] = model.solref_limit[1]
    buffer[meta_offset + MODEL_META_IDX_SOLIMP_LIMIT_0] = model.solimp_limit[0]
    buffer[meta_offset + MODEL_META_IDX_SOLIMP_LIMIT_1] = model.solimp_limit[1]
    buffer[meta_offset + MODEL_META_IDX_SOLIMP_LIMIT_2] = model.solimp_limit[2]
    buffer[meta_offset + MODEL_META_IDX_SOLIMP_LIMIT_3] = model.solimp_limit[3]
    buffer[meta_offset + MODEL_META_IDX_SOLIMP_LIMIT_4] = model.solimp_limit[4]
    # Friction cone model
    buffer[meta_offset + MODEL_META_IDX_IMPRATIO] = model.impratio
    # Equality constraints
    buffer[meta_offset + MODEL_META_IDX_NEQUALITY] = Scalar[DTYPE](
        model.num_equality
    )
    # Fixed tendons
    buffer[meta_offset + MODEL_META_IDX_NTENDON] = Scalar[DTYPE](
        model.num_tendons
    )


fn copy_invweight0_to_buffer[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
](
    model: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
    ],
    buffer: HostBuffer[DTYPE],
):
    """Copy body_invweight0 and dof_invweight0 from Model to GPU buffer.

    Must be called after copy_model_to_buffer. The buffer must be allocated
    with model_size_with_invweight (not model_size) to have enough space.

    Args:
        model: Source model (must have invweight0 computed).
        buffer: Destination buffer.
    """
    # Copy body_invweight0[NBODY*2]
    var bw_offset = model_body_invweight0_offset[
        NBODY, NJOINT, NGEOM, MAX_EQUALITY, MAX_TENDON
    ]()
    for i in range(NBODY * 2):
        buffer[bw_offset + i] = model.body_invweight0[i]

    # Copy dof_invweight0[NV]
    var dw_offset = model_dof_invweight0_offset[
        NBODY, NJOINT, NGEOM, MAX_EQUALITY, MAX_TENDON
    ]()
    for i in range(NV):
        buffer[dw_offset + i] = model.dof_invweight0[i]


fn copy_geoms_to_buffer[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
](
    model: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
    ],
    buffer: HostBuffer[DTYPE],
):
    """Copy unified geom data from Model to GPU buffer.

    Args:
        model: Source model with geom arrays.
        buffer: Destination buffer (must have room for NGEOM geoms).
    """
    for g in range(NGEOM):
        var offset = model_geom_offset[NBODY, NJOINT](g)
        buffer[offset + GEOM_IDX_TYPE] = Scalar[DTYPE](model.geom_type[g])
        buffer[offset + GEOM_IDX_BODY] = Scalar[DTYPE](model.geom_body[g])
        buffer[offset + GEOM_IDX_POS_X] = model.geom_pos[g * 3 + 0]
        buffer[offset + GEOM_IDX_POS_Y] = model.geom_pos[g * 3 + 1]
        buffer[offset + GEOM_IDX_POS_Z] = model.geom_pos[g * 3 + 2]
        buffer[offset + GEOM_IDX_QUAT_X] = model.geom_quat[g * 4 + 0]
        buffer[offset + GEOM_IDX_QUAT_Y] = model.geom_quat[g * 4 + 1]
        buffer[offset + GEOM_IDX_QUAT_Z] = model.geom_quat[g * 4 + 2]
        buffer[offset + GEOM_IDX_QUAT_W] = model.geom_quat[g * 4 + 3]
        buffer[offset + GEOM_IDX_RADIUS] = model.geom_radius[g]
        buffer[offset + GEOM_IDX_HALF_LENGTH] = model.geom_half_length[g]
        buffer[offset + GEOM_IDX_HALF_X] = model.geom_half_x[g]
        buffer[offset + GEOM_IDX_HALF_Y] = model.geom_half_y[g]
        buffer[offset + GEOM_IDX_HALF_Z] = model.geom_half_z[g]
        buffer[offset + GEOM_IDX_FRICTION] = model.geom_friction[g]
        buffer[offset + GEOM_IDX_CONTYPE] = Scalar[DTYPE](model.geom_contype[g])
        buffer[offset + GEOM_IDX_CONAFFINITY] = Scalar[DTYPE](
            model.geom_conaffinity[g]
        )
        buffer[offset + GEOM_IDX_CONDIM] = Scalar[DTYPE](model.geom_condim[g])
        buffer[offset + GEOM_IDX_FRICTION_SPIN] = model.geom_friction_spin[g]
        buffer[offset + GEOM_IDX_FRICTION_ROLL] = model.geom_friction_roll[g]
        buffer[offset + GEOM_IDX_RBOUND] = model.geom_rbound[g]
        buffer[offset + GEOM_IDX_SOLREF_0] = model.geom_solref[g * 2 + 0]
        buffer[offset + GEOM_IDX_SOLREF_1] = model.geom_solref[g * 2 + 1]
        buffer[offset + GEOM_IDX_SOLIMP_0] = model.geom_solimp[g * 5 + 0]
        buffer[offset + GEOM_IDX_SOLIMP_1] = model.geom_solimp[g * 5 + 1]
        buffer[offset + GEOM_IDX_SOLIMP_2] = model.geom_solimp[g * 5 + 2]
        buffer[offset + GEOM_IDX_SOLIMP_3] = model.geom_solimp[g * 5 + 3]
        buffer[offset + GEOM_IDX_SOLIMP_4] = model.geom_solimp[g * 5 + 4]
        buffer[offset + GEOM_IDX_MARGIN] = model.geom_margin[g]


fn copy_equality_to_buffer[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    MAX_EQUALITY: Int = 0,
](
    model: Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, MAX_EQUALITY
    ],
    buffer: HostBuffer[DTYPE],
):
    """Copy equality constraint data from Model to GPU buffer.

    Args:
        model: Source model with equality constraints.
        buffer: Destination buffer (must have room for equality data).
    """
    for e in range(model.num_equality):
        var eq = model.equality_constraints[e]
        var offset = model_equality_offset[NBODY, NJOINT, NGEOM](e)
        buffer[offset + EQ_IDX_TYPE] = Scalar[DTYPE](eq.eq_type)
        buffer[offset + EQ_IDX_BODY_A] = Scalar[DTYPE](eq.body_a)
        buffer[offset + EQ_IDX_BODY_B] = Scalar[DTYPE](eq.body_b)
        buffer[offset + EQ_IDX_ANCHOR_AX] = eq.anchor_a_x
        buffer[offset + EQ_IDX_ANCHOR_AY] = eq.anchor_a_y
        buffer[offset + EQ_IDX_ANCHOR_AZ] = eq.anchor_a_z
        buffer[offset + EQ_IDX_ANCHOR_BX] = eq.anchor_b_x
        buffer[offset + EQ_IDX_ANCHOR_BY] = eq.anchor_b_y
        buffer[offset + EQ_IDX_ANCHOR_BZ] = eq.anchor_b_z
        buffer[offset + EQ_IDX_RELPOSE_X] = eq.relpose_x
        buffer[offset + EQ_IDX_RELPOSE_Y] = eq.relpose_y
        buffer[offset + EQ_IDX_RELPOSE_Z] = eq.relpose_z
        buffer[offset + EQ_IDX_RELPOSE_W] = eq.relpose_w
        buffer[offset + EQ_IDX_SOLREF_0] = eq.solref_0
        buffer[offset + EQ_IDX_SOLREF_1] = eq.solref_1
        buffer[offset + EQ_IDX_SOLIMP_0] = eq.solimp_0
        buffer[offset + EQ_IDX_SOLIMP_1] = eq.solimp_1
        buffer[offset + EQ_IDX_SOLIMP_2] = eq.solimp_2
        buffer[offset + EQ_IDX_SOLIMP_3] = eq.solimp_3
        buffer[offset + EQ_IDX_SOLIMP_4] = eq.solimp_4


fn copy_tendons_to_buffer[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
](
    model: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
    ],
    buffer: HostBuffer[DTYPE],
):
    """Copy fixed tendon data from Model to GPU buffer.

    Args:
        model: Source model with tendons.
        buffer: Destination buffer (must have room for tendon data).
    """
    for t in range(model.num_tendons):
        var ten = model.tendons[t]
        var offset = model_tendon_offset[NBODY, NJOINT, NGEOM, MAX_EQUALITY](t)
        buffer[offset + TENDON_IDX_NUM_JOINTS] = Scalar[DTYPE](ten.num_joints)
        buffer[offset + TENDON_IDX_JOINT_0] = Scalar[DTYPE](ten.joint_idx_0)
        buffer[offset + TENDON_IDX_JOINT_1] = Scalar[DTYPE](ten.joint_idx_1)
        buffer[offset + TENDON_IDX_JOINT_2] = Scalar[DTYPE](ten.joint_idx_2)
        buffer[offset + TENDON_IDX_JOINT_3] = Scalar[DTYPE](ten.joint_idx_3)
        buffer[offset + TENDON_IDX_COEF_0] = ten.coef_0
        buffer[offset + TENDON_IDX_COEF_1] = ten.coef_1
        buffer[offset + TENDON_IDX_COEF_2] = ten.coef_2
        buffer[offset + TENDON_IDX_COEF_3] = ten.coef_3
        buffer[offset + TENDON_IDX_LENGTH_REF] = ten.length_ref
        buffer[offset + TENDON_IDX_SOLREF_0] = ten.solref_0
        buffer[offset + TENDON_IDX_SOLREF_1] = ten.solref_1
        buffer[offset + TENDON_IDX_SOLIMP_0] = ten.solimp_0
        buffer[offset + TENDON_IDX_SOLIMP_1] = ten.solimp_1
        buffer[offset + TENDON_IDX_SOLIMP_2] = ten.solimp_2
        buffer[offset + TENDON_IDX_SOLIMP_3] = ten.solimp_3
        buffer[offset + TENDON_IDX_SOLIMP_4] = ten.solimp_4


# =============================================================================
# Copy Data to/from Buffer
# =============================================================================


fn copy_data_to_buffer[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
](
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    buffer: HostBuffer[DTYPE],
    env_idx: Int,
):
    """Copy Data to a specific environment slot in state buffer.

    Args:
        data: Source data.
        buffer: Destination buffer.
        env_idx: Environment index in buffer.
    """
    comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
    var base = env_idx * STATE_SIZE

    # Copy qpos
    for i in range(NQ):
        buffer[base + qpos_offset[NQ, NV]() + i] = data.qpos[i]

    # Copy qvel
    for i in range(NV):
        buffer[base + qvel_offset[NQ, NV]() + i] = data.qvel[i]

    # Copy qacc
    for i in range(NV):
        buffer[base + qacc_offset[NQ, NV]() + i] = data.qacc[i]

    # Copy qfrc
    for i in range(NV):
        buffer[base + qfrc_offset[NQ, NV]() + i] = data.qfrc[i]

    # Copy xpos
    for i in range(NBODY * 3):
        buffer[base + xpos_offset[NQ, NV, NBODY]() + i] = data.xpos[i]

    # Copy xquat
    for i in range(NBODY * 4):
        buffer[base + xquat_offset[NQ, NV, NBODY]() + i] = data.xquat[i]

    # Copy xipos
    for i in range(NBODY * 3):
        buffer[base + xipos_offset[NQ, NV, NBODY]() + i] = data.xipos[i]

    # Copy xvel
    for i in range(NBODY * 3):
        buffer[base + xvel_offset[NQ, NV, NBODY]() + i] = data.xvel[i]

    # Copy xangvel
    for i in range(NBODY * 3):
        buffer[base + xangvel_offset[NQ, NV, NBODY]() + i] = data.xangvel[i]

    # Copy metadata
    var meta_offset = base + metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
    buffer[meta_offset] = Scalar[DTYPE](data.num_contacts)


fn copy_buffer_to_data[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
](
    buffer: UnsafePointer[Scalar[DTYPE]],
    mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    env_idx: Int,
):
    """Copy state buffer slot to Data.

    Args:
        buffer: Source buffer.
        data: Destination data.
        env_idx: Environment index in buffer.
    """
    comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
    var base = env_idx * STATE_SIZE

    # Copy qpos
    for i in range(NQ):
        data.qpos[i] = buffer[base + qpos_offset[NQ, NV]() + i]

    # Copy qvel
    for i in range(NV):
        data.qvel[i] = buffer[base + qvel_offset[NQ, NV]() + i]

    # Copy qacc
    for i in range(NV):
        data.qacc[i] = buffer[base + qacc_offset[NQ, NV]() + i]

    # Copy qfrc
    for i in range(NV):
        data.qfrc[i] = buffer[base + qfrc_offset[NQ, NV]() + i]

    # Copy xpos
    for i in range(NBODY * 3):
        data.xpos[i] = buffer[base + xpos_offset[NQ, NV, NBODY]() + i]

    # Copy xquat
    for i in range(NBODY * 4):
        data.xquat[i] = buffer[base + xquat_offset[NQ, NV, NBODY]() + i]

    # Copy xipos
    for i in range(NBODY * 3):
        data.xipos[i] = buffer[base + xipos_offset[NQ, NV, NBODY]() + i]

    # Copy xvel
    for i in range(NBODY * 3):
        data.xvel[i] = buffer[base + xvel_offset[NQ, NV, NBODY]() + i]

    # Copy xangvel
    for i in range(NBODY * 3):
        data.xangvel[i] = buffer[base + xangvel_offset[NQ, NV, NBODY]() + i]

    # Copy metadata
    var meta_offset = base + metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
    data.num_contacts = Int(buffer[meta_offset])
