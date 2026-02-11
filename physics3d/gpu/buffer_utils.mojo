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
    BODY_IDX_GEOM_TYPE,
    BODY_IDX_RADIUS,
    BODY_IDX_HALF_LENGTH,
    BODY_IDX_HALF_X,
    BODY_IDX_HALF_Y,
    BODY_IDX_HALF_Z,
    BODY_IDX_CONTYPE,
    BODY_IDX_CONAFFINITY,
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
    MODEL_META_IDX_NBODY,
    MODEL_META_IDX_NJOINT,
    MODEL_META_IDX_GRAVITY_X,
    MODEL_META_IDX_GRAVITY_Y,
    MODEL_META_IDX_GRAVITY_Z,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_GROUND_Z,
    MODEL_META_IDX_FRICTION,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
    MODEL_META_IDX_SOLREF_LIMIT_0,
    MODEL_META_IDX_SOLREF_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_0,
    MODEL_META_IDX_SOLIMP_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_2,
    MODEL_META_IDX_GROUND_CONTYPE,
    MODEL_META_IDX_GROUND_CONAFFINITY,
)
from ..types import Model, Data

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
](ctx: DeviceContext) raises -> HostBuffer[DTYPE]:
    """Allocate host buffer for GC model.

    The model buffer contains static configuration shared by all environments.

    Returns:
        Pointer to allocated buffer.
    """
    comptime MODEL_SIZE = model_size[NBODY, NJOINT]()
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
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
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
        buffer[offset + BODY_IDX_GEOM_TYPE] = Scalar[DTYPE](
            model.body_geom_type[body]
        )
        buffer[offset + BODY_IDX_RADIUS] = model.body_radius[body]
        buffer[offset + BODY_IDX_HALF_LENGTH] = model.body_half_length[body]
        buffer[offset + BODY_IDX_HALF_X] = model.body_half_x[body]
        buffer[offset + BODY_IDX_HALF_Y] = model.body_half_y[body]
        buffer[offset + BODY_IDX_HALF_Z] = model.body_half_z[body]
        buffer[offset + BODY_IDX_CONTYPE] = Scalar[DTYPE](
            model.body_contype[body]
        )
        buffer[offset + BODY_IDX_CONAFFINITY] = Scalar[DTYPE](
            model.body_conaffinity[body]
        )

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
    buffer[meta_offset + MODEL_META_IDX_GROUND_Z] = model.ground_z
    buffer[meta_offset + MODEL_META_IDX_FRICTION] = model.friction
    # solref/solimp contact
    buffer[meta_offset + MODEL_META_IDX_SOLREF_CONTACT_0] = model.solref_contact[0]
    buffer[meta_offset + MODEL_META_IDX_SOLREF_CONTACT_1] = model.solref_contact[1]
    buffer[meta_offset + MODEL_META_IDX_SOLIMP_CONTACT_0] = model.solimp_contact[0]
    buffer[meta_offset + MODEL_META_IDX_SOLIMP_CONTACT_1] = model.solimp_contact[1]
    buffer[meta_offset + MODEL_META_IDX_SOLIMP_CONTACT_2] = model.solimp_contact[2]
    # solref/solimp limit
    buffer[meta_offset + MODEL_META_IDX_SOLREF_LIMIT_0] = model.solref_limit[0]
    buffer[meta_offset + MODEL_META_IDX_SOLREF_LIMIT_1] = model.solref_limit[1]
    buffer[meta_offset + MODEL_META_IDX_SOLIMP_LIMIT_0] = model.solimp_limit[0]
    buffer[meta_offset + MODEL_META_IDX_SOLIMP_LIMIT_1] = model.solimp_limit[1]
    buffer[meta_offset + MODEL_META_IDX_SOLIMP_LIMIT_2] = model.solimp_limit[2]
    # Ground collision filtering
    buffer[meta_offset + MODEL_META_IDX_GROUND_CONTYPE] = Scalar[DTYPE](
        model.ground_contype
    )
    buffer[meta_offset + MODEL_META_IDX_GROUND_CONAFFINITY] = Scalar[DTYPE](
        model.ground_conaffinity
    )


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

    # Copy xvel
    for i in range(NBODY * 3):
        data.xvel[i] = buffer[base + xvel_offset[NQ, NV, NBODY]() + i]

    # Copy xangvel
    for i in range(NBODY * 3):
        data.xangvel[i] = buffer[base + xangvel_offset[NQ, NV, NBODY]() + i]

    # Copy metadata
    var meta_offset = base + metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
    data.num_contacts = Int(buffer[meta_offset])
