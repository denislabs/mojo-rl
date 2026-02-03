"""GPU buffer utilities for Generalized Coordinates engine.

Provides functions to create and manage GPU buffers for the GC engine.
Supports both host (CPU) and device (GPU) buffers.

Buffer layout per environment:
  [qpos: NQ | qvel: NV | qacc: NV | qfrc: NV |
   xpos: NBODY*3 | xquat: NBODY*4 | xvel: NBODY*3 | xangvel: NBODY*3 |
   contacts: MAX_CONTACTS*12 | metadata: 4]
"""

from memory import UnsafePointer
from .constants import (
    gc_state_size,
    gc_qpos_offset,
    gc_qvel_offset,
    gc_qacc_offset,
    gc_qfrc_offset,
    gc_xpos_offset,
    gc_xquat_offset,
    gc_xvel_offset,
    gc_xangvel_offset,
    gc_contacts_offset,
    gc_metadata_offset,
    GC_CONTACT_SIZE,
    GC_METADATA_SIZE,
    gc_model_size,
    gc_model_body_offset,
    gc_model_joint_offset,
    gc_model_metadata_offset,
    GC_MODEL_BODY_SIZE,
    GC_MODEL_JOINT_SIZE,
    GC_MODEL_META_SIZE,
    GC_BODY_IDX_MASS,
    GC_BODY_IDX_INV_MASS,
    GC_BODY_IDX_IXX,
    GC_BODY_IDX_IYY,
    GC_BODY_IDX_IZZ,
    GC_BODY_IDX_INV_IXX,
    GC_BODY_IDX_INV_IYY,
    GC_BODY_IDX_INV_IZZ,
    GC_BODY_IDX_POS_X,
    GC_BODY_IDX_POS_Y,
    GC_BODY_IDX_POS_Z,
    GC_BODY_IDX_QUAT_X,
    GC_BODY_IDX_QUAT_Y,
    GC_BODY_IDX_QUAT_Z,
    GC_BODY_IDX_QUAT_W,
    GC_BODY_IDX_PARENT,
    GC_BODY_IDX_GEOM_TYPE,
    GC_BODY_IDX_RADIUS,
    GC_BODY_IDX_HALF_LENGTH,
    GC_BODY_IDX_HALF_X,
    GC_BODY_IDX_HALF_Y,
    GC_BODY_IDX_HALF_Z,
    GC_JOINT_IDX_TYPE,
    GC_JOINT_IDX_BODY_ID,
    GC_JOINT_IDX_QPOS_ADR,
    GC_JOINT_IDX_DOF_ADR,
    GC_JOINT_IDX_POS_X,
    GC_JOINT_IDX_POS_Y,
    GC_JOINT_IDX_POS_Z,
    GC_JOINT_IDX_AXIS_X,
    GC_JOINT_IDX_AXIS_Y,
    GC_JOINT_IDX_AXIS_Z,
    GC_JOINT_IDX_TAU_LIMIT,
    GC_MODEL_META_IDX_NBODY,
    GC_MODEL_META_IDX_NJOINT,
    GC_MODEL_META_IDX_GRAVITY_X,
    GC_MODEL_META_IDX_GRAVITY_Y,
    GC_MODEL_META_IDX_GRAVITY_Z,
    GC_MODEL_META_IDX_TIMESTEP,
    GC_MODEL_META_IDX_GROUND_Z,
    GC_MODEL_META_IDX_FRICTION,
)
from ..types import ModelGC, DataGC


# =============================================================================
# Host Buffer Creation
# =============================================================================


fn create_gc_state_buffer[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    MAX_CONTACTS: Int,
    BATCH: Int,
]() -> UnsafePointer[Scalar[DTYPE]]:
    """Allocate host buffer for GC state.

    Args:
        DTYPE: Data type (float32 or float64).
        NQ: Total qpos dimension.
        NV: Total qvel dimension.
        NBODY: Number of bodies.
        MAX_CONTACTS: Maximum contacts.
        BATCH: Number of environments.

    Returns:
        Pointer to allocated buffer.
    """
    comptime state_size = gc_state_size[NQ, NV, NBODY, MAX_CONTACTS]()
    var total_size = state_size * BATCH
    var buffer = UnsafePointer[Scalar[DTYPE]].alloc(total_size)

    # Initialize to zero
    for i in range(total_size):
        buffer[i] = Scalar[DTYPE](0)

    return buffer


fn create_gc_model_buffer[
    DTYPE: DType,
    NBODY: Int,
    NJOINT: Int,
]() -> UnsafePointer[Scalar[DTYPE]]:
    """Allocate host buffer for GC model.

    The model buffer contains static configuration shared by all environments.

    Returns:
        Pointer to allocated buffer.
    """
    comptime model_size = gc_model_size[NBODY, NJOINT]()
    var buffer = UnsafePointer[Scalar[DTYPE]].alloc(model_size)

    # Initialize to zero
    for i in range(model_size):
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
    model: ModelGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    buffer: UnsafePointer[Scalar[DTYPE]],
):
    """Copy ModelGC data to a flat buffer for GPU.

    Args:
        model: Source model.
        buffer: Destination buffer (must be at least gc_model_size bytes).
    """
    # Copy body data
    for body in range(NBODY):
        var offset = gc_model_body_offset(body)
        buffer[offset + GC_BODY_IDX_MASS] = model.body_mass[body]
        buffer[offset + GC_BODY_IDX_INV_MASS] = model.body_inv_mass[body]
        buffer[offset + GC_BODY_IDX_IXX] = model.body_inertia[body * 3 + 0]
        buffer[offset + GC_BODY_IDX_IYY] = model.body_inertia[body * 3 + 1]
        buffer[offset + GC_BODY_IDX_IZZ] = model.body_inertia[body * 3 + 2]
        buffer[offset + GC_BODY_IDX_INV_IXX] = model.body_inv_inertia[body * 3 + 0]
        buffer[offset + GC_BODY_IDX_INV_IYY] = model.body_inv_inertia[body * 3 + 1]
        buffer[offset + GC_BODY_IDX_INV_IZZ] = model.body_inv_inertia[body * 3 + 2]
        buffer[offset + GC_BODY_IDX_POS_X] = model.body_pos[body * 3 + 0]
        buffer[offset + GC_BODY_IDX_POS_Y] = model.body_pos[body * 3 + 1]
        buffer[offset + GC_BODY_IDX_POS_Z] = model.body_pos[body * 3 + 2]
        buffer[offset + GC_BODY_IDX_QUAT_X] = model.body_quat[body * 4 + 0]
        buffer[offset + GC_BODY_IDX_QUAT_Y] = model.body_quat[body * 4 + 1]
        buffer[offset + GC_BODY_IDX_QUAT_Z] = model.body_quat[body * 4 + 2]
        buffer[offset + GC_BODY_IDX_QUAT_W] = model.body_quat[body * 4 + 3]
        buffer[offset + GC_BODY_IDX_PARENT] = Scalar[DTYPE](model.body_parent[body])
        buffer[offset + GC_BODY_IDX_GEOM_TYPE] = Scalar[DTYPE](model.body_geom_type[body])
        buffer[offset + GC_BODY_IDX_RADIUS] = model.body_radius[body]
        buffer[offset + GC_BODY_IDX_HALF_LENGTH] = model.body_half_length[body]
        buffer[offset + GC_BODY_IDX_HALF_X] = model.body_half_x[body]
        buffer[offset + GC_BODY_IDX_HALF_Y] = model.body_half_y[body]
        buffer[offset + GC_BODY_IDX_HALF_Z] = model.body_half_z[body]

    # Copy joint data
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var offset = gc_model_joint_offset[NBODY](j)
        buffer[offset + GC_JOINT_IDX_TYPE] = Scalar[DTYPE](joint.jnt_type)
        buffer[offset + GC_JOINT_IDX_BODY_ID] = Scalar[DTYPE](joint.body_id)
        buffer[offset + GC_JOINT_IDX_QPOS_ADR] = Scalar[DTYPE](joint.qpos_adr)
        buffer[offset + GC_JOINT_IDX_DOF_ADR] = Scalar[DTYPE](joint.dof_adr)
        buffer[offset + GC_JOINT_IDX_POS_X] = joint.pos_x
        buffer[offset + GC_JOINT_IDX_POS_Y] = joint.pos_y
        buffer[offset + GC_JOINT_IDX_POS_Z] = joint.pos_z
        buffer[offset + GC_JOINT_IDX_AXIS_X] = joint.axis_x
        buffer[offset + GC_JOINT_IDX_AXIS_Y] = joint.axis_y
        buffer[offset + GC_JOINT_IDX_AXIS_Z] = joint.axis_z
        buffer[offset + GC_JOINT_IDX_TAU_LIMIT] = joint.tau_limit

    # Copy metadata
    var meta_offset = gc_model_metadata_offset[NBODY, NJOINT]()
    buffer[meta_offset + GC_MODEL_META_IDX_NBODY] = Scalar[DTYPE](NBODY)
    buffer[meta_offset + GC_MODEL_META_IDX_NJOINT] = Scalar[DTYPE](model.num_joints)
    buffer[meta_offset + GC_MODEL_META_IDX_GRAVITY_X] = model.gravity[0]
    buffer[meta_offset + GC_MODEL_META_IDX_GRAVITY_Y] = model.gravity[1]
    buffer[meta_offset + GC_MODEL_META_IDX_GRAVITY_Z] = model.gravity[2]
    buffer[meta_offset + GC_MODEL_META_IDX_TIMESTEP] = model.timestep
    buffer[meta_offset + GC_MODEL_META_IDX_GROUND_Z] = model.ground_z
    buffer[meta_offset + GC_MODEL_META_IDX_FRICTION] = model.friction


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
    data: DataGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    buffer: UnsafePointer[Scalar[DTYPE]],
    env_idx: Int,
):
    """Copy DataGC to a specific environment slot in state buffer.

    Args:
        data: Source data.
        buffer: Destination buffer.
        env_idx: Environment index in buffer.
    """
    comptime state_size = gc_state_size[NQ, NV, NBODY, MAX_CONTACTS]()
    var base = env_idx * state_size

    # Copy qpos
    for i in range(NQ):
        buffer[base + gc_qpos_offset[NQ, NV]() + i] = data.qpos[i]

    # Copy qvel
    for i in range(NV):
        buffer[base + gc_qvel_offset[NQ, NV]() + i] = data.qvel[i]

    # Copy qacc
    for i in range(NV):
        buffer[base + gc_qacc_offset[NQ, NV]() + i] = data.qacc[i]

    # Copy qfrc
    for i in range(NV):
        buffer[base + gc_qfrc_offset[NQ, NV]() + i] = data.qfrc[i]

    # Copy xpos
    for i in range(NBODY * 3):
        buffer[base + gc_xpos_offset[NQ, NV, NBODY]() + i] = data.xpos[i]

    # Copy xquat
    for i in range(NBODY * 4):
        buffer[base + gc_xquat_offset[NQ, NV, NBODY]() + i] = data.xquat[i]

    # Copy xvel
    for i in range(NBODY * 3):
        buffer[base + gc_xvel_offset[NQ, NV, NBODY]() + i] = data.xvel[i]

    # Copy xangvel
    for i in range(NBODY * 3):
        buffer[base + gc_xangvel_offset[NQ, NV, NBODY]() + i] = data.xangvel[i]

    # Copy metadata
    var meta_offset = base + gc_metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
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
    mut data: DataGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    env_idx: Int,
):
    """Copy state buffer slot to DataGC.

    Args:
        buffer: Source buffer.
        data: Destination data.
        env_idx: Environment index in buffer.
    """
    comptime state_size = gc_state_size[NQ, NV, NBODY, MAX_CONTACTS]()
    var base = env_idx * state_size

    # Copy qpos
    for i in range(NQ):
        data.qpos[i] = buffer[base + gc_qpos_offset[NQ, NV]() + i]

    # Copy qvel
    for i in range(NV):
        data.qvel[i] = buffer[base + gc_qvel_offset[NQ, NV]() + i]

    # Copy qacc
    for i in range(NV):
        data.qacc[i] = buffer[base + gc_qacc_offset[NQ, NV]() + i]

    # Copy qfrc
    for i in range(NV):
        data.qfrc[i] = buffer[base + gc_qfrc_offset[NQ, NV]() + i]

    # Copy xpos
    for i in range(NBODY * 3):
        data.xpos[i] = buffer[base + gc_xpos_offset[NQ, NV, NBODY]() + i]

    # Copy xquat
    for i in range(NBODY * 4):
        data.xquat[i] = buffer[base + gc_xquat_offset[NQ, NV, NBODY]() + i]

    # Copy xvel
    for i in range(NBODY * 3):
        data.xvel[i] = buffer[base + gc_xvel_offset[NQ, NV, NBODY]() + i]

    # Copy xangvel
    for i in range(NBODY * 3):
        data.xangvel[i] = buffer[base + gc_xangvel_offset[NQ, NV, NBODY]() + i]

    # Copy metadata
    var meta_offset = base + gc_metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
    data.num_contacts = Int(buffer[meta_offset])


# =============================================================================
# Free Buffers
# =============================================================================


fn free_gc_buffer[DTYPE: DType](buffer: UnsafePointer[Scalar[DTYPE]]):
    """Free a GC buffer."""
    buffer.free()
