"""Physics3D v2 GPU Buffer Utilities - Multi-body state buffer management.

Provides functions to:
1. Initialize state buffers with default values
2. Copy from CPU Model/Data to GPU buffers
3. Copy from GPU buffers back to CPU Data
4. Helper functions for accessing state fields


"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from ..types import Model, Data
from .constants import (
    BODY_STATE_SIZE,
    CONTACT_STATE_SIZE,
    METADATA_SIZE,
    MODEL_BODY_SIZE,
    MODEL_IDX_MASS,
    MODEL_IDX_INV_MASS,
    MODEL_IDX_RADIUS,
    MODEL_IDX_IXX,
    MODEL_IDX_IYY,
    MODEL_IDX_IZZ,
    MODEL_IDX_INV_IXX,
    MODEL_IDX_INV_IYY,
    MODEL_IDX_INV_IZZ,
    MODEL_IDX_GEOM_TYPE,
    MODEL_IDX_HALF_LENGTH,
    MODEL_IDX_HALF_X,
    MODEL_IDX_HALF_Y,
    MODEL_IDX_HALF_Z,
    GEOM_SPHERE,
    BODY_IDX_PX,
    BODY_IDX_PY,
    BODY_IDX_PZ,
    BODY_IDX_QX,
    BODY_IDX_QY,
    BODY_IDX_QZ,
    BODY_IDX_QW,
    BODY_IDX_VX,
    BODY_IDX_VY,
    BODY_IDX_VZ,
    BODY_IDX_WX,
    BODY_IDX_WY,
    BODY_IDX_WZ,
    BODY_IDX_AX,
    BODY_IDX_AY,
    BODY_IDX_AZ,
    BODY_IDX_ALPHA_X,
    BODY_IDX_ALPHA_Y,
    BODY_IDX_ALPHA_Z,
    BODY_IDX_FX,
    BODY_IDX_FY,
    BODY_IDX_FZ,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_DIST,
    CONTACT_IDX_IMPULSE_N,
    CONTACT_IDX_IMPULSE_T1,
    CONTACT_IDX_IMPULSE_T2,
    META_IDX_NUM_CONTACTS,
    compute_state_size,
    body_offset,
    contact_offset,
    metadata_offset,
)

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
    GC_JOINT_IDX_RANGE_MIN,
    GC_JOINT_IDX_RANGE_MAX,
    GC_JOINT_IDX_ARMATURE,
    GC_JOINT_IDX_DAMPING,
    GC_JOINT_IDX_STIFFNESS,
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
# State Buffer Initialization
# =============================================================================


fn init_state_host_buffer[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, BATCH: Int
](ctx: DeviceContext) raises -> HostBuffer[DTYPE]:
    """Create and initialize a host buffer with default state values.

    Each environment gets:
    - All bodies at origin with identity quaternion
    - Zero velocities
    - Zero accelerations
    - Zero forces
    - No active contacts

    Returns:
        HostBuffer of size [BATCH * STATE_SIZE].
    """
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var host_buf = ctx.enqueue_create_host_buffer[DTYPE](BATCH * STATE_SIZE)

    # Initialize each environment
    for env in range(BATCH):
        var env_base = env * STATE_SIZE

        # Initialize each body
        for body in range(NUM_BODIES):
            var body_base = env_base + body_offset[NUM_BODIES, MAX_CONTACTS](
                body
            )

            # Position at origin
            host_buf[body_base + BODY_IDX_PX] = Scalar[DTYPE](0)
            host_buf[body_base + BODY_IDX_PY] = Scalar[DTYPE](0)
            host_buf[body_base + BODY_IDX_PZ] = Scalar[DTYPE](0)

            # Identity quaternion [0, 0, 0, 1]
            host_buf[body_base + BODY_IDX_QX] = Scalar[DTYPE](0)
            host_buf[body_base + BODY_IDX_QY] = Scalar[DTYPE](0)
            host_buf[body_base + BODY_IDX_QZ] = Scalar[DTYPE](0)
            host_buf[body_base + BODY_IDX_QW] = Scalar[DTYPE](1)

            # Zero velocities
            host_buf[body_base + BODY_IDX_VX] = Scalar[DTYPE](0)
            host_buf[body_base + BODY_IDX_VY] = Scalar[DTYPE](0)
            host_buf[body_base + BODY_IDX_VZ] = Scalar[DTYPE](0)
            host_buf[body_base + BODY_IDX_WX] = Scalar[DTYPE](0)
            host_buf[body_base + BODY_IDX_WY] = Scalar[DTYPE](0)
            host_buf[body_base + BODY_IDX_WZ] = Scalar[DTYPE](0)

            # Zero accelerations
            host_buf[body_base + BODY_IDX_AX] = Scalar[DTYPE](0)
            host_buf[body_base + BODY_IDX_AY] = Scalar[DTYPE](0)
            host_buf[body_base + BODY_IDX_AZ] = Scalar[DTYPE](0)
            host_buf[body_base + BODY_IDX_ALPHA_X] = Scalar[DTYPE](0)
            host_buf[body_base + BODY_IDX_ALPHA_Y] = Scalar[DTYPE](0)
            host_buf[body_base + BODY_IDX_ALPHA_Z] = Scalar[DTYPE](0)

            # Zero forces
            host_buf[body_base + BODY_IDX_FX] = Scalar[DTYPE](0)
            host_buf[body_base + BODY_IDX_FY] = Scalar[DTYPE](0)
            host_buf[body_base + BODY_IDX_FZ] = Scalar[DTYPE](0)

        # Initialize contacts as inactive
        for c in range(MAX_CONTACTS):
            var c_base = env_base + contact_offset[NUM_BODIES, MAX_CONTACTS](c)
            host_buf[c_base + CONTACT_IDX_BODY_A] = Scalar[DTYPE](-1)
            host_buf[c_base + CONTACT_IDX_BODY_B] = Scalar[DTYPE](-1)
            host_buf[c_base + CONTACT_IDX_POS_X] = Scalar[DTYPE](0)
            host_buf[c_base + CONTACT_IDX_POS_Y] = Scalar[DTYPE](0)
            host_buf[c_base + CONTACT_IDX_POS_Z] = Scalar[DTYPE](0)
            host_buf[c_base + CONTACT_IDX_NX] = Scalar[DTYPE](0)
            host_buf[c_base + CONTACT_IDX_NY] = Scalar[DTYPE](0)
            host_buf[c_base + CONTACT_IDX_NZ] = Scalar[DTYPE](1)
            host_buf[c_base + CONTACT_IDX_DIST] = Scalar[DTYPE](0)
            host_buf[c_base + CONTACT_IDX_IMPULSE_N] = Scalar[DTYPE](0)
            host_buf[c_base + CONTACT_IDX_IMPULSE_T1] = Scalar[DTYPE](0)
            host_buf[c_base + CONTACT_IDX_IMPULSE_T2] = Scalar[DTYPE](0)

        # Initialize metadata
        var meta_base = env_base + metadata_offset[NUM_BODIES, MAX_CONTACTS]()
        host_buf[meta_base + META_IDX_NUM_CONTACTS] = Scalar[DTYPE](0)

    return host_buf^


# =============================================================================
# Model Buffer Creation
# =============================================================================


fn create_model_host_buffer[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](
    ctx: DeviceContext, model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS]
) raises -> HostBuffer[DTYPE]:
    """Create a host buffer with model data.

    Layout per body (14 floats):
    [mass, inv_mass, radius, ixx, iyy, izz, inv_ixx, inv_iyy, inv_izz,
     geom_type, half_length, half_x, half_y, half_z]

    Returns:
        HostBuffer of size [NUM_BODIES * MODEL_BODY_SIZE].
    """
    var host_buf = ctx.enqueue_create_host_buffer[DTYPE](
        NUM_BODIES * MODEL_BODY_SIZE
    )

    for i in range(NUM_BODIES):
        var base = i * MODEL_BODY_SIZE
        host_buf[base + MODEL_IDX_MASS] = model.masses[i]
        host_buf[base + MODEL_IDX_INV_MASS] = model.inv_masses[i]
        host_buf[base + MODEL_IDX_RADIUS] = model.radii[i]
        host_buf[base + MODEL_IDX_IXX] = model.inertias[i * 3 + 0]
        host_buf[base + MODEL_IDX_IYY] = model.inertias[i * 3 + 1]
        host_buf[base + MODEL_IDX_IZZ] = model.inertias[i * 3 + 2]
        host_buf[base + MODEL_IDX_INV_IXX] = model.inv_inertias[i * 3 + 0]
        host_buf[base + MODEL_IDX_INV_IYY] = model.inv_inertias[i * 3 + 1]
        host_buf[base + MODEL_IDX_INV_IZZ] = model.inv_inertias[i * 3 + 2]
        host_buf[base + MODEL_IDX_GEOM_TYPE] = Scalar[DTYPE](
            model.geom_types[i]
        )
        host_buf[base + MODEL_IDX_HALF_LENGTH] = model.half_lengths[i]
        host_buf[base + MODEL_IDX_HALF_X] = model.half_x[i]
        host_buf[base + MODEL_IDX_HALF_Y] = model.half_y[i]
        host_buf[base + MODEL_IDX_HALF_Z] = model.half_z[i]

    return host_buf^


fn init_model_host_buffer[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](ctx: DeviceContext) raises -> HostBuffer[DTYPE]:
    """Create an empty model host buffer for manual initialization.

    Layout per body (14 floats):
    [mass, inv_mass, radius, ixx, iyy, izz, inv_ixx, inv_iyy, inv_izz,
     geom_type, half_length, half_x, half_y, half_z]

    Use for testing when you want to set model properties directly rather than
    from a CPU Model struct.

    Returns:
        HostBuffer of size [NUM_BODIES * MODEL_BODY_SIZE], zero-initialized.
    """
    var host_buf = ctx.enqueue_create_host_buffer[DTYPE](
        NUM_BODIES * MODEL_BODY_SIZE
    )

    # Initialize all values with defaults (sphere geometry)
    for i in range(NUM_BODIES):
        var base = i * MODEL_BODY_SIZE
        host_buf[base + MODEL_IDX_MASS] = Scalar[DTYPE](0)
        host_buf[base + MODEL_IDX_INV_MASS] = Scalar[DTYPE](0)
        host_buf[base + MODEL_IDX_RADIUS] = Scalar[DTYPE](0)
        host_buf[base + MODEL_IDX_IXX] = Scalar[DTYPE](0)
        host_buf[base + MODEL_IDX_IYY] = Scalar[DTYPE](0)
        host_buf[base + MODEL_IDX_IZZ] = Scalar[DTYPE](0)
        host_buf[base + MODEL_IDX_INV_IXX] = Scalar[DTYPE](0)
        host_buf[base + MODEL_IDX_INV_IYY] = Scalar[DTYPE](0)
        host_buf[base + MODEL_IDX_INV_IZZ] = Scalar[DTYPE](0)
        host_buf[base + MODEL_IDX_GEOM_TYPE] = Scalar[DTYPE](GEOM_SPHERE)
        host_buf[base + MODEL_IDX_HALF_LENGTH] = Scalar[DTYPE](0)
        host_buf[base + MODEL_IDX_HALF_X] = Scalar[DTYPE](0)
        host_buf[base + MODEL_IDX_HALF_Y] = Scalar[DTYPE](0)
        host_buf[base + MODEL_IDX_HALF_Z] = Scalar[DTYPE](0)

    return host_buf^


# =============================================================================
# CPU to GPU Data Transfer
# =============================================================================


fn copy_data_to_host_buffer[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](
    mut host_buf: HostBuffer[DTYPE],
    env: Int,
    data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS],
):
    """Copy CPU Data to a specific environment slot in host buffer.

    Args:
        host_buf: Target host buffer.
        env: Environment index within the batch.
        data: CPU data to copy.
    """
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var env_base = env * STATE_SIZE

    # Copy body states
    for body in range(NUM_BODIES):
        var body_base = env_base + body_offset[NUM_BODIES, MAX_CONTACTS](body)

        # Position
        host_buf[body_base + BODY_IDX_PX] = data.positions[body * 3 + 0]
        host_buf[body_base + BODY_IDX_PY] = data.positions[body * 3 + 1]
        host_buf[body_base + BODY_IDX_PZ] = data.positions[body * 3 + 2]

        # Quaternion
        host_buf[body_base + BODY_IDX_QX] = data.quaternions[body * 4 + 0]
        host_buf[body_base + BODY_IDX_QY] = data.quaternions[body * 4 + 1]
        host_buf[body_base + BODY_IDX_QZ] = data.quaternions[body * 4 + 2]
        host_buf[body_base + BODY_IDX_QW] = data.quaternions[body * 4 + 3]

        # Velocity
        host_buf[body_base + BODY_IDX_VX] = data.velocities[body * 3 + 0]
        host_buf[body_base + BODY_IDX_VY] = data.velocities[body * 3 + 1]
        host_buf[body_base + BODY_IDX_VZ] = data.velocities[body * 3 + 2]

        # Angular velocity
        host_buf[body_base + BODY_IDX_WX] = data.angular_velocities[
            body * 3 + 0
        ]
        host_buf[body_base + BODY_IDX_WY] = data.angular_velocities[
            body * 3 + 1
        ]
        host_buf[body_base + BODY_IDX_WZ] = data.angular_velocities[
            body * 3 + 2
        ]

    # Copy contacts (if any)
    var num_contacts = data.num_contacts
    for c in range(num_contacts):
        var c_base = env_base + contact_offset[NUM_BODIES, MAX_CONTACTS](c)
        host_buf[c_base + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
            data.contacts[c].body_a
        )
        host_buf[c_base + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
            data.contacts[c].body_b
        )
        host_buf[c_base + CONTACT_IDX_POS_X] = data.contacts[c].pos_x
        host_buf[c_base + CONTACT_IDX_POS_Y] = data.contacts[c].pos_y
        host_buf[c_base + CONTACT_IDX_POS_Z] = data.contacts[c].pos_z
        host_buf[c_base + CONTACT_IDX_NX] = data.contacts[c].normal_x
        host_buf[c_base + CONTACT_IDX_NY] = data.contacts[c].normal_y
        host_buf[c_base + CONTACT_IDX_NZ] = data.contacts[c].normal_z
        host_buf[c_base + CONTACT_IDX_DIST] = data.contacts[c].dist
        host_buf[c_base + CONTACT_IDX_IMPULSE_N] = data.contacts[c].impulse_n
        host_buf[c_base + CONTACT_IDX_IMPULSE_T1] = data.contacts[c].impulse_t1
        host_buf[c_base + CONTACT_IDX_IMPULSE_T2] = data.contacts[c].impulse_t2

    # Set metadata
    var meta_base = env_base + metadata_offset[NUM_BODIES, MAX_CONTACTS]()
    host_buf[meta_base + META_IDX_NUM_CONTACTS] = Scalar[DTYPE](num_contacts)


# =============================================================================
# GPU to CPU Data Transfer
# =============================================================================


fn copy_host_buffer_to_data[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](
    host_buf: HostBuffer[DTYPE],
    env: Int,
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS],
):
    """Copy data from host buffer to CPU Data structure.

    Args:
        host_buf: Source host buffer.
        env: Environment index within the batch.
        data: Target CPU data structure.
    """
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var env_base = env * STATE_SIZE

    # Copy body states
    for body in range(NUM_BODIES):
        var body_base = env_base + body_offset[NUM_BODIES, MAX_CONTACTS](body)

        # Position
        data.positions[body * 3 + 0] = host_buf[body_base + BODY_IDX_PX]
        data.positions[body * 3 + 1] = host_buf[body_base + BODY_IDX_PY]
        data.positions[body * 3 + 2] = host_buf[body_base + BODY_IDX_PZ]

        # Quaternion
        data.quaternions[body * 4 + 0] = host_buf[body_base + BODY_IDX_QX]
        data.quaternions[body * 4 + 1] = host_buf[body_base + BODY_IDX_QY]
        data.quaternions[body * 4 + 2] = host_buf[body_base + BODY_IDX_QZ]
        data.quaternions[body * 4 + 3] = host_buf[body_base + BODY_IDX_QW]

        # Velocity
        data.velocities[body * 3 + 0] = host_buf[body_base + BODY_IDX_VX]
        data.velocities[body * 3 + 1] = host_buf[body_base + BODY_IDX_VY]
        data.velocities[body * 3 + 2] = host_buf[body_base + BODY_IDX_VZ]

        # Angular velocity
        data.angular_velocities[body * 3 + 0] = host_buf[
            body_base + BODY_IDX_WX
        ]
        data.angular_velocities[body * 3 + 1] = host_buf[
            body_base + BODY_IDX_WY
        ]
        data.angular_velocities[body * 3 + 2] = host_buf[
            body_base + BODY_IDX_WZ
        ]

    # Get number of contacts from metadata
    var meta_base = env_base + metadata_offset[NUM_BODIES, MAX_CONTACTS]()
    var num_contacts = Int(host_buf[meta_base + META_IDX_NUM_CONTACTS])
    data.num_contacts = num_contacts

    # Copy contacts
    for c in range(num_contacts):
        var c_base = env_base + contact_offset[NUM_BODIES, MAX_CONTACTS](c)
        data.contacts[c].body_a = Int(host_buf[c_base + CONTACT_IDX_BODY_A])
        data.contacts[c].body_b = Int(host_buf[c_base + CONTACT_IDX_BODY_B])
        data.contacts[c].pos_x = host_buf[c_base + CONTACT_IDX_POS_X]
        data.contacts[c].pos_y = host_buf[c_base + CONTACT_IDX_POS_Y]
        data.contacts[c].pos_z = host_buf[c_base + CONTACT_IDX_POS_Z]
        data.contacts[c].normal_x = host_buf[c_base + CONTACT_IDX_NX]
        data.contacts[c].normal_y = host_buf[c_base + CONTACT_IDX_NY]
        data.contacts[c].normal_z = host_buf[c_base + CONTACT_IDX_NZ]
        data.contacts[c].dist = host_buf[c_base + CONTACT_IDX_DIST]
        data.contacts[c].impulse_n = host_buf[c_base + CONTACT_IDX_IMPULSE_N]
        data.contacts[c].impulse_t1 = host_buf[c_base + CONTACT_IDX_IMPULSE_T1]
        data.contacts[c].impulse_t2 = host_buf[c_base + CONTACT_IDX_IMPULSE_T2]


# =============================================================================
# Convenience Accessors
# =============================================================================


fn set_body_position[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](
    mut host_buf: HostBuffer[DTYPE],
    env: Int,
    body: Int,
    x: Scalar[DTYPE],
    y: Scalar[DTYPE],
    z: Scalar[DTYPE],
):
    """Set position for a specific body in a specific environment."""
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var body_base = env * STATE_SIZE + body_offset[NUM_BODIES, MAX_CONTACTS](
        body
    )
    host_buf[body_base + BODY_IDX_PX] = x
    host_buf[body_base + BODY_IDX_PY] = y
    host_buf[body_base + BODY_IDX_PZ] = z


fn set_body_velocity[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](
    mut host_buf: HostBuffer[DTYPE],
    env: Int,
    body: Int,
    vx: Scalar[DTYPE],
    vy: Scalar[DTYPE],
    vz: Scalar[DTYPE],
):
    """Set linear velocity for a specific body in a specific environment."""
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var body_base = env * STATE_SIZE + body_offset[NUM_BODIES, MAX_CONTACTS](
        body
    )
    host_buf[body_base + BODY_IDX_VX] = vx
    host_buf[body_base + BODY_IDX_VY] = vy
    host_buf[body_base + BODY_IDX_VZ] = vz


fn get_body_position[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](
    host_buf: HostBuffer[DTYPE],
    env: Int,
    body: Int,
) -> Tuple[
    Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]
]:
    """Get position for a specific body in a specific environment."""
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var body_base = env * STATE_SIZE + body_offset[NUM_BODIES, MAX_CONTACTS](
        body
    )
    return (
        host_buf[body_base + BODY_IDX_PX],
        host_buf[body_base + BODY_IDX_PY],
        host_buf[body_base + BODY_IDX_PZ],
    )


fn get_body_velocity[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](
    host_buf: HostBuffer[DTYPE],
    env: Int,
    body: Int,
) -> Tuple[
    Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]
]:
    """Get velocity for a specific body in a specific environment."""
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var body_base = env * STATE_SIZE + body_offset[NUM_BODIES, MAX_CONTACTS](
        body
    )
    return (
        host_buf[body_base + BODY_IDX_VX],
        host_buf[body_base + BODY_IDX_VY],
        host_buf[body_base + BODY_IDX_VZ],
    )


fn get_body_z[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](host_buf: HostBuffer[DTYPE], env: Int, body: Int) -> Scalar[DTYPE]:
    """Get z position for a specific body."""
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var body_base = env * STATE_SIZE + body_offset[NUM_BODIES, MAX_CONTACTS](
        body
    )
    return host_buf[body_base + BODY_IDX_PZ]


fn get_body_vz[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](host_buf: HostBuffer[DTYPE], env: Int, body: Int) -> Scalar[DTYPE]:
    """Get z velocity for a specific body."""
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var body_base = env * STATE_SIZE + body_offset[NUM_BODIES, MAX_CONTACTS](
        body
    )
    return host_buf[body_base + BODY_IDX_VZ]


fn get_num_contacts[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](host_buf: HostBuffer[DTYPE], env: Int) -> Int:
    """Get number of active contacts for an environment."""
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var meta_base = (
        env * STATE_SIZE + metadata_offset[NUM_BODIES, MAX_CONTACTS]()
    )
    return Int(host_buf[meta_base + META_IDX_NUM_CONTACTS])


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
](ctx: DeviceContext) raises -> HostBuffer[DTYPE]:
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
    var buffer = ctx.enqueue_create_host_buffer[DTYPE](total_size)

    # Initialize to zero
    for i in range(total_size):
        buffer[i] = Scalar[DTYPE](0)

    return buffer


fn create_gc_model_buffer[
    DTYPE: DType,
    NBODY: Int,
    NJOINT: Int,
](ctx: DeviceContext) raises -> HostBuffer[DTYPE]:
    """Allocate host buffer for GC model.

    The model buffer contains static configuration shared by all environments.

    Returns:
        Pointer to allocated buffer.
    """
    comptime model_size = gc_model_size[NBODY, NJOINT]()
    var buffer = ctx.enqueue_create_host_buffer[DTYPE](model_size)

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
    buffer: HostBuffer[DTYPE],
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
        buffer[offset + GC_BODY_IDX_INV_IXX] = model.body_inv_inertia[
            body * 3 + 0
        ]
        buffer[offset + GC_BODY_IDX_INV_IYY] = model.body_inv_inertia[
            body * 3 + 1
        ]
        buffer[offset + GC_BODY_IDX_INV_IZZ] = model.body_inv_inertia[
            body * 3 + 2
        ]
        buffer[offset + GC_BODY_IDX_POS_X] = model.body_pos[body * 3 + 0]
        buffer[offset + GC_BODY_IDX_POS_Y] = model.body_pos[body * 3 + 1]
        buffer[offset + GC_BODY_IDX_POS_Z] = model.body_pos[body * 3 + 2]
        buffer[offset + GC_BODY_IDX_QUAT_X] = model.body_quat[body * 4 + 0]
        buffer[offset + GC_BODY_IDX_QUAT_Y] = model.body_quat[body * 4 + 1]
        buffer[offset + GC_BODY_IDX_QUAT_Z] = model.body_quat[body * 4 + 2]
        buffer[offset + GC_BODY_IDX_QUAT_W] = model.body_quat[body * 4 + 3]
        buffer[offset + GC_BODY_IDX_PARENT] = Scalar[DTYPE](
            model.body_parent[body]
        )
        buffer[offset + GC_BODY_IDX_GEOM_TYPE] = Scalar[DTYPE](
            model.body_geom_type[body]
        )
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
        buffer[offset + GC_JOINT_IDX_RANGE_MIN] = joint.range_min
        buffer[offset + GC_JOINT_IDX_RANGE_MAX] = joint.range_max
        buffer[offset + GC_JOINT_IDX_ARMATURE] = joint.armature
        buffer[offset + GC_JOINT_IDX_DAMPING] = joint.damping
        buffer[offset + GC_JOINT_IDX_STIFFNESS] = joint.stiffness

    # Copy metadata
    var meta_offset = gc_model_metadata_offset[NBODY, NJOINT]()
    buffer[meta_offset + GC_MODEL_META_IDX_NBODY] = Scalar[DTYPE](NBODY)
    buffer[meta_offset + GC_MODEL_META_IDX_NJOINT] = Scalar[DTYPE](
        model.num_joints
    )
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
    buffer: HostBuffer[DTYPE],
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
