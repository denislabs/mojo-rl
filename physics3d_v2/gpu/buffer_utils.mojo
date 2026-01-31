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
            var body_base = env_base + body_offset[NUM_BODIES, MAX_CONTACTS](body)

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
    """Create a host buffer with model data (masses, inv_masses, radii).

    Layout per body: [mass, inv_mass, radius, ixx, iyy, izz, inv_ixx, inv_iyy, inv_izz]
    Total: 9 floats per body.

    Returns:
        HostBuffer of size [NUM_BODIES * 9].
    """
    comptime MODEL_BODY_SIZE = 9
    var host_buf = ctx.enqueue_create_host_buffer[DTYPE](NUM_BODIES * MODEL_BODY_SIZE)

    for i in range(NUM_BODIES):
        var base = i * MODEL_BODY_SIZE
        host_buf[base + 0] = model.masses[i]
        host_buf[base + 1] = model.inv_masses[i]
        host_buf[base + 2] = model.radii[i]
        host_buf[base + 3] = model.inertias[i * 3 + 0]
        host_buf[base + 4] = model.inertias[i * 3 + 1]
        host_buf[base + 5] = model.inertias[i * 3 + 2]
        host_buf[base + 6] = model.inv_inertias[i * 3 + 0]
        host_buf[base + 7] = model.inv_inertias[i * 3 + 1]
        host_buf[base + 8] = model.inv_inertias[i * 3 + 2]

    return host_buf^


fn init_model_host_buffer[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](ctx: DeviceContext) raises -> HostBuffer[DTYPE]:
    """Create an empty model host buffer for manual initialization.

    Layout per body: [mass, inv_mass, radius, ixx, iyy, izz, inv_ixx, inv_iyy, inv_izz]
    Total: 9 floats per body.

    Use for testing when you want to set model properties directly rather than
    from a CPU Model struct.

    Returns:
        HostBuffer of size [NUM_BODIES * 9], zero-initialized.
    """
    comptime MODEL_BODY_SIZE = 9
    var host_buf = ctx.enqueue_create_host_buffer[DTYPE](NUM_BODIES * MODEL_BODY_SIZE)

    # Zero-initialize all values
    for i in range(NUM_BODIES * MODEL_BODY_SIZE):
        host_buf[i] = Scalar[DTYPE](0)

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
        host_buf[body_base + BODY_IDX_WX] = data.angular_velocities[body * 3 + 0]
        host_buf[body_base + BODY_IDX_WY] = data.angular_velocities[body * 3 + 1]
        host_buf[body_base + BODY_IDX_WZ] = data.angular_velocities[body * 3 + 2]

    # Copy contacts (if any)
    var num_contacts = data.num_contacts
    for c in range(num_contacts):
        var c_base = env_base + contact_offset[NUM_BODIES, MAX_CONTACTS](c)
        host_buf[c_base + CONTACT_IDX_BODY_A] = Scalar[DTYPE](data.contacts[c].body_a)
        host_buf[c_base + CONTACT_IDX_BODY_B] = Scalar[DTYPE](data.contacts[c].body_b)
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
        data.angular_velocities[body * 3 + 0] = host_buf[body_base + BODY_IDX_WX]
        data.angular_velocities[body * 3 + 1] = host_buf[body_base + BODY_IDX_WY]
        data.angular_velocities[body * 3 + 2] = host_buf[body_base + BODY_IDX_WZ]

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
    var body_base = env * STATE_SIZE + body_offset[NUM_BODIES, MAX_CONTACTS](body)
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
    var body_base = env * STATE_SIZE + body_offset[NUM_BODIES, MAX_CONTACTS](body)
    host_buf[body_base + BODY_IDX_VX] = vx
    host_buf[body_base + BODY_IDX_VY] = vy
    host_buf[body_base + BODY_IDX_VZ] = vz


fn get_body_position[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](
    host_buf: HostBuffer[DTYPE],
    env: Int,
    body: Int,
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Get position for a specific body in a specific environment."""
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var body_base = env * STATE_SIZE + body_offset[NUM_BODIES, MAX_CONTACTS](body)
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
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Get velocity for a specific body in a specific environment."""
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var body_base = env * STATE_SIZE + body_offset[NUM_BODIES, MAX_CONTACTS](body)
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
    var body_base = env * STATE_SIZE + body_offset[NUM_BODIES, MAX_CONTACTS](body)
    return host_buf[body_base + BODY_IDX_PZ]


fn get_body_vz[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](host_buf: HostBuffer[DTYPE], env: Int, body: Int) -> Scalar[DTYPE]:
    """Get z velocity for a specific body."""
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var body_base = env * STATE_SIZE + body_offset[NUM_BODIES, MAX_CONTACTS](body)
    return host_buf[body_base + BODY_IDX_VZ]


fn get_num_contacts[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
](host_buf: HostBuffer[DTYPE], env: Int) -> Int:
    """Get number of active contacts for an environment."""
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var meta_base = env * STATE_SIZE + metadata_offset[NUM_BODIES, MAX_CONTACTS]()
    return Int(host_buf[meta_base + META_IDX_NUM_CONTACTS])
