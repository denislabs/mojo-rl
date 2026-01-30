"""Physics3D v2 GPU Utilities - State buffer initialization and conversion.

Provides functions to:
1. Initialize state buffers with default values
2. Copy from CPU Model/Data to GPU buffer
3. Copy from GPU buffer to CPU Data
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from .constants import (
    STATE_SIZE,
    GEOM_SPHERE,
    IDX_X,
    IDX_Y,
    IDX_Z,
    IDX_QX,
    IDX_QY,
    IDX_QZ,
    IDX_QW,
    IDX_VX,
    IDX_VY,
    IDX_VZ,
    IDX_WX,
    IDX_WY,
    IDX_WZ,
    IDX_AX,
    IDX_AY,
    IDX_AZ,
    IDX_ALPHA_X,
    IDX_ALPHA_Y,
    IDX_ALPHA_Z,
    IDX_FX,
    IDX_FY,
    IDX_FZ,
    IDX_TAU_X,
    IDX_TAU_Y,
    IDX_TAU_Z,
    IDX_XPOS_X,
    IDX_XPOS_Y,
    IDX_XPOS_Z,
    IDX_CONTACT_ACTIVE,
    IDX_CONTACT_DEPTH,
    IDX_CONTACT_NX,
    IDX_CONTACT_NY,
    IDX_CONTACT_NZ,
    IDX_CONTACT_PX,
    IDX_CONTACT_PY,
    IDX_CONTACT_PZ,
)


fn init_state_host_buffer[
    DTYPE: DType, BATCH: Int
](ctx: DeviceContext) raises -> HostBuffer[DTYPE]:
    """Create and initialize a host buffer with default state values.

    Each environment gets:
    - Position at origin (0, 0, 0)
    - Identity quaternion (0, 0, 0, 1)
    - Zero velocities
    - Zero accelerations
    - Zero forces
    - No active contact

    Returns:
        HostBuffer of size [BATCH * STATE_SIZE].
    """
    var host_buf = ctx.enqueue_create_host_buffer[DTYPE](BATCH * STATE_SIZE)

    # Initialize each environment
    for env in range(BATCH):
        var base = env * STATE_SIZE

        # qpos: position at origin, identity quaternion
        host_buf[base + IDX_X] = Scalar[DTYPE](0)
        host_buf[base + IDX_Y] = Scalar[DTYPE](0)
        host_buf[base + IDX_Z] = Scalar[DTYPE](0)
        host_buf[base + IDX_QX] = Scalar[DTYPE](0)
        host_buf[base + IDX_QY] = Scalar[DTYPE](0)
        host_buf[base + IDX_QZ] = Scalar[DTYPE](0)
        host_buf[base + IDX_QW] = Scalar[DTYPE](1)

        # qvel: zero velocities
        host_buf[base + IDX_VX] = Scalar[DTYPE](0)
        host_buf[base + IDX_VY] = Scalar[DTYPE](0)
        host_buf[base + IDX_VZ] = Scalar[DTYPE](0)
        host_buf[base + IDX_WX] = Scalar[DTYPE](0)
        host_buf[base + IDX_WY] = Scalar[DTYPE](0)
        host_buf[base + IDX_WZ] = Scalar[DTYPE](0)

        # qacc: zero accelerations
        host_buf[base + IDX_AX] = Scalar[DTYPE](0)
        host_buf[base + IDX_AY] = Scalar[DTYPE](0)
        host_buf[base + IDX_AZ] = Scalar[DTYPE](0)
        host_buf[base + IDX_ALPHA_X] = Scalar[DTYPE](0)
        host_buf[base + IDX_ALPHA_Y] = Scalar[DTYPE](0)
        host_buf[base + IDX_ALPHA_Z] = Scalar[DTYPE](0)

        # qfrc: zero applied forces
        host_buf[base + IDX_FX] = Scalar[DTYPE](0)
        host_buf[base + IDX_FY] = Scalar[DTYPE](0)
        host_buf[base + IDX_FZ] = Scalar[DTYPE](0)
        host_buf[base + IDX_TAU_X] = Scalar[DTYPE](0)
        host_buf[base + IDX_TAU_Y] = Scalar[DTYPE](0)
        host_buf[base + IDX_TAU_Z] = Scalar[DTYPE](0)

        # xpos: zero (will be computed)
        host_buf[base + IDX_XPOS_X] = Scalar[DTYPE](0)
        host_buf[base + IDX_XPOS_Y] = Scalar[DTYPE](0)
        host_buf[base + IDX_XPOS_Z] = Scalar[DTYPE](0)

        # contact: inactive
        host_buf[base + IDX_CONTACT_ACTIVE] = Scalar[DTYPE](0)
        host_buf[base + IDX_CONTACT_DEPTH] = Scalar[DTYPE](0)
        host_buf[base + IDX_CONTACT_NX] = Scalar[DTYPE](0)
        host_buf[base + IDX_CONTACT_NY] = Scalar[DTYPE](0)
        host_buf[base + IDX_CONTACT_NZ] = Scalar[DTYPE](1)
        host_buf[base + IDX_CONTACT_PX] = Scalar[DTYPE](0)
        host_buf[base + IDX_CONTACT_PY] = Scalar[DTYPE](0)
        host_buf[base + IDX_CONTACT_PZ] = Scalar[DTYPE](0)

    return host_buf^


fn set_position[
    DTYPE: DType
](
    mut host_buf: HostBuffer[DTYPE],
    env: Int,
    x: Scalar[DTYPE],
    y: Scalar[DTYPE],
    z: Scalar[DTYPE],
):
    """Set position for a specific environment in host buffer."""
    var base = env * STATE_SIZE
    host_buf[base + IDX_X] = x
    host_buf[base + IDX_Y] = y
    host_buf[base + IDX_Z] = z


fn set_velocity[
    DTYPE: DType
](
    mut host_buf: HostBuffer[DTYPE],
    env: Int,
    vx: Scalar[DTYPE],
    vy: Scalar[DTYPE],
    vz: Scalar[DTYPE],
):
    """Set linear velocity for a specific environment in host buffer."""
    var base = env * STATE_SIZE
    host_buf[base + IDX_VX] = vx
    host_buf[base + IDX_VY] = vy
    host_buf[base + IDX_VZ] = vz


fn get_position[
    DTYPE: DType
](
    host_buf: HostBuffer[DTYPE], env: Int
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Get position for a specific environment from host buffer."""
    var base = env * STATE_SIZE
    return (
        host_buf[base + IDX_X],
        host_buf[base + IDX_Y],
        host_buf[base + IDX_Z],
    )


fn get_velocity[
    DTYPE: DType
](
    host_buf: HostBuffer[DTYPE], env: Int
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Get velocity for a specific environment from host buffer."""
    var base = env * STATE_SIZE
    return (
        host_buf[base + IDX_VX],
        host_buf[base + IDX_VY],
        host_buf[base + IDX_VZ],
    )


fn get_z[DTYPE: DType](host_buf: HostBuffer[DTYPE], env: Int) -> Scalar[DTYPE]:
    """Get z position for a specific environment from host buffer."""
    var base = env * STATE_SIZE
    return host_buf[base + IDX_Z]


fn get_vz[DTYPE: DType](host_buf: HostBuffer[DTYPE], env: Int) -> Scalar[DTYPE]:
    """Get z velocity for a specific environment from host buffer."""
    var base = env * STATE_SIZE
    return host_buf[base + IDX_VZ]


fn is_contact_active[
    DTYPE: DType
](host_buf: HostBuffer[DTYPE], env: Int) -> Bool:
    """Check if contact is active for a specific environment."""
    var base = env * STATE_SIZE
    return host_buf[base + IDX_CONTACT_ACTIVE] >= Scalar[DTYPE](0.5)


fn get_contact_depth[
    DTYPE: DType
](host_buf: HostBuffer[DTYPE], env: Int) -> Scalar[DTYPE]:
    """Get contact depth for a specific environment."""
    var base = env * STATE_SIZE
    return host_buf[base + IDX_CONTACT_DEPTH]
