"""Physics3D v2 GPU - Multi-body batched physics simulation on GPU.

Provides buffer utilities for parallel physics simulation across many environments.
GPU simulation is done via the ImpulseIntegrator and PGSIntegrator classes.

Example usage:
    from gpu.host import DeviceContext
    from physics3d_v2.gpu import (
        init_state_host_buffer,
        init_model_host_buffer,
        set_body_position,
        get_body_position,
        compute_state_size,
    )
    from physics3d_v2 import ImpulseIntegrator, PGSIntegrator

    # Setup
    var ctx = DeviceContext()
    var host_state = init_state_host_buffer[DTYPE, NUM_BODIES, MAX_CONTACTS, BATCH](ctx)
    var host_model = init_model_host_buffer[DTYPE, NUM_BODIES, MAX_CONTACTS](ctx)

    # Configure model (mass, inv_mass, radius, inertia...)
    host_model[0] = 1.0  # mass
    host_model[1] = 1.0  # inv_mass
    host_model[2] = 0.1  # radius
    # ... etc

    # Set initial positions
    set_body_position[DTYPE, NUM_BODIES, MAX_CONTACTS](host_state, env=0, body=0, x=0, y=0, z=1)

    # Transfer to GPU
    comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS]()
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](NUM_BODIES * 9)
    ctx.enqueue_copy(state_buf, host_state)
    ctx.enqueue_copy(model_buf, host_model)
    ctx.synchronize()

    # Run simulation using ImpulseIntegrator or PGSIntegrator
    ImpulseIntegrator.simulate_gpu[DTYPE, NUM_BODIES, MAX_CONTACTS, BATCH](
        ctx, state_buf, model_buf, num_steps=100,
        dt=0.01, gravity_z=-9.81, ground_z=0.0, restitution=0.5, friction=0.5
    )
    ctx.synchronize()

    # Transfer back to host
    ctx.enqueue_copy(host_state, state_buf)
    ctx.synchronize()

    # Read results
    var pos = get_body_position[DTYPE, NUM_BODIES, MAX_CONTACTS](host_state, env=0, body=0)
"""

# Constants for buffer layout
from .constants import (
    # GPU configuration
    TPB,
    TILE,
    # Per-body state layout
    BODY_STATE_SIZE,
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
    # Per-contact state layout
    CONTACT_STATE_SIZE,
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
    # Metadata
    METADATA_SIZE,
    META_IDX_NUM_CONTACTS,
    # Helper functions
    compute_state_size,
    body_offset,
    contact_offset,
    metadata_offset,
    # Geometry types
    GEOM_PLANE,
    GEOM_SPHERE,
    # Physics defaults
    DEFAULT_GRAVITY_Z,
    DEFAULT_TIMESTEP,
    DEFAULT_RESTITUTION,
    DEFAULT_BAUMGARTE,
    DEFAULT_SLOP,
    # Legacy constants (for backward compatibility)
    LEGACY_STATE_SIZE,
    IDX_X,
    IDX_Y,
    IDX_Z,
    IDX_VX,
    IDX_VY,
    IDX_VZ,
)

# Buffer utilities
from .buffer_utils import (
    init_state_host_buffer,
    init_model_host_buffer,
    create_model_host_buffer,
    copy_data_to_host_buffer,
    copy_host_buffer_to_data,
    set_body_position,
    set_body_velocity,
    get_body_position,
    get_body_velocity,
    get_body_z,
    get_body_vz,
    get_num_contacts,
    # GC buffer utilities
    create_gc_state_buffer,
    create_gc_model_buffer,
    copy_model_to_buffer,
    copy_data_to_buffer,
    copy_buffer_to_data,
)

# GC GPU kernels (main step kernel and integration-specific kernels in gc_kernels)
from .gc_kernels import (
    step_gc_kernel,
    detect_ground_contacts_gpu,
    compute_contact_forces_gpu,
    integrate_gc_gpu,
    normalize_qpos_quaternions_gpu,
)

# GC GPU kernels colocated with CPU implementations
from ..kinematics.quat_math import (
    gpu_quat_mul,
    gpu_quat_rotate,
    gpu_axis_angle_to_quat,
    gpu_quat_normalize,
)
from ..kinematics.forward_kinematics import (
    forward_kinematics_gpu,
    compute_body_velocities_gpu,
)
from ..dynamics.mass_matrix import compute_mass_matrix_diagonal_gpu
from ..dynamics.bias_forces import compute_bias_forces_gpu
