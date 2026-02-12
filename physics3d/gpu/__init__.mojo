"""Physics3D GPU - Batched physics simulation on GPU."""

# Constants for buffer layout
from .constants import (
    # GPU configuration
    TPB,
    TILE,
    # State buffer layout
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
    model_size,
    model_body_offset,
    model_joint_offset,
    model_metadata_offset,
    # Buffer layout constants
    CONTACT_SIZE,
    METADATA_SIZE,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    # Physics defaults
    DEFAULT_GRAVITY_Z,
    DEFAULT_TIMESTEP,
    DEFAULT_RESTITUTION,
    # solref/solimp metadata indices
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
    # Unified geom constants
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
    model_geom_offset,
)

# Buffer utilities
from .buffer_utils import (
    create_state_buffer,
    create_model_buffer,
    copy_model_to_buffer,
    copy_geoms_to_buffer,
    copy_data_to_buffer,
    copy_buffer_to_data,
)


# GPU kernels colocated with CPU implementations
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
from ..dynamics.mass_matrix import (
    compute_mass_matrix_diagonal_gpu,
    compute_mass_matrix_full_gpu,
    ldl_factor_gpu,
    ldl_solve_gpu,
    compute_M_inv_from_ldl_gpu,
)
from ..dynamics.bias_forces import compute_bias_forces_gpu
from ..dynamics.jacobian import compute_composite_inertia_gpu
