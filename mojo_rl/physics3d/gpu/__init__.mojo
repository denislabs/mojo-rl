"""Physics3D GPU layer: the fields record-layout constants (per-record
sizes + column indices — the slab OFFSET tables died at the G5 sunset) and
the cfrc_ext / cvel per-field kernels."""

# Constants for buffer layout
from .constants import (
    # GPU configuration
    TPB,
    TILE,
    # Record sizes + per-record column indices (the fields record layout)
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
)

# GPU kernels colocated with CPU implementations
from ..kinematics.quat_math import (
    gpu_quat_mul,
    gpu_quat_rotate,
    gpu_axis_angle_to_quat,
    gpu_quat_normalize,
)
# Legacy GPU mass_matrix/jacobian (CRBA/LDL) deleted at the fields sunset.

# Phase-2 post-substep GPU kernels
from .cfrc_ext_gpu import compute_cfrc_ext_fields
from .cvel_gpu import compute_cvel_fields
