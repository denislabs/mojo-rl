"""Physics3D v2 GPU - Batched physics simulation on GPU.

Provides a fused physics kernel following the 1 environment = 1 thread pattern.
All state is stored in flat LayoutTensor[BATCH, STATE_SIZE] buffers.
"""

from .constants import (
    STATE_SIZE,
    QPOS_OFFSET,
    QVEL_OFFSET,
    QACC_OFFSET,
    QFRC_OFFSET,
    XPOS_OFFSET,
    CONTACT_OFFSET,
    # Individual field indices
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
from .layout import Physics3DV2Layout
from .kernel import Physics3DV2Kernel, step_gpu, step_gpu_batched
from .utils import (
    init_state_host_buffer,
    set_position,
    set_velocity,
    get_position,
    get_velocity,
    get_z,
    get_vz,
    is_contact_active,
    get_contact_depth,
)
