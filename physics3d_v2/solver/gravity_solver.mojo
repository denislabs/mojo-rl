from layout import LayoutTensor, Layout
from ..gpu.constants import BODY_IDX_VZ, body_offset


@always_inline
fn apply_gravity_gpu[
    DTYPE: DType,
    NUM_BODIES: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    dt: Scalar[DTYPE],
    gravity_z: Scalar[DTYPE],
):
    """Apply gravity to all body velocities."""
    for i in range(NUM_BODIES):
        var b_off = body_offset[NUM_BODIES, MAX_CONTACTS](i)
        var vz = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_VZ])
        state[env, b_off + BODY_IDX_VZ] = vz + dt * gravity_z
