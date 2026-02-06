from layout import LayoutTensor, Layout
from ..gpu.constants import BODY_IDX_VZ, body_offset


@always_inline
fn apply_gravity_gpu[
    DTYPE: DType,
    NUM_BODIES: Int,
    MAX_CONTACTS: Int,
    MAX_JOINTS: Int = 0,
    STATE_SIZE: Int = 0,
    BATCH: Int = 1,
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
        var b_off = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](i)
        var vz = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_VZ])
        state[env, b_off + BODY_IDX_VZ] = vz + dt * gravity_z
