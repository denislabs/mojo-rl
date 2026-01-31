from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from math import sqrt
from ..gpu.constants import (
    BODY_IDX_PX,
    BODY_IDX_PY,
    BODY_IDX_PZ,
    BODY_IDX_VX,
    BODY_IDX_VY,
    BODY_IDX_VZ,
    BODY_IDX_WX,
    BODY_IDX_WY,
    BODY_IDX_WZ,
    BODY_IDX_QX,
    BODY_IDX_QY,
    BODY_IDX_QZ,
    BODY_IDX_QW,
    body_offset,
)


@always_inline
fn integrate_positions_kernel[
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
):
    """Integrate positions using semi-implicit Euler."""
    for i in range(NUM_BODIES):
        var b_off = body_offset[NUM_BODIES, MAX_CONTACTS](i)

        # Linear position
        var px = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_PX])
        var py = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_PY])
        var pz = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_PZ])
        var vx = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_VX])
        var vy = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_VY])
        var vz = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_VZ])

        state[env, b_off + BODY_IDX_PX] = px + dt * vx
        state[env, b_off + BODY_IDX_PY] = py + dt * vy
        state[env, b_off + BODY_IDX_PZ] = pz + dt * vz

        # Quaternion integration: q' = q + 0.5*dt*ω⊗q
        var half_dt = Scalar[DTYPE](0.5) * dt
        var wx = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_WX])
        var wy = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_WY])
        var wz = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_WZ])
        var qx = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_QX])
        var qy = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_QY])
        var qz = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_QZ])
        var qw = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_QW])

        var qx_new = qx + half_dt * (wx * qw + wy * qz - wz * qy)
        var qy_new = qy + half_dt * (-wx * qz + wy * qw + wz * qx)
        var qz_new = qz + half_dt * (wx * qy - wy * qx + wz * qw)
        var qw_new = qw + half_dt * (-wx * qx - wy * qy - wz * qz)

        # Normalize quaternion
        var norm_sq = (
            qx_new * qx_new
            + qy_new * qy_new
            + qz_new * qz_new
            + qw_new * qw_new
        )
        if norm_sq > Scalar[DTYPE](1e-10):
            var inv_norm = Scalar[DTYPE](1.0) / sqrt(norm_sq)
            state[env, b_off + BODY_IDX_QX] = qx_new * inv_norm
            state[env, b_off + BODY_IDX_QY] = qy_new * inv_norm
            state[env, b_off + BODY_IDX_QZ] = qz_new * inv_norm
            state[env, b_off + BODY_IDX_QW] = qw_new * inv_norm
