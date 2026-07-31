"""Spatial (site-routed) tendon length and Jacobian — MuJoCo's `mj_tendon`.

Port of the SPATIAL branch of `mj_tendon`
(references/mujoco-3.3.6/src/engine/engine_core_smooth.c:651-856), restricted
to site-to-site routing:

    length   = sum_k |p_{k+1} - p_k|
    ten_J[:] = sum_k dif_k . (jacp(p_{k+1}, body_{k+1}) - jacp(p_k, body_k))

with `dif_k = normalize(p_{k+1} - p_k)`.

THREE THINGS THAT LOOK LIKE DETAILS AND ARE NOT:

1. The two endpoint Jacobians are evaluated at DIFFERENT POINTS on DIFFERENT
   BODIES. A contact Jacobian gets away with one point because both bodies
   touch there; a tendon segment does not. Hence two calls to
   `_contact_jacobian_row` rather than one bilateral call.

2. MuJoCo accumulates a segment's moment ONLY when `wbody[k] != wbody[k+1]`
   (`engine_core_smooth.c:793`). A segment between two sites on the same body
   has constant length, so it contributes to `length` but must contribute
   nothing to `ten_J`. Skipping the guard would add two Jacobians that cancel
   only in exact arithmetic.

3. Site world positions are RECOMPUTED here from `xpos`/`xquat`, not read from
   `Data.site_xpos` which FK has already filled. That is deliberate: `Data`
   sizes `site_xpos` as `BATCH * NSITE * 3`, so passing it into the solvers
   means binding a ZERO-BYTE tensor on every site-less model, which crashes at
   bind (it took out three solver tests before this was rewired). See
   `_site_world`.

Wrap geoms and pulleys are not supported; `full_parser` rejects them at parse
time, so nothing here has to guess.

FIXED tendons are not handled here — they are `length = sum coef_i * qpos_i`
with a trivial Jacobian, and live in `constraints/equality_tendon.mojo`.
"""

from std.collections import InlineArray
from std.math import sqrt
from layout import Layout, LayoutTensor

from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_SITE_SIZE,
    MODEL_TENDON_SIZE,
    SITE_IDX_BODY,
    SITE_IDX_POS_X,
    SITE_IDX_POS_Y,
    SITE_IDX_POS_Z,
    TENDON_IDX_NUM_SITES,
    TENDON_IDX_SITE_0,
)
from ..constraints.contact_solve import _contact_jacobian_row
from ..kinematics.quat_math import gpu_quat_rotate


@always_inline
def _site_world[
    DTYPE: DType, NBODY: Int, NSITE: Int, BATCH: Int
](
    env: Int,
    site_idx: Int,
    s_body: Int,
    sites: LayoutTensor[
        DTYPE, Layout.row_major(NSITE, MODEL_SITE_SIZE), MutAnyOrigin
    ],
    xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """A site's world position, recomputed from `xpos`/`xquat`.

    Identical arithmetic to `kinematics/forward_kinematics._fk_site`, which has
    already written this into `Data.site_xpos`. It is recomputed rather than
    read because threading `site_xpos` into the solvers means binding a tensor
    that is EMPTY on every site-less model (walker, cheetah, ant, ...), and
    `Data` sizes it `BATCH * NSITE * 3`. Passing that operand crashed three
    solver tests at bind. `xpos`/`xquat` are always non-empty and the solver
    already receives both, so this costs one quaternion rotation per waypoint
    and removes an entire class of empty-operand failure.
    """
    var sp_x = rebind[Scalar[DTYPE]](sites[site_idx, SITE_IDX_POS_X])
    var sp_y = rebind[Scalar[DTYPE]](sites[site_idx, SITE_IDX_POS_Y])
    var sp_z = rebind[Scalar[DTYPE]](sites[site_idx, SITE_IDX_POS_Z])
    var bqx = rebind[Scalar[DTYPE]](xquat[env, s_body * 4 + 0])
    var bqy = rebind[Scalar[DTYPE]](xquat[env, s_body * 4 + 1])
    var bqz = rebind[Scalar[DTYPE]](xquat[env, s_body * 4 + 2])
    var bqw = rebind[Scalar[DTYPE]](xquat[env, s_body * 4 + 3])
    var rot = gpu_quat_rotate(bqx, bqy, bqz, bqw, sp_x, sp_y, sp_z)
    return (
        rebind[Scalar[DTYPE]](xpos[env, s_body * 3 + 0]) + rot[0],
        rebind[Scalar[DTYPE]](xpos[env, s_body * 3 + 1]) + rot[1],
        rebind[Scalar[DTYPE]](xpos[env, s_body * 3 + 2]) + rot[2],
    )


@always_inline
def _sqrt_pos[DTYPE: DType](v: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """sqrt clamped at 0 — the argument is a squared norm, so a tiny negative
    can only be rounding.

    Must be `sqrt`, not `v ** 0.5`: pow is evaluated as exp(0.5*log v) and
    loses ~5 digits. With pow this returned ten_length 0.2920000000126963 and
    ten_J 0.9999999999565194 where MuJoCo has 0.292 and 1 — a 4e-11 relative
    error in float64, which is 5 orders worse than the 1e-16 the arithmetic
    is entitled to and would have been mistaken for a real modelling gap.
    """
    if v <= Scalar[DTYPE](0):
        return Scalar[DTYPE](0)
    return sqrt(v)


def spatial_tendon_length_jac[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    NSITE: Int,
    NTENDON: Int,
    V_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    t_i: Int,
    tendons: LayoutTensor[
        DTYPE, Layout.row_major(NTENDON, MODEL_TENDON_SIZE), MutAnyOrigin
    ],
    sites: LayoutTensor[
        DTYPE, Layout.row_major(NSITE, MODEL_SITE_SIZE), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    mmeta: LayoutTensor[DTYPE, Layout.row_major(MODEL_META_SIZE), MutAnyOrigin],
    subtree_com: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    cdof: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    mut J_row: InlineArray[Scalar[DTYPE], V_SIZE],
) -> Scalar[DTYPE]:
    """Length of spatial tendon `t_i`, with its dense moment arm in `J_row`.

    `J_row` is zeroed here, then accumulated over segments.
    """
    for i in range(V_SIZE):
        J_row[i] = Scalar[DTYPE](0)

    var length = Scalar[DTYPE](0)

    var nsites = Int(
        rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_NUM_SITES])
    )
    if nsites < 2:
        return length

    var seg_J = InlineArray[Scalar[DTYPE], V_SIZE](fill=Scalar[DTYPE](0))

    for k in range(nsites - 1):
        var s0 = Int(
            rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_SITE_0 + k])
        )
        var s1 = Int(
            rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_SITE_0 + k + 1])
        )
        if s0 < 0 or s1 < 0:
            continue

        var b0 = Int(rebind[Scalar[DTYPE]](sites[s0, SITE_IDX_BODY]))
        var b1 = Int(rebind[Scalar[DTYPE]](sites[s1, SITE_IDX_BODY]))
        var p0 = _site_world[DTYPE, NBODY, NSITE, BATCH](
            env, s0, b0, sites, xpos, xquat
        )
        var p1 = _site_world[DTYPE, NBODY, NSITE, BATCH](
            env, s1, b1, sites, xpos, xquat
        )
        var p0x = p0[0]
        var p0y = p0[1]
        var p0z = p0[2]
        var p1x = p1[0]
        var p1y = p1[1]
        var p1z = p1[2]

        var dx = p1x - p0x
        var dy = p1y - p0y
        var dz = p1z - p0z
        var seg_len = _sqrt_pos[DTYPE](dx * dx + dy * dy + dz * dz)
        length += seg_len

        # Same body => constant segment length => no moment (see docstring).
        if b0 == b1:
            continue

        # A degenerate segment has no direction; MuJoCo's mju_normalize3
        # leaves the vector zero, which contributes nothing.
        if seg_len <= Scalar[DTYPE](0):
            continue
        var inv = Scalar[DTYPE](1) / seg_len
        var ux = dx * inv
        var uy = dy * inv
        var uz = dz * inv

        # + dif . jacp(p1, b1)
        _contact_jacobian_row[DTYPE, NV, NBODY, NJOINT, V_SIZE, BATCH](
            env, subtree_com, joints, bodies, mmeta, cdof,
            b1, 0, p1x, p1y, p1z, ux, uy, uz, seg_J,
        )
        for i in range(V_SIZE):
            J_row[i] += seg_J[i]

        # - dif . jacp(p0, b0)
        _contact_jacobian_row[DTYPE, NV, NBODY, NJOINT, V_SIZE, BATCH](
            env, subtree_com, joints, bodies, mmeta, cdof,
            b0, 0, p0x, p0y, p0z, ux, uy, uz, seg_J,
        )
        for i in range(V_SIZE):
            J_row[i] -= seg_J[i]

    return length
