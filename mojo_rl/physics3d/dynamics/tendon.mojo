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

4. A waypoint is a SITE OR A WRAP GEOM, and a geom is consumed together with
   the site TWO entries along (`engine_core_smooth.c:1022`): `site-geom-site`
   is one step producing THREE sub-segments, not two steps producing two.
   `j` then advances by 2. ⚠ It advances by 2 whenever the entry WAS a geom,
   even when `mju_wrap` declined to wrap — a geom the tendon happens to clear
   is skipped, not routed through.

`<pulley>` is not supported; `full_parser` rejects it at parse time, so
nothing here has to guess.

FIXED tendons are not handled here — they are `length = sum coef_i * qpos_i`
with a trivial Jacobian, and live in `constraints/equality_tendon.mojo`.
"""

from std.collections import InlineArray
from std.math import sqrt
from layout import Layout, LayoutTensor
from ..fields.scratch import Scratch

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
    TENDON_IDX_NUM_WRAPS,
    TENDON_IDX_WOBJ_0,
    TENDON_IDX_WTYPE_0,
    TENDON_IDX_WPRM_0,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_RADIUS,
    WRAP_SITE,
    WRAP_SPHERE,
    WRAP_CYLINDER,
)
from .wrap import mju_wrap, WrapOut
from ..fields import DimsLike
from .jac_contact_row import _contact_jacobian_row
from ..kinematics.quat_math import gpu_quat_rotate, gpu_quat_mul


@always_inline
def _site_world[
    DTYPE: DType,
    BATCH: Int,
    D: DimsLike,
    L_SITES: Layout,
    L_XPOS: Layout,
    L_XQUAT: Layout](
    env: Int,
    site_idx: Int,
    s_body: Int,
    dims: D,
    sites: LayoutTensor[
        DTYPE, L_SITES, MutAnyOrigin
    ],
    xpos: LayoutTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: LayoutTensor[
        DTYPE, L_XQUAT, MutAnyOrigin
    ],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """A site's world position, recomputed from `xpos`/`xquat`.

    Identical arithmetic to `kinematics/forward_kinematics._fk_site`, which has
    already written this into `Data.site_xpos`. It is recomputed rather than
    read because threading `site_xpos` into the solvers means binding a tensor
    that is EMPTY on every site-less model (walker, cheetah, ant, ...), and
    `Data` sizes it `BATCH * nsite * 3`. Passing that operand crashed three
    solver tests at bind. `xpos`/`xquat` are always non-empty and the solver
    already receives both, so this costs one quaternion rotation per waypoint
    and removes an entire class of empty-operand failure.
    """
    var nsite = dims.get_nsite()
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


struct GeomFrame[DTYPE: DType](Copyable, ImplicitlyCopyable, Movable):
    """A wrap geom's world pose: position plus a ROW-MAJOR 3x3 rotation."""

    var px: Scalar[Self.DTYPE]
    var py: Scalar[Self.DTYPE]
    var pz: Scalar[Self.DTYPE]
    var m00: Scalar[Self.DTYPE]
    var m01: Scalar[Self.DTYPE]
    var m02: Scalar[Self.DTYPE]
    var m10: Scalar[Self.DTYPE]
    var m11: Scalar[Self.DTYPE]
    var m12: Scalar[Self.DTYPE]
    var m20: Scalar[Self.DTYPE]
    var m21: Scalar[Self.DTYPE]
    var m22: Scalar[Self.DTYPE]

    def __init__(out self):
        self.px = Scalar[Self.DTYPE](0)
        self.py = Scalar[Self.DTYPE](0)
        self.pz = Scalar[Self.DTYPE](0)
        self.m00 = Scalar[Self.DTYPE](1)
        self.m01 = Scalar[Self.DTYPE](0)
        self.m02 = Scalar[Self.DTYPE](0)
        self.m10 = Scalar[Self.DTYPE](0)
        self.m11 = Scalar[Self.DTYPE](1)
        self.m12 = Scalar[Self.DTYPE](0)
        self.m20 = Scalar[Self.DTYPE](0)
        self.m21 = Scalar[Self.DTYPE](0)
        self.m22 = Scalar[Self.DTYPE](1)


def _geom_world_frame[
    DTYPE: DType,
    BATCH: Int,
    D: DimsLike,
    L_GEOMS: Layout,
    L_XPOS: Layout,
    L_XQUAT: Layout,
](
    env: Int,
    g_id: Int,
    dims: D,
    geoms: LayoutTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: LayoutTensor[DTYPE, L_XQUAT, MutAnyOrigin],
) -> GeomFrame[DTYPE]:
    """A wrap geom's world frame, MuJoCo's `geom_xpos` / `geom_xmat`.

    ⚠⚠ THE MATRIX IS BUILT BY ROTATING THE THREE BASIS VECTORS, not from the
    quaternion-to-matrix formula. Column `j` is `quat_rotate(q, e_j)` by
    definition, so this cannot disagree with `gpu_quat_rotate` about the
    (x, y, z, w) storage order or the active/passive convention — and a
    transposed rotation here is the kind of error that leaves the wrap
    correct on a symmetric pulley and wrong on a tilted one, which is a hard
    thing to see. Three rotations instead of ~20 multiply-adds, once per wrap.
    """
    var gb = Int(rebind[Scalar[DTYPE]](geoms[g_id, GEOM_IDX_BODY]))
    if gb < 0:
        gb = 0

    var bqx = rebind[Scalar[DTYPE]](xquat[env, gb * 4 + 0])
    var bqy = rebind[Scalar[DTYPE]](xquat[env, gb * 4 + 1])
    var bqz = rebind[Scalar[DTYPE]](xquat[env, gb * 4 + 2])
    var bqw = rebind[Scalar[DTYPE]](xquat[env, gb * 4 + 3])

    var lx = rebind[Scalar[DTYPE]](geoms[g_id, GEOM_IDX_POS_X])
    var ly = rebind[Scalar[DTYPE]](geoms[g_id, GEOM_IDX_POS_Y])
    var lz = rebind[Scalar[DTYPE]](geoms[g_id, GEOM_IDX_POS_Z])
    var lqx = rebind[Scalar[DTYPE]](geoms[g_id, GEOM_IDX_QUAT_X])
    var lqy = rebind[Scalar[DTYPE]](geoms[g_id, GEOM_IDX_QUAT_Y])
    var lqz = rebind[Scalar[DTYPE]](geoms[g_id, GEOM_IDX_QUAT_Z])
    var lqw = rebind[Scalar[DTYPE]](geoms[g_id, GEOM_IDX_QUAT_W])

    var q = gpu_quat_mul(bqx, bqy, bqz, bqw, lqx, lqy, lqz, lqw)
    var off = gpu_quat_rotate(bqx, bqy, bqz, bqw, lx, ly, lz)

    var c0 = gpu_quat_rotate(
        q[0], q[1], q[2], q[3],
        Scalar[DTYPE](1), Scalar[DTYPE](0), Scalar[DTYPE](0),
    )
    var c1 = gpu_quat_rotate(
        q[0], q[1], q[2], q[3],
        Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0),
    )
    var c2 = gpu_quat_rotate(
        q[0], q[1], q[2], q[3],
        Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1),
    )

    var f = GeomFrame[DTYPE]()
    f.px = rebind[Scalar[DTYPE]](xpos[env, gb * 3 + 0]) + off[0]
    f.py = rebind[Scalar[DTYPE]](xpos[env, gb * 3 + 1]) + off[1]
    f.pz = rebind[Scalar[DTYPE]](xpos[env, gb * 3 + 2]) + off[2]
    f.m00 = c0[0]
    f.m10 = c0[1]
    f.m20 = c0[2]
    f.m01 = c1[0]
    f.m11 = c1[1]
    f.m21 = c1[2]
    f.m02 = c2[0]
    f.m12 = c2[1]
    f.m22 = c2[2]
    return f^


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
    V_CAP: Int,
    BATCH: Int,
    # ⚠ A PROVIDER ITS OWN BODY DOES NOT NEED. Nothing here reads a dimension
    # directly, so the sweep gave this declaration no `D` — and then `_site_
    # world`, which it calls, needed one. A caller's provider requirement is
    # the TRANSITIVE closure of its callees', not what its own lines mention.
    D: DimsLike,
    L_TENDONS: Layout,
    L_SITES: Layout,
    L_GEOMS: Layout,
    L_BODIES: Layout,
    L_JOINTS: Layout,
    L_MMETA: Layout,
    L_SUBTREE_COM: Layout,
    L_CDOF: Layout,
    L_XQUAT: Layout,
](
    env: Int,
    t_i: Int,
    dims: D,
    tendons: LayoutTensor[
        DTYPE, L_TENDONS, MutAnyOrigin
    ],
    sites: LayoutTensor[
        DTYPE, L_SITES, MutAnyOrigin
    ],
    # ⚠ THREADED IN FOR THE WRAP GEOMS, and it had to be: a wrap object's
    # world frame is its body's pose composed with its OWN local pos/quat, and
    # its radius is `GEOM_IDX_RADIUS`. The alternative — copying those nine
    # numbers into the tendon record beside each wrap entry — would be a
    # second spelling of the geom, and a geom resized in the studio would
    # leave the tendon routing round its old size.
    geoms: LayoutTensor[
        DTYPE, L_GEOMS, MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, L_BODIES, MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, L_JOINTS, MutAnyOrigin
    ],
    mmeta: LayoutTensor[DTYPE, L_MMETA, MutAnyOrigin],
    subtree_com: LayoutTensor[
        DTYPE, L_SUBTREE_COM, MutAnyOrigin
    ],
    cdof: LayoutTensor[DTYPE, L_CDOF, MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, L_SUBTREE_COM, MutAnyOrigin],
    xquat: LayoutTensor[
        DTYPE, L_XQUAT, MutAnyOrigin
    ],
    mut J_row: Scratch[Scalar[DTYPE], V_CAP],
) -> Scalar[DTYPE]:
    """Length of spatial tendon `t_i`, with its dense moment arm in `J_row`.

    `J_row` is zeroed here, then accumulated over segments.
    """
    var nv = dims.get_nv()
    for i in range(nv):
        J_row[i] = Scalar[DTYPE](0)

    var length = Scalar[DTYPE](0)

    var nwrap = Int(
        rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_NUM_WRAPS])
    )
    if nwrap < 2:
        return length

    var seg_J = Scratch[Scalar[DTYPE], V_CAP](nv, fill=Scalar[DTYPE](0))

    # ── the waypoint walk ────────────────────────────────────────────────
    # ⚠ `while`, NOT `for`. The step is 1 or 2 depending on whether entry
    # `j+1` is a wrap geom, and a `for k in range(nwrap-1)` cannot express
    # that — it would evaluate the geom entry as if it were a site, routing
    # the tendon THROUGH the pulley's centre.
    var j = 0
    while j < nwrap - 1:
        var s0 = Int(
            rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_WOBJ_0 + j])
        )
        var t1 = Int(
            rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_WTYPE_0 + j + 1])
        )
        var s1 = Int(
            rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_WOBJ_0 + j + 1])
        )
        if s0 < 0:
            j += 1
            continue

        var b0 = Int(rebind[Scalar[DTYPE]](sites[s0, SITE_IDX_BODY]))
        var p0 = _site_world[DTYPE, BATCH](
            env, s0, b0, dims, sites, xpos, xquat
        )

        var is_wrap = t1 == WRAP_SPHERE or t1 == WRAP_CYLINDER
        var g_id = -1
        var w = WrapOut[DTYPE]()

        if is_wrap:
            # The wrap object is at j+1; the segment's FAR SITE is at j+2.
            g_id = s1
            if j + 2 >= nwrap:
                break
            s1 = Int(
                rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_WOBJ_0 + j + 2])
            )
            var side_id = Int(
                rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_WPRM_0 + j + 1])
            )
            if s1 >= 0 and g_id >= 0:
                var b1p = Int(rebind[Scalar[DTYPE]](sites[s1, SITE_IDX_BODY]))
                var p1p = _site_world[DTYPE, BATCH](
                    env, s1, b1p, dims, sites, xpos, xquat
                )
                var gf = _geom_world_frame[DTYPE, BATCH](
                    env, g_id, dims, geoms, xpos, xquat
                )
                var has_side = side_id >= 0
                var sx = Scalar[DTYPE](0)
                var sy = Scalar[DTYPE](0)
                var sz = Scalar[DTYPE](0)
                if has_side:
                    var sb = Int(
                        rebind[Scalar[DTYPE]](sites[side_id, SITE_IDX_BODY])
                    )
                    var sp = _site_world[DTYPE, BATCH](
                        env, side_id, sb, dims, sites, xpos, xquat
                    )
                    sx = sp[0]
                    sy = sp[1]
                    sz = sp[2]
                var radius = rebind[Scalar[DTYPE]](
                    geoms[g_id, GEOM_IDX_RADIUS]
                )
                w = mju_wrap[DTYPE](
                    p0[0], p0[1], p0[2],
                    p1p[0], p1p[1], p1p[2],
                    gf.px, gf.py, gf.pz,
                    gf.m00, gf.m01, gf.m02,
                    gf.m10, gf.m11, gf.m12,
                    gf.m20, gf.m21, gf.m22,
                    radius,
                    WRAP_SPHERE if t1 == WRAP_SPHERE else WRAP_CYLINDER,
                    has_side, sx, sy, sz,
                )

        if s1 < 0:
            j += 2 if is_wrap else 1
            continue

        var b1 = Int(rebind[Scalar[DTYPE]](sites[s1, SITE_IDX_BODY]))
        var p1 = _site_world[DTYPE, BATCH](
            env, s1, b1, dims, sites, xpos, xquat
        )

        # ── the sub-segments: one straight run, or chord-arc-chord ───────
        # `wbody[k]` is the body each waypoint belongs to; the arc's two ends
        # both belong to the WRAP GEOM's body, which is what makes a tendon
        # sliding over a fixed pulley contribute no moment there.
        var nseg = 1
        var ax = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
        var ay = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
        var az = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
        var ab = InlineArray[Int, 4](fill=0)

        ax[0] = p0[0]
        ay[0] = p0[1]
        az[0] = p0[2]
        ab[0] = b0

        if w.wlen < 0:
            ax[1] = p1[0]
            ay[1] = p1[1]
            az[1] = p1[2]
            ab[1] = b1
            length += _sqrt_pos[DTYPE](
                (p1[0] - p0[0]) * (p1[0] - p0[0])
                + (p1[1] - p0[1]) * (p1[1] - p0[1])
                + (p1[2] - p0[2]) * (p1[2] - p0[2])
            )
        else:
            nseg = 3
            var gb = Int(
                rebind[Scalar[DTYPE]](geoms[g_id, GEOM_IDX_BODY])
            )
            if gb < 0:
                gb = 0
            ax[1] = w.p0x
            ay[1] = w.p0y
            az[1] = w.p0z
            ab[1] = gb
            ax[2] = w.p1x
            ay[2] = w.p1y
            az[2] = w.p1z
            ab[2] = gb
            ax[3] = p1[0]
            ay[3] = p1[1]
            az[3] = p1[2]
            ab[3] = b1
            # ⚠ THE ARC ITSELF IS `w.wlen`, NOT `|w0 - w1|`. Using the chord
            # would shorten every wrapped tendon by the bulge — plausible,
            # monotone in the same direction, and invisible without a gate.
            length += (
                _sqrt_pos[DTYPE](
                    (ax[1] - ax[0]) * (ax[1] - ax[0])
                    + (ay[1] - ay[0]) * (ay[1] - ay[0])
                    + (az[1] - az[0]) * (az[1] - az[0])
                )
                + w.wlen
                + _sqrt_pos[DTYPE](
                    (ax[3] - ax[2]) * (ax[3] - ax[2])
                    + (ay[3] - ay[2]) * (ay[3] - ay[2])
                    + (az[3] - az[2]) * (az[3] - az[2])
                )
            )

        for k in range(nseg):
            # Same body => constant segment length => no moment (see the
            # module docstring). The ARC's own sub-segment always hits this,
            # both ends being on the wrap geom's body.
            if ab[k] == ab[k + 1]:
                continue
            var dx = ax[k + 1] - ax[k]
            var dy = ay[k + 1] - ay[k]
            var dz = az[k + 1] - az[k]
            var sl = _sqrt_pos[DTYPE](dx * dx + dy * dy + dz * dz)
            if sl <= Scalar[DTYPE](0):
                continue
            var inv = Scalar[DTYPE](1) / sl
            var ux = dx * inv
            var uy = dy * inv
            var uz = dz * inv

            _contact_jacobian_row[DTYPE, V_CAP](
                env, subtree_com, joints, bodies, mmeta, cdof,
                ab[k + 1], 0, ax[k + 1], ay[k + 1], az[k + 1],
                ux, uy, uz, seg_J, nv,
            )
            for i in range(nv):
                J_row[i] += seg_J[i]

            _contact_jacobian_row[DTYPE, V_CAP](
                env, subtree_com, joints, bodies, mmeta, cdof,
                ab[k], 0, ax[k], ay[k], az[k], ux, uy, uz, seg_J, nv,
            )
            for i in range(nv):
                J_row[i] -= seg_J[i]

        j += 2 if is_wrap else 1

    return length
