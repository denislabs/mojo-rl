"""Tendon limit rows — MuJoCo's `mjCNSTR_LIMIT_TENDON`.

A tendon whose `<spatial>`/`<fixed>` element carries `limited="true"` and a
`range` gets a one-sided row per violated side, exactly like a joint limit but
with a DENSE Jacobian:

    dist = side * (range[(side+1)/2] - ten_length)      side = -1 (lo), +1 (hi)
    jac  = -side * ten_J                                (engine_core_constraint.c:814)
    D    = 1 / R,  R = (1-imp)/imp * tendon_invweight0

`tendon_invweight0` is `J M^-1 J^T` at qpos0 and is the row's `diagApprox`
(`engine_core_constraint.c:1130`) — the same slot `dof_invweight0` fills for a
joint limit. Getting it wrong is a silent multiplicative error on the whole
constraint force, which is exactly how bug 20 hid for weeks; the ball_in_cup
gate diffs it against MuJoCo directly.

WHY THIS IS A ROW AND NOT A POST-PASS. Commit 04a7c508 moved joint limits and
dry friction INTO the Newton/CG systems after proving that solving them
sequentially after the contacts was the whole of dm_control finger's contact
residual (0.216 N of force error, 0.0390 gated). ball_in_cup has exactly the
same shape — a caught ball rests on the cup's capsules (contacts) while the
string is taut (this row), on shared dofs — so building it as a post-pass
would have reproduced the bug we just finished removing.

SCOPE: PYRAMIDAL only. The pyramidal solvers already carry a dense `Je` edge
list, so a dense row costs nothing structurally. The ELLIPTIC core stores its
scalar rows as `(dof, sign)` precisely to stay under Metal's local-memory
ceiling (see constraints/scalar_rows.mojo), and `MAX_TENDON * NV` extra floats
there is a change that needs its own measurement. No elliptic model has a
limited tendon; `ModelDefFromXML` RAISES on the combination rather than
letting it pass silently.

Also SOLVER-scoped: the rows are built in `solver/newton_solve.mojo`, which is
the env default (`Phyics3dEnv.SOLVER = "newton"`). `cg_solve` is elliptic-only
and so is covered by the same raise. The PGS solvers (`contact_solve`,
`island_pgs_solve`) would silently omit the row; they are already the two
paths that omit the joint-limit/friction rows for the same structural reason
(see constraints/scalar_rows.mojo), and nothing gated or trained selects them.
"""

from std.collections import InlineArray
from std.math import pow
from layout import Layout, LayoutTensor


from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_SITE_SIZE,
    MODEL_TENDON_SIZE,
    MODEL_META_IDX_NTENDON,
    JOINT_IDX_DOF_ADR,
    TENDON_KIND_SPATIAL,
    TENDON_IDX_KIND,
    TENDON_IDX_LIMITED,
    TENDON_IDX_RANGE_MIN,
    TENDON_IDX_RANGE_MAX,
    TENDON_IDX_MARGIN,
    TENDON_IDX_INVWEIGHT0,
    TENDON_IDX_NUM_JOINTS,
    TENDON_IDX_JOINT_0,
    TENDON_IDX_COEF_0,
    TENDON_IDX_SOLREF_LIM_0,
    TENDON_IDX_SOLREF_LIM_1,
    TENDON_IDX_SOLIMP_LIM_0,
    TENDON_IDX_SOLIMP_LIM_1,
    TENDON_IDX_SOLIMP_LIM_2,
    TENDON_IDX_SOLIMP_LIM_3,
    TENDON_IDX_SOLIMP_LIM_4,
)
from ..dynamics.tendon import spatial_tendon_length_jac


comptime MJ_MINIMP: Float64 = 0.0001
comptime MJ_MAXIMP: Float64 = 0.9999


@always_inline
def _clamp_imp[DTYPE: DType](v: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """MuJoCo clamps every impedance to [mjMINIMP, mjMAXIMP]
    (engine_core_constraint.c). Omitting this was bug 21."""
    if v < Scalar[DTYPE](MJ_MINIMP):
        return Scalar[DTYPE](MJ_MINIMP)
    if v > Scalar[DTYPE](MJ_MAXIMP):
        return Scalar[DTYPE](MJ_MAXIMP)
    return v


@always_inline
def _solimp[
    DTYPE: DType
](
    pen: Scalar[DTYPE],
    dmin: Scalar[DTYPE],
    dmax: Scalar[DTYPE],
    width: Scalar[DTYPE],
    midpoint: Scalar[DTYPE],
    power: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """MuJoCo's piecewise-power impedance ramp over the violation depth."""
    if dmin == dmax or width <= Scalar[DTYPE](0):
        return _clamp_imp[DTYPE](Scalar[DTYPE](0.5) * (dmin + dmax))
    var x = pen / width
    if x <= Scalar[DTYPE](0):
        return _clamp_imp[DTYPE](dmin)
    if x >= Scalar[DTYPE](1):
        return _clamp_imp[DTYPE](dmax)
    var y: Scalar[DTYPE]
    if power == Scalar[DTYPE](1):
        y = x
    elif x <= midpoint:
        y = pow(x, power) / pow(midpoint, power - Scalar[DTYPE](1))
    else:
        y = Scalar[DTYPE](1) - pow(Scalar[DTYPE](1) - x, power) / pow(
            Scalar[DTYPE](1) - midpoint, power - Scalar[DTYPE](1)
        )
    return _clamp_imp[DTYPE](dmin + y * (dmax - dmin))


def build_tendon_limit_rows[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    NSITE: Int,
    NTENDON: Int,
    V_SIZE: Int,
    ME: Int,
    BATCH: Int,
](
    env: Int,
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
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
    m_inv: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    mut Je: InlineArray[Scalar[DTYPE], ME * V_SIZE],
    mut De: InlineArray[Scalar[DTYPE], ME],
    mut bias_e: InlineArray[Scalar[DTYPE], ME],
    mut num_edges: Int,
):
    """Append a row per violated tendon limit side to the pyramidal edge list.

    Rows are ONE-SIDED (force >= 0), so the caller leaves `kind_e` at
    SROW_LIMIT and `R_e`/`floss_e` at 0, exactly as it does for joint limits.
    """
    comptime if NTENDON == 0:
        return

    var nten = Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NTENDON]))
    if nten == 0:
        return
    if nten > NTENDON:
        nten = NTENDON

    var tJ = InlineArray[Scalar[DTYPE], V_SIZE](fill=Scalar[DTYPE](0))

    for t in range(nten):
        if Int(rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_LIMITED])) == 0:
            continue

        # --- length + moment arm ------------------------------------------
        var ten_len = Scalar[DTYPE](0)
        var kind = Int(rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_KIND]))
        if kind == TENDON_KIND_SPATIAL:
            ten_len = spatial_tendon_length_jac[
                DTYPE, NV, NBODY, NJOINT, NSITE, NTENDON, V_SIZE, BATCH
            ](
                env, t, tendons, sites, bodies, joints, mmeta, subtree_com,
                cdof, xpos, xquat, tJ,
            )
        else:
            # A fixed tendon's length needs qpos, which this builder is not
            # given; no gated model has a limited fixed tendon, and letting it
            # through as length 0 would fabricate a permanent violation.
            continue

        # --- ten_J . qvel --------------------------------------------------
        var ten_vel = Scalar[DTYPE](0)
        for i in range(NV):
            ten_vel += tJ[i] * rebind[Scalar[DTYPE]](qvel[env, i])

        # --- K = J M^-1 J^T at the CURRENT pose ----------------------------
        var K_t = Scalar[DTYPE](0)
        for a in range(NV):
            if tJ[a] == Scalar[DTYPE](0):
                continue
            var acc = Scalar[DTYPE](0)
            for b in range(NV):
                acc += rebind[Scalar[DTYPE]](m_inv[env, a * NV + b]) * tJ[b]
            K_t += tJ[a] * acc
        if K_t < Scalar[DTYPE](1e-10):
            K_t = Scalar[DTYPE](1e-10)

        var rmin = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_RANGE_MIN])
        var rmax = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_RANGE_MAX])
        var margin = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_MARGIN])

        var tc = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLREF_LIM_0])
        var dr = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLREF_LIM_1])
        var dmin = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLIMP_LIM_0])
        var dmax = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLIMP_LIM_1])
        var width = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLIMP_LIM_2])
        var midpt = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLIMP_LIM_3])
        var power = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLIMP_LIM_4])

        var K_spring = Scalar[DTYPE](1) / (dmax * dmax * tc * tc * dr * dr)
        var B_damp = Scalar[DTYPE](2) / (dmax * tc)

        var diag = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_INVWEIGHT0])
        if diag < Scalar[DTYPE](1e-10):
            diag = K_t

        # side = -1 (lower), +1 (upper); `sign` below is MuJoCo's -side, so
        # the row's Jacobian is `sign * ten_J`.
        for s in range(2):
            if num_edges >= ME:
                break
            var side = Scalar[DTYPE](-1) if s == 0 else Scalar[DTYPE](1)
            var bound = rmin if s == 0 else rmax
            var dist = side * (bound - ten_len)
            if dist >= margin:
                continue

            var sign = -side
            var pen = -dist
            var v_lim = sign * ten_vel

            var imp = _solimp[DTYPE](pen, dmin, dmax, width, midpt, power)
            var R = (Scalar[DTYPE](1) - imp) / imp * diag
            if R < Scalar[DTYPE](1e-14):
                R = Scalar[DTYPE](1e-14)

            for i in range(NV):
                Je[num_edges * NV + i] = sign * tJ[i]

            # Same inv_K round-trip the joint-limit rows use, so both kinds of
            # limit land on bit-identical D for identical (K, R).
            var inv_K = Scalar[DTYPE](1) / (K_t + R)
            var R_recov = Scalar[DTYPE](1) / inv_K - K_t
            if R_recov < Scalar[DTYPE](1e-14):
                R_recov = Scalar[DTYPE](1e-14)
            De[num_edges] = Scalar[DTYPE](1) / R_recov
            bias_e[num_edges] = B_damp * v_lim - K_spring * imp * pen
            num_edges += 1
