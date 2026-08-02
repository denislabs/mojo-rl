"""The contact tangent frame — MuJoCo's `mju_makeFrame`, in ONE place.

`CONTACT_IDX_FRAME_T1_*` in the contact record is a **hint**, not a tangent.
Only the PLANE/CAPSULE branches write it (they store the capsule axis, so the
tangent basis lines up with the geometry) — four sites in
`collision/contact_detection.mojo`, and NOT capsule/capsule, capsule/sphere or
capsule/box, which despite the name "capsule narrow phase" leave the slot
alone like every other pair. Whatever the previous occupant of that contact
slot left there is what a reader gets. Turning it into a tangent takes three
steps that the raw field has NOT had applied to it:

  * fall back to a default axis when the hint is absent (`|hint|^2 < 0.25`,
    MuJoCo's own test in `mju_makeFrame`, engine_util_spatial.c:538),
  * Gram-Schmidt it against the normal — the capsule axis is not orthogonal
    to the contact normal in general,
  * normalize, with a second fallback for a hint parallel to the normal.

Reading the field directly gives an unnormalized, non-orthogonal, possibly
stale vector. That is exactly the bug this module was extracted to kill:
`dynamics/rne_post.mojo` and `gpu/cfrc_ext_gpu.mojo` both did it, so every
contact force they mapped back into world coordinates had the right normal
component (that one only needs `n`) and a garbage tangential one. It was
invisible on `qacc` — the solvers build their own frame correctly — and
surfaced only once quadruped asked for a force/torque SENSOR reading. Its
other consumer is Ant's `contact_cost`.

Three solver sites built this correctly but each from its own copy of the
same forty lines (`constraints/contact_solve.mojo` twice,
`solver/island_pgs_solve.mojo` once). They now all call this, so a frame
convention change lands everywhere at once or not at all.

THE DEFAULT AXIS IS MuJoCo'S: `(0,1,0)` unless `|n_y| >= 0.5`, in which case
`(0,0,1)` (`engine_util_spatial.c:517-525`). It used to be "the least-aligned
basis axis", which for a floor contact made our `t1` the world x-axis where
MuJoCo's is the world y-axis — a 90-degree rotation about the normal (task
#25, fixed 2026-08-03).

⚠ HOW VISIBLE THAT ROTATION IS DEPENDS ON THE CONE, and the easy answer is
wrong. Under ISOTROPIC friction:

  * ELLIPTIC — the cone's cross-section is a CIRCLE, invariant under ANY
    rotation about the normal. Nothing to see.
  * PYRAMIDAL — the cross-section is the SQUARE `|f_t1| <= mu*f_n`,
    `|f_t2| <= mu*f_n`, which is invariant under a 90-degree rotation and
    NOT under any other. So whether the old rule and MuJoCo's differ
    observably depends on the ANGLE between them, not merely on isotropy.

The two rules differ by exactly 90 degrees when the normal is axis-aligned —
every floor contact, which is why the standing quadruped's force/torque
sensors do not move — but by a general angle otherwise. Measured: for
`n = (0.3, 0.4, sqrt(.75))` the old rule projects `x` (least-aligned) and
MuJoCo projects `y` (`|n_y| < 0.5`), and those two projections are
**97.9 degrees** apart, against exactly 90 for `n = (0,0,1)`. A PYRAMIDAL
model with tilted contact normals therefore CAN see a real difference,
bounded by the pyramid's own approximation error to the circle. Do not
describe this fix as a pure relabeling.

What IS invariant regardless: the per-direction labels' only consumers
(`rne_post` / `cfrc_ext`) sum `t1` and `t2` back into a world-frame
resultant. And anisotropy cannot arise by accident — a geom pair's contact
friction is built as `[fri0, fri0, fri1, fri2, fri2]`
(`engine_collision_driver.c:1483-1487`), so slide1 == slide2 and
roll1 == roll2 identically, and only an explicit `<pair>` with a
five-component `friction=` can break that. There is no `<pair>` and no
`condim="6"` anywhere in `mojo_rl/`.

`tests/physics3d/test_contact_frame_vs_mujoco.mojo` gates the whole function
against MuJoCo's own `contact.frame`, on live contacts, across both default
branches and the capsule-hint path.
"""

from std.math import sqrt


@always_inline
def _default_axis[
    DTYPE: DType
](ny: Scalar[DTYPE]) -> InlineArray[Scalar[DTYPE], 3]:
    """`mju_makeFrame`'s undefined-yaxis case, engine_util_spatial.c:517-525.

    `(0,1,0)` unless the normal is itself close to y, in which case `(0,0,1)`.
    MuJoCo spells the test `frame[1] < 0.5 && frame[1] > -0.5`, i.e. strictly
    inside ±0.5 keeps y — so exactly ±0.5 takes the z branch, and the
    comparison below is written to match that boundary rather than the more
    natural `abs(ny) < 0.5`.
    """
    var out = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    if ny < Scalar[DTYPE](0.5) and ny > Scalar[DTYPE](-0.5):
        out[1] = Scalar[DTYPE](1)
    else:
        out[2] = Scalar[DTYPE](1)
    return out


@always_inline
def contact_tangent_frame[
    DTYPE: DType
](
    nx: Scalar[DTYPE],
    ny: Scalar[DTYPE],
    nz: Scalar[DTYPE],
    hint_x_in: Scalar[DTYPE],
    hint_y_in: Scalar[DTYPE],
    hint_z_in: Scalar[DTYPE],
) -> InlineArray[Scalar[DTYPE], 6]:
    """Orthonormal (t1, t2) for a contact normal, from a possibly-absent hint.

    Returns `[t1x, t1y, t1z, t2x, t2y, t2z]` with `t2 = n x t1`. The normal is
    assumed already normalized (every narrow phase writes it that way).
    """
    var hint_x = hint_x_in
    var hint_y = hint_y_in
    var hint_z = hint_z_in

    # No hint (non-capsule pair, or a stale slot): default axis.
    if hint_x * hint_x + hint_y * hint_y + hint_z * hint_z < Scalar[DTYPE](
        0.25
    ):
        var d = _default_axis[DTYPE](ny)
        hint_x = d[0]
        hint_y = d[1]
        hint_z = d[2]

    # Gram-Schmidt against the normal.
    var dot_nh = nx * hint_x + ny * hint_y + nz * hint_z
    var t1x = hint_x - dot_nh * nx
    var t1y = hint_y - dot_nh * ny
    var t1z = hint_z - dot_nh * nz
    var t1_mag = sqrt(t1x * t1x + t1y * t1y + t1z * t1z)

    # Hint parallel to the normal — nothing survived; take the default axis.
    #
    # ⚠ MuJoCo DOES NOT DO THIS. `mju_makeFrame` normalizes the collapsed
    # yaxis and `mju_normalize3` rewrites a zero vector as (1,0,0)
    # (engine_util_spatial.c:530), so MuJoCo's t1 is (1,0,0) here — which is
    # orthogonal to the normal only by luck and is exactly the normal itself
    # when a capsule lies along x and contacts on its end cap. Ours re-defaults
    # instead, which always yields a valid frame. Deliberately NOT faithful:
    # reproducing it can hand the solver a degenerate basis. The divergence is
    # unreachable through a geom pair anyway — only a capsule narrow phase
    # writes a hint, and it writes the capsule AXIS, so this fires only for an
    # end-cap contact along that axis.
    if t1_mag < Scalar[DTYPE](1e-10):
        var d = _default_axis[DTYPE](ny)
        hint_x = d[0]
        hint_y = d[1]
        hint_z = d[2]
        dot_nh = nx * hint_x + ny * hint_y + nz * hint_z
        t1x = hint_x - dot_nh * nx
        t1y = hint_y - dot_nh * ny
        t1z = hint_z - dot_nh * nz
        t1_mag = sqrt(t1x * t1x + t1y * t1y + t1z * t1z)

    if t1_mag > Scalar[DTYPE](1e-10):
        t1x = t1x / t1_mag
        t1y = t1y / t1_mag
        t1z = t1z / t1_mag

    var out = InlineArray[Scalar[DTYPE], 6](uninitialized=True)
    out[0] = t1x
    out[1] = t1y
    out[2] = t1z
    # t2 = n x t1
    out[3] = ny * t1z - nz * t1y
    out[4] = nz * t1x - nx * t1z
    out[5] = nx * t1y - ny * t1x
    return out
