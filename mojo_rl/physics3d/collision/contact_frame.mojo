"""The contact tangent frame — MuJoCo's `mju_makeFrame`, in ONE place.

`CONTACT_IDX_FRAME_T1_*` in the contact record is a **hint**, not a tangent.
Only the capsule narrow phases write it (they store the capsule axis, so the
tangent basis lines up with the geometry); every other primitive pair leaves
it alone, which means it is whatever the previous occupant of that contact
slot left there. Turning it into a tangent takes three steps that the raw
field has NOT had applied to it:

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

⚠ Our default axis is NOT MuJoCo's. MuJoCo picks `(0,1,0)` unless
`|n_y| >= 0.5`, in which case `(0,0,1)`; we pick the least-aligned basis
axis. For a floor contact that makes our `t1` the world x-axis where
MuJoCo's is the world y-axis — a 90-degree rotation about the normal. With
isotropic friction that is a relabeling the physics cannot see (both the
friction cone and the pyramid's four edges are invariant under it, and the
world-frame force is identical), which is why it has never shown up. It is
NOT invisible for anisotropic friction, where `friction[0]` and
`friction[1]` name specific directions. Kept as-is deliberately: changing it
perturbs the last bits of every contact golden in the suite, so it belongs
in its own change, not this one.
"""

from std.math import sqrt


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
        var abs_nx = abs(nx)
        var abs_ny = abs(ny)
        var abs_nz = abs(nz)
        if abs_nx <= abs_ny and abs_nx <= abs_nz:
            hint_x = Scalar[DTYPE](1)
            hint_y = Scalar[DTYPE](0)
            hint_z = Scalar[DTYPE](0)
        elif abs_ny <= abs_nz:
            hint_x = Scalar[DTYPE](0)
            hint_y = Scalar[DTYPE](1)
            hint_z = Scalar[DTYPE](0)
        else:
            hint_x = Scalar[DTYPE](0)
            hint_y = Scalar[DTYPE](0)
            hint_z = Scalar[DTYPE](1)

    # Gram-Schmidt against the normal.
    var dot_nh = nx * hint_x + ny * hint_y + nz * hint_z
    var t1x = hint_x - dot_nh * nx
    var t1y = hint_y - dot_nh * ny
    var t1z = hint_z - dot_nh * nz
    var t1_mag = sqrt(t1x * t1x + t1y * t1y + t1z * t1z)

    # Hint parallel to the normal — nothing survived; take the default axis.
    if t1_mag < Scalar[DTYPE](1e-10):
        var abs_nx = abs(nx)
        var abs_ny = abs(ny)
        var abs_nz = abs(nz)
        if abs_nx <= abs_ny and abs_nx <= abs_nz:
            hint_x = Scalar[DTYPE](1)
            hint_y = Scalar[DTYPE](0)
            hint_z = Scalar[DTYPE](0)
        elif abs_ny <= abs_nz:
            hint_x = Scalar[DTYPE](0)
            hint_y = Scalar[DTYPE](1)
            hint_z = Scalar[DTYPE](0)
        else:
            hint_x = Scalar[DTYPE](0)
            hint_y = Scalar[DTYPE](0)
            hint_z = Scalar[DTYPE](1)
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
