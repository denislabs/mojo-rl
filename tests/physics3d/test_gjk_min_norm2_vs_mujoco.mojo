"""GJK's convergence floor is a threshold on |v| SQUARED, so a "tiny" constant
is a LARGE distance — and a separated pair under it is reported as a contact.

    pixi run mojo run -I . tests/physics3d/test_gjk_min_norm2_vs_mujoco.mojo

WHAT WENT WRONG. The loop that ends GJK read

    if v_dot_v < GJK_TOLERANCE:      # GJK_TOLERANCE = 1e-10
        break

and the separated-vs-penetrating classification after it compared `dist_sq`
against the same constant. `v_dot_v` is |v| SQUARED, so 1e-10 is a distance
floor of `sqrt(1e-10)` = **1e-5 m = 10 microns**. Every convex pair separated
by less than 10 microns converged "to the origin", was classified PENETRATING,
and was handed to EPA — which returns `-0.0`. A real gap became an invented
contact, and an invented contact is a constraint row the solver cannot tell
from a true one.

⚠⚠ 1e-10 LOOKS LIKE A TIGHT TOLERANCE. It is not a tolerance on a distance, it
is a tolerance on a distance squared, and nothing at the call site says so.

MUJOCO DOES NOT USE A CONSTANT (`engine_collision_gjk.c:212-225`):

    mjtNum tol2      = status->tolerance * status->tolerance;
    mjtNum min_norm2 = discreteGeoms(obj1, obj2) ? mjMINVAL2 : tol2;
    ...
    if ((x_norm = dot3(x_k, x_k)) < min_norm2) break;

`tolerance` is `<option ccd_tolerance>`, default 1e-6, so the floor is 1e-6 m
for a SMOOTH pair and `mjMINVAL` = 1e-15 m for a POLYTOPE pair — 10x and 1e10x
tighter than the constant we had. It is the SAME `discreteGeoms` switch that
`_epa_tolerance` in `gjk.mojo` already implements; that one crossed to EPA and
never to GJK, and `_gjk_min_norm2` is the other half.

⚠ A NON-ZERO MARGIN MAKES A DISCRETE PAIR SMOOTH — MuJoCo's first line, and
load-bearing here: the duplo stud declares `margin=1e-4`, so stud-vs-flange is
a SMOOTH pair and its floor is `ccd_tolerance`, not `mjMINVAL`.

THE FIXTURE IS THE REAL ONE, AND ITS ANSWER IS EXACT. These are the `stud` and
`flange` geoms of `dm_control`'s `duplo2x4.xml` verbatim, in the configuration
`manipulation/reassemble_5_bricks_random_order` starts in. The cylinder's axis
is z and the box's nearest feature is its +x FACE, so the signed gap is

    (box_face_x - cyl_axis_x) - r

with NO approximation and no reference implementation in the loop. In the model
that is `0.00465 - 0.004647` = **3.0 microns** — a fifth of the old floor.
MuJoCo reports +3.02e-06 there; we reported -6.33e-06 on all 48 such pairs,
penetration where there is a gap.

⚠ THE CONTROL IS IN THE TABLE. The two widest gaps are asserted to be what
they already were: 2e-5 and 5e-5 sit ABOVE the old 1e-5 floor and are
bit-identical before and after the fix. A change that perturbed GJK globally
rather than under the floor would move them.

⚠⚠ THIS DOES NOT ASSERT THAT THE SEPARATED BRANCH IS ACCURATE, BECAUSE IT IS
NOT. Over 200 gaps in 5e-7..1e-4 the fixed code still exceeds 1e-6 of error on
153 of them (worst 5.2e-5, near gap 5.2e-5) and still returns `-0.0` at 8.5e-6
and 5.2e-5. That residual is a DIFFERENT mechanism — it is insensitive to
`ccd_tolerance` (1e-6..1e-14) and to the iteration cap (100..1000), and
disabling `_gjk_intersect` makes it worse, so it lives in the distance
subalgorithm. It is tracked as the open half of task #81; the `5.0e-05` row
below is left in the table with a loose bound precisely so this file records
it rather than hiding it.

⚠ EVERY OTHER COLLISION GATE PASSED WITH THE BUG IN. `test_gjk_simplex`,
`test_gjk_float32_no_phantom_contacts`, `test_within_margin_convex_contacts`,
`test_mesh_collision`, `test_mesh_manifold_vs_mujoco`, `test_narrow_phase_pairs`,
`test_epa_optimality_cylinder_mesh`, `test_box_box_sweep`,
`test_capsule_box_sweep`, `test_sawyer_mesh_rest_vs_mujoco` and
`test_mesh_polygons_vs_mujoco` are all GREEN both before and after. None of
them poses a convex pair separated by less than 10 microns.
"""
from std.math import abs
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.collision.gjk import gjk_epa
from mojo_rl.physics3d.constants import GEOM_BOX, GEOM_CYLINDER

comptime DT = DType.float64
comptime LV = Layout.row_major(1, 3)
comptime LA = Layout.row_major(1)

# `duplo2x4.xml`, class `stud`: cylinder, size ".0047 .0023", margin 1e-4. The
# radius is 0.004647 and NOT the 0.0047 in the file: dm_control's `Duplo`
# overwrites it in `initialize_episode_mjcf` from `_STUD_SIZE_PARAMS`, and with
# `variation=0` that is the lower quartile for (easy_align=False, flanges=True).
comptime R_STUD = 0.004647
comptime HL_STUD = 0.0023
# class `flange`: box, size ".0008 .00055 .0087".
comptime HX_FLANGE = 0.0008
comptime HY_FLANGE = 0.00055
comptime HZ_FLANGE = 0.0087
# z offset between the two in the assembled tower: the flange belongs to the
# brick above, whose origin sits 0.0192 higher.
comptime DZ = 0.00655
comptime MARGIN = 1e-4
comptime CCD_TOL = 1e-6
comptime CCD_ITER = 35


def probe(
    gap: Float64,
    mv: LayoutTensor[DT, LV, MutAnyOrigin],
    ma: LayoutTensor[DT, LA, MutAnyOrigin],
    me: LayoutTensor[DT, LA, MutAnyOrigin],
) -> Float64:
    """Signed distance our narrow phase reports for a stud/flange pair whose
    true face-to-surface gap is `gap`."""
    var dx = Scalar[DT](gap + HX_FLANGE + R_STUD)
    var res = gjk_epa[DT](
        GEOM_CYLINDER,
        Scalar[DT](0), Scalar[DT](0), Scalar[DT](0),
        Scalar[DT](0), Scalar[DT](0), Scalar[DT](0), Scalar[DT](1),
        Scalar[DT](R_STUD), Scalar[DT](HL_STUD),
        Scalar[DT](0), Scalar[DT](0), Scalar[DT](0),
        mv, ma, me, 0, 0,
        GEOM_BOX,
        dx, Scalar[DT](0), Scalar[DT](DZ),
        Scalar[DT](0), Scalar[DT](0), Scalar[DT](0), Scalar[DT](1),
        Scalar[DT](0), Scalar[DT](0),
        Scalar[DT](HX_FLANGE), Scalar[DT](HY_FLANGE), Scalar[DT](HZ_FLANGE),
        0, 0,
        Scalar[DT](CCD_TOL), CCD_ITER, Scalar[DT](MARGIN),
    )
    return Float64(res[0])


def main() raises:
    var mbuf = List[Scalar[DT]](length=3, fill=Scalar[DT](0))
    var mv = LayoutTensor[DT, LV, MutAnyOrigin](
        mbuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]()
    )
    var abuf = List[Scalar[DT]](length=1, fill=Scalar[DT](-1))
    var ma = LayoutTensor[DT, LA, MutAnyOrigin](
        abuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]()
    )
    var ebuf = List[Scalar[DT]](length=1, fill=Scalar[DT](-1))
    var me = LayoutTensor[DT, LA, MutAnyOrigin](
        ebuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]()
    )

    # (true gap, tolerance on |reported - true|). Every gap here is UNDER the
    # old 1e-5 floor except the last two, which are the control.
    var gaps = List[Float64]()
    var tols = List[Float64]()
    gaps.append(2.0e-06); tols.append(1e-6)
    gaps.append(3.0e-06); tols.append(1e-6)   # <- the model's own separation
    gaps.append(4.0e-06); tols.append(1e-6)
    gaps.append(5.0e-06); tols.append(1e-6)
    gaps.append(7.0e-06); tols.append(1e-6)
    gaps.append(1.0e-05); tols.append(1e-6)
    gaps.append(2.0e-05); tols.append(1e-9)   # control: above the old floor
    gaps.append(5.0e-05); tols.append(1e-5)   # control + open residual (#81)

    print("  true gap        reported          error    tol")
    var failures = 0
    for i in range(len(gaps)):
        var g = gaps[i]
        var got = probe(g, mv, ma, me)
        var err = got - g
        var ok = got > 0.0 and abs(err) <= tols[i]
        print(
            "  ", g, " ", got, " ", err, " ", tols[i], " ",
            "ok" if ok else "FAIL",
        )
        if not ok:
            failures += 1
            if got <= 0.0:
                print(
                    "     ^ a PHANTOM CONTACT: a pair", g,
                    "m apart reported as touching/penetrating",
                )

    _ = mbuf^
    _ = abuf^
    _ = ebuf^

    if failures != 0:
        raise Error(
            String(failures)
            + " of "
            + String(len(gaps))
            + " separations wrong — GJK's floor is a threshold on |v| SQUARED;"
            + " see `_gjk_min_norm2` in physics3d/collision/gjk.mojo"
        )
    print("PASS: all", len(gaps), "separations correct")
