"""Cylinder pairs go to the convex query, not to a capsule in disguise.

    pixi run mojo run -I . tests/physics3d/test_cylinder_pairs_vs_mujoco.mojo

WHAT WENT WRONG. `cylinder_capsule` and `cylinder_cylinder` compute

    dist = axis_to_axis_distance - r1 - r2

which is the CAPSULE-capsule formula. It rounds the cylinder's flat end caps
into hemispheres, so the modelled surface bulges a full radius past where the
cylinder actually ends. The error is exactly `-r` in EVERY configuration,
separated or penetrating — the same defect `cylinder_box` had, documented in
`contact_detection.mojo` and fixed there by routing to `gjk_epa`, but never
carried across to the other two.

⚠ MuJoCo'S OWN TABLE IS THE SPECIFICATION. `mjCOLLISIONFUNC`
(`engine_collision_driver.c:47-56`) gives CYLINDER exactly two primitives —
`mjc_SphereCylinder` and `mjc_PlaneCylinder`. Every other cylinder pair,
including CAPSULE x CYLINDER and CYLINDER x CYLINDER, is `mjc_Convex`.

MEASURED on Menagerie's sharpa_wave, a hand with 5 cylinders and 15 capsules,
at its reference pose:

    contacts at qpos0        : 4 (up to 6 mm deep)  ->  0   (MuJoCo: 0)
    100 steps, max |qpos-MJ| : 3.869e-01            ->  1.231e-03

⚠ THE SPURIOUS CONTACTS WERE BETWEEN FINGERS THAT NEVER TOUCH. MuJoCo's own
`mj_geomDistance` puts the closest collidable pair at +0.004115 m — SEPARATED —
where we reported -0.006069 m of penetration. Nothing was wrong with the
filtering (the pairs are neither excluded nor parent-child), nothing was wrong
with the poses (FK matched to 3e-17); the narrow phase was answering a
question about a different shape.

⚠ AND IT MOVED unitree_h1 ONTO MuJoCo'S NUMBER. h1 has 6 cylinders and 20
capsules; its 300-step zmax was 1.059960760768038 against MuJoCo's
1.059960760000 and is now exact. A 7.7e-10 discrepancy that turned out to be
this bug, quietly, on a model nobody suspected.

⚠ THREE COPIES OF THE DISPATCH TABLE HAD TO MOVE TOGETHER:
`collision/contact_detection.mojo`, `collision/broadphase_sap.mojo` and
`collision/multi_ccd.mojo`. The last one's docstring already said so — "when a
branch here changes, this file changes with it" — because its four perturbed
manifold extensions were built from the same capsule reduction, and no gate
could see it since MuJoCo copies `con[0].dist` onto every extra row.
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.studio.stepping import StudioIntegPyr
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, CONTACT_SIZE, CONTACT_IDX_DIST,
)

comptime DT = DType.float64
comptime SHARPA = String(
    "references/mujoco_menagerie-main/sharpa_wave/scene_left.xml"
)


# A capsule END-ON above a cylinder's FLAT TOP FACE — the one configuration
# where rounding the cap changes the answer, and where the error is exactly
# the cylinder's radius. Gravity off so the pose under test is the pose
# written down.
#
# ⚠ THE CAPSULE NEEDS A `<freejoint/>` AND THAT IS NOT DECORATION. Two bodies
# with no joints are both WELDED TO THE WORLD, and MuJoCo filters a pair whose
# bodies share a weld id — so a fixture without it reports ncon 0 in both
# engines, for a reason that has nothing to do with the geometry under test.
# The negative control below is what caught that; `mj_geomDistance` does not
# apply the filter, so probing distances alone would not have.
#
# Geometry: the cylinder spans z in [-0.1, +0.1] with r = 0.05. The capsule's
# lowest point is at `z - 0.12`, so the true gap is `z - 0.22`, and MuJoCo's
# `mj_geomDistance` returns exactly that (verified with the freejoint present:
# ncon 0 at z = 0.24, and dist 0.0 / -0.01 / -0.02 at z = 0.22 / 0.21 / 0.20).
#
# The capsule reduction instead computes `(z - 0.2) - 0.05 - 0.02 = z - 0.27`,
# i.e. `-0.05 = -cyl_r` too deep, everywhere.
def _xml(zc: String) -> String:
    return String(
        "<mujoco><option gravity='0 0 0'/><worldbody>"
        "<body><geom type='cylinder' size='0.05 0.1' mass='1'/></body>"
        "<body pos='0 0 "
    ) + zc + String(
        "'><freejoint/><geom type='capsule' size='0.02 0.1' mass='1'/></body>"
        "</worldbody></mujoco>"
    )


def _probe(xml: String) raises -> Tuple[Int, Float64]:
    """(ncon, first contact dist) after one step."""
    var fmd = parse_xml_full(xml, String(""))
    var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=0)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
    var integ = StudioIntegPyr(dims)
    integ.step["cpu"](d, m)
    var nc = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
    var dist = 0.0
    if nc > 0:
        dist = Float64(d.contacts.data[CONTACT_IDX_DIST])
    return (nc, dist)


def test_capsule_over_a_cylinder_flat_face() raises:
    """The configuration the capsule reduction gets wrong by exactly -r."""
    print("=== capsule end-on above a cylinder's flat face ===")
    # ⚠ THE SEPARATED CASE IS THE ONE THAT NAMES THE BUG. MuJoCo puts these
    # 2 cm APART; the reduction reports 3 cm of penetration, so the model
    # manufactures a contact out of clear air.
    var far = _probe(_xml(String("0.24")))
    print("  z=0.24 (MuJoCo: 0.02 m APART) -> ncon", far[0])
    assert_true(
        far[0] == 0,
        "a capsule 2 cm above a cylinder's flat face must not touch it; we"
        " report " + String(far[0]) + " contact(s) at dist "
        + String(far[1])
        + ". The capsule reduction rounds the cylinder's end cap into a"
        " hemisphere, which reaches a full radius (0.05) too far.",
    )
    # ⚠ AND THE NEGATIVE CONTROL: a real overlap must still be found, at the
    # right depth. Without this, "return no contacts" would pass the row above.
    var near = _probe(_xml(String("0.21")))
    print("  z=0.21 (MuJoCo: 0.01 m DEEP)  -> ncon", near[0],
          " dist", near[1])
    assert_true(
        near[0] > 0,
        "a capsule overlapping the cylinder by 1 cm must produce a contact;"
        " we found none",
    )
    assert_true(
        abs(near[1] - (-0.01)) < 1e-6,
        "the penetration depth is " + String(near[1])
        + " where the geometry (and MuJoCo's mj_geomDistance) says -0.01."
        " An answer near -0.06 is -0.01 minus the cylinder radius, i.e. the"
        " capsule reduction.",
    )
    print("  PASS")


def test_sharpa_wave_has_no_contacts_at_its_reference_pose() raises:
    """The model it was found on. MuJoCo reports ncon 0 there.

    ⚠ `ncon == 0` IS THE WHOLE ASSERTION AND IT IS EXACT, not a tolerance.
    Fingers that do not touch must produce nothing; the closest collidable
    pair is +0.004115 m apart by MuJoCo's own `mj_geomDistance`.
    """
    print("=== sharpa_wave: no contacts at qpos0 ===")
    var src = read_model_source(SHARPA)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var verts = 262144
    var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    var tries = 0
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") == -1 or tries > 24:
                raise e
            tries += 1
            verts = verts * 2
            dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
    var integ = StudioIntegPyr(dims)
    integ.step["cpu"](d, m)
    var nc = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
    var deepest = 0.0
    for k in range(nc):
        var dd = Float64(d.contacts.data[k * CONTACT_SIZE + CONTACT_IDX_DIST])
        if dd < deepest:
            deepest = dd
    print("  ncon", nc, " (MuJoCo 0)  deepest", deepest)
    assert_true(
        nc == 0,
        "sharpa_wave has " + String(nc)
        + " contact(s) at its reference pose, deepest " + String(deepest)
        + " m. MuJoCo has none: no two collidable geoms are closer than"
        " +0.004115 m. Unfixed this was 4 contacts up to 6 mm deep, between"
        " fingers that never touch.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
