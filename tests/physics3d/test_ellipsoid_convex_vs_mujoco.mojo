"""An ELLIPSOID that is not against a plane — vs MuJoCo 3.10.0.

    pixi run mojo run -I . tests/physics3d/test_ellipsoid_convex_vs_mujoco.mojo

Row ELLIPSOID of `mjCOLLISIONFUNC` (`engine_collision_driver.c:45`) is
`mjc_Convex` against ELLIPSOID, CYLINDER, BOX and MESH, and column ELLIPSOID is
`mjc_Convex` from SPHERE and CAPSULE down. Only `mjc_PlaneConvex` is separate.
This engine had ONLY the plane case, and everything else about an ellipsoid was
absent in three independent places, each of them silent:

  1. **`_support` had no ellipsoid branch**, and its fallback returns the
     geom's CENTRE for a type it does not know — so an ellipsoid entered
     GJK/EPA as a zero-radius DOT. No error, no warning, no contact.
  2. **the SAP broadphase's `_aabb_half_extents` had no ellipsoid branch**, and
     ITS fallback was `radius`, which for an ellipsoid is `size[0]` — the x
     semi-axis. flybody's labrum ellipsoids are `0.0035 0.00875 0.0131`, so
     their AABB came out **3.7x too small on z**.
  3. **the SAP AABB omitted the geoms' own `margin`.** MuJoCo's `filterBox`
     and `mj_filterSphere` are both called WITH the pair's margin; ours folded
     in only a `<contact><pair margin=>`. A pair separated by less than its
     margin but more than its extents never reached the narrow phase.

⚠⚠ ALL THREE HAD TO GO. Any one of them alone drops the contact, and the two
narrow phases disagree — the naive path (`ngeom < 16`) has no AABB stage at
all, so (2) and (3) are invisible below the threshold and (1) is not. This file
therefore runs the SAME fixture twice, once padded past `SAP_THRESHOLD`, and a
failure on one variant and not the other names the path.

⚠ MEASURED, flybody one step from its keyframe against MuJoCo 3.10.0: worst
|d(qpos)| **2.658e-03 -> 1.422e-04**, and the Menagerie board's count above
1e-3 goes 3 to 2. That scene had been filed against `<adhesion>`; most of its
residual was the labrum pair not colliding at all.
"""

from std.math import abs as math_abs, sqrt
from std.python import Python
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.studio.stepping import StudioIntegPyr
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, CONTACT_SIZE,
    CONTACT_IDX_POS_X, CONTACT_IDX_POS_Y, CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX, CONTACT_IDX_NY, CONTACT_IDX_NZ, CONTACT_IDX_DIST,
)

comptime DT = DType.float64

# ⚠ THE SIZES ARE NOT LARGEST-FIRST — `0.0035 0.00875 0.0131`, flybody's own —
# because `radius` is `size[0]` and an AABB built from it is only wrong when
# the x semi-axis is not the largest. Both geoms are ROTATED so the AABB
# formula is exercised rather than the axis-aligned special case, and the pair
# is SEPARATED by 1.3e-04 inside a margin of 1.0e-03 so the margin fold-in
# decides whether it is a candidate at all.
comptime PAIR = String(
    """
    <body name="a" pos="0 0 0.5" euler="0.3 -0.4 0.2">
      <freejoint/>
      <geom name="ea" type="ellipsoid" size="0.0035 0.00875 0.0131"
            margin="0.0005"/>
    </body>
    <body name="b" pos="0 0 0.516" euler="-0.2 0.5 0.1">
      <freejoint/>
      <geom name="eb" type="ellipsoid" size="0.0035 0.00875 0.0131"
            margin="0.0005"/>
    </body>"""
)

# ⚠ The margin fixture is AXIS-ALIGNED on purpose. A rotated SEPARATED pair
# goes through the GJK DISTANCE subalgorithm, whose residual is a known open
# item — measured here at 1.4e-04 on `dist` and 15 degrees on the normal,
# enough to report a 1.3e-04 gap as a 1.2e-05 penetration. That is not what
# this file is about, and pinning it here would either encode the residual as
# correct or leave a permanently red gate. Axis-aligned, the same query agrees
# with MuJoCo to 7e-08.
comptime PAIR_MARGIN = String(
    """
    <body name="a" pos="0 0 0.5">
      <freejoint/>
      <geom name="ea" type="ellipsoid" size="0.0035 0.00875 0.0131"
            margin="0.0005"/>
    </body>
    <body name="b" pos="0.0071 0 0.5">
      <freejoint/>
      <geom name="eb" type="ellipsoid" size="0.0035 0.00875 0.0131"
            margin="0.0005"/>
    </body>"""
)


def _pad(n: Int) -> String:
    """`n` far-away spheres — enough of them pushes `ngeom` over
    `SAP_THRESHOLD` (16) and `detect_contacts_auto` switches narrow phase."""
    var out = String("")
    for i in range(n):
        out += String(
            '\n    <body name="p', i, '" pos="', 2.0 + Float64(i) * 0.2,
            ' 0 0.5"><freejoint/><geom name="pg', i,
            '" type="sphere" size="0.01"/></body>',
        )
    return out^


def _xml(body: String, npad: Int) raises -> String:
    return String(
        '<mujoco>\n  <compiler angle="radian"/>\n'
        '  <option timestep="0.002" integrator="Euler"/>\n  <worldbody>',
        body, _pad(npad), "\n  </worldbody>\n</mujoco>",
    )


def _ours(xml: String) raises -> Tuple[List[Float64], List[Float64],
                                       List[Float64]]:
    """One step; returns (xyz per contact, dist per contact, normal per)."""
    var fmd = parse_xml_full(expand_mjcf(xml, String("")), String(""))
    var dims = dims_from_flat(fmd, max_contacts=32, nmesh_verts=0)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
        d.qfrc.data[i] = Scalar[DT](0)
    var integ = StudioIntegPyr(dims)
    integ.step["cpu"](d, m)

    var nc = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
    var pos = List[Float64]()
    var dst = List[Float64]()
    var nrm = List[Float64]()
    for k in range(nc):
        var o = k * CONTACT_SIZE
        pos.append(Float64(d.contacts.data[o + CONTACT_IDX_POS_X]))
        pos.append(Float64(d.contacts.data[o + CONTACT_IDX_POS_Y]))
        pos.append(Float64(d.contacts.data[o + CONTACT_IDX_POS_Z]))
        dst.append(Float64(d.contacts.data[o + CONTACT_IDX_DIST]))
        nrm.append(Float64(d.contacts.data[o + CONTACT_IDX_NX]))
        nrm.append(Float64(d.contacts.data[o + CONTACT_IDX_NY]))
        nrm.append(Float64(d.contacts.data[o + CONTACT_IDX_NZ]))
    return (pos^, dst^, nrm^)


def _check(
    body: String, npad: Int, label: String, tol_pos: Float64,
    tol_dist: Float64, tol_ang: Float64,
) raises:
    var xml = _xml(body, npad)
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(xml)
    var md = mujoco.MjData(m)
    _ = mujoco.mj_forward(m, md)
    var mj_n = Int(py=md.ncon)
    var ngeom = Int(py=m.ngeom)

    print("=== ", label, ": ngeom", ngeom, " MuJoCo ncon", mj_n)
    assert_true(
        mj_n == 1,
        "the fixture must give MuJoCo exactly ONE contact; got "
        + String(mj_n) + " — it changed and no longer measures anything",
    )
    var c = md.contact[0]
    assert_true(
        Int(py=c.exclude) == 0,
        "the fixture must not rely on an EXCLUDED contact (`dist >="
        " includemargin`) — those carry no constraint row",
    )
    var mj_d = Float64(py=c.dist)
    var mj_p = List[Float64]()
    var mj_n3 = List[Float64]()
    for k in range(3):
        mj_p.append(Float64(py=c.pos[k]))
        mj_n3.append(Float64(py=c.frame[k]))

    var got = _ours(xml)
    var pos = got[0].copy()
    var dst = got[1].copy()
    var nrm = got[2].copy()
    print("  mj   dist", mj_d, " pos", mj_p[0], mj_p[1], mj_p[2])
    for i in range(len(dst)):
        print(
            "  ours dist", dst[i],
            " pos", pos[3 * i], pos[3 * i + 1], pos[3 * i + 2],
        )
    assert_true(
        len(dst) == 1,
        String("ncon ") + String(len(dst)) + " != 1 on the " + label
        + " path. ZERO means the ellipsoid never reached `mjc_Convex` — the"
        " GJK support fallback (a point at the centre), the SAP AABB"
        " fallback (`size[0]`), or the SAP AABB's missing margin.",
    )

    var dp = sqrt(
        (pos[0] - mj_p[0]) * (pos[0] - mj_p[0])
        + (pos[1] - mj_p[1]) * (pos[1] - mj_p[1])
        + (pos[2] - mj_p[2]) * (pos[2] - mj_p[2])
    )
    var dd = math_abs(dst[0] - mj_d)
    # The record's normal is `body_b -> body_a`, MuJoCo's frame is
    # `geom1 -> geom2`, so compare the ANGLE and ignore the sign.
    var dot = (
        nrm[0] * mj_n3[0] + nrm[1] * mj_n3[1] + nrm[2] * mj_n3[2]
    )
    var adot = math_abs(dot)
    if adot > 1.0:
        adot = 1.0
    print("  |d(pos)|", dp, " |d(dist)|", dd, " 1-|dot(n)|", 1.0 - adot)
    # ⚠ NOT machine precision, and deliberately so: an ellipsoid has no flat
    # feature, so both engines are reading an EPA/GJK iterate rather than a
    # closed form. The tolerances are per-fixture and each one is measured.
    assert_true(
        dp < tol_pos and dd < tol_dist and (1.0 - adot) < tol_ang,
        String("the contact does not agree with MuJoCo: |d(pos)| ")
        + String(dp) + ", |d(dist)| " + String(dd) + ", 1-|dot(n)| "
        + String(1.0 - adot),
    )


def test_ellipsoid_pair_on_the_naive_path() raises:
    """`ngeom` 2 — below `SAP_THRESHOLD`, so no AABB stage at all.

    This variant isolates the GJK SUPPORT function: it is the only one of the
    three defects that can bite here. Penetrating, so the number being
    compared is EPA's and not the distance subalgorithm's.
    """
    _check(PAIR, 0, "naive, penetrating", 1e-04, 1e-04, 1e-03)


def test_ellipsoid_pair_on_the_sap_path() raises:
    """`ngeom` 22 — over `SAP_THRESHOLD`, so the AABB sweep runs.

    Failing here while the naive variant passes means the BROADPHASE dropped
    the pair, and with this fixture that is the ellipsoid AABB: its
    `0.0035 0.00875 0.0131` is not largest-first, so a bound built from
    `size[0]` is 3.7x too small on z.
    """
    _check(PAIR, 20, "sap, penetrating", 1e-04, 1e-04, 1e-03)


def test_a_within_margin_pair_survives_the_sap_sweep() raises:
    """SEPARATED by 1.0e-04 inside a margin of 1.0e-03, on the SAP path.

    ⚠ THIS ONE IS ABOUT THE MARGIN AND NOTHING ELSE. The two AABBs do not
    touch — they are 1.0e-04 apart — so the pair is a candidate only if each
    geom's own `margin` widens its box, which is what MuJoCo's `filterBox` and
    `mj_filterSphere` are called with and what this sweep used to omit.
    flybody's labrum pair is exactly this shape, and it had no contact at all.
    """
    _check(PAIR_MARGIN, 20, "sap, within margin", 1e-06, 1e-06, 1e-03)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
