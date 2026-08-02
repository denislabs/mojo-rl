"""The contact TANGENT FRAME against MuJoCo's own `contact.frame`.

`collision/contact_frame.mojo::contact_tangent_frame` is our `mju_makeFrame`
(engine_util_spatial.c:508). Every solver and every `cfrc_ext` consumer routes
through it, so a convention change there lands everywhere at once — and until
2026-08-03 the convention was wrong: the no-hint default axis was "the
least-aligned basis axis" where MuJoCo's is `(0,1,0)` unless `|n_y| >= 0.5`,
in which case `(0,0,1)`. For a floor contact that is a 90-degree rotation of
the tangent basis about the normal (task #25).

WHY NOTHING EVER CAUGHT IT, and why this file is the only thing that can.
With ISOTROPIC friction a rotation about the normal is unobservable: the
elliptic cone is a circle, the pyramid's four edges map onto themselves, and
the world-frame force is identical. `rne_post`/`cfrc_ext` sum t1 and t2 back
into a resultant, so they cannot see it either. And anisotropy cannot arise by
accident — a geom pair's contact friction is built as
`[fri0, fri0, fri1, fri2, fri2]` (engine_collision_driver.c:1483), so
slide1 == slide2 and roll1 == roll2 identically; only an explicit `<pair>`
with a five-component `friction=` breaks that, and there is none in the tree.
So no dynamical quantity discriminates. Only comparing the FRAME ITSELF does.

WHAT THE POSES COVER. Five sphere/sphere groups whose contact normals are
placed to exercise BOTH default branches and the `0.5` boundary between them
(MuJoCo's test is `frame[1] < 0.5 && frame[1] > -0.5`, so exactly ±0.5 takes
the z branch), two capsule pairs, and a tilted capsule on the floor PLANE —
the plane/capsule branches being the only ones that write a HINT, and so the
only route to the Gram-Schmidt path rather than the default. The boundary
itself is covered by a direct call, since no pose in the tree reaches it.

⚠ t1 IS SIGN-INVARIANT UNDER THE NORMAL, t2 IS NOT. `t1 = normalize(d - (n.d)n)`
is unchanged by `n -> -n`, and our normal points `body_b -> body_a` where
MuJoCo's points `geom1 -> geom2`, so t1 is compared directly and t2 against
`s * mj_t2` for the sign `s` that relates the two normals. For the HINT groups
t1 is compared only up to collinearity: both engines feed in the capsule axis,
but nothing fixes which end of it, so an exact match there would be asserting
a coincidence rather than the convention.

Run: pixi run mojo run -I . tests/physics3d/test_contact_frame_vs_mujoco.mojo
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from std.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.collision.contact_frame import contact_tangent_frame
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE,
    META_IDX_NUM_CONTACTS,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_FRAME_T1_X,
    CONTACT_IDX_FRAME_T1_Y,
    CONTACT_IDX_FRAME_T1_Z,
)

comptime DTYPE = DType.float64

# Each group is a colliding pair 1 m from its neighbours, penetrating 5 mm
# along the offset direction, so the contact normal IS that direction.
#
#   g0  offset +z              n_y = 0        -> default (0,1,0)
#   g1  offset +y              |n_y| = 1      -> default (0,0,1)
#   g2  offset +x              n_y = 0        -> default (0,1,0)
#   g3  offset (0,.707,.707)   |n_y| = .707   -> default (0,0,1)
#   g4  offset (0,.4,.9165)    |n_y| = .4     -> default (0,1,0), just inside
#   g5  capsule/capsule, perpendicular axes
#   g6  capsule/sphere
#   g7  TILTED capsule on the floor PLANE       -> the only HINT path
#
# g3/g4 straddle the 0.5 boundary deliberately: with the OLD "least-aligned
# axis" rule g4's default was x (|n_x| = 0 is smallest) where MuJoCo's is y,
# and g3's was x where MuJoCo's is z. Either would have passed a test that
# only looked at floor contacts.
#
# ⚠ ONLY THE PLANE/CAPSULE BRANCHES WRITE THE HINT. The module docstring says
# "the capsule narrow phases", which reads as any pair with a capsule in it;
# `collision/contact_detection.mojo` writes `CONTACT_IDX_FRAME_T1_*` at four
# sites and all four are plane/capsule (both geom orderings, both endpoints).
# g5 and g6 therefore take the DEFAULT axis despite containing capsules, and
# an earlier draft of this file asserted two hint groups and got zero. g7 is
# tilted so that only its lower cap reaches the floor — a level capsule gives
# TWO contacts and breaks the one-per-group matching below.
#
# ⚠ g5's capsules are PERPENDICULAR, not parallel. Parallel centrelines make
# the closest-point pair a segment rather than a point, and a pose sitting on
# that tie can disagree with MuJoCo in either engine's favour — the trap
# recorded as invariant 6 in docs/DM_CONTROL_PORT.md.
comptime FRAME_XML = """
<mujoco model="frames">
  <option timestep="0.002" gravity="0 0 0"/>
  <default>
    <geom friction="1 0.005 0.0001" solimp="0.9 0.95 0.001" solref="0.02 1"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 .1" pos="0 0 0"/>
    <body name="g0a" pos="0.0 0 0.5">
      <joint name="j0a" type="slide" axis="1 0 0"/>
      <geom name="c0a" type="sphere" size=".05"/>
    </body>
    <body name="g0b" pos="0.0 0 0.595">
      <joint name="j0b" type="slide" axis="1 0 0"/>
      <geom name="c0b" type="sphere" size=".05"/>
    </body>

    <body name="g1a" pos="1.0 0 0.5">
      <joint name="j1a" type="slide" axis="1 0 0"/>
      <geom name="c1a" type="sphere" size=".05"/>
    </body>
    <body name="g1b" pos="1.0 0.095 0.5">
      <joint name="j1b" type="slide" axis="1 0 0"/>
      <geom name="c1b" type="sphere" size=".05"/>
    </body>

    <body name="g2a" pos="2.0 0 0.5">
      <joint name="j2a" type="slide" axis="0 1 0"/>
      <geom name="c2a" type="sphere" size=".05"/>
    </body>
    <body name="g2b" pos="2.095 0 0.5">
      <joint name="j2b" type="slide" axis="0 1 0"/>
      <geom name="c2b" type="sphere" size=".05"/>
    </body>

    <body name="g3a" pos="3.0 0 0.5">
      <joint name="j3a" type="slide" axis="1 0 0"/>
      <geom name="c3a" type="sphere" size=".05"/>
    </body>
    <body name="g3b" pos="3.0 0.06717514421271702 0.56717514421271702">
      <joint name="j3b" type="slide" axis="1 0 0"/>
      <geom name="c3b" type="sphere" size=".05"/>
    </body>

    <body name="g4a" pos="4.0 0 0.5">
      <joint name="j4a" type="slide" axis="1 0 0"/>
      <geom name="c4a" type="sphere" size=".05"/>
    </body>
    <body name="g4b" pos="4.0 0.038 0.5870689382041610">
      <joint name="j4b" type="slide" axis="1 0 0"/>
      <geom name="c4b" type="sphere" size=".05"/>
    </body>

    <body name="g5a" pos="5.0 0 0.5">
      <joint name="j5a" type="slide" axis="1 0 0"/>
      <geom name="c5a" type="capsule" size=".04" fromto="0 -.06 0 0 .06 0"/>
    </body>
    <body name="g5b" pos="5.0 0 0.575">
      <joint name="j5b" type="slide" axis="0 1 0"/>
      <geom name="c5b" type="capsule" size=".04" fromto="-.06 0 0 .06 0 0"/>
    </body>

    <body name="g6a" pos="6.0 0 0.5">
      <joint name="j6a" type="slide" axis="1 0 0"/>
      <geom name="c6a" type="capsule" size=".04" fromto="0 -.06 0 0 .06 0"/>
    </body>
    <body name="g6b" pos="6.0 0 0.585">
      <joint name="j6b" type="slide" axis="1 0 0"/>
      <geom name="c6b" type="sphere" size=".05"/>
    </body>

    <body name="g7a" pos="7.0 0 0.095">
      <joint name="j7a" type="slide" axis="1 0 0"/>
      <geom name="c7a" type="capsule" size=".04" fromto="-.06 0 -.06 .06 0 .06"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime fp = parse_xml(FRAME_XML)
comptime FM = ModelDefFromXML[
    xml=FRAME_XML,
    nbody=fp.NBODY, njoint=fp.NJOINT, nq=fp.NQ, nv=fp.NV,
    ngeom=fp.NGEOM, nact=fp.NACT, ntex=fp.NTEX, nmat=fp.NMAT,
    nlight=fp.NLIGHT, ncam=fp.NCAM, nsite=fp.NSITE,
    max_tendon=fp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=32,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=fp.TIMESTEP,
]

comptime NGROUPS: Int = 8
comptime N_HINT_GROUPS: Int = 1  # g7 — plane/capsule is the only writer
comptime TOL_FRAME: Float64 = 1e-12
# `t1` is a function of the NORMAL, so it can be no more exact than the normal
# it is built from. When this file was written the two capsule-first groups
# arrived with a normal ~1e-10 off MuJoCo's while every other group was exact,
# so each group's bound was `max(TOL_FRAME, 8 * its own measured normal
# error)` — a measured allowance rather than a constant picked to make a
# failure go away — and the note here said "#49 closing will show up as the
# slack going unused".
#
# #49 CLOSED 2026-08-03 and the slack went unused, so it is gone. The cause
# was never the narrow phase: `quat_math.mojo` normalized quaternions as
# `1/sqrt(norm_sq + 1e-10)`, leaving every body quaternion 5e-11 short of
# unit, so a capsule's world AXIS (its local quaternion rotated by the body's)
# was not unit either. Spheres have no axis, which is the whole reason it
# looked like a capsule-specific narrow-phase defect. Now every group is
# exact: worst normal error 4.44e-16, worst t1 4.44e-16, worst t2 1.11e-16.
comptime TOL_NORMAL: Float64 = 1e-14

comptime Dat = Data[DTYPE, FM.NQ, FM.NV, FM.NBODY, FM.MAX_CONTACTS, FM.NSITE, 1]
comptime Mod = Model[
    DTYPE, FM.NV, FM.NBODY, FM.NJOINT, FM.NGEOM, FM.MAX_EQUALITY,
    FM.MAX_TENDON, FM.NSITE, FM.NEXCLUDE, 0,
]


def test_default_axis_matches_mju_makeFrame() raises:
    """The no-hint branch alone, including the ±0.5 boundary.

    Called directly rather than through a model, because the boundary itself
    is not reachable by any pose in the tree and a rule that is only ever
    tested at `n_y = 0` and `n_y = 1` is not tested at all.
    """
    print("--- contact frame: the default axis rule ---")
    # (n, expected t1) with no hint. Expected values are `mju_makeFrame`'s
    # (0,1,0)/(0,0,1) choice, Gram-Schmidted against n and normalized.
    # n = +z: |n_y| = 0 -> y branch, already orthogonal.
    var f = contact_tangent_frame[DTYPE](0, 0, 1, 0, 0, 0)
    assert_true(
        abs(Float64(f[0])) < TOL_FRAME and abs(Float64(f[1]) - 1) < TOL_FRAME
        and abs(Float64(f[2])) < TOL_FRAME,
        "n=+z must give t1=(0,1,0); the old rule gave (1,0,0)",
    )
    # n = -z: the rule is symmetric in the normal's sign.
    var fm = contact_tangent_frame[DTYPE](0, 0, -1, 0, 0, 0)
    assert_true(
        abs(Float64(fm[0])) < TOL_FRAME and abs(Float64(fm[1]) - 1) < TOL_FRAME
        and abs(Float64(fm[2])) < TOL_FRAME,
        "n=-z must give the same t1 as n=+z",
    )
    # n = +y: |n_y| = 1 >= 0.5 -> z branch.
    var fy = contact_tangent_frame[DTYPE](0, 1, 0, 0, 0, 0)
    assert_true(
        abs(Float64(fy[0])) < TOL_FRAME and abs(Float64(fy[1])) < TOL_FRAME
        and abs(Float64(fy[2]) - 1) < TOL_FRAME,
        "n=+y must give t1=(0,0,1)",
    )
    # n = +x: |n_y| = 0 -> y branch.
    var fx = contact_tangent_frame[DTYPE](1, 0, 0, 0, 0, 0)
    assert_true(
        abs(Float64(fx[0])) < TOL_FRAME and abs(Float64(fx[1]) - 1) < TOL_FRAME
        and abs(Float64(fx[2])) < TOL_FRAME,
        "n=+x must give t1=(0,1,0); the old rule gave (0,0,1) here",
    )
    # THE BOUNDARY. MuJoCo's test is `frame[1] < 0.5 && frame[1] > -0.5`, so
    # n_y = 0.5 EXACTLY takes the z branch and 0.5-eps takes the y branch.
    var s = sqrt(0.75)  # n = (0, .5, sqrt(.75)) is a unit vector
    var f_at = contact_tangent_frame[DTYPE](0, 0.5, Scalar[DTYPE](s), 0, 0, 0)
    # z branch: t1 = normalize((0,0,1) - n_z*n) -> the in-plane direction with
    # a POSITIVE z component and a negative y component.
    assert_true(
        Float64(f_at[1]) < 0 and Float64(f_at[2]) > 0,
        "n_y = 0.5 exactly must take the (0,0,1) branch — MuJoCo's test is"
        " strict on both sides",
    )
    var f_below = contact_tangent_frame[DTYPE](
        0, Scalar[DTYPE](0.5 - 1e-9), Scalar[DTYPE](s), 0, 0, 0
    )
    assert_true(
        Float64(f_below[1]) > 0,
        "n_y just below 0.5 must take the (0,1,0) branch",
    )
    print("  PASS: y branch, z branch, and the 0.5 boundary on both sides")


def _build() raises -> Mod:
    var ctx = DeviceContext()
    var mf = Mod()
    FM.init_fields[DTYPE, 0](ctx, mf)
    return mf^


def test_contact_frame_matches_mujoco() raises:
    """Our (t1, t2) against MuJoCo's `contact.frame` on live contacts."""
    print("--- contact frame vs MuJoCo ---")
    var mf = _build()
    var d = Dat()
    FM.reset_data(d)
    forward_kinematics["cpu"](d, mf)
    detect_contacts["cpu"](d, mf)

    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(String(FRAME_XML))
    var dat = mujoco.MjData(m)
    mujoco.mj_forward(m, dat)

    var n_ours = Int(d.meta.data[META_IDX_NUM_CONTACTS])
    var n_mj = Int(py=dat.ncon)
    print("  contacts: ours", n_ours, " MuJoCo", n_mj)
    assert_true(
        n_ours == NGROUPS and n_mj == NGROUPS,
        String("expected one contact per group (") + String(NGROUPS)
        + ") but got ours=" + String(n_ours) + " MuJoCo=" + String(n_mj)
        + " — every pair here is single-point, so a different count means the"
        " model changed and the group->contact matching below is invalid",
    )

    # MuJoCo's contacts, indexed by group (x position rounds to the group id).
    var mj_group = List[Int]()
    for _ in range(NGROUPS):
        mj_group.append(-1)
    for c in range(n_mj):
        var con = dat.contact[c]
        var gx = Float64(py=con.pos[0])
        var g = Int(gx + 0.5)
        assert_true(
            g >= 0 and g < NGROUPS and abs(gx - Float64(g)) < 0.3,
            String("MuJoCo contact ") + String(c) + " at x=" + String(gx)
            + " is not inside any group",
        )
        assert_true(
            mj_group[g] == -1,
            String("two MuJoCo contacts in group ") + String(g),
        )
        mj_group[g] = c

    var n_hint_seen = 0
    var n_default_seen = 0
    var worst_t1 = Float64(0)
    var worst_t2 = Float64(0)
    var worst_n = Float64(0)
    for c in range(n_ours):
        var b = c * CONTACT_SIZE
        var gx = Float64(d.contacts.data[b + CONTACT_IDX_POS_X])
        var g = Int(gx + 0.5)
        assert_true(
            g >= 0 and g < NGROUPS and abs(gx - Float64(g)) < 0.3,
            String("our contact ") + String(c) + " at x=" + String(gx)
            + " is not inside any group",
        )
        var mc = mj_group[g]
        assert_true(mc >= 0, String("no MuJoCo contact in group ") + String(g))
        var con = dat.contact[mc]

        var nx = Float64(d.contacts.data[b + CONTACT_IDX_NX])
        var ny = Float64(d.contacts.data[b + CONTACT_IDX_NY])
        var nz = Float64(d.contacts.data[b + CONTACT_IDX_NZ])
        var hx = Float64(d.contacts.data[b + CONTACT_IDX_FRAME_T1_X])
        var hy = Float64(d.contacts.data[b + CONTACT_IDX_FRAME_T1_Y])
        var hz = Float64(d.contacts.data[b + CONTACT_IDX_FRAME_T1_Z])
        var has_hint = (hx * hx + hy * hy + hz * hz) >= 0.25

        var mnx = Float64(py=con.frame[0])
        var mny = Float64(py=con.frame[1])
        var mnz = Float64(py=con.frame[2])
        # Our normal runs body_b -> body_a, MuJoCo's geom1 -> geom2, so the
        # two agree up to a sign this file does not try to predict — that is
        # test_narrow_phase_pairs' job. Here it only fixes t2's sign.
        var dot_n = nx * mnx + ny * mny + nz * mnz
        assert_true(
            abs(abs(dot_n) - 1.0) < TOL_NORMAL,
            String("group ") + String(g) + ": our normal is not collinear"
            " with MuJoCo's, so the frames are not comparable",
        )
        var s = 1.0 if dot_n > 0 else -1.0
        # How far the INPUT is off, which bounds how exact the output can be.
        var n_err = max(
            abs(nx - s * mnx), max(abs(ny - s * mny), abs(nz - s * mnz))
        )
        worst_n = max(worst_n, n_err)
        assert_true(
            n_err < TOL_NORMAL,
            String("group ") + String(g) + ": normal is " + String(n_err)
            + " off MuJoCo's — a narrow-phase regression, not a frame one",
        )
        print("    g", g, " hint", has_hint, " normal err", n_err)

        var f = contact_tangent_frame[DTYPE](
            Scalar[DTYPE](nx), Scalar[DTYPE](ny), Scalar[DTYPE](nz),
            Scalar[DTYPE](hx), Scalar[DTYPE](hy), Scalar[DTYPE](hz),
        )
        var t1x = Float64(f[0])
        var t1y = Float64(f[1])
        var t1z = Float64(f[2])
        var t2x = Float64(f[3])
        var t2y = Float64(f[4])
        var t2z = Float64(f[5])

        var m1x = Float64(py=con.frame[3])
        var m1y = Float64(py=con.frame[4])
        var m1z = Float64(py=con.frame[5])
        var m2x = Float64(py=con.frame[6])
        var m2y = Float64(py=con.frame[7])
        var m2z = Float64(py=con.frame[8])

        if has_hint:
            n_hint_seen += 1
            # Both engines seed from the capsule AXIS and nothing fixes which
            # end of it, so collinearity is the whole claim here.
            var dt1 = t1x * m1x + t1y * m1y + t1z * m1z
            var e1 = abs(abs(dt1) - 1.0)
            worst_t1 = max(worst_t1, e1)
            assert_true(
                e1 < TOL_FRAME,
                String("group ") + String(g) + " (hint): our t1 is not"
                " collinear with MuJoCo's — Gram-Schmidt or the hint differs",
            )
            var dt2 = t2x * m2x + t2y * m2y + t2z * m2z
            var e2 = abs(abs(dt2) - 1.0)
            worst_t2 = max(worst_t2, e2)
            assert_true(
                e2 < TOL_FRAME,
                String("group ") + String(g) + " (hint): t2 is not collinear"
                " with MuJoCo's",
            )
        else:
            n_default_seen += 1
            # `t1 = normalize(d - (n.d)n)` is invariant under `n -> -n`, so
            # this is an EXACT comparison, not a collinear one.
            var e1 = max(
                abs(t1x - m1x), max(abs(t1y - m1y), abs(t1z - m1z))
            )
            worst_t1 = max(worst_t1, e1)
            assert_true(
                e1 < TOL_FRAME,
                String("group ") + String(g) + ": t1 = (" + String(t1x) + ", "
                + String(t1y) + ", " + String(t1z) + ") but MuJoCo has ("
                + String(m1x) + ", " + String(m1y) + ", " + String(m1z) + ")",
            )
            # t2 = n x t1 flips with the normal.
            var e2 = max(
                abs(t2x - s * m2x),
                max(abs(t2y - s * m2y), abs(t2z - s * m2z)),
            )
            worst_t2 = max(worst_t2, e2)
            assert_true(
                e2 < TOL_FRAME,
                String("group ") + String(g) + ": t2 disagrees with MuJoCo's"
                " once the normal sign is accounted for",
            )

    assert_true(
        n_hint_seen == N_HINT_GROUPS,
        String("only ") + String(n_hint_seen) + " contacts carried a hint —"
        " the Gram-Schmidt path is no longer covered",
    )
    assert_true(
        n_default_seen == NGROUPS - N_HINT_GROUPS,
        String("only ") + String(n_default_seen) + " contacts took the default"
        " axis — the branch this file exists for is no longer covered",
    )
    print("  worst t1 err =", worst_t1, " worst t2 err =", worst_t2,
          " worst normal err =", worst_n,
          " (", n_default_seen, "default,", n_hint_seen, "hint )")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
