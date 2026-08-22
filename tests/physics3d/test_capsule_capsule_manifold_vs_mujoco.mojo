"""Capsule/capsule is a MANIFOLD when the axes are parallel — vs MuJoCo 3.10.0.

    pixi run mojo run -I . \
        tests/physics3d/test_capsule_capsule_manifold_vs_mujoco.mojo

`mjraw_CapsuleCapsule` (`engine_collision_primitive.c:426`) has two branches:

  * **non-parallel** (`|det| >= mjMINVAL`) — one closest-point pair, clipped to
    both segments, handed to `mjraw_SphereSphere`. ONE contact.
  * **parallel** — the closest-point pair is not unique, and MuJoCo walks FOUR
    candidate ends (`x1 = +1`, `x1 = -1`, `x2 = +1`, `x2 = -1`), each paired
    with the other segment's clipped projection, stopping as soon as TWO come
    back within `margin`.

⚠⚠ WHY THIS NEEDS A GATE OF ITS OWN. One point is a perfectly reasonable
answer to "where do these two capsules touch", it is what a closest-point query
returns, and every number in it agrees with MuJoCo's first point. What it
cannot express is that two parallel capsules resting on each other do not
PIVOT — the same defect a box on one contact point has, and the reason
`box_capsule_manifold` exists. The contact-set sweep found it on `i2rt_yam`,
whose two finger capsules are exactly parallel and exactly touching: MuJoCo
reports 2 contacts there and this engine reported 1.

⚠ The degenerate-normal rule is gated here too, by the second pair. When the
two centrelines CROSS, the closest points coincide, the separation direction is
undefined, and `mjraw_SphereSphere` takes the CROSS PRODUCT OF THE TWO z AXES
(and `mju_normalize3`'s `(1,0,0)` when that is zero as well). Getting it from
`centre_B - centre_A` instead is a documented past defect of this file — see
`capsule_capsule`'s own comment.

⚠ THE STEP COUNTS DIFFER ON PURPOSE. MuJoCo's `d.contact` after `mj_forward`
is detection at the CURRENT pose; ours after one step is detection at the pose
that step STARTED from. Both therefore look at the fixture's rest pose.
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
from mojo_rl.physics3d.collision.collision_primitives import (
    capsule_capsule_manifold, CC_MAX_POINTS,
)

comptime DT = DType.float64

# `ca`/`cb` are PARALLEL, overlap by 0.02, and are offset along their own axis
# so that both of MuJoCo's clips do work. `cc`/`ce` CROSS at 0.6 rad, which is
# the non-parallel branch AND the coincident-centres normal rule.
comptime XML = String(
    """<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="Euler"/>
  <worldbody>
    <body name="a" pos="0 0 0.5">
      <freejoint/>
      <geom name="ca" type="capsule" fromto="-0.2 0 0  0.2 0 0" size="0.05"/>
    </body>
    <body name="b" pos="0 0.08 0.5">
      <freejoint/>
      <geom name="cb" type="capsule" fromto="-0.15 0 0  0.25 0 0" size="0.05"/>
    </body>
    <body name="c" pos="0 -0.5 0.5" euler="0 0 0.6">
      <freejoint/>
      <geom name="cc" type="capsule" fromto="-0.2 0 0  0.2 0 0" size="0.05"/>
    </body>
    <body name="e" pos="0 -0.42 0.5">
      <freejoint/>
      <geom name="ce" type="capsule" fromto="-0.2 0 0  0.2 0 0" size="0.05"/>
    </body>
  </worldbody>
</mujoco>"""
)


def _ours() raises -> Tuple[List[Float64], List[Float64]]:
    """One step of the fixture; returns (flat xyz per contact, dist per)."""
    var fmd = parse_xml_full(expand_mjcf(XML, String("")), String(""))
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
    for k in range(nc):
        var o = k * CONTACT_SIZE
        pos.append(Float64(d.contacts.data[o + CONTACT_IDX_POS_X]))
        pos.append(Float64(d.contacts.data[o + CONTACT_IDX_POS_Y]))
        pos.append(Float64(d.contacts.data[o + CONTACT_IDX_POS_Z]))
        dst.append(Float64(d.contacts.data[o + CONTACT_IDX_DIST]))
    return (pos^, dst^)


def test_parallel_capsules_give_two_contacts() raises:
    """Ours against MuJoCo compiling the SAME string, contact by contact."""
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(XML)
    var md = mujoco.MjData(m)
    _ = mujoco.mj_forward(m, md)
    var mj_n = Int(py=md.ncon)

    var mj_pos = List[Float64]()
    var mj_dist = List[Float64]()
    for i in range(mj_n):
        var c = md.contact[i]
        # ⚠ An EXCLUDED contact (`dist >= includemargin`) carries no
        # constraint row; the fixture has none, and this is the check that
        # says so rather than an assumption.
        assert_true(
            Int(py=c.exclude) == 0,
            "the fixture must not rely on an excluded contact",
        )
        for k in range(3):
            mj_pos.append(Float64(py=c.pos[k]))
        mj_dist.append(Float64(py=c.dist))

    var got = _ours()
    var pos = got[0].copy()
    var dst = got[1].copy()
    var n = len(dst)

    print("=== capsule/capsule fixture: MuJoCo", mj_n, " ours", n)
    for i in range(mj_n):
        print(
            "  mj  ", mj_pos[3 * i], mj_pos[3 * i + 1], mj_pos[3 * i + 2],
            " dist", mj_dist[i],
        )
    for i in range(n):
        print(
            "  ours", pos[3 * i], pos[3 * i + 1], pos[3 * i + 2],
            " dist", dst[i],
        )

    assert_true(
        mj_n == 3,
        "MuJoCo must see 3 contacts on this fixture (2 from the PARALLEL pair"
        " + 1 from the crossing pair); got " + String(mj_n)
        + " — the fixture changed and the gate no longer measures the"
        " manifold",
    )
    assert_true(
        n == mj_n,
        "ncon " + String(n) + " != MuJoCo " + String(mj_n)
        + ". TWO here means the parallel pair produced ONE point, which is"
        " what a plain closest-point query gives and what this engine did"
        " before `capsule_capsule_manifold`.",
    )

    # Match by nearest position — the two broadphases order contacts
    # differently and comparing index i to index i reports garbage.
    var used = List[Bool](length=n, fill=False)
    var worst_p = 0.0
    var worst_d = 0.0
    for i in range(mj_n):
        var best = -1
        var bestr = 1e30
        for j in range(n):
            if used[j]:
                continue
            var dx = pos[3 * j + 0] - mj_pos[3 * i + 0]
            var dy = pos[3 * j + 1] - mj_pos[3 * i + 1]
            var dz = pos[3 * j + 2] - mj_pos[3 * i + 2]
            var r = sqrt(dx * dx + dy * dy + dz * dz)
            if r < bestr:
                bestr = r
                best = j
        assert_true(best >= 0, "no contact left to match")
        used[best] = True
        if bestr > worst_p:
            worst_p = bestr
        var dd = math_abs(dst[best] - mj_dist[i])
        if dd > worst_d:
            worst_d = dd
    print("  worst |d(pos)|", worst_p, " worst |d(dist)|", worst_d)
    assert_true(
        worst_p < 1e-12 and worst_d < 1e-12,
        "matched contacts must agree with MuJoCo to machine precision;"
        " worst |d(pos)| = " + String(worst_p)
        + ", worst |d(dist)| = " + String(worst_d),
    )


def test_the_manifold_primitive_directly() raises:
    """`i2rt_yam`'s own pair, at the record level.

    Two capsules of radius 0.01 and half-length 0.02, parallel, 0.02 apart —
    i.e. EXACTLY touching. ⚠ This one is asserted on the primitive rather than
    through the engine because `dist` lands on 0 to within 4e-17 and the
    `dist >= margin` cut then reads FK rounding rather than geometry: MuJoCo's
    own value there is -4.163e-17 and ours is +0.0. The two points and their
    positions are not in doubt, and those are what this checks.
    """
    var d1 = InlineArray[Scalar[DT], CC_MAX_POINTS](fill=Scalar[DT](0))
    var p1 = InlineArray[Scalar[DT], 3 * CC_MAX_POINTS](fill=Scalar[DT](0))
    var n1 = InlineArray[Scalar[DT], 3 * CC_MAX_POINTS](fill=Scalar[DT](0))
    # yam's z axis (-0.7241379..., 0, 0.6896551...) as a rotation about -y.
    var c = Float64(0.689655172413793)
    var qw = sqrt((1.0 + c) * 0.5)
    var qy = -sqrt((1.0 - c) * 0.5)
    var n = capsule_capsule_manifold[DT](
        0.33475484, 0.01, 0.37260463, 0.0, qy, 0.0, qw, 0.02, 0.01,
        0.33475484, -0.01, 0.37260463, 0.0, qy, 0.0, qw, 0.02, 0.01,
        Scalar[DT](0.0), d1, p1, n1,
    )
    print("=== yam's finger pair, primitive level: points", n)
    for i in range(n):
        print(
            "  dist", Float64(d1[i]),
            " pos", Float64(p1[3 * i]), Float64(p1[3 * i + 1]),
            Float64(p1[3 * i + 2]),
            " n", Float64(n1[3 * i]), Float64(n1[3 * i + 1]),
            Float64(n1[3 * i + 2]),
        )
    assert_true(
        n == 2,
        "parallel capsules must give TWO points; got " + String(n),
    )
    # MuJoCo's own two contact positions for this pair.
    var want_x0 = 0.320272078
    var want_x1 = 0.349237595
    var lo = Float64(p1[0]) if Float64(p1[0]) < Float64(p1[3]) else Float64(
        p1[3]
    )
    var hi = Float64(p1[0]) if Float64(p1[0]) > Float64(p1[3]) else Float64(
        p1[3]
    )
    assert_true(
        math_abs(lo - want_x0) < 1e-8 and math_abs(hi - want_x1) < 1e-8,
        "the two points must land where MuJoCo puts them (x = "
        + String(want_x0) + " and " + String(want_x1) + "); got "
        + String(lo) + " and " + String(hi),
    )
    for i in range(2):
        assert_true(
            math_abs(Float64(n1[3 * i + 1]) + 1.0) < 1e-12,
            "both normals point from A to B, i.e. -y here",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
