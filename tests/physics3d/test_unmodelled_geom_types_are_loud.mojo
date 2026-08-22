"""`<geom type="hfield">` and `type="sdf"` are SPHERES here, and now they say so.

    pixi run mojo run -I . tests/physics3d/test_unmodelled_geom_types_are_loud.mojo

WHAT IT WAS. `_geom_type_from_str` ends in `return _GEOM_SPHERE  # default`,
and `hfield` and `sdf` fall through to it. A heightfield therefore collides as
a BALL of radius `size[0]` — not an approximation of terrain, a completely
different solid — and nothing anywhere said so. The function's own comment has
carried the warning since `ellipsoid` was fixed out of the same default:

    ⚠ THE DEFAULT IS A SILENT SUBSTITUTION, not an error. `ellipsoid` used
    to land here, which cost fish its whole mass distribution (bug 26).

MEASURED, `google_barkour_vb/scene_hfield_mjx` at its keyframe: MuJoCo emits
8 contacts and we emit 4, on 6 different body pairs, **2.219e-01** apart in
depth and **81.1 deg** apart in normal. That is the worst row of the whole
contact-set column and the only one that is a missing CAPABILITY rather than a
numerical difference.

⚠ THIS FILE DOES NOT FIX THAT. It makes the substitution audible, which is
what the rest of §11.7's list does for the other known gaps. Implementing
`mjc_HFieldConvex` is a subsystem, not a constant.

⚠ THE COUNT IS OVER THE DOCUMENT, not off the element — a
`<default><geom type="hfield"/></default>` is a legal spelling and an
element-only read would miss it, which is the trap this parser has been bitten
by repeatedly. `<hfield>` inside `<asset>` is deliberately NOT counted:
declaring the asset is harmless, it is a GEOM naming the type that gets
substituted.
"""

from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import read_model_source

comptime HFIELD_SCENE = String(
    "references/mujoco_menagerie-main/google_barkour_vb/scene_hfield_mjx.xml"
)
comptime CLEAN_SCENE = String(
    "references/mujoco_menagerie-main/kinova_gen3/scene.xml"
)

# `<default><geom type="hfield"/></default>` — the spelling an element-only
# read misses. Neither geom names a type of its own.
comptime DEFAULT_CLASS_XML = String(
    """
<mujoco model="hfield in a default">
  <default>
    <default class="terrain">
      <geom type="hfield" hfield="hf"/>
    </default>
  </default>
  <asset>
    <hfield name="hf" nrow="4" ncol="4" size="1 1 0.2 0.1"/>
  </asset>
  <worldbody>
    <geom class="terrain" name="g0"/>
    <geom class="terrain" name="g1" pos="3 0 0"/>
  </worldbody>
</mujoco>
"""
)


def test_a_hfield_geom_is_counted() raises:
    """The one Menagerie scene that has one."""
    print("=== `hfield` geoms are counted, not silently substituted ===")
    var src = read_model_source(HFIELD_SCENE)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    print("  unmodelled_geom_types =", fmd.unmodelled_geom_types, " (want 1)")
    assert_true(
        fmd.unmodelled_geom_types >= 1,
        "google_barkour_vb's heightfield geom was not counted, so the engine"
        " substitutes a SPHERE of radius size[0] for it and says nothing —"
        " which is how a 2.219e-01 contact-depth error stayed invisible",
    )


def test_a_model_without_one_stays_quiet() raises:
    """⚠ NON-VACUITY FROM THE OTHER SIDE: a diagnostic that always fires is
    not a diagnostic. `kinova_gen3` has meshes and primitives and no
    heightfield."""
    print("=== a model with no hfield/sdf geom counts ZERO ===")
    var src = read_model_source(CLEAN_SCENE)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    print("  unmodelled_geom_types =", fmd.unmodelled_geom_types, " (want 0)")
    assert_true(
        fmd.unmodelled_geom_types == 0,
        "kinova_gen3 has no `hfield`/`sdf` geom but the counter reported "
        + String(fmd.unmodelled_geom_types)
        + " — this diagnostic would then fire on every model and mean nothing",
    )


def test_the_default_class_spelling_is_counted() raises:
    """⚠⚠ THE `<default>` CHAIN IS THE #1 SOURCE OF MISSED ATTRIBUTES HERE.

    Two geoms inherit `type="hfield"` from a class and name no type of their
    own. A count taken off the element reports ZERO on this model.
    """
    print("=== `type=\"hfield\"` inside a `<default>` still counts ===")
    var fmd = parse_xml_full(DEFAULT_CLASS_XML, String("."))
    print("  unmodelled_geom_types =", fmd.unmodelled_geom_types, " (want 1)")
    assert_true(
        fmd.unmodelled_geom_types >= 1,
        "a `<geom type=\"hfield\">` declared in a `<default>` class was not"
        " counted; the two geoms that inherit it collide as spheres in"
        " silence",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
