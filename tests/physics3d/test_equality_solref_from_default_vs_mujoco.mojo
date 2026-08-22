"""`<default><equality solref=.../></default>` — the ROOT class nobody read.

    pixi run mojo run -I . tests/physics3d/test_equality_solref_from_default_vs_mujoco.mojo

WHAT IT WAS. `_fill_equality_solparams` read `solref` / `solimp` /
`torquescale` off the equality element's OWN opening tag and nowhere else.
MJCF's top-level `<default>` block is the class "main" and every element that
names no `class` inherits from it, so a model that puts its constraint
parameters there got MuJoCo's built-in defaults instead.

agility_cassie is that model:

    <default>
      <equality solref="0.005 1"/>
      ...
    <equality>
      <connect body1="left-plantar-rod" body2="left-foot" anchor="0.35012 0 0"/>
      ... x4

Four closed-loop rows running a constraint time constant of **0.02 instead of
0.005** — four times too slow. `mjModel.eq_solref` says `[0.005, 1]`; we said
`[0.02, 1]`.

⚠ THE RULE EXISTED IN THIS PARSER TWICE AND DRIFTED. The
`<equality><tendon>` branch 900 lines further down already consulted
`_default_class_tag(...)` for exactly these two attributes. The connect/weld/
joint branch never did. Both go through one function now.

⚠ AND `_default_class_tag` COULD NOT HAVE FIXED IT ALONE: it returns "" for an
empty class name, so it reads the root block for NO element. `_root_default_tag`
is the missing half.

MEASURED (`sweepN.py 1`, seed 2024):

    agility_cassie   4.460e-04  ->  2.220e-16      (it leaves the board)
    board            76 -> 77 of 85 at or below 1e-9

⚠ The 4.460e-04 is what was left after `10b372aa` fixed the ball joint
underneath it; before that the same scene was reading 4.460e-04 for two
independent reasons. The §9.8.3 ablation had named EQUALITY as the driver and
was right.

SURVEY. Exactly two Menagerie models put equality parameters in a `<default>`,
both on the ROOT class: agility_cassie (`solref="0.005 1"`) and
apptronik_apollo (`solref="0.005 1" solimp="0.99 0.999 0.00001"`). apollo has
`neq == 0`, so cassie is the only one this moves — which is also why the
survey matters: a defect with one live instance is one nobody trips over
twice.
"""

from std.math import abs
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import read_model_source

comptime CASSIE = String(
    "references/mujoco_menagerie-main/agility_cassie/scene.xml"
)

# MuJoCo 3.10.0, `m.eq_solref` / `m.eq_solimp` for all four of cassie's
# connects. The solimp is MuJoCo's own default — cassie declares only solref,
# so it is here to prove the default half still comes through.
comptime MJ_SOLREF_0 = 0.005
comptime MJ_SOLREF_1 = 1.0
comptime MJ_SOLIMP_0 = 0.9
comptime MJ_SOLIMP_1 = 0.95

# MuJoCo's built-in equality solref, i.e. what every one of cassie's rows was
# getting. Named so the assertion can say "this is the value it must NOT be".
comptime MJ_BUILTIN_SOLREF_0 = 0.02

# A named class, which the other half of the chain has to reach. Two connects:
# one takes the class, one takes the root.
comptime CLASS_XML = String(
    """
<mujoco model="equality classes">
  <default>
    <equality solref="0.005 1"/>
    <default class="stiff">
      <equality solref="0.001 1" solimp="0.95 0.99 0.002"/>
    </default>
  </default>
  <worldbody>
    <body name="a" pos="0 0 1"><freejoint/><geom size="0.1"/></body>
    <body name="b" pos="0 0 2"><freejoint/><geom size="0.1"/></body>
    <body name="c" pos="0 0 3"><freejoint/><geom size="0.1"/></body>
  </worldbody>
  <equality>
    <connect body1="a" body2="b" anchor="0 0 0"/>
    <connect body1="b" body2="c" anchor="0 0 0" class="stiff"/>
  </equality>
</mujoco>
"""
)

# No `<default>` at all — MuJoCo's built-in must survive the change.
comptime BARE_XML = String(
    """
<mujoco model="equality bare">
  <worldbody>
    <body name="a" pos="0 0 1"><freejoint/><geom size="0.1"/></body>
    <body name="b" pos="0 0 2"><freejoint/><geom size="0.1"/></body>
  </worldbody>
  <equality>
    <connect body1="a" body2="b" anchor="0 0 0"/>
  </equality>
</mujoco>
"""
)


def test_the_elements_themselves_declare_no_solref() raises:
    """⚠⚠ NON-VACUITY. If cassie's `<connect>` tags carried `solref` of their
    own, the element-only read would already have been right and every row
    below would pass without testing anything.
    """
    print("=== cassie's <connect> tags carry NO solref of their own ===")
    var src = read_model_source(CASSIE)
    var xml = expand_mjcf(src[0], src[1])
    var eq_at = xml.find("<equality>")
    assert_true(eq_at >= 0, "cassie has no <equality> section")
    var eq_end = xml.find("</equality>", eq_at)
    var sec = String(xml[byte = eq_at : eq_end])
    var has = sec.find("solref") >= 0
    print("  '<equality>...</equality>' contains 'solref':", has, " (want False)")
    assert_true(
        not has,
        "cassie's equality ELEMENTS now declare solref themselves, so this"
        " file no longer tests the <default> chain at all — find another"
        " model or write a synthetic one",
    )
    var d_at = xml.find("<default>")
    assert_true(
        d_at >= 0,
        "cassie has no root <default> block; this file's premise is gone",
    )


def test_cassie_takes_solref_from_the_root_default() raises:
    """The defect, stated against `mjModel.eq_solref`."""
    print("=== cassie's four connects read solref from the ROOT <default> ===")
    var src = read_model_source(CASSIE)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    print("  neq =", len(fmd.equalities), " (MuJoCo: 4)")
    assert_true(
        len(fmd.equalities) == 4,
        "cassie should parse four equality constraints, got "
        + String(len(fmd.equalities)),
    )
    for i in range(len(fmd.equalities)):
        var e = fmd.equalities[i]
        print(
            "  eq", i, " solref", e.solref_0, e.solref_1,
            " solimp", e.solimp_0, e.solimp_1,
        )
        assert_true(
            abs(e.solref_0 - MJ_SOLREF_0) < 1e-15,
            String("equality ") + String(i) + " solref[0] is "
            + String(e.solref_0) + ", MuJoCo says " + String(MJ_SOLREF_0)
            + ". " + String(MJ_BUILTIN_SOLREF_0) + " means the root"
            " <default><equality/> was never read and the row is running a"
            " constraint time constant FOUR TIMES too slow.",
        )
        assert_true(
            abs(e.solref_1 - MJ_SOLREF_1) < 1e-15,
            String("equality ") + String(i) + " solref[1] is "
            + String(e.solref_1),
        )
        # cassie declares no solimp, so these must still be MuJoCo's defaults —
        # a fallback chain that overwrites what it should leave alone is the
        # other way to get this wrong.
        assert_true(
            abs(e.solimp_0 - MJ_SOLIMP_0) < 1e-15
            and abs(e.solimp_1 - MJ_SOLIMP_1) < 1e-15,
            String("equality ") + String(i) + " solimp is ("
            + String(e.solimp_0) + ", " + String(e.solimp_1)
            + "), MuJoCo's default is (0.9, 0.95) — cassie declares none",
        )


def test_a_named_class_beats_the_root_and_the_root_still_applies() raises:
    """Both halves of the chain, on one model.

    ⚠ The second connect names `class="stiff"`, which itself sits INSIDE the
    root block. The first names nothing. They must come out different.
    """
    print("=== element class beats root; root still reaches the classless ===")
    var fmd = parse_xml_full(CLASS_XML, String("."))
    assert_true(
        len(fmd.equalities) == 2,
        "expected two equality constraints, got " + String(len(fmd.equalities)),
    )
    var e0 = fmd.equalities[0]
    var e1 = fmd.equalities[1]
    print("  no class : solref", e0.solref_0, e0.solref_1,
          " solimp0", e0.solimp_0)
    print("  'stiff'  : solref", e1.solref_0, e1.solref_1,
          " solimp0", e1.solimp_0)
    assert_true(
        abs(e0.solref_0 - 0.005) < 1e-15,
        "the classless connect should take the ROOT default 0.005, got "
        + String(e0.solref_0),
    )
    assert_true(
        abs(e1.solref_0 - 0.001) < 1e-15,
        "the connect naming class='stiff' should take 0.001, got "
        + String(e1.solref_0)
        + " — a chain that stops at the root reads 0.005 here",
    )
    assert_true(
        abs(e1.solimp_0 - 0.95) < 1e-15,
        "class='stiff' declares solimp too; got solimp[0] "
        + String(e1.solimp_0),
    )


def test_no_default_block_keeps_mujocos_builtin() raises:
    """⚠ THE OTHER FAILURE MODE: a fallback that invents a value. With no
    `<default>` anywhere, `_root_default_tag` must return nothing and the
    built-in 0.02 must stand."""
    print("=== no <default> at all: MuJoCo's built-in 0.02 survives ===")
    var fmd = parse_xml_full(BARE_XML, String("."))
    assert_true(len(fmd.equalities) == 1, "expected one equality constraint")
    var e = fmd.equalities[0]
    print("  solref", e.solref_0, e.solref_1, " (want 0.02 1)")
    assert_true(
        abs(e.solref_0 - MJ_BUILTIN_SOLREF_0) < 1e-15
        and abs(e.solref_1 - 1.0) < 1e-15,
        "with no <default> block the equality should keep MuJoCo's built-in"
        " solref (0.02, 1); got (" + String(e.solref_0) + ", "
        + String(e.solref_1) + ")",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
