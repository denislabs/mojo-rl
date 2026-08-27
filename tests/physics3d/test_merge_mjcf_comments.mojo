"""`merge_mjcf` must not lose a section to a comment that mentions its tag.

    pixi run mojo run -I . tests/physics3d/test_merge_mjcf_comments.mojo

`_extract_section_inner` depth-counts `"<" + tag` over RAW TEXT. Until
2026-08-13 `merge_mjcf` fed it unstripped input, so a comment merely MENTIONING
a section tag counted as an opener, the depth never balanced, and the section
came out EMPTY — with no diagnostic, because a dropped section is not an error.
MuJoCo then rejects the merged model with "unknown default class name", or
worse, accepts a model that is quietly missing its constraints.

⚠ NESTING IS IRRELEVANT, and this was first filed as "merge_mjcf cannot do
nested `<default>` blocks". It can. The CLEAN case below is nested and merges
fine; the only difference in the failing case is one comment line.

⚠⚠ THE ASSERTION IS `mjModel` COUNTS FROM THE MERGED STRING, not "is the
substring present". Searching the merged text for `<default>` would pass on a
section that survived but lost its contents, and the original failure mode
included content resurfacing INSIDE `<asset>`. Compiling both merges and
diffing the counts is the only check that covers both.

⚠ ONE POISONED COMMENT PER CASE, each naming a DIFFERENT section. All four
sections here have been lost by this function before — `<tendon>` (fish,
missing from the accumulator list), `<equality>` (quadruped's leg couplings,
via a self-closing `<equality/>`), `<contact>` (dropped on a stale claim), and
now `<default>`. A single fixture with all four poisoned at once would pass the
moment ANY one of them worked.

THE PAIR OF FIXTURES IS THE POINT: `_clean` and each poisoned variant differ by
exactly one comment, so a count difference can only come from the comment.
"""

from std.python import Python
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser import merge_mjcf

# A robot with all four sections a comment could poison. Each variant injects
# ONE comment INSIDE the named section.
#
# ⚠⚠ POSITION MATTERS, AND THE FIRST VERSION OF THIS FILE GOT IT WRONG.
# A comment placed AFTER a section has already closed is harmless — the
# scanner extracted the real section before reaching it — so `<default>`
# passed even with the fix reverted, which is precisely the case that motivated
# the fix. SO-101's real comment sat BETWEEN two nested `<default>` classes,
# i.e. INSIDE the block, where the depth counter is still open. Every poisoned
# comment below is inside its section for that reason.



def _mk(default_c: String, tendon_c: String, equality_c: String,
        contact_c: String) -> String:
    """One model, with an optional comment inside each of four sections."""
    return String(
        """<mujoco>
  <default>
    <default class="servo"><position kp="37.5"/></default>
""", default_c, """    <default class="link"><joint damping="0.5"/></default>
  </default>
  <worldbody>
    <body name="a"><joint name="ja" type="hinge" axis="0 1 0" class="link"/>
      <geom type="box" size=".1 .1 .1"/></body>
    <body name="b" pos="0 0 .5"><joint name="jb" type="hinge" axis="0 1 0" class="link"/>
      <geom type="box" size=".1 .1 .1"/></body>
  </worldbody>
  <tendon>
""", tendon_c, """    <fixed name="t1"><joint joint="ja" coef="1"/><joint joint="jb" coef="-1"/></fixed>
  </tendon>
  <equality>
""", equality_c, """    <weld body1="a" body2="b"/>
  </equality>
  <contact>
""", contact_c, """    <exclude body1="a" body2="b"/>
  </contact>
  <actuator><position class="servo" name="m" joint="ja" ctrlrange="-1 1"/></actuator>
</mujoco>""",
    )


comptime NO = ""
comptime CLEAN = _mk(NO, NO, NO, NO)
comptime PLAIN = _mk("    <!-- an ordinary remark, no angle brackets -->\n", NO, NO, NO)
comptime P_DEFAULT = _mk("    <!-- merged from a second top-level <default>; see the bake -->\n", NO, NO, NO)
comptime P_TENDON = _mk(NO, "    <!-- this <tendon> ties ja to jb -->\n", NO, NO)
comptime P_EQUALITY = _mk(NO, NO, "    <!-- the <equality> below couples the links -->\n", NO)
comptime P_CONTACT = _mk(NO, NO, NO, "    <!-- this <contact> section excludes a/b -->\n")

comptime TASK = """<mujoco><worldbody>
  <body name="target" mocap="true" pos="0 0 1">
    <geom type="sphere" size="0.01" contype="0" conaffinity="0"/></body>
</worldbody></mujoco>"""


def _counts(xml: String) raises -> List[Int]:
    """`mjModel` counts for a merged string. Compiling IS the check: before the
    fix, the poisoned `<default>` merge did not compile at all."""
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(xml)
    var out = List[Int]()
    out.append(Int(py=m.nbody))
    out.append(Int(py=m.njnt))
    out.append(Int(py=m.ngeom))
    out.append(Int(py=m.nu))
    out.append(Int(py=m.neq))
    out.append(Int(py=m.ntendon))
    out.append(Int(py=m.nexclude))
    return out^


def _same_as_clean(name: String, poisoned: String) raises:
    var want = _counts(merge_mjcf(CLEAN, TASK))
    var got = _counts(merge_mjcf(poisoned, TASK))
    var labels = List[String]()
    labels.append(String("nbody"))
    labels.append(String("njnt"))
    labels.append(String("ngeom"))
    labels.append(String("nu"))
    labels.append(String("neq"))
    labels.append(String("ntendon"))
    labels.append(String("nexclude"))
    print("  ", name, " nbody/njnt/ngeom/nu/neq/ntendon/nexclude =",
          got[0], got[1], got[2], got[3], got[4], got[5], got[6])
    for i in range(len(want)):
        assert_true(
            want[i] == got[i],
            name + ": " + labels[i] + " is " + String(got[i])
            + " but the comment-free merge gives " + String(want[i])
            + " — the two inputs differ by ONE COMMENT, so the section was"
            " lost to `_extract_section_inner`'s depth counter",
        )


def test_clean_merge_is_the_baseline() raises:
    """The control. Nested default classes, no comments — and it must compile,
    or every comparison below is against a broken reference."""
    var c = _counts(merge_mjcf(CLEAN, TASK))
    print("  clean baseline  nbody", c[0], " neq", c[4], " ntendon", c[5],
          " nexclude", c[6])
    assert_true(c[4] == 1, "baseline lost its <equality>")
    assert_true(c[5] == 1, "baseline lost its <tendon>")
    assert_true(c[6] == 1, "baseline lost its <contact><exclude>")
    assert_true(c[3] == 1, "baseline lost its <actuator>")


def test_plain_comment_is_harmless() raises:
    """A comment with no angle brackets. Pins that the fix is about TAG-LIKE
    text, not about comments existing."""
    _same_as_clean("plain comment ", PLAIN)


def test_comment_naming_default() raises:
    """THE ONE THAT BROKE SO-101 — a comment BETWEEN two nested classes.

    ⚠ The comment must be INSIDE the block. Placed after `</default>` this
    case passes even with the fix reverted, because the real section was
    already extracted; the first version of this file made exactly that
    mistake and the gate proved nothing for its motivating case.
    """
    _same_as_clean("names <default>", P_DEFAULT)


def test_comment_naming_equality() raises:
    """Sibling section. quadruped lost its leg couplings to the same counter,
    from a self-closing `<equality/>` rather than a comment."""
    _same_as_clean("names <equality>", P_EQUALITY)


def test_comment_naming_tendon() raises:
    """Sibling section. fish shipped with a dropped `<tendon>` for a day."""
    _same_as_clean("names <tendon>  ", P_TENDON)


def test_comment_naming_contact() raises:
    """Sibling section. `<contact>` was dropped from the accumulator list on a
    stale claim until 2026-08-03."""
    _same_as_clean("names <contact> ", P_CONTACT)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
