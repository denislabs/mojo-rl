"""`<position kp/kv>` resolved through `<default>` classes — vs MuJoCo.

    pixi run mojo run -I . tests/physics3d/test_position_gain_defaults.mojo

WHAT THIS GATES. `xml_parser`'s actuator loop read `<position>`'s `kp` and `kv`
with `_extract_attr(tag, ...)` — the ELEMENT ONLY — while every neighbouring
attribute (`gear`, `ctrlrange`, `forcerange`, `forcelimited`, `gaintype`,
`biastype`, `gainprm`, `biasprm`, and `<velocity>`'s own `kv`) already resolved
element -> class chain -> root `<default>`. A gain declared in a class was
silently replaced by MuJoCo's default of kp 1 / kv 0.

⚠⚠ EVERY FIXTURE HERE PUTS THE GAIN **ONLY** IN A CLASS, AND NEVER USES 1.
Both constraints are load-bearing:

  · A fixture that writes `kp` on the actuator tag passes with the defect
    present AND absent — it gates nothing. Nine of the ten `<position>`
    actuators in the tree are spelled that way, which is exactly why no
    existing model could serve as the gate (measured: fish, manipulator,
    sawyer and both SO-ARMs all compared EXACT against `mjModel` before the
    fix as well as after).
  · MuJoCo's kp default IS 1, so "the class was never consulted" and "the
    class said 1" are indistinguishable in the output. A gain of 1 would make
    a green result meaningless.

⚠ THE REFERENCE IS MuJoCo, not a hand-written expectation. Each case compiles
the same XML string with the 3.10.0 runtime and reads `actuator_gainprm[0]`
and `-actuator_biasprm[2]`, which is what our `motor_kp` / `motor_kv` mean.
Tolerance is 0.0 — both sides parsed the same literal text.

⚠ WHAT IT GATES NOW. The RUNTIME parser (`full_parser` -> `FlatModelDef` ->
`build_spec_fields` -> `SpecFields`), which is where actuators come from since
phase 1a. This file used to say "only the COMPTIME parser; `full_parser`
carries no actuator table at all" — that was true when it was written and
stopped being true at 1a.1, which gave `full_parser` class resolution on the
actuator path (`224135af`) and taught `<default>` blocks to see `<position>`
at all. Still only `<position>`; `<velocity>` and `<general>` are re-gated
here as controls.
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser import parse_xml_full
from mojo_rl.physics3d.parser.fields_build import build_spec_fields
from mojo_rl.physics3d.fields import SpecFields
from mojo_rl.physics3d.gpu.constants import ACT_IDX_KP, ACT_IDX_KV


# ── A: the gain is on the actuator's OWN class, a direct child of <default>.
#
# ⚠ THE DISCRIMINATING CASE. One class, no nesting, no comments, no second
# block. It failed before the fix, which refutes on its own every explanation
# involving chains, nesting or comments — all of which were filed as the cause
# before this fixture existed.
comptime X_OWN_CLASS = """<mujoco>
  <default>
    <default class="servo">
      <position kp="998.22" kv="2.731" forcerange="-2.94 2.94"/>
    </default>
  </default>
  <worldbody><body>
    <joint name="j" axis="0 1 0" range="-1 1"/>
    <geom type="box" size=".1 .1 .1"/>
  </body></worldbody>
  <actuator><position class="servo" name="a" joint="j" ctrlrange="-1 1"/></actuator>
</mujoco>"""

# ── B: the gain is on the GRANDPARENT class; the actuator names a nested
# child that sets something else. This is SO-ARM100's shape.
comptime X_GRANDPARENT = """<mujoco>
  <default>
    <default class="arm">
      <position kp="50" forcerange="-3.5 3.5"/>
      <default class="rot">
        <joint axis="0 1 0" range="-1.92 1.92"/>
      </default>
    </default>
  </default>
  <worldbody><body>
    <joint name="j" class="rot"/>
    <geom type="box" size=".1 .1 .1"/>
  </body></worldbody>
  <actuator><position class="rot" name="a" joint="j" ctrlrange="-1 1"/></actuator>
</mujoco>"""

# ── C: the gain is at the ROOT `<default>` level, on no class at all.
comptime X_ROOT = """<mujoco>
  <default>
    <position kp="37.5" kv="4.25"/>
  </default>
  <worldbody><body>
    <joint name="j" axis="0 1 0" range="-1 1"/>
    <geom type="box" size=".1 .1 .1"/>
  </body></worldbody>
  <actuator><position name="a" joint="j" ctrlrange="-1 1"/></actuator>
</mujoco>"""

# ── D: the ELEMENT overrides the class. The fix must not invert precedence —
# resolving 3-way is only correct if the element still wins.
comptime X_ELEMENT_WINS = """<mujoco>
  <default>
    <default class="servo">
      <position kp="998.22" kv="2.731"/>
    </default>
  </default>
  <worldbody><body>
    <joint name="j" axis="0 1 0" range="-1 1"/>
    <geom type="box" size=".1 .1 .1"/>
  </body></worldbody>
  <actuator>
    <position class="servo" name="a" joint="j" kp="12.5" kv="0.75" ctrlrange="-1 1"/>
  </actuator>
</mujoco>"""

# ── E: NOTHING sets the gains anywhere. MuJoCo's documented defaults are
# kp 1 / kv 0, and the fix must not disturb them.
comptime X_DEFAULTS = """<mujoco>
  <worldbody><body>
    <joint name="j" axis="0 1 0" range="-1 1"/>
    <geom type="box" size=".1 .1 .1"/>
  </body></worldbody>
  <actuator><position name="a" joint="j" ctrlrange="-1 1"/></actuator>
</mujoco>"""

# ── F: control — `<velocity>`'s kv was ALREADY 3-way. Re-gated so a future
# refactor of the shared helper cannot break it while fixing something else.
comptime X_VELOCITY_CLASS = """<mujoco>
  <default>
    <default class="vel">
      <velocity kv="7.25"/>
    </default>
  </default>
  <worldbody><body>
    <joint name="j" axis="0 1 0" range="-1 1"/>
    <geom type="box" size=".1 .1 .1"/>
  </body></worldbody>
  <actuator><velocity class="vel" name="a" joint="j" ctrlrange="-1 1"/></actuator>
</mujoco>"""

# ⚠ THE SIX `comptime` PARSES ARE GONE WITH THE COMPTIME PARSER. Each fixture
# is now read at RUNTIME, once, inside the test that uses it — which is also
# six fewer comptime-interpreted XML scans in this file's build.
#
# Every fixture is one actuator, one joint, one dof, no tendons, no keyframes,
# so all six record dims are 1. `build_spec_fields` checks the actuator count
# EXACTLY and the rest as capacities, so a fixture that grew a second actuator
# would raise here rather than be silently truncated.
def _gains(xml: String) raises -> Tuple[Float64, Float64]:
    """`(kp, kv)` of actuator 0, off the records the engine reads."""
    var fmd = parse_xml_full(xml)
    var sf = SpecFields[DType.float64, 1, 1, 1, 1, 1, 1]()
    build_spec_fields[DType.float64, 1, 1, 1, 1, 1, 1](fmd, sf)
    return (
        Float64(sf.actuators.data[ACT_IDX_KP]),
        Float64(sf.actuators.data[ACT_IDX_KV]),
    )


def _check(name: String, xml: String, kp: Float64, kv: Float64) raises:
    """Diff one model's actuator 0 against MuJoCo compiled from the same text.

    ⚠ `motor_kp` IS `gainprm[0]` and `motor_kv` IS `-biasprm[2]`; that is the
    mapping MuJoCo compiles a `<position>` into, and reading the pair any other
    way would gate a different quantity.
    """
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(xml)
    var rkp = Float64(py=m.actuator_gainprm[0][0])
    var rkv = -Float64(py=m.actuator_biasprm[0][2])
    print("  ", name, " ours kp", kp, "kv", kv, "  MuJoCo kp", rkp, "kv", rkv)
    assert_true(
        abs(kp - rkp) == 0.0,
        name + ": kp is " + String(kp) + " but MuJoCo compiles " + String(rkp)
        + " — a value of 1.0 means the <default> class was never consulted",
    )
    assert_true(
        abs(kv - rkv) == 0.0,
        name + ": kv is " + String(kv) + " but MuJoCo compiles " + String(rkv),
    )


def test_gain_on_own_class() raises:
    """The discriminating case: one class, no nesting. Was kp 1.0."""
    var g = _gains(X_OWN_CLASS)
    _check("own-class    ", X_OWN_CLASS, g[0], g[1])


def test_gain_on_grandparent_class() raises:
    """SO-ARM100's shape — the actuator names a nested child of the setter."""
    var g = _gains(X_GRANDPARENT)
    _check("grandparent  ", X_GRANDPARENT, g[0], g[1])


def test_gain_at_root_default() raises:
    """The third level: `<default><position .../></default>`, no class."""
    var g = _gains(X_ROOT)
    _check("root-default ", X_ROOT, g[0], g[1])


def test_element_still_wins_over_class() raises:
    """Precedence, which resolving 3-way could plausibly have inverted."""
    var g = _gains(X_ELEMENT_WINS)
    _check("element-wins ", X_ELEMENT_WINS, g[0], g[1])


def test_untouched_defaults_are_mujocos() raises:
    """kp 1 / kv 0 when nothing sets them — the fallback must survive."""
    var g = _gains(X_DEFAULTS)
    _check("no-decl      ", X_DEFAULTS, g[0], g[1])


def test_velocity_class_still_resolves() raises:
    """Control: `<velocity>` was already 3-way and must stay that way.

    ⚠ For `<velocity>` BOTH slots carry K — `gainprm[0]` and `-biasprm[2]` are
    the same number — so this also checks the shared helper did not start
    handing the two branches different answers.
    """
    var g = _gains(X_VELOCITY_CLASS)
    _check("velocity-cls ", X_VELOCITY_CLASS, g[0], g[1])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
