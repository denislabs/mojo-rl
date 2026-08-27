"""The ROOT `<default><motor ctrlrange>` rule — the env's scalar action bounds.

⚠ THIS FILE REPLACED A DIFFERENTIAL GATE THAT HAD FINISHED ITS JOB. Until
phase 1b.5 it compared `full_parser`'s `default_motor_ctrl_min/max` against
the comptime `_xml_default_motor_ctrlrange` over all 56 shipped models, to
prove that moving `CTRL_MIN`/`CTRL_MAX` out of comptime moved no VALUE. It
proved exactly that — 56 models, zero moved — and then lost its oracle when
the comptime MJCF strings were deleted, because a `comptime` scan needs a
`comptime` string. What survives here is the RULE.

⚠⚠ THE VALUE THIS PINS IS KNOWN TO BE WRONG ON SOME MODELS, ON PURPOSE. It is
a model-wide SUMMARY, not the clamp: `apply_actions` clamps each actuator to
its OWN range, while this pair only sizes the box a policy is told to sample
from. Measured against dm_control's `action_spec`, `reach_site_features`
advertises (-1, 1) where the real bounds are ±0.6283, ±0.8378 and ±5.0.
Correcting that changes the action scaling of every shipped env and is owed
its own before/after measurement — so this gate pins the rule AS IT IS.
`test_per_actuator_action_bounds` is where the wrongness is measured on
purpose, against MuJoCo.

WHAT IS PINNED:

  1. a ROOT `<default><motor ctrlrange="lo hi"/>` supplies the pair
  2. ABSENT is (-1, 1)
  3. ⚠ A NAMED `<default class="...">` MUST NOT LEAK. `_root_defaults` strips
     the class blocks, and without that a `<motor>` inside a class would apply
     globally AND a top-level `<motor>` declared after the first class block
     would be missed entirely. swimmer paid for that once at 2000x — see the
     note on `_xml_default_motor_gear`.
  4. `<general>` is accepted as well as `<motor>` — ⚠ AND THIS IS A REAL
     DIFFERENCE FROM THE READER THIS FILE REPLACED. The comptime scanner
     looked ONLY for `<motor>`; `full_parser` also accepts `<general>`,
     `<position>` and `<velocity>`. No model in the tree puts its root
     ctrlrange on a `<general>` tag, so the 56-model differential could not
     see it. It is pinned here so the behaviour is at least WRITTEN DOWN.

Run: pixi run mojo run -I . tests/physics3d/test_ctrl_range_source.mojo
"""

from std.math import abs
from std.testing import assert_true

from mojo_rl.physics3d.parser import parse_xml_full


comptime _BODY = """
  <worldbody>
    <body name="b" pos="0 0 1">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom name="g" type="capsule" size=".05 .2"/>
    </body>
  </worldbody>
  <actuator><motor name="m" joint="j" gear="1"/></actuator>
</mujoco>"""

comptime _NO_DEFAULT = "<mujoco model=\"none\">" + _BODY

comptime _ROOT_MOTOR = """<mujoco model="root">
  <default><motor ctrlrange="-.4 .4"/></default>""" + _BODY

comptime _ROOT_GENERAL = """<mujoco model="general">
  <default><general ctrlrange="-2 3"/></default>""" + _BODY

# ⚠ The class block must NOT supply the model-wide pair.
comptime _CLASS_ONLY = """<mujoco model="classonly">
  <default>
    <default class="act"><motor ctrlrange="-9 9"/></default>
  </default>""" + _BODY

# ⚠ A root `<motor>` declared AFTER a class block must still be seen.
comptime _ROOT_AFTER_CLASS = """<mujoco model="after">
  <default>
    <default class="act"><motor ctrlrange="-9 9"/></default>
    <motor ctrlrange="-.7 .7"/>
  </default>""" + _BODY


struct Tally(Copyable, Movable):
    var checks: Int
    var bad: Int

    def __init__(out self):
        self.checks = 0
        self.bad = 0


def _pair(
    mut t: Tally, what: String, xml: String, lo: Float64, hi: Float64
) raises:
    var fmd = parse_xml_full(xml)
    t.checks += 1
    if (
        abs(fmd.default_motor_ctrl_min - lo) > 1e-12
        or abs(fmd.default_motor_ctrl_max - hi) > 1e-12
    ):
        t.bad += 1
        print(
            "  FAIL", what, ": got (", fmd.default_motor_ctrl_min, ",",
            fmd.default_motor_ctrl_max, ")  want (", lo, ",", hi, ")",
        )


def main() raises:
    var t = Tally()
    print("=== root <default><motor ctrlrange> ===")

    _pair(t, "absent -> (-1, 1)", String(_NO_DEFAULT), -1.0, 1.0)
    _pair(t, "root <motor>", String(_ROOT_MOTOR), -0.4, 0.4)
    _pair(t, "root <general> is accepted", String(_ROOT_GENERAL), -2.0, 3.0)
    # ⚠ The two class rows are the ones with a history behind them.
    _pair(t, "class-only must NOT leak", String(_CLASS_ONLY), -1.0, 1.0)
    _pair(t, "root AFTER a class block", String(_ROOT_AFTER_CLASS), -0.7, 0.7)

    print()
    print("checks:", t.checks, " failures:", t.bad)
    assert_true(
        t.bad == 0,
        String(t.bad) + " root-default ctrlrange rule(s) wrong — these are"
        " the env's advertised action bounds",
    )
    print()
    print("PASS")
