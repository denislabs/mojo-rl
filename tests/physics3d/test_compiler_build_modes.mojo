"""`<compiler>` build modes — the parse rules, pinned against MuJoCo's defaults.

⚠ THIS FILE REPLACED A DIFFERENTIAL GATE, and the replacement is the better
one. Until phase 1b.5 it compared `full_parser`'s record against the comptime
scanners (`_xml_compiler_inertiafromgeom` and friends) over 16 shipped models.
That gate did its job — it proved the switch-over to `FlatModelDef` moved no
value — and then lost its oracle when the comptime MJCF strings were deleted,
because a `comptime` scan needs a `comptime` string.

⚠⚠ IT WAS ALSO BLIND IN THE WAY THAT MATTERS. Two readers of ours agreeing
proves nothing about either, and `inertiafromgeom` has exactly that history:
its default is AUTO, not off, and getting it wrong gave dm_control's pendulum
~1/21 of its true inertia while both readers agreed perfectly. What is worth
gating is the RULE against MuJoCo's documented behaviour, which is what this
file does.

WHAT IS PINNED — the four rules, each on a fixture that isolates it:

  1. ABSENT means AUTO (2), not off. This is the one that has already cost a
     real bug, and the one a reader is most likely to "simplify" to 0.
  2. `true` -> 1, `false` -> 0, `auto` -> 2.
  3. `inertiagrouprange="lo hi"` parses to (lo, hi); absent is (0, 5).
  4. `settotalmass` ABSENT is -1.0, not 0.0 — `settotalmass="0"` is a legal
     request and must not read as "not specified".

⚠ WHAT THIS FILE CANNOT SEE, stated so nobody mistakes it for full coverage:
MuJoCo does not RETAIN `inertiafromgeom` — the compiler consumes it and only
the resulting masses survive into `mjModel` — so the flag itself has no
parity oracle anywhere. What pins the SEMANTICS is the effect, gated
elsewhere and against MuJoCo:

  * `test_xml_full_parser` asserts half-cheetah's torso mass equals
    `mjModel.body_mass[1]` to 1e-12 — the inertiafromgeom=auto path, and the
    assertion that caught a call site passing 0 for an XML that says nothing.
  * `test_jaco_mesh_body_inertia_vs_mujoco` covers the mesh + boundmass path.
  * dm_control's cheetah covers `settotalmass` through its masses: it declares
    `settotalmass="14"` and NO inertiafromgeom, so it takes the AUTO default
    and the rescale runs — 21.18 kg -> 14.0, confirmed against the runtime.

Run: pixi run mojo run -I . tests/physics3d/test_compiler_build_modes.mojo
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
</mujoco>"""

comptime _NO_COMPILER_TAG = "<mujoco model=\"none\">" + _BODY
comptime _EMPTY_COMPILER = (
    "<mujoco model=\"empty\"><compiler angle=\"radian\"/>" + _BODY
)
comptime _IFG_TRUE = (
    "<mujoco model=\"t\"><compiler inertiafromgeom=\"true\"/>" + _BODY
)
comptime _IFG_FALSE = (
    "<mujoco model=\"f\"><compiler inertiafromgeom=\"false\"/>" + _BODY
)
comptime _IFG_AUTO = (
    "<mujoco model=\"a\"><compiler inertiafromgeom=\"auto\"/>" + _BODY
)
comptime _IGR = (
    "<mujoco model=\"igr\"><compiler inertiagrouprange=\"4 5\"/>" + _BODY
)
comptime _STM = (
    "<mujoco model=\"stm\"><compiler settotalmass=\"14\"/>" + _BODY
)
comptime _STM_ZERO = (
    "<mujoco model=\"stm0\"><compiler settotalmass=\"0\"/>" + _BODY
)


struct Tally(Copyable, Movable):
    var checks: Int
    var bad: Int

    def __init__(out self):
        self.checks = 0
        self.bad = 0


def _eq_i(mut t: Tally, what: String, got: Int, want: Int) raises:
    t.checks += 1
    if got != want:
        t.bad += 1
        print("  FAIL", what, ": got", got, " want", want)


def _eq_f(mut t: Tally, what: String, got: Float64, want: Float64) raises:
    t.checks += 1
    if abs(got - want) > 1e-12:
        t.bad += 1
        print("  FAIL", what, ": got", got, " want", want)


def main() raises:
    var t = Tally()
    print("=== <compiler> build modes: the parse rules ===")

    # ── rule 1: ABSENT means AUTO, at both spellings of absent ───────────
    var none = parse_xml_full(String(_NO_COMPILER_TAG))
    var empty = parse_xml_full(String(_EMPTY_COMPILER))
    _eq_i(t, "no <compiler> tag -> inertiafromgeom", none.inertiafromgeom, 2)
    _eq_i(t, "<compiler> without the attr", empty.inertiafromgeom, 2)

    # ── rule 2: the three spellings ──────────────────────────────────────
    _eq_i(t, 'inertiafromgeom="true"',
          parse_xml_full(String(_IFG_TRUE)).inertiafromgeom, 1)
    _eq_i(t, 'inertiafromgeom="false"',
          parse_xml_full(String(_IFG_FALSE)).inertiafromgeom, 0)
    _eq_i(t, 'inertiafromgeom="auto"',
          parse_xml_full(String(_IFG_AUTO)).inertiafromgeom, 2)

    # ── rule 3: inertiagrouprange ────────────────────────────────────────
    var igr = parse_xml_full(String(_IGR))
    _eq_i(t, 'inertiagrouprange="4 5" min', igr.inertiagrouprange_min, 4)
    _eq_i(t, 'inertiagrouprange="4 5" max', igr.inertiagrouprange_max, 5)
    _eq_i(t, "inertiagrouprange absent min", none.inertiagrouprange_min, 0)
    _eq_i(t, "inertiagrouprange absent max", none.inertiagrouprange_max, 5)

    # ── rule 4: settotalmass, and the -1.0 vs 0.0 distinction ────────────
    _eq_f(t, 'settotalmass="14"',
          parse_xml_full(String(_STM)).settotalmass, 14.0)
    _eq_f(t, "settotalmass absent", none.settotalmass, -1.0)
    # ⚠ THE ROW THAT MAKES THE SENTINEL MEAN SOMETHING. If absent were 0.0
    # this would be indistinguishable from an explicit zero.
    _eq_f(t, 'settotalmass="0" (explicit)',
          parse_xml_full(String(_STM_ZERO)).settotalmass, 0.0)

    print()
    print("checks:", t.checks, " failures:", t.bad)
    assert_true(
        t.bad == 0, String(t.bad) + " <compiler> parse rule(s) wrong"
    )
    print()
    print("PASS")
