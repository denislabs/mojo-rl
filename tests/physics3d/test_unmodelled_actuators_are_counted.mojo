"""Every `<actuator>` child this parser skips must be COUNTED.

    pixi run mojo run -I . tests/physics3d/test_unmodelled_actuators_are_counted.mojo

WHAT WAS THERE. `_fill_actuators` scans for four spellings — `<motor>`,
`<general>`, `<position>`, `<velocity>`. MJCF has ten. Anything else in the
section was skipped with no record and no message, so the model loaded, built,
stepped, and consumed a control vector of the wrong length.

⚠⚠ A SKIPPED ACTUATOR DOES NOT FAIL, IT SHIFTS. Callers index `ctrl` by
position, so every index past the first missing actuator lands on a DIFFERENT
actuator than the one the policy meant. A trained policy handed to this engine
would drive the wrong joints and look like a bad policy.

⚠ IT WAS FOUND BY DIFFING COUNTS, NOT BY ANYTHING THE PARSER SAID. Across the
85 loadable Menagerie scenes exactly two disagree with the runtime's `nu`:

    flybody        MuJoCo nu 78, ours 70    eight <adhesion>
    shadow_dexee   MuJoCo nu 12, ours  0    twelve <plugin plugin="mujoco.pid">

flybody's whole `qfrc_actuator` residual against MuJoCo is those eight: its
`<adhesion>` actuators use a BODY transmission, so each one loads every dof in
its chain — `femur_T2_right` reads -0.120000 here (exactly `gain * ctrl` from
its own servo) against MuJoCo's -0.085509, and the difference is the claw pad
pulling on the same chain.

WHAT THIS GATES. Not the feature — the BOOKKEEPING. `nact + unmodelled ==
MuJoCo's nu` is an invariant that survives implementing any of these: the day
`<adhesion>` lands, flybody goes to 78 + 0 and the sum still holds.
"""

from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import read_model_source

comptime FLYBODY = String(
    "references/mujoco_menagerie-main/flybody/scene.xml"
)
comptime DEXEE = String(
    "references/mujoco_menagerie-main/shadow_dexee/scene.xml"
)
# ⚠ THE NEGATIVE CONTROL, and it must be a REAL model with many actuators.
# Without it this file passes against a parser that counts every actuator as
# unmodelled.
comptime G1 = String(
    "references/mujoco_menagerie-main/unitree_g1/scene.xml"
)

# MuJoCo 3.10.0 `m.nu`.
comptime MJ_NU_FLYBODY = 78
comptime MJ_NU_DEXEE = 12
comptime MJ_NU_G1 = 29


def _counts(path: String) raises -> List[Int]:
    """`[len(actuators), unmodelled_actuators]` straight off the parser."""
    var src = read_model_source(path)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var out = List[Int]()
    out.append(len(fmd.actuators))
    out.append(fmd.unmodelled_actuators)
    return out^


def test_skipped_actuators_are_counted() raises:
    """`nact + unmodelled == nu`, on the two models that are short and one
    that is not."""
    print("=== nact + unmodelled vs MuJoCo nu ===")
    var paths: List[String] = [FLYBODY, DEXEE, G1]
    var want: List[Int] = [MJ_NU_FLYBODY, MJ_NU_DEXEE, MJ_NU_G1]
    var names: List[String] = [
        String("flybody"), String("shadow_dexee"), String("unitree_g1"),
    ]
    for i in range(len(paths)):
        var c = _counts(paths[i])
        print(
            "  ", names[i], " nact", c[0], " unmodelled", c[1],
            " sum", c[0] + c[1], "  MuJoCo nu", want[i],
        )
        assert_true(
            c[0] + c[1] == want[i],
            names[i] + ": nact " + String(c[0]) + " + unmodelled "
            + String(c[1]) + " = " + String(c[0] + c[1])
            + ", but MuJoCo reports nu " + String(want[i])
            + ". Some `<actuator>` child is being dropped without being"
            " counted, so a control vector sized for MuJoCo is misaligned"
            " from the first missing actuator onwards.",
        )
    # ⚠ AND THE COUNT MUST BE ZERO WHERE NOTHING IS SKIPPED. The sum above is
    # satisfied by a parser that skips everything and counts everything;
    # g1 declares 29 `<position>` servos and none of them may land here.
    var g1c = _counts(G1)
    assert_true(
        g1c[1] == 0 and g1c[0] == MJ_NU_G1,
        "unitree_g1 declares " + String(MJ_NU_G1) + " ordinary `<position>`"
        " actuators and none of them is an unmodelled type; got nact "
        + String(g1c[0]) + " and unmodelled " + String(g1c[1]),
    )
    # ⚠ AND NONZERO WHERE SOMETHING IS. Without this the field could simply
    # never be written and the two rows above would still pass on a parser
    # that also happened to parse adhesion.
    var fbc = _counts(FLYBODY)
    var dxc = _counts(DEXEE)
    assert_true(
        fbc[0] + fbc[1] == MJ_NU_FLYBODY and dxc[0] + dxc[1] == MJ_NU_DEXEE,
        "the two short models must still add up",
    )
    assert_true(
        fbc[1] > 0 or fbc[0] == MJ_NU_FLYBODY,
        "flybody is either fully parsed (nact 78) or short AND counted;"
        " nact " + String(fbc[0]) + " unmodelled " + String(fbc[1]),
    )
    assert_true(
        dxc[1] > 0 or dxc[0] == MJ_NU_DEXEE,
        "shadow_dexee is either fully parsed (nact 12) or short AND counted;"
        " nact " + String(dxc[0]) + " unmodelled " + String(dxc[1]),
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
