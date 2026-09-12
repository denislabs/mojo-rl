"""Every env's hand-counted site index, checked against the model (AUD-23).

The environment configs address sensors by literals derived by reading the
MJCF and counting in worldbody DFS order — `TOUCH_TOE_SITE_IDX = 0`,
`TORSO_SITE_IDX = 24`, `ESCAPE_RF_SITE_0 = 3`, `DOG_SITE_PALM_L = 9`, and
base-plus-stride arithmetic over them. The parser's own comment on the site
name table says what goes wrong: "sensors are addressed BY SITE INDEX, so a
permuted site array reads the wrong sensor." Inserting one site in a task
fragment shifts twenty rangefinders and nothing raises.

⚠⚠ THIS GATE IS THE DIRECT PIN THOSE LITERALS NEVER HAD. They were checked
only INDIRECTLY — a `test_*_vs_dm_control` parity test would eventually drift
if an index moved, but it compares OBSERVATIONS, so the failure arrives as a
numeric mismatch in a 50-dimensional vector rather than as "site index 24 is
not the torso any more". Here each literal is compared against what the
model's own `<sensor>` table resolves the sensor's name to, which is the same
number MuJoCo's `mj_name2id` would give.

⚠ IT CHECKS, IT DOES NOT REPLACE. Swapping the literals for runtime lookups
would touch obs-extraction code that a dozen parity tests pin numerically, for
no behaviour change — the literals are right today, and this file is what says
so on every run. The replacement is a separate change with its own risk; the
silent-drift hazard is closed either way.

Run with:
    pixi run mojo run -I . tests/physics3d/test_env_site_literals_match_sensors.mojo
"""

from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.flat_model import FlatModelDef

from mojo_rl.envs.dm_control.hopper.hopper_xml import (
    TOUCH_TOE_SITE_IDX,
    TOUCH_HEEL_SITE_IDX,
    TORSO_BODY_IDX as HOPPER_TORSO_BODY_IDX,
)
from mojo_rl.envs.dm_control.quadruped.quadruped_xml import (
    TORSO_SITE_IDX,
    TOE_SITE_0,
    TORSO_BODY_IDX as QUAD_TORSO_BODY_IDX,
)
from mojo_rl.envs.dm_control.quadruped.quadruped_escape_config import (
    ESCAPE_RF_SITE_0,
    ESCAPE_N_RF,
)
from mojo_rl.envs.dm_control.dog.dog_xml import (
    DOG_SITE_PALM_L,
    DOG_SITE_PALM_R,
    DOG_SITE_SOLE_L,
    DOG_SITE_SOLE_R,
    DOG_SITE_HEAD,
    DOG_TORSO_BODY_IDX,
)


def _load(path: String) raises -> FlatModelDef:
    var xml = String()
    with open(path, "r") as f:
        xml = f.read()
    # Assets resolve against the model file's directory, MuJoCo's own rule.
    var slash = path.rfind("/")
    var base = String(".") if slash < 0 else String(path[byte=0:slash])
    return parse_xml_full(xml, base)


def _check_site(
    fmd: FlatModelDef, sensor: String, literal: Int, label: String,
    mut checked: Int,
) raises:
    var resolved = fmd.sensor_site_by_name(sensor)
    print("  ", label, "literal", literal, " model says", resolved,
          " sensor '" + sensor + "'")
    assert_true(
        resolved == literal,
        label + ": the config uses site index " + String(literal)
        + " but <sensor name='" + sensor + "'> resolves to site "
        + String(resolved) + ". A site was added, removed or reordered and"
        " this env is now reading a DIFFERENT site's sensor.",
    )
    checked += 1


def _check_body(
    fmd: FlatModelDef, sensor: String, literal: Int, label: String,
    mut checked: Int,
) raises:
    var resolved = fmd.sensor_body_by_name(sensor)
    print("  ", label, "literal", literal, " model says", resolved,
          " sensor '" + sensor + "'")
    assert_true(
        resolved == literal,
        label + ": the config uses body index " + String(literal)
        + " but <sensor name='" + sensor + "'> resolves to body "
        + String(resolved),
    )
    checked += 1


def test_hopper_literals() raises:
    print("=== hopper ===")
    var fmd = _load(String("mojo_rl/envs/dm_control/assets/hopper.xml"))
    var n = 0
    _check_site(fmd, String("touch_toe"), TOUCH_TOE_SITE_IDX,
                String("TOUCH_TOE_SITE_IDX"), n)
    _check_site(fmd, String("touch_heel"), TOUCH_HEEL_SITE_IDX,
                String("TOUCH_HEEL_SITE_IDX"), n)
    _check_body(fmd, String("torso_subtreelinvel"), HOPPER_TORSO_BODY_IDX,
                String("TORSO_BODY_IDX"), n)
    assert_true(n == 3, "expected 3 hopper checks, ran " + String(n))


def test_quadruped_literals() raises:
    print("=== quadruped (walk) ===")
    var fmd = _load(
        String("mojo_rl/envs/dm_control/assets/quadruped_walk.xml")
    )
    var n = 0
    # The three IMU sensors all sit on the torso site, so all three must
    # resolve to the same literal — which is itself a check that the config's
    # single `TORSO_SITE` parameter is right for all of them.
    for s in [String("imu_accel"), String("imu_gyro"), String("velocimeter")]:
        _check_site(fmd, s, TORSO_SITE_IDX, String("TORSO_SITE_IDX"), n)
    _check_body(fmd, String("imu_accel"), QUAD_TORSO_BODY_IDX,
                String("TORSO_BODY_IDX"), n)

    # ⚠ THE STRIDE, NOT JUST THE BASE. `quadruped_config` reads the four toe
    # force/torque pairs as `TOE_SITE_0 + t`, so a base that is right with a
    # stride that is not would still read three wrong sites.
    var toes = [
        String("force_toe_front_left"), String("force_toe_front_right"),
        String("force_toe_back_right"), String("force_toe_back_left"),
    ]
    for t in range(len(toes)):
        _check_site(fmd, toes[t], TOE_SITE_0 + t,
                    String("TOE_SITE_0 + ") + String(t), n)
    assert_true(n == 8, "expected 8 quadruped checks, ran " + String(n))


def test_quadruped_escape_rangefinders() raises:
    """The twenty rangefinders — the case the audit names explicitly.

    `quadruped_escape_config` reads `ESCAPE_RF_SITE_0 + i` for twenty
    contiguous sites, and its own comment says a task fragment that inserted a
    site "would otherwise shift every rangefinder silently". Each one is
    checked here against its sensor's name.
    """
    print("=== quadruped escape: 20 rangefinders ===")
    var fmd = _load(
        String("mojo_rl/envs/dm_control/assets/quadruped_escape.xml")
    )
    # ⚠ THE NAMES ARE GRID COORDINATES, NOT A FLAT INDEX. They run
    # rf_00..rf_04, rf_10..rf_14, rf_20..rf_24, rf_30..rf_34 — four rows of
    # five, `rf_<row><col>`. Reading them as a zero-padded counter gives
    # `rf_05`, which does not exist; the config's `ESCAPE_RF_SITE_0 + i` walks
    # the sites in declaration order, which is this grid flattened row-major.
    var n = 0
    for i in range(ESCAPE_N_RF):
        var nm = String("rf_") + String(i // 5) + String(i % 5)
        _check_site(fmd, nm, ESCAPE_RF_SITE_0 + i,
                    String("ESCAPE_RF_SITE_0 + ") + String(i), n)
    assert_true(
        n == ESCAPE_N_RF,
        "expected " + String(ESCAPE_N_RF) + " rangefinder checks, ran "
        + String(n),
    )
    # ⚠ NON-VACUITY: twenty checks of a zero-length loop would also "pass".
    assert_true(ESCAPE_N_RF == 20, "the escape task should declare 20"
                " rangefinders, ESCAPE_N_RF = " + String(ESCAPE_N_RF))


def test_dog_literals() raises:
    print("=== dog (stand/walk) ===")
    var fmd = _load(
        String("mojo_rl/envs/dm_control/assets/dog_stand_walk.xml")
    )
    var n = 0
    _check_site(fmd, String("palm_L"), DOG_SITE_PALM_L,
                String("DOG_SITE_PALM_L"), n)
    _check_site(fmd, String("palm_R"), DOG_SITE_PALM_R,
                String("DOG_SITE_PALM_R"), n)
    _check_site(fmd, String("sole_L"), DOG_SITE_SOLE_L,
                String("DOG_SITE_SOLE_L"), n)
    _check_site(fmd, String("sole_R"), DOG_SITE_SOLE_R,
                String("DOG_SITE_SOLE_R"), n)
    # head carries the accelerometer / velocimeter / gyro triple.
    _check_site(fmd, String("accelerometer"), DOG_SITE_HEAD,
                String("DOG_SITE_HEAD"), n)
    _check_body(fmd, String("torso_linvel"), DOG_TORSO_BODY_IDX,
                String("DOG_TORSO_BODY_IDX"), n)
    assert_true(n == 6, "expected 6 dog checks, ran " + String(n))


def test_a_wrong_literal_would_be_caught() raises:
    """The negative control: `_check_site` must FAIL on a wrong index.

    ⚠ WITHOUT THIS THE WHOLE FILE COULD BE VACUOUS. If `sensor_site_by_name`
    returned the literal it was handed — or if `_check_site` swallowed its
    assertion — every test above would pass while checking nothing.
    """
    print("=== negative control: a planted wrong index is caught ===")
    var fmd = _load(String("mojo_rl/envs/dm_control/assets/hopper.xml"))
    var n = 0
    var caught = False
    try:
        _check_site(fmd, String("touch_toe"), TOUCH_TOE_SITE_IDX + 1,
                    String("PLANTED"), n)
    except:
        caught = True
    assert_true(
        caught,
        "a deliberately wrong site index was NOT caught — every check in this"
        " file is vacuous",
    )
    print("  planted off-by-one caught: ok")

    # And an unknown sensor name must raise rather than return 0.
    var caught2 = False
    try:
        _ = fmd.sensor_site_by_name(String("no_such_sensor"))
    except:
        caught2 = True
    assert_true(caught2, "an unknown sensor name must raise")
    print("  unknown sensor name raises: ok")


def main() raises:
    var suite = TestSuite()
    suite.test[test_hopper_literals]()
    suite.test[test_quadruped_literals]()
    suite.test[test_quadruped_escape_rangefinders]()
    suite.test[test_dog_literals]()
    suite.test[test_a_wrong_literal_would_be_caught]()
    suite^.run()
