"""The scan-and-print list: every accepted-but-unread MJCF attribute says so.

    pixi run mojo run -I . tests/physics3d/test_silent_attrs_are_loud.mojo

`docs/PHYSICS3D_MUJOCO_312_AUDIT.md` found ~30 attributes and elements this
loader accepts and never reads — the model loads, runs, and is quietly not
MuJoCo's. Recommendation 1 of that audit is one list that counts them at
load and prints one line per audit id, raising for the rows that are wrong
physics on a model that loads today. This gate is that list's contract:

  1. a model spelling every silent row reports every audit id;
  2. a clean model reports NOTHING — the row that lands must be deleted from
     the scan, or this test goes red on it (a warning that outlives its gap
     teaches people to ignore warnings);
  3. the wrong-physics spellings RAISE, naming their audit id;
  4. AUD-01, the one row FIXED rather than made loud: `jnt_limited` resolves
     as MuJoCo's `islimited()` does, gated against the 3.12 runtime.
"""

from std.testing import assert_true, TestSuite
from std.python import Python

from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.parser.expander import expand_mjcf


# One of every silent row. It must LOAD (nothing here raises) — the point is
# the report, not a refusal.
comptime SILENT_XML = String(
    """
<mujoco model="every silent row">
  <compiler alignfree="true" balanceinertia="true" discardvisual="true" fusestatic="true"/>
  <option wind="1 0 0" magnetic="0 0 1" o_margin="0.01" sleep_tolerance="0.1" actuatorgroupdisable="1">
    <flag clampctrl="disable" actuation="disable" filterparent="disable" refsafe="disable"
          spring="disable" passive="enable" autoreset="disable" nativeccd="disable"
          override="enable" energy="enable" diagexact="enable"/>
  </option>
  <statistic meaninertia="2"/>
  <asset>
    <mesh name="inline" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
    <hfield name="hf" nrow="2" ncol="2" size="1 1 0.1 0.01" elevation="0 0 0 1"/>
  </asset>
  <default>
    <pair condim="6"/>
    <default class="p">
      <pair solref="0.005 1"/>
    </default>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" shellinertia="true" fitscale="2"
          adhesion="1" surfacevel="1 0 0 0 0 0"/>
    <body name="a" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0" range="-1 1" margin="0.1" stiffness="1 2" damping="3 4"/>
      <geom name="ga" type="sphere" size="0.1" fluidshape="ellipsoid" fluidcoef="0.5 0.25 1.5 1 1"/>
      <body name="b" pos="0 0 0.5">
        <joint name="j2" type="ball" range="0 30"/>
        <geom name="gb" type="sphere" size="0.1"/>
      </body>
    </body>
  </worldbody>
  <contact>
    <pair geom1="ga" geom2="gb" class="p" adhesion="2" solreffriction="0.01 1"/>
  </contact>
  <tendon>
    <fixed name="t" damping="3 0.4" stiffness="2 0.7" frictionloss="0.5" armature="0.1"
           solreffriction="0.02 1" solimpfriction="0.9 0.95 0.001" actuatorfrcrange="-1 1">
      <joint joint="j1" coef="1"/>
    </fixed>
  </tendon>
  <actuator>
    <general name="m" joint="j1" actlimited="true" actrange="-1 1" actearly="true" delay="0.01" nsample="4" damping="1" armature="2"/>
    <position name="s" joint="j1" kp="10" timeconst="0.05"/>
  </actuator>
  <deformable>
    <flex name="cloth" dim="2"/>
  </deformable>
</mujoco>
"""
)

comptime CLEAN_XML = String(
    """
<mujoco model="clean">
  <option timestep="0.002" integrator="implicitfast">
    <flag contact="disable" multiccd="disable"/>
  </option>
  <default>
    <joint limited="true" range="-1 1" damping="0.1"/>
    <geom type="capsule" size="0.05 0.2" friction="1 0.005 0.0001" margin="0.001" gap="0.001"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="a" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0" stiffness="1" springref="0.1"/>
      <geom name="ga"/>
      <site name="s1" pos="0 0 0.2"/>
    </body>
  </worldbody>
  <equality>
    <connect body1="a" body2="world" anchor="0 0 1" active="true"/>
  </equality>
  <tendon>
    <fixed name="t" stiffness="1" springlength="0" limited="true" range="-1 1">
      <joint joint="j1" coef="1"/>
    </fixed>
  </tendon>
  <actuator>
    <motor name="m" joint="j1" gear="2" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""
)


def _one_joint(joint_line: String) -> String:
    return (
        String(
            """
<mujoco model="one">
  <worldbody>
    <body name="a" pos="0 0 1">
      """
        )
        + joint_line
        + String(
            """
      <geom name="ga" type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""
        )
    )


comptime RAISE_DYNTYPE = String(
    """
<mujoco model="dyntype">
  <worldbody>
    <body name="a" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom name="ga" type="sphere" size="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <general name="m" joint="j1" dyntype="muscle" dynprm="0.01 0.04"/>
  </actuator>
</mujoco>
"""
)

comptime RAISE_ACTIVE = String(
    """
<mujoco model="active">
  <worldbody>
    <body name="a" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom name="ga" type="sphere" size="0.1"/>
    </body>
  </worldbody>
  <equality>
    <weld body1="a" body2="world" active="false"/>
  </equality>
</mujoco>
"""
)

comptime RAISE_TENDON_MARGIN = String(
    """
<mujoco model="tendon margin">
  <worldbody>
    <body name="a" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom name="ga" type="sphere" size="0.1"/>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t" limited="true" range="-1 0.3" margin="0.2">
      <joint joint="j1" coef="1"/>
    </fixed>
  </tendon>
</mujoco>
"""
)

comptime RAISE_ATTACH_FRAME = String(
    """
<mujoco model="attach frame">
  <asset>
    <model name="sub" file="does_not_matter.xml"/>
  </asset>
  <worldbody>
    <attach model="sub" frame="f" prefix="s_"/>
  </worldbody>
</mujoco>
"""
)

# AUD-01: four spellings of `limited`, all resolved by MuJoCo's `islimited()`.
comptime LIMITED_XML = String(
    """
<mujoco model="limited">
  <default>
    <default class="unlim">
      <joint limited="false" range="-60 60"/>
    </default>
    <default class="ranged">
      <joint range="-30 30"/>
    </default>
    <default class="degenerate">
      <joint range="0 0"/>
    </default>
  </default>
  <worldbody>
    <body name="a" pos="0 0 1">
      <joint name="class_says_false" class="unlim" type="hinge" axis="0 1 0"/>
      <geom name="ga" type="sphere" size="0.1"/>
      <body name="b" pos="0 0 0.2">
        <joint name="class_range_auto" class="ranged" type="hinge" axis="0 1 0"/>
        <geom name="gb" type="sphere" size="0.1"/>
        <body name="c" pos="0 0 0.2">
          <joint name="degenerate_range" class="degenerate" type="hinge" axis="0 1 0"/>
          <geom name="gc" type="sphere" size="0.1"/>
          <body name="d" pos="0 0 0.2">
            <joint name="element_range" type="hinge" axis="0 1 0" range="-1 1"/>
            <geom name="gd" type="sphere" size="0.1"/>
            <body name="e" pos="0 0 0.2">
              <joint name="element_false_class_range" class="ranged" type="hinge" axis="0 1 0" limited="false"/>
              <geom name="ge" type="sphere" size="0.1"/>
              <body name="f" pos="0 0 0.2">
                <joint name="slide_reversed" type="slide" axis="0 0 1" range="1 -1"/>
                <geom name="gf" type="sphere" size="0.1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
)


def _has(ids: List[String], aud: String) -> Bool:
    for i in range(len(ids)):
        if ids[i] == aud:
            return True
    return False


def _parse_raises_naming(xml: String, needle: String) -> Bool:
    try:
        _ = parse_xml_full(xml, String("."))
        return False
    except e:
        var msg = String(e)
        if msg.find(needle) == -1:
            print("  raised, but not with", needle, ":", msg)
            return False
        return True


def test_every_silent_row_reports_its_audit_id() raises:
    print("=== one of every silent row: each audit id reported ===")
    var fmd = parse_xml_full(SILENT_XML, String("."))
    print("  silent_attrs =", fmd.silent_attrs, " ids =", len(fmd.silent_attr_ids))
    var want: List[String] = [
        String("AUD-08"), String("AUD-12"),
        String("AUD-15"),
        # ⚠ AUD-21 IS GONE FROM THIS LIST ON PURPOSE, like AUD-27 and AUD-34
        # below. `<position timeconst>` is READ as of 2026-09-13 — it
        # resolves to `dyntype=filterexact` with `dynprm[0] = timeconst`,
        # gated by `test_actuator_dyntype_vs_mujoco` — so a scan that still
        # reported it would name a defect that no longer exists. The fixture
        # keeps its `timeconst=` attribute so a regression that silently
        # stopped reading it has to come back through that gate.
        #
        # ⚠ AUD-02 BELOW IS STILL LIVE BUT NARROWER: `actlimited` and
        # `actrange` are read now (the `mj_nextActivation` clamp) and
        # `actearly` is not, which is what the fixture's `actearly="true"`
        # keeps firing.
        # ⚠ AUD-14 IS GONE FROM THIS LIST ON PURPOSE, like AUD-27 and AUD-34:
        # `<hfield elevation>` is READ as of 2026-09-13 and gated by
        # `test_hfield_elevation_vs_mujoco`.
        # AUD-23 is GONE from this list on purpose: `<sensor>` is parsed now,
        # and an element this engine does not model RAISES rather than being
        # counted. Its coverage moved to `test_sensor_table_vs_mujoco`.
        String("AUD-24"), String("AUD-25"), String("AUD-26"),
        # ⚠ AUD-27 IS GONE FROM THIS LIST ON PURPOSE. `<option wind>` is
        # READ as of 2026-09-13 — it reaches the fluid model, gated by
        # `test_wind_vs_mujoco` — so a scan that still reported it would be
        # naming a defect that no longer exists. The fixture below keeps its
        # `wind=` attribute so that a regression which silently stopped
        # reading it would have to come back through that gate.
        String("AUD-28"), String("AUD-37"), String("AUD-02"),
        # ⚠ AUD-34 IS GONE FROM THIS LIST ON PURPOSE, like AUD-27 above:
        # `<flag filterparent="disable">` is READ as of 2026-09-13 and gated
        # by `test_filterparent_vs_mujoco`. The fixture keeps the attribute
        # so a regression that stopped reading it has to come back through
        # that gate.
        # AUD-54 — the 3.12 POLYNOMIAL stiffness/damping, which the fixture's
        # `<fixed damping="3 0.4" stiffness="2 0.7">` states. Unlike its
        # neighbours in this list the ATTRIBUTE is read: what is dropped is
        # the higher coefficient, so the row has to be counted separately
        # from the AUD-08 "not read at all" family beside it.
        String("AUD-54"),
    ]
    for i in range(len(want)):
        assert_true(
            _has(fmd.silent_attr_ids, want[i]),
            "the scan did not report " + want[i] + " on a model that spells"
            " its attribute — that row is silent again",
        )
    # every declaration in the fixture, at least once each
    assert_true(
        fmd.silent_attrs >= 40,
        "expected at least 40 silent declarations counted, got "
        + String(fmd.silent_attrs),
    )
    # the 3.11/3.12 actuator elements are counted with the older ones
    var fmd2 = parse_xml_full(
        RAISE_DYNTYPE.replace(
            String("<general name=\"m\" joint=\"j1\" dyntype=\"muscle\" dynprm=\"0.01 0.04\"/>"),
            String("<motor name=\"m\" joint=\"j1\"/><pid name=\"p\" joint=\"j1\" kp=\"5\"/><dcmotor name=\"d\" joint=\"j1\"/>"),
        ),
        String("."),
    )
    assert_true(
        fmd2.unmodelled_actuators == 2 and len(fmd2.actuators) == 1,
        "<pid> and <dcmotor> must be COUNTED as unmodelled (got "
        + String(fmd2.unmodelled_actuators) + ", nact "
        + String(len(fmd2.actuators)) + ")",
    )


def test_a_clean_model_reports_nothing() raises:
    """⚠ THE ROW THAT RETIRES ITSELF. Every attribute in CLEAN_XML is one the
    loader reads. If a landed feature is still on the scan, this is where it
    shows."""
    print("=== a clean model: zero silent rows ===")
    var fmd = parse_xml_full(CLEAN_XML, String("."))
    print("  silent_attrs =", fmd.silent_attrs)
    assert_true(
        fmd.silent_attrs == 0 and len(fmd.silent_attr_ids) == 0,
        "a model using only modelled attributes tripped the silent-attr"
        " scan: " + String(fmd.silent_attrs) + " hit(s)",
    )


comptime RAISE_EQ_ACCEL_SENSOR = String(
    """<mujoco model="eq_plus_accel_sensor">
  <compiler angle="radian"/>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
      <site name="s" pos="0.2 0 0"/>
    </body>
    <body name="b2" pos="0 0.4 1">
      <joint name="j2" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
    </body>
  </worldbody>
  <equality>
    <connect body1="b1" body2="b2" anchor="0 0 0"/>
  </equality>
  <sensor>
    <force name="f" site="s"/>
  </sensor>
</mujoco>
"""
)


def test_wrong_physics_rows_raise() raises:
    print("=== the wrong-physics spellings refuse to load, by audit id ===")
    # ⚠ `muscle`, NOT `integrator` — THE REFUSAL WAS NARROWED, NOT REMOVED.
    # `integrator`, `filter` and `filterexact` are modelled as of 2026-09-13
    # (AUD-02/AUD-21, `dynamics/activation.next_activation`); this fixture
    # moved to a dyntype that is still a DIFFERENT ODE rather than a
    # different integration of the same one. The arm below is what keeps the
    # narrowing honest.
    assert_true(
        _parse_raises_naming(RAISE_DYNTYPE, String("AUD-02")),
        "dyntype=\"muscle\" loaded as a stateless actuator",
    )
    # ...and the three that ARE modelled must LOAD, each with an activation.
    # Without this the refusal above could be satisfied by refusing every
    # dyntype, which is what it used to do.
    for _dt in [String("integrator"), String("filter"),
                String("filterexact")]:
        var _ok = parse_xml_full(
            RAISE_DYNTYPE.replace(String("muscle"), _dt).replace(
                String("0.01 0.04"), String("0.1")
            ),
            String("."),
        )
        assert_true(
            _ok.na == 1,
            "dyntype=\"" + _dt + "\" built " + String(_ok.na)
            + " activations, not 1 — a modelled dyntype that owns no `act`"
            " takes its force from `ctrl`, which is AUD-02's defect",
        )
    # AUD-04 LANDED 2026-09-12: an inactive equality loads and builds nothing
    # (gated vs MuJoCo in test_equality_active_pair_class_frame_vs_mujoco).
    var inactive = parse_xml_full(RAISE_ACTIVE, String("."))
    assert_true(
        len(inactive.equalities) == 0 and inactive.inactive_equalities == 1,
        "<weld active=\"false\"> must be dropped and counted",
    )
    # AUD-29 LANDED 2026-09-12: a direct-form solreflimit loads and is kept
    # as written (gated vs MuJoCo in test_joint_margin_and_direct_solref_vs_mujoco).
    var direct = parse_xml_full(
        _one_joint(
            String(
                "<joint name=\"j1\" type=\"hinge\" axis=\"0 1 0\""
                " range=\"-1 1\" solreflimit=\"-1000 -50\"/>"
            )
        ),
        String("."),
    )
    assert_true(
        direct.joints[0].solref_limit_0 == -1000.0
        and direct.joints[0].solref_limit_1 == -50.0,
        "a direct-form solreflimit was not kept as written",
    )
    # AUD-38 LANDED 2026-09-12: a limited tendon with margin loads.
    _ = parse_xml_full(RAISE_TENDON_MARGIN, String("."))
    # ball-joint springs LANDED 2026-09-12 (gated vs MuJoCo in
    # test_ball_spring_and_tendon_damping_vs_mujoco); a FREE-joint spring
    # still needs the body's XML pose and is refused.
    assert_true(
        _parse_raises_naming(
            _one_joint(
                String("<joint name=\"j1\" type=\"free\" stiffness=\"10\"/>")
            ),
            String("AUD-43"),
        ),
        "a free-joint spring loaded without its spring pose",
    )
    var attach_raised = False
    try:
        _ = expand_mjcf(RAISE_ATTACH_FRAME, String("."))
    except e:
        attach_raised = String(e).find("AUD-13") != -1
        if not attach_raised:
            print("  attach raised without AUD-13:", String(e))
    assert_true(attach_raised, "<attach frame=...> was spliced whole")


def test_a_sensor_under_a_closed_loop_is_served_now() raises:
    """AUD-48, CLOSED. Every sensor under a connect/weld is served.

    ⚠⚠ THIS TEST USED TO ASSERT THE OPPOSITE, AND BOTH HALVES OF THAT WERE
    WRONG FOR DIFFERENT REASONS:

      * the ACCELEROMETER never read `cfrc_ext` at all
        (`mj_objectAcceleration`, engine_core_util.c:909), so unserving it was
        over-firing;
      * the FORCE and TORQUE sensors do read it, through `cfrc_int`, and the
        term they were missing is now computed — the Newton solver retains its
        equality row forces in `Data.efc_eq_force` and `_cfrc_ext_env` walks
        them with MuJoCo's own cursor.

    ⚠ THE DECISION IS NO LONGER AT LOAD, AND COULD NOT HAVE BEEN. PGS, CG and
    the island solver still discard those forces, and a parser cannot see
    which solver an integrator will be instantiated with. `_cfrc_ext_env`
    compares the row count the step's solver retained
    (`META_IDX_EQ_FORCE_LIVE`) against what the equality table implies and
    NaNs the affected bodies when they disagree — loud on exactly the sensors
    that would be wrong. `test_closed_loop_sensors_vs_mujoco` is where the
    values are checked against MuJoCo.
    """
    print("=== AUD-48: everything under a <connect> is served ===")
    var fmd = parse_xml_full(RAISE_EQ_ACCEL_SENSOR, String("."))
    assert_true(len(fmd.sensors) == 1, "expected one sensor")
    print("  force sensor served =", fmd.sensors[0].served, " (want 1)")
    assert_true(
        fmd.sensors[0].served,
        "the force sensor under a connect is computed now and must be served",
    )
    assert_true(
        not _has(fmd.silent_attr_ids, String("AUD-48")),
        "AUD-48 is implemented; the loader must not report a row for it",
    )

    print("  control: the same connect with an ACCELEROMETER")
    var accel = parse_xml_full(
        RAISE_EQ_ACCEL_SENSOR.replace(
            String("<force name=\"f\" site=\"s\"/>"),
            String("<accelerometer name=\"a\" site=\"s\"/>"),
        ),
        String("."),
    )
    assert_true(
        len(accel.sensors) == 1 and accel.sensors[0].served,
        "an accelerometer under a connect must be served",
    )

    print("  control: the same sensor under a JOINT equality")
    var joint_eq = parse_xml_full(
        RAISE_EQ_ACCEL_SENSOR.replace(
            String("<connect body1=\"b1\" body2=\"b2\" anchor=\"0 0 0\"/>"),
            String("<joint joint1=\"j1\" joint2=\"j2\"/>"),
        ),
        String("."),
    )
    assert_true(
        len(joint_eq.equalities) == 1,
        "the joint-equality control did not parse its equality",
    )
    assert_true(
        joint_eq.sensors[0].served,
        "a JOINT equality contributes nothing to `cfrc_ext` and never did;"
        " the sensor under it must stay served",
    )


def test_jnt_limited_resolves_like_mujoco() raises:
    """AUD-01 — `limited` through the class chain, and \"auto\" is
    `range[0] < range[1]` (user_objects.cc:185-187), against 3.12."""
    print("=== AUD-01: jnt_limited vs MuJoCo on six spellings ===")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(LIMITED_XML)
    var fmd = parse_xml_full(LIMITED_XML, String("."))
    var nj = Int(py=m.njnt)
    assert_true(
        nj == 6 and len(fmd.joints) == 6,
        "fixture must expose six joints (MuJoCo " + String(nj) + ", ours "
        + String(len(fmd.joints)) + ")",
    )
    for j in range(nj):
        var mj_lim = Int(py=m.jnt_limited[j]) != 0
        var ours = fmd.joints[j].is_limited
        var lo = Float64(py=m.jnt_range[j][0])
        var hi = Float64(py=m.jnt_range[j][1])
        print(
            "  ", fmd.joint_names[j], " mujoco limited=", mj_lim, " ours=",
            ours, " mj range=", lo, hi, " ours=", fmd.joints[j].range_min,
            fmd.joints[j].range_max,
        )
        assert_true(
            mj_lim == ours,
            "jnt_limited differs from MuJoCo on `" + fmd.joint_names[j] + "`",
        )
        if mj_lim:
            assert_true(
                abs(fmd.joints[j].range_min - lo) < 1e-9
                and abs(fmd.joints[j].range_max - hi) < 1e-9,
                "range differs from MuJoCo on `" + fmd.joint_names[j] + "`",
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
