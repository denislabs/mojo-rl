"""The `<sensor>` front end's TABLE vs a live MuJoCo 3.12 `MjModel` (AUD-23).

`parser/full_parser._fill_sensors` resolves every `<sensor>` element into a
`SensorData` row whose columns are MuJoCo's own: `sensor_type`,
`sensor_objtype`, `sensor_objid`, `sensor_dim`, `sensor_adr`,
`sensor_datatype`, `sensor_needstage`, `sensor_cutoff`. This gate reads all
eight off a live `MjModel` built from the SAME XML and compares them column by
column — so the enum values, the per-type dim/datatype/stage rules and the
declaration-order address layout are all pinned to the runtime rather than to
a transcription of `user_objects.cc`.

⚠⚠ WHY A TABLE GATE AND NOT A VALUE GATE. The readings themselves already
have gates (`test_rne_post_sensors_vs_mujoco`, `test_subtree_linvel_vs_mujoco`,
`test_rangefinder_vs_mujoco`) and those compare NUMBERS. What had no gate, and
what AUD-23 is actually about, is the ADDRESSING: which site a named sensor
reads, and where its values land in `sensordata`. That is what the environment
configs used to encode as hand-counted literals — `TOUCH_TOE_SITE_IDX = 0`,
`TORSO_SITE_IDX = 24`, `ESCAPE_RF_SITE_0 = 3` — each derived by counting sites
in worldbody DFS order and pinned only indirectly. A wrong index there does not
crash; it reads a different site's sensor, which is a plausible number.

⚠ THE THREE NEGATIVE TESTS ARE THE POINT OF THE OTHER HALF. An unmodelled
sensor element, a 3.12 delay attribute and a camera-form rangefinder must all
REFUSE the model. Accepting any of them would return a value of the right
shape: an unmodelled element shifts every later `adr`, a dropped `delay`
returns the undelayed reading, and a dropped `camera`/`data` truncates the
sensor's dim.

Run with:
    pixi run mojo run -I . tests/physics3d/test_sensor_table_vs_mujoco.mojo
"""

from std.python import Python, PythonObject
from std.math import abs
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full


# A model exercising all eight modelled elements, both attachment kinds, a
# cutoff on each datatype, and — deliberately — sites declared BOTH in the
# worldbody and inside bodies, because that is the ordering rule
# (`_stable_group_by_body_sites`) that makes a hand-counted index wrong.
comptime SENSOR_XML = """
<mujoco model="sensor_table">
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="torso" pos="0 0 1">
      <freejoint name="root"/>
      <geom name="tgeom" type="box" size=".1 .1 .1"/>
      <site name="imu" pos="0 0 .1" size=".05"/>
      <site name="rf" pos="0 0 -.1" size=".02"/>
      <body name="shin" pos=".3 0 0">
        <joint name="knee" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom name="sgeom" type="capsule" fromto="0 0 0 .2 0 0" size=".03"/>
        <site name="foot" pos=".2 0 0" size=".04"/>
      </body>
    </body>
    <!-- ⚠ DECLARED LAST IN THE TEXT, NUMBERED FIRST BY MuJoCo. The worldbody
         owns this site, and MuJoCo emits sites grouped by body with body 0's
         first — so text order is (imu, rf, foot, world_ref) and element order
         is (world_ref, imu, rf, foot). A hand-counted index reads the text. -->
    <site name="world_ref" pos="0 0 2" size=".05"/>
  </worldbody>
  <sensor>
    <touch name="foot_touch" site="foot" cutoff="50"/>
    <accelerometer name="imu_acc" site="imu"/>
    <velocimeter name="imu_vel" site="imu" cutoff="7.5"/>
    <gyro name="imu_gyro" site="imu"/>
    <force name="foot_force" site="foot"/>
    <torque name="foot_torque" site="foot"/>
    <rangefinder name="down" site="rf"/>
    <subtreelinvel name="torso_vel" body="torso"/>
  </sensor>
</mujoco>
"""

def _names() -> List[String]:
    """The fixture's sensor names in declaration order.

    A function rather than a `comptime` array: a `comptime Array[String, N]`
    is not `ImplicitlyCopyable`, so it cannot be materialised into a runtime
    loop.
    """
    return [
        String("foot_touch"), String("imu_acc"), String("imu_vel"),
        String("imu_gyro"), String("foot_force"), String("foot_torque"),
        String("down"), String("torso_vel"),
    ]


def _mj_model(mujoco: PythonObject, xml: String) raises -> PythonObject:
    return mujoco.MjModel.from_xml_string(PythonObject(xml))


def test_sensor_table_matches_mujoco() raises:
    print("=== <sensor> table vs MuJoCo, column by column ===")
    var mujoco = Python.import_module("mujoco")
    print("  mujoco", String(mujoco.__version__))
    var m = _mj_model(mujoco, SENSOR_XML)
    var fmd = parse_xml_full(SENSOR_XML, String("."))
    var names = _names()

    var n_mj = Int(py=m.nsensor)
    assert_true(
        len(fmd.sensors) == n_mj,
        "sensor COUNT: ours " + String(len(fmd.sensors)) + " vs MuJoCo "
        + String(n_mj),
    )
    assert_true(n_mj == 8, "fixture should declare 8 sensors, MuJoCo saw "
                + String(n_mj))

    print(
        "  name          type objtype objid dim  adr dtype stage  cutoff"
        "   (ours == MuJoCo)"
    )
    var compared = 0
    for i in range(n_mj):
        var sd = fmd.sensors[i]
        var mj_type = Int(py=m.sensor_type[i])
        var mj_objtype = Int(py=m.sensor_objtype[i])
        var mj_objid = Int(py=m.sensor_objid[i])
        var mj_dim = Int(py=m.sensor_dim[i])
        var mj_adr = Int(py=m.sensor_adr[i])
        var mj_dtype = Int(py=m.sensor_datatype[i])
        var mj_stage = Int(py=m.sensor_needstage[i])
        var mj_cut = Float64(py=m.sensor_cutoff[i])

        print(
            "  ", names[i], sd.sensor_type, sd.objtype, sd.objid, sd.dim,
            sd.adr, sd.datatype, sd.needstage, sd.cutoff,
        )

        assert_true(
            fmd.sensor_names[i] == names[i],
            "sensor " + String(i) + " NAME: ours '" + fmd.sensor_names[i]
            + "' vs expected '" + names[i] + "'",
        )
        assert_true(
            sd.sensor_type == mj_type,
            names[i] + " type: ours " + String(sd.sensor_type) + " vs MuJoCo "
            + String(mj_type),
        )
        assert_true(
            sd.objtype == mj_objtype,
            names[i] + " objtype: ours " + String(sd.objtype) + " vs MuJoCo "
            + String(mj_objtype),
        )
        # ⚠ THE COLUMN THE HAND-COUNTED LITERALS GOT WRONG. `sensor_objid` is
        # a SITE index for seven of these and a BODY index for the eighth, and
        # both orderings are MuJoCo's grouped-by-body ones, not XML text order.
        assert_true(
            sd.objid == mj_objid,
            names[i] + " objid: ours " + String(sd.objid) + " vs MuJoCo "
            + String(mj_objid) + " — this is the addressing bug AUD-23 is"
            " about; a wrong index here reads a different site's sensor",
        )
        assert_true(
            sd.dim == mj_dim,
            names[i] + " dim: ours " + String(sd.dim) + " vs MuJoCo "
            + String(mj_dim),
        )
        assert_true(
            sd.adr == mj_adr,
            names[i] + " adr: ours " + String(sd.adr) + " vs MuJoCo "
            + String(mj_adr),
        )
        assert_true(
            sd.datatype == mj_dtype,
            names[i] + " datatype: ours " + String(sd.datatype)
            + " vs MuJoCo " + String(mj_dtype)
            + " — AUD-47 recorded rangefinder as POSITIVE and it is REAL",
        )
        assert_true(
            sd.needstage == mj_stage,
            names[i] + " needstage: ours " + String(sd.needstage)
            + " vs MuJoCo " + String(mj_stage),
        )
        assert_true(
            abs(sd.cutoff - mj_cut) < 1e-12,
            names[i] + " cutoff: ours " + String(sd.cutoff) + " vs MuJoCo "
            + String(mj_cut),
        )
        compared += 1

    # ⚠ ROWS COMPARED, PRINTED BESIDE ROWS DIFFERING. "0 mismatches" over an
    # empty loop is the default failure mode of a gate like this one.
    print("  rows compared:", compared, " differing: 0")
    assert_true(compared == 8, "expected 8 rows compared, got "
                + String(compared))

    var mj_nsd = Int(py=m.nsensordata)
    assert_true(
        fmd.nsensordata() == mj_nsd,
        "nsensordata: ours " + String(fmd.nsensordata()) + " vs MuJoCo "
        + String(mj_nsd),
    )
    print("  nsensordata:", fmd.nsensordata(), "== MuJoCo", mj_nsd)


def test_lookup_by_name_beats_a_counted_index() raises:
    """The name API resolves to the same site MuJoCo's `mj_name2id` does.

    This is the test that would have caught a shifted site table: it asks for
    the sensor BY NAME and compares the site it lands on against MuJoCo's, for
    every sensor in the fixture. The fixture declares the worldbody's own site
    LAST in the text, where MuJoCo numbers it FIRST, so text order and element
    order genuinely disagree — and that disagreement is asserted below rather
    than assumed.
    """
    print("=== sensor -> site/body by NAME vs mj_name2id ===")
    var mujoco = Python.import_module("mujoco")
    var m = _mj_model(mujoco, SENSOR_XML)
    var fmd = parse_xml_full(SENSOR_XML, String("."))
    var names = _names()

    # ⚠ THE ORDERING TRAP, ASSERTED RATHER THAN ASSUMED. `world_ref` is the
    # LAST site in the XML text and the FIRST in MuJoCo's numbering, because
    # sites are grouped by body and the worldbody is body 0. If this ever
    # stopped being true the fixture would still pass every column above while
    # no longer testing the thing it was built to test.
    var mj_world_ref = Int(py=
        mujoco.mj_name2id(m, Int(py=mujoco.mjtObj.mjOBJ_SITE.value),
                          PythonObject(String("world_ref")))
    )
    var mj_imu = Int(py=
        mujoco.mj_name2id(m, Int(py=mujoco.mjtObj.mjOBJ_SITE.value),
                          PythonObject(String("imu")))
    )
    print("  mj site id: world_ref =", mj_world_ref, " imu =", mj_imu,
          " (world_ref is LAST in the XML text)")
    assert_true(
        mj_world_ref == 0 and mj_imu == 1,
        "the fixture no longer exercises the grouped-by-body site ordering:"
        " world_ref is declared last in the text and must be site 0, imu"
        " must be 1; got " + String(mj_world_ref) + " and " + String(mj_imu),
    )

    var checked = 0
    for i in range(len(names)):
        var nm = names[i]
        var idx = fmd.sensor_index_by_name(nm)
        assert_true(idx == i, "sensor_index_by_name('" + nm + "') = "
                    + String(idx) + ", want " + String(i))
        var mj_objid = Int(py=m.sensor_objid[i])
        if fmd.sensors[i].objtype == 6:  # SENSOBJ_SITE
            var ours = fmd.sensor_site_by_name(nm)
            assert_true(
                ours == mj_objid,
                nm + ": sensor_site_by_name -> " + String(ours)
                + " but MuJoCo's sensor_objid is " + String(mj_objid),
            )
        else:
            var ours_b = fmd.sensor_body_by_name(nm)
            assert_true(
                ours_b == mj_objid,
                nm + ": sensor_body_by_name -> " + String(ours_b)
                + " but MuJoCo's sensor_objid is " + String(mj_objid),
            )
        var ours_adr = fmd.sensor_adr_by_name(nm)
        assert_true(
            ours_adr == Int(py=m.sensor_adr[i]),
            nm + ": sensor_adr_by_name -> " + String(ours_adr) + " vs MuJoCo "
            + String(Int(py=m.sensor_adr[i])),
        )
        checked += 1
    print("  names resolved and cross-checked:", checked)
    assert_true(checked == 8, "expected 8 names checked")

    # A name that is not there must RAISE, not return 0 — the whole point of
    # replacing a literal with a lookup.
    var raised = False
    try:
        _ = fmd.sensor_site_by_name(String("no_such_sensor"))
    except:
        raised = True
    assert_true(raised, "an unknown sensor name must raise, not return a"
                " sentinel index")
    print("  unknown name raises: ok")

    # The body-attached sensor has no site index to give, and saying so beats
    # handing back `objid` as if it were one.
    var raised2 = False
    try:
        _ = fmd.sensor_site_by_name(String("torso_vel"))
    except:
        raised2 = True
    assert_true(raised2, "sensor_site_by_name on a body-attached sensor must"
                " raise rather than return its body index")
    print("  site lookup on a body sensor raises: ok")


def _refuses(xml: String) -> Bool:
    try:
        var fmd = parse_xml_full(xml, String("."))
        _ = len(fmd.sensors)
        return False
    except:
        return True


def test_unservable_sensors_refuse_to_load() raises:
    print("=== the refusals: unmodelled, 3.12 delay, camera rangefinder ===")

    comptime HEAD = """
<mujoco>
  <worldbody>
    <body name="b"><freejoint name="root"/><geom type="box" size=".1 .1 .1"/>
    <site name="s" size=".05"/></body>
  </worldbody>
  <sensor>
"""
    comptime TAIL = """
  </sensor>
</mujoco>
"""
    # Each row: one <sensor> child that must refuse the whole model.
    var rows: List[String] = [
        # data-dependent width — the one family that still refuses, because a
        # row for it would have to GUESS `dim` and corrupt every later `adr`.
        String('<user name="u" dim="3" objtype="site" objname="s"'
               ' needstage="pos"/>'),
        # 3.12's delay / interval path (changelog 6419534b)
        String('<gyro name="g" site="s" delay="0.01"/>'),
        String('<gyro name="g" site="s" interval="0.02"/>'),
        String('<gyro name="g" site="s" nsample="4"/>'),
        # 3.12's rangefinder forms (9d646e65, ed15493a)
        String('<rangefinder name="r" camera="cam"/>'),
        String('<rangefinder name="r" site="s" data="dist normal"/>'),
        # a reference that does not resolve
        String('<gyro name="g" site="nope"/>'),
        String('<subtreelinvel name="sl" body="nope"/>'),
        String('<jointpos name="jp" joint="nope"/>'),
        # a modelled element missing its object
        String('<touch name="t"/>'),
        String('<jointvel name="jv"/>'),
        # ⚠ A JOINT SENSOR ON A MULTI-DOF JOINT. MuJoCo's own compiler
        # refuses this (`user_objects.cc:7902`, "joint must be slide or hinge
        # in sensor") — the reading is one scalar and a freejoint has no
        # single qpos to report. HEAD's body carries an unnamed freejoint, so
        # this row names its own.
        String('<jointpos name="jp" joint="root"/>'),
    ]
    var refused = 0
    for i in range(len(rows)):
        var xml = String(HEAD) + rows[i] + String(TAIL)
        var r = _refuses(xml)
        print("  ", "REFUSED " if r else "ACCEPTED", rows[i])
        assert_true(
            r,
            "this <sensor> child must refuse the model, and it loaded: "
            + rows[i],
        )
        refused += 1
    print("  rows that must refuse:", len(rows), " refused:", refused)

    # ⚠ NON-VACUITY. If the harness refused everything — a typo in HEAD, say —
    # every row above would "pass". The same skeleton with a SERVED sensor has
    # to load.
    var ok_xml = (
        String(HEAD) + String('<gyro name="g" site="s"/>') + String(TAIL)
    )
    assert_true(
        not _refuses(ok_xml),
        "the control model with a served <gyro> must LOAD; if it does not,"
        " every refusal above is vacuous",
    )
    print("  control (a served <gyro>) loads: ok — the refusals are not vacuous")


# A model whose sensor list INTERLEAVES served and unserved elements, so an
# unserved row sits between two served ones and its width has to be right for
# the later one's `adr` to be. This is the shape `dog`, `swimmer`, `finger`
# and `quadruped` actually have.
comptime MIXED_XML = """
<mujoco model="mixed_sensors">
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="torso" pos="0 0 1">
      <freejoint name="root"/>
      <geom name="tgeom" type="box" size=".1 .1 .1"/>
      <site name="imu" pos="0 0 .1" size=".05"/>
      <body name="shin" pos=".3 0 0">
        <joint name="knee" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom name="sgeom" type="capsule" fromto="0 0 0 .2 0 0" size=".03"/>
        <site name="foot" pos=".2 0 0" size=".04"/>
      </body>
    </body>
  </worldbody>
  <sensor>
    <gyro name="g" site="imu"/>
    <framequat name="fq" objtype="site" objname="imu"/>
    <jointpos name="jp" joint="knee"/>
    <touch name="t" site="foot"/>
    <subtreeangmom name="sam" body="torso"/>
    <framexaxis name="fx" objtype="site" objname="foot"/>
    <velocimeter name="v" site="imu"/>
  </sensor>
</mujoco>
"""


def test_unserved_sensors_are_addressed_not_dropped() raises:
    """An unserved sensor holds its slot, so later `adr`s stay MuJoCo-exact.

    ⚠ THIS IS THE TEST THAT DISTINGUISHES THE THREE POSSIBLE DESIGNS. Dropping
    the unserved `subtreeangmom` would leave `framexaxis`'s `adr` at 9 instead
    of 12 and `velocimeter`'s at 12 instead of 15 — plausible numbers pointing
    at another sensor's values. Refusing the model would regress four
    dm_control models. Addressing it keeps every offset right and makes the
    READ raise instead.

    ⚠ THE COUNTS BELOW MOVE AS THE TAIL LANDS, AND THEY ARE PINNED ANYWAY.
    `jointpos` was served on 2026-09-13 and the frame family the same day
    (audit §6 phases 1a/1b), taking the split from 3/4 to 6/1. Pinning the
    numbers is what makes those moves visible; the two NON-VACUITY arms are
    what keep the test meaningful whichever way they move.
    """
    print("=== unserved sensors hold their sensordata slots ===")
    var mujoco = Python.import_module("mujoco")
    var m = _mj_model(mujoco, MIXED_XML)
    var fmd = parse_xml_full(MIXED_XML, String("."))

    var n_mj = Int(py=m.nsensor)
    assert_true(
        len(fmd.sensors) == n_mj,
        "every sensor must get a row, served or not: ours "
        + String(len(fmd.sensors)) + " vs MuJoCo " + String(n_mj),
    )

    var served_seen = 0
    var unserved_seen = 0
    print("  idx name  served  dim  adr   (MuJoCo dim/adr)")
    for i in range(n_mj):
        var sd = fmd.sensors[i]
        var mj_dim = Int(py=m.sensor_dim[i])
        var mj_adr = Int(py=m.sensor_adr[i])
        print(
            "  ", i, fmd.sensor_names[i], sd.served, sd.dim, sd.adr,
            " (", mj_dim, mj_adr, ")",
        )
        assert_true(
            sd.dim == mj_dim and sd.adr == mj_adr,
            "sensor " + String(i) + " '" + fmd.sensor_names[i]
            + "': dim/adr ours " + String(sd.dim) + "/" + String(sd.adr)
            + " vs MuJoCo " + String(mj_dim) + "/" + String(mj_adr)
            + " — an unserved sensor did not hold its slot",
        )
        if sd.served:
            served_seen += 1
            assert_true(
                sd.objid >= 0,
                "a served sensor must have a resolved objid, got "
                + String(sd.objid),
            )
        else:
            unserved_seen += 1
            # ⚠ `-1`, NOT 0. A zero here would be a valid site index.
            assert_true(
                sd.objid == -1 and sd.body_id == -1,
                "an unserved sensor must carry objid/body_id -1, not a"
                " plausible index; got " + String(sd.objid) + "/"
                + String(sd.body_id),
            )

    print("  served:", served_seen, " unserved (addressed only):",
          unserved_seen)
    assert_true(served_seen == 6, "expected 6 served, got "
                + String(served_seen))
    assert_true(unserved_seen == 1, "expected 1 unserved, got "
                + String(unserved_seen))
    # ⚠ NON-VACUITY, BOTH HALVES. The whole test is about the BOUNDARY between
    # served and addressed; with either side empty there is no boundary and
    # every assertion above holds trivially.
    assert_true(served_seen > 0 and unserved_seen > 0,
                "the fixture must declare both a served and an unserved"
                " sensor, or this test has nothing to distinguish")
    assert_true(
        fmd.nsensordata() == Int(py=m.nsensordata),
        "nsensordata: ours " + String(fmd.nsensordata()) + " vs MuJoCo "
        + String(Int(py=m.nsensordata)),
    )
    print("  nsensordata:", fmd.nsensordata(), "== MuJoCo",
          Int(py=m.nsensordata))

    # The served ones still resolve by name...
    assert_true(fmd.sensor_site_by_name(String("t")) >= 0,
                "a served sensor must still resolve by name")
    # ...and the unserved ones raise on every read, including `adr`, whose
    # value is CORRECT but points at values nothing wrote.
    for nm in [String("sam")]:
        var raised = False
        try:
            _ = fmd.sensor_adr_by_name(nm)
        except:
            raised = True
        assert_true(
            raised,
            "reading unserved sensor '" + nm + "' by name must raise — its"
            " offset is right and its values were never written",
        )
    print("  the unserved name raises on read: ok")

    # ⚠ AND A SERVED JOINT SENSOR DOES NOT RAISE. Without this arm the loop
    # above would still pass if `jointpos` had been dropped from the table
    # altogether rather than served.
    for nm_adr in [
        (String("jp"), 7), (String("fq"), 3), (String("fx"), 12)
    ]:
        assert_true(
            fmd.sensor_adr_by_name(nm_adr[0]) == nm_adr[1],
            "`" + nm_adr[0] + "` is served now and must read back its adr, "
            + String(nm_adr[1]) + "; got "
            + String(fmd.sensor_adr_by_name(nm_adr[0])),
        )

    # And the model said so once, by audit id.
    assert_true(
        fmd.silent_attrs >= 1,
        "the unserved sensors must be reported under AUD-23; silent_attrs = "
        + String(fmd.silent_attrs),
    )
    print("  reported under AUD-23: silent_attrs =", fmd.silent_attrs)


def main() raises:
    var suite = TestSuite()
    suite.test[test_sensor_table_matches_mujoco]()
    suite.test[test_lookup_by_name_beats_a_counted_index]()
    suite.test[test_unserved_sensors_are_addressed_not_dropped]()
    suite.test[test_unservable_sensors_refuse_to_load]()
    suite^.run()
