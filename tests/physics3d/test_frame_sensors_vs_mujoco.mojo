"""`sensors/frame.mojo` vs MuJoCo's `d.sensordata` (AUD-23, audit §6 1b + 1c).

`framepos`, `framequat`, the three axis sensors and the two velocity sensors,
over the four object types this loader resolves and both reference forms. 212
declarations across `mojo_rl/envs`, Menagerie and dm_control read these; until
2026-09-13 every one of them read back the NaN `Data` fills `sensordata` with.

⚠ THE TWO VELOCITY ROWS ARE A DIFFERENT STAGE. `framelinvel`/`frameangvel` are
`mj_sensorVel`, not `mj_sensorPos`; they read `xvel`/`xangvel`, which the step
overwrites. The harness below snapshots the state BEFORE stepping for exactly
that reason.

⚠⚠ THE FIXTURE MUST SEPARATE `body` FROM `xbody`, OR IT PROVES ALMOST NOTHING.
They are two frames of the same body — the body frame and the INERTIAL frame —
and they coincide whenever the centre of mass sits at the body origin with the
principal axes aligned, which is true of every centred sphere and box anyone
reaches for when writing a fixture. `link` here carries a capsule with
`fromto="0 0 0 .25 0 0"`, so its `ipos` is (.125, 0, 0) and its `iquat` rotates
z onto x: the two frames differ in BOTH position and orientation, and
`test_the_two_body_frames_are_not_the_same_frame` asserts that before anything
else is believed.

⚠ AIRBORNE, LIKE `test_sensordata_vs_mujoco`. Nothing here depends on the
solver; a contact would only add a place for the two engines to disagree about
something that is not a frame.

Run with:
    pixi run mojo run -I . tests/physics3d/test_frame_sensors_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.model.model_dims import ModelDims
from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.integrator.euler import EulerIntegrator

comptime DTYPE = DType.float64

# ⚠ EVERY OBJECT NAMED HERE IS OFF-AXIS AND OFF-ORIGIN ON PURPOSE. A site at
# the body origin with no `quat` makes `framepos objtype="site"` agree with
# `framepos objtype="xbody"` for the wrong reason, and a wrong objtype
# dispatch would pass.
comptime FR_XML = """
<mujoco model="frame sensors">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="torso" pos="0 0 2.0">
      <freejoint name="root"/>
      <geom name="gt" type="box" size="0.12 0.1 0.08" density="700"/>
      <site name="imu" pos="0.03 0.02 0.05" size="0.02"
            euler="0 25 40"/>
      <body name="link" pos="0.2 0 0" euler="0 0 15">
        <joint name="el" type="hinge" axis="0 1 0"/>
        <geom name="gl" type="capsule" fromto="0 0 0 0.25 0 0" size="0.03"
              density="900"/>
        <site name="wrist" pos="0.25 0.01 0" size="0.02" euler="10 0 0"/>
      </body>
    </body>
  </worldbody>
  <sensor>
    <framepos name="p_site" objtype="site" objname="wrist"/>
    <framequat name="q_site" objtype="site" objname="imu"/>
    <framexaxis name="x_xbody" objtype="xbody" objname="link"/>
    <frameyaxis name="y_body" objtype="body" objname="link"/>
    <framezaxis name="z_geom" objtype="geom" objname="gl"/>
    <framepos name="p_body" objtype="body" objname="link"/>
    <framequat name="q_geom" objtype="geom" objname="gl"/>
    <framepos name="p_rel" objtype="geom" objname="gl"
              reftype="site" refname="imu"/>
    <framequat name="q_rel" objtype="xbody" objname="link"
               reftype="xbody" refname="torso"/>
    <framexaxis name="x_rel" objtype="site" objname="wrist"
                reftype="body" refname="torso"/>
    <framelinvel name="lv_site" objtype="site" objname="wrist"/>
    <frameangvel name="av_geom" objtype="geom" objname="gl"/>
    <framelinvel name="lv_rel" objtype="site" objname="wrist"
                 reftype="xbody" refname="torso"/>
    <frameangvel name="av_rel" objtype="xbody" objname="link"
                 reftype="site" refname="imu"/>
  </sensor>
</mujoco>
"""

comptime fp = parse_xml(FR_XML)
comptime FM = ModelDefFromXML[
    xml=FR_XML,
    nbody=fp.NBODY, njoint=fp.NJOINT, nq=fp.NQ, nv=fp.NV,
    ngeom=fp.NGEOM, nact=fp.NACT, ntex=fp.NTEX, nmat=fp.NMAT,
    nlight=fp.NLIGHT, ncam=fp.NCAM, nsite=fp.NSITE,
    # Fourteen sensors; 3+4+3+3+3+3+4+3+4+3 + 3+3+3+3 = 45 values.
    nsensor=14, nsensordata=45,
    max_tendon=fp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=fp.TIMESTEP,
]

comptime FMD = ModelDims[FM]
comptime Dat = Data[DTYPE, FMD, 1]
comptime Mod = Model[DTYPE, FMD]
comptime Integ = EulerIntegrator[
    DTYPE, FMD, FM.CONE_TYPE, 1, SOLVER="newton", RNE_POST=True
]


def _names() -> List[String]:
    return [
        String("p_site"), String("q_site"), String("x_xbody"),
        String("y_body"), String("z_geom"), String("p_body"),
        String("q_geom"), String("p_rel"), String("q_rel"), String("x_rel"),
        String("lv_site"), String("av_geom"), String("lv_rel"),
        String("av_rel"),
    ]


def _run(
    mut d: Dat, mut mf: Mod, ctx: DeviceContext
) raises -> Tuple[List[Float64], List[Float64]]:
    """Airborne, tilted, moving — then one step, which runs the passes.

    ⚠ SNAPSHOT BEFORE THE STEP. `step` integrates; the position-stage sensors
    were evaluated at the PRE-integration pose, so that is the state MuJoCo
    must be forwarded to. `test_sensordata_vs_mujoco` records what happens
    when this is got wrong.
    """
    var sf = FM.make_spec_fields[DTYPE]()
    FM.init_fields[DTYPE](ctx, mf)
    FM.reset_data(sf, d)
    d.qpos.data[2] = Scalar[DTYPE](2.0)
    # A tilt that is not a multiple of 90 degrees about any axis. w first.
    d.qpos.data[3] = Scalar[DTYPE](0.9238795325112867)
    d.qpos.data[4] = Scalar[DTYPE](0.2209424194365075)
    d.qpos.data[5] = Scalar[DTYPE](0.2209424194365075)
    d.qpos.data[6] = Scalar[DTYPE](0.2209424194365075)
    d.qpos.data[7] = Scalar[DTYPE](0.6)  # elbow
    d.qvel.data[0] = Scalar[DTYPE](0.7)
    d.qvel.data[1] = Scalar[DTYPE](-0.4)
    d.qvel.data[2] = Scalar[DTYPE](1.1)
    d.qvel.data[3] = Scalar[DTYPE](0.9)
    d.qvel.data[4] = Scalar[DTYPE](-1.3)
    d.qvel.data[5] = Scalar[DTYPE](0.5)
    d.qvel.data[6] = Scalar[DTYPE](2.0)

    var qpos = List[Float64]()
    var qvel = List[Float64]()
    for i in range(FM.NQ):
        qpos.append(Float64(d.qpos.data[i]))
    for i in range(FM.NV):
        qvel.append(Float64(d.qvel.data[i]))

    var integ = Integ()
    # No explicit `sensor_pos` call: `step` runs the passes, and this gate is
    # about the wiring as much as the arithmetic.
    integ.step["cpu"](d, mf)
    return (qpos^, qvel^)


def _mj_at(
    mujoco: PythonObject, qpos: List[Float64], qvel: List[Float64],
) raises -> PythonObject:
    var m = mujoco.MjModel.from_xml_string(PythonObject(String(FR_XML)))
    var dat = mujoco.MjData(m)
    for i in range(len(qpos)):
        dat.qpos[i] = qpos[i]
    for i in range(len(qvel)):
        dat.qvel[i] = qvel[i]
    mujoco.mj_forward(m, dat)
    return dat^


def test_the_two_body_frames_are_not_the_same_frame() raises:
    """`objtype="body"` is the INERTIAL frame; `objtype="xbody"` is not.

    ⚠⚠ WITHOUT THIS THE WHOLE FILE IS NEARLY VACUOUS. The two frames coincide
    on a centred, axis-aligned body, and a dispatch that confused them would
    then agree with MuJoCo on every row. This reads `link`'s `ipos` and
    `iquat` off a live `MjModel` and asserts both are non-trivial, so the
    `body`/`xbody` rows below are separating something real.
    """
    print("=== body and xbody are different frames of `link` ===")
    var mujoco = Python.import_module("mujoco")
    print("  mujoco", String(mujoco.__version__))
    var m = mujoco.MjModel.from_xml_string(PythonObject(String(FR_XML)))
    var O = mujoco.mjtObj
    var b = Int(py=mujoco.mj_name2id(m, O.mjOBJ_BODY, PythonObject("link")))
    assert_true(b > 0, "the fixture no longer declares a body named `link`")

    var ipos_n = 0.0
    for k in range(3):
        ipos_n += abs(Float64(py=m.body_ipos[b][k]))
    # iquat is (w, x, y, z); a non-identity one has a nonzero vector part.
    var iq_v = 0.0
    for k in range(1, 4):
        iq_v += abs(Float64(py=m.body_iquat[b][k]))
    print("  link ipos |.|1 =", ipos_n, "  iquat vector |.|1 =", iq_v)
    assert_true(
        ipos_n > 1e-3,
        "`link`'s centre of mass sits at its origin — the body and xbody"
        " POSITION rows would agree even under a wrong dispatch",
    )
    assert_true(
        iq_v > 1e-3,
        "`link`'s inertia frame is axis-aligned with its body frame — the"
        " body and xbody ORIENTATION rows would agree even under a wrong"
        " dispatch",
    )


def test_frame_sensors_match_mujoco() raises:
    """All five kinds, four object types, both reference forms."""
    print("=== frame sensors vs MuJoCo ===")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(PythonObject(String(FR_XML)))

    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    var st = _run(d, mf, ctx)
    var qpos = st[0].copy()
    var qvel = st[1].copy()
    var dat = _mj_at(mujoco, qpos, qvel)

    assert_true(
        Int(py=dat.ncon) == 0,
        "the fixture must stay AIRBORNE — MuJoCo reports "
        + String(Int(py=dat.ncon)) + " contacts",
    )

    var names = _names()
    assert_true(
        Int(py=m.nsensor) == len(names),
        "MuJoCo sees " + String(Int(py=m.nsensor)) + " sensors, the name list"
        " has " + String(len(names)),
    )

    var compared = 0
    var nonzero = 0
    var worst = 0.0
    print("  sensor     adr dim   ours[0]              MuJoCo[0]")
    for i in range(len(names)):
        var adr = Int(py=m.sensor_adr[i])
        var dim = Int(py=m.sensor_dim[i])
        for k in range(dim):
            var o = Float64(d.sensordata.data[adr + k])
            var t = Float64(py=dat.sensordata[adr + k])
            var diff = abs(o - t)
            if diff > worst:
                worst = diff
            if abs(t) > 1e-9:
                nonzero += 1
            assert_true(
                diff <= 1e-9,
                names[i] + "[" + String(k) + "] (adr " + String(adr + k)
                + "): ours " + String(o) + " vs MuJoCo " + String(t)
                + ", |d| " + String(diff),
            )
            compared += 1
        print("  ", names[i], adr, dim,
              Float64(d.sensordata.data[adr]),
              Float64(py=dat.sensordata[adr]))

    print("  values compared:", compared, " differing: 0  worst |d| =", worst)
    print("  values MuJoCo reports NONZERO:", nonzero, "/", compared)
    assert_true(compared == 45,
                "expected 45 values, compared " + String(compared))
    # ⚠ NON-VACUITY. A `sensordata` of zeros on both sides passes every row.
    assert_true(
        nonzero >= 37,
        "only " + String(nonzero) + " of " + String(compared) + " values are"
        " nonzero on MuJoCo's side — the fixture has stopped exercising the"
        " sensors",
    )


def test_the_relative_form_is_not_the_global_one() raises:
    """A reference frame CHANGES the reading, and the gate proves it does.

    ⚠⚠ THE FAILURE THIS CATCHES IS THE MOST LIKELY ONE IN THE WHOLE FAMILY.
    `reftype`/`refname` are optional, so a loader that parsed them and then
    did nothing would report the GLOBAL quantity — right units, right
    magnitude, wrong frame — and would agree with MuJoCo on every sensor that
    declares no reference, which is every sensor in every model in this tree
    today. The agreement in the test above is therefore not evidence about the
    relative path unless the references BIND.

    So: take MuJoCo's own output at this state, take it again on the same
    document with every `reftype`/`refname` pair deleted, and count how many
    values move. The three relative rows carry ten values between them; if
    fewer than six move, the fixture's reference frames are degenerate.
    """
    print("=== the reference frame actually moves the reading ===")
    var mujoco = Python.import_module("mujoco")

    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    var st = _run(d, mf, ctx)
    var qpos = st[0].copy()
    var qvel = st[1].copy()
    var dat = _mj_at(mujoco, qpos, qvel)

    var no_ref = String(FR_XML)
    no_ref = no_ref.replace(
        String('reftype="site" refname="imu"'), String("")
    )
    no_ref = no_ref.replace(
        String('reftype="xbody" refname="torso"'), String("")
    )
    no_ref = no_ref.replace(
        String('reftype="body" refname="torso"'), String("")
    )
    no_ref = no_ref.replace(
        String('reftype="xbody" refname="torso"'), String("")
    )
    no_ref = no_ref.replace(
        String('reftype="site" refname="imu"'), String("")
    )
    assert_true(
        no_ref.find(String("reftype")) == -1,
        "the reference-stripping edit missed a reftype — the control below"
        " would then compare the model with itself",
    )

    var m2 = mujoco.MjModel.from_xml_string(PythonObject(no_ref))
    var d2 = mujoco.MjData(m2)
    for i in range(len(qpos)):
        d2.qpos[i] = qpos[i]
    for i in range(len(qvel)):
        d2.qvel[i] = qvel[i]
    mujoco.mj_forward(m2, d2)

    var changed = 0
    for k in range(45):
        if abs(
            Float64(py=dat.sensordata[k]) - Float64(py=d2.sensordata[k])
        ) > 1e-9:
            changed += 1
    print("  values the reference frames change, on MuJoCo's own output:",
          changed, "/ 45")
    assert_true(
        changed >= 12,
        "dropping every reftype/refname changed only " + String(changed)
        + " of 45 values — the fixture's reference frames are degenerate and"
        " the relative path is not being tested",
    )

    # ⚠ AND OURS MUST BE THE RELATIVE READING, NOT THE GLOBAL ONE. The rows
    # above already compare against `dat`; this names the alternative
    # explicitly, so a regression prints the value the bug would produce.
    var O = mujoco.mjtObj
    var m = mujoco.MjModel.from_xml_string(PythonObject(String(FR_XML)))
    for nm in [String("p_rel"), String("q_rel"), String("x_rel"),
               String("lv_rel"), String("av_rel")]:
        var sid = Int(py=mujoco.mj_name2id(m, O.mjOBJ_SENSOR,
                                           PythonObject(nm)))
        var adr = Int(py=m.sensor_adr[sid])
        var dim = Int(py=m.sensor_dim[sid])
        var d_rel = 0.0
        var d_glob = 0.0
        for k in range(dim):
            var ours = Float64(d.sensordata.data[adr + k])
            d_rel += abs(ours - Float64(py=dat.sensordata[adr + k]))
            d_glob += abs(ours - Float64(py=d2.sensordata[adr + k]))
        print("  ", nm, " |ours - relative| =", d_rel,
              "  |ours - global| =", d_glob)
        assert_true(
            d_rel <= 1e-9,
            nm + ": ours does not match MuJoCo's relative reading (|d| "
            + String(d_rel) + ")",
        )
        assert_true(
            d_glob > 1e-6,
            nm + ": ours equals MuJoCo's GLOBAL reading — the reference"
            " frame is being parsed and then ignored",
        )


def test_a_camera_frame_sensor_is_unserved_not_refused() raises:
    """`objtype="camera"` has no name table here, so the row stays addressed.

    ⚠ THIS IS THE `served`-PER-ROW CASE, and it is the first one in the
    loader. Every other element is served or not by its NAME; a frame sensor
    is served for four object types and not for the fifth. The row keeps
    MuJoCo's `dim` and `adr` so that every sensor AFTER it still reports the
    right offset, and the AUD-23 counter names it.
    """
    print("=== a camera frame sensor is addressed, not served ===")
    var xml = String(
        "<mujoco><worldbody>"
        "<body name='b' pos='0 0 1'>"
        "<joint name='h' type='hinge' axis='0 1 0'/>"
        "<geom name='g' type='sphere' size='0.1'/>"
        "<site name='s' pos='0.1 0 0' size='0.01'/>"
        "<camera name='c' pos='0 -0.5 0'/>"
        "</body></worldbody>"
        "<sensor>"
        "<framepos name='pc' objtype='camera' objname='c'/>"
        "<framepos name='ps' objtype='site' objname='s'/>"
        "</sensor></mujoco>"
    )
    # ⚠ THE ORACLE LOADS IT. Our row is a gap, not a disagreement.
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(PythonObject(xml))
    assert_true(Int(py=m.nsensor) == 2, "MuJoCo should see two sensors")

    var fmd = parse_xml_full(xml, String("."))
    assert_true(len(fmd.sensors) == 2,
                "both rows must exist; got " + String(len(fmd.sensors)))
    assert_true(
        not fmd.sensors[0].served and fmd.sensors[0].objid == -1,
        "the camera row must be addressed and NOT served",
    )
    assert_true(
        fmd.sensors[1].served and fmd.sensors[1].objid >= 0,
        "the site row must be served — otherwise the camera verdict is about"
        " the element, not the objtype",
    )
    # And its slot is still MuJoCo's.
    assert_true(
        fmd.sensors[1].adr == Int(py=m.sensor_adr[1]),
        "the sensor AFTER the unserved one must keep MuJoCo's adr: ours "
        + String(fmd.sensors[1].adr) + " vs " + String(Int(py=m.sensor_adr[1])),
    )
    var raised = False
    try:
        _ = fmd.sensor_adr_by_name(String("pc"))
    except:
        raised = True
    assert_true(raised, "reading the unserved camera row by name must raise")
    print("  camera row addressed (adr kept, read raises), site row served")


def test_the_frame_refusals() raises:
    """Half an object reference, an unknown objtype, an unknown name.

    MuJoCo refuses each of these; so must we. `objtype` without `objname` is
    an error in the reference IN BOTH DIRECTIONS
    (xml_native_reader.cc:3045-3059), which is easy to read as a default and
    is not one.
    """
    print("=== the frame sensors' refusals ===")
    comptime HEAD = """
<mujoco><worldbody><body name="b" pos="0 0 1">
<joint name="h" type="hinge" axis="0 1 0"/>
<geom name="g" type="sphere" size="0.1"/>
<site name="s" pos="0.1 0 0" size="0.01"/>
</body></worldbody><sensor>
"""
    comptime TAIL = """
</sensor></mujoco>
"""
    var rows: List[String] = [
        String('<framepos name="a" objtype="site"/>'),
        String('<framepos name="a" objname="s"/>'),
        String('<framepos name="a" objtype="joint" objname="h"/>'),
        String('<framepos name="a" objtype="site" objname="nope"/>'),
        String('<framequat name="a" objtype="body" objname="nope"/>'),
        String('<framexaxis name="a" objtype="geom" objname="nope"/>'),
        String('<framepos name="a" objtype="site" objname="s"'
               ' reftype="site"/>'),
        String('<framepos name="a" objtype="site" objname="s"'
               ' refname="s"/>'),
    ]
    var refused = 0
    for i in range(len(rows)):
        var xml = String(HEAD) + rows[i] + String(TAIL)
        var r = False
        try:
            _ = parse_xml_full(xml, String("."))
        except:
            r = True
        print("  ", "REFUSED " if r else "ACCEPTED", rows[i])
        assert_true(r, "this row must refuse the model: " + rows[i])
        refused += 1
    print("  rows that must refuse:", len(rows), " refused:", refused)

    # ⚠ THE CONTROL. Without it every refusal above could be about the
    # fixture's header rather than the row.
    var ok_xml = (
        String(HEAD)
        + String('<framepos name="a" objtype="site" objname="s"/>')
        + String(TAIL)
    )
    var fmd = parse_xml_full(ok_xml, String("."))
    assert_true(
        len(fmd.sensors) == 1 and fmd.sensors[0].served,
        "the well-formed row must load and be served",
    )
    print("  control (a well-formed framepos) loads and is served: ok")


def main() raises:
    var suite = TestSuite()
    suite.test[test_the_two_body_frames_are_not_the_same_frame]()
    suite.test[test_frame_sensors_match_mujoco]()
    suite.test[test_the_relative_form_is_not_the_global_one]()
    suite.test[test_a_camera_frame_sensor_is_unserved_not_refused]()
    suite.test[test_the_frame_refusals]()
    suite^.run()
