"""`<keyframe><key>` — parsed, and NOT applied by the default reset.

`keyframe` and `<key` appeared NOWHERE in `mojo_rl/physics3d/parser/`: the
section was not merged, not counted and not read. ToddlerBot's reference env
resets from `keyframe("home").qpos`, whose values differ from `qpos0` in 26 of
51 slots by up to 1.5708 rad, and which sets 18 of its 30 controls. Without it
we reset a standing humanoid into a different posture and nothing raises.

⚠⚠ THE LOAD-BEARING DESIGN POINT, AND IT IS THE OPPOSITE OF THE OBVIOUS ONE.
Measured on the 3.10.0 runtime:

    m has nkey=1 with key_qpos[0] != qpos0
    mj_resetData(m, d)          -> d.qpos == qpos0        (NOT the keyframe)
    mj_resetDataKeyframe(m,d,0) -> d.qpos == key_qpos[0]

So a keyframe is NOT a reset pose. A `reset_data` that silently "preferred" a
keyframe would diverge from MuJoCo on every model that declares one — 66 of
Menagerie's models — which is the shape of the `ctrlrange` fallback that
became a hard clamp. `reset_data` is therefore unchanged and
`reset_data_keyframe` is a separate, explicit entry point. The third test
below is what holds that line.

⚠ Wrong-length attributes are REJECTED rather than padded. MuJoCo pads a SHORT
one, but not from qpos0: for a model whose `qpos0[7]` is 0.00436332 (a
`ref="0.25"` in degrees) a short `qpos` comes back carrying 0.25 in that slot
— the raw attribute value, before unit conversion. Across Menagerie's 66
keyframed models 145 of 145 attributes are exactly full length and none is
short, so nothing real depends on reproducing that.

⚠ Counts come from loading models with MuJoCo. A `<key>` inside an XML COMMENT
is a live hazard here: `rethink_robotics_sawyer/sawyer.xml` carries a
commented-out second `<key name="home">` one slot longer than nq, and a text
scan reads it as a real over-length keyframe.

Run with:
    pixi run mojo run -I . tests/physics3d/test_keyframe_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf
from mojo_rl.physics3d.fields import Data, Dims
from mojo_rl.physics3d.model.model_dims import ModelDims

comptime DTYPE = DType.float64

# Two keyframes, so an implementation that ignores the index cannot pass, and
# they differ in WHICH attributes they carry:
#
#   home  — qpos + qvel + ctrl + time, every one of them non-default
#   part  — qpos ONLY, so qvel/ctrl must come back ZERO rather than inherited
#           from `home` or from the model
#
# ⚠ `j1` carries `ref="0.25"` with the default degree units, so qpos0[1] is
# 0.004363 rad while any raw reading of the attribute is 0.25. That is the
# exact slot where MuJoCo's short-qpos padding leaks raw spec state, and it
# also makes qpos0 != 0 so "reset wrote zeros" cannot masquerade as a pass.
comptime XML = String(
    """<mujoco model="keyframe_matrix">
  <option timestep="0.001" gravity="0 0 0"/>
  <worldbody>
    <body name="b0" pos="0 0 0"><joint name="j0" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/></body>
    <body name="b1" pos="0 1 0">
      <joint name="j1" type="hinge" axis="0 0 1" ref="0.25"/>
      <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/></body>
    <body name="b2" pos="0 2 0"><joint name="j2" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/></body>
  </worldbody>
  <actuator>
    <motor name="a0" joint="j0" ctrlrange="-5 5"/>
    <motor name="a1" joint="j1" ctrlrange="-5 5"/>
    <motor name="a2" joint="j2" ctrlrange="-5 5"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0.11 0.22 0.33" qvel="1.5 -2.5 3.5"
         ctrl="0.7 -0.8 0.9" time="0.125"/>
    <key name="part" qpos="-0.4 0.5 -0.6"/>
  </keyframe>
</mujoco>"""
)

comptime pm = parse_xml(XML)
comptime M = ModelDefFromXML[
    xml=XML,
    nbody=pm.NBODY, njoint=pm.NJOINT, nq=pm.NQ, nv=pm.NV,
    ngeom=pm.NGEOM, nact=pm.NACT, ntex=pm.NTEX, nmat=pm.NMAT,
    nlight=pm.NLIGHT, ncam=pm.NCAM, nsite=pm.NSITE,
    max_contacts=8,
    obs_dim_override=1, obs_qpos_skip=0,
    timestep=pm.TIMESTEP,
    # MuJoCo `m->nkey`. Hand-supplied since 1a.4(c); `init_fields`
    # asserts it against the parsed XML.
    nkey = 2,
]
comptime MD = ModelDims[M]

comptime TOL: Float64 = 1e-12

# ---- The MERGE leg -------------------------------------------------------
#
# ⚠ THIS IS THE SHAPE THAT ACTUALLY BROKE ToddlerBot, and it is invisible to
# every test above. Its `<key name="home">` lives in the INCLUDED robot file
# while `scene.xml` is what gets loaded, and `<keyframe>` was not in
# `merge_mjcf`'s accumulator list — so the section was dropped before any
# parser saw it. A single-file fixture cannot catch that: the parser was
# never the thing at fault.
comptime SCENE_XML = String(
    """<mujoco model="scene">
  <option timestep="0.001" gravity="0 0 0"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
  </worldbody>
</mujoco>"""
)
comptime ROBOT_XML = String(
    """<mujoco model="robot">
  <worldbody>
    <body name="rb" pos="0 0 1"><joint name="rj" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/></body>
  </worldbody>
  <actuator><motor name="ra" joint="rj" ctrlrange="-5 5"/></actuator>
  <keyframe>
    <key name="home" qpos="1.234" ctrl="0.75"/>
  </keyframe>
</mujoco>"""
)

comptime MERGED_XML = merge_mjcf(SCENE_XML, ROBOT_XML)
comptime mp = parse_xml(MERGED_XML)
comptime MM = ModelDefFromXML[
    xml=MERGED_XML,
    nbody=mp.NBODY, njoint=mp.NJOINT, nq=mp.NQ, nv=mp.NV,
    ngeom=mp.NGEOM, nact=mp.NACT, ntex=mp.NTEX, nmat=mp.NMAT,
    nlight=mp.NLIGHT, ncam=mp.NCAM, nsite=mp.NSITE,
    max_contacts=8,
    obs_dim_override=1, obs_qpos_skip=0,
    timestep=mp.TIMESTEP,
    # MuJoCo `m->nkey`. Hand-supplied since 1a.4(c); `init_fields`
    # asserts it against the parsed XML.
    nkey = 1,
]


def test_keyframe_tables_match_mujoco() raises:
    """Our recorded keys against `mjModel.key_qpos/qvel/ctrl/time`."""
    var sf = M.make_spec_fields[DTYPE]()
    print("=== keyframe: tables vs MuJoCo ===")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(XML)

    var nkey = Int(py=m.nkey)
    print("   nkey ours", M.nkey, " MuJoCo", nkey)
    assert_true(
        nkey == 2 and M.nkey == 2,
        "the fixture must carry both keys (MuJoCo nkey=" + String(nkey)
        + ", ours=" + String(M.nkey) + "). 0 here means the <keyframe>"
        " section never reached the parser",
    )

    var worst = 0.0
    for k in range(nkey):
        for i in range(Int(py=m.nq)):
            var d = abs(M.key_qpos_at[DTYPE](sf, k, i) - Float64(py=m.key_qpos[k][i]))
            if d > worst:
                worst = d
        for i in range(Int(py=m.nv)):
            var d = abs(M.key_qvel_at[DTYPE](sf, k, i) - Float64(py=m.key_qvel[k][i]))
            if d > worst:
                worst = d
        for i in range(Int(py=m.nu)):
            var d = abs(M.key_ctrl_at[DTYPE](sf, k, i) - Float64(py=m.key_ctrl[k][i]))
            if d > worst:
                worst = d
        var dt = abs(M.key_time_at[DTYPE](sf, k) - Float64(py=m.key_time[k]))
        if dt > worst:
            worst = dt
        print(
            "   key", k,
            " qpos[0]", M.key_qpos_at[DTYPE](sf, k, 0),
            " qvel[0]", M.key_qvel_at[DTYPE](sf, k, 0),
            " ctrl[0]", M.key_ctrl_at[DTYPE](sf, k, 0),
            " time", M.key_time_at[DTYPE](sf, k),
        )

    print("   worst |d| =", worst)
    assert_true(worst <= TOL, "keyframe tables differ from MuJoCo by "
                + String(worst))
    print("  PASS")


def test_absent_attributes_follow_mujocos_defaults() raises:
    """Key `part` omits qvel and ctrl: MuJoCo fills ZEROS, not `home`'s values.

    ⚠ This is the leg that distinguishes "absent" from "all zeros". An
    implementation that carried the previous key's values forward, or that
    inherited the model's, would pass the table test above only by accident of
    ordering — so it is asserted against MuJoCo AND against `home` explicitly.
    """
    var sf = M.make_spec_fields[DTYPE]()
    print("=== keyframe: absent attributes ===")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(XML)
    var kpart = Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "part"))
    var khome = Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    print("   'home' index", khome, "  'part' index", kpart)

    # Vacuity: `home` must be non-zero where `part` is absent, or "zero"
    # proves nothing.
    var home_nonzero = False
    for i in range(Int(py=m.nv)):
        if abs(M.key_qvel_at[DTYPE](sf, khome, i)) > 1e-9:
            home_nonzero = True
    assert_true(
        home_nonzero,
        "'home' has an all-zero qvel, so 'part' returning zeros no longer"
        " distinguishes absent from inherited",
    )

    for i in range(Int(py=m.nv)):
        assert_true(
            abs(M.key_qvel_at[DTYPE](sf, kpart, i)) <= TOL,
            "key 'part' omits qvel; MuJoCo fills zeros but ours returned "
            + String(M.key_qvel_at[DTYPE](sf, kpart, i)),
        )
    for i in range(Int(py=m.nu)):
        assert_true(
            abs(M.key_ctrl_at[DTYPE](sf, kpart, i)) <= TOL,
            "key 'part' omits ctrl; MuJoCo fills zeros but ours returned "
            + String(M.key_ctrl_at[DTYPE](sf, kpart, i)),
        )
    print("  PASS")


def test_default_reset_is_qpos0_not_the_keyframe() raises:
    """⚠⚠ THE LOAD-BEARING LEG. `reset_data` must still write qpos0.

    MuJoCo's `mj_resetData` ignores keyframes entirely; only
    `mj_resetDataKeyframe` applies one. If a future change makes the default
    reset "prefer" a declared keyframe — which reads like an improvement — the
    engine silently starts every one of Menagerie's 66 keyframed models from a
    different pose than MuJoCo. This test is what makes that a red build."""
    var sf = M.make_spec_fields[DTYPE]()
    print("=== keyframe: default reset is NOT the keyframe ===")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(XML)
    var md = mujoco.MjData(m)
    _ = mujoco.mj_resetData(m, md)

    var d = Data[DTYPE, MD, 1]()
    M.reset_data[DTYPE](sf, d)

    # Vacuity: the keyframe must actually differ from qpos0, or "we match
    # qpos0" and "we match the keyframe" are the same statement.
    var spread = 0.0
    for i in range(Int(py=m.nq)):
        var s = abs(Float64(py=m.key_qpos[0][i]) - Float64(py=m.qpos0[i]))
        if s > spread:
            spread = s
    print("   max |key_qpos[0] - qpos0| =", spread)
    assert_true(
        spread > 0.1,
        "the fixture's keyframe no longer differs meaningfully from qpos0, so"
        " this test cannot tell the two reset behaviours apart",
    )

    var worst = 0.0
    for i in range(Int(py=m.nq)):
        var dd = abs(Float64(d.qpos.data[i]) - Float64(py=md.qpos[i]))
        if dd > worst:
            worst = dd
    print("   |reset_data - mj_resetData| =", worst)
    assert_true(
        worst <= TOL,
        "reset_data no longer matches mj_resetData (worst " + String(worst)
        + "). ⚠ if this went red because reset_data now applies a keyframe,"
        " that is the regression this test exists for — MuJoCo's mj_resetData"
        " writes qpos0 even when nkey > 0",
    )
    print("  PASS")


def test_reset_data_keyframe_matches_mj_resetDataKeyframe() raises:
    """The explicit path, against `mj_resetDataKeyframe`, for BOTH keys."""
    var sf = M.make_spec_fields[DTYPE]()
    print("=== keyframe: reset_data_keyframe vs mj_resetDataKeyframe ===")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(XML)
    var md = mujoco.MjData(m)

    var worst = 0.0
    for k in range(Int(py=m.nkey)):
        _ = mujoco.mj_resetDataKeyframe(m, md, k)
        var d = Data[DTYPE, MD, 1]()
        M.reset_data_keyframe[DTYPE](sf, d, k)
        for i in range(Int(py=m.nq)):
            var dd = abs(Float64(d.qpos.data[i]) - Float64(py=md.qpos[i]))
            if dd > worst:
                worst = dd
        for i in range(Int(py=m.nv)):
            var dv = abs(Float64(d.qvel.data[i]) - Float64(py=md.qvel[i]))
            if dv > worst:
                worst = dv
        print("   key", k, " worst so far", worst)

    assert_true(
        worst <= TOL,
        "reset_data_keyframe differs from mj_resetDataKeyframe by "
        + String(worst),
    )
    print("  PASS")


def test_keyframe_survives_merge_mjcf() raises:
    """A keyframe declared in an INCLUDED file must survive the merge.

    ⚠ THE ToddlerBot SHAPE. `<keyframe>` was missing from `merge_mjcf`'s
    accumulator list, so a key declared in the robot file vanished when the
    scene that includes it was loaded — silently, because a model without a
    keyframe is simply a model that resets to qpos0. Every other test in this
    file passes a single-file fixture straight to the parser and cannot see
    it. This is the fourth section dropped this way, after <tendon>,
    <option>'s <flag> children and <contact>."""
    var sf = MM.make_spec_fields[DTYPE]()
    print("=== keyframe: survives merge_mjcf ===")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")
    # ⚠⚠ THE REFERENCE IS THE ROBOT FILE, NOT THE MERGED STRING. Compiling
    # MuJoCo from our own merge makes the drop invisible BY CONSTRUCTION: with
    # the accumulator disabled, MuJoCo reads the same keyframe-less text we
    # produced and reports nkey 0 as well, so "ours == MuJoCo" stays true
    # while both are wrong. Anchoring on ROBOT_XML — which our merge did not
    # produce — is what makes this leg able to fail. (`DM_CONTROL_PORT.md`
    # calls this out for layer-2 gates generally; it bites here exactly.)
    var ref_m = mujoco.MjModel.from_xml_string(ROBOT_XML)
    assert_true(
        Int(py=ref_m.nkey) == 1,
        "ROBOT_XML itself declares no keyframe — the fixture drifted and this"
        " test can no longer detect a dropped section",
    )

    var m = mujoco.MjModel.from_xml_string(MERGED_XML)
    print("   merged nkey ours", MM.nkey, " MuJoCo(merged)", Int(py=m.nkey),
          " MuJoCo(robot only)", Int(py=ref_m.nkey))
    assert_true(
        Int(py=m.nkey) == 1,
        "⚠ merge_mjcf DROPPED the <keyframe> section: MuJoCo reads our merged"
        " text and finds nkey=0, while the robot file it came from declares"
        " 1. Check `all_keyframe` in xml_parser.merge_mjcf — a section missing"
        " from the accumulator list is dropped with no diagnostic",
    )
    assert_true(
        MM.nkey == 1,
        "the keyframe did NOT survive merge_mjcf (ours nkey=" + String(MM.nkey)
        + "). ⚠ check `all_keyframe` in xml_parser.merge_mjcf — a section"
        " missing from the accumulator list is dropped with no diagnostic",
    )

    var worst = 0.0
    for i in range(Int(py=m.nq)):
        var d = abs(MM.key_qpos_at[DTYPE](sf, 0, i) - Float64(py=m.key_qpos[0][i]))
        if d > worst:
            worst = d
    for i in range(Int(py=m.nu)):
        var d = abs(MM.key_ctrl_at[DTYPE](sf, 0, i) - Float64(py=m.key_ctrl[0][i]))
        if d > worst:
            worst = d
    print("   merged key qpos[0]", MM.key_qpos_at[DTYPE](sf, 0, 0),
          " ctrl[0]", MM.key_ctrl_at[DTYPE](sf, 0, 0), " worst |d|", worst)
    assert_true(worst <= TOL, "merged keyframe differs from MuJoCo by "
                + String(worst))
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
