"""`<option><flag .../></option>` — the disable bits, and the two we half-did.

    pixi run mojo run -I . tests/physics3d/test_option_disable_flags_vs_mujoco.mojo

WHAT WAS WRONG. This engine has no runtime disable word, so `mjtDisableBit` is
honoured by EDITING THE MODEL at parse time. That block sat beside the
worldbody walk, where it could reach `result.geoms` and `result.joints` and
nothing else — and `pairs`, `equalities` and `tendons` are all filled AFTER
it. So:

⚠⚠ `<flag constraint="disable"/>` LEFT EQUALITY, DRY-FRICTION AND TENDON-LIMIT
ROWS LIVE. MuJoCo's `mjDSBL_CONSTRAINT` makes `mj_makeConstraint` return with
`nefc == 0` — every row of every type. We dropped contacts and joint limits
and kept the rest.

⚠⚠ `<flag contact="disable"/>` LEFT `<contact><pair>` COLLIDING. Zeroing
`contype`/`conaffinity` switches off the MASK-based path, and **an explicit
pair bypasses that mask** — that is what a pair is for. apptronik_apollo
declares six.

WHAT IT COST, WHICH IS THE POINT. An ablation is the primary tool for the rest
of the Menagerie board: switch a subsystem off in BOTH engines and see what is
left. Run with `constraint="disable"` it reported the residual GROWING on six
of eight models — google_robot 1.7e-09 -> 1.5e-04, robot_soccer_kit 1.6e-08 ->
7.9e-05 — which is exactly what you measure when the reference drops every row
and you drop a third of them. **A broken ablation does not fail, it lies**, and
it lies in the direction of "your smooth dynamics are wrong", which is the most
expensive wrong answer available. With the flags fixed the same ablation reads
1.58e-17 / 3.04e-18 / 9.76e-19 and correctly names contacts, equality and
friction on the models that still differ.

⚠ NO MENAGERIE MODEL SETS ANY OF THESE, so the sweep does not move. Two
dm_control models do — acrobot and fish, both `constraint="disable"` — and
neither declares an equality, a `frictionloss` or a `<pair>`, so their gates
are unaffected either way. That is the shape of this defect: invisible to
every model in the tree and fatal to the instrument.

THE FIXTURE is built so each flag changes the answer on its own:

* the ball's geoms are `contype="0" conaffinity="0"`, so **only** the explicit
  `<pair>` can collide it with the floor — `contact` is decisive, and a build
  that keeps the pair still reports `ncon 1`.
* `ja` starts at 0.205 against `range="-0.2 0.2"`, i.e. already OUTSIDE, so
  the limit row is active rather than merely present.
* `<equality joint polycoef="0 2 0 0 0">` couples `ja` and `jb` at 2:1, so a
  dropped equality moves both.
* `jc` slides with `frictionloss="0.5"` and a velocity, so friction shows in
  the third digit of a dof nothing else touches.

MuJoCo's `nefc` for the six variants is 7 / 3 / 6 / 6 / 6 / 0 — every flag
removes something, and `constraint` removes everything.
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.studio.stepping import (
    StudioImpFastPyr, StudioImpFastEll, StudioIntegPyr, StudioIntegEll,
    studio_cone_of, studio_uses_implicit,
)
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.gpu.constants import KEY_IDX_NQPOS, KEY_IDX_NQVEL

comptime DT = DType.float64

comptime BODY = String(
    """
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" contype="0" conaffinity="0"/>
    <body name="ball" pos="0 0 0.049">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.05" mass="1"
            contype="0" conaffinity="0"/>
    </body>
    <body name="a" pos="1 0 0.5">
      <joint name="ja" type="hinge" axis="0 1 0" range="-0.2 0.2"
             limited="true"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"
            contype="0" conaffinity="0"/>
      <body name="b" pos="0.3 0 0">
        <joint name="jb" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"
              contype="0" conaffinity="0"/>
      </body>
    </body>
    <body name="c" pos="2 0 0.5">
      <joint name="jc" type="slide" axis="1 0 0" frictionloss="0.5"/>
      <geom type="box" size=".05 .05 .05" mass="1"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="floor" geom2="ball"/>
  </contact>
  <equality>
    <joint joint1="ja" joint2="jb" polycoef="0 2 0 0 0"/>
  </equality>
  <keyframe>
    <key qpos="0 0 0.049 1 0 0 0  0.205 0.41 0"
         qvel="0 0 -0.4 0 0 0  2.0 4.0 1.0"/>
  </keyframe>
</mujoco>"""
)


def _xml(flag: String) raises -> String:
    """The fixture with one `<flag NAME="disable"/>`, or none for "".""" 
    var opt = String('<option timestep="0.002"/>')
    if flag != "":
        opt = String(
            '<option timestep="0.002"><flag ', flag,
            '="disable"/></option>',
        )
    return String('<mujoco>\n  <compiler angle="radian"/>\n  ', opt, BODY)


def _mj(flag: String) raises -> List[Float64]:
    """MuJoCo 3.10.0 `qpos` after one step from keyframe 0."""
    if flag == "":
        return [
            1.938224570756685e-20, -1.938224570756685e-20, 0.048368038, 1.0,
            3.060198456813736e-19, 0.0, 0.0,
            0.2085764036456166, 0.41362189546507544, 0.001998,
        ]
    if flag == "contact":
        return [
            0.0, 0.0, 0.048160760000000004, 1.0, 0.0, 0.0, 0.0,
            0.2085764036456166, 0.41362189546507544, 0.001998,
        ]
    if flag == "equality":
        return [
            -1.2553826544170728e-20, 1.2553826544170728e-20, 0.048368038, 1.0,
            1.9820820144995098e-19, 0.0, 0.0,
            0.20816866911258775, 0.42002968726734485, 0.001998,
        ]
    if flag == "frictionloss":
        return [
            1.938224570756685e-20, -1.938224570756685e-20, 0.048368038, 1.0,
            3.060198456813736e-19, 0.0, 0.0,
            0.2085764036456166, 0.41362189546507544, 0.002,
        ]
    if flag == "limit":
        return [
            1.938224570756685e-20, -1.938224570756685e-20, 0.048368038, 1.0,
            3.060198456813736e-19, 0.0, 0.0,
            0.21033575644143954, 0.41422151301674287, 0.001998,
        ]
    # constraint — nefc 0, every row gone
    return [
        0.0, 0.0, 0.048160760000000004, 1.0, 0.0, 0.0, 0.0,
        0.20920209137377566, 0.41768002105006996, 0.002,
    ]


def _step_once(xml: String) raises -> List[Float64]:
    var fmd = parse_xml_full(expand_mjcf(xml, String("")), String(""))
    var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=0)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var d = Data[DT, DynDims, 1](dims)
    assert_true(dims.get_nkey() > 0, "the fixture must carry its keyframe")
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    var nqp = Int(Float64(sf.key_meta.data[KEY_IDX_NQPOS]))
    for i in range(min(nqp, nq)):
        d.qpos.data[i] = sf.key_qpos.data[i]
    var nqv = Int(Float64(sf.key_meta.data[KEY_IDX_NQVEL]))
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)
    for i in range(min(nqv, nv)):
        d.qvel.data[i] = sf.key_qvel.data[i]
    for i in range(nv):
        d.qfrc.data[i] = Scalar[DT](0)
    var use_imp = studio_uses_implicit(fmd)
    var cone = studio_cone_of(fmd)
    var imp_e = StudioImpFastEll(dims)
    var imp_p = StudioImpFastPyr(dims)
    var ell = StudioIntegEll(dims)
    var pyr = StudioIntegPyr(dims)
    if use_imp:
        if cone == ConeType.ELLIPTIC:
            imp_e.step["cpu"](d, m)
        else:
            imp_p.step["cpu"](d, m)
    else:
        if cone == ConeType.ELLIPTIC:
            ell.step["cpu"](d, m)
        else:
            pyr.step["cpu"](d, m)
    var out = List[Float64]()
    for i in range(nq):
        out.append(Float64(d.qpos.data[i]))
    return out^


def _worst(got: List[Float64], want: List[Float64]) -> Float64:
    var w = 0.0
    for i in range(len(want)):
        var e = abs(got[i] - want[i])
        if e > w:
            w = e
    return w


def _check(flag: String, tol: Float64) raises:
    var label = flag if flag != "" else String("(no flag)")
    var want = _mj(flag)
    var got = _step_once(_xml(flag))
    assert_true(
        len(got) == 10,
        "the fixture must build 10 qpos slots; got " + String(len(got)),
    )
    var worst = _worst(got, want)
    print("  ", label, " worst |d(qpos)| =", worst)
    assert_true(
        worst < tol,
        "with <flag " + label + '="disable"> we are ' + String(worst)
        + " from MuJoCo. Matching the NO-FLAG row instead means the flag"
        " reached nothing; matching it PARTLY means only some row types were"
        " dropped, which is what `constraint` used to do.",
    )


def test_no_flag_is_the_control() raises:
    """Every row type live — the row the others are measured against."""
    print("=== the fixture with no flag at all ===")
    _check(String(""), 1e-12)
    print("  PASS")


def test_contact_disable_drops_explicit_pairs() raises:
    """⚠ THE PAIR BYPASSES `contype` — that is the whole point of a pair."""
    print("=== <flag contact=\"disable\"> ===")
    # ⚠ VACUITY: the ball's z must DIFFER from the no-flag row, or the fixture
    # never had a contact to drop.
    var with_c = _mj(String(""))
    var no_c = _mj(String("contact"))
    print("   ball z: contact on", with_c[2], " off", no_c[2])
    assert_true(
        abs(with_c[2] - no_c[2]) > 1e-6,
        "the reference answers agree with and without the contact; the pair"
        " is not colliding in MuJoCo either and the fixture proves nothing",
    )
    _check(String("contact"), 1e-12)
    print("  PASS")


def test_equality_frictionloss_and_limit_each_drop_their_rows() raises:
    """Three flags, three row types, one file."""
    print("=== <flag equality|frictionloss|limit=\"disable\"> ===")
    _check(String("equality"), 1e-12)
    _check(String("frictionloss"), 1e-12)
    _check(String("limit"), 1e-12)
    print("  PASS")


def test_constraint_disable_leaves_nothing() raises:
    """`mjDSBL_CONSTRAINT` is TOTAL — MuJoCo reports `nefc == 0`."""
    print("=== <flag constraint=\"disable\"> ===")
    # ⚠ It must differ from EVERY single-flag row, or "total" is not tested.
    var total = _mj(String("constraint"))
    for f in [String("contact"), String("equality"), String("frictionloss"),
              String("limit")]:
        var one = _mj(f)
        assert_true(
            _worst(total, one) > 1e-9,
            "the `constraint` reference equals the `" + f + "` one; the"
            " fixture cannot tell TOTAL from that single flag",
        )
    _check(String("constraint"), 1e-12)
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
