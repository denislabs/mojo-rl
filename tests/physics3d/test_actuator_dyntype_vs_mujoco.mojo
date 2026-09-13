"""AUD-02 / AUD-21 — the three integrable activation dynamics, vs MuJoCo.

    pixi run mojo run -I . tests/physics3d/test_actuator_dyntype_vs_mujoco.mojo

An actuator with a `dyntype` owns an activation variable and MuJoCo applies
its force from THAT, not from `ctrl`. Only `filter` was modelled; `integrator`
and `filterexact` raised at load, so `aloha/filtered_cartesian_actuators.xml`
— the one model in this tree that declares `filterexact` — could not be loaded
at all. `<position timeconst>` is the same feature under a second spelling
(AUD-21): `mjs_setToPosition` (user_api.cc:1291-1294) writes
`dynprm[0] = timeconst` and `dyntype = timeconst == 0 ? NONE : FILTEREXACT`.

⚠⚠ `filterexact` IS NOT A REFINEMENT OF `filter`, IT IS A DIFFERENT ANSWER,
and this fixture gives the two the SAME tau so that is the only thing between
them. MuJoCo, `tau = 0.1`, `ctrl = 1`, `h = 0.002`, one step:

    filter        act = 0.02          (Euler:  act += act_dot*h)
    filterexact   act = 0.01980133    (exact:  act += act_dot*tau*(1-e^-h/tau))

1% apart at step one and compounding. A gate that used a different tau for
each could not tell the exact rule from the Euler one.

⚠ THE `integrator` ARM SATURATES ON PURPOSE. `act_dot = ctrl` with ctrl 1 and
h 0.002 reaches `actrange`'s 0.4 at step 200, and the rollout runs 300 — so
`actlimited` is exercised for a third of it. Without that the clamp is dead
code that no number here would notice.

⚠⚠ EXPOSURE, MEASURED, AND IT IS ZERO. `filterexact` appears once in the
tree, in `aloha/filtered_cartesian_actuators.xml` — which is a FRAGMENT that
MuJoCo itself refuses standalone ("transmission target 'left/gripper' not
found") and that only two MJX patches reference. `integrator` appears only in
a dm_control TEST asset; `timeconst`, `actlimited` and `actrange` only in
dm_control's SCHEMA, i.e. in no model at all. So this lands the feature and
narrows a refusal; it does not unblock a model. `muscle`, `dcmotor`, `pid`
and `user` still raise, and each is a different ODE rather than a different
integration of the same one.

⚠ THE CLASS-CHAIN ARM IS THE ONE SHAPE A REAL FILE USES. aloha states its
`dyntype="filterexact" dynprm="0.5"` in a `<default class="act">` block, not
on the element, so the element-only path this file would otherwise gate is
the path no model takes.
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields

comptime DT = DType.float64
comptime N_STEPS = 300
comptime SAT_STEP = 200  # ctrl 1 * 0.002 reaches actrange 0.4 here

# Four independent hinges, one actuator each, gravity off: `act` is then a
# pure function of the dynamics and nothing else can move it.
comptime XML = """
<mujoco model="dyntypes">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="b1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
    </body>
    <body name="b2" pos="0 0.5 0">
      <joint name="j2" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
    </body>
    <body name="b3" pos="0 1.0 0">
      <joint name="j3" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
    </body>
    <body name="b4" pos="0 1.5 0">
      <joint name="j4" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
    </body>
  </worldbody>
  <actuator>
    <general name="a_fe" joint="j1" dyntype="filterexact" dynprm="0.1"
             gaintype="fixed" gainprm="5" biastype="none"/>
    <general name="a_f"  joint="j2" dyntype="filter" dynprm="0.1"
             gaintype="fixed" gainprm="5" biastype="none"/>
    <general name="a_i"  joint="j3" dyntype="integrator" dynprm="1"
             gaintype="fixed" gainprm="5" biastype="none"
             actlimited="true" actrange="-0.4 0.4"/>
    <position name="a_tc" joint="j4" kp="20" timeconst="0.05"/>
  </actuator>
</mujoco>
"""

comptime N_ACT = 4
comptime CTRL_0: Float64 = 1.0
comptime CTRL_3: Float64 = 0.5


def _ctrl() -> List[Float64]:
    var c = List[Float64]()
    c.append(CTRL_0)
    c.append(CTRL_0)
    c.append(CTRL_0)
    c.append(CTRL_3)
    return c^


def _ours() raises -> Tuple[List[Float64], List[Float64]]:
    """`act` after every step, flattened, plus the final `qfrc`.

    ⚠ `apply_actions_fields` ONLY. The activation ODE is actuation, not
    integration — stepping the whole engine would add contact and passive
    terms to `qfrc` and make a mismatch ambiguous between the two.
    """
    var base = String(".")
    var fmd = parse_xml_full(expand_mjcf(XML, base), base)
    var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=64)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var nv = dims.get_nv()
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)

    var na = fmd.na
    var act = List[Scalar[DT]](
        length=na if na > 0 else 1, fill=Scalar[DT](0)
    )
    var ctrl = _ctrl()
    var trace = List[Float64]()
    for _s in range(N_STEPS):
        for i in range(nv):
            d.qfrc.data[i] = Scalar[DT](0)
        apply_actions_fields[DT](sf, d, ctrl, act, fmd.timestep)
        for k in range(N_ACT):
            trace.append(Float64(act[k]))
    var qfrc = List[Float64]()
    for i in range(nv):
        qfrc.append(Float64(d.qfrc.data[i]))
    return (trace^, qfrc^)


def _theirs() raises -> Tuple[List[Float64], List[Float64], Int]:
    """MuJoCo's `act` after every step, plus the final `qfrc_actuator`.

    ⚠ `mj_forward` THEN the activation advance, not `mj_step`: the bodies are
    free to rotate under the actuator force, and after 300 steps two engines'
    joint angles would differ by integration round-off, which would then feed
    back into a `<position>` actuator's force. `qvel` is held at 0 on both
    sides so the comparison stays about `act`.
    """
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(PythonObject(String(XML)))
    var d = mujoco.MjData(m)
    var c = _ctrl()
    for i in range(N_ACT):
        d.ctrl[i] = c[i]
    var trace = List[Float64]()
    for _s in range(N_STEPS):
        for i in range(Int(py=m.nv)):
            d.qvel[i] = 0.0
            d.qpos[i] = 0.0
        mujoco.mj_step(m, d)
        for k in range(N_ACT):
            trace.append(Float64(py=d.act[k]))
    var qfrc = List[Float64]()
    for i in range(Int(py=m.nv)):
        qfrc.append(Float64(py=d.qfrc_actuator[i]))
    return (trace^, qfrc^, Int(py=m.na))


def test_mujoco_gives_the_four_dyntypes_four_answers() raises:
    """⚠ RUN FIRST. `filterexact` and `filter` share a tau here, so if MuJoCo
    gave them the same activation the exact-vs-Euler distinction would be
    untestable and every number below would pass on either rule.

    Also pins `<position timeconst>` as FILTEREXACT with `dynprm[0]` = the
    time constant, which is the whole of AUD-21.
    """
    print("=== MuJoCo's four dyntypes are four different activations ===")
    var mujoco = Python.import_module("mujoco")
    print("  mujoco", String(mujoco.__version__))
    var m = mujoco.MjModel.from_xml_string(PythonObject(String(XML)))
    print("  na =", Int(py=m.na), " dyntypes =",
          Int(py=m.actuator_dyntype[0]), Int(py=m.actuator_dyntype[1]),
          Int(py=m.actuator_dyntype[2]), Int(py=m.actuator_dyntype[3]))
    assert_true(
        Int(py=m.na) == N_ACT,
        "MuJoCo gives this model " + String(Int(py=m.na)) + " activations,"
        " not " + String(N_ACT) + " — the fixture is not what it claims",
    )
    # `<position timeconst>` must have become FILTEREXACT (3), tau 0.05.
    assert_true(
        Int(py=m.actuator_dyntype[3]) == 3,
        "the <position timeconst> actuator is dyntype "
        + String(Int(py=m.actuator_dyntype[3])) + ", not FILTEREXACT (3)."
        " AUD-21's premise is wrong and this file is testing the wrong thing",
    )
    assert_true(
        abs(Float64(py=m.actuator_dynprm[3][0]) - 0.05) < 1e-15,
        "the <position timeconst> actuator's dynprm[0] is not its timeconst",
    )

    var t = _theirs()
    var fe = t[0][0]
    var fl = t[0][1]
    print("  step 1: filterexact =", fe, " filter =", fl,
          " |d| =", abs(fe - fl))
    assert_true(
        abs(fe - fl) > 1e-5,
        "filterexact and filter agree to " + String(abs(fe - fl))
        + " at step 1 on the same tau — Euler and the exact rule cannot be"
        " told apart on this fixture",
    )


def test_the_activation_clamp_actually_binds() raises:
    """⚠ THE `actlimited` ARM IS DEAD CODE UNLESS THE ACTIVATION SATURATES.

    `act_dot = ctrl = 1` at `h = 0.002` reaches the 0.4 bound at step 200.
    MuJoCo's own trace must show it pinned there afterwards, or the clamp is
    never reached and its port is untested.
    """
    print("=== the integrator's activation saturates at actrange ===")
    var t = _theirs()
    var before = t[0][(SAT_STEP - 20) * N_ACT + 2]
    var after = t[0][(N_STEPS - 1) * N_ACT + 2]
    print("  MuJoCo act[integrator]: step", SAT_STEP - 20, "=", before,
          " step", N_STEPS - 1, "=", after)
    assert_true(
        before < 0.4 - 1e-9,
        "the integrator is already clamped " + String(20) + " steps before"
        " the expected saturation — the fixture never shows the unclamped"
        " regime",
    )
    assert_true(
        abs(after - 0.4) < 1e-12,
        "the integrator's activation ends at " + String(after) + ", not at"
        " actrange's 0.4 — `actlimited` is not binding and its port is"
        " untested",
    )


def test_our_activations_match_mujoco() raises:
    """All four activations, every step of the rollout."""
    print("=== <general dyntype> + <position timeconst> vs MuJoCo ===")
    var o = _ours()
    var t = _theirs()
    assert_true(
        len(o[0]) == len(t[0]),
        "trace lengths differ: ours " + String(len(o[0])) + " vs MuJoCo "
        + String(len(t[0])),
    )
    var worst = 0.0
    var worst_k = -1
    var worst_s = -1
    for s in range(N_STEPS):
        for k in range(N_ACT):
            var dd = abs(o[0][s * N_ACT + k] - t[0][s * N_ACT + k])
            if dd > worst:
                worst = dd
                worst_k = k
                worst_s = s
    for k in range(N_ACT):
        print("  actuator", k, " final: ours =",
              o[0][(N_STEPS - 1) * N_ACT + k], " MuJoCo =",
              t[0][(N_STEPS - 1) * N_ACT + k])
    print("  worst |d act| over", N_STEPS, "steps =", worst,
          " (actuator", worst_k, "step", worst_s, ")")
    assert_true(
        worst <= 1e-13,
        "activation " + String(worst_k) + " diverges by " + String(worst)
        + " at step " + String(worst_s),
    )


comptime XML_CLASS = """
<mujoco model="dyntype via class">
  <option timestep="0.002" gravity="0 0 0"/>
  <default>
    <default class="act">
      <general dyntype="filterexact" dynprm="0.1" gaintype="fixed"
               gainprm="5" biastype="none"/>
    </default>
  </default>
  <worldbody>
    <body name="b1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
    </body>
  </worldbody>
  <actuator>
    <general name="a" class="act" joint="j1"/>
  </actuator>
</mujoco>
"""


def test_the_dyntype_is_read_through_the_class_chain() raises:
    """aloha's shape: `dyntype` in a `<default>`, nothing on the element.

    ⚠ THE ELEMENT PATH AND THE CLASS PATH ARE DIFFERENT CODE. The element
    reads `_extract_attr(tag, "dyntype")`; the class reads
    `eff.motor_dyntype_s`, merged down the chain. A gate that only spelled
    the attribute on the element would leave the path every real file uses
    untested — dog and quadruped both declare `dyntype="filter"` in a class
    for the same reason.
    """
    print("=== dyntype resolved through a <default class=> chain ===")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(PythonObject(String(XML_CLASS)))
    assert_true(
        Int(py=m.na) == 1 and Int(py=m.actuator_dyntype[0]) == 3,
        "MuJoCo does not give the class-declared actuator FILTEREXACT — the"
        " fixture is not testing what it claims",
    )
    var d = mujoco.MjData(m)
    d.ctrl[0] = CTRL_0
    mujoco.mj_step(m, d)
    var want = Float64(py=d.act[0])

    var base = String(".")
    var fmd = parse_xml_full(expand_mjcf(XML_CLASS, base), base)
    var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=64)
    var mf = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, mf)
    var sf = spec_fields_runtime[DT](fmd, dims, mf)
    var dd = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        dd.qpos.data[i] = sf.qpos0.data[i]
    var act = List[Scalar[DT]](length=1, fill=Scalar[DT](0))
    var ctrl = List[Float64]()
    ctrl.append(CTRL_0)
    for i in range(dims.get_nv()):
        dd.qfrc.data[i] = Scalar[DT](0)
    apply_actions_fields[DT](sf, dd, ctrl, act, fmd.timestep)
    var got = Float64(act[0])
    print("  after one step: ours =", got, " MuJoCo =", want)
    assert_true(
        fmd.na == 1,
        "the class-declared dyntype produced " + String(fmd.na)
        + " activations, not 1 — the class chain is not being read",
    )
    assert_true(
        abs(got - want) <= 1e-15,
        "ours " + String(got) + " vs MuJoCo " + String(want),
    )
    # The named wrong answer: Euler on the same tau.
    var euler = CTRL_0 / 0.1 * 0.002
    assert_true(
        abs(got - euler) > 1e-6,
        "ours is " + String(got) + ", which is Euler's " + String(euler)
        + " — the class chain delivered `filter`, not `filterexact`",
    )


def test_the_force_comes_from_act_not_ctrl() raises:
    """The point of the whole feature.

    ⚠ IF THE FORCE CAME FROM `ctrl`, `a_fe` WOULD READ `5 * 1 = 5` from step
    one. It reads `5 * act`, which is 0.099 at step 2 and climbs. Naming the
    wrong value is what makes this arm evidence rather than a restatement of
    the test above.
    """
    print("=== the force is gain*act, not gain*ctrl ===")
    var o = _ours()
    var t = _theirs()
    var worst = 0.0
    for i in range(len(o[1])):
        var dd = abs(o[1][i] - t[1][i])
        if dd > worst:
            worst = dd
        print("  dof", i, " ours =", o[1][i], " MuJoCo =", t[1][i])
    assert_true(
        worst <= 1e-11,
        "qfrc_actuator differs by " + String(worst),
    )
    # The named wrong answer: gain * ctrl on the first actuator.
    var from_ctrl = 5.0 * CTRL_0
    assert_true(
        abs(o[1][0] - from_ctrl) > 1e-3,
        "actuator 0's force is " + String(o[1][0]) + ", which is gain*ctrl ="
        + String(from_ctrl) + ". The activation is being ignored and the"
        " force taken from the control — AUD-02's exact failure",
    )


def main() raises:
    var suite = TestSuite()
    suite.test[test_mujoco_gives_the_four_dyntypes_four_answers]()
    suite.test[test_the_activation_clamp_actually_binds]()
    suite.test[test_our_activations_match_mujoco]()
    suite.test[test_the_dyntype_is_read_through_the_class_chain]()
    suite.test[test_the_force_comes_from_act_not_ctrl]()
    suite^.run()
