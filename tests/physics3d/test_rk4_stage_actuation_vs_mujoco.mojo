"""RK4 re-evaluates the actuators at every stage — vs MuJoCo 3.10.0.

WHY THIS EXISTS. `mj_RungeKutta` runs `mj_forwardSkip` at each of its four
stages, and that recomputes `qfrc_actuator` at the stage's `qpos`/`qvel`.
`RK4Integrator.step` integrated the `d.qfrc` the driver applied at the
step's start through all four stages. A joint `<motor>` is `gear*ctrl` at
every stage, so every RK4 model in the Gym suite was exact; a `<position>`
servo reads the stage's `qpos`, and a body-fixed thrust (crazyflie's
`<general site=...>` rotors) the stage's orientation. A Python replica of
`mj_RungeKutta` with the actuator frozen at stage 0 matched the old entry
to 1.3e-23; crazyflie sat at 1.9e-08 on the fifty-step board for it
(PERFORMANCE.md §13.34-35). `step_actuated` re-applies
`apply_actions_fields` + `apply_pose_transmission` at stages 1-3.

WHAT IT GATES. Two fixtures, runtime path, pyramidal RK4:
  * a pendulum under a `<position>` servo with a large timestep, so the
    stage-to-stage change of `kp*(ctrl - q)` is far above rounding;
  * a free body with a site-transmission wrench whose thrust is off the
    body's axis, so the body tumbles and the world-frame force rotates
    within the step.
For each: `|d(qvel)|` after 1, 5 and 30 steps with `step_actuated` must be
rounding-level, AND the frozen `step` must DIFFER from MuJoCo by more than
the tolerance at 5 steps — the control arm that proves the fixture can
see the mechanism (a motor-only fixture would pass both).

Run: pixi run mojo run -I . tests/physics3d/test_rk4_stage_actuation_vs_mujoco.mojo
"""
from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.dynamics.pose_transmission import apply_pose_transmission
from mojo_rl.physics3d.studio.stepping import StudioRk4Pyr

comptime DT = DType.float64

comptime SERVO = """
<mujoco model="rk4_servo">
  <option timestep="0.02" integrator="RK4" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="arm" pos="0 0 1">
      <joint name="j" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.02" mass="0.5"/>
    </body>
  </worldbody>
  <actuator>
    <position name="p" joint="j" kp="30" ctrlrange="-3 3"/>
  </actuator>
</mujoco>
"""

comptime THRUST = """
<mujoco model="rk4_thrust">
  <option timestep="0.005" integrator="RK4" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="craft" pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size="0.05 0.03 0.01" mass="0.1"/>
      <site name="rotor" pos="0.04 0.02 0" quat="0.96 0.2 0.2 0"/>
    </body>
  </worldbody>
  <actuator>
    <general name="t" site="rotor" gear="0 0 1 0 0 0.02" ctrlrange="0 5"/>
  </actuator>
</mujoco>
"""

comptime TOL: Float64 = 5e-12


def _ours[XML: String](nstep: Int, ctrl: Float64, actuated: Bool) raises -> List[Float64]:
    var fmd = parse_xml_full(materialize[XML](), String("."))
    var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=64)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var d = Data[DT, DynDims, 1](dims)
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    var actions = List[Float64](length=1, fill=ctrl)
    var act = List[Scalar[DT]](length=1, fill=Scalar[DT](0))
    var integ = StudioRk4Pyr(dims)
    for _ in range(nstep):
        # the driver's application at the start state — stage 0 either way
        for i in range(nv):
            d.qfrc.data[i] = Scalar[DT](0)
        apply_actions_fields[DT](sf, d, actions, act, fmd.timestep)
        apply_pose_transmission[DT](sf, m, d, integ.scratch, actions, act, fmd.timestep)
        if actuated:
            integ.step_actuated["cpu"](d, m, sf, actions, act, fmd.timestep)
        else:
            integ.step["cpu"](d, m)
    var out = List[Float64]()
    for i in range(nv):
        out.append(Float64(d.qvel.data[i]))
    return out^


def _mj[XML: String](nstep: Int, ctrl: Float64) raises -> List[Float64]:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[XML]())
    var d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.ctrl[0] = ctrl
    for _ in range(nstep):
        mujoco.mj_step(m, d)
    var qv = d.qvel.flatten().tolist()
    var out = List[Float64]()
    for i in range(Int(py=m.nv)):
        out.append(Float64(py=qv[i]))
    return out^


def _maxdiff(a: List[Float64], b: List[Float64]) -> Float64:
    var w = 0.0
    for i in range(min(len(a), len(b))):
        var e = abs(a[i] - b[i])
        if e > w:
            w = e
    return w


def _check[XML: String](name: String, ctrl: Float64) raises:
    print("=== RK4 per-stage actuation vs MuJoCo:", name, "===")
    var frozen5 = _maxdiff(_ours[XML](5, ctrl, False), _mj[XML](5, ctrl))
    print("   control arm — frozen qfrc, 5 steps: |d(qvel)| =", frozen5)
    assert_true(
        frozen5 > 100.0 * TOL,
        name + ": the frozen-qfrc control arm agrees with MuJoCo — this"
        " fixture cannot see per-stage actuation and the gate is blind",
    )
    for n in [1, 5, 30]:
        var e = _maxdiff(_ours[XML](n, ctrl, True), _mj[XML](n, ctrl))
        print("  ", n, "step(s): |d(qvel)| =", e)
        assert_true(
            e <= TOL,
            name + " after " + String(n) + " step(s) disagrees with MuJoCo by "
            + String(e) + " (tol " + String(TOL) + ") — a stage is not"
            " re-evaluating the actuator at its own state",
        )
    print("  PASS")


def test_position_servo() raises:
    _check[SERVO](String("position servo, dt 0.02"), 1.2)


def test_site_thrust_on_a_free_body() raises:
    _check[THRUST](String("off-axis site thrust, free body"), 3.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
