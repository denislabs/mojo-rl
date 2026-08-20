"""No engine may invent a speed limit MuJoCo does not have.

    pixi run mojo run -I . tests/physics3d/test_no_invented_velocity_clamp_vs_mujoco.mojo

WHAT WAS THERE. All three integrators finished their velocity update with

    var qvel_max = Scalar[DTYPE](100.0)
    ...
    elif qvel_new >  qvel_max: qvel_new =  qvel_max
    elif qvel_new < -qvel_max: qvel_new = -qvel_max

A hard saturation at 100 rad/s, silent, on every dof, every step.

WHAT MUJOCO DOES. `mj_checkVel` (`engine_forward.c`) is its ONLY velocity
guard, and it does not rescale anything:

    for (int j=0; j < nv; j++)
      if (mju_isBad(d->qvel[i])) { mj_warning(d, mjWARN_BADQVEL, i);
                                   mj_resetData(m, d); return; }

`mju_isBad` is NaN, inf, or `|v| > mjMAXVAL` — and mjMAXVAL is **1e10**
(`mjmodel.h:25`), not 100. Below that bound MuJoCo integrates whatever the
dynamics produce; above it, it warns and resets the entire state. A clamp is
neither behaviour.

⚠⚠ IT IS INVISIBLE UNTIL SOMETHING GOES FAST, which is why it survived every
gate in this tree. Across the 85 loadable Menagerie scenes, exactly ONE
reaches it one step out of its own keyframe — and it reaches it hard.
kinova_gen3 ships `home` with its base and shoulder collision hulls 12 mm
interpenetrated, so MuJoCo's very first step answers `qacc = -12405` and
`|qvel|max = 165.583` at the ctrl this file holds. Ours stopped dead on
100.0000.

⚠ THE FINGERPRINT IS A ROUND NUMBER IN A POSITION. kinova's `joint_7` starts
at pi/2 and our first step put it at 1.37079633 — exactly `pi/2 - 0.2`, and
0.2 is `100 * dt`. A joint that lands a suspiciously round distance from
where it began is a clamp, not a force.

MEASURED CONSEQUENCE (worst |d(qpos)| vs MuJoCo 3.10.0, same control fed to
both engines, one step from keyframe 0):

    kinova_gen3   1.317e-01 -> 4.351e-02

⚠ THE RESIDUAL IS A DIFFERENT DEFECT and this gate does not claim it: MuJoCo
finds four contacts between those two hulls where we find one. That is why
the real-model half below asserts the PROPERTY (the velocity is no longer
pinned to 100) rather than a qpos value it cannot yet reach.

⚠ ALL THREE INTEGRATORS, AND RK4 TWICE. Euler, implicitfast and RK4 each
carried the clamp, and RK4 carried a second one on the COMBINED velocity it
integrates position with — so its `qpos` was clamped on a path its `qvel`
never took. The fixture below runs all three.
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.integrator.implicit import ImplicitIntegrator
from mojo_rl.physics3d.integrator.rk4 import RK4Integrator
from mojo_rl.physics3d.types import ConeType

comptime DT = DType.float64

# Two hinges that differ ONLY in gear. `fast` is geared 1000 and passes the
# old bound three times over in a single step; `slow` is geared 1 and never
# comes near it.
#
# ⚠ THE SLOW TWIN IS THE NEGATIVE CONTROL, and it is not decoration: raising
# the bound is a one-line change that could just as easily have been a
# one-line change to the arithmetic. `slow` pins the ordinary path while
# `fast` moves.
#
# ⚠ `gravity="0 0 0"` and no damping, so the answer is exactly
# `qacc = gear*ctrl / I` with `I = m*(0.2^2+0.2^2)/12 = 0.0066667` — 150000
# rad/s^2, i.e. 300 rad/s after one 2 ms step. Nothing about the reference
# values depends on a solver.
comptime _BODY = String(
    """
  <worldbody>
    <body><joint name="fast" type="hinge" axis="0 0 1"/>
      <geom type="box" size=".1 .1 .1" mass="1"/></body>
    <body pos="1 0 0"><joint name="slow" type="hinge" axis="0 0 1"/>
      <geom type="box" size=".1 .1 .1" mass="1"/></body>
  </worldbody>
  <actuator>
    <motor joint="fast" gear="1000"/>
    <motor joint="slow" gear="1"/>
  </actuator>
</mujoco>"""
)

comptime XML_EULER = String(
    """<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="Euler" gravity="0 0 0"/>"""
) + _BODY
comptime XML_IMPFAST = String(
    """<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 0"/>"""
) + _BODY
comptime XML_RK4 = String(
    """<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="RK4" gravity="0 0 0"/>"""
) + _BODY

# MuJoCo 3.10.0, ctrl = (1, 1) held, after N `mj_step`s. Seventeen digits, so
# the tolerance below is set by the arithmetic and not by a transcription.
#
# ⚠ RK4's `qpos` IS NOT EULER'S while its `qvel` IS. Under a constant force
# RK4's four stages average to half the Euler position increment, and that
# half is exactly what its second clamp used to eat.
comptime MJ_E1_FAST = 0.5999999999999999
comptime MJ_E1_SLOW = 0.0005999999999999998
comptime MJ_E3_FAST = 3.599999999999999
comptime MJ_E3_SLOW = 0.003599999999999999
comptime MJ_R1_FAST = 0.29999999999999993
comptime MJ_R1_SLOW = 0.0002999999999999999
comptime MJ_R3_FAST = 2.6999999999999997
comptime MJ_R3_SLOW = 0.0026999999999999993
comptime MJ_V1_FAST = 299.99999999999994
comptime MJ_V1_SLOW = 0.29999999999999993
comptime MJ_V3_FAST = 899.9999999999999
comptime MJ_V3_SLOW = 0.8999999999999998

comptime KINOVA = String(
    "references/mujoco_menagerie-main/kinova_gen3/scene.xml"
)

comptime _EULER = EulerIntegrator[DT, DynDims, ConeType.PYRAMIDAL, 1, "newton"]
comptime _IMPFAST = ImplicitIntegrator[
    DT, DynDims, ConeType.PYRAMIDAL, 1, "newton", SKIP_RNE_DERIV=True,
]
comptime _RK4 = RK4Integrator[DT, DynDims, ConeType.PYRAMIDAL, 1, "newton"]






def _run_euler(xml: String, nstep: Int) raises -> List[Float64]:
    """`qpos` then `qvel`, concatenated, after `nstep` steps at ctrl 1."""
    var fmd = parse_xml_full(expand_mjcf(xml, String("")), String(""))
    var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=0)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
    var nact = dims.get_nact()
    var actions = List[Float64](length=nact, fill=1.0)
    var act = List[Scalar[DT]](length=nact, fill=Scalar[DT](0))
    var integ = _EULER(dims)
    for _ in range(nstep):
        for i in range(dims.get_nv()):
            d.qfrc.data[i] = Scalar[DT](0)
        apply_actions_fields[DT](sf, d, actions, act, fmd.timestep)
        integ.step["cpu"](d, m)
    var out = List[Float64]()
    for i in range(dims.get_nq()):
        out.append(Float64(d.qpos.data[i]))
    for i in range(dims.get_nv()):
        out.append(Float64(d.qvel.data[i]))
    return out^


def _run_impfast(xml: String, nstep: Int) raises -> List[Float64]:
    var fmd = parse_xml_full(expand_mjcf(xml, String("")), String(""))
    var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=0)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
    var nact = dims.get_nact()
    var actions = List[Float64](length=nact, fill=1.0)
    var act = List[Scalar[DT]](length=nact, fill=Scalar[DT](0))
    var integ = _IMPFAST(dims)
    for _ in range(nstep):
        for i in range(dims.get_nv()):
            d.qfrc.data[i] = Scalar[DT](0)
        apply_actions_fields[DT](sf, d, actions, act, fmd.timestep)
        integ.step["cpu"](d, m)
    var out = List[Float64]()
    for i in range(dims.get_nq()):
        out.append(Float64(d.qpos.data[i]))
    for i in range(dims.get_nv()):
        out.append(Float64(d.qvel.data[i]))
    return out^


def _run_rk4(xml: String, nstep: Int) raises -> List[Float64]:
    var fmd = parse_xml_full(expand_mjcf(xml, String("")), String(""))
    var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=0)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
    var nact = dims.get_nact()
    var actions = List[Float64](length=nact, fill=1.0)
    var act = List[Scalar[DT]](length=nact, fill=Scalar[DT](0))
    var integ = _RK4(dims)
    for _ in range(nstep):
        for i in range(dims.get_nv()):
            d.qfrc.data[i] = Scalar[DT](0)
        apply_actions_fields[DT](sf, d, actions, act, fmd.timestep)
        integ.step["cpu"](d, m)
    var out = List[Float64]()
    for i in range(dims.get_nq()):
        out.append(Float64(d.qpos.data[i]))
    for i in range(dims.get_nv()):
        out.append(Float64(d.qvel.data[i]))
    return out^


def _check(
    name: String, got: List[Float64],
    qf: Float64, qs: Float64, vf: Float64, vs: Float64,
) raises -> Float64:
    assert_true(
        len(got) == 4,
        "the fixture must have two hinges (2 qpos + 2 qvel); got "
        + String(len(got)) + " values — the gate would be vacuous",
    )
    print(
        "  ", name, " qpos", got[0], got[1], "  qvel", got[2], got[3],
    )
    var worst = abs(got[0] - qf)
    if abs(got[1] - qs) > worst:
        worst = abs(got[1] - qs)
    if abs(got[2] - vf) > worst:
        worst = abs(got[2] - vf)
    if abs(got[3] - vs) > worst:
        worst = abs(got[3] - vs)
    # ⚠ THE FAST DOF FIRST, with the number the clamp would have produced
    # spelled out, so a failure here reads as "still clamped" and not as a
    # generic mismatch.
    assert_true(
        abs(got[2] - vf) < 1e-9,
        name + ": fast qvel is " + String(got[2]) + " but MuJoCo integrates"
        " to " + String(vf) + ". A value pinned at 100 is the invented clamp;"
        " anything else is an arithmetic regression.",
    )
    assert_true(
        abs(got[0] - qf) < 1e-9,
        name + ": fast qpos is " + String(got[0]) + " but MuJoCo reaches "
        + String(qf) + ".",
    )
    # ⚠ THE NEGATIVE CONTROL. The slow hinge never approaches any bound, so it
    # must be untouched by whatever the fast one exposed.
    assert_true(
        abs(got[3] - vs) < 1e-12 and abs(got[1] - qs) < 1e-12,
        name + ": the SLOW hinge moved. It peaks at " + String(vs)
        + " rad/s and cannot reach any velocity bound, so a mismatch here"
        " means the ordinary integration path changed: qpos " + String(got[1])
        + " vs " + String(qs) + ", qvel " + String(got[3]) + " vs "
        + String(vs),
    )
    return worst


def test_all_three_integrators_pass_100_rad_per_s() raises:
    """300 and 900 rad/s, in Euler, implicitfast and RK4."""
    print("=== a hinge geared past the old bound ===")
    var worst = 0.0
    var w = _check(
        String("euler   n=1"), _run_euler(XML_EULER, 1),
        MJ_E1_FAST, MJ_E1_SLOW, MJ_V1_FAST, MJ_V1_SLOW,
    )
    if w > worst:
        worst = w
    w = _check(
        String("euler   n=3"), _run_euler(XML_EULER, 3),
        MJ_E3_FAST, MJ_E3_SLOW, MJ_V3_FAST, MJ_V3_SLOW,
    )
    if w > worst:
        worst = w
    # ⚠ implicitfast REACHES THE SAME NUMBERS HERE ON PURPOSE. With no damping
    # and no actuator kv, `M_hat` IS `M`, so any difference between the two
    # rows would be the integrator disagreeing with itself.
    w = _check(
        String("impfast n=1"), _run_impfast(XML_IMPFAST, 1),
        MJ_E1_FAST, MJ_E1_SLOW, MJ_V1_FAST, MJ_V1_SLOW,
    )
    if w > worst:
        worst = w
    w = _check(
        String("impfast n=3"), _run_impfast(XML_IMPFAST, 3),
        MJ_E3_FAST, MJ_E3_SLOW, MJ_V3_FAST, MJ_V3_SLOW,
    )
    if w > worst:
        worst = w
    # ⚠ RK4's qpos IS HALF EULER'S, and that is the second clamp's target.
    w = _check(
        String("rk4     n=1"), _run_rk4(XML_RK4, 1),
        MJ_R1_FAST, MJ_R1_SLOW, MJ_V1_FAST, MJ_V1_SLOW,
    )
    if w > worst:
        worst = w
    w = _check(
        String("rk4     n=3"), _run_rk4(XML_RK4, 3),
        MJ_R3_FAST, MJ_R3_SLOW, MJ_V3_FAST, MJ_V3_SLOW,
    )
    if w > worst:
        worst = w
    print("  worst |diff| over all six rows", worst)
    print("  PASS")


def test_kinova_first_step_is_not_pinned_at_100() raises:
    """The real model that found it, from its own `home` keyframe.

    ⚠ THIS ASSERTS A PROPERTY, NOT A VALUE. MuJoCo answers |qvel|max 165.583
    here; we answer 155.79, and the gap is a SEPARATE defect — four contacts
    between the base and shoulder hulls where we detect one. Pinning 165.583
    would make this file fail for the contact gap and stop saying anything
    about the clamp. What it does state is what the clamp violated: the
    engine integrates past 100 rad/s when the dynamics say so.
    """
    print("=== kinova_gen3, one step from `home` ===")
    var src = read_model_source(KINOVA)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    var tries = 0
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") == -1 or tries > 24:
                raise e
            tries += 1
            verts = verts * 2
            dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var d = Data[DT, DynDims, 1](dims)
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    assert_true(
        dims.get_nkey() > 0,
        "kinova_gen3 must carry a keyframe — without it this gate starts"
        " from a pose that does not self-penetrate and measures nothing",
    )
    for i in range(nq):
        d.qpos.data[i] = sf.key_qpos.data[i]
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)
    var nact = dims.get_nact()
    var actions = List[Float64](length=nact, fill=0.4)
    var act = List[Scalar[DT]](length=nact, fill=Scalar[DT](0))
    var integ = _IMPFAST(dims)
    for i in range(nv):
        d.qfrc.data[i] = Scalar[DT](0)
    apply_actions_fields[DT](sf, d, actions, act, fmd.timestep)
    integ.step["cpu"](d, m)
    var vmax = 0.0
    for i in range(nv):
        var v = abs(Float64(d.qvel.data[i]))
        if v > vmax:
            vmax = v
    print("  |qvel|max", vmax, "   (MuJoCo 3.10.0: 165.5830739476927)")
    assert_true(
        abs(vmax - 100.0) > 1e-6,
        "|qvel|max is " + String(vmax) + " — exactly the invented bound."
        " MuJoCo reaches 165.583 on this step.",
    )
    assert_true(
        vmax > 120.0,
        "|qvel|max is " + String(vmax) + ", well short of the 165.583 MuJoCo"
        " integrates to. Some bound is still truncating this dof.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
