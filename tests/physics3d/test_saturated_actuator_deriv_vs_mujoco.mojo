"""A SATURATED actuator contributes nothing to the implicit `qDeriv`.

    pixi run mojo run -I . tests/physics3d/test_saturated_actuator_deriv_vs_mujoco.mojo

WHAT WAS MISSING. `mjd_actuator_vel` opens by skipping any actuator whose
force is clamped (`engine_derivative.c`):

    // skip if force is clamped by forcerange
    if (m->actuator_forcelimited[i]) {
      mjtNum force = d->actuator_force[i];
      if (force <= range[0] || force >= range[1]) continue;
    }

A saturated servo's force is PINNED at the bound, so it no longer depends on
velocity and its `-kv` term does not belong in `M_hat = M - dt*qDeriv`. We
carried `Model.dof_actdamp`, baked from `kv` at BUILD time, and added it
unconditionally — over-damping every saturated dof, every step.

⚠⚠ THE QUANTITY IS STATE-DEPENDENT, WHICH IS WHY A MODEL FIELD COULD NOT HOLD
IT. Whether an actuator is saturated changes with `ctrl` and with `qpos`, so
the fix is a per-step `Data.dof_actdamp` written by `apply_actions_fields`,
plus `META_IDX_ACTDAMP_LIVE` saying it was written. A step taken with no
actuation call at all leaves the flag down and falls back to the model-time
value — which is correct there, because nothing can be saturated.

⚠ MEASURED ON MUJOCO'S OWN `d.qDeriv`, which is only computed inside
`mj_implicitSkip` and is therefore all ZERO after `mj_forward` — it has to be
read after `mj_step`. On rainbow_robotics rby1, whose 24 position servos are
`forcerange="-270 270"` and saturate at `qpos0`, its diagonal reads

    dofs 0-5  (free base)   -5      joint damping alone
    dofs 6,7  (wheels)      -4005   damping + kv 4000, NOT clamped
    dofs 8+   (servos)      -5      SATURATED: no actuator term at all

against our -405 on every servo dof.

MEASURED CONSEQUENCE, worst |d(qpos)| against MuJoCo 3.10.0 under a fixed
random control sequence fed to both engines, 20 steps from each model's
keyframe:

    universal_robots_ur5e   1.488e-01 -> 2.220e-16
    franka_emika_panda      1.159e-01 -> 7.584e-03
    ufactory_xarm7          6.723e-02 -> 3.567e-07
    robotstudio_so101       2.167e-02 -> 3.469e-17
    google_barkour_vb       1.012e-02 -> 2.220e-16
    rethink_robotics_sawyer 3.384e-03 -> 3.469e-18

27 models in this tree pair a saturating actuator with an implicit integrator.
⚠ panda's 7.6e-03 is a separate residual this does not reach.

⚠ IT ONLY BITES UNDER SATURATION, so a model driven gently is exact either
way — which is why every gate here stayed green. `dampratio`, `actuatorfrcrange`
and now this are three defects on the same fault line: the actuator path is
correct at rest and wrong under drive.

⚠ TWO SITES, BOTH WIRED: `apply_actions_fields` (CPU) and
`apply_actions_kernel_gpu`, with the argument in the `ModelDefLike` TRAIT so
the compiler refuses a caller that forgets it. The batched env builds Euler
and RK4, neither of which reads the array — it is filled anyway so a future
implicit caller cannot silently get the model-time value.
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
from mojo_rl.physics3d.studio.stepping import StudioImpFastPyr

comptime DT = DType.float64

# Two identical hinges. `a`'s servo is `forcerange="-2 2"` and saturates at
# ctrl 0.5 (kp 4000 would ask for ~2000 N.m); `b`'s is unlimited and does not.
# ⚠ THE UNSATURATED TWIN IS THE POINT — it holds the `kv` term fixed while the
# saturated one loses it, so a fix that simply dropped actuator damping
# everywhere fails this file.
comptime XML = String(
    """<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <body><joint name="a" type="hinge" axis="0 0 1" damping="5"/>
      <geom type="box" size=".1 .1 .1" mass="1"/></body>
    <body pos="1 0 0"><joint name="b" type="hinge" axis="0 0 1" damping="5"/>
      <geom type="box" size=".1 .1 .1" mass="1"/></body>
  </worldbody>
  <actuator>
    <position joint="a" kp="4000" kv="400" forcelimited="true"
              forcerange="-2 2"/>
    <position joint="b" kp="4000" kv="400"/>
  </actuator>
</mujoco>"""
)

# MuJoCo 3.10.0, ctrl = (0.5, 0.5) held, `qpos` after N `mj_step`s. Its own
# `qDeriv` diagonal for this fixture is (-5, -405): damping alone on the
# saturated joint, damping + kv on the other.
#
# ⚠ SEVENTEEN SIGNIFICANT DIGITS, AND THE TOLERANCE IS 1e-15 BECAUSE OF IT.
# The first draft pasted these from a `precision=12` print and then had to
# allow 1e-12 — a bound set by MY TRANSCRIPTION, not by the arithmetic, and
# wide enough to hide a real regression in the term being gated.
comptime MJ1_A = 0.0004799999999999999
comptime MJ1_B = 0.0097959183673469417
comptime MJ5_A = 0.0034721280000000001
comptime MJ5_B = 0.047400912644058171
comptime MJ20_A = 0.015466666672530728
comptime MJ20_B = 0.16447556958197138

comptime BARKOUR = String(
    "references/mujoco_menagerie-main/google_barkour_vb/scene.xml"
)

# MuJoCo 3.10.0, keyframe 0, ctrl 0.4 held, `qpos` after 20 `mj_step`s.
comptime MJQ_0 = -0.00016550990361186273
comptime MJQ_1 = -1.7386315435508103e-05
comptime MJQ_2 = 0.28077411746013853
comptime MJQ_3 = 0.99999983859411812
comptime MJQ_4 = -3.1577861838468056e-05
comptime MJQ_5 = 0.00056714468683647507
comptime MJQ_6 = -1.2707482284673752e-05
comptime MJQ_7 = 0.15328794179853586
comptime MJQ_8 = 0.43331442146635263
comptime MJQ_9 = 0.72904791223071252
comptime MJQ_10 = 0.15503305972937453
comptime MJQ_11 = 0.43387177649328074
comptime MJQ_12 = 0.72934399507598791
comptime MJQ_13 = 0.15337440215131226
comptime MJQ_14 = 0.43342565099704361
comptime MJQ_15 = 0.72906493890439328
comptime MJQ_16 = 0.15508915918734384
comptime MJQ_17 = 0.43401223456662147
comptime MJQ_18 = 0.72940160965946788


def _run(nstep: Int) raises -> List[Float64]:
    var fmd = parse_xml_full(expand_mjcf(XML, String("")), String(""))
    var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=0)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
    var nact = dims.get_nact()
    var actions = List[Float64](length=nact, fill=0.5)
    var act = List[Scalar[DT]](length=nact, fill=Scalar[DT](0))
    var integ = StudioImpFastPyr(dims)
    for _ in range(nstep):
        for i in range(dims.get_nv()):
            d.qfrc.data[i] = Scalar[DT](0)
        apply_actions_fields[DT](sf, d, actions, act, fmd.timestep)
        integ.step["cpu"](d, m)
    var out = List[Float64]()
    for i in range(dims.get_nq()):
        out.append(Float64(d.qpos.data[i]))
    return out^


def test_saturated_and_unsaturated_match_mujoco() raises:
    """One saturated servo, one not, in the same model.

    ⚠ EXPECTED VALUES ARE MUJOCO'S `qpos` after N `mj_step`s at the same
    constant ctrl — not this engine's output.
    """
    print("=== saturated vs unsaturated servo, implicitfast ===")
    var want_a: List[Float64] = [MJ1_A, MJ5_A, MJ20_A]
    var want_b: List[Float64] = [MJ1_B, MJ5_B, MJ20_B]
    var steps: List[Int] = [1, 5, 20]
    var worst_a = 0.0
    var worst_b = 0.0
    for k in range(3):
        var got = _run(steps[k])
        assert_true(
            len(got) == 2,
            "the fixture must have two joints — the gate would be vacuous",
        )
        var ea = abs(got[0] - want_a[k])
        var eb = abs(got[1] - want_b[k])
        if ea > worst_a:
            worst_a = ea
        if eb > worst_b:
            worst_b = eb
        print(
            "  ", steps[k], "steps  a(saturated)", got[0], " mj", want_a[k],
            "   b(free)", got[1], " mj", want_b[k],
        )
    print("  worst |d| saturated", worst_a, "  unsaturated", worst_b)
    assert_true(
        worst_a < 1e-15,
        "the SATURATED servo must match MuJoCo; worst |d| = "
        + String(worst_a)
        + ". Its force is pinned at the +-2 bound, so `mjd_actuator_vel`"
        " skips it and its `kv` must NOT enter M_hat. Adding it over-damps"
        " the dof and the joint moves too little.",
    )
    # ⚠ THE NEGATIVE CONTROL, and the row that stops the obvious wrong fix:
    # dropping actuator damping everywhere would make THIS one wrong while
    # the row above went green.
    assert_true(
        worst_b < 1e-15,
        "the UNSATURATED servo must ALSO match MuJoCo; worst |d| = "
        + String(worst_b)
        + ". Its `kv` term is live and must stay in M_hat.",
    )
    print("  PASS")


def test_barkour_matches_mujoco_under_saturating_drive() raises:
    """A real model, whose twelve servos are all `forcerange`-limited.

    ⚠ EXPECTED VALUES ARE MUJOCO'S `qpos` after 20 `mj_step`s from keyframe 0
    at a constant ctrl of 0.4 on all twelve actuators.
    """
    print("=== google_barkour_vb under saturating drive ===")
    var src = read_model_source(BARKOUR)
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
    assert_true(
        dims.get_nkey() > 0,
        "barkour ships a `home` keyframe and this gate needs it",
    )
    var nq = dims.get_nq()
    var d = Data[DT, DynDims, 1](dims)
    for i in range(nq):
        d.qpos.data[i] = sf.key_qpos.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
    var nact = dims.get_nact()
    var actions = List[Float64](length=nact, fill=0.4)
    var act = List[Scalar[DT]](length=nact, fill=Scalar[DT](0))
    var integ = StudioImpFastPyr(dims)
    for _ in range(20):
        for i in range(dims.get_nv()):
            d.qfrc.data[i] = Scalar[DT](0)
        apply_actions_fields[DT](sf, d, actions, act, fmd.timestep)
        integ.step["cpu"](d, m)

    # MuJoCo's own `qpos` for this run. Slots 0..6 are the free joint.
    var want: List[Float64] = [
        MJQ_0, MJQ_1, MJQ_2, MJQ_3, MJQ_4, MJQ_5, MJQ_6,
        MJQ_7, MJQ_8, MJQ_9, MJQ_10, MJQ_11, MJQ_12, MJQ_13,
        MJQ_14, MJQ_15, MJQ_16, MJQ_17, MJQ_18,
    ]
    assert_true(
        len(want) == nq,
        "the expected vector must cover every qpos slot — nq is " + String(nq),
    )
    var worst = 0.0
    for i in range(nq):
        var e = abs(Float64(d.qpos.data[i]) - want[i])
        if e > worst:
            worst = e
    print("  worst |d(qpos)| =", worst)
    # ⚠ VACUITY. The legs must have MOVED off the keyframe.
    assert_true(
        abs(Float64(d.qpos.data[8]) - Float64(sf.key_qpos.data[8])) > 1e-3,
        "the model did not move — the gate would compare a pose neither"
        " engine integrated",
    )
    assert_true(
        worst < 1e-12,
        "barkour must match MuJoCo; worst |d(qpos)| = " + String(worst)
        + ". Before saturation was honoured this was 1.012e-02.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
