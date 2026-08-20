"""`<joint actuatorfrcrange>` — MuJoCo's SECOND actuator force clamp.

    pixi run mojo run -I . tests/physics3d/test_jnt_actfrcrange_vs_mujoco.mojo

WHAT WAS MISSING. `mj_fwdActuation` clamps the actuator force TWICE:

    clampVec(force, m->actuator_forcerange, m->actuator_forcelimited, nu, NULL);
                                                    // engine_forward.c:417
    clampVec(d->qfrc_actuator, m->jnt_actfrcrange, m->jnt_actfrclimited,
             m->njnt, m->jnt_dofadr);               // engine_forward.c:477

The first is per-ACTUATOR, on that actuator's SCALAR force, before the moment
— we had it. The second is per-JOINT, on the ACCUMULATED `qfrc_actuator` at
the joint's DOF address, after every actuator has contributed — we had no
trace of it. `actuatorfrcrange` was parsed nowhere.

⚠⚠ HAVING ONE IS NOT HAVING THE OTHER, AND THE MODELS THAT NEED IT MOSTLY
DECLARE ONLY THE SECOND. On unitree_g1 `actuator_forcelimited` is FALSE on all
29 actuators while `jnt_actfrclimited` is TRUE on 29 of 30 joints — so the
clamp we implemented never fires on g1 and the one we did not implement is the
only force limit that model has. 481 of this tree's 2519 joints declare one,
across 20 robots.

MEASURED, `qfrc_actuator` at `qpos0` under a fixed random `ctrl`, worst
|d| against MuJoCo 3.10.0 across every model that declares it — this is the
FORCE, so no integration or chaos is involved:

    pal_tiago (velocity)  7.940e+04 -> 0.000e+00
    pal_tiago_dual        5.937e+04 -> 0.000e+00
    pal_talos             9.126e+03 -> 5.884e-15
    franka_fr3            3.488e+03 -> 0.000e+00
    flexiv_rizon4         4.116e+02 -> 0.000e+00
    unitree_g1            4.705e+02 -> 0.000e+00
    aloha                 6.652e+01 -> 0.000e+00
    booster_t1            5.621e+01 -> 0.000e+00
    trossen_wxai          3.949e+01 -> 0.000e+00
    sharpa_wave           1.692e+00 -> 0.000e+00

i.e. up to 79 kN.m of phantom torque on a single dof. (apollo, spot,
berkeley_humanoid, fourier_n1 and pndbotics_adam_lite read 0 both ways at that
particular sample — their random `ctrl` did not push any joint past its limit.
A sweep that reported only those five would have exonerated the defect.)

⚠ IT ONLY BITES UNDER DRIVE, which is why g1 looked perfect. From its `stand`
keyframe, stepped with the keyframe's own `ctrl`, g1 agreed with MuJoCo to
1.1e-15 over 100 steps BEFORE this fix — the servos sit at their setpoints and
never ask for more torque than the joint allows. Drive it with the studio's
random policy from `qpos0` and the wrists, which are `kp=500` servos over a
+-1.61 rad range clamped to +-5 N.m, ask for ~800 N.m:

    worst |d(qpos)| vs MuJoCo, same random ctrl sequence fed to both
                 before      after
      1 step     3.746e-02   2.220e-16
     20 steps    1.501e-01   4.302e-16

"Fine at rest, wrong under a policy" is the same fingerprint `dampratio` had.

⚠ NO DEGREE CONVERSION — verified against the runtime, not assumed. In a
`<compiler angle="degree"/>` model, `range="-90 90"` compiles to +-1.5708 rad
while `actuatorfrcrange="-5 5"` stays +-5: one is an angle, the other a
torque. Applying `deg_factor` to it would have scaled g1's +-5 N.m to +-0.087.

⚠ TWO CLAMP SITES, BOTH WIRED. `apply_actions_fields` (CPU) and
`apply_actions_kernel_gpu` — this file's `ctrllimited` predecessor records why:
fixing one alone leaves the two targets computing different forces from the
same action. No in-repo env model declares `actuatorfrcrange` (audited: 0 of
60), so the GPU path has nothing to clamp today and everything to clamp the
moment someone trains on a Menagerie port.

⚠ THE ORDER MATTERS. The clamp sits AFTER the actuator loop and BEFORE the
fixed-tendon springs. A spring is `qfrc_passive` and is NOT subject to this
limit; `d.qfrc` is our single accumulator for both, so position is the only
thing keeping them separable.
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

comptime DT = DType.float64

comptime G1 = String("references/mujoco_menagerie-main/unitree_g1/scene.xml")

# Four joints, four spellings. `<compiler angle="degree"/>` is LOAD-BEARING
# here for the opposite reason it usually is: it proves the torque is NOT
# converted while the range beside it IS.
comptime XML = String(
    """<mujoco>
  <compiler angle="degree"/>
  <default>
    <default class="klass">
      <joint actuatorfrcrange="-3 3"/>
    </default>
    <default class="zero">
      <joint actuatorfrcrange="0 0"/>
    </default>
  </default>
  <worldbody>
    <body>
      <joint name='a' type='hinge' axis='0 0 1' range='-90 90'
             actuatorfrcrange='-5 5'/>
      <geom type='box' size='.1 .1 .1' mass='1'/>
      <body pos='0 0 .3'>
        <joint name='b' class='klass' type='hinge' axis='0 0 1'/>
        <geom type='box' size='.1 .1 .1' mass='1'/>
        <body pos='0 0 .3'>
          <joint name='c' class='zero' type='hinge' axis='0 0 1'/>
          <geom type='box' size='.1 .1 .1' mass='1'/>
          <body pos='0 0 .3'>
            <joint name='d' type='hinge' axis='0 0 1'/>
            <geom type='box' size='.1 .1 .1' mass='1'/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint='a' kp='1000'/>
    <position joint='b' kp='1000'/>
    <position joint='c' kp='1000'/>
    <position joint='d' kp='1000'/>
  </actuator>
</mujoco>"""
)


def test_actfrcrange_is_parsed_on_both_paths() raises:
    """Element, class, and the two negative controls.

    ⚠ EXPECTED VALUES ARE MUJOCO'S OWN `jnt_actfrcrange` /
    `jnt_actfrclimited`, read off the 3.10.0 runtime for this fixture:
    joint a [-5, 5] limited, b [-3, 3] limited, c [0, 0] UNLIMITED (the
    undefined marker), d [0, 0] unlimited.
    """
    print("=== <joint actuatorfrcrange> parsing ===")
    var fmd = parse_xml_full(expand_mjcf(XML, String("")), String(""))
    assert_true(
        len(fmd.joints) == 4,
        "fixture did not parse four joints — the gate would be vacuous",
    )
    for i in range(4):
        ref j = fmd.joints[i]
        print(
            "  joint", i, " actfrc_limited", j.is_actfrc_limited,
            " [", j.actfrc_min, ",", j.actfrc_max, "]",
            "   range [", j.range_min, ",", j.range_max, "]",
        )

    ref a = fmd.joints[0]
    assert_true(
        a.is_actfrc_limited and abs(a.actfrc_min - (-5.0)) < 1e-12
        and abs(a.actfrc_max - 5.0) < 1e-12,
        "an element `actuatorfrcrange='-5 5'` must give limited [-5, 5], got "
        + String(a.is_actfrc_limited) + " [" + String(a.actfrc_min) + ", "
        + String(a.actfrc_max) + "]",
    )
    # ⚠ THE DEGREE CONTROL, on the SAME joint. `range='-90 90'` must convert
    # to +-1.5708 rad while the torque beside it does not move. Verified
    # against the runtime: MuJoCo reports exactly this pair.
    assert_true(
        abs(a.range_min - (-1.5707963267948966)) < 1e-9,
        "`range='-90 90'` under `angle='degree'` must be +-1.5708 rad — if it"
        " is not, this fixture cannot say anything about the torque, got "
        + String(a.range_min),
    )
    ref b = fmd.joints[1]
    assert_true(
        b.is_actfrc_limited and abs(b.actfrc_min - (-3.0)) < 1e-12
        and abs(b.actfrc_max - 3.0) < 1e-12,
        "a `<default class>` `actuatorfrcrange='-3 3'` must reach the joint —"
        " Menagerie states it in a class as often as inline; got "
        + String(b.is_actfrc_limited) + " [" + String(b.actfrc_min) + ", "
        + String(b.actfrc_max) + "]",
    )
    # ⚠ NEGATIVE CONTROL 1. `"0 0"` is MuJoCo's undefined marker.
    ref c = fmd.joints[2]
    assert_true(
        not c.is_actfrc_limited,
        "`actuatorfrcrange='0 0'` is the UNDEFINED marker, not a zero-width"
        " range — clamping to [0, 0] would deliver ZERO torque where MuJoCo"
        " delivers the full command",
    )
    # ⚠ NEGATIVE CONTROL 2, the load-bearing one: without it this file would
    # pass against an implementation that limited every joint.
    ref d = fmd.joints[3]
    assert_true(
        not d.is_actfrc_limited,
        "a joint declaring no `actuatorfrcrange` anywhere must stay UNLIMITED",
    )
    print("  PASS")


def test_g1_actfrcrange_matches_mujoco() raises:
    """The real model, against MuJoCo's own `jnt_actfrcrange`.

    ⚠ MEASURED ON THE 3.10.0 RUNTIME: g1 reports `jnt_actfrclimited` true on
    29 of its 30 joints (the free root is the exception) and its three wrist
    joints per arm are +-25, +-5, +-5 N.m.
    """
    print("=== unitree_g1 jnt_actfrcrange ===")
    var src = read_model_source(G1)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var n_lim = 0
    var tightest = 1e30
    for i in range(len(fmd.joints)):
        ref j = fmd.joints[i]
        if j.is_actfrc_limited:
            n_lim += 1
            var w = abs(j.actfrc_max)
            if w < tightest:
                tightest = w
    print("  actfrclimited on", n_lim, "of", len(fmd.joints),
          " tightest |frc|", tightest, " (MuJoCo: 29 of 30, 5)")
    assert_true(
        len(fmd.joints) == 30 and n_lim == 29,
        "MuJoCo reports jnt_actfrclimited true on 29 of g1's 30 joints; we"
        " have " + String(n_lim) + " of " + String(len(fmd.joints)),
    )
    assert_true(
        abs(tightest - 5.0) < 1e-12,
        "g1's wrist pitch/yaw joints declare `actuatorfrcrange='-5 5'`, so"
        " the tightest bound must be 5 N.m; got " + String(tightest)
        + ". A value near 0.087 means `deg_factor` was applied to a TORQUE.",
    )
    print("  PASS")


def test_g1_qfrc_actuator_matches_mujoco() raises:
    """The FORCE, which is the only thing the parse is for.

    ⚠⚠ THIS IS WHAT THE RECORD TEST CANNOT CARRY. The observable is
    `qfrc_actuator` after every actuator has contributed, which is where
    MuJoCo's second clamp acts — a parser that read the range perfectly and a
    force path that ignored it passes the two tests above and fails here.

    ⚠ THE CTRL IS CHOSEN TO SATURATE. g1's actuators are `<position kp=500>`
    over `inheritrange` ctrlranges, so a command at the end of a wrist's
    +-1.61 rad range asks for roughly 800 N.m against a +-5 N.m joint. At
    `ctrl = 0` from `qpos0` every servo is already at its setpoint, the clamp
    never fires, and this gate would be vacuous — which is exactly why g1
    matched MuJoCo to 1.1e-15 for 100 steps while the defect was live.

    ⚠ EXPECTED VALUES ARE MUJOCO'S `d.qfrc_actuator`, computed by
    `mj_forward` on the 3.10.0 runtime at `qpos0` with this same ctrl.
    """
    print("=== unitree_g1 qfrc_actuator under a saturating ctrl ===")
    var src = read_model_source(G1)
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
    var nv = dims.get_nv()
    var nact = dims.get_nact()

    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)
        d.qfrc.data[i] = Scalar[DT](0)

    # ctrl = +1 rad on every actuator: inside every ctrlrange (so the
    # ctrllimited clamp is NOT what is being measured) and far past every
    # actuatorfrcrange at kp = 500.
    var actions = List[Float64](length=nact, fill=1.0)
    var act = List[Scalar[DT]](length=nact, fill=Scalar[DT](0))
    apply_actions_fields[DT](sf, d, actions, act, fmd.timestep)

    # MuJoCo `d.qfrc_actuator` for this exact state and ctrl. The six free-dof
    # entries are zero; every actuated dof saturates at its joint's bound.
    var want: List[Float64] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
        88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
        88.0, 50.0, 50.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
    ]
    assert_true(
        len(want) == nv,
        "the expected vector must cover every dof — nv is " + String(nv)
        + " and the literal has " + String(len(want)),
    )
    var worst = 0.0
    var n_sat = 0
    for i in range(nv):
        var got = Float64(d.qfrc.data[i])
        var e = abs(got - want[i])
        if e > worst:
            worst = e
        if want[i] != 0.0:
            n_sat += 1
    print("  worst |d(qfrc_actuator)| =", worst, " over", nv, "dofs,",
          n_sat, "saturating")
    # ⚠ VACUITY. If nothing saturates, an engine with no clamp at all passes.
    assert_true(
        n_sat >= 29,
        "every actuated dof must be AT its joint's bound for this gate to"
        " mean anything — only " + String(n_sat) + " are",
    )
    assert_true(
        worst < 1e-9,
        "qfrc_actuator must match MuJoCo's; worst |d| = " + String(worst)
        + ". Before `jnt_actfrcrange` existed here this was 4.705e+02 on g1"
        " and 7.940e+04 on pal_tiago — the servo had no ceiling at all.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
