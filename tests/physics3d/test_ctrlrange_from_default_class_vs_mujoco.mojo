"""A `ctrlrange` stated in a `<default>` class must actually CLAMP.

    pixi run mojo run -I . tests/physics3d/test_ctrlrange_from_default_class_vs_mujoco.mojo

WHAT WAS MISSING. `ctrllimited` defaults to "auto" — limited iff a range was
DEFINED — and the rule was written INLINE, twice. The element path had it
(and was fixed once, for the `"0 0"` marker); the `<default>` block read
`ctrlrange` into the class's min/max and never touched `motor_ctrl_limited`.
So a range stated in a CLASS produced a bound that NOTHING CLAMPED AGAINST.

⚠⚠ THE RANGE WAS RIGHT THE WHOLE TIME, WHICH IS WHY IT HID. Swept across all
83 loadable Menagerie scenes, 1192 actuators, against MuJoCo 3.10.0's own
`actuator_ctrllimited` / `actuator_ctrlrange`:

    ctrllimited WRONG : 322 / 1192   (27 robots, ALL false-for-true)
    ctrlrange   WRONG : 0   / 1192

Every bound was parsed correctly and carried all the way to the force law,
where `apply_actions_fields` guards its clamp on `ACT_IDX_CTRL_LIMITED` —
correctly, because an unlimited actuator must NOT be squeezed into the
fallback range. A record dump compares equal on the numbers that matter and
the flag reads as a detail. It is not one: it is the difference between
having a clamp and not having one.

⚠ THE 27: aloha, anymal_b, anymal_c, fourier_n1, franka_emika_panda,
barkour_v0, barkour_vb, hello_robot_stretch{,_3}, i2rt_yam, kuka_iiwa_14,
leap_hand, robotis_op3, shadow_hand, skydio_x2, tetheria_aero_hand_open,
trossen_vx300s, ufactory_xarm7, umi_gripper, unitree_a1, unitree_go1,
unitree_go2, ur5e, ur10e, wonik_allegro — i.e. most of the quadrupeds and
most of the hands.

⚠ IT ONLY BITES WHEN `ctrl` IS OUT OF RANGE, the same reason `inheritrange`
survived. Measured on `google_barkour_vb`, whose knee class says
`<general ctrlrange="0.1 2.34346"/>` while both `ctrl` and `qpos0` are 0:
MuJoCo clamps to 0.1 and reports `actuator_force` 5 N.m on each of the four
knees. We reported 0 — on all twelve actuators, every step. Stepped from
`qpos0`, the worst `qpos` divergence against MuJoCo went

    50 steps  1.725e-02 -> 3.412e-03
   100 steps  1.095e-01 -> 1.125e-03

THE FIX is `_apply_ctrlrange`, the twin of `_apply_forcerange` — which was
written as a shared helper from the start and never drifted. Both callers now
go through one function, so the two paths cannot disagree again.

⚠ `autolimits` IS ASSUMED TRUE, matching MuJoCo's own default and the element
path's long-standing behaviour. Audited: zero files in this tree (Menagerie
included) set `autolimits="false"`.
"""

from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.fields import Data, Model, DynDims

comptime DT = DType.float64

comptime BARKOUR = String(
    "references/mujoco_menagerie-main/google_barkour_vb/scene.xml"
)

# Six actuators on one joint. Every one states its range a different way, so
# a single wrong precedence rule cannot satisfy all six.
#
# ⚠ `<compiler angle="radian"/>` IS LOAD-BEARING — MJCF defaults to DEGREES
# and a fixture without it has bitten this tree before.
comptime XML = String(
    """<mujoco>
  <compiler angle="radian"/>
  <default>
    <default class="klass">
      <general ctrlrange="-2 0.5"/>
    </default>
    <default class="zerorange">
      <general ctrlrange="0 0"/>
    </default>
    <default class="unlimited">
      <general ctrlrange="-2 0.5" ctrllimited="false"/>
    </default>
  </default>
  <worldbody>
    <body>
      <joint name='j' type='hinge' axis='0 0 1'/>
      <geom type='box' size='.1 .1 .1' mass='2'/>
    </body>
  </worldbody>
  <actuator>
    <general name='a' class='klass'     joint='j'/>
    <general name='b' joint='j' ctrlrange='-3 4'/>
    <general name='c' class='klass'     joint='j' ctrlrange='-3 4'/>
    <general name='d' class='zerorange' joint='j'/>
    <general name='e' class='unlimited' joint='j'/>
    <general name='f' joint='j'/>
  </actuator>
</mujoco>"""
)


def _build(
    xml: String, base: String
) raises -> Tuple[List[Float64], List[Float64], List[Float64]]:
    """(limited, lo, hi) per actuator, through the runtime path the studio uses.

    ⚠ THE MESH BUDGET IS DISCOVERED, NOT GUESSED — the same retry-on-raise
    loop the studio's loader runs, so this gate does not go red the next time
    a Menagerie mesh grows.
    """
    var fmd = parse_xml_full(expand_mjcf(xml, base), base)
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
    var lim = List[Float64]()
    var lo = List[Float64]()
    var hi = List[Float64]()
    for i in range(len(fmd.actuators)):
        ref a = fmd.actuators[i]
        lim.append(1.0 if a.is_ctrl_limited else 0.0)
        lo.append(a.ctrl_min)
        hi.append(a.ctrl_max)
    return (lim^, lo^, hi^)


def test_class_ctrlrange_is_limited() raises:
    """The class path, its precedence, and the two negative controls.

    ⚠ EXPECTED VALUES ARE MUJOCO'S RULE, not our output: "auto" means limited
    iff a range was defined, `"0 0"` is the undefined marker, an explicit
    `ctrllimited` wins, and an element range overrides the class's.
    """
    print("=== ctrlrange from a <default> class ===")
    var _r = _build(XML, String(""))
    # ⚠ COPIED OUT OF THE TUPLE FIRST. Two interior references into one tuple
    # inside a single call is an invalidated-reference error in Mojo nightly.
    var lim = _r[0].copy()
    var lo = _r[1].copy()
    var hi = _r[2].copy()
    assert_true(
        len(lim) == 6,
        "fixture did not parse six actuators — the gate would be vacuous",
    )
    var names: List[String] = [
        "a class-range", "b element-range", "c element-over-class",
        "d class 0 0", "e class ctrllimited=false", "f no range",
    ]
    for i in range(6):
        print(
            "  ", names[i], " limited", lim[i] != 0.0,
            " [", lo[i], ",", hi[i], "]",
        )

    # THE BUG. A range stated only in the class must be limited AND keep its
    # bounds.
    assert_true(
        lim[0] != 0.0 and abs(lo[0] - (-2.0)) < 1e-12
        and abs(hi[0] - 0.5) < 1e-12,
        "`<default class><general ctrlrange='-2 0.5'/>` must give"
        " ctrllimited=true and [-2, 0.5] — got limited "
        + String(lim[0] != 0.0) + " [" + String(lo[0]) + ", "
        + String(hi[0]) + "]. The range was always parsed; it is the FLAG"
        " that was missing, and `apply_actions_fields` clamps only when the"
        " flag is set.",
    )
    # The element path, which already worked — kept so a regression that
    # broke it while fixing the class path cannot pass.
    assert_true(
        lim[1] != 0.0 and abs(lo[1] - (-3.0)) < 1e-12
        and abs(hi[1] - 4.0) < 1e-12,
        "an element `ctrlrange='-3 4'` must give ctrllimited=true and [-3, 4]",
    )
    # Precedence: element over class.
    assert_true(
        lim[2] != 0.0 and abs(lo[2] - (-3.0)) < 1e-12
        and abs(hi[2] - 4.0) < 1e-12,
        "an element `ctrlrange` must OVERRIDE the class's, giving [-3, 4];"
        " got [" + String(lo[2]) + ", " + String(hi[2]) + "]",
    )
    # ⚠ NEGATIVE CONTROL 1. `"0 0"` is MuJoCo's undefined marker, and this
    # rule already existed on the element path — the class path must not
    # acquire the flag without it.
    assert_true(
        lim[3] == 0.0,
        "`ctrlrange='0 0'` in a class is MuJoCo's UNDEFINED marker, not a"
        " zero-width range — it must stay unlimited, or the actuator is"
        " clamped to [0, 0] and delivers ZERO FORCE",
    )
    # ⚠ NEGATIVE CONTROL 2. An explicit `ctrllimited="false"` beside a range.
    assert_true(
        lim[4] == 0.0,
        "an explicit `ctrllimited='false'` in the class must override the"
        " range's implied limit",
    )
    # ⚠ NEGATIVE CONTROL 3, AND THE ONE THAT MATTERS MOST. Without it this
    # file would pass against an implementation that marked EVERY actuator
    # limited — which is precisely the failure the `"0 0"` fix was undoing.
    assert_true(
        lim[5] == 0.0,
        "an actuator with no range anywhere must stay UNLIMITED; MuJoCo"
        " leaves it unclamped and squeezing it into the fallback range is the"
        " opposite defect",
    )
    print("  PASS")


def test_barkour_ctrllimited_matches_mujoco() raises:
    """The real model, against MuJoCo's own numbers.

    ⚠ MEASURED ON THE 3.10.0 RUNTIME: barkour reports
    `actuator_ctrllimited` true on all twelve, and the knee actuators'
    `ctrlrange` is [0.1, 2.34346]. Every one of those ranges is stated in a
    `<default>` class, which is why all twelve were unlimited here.
    """
    print("=== google_barkour_vb ctrllimited ===")
    var src = read_model_source(BARKOUR)
    var _r = _build(src[0], src[1])
    var lim = _r[0].copy()
    var lo = _r[1].copy()
    var hi = _r[2].copy()
    var n_lim = 0
    for i in range(len(lim)):
        if lim[i] != 0.0:
            n_lim += 1
    print("  ctrllimited on", n_lim, "of", len(lim))
    print("  knee ctrlrange [", lo[2], ",", hi[2],
          "]  (MuJoCo: [0.1, 2.34346])")
    assert_true(
        len(lim) == 12 and n_lim == 12,
        "MuJoCo reports actuator_ctrllimited true on all 12 of barkour's"
        " actuators; we have " + String(n_lim) + " of " + String(len(lim)),
    )
    assert_true(
        abs(lo[2] - 0.1) < 1e-9 and abs(hi[2] - 2.34346) < 1e-9,
        "barkour's knee ctrlrange must be MuJoCo's [0.1, 2.34346], got ["
        + String(lo[2]) + ", " + String(hi[2]) + "]",
    )
    print("  PASS")


def test_barkour_knee_torque_matches_mujoco() raises:
    """The FORCE, which is the only reason the flag matters.

    ⚠⚠ THIS IS THE ROW THE RECORD GATES ABOVE CANNOT CARRY. `ctrl_min` and
    `ctrl_max` were correct before the fix; a gate that reads them back
    passes either way. What changed is whether `apply_actions_fields` clamps
    against them, and the only way to see that is to ask for the force.

    ⚠ EXPECTED VALUES ARE MUJOCO'S OWN `qfrc_actuator` at `qpos0` with
    `ctrl = 0`, read off the 3.10.0 runtime: zero on the six free-joint dofs
    and on every abduction and hip dof, and 5 N.m on each of the four knees —
    `kp * (clamp(0, 0.1, 2.34346) - 0)` = `50 * 0.1`.
    """
    print("=== barkour qfrc_actuator at qpos0, ctrl = 0 ===")
    var src = read_model_source(BARKOUR)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=verts)
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
            dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
        d.qfrc.data[i] = Scalar[DT](0)

    var nact = dims.get_nact()
    var actions = List[Float64](length=nact, fill=0.0)
    var act = List[Scalar[DT]](length=nact, fill=Scalar[DT](0))
    apply_actions_fields[DT](sf, d, actions, act, fmd.timestep)

    var line = String("")
    for i in range(dims.get_nv()):
        line += String(Float64(d.qfrc.data[i])) + " "
    print("  qfrc_actuator", line)

    # Knee dofs are 8, 11, 14, 17 — the third of each leg's three.
    var knees: List[Int] = [8, 11, 14, 17]
    for k in range(len(knees)):
        var got = Float64(d.qfrc.data[knees[k]])
        assert_true(
            abs(got - 5.0) < 1e-12,
            "knee dof " + String(knees[k]) + " must carry MuJoCo's 5 N.m"
            " (kp 50 times the clamped ctrl 0.1); got " + String(got)
            + ". A zero here means `ctrl` reached the force law UNCLAMPED —"
            " the actuator is not marked ctrllimited.",
        )
    # ⚠ THE NEGATIVE CONTROL. Every OTHER dof must stay at zero: `ctrl` is 0
    # and `qpos0` is 0, so abduction and hip are inside their ranges and a
    # clamp that fired on them would be a different bug entirely.
    for i in range(dims.get_nv()):
        var is_knee = i == 8 or i == 11 or i == 14 or i == 17
        if not is_knee:
            assert_true(
                abs(Float64(d.qfrc.data[i])) < 1e-12,
                "dof " + String(i) + " must carry no actuator force —"
                " MuJoCo's qfrc_actuator is 0 there; got "
                + String(Float64(d.qfrc.data[i])),
            )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
