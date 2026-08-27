"""Three defects in one chain, all on `tetheria_aero_hand_open`.

    pixi run mojo run -I . tests/physics3d/test_spatial_tendon_transmission_vs_mujoco.mojo

That scene was the worst in the Menagerie sweep at 2.590e-01, and it took
three independent fixes to bring it to 5.551e-16 — each one hidden behind the
one before it, which is why they arrive together.

1. THE ACTUATOR TRANSMISSION APPLIED NO FORCE. `_fill_actuator_transmission`
   resolves a `tendon=` actuator by copying the tendon's `(joint, coef)` list.
   Only a **FIXED** tendon has one; a SPATIAL tendon is a polyline through
   sites, so `num_joints` is 0, the walk wrote nothing, and the actuator came
   out with `trn_n = 0` — a slot in `nact`, a consumed control, and zero
   force. Six of tetheria's seven actuators are
   `<position tendon="if_tendon0" kp="10000"/>`, so the hand had no tendons
   pulling it at all. Its length and moment arm depend on the POSE, so they
   cannot live in those triples: `dynamics/pose_transmission.mojo` evaluates
   them after forward kinematics instead. That took `|d(qfrc_actuator)|` from
   4.14e+00 to 1.15e-14 and the scene from 2.590e-01 to 1.002e-01.

2. `<option><flag eulerdamp="disable"/></option>` WAS NOT PARSED. With that
   flag `mj_EulerSkip` integrates velocity EXPLICITLY (`qvel += h*qacc`);
   without it, it solves `(M + h*diag(B)) qacc' = M qacc` first. We always
   solved. On this model `h*B/M` is 0.625 — `h = 0.01`, `dof_damping = 0.1`,
   `M_ii ~ 1.6e-03` — so every velocity came out at 61.5% of MuJoCo's.
   `_option_flag_disabled` had existed since the `multiccd` work and was
   simply never asked about this name. 1.002e-01 -> 8.022e-03.

   ⚠ THE ARITHMETIC IS WHAT IDENTIFIED IT. MuJoCo's `qvel` after one step was
   EXACTLY `h * qacc` to the last digit on a model with `dof_damping = 0.1`,
   which is only possible if the damping solve did not run.

3. `<default><tendon .../></default>` WAS NOT RESOLVED. Every tendon
   attribute was read off the element's own opening tag, and this model keeps
   `stiffness` and `springlength` in `<default class="distal_spring">`. So its
   eight spring tendons had stiffness 0 and pulled on nothing. The `<default>`
   chain is the single largest source of defects in this parser — geom `type`,
   geom `material`, actuator tags, joint ranges have all been here — and the
   cure is the same each time: come through `NamedDefaultsList`, which already
   resolves a class against its parents, not through the element tag.
   8.022e-03 -> 5.551e-16.

⚠ THE ORDER MATTERS FOR ANYONE BISECTING THIS. Defect 2 is invisible while
defect 1 is live (a hand with no actuator force barely moves, so 61.5% of
nothing is nothing) and defect 3 is invisible while 2 is live (the spring
moments are ~0.16 against actuator moments of ~4). Each fix is what made the
next measurable — which is also why the first two "did not fix it" and were
right anyway.

MEASURED. Menagerie step-1 sweep: 67 -> 68 of 85 scenes at or below 1e-9, and
4 -> 3 above 1e-3. tetheria is the only scene that moves; every other
tendon-carrying scene is unchanged to the digit.
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
from mojo_rl.physics3d.fields.dynamics_scratch import DynamicsScratch
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.dynamics.pose_transmission import (
    apply_pose_transmission,
)
from mojo_rl.physics3d.studio.stepping import StudioIntegPyr
from mojo_rl.physics3d.gpu.constants import (
    KEY_META_SIZE, KEY_IDX_NQPOS,
)

comptime DT = DType.float64

comptime TETHERIA = String(
    "references/mujoco_menagerie-main/tetheria_aero_hand_open/scene_right.xml"
)

# ── fixture for defect 2: one damped hinge, the flag on and off ───────────
# `h*B` is 1e-03 against an `M_ii` of about 2.4e-03, so the two answers are
# 18% apart — far outside any tolerance and far inside the range where both
# integrators are perfectly stable.
comptime _ED_BODY = String(
    """
  <worldbody>
    <body pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.1"
             armature="0.0015591"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
    </body>
  </worldbody>
"""
)
comptime XML_ED_ON = String(
    '<mujoco><option timestep="0.01" integrator="Euler"/>'
    + _ED_BODY + "</mujoco>"
)
comptime XML_ED_OFF = String(
    '<mujoco><option timestep="0.01" integrator="Euler">'
    + '<flag eulerdamp="disable"/></option>'
    + _ED_BODY + "</mujoco>"
)
# MuJoCo 3.10.0, one `mj_step` from qpos0.
comptime MJ_ED_ON_QVEL = 0.4194890950287431
comptime MJ_ED_OFF_QVEL = 0.4935893950186796

# ── fixture for defects 1 and 3: one spatial tendon, driven and sprung ────
# `spr` takes BOTH its stiffness and its rest length from a `<default>`
# class — tetheria's own numbers — so a parser that reads only the element
# tag gives it stiffness 0. `pull` is driven by a `<position tendon=>`,
# which has no `(joint, coef)` list to walk.
comptime XML_SPATIAL = String(
    """<mujoco>
  <option timestep="0.002"/>
  <default>
    <default class="spring_cls">
      <tendon stiffness="4000" springlength="0.021336"/>
    </default>
  </default>
  <worldbody>
    <site name="anchor" pos="0 0 0.4" size="0.002"/>
    <body pos="0 0 0.3">
      <joint name="h" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.01"/>
      <site name="tip" pos="0.15 0 0" size="0.002"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="pull">
      <site site="tip"/>
      <site site="anchor"/>
    </spatial>
    <spatial name="spr" class="spring_cls">
      <site site="tip"/>
      <site site="anchor"/>
    </spatial>
  </tendon>
  <actuator>
    <position name="a" tendon="pull" kp="500" ctrlrange="-1 1"/>
  </actuator>
</mujoco>"""
)
# MuJoCo 3.10.0 at qpos0 with ctrl = 0.05. The tendon is 0.18027756 long.
comptime MJ_SP_QFRC_ACT = -5.4198742570881727
comptime MJ_SP_QFRC_PAS = -52.898949968461738
comptime MJ_SP_TEN_LENGTH = 0.18027756377319946

# ── tetheria, 20 steps at its own keyframe ctrl ───────────────────────────
# ⚠ FUNCTIONS, NOT `comptime` LISTS. A `comptime List[Float64]` cannot be
# materialized to runtime (it is not `ImplicitlyCopyable`), so the reference
# vectors are built where they are used.
def _mj_teth() -> List[Float64]:
    """MuJoCo 3.10.0 `qpos` after ONE `mj_step` from keyframe 0 at
    `_teth_ctrl()` — the keyframe's own `ctrl`."""
    return [
        1.5003428698870094, 0.30491341042682246, 0.30821461505582148,
        1.5000726265991515, 0.35645111806611302, 0.36001106837584002,
        1.4980817263092769, 0.32129538235825594, 0.324719586625624,
        1.4950606854955284, 0.46846693757571339, 0.47197345050719602,
        0.74635111430292334, 0.25978929319290806, 0.12269943168364987,
        0.93075122016379308,
    ]
def _teth_ctrl() -> List[Float64]:
    return [0.09, 0.09, 0.09, 0.09, 0.75, 0.035, 0.1]


def _qfrc_after_actuation(
    xml: String, ctrl: List[Float64]
) raises -> List[Float64]:
    """`d.qfrc` after BOTH actuation passes, at qpos0.

    This is the sum MuJoCo splits across `qfrc_actuator` and the tendon-spring
    half of `qfrc_passive`; the fixture has no joint damping or stiffness, so
    nothing else lands in either.
    """
    var fmd = parse_xml_full(expand_mjcf(xml, String("")), String(""))
    var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=0)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
    var act = List[Scalar[DT]](
        length=dims.get_nact() if dims.get_nact() > 0 else 1,
        fill=Scalar[DT](0),
    )
    var sc = DynamicsScratch[DT, DynDims, 1](dims)
    apply_actions_fields[DT](sf, d, ctrl, act, fmd.timestep)
    apply_pose_transmission[DT](sf, m, d, sc, ctrl, act, fmd.timestep)
    var out = List[Float64]()
    for i in range(dims.get_nv()):
        out.append(Float64(d.qfrc.data[i]))
    return out^


def _qvel_after_one_step(xml: String) raises -> List[Float64]:
    """One Euler step from qpos0, no actuation."""
    var fmd = parse_xml_full(expand_mjcf(xml, String("")), String(""))
    var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=0)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
        d.qfrc.data[i] = Scalar[DT](0)
    var integ = StudioIntegPyr(dims)
    integ.step["cpu"](d, m)
    var out = List[Float64]()
    for i in range(dims.get_nv()):
        out.append(Float64(d.qvel.data[i]))
    return out^


def test_eulerdamp_flag_is_honoured() raises:
    """`<flag eulerdamp="disable"/>` must switch the damping solve OFF."""
    print("=== <option><flag eulerdamp> ===")
    var on = _qvel_after_one_step(XML_ED_ON)
    var off = _qvel_after_one_step(XML_ED_OFF)
    assert_true(
        len(on) == 1 and len(off) == 1,
        "the fixture is one hinge — got " + String(len(on)),
    )
    print("  flag absent  ours", on[0], " mj", MJ_ED_ON_QVEL)
    print("  flag disable ours", off[0], " mj", MJ_ED_OFF_QVEL)
    # ⚠ VACUITY, AND IT IS THE WHOLE POINT. If the flag were still ignored
    # both rows would read the SAME number and both comparisons could not
    # both pass. Assert the reference pair differs first, so a future
    # fixture that stops discriminating fails here rather than passing
    # everything.
    assert_true(
        abs(MJ_ED_ON_QVEL - MJ_ED_OFF_QVEL) > 0.05,
        "the two reference velocities must differ or this gate is vacuous",
    )
    assert_true(
        abs(on[0] - MJ_ED_ON_QVEL) < 1e-12,
        "WITHOUT the flag the implicit damping solve must run; ours "
        + String(on[0]) + " vs MuJoCo " + String(MJ_ED_ON_QVEL),
    )
    assert_true(
        abs(off[0] - MJ_ED_OFF_QVEL) < 1e-12,
        "WITH `eulerdamp=disable` MuJoCo integrates velocity explicitly"
        " (`qvel += h*qacc`); ours " + String(off[0]) + " vs MuJoCo "
        + String(MJ_ED_OFF_QVEL) + ". Reading the flag's own value here"
        " means checking `_option_flag_disabled(xml, \"eulerdamp\")` reaches"
        " `MODEL_META_IDX_EULERDAMP_DISABLED` and that `_finalize_env` takes"
        " its explicit branch.",
    )
    print("  PASS")


def test_spatial_tendon_actuator_and_spring_pull() raises:
    """A `<position tendon=>` on a SPATIAL tendon must apply force, and a
    `<spatial>` spring must take its parameters from its `<default>` class."""
    print("=== spatial tendon: transmission + default class ===")
    var ctrl: List[Float64] = [0.05]
    var got = _qfrc_after_actuation(XML_SPATIAL, ctrl)
    assert_true(
        len(got) == 1,
        "the fixture is one hinge — got " + String(len(got)),
    )
    var want = MJ_SP_QFRC_ACT + MJ_SP_QFRC_PAS
    print("  ours qfrc     ", got[0])
    print("  mj  actuator  ", MJ_SP_QFRC_ACT)
    print("  mj  passive   ", MJ_SP_QFRC_PAS, " (the class-defaulted spring)")
    print("  mj  sum       ", want)
    # ⚠ THE TWO HALVES ARE BOTH LARGE AND OPPOSITE IN ORIGIN. Dropping the
    # actuator leaves -52.90; dropping the spring leaves -5.42; dropping both
    # leaves 0. No two of those are within 1e-9 of the sum, so this single
    # comparison cannot pass with either half missing.
    assert_true(
        abs(MJ_SP_QFRC_ACT) > 1.0 and abs(MJ_SP_QFRC_PAS) > 1.0,
        "both halves must be large or the gate cannot separate them",
    )
    # ⚠ RELATIVE, BECAUSE THE TWO HALVES ARE HUNDREDS OF NEWTONS. The
    # spring is 4000 N/m stretched 0.159 m past its rest length, so the
    # scalar force is -635.75 and the moment arm 0.0832; a 1e-10 relative
    # difference in a moment arm computed through `cdof` rather than through
    # MuJoCo's own site Jacobian lands at 6.6e-09 absolute. The failure this
    # gate is for is four orders larger — a MISSING half.
    var rel = abs(got[0] - want) / abs(want)
    print("  relative", rel)
    assert_true(
        rel < 1e-9,
        "ours " + String(got[0]) + " vs MuJoCo " + String(want)
        + ". 0 means neither half ran (the actuator resolved to `trn_n = 0`"
        " and the spring to `stiffness = 0`); -52.90 means the actuator is"
        " missing; -5.42 means the `<default class>` did not reach the"
        " tendon.",
    )
    print("  PASS")


def test_tetheria_matches_mujoco() raises:
    """The scene all three defects were found on, one step.

    ⚠ ONE STEP, AND NOT TWENTY, FOR A REASON WORTH READING. At step 1 this
    model matches MuJoCo to 4.4e-16 — which is what the Menagerie sweep
    measures and what these three fixes buy. At step 20 it is 1.5e-03, and
    that residual is NOT any of them and NOT a physics term at all: it is
    `<option iterations="5" ls_iterations="8">`, which nothing in this engine
    reads. We run a hardcoded `NEWTON_ITER_GPU = 200` / `LINESEARCH_ITER = 50`
    and therefore converge FURTHER than the reference does.

    The proof is one line on MuJoCo's side, with nothing rebuilt here:

        m.opt.iterations, m.opt.ls_iterations = 200, 50
        -> |ours - mj| = 1.887e-15   (5, 8 gives 1.545e-03)

    ⚠ AND THE FALSE TRAIL IS WORTH RECORDING. Rewriting `frictionloss="0.02"`
    to `"0"` in a copy of the model gives 3.3e-15 at step 40, which reads as
    "the dry-friction rows are wrong". They are not — their `R` and `aref`
    match `d.efc_R` / `d.efc_aref` exactly, and their state test is MuJoCo's
    (`engine_solver.c`: `-Rf < x < Rf` quadratic, else linear). Friction is
    simply what an UNDER-converged solve is sensitive to, so removing it
    removes the symptom. The give-away that it was not a friction magnitude
    error: scaling `frictionloss` by 10 either way (0.002 / 0.2) made step 2
    EXACT again and moved the onset to step 20 — a proportional error would
    have scaled, not moved.

    ⚠⚠ AND HONOURING THE BUDGET IS **NOT** THE FIX — MEASURED, DO NOT REDO IT.
    Parsing all four attributes and clamping both loops to them was written
    and thrown away: the Menagerie sweep went from 68 to 65 scenes at or below
    1e-9, and tetheria's own step 1 went from 4.4e-16 to 1.9e-04. The reason
    is that our iterate at 5 is not MuJoCo's iterate at 5. MuJoCo CONVERGES
    within its 5 (its step-1 answer at 5/8 equals its answer at 200/50 to
    4.4e-16, which is also ours); we do not. So truncating at the file's count
    lands us on an unconverged point that MuJoCo never visits, while running
    to convergence lands us on the point MuJoCo reaches.

    That makes the real defect our CONVERGENCE RATE, not the budget: our
    Newton needs more iterations than MuJoCo's to reach the same tolerance,
    and the 1.5e-03 is the steps where MuJoCo runs out of budget first and
    stops somewhere we do not. Warm-starting is the obvious suspect — MuJoCo
    seeds `qacc` from `d->qacc_warmstart` (`mjDSBL_WARMSTART` is off by
    default) and we start from `qacc_constrained` every step — but that is a
    solver investigation, not a parser one. Nine of 88 scenes set a
    non-default budget; our own default (200) is not MuJoCo's (100) either,
    and neither number matters until the rate does.
    """
    print("=== tetheria_aero_hand_open/scene_right, one step ===")
    var MJ_TETH = _mj_teth()
    var TETH_CTRL = _teth_ctrl()
    var src = read_model_source(TETHERIA)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=131072)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var d = Data[DT, DynDims, 1](dims)
    # From keyframe 0 — the pose the sweep uses, and the one whose spring
    # tendons are stretched. At `qpos0` they sit at their rest length and
    # defect 3 is invisible.
    assert_true(
        dims.get_nkey() > 0,
        "this scene must carry a keyframe — the gate measures from it",
    )
    var nqp = Int(Float64(sf.key_meta.data[KEY_IDX_NQPOS]))
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(min(nqp, nq)):
        d.qpos.data[i] = sf.key_qpos.data[i]
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)
    var act = List[Scalar[DT]](length=dims.get_nact(), fill=Scalar[DT](0))
    var sc = DynamicsScratch[DT, DynDims, 1](dims)
    var integ = StudioIntegPyr(dims)
    for _ in range(1):
        for i in range(nv):
            d.qfrc.data[i] = Scalar[DT](0)
        apply_actions_fields[DT](sf, d, TETH_CTRL, act, fmd.timestep)
        apply_pose_transmission[DT](
            sf, m, d, sc, TETH_CTRL, act, fmd.timestep
        )
        integ.step["cpu"](d, m)

    assert_true(
        len(MJ_TETH) == nq,
        "the expected vector must cover every qpos slot — nq is "
        + String(nq),
    )
    var worst = 0.0
    var wi = 0
    for i in range(nq):
        var e = abs(Float64(d.qpos.data[i]) - MJ_TETH[i])
        if e > worst:
            worst = e
            wi = i
    print("  worst |d(qpos)| =", worst, " at dof", wi)
    print("  ours", Float64(d.qpos.data[wi]), " mj", MJ_TETH[wi])
    # ⚠ VACUITY. The hand must have CLOSED — every finger's first joint
    # travels ~0.3 rad off the keyframe under this control. A model whose
    # actuators do nothing sits within 1e-3 of where it started.
    var moved = abs(Float64(d.qpos.data[0]) - Float64(sf.key_qpos.data[0]))
    print("  dof 0 moved", moved, "rad off the keyframe")
    assert_true(
        moved > 0.05,
        "the hand did not close — the gate would be comparing a pose"
        " neither engine integrated. One step at this control takes dof 0"
        " from 1.57335 to 1.50034, i.e. 0.073 rad; it moved "
        + String(moved) + ".",
    )
    assert_true(
        worst < 1e-12,
        "tetheria must match MuJoCo; worst |d(qpos)| = " + String(worst)
        + ". Before these three fixes the step-1 figure was 2.590e-01.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
