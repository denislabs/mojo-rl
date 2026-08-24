"""The studio steps RK4 when the file says RK4 — and Euler is measurably wrong.

WHY THIS EXISTS
===============
`test_studio_honours_option_cone` gates the CONE axis of the studio's
dispatch. The INTEGRATOR axis had a third value nobody built: the studio
carried `EulerIntegrator` and `ImplicitIntegrator` and selected between them
with a BOOLEAN (`studio_uses_implicit`), so `RK4` — which is not implicit —
fell out of the `else` and was stepped with Euler.

⚠⚠ AND IT WAS WARNED ABOUT, WHICH IS WHY IT SURVIVED. `studio_integrator_warning`
printed "the studio builds Euler and implicitfast only and is stepping with
EULER. Expect a less accurate trajectory" — a true sentence that made the
substitution look accounted for. It was not: the fidelity harness that mirrors
the studio's path drove `bitcraze_crazyflie_2` through the same `else`, and
the resulting 9.200e-06 sat at #5 on the Menagerie board attributed to a
defect in the SITE transmission. A warning is not a substitute for stepping
the file.

⚠ THE SIGNATURE THAT NAMED IT, and the reason the arms below compare against
MuJoCo rather than against each other. In free flight the acceleration is
nearly constant; semi-implicit Euler then moves `a*dt^2` and RK4 moves half of
that, so ours came out EXACTLY 2x the reference in EVERY dof — a ratio, not a
drift. Two integrators that merely "differ" would satisfy a self-comparison.

THREE ARMS:

  1. the SELECTION — `studio_integrator_of` against MuJoCo's `m.opt.integrator`
     on one real model of each of the three values, so the arm is not a
     constant.

  2. ⚠⚠ NOT INERT, ON A MODEL WITH NO CONTACTS. crazyflie, one step from
     qpos0. RK4 must reach MuJoCo; EULER MUST NOT. The second half is what
     keeps this from passing on a build where both aliases are the same
     integrator.

  3. ⚠ AND ON A MODEL THAT TOUCHES THE GROUND. `StudioRk4Pyr` solves contacts
     INSIDE its stage loop (four detect+solve passes per step), which arm 2
     never exercises — crazyflie flies. hopper is RK4, pyramidal, and has two
     live contacts by step 60.

Measured here (MuJoCo 3.10.0, both engines driven by the same control):

    crazyflie, 1 step    EULER 1.1015e-06     RK4 5.1546e-13
    hopper,   60 steps   EULER 3.8885e-03     RK4 1.2490e-16

Run: pixi run mojo run -I . tests/physics3d/test_studio_honours_option_rk4.mojo
"""

from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
    spec_fields_runtime, read_model_source,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.fields.dynamics_scratch import DynamicsScratch
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.dynamics.pose_transmission import (
    apply_pose_transmission,
)
from mojo_rl.physics3d.types import ConeType, IntegratorType
from mojo_rl.physics3d.studio.stepping import (
    StudioIntegPyr, StudioRk4Pyr, studio_integrator_of, studio_uses_rk4,
    studio_cone_of, studio_integrator_warning, STUDIO_DT,
)


comptime DT = STUDIO_DT

# ⚠ THE GOLDENS ARE MuJoCo'S, read with
#     m = mujoco.MjModel.from_xml_path(P); m.opt.integrator
# (0 EULER, 1 RK4, 2 IMPLICIT, 3 IMPLICITFAST). One real model per value.
comptime RK4_MODEL = String(
    "references/mujoco_menagerie-main/bitcraze_crazyflie_2/scene.xml"
)
comptime IMPFAST_MODEL = String(
    "references/mujoco_menagerie-main/hello_robot_stretch/scene.xml"
)
comptime EULER_MODEL = String(
    "references/mujoco_menagerie-main/agility_cassie/scene.xml"
)
# The contact arm: RK4 + pyramidal + a foot on the floor.
comptime RK4_CONTACT_MODEL = String("mojo_rl/envs/hopper/assets/hopper.xml")


def _mj_crazyflie_1() -> List[Float64]:
    """MuJoCo `qpos` after 1 step from qpos0 at ctrl (.25, -.2, .15, -.1)."""
    return [
        -3.86591754744593208e-13,
        -5.15455689591950211e-13,
        +9.99988985189356749e-02,
        +9.99999999999994116e-01,
        +8.35037627611809449e-08,
        -6.26278211319954297e-08,
        +3.09147520910410893e-08,
    ]


def _mj_hopper_60() -> List[Float64]:
    """MuJoCo `qpos` after 60 steps from qpos0 at ctrl (.3, -.2, .1)."""
    return [
        -1.97659083869505536e-02,
        +1.19266275606377170e+00,
        -1.11804551264358870e-01,
        +1.16316582075055006e-03,
        -2.42184637210970582e-01,
        +1.36890231673449742e-01,
    ]


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def eq(mut self, got: Int, want: Int, msg: String):
        self.checks += 1
        if got == want:
            print("  ok:", msg, "=", got)
        else:
            self.fails += 1
            print("  FAIL:", msg, "got", got, "want", want)

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def _drive[ARM: StaticString](
    path: String, ctrl: List[Float64], nstep: Int
) raises -> List[Float64]:
    """`qpos` after `nstep` steps from qpos0, driven by a constant `ctrl`.

    ⚠ THE STUDIO'S OWN ALIASES, not a locally spelled pair — a test that
    instantiated its own `RK4Integrator` would pass while the studio's branch
    was wired to Euler, which is the exact failure this gate exists for.
    """
    var src = read_model_source(path)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=65536)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var d = Data[DT, DynDims, 1](dims)
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)
    var act = List[Scalar[DT]](
        length=dims.get_nact() if dims.get_nact() > 0 else 1,
        fill=Scalar[DT](0),
    )
    var sc = DynamicsScratch[DT, DynDims, 1](dims)

    # ⚠ THE ACTUATION IS THE HARNESS'S, ONCE PER STEP. `apply_pose_transmission`
    # is deliberately outside the integrator (see its header): under Euler that
    # is exact, under RK4 it freezes a pose-dependent moment at stage 0. That
    # residual is 3.3e-13 on crazyflie — four orders below the Euler
    # substitution this gate measures, and NOT what these tolerances test.
    comptime if ARM == "rk4":
        var integ = StudioRk4Pyr(dims)
        for _ in range(nstep):
            for i in range(nv):
                d.qfrc.data[i] = Scalar[DT](0)
            apply_actions_fields[DT](sf, d, ctrl, act, fmd.timestep)
            apply_pose_transmission[DT](sf, m, d, sc, ctrl, act, fmd.timestep)
            integ.step["cpu"](d, m)
    else:
        var integ = StudioIntegPyr(dims)
        for _ in range(nstep):
            for i in range(nv):
                d.qfrc.data[i] = Scalar[DT](0)
            apply_actions_fields[DT](sf, d, ctrl, act, fmd.timestep)
            apply_pose_transmission[DT](sf, m, d, sc, ctrl, act, fmd.timestep)
            integ.step["cpu"](d, m)

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


def main() raises:
    var t = Tally()
    print("=== the studio steps the integrator the model declares ===")

    # ── 1. the selection ─────────────────────────────────────────────────
    print("--- selection, against MuJoCo's m.opt.integrator ---")
    var f_rk4 = parse_model_runtime(RK4_MODEL)
    var f_imp = parse_model_runtime(IMPFAST_MODEL)
    var f_eul = parse_model_runtime(EULER_MODEL)
    t.eq(studio_integrator_of(f_rk4), IntegratorType.RK4,
         "bitcraze_crazyflie_2 (<option integrator='RK4'>)")
    t.eq(studio_integrator_of(f_imp), IntegratorType.IMPLICITFAST,
         "hello_robot_stretch (<option integrator='implicitfast'>)")
    t.eq(studio_integrator_of(f_eul), IntegratorType.EULER,
         "agility_cassie (says nothing -> MuJoCo's default)")
    t.truth(studio_uses_rk4(f_rk4) and not studio_uses_rk4(f_eul),
            "`studio_uses_rk4` agrees with the selector it wraps")

    # ⚠ NON-VACUITY: three models that all read the same would make every
    # line above a constant.
    t.truth(
        studio_integrator_of(f_rk4) != studio_integrator_of(f_imp)
        and studio_integrator_of(f_imp) != studio_integrator_of(f_eul)
        and studio_integrator_of(f_rk4) != studio_integrator_of(f_eul),
        "the three models actually select THREE different integrators",
    )

    # An RK4 model we step as declared draws no warning; the pyramidal-cone
    # substitution is the only thing left on this axis to name.
    t.eq(studio_cone_of(f_rk4), ConeType.PYRAMIDAL,
         "the RK4 model is pyramidal (the only cone this arm builds)")
    t.truth(studio_integrator_warning(f_rk4).byte_length() == 0,
            "an RK4 + pyramidal model draws no warning any more")

    # ── 2. not inert, contact-free ───────────────────────────────────────
    print("--- crazyflie, 1 step from qpos0, vs MuJoCo 3.10.0 ---")
    var cf_ctrl: List[Float64] = [0.25, -0.2, 0.15, -0.1]
    var want_cf = _mj_crazyflie_1()
    var cf_rk4 = _drive["rk4"](RK4_MODEL, cf_ctrl, 1)
    var cf_eul = _drive["euler"](RK4_MODEL, cf_ctrl, 1)
    var e_cf_rk4 = _worst(cf_rk4, want_cf)
    var e_cf_eul = _worst(cf_eul, want_cf)
    print("    RK4   worst |d qpos| =", e_cf_rk4)
    print("    EULER worst |d qpos| =", e_cf_eul)
    t.truth(e_cf_rk4 < 1e-11, "RK4 reaches MuJoCo (< 1e-11)")
    t.truth(e_cf_eul > 1e-7,
            "EULER does NOT (> 1e-7) — the two aliases are different"
            " integrators, and the selection is what decides the answer")

    # ── 3. and with contacts inside the stage loop ───────────────────────
    print("--- hopper, 60 steps from qpos0 (2 live contacts), vs MuJoCo ---")
    var hp_ctrl: List[Float64] = [0.3, -0.2, 0.1]
    var want_hp = _mj_hopper_60()
    var hp_rk4 = _drive["rk4"](RK4_CONTACT_MODEL, hp_ctrl, 60)
    var hp_eul = _drive["euler"](RK4_CONTACT_MODEL, hp_ctrl, 60)
    var e_hp_rk4 = _worst(hp_rk4, want_hp)
    var e_hp_eul = _worst(hp_eul, want_hp)
    print("    RK4   worst |d qpos| =", e_hp_rk4)
    print("    EULER worst |d qpos| =", e_hp_eul)
    t.truth(e_hp_rk4 < 1e-12,
            "RK4 reaches MuJoCo through 60 steps and two contacts (< 1e-12)")
    t.truth(e_hp_eul > 1e-4, "EULER does NOT (> 1e-4)")

    # ⚠ AND BOTH MUST BE FINITE. "They differ" is also true when one of them
    # diverged, which would make the non-inertness arms pass on a broken build.
    var finite = True
    for i in range(len(hp_rk4)):
        var a = hp_rk4[i]
        var b = hp_eul[i]
        if not (a == a) or not (b == b) or abs(a) > 1e12 or abs(b) > 1e12:
            finite = False
    t.truth(finite, "both integrators stayed finite (control)")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error(
            "test_studio_honours_option_rk4: " + String(t.fails) + " failed"
        )
