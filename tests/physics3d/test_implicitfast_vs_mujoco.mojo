"""`<option integrator="implicitfast">` — the servo damping that belongs in M.

    pixi run mojo run -I . tests/physics3d/test_implicitfast_vs_mujoco.mojo

WHAT WAS MISSING, IN TWO PARTS.

  1. `integrator` was parsed but nothing dispatched on it, so the studio
     stepped every model with explicit Euler — including the ones that ask
     for `implicitfast`, which is most of Menagerie's legged robots.
  2. Our implicit integrators were implicit in the passive damping only.
     MuJoCo's `mjd_smooth_vel` is actuator + passive + (optional) RNE
     (`engine_derivative.c:1985`); `mjd_actuator_vel` — the `-kv` of a
     `<position>`/`<velocity>` servo, entering as `J^T diag(kv) J` — had no
     port at all.

⚠⚠ PART 2 IS WHAT MAKES PART 1 WORTH ANYTHING, and spot is the proof: its
`dof_damping` is 0 and its `dof_armature` is 0, so its ONLY damping anywhere
is the actuators' `kv=40`. With `SKIP_RNE_DERIV` set and `dof_actdamp` zero,
qDeriv is identically zero and `M_hat = M` — the implicit step reduces
exactly to Euler's. So switching spot to "implicit" without the actuator term
would have changed nothing at all, while looking like a fix.

MEASURED AT SPOT'S FIRST STEP. Both engines agree on everything upstream —
`qfrc_actuator` 127.824 and `qacc` 18300.23 to every printed digit — so the
integrator is the single variable:

    MuJoCo implicitfast : |qvel|max  2.8516
    MuJoCo Euler        : |qvel|max 36.6005
    ours, Euler         : |qvel|max 36.600467      <- our Euler is CORRECT
    ours, implicitfast  : |qvel|max  2.851622      <- and now so is this

⚠ SO THE EULER ROW IS NOT A LEFTOVER. It is the negative control: it proves
the fixture moves for the reason claimed, and it pins Euler against a
"fix" that quietly made every integrator implicit. MuJoCo-Euler is unstable
here too (|qvel|max 136.5 after 1500 steps) — Euler is not being blamed for
being wrong, it is being blamed for being asked.

⚠ WHY IT LOOKS LIKE A CONTACT BUG AND IS NOT. The reported symptom is "spot
bounces and flies as soon as it touches the ground". At step 0, before any
contact exists (`ncon 0`), the robot already carries 36.6 rad/s: the servos
are fighting the knee limits — `inheritrange` clamps a commanded 0 to -0.254
— and explicit Euler amplifies that instead of damping it. The floor is where
the energy becomes visible, not where it comes from.
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.studio.stepping import (
    StudioIntegEll, StudioImpFastEll, studio_cone_of, studio_uses_implicit,
)
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.types import ConeType, IntegratorType

comptime DT = DType.float64
comptime SPOT = String(
    "references/mujoco_menagerie-main/boston_dynamics_spot/scene.xml"
)


struct Run(Copyable, Movable):
    var qvel0: Float64
    """|qvel|max after the FIRST step — where the integrators separate."""
    var zmax: Float64
    var zfinal: Float64

    def __init__(out self, qvel0: Float64, zmax: Float64, zfinal: Float64):
        self.qvel0 = qvel0
        self.zmax = zmax
        self.zfinal = zfinal


def _drop[IMPLICIT: Bool](nstep: Int) raises -> Run:
    """Spot, dropped, on the runtime path, driven by nothing.

    ⚠ ZERO ACTIONS AND IT STILL MATTERS. `ctrl = 0` is not "no actuator": the
    knees' `ctrlrange` is [-2.793, -0.254], so MuJoCo clamps 0 to -0.254 and
    every servo pulls against its own limit at reset. That is the excitation.
    """
    var src = read_model_source(SPOT)
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
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)

    # ⚠ THE SAME ALIASES THE STUDIO STEPS WITH, imported from the package
    # rather than respelled — a gate that built its own integrator would pass
    # while the studio's two were wired backwards.
    var euler = StudioIntegEll(dims)
    var impf = StudioImpFastEll(dims)
    var nact = dims.get_nact()
    var actions = List[Float64](length=nact, fill=0.0)
    var act = List[Scalar[DT]](length=nact, fill=Scalar[DT](0))

    var qvel0 = 0.0
    var zmax = -1e30
    for t in range(nstep):
        apply_actions_fields[DT](sf, d, actions, act, fmd.timestep)
        comptime if IMPLICIT:
            impf.step["cpu"](d, m)
        else:
            euler.step["cpu"](d, m)
        if t == 0:
            for i in range(dims.get_nv()):
                var v = abs(Float64(d.qvel.data[i]))
                if v > qvel0:
                    qvel0 = v
        var z = Float64(d.qpos.data[2])
        if z > zmax:
            zmax = z
    return Run(qvel0, zmax, Float64(d.qpos.data[2]))


def test_spot_asks_for_implicitfast_and_gets_it() raises:
    """The dispatch — parsed is not the same as honoured."""
    print("=== spot's <option integrator> reaches the dispatch ===")
    var src = read_model_source(SPOT)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    print("  integrator", fmd.integrator, " cone", studio_cone_of(fmd),
          " -> implicit?", studio_uses_implicit(fmd))
    assert_true(
        fmd.integrator == IntegratorType.IMPLICITFAST,
        "spot.xml says integrator='implicitfast'; the parser reports "
        + String(fmd.integrator),
    )
    # ⚠ THE ROW THAT WOULD HAVE CAUGHT THE ORIGINAL BUG. The value was
    # already parsed correctly and NOTHING READ IT, so an assertion on the
    # parse alone stays green while every model steps with Euler.
    assert_true(
        studio_uses_implicit(fmd),
        "spot asks for implicitfast but the studio's dispatch would step it"
        " with Euler — which is the defect: the option was parsed and unread.",
    )
    print("  PASS")


def test_first_step_matches_mujoco_for_both_integrators() raises:
    """One step, two integrators, two MuJoCo numbers.

    ⚠ THE FIRST STEP IS THE WHOLE MEASUREMENT and later steps are not. Both
    engines start from the same state with the same forces, so step 0 is a
    clean comparison; by step 100 the trajectories have diverged through the
    contact solver and any disagreement there is a different question.
    """
    print("=== spot, first step, vs MuJoCo ===")
    var e = _drop[False](1)
    var f = _drop[True](1)
    print("  Euler        |qvel|max", e.qvel0, "  (MuJoCo Euler        36.6005)")
    print("  implicitfast |qvel|max", f.qvel0, "  (MuJoCo implicitfast  2.8516)")
    assert_true(
        abs(e.qvel0 - 36.6005) < 1e-3,
        "our EULER step must still reproduce MuJoCo's Euler (36.6005); got "
        + String(e.qvel0)
        + ". This row is the negative control — if it moved, the change"
        " altered Euler, which must stay exactly what MuJoCo's Euler is.",
    )
    assert_true(
        abs(f.qvel0 - 2.8516) < 1e-3,
        "our IMPLICITFAST step must reproduce MuJoCo's implicitfast (2.8516);"
        " got " + String(f.qvel0)
        + ". A value near 36.6 means M_hat came out equal to M — i.e."
        " `dof_actdamp` is zero and `mjd_actuator_vel` is not reaching"
        " qDeriv, which is the half of this fix that does the work.",
    )
    print("  PASS")


def test_spot_stands_instead_of_flying() raises:
    """The reported symptom, as a number.

    ⚠ THE BOUND IS DELIBERATELY LOOSE. The bug is not "settles a centimetre
    low", it is a 90 kg robot passing 18 metres, so a generous threshold
    cannot be satisfied by a subtly wrong solve and cannot go red on a
    legitimate numerical change. Spot is dropped from 0.75 m and MuJoCo
    settles it at 0.654.
    """
    print("=== spot stays on the ground ===")
    var f = _drop[True](1500)
    print("  implicitfast  zmax", f.zmax, " final z", f.zfinal,
          "  (MuJoCo: 0.750 / 0.654)")
    assert_true(
        f.zmax < 1.0,
        "spot rose to " + String(f.zmax)
        + " m from a 0.75 m drop — it is being driven off the ground."
        " Stepped with Euler this reads 18.36.",
    )
    assert_true(
        abs(f.zfinal - 0.654) < 0.05,
        "spot should settle standing at MuJoCo's 0.654 m; it ended at "
        + String(f.zfinal),
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
