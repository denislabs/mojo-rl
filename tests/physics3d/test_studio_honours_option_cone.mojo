"""The studio steps the cone the MODEL declares — and the two differ.

WHY THIS EXISTS
===============
`test_option_solver_choice_is_parsed` gates that `<option cone/solver>` is
READ. Reading it changes nothing on its own: the studio was passing neither
parameter, so `EulerIntegrator` fell back to its own defaults — ELLIPTIC and
`"pgs"` — while MuJoCo 3.10.0 defaults to PYRAMIDAL and NEWTON. Every model
opened in the studio, including the 33 Menagerie scenes that are pyramidal
precisely because they say nothing, simulated with a friction cone and a
solver the reference does not use.

TWO ARMS, AND THE SECOND IS THE ONE THAT BITES:

  1. the SELECTION — `studio_cone_of(fmd)` returns what MuJoCo reports for the
     same file. Gated on real models at both ends of the split.

  2. ⚠⚠ the CONE PARAMETER IS NOT INERT. Selecting correctly between two
     integrators that behave identically would pass every arm of (1) while
     changing no physics — and a pair wired BACKWARDS would too. So the two
     are stepped from the same state on a model with contacts and their
     results must DIFFER. Without this the whole feature could be a no-op.

⚠ THE ALIASES COME FROM `studio.stepping`, the same module the studio steps
with. A test spelling its own pair would pass while the studio's two were
swapped.

Run: pixi run mojo run -I . tests/physics3d/test_studio_honours_option_cone.mojo
"""

from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.studio.stepping import (
    StudioIntegPyr, StudioIntegEll, studio_cone_of, studio_solver_warning,
    STUDIO_DT,
)


# ⚠ THE GOLDENS ARE MuJoCo'S, read with
#     m = mujoco.MjModel.from_xml_path(P); m.opt.cone   # 0 pyramidal, 1 elliptic
# from each model's own directory. Both ends of the split, both real.
# ⚠ `agility_cassie`, NOT `unitree_go2` — my first draft used go2 "because it
# is a quadruped that says nothing", and go2.xml says `cone="elliptic"`. The
# gate failed and the PARSER was right; the golden was wrong. Every exemplar
# here was read off MuJoCo, not off a memory of the distribution.
comptime PYRAMIDAL_MODEL = String(
    "references/mujoco_menagerie-main/agility_cassie/scene.xml"
)
comptime ELLIPTIC_MODEL = String(
    "references/mujoco_menagerie-main/aloha/scene.xml"
)
# Contacts from the first step, which is what makes the cone observable at all.
comptime CONTACT_MODEL = String("mojo_rl/envs/ant/assets/ant.xml")


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


def main() raises:
    var t = Tally()
    print("=== the studio steps the cone the model declares ===")

    # ── 1. the selection ─────────────────────────────────────────────────
    print("--- selection, against MuJoCo's m.opt.cone ---")
    var pyr = parse_model_runtime(PYRAMIDAL_MODEL)
    var ell = parse_model_runtime(ELLIPTIC_MODEL)
    t.eq(studio_cone_of(pyr), ConeType.PYRAMIDAL,
         "agility_cassie (says nothing -> MuJoCo's default)")
    t.eq(studio_cone_of(ell), ConeType.ELLIPTIC,
         "aloha (<option cone='elliptic'>)")

    # ⚠ NON-VACUITY: if both models read the same, the selection is a constant
    # and every arm above is decoration.
    t.truth(studio_cone_of(pyr) != studio_cone_of(ell),
            "the two models actually select DIFFERENT cones")

    # A solver we do not build is named, not swallowed.
    t.truth(studio_solver_warning(pyr).byte_length() == 0,
            "a Newton model draws no warning")

    # ── 2. the cone parameter is not inert ───────────────────────────────
    # Step the SAME model from the SAME state with each integrator. A cone
    # that reached nothing would give identical qvel and this arm would fail.
    print("--- the two integrators disagree on a model with contacts ---")
    var fmd = parse_model_runtime(CONTACT_MODEL)
    var dims = dims_from_flat(fmd)
    var nv = dims.get_nv()

    var m_a = Model[STUDIO_DT, DynDims](dims)
    build_model_runtime[STUDIO_DT](fmd, dims, m_a)
    var m_b = Model[STUDIO_DT, DynDims](dims)
    build_model_runtime[STUDIO_DT](fmd, dims, m_b)

    var d_a = Data[STUDIO_DT, DynDims, 1](dims)
    var d_b = Data[STUDIO_DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d_a.qpos.data[i] = Scalar[STUDIO_DT](fmd.qpos0[i])
        d_b.qpos.data[i] = Scalar[STUDIO_DT](fmd.qpos0[i])
    # A sideways push, so the FRICTION cone — not just the normal — is what
    # the two solvers have to disagree about.
    for i in range(nv):
        var v = Scalar[STUDIO_DT](0.35) if i % 3 == 0 else Scalar[STUDIO_DT](0)
        d_a.qvel.data[i] = v
        d_b.qvel.data[i] = v

    var ip = StudioIntegPyr(dims)
    var ie = StudioIntegEll(dims)
    for _ in range(12):
        ip.step["cpu"](d_a, m_a)
        ie.step["cpu"](d_b, m_b)

    var worst = Float64(0)
    for i in range(nv):
        var dd = Float64(d_a.qvel.data[i]) - Float64(d_b.qvel.data[i])
        if dd < 0:
            dd = -dd
        if dd > worst:
            worst = dd
    print("    max |qvel_pyramidal - qvel_elliptic| =", worst)
    t.truth(worst > 1e-9,
            "the cone parameter REACHES the solve (the two results differ)")

    # ⚠ AND BOTH MUST BE FINITE. "They differ" is also true when one of them
    # diverged, which would make this arm pass on a broken integrator.
    var finite = True
    for i in range(nv):
        var a = Float64(d_a.qvel.data[i])
        var b = Float64(d_b.qvel.data[i])
        if not (a == a) or not (b == b) or a > 1e12 or b > 1e12:
            finite = False
    t.truth(finite, "both integrators stayed finite (control)")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error(
            "test_studio_honours_option_cone: " + String(t.fails) + " failed"
        )
