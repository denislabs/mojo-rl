"""Which integrator the studio builds, and which one a model gets.

⚠⚠ `EulerIntegrator`'s OWN DEFAULTS ARE NOT MuJoCo's. It defaults to
`CONE_TYPE = ELLIPTIC` and `SOLVER = "pgs"`; MuJoCo 3.10.0 defaults to
PYRAMIDAL and NEWTON — measured, on a model whose `<option>` says nothing:

    m.opt.cone   == 0  (mjCONE_PYRAMIDAL)
    m.opt.solver == 2  (mjSOL_NEWTON)

A tool that passes neither parameter therefore runs a friction cone and a
solver the reference does not use, on every model it opens. That is what the
studio did, including on the 33 Menagerie scenes that are pyramidal precisely
because they say nothing.

⚠ CONE IS WHAT VARIES; SOLVER DOES NOT. Measured across 114 models — all 57
Menagerie scenes and all 57 loadable in-repo ones — the solver is NEWTON in
112 and PGS in 2, while the cone splits 67 pyramidal / 45 elliptic. So the
studio builds a cone PAIR per integrator and picks by the model's declared
cone, and a model asking for a solver we did not build gets a warning naming
what it asked for rather than a silent substitution.

⚠⚠ AND `integrator` VARIES THREE WAYS, NOT TWO. Re-measured over 131 loadable
models (85 Menagerie scenes + 46 in-repo): 67 EULER, 50 IMPLICITFAST, **14
RK4**. The RK4 arm was the one the studio did not build, so it went to Euler
with a warning — see `StudioRk4Pyr` for what that cost. `studio_integrator_of`
is now the single selector and returns all three; the old boolean could not.

⚠ THIS LIVES IN THE PACKAGE, NOT IN `examples/physics_studio.mojo`, so the
gate can import the SAME aliases the studio steps with. A test that spelled
its own pair would pass while the studio's two were wired backwards — the
selection and the thing selected have to come from one place.

Measured side-effect, on the same models before and after (interleaved, min of
two runs): `unitree_go2` 921 -> 143 us/step and `aloha` 2588 -> 2423 us/step.
Matching the reference is also the faster path here.
"""

from ..fields import DynDims
from ..integrator.euler import EulerIntegrator
from ..integrator.implicit import ImplicitIntegrator
from ..integrator.rk4 import RK4Integrator
from ..parser.flat_model import FlatModelDef
from ..types import ConeType, SolverType, IntegratorType


comptime STUDIO_DT = DType.float64

# ⚠⚠ `MAX_CONDIM = 6`, NOT 3, AND NOT A FIFTH/SIXTH INTEGRATOR EITHER.
# `_contact_solve_env` clamps each contact's OWN condim down to this bound
# silently, so a studio built at 3 solved every `condim="6"` foot as condim 3
# — torsional and rolling friction dropped, with no diagnostic. 46 of the 142
# models in this tree need more than 3 (29 Menagerie scenes and 17 in-repo
# assets), including spot, apollo, all four unitree quadrupeds, both shadow
# hands and every dm_control dog.
#
# ⚠ SIX IS NOT FREE, AND THE FIRST VERSION OF THIS NOTE CLAIMED IT WAS. The
# row builders do loop over each CONTACT's own condim, so the extra rows are
# not built for a condim-3 contact — but `MAX_CONDIM` also sizes the region
# they are built INTO and the PYRAMIDAL edge count `2*(MAX_CONDIM-1)` per
# SLOT, and that shows up. Measured interleaved in one process, min of five
# rounds, 300 steps:
#
#     google_barkour_vb (needs 3)   118.3 -> 132.5 us/step   x1.12
#     boston_dynamics_spot (needs 6) 69.9 ->  86.6 us/step   x1.24
#
# ⚠ ONE BOUND FOR EVERY MODEL ANYWAY, rather than a third dispatch axis on top
# of cone and integrator: that would take the studio's four integrator
# instantiations to EIGHT, which this tree has repeatedly paid for in compile
# time, to save 12% on a tool that steps at 8 kHz and renders at 60 Hz.
# `MAX_CONDIM=6` is a superset of the answer 3 gives — verified, not assumed:
# barkour's 100-step trajectory against MuJoCo is unchanged to the last digit.
# A batched trainer sizing its own integrator should still pass the model's
# own `fmd.max_condim` and pay nothing.
comptime STUDIO_MAX_CONDIM = 6

# ⚠⚠ AN ENABLE, NOT A COUNT — AND WITHOUT IT THE STUDIO NEVER RAN
# `mj_solNoSlip`. `solve_newton`'s `NOSLIP_ITER` used to BE the number of
# friction-only sweeps, which meant the number had to be known when the
# integrator was instantiated. The studio instantiates its five integrators at
# compile time and learns which file it is opening at run time, so there was no
# value it could pass: it defaulted to 0 and the pass was off for every model,
# whatever `<option noslip_iterations>` said.
#
# The count now travels in `MODEL_META_IDX_NOSLIP_ITERATIONS`, written by
# `fields_build` from the model's own `<option>`, and this parameter only
# decides whether the code is EMITTED. So 1 here means "build the pass in", not
# "run one sweep": a model that asks for 5 runs 5, and a model that asks for
# nothing still runs none.
#
# ⚠ NOT A SIXTH AND SEVENTH INTEGRATOR. This is the axis `MAX_CONDIM` above
# argued its way out of doubling, for the same reason — an enable that is
# always on costs one instantiation, a count would have cost a pair per
# distinct value in the tree.
#
# Measured across the SEVEN Menagerie scenes that set `<option
# noslip_iterations>`, as how far the pass moves qpos in one step in each
# engine — the quantity that is well-posed whatever else the model disagrees
# about:
#
#     robot_soccer_kit     ours 7.062e-08   MuJoCo 7.062e-08   ratio 1.0
#     shadow_dexee         ours 1.442e-14   MuJoCo 1.409e-14   ratio 1.0
#     flybody              ours 1.927e-15   MuJoCo 1.926e-15   ratio 1.0
#     hello_robot_stretch  ours 0           MuJoCo 0
#     umi_gripper          ours 0           MuJoCo 0
#     dynamixel_2r         ours 0           MuJoCo 1.735e-18
#     hello_robot_stretch_3 ours 2.244e-03  MuJoCo 9.765e-05   ratio 23
#
# `robot_soccer_kit`'s whole board residual was this pass: it went 7.062e-08 ->
# 7.589e-19 against MuJoCo, and it covers the PYRAMIDAL branch while dexee and
# flybody cover the ELLIPTIC one.
#
# ⚠⚠ AND `hello_robot_stretch_3` GOT WORSE — 4.419e-03 -> 6.649e-03 — WHICH IS
# NOT THIS PASS BEING WRONG. It is the tree's #1 board row already, at
# 4.4e-03 with the pass off, and `mj_solNoSlip` redistributes friction WITHIN
# the contact set it is handed. Hand it a set whose normals are 156-of-12850
# different and it redistributes to different places; the ratio-1.0 rows above
# are the control that says the routine agrees when its inputs do. The residual
# also grows monotonically with the iteration count (1 -> 6.383e-03, 10 ->
# 7.117e-03) and is IDENTICAL at `noslip_tolerance` 1e-6 and 0, so neither the
# stopping rule nor a convergence failure is in it. Enabling the pass makes
# that scene a 23x AMPLIFIER of its own upstream defect, which is a better
# probe than the quiet 4.4e-03 was.
comptime STUDIO_NOSLIP = 1

comptime StudioIntegPyr = EulerIntegrator[
    STUDIO_DT, DynDims, ConeType.PYRAMIDAL, 1, "newton",
    MAX_CONDIM=STUDIO_MAX_CONDIM, NOSLIP_ITER=STUDIO_NOSLIP,
]
comptime StudioIntegEll = EulerIntegrator[
    STUDIO_DT, DynDims, ConeType.ELLIPTIC, 1, "newton",
    MAX_CONDIM=STUDIO_MAX_CONDIM, NOSLIP_ITER=STUDIO_NOSLIP,
]

# ⚠⚠ AND THE IMPLICIT PAIR, BECAUSE THE FILE ASKS FOR IT. `integrator` was
# the one `<option>` the studio still ignored, and it is not cosmetic: spot
# and g1 both say `implicitfast`, both are driven by `<position>` servos, and
# spot's `dof_damping` is 0 — so its ONLY damping is the actuator's `kv=40`,
# which explicit Euler integrates unstably. Measured at spot's first step,
# same forces (qfrc_actuator 127.824) and same qacc (18300.2) in both
# engines:
#
#     MuJoCo implicitfast : |qvel|max  2.8516
#     MuJoCo Euler        : |qvel|max 36.6005
#     ours   (Euler)      : |qvel|max 36.600467   <- exact agreement
#
# i.e. our Euler is RIGHT and was being asked the wrong question. Stepped
# with Euler anyway, spot leaves the ground and passes 18 m.
comptime StudioImpFastPyr = ImplicitIntegrator[
    STUDIO_DT, DynDims, ConeType.PYRAMIDAL, 1, "newton", SKIP_RNE_DERIV=True,
    MAX_CONDIM=STUDIO_MAX_CONDIM, NOSLIP_ITER=STUDIO_NOSLIP,
]
comptime StudioImpFastEll = ImplicitIntegrator[
    STUDIO_DT, DynDims, ConeType.ELLIPTIC, 1, "newton", SKIP_RNE_DERIV=True,
    MAX_CONDIM=STUDIO_MAX_CONDIM, NOSLIP_ITER=STUDIO_NOSLIP,
]

# ⚠⚠ AND RK4, THE LAST SUBSTITUTION LEFT — AND IT WAS WORTH 9.200e-06.
# `bitcraze_crazyflie_2` sat at #5 on the Menagerie board and the residual was
# read as a defect in the SITE transmission (an actuator whose moment turns
# with the body, frozen at RK4's stage 0). It was not. The scene ships
# `<option integrator="RK4">`, the studio stepped it with Euler, and the
# signature says so outright: in free flight the acceleration is very nearly
# constant, semi-implicit Euler moves `a*dt^2` where RK4 moves `a*dt^2/2`, and
# ours came out EXACTLY 2x the reference in every dof. Measured, one step from
# the keyframe with the board's own random controls:
#
#     ours Euler   (as the studio stepped it)      9.2004e-06
#     ours RK4     (this alias)                    3.3142e-13
#     MuJoCo forced to Euler, model untouched      5.2940e-23
#
# The last line is the control: nothing but the integrator moves it.
#
# ⚠ 14 OF THE 131 MODELS IN THIS TREE ASK FOR RK4, NOT ONE. Menagerie has
# exactly one (crazyflie), but the Gym locomotion suite is RK4 to a model —
# ant, hopper, walker2d, swimmer, reacher, both humanoids and both inverted
# pendulums — plus four dm_control models. Their ENVS were already stepping
# RK4 (`*_config.mojo` sizes `rk4_extra_workspace_size`); it was the studio,
# and the fidelity harness that mirrors it, that were not. On the in-repo
# models the substitution is far bigger than crazyflie's, because they are in
# CONTACT — hopper, 60 steps from qpos0 at a constant ctrl, two live contacts:
#
#     ours Euler                                   3.8885e-03
#     ours RK4                                     1.2490e-16
#
# which is also the arm that proves the per-stage contact solve inside
# `RK4Integrator.step` works: crazyflie flies and never touches anything.
#
# ⚠ ONE CONE, NOT TWO, AND THAT IS THE `studio_solver_warning` PRECEDENT
# RATHER THAN AN OVERSIGHT. Measured over the same 131 models, ALL 14 RK4
# models are PYRAMIDAL and none is elliptic — the cone varies on the Euler and
# implicitfast axes (67/45 over the tree) and does not vary here. So an `RK4`
# + `cone="elliptic"` model gets a WARNING naming what it asked for and what it
# got, exactly as a model asking for PGS or CG does. Build the second one here
# and add its branch below the day that census stops reading 14/0.
#
# ⚠ THIS ONE COSTS 12 s ON THE STUDIO'S BUILD — 135 -> 147 s, interleaved
# old/new on one machine, first round only (Mojo's build cache takes the
# second round of BOTH to 29 s and would report the difference as zero). The
# cost of a second cone was NOT measured; do not assume it is another 12.
comptime StudioRk4Pyr = RK4Integrator[
    STUDIO_DT, DynDims, ConeType.PYRAMIDAL, 1, "newton",
    MAX_CONDIM=STUDIO_MAX_CONDIM, NOSLIP_ITER=STUDIO_NOSLIP,
]


def studio_cone_of(fmd: FlatModelDef) -> Int:
    """The cone the studio will step this model with.

    Trivial by design — the point is that it is ONE function, so the studio's
    dispatch and the gate cannot disagree about which way round the two
    integrators go.
    """
    return (
        ConeType.ELLIPTIC if fmd.cone == ConeType.ELLIPTIC
        else ConeType.PYRAMIDAL
    )


def studio_solver_warning(fmd: FlatModelDef) -> String:
    """"" when the model's solver is one we build, else what to tell the user.

    ⚠ NAMED, NOT SWALLOWED. `<option solver="PGS">` is a real request and two
    models in the tree make it; stepping them with Newton without saying so
    would make the studio quietly disagree with the file it is showing.
    """
    if fmd.solver == SolverType.NEWTON:
        return String("")
    var asked = String("PGS")
    if fmd.solver == SolverType.CG:
        asked = String("CG")
    elif fmd.solver != SolverType.PGS:
        asked = String("solver #") + String(fmd.solver)
    return (
        "Warning: this model asks for <option solver='" + asked + "'>; the"
        " studio builds NEWTON only and is stepping with that."
    )


def studio_condim_warning(fmd: FlatModelDef) -> String:
    """"" when the studio's `MAX_CONDIM` covers this model, else what to say.

    ⚠⚠ THE RECORDER EXISTED AND NOTHING READ IT. `fmd.max_condim` has carried
    "the condim this model needs" since spot's `condim="6"` feet were found
    being solved as condim 3, precisely so a caller could compare it against
    the bound it built — and no caller ever did. `_contact_solve_env` clamps
    silently, so the studio ran 46 of the tree's 142 models with their
    torsional and rolling friction dropped and said nothing. Recording a
    requirement is not checking it.

    ⚠ THIS SHOULD BE UNREACHABLE TODAY, since `STUDIO_MAX_CONDIM` is 6 and
    MuJoCo's condim domain is {1, 3, 4, 6}. That is the point: it is the
    assertion that the bound above still covers the domain, sitting where the
    user would be told if it ever stopped.
    """
    if fmd.max_condim <= STUDIO_MAX_CONDIM:
        return String("")
    return (
        "Warning: this model needs condim " + String(fmd.max_condim)
        + " and the studio solves contacts at " + String(STUDIO_MAX_CONDIM)
        + "; the extra friction rows are dropped, so it will spin and roll"
        " with less resistance than the file asks for."
    )


def studio_integrator_of(fmd: FlatModelDef) -> Int:
    """The `IntegratorType` the studio will actually STEP this model with.

    ⚠ ONE FUNCTION, for the same reason `studio_cone_of` is one: the studio's
    dispatch, the fidelity harness that mirrors it and the gate must not be
    able to disagree about which integrator a model gets. It returns what is
    STEPPED, not what the file said — `implicit` maps to `IMPLICITFAST` here,
    and `studio_integrator_warning` is where the user is told so.

    ⚠⚠ IT RETURNS THREE VALUES NOW AND IT USED TO RETURN TWO. The predicate
    below (`studio_uses_implicit`) was the whole of the dispatch, so every
    caller was written as `if implicit: ... else: EULER` — and RK4, which is
    not implicit, fell out of that `else` into Euler. A boolean cannot carry a
    third case; that is why this replaced it rather than gaining a sibling.
    """
    if fmd.integrator == IntegratorType.RK4:
        return IntegratorType.RK4
    if (
        fmd.integrator == IntegratorType.IMPLICITFAST
        or fmd.integrator == IntegratorType.IMPLICIT
    ):
        return IntegratorType.IMPLICITFAST
    return IntegratorType.EULER


def studio_uses_implicit(fmd: FlatModelDef) -> Bool:
    """True when this model must be stepped by the implicit pair.

    Kept as a name over `studio_integrator_of` — NOT as a second rule. A
    caller that still branches on this alone silently steps an RK4 model with
    Euler, which is the defect this trio was split to end.
    """
    return studio_integrator_of(fmd) == IntegratorType.IMPLICITFAST


def studio_uses_rk4(fmd: FlatModelDef) -> Bool:
    """True when this model must be stepped by `StudioRk4Pyr`."""
    return studio_integrator_of(fmd) == IntegratorType.RK4


def studio_integrator_warning(fmd: FlatModelDef) -> String:
    """"" when we step what the file asked for, else what to tell the user.

    ⚠ TWO SUBSTITUTIONS ARE POSSIBLE AND BOTH ARE NAMED. `implicit` gets the
    IMPLICITFAST pair — the same M_hat minus the dense RNE velocity
    derivative, so it is damped correctly and merely less exact in the
    Coriolis terms. `RK4` + `cone="elliptic"` gets RK4 with the PYRAMIDAL cone,
    because that is the only RK4 instantiation built (see `StudioRk4Pyr`: the
    combination occurs in 0 of the tree's 131 models). Neither is silent; a
    substitution nobody is told about is how a tool ends up disagreeing with
    the file it is displaying.

    ⚠⚠ `RK4` ITSELF NO LONGER WARNS, AND THE WARNING IS NOT WHAT WAS WRONG.
    This function named the substitution correctly for as long as it existed —
    "Expect a less accurate trajectory" — and the substitution still put
    crazyflie at #5 on the Menagerie board with a residual that was filed
    against the site transmission instead. A warning is not a substitute for
    stepping the file.
    """
    if fmd.integrator == IntegratorType.EULER:
        return String("")
    if fmd.integrator == IntegratorType.IMPLICITFAST:
        return String("")
    if fmd.integrator == IntegratorType.IMPLICIT:
        return String(
            "Note: this model asks for <option integrator='implicit'>; the"
            " studio is stepping it with IMPLICITFAST — the same implicit"
            " damping, without the dense RNE velocity derivative."
        )
    if fmd.integrator == IntegratorType.RK4:
        if fmd.cone == ConeType.ELLIPTIC:
            return String(
                "Warning: this model asks for <option integrator='RK4'"
                " cone='elliptic'>; the studio builds RK4 with the PYRAMIDAL"
                " cone only and is stepping with that."
            )
        return String("")
    return (
        "Warning: unknown <option integrator> #" + String(fmd.integrator)
        + "; stepping with EULER."
    )
