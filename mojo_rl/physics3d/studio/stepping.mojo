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
studio builds TWO integrators and picks by the model's declared cone, and a
model asking for a solver we did not build gets a warning naming what it asked
for rather than a silent substitution.

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
from ..parser.flat_model import FlatModelDef
from ..types import ConeType, SolverType, IntegratorType


comptime STUDIO_DT = DType.float64

comptime StudioIntegPyr = EulerIntegrator[
    STUDIO_DT, DynDims, ConeType.PYRAMIDAL, 1, "newton", MAX_CONDIM=3
]
comptime StudioIntegEll = EulerIntegrator[
    STUDIO_DT, DynDims, ConeType.ELLIPTIC, 1, "newton", MAX_CONDIM=3
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
    STUDIO_DT, DynDims, ConeType.PYRAMIDAL, 1, "newton", SKIP_RNE_DERIV=True
]
comptime StudioImpFastEll = ImplicitIntegrator[
    STUDIO_DT, DynDims, ConeType.ELLIPTIC, 1, "newton", SKIP_RNE_DERIV=True
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


def studio_uses_implicit(fmd: FlatModelDef) -> Bool:
    """True when this model must be stepped by the implicit pair.

    ⚠ ONE FUNCTION, for the same reason `studio_cone_of` is one: the studio's
    dispatch and the gate must not be able to disagree about which integrator
    a model gets.

    `implicit` maps here too — see `studio_integrator_warning`.
    """
    return (
        fmd.integrator == IntegratorType.IMPLICITFAST
        or fmd.integrator == IntegratorType.IMPLICIT
    )


def studio_integrator_warning(fmd: FlatModelDef) -> String:
    """"" when we step what the file asked for, else what to tell the user.

    ⚠ TWO SUBSTITUTIONS ARE POSSIBLE AND BOTH ARE NAMED. `implicit` gets the
    IMPLICITFAST pair — the same M_hat minus the dense RNE velocity
    derivative, so it is damped correctly and merely less exact in the
    Coriolis terms. `RK4` gets Euler, which is a real difference in accuracy
    rather than in stability. Neither is silent; a substitution nobody is told
    about is how a tool ends up disagreeing with the file it is displaying.
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
        return String(
            "Warning: this model asks for <option integrator='RK4'>; the"
            " studio builds Euler and implicitfast only and is stepping with"
            " EULER. Expect a less accurate trajectory, not an unstable one."
        )
    return (
        "Warning: unknown <option integrator> #" + String(fmd.integrator)
        + "; stepping with EULER."
    )
