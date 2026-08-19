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
from ..parser.flat_model import FlatModelDef
from ..types import ConeType, SolverType


comptime STUDIO_DT = DType.float64

comptime StudioIntegPyr = EulerIntegrator[
    STUDIO_DT, DynDims, ConeType.PYRAMIDAL, 1, "newton", MAX_CONDIM=3
]
comptime StudioIntegEll = EulerIntegrator[
    STUDIO_DT, DynDims, ConeType.ELLIPTIC, 1, "newton", MAX_CONDIM=3
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
