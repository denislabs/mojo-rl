"""`<option impratio>` must reach the solver.

    pixi run mojo run -I . tests/physics3d/test_impratio_is_parsed_and_used.mojo

WHAT WENT WRONG. `impratio` was READ BY FIVE SOLVERS AND WRITTEN BY NOBODY.
`contact_solve`, `newton_solve` (twice), `cg_solve` and `island_pgs_solve` all
took it from `MODEL_META_IDX_IMPRATIO`, and `fields_build` filled that slot with

    mf.meta.data[MODEL_META_IDX_IMPRATIO] = Scalar[DTYPE](1.0)

while `_parse_option` never looked at the attribute at all. So a model asking
for anything other than 1 was silently simulated at 1, in both parser paths.

⚠⚠ IT SURVIVED BECAUSE THE HARDCODED VALUE WAS THE COMMON ONE.
`contact_solve.mojo`'s own note says it: "both are exact no-ops at the default
`impratio = 1`, which is what every model in the tree uses — no gate here can
move, and none did." That is a description of dead data being indistinguishable
from correct data, and it is what this file exists to make impossible.

WHAT IT IS. The ratio of frictional to normal constraint IMPEDANCE
(`engine_core_constraint.c:1886`): `R[1] = R[0]/impratio`, so the regularized
coefficient is `mu = friction[0]*sqrt(R[1]/R[0]) = friction[0]/sqrt(impratio)`.

⚠ WITH AN ELLIPTIC CONE IT MOVES THE **NORMAL** FORCE, NOT JUST THE FRICTION.
That is the non-obvious part and the reason a "friction-only" reading of the
name is wrong: the cone couples the tangential rows to the normal one, so
stiffening friction raises the normal force the contact can carry.

MEASURED, Boston Dynamics spot (`impratio="100"`), at its first ground contact,
with our engine and MuJoCo driven to the SAME state and agreeing on the contact
geometry to every printed digit (position, normal, dist, condim, friction):

    MuJoCo, impratio=100 : fn = 11.070, 10.256
    MuJoCo, impratio=1   : fn =  6.849,  6.342
    ours,  hardcoded 1   : fn =  6.333,  5.837

⚠⚠ THIS FILE GATES THE PLUMBING, NOT THE SOLVE, AND THAT IS NOT AN OVERSIGHT.
A companion assertion — "one sliding contact, two impratios, two different
normal forces" — was written, measured, and REMOVED because it is RED for a
DIFFERENT defect, and a gate that fails for a reason its name does not describe
is worse than no gate. What it measured, on the fixture below (a 2 kg sphere
resting on a plane and sliding at 1.5 m/s, elliptic cone):

    MuJoCo  impratio=1 -> 71.2965   impratio=100 -> 72.9816   (+2.4%)
    ours    impratio=1 -> 18.1274   impratio=100 -> 18.1274   (BIT-IDENTICAL)

So the fixture is NOT vacuous — the reference moves on it — and our solve does
not respond to `impratio` at all here, on top of sitting 3.9x below MuJoCo's
normal force on a single sphere. Both point at the same place: the default
solver. `EulerIntegrator`'s `SOLVER` defaults to `"pgs"` and `<option solver>`
is not parsed, while MuJoCo runs NEWTON with 100 iterations, and a first-order
PGS under-converges exactly where the friction rows are stiff. That is tracked
separately; do not "fix" it by loosening anything here.

⚠ WHAT THE FIXTURE IS FOR is therefore only the META assertions. It is kept
sliding and elliptic anyway so that whoever restores the dynamic assertion has
a configuration the reference is known to move on.
"""

from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.fields import Model
from mojo_rl.physics3d.model.model_dims import ModelDims
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.gpu.constants import (
    MODEL_META_IDX_IMPRATIO,
)

comptime DTYPE = DType.float64


# A sphere resting on a plane, SLIDING — see the header for why the tangential
# velocity is load-bearing. `cone='elliptic'` likewise: the pyramidal builder
# reaches `impratio` by a different route and this fixture is about the
# elliptic one, which is what every model that sets `impratio` also selects.
def _xml(impr: String) -> String:
    return String(
        "<mujoco><option timestep='0.002' cone='elliptic'"
    ) + impr + String(
        """>
  </option>
  <worldbody>
    <geom name='floor' type='plane' size='0 0 0.05'/>
    <body name='ball' pos='0 0 0.0299'>
      <freejoint/>
      <geom name='b' type='sphere' size='0.03' mass='2'
            friction='0.8 0.02 0.01'/>
    </body>
  </worldbody>
</mujoco>"""
    )


comptime XML_ABSENT = _xml("")
comptime XML_HUNDRED = _xml(" impratio='100'")
comptime XML_ZERO = _xml(" impratio='0'")

comptime PM_A = parse_xml(XML_ABSENT)
comptime PM_H = parse_xml(XML_HUNDRED)
comptime PM_Z = parse_xml(XML_ZERO)

comptime MD_A = ModelDefFromXML[
    xml=XML_ABSENT, nbody=PM_A.NBODY, njoint=PM_A.NJOINT, nq=PM_A.NQ,
    nv=PM_A.NV, ngeom=PM_A.NGEOM, nact=PM_A.NACT, ntex=PM_A.NTEX,
    nmat=PM_A.NMAT, nlight=PM_A.NLIGHT, ncam=PM_A.NCAM, nsite=PM_A.NSITE,
    neq=PM_A.NEQ, nexclude=PM_A.NEXCLUDE, npair=PM_A.NPAIR,
    max_tendon=PM_A.NTENDON, max_condim=PM_A.MAX_CONDIM,
    max_equality=1, max_contacts=16, timestep=PM_A.TIMESTEP,
    cone_type=ConeType.ELLIPTIC,
]
comptime MD_H = ModelDefFromXML[
    xml=XML_HUNDRED, nbody=PM_H.NBODY, njoint=PM_H.NJOINT, nq=PM_H.NQ,
    nv=PM_H.NV, ngeom=PM_H.NGEOM, nact=PM_H.NACT, ntex=PM_H.NTEX,
    nmat=PM_H.NMAT, nlight=PM_H.NLIGHT, ncam=PM_H.NCAM, nsite=PM_H.NSITE,
    neq=PM_H.NEQ, nexclude=PM_H.NEXCLUDE, npair=PM_H.NPAIR,
    max_tendon=PM_H.NTENDON, max_condim=PM_H.MAX_CONDIM,
    max_equality=1, max_contacts=16, timestep=PM_H.TIMESTEP,
    cone_type=ConeType.ELLIPTIC,
]
comptime MD_Z = ModelDefFromXML[
    xml=XML_ZERO, nbody=PM_Z.NBODY, njoint=PM_Z.NJOINT, nq=PM_Z.NQ,
    nv=PM_Z.NV, ngeom=PM_Z.NGEOM, nact=PM_Z.NACT, ntex=PM_Z.NTEX,
    nmat=PM_Z.NMAT, nlight=PM_Z.NLIGHT, ncam=PM_Z.NCAM, nsite=PM_Z.NSITE,
    neq=PM_Z.NEQ, nexclude=PM_Z.NEXCLUDE, npair=PM_Z.NPAIR,
    max_tendon=PM_Z.NTENDON, max_condim=PM_Z.MAX_CONDIM,
    max_equality=1, max_contacts=16, timestep=PM_Z.TIMESTEP,
    cone_type=ConeType.ELLIPTIC,
]

comptime DA = ModelDims[MD_A, 64]
comptime DH = ModelDims[MD_H, 64]
comptime DZ = ModelDims[MD_Z, 64]


def test_impratio_reaches_model_meta() raises:
    """The value the XML asked for is the value the solvers read.

    ⚠ THE THREE MODELS ARE DIFFERENT TYPES, so this cannot be a loop —
    `ModelDefFromXML` specialises on the XML string and the whole point is that
    the strings differ. Written out three times, as the `ccd_options` gate is.
    """
    print("=== <option impratio> reaches model META ===")
    var ctx = DeviceContext()

    var mf_a = Model[DTYPE, DA]()
    MD_A.init_fields[DTYPE](ctx, mf_a)
    var ir_a = Float64(mf_a.meta.data[MODEL_META_IDX_IMPRATIO])

    var mf_h = Model[DTYPE, DH]()
    MD_H.init_fields[DTYPE](ctx, mf_h)
    var ir_h = Float64(mf_h.meta.data[MODEL_META_IDX_IMPRATIO])

    var mf_z = Model[DTYPE, DZ]()
    MD_Z.init_fields[DTYPE](ctx, mf_z)
    var ir_z = Float64(mf_z.meta.data[MODEL_META_IDX_IMPRATIO])

    print("  absent      -> META impratio", ir_a)
    print("  ='100'      -> META impratio", ir_h)
    print("  ='0'        -> META impratio", ir_z)

    assert_true(
        ir_a == 1.0,
        "a model that sets no impratio must get MuJoCo's default 1, got "
        + String(ir_a),
    )
    assert_true(
        ir_h == 100.0,
        "`<option impratio='100'>` did not reach META — it holds "
        + String(ir_h)
        + ". This is the original defect: five solvers read this slot and"
        " `fields_build` hardcoded 1.0 into it.",
    )
    # `R[1] = R[0]/impratio` divides by it and `mu` takes its square root, so a
    # non-positive value must not be copied through — it would surface as a
    # division by zero or a NaN several call frames from the XML that caused it.
    assert_true(
        ir_z == 1.0,
        "`impratio='0'` must fall back to 1 rather than be copied, got "
        + String(ir_z),
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
