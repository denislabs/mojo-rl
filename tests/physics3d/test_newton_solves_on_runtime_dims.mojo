"""Newton must produce contact forces on the RUNTIME-DIMS path, not just the
comptime one.

    pixi run mojo run -I . tests/physics3d/test_newton_solves_on_runtime_dims.mojo

WHAT WENT WRONG. `newton_solve` sized its workspace guard from a COMPTIME
member:

    comptime SOLVER_WS = 81 * MC + 12 * MC * D.NV

On a DYNAMIC provider `D.NV` is `DIM_POISON = -1` and `MC` floors to 1, so that
is `81 - 12` = **69 scalars for every model, whatever its size**. The guard it
fed then compared the real region against 69 and took its early `return`:

    if ws_end_elliptic(max_contacts, nv, MAX_CONDIM) > SOLVER_WS:
        print("FATAL: ..."); return

so `_newton_solve_env` COMPUTED NO CONTACT FORCE AT ALL on every runtime-loaded
model. It is the same unconverted `81*MC + 12*MC*D.NV` literal that
`ContactScratch` allocated with until 2026-08-18; that copy was fixed and this
consumer was not.

⚠⚠ IT PRINTED "FATAL" ON EVERY STEP AND STILL WENT UNNOTICED FOR MONTHS, which
is the part worth remembering. The message names `MAX_CONDIM` — a cause that
had nothing to do with it — and the symptom is a body falling through the floor,
which reads as a physics bug and sends you to the collision code. Anyone
diagnosing it while filtering stdout (`2>/dev/null`, a `grep` for the numbers
they expected) never sees the engine saying exactly what is wrong.

MEASURED, a 5 kg sphere dropped 0.30 m onto a plane, runtime dims:

    PGS    : zmin 0.0225032, final 0.0330415   (MuJoCo: 0.02250 / 0.03304)
    Newton : zmin -43.87,    final -43.87      <- exactly free fall, 0.5*9.81*3^2

⚠ IT MATTERS BECAUSE THE RUNTIME PATH IS THE STUDIO'S, and MuJoCo's DEFAULT
solver is Newton (`m.opt.solver == 2`). Every model opened in the studio was
therefore restricted to a solver MuJoCo does not use by default, and anyone
selecting the reference's own solver got a scene that fell through the ground.

⚠ THE ASSERTION IS "NEWTON AGREES WITH PGS", NOT A GOLDEN. Both solvers solve
the same problem, so on a single frictionless-ish resting contact they must
land in the same place; pinning a number instead would freeze whatever this
build produces and would keep passing if Newton went back to returning early
(its answer would simply become a different frozen number). The fall-through
assertion is kept separately and explicitly, because that is the actual failure
mode and it must be named.
"""

from std.math import abs
from max.gpu.host import DeviceContext

from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.gpu.constants import META_IDX_NUM_CONTACTS

comptime DT = DType.float64
comptime N_STEP = 1500

# ⚠ BUILT FROM A STRING, NOT A FILE. `parse_xml_full` takes the text, so the
# fixture needs no asset on disk and cannot fail for a path reason on someone
# else's checkout. `solref`/`solimp` are the duplo/spot foot values — a STIFF
# contact, which is where a workspace bug shows rather than hides.
comptime BALL_XML = String(
    """<mujoco><option timestep='0.002' cone='elliptic'/>
  <worldbody>
    <geom name='floor' type='plane' size='0 0 0.05'/>
    <body name='ball' pos='0 0 0.30'>
      <freejoint/>
      <geom name='b' type='sphere' size='0.036' mass='5'
            solref='0.004 1' solimp='0.015 1 0.036' friction='0.8 0.02 0.01'/>
    </body>
  </worldbody>
</mujoco>"""
)


def _drop[SOLV: StaticString]() raises -> Tuple[Float64, Float64, Int]:
    """Drop the ball on the RUNTIME-DIMS path; return (zmin, zfinal, ncon)."""
    var fmd = parse_xml_full(BALL_XML, String(""))
    var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=0)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    var integ = EulerIntegrator[
        DT, DynDims, CONE_TYPE=ConeType.ELLIPTIC, BATCH=1, SOLVER=SOLV,
        MAX_CONDIM=3,
    ](dims)
    d.qpos.data[2] = Scalar[DT](0.30)
    d.qpos.data[3] = Scalar[DT](1)
    var zmin = 1e30
    for _ in range(N_STEP):
        integ.step["cpu"](d, m)
        var z = Float64(d.qpos.data[2])
        if z < zmin:
            zmin = z
    return (
        zmin,
        Float64(d.qpos.data[2]),
        Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS])),
    )


def test_newton_does_not_fall_through_on_runtime_dims() raises:
    """The actual failure mode, named: no contact force at all."""
    print("=== Newton on runtime dims: does the ball stop? ===")
    var r = _drop["newton"]()
    print("  newton  zmin", r[0], " final", r[1], " ncon", r[2])
    # ⚠ THE THRESHOLD IS GENEROUS ON PURPOSE. The bug is not "slightly wrong
    # height", it is 44 METRES of free fall, so a loose bound cannot be
    # satisfied by a subtly broken solve and cannot go red on a legitimate
    # numerical change.
    assert_true(
        r[1] > 0.0,
        "the ball ended at "
        + String(r[1])
        + " m — BELOW THE FLOOR. Newton produced no contact force at all;"
        " check the workspace guard in `_newton_solve_env`, which returns"
        " early when the region does not fit and whose bound was a comptime"
        " literal that is 69 on every dynamic provider.",
    )
    assert_true(
        r[2] > 0,
        "no contact at rest — the fixture is not resting on the plane and the"
        " gate would be vacuous",
    )
    print("  PASS")


def test_newton_agrees_with_pgs_on_runtime_dims() raises:
    """Two solvers, one problem, one answer.

    ⚠ AN AGREEMENT, NOT A GOLDEN — see the header. This is what catches a
    Newton that runs but solves the wrong thing, which the fall-through
    assertion above would let pass.
    """
    print("=== Newton agrees with PGS on runtime dims ===")
    var n = _drop["newton"]()
    var p = _drop["pgs"]()
    print("  newton  zmin", n[0], " final", n[1])
    print("  pgs     zmin", p[0], " final", p[1])
    assert_true(
        abs(n[1] - p[1]) < 1e-6 and abs(n[0] - p[0]) < 1e-6,
        "Newton and PGS disagree on the same runtime-dims model: final "
        + String(n[1])
        + " vs "
        + String(p[1])
        + ", zmin "
        + String(n[0])
        + " vs "
        + String(p[0]),
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
