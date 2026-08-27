"""The implicit integrators must solve constraints against M, then re-solve.

    pixi run mojo run -I . tests/physics3d/test_implicit_solves_constraints_against_M.mojo

MuJoCo's `mj_step` is two stages (`engine_forward.c:1983` then `:2003`):

    mj_forward   : build AND SOLVE the constraint rows against the PLAIN M
                   -> qfrc_constraint
    mj_implicit  : form M_hat = M - dt*qDeriv and RE-SOLVE
                   qacc = M_hat^-1 (qfrc_smooth + qfrc_constraint)

`ImplicitIntegrator` used to form M_hat FIRST and hand `M_hat^-1` to the
solver, so every constraint row was solved against a mass matrix the reference
never uses.

⚠⚠ IT IS INVISIBLE UNTIL A ROW CARRIES FORCE. With no active rows the two
orderings agree exactly — which is why spot's implicitfast FIRST step matched
MuJoCo to 2.851622 while its fiftieth was 2.4e-05 out. Any gate that only
checked step 1, or only checked a model with nothing in contact, would have
stayed green through all of it.

MEASURED (MuJoCo 3.10.0):

    spot,   qpos[2] after  50 steps : 0.6999373423 -> 0.6999129145
                                      (MuJoCo      0.699912914482164)
    sharpa, qpos[1] after 100 steps : -0.0026270485 -> -0.0026197515
                                      (MuJoCo      -0.002619751486541)

i.e. errors of 2.44e-05 and 7.30e-06 became 2e-16. Across all dofs spot's
50-step worst was 3.631e-02 and is now 2.220e-16.

⚠ NEITHER MODEL HAS A CONTACT AT THE MOMENT IT MATTERS. spot reports ncon 0
at step 50; its live rows are JOINT LIMITS, because `inheritrange` clamps a
commanded 0 into the knee's [-2.793, -0.254] and the servo holds it there.
sharpa_wave has ncon 0 throughout; its live rows are the 22 dof-FRICTION rows
its `frictionloss` declares. "Constraint" here is not a synonym for "contact",
and picking two models that are not touching anything is deliberate.

⚠ `M * qacc_constrained` IS `qfrc_smooth + qfrc_constraint`. The solvers hand
back an ACCELERATION, so multiplying by the same M they were solved against
recovers the force to re-solve with — no solver has to expose
`qfrc_constraint` separately. It must happen before `_msub_qderiv_env` turns
M into M_hat in place, which is why the re-solve block sits where it does.
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
    StudioImpFastEll, StudioImpFastPyr, studio_cone_of,
)
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.types import ConeType

comptime DT = DType.float64
comptime SPOT = String(
    "references/mujoco_menagerie-main/boston_dynamics_spot/scene.xml"
)
comptime SHARPA = String(
    "references/mujoco_menagerie-main/sharpa_wave/scene_left.xml"
)

# ⚠ THE NEGATIVE CONTROL, AND IT IS NOT DECORATION. The re-solve adds a second
# factorization and a second solve to every implicit step; a model with NO
# constraint rows must come out of it completely unchanged, and free fall is
# the one trajectory whose answer is known without a reference: qacc = -g
# exactly, so qvel after one step is -g*dt = -0.01962.
comptime FREEFALL = String(
    "<mujoco><option integrator='implicitfast' timestep='0.002'/>"
    "<worldbody><body pos='0 0 5'><freejoint/>"
    "<geom type='sphere' size='0.1' mass='1'/></body></worldbody></mujoco>"
)


def _run(xml: String, base: String, nstep: Int) raises -> List[Float64]:
    """Step through the runtime path with the actuators driven at ctrl = 0."""
    var fmd = parse_xml_full(expand_mjcf(xml, base), base)
    var verts = 262144
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

    var ell = StudioImpFastEll(dims)
    var pyr = StudioImpFastPyr(dims)
    var is_ell = studio_cone_of(fmd) == ConeType.ELLIPTIC
    var nact = dims.get_nact()
    # ⚠ THE ACTUATORS RUN AT ctrl = 0, WHICH IS NOT THE SAME AS NOT RUNNING
    # THEM. A `<position>` servo still produces `kp*(0 - qpos) - kv*qvel` the
    # moment the joint leaves its reference, and MuJoCo computes
    # `qfrc_actuator` on every step. Omitting it here made sharpa_wave's error
    # read 1.19e-03 when the engine was already at 7.30e-06 — a property of
    # the harness, mistaken for a property of the solver.
    var actions = List[Float64](length=nact if nact > 0 else 1, fill=0.0)
    var act = List[Scalar[DT]](
        length=nact if nact > 0 else 1, fill=Scalar[DT](0)
    )
    for _ in range(nstep):
        if nact > 0:
            for i in range(dims.get_nv()):
                d.qfrc.data[i] = Scalar[DT](0)
            apply_actions_fields[DT](sf, d, actions, act, fmd.timestep)
        if is_ell:
            ell.step["cpu"](d, m)
        else:
            pyr.step["cpu"](d, m)
    var out = List[Float64]()
    for i in range(dims.get_nq()):
        out.append(Float64(d.qpos.data[i]))
    return out^


def test_free_fall_is_untouched_by_the_re_solve() raises:
    """No constraint rows: the answer must still be exactly -g*dt."""
    print("=== free fall under implicitfast (no rows) ===")
    var q = _run(FREEFALL, String(""), 1)
    var z = q[2]
    # qpos starts at 5.0; after one step z = 5 - g*dt*dt.
    var want = 5.0 - 9.81 * 0.002 * 0.002
    print("  z after 1 step", z, " (analytic", want, ")")
    assert_true(
        abs(z - want) < 1e-15,
        "free fall must be exact under implicitfast; got " + String(z)
        + " against the analytic " + String(want)
        + ". The re-solve adds a factorization and a solve to every step and"
        " must be a no-op when nothing constrains the model.",
    )
    print("  PASS")


def test_spot_limit_rows_match_mujoco_at_50_steps() raises:
    """Live JOINT-LIMIT rows, no contacts.

    ⚠ MuJoCo reports ncon 0 here. spot's knees are held against their limit by
    the servo, so the rows that expose the ordering are limits, not contacts.
    """
    print("=== spot, 50 steps: limit rows against the plain M ===")
    var src = read_model_source(SPOT)
    var q = _run(src[0], src[1], 50)
    print("  qpos[2]", q[2], " (MuJoCo 0.699912914482164)")
    assert_true(
        abs(q[2] - 0.699912914482164) < 1e-9,
        "spot's height after 50 steps is " + String(q[2])
        + " against MuJoCo's 0.699912914482164. Solving the limit rows against"
        " M_hat instead of M put this at 0.6999373423 — an error of 2.44e-05,"
        " and 3.63e-02 across all dofs.",
    )
    print("  PASS")


def test_sharpa_friction_rows_match_mujoco_at_100_steps() raises:
    """Live dof-FRICTION rows, no contacts, no limits reached."""
    print("=== sharpa_wave, 100 steps: friction rows against the plain M ===")
    var src = read_model_source(SHARPA)
    var q = _run(src[0], src[1], 100)
    print("  qpos[1]", q[1], " (MuJoCo -0.002619751486541)")
    assert_true(
        abs(q[1] - (-0.002619751486541)) < 1e-9,
        "sharpa_wave's thumb CMC_AA after 100 steps is " + String(q[1])
        + " against MuJoCo's -0.002619751486541. Solving the 22 friction rows"
        " against M_hat instead of M put this at -0.0026270485.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
