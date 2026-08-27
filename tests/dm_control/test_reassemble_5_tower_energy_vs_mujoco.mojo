"""A settled `reassemble_5` tower must not gain energy MuJoCo does not.

⚠⚠ THIS IS THE GATE `PHYSICS3D_CONTACT_FIDELITY_REASSEMBLE5.md` §8 ASKED FOR
AND NOBODY WROTE. Its words: "There is currently no gate anywhere that would
catch a 10,000x energy excess on a settled tower." For weeks there was not —
`test_reassemble_5_bricks_vs_dm_control` is green throughout and gates the
WIRING (build_stack, reset, the per-episode relabeling), not the tower's
behaviour, and `metric.mojo` / `shake12.mojo` are harnesses that nothing runs.

The hunt that closed the symptom found TWO UNRELATED MECHANISMS with the same
signature — a duplicated contact row from an ulp in a quaternion, and a
face-vs-edge axis tie decided by float32 rounding. That is the argument for
gating the SYMPTOM and not only the two mechanisms: a third way of pumping
energy into the tower would otherwise be caught by luck.

WHAT IS MEASURED

Peak kinetic energy of the free bricks, `0.5 v^T M v` restricted to their dofs,
over 200 substeps of ZERO control from a tower that starts at EXACT rest. A
settled tower has none to gain; whatever appears was injected by the solver.

⚠⚠ AGAINST MuJoCo'S ABSOLUTE NUMBER, NOT AGAINST OUR OWN float64. Every metric
in the original hunt — including a purpose-built one — scored float32 against
our float64. Both arms carried the port bug, so the comparison was structurally
blind and reported a healthy 3-6x while both sat four orders from the reference.
The reference leg below runs dm_control itself.

⚠ AND BOTH DTYPES ARE MEASURED. The two mechanisms fixed here hit DIFFERENT
dtypes on different towers — the quaternion one at float64, the axis tie at
float32 — and each tower was clean in the other precision. A single-dtype gate
would have missed one of them.

⚠ THE DISTRIBUTION, NOT THE MEAN. `metric.mojo` averages five towers and
reported `f64 mean 1.4e-06` while four of its five were at 5e-09 and one at
7.2e-06; an outlier and a population averaged together describe neither. This
compares WORST against WORST.

THE TOLERANCE, AND WHY IT IS LOOSE

The towers are not the same towers: MuJoCo draws them with its own RNG and we
with ours, so this compares two distributions, and MuJoCo's own spread is 3x
(3.9e-09 .. 1.19e-08 over five towers). `MAX_RATIO = 20` sits an order above
that spread and more than an order BELOW the smallest historical failure — the
regressions this exists to catch were 600x and 10000x. A tight tolerance here
would be measuring RNG.

⚠ NEVER `pip install dm_control`. Its setup.py wants `mujoco>=3.11.0` and would
drag the runtime up from 3.10.0, silently re-baselining every dm_control gate.
`manipulation_ref._bootstrap()` puts the vendored tree on the path instead.

Run:
    pixi run mojo run -I . tests/dm_control/test_reassemble_5_tower_energy_vs_mujoco.mojo
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.manipulation_reassemble5_def import Reassemble5Model
from mojo_rl.envs.dm_control.manipulation_reassemble5_config import (
    Reassemble5Config,
)
from mojo_rl.envs.dm_control.manipulation_stack_fixed import BRICK_DOF_ADR_0
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.physics3d.gpu.constants import META_IDX_NUM_CONTACTS

comptime E32 = Phyics3dEnv[Reassemble5Model, Reassemble5Config, DType.float32, False]
comptime E64 = Phyics3dEnv[Reassemble5Model, Reassemble5Config, DType.float64, False]
comptime NQ = Reassemble5Model.NQ
comptime NV = Reassemble5Model.NV

comptime TASK = String("reassemble_5_bricks_random_order_features")

# ⚠ 200 SUBSTEPS IS 0.4 s AND THAT IS DELIBERATE — short enough that the
# unactuated arm, which is sagging under gravity and legitimately diverges,
# never reaches the tower. Beyond that the brick energy stops being
# attributable to the contact solve.
comptime N_SUB: Int = 200
comptime N_TOWERS: Int = 6
comptime N_REF_TOWERS: Int = 5
comptime MAX_RATIO: Float64 = 20.0
# A tower whose contact set collapsed is not the fixture this measures.
comptime MIN_CONTACTS: Int = 50


def _dump[DTYPE: DType, //](src: List[Scalar[DTYPE]], mut o: List[Float64]):
    for i in range(len(src)):
        o.append(Float64(src[i]))


def _load_into[DTYPE: DType, //](mut dst: List[Scalar[DTYPE]], src: List[Float64]):
    for i in range(min(len(dst), len(src))):
        dst[i] = Scalar[DTYPE](src[i])


def _brick_ke[DTYPE: DType, //](
    mut env: Phyics3dEnv[Reassemble5Model, Reassemble5Config, DTYPE, False]
) -> Float64:
    """`0.5 v^T M v` over the BRICK dofs only.

    ⚠ THE ARM IS EXCLUDED AND MUST BE. It is unactuated and sagging under
    gravity, so it legitimately gains energy and legitimately diverges
    chaotically; including it buries the tower's signal. Measured during the
    hunt: arm dofs agreed to 3.7e-07 while brick dofs were 80% apart, in the
    same solve.
    """
    ref M = env.integ_euler.scratch.M
    var ke = 0.0
    for i in range(BRICK_DOF_ADR_0, NV):
        var vi = Float64(env.d.qvel.data[i])
        for j in range(BRICK_DOF_ADR_0, NV):
            ke += 0.5 * vi * Float64(M.data[i * NV + j]) * Float64(env.d.qvel.data[j])
    return ke


def _mujoco_peak_brick_ke(mut worst: Float64, mut initial: Float64) raises:
    """dm_control's own peak brick KE from an at-rest tower.

    ⚠ `Physics` IS WEAKREF-BACKED AND `reset()` RECOMPILES, so the model and
    data handles must be RE-FETCHED after every reset or the next attribute
    access raises `ReferenceError` — from the line that READS the model, not
    the one that dropped it.

    ⚠ ZERO `qvel` TOO. Our reset leaves the tower at exact rest, so MuJoCo's
    velocities are zeroed as well or the two runs do not start in the same
    state and the comparison is not about stepping.
    """
    var sys = Python.import_module("sys")
    _ = sys.path.insert(0, "tests/dm_control")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var R = Python.import_module("manipulation_ref")
    _ = R._bootstrap()
    var np = Python.import_module("numpy")

    var env = R._load(TASK)
    worst = 0.0
    initial = 0.0
    for _t in range(N_REF_TOWERS):
        _ = env.reset()
        var phys = env.physics
        var m = phys.model
        var d = phys.data
        d.ctrl[:] = 0.0
        d.qvel[:] = 0.0
        _ = phys.forward()

        # Every FREE joint is a brick; `mjJNT_FREE` is type 0.
        var free = List[Int]()
        var mass = List[Float64]()
        for j in range(Int(py=m.njnt)):
            if Int(py=m.jnt_type[j]) == 0:
                free.append(Int(py=m.jnt_dofadr[j]))
                mass.append(Float64(py=m.body_mass[m.jnt_bodyid[j]]))

        var ke0 = 0.0
        for k in range(len(free)):
            var a = free[k]
            var vv = 0.0
            for c in range(3):
                var x = Float64(py=d.qvel[a + c])
                vv += x * x
            ke0 += 0.5 * mass[k] * vv
        if ke0 > initial:
            initial = ke0

        var peak = 0.0
        for _s in range(N_SUB):
            d.ctrl[:] = 0.0
            _ = phys.step()
            var ke = 0.0
            for k in range(len(free)):
                var a = free[k]
                var vv = 0.0
                for c in range(3):
                    var x = Float64(py=d.qvel[a + c])
                    vv += x * x
                ke += 0.5 * mass[k] * vv
            if ke > peak:
                peak = ke
        print("     MuJoCo tower peak brick KE", peak)
        if peak > worst:
            worst = peak


def test_reassemble_5_tower_energy_matches_mujoco() raises:
    """Worst-tower peak brick KE, ours vs MuJoCo's, at BOTH dtypes."""
    print("=== reassemble_5: a settled tower must not gain energy ===")
    var mj_worst = 0.0
    var mj_initial = 0.0
    _mujoco_peak_brick_ke(mj_worst, mj_initial)
    print("  MuJoCo worst peak brick KE:", mj_worst,
          "  (initial:", mj_initial, ")")

    # NON-VACUITY ON THE REFERENCE. A reference that reports zero would make
    # every ratio below infinite or undefined, and one that starts with energy
    # is not the at-rest tower this measures.
    assert_true(
        mj_worst > 0.0,
        String("the reference reported ZERO peak brick KE — dm_control is not"
               " stepping the tower, so nothing below means anything"),
    )
    assert_true(
        mj_initial == 0.0,
        String("the reference tower does NOT start at rest (initial brick KE ")
        + String(mj_initial)
        + "). Then its peak is an initial transient rather than energy the"
        " solver injected, and so is ours.",
    )

    var ctx = DeviceContext()
    var seed = E32(ctx)
    var worst32 = 0.0
    var worst64 = 0.0
    var min_ncon = 1 << 30
    var max_initial = 0.0

    for _t in range(N_TOWERS):
        # ⚠ ONE DTYPE PER ENV, AND ONE RNG. Both arms are generated by the SAME
        # seed env and transplanted, because stepping two envs in one process
        # makes the second resume the global RNG where the first stopped and
        # the two precisions never see the same tower.
        _ = seed.reset()
        for _ in range(250):
            _ = seed.step(E32.ActionType())
        _ = seed.reset()

        var q = List[Float64]()
        var v = List[Float64]()
        for i in range(NQ):
            q.append(Float64(seed.d.qpos.data[i]))
        for i in range(NV):
            v.append(Float64(seed.d.qvel.data[i]))
        # ⚠ THE MODEL FIELDS TRAVEL TOO. The welded brick has no freejoint, so
        # `build_stack` writes its pose to the MODEL. Replaying `qpos` alone
        # rebuilds the free bricks around a STALE baked tower, which
        # interpenetrates and produces a violent transient in both precisions —
        # an artefact that reads exactly like the bug this gates.
        var bo = List[Float64]()
        var ge = List[Float64]()
        var si = List[Float64]()
        var jo = List[Float64]()
        _dump(seed.mf.bodies.data, bo)
        _dump(seed.mf.geoms.data, ge)
        _dump(seed.mf.sites.data, si)
        _dump(seed.mf.joints.data, jo)

        var f32 = E32(ctx, 1000000, 1)
        var f64 = E64(ctx, 1000000, 1)
        _load_into(f32.mf.bodies.data, bo)
        _load_into(f32.mf.geoms.data, ge)
        _load_into(f32.mf.sites.data, si)
        _load_into(f32.mf.joints.data, jo)
        _load_into(f64.mf.bodies.data, bo)
        _load_into(f64.mf.geoms.data, ge)
        _load_into(f64.mf.sites.data, si)
        _load_into(f64.mf.joints.data, jo)
        f32.set_state(q, v)
        f64.set_state(q, v)

        var i32 = _brick_ke(f32)
        var i64 = _brick_ke(f64)
        if i32 > max_initial:
            max_initial = i32
        if i64 > max_initial:
            max_initial = i64

        var p32 = 0.0
        var p64 = 0.0
        for _s in range(N_SUB):
            _ = f32.step(E32.ActionType())
            _ = f64.step(E64.ActionType())
            var a = _brick_ke(f32)
            if a > p32:
                p32 = a
            var b = _brick_ke(f64)
            if b > p64:
                p64 = b
            var n32 = Int(Float64(f32.d.meta.data[META_IDX_NUM_CONTACTS]))
            if n32 < min_ncon:
                min_ncon = n32
        print("   tower", _t, " f32 peak", p32, " f64 peak", p64)
        if p32 > worst32:
            worst32 = p32
        if p64 > worst64:
            worst64 = p64

    print("  ours worst peak brick KE:  f32", worst32, "  f64", worst64)
    print("  ratio to MuJoCo:  f32", worst32 / mj_worst,
          "  f64", worst64 / mj_worst)
    print("  min contacts over the window:", min_ncon,
          "   worst initial brick KE:", max_initial)

    # NON-VACUITY ON OUR SIDE, both halves. A tower that has fallen apart has
    # no contacts to solve, and one that starts with energy is measuring its
    # own reset rather than the solver.
    assert_true(
        min_ncon >= MIN_CONTACTS,
        String("our tower dropped to ") + String(min_ncon)
        + " contacts during the window — it is not a stacked tower any more,"
        " so a low energy reading proves nothing.",
    )
    assert_true(
        max_initial == 0.0,
        String("our tower does not start at EXACT rest (initial brick KE ")
        + String(max_initial)
        + "). Then the peak below is dominated by a transient we brought with"
        " us and the comparison is not about stepping at all.",
    )

    assert_true(
        worst64 <= MAX_RATIO * mj_worst,
        String("float64: a settled tower gained ") + String(worst64)
        + " J against MuJoCo's " + String(mj_worst) + " ("
        + String(worst64 / mj_worst)
        + "x). A tower under zero control has no energy to gain, so this is"
        " the contact solve injecting it. Two mechanisms have done this"
        " before, both at only one dtype: a duplicated constraint row from an"
        " ulp in a body quaternion, and a box/box face-vs-edge axis tie. See"
        " PHYSICS3D_CONTACT_FIDELITY_REASSEMBLE5.md §3.",
    )
    assert_true(
        worst32 <= MAX_RATIO * mj_worst,
        String("float32: a settled tower gained ") + String(worst32)
        + " J against MuJoCo's " + String(mj_worst) + " ("
        + String(worst32 / mj_worst)
        + "x). GPU training is float32-only, so 'run float64' is not an"
        " available escape here. See the float64 message for the two known"
        " mechanisms.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
