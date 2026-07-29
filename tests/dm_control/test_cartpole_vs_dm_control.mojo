"""dm_control `cartpole` parity: our envs vs MuJoCo + the reference task.

Same three layers as the pendulum test (physics / observation / reward), run
for the 1-pole model against `suite/cartpole.xml` verbatim, plus a model-build
check for the 2- and 3-pole variants that the reference generates with lxml.

What this specifically exercises beyond pendulum:
  - RK4 (cartpole sets integrator="RK4"; pendulum takes MuJoCo's Euler default)
  - a multi-body chain and a limited slide joint
  - BOTH reward forms — sparse (product of zero-margin indicators) and dense
    (four factors, including a `quadratic` sigmoid on the control)
  - MJCF default-class inheritance: every pole joint and geom is defined
    entirely by `<default class="pole">` reached through `childclass`

See the pendulum test for why MuJoCo is driven directly rather than through
the dm_control package.

Run with:
    pixi run mojo run -I . tests/dm_control/test_cartpole_vs_dm_control.mojo
"""

from std.math import abs, sin, pi
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.envs.dm_control.cartpole import (
    DMCartpoleSwingup,
    DMCartpoleSwingupSparse,
    DMCartpole1Model,
    DMCartpole2Model,
    DMCartpole3Model,
    dm_cartpole2_xml,
    dm_cartpole3_xml,
)
from mojo_rl.physics3d.fields import Model
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_IPOS_X,
    BODY_IDX_IXX,
)
from std.gpu.host import DeviceContext


comptime EnvDense = DMCartpoleSwingup[DType.float64]
comptime EnvSparse = DMCartpoleSwingupSparse[DType.float64]

comptime REF_XML: StaticString = (
    "references/dm_control-main/dm_control/suite/cartpole.xml"
)
comptime REF_PATH: StaticString = "references/dm_control-main"

# See the pendulum test for the two-bound rationale and the known ~1e-10
# relative qacc residual. RK4 accumulates a little faster than Euler here.
comptime N_EARLY: Int = 5
comptime STATE_TOL_EARLY: Float64 = 1e-9
comptime STATE_TOL: Float64 = 1e-5
comptime OBS_TOL: Float64 = 1e-5
# Dense reward is smooth, so it tracks the state error. Sparse reward is a
# product of hard indicators and must match exactly away from the boundary.
comptime REWARD_TOL: Float64 = 1e-5

comptime N_STEPS: Int = 100


def _action_at(step: Int) -> Float64:
    return 0.8 * sin(0.21 * Float64(step)) + 0.2 * sin(1.3 * Float64(step))


def _setup() raises -> PythonObject:
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var rw = Python.import_module("dm_control.utils.rewards")
    var model = mujoco.MjModel.from_xml_path(String(REF_XML))
    var data = mujoco.MjData(model)
    # Flat views of the reference reward pieces, to avoid building Python
    # tuples from Mojo (see the rewards parity test).
    var tol = Python.evaluate(
        "lambda rw: lambda x, lo, hi, m, s, v: float("
        "rw.tolerance(x, bounds=(lo, hi), margin=m, sigmoid=s,"
        " value_at_margin=v))"
    )(rw)
    return Python.tuple(mujoco, model, data, tol)


def _ref_sparse_reward(
    tol: PythonObject, cart: Float64, zz: Float64
) raises -> Float64:
    """`Balance._get_reward(sparse=True)` for one pole."""
    var cart_in = Float64(
        py=tol(cart, -0.25, 0.25, 0.0, String("gaussian"), 0.1)
    )
    var angle_in = Float64(
        py=tol(zz, 0.995, 1.0, 0.0, String("gaussian"), 0.1)
    )
    return cart_in * angle_in


def _ref_dense_reward(
    tol: PythonObject,
    cart: Float64,
    zz: Float64,
    ctrl: Float64,
    ang_vel: Float64,
) raises -> Float64:
    """`Balance._get_reward(sparse=False)` for one pole (angular_vel is a
    single element, so `.min()` is that element)."""
    var upright = (zz + 1.0) / 2.0
    var centered = Float64(
        py=tol(cart, 0.0, 0.0, 2.0, String("gaussian"), 0.1)
    )
    centered = (1.0 + centered) / 2.0
    var small_control = Float64(
        py=tol(ctrl, 0.0, 0.0, 1.0, String("quadratic"), 0.0)
    )
    small_control = (4.0 + small_control) / 5.0
    var small_velocity = Float64(
        py=tol(ang_vel, 0.0, 0.0, 5.0, String("gaussian"), 0.1)
    )
    small_velocity = (1.0 + small_velocity) / 2.0
    return upright * small_control * small_velocity * centered


def test_cartpole_1pole_matches_mujoco() raises:
    var handle = _setup()
    var mujoco = handle[0]
    var model = handle[1]
    var data = handle[2]
    var tol = handle[3]

    var max_state = 0.0
    var max_state_early = 0.0
    var max_obs = 0.0
    var max_rd = 0.0
    var max_rs = 0.0

    var inits = [
        (0.0, 0.0, 0.0, 0.0),        # upright, at rest
        (0.0, pi, 0.0, 0.0),         # hanging (swing-up start)
        (0.15, 0.05, 0.0, 0.0),      # near balance, cart off-centre
        (-0.3, 0.6, 0.4, -1.0),      # off-bounds cart, moving
        (0.05, -0.2, -0.2, 2.0),     # generic
    ]

    for init in inits:
        var q0 = init[0]
        var q1 = init[1]
        var v0 = init[2]
        var v1 = init[3]

        mujoco.mj_resetData(model, data)
        data.qpos[0] = q0
        data.qpos[1] = q1
        data.qvel[0] = v0
        data.qvel[1] = v1
        mujoco.mj_forward(model, data)

        var env_d = EnvDense()
        _ = env_d.reset()
        env_d.set_state([q0, q1], [v0, v1])
        var env_s = EnvSparse()
        _ = env_s.reset()
        env_s.set_state([q0, q1], [v0, v1])

        for step in range(N_STEPS):
            var a = _action_at(step)

            data.ctrl[0] = a
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            var act = EnvDense.ActionType()
            act.data[0] = a
            var out_d = env_d.step(act)
            var act2 = EnvSparse.ActionType()
            act2.data[0] = a
            var out_s = env_s.step(act2)

            # 1. physics
            var ref_q0 = Float64(py=data.qpos[0])
            var ref_q1 = Float64(py=data.qpos[1])
            var ref_v0 = Float64(py=data.qvel[0])
            var ref_v1 = Float64(py=data.qvel[1])
            var dq = abs(ref_q0 - Float64(env_d.d.qpos.data[0]))
            var dq1 = abs(ref_q1 - Float64(env_d.d.qpos.data[1]))
            var dv = abs(ref_v0 - Float64(env_d.d.qvel.data[0]))
            var dv1 = abs(ref_v1 - Float64(env_d.d.qvel.data[1]))
            var worst = dq
            if dq1 > worst:
                worst = dq1
            if dv > worst:
                worst = dv
            if dv1 > worst:
                worst = dv1
            if worst > max_state:
                max_state = worst
            if step < N_EARLY and worst > max_state_early:
                max_state_early = worst

            # 2. observation: [cart, zz, xz, qvel0, qvel1]
            var ref_zz = Float64(py=data.xmat[2][8])
            var ref_xz = Float64(py=data.xmat[2][2])
            var obs = out_d[0]
            var od = abs(ref_q0 - Float64(obs.data[0]))
            var o1 = abs(ref_zz - Float64(obs.data[1]))
            var o2 = abs(ref_xz - Float64(obs.data[2]))
            var o3 = abs(ref_v0 - Float64(obs.data[3]))
            var o4 = abs(ref_v1 - Float64(obs.data[4]))
            for o in [od, o1, o2, o3, o4]:
                if o > max_obs:
                    max_obs = o

            # 3. rewards, both forms
            var rd = abs(
                _ref_dense_reward(tol, ref_q0, ref_zz, a, ref_v1)
                - Float64(out_d[1])
            )
            if rd > max_rd:
                max_rd = rd
            var rs = abs(
                _ref_sparse_reward(tol, ref_q0, ref_zz) - Float64(out_s[1])
            )
            if rs > max_rs:
                max_rs = rs

    print("cartpole-1pole vs MuJoCo,", len(inits), "x", N_STEPS, "steps:")
    print(
        "  max |d(state)| first", N_EARLY, "=", max_state_early,
        " (bound ", STATE_TOL_EARLY, ")",
    )
    print("  max |d(state)| all     =", max_state, " (bound ", STATE_TOL, ")")
    print("  max |d(obs)|           =", max_obs, " (bound ", OBS_TOL, ")")
    print("  max |d(reward dense)|  =", max_rd, " (bound ", REWARD_TOL, ")")
    print("  max |d(reward sparse)| =", max_rs, " (bound ", REWARD_TOL, ")")

    assert_true(
        max_state_early <= STATE_TOL_EARLY, "physics deviated early"
    )
    assert_true(max_state <= STATE_TOL, "physics drifted")
    assert_true(max_obs <= OBS_TOL, "observation deviated")
    assert_true(max_rd <= REWARD_TOL, "dense reward deviated")
    assert_true(max_rs <= REWARD_TOL, "sparse reward deviated")


def test_cartpole_multipole_models_match_mujoco() raises:
    """2- and 3-pole variants: dims + per-body inertial properties."""
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")

    var ctx = DeviceContext()
    var worst = 0.0

    # --- 2 poles ---
    var m2 = mujoco.MjModel.from_xml_string(String(dm_cartpole2_xml))
    assert_true(
        Int(py=m2.nbody) == DMCartpole2Model.NBODY,
        "2-pole nbody mismatch",
    )
    assert_true(Int(py=m2.nq) == DMCartpole2Model.NQ, "2-pole nq mismatch")

    var mf2 = Model[
        DType.float64,
        DMCartpole2Model.NV,
        DMCartpole2Model.NBODY,
        DMCartpole2Model.NJOINT,
        DMCartpole2Model.NGEOM,
        DMCartpole2Model.MAX_EQUALITY,
        DMCartpole2Model.MAX_TENDON,
        DMCartpole2Model.NSITE,
        DMCartpole2Model.NEXCLUDE,
        0,
    ]()
    DMCartpole2Model.init_fields[DType.float64, 0](ctx, mf2)
    worst = _cmp_bodies(mf2.bodies.data, m2, DMCartpole2Model.NBODY, worst)

    # --- 3 poles ---
    var m3 = mujoco.MjModel.from_xml_string(String(dm_cartpole3_xml))
    assert_true(
        Int(py=m3.nbody) == DMCartpole3Model.NBODY,
        "3-pole nbody mismatch",
    )
    var mf3 = Model[
        DType.float64,
        DMCartpole3Model.NV,
        DMCartpole3Model.NBODY,
        DMCartpole3Model.NJOINT,
        DMCartpole3Model.NGEOM,
        DMCartpole3Model.MAX_EQUALITY,
        DMCartpole3Model.MAX_TENDON,
        DMCartpole3Model.NSITE,
        DMCartpole3Model.NEXCLUDE,
        0,
    ]()
    DMCartpole3Model.init_fields[DType.float64, 0](ctx, mf3)
    worst = _cmp_bodies(mf3.bodies.data, m3, DMCartpole3Model.NBODY, worst)

    print("cartpole 2/3-pole model build: max |d(mass,ipos,inertia)| =", worst)
    assert_true(worst <= 1e-12, "generated pole chain differs from MuJoCo")


def _cmp_bodies(
    bodies: List[Float64],
    mjmodel: PythonObject,
    nbody: Int,
    running_worst: Float64,
) raises -> Float64:
    """Compare mass / ipos / diagonal inertia for every body."""
    var worst = running_worst
    for b in range(nbody):
        var base = b * MODEL_BODY_SIZE
        var dm = abs(bodies[base + BODY_IDX_MASS] - Float64(py=mjmodel.body_mass[b]))
        if dm > worst:
            worst = dm
        for k in range(3):
            var dp = abs(
                bodies[base + BODY_IDX_IPOS_X + k]
                - Float64(py=mjmodel.body_ipos[b][k])
            )
            if dp > worst:
                worst = dp
            var di = abs(
                bodies[base + BODY_IDX_IXX + k]
                - Float64(py=mjmodel.body_inertia[b][k])
            )
            if di > worst:
                worst = di
    return worst


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
