"""dm_control `cheetah` parity: our env vs MuJoCo + the reference task.

Same layers as the walker test (model / physics / observation / reward), for
`suite/cheetah.xml` verbatim.

What this exercises beyond walker:
  - `euler="0 <deg> 0"` geom orientation on seven geoms, added to the parser
    2026-07-29. `test_cheetah_geom_quats_match_mujoco` compares our resolved
    geom quaternions against MuJoCo's `model.geom_quat` element by element,
    which is the sharpest possible gate on that path — inertia alone would
    hide a sign or axis-order error, since a capsule's inertia is symmetric
    under some of the wrong answers.
  - `<compiler settotalmass="14">`, which rescales every mass post-hoc.
  - MuJoCo's DEGREE default for `<compiler angle>`: cheetah.xml omits the
    attribute and states its joint ranges in degrees.
  - joint `stiffness` — a passive spring, which none of pendulum, cartpole or
    walker uses.

See the pendulum test for why MuJoCo is driven directly rather than through
the dm_control package.

Run with:
    pixi run mojo run -I . tests/dm_control/test_cheetah_vs_dm_control.mojo
"""

from std.math import abs, sin, inf
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.cheetah import (
    DMCheetahRun,
    DMCheetahModel,
    TORSO_BODY_IDX,
    RUN_SPEED,
)
from mojo_rl.physics3d.fields import Model, Dims
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_IPOS_X,
    BODY_IDX_IXX,
    MODEL_GEOM_SIZE,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
)


comptime Env = DMCheetahRun[DType.float64]

comptime REF_XML: StaticString = (
    "references/dm_control-main/dm_control/suite/cheetah.xml"
)
comptime REF_PATH: StaticString = "references/dm_control-main"

comptime NQ: Int = 9
comptime NV: Int = 9
comptime NBODY: Int = 8
comptime NGEOM: Int = 9
comptime NACT: Int = 6
# cheetah.py passes no control_timestep, so one env step is one physics step.
comptime FRAME_SKIP: Int = 1

# Same two-regime split as walker: tight while every joint is inside its
# range, loose once the joint-limit constraint engages. See that test for the
# rationale.
comptime STATE_TOL_SMOOTH: Float64 = 1e-8
comptime OBS_TOL_SMOOTH: Float64 = 1e-8
comptime REWARD_TOL_SMOOTH: Float64 = 1e-8
comptime STATE_TOL_ALL: Float64 = 1e-6
comptime REWARD_TOL_ALL: Float64 = 1e-6
# Only 5, unlike walker's 25, and that is a property of the model rather than
# a weak test: `fthigh` has range [-57, +0.4] degrees and a 180 N.m/rad
# passive spring pulling it toward 0, so it is driven onto its upper bound
# within about a quarter period (~5 control steps) from any start. There is no
# initial state from which cheetah stays limit-free for long.
comptime MIN_SMOOTH_STEPS: Int = 5
comptime LIMIT_MARGIN: Float64 = 0.02

comptime AMP_AIR: Float64 = 0.05
comptime N_STEPS: Int = 60


def _action_at(step: Int, k: Int) -> Float64:
    return AMP_AIR * sin(0.23 * Float64(step) + 0.7 * Float64(k))


def _setup() raises -> PythonObject:
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var rw = Python.import_module("dm_control.utils.rewards")
    var model = mujoco.MjModel.from_xml_path(String(REF_XML))
    var data = mujoco.MjData(model)
    var tol = Python.evaluate(
        "lambda rw: lambda x, lo, hi, m, s, v: float("
        "rw.tolerance(x, bounds=(lo, hi), margin=m, sigmoid=s,"
        " value_at_margin=v))"
    )(rw)
    return Python.tuple(mujoco, model, data, tol)


def _ref_reward(tol: PythonObject, speed: Float64) raises -> Float64:
    """`Cheetah.get_reward`."""
    return Float64(
        py=tol(
            speed,
            RUN_SPEED,
            Float64(py=Python.evaluate("float('inf')")),
            RUN_SPEED,
            String("linear"),
            0.0,
        )
    )


def _build_model() raises -> Model[DType.float64, Dims[nv=DMCheetahModel.NV, nbody=DMCheetahModel.NBODY, njoint=DMCheetahModel.NJOINT, ngeom=DMCheetahModel.NGEOM, nequality=DMCheetahModel.MAX_EQUALITY, ntendon=DMCheetahModel.MAX_TENDON, nsite=DMCheetahModel.NSITE, nexclude=DMCheetahModel.NEXCLUDE, nmesh_verts=0]]:
    var ctx = DeviceContext()
    var mf = Model[DType.float64, Dims[nv=DMCheetahModel.NV, nbody=DMCheetahModel.NBODY, njoint=DMCheetahModel.NJOINT, ngeom=DMCheetahModel.NGEOM, nequality=DMCheetahModel.MAX_EQUALITY, ntendon=DMCheetahModel.MAX_TENDON, nsite=DMCheetahModel.NSITE, nexclude=DMCheetahModel.NEXCLUDE, nmesh_verts=0]]()
    DMCheetahModel.init_fields[DType.float64, 0](ctx, mf)
    return mf^


def test_cheetah_geom_quats_match_mujoco() raises:
    """Every geom's resolved orientation vs MuJoCo's `model.geom_quat`.

    This is the direct gate on `euler=` parsing. Seven of the nine geoms carry
    `euler="0 <deg> 0"` with angles from +50 to -218 degrees — well outside the
    small-angle regime, which is why the sin/cos helper needed range reduction.

    Quaternions are compared up to sign (q and -q are the same rotation), and
    ours are stored [x,y,z,w] against MuJoCo's [w,x,y,z].
    """
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path("mojo_rl/envs/dm_control/assets/cheetah.xml")
    var mf = _build_model()

    var worst = 0.0
    for g in range(NGEOM):
        var base = g * MODEL_GEOM_SIZE
        var qx = mf.geoms.data[base + GEOM_IDX_QUAT_X]
        var qy = mf.geoms.data[base + GEOM_IDX_QUAT_Y]
        var qz = mf.geoms.data[base + GEOM_IDX_QUAT_Z]
        var qw = mf.geoms.data[base + GEOM_IDX_QUAT_W]
        var rw_ = Float64(py=m.geom_quat[g][0])
        var rx = Float64(py=m.geom_quat[g][1])
        var ry = Float64(py=m.geom_quat[g][2])
        var rz = Float64(py=m.geom_quat[g][3])
        # Align sign via the dot product before differencing.
        var dot = qw * rw_ + qx * rx + qy * ry + qz * rz
        var s = 1.0 if dot >= 0.0 else -1.0
        var d = abs(qw - s * rw_)
        for e in [abs(qx - s * rx), abs(qy - s * ry), abs(qz - s * rz)]:
            if e > d:
                d = e
        print("  geom", g, "max |d(quat)| =", d)
        if d > worst:
            worst = d

    print("cheetah geom quats: worst =", worst)
    assert_true(worst <= 1e-12, "geom orientation differs from MuJoCo")


def test_cheetah_model_matches_mujoco() raises:
    """Dims + per-body mass / CoM / inertia, i.e. euler + settotalmass."""
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path("mojo_rl/envs/dm_control/assets/cheetah.xml")

    assert_true(Int(py=m.nbody) == DMCheetahModel.NBODY, "nbody mismatch")
    assert_true(Int(py=m.njnt) == DMCheetahModel.NJOINT, "njnt mismatch")
    assert_true(Int(py=m.nq) == DMCheetahModel.NQ, "nq mismatch")
    assert_true(Int(py=m.ngeom) == DMCheetahModel.NGEOM, "ngeom mismatch")
    assert_true(Int(py=m.nu) == DMCheetahModel.nact, "nu mismatch")

    var mf = _build_model()
    var worst = 0.0
    var total_mass = 0.0
    for b in range(NBODY):
        var base = b * MODEL_BODY_SIZE
        total_mass += mf.bodies.data[base + BODY_IDX_MASS]
        var dm = abs(
            mf.bodies.data[base + BODY_IDX_MASS] - Float64(py=m.body_mass[b])
        )
        if dm > worst:
            worst = dm
        for k in range(3):
            var dp = abs(
                mf.bodies.data[base + BODY_IDX_IPOS_X + k]
                - Float64(py=m.body_ipos[b][k])
            )
            if dp > worst:
                worst = dp
            var di = abs(
                mf.bodies.data[base + BODY_IDX_IXX + k]
                - Float64(py=m.body_inertia[b][k])
            )
            if di > worst:
                worst = di

    print("cheetah model build: max |d(mass,ipos,inertia)| =", worst)
    print("  total mass =", total_mass, " (settotalmass=14)")
    assert_true(worst <= 1e-12, "cheetah model differs from MuJoCo")
    assert_true(abs(total_mass - 14.0) <= 1e-9, "settotalmass not applied")


def test_cheetah_airborne_matches_mujoco() raises:
    """Physics / obs / reward with the cheetah lifted clear of the ground."""
    var handle = _setup()
    var mujoco = handle[0]
    var model = handle[1]
    var data = handle[2]
    var tol = handle[3]

    var max_state = 0.0
    var max_state_smooth = 0.0
    var max_obs_smooth = 0.0
    var max_r = 0.0
    var max_r_smooth = 0.0
    var min_smooth_steps = N_STEPS

    # qpos = [rootx, rootz, rooty, bthigh, bshin, bfoot, fthigh, fshin, ffoot].
    # rootz lifts the body; leg angles start inside their ranges.
    var inits = [
        [0.0, 2.0, 0.0, 0.2, 0.0, -1.0, -0.3, -0.2, 0.0],
        [0.0, 3.0, 0.3, -0.2, 0.3, -2.0, -0.6, 0.2, 0.2],
        [0.0, 2.5, -0.4, 0.5, -0.4, -1.5, -0.1, -0.5, -0.2],
    ]

    for init in inits:
        mujoco.mj_resetData(model, data)
        for i in range(NQ):
            data.qpos[i] = init[i]
        mujoco.mj_forward(model, data)

        var env = Env()
        _ = env.reset()
        var qs = List[Float64]()
        var vs = List[Float64]()
        for i in range(NQ):
            qs.append(init[i])
        for _ in range(NV):
            vs.append(0.0)
        env.set_state(qs, vs)

        var smooth = True
        var smooth_steps = 0

        for step in range(N_STEPS):
            var act = Env.ActionType()
            for k in range(NACT):
                var a = _action_at(step, k)
                data.ctrl[k] = a
                act.data[k] = a
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            var out = env.step(act)

            # Smooth while no joint is at (or within a margin of) a bound —
            # checked before accumulating, see the walker test.
            for j in range(NQ):
                var lo = Float64(py=model.jnt_range[j][0])
                var hi = Float64(py=model.jnt_range[j][1])
                if lo == hi:
                    continue
                var q = Float64(py=data.qpos[j])
                if q < lo + LIMIT_MARGIN or q > hi - LIMIT_MARGIN:
                    smooth = False

            for i in range(NQ):
                var dq = abs(
                    Float64(py=data.qpos[i]) - Float64(env.d.qpos.data[i])
                )
                if dq > max_state:
                    max_state = dq
                if smooth and dq > max_state_smooth:
                    max_state_smooth = dq
            for i in range(NV):
                var dv = abs(
                    Float64(py=data.qvel[i]) - Float64(env.d.qvel.data[i])
                )
                if dv > max_state:
                    max_state = dv
                if smooth and dv > max_state_smooth:
                    max_state_smooth = dv
            if smooth:
                smooth_steps = step + 1

            # observation: qpos[1:] then qvel
            var obs = out[0]
            var oi = 0
            for i in range(1, NQ):
                var d_o = abs(
                    Float64(py=data.qpos[i]) - Float64(obs.data[oi])
                )
                if smooth and d_o > max_obs_smooth:
                    max_obs_smooth = d_o
                oi += 1
            for i in range(NV):
                var d_o = abs(
                    Float64(py=data.qvel[i]) - Float64(obs.data[oi])
                )
                if smooth and d_o > max_obs_smooth:
                    max_obs_smooth = d_o
                oi += 1

            # reward, off the subtreelinvel sensor
            var d_r = abs(
                _ref_reward(tol, Float64(py=data.sensordata[0]))
                - Float64(out[1])
            )
            if d_r > max_r:
                max_r = d_r
            if smooth and d_r > max_r_smooth:
                max_r_smooth = d_r

        if smooth_steps < min_smooth_steps:
            min_smooth_steps = smooth_steps

    print("cheetah (airborne) vs MuJoCo,", len(inits), "x", N_STEPS, "steps:")
    print("  shortest smooth prefix =", min_smooth_steps, "steps")
    print(
        "  smooth: max |d(state)| =", max_state_smooth,
        " |d(obs)| =", max_obs_smooth, " |d(reward)| =", max_r_smooth,
    )
    print(
        "  all:    max |d(state)| =", max_state, " |d(reward)| =", max_r,
    )

    assert_true(
        min_smooth_steps >= MIN_SMOOTH_STEPS,
        "smooth prefix too short — the tight bounds prove nothing",
    )
    assert_true(max_state_smooth <= STATE_TOL_SMOOTH, "physics deviated")
    assert_true(max_obs_smooth <= OBS_TOL_SMOOTH, "observation deviated")
    assert_true(max_r_smooth <= REWARD_TOL_SMOOTH, "reward deviated")
    assert_true(max_state <= STATE_TOL_ALL, "physics drifted past the limits")
    assert_true(max_r <= REWARD_TOL_ALL, "reward drifted past the limits")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
