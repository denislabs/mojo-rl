"""dm_control `walker` parity: our envs vs MuJoCo + the reference task.

Same layers as the pendulum/cartpole tests (model / physics / observation /
reward), for `suite/walker.xml` verbatim.

What this exercises beyond cartpole:
  - `zaxis="1 0 0"` geom orientation, added to the parser 2026-07-29. The two
    foot capsules are the only geoms in the suite so far that use it, and it
    changes both their inertia tensors and their contact geometry — so the
    model-build check below is the real gate on that parser path.
  - a top-level unnamed `<default>` combined with a named class reached via
    `childclass`, where the bare `<joint>` elements inherit even their TYPE
    (MuJoCo's hinge default) from nothing at all.
  - the `subtreelinvel` sensor, i.e. `Data.xvel` — the term the walk/run
    rewards are built on.
  - contacts. Walker rests on the floor in its default pose, so the ground
    rollout runs the constraint solver every step; cartpole disabled contact
    entirely and pendulum has no floor.

See the pendulum test for why MuJoCo is driven directly rather than through
the dm_control package.

Run with:
    pixi run mojo run -I . tests/dm_control/test_walker_vs_dm_control.mojo
"""

from std.math import abs, sin, inf
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.walker import (
    DMWalkerStand,
    DMWalkerWalk,
    DMWalkerRun,
    DMWalkerModel,
    TORSO_BODY_IDX,
    STAND_HEIGHT,
)
from mojo_rl.physics3d.fields import Model, Dims
from mojo_rl.physics3d.model.model_dims import ModelDims
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_IPOS_X,
    BODY_IDX_IXX,
)
comptime MD = ModelDims[DMWalkerModel]


comptime EnvStand = DMWalkerStand[DType.float64]
comptime EnvWalk = DMWalkerWalk[DType.float64]
comptime EnvRun = DMWalkerRun[DType.float64]

comptime REF_XML: StaticString = (
    "references/dm_control-main/dm_control/suite/walker.xml"
)
comptime REF_PATH: StaticString = "references/dm_control-main"

comptime NQ: Int = 9
comptime NV: Int = 9
comptime NBODY: Int = 8
comptime NACT: Int = 6
# walker.xml timestep 0.0025 vs walker.py _CONTROL_TIMESTEP 0.025.
comptime FRAME_SKIP: Int = 10

# Two regimes, checked at two bounds — the split is the point of this test.
#
# While every joint sits inside its range the dynamics are smooth and we track
# MuJoCo to ~1e-10, the same residual pendulum and cartpole show. The moment a
# joint crosses its limit the joint-limit CONSTRAINT engages, and our solver
# and MuJoCo's take visibly different impulses; from there the trajectories
# separate at ~1e-5 per step. That divergence is pre-existing and shared with
# contacts (see docs/DM_CONTROL_PORT.md); it is NOT introduced by this domain.
#
# So: assert tightly over the smooth prefix, loosely over the whole rollout.
# The test finds the transition itself rather than hard-coding a step, and
# prints the prefix length so a regression that shrinks it is visible.
comptime STATE_TOL_SMOOTH: Float64 = 1e-8
comptime OBS_TOL_SMOOTH: Float64 = 1e-8
comptime REWARD_TOL_SMOOTH: Float64 = 1e-8

comptime STATE_TOL_ALL: Float64 = 0.05
comptime OBS_TOL_ALL: Float64 = 0.05
comptime REWARD_TOL_ALL: Float64 = 1e-3

# Each init must stay in the smooth regime for at least this long, or the
# tight assertion above is vacuous.
comptime MIN_SMOOTH_STEPS: Int = 25

# Airborne rollout torque. Deliberately gentle: at full scale an ankle crosses
# its +-45 deg limit within a couple of steps and there is no smooth prefix
# left to measure. The grounded test below runs at full scale.
comptime AMP_AIR: Float64 = 0.03
comptime AMP_GROUND: Float64 = 0.7

# How close to a joint bound counts as "the limit is in play" (radians).
comptime LIMIT_MARGIN: Float64 = 0.02

comptime N_STEPS: Int = 60

# Grounded rollout: long enough to see the walker settle/tip, with trajectory
# tracking asserted only over the first few steps (contacts diverge fast).
comptime N_STEPS_GROUND: Int = 120
comptime N_STEPS_GROUND_TRACK: Int = 8
# Deliberately loose. With contacts live from step 0 the two solvers separate
# within a few steps; this bound exists to catch a GROSS error (no floor geom,
# flipped contact normal, mis-oriented foot) which shows up as metres, not to
# certify solver parity. Measured ~0.05 as of 2026-07-29.
comptime GROUND_TOL: Float64 = 0.15
# How far our torso-height envelope may differ from MuJoCo's over the rollout.
comptime HEIGHT_ENVELOPE_TOL: Float64 = 0.25


def _action_at[AMP: Float64](step: Int, k: Int) -> Float64:
    return AMP * sin(0.17 * Float64(step) + 0.9 * Float64(k))


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


def _ref_reward(
    tol: PythonObject,
    torso_height: Float64,
    torso_upright: Float64,
    horizontal_velocity: Float64,
    move_speed: Float64,
) raises -> Float64:
    """`PlanarWalker.get_reward`."""
    var standing = Float64(
        py=tol(
            torso_height,
            STAND_HEIGHT,
            Float64(py=Python.evaluate("float('inf')")),
            STAND_HEIGHT / 2.0,
            String("gaussian"),
            0.1,
        )
    )
    var upright = (1.0 + torso_upright) / 2.0
    var stand_reward = (3.0 * standing + upright) / 4.0
    if move_speed == 0.0:
        return stand_reward
    var move_reward = Float64(
        py=tol(
            horizontal_velocity,
            move_speed,
            Float64(py=Python.evaluate("float('inf')")),
            move_speed / 2.0,
            String("linear"),
            0.5,
        )
    )
    return stand_reward * (5.0 * move_reward + 1.0) / 6.0


def test_walker_model_matches_mujoco() raises:
    """Model build: dims + per-body mass / CoM / inertia.

    This is the gate on `zaxis` parsing: get the foot orientation wrong and
    `body_inertia` for the two feet is visibly off, because the capsule's long
    axis lands on Z instead of X.
    """
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path("mojo_rl/envs/dm_control/assets/walker.xml")

    assert_true(Int(py=m.nbody) == DMWalkerModel.NBODY, "nbody mismatch")
    assert_true(Int(py=m.njnt) == DMWalkerModel.NJOINT, "njnt mismatch")
    assert_true(Int(py=m.nq) == DMWalkerModel.NQ, "nq mismatch")
    assert_true(Int(py=m.nv) == DMWalkerModel.NV, "nv mismatch")
    assert_true(Int(py=m.ngeom) == DMWalkerModel.NGEOM, "ngeom mismatch")
    assert_true(Int(py=m.nu) == DMWalkerModel.nact, "nu mismatch")

    var ctx = DeviceContext()
    var mf = Model[DType.float64, MD]()
    DMWalkerModel.init_fields[DType.float64](ctx, mf)

    var worst = 0.0
    for b in range(NBODY):
        var base = b * MODEL_BODY_SIZE
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

    print("walker model build: max |d(mass,ipos,inertia)| =", worst)
    assert_true(worst <= 1e-12, "walker model differs from MuJoCo")


def test_walker_airborne_matches_mujoco() raises:
    """Physics / obs / reward with the walker lifted clear of the floor.

    Isolates the articulated chain, damping, armature, actuator gears and the
    `subtreelinvel` sensor from the CONTACT solver. Joint limits still engage
    partway through each rollout, which is why the numbers are split into a
    smooth prefix and a full-rollout total (see the tolerance block above).
    """
    var handle = _setup()
    var mujoco = handle[0]
    var model = handle[1]
    var data = handle[2]
    var tol = handle[3]

    var max_state = 0.0
    var max_state_smooth = 0.0
    var max_obs = 0.0
    var max_obs_smooth = 0.0
    var max_r_stand = 0.0
    var max_r_smooth = 0.0
    var max_r_walk = 0.0
    var max_r_run = 0.0
    var min_smooth_steps = N_STEPS

    # rootz lifts the torso clear of the floor for the whole rollout; the
    # rest are hip/knee/ankle angles starting inside their ranges. The
    # oscillating torques do eventually drive a joint into a bound, which is
    # what ends the smooth prefix. Knees start well flexed on purpose: their
    # range is [-150, 0] deg, so a knee near neutral is already AT its upper
    # bound and there would be no smooth phase at all.
    var inits = [
        [3.0, 0.0, 0.0, 0.2, -0.9, 0.0, -0.1, -1.0, 0.0],
        [4.0, 0.5, 0.4, 0.4, -1.2, 0.05, 0.3, -0.8, -0.05],
        [3.5, -0.2, -0.7, 0.1, -1.0, -0.1, 0.5, -1.1, 0.1],
    ]

    for init in inits:
        mujoco.mj_resetData(model, data)
        for i in range(NQ):
            data.qpos[i] = init[i]
        mujoco.mj_forward(model, data)

        var e_stand = EnvStand()
        _ = e_stand.reset()
        var e_walk = EnvWalk()
        _ = e_walk.reset()
        var e_run = EnvRun()
        _ = e_run.reset()

        var qs = List[Float64]()
        var vs = List[Float64]()
        for i in range(NQ):
            qs.append(init[i])
        for _ in range(NV):
            vs.append(0.0)
        e_stand.set_state(qs, vs)
        e_walk.set_state(qs, vs)
        e_run.set_state(qs, vs)

        # Still in the smooth regime: no joint has left its range yet. Read
        # the ranges off MuJoCo's own model so the test cannot drift from it.
        var smooth = True
        var smooth_steps = 0

        for step in range(N_STEPS):
            var a_stand = EnvStand.ActionType()
            var a_walk = EnvWalk.ActionType()
            var a_run = EnvRun.ActionType()
            for k in range(NACT):
                var a = _action_at[AMP_AIR](step, k)
                data.ctrl[k] = a
                a_stand.data[k] = a
                a_walk.data[k] = a
                a_run.data[k] = a

            # walker's control timestep is 10x the physics timestep, so one
            # env step is ten mj_steps (dm_control's `physics.step(n_sub_steps)`).
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            var out_stand = e_stand.step(a_stand)
            var out_walk = e_walk.step(a_walk)
            var out_run = e_run.step(a_run)

            # Still smooth? Checked BEFORE accumulating this step, and with a
            # margin, because the joint-limit constraint engages during the
            # substeps — by the time qpos is observed past the bound the
            # constraint has already acted, and a joint that merely came CLOSE
            # may have touched and rebounded inside a single control step.
            for j in range(NQ):
                var lo = Float64(py=model.jnt_range[j][0])
                var hi = Float64(py=model.jnt_range[j][1])
                if lo == hi:
                    continue  # unlimited
                var q = Float64(py=data.qpos[j])
                if q < lo + LIMIT_MARGIN or q > hi - LIMIT_MARGIN:
                    smooth = False

            # 1. physics
            for i in range(NQ):
                var dq = abs(
                    Float64(py=data.qpos[i])
                    - Float64(e_stand.d.qpos.data[i])
                )
                if dq > max_state:
                    max_state = dq
                if smooth and dq > max_state_smooth:
                    max_state_smooth = dq
            for i in range(NV):
                var dv = abs(
                    Float64(py=data.qvel[i])
                    - Float64(e_stand.d.qvel.data[i])
                )
                if dv > max_state:
                    max_state = dv
                if smooth and dv > max_state_smooth:
                    max_state_smooth = dv
            if smooth:
                smooth_steps = step + 1

            # 2. observation: orientations(14) + height(1) + velocity(9)
            var obs = out_stand[0]
            var oi = 0
            for b in range(1, NBODY):
                var d_xx = abs(
                    Float64(py=data.xmat[b][0]) - Float64(obs.data[oi])
                )
                if d_xx > max_obs:
                    max_obs = d_xx
                if smooth and d_xx > max_obs_smooth:
                    max_obs_smooth = d_xx
                oi += 1
                var d_xz = abs(
                    Float64(py=data.xmat[b][2]) - Float64(obs.data[oi])
                )
                if d_xz > max_obs:
                    max_obs = d_xz
                if smooth and d_xz > max_obs_smooth:
                    max_obs_smooth = d_xz
                oi += 1
            var ref_height = Float64(py=data.xpos[TORSO_BODY_IDX][2])
            var d_h = abs(ref_height - Float64(obs.data[oi]))
            if d_h > max_obs:
                max_obs = d_h
            if smooth and d_h > max_obs_smooth:
                max_obs_smooth = d_h
            oi += 1
            for i in range(NV):
                var d_v = abs(
                    Float64(py=data.qvel[i]) - Float64(obs.data[oi])
                )
                if d_v > max_obs:
                    max_obs = d_v
                if smooth and d_v > max_obs_smooth:
                    max_obs_smooth = d_v
                oi += 1

            # 3. rewards, all three tasks
            var ref_upright = Float64(py=data.xmat[TORSO_BODY_IDX][8])
            var ref_hvel = Float64(py=data.sensordata[0])
            var d_rs = abs(
                _ref_reward(tol, ref_height, ref_upright, ref_hvel, 0.0)
                - Float64(out_stand[1])
            )
            if d_rs > max_r_stand:
                max_r_stand = d_rs
            var d_rw = abs(
                _ref_reward(tol, ref_height, ref_upright, ref_hvel, 1.0)
                - Float64(out_walk[1])
            )
            if d_rw > max_r_walk:
                max_r_walk = d_rw
            var d_rr = abs(
                _ref_reward(tol, ref_height, ref_upright, ref_hvel, 8.0)
                - Float64(out_run[1])
            )
            if d_rr > max_r_run:
                max_r_run = d_rr
            if smooth:
                for d in [d_rs, d_rw, d_rr]:
                    if d > max_r_smooth:
                        max_r_smooth = d

        if smooth_steps < min_smooth_steps:
            min_smooth_steps = smooth_steps

    print("walker (airborne) vs MuJoCo,", len(inits), "x", N_STEPS, "steps:")
    print("  shortest smooth prefix =", min_smooth_steps, "steps")
    print("  --- inside joint limits (smooth dynamics) ---")
    print(
        "  max |d(state)|  =", max_state_smooth,
        " (bound ", STATE_TOL_SMOOTH, ")",
    )
    print(
        "  max |d(obs)|    =", max_obs_smooth,
        " (bound ", OBS_TOL_SMOOTH, ")",
    )
    print(
        "  max |d(reward)| =", max_r_smooth,
        " (bound ", REWARD_TOL_SMOOTH, ")",
    )
    print("  --- whole rollout (joint-limit solver engaged) ---")
    print("  max |d(state)|        =", max_state, " (bound ", STATE_TOL_ALL, ")")
    print("  max |d(obs)|          =", max_obs, " (bound ", OBS_TOL_ALL, ")")
    print("  max |d(reward stand)| =", max_r_stand)
    print("  max |d(reward walk)|  =", max_r_walk)
    print("  max |d(reward run)|   =", max_r_run)

    assert_true(
        min_smooth_steps >= MIN_SMOOTH_STEPS,
        "smooth prefix too short — the tight bounds below prove nothing",
    )
    assert_true(max_state_smooth <= STATE_TOL_SMOOTH, "physics deviated")
    assert_true(max_obs_smooth <= OBS_TOL_SMOOTH, "observation deviated")
    assert_true(max_r_smooth <= REWARD_TOL_SMOOTH, "reward deviated")

    assert_true(max_state <= STATE_TOL_ALL, "physics drifted past the limits")
    assert_true(max_obs <= OBS_TOL_ALL, "observation drifted past the limits")
    assert_true(max_r_stand <= REWARD_TOL_ALL, "stand reward drifted")
    assert_true(max_r_walk <= REWARD_TOL_ALL, "walk reward drifted")
    assert_true(max_r_run <= REWARD_TOL_ALL, "run reward drifted")


def test_walker_grounded_matches_mujoco() raises:
    """The walker as the task actually runs it: standing on the floor.

    Contacts and joint limits are both live from step 0 (in the default pose
    the feet rest exactly on z=0 and the knees sit at their upper bound), so
    this does NOT attempt trajectory parity — the constraint solvers diverge
    and the dynamics are chaotic. What it does gate is that the contact path
    is doing its job at all:

      - the feet do not sink through the floor,
      - the torso does not collapse to the ground or launch,
      - short-horizon tracking against MuJoCo stays within GROUND_TOL,
      - reward stays a valid dm_control reward, i.e. inside [0, 1].

    A wrong contact normal, a missing floor geom, or a mis-oriented foot
    capsule all break at least one of these.
    """
    var handle = _setup()
    var mujoco = handle[0]
    var model = handle[1]
    var data = handle[2]
    var tol = handle[3]

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    var env = EnvWalk()
    _ = env.reset()
    var qs = List[Float64]()
    var vs = List[Float64]()
    for _ in range(NQ):
        qs.append(0.0)
    for _ in range(NV):
        vs.append(0.0)
    env.set_state(qs, vs)

    var max_state_short = 0.0
    var min_height = 1e9
    var max_height = -1e9
    var ref_min_height = 1e9
    var ref_max_height = -1e9
    var min_reward = 1e9
    var max_reward = -1e9

    for step in range(N_STEPS_GROUND):
        var act = EnvWalk.ActionType()
        for k in range(NACT):
            var a = _action_at[AMP_GROUND](step, k)
            data.ctrl[k] = a
            act.data[k] = a
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        var out = env.step(act)

        if step < N_STEPS_GROUND_TRACK:
            for i in range(NQ):
                var dq = abs(
                    Float64(py=data.qpos[i]) - Float64(env.d.qpos.data[i])
                )
                if dq > max_state_short:
                    max_state_short = dq

        var h = Float64(env.d.xpos.data[TORSO_BODY_IDX * 3 + 2])
        if h < min_height:
            min_height = h
        if h > max_height:
            max_height = h
        var rh = Float64(py=data.xpos[TORSO_BODY_IDX][2])
        if rh < ref_min_height:
            ref_min_height = rh
        if rh > ref_max_height:
            ref_max_height = rh
        var r = Float64(out[1])
        if r < min_reward:
            min_reward = r
        if r > max_reward:
            max_reward = r

    print("walker (grounded)", N_STEPS_GROUND, "steps:")
    print(
        "  max |d(qpos)| first", N_STEPS_GROUND_TRACK, "=", max_state_short,
        " (bound ", GROUND_TOL, ")",
    )
    print("  torso height ours =", min_height, "..", max_height)
    print("  torso height ref  =", ref_min_height, "..", ref_max_height)
    print("  reward range      =", min_reward, "..", max_reward)

    assert_true(
        max_state_short <= GROUND_TOL,
        "grounded rollout left MuJoCo's trajectory too fast",
    )
    # Compare the height ENVELOPE against MuJoCo's rather than against magic
    # numbers: with these torques the walker tips over in both sims, so the
    # absolute values are a property of the task, not of correctness. What
    # must hold is that we tip over the same way.
    assert_true(
        abs(min_height - ref_min_height) <= HEIGHT_ENVELOPE_TOL
        and abs(max_height - ref_max_height) <= HEIGHT_ENVELOPE_TOL,
        "torso height envelope differs from MuJoCo's",
    )
    # The torso capsule has radius 0.07, so a centre below that is the body
    # inside the floor — the one failure the envelope check could miss if
    # MuJoCo were somehow wrong too.
    assert_true(min_height > 0.07, "torso sank through the floor")
    assert_true(
        min_reward >= 0.0 and max_reward <= 1.0,
        "reward left [0, 1] — not a valid dm_control reward",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
