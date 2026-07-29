"""dm_control `pendulum-swingup` parity: our env vs MuJoCo + the reference task.

Three layers are gated, innermost first:

  1. PHYSICS — (qpos, qvel) trajectory vs MuJoCo stepping the SAME
     `pendulum.xml`, from identical initial states under an identical action
     sequence. This is the real engine test.
  2. OBSERVATION — our `custom_extract_obs_cpu` vs MuJoCo's own
     `data.xmat[pole]` columns zz/xz plus qvel, i.e. `SwingUp.get_observation`.
     Catches quaternion-convention and column-index mistakes in `xmat_elem`.
  3. REWARD — ours vs the reference `rewards.tolerance` fed MuJoCo's xmat_zz,
     i.e. `SwingUp.get_reward`.

Why MuJoCo directly and not the dm_control package: dm-control 1.0.41 needs
mujoco >= 3.11 (its bindings reference `mjModel.flex_bandwidth`) and
conda-forge tops out at 3.10, so importing `dm_control.suite` fails. The XML
and `rewards.py` are consumed straight from `references/dm_control-main`
instead — `rewards.py` is pure numpy — and the ~6 lines of task glue
(which xmat columns, which bounds) are transcribed here from
`suite/pendulum.py`. See docs/DM_CONTROL_PORT.md, Stage 0.

READ POINT: after `mj_step`, MuJoCo's `data.xmat` still reflects the
PRE-integration qpos (xmat is computed in mj_step1), which is also what our
engine does by default and what Gymnasium is calibrated against. dm_control
instead syncs derived fields to the integrated state before the task reads
them, so the port sets `SYNC_FK_AFTER_STEP = True` and this test calls
`mj_forward` after each `mj_step`. Without both, orientation-based
observations and rewards are silently one control step stale.
qpos/qvel need no such treatment.

Run with:
    pixi run mojo run -I . tests/dm_control/test_pendulum_vs_dm_control.mojo
"""

from std.math import abs, sin, pi
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.envs.dm_control.pendulum import (
    DMPendulum,
    DMPendulumModel,
    COSINE_BOUND,
)


comptime Env = DMPendulum[DType.float64]

comptime REF_XML: StaticString = (
    "references/dm_control-main/dm_control/suite/pendulum.xml"
)
comptime REF_PATH: StaticString = "references/dm_control-main"

# Two bounds, because a single one cannot distinguish "our integrator is
# wrong" from "small seed error amplified over a 2-second rollout".
#   EARLY: the first few steps, where nothing has accumulated. A genuine bug
#          shows up here at 1e-2..1e0 — the inertiafromgeom default bug was
#          5e-1 at step 1, so this still catches real errors by 7+ orders.
#   LATE:  the full 100 steps, bounding the drift rate.
#
# KNOWN RESIDUAL (unresolved, tracked in docs/DM_CONTROL_PORT.md): our qacc
# differs from MuJoCo's by ~1e-10 RELATIVE at step 0, which then grows roughly
# linearly. Ruled out as causes: model constants (mass/ipos/inertia agree to
# 3e-19 absolute) and joint damping (the seed error is unchanged with
# damping="0"). Remaining leads are the RNE bias-force accumulation and the
# constraint solve running when no constraint is active. It is far below
# anything dynamically meaningful — reward matches EXACTLY over all 600 steps
# tested — but it is NOT float64 rounding, so it should be explained before
# porting domains with stiffer dynamics.
comptime N_EARLY: Int = 5
comptime STATE_TOL_EARLY: Float64 = 1e-9
comptime STATE_TOL: Float64 = 1e-6
comptime OBS_TOL: Float64 = 1e-6
# Reward is a hard 0/1 indicator, so it must match EXACTLY except within a
# hair of the 8-degree boundary; the trajectories tested stay clear of it.
comptime REWARD_TOL: Float64 = 0.0

comptime N_STEPS: Int = 100


def _action_at(step: Int) -> Float64:
    """Deterministic, non-trivial torque sequence in [-1, 1]. Both engines
    consume exactly these values, so the comparison never depends on RNG."""
    return 0.9 * sin(0.3 * Float64(step)) + 0.1 * sin(1.7 * Float64(step))


def _mujoco_setup() raises -> PythonObject:
    """`(model, data, rewards_module)` for the reference pendulum."""
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var rw = Python.import_module("dm_control.utils.rewards")
    var model = mujoco.MjModel.from_xml_path(String(REF_XML))
    var data = mujoco.MjData(model)
    return Python.tuple(mujoco, model, data, rw)


def _run_one_trajectory(
    handle: PythonObject,
    init_angle: Float64,
    init_vel: Float64,
    mut max_state: Float64,
    mut max_state_early: Float64,
    mut max_obs: Float64,
    mut max_reward: Float64,
) raises:
    """Step both engines in lockstep from the same initial state."""
    var mujoco = handle[0]
    var model = handle[1]
    var data = handle[2]
    var rw = handle[3]

    # --- reference: reset to the requested state ---
    mujoco.mj_resetData(model, data)
    data.qpos[0] = init_angle
    data.qvel[0] = init_vel
    mujoco.mj_forward(model, data)

    # --- ours: same state ---
    var env = Env()
    _ = env.reset()
    env.set_state([init_angle], [init_vel])

    for step in range(N_STEPS):
        var a = _action_at(step)

        # Reference step: set control, integrate, then refresh derived
        # quantities so xmat matches the post-integration qpos.
        data.ctrl[0] = a
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        # Ours.
        var act = Env.ActionType()
        act.data[0] = a
        var out = env.step(act)
        var obs = out[0]
        var reward = Float64(out[1])

        # 1. physics state
        var ref_q = Float64(py=data.qpos[0])
        var ref_v = Float64(py=data.qvel[0])
        var our_q = Float64(env.d.qpos.data[0])
        var our_v = Float64(env.d.qvel.data[0])
        var dq = abs(ref_q - our_q)
        var dv = abs(ref_v - our_v)
        if dq > max_state:
            max_state = dq
        if dv > max_state:
            max_state = dv

        # 2. observation — `data.xmat` is shaped (nbody, 9), row-major within
        #    a body: zz = column 8, xz = column 2. Body 1 is "pole".
        var ref_zz = Float64(py=data.xmat[1][8])
        var ref_xz = Float64(py=data.xmat[1][2])
        var d0 = abs(ref_zz - Float64(obs.data[0]))
        var d1 = abs(ref_xz - Float64(obs.data[1]))
        var d2 = abs(ref_v - Float64(obs.data[2]))
        if d0 > max_obs:
            max_obs = d0
        if d1 > max_obs:
            max_obs = d1
        if d2 > max_obs:
            max_obs = d2

        # 3. reward — SwingUp.get_reward: tolerance(zz, (COSINE_BOUND, 1)).
        var ref_reward = Float64(
            py=rw.tolerance(ref_zz, bounds=Python.tuple(COSINE_BOUND, 1.0))
        )
        var dr = abs(ref_reward - reward)
        if dr > max_reward:
            max_reward = dr
        if dr > REWARD_TOL:
            print(
                "  reward mismatch @step", step,
                " zz=", ref_zz, " ours=", reward, " ref=", ref_reward,
            )

        # Early steps carry essentially no accumulated error, so they pin the
        # integrator itself; later steps only bound the drift rate.
        if step < N_EARLY:
            if dq > max_state_early:
                max_state_early = dq
            if dv > max_state_early:
                max_state_early = dv


def test_pendulum_matches_mujoco_and_reference_task() raises:
    var handle = _mujoco_setup()

    var max_state = 0.0
    var max_state_early = 0.0
    var max_obs = 0.0
    var max_reward = 0.0

    # Initial angles spanning hanging-down, sideways, and near-upright, with
    # and without initial spin — the swing-up task visits all of these.
    var inits = [
        (0.0, 0.0),        # upright, at rest (reward 1 region)
        (pi, 0.0),         # hanging down, at rest
        (0.5 * pi, 0.0),   # horizontal
        (-0.5 * pi, 2.0),  # horizontal with spin
        (0.1, -3.0),       # near-upright, fast
        (2.5, 1.25),       # generic
    ]
    for init in inits:
        _run_one_trajectory(
            handle,
            init[0],
            init[1],
            max_state,
            max_state_early,
            max_obs,
            max_reward,
        )

    print(
        "pendulum vs MuJoCo over ", len(inits), " x ", N_STEPS, " steps:",
    )
    print(
        "  max |d(qpos,qvel)| first", N_EARLY, "steps =", max_state_early,
        " (bound ", STATE_TOL_EARLY, ")",
    )
    print("  max |d(qpos,qvel)| all steps  =", max_state, " (bound ", STATE_TOL, ")")
    print("  max |d(obs)|       =", max_obs, " (bound ", OBS_TOL, ")")
    print("  max |d(reward)|    =", max_reward, " (bound ", REWARD_TOL, ")")

    assert_true(
        max_state_early <= STATE_TOL_EARLY,
        "physics deviated from MuJoCo in the first steps",
    )
    assert_true(max_state <= STATE_TOL, "physics drifted from MuJoCo")
    assert_true(max_obs <= OBS_TOL, "observation deviated from the reference")
    assert_true(max_reward <= REWARD_TOL, "reward deviated from the reference")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
