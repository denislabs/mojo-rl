"""dm_control `acrobot` parity: our env vs MuJoCo + the reference task.

Same layers as the other domain tests (model / physics / observation /
reward), for `suite/acrobot.xml` verbatim.

What this exercises beyond the earlier domains:
  - `<flag constraint="disable"/>`, added to the parser alongside this port.
    It is not cosmetic: the lower arm sweeps a metre BELOW the floor plane, so
    with the constraint solver live the swing-up dynamics are wrong.
  - `site_xpos`, which drives the reward here (tip-to-target distance). No
    previously ported domain read a site position.
  - a two-link chain under RK4 with NO joint limits and NO contacts, so unlike
    walker and cheetah there is no limit regime to split on. The split here is
    on HORIZON instead: an unactuated double pendulum is chaotic, so the
    engine's known ~4e-11 relative residual on qacc (see the note in
    docs/DM_CONTROL_PORT.md — it is pre-existing and shared by every domain)
    grows by roughly three decades over 200 steps. The short gate is the one
    that would catch a real port bug; the long gate only bounds the growth.

See the pendulum test for why MuJoCo is driven directly rather than through
the dm_control package.

Run with:
    pixi run mojo run -I . tests/dm_control/test_acrobot_vs_dm_control.mojo
"""

from std.math import abs, sin, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.acrobot import (
    DMAcrobotSwingup,
    DMAcrobotSwingupSparse,
    DMAcrobotModel,
    TARGET_RADIUS,
    UPPER_ARM_BODY_IDX,
    LOWER_ARM_BODY_IDX,
    TARGET_SITE_IDX,
    TIP_SITE_IDX,
)
from mojo_rl.physics3d.fields import Model, Dims
from mojo_rl.physics3d.model.model_dims import ModelDims
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_IPOS_X,
    BODY_IDX_IXX,
    MODEL_GEOM_SIZE,
    GEOM_IDX_CONTYPE,
    GEOM_IDX_CONAFFINITY,
)
comptime MD = ModelDims[DMAcrobotModel]


comptime Env = DMAcrobotSwingup[DType.float64]
comptime EnvSparse = DMAcrobotSwingupSparse[DType.float64]

comptime REF_XML: StaticString = (
    "references/dm_control-main/dm_control/suite/acrobot.xml"
)
comptime REF_PATH: StaticString = "references/dm_control-main"

comptime NQ: Int = 2
comptime NV: Int = 2
comptime NBODY: Int = 3
comptime NGEOM: Int = 4
comptime NSITE: Int = 2
comptime NACT: Int = 1
# acrobot.py passes no control_timestep, so one env step is one physics step.
comptime FRAME_SKIP: Int = 1

# No limits and no contacts, so nothing switches regime — the split is purely
# on horizon, because the chaotic dynamics amplify the engine's residual.
# Measured on 2026-07-29: short 8.1e-10, full 1.7e-8. Bounds are ~5x those.
comptime N_SHORT: Int = 25
comptime STATE_TOL_SHORT: Float64 = 5e-9
comptime OBS_TOL_SHORT: Float64 = 5e-9
comptime SITE_TOL_SHORT: Float64 = 2e-9
comptime STATE_TOL: Float64 = 1e-7
comptime SITE_TOL: Float64 = 1e-8
comptime OBS_TOL: Float64 = 1e-7
comptime REWARD_TOL: Float64 = 1e-9

comptime AMP: Float64 = 0.8
comptime N_STEPS: Int = 200


def _action_at(step: Int) -> Float64:
    return AMP * sin(0.19 * Float64(step))


def _setup() raises -> PythonObject:
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var rw = Python.import_module("dm_control.utils.rewards")
    var model = mujoco.MjModel.from_xml_path(String(REF_XML))
    var data = mujoco.MjData(model)
    var tol = Python.evaluate(
        "lambda rw: lambda x, lo, hi, m: float("
        "rw.tolerance(x, bounds=(lo, hi), margin=m))"
    )(rw)
    return Python.tuple(mujoco, model, data, tol)


def _build_model() raises -> Model[DType.float64, MD]:
    var ctx = DeviceContext()
    var mf = Model[DType.float64, MD]()
    DMAcrobotModel.init_fields[DType.float64](ctx, mf)
    return mf^


def test_acrobot_model_matches_mujoco() raises:
    """Dims, per-body mass / CoM / inertia, the target radius, and the
    constraint flag.

    `TARGET_RADIUS` is a hand-carried constant (site sizes are render-only in
    our model records), so it is asserted against `model.site_size` here —
    otherwise a change to acrobot.xml would silently desync the reward.
    """
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path("mojo_rl/envs/dm_control/assets/acrobot.xml")

    assert_true(Int(py=m.nbody) == DMAcrobotModel.NBODY, "nbody mismatch")
    assert_true(Int(py=m.njnt) == DMAcrobotModel.NJOINT, "njnt mismatch")
    assert_true(Int(py=m.nq) == DMAcrobotModel.NQ, "nq mismatch")
    assert_true(Int(py=m.ngeom) == DMAcrobotModel.NGEOM, "ngeom mismatch")
    assert_true(Int(py=m.nsite) == DMAcrobotModel.NSITE, "nsite mismatch")
    assert_true(Int(py=m.nu) == DMAcrobotModel.nact, "nu mismatch")

    # Our site indices are asserted against MuJoCo's name table rather than
    # assumed from DFS order.
    var target_id = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target")
    )
    var tip_id = Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip"))
    assert_true(target_id == TARGET_SITE_IDX, "target site index mismatch")
    assert_true(tip_id == TIP_SITE_IDX, "tip site index mismatch")

    var upper_id = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "upper_arm")
    )
    var lower_id = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "lower_arm")
    )
    assert_true(upper_id == UPPER_ARM_BODY_IDX, "upper_arm index mismatch")
    assert_true(lower_id == LOWER_ARM_BODY_IDX, "lower_arm index mismatch")

    var ref_radius = Float64(py=m.site_size[target_id][0])
    assert_true(
        abs(ref_radius - TARGET_RADIUS) <= 1e-15,
        "TARGET_RADIUS is out of sync with acrobot.xml's site_size",
    )

    var mf = _build_model()
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

    print("acrobot model build: max |d(mass,ipos,inertia)| =", worst)
    assert_true(worst <= 1e-12, "acrobot model differs from MuJoCo")

    # `<flag constraint="disable"/>` must have cleared every collision mask,
    # which is how our engine expresses "no constraint rows".
    for g in range(NGEOM):
        var base = g * MODEL_GEOM_SIZE
        assert_true(
            mf.geoms.data[base + GEOM_IDX_CONTYPE] == 0
            and mf.geoms.data[base + GEOM_IDX_CONAFFINITY] == 0,
            "constraint=disable did not clear the collision mask",
        )


def test_acrobot_matches_mujoco() raises:
    """Physics, site positions, observation and both rewards.

    The arms are started from several angles including ones that put the lower
    arm through the floor plane, which is exactly where a mishandled
    `constraint="disable"` would show up.
    """
    var handle = _setup()
    var mujoco = handle[0]
    var model = handle[1]
    var data = handle[2]
    var tol = handle[3]

    var max_state = 0.0
    var max_site = 0.0
    var max_obs = 0.0
    var max_state_short = 0.0
    var max_site_short = 0.0
    var max_obs_short = 0.0
    var max_r_dense = 0.0
    var max_r_sparse = 0.0
    var min_dist = 1e9
    var max_dist = 0.0

    # (shoulder, elbow) — the last two swing the lower arm below the floor.
    var inits = [
        [0.3, -0.5],
        [2.4, 1.7],
        [3.0, -2.8],
        [-1.2, 2.9],
    ]

    for init in inits:
        mujoco.mj_resetData(model, data)
        for i in range(NQ):
            data.qpos[i] = init[i]
        mujoco.mj_forward(model, data)

        var env = Env()
        var env_sparse = EnvSparse()
        _ = env.reset()
        _ = env_sparse.reset()
        var qs = List[Float64]()
        var vs = List[Float64]()
        for i in range(NQ):
            qs.append(init[i])
        for _ in range(NV):
            vs.append(0.0)
        env.set_state(qs, vs)
        env_sparse.set_state(qs, vs)

        for step in range(N_STEPS):
            var a = _action_at(step)
            var act = Env.ActionType()
            act.data[0] = a
            var act_s = EnvSparse.ActionType()
            act_s.data[0] = a
            data.ctrl[0] = a

            for _ in range(FRAME_SKIP):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            var out = env.step(act)
            var out_s = env_sparse.step(act_s)

            var short = step < N_SHORT

            for i in range(NQ):
                var dq = abs(
                    Float64(py=data.qpos[i]) - Float64(env.d.qpos.data[i])
                )
                if dq > max_state:
                    max_state = dq
                if short and dq > max_state_short:
                    max_state_short = dq
            for i in range(NV):
                var dv = abs(
                    Float64(py=data.qvel[i]) - Float64(env.d.qvel.data[i])
                )
                if dv > max_state:
                    max_state = dv
                if short and dv > max_state_short:
                    max_state_short = dv

            # site_xpos — the reward's only input
            for s in range(NSITE):
                for k in range(3):
                    var ds = abs(
                        Float64(py=data.site_xpos[s][k])
                        - Float64(env.d.site_xpos.data[s * 3 + k])
                    )
                    if ds > max_site:
                        max_site = ds
                    if short and ds > max_site_short:
                        max_site_short = ds

            # observation: (xz upper, xz lower, zz upper, zz lower, qvel x2)
            var obs = out[0]
            var ref_obs = [
                Float64(py=data.xmat[UPPER_ARM_BODY_IDX][2]),
                Float64(py=data.xmat[LOWER_ARM_BODY_IDX][2]),
                Float64(py=data.xmat[UPPER_ARM_BODY_IDX][8]),
                Float64(py=data.xmat[LOWER_ARM_BODY_IDX][8]),
                Float64(py=data.qvel[0]),
                Float64(py=data.qvel[1]),
            ]
            for i in range(6):
                var d_o = abs(ref_obs[i] - Float64(obs.data[i]))
                if d_o > max_obs:
                    max_obs = d_o
                if short and d_o > max_obs_short:
                    max_obs_short = d_o

            # reward — `Physics.to_target` then `rewards.tolerance`
            var dx = Float64(py=data.site_xpos[TARGET_SITE_IDX][0]) - Float64(
                py=data.site_xpos[TIP_SITE_IDX][0]
            )
            var dy = Float64(py=data.site_xpos[TARGET_SITE_IDX][1]) - Float64(
                py=data.site_xpos[TIP_SITE_IDX][1]
            )
            var dz = Float64(py=data.site_xpos[TARGET_SITE_IDX][2]) - Float64(
                py=data.site_xpos[TIP_SITE_IDX][2]
            )
            var dist = sqrt(dx * dx + dy * dy + dz * dz)
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist

            var d_rd = abs(
                Float64(py=tol(dist, 0.0, TARGET_RADIUS, 1.0))
                - Float64(out[1])
            )
            if d_rd > max_r_dense:
                max_r_dense = d_rd
            var d_rs = abs(
                Float64(py=tol(dist, 0.0, TARGET_RADIUS, 0.0))
                - Float64(out_s[1])
            )
            if d_rs > max_r_sparse:
                max_r_sparse = d_rs

    print("acrobot vs MuJoCo,", len(inits), "x", N_STEPS, "steps:")
    print(
        "  first", N_SHORT, "steps: max |d(state)| =", max_state_short,
        " |d(site_xpos)| =", max_site_short, " |d(obs)| =", max_obs_short,
    )
    print("  all:    max |d(state)| =", max_state, " |d(site_xpos)| =", max_site)
    print("          max |d(obs)| =", max_obs)
    print(
        "  max |d(reward)| dense =", max_r_dense,
        " sparse =", max_r_sparse,
    )
    print("  tip-to-target range =", min_dist, "..", max_dist)

    assert_true(max_state_short <= STATE_TOL_SHORT, "physics deviated early")
    assert_true(max_site_short <= SITE_TOL_SHORT, "site_xpos deviated early")
    assert_true(max_obs_short <= OBS_TOL_SHORT, "observation deviated early")
    assert_true(max_state <= STATE_TOL, "physics deviated")
    assert_true(max_site <= SITE_TOL, "site_xpos deviated")
    assert_true(max_obs <= OBS_TOL, "observation deviated")
    assert_true(max_r_dense <= REWARD_TOL, "dense reward deviated")
    assert_true(max_r_sparse <= REWARD_TOL, "sparse reward deviated")
    # The rollouts must actually reach the target sphere, or the sparse gate
    # is comparing 0 against 0 for 800 steps and proves nothing.
    assert_true(
        min_dist < TARGET_RADIUS,
        "no rollout reached the target — the sparse reward gate is vacuous",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
