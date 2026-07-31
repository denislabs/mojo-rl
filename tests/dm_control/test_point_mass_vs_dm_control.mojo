"""dm_control `point_mass-easy` parity: our env vs MuJoCo + the reference task.

Same layers as the other domain tests (model / physics / observation /
reward). MuJoCo is driven from the UNMODIFIED `suite/point_mass.xml`, tendons
and all, and so are we — this port used to substitute two joint motors for the
two identity-coef fixed tendons, and that rewrite is gone now that
`apply_actions` resolves a tendon transmission directly. The numbers below did
not move by one digit when it was removed, which is the substitution's epitaph:
it was exact, and it is no longer needed.

`hard`, which randomizes those coefficients per episode and so cannot be
expressed by any substitution, is gated separately in
`test_point_mass_hard_vs_dm_control.mojo`.

What this exercises beyond the earlier domains:
  - `geom_xpos`, added to `physics3d/kinematics` with this port. The reward is
    a geom-to-geom distance, and both geoms are involved: `target` is
    world-attached (the `body == 0` shortcut) and `pointmass` rides a moving
    body, so a bug in either branch is fatal to the reward.
  - limited SLIDE joints (range +-.29 m). Those ranges must NOT be scaled by
    the degree->radian factor; if they were, the mass would be pinned inside
    +-.005 m and every rollout would look identical.

Run with:
    pixi run mojo run -I . tests/dm_control/test_point_mass_vs_dm_control.mojo
"""

from std.math import abs, sin, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from std.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.point_mass import (
    DMPointMassEasy,
    DMPointMassModel,
    dm_point_mass_xml,
    POINTMASS_GEOM_IDX,
    TARGET_GEOM_IDX,
    TARGET_SIZE,
)
from mojo_rl.physics3d.fields import Model
from mojo_rl.physics3d.kinematics.geom_xpos import geom_xpos
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS,
    MODEL_JOINT_SIZE,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    MODEL_GEOM_SIZE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_Z,
)


comptime Env = DMPointMassEasy[DType.float64]

comptime REF_XML: StaticString = (
    "references/dm_control-main/dm_control/suite/point_mass.xml"
)
comptime REF_PATH: StaticString = "references/dm_control-main"

comptime NQ: Int = 2
comptime NV: Int = 2
comptime NBODY: Int = 2
comptime NGEOM: Int = 7
comptime NACT: Int = 2
# point_mass.py passes no control_timestep, so one env step is one physics step.
comptime FRAME_SKIP: Int = 1

comptime STATE_TOL: Float64 = 1e-9
comptime GEOM_TOL: Float64 = 1e-9
comptime OBS_TOL: Float64 = 1e-9
comptime REWARD_TOL: Float64 = 1e-9

# MuJoCo's own indices for the same two geoms (it sorts by body, we sort by
# XML text order — see `point_mass_xml`). Pinned in the model test.
comptime REF_TARGET_GEOM_IDX: Int = 5
comptime REF_POINTMASS_GEOM_IDX: Int = 6

comptime AMP: Float64 = 0.9
comptime N_STEPS: Int = 150


def _action_at(step: Int, k: Int) -> Float64:
    return AMP * sin(0.11 * Float64(step) + 1.6 * Float64(k))


def _setup() raises -> PythonObject:
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var rw = Python.import_module("dm_control.utils.rewards")
    # The REFERENCE model — fixed tendons and tendon actuators intact.
    var model = mujoco.MjModel.from_xml_path(String(REF_XML))
    var data = mujoco.MjData(model)
    var tol = Python.evaluate(
        "lambda rw: lambda x, lo, hi, m, s, v: float("
        "rw.tolerance(x, bounds=(lo, hi), margin=m, sigmoid=s,"
        " value_at_margin=v))"
    )(rw)
    return Python.tuple(mujoco, model, data, tol)


def _build_model() raises -> Model[
    DType.float64,
    DMPointMassModel.NV,
    DMPointMassModel.NBODY,
    DMPointMassModel.NJOINT,
    DMPointMassModel.NGEOM,
    DMPointMassModel.MAX_EQUALITY,
    DMPointMassModel.MAX_TENDON,
    DMPointMassModel.NSITE,
    DMPointMassModel.NEXCLUDE,
    0,
]:
    var ctx = DeviceContext()
    var mf = Model[
        DType.float64,
        DMPointMassModel.NV,
        DMPointMassModel.NBODY,
        DMPointMassModel.NJOINT,
        DMPointMassModel.NGEOM,
        DMPointMassModel.MAX_EQUALITY,
        DMPointMassModel.MAX_TENDON,
        DMPointMassModel.NSITE,
        DMPointMassModel.NEXCLUDE,
        0,
    ]()
    DMPointMassModel.init_fields[DType.float64, 0](ctx, mf)
    return mf^


def test_point_mass_model_matches_mujoco() raises:
    """Dims, masses, joint ranges, the target radius, and the geom indices.

    The joint-range assertion is the gate on the slide-vs-hinge angle fix: in
    degrees these would come out at +-0.00506 rad instead of +-0.29 m.
    """
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    # Both sides now build the same XML; this reads it from disk so a drift
    # between our embedded copy and the reference file still fails here.
    var m = mujoco.MjModel.from_xml_path(String(REF_XML))

    assert_true(Int(py=m.nbody) == DMPointMassModel.NBODY, "nbody mismatch")
    assert_true(Int(py=m.njnt) == DMPointMassModel.NJOINT, "njnt mismatch")
    assert_true(Int(py=m.nq) == DMPointMassModel.NQ, "nq mismatch")
    assert_true(Int(py=m.ngeom) == DMPointMassModel.NGEOM, "ngeom mismatch")
    assert_true(Int(py=m.nu) == DMPointMassModel.nact, "nu mismatch")

    # MuJoCo sorts geoms by body, our parser by XML text order, so these two
    # models genuinely disagree on the indices — see `point_mass_xml`. Pin
    # BOTH, so a future reordering on either side fails loudly here rather
    # than silently swapping which geom the reward measures.
    var pm_id = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pointmass")
    )
    var tg_id = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target")
    )
    assert_true(pm_id == 6, "MuJoCo's pointmass geom index moved")
    assert_true(tg_id == 5, "MuJoCo's target geom index moved")

    var mf_idx = _build_model()
    assert_true(
        Int(mf_idx.geoms.data[POINTMASS_GEOM_IDX * MODEL_GEOM_SIZE
                              + GEOM_IDX_BODY]) == 1,
        "our pointmass geom index is not the one on the moving body",
    )
    assert_true(
        Int(mf_idx.geoms.data[TARGET_GEOM_IDX * MODEL_GEOM_SIZE
                              + GEOM_IDX_BODY]) == 0,
        "our target geom index is not world-attached",
    )
    assert_true(
        abs(
            Float64(
                mf_idx.geoms.data[TARGET_GEOM_IDX * MODEL_GEOM_SIZE
                                  + GEOM_IDX_POS_Z]
            )
            - 0.01
        ) <= 1e-15,
        "our target geom is not the one at z = .01",
    )

    var ref_size = Float64(py=m.geom_size[tg_id][0])
    assert_true(
        abs(ref_size - TARGET_SIZE) <= 1e-15,
        "TARGET_SIZE is out of sync with point_mass.xml's geom_size",
    )

    var mf = _build_model()
    var worst = 0.0
    for b in range(NBODY):
        var dm = abs(
            mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_MASS]
            - Float64(py=m.body_mass[b])
        )
        if dm > worst:
            worst = dm
    print("point_mass model build: max |d(mass)| =", worst)
    assert_true(worst <= 1e-15, "point_mass masses differ from MuJoCo")

    var worst_range = 0.0
    for j in range(NQ):
        var lo = Float64(
            mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN]
        )
        var hi = Float64(
            mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MAX]
        )
        var dlo = abs(lo - Float64(py=m.jnt_range[j][0]))
        var dhi = abs(hi - Float64(py=m.jnt_range[j][1]))
        if dlo > worst_range:
            worst_range = dlo
        if dhi > worst_range:
            worst_range = dhi
    print("  max |d(jnt_range)| =", worst_range)
    assert_true(
        worst_range <= 1e-15,
        "slide joint ranges differ — degree conversion leaked onto a slide?",
    )


def test_point_mass_matches_mujoco() raises:
    """Physics, geom_xpos, observation and reward against the tendon model."""
    var handle = _setup()
    var mujoco = handle[0]
    var model = handle[1]
    var data = handle[2]
    var tol = handle[3]

    var max_state = 0.0
    var max_geom = 0.0
    var max_obs = 0.0
    var max_r = 0.0
    var min_dist = 1e9
    var max_dist = 0.0
    var hit_limit = False

    # The reward is ~0 beyond about 5 cm (margin = the 1.5 cm target radius),
    # so a run that only ever wanders the far corners exercises `near_target`
    # in its flat tail. The last two inits start on top of the target so the
    # rollout sweeps the peak of the tolerance curve as well.
    var inits = [
        [0.1, -0.05],
        [-0.2, 0.15],
        [0.25, 0.25],
        [0.005, -0.008],
        [-0.02, 0.01],
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

            for i in range(NQ):
                var dq = abs(
                    Float64(py=data.qpos[i]) - Float64(env.d.qpos.data[i])
                )
                if dq > max_state:
                    max_state = dq
                # Track whether the run ever loads a joint limit — the limit
                # constraint is the one place our solver and MuJoCo's are
                # known to differ, so it belongs in the report.
                var lo = Float64(py=model.jnt_range[i][0])
                var hi = Float64(py=model.jnt_range[i][1])
                var q = Float64(py=data.qpos[i])
                if q <= lo + 1e-6 or q >= hi - 1e-6:
                    hit_limit = True
            for i in range(NV):
                var dv = abs(
                    Float64(py=data.qvel[i]) - Float64(env.d.qvel.data[i])
                )
                if dv > max_state:
                    max_state = dv

            # observation: qpos then qvel
            var obs = out[0]
            for i in range(NQ):
                var d_o = abs(
                    Float64(py=data.qpos[i]) - Float64(obs.data[i])
                )
                if d_o > max_obs:
                    max_obs = d_o
            for i in range(NV):
                var d_o = abs(
                    Float64(py=data.qvel[i]) - Float64(obs.data[NQ + i])
                )
                if d_o > max_obs:
                    max_obs = d_o

            # geom_xpos for the two geoms the reward reads
            # Paired by NAME, not by index: (our index, MuJoCo's index).
            var geom_pairs = [
                [POINTMASS_GEOM_IDX, REF_POINTMASS_GEOM_IDX],
                [TARGET_GEOM_IDX, REF_TARGET_GEOM_IDX],
            ]
            for pair in geom_pairs:
                var ours = geom_xpos(env.d, env.mf.geoms.data, pair[0])
                var ref_g = [
                    Float64(py=data.geom_xpos[pair[1]][0]),
                    Float64(py=data.geom_xpos[pair[1]][1]),
                    Float64(py=data.geom_xpos[pair[1]][2]),
                ]
                var mine = [ours[0], ours[1], ours[2]]
                for k in range(3):
                    var dg = abs(ref_g[k] - mine[k])
                    if dg > max_geom:
                        max_geom = dg

            # reward = near_target * small_control
            var dx = Float64(py=data.geom_xpos[REF_TARGET_GEOM_IDX][0]) - Float64(
                py=data.geom_xpos[REF_POINTMASS_GEOM_IDX][0]
            )
            var dy = Float64(py=data.geom_xpos[REF_TARGET_GEOM_IDX][1]) - Float64(
                py=data.geom_xpos[REF_POINTMASS_GEOM_IDX][1]
            )
            var dz = Float64(py=data.geom_xpos[REF_TARGET_GEOM_IDX][2]) - Float64(
                py=data.geom_xpos[REF_POINTMASS_GEOM_IDX][2]
            )
            var dist = sqrt(dx * dx + dy * dy + dz * dz)
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist

            var near_target = Float64(
                py=tol(
                    dist, 0.0, TARGET_SIZE, TARGET_SIZE, String("gaussian"), 0.1
                )
            )
            var acc = 0.0
            for k in range(NACT):
                acc += Float64(
                    py=tol(
                        Float64(py=data.ctrl[k]),
                        0.0,
                        0.0,
                        1.0,
                        String("quadratic"),
                        0.0,
                    )
                )
            var small_control = (acc / Float64(NACT) + 4.0) / 5.0
            var d_r = abs(near_target * small_control - Float64(out[1]))
            if d_r > max_r:
                max_r = d_r

    print("point_mass vs MuJoCo,", len(inits), "x", N_STEPS, "steps:")
    print("  max |d(state)| =", max_state, " |d(geom_xpos)| =", max_geom)
    print("  max |d(obs)| =", max_obs, " |d(reward)| =", max_r)
    print("  mass-to-target range =", min_dist, "..", max_dist)
    print("  loaded a joint limit:", hit_limit)

    assert_true(max_state <= STATE_TOL, "physics deviated")
    assert_true(max_geom <= GEOM_TOL, "geom_xpos deviated")
    assert_true(max_obs <= OBS_TOL, "observation deviated")
    assert_true(max_r <= REWARD_TOL, "reward deviated")
    # The reward is flat 0 outside a 3 cm ball, so a rollout that never gets
    # near the target would gate almost nothing.
    assert_true(
        min_dist < 4.0 * TARGET_SIZE,
        "no rollout approached the target — the reward gate is nearly vacuous",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
