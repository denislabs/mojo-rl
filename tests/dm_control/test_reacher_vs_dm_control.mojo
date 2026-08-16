"""dm_control `reacher` parity (easy + hard): our env vs MuJoCo + the reference.

The same four layers as the other domain tests (model / physics / observation
/ reward), plus one job unique to this domain: proving the MOCAP-BODY
SUBSTITUTION.

dm_control moves the target by writing `model.geom_pos['target']` at every
reset. Our `fields.Model` is shared and unbatched, so we park the target on a
mocap body and write per-env `d.mocap_pos` instead (see `reacher_xml`). This
test drives MuJoCo from the UNMODIFIED `suite/reacher.xml` and applies the
reference's own model write, while our env gets the same coordinate through
the mocap path. Any disagreement between the two mechanisms — a target left at
its XML position, an FK pass that overwrites the mocap pose, a stale
`geom_xpos` — shows up immediately as a `to_target` divergence.

Also gated here:
  - Both branches of `randomize_limited_and_rotational_joints`. reacher is the
    first ported domain with one limited joint (`wrist`, +-160 deg) and one
    unlimited one (`shoulder`), so the reset test covers the full-circle
    branch and the range branch at once.
  - The sparse reward's radius. `easy` and `hard` differ ONLY in
    `TARGET_SIZE`, which we carry as a config comptime rather than the
    reference's per-episode `geom_size` write; the rollout runs both and
    checks each against dm_control's own `rewards.tolerance` at the same
    distance.

Run with:
    pixi run mojo run -I . tests/dm_control/test_reacher_vs_dm_control.mojo
"""

from std.math import abs, sin, cos, sqrt, pi
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.reacher import (
    DMReacherEasy,
    DMReacherHard,
    DMReacherModel,
    DMReacherHardModel,
    DMReacherConfig,
    FINGER_BODY_IDX,
    TARGET_BODY_IDX,
    FINGER_GEOM_IDX,
    TARGET_GEOM_IDX,
    FINGER_SIZE,
    TARGET_Z,
)
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.physics3d.model import ModelDefLike
from mojo_rl.physics3d.fields import Model, Dims
from mojo_rl.physics3d.kinematics.geom_xpos import geom_xpos
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_MOCAP,
    MODEL_JOINT_SIZE,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    MODEL_GEOM_SIZE,
    GEOM_IDX_BODY,
)


comptime REF_XML: StaticString = (
    "references/dm_control-main/dm_control/suite/reacher.xml"
)
comptime REF_PATH: StaticString = "references/dm_control-main"

comptime NQ: Int = 2
comptime NV: Int = 2
comptime NACT: Int = 2
comptime NGEOM: Int = 10
# reacher.py passes no control_timestep, so one env step is one physics step.
comptime FRAME_SKIP: Int = 1

# Our model carries ONE MORE BODY than the reference: the mocap target. The
# arm chain is declared first so bodies 1..3 line up with MuJoCo's.
comptime REF_NBODY: Int = 4
comptime NBODY_SHARED: Int = 4  # world + arm + hand + finger

comptime STATE_TOL: Float64 = 1e-9
comptime GEOM_TOL: Float64 = 1e-9
comptime OBS_TOL: Float64 = 1e-9
comptime REWARD_TOL: Float64 = 1e-12

# MuJoCo's indices for the two geoms the task reads. It sorts geoms by body id
# and `target` is world-attached, so it lands AHEAD of the arm chain; our
# parser numbers in XML text order and puts it last. Both are pinned below.
comptime REF_TARGET_GEOM_IDX: Int = 6
comptime REF_FINGER_GEOM_IDX: Int = 9

comptime BIG_TARGET: Float64 = 0.05  # _BIG_TARGET  (easy)
comptime SMALL_TARGET: Float64 = 0.015  # _SMALL_TARGET (hard)

comptime AMP: Float64 = 0.8
comptime N_STEPS: Int = 200

# `initialize_episode` bounds, mirrored from reacher.py for the reset test.
comptime R_MIN: Float64 = 0.05
comptime R_MAX: Float64 = 0.20


def _action_at(step: Int, k: Int) -> Float64:
    return AMP * sin(0.09 * Float64(step) + 1.9 * Float64(k))


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


def _build_model() raises -> Model[DType.float64, Dims[nv=DMReacherModel.NV, nbody=DMReacherModel.NBODY, njoint=DMReacherModel.NJOINT, ngeom=DMReacherModel.NGEOM, nequality=DMReacherModel.MAX_EQUALITY, ntendon=DMReacherModel.MAX_TENDON, nsite=DMReacherModel.NSITE, nexclude=DMReacherModel.NEXCLUDE, nmesh_verts=0]]:
    var ctx = DeviceContext()
    var mf = Model[DType.float64, Dims[nv=DMReacherModel.NV, nbody=DMReacherModel.NBODY, njoint=DMReacherModel.NJOINT, ngeom=DMReacherModel.NGEOM, nequality=DMReacherModel.MAX_EQUALITY, ntendon=DMReacherModel.MAX_TENDON, nsite=DMReacherModel.NSITE, nexclude=DMReacherModel.NEXCLUDE, nmesh_verts=0]]()
    DMReacherModel.init_fields[DType.float64, 0](ctx, mf)
    return mf^


def test_reacher_model_matches_mujoco() raises:
    """Dims, masses, joint ranges, geom indices, and the mocap substitution.

    `wrist` is `range="-160 160"` with the file's default `<compiler angle>`,
    i.e. DEGREES. If the degree->radian conversion were skipped the arm would
    fold at +-160 rad (unlimited in practice) and every rollout would still
    look plausible — hence the explicit range assertion.
    """
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path(String(REF_XML))

    assert_true(Int(py=m.njnt) == DMReacherModel.NJOINT, "njnt mismatch")
    assert_true(Int(py=m.nq) == DMReacherModel.NQ, "nq mismatch")
    assert_true(Int(py=m.nv) == DMReacherModel.NV, "nv mismatch")
    assert_true(Int(py=m.ngeom) == DMReacherModel.NGEOM, "ngeom mismatch")
    assert_true(Int(py=m.nu) == DMReacherModel.nact, "nu mismatch")

    # The one intended structural difference: our target rides a mocap body,
    # so we carry exactly one body more than the reference and it must be
    # flagged as mocap. Assert both halves — an unflagged body would be swept
    # by FK and the target would follow the world instead of the episode.
    assert_true(Int(py=m.nbody) == REF_NBODY, "reference nbody moved")
    assert_true(
        DMReacherModel.NBODY == REF_NBODY + 1,
        "ours should carry exactly one extra body (the mocap target)",
    )

    var mf = _build_model()
    assert_true(
        Int(mf.bodies.data[TARGET_BODY_IDX * MODEL_BODY_SIZE + BODY_IDX_MOCAP])
        == 1,
        "the target body is not flagged mocap — FK would overwrite it",
    )
    for b in range(DMReacherModel.NBODY):
        if b == TARGET_BODY_IDX:
            continue
        assert_true(
            Int(mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_MOCAP]) == 0,
            "an arm body is flagged mocap",
        )

    # Geom indices. Ours and MuJoCo's genuinely disagree (see `reacher_xml`);
    # pin BOTH so a reordering on either side fails loudly here rather than
    # silently swapping which geom the reward measures.
    var tg_id = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target")
    )
    var fg_id = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger")
    )
    assert_true(tg_id == REF_TARGET_GEOM_IDX, "MuJoCo's target geom moved")
    assert_true(fg_id == REF_FINGER_GEOM_IDX, "MuJoCo's finger geom moved")
    assert_true(
        Int(mf.geoms.data[TARGET_GEOM_IDX * MODEL_GEOM_SIZE + GEOM_IDX_BODY])
        == TARGET_BODY_IDX,
        "our target geom does not ride the mocap body",
    )
    assert_true(
        Int(mf.geoms.data[FINGER_GEOM_IDX * MODEL_GEOM_SIZE + GEOM_IDX_BODY])
        == FINGER_BODY_IDX,
        "our finger geom does not ride the finger body",
    )

    # Reward radii, lifted from the XML into comptimes on our side.
    assert_true(
        abs(Float64(py=m.geom_size[fg_id][0]) - FINGER_SIZE) <= 1e-15,
        "FINGER_SIZE is out of sync with reacher.xml",
    )
    assert_true(
        abs(Float64(py=m.geom_size[tg_id][0]) - BIG_TARGET) <= 1e-15,
        "reacher.xml's target radius is no longer _BIG_TARGET",
    )
    assert_true(
        abs(Float64(py=m.geom_pos[tg_id][2]) - TARGET_Z) <= 1e-15,
        "TARGET_Z is out of sync with reacher.xml",
    )

    # Masses, over the bodies both models share.
    var worst = 0.0
    for b in range(NBODY_SHARED):
        var dm = abs(
            mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_MASS]
            - Float64(py=m.body_mass[b])
        )
        if dm > worst:
            worst = dm
    print("reacher model build: max |d(mass)| =", worst)
    assert_true(worst <= 1e-15, "reacher masses differ from MuJoCo")

    # Joint ranges. MuJoCo reports 0..0 for an unlimited joint; ours uses the
    # +-1e10 sentinel, so only the limited one is comparable.
    var lo = Float64(mf.joints.data[1 * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN])
    var hi = Float64(mf.joints.data[1 * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MAX])
    var d_lo = abs(lo - Float64(py=m.jnt_range[1][0]))
    var d_hi = abs(hi - Float64(py=m.jnt_range[1][1]))
    print("  wrist range =", lo, "..", hi, " max |d| =", max(d_lo, d_hi))
    assert_true(
        max(d_lo, d_hi) <= 1e-15,
        "wrist range differs — degree->radian conversion missing?",
    )
    var s_lo = Float64(mf.joints.data[0 * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN])
    assert_true(
        s_lo <= -1e9,
        "shoulder is not unlimited — the reset would take the wrong branch",
    )


def _rollout[
    MODEL: ModelDefLike, TSIZE: Float64
]() raises -> List[Float64]:
    """One easy/hard rollout set; returns [state, geom, obs, reward, r_min,
    r_max, hit_limit] as floats so both parameterizations report identically.

    ⚠ THE MODEL IS A PARAMETER because `hard` no longer shares `easy`'s. They
    differ only in the target's radius, which is INERT (contact is disabled
    model-wide) — so this rollout cannot tell them apart, and that is exactly
    why the parameter has to be here rather than hardcoded: the numbers below
    would stay green while the env that ships drifted away from the one under
    test. What the radius DOES reach is the renderer, which reads it at compile
    time; see `reacher_xml.DMReacherHardModel`.
    """
    comptime EnvT = Phyics3dEnv[
        MODEL, DMReacherConfig[TSIZE], DType.float64, False
    ]

    var handle = _setup()
    var mujoco = handle[0]
    var model = handle[1]
    var data = handle[2]
    var tol = handle[3]

    var tg_id = Int(
        py=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target")
    )

    var max_state = 0.0
    var max_geom = 0.0
    var max_obs = 0.0
    var max_r = 0.0
    var r_min = 1e9
    var r_max = -1e9
    var hit_limit = 0.0

    # (shoulder, wrist, target_x, target_y). The reward is a HARD indicator, so
    # a rollout that never overlaps the target gates only the zero branch. With
    # both joints at 0 the finger sits at (0.24, 0) — outside the arena wall,
    # but the target radius is unconstrained by it — so the last two inits park
    # the target on the finger and the run sweeps both sides of the threshold,
    # for `hard`'s 1.5 cm disc as well as `easy`'s 5 cm one.
    var inits = [
        [0.0, 0.0, 0.10, 0.15],
        [1.2, -0.7, -0.12, 0.08],
        [-2.5, 1.4, 0.06, -0.17],
        [0.0, 0.0, 0.24, 0.0],
        [0.4, 0.3, 0.20, 0.09],
    ]

    for init in inits:
        var tx = init[2]
        var ty = init[3]

        # Reference: the model write dm_control's `initialize_episode` does.
        mujoco.mj_resetData(model, data)
        model.geom_pos[tg_id][0] = tx
        model.geom_pos[tg_id][1] = ty
        for i in range(NQ):
            data.qpos[i] = init[i]
        mujoco.mj_forward(model, data)

        # Ours: the same coordinate through the per-env mocap path.
        var env = EnvT()
        _ = env.reset()
        env.d.mocap_pos.data[TARGET_BODY_IDX * 3 + 0] = tx
        env.d.mocap_pos.data[TARGET_BODY_IDX * 3 + 1] = ty
        env.d.mocap_pos.data[TARGET_BODY_IDX * 3 + 2] = TARGET_Z
        var qs = List[Float64]()
        var vs = List[Float64]()
        for i in range(NQ):
            qs.append(init[i])
        for _ in range(NV):
            vs.append(0.0)
        env.set_state(qs, vs)  # re-syncs mocap -> xpos, then FK

        for step in range(N_STEPS):
            var act = EnvT.ActionType()
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
            for i in range(NV):
                var dv = abs(
                    Float64(py=data.qvel[i]) - Float64(env.d.qvel.data[i])
                )
                if dv > max_state:
                    max_state = dv
            # The wrist limit is the one constraint our solver and MuJoCo's are
            # known to disagree on, so report whether the run ever loads it.
            var wq = Float64(py=data.qpos[1])
            var wlo = Float64(py=model.jnt_range[1][0])
            var whi = Float64(py=model.jnt_range[1][1])
            if wq <= wlo + 1e-6 or wq >= whi - 1e-6:
                hit_limit = 1.0

            # geom_xpos, paired by NAME: (our index, MuJoCo's index).
            var geom_pairs = [
                [FINGER_GEOM_IDX, REF_FINGER_GEOM_IDX],
                [TARGET_GEOM_IDX, REF_TARGET_GEOM_IDX],
            ]
            for pair in geom_pairs:
                var ours = geom_xpos(env.d, env.mf.geoms.data, pair[0])
                var mine = [ours[0], ours[1], ours[2]]
                for k in range(3):
                    var dg = abs(
                        Float64(py=data.geom_xpos[pair[1]][k]) - mine[k]
                    )
                    if dg > max_geom:
                        max_geom = dg

            # to_target and the observation: qpos, to_target, qvel.
            var ttx = Float64(py=data.geom_xpos[REF_TARGET_GEOM_IDX][0]) - Float64(
                py=data.geom_xpos[REF_FINGER_GEOM_IDX][0]
            )
            var tty = Float64(py=data.geom_xpos[REF_TARGET_GEOM_IDX][1]) - Float64(
                py=data.geom_xpos[REF_FINGER_GEOM_IDX][1]
            )
            var obs = out[0]
            var ref_obs = [
                Float64(py=data.qpos[0]),
                Float64(py=data.qpos[1]),
                ttx,
                tty,
                Float64(py=data.qvel[0]),
                Float64(py=data.qvel[1]),
            ]
            for i in range(6):
                var d_o = abs(ref_obs[i] - Float64(obs.data[i]))
                if d_o > max_obs:
                    max_obs = d_o

            # reward = tolerance(||to_target||, (0, target + finger)), margin 0.
            var dist = sqrt(ttx * ttx + tty * tty)
            var radii = TSIZE + FINGER_SIZE
            var ref_r = Float64(
                py=tol(dist, 0.0, radii, 0.0, String("gaussian"), 0.1)
            )
            var d_r = abs(ref_r - Float64(out[1]))
            if d_r > max_r:
                max_r = d_r
            if ref_r < r_min:
                r_min = ref_r
            if ref_r > r_max:
                r_max = ref_r

    return [max_state, max_geom, max_obs, max_r, r_min, r_max, hit_limit]


def _check[MODEL: ModelDefLike, TSIZE: Float64](label: String) raises:
    var r = _rollout[MODEL, TSIZE]()
    print("reacher", label, "vs MuJoCo, 5 x", N_STEPS, "steps:")
    print("  max |d(state)| =", r[0], " |d(geom_xpos)| =", r[1])
    print("  max |d(obs)| =", r[2], " |d(reward)| =", r[3])
    print("  reward range =", r[4], "..", r[5], " loaded wrist limit:", r[6] > 0)

    assert_true(r[0] <= STATE_TOL, "physics deviated")
    assert_true(r[1] <= GEOM_TOL, "geom_xpos / mocap target deviated")
    assert_true(r[2] <= OBS_TOL, "observation deviated")
    assert_true(r[3] <= REWARD_TOL, "reward deviated")
    # A hard indicator gates nothing if the run only ever sees one side of it.
    assert_true(r[4] == 0.0, "no rollout step scored 0 — reward gate is weak")
    assert_true(r[5] == 1.0, "no rollout step scored 1 — reward gate is weak")


def test_reacher_easy_matches_mujoco() raises:
    """Physics, geom_xpos through the mocap target, observation and reward."""
    _check[DMReacherModel, BIG_TARGET]("easy")


def test_reacher_hard_matches_mujoco() raises:
    """Same rollout at `hard`'s 1.5 cm radius — the only difference between the
    two registered tasks, and the one we carry as a comptime instead of the
    reference's per-episode `geom_size` write."""
    _check[DMReacherHardModel, SMALL_TARGET]("hard")


def test_reacher_reset_randomization() raises:
    """`initialize_episode`: joint randomization plus the polar target.

    reacher is the first ported domain with one LIMITED and one UNLIMITED
    joint, so both branches of `randomize_limited_and_rotational_joints` run
    here. The gate is distributional (bounds + coverage), not a seed match —
    we do not reproduce numpy's RandomState (gap G9).
    """
    comptime N_TRIALS: Int = 400
    var env = DMReacherEasy[DType.float64]()

    var wrist_lo = Float64(
        env.mf.joints.data[1 * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN]
    )
    var wrist_hi = Float64(
        env.mf.joints.data[1 * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MAX]
    )

    var r_lo = 1e9
    var r_hi = -1e9
    var sh_lo = 1e9
    var sh_hi = -1e9
    var wr_lo = 1e9
    var wr_hi = -1e9
    var bad_z = 0
    var quadrants = [0, 0, 0, 0]

    for _ in range(N_TRIALS):
        _ = env.reset()
        var tp = geom_xpos(env.d, env.mf.geoms.data, TARGET_GEOM_IDX)
        var rad = sqrt(tp[0] * tp[0] + tp[1] * tp[1])
        if rad < r_lo:
            r_lo = rad
        if rad > r_hi:
            r_hi = rad
        if abs(tp[2] - TARGET_Z) > 1e-15:
            bad_z += 1
        var qi = 0
        if tp[0] >= 0.0:
            qi += 1
        if tp[1] >= 0.0:
            qi += 2
        quadrants[qi] += 1

        var sh = Float64(env.d.qpos.data[0])
        var wr = Float64(env.d.qpos.data[1])
        if sh < sh_lo:
            sh_lo = sh
        if sh > sh_hi:
            sh_hi = sh
        if wr < wr_lo:
            wr_lo = wr
        if wr > wr_hi:
            wr_hi = wr

    print("reacher reset over", N_TRIALS, "episodes:")
    print("  target radius", r_lo, "..", r_hi, " (want", R_MIN, "..", R_MAX, ")")
    print("  shoulder", sh_lo, "..", sh_hi, " (want -pi..pi)")
    print("  wrist   ", wr_lo, "..", wr_hi, " (want", wrist_lo, "..", wrist_hi, ")")
    print("  target quadrant counts", quadrants[0], quadrants[1],
          quadrants[2], quadrants[3])

    assert_true(bad_z == 0, "target z drifted off the arena plane")
    assert_true(
        r_lo >= R_MIN - 1e-12 and r_hi <= R_MAX + 1e-12,
        "target radius left [.05, .20]",
    )
    # Coverage: 400 draws should fill most of each range.
    assert_true(
        r_lo < R_MIN + 0.02 and r_hi > R_MAX - 0.02,
        "target radius does not span its range",
    )
    for q in range(4):
        assert_true(
            quadrants[q] > 40,
            "target angle is not uniform over the circle",
        )
    assert_true(
        sh_lo >= -pi - 1e-12 and sh_hi <= pi + 1e-12,
        "shoulder left [-pi, pi] — the unlimited branch is wrong",
    )
    assert_true(
        sh_lo < -pi + 0.2 and sh_hi > pi - 0.2,
        "shoulder does not span the full circle",
    )
    assert_true(
        wr_lo >= wrist_lo - 1e-12 and wr_hi <= wrist_hi + 1e-12,
        "wrist left its range — the limited branch is wrong",
    )
    assert_true(
        wr_lo < wrist_lo + 0.2 and wr_hi > wrist_hi - 0.2,
        "wrist does not span its range",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
