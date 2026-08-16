"""dm_control `quadruped fetch` vs the reference — model, observation, reward.

The twelve dims `fetch` adds to `quadruped`'s 78 are `ball_state` (9) and
`target_position` (3), and all twelve are frame conversions that stay finite
and plausible-looking when they are wrong. That is what this file is for.

⚠ THE FRAME DIRECTION IS THE WHOLE RISK. `ball_state` and `target_position`
both end in `v.dot(torso_frame)` with `torso_frame = xmat['torso'].reshape(3,3)`
UNTRANSPOSED. Under numpy's row-vector convention that is `R^T v` — world to
body. `R v` is equally plausible to write, differs only by a transpose, and
produces twelve numbers of the right magnitude with the right units. The pose
below is chosen with a NON-TRIVIAL TORSO ROTATION for exactly that reason: at
identity orientation `R = R^T` and the test would pass either way.

⚠ THE SITE IDS ARE NOT walk/run's. `<site name="target">` is declared before
the torso body, so it takes id 0 and pushes every other site up by one —
`torso` 24 -> 25, the toes 25.. -> 26... A gate that reused the walk constants
would read the velocimeter off a pupil site and still get finite numbers.

⚠ THE REWARD IS ZERO OVER MOST OF THE ARENA and that is CORRECT: both terms
use a linear sigmoid with `value_at_margin=0`, so anything beyond
`arena_radius = 15*sqrt(2) = 21.2` scores exactly 0. A smoke test that spawns
the ball at random therefore sees `reward == 0.0` on almost every step and
proves nothing. The reward cases below PIN the ball at chosen distances and
assert both a zero and a non-zero.

Run: pixi run mojo run -I . tests/dm_control/test_quadruped_fetch_vs_dm_control.mojo
"""

from std.math import abs, sqrt, cos, sin
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.quadruped import (
    DMQuadrupedFetch,
    QUADRUPED_FETCH_OBS_DIM,
    FETCH_TARGET_SITE_IDX,
    FETCH_WORKSPACE_SITE_IDX,
    FETCH_TORSO_SITE_IDX,
    FETCH_TOE_SITE_0,
    FETCH_BALL_BODY_IDX,
    FETCH_BALL_QPOS_0,
    FETCH_BALL_DOF_0,
    qfp,
)

comptime OBS_TOL: Float64 = 1e-9
comptime NQ: Int = 30
comptime NV: Int = 28


def _mj() raises -> PythonObject:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path("mojo_rl/envs/dm_control/assets/quadruped_fetch.xml")
    return Python.tuple(mujoco, m, mujoco.MjData(m))


def _pose() raises -> Tuple[List[Float64], List[Float64]]:
    """A deliberately asymmetric, airborne pose.

    The torso carries a rotation about all three axes so that `R != R^T`, and
    the ball sits off-axis with both linear and angular velocity so that every
    one of the nine `ball_state` numbers is distinct and non-zero. Airborne so
    `ncon == 0` and the comparison is not entangled with the solver.
    """
    var qpos = List[Float64]()
    for _ in range(NQ):
        qpos.append(0.0)
    var qvel = List[Float64]()
    for _ in range(NV):
        qvel.append(0.0)

    # torso free joint: [x, y, z, qw, qx, qy, qz] — w FIRST.
    qpos[0] = 0.4
    qpos[1] = -0.7
    qpos[2] = 1.3
    var q = [0.8446, 0.1913, -0.4619, 0.1913]
    var n = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    for k in range(4):
        qpos[3 + k] = q[k] / n
    # a few leg hinges off zero so the model is not in its rest pose
    qpos[7] = 0.21
    qpos[9] = -0.35
    qpos[12] = 0.44

    # ball free joint at the tail of the state vector
    qpos[FETCH_BALL_QPOS_0 + 0] = 2.6
    qpos[FETCH_BALL_QPOS_0 + 1] = 1.1
    qpos[FETCH_BALL_QPOS_0 + 2] = 0.9
    qpos[FETCH_BALL_QPOS_0 + 3] = 1.0

    qvel[0] = 0.33
    qvel[1] = -0.21
    qvel[2] = 0.15
    qvel[3] = 0.5
    qvel[4] = -0.25
    qvel[5] = 0.8
    qvel[FETCH_BALL_DOF_0 + 0] = -1.7
    qvel[FETCH_BALL_DOF_0 + 1] = 2.3
    qvel[FETCH_BALL_DOF_0 + 2] = -0.6
    qvel[FETCH_BALL_DOF_0 + 3] = 3.1
    qvel[FETCH_BALL_DOF_0 + 4] = -1.4
    qvel[FETCH_BALL_DOF_0 + 5] = 2.2
    return (qpos^, qvel^)


def test_fetch_model_matches_mujoco() raises:
    """Dims and the four indices the config reads state through."""
    print("--- fetch: model vs MuJoCo ---")
    var h = _mj()
    var mujoco = h[0]
    var m = h[1]
    var O = mujoco.mjtObj

    assert_true(Int(py=m.nq) == qfp.NQ, "nq mismatch")
    assert_true(Int(py=m.nv) == qfp.NV, "nv mismatch")
    assert_true(Int(py=m.nbody) == qfp.NBODY, "nbody mismatch")
    assert_true(Int(py=m.nsite) == qfp.NSITE, "nsite mismatch")
    assert_true(Int(py=m.ngeom) == qfp.NGEOM, "ngeom mismatch")

    # ⚠ These are the constants the config indexes with. Counting them by hand
    # off the XML is what `<default>` blocks make unreliable; they come from a
    # compiled mjModel and this asserts they still do.
    assert_true(
        Int(py=mujoco.mj_name2id(m, O.mjOBJ_SITE, "target"))
        == FETCH_TARGET_SITE_IDX,
        "target site id moved — every site index shifts with it",
    )
    assert_true(
        Int(py=mujoco.mj_name2id(m, O.mjOBJ_SITE, "workspace"))
        == FETCH_WORKSPACE_SITE_IDX, "workspace site id moved")
    assert_true(
        Int(py=mujoco.mj_name2id(m, O.mjOBJ_SITE, "torso"))
        == FETCH_TORSO_SITE_IDX, "torso site id moved")
    assert_true(
        Int(py=mujoco.mj_name2id(m, O.mjOBJ_SITE, "toe_front_left"))
        == FETCH_TOE_SITE_0, "first toe site id moved")
    assert_true(
        Int(py=mujoco.mj_name2id(m, O.mjOBJ_BODY, "ball"))
        == FETCH_BALL_BODY_IDX, "ball body id moved")

    var jid = mujoco.mj_name2id(m, O.mjOBJ_JOINT, "ball_root")
    assert_true(
        Int(py=m.jnt_qposadr[jid]) == FETCH_BALL_QPOS_0
        and Int(py=m.jnt_dofadr[jid]) == FETCH_BALL_DOF_0,
        "ball_root is no longer the last joint — the state layout changed",
    )

    # The ball is the reason the pyramid had to be generalised; if this ever
    # reads 3 the model stopped exercising rolling friction.
    var bg = mujoco.mj_name2id(m, O.mjOBJ_GEOM, "ball")
    assert_true(
        Int(py=m.geom_condim[bg]) == 6,
        "the ball is no longer condim 6 — rolling friction is not being tested"
        " and `max_condim` on the model def is now wrong",
    )
    assert_true(
        qfp.MAX_CONDIM == 6,
        "parse_xml's condim scan no longer sees the ball's condim=6, so the"
        " pyramidal edge list would be sized for 4 edges and the torsional and"
        " rolling rows would be silently dropped",
    )

    # Four tilted plane walls must survive `walls_and_ball=True`.
    var nwall = 0
    for nm in [
        String("wall_px"), String("wall_py"),
        String("wall_nx"), String("wall_ny"),
    ]:
        if Int(py=mujoco.mj_name2id(m, O.mjOBJ_GEOM, nm)) >= 0:
            nwall += 1
    assert_true(nwall == 4, "the arena walls are missing")
    print("  PASS: dims, ids, condim 6, 4 walls")


def test_fetch_ball_state_and_target_match_reference() raises:
    """The twelve fetch-only dims, against the reference formulas."""
    print("--- fetch: ball_state + target_position ---")
    var st = _pose()
    var qpos = st[0].copy()
    var qvel = st[1].copy()

    var env = DMQuadrupedFetch[DType.float64](DeviceContext(), 1000, 1)
    _ = env.reset()
    env.set_state(qpos, qvel)
    var a = type_of(env).ActionType()
    _ = env.step(a)
    env.set_state(qpos, qvel)
    var obs = env.get_obs_list()
    assert_true(
        len(obs) == QUADRUPED_FETCH_OBS_DIM,
        "fetch observation is not 90 long",
    )

    var h = _mj()
    var mujoco = h[0]
    var m = h[1]
    var dat = h[2]
    for i in range(NQ):
        dat.qpos[i] = qpos[i]
    for i in range(NV):
        dat.qvel[i] = qvel[i]
    mujoco.mj_forward(m, dat)
    assert_true(Int(py=dat.ncon) == 0, "pose must be contact-free")

    var np = Python.import_module("numpy")
    var O = mujoco.mjtObj
    var tb = mujoco.mj_name2id(m, O.mjOBJ_BODY, "torso")
    var bb = mujoco.mj_name2id(m, O.mjOBJ_BODY, "ball")
    var tg = mujoco.mj_name2id(m, O.mjOBJ_SITE, "target")

    # torso_frame = xmat['torso'].reshape(3,3); v.dot(torso_frame)
    var R = np.array(dat.xmat[tb]).reshape(3, 3)
    var torso_pos = np.array(dat.xpos[tb])
    var ball_pos = np.array(dat.xpos[bb])

    var rel_pos = np.dot(np.subtract(ball_pos, torso_pos), R)
    var v_lin = np.array(
        [
            dat.qvel[FETCH_BALL_DOF_0 + 0] - dat.qvel[0],
            dat.qvel[FETCH_BALL_DOF_0 + 1] - dat.qvel[1],
            dat.qvel[FETCH_BALL_DOF_0 + 2] - dat.qvel[2],
        ]
    )
    var rel_vel = np.dot(v_lin, R)
    var v_rot = np.array(
        [
            dat.qvel[FETCH_BALL_DOF_0 + 3],
            dat.qvel[FETCH_BALL_DOF_0 + 4],
            dat.qvel[FETCH_BALL_DOF_0 + 5],
        ]
    )
    var rot_vel = np.dot(v_rot, R)
    var tgt = np.dot(
        np.subtract(np.array(dat.site_xpos[tg]), torso_pos), R
    )

    var want = List[Float64]()
    for k in range(3):
        want.append(Float64(py=rel_pos[k]))
    for k in range(3):
        want.append(Float64(py=rel_vel[k]))
    for k in range(3):
        want.append(Float64(py=rot_vel[k]))
    for k in range(3):
        want.append(Float64(py=tgt[k]))

    # NON-VACUITY: with an identity torso rotation `R == R^T` and a transposed
    # implementation would agree. Require the pose to actually distinguish
    # them, by checking R is not symmetric.
    var asym = Float64(0)
    for i in range(3):
        for j in range(3):
            var e = abs(
                Float64(py=R[i][j]) - Float64(py=R[j][i])
            )
            if e > asym:
                asym = e
    print("  torso frame asymmetry (must be >> 0):", asym)
    assert_true(
        asym > 0.1,
        "the torso rotation is too close to symmetric — R and R^T would give"
        " the same answer and this test would not detect a transposed frame",
    )

    var worst = Float64(0)
    var worst_i = 0
    for k in range(12):
        var e = abs(Float64(obs[78 + k]) - want[k])
        if e > worst:
            worst = e
            worst_i = k
    print("  worst |err| over the 12 fetch dims =", worst, "at", worst_i)
    print("    ours =", Float64(obs[78 + worst_i]), " ref =", want[worst_i])
    assert_true(worst < OBS_TOL, "ball_state / target_position diverge")
    print("  PASS: 9 ball_state + 3 target_position")


def test_fetch_reward_matches_reference() raises:
    """Both reward terms, at a scoring pose and at an out-of-margin one."""
    print("--- fetch: reward ---")
    var np = Python.import_module("numpy")

    comptime ARENA = 15.0 * 1.4142135623730951
    comptime WORKSPACE_R = 0.3
    comptime BALL_R = 0.15
    comptime TARGET_R = 0.4

    def _lin(x: Float64, lo: Float64, hi: Float64, margin: Float64)
            -> Float64:
        """`rewards.tolerance(..., sigmoid='linear', value_at_margin=0)`."""
        if x >= lo and x <= hi:
            return 1.0
        var d = (lo - x if x < lo else x - hi) / margin
        return 0.0 if d >= 1.0 else 1.0 - d

    var env = DMQuadrupedFetch[DType.float64](DeviceContext(), 1000, 1)

    # --- case 1: ball parked ON the target, quadruped right beside it -------
    var st = _pose()
    var qpos = st[0].copy()
    var qvel = st[1].copy()
    # Upright torso so the upright term is 1 and the two distance terms are
    # what is actually under test.
    qpos[3] = 1.0
    qpos[4] = 0.0
    qpos[5] = 0.0
    qpos[6] = 0.0
    qpos[0] = 0.6
    qpos[1] = 0.0
    qpos[FETCH_BALL_QPOS_0 + 0] = 0.0
    qpos[FETCH_BALL_QPOS_0 + 1] = 0.0
    qpos[FETCH_BALL_QPOS_0 + 2] = 0.15
    for i in range(NV):
        qvel[i] = 0.0

    _ = env.reset()
    env.set_state(qpos, qvel)
    var a = type_of(env).ActionType()
    var res_near = env.step(a)

    # ⚠ THE REFERENCE IS BUILT FROM OUR OWN POST-STEP STATE, not from the pose
    # that was set. `step` advances the physics before computing the reward, so
    # comparing against `mj_forward` at the PINNED pose is off by one substep —
    # it showed up here as a 1.2e-3 disagreement that looks like a formula bug
    # and is not one. What this test is for is the REWARD FORMULA; the dynamics
    # that produced the state are gated by
    # tests/dm_control/test_quadruped_vs_dm_control.mojo.
    var post_q = List[Float64]()
    for i in range(NQ):
        post_q.append(Float64(env.d.qpos.data[i]))
    var post_v = List[Float64]()
    for i in range(NV):
        post_v.append(Float64(env.d.qvel.data[i]))

    var h = _mj()
    var mujoco = h[0]
    var m = h[1]
    var dat = h[2]
    for i in range(NQ):
        dat.qpos[i] = post_q[i]
    for i in range(NV):
        dat.qvel[i] = post_v[i]
    mujoco.mj_forward(m, dat)

    var O = mujoco.mjtObj
    var bb = mujoco.mj_name2id(m, O.mjOBJ_BODY, "ball")
    var ws = mujoco.mj_name2id(m, O.mjOBJ_SITE, "workspace")
    var tg = mujoco.mj_name2id(m, O.mjOBJ_SITE, "target")
    var bx = Float64(py=dat.xpos[bb][0])
    var by = Float64(py=dat.xpos[bb][1])
    var s2b = sqrt(
        (Float64(py=dat.site_xpos[ws][0]) - bx) ** 2
        + (Float64(py=dat.site_xpos[ws][1]) - by) ** 2
    )
    var b2t = sqrt(
        (Float64(py=dat.site_xpos[tg][0]) - bx) ** 2
        + (Float64(py=dat.site_xpos[tg][1]) - by) ** 2
    )
    var reach = _lin(s2b, 0.0, WORKSPACE_R + BALL_R, ARENA)
    var fetchr = _lin(b2t, 0.0, TARGET_R, ARENA)
    var upright = Float64(py=dat.xmat[1][8])
    var up_r = _lin(upright, 1.0, 1.0e300, 2.0)
    var want_near = up_r * reach * (0.5 + 0.5 * fetchr)

    print("  near: self_to_ball", s2b, " ball_to_target", b2t)
    print("        reach", reach, " fetch", fetchr, " upright", up_r)
    print("        ours", Float64(res_near[1]), " ref", want_near)

    # NON-VACUITY: this pose must actually score, or the comparison is a
    # comparison of two zeros — which is what a randomly-spawned ball gives on
    # almost every step.
    assert_true(
        want_near > 0.4,
        "the 'near' pose does not score — it no longer distinguishes a"
        " working reward from one that returns 0 everywhere",
    )
    assert_true(
        abs(Float64(res_near[1]) - want_near) < OBS_TOL,
        "fetch reward diverges at the scoring pose",
    )

    # --- case 2: ball outside the margin, reward must be exactly 0 ----------
    # ⚠ IT TAKES OPPOSITE CORNERS. `arena_radius` is 15*sqrt(2) = 21.2, which
    # is the corner-to-CENTRE distance — a ball in one corner with the
    # quadruped anywhere near the middle is still inside the margin and scores
    # a small non-zero. The first draft put the ball at (14, 14) with the torso
    # near the origin and got 0.073, which is the correct reward for that pose
    # and not the branch under test. Only a corner-to-corner separation
    # (~38 m) actually clears 21.2.
    qpos[0] = -13.0
    qpos[1] = -13.0
    qpos[FETCH_BALL_QPOS_0 + 0] = 14.0
    qpos[FETCH_BALL_QPOS_0 + 1] = 14.0
    env.set_state(qpos, qvel)
    var res_far = env.step(a)
    print("  far:  ours", Float64(res_far[1]), " (must be exactly 0)")
    assert_true(
        Float64(res_far[1]) == 0.0,
        "beyond arena_radius the linear sigmoid with value_at_margin=0 must"
        " give exactly 0",
    )
    print("  PASS: both reward branches")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
