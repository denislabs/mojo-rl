"""dm_control `point_mass-hard` parity: our env vs MuJoCo + the reference task.

`hard` differs from `easy` in exactly one thing — `initialize_episode` writes a
random 2x2 mixing matrix into `model.wrap_prm`, so each control drives a random
linear combination of `root_x`/`root_y`. Everything downstream (observation,
reward, horizon, model) is `easy`'s and is already gated by
`test_point_mass_vs_dm_control.mojo`; this file gates the mixing.

WHAT IS AND IS NOT REPRODUCIBLE. dm_control draws the two directions from its
own `np.random.RandomState`, which we cannot replay, so an episode-for-episode
match against the reference is not on offer and is not the point. The gate is
split accordingly:

  * `test_..._transmission_matches_mujoco` — write the SAME coefficients into
    both engines and step them in lockstep. This is the part that has to be
    exact: it is the physics of a fixed-tendon transmission under a
    non-identity, non-orthogonal mixing, which no previously ported model
    exercises (fish's tendons are springs, ball_in_cup's is a spatial limit,
    and `easy`'s coefs are the identity).
  * `test_..._randomization_matches_reference` — gate our SAMPLER against the
    reference's stated distribution rather than its draws: unit norm, the
    `|dot| > 0.9` rejection, per-episode variation, and that the mixing
    actually reaches the physics.

The second test is the one that would have caught the failure this task is
really about. A `hard` config that inherited `MODEL_DEF.apply_actions` would
read the COMPTIME transmission tables, never see the randomized coefs, and
behave exactly like `easy` — a working env for the wrong task, with no error
anywhere. So it asserts the response direction tracks `dir1`, not merely that
the numbers in the model records changed.

Run with:
    pixi run mojo run -I . tests/dm_control/test_point_mass_hard_vs_dm_control.mojo
"""

from std.math import abs, sin, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.envs.dm_control.point_mass import (
    DMPointMassHard,
    DMPointMassModel,
    T1_TENDON_IDX,
    T2_TENDON_IDX,
    TARGET_SIZE,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_TENDON_SIZE,
    TENDON_IDX_NUM_JOINTS,
    TENDON_IDX_JOINT_0,
    TENDON_IDX_COEF_0,
    TENDON_IDX_KIND,
    TENDON_KIND_FIXED,
)

# ⚠ THE WRAP STRIDE IS A CONSTANT, NOT A LITERAL. These tables are
# `[actuator * MAX_COMPTIME_TENDON_WRAPS + k]`; the cap moved 4 -> 16
# with defect 17 and a hardcoded 4 here silently reads the wrong slot.
from mojo_rl.physics3d.parser.xml_parser import MAX_COMPTIME_TENDON_WRAPS
from mojo_rl.physics3d.gpu.constants import (
    MODEL_ACTUATOR_SIZE,
    ACT_IDX_GEAR,
    ACT_IDX_TRN_N,
    ACT_IDX_TRN_COEF_0,
)


comptime Env = DMPointMassHard[DType.float64]

comptime REF_XML: StaticString = (
    "references/dm_control-main/dm_control/suite/point_mass.xml"
)
comptime REF_PATH: StaticString = "references/dm_control-main"

comptime NQ: Int = 2
comptime NV: Int = 2
comptime NACT: Int = 2
comptime NTENDON: Int = 2
comptime FRAME_SKIP: Int = 1

comptime STATE_TOL: Float64 = 1e-9
comptime OBS_TOL: Float64 = 1e-9
comptime REWARD_TOL: Float64 = 1e-9

comptime AMP: Float64 = 0.9
comptime N_STEPS: Int = 150

# `PointMass.initialize_episode`'s rejection threshold.
comptime PARALLEL_COS: Float64 = 0.9


def _action_at(step: Int, k: Int) -> Float64:
    return AMP * sin(0.11 * Float64(step) + 1.6 * Float64(k))


def _set_our_coefs(mut env: Env, d1x: Float64, d1y: Float64,
                   d2x: Float64, d2y: Float64):
    """Overwrite what `custom_reset_model_cpu` drew, so both engines mix the
    same way. Must run AFTER `reset()`, which is what randomizes them."""
    var t0 = T1_TENDON_IDX * MODEL_TENDON_SIZE
    env.mf.tendons.data[t0 + TENDON_IDX_COEF_0 + 0] = d1x
    env.mf.tendons.data[t0 + TENDON_IDX_COEF_0 + 1] = d1y
    var t1 = T2_TENDON_IDX * MODEL_TENDON_SIZE
    env.mf.tendons.data[t1 + TENDON_IDX_COEF_0 + 0] = d2x
    env.mf.tendons.data[t1 + TENDON_IDX_COEF_0 + 1] = d2y


def test_point_mass_hard_model_matches_mujoco() raises:
    """The wrap layout the randomizer indexes into, on both sides.

    `physics.model.wrap_prm[[0, 1]] = dir1` is a raw index into a FLAT array of
    wrap objects, so the reference's line is only correct if tendon 0's two
    joint wraps really are entries 0 and 1, in (root_x, root_y) order. Nothing
    in the XML guarantees that beyond declaration order, and nothing in either
    engine would complain if it changed — the mass would simply be driven along
    a transposed basis. Hence pinning it explicitly on both sides.
    """
    var sf = DMPointMassModel.make_spec_fields[DType.float64]()
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path(String(REF_XML))

    assert_true(Int(py=m.ntendon) == NTENDON, "ntendon mismatch")
    assert_true(Int(py=m.nwrap) == 4, "nwrap mismatch (2 tendons x 2 joints)")
    assert_true(
        Int(py=m.ntendon) == DMPointMassModel.MAX_TENDON,
        "our MAX_TENDON does not match MuJoCo's ntendon",
    )

    var jx = Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root_x"))
    var jy = Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root_y"))

    # wrap_objid / wrap_prm, flat over tendons: t1's (root_x, root_y) then
    # t2's. These four entries are literally what `hard` overwrites.
    var want_obj = [jx, jy, jx, jy]
    var want_prm = [1.0, 0.0, 0.0, 1.0]
    for w in range(4):
        assert_true(
            Int(py=m.wrap_objid[w]) == want_obj[w],
            "MuJoCo's wrap order moved — wrap_prm[[0,1]] no longer means t1",
        )
        assert_true(
            abs(Float64(py=m.wrap_prm[w]) - want_prm[w]) <= 1e-15,
            "MuJoCo's wrap_prm is not the XML's identity mixing",
        )

    # Each actuator drives the tendon of the same index — the pairing the
    # `hard` config's apply hook assumes (actuator a -> tendon a).
    for a in range(NACT):
        assert_true(
            Int(py=m.actuator_trntype[a])
            == Int(py=mujoco.mjtTrn.mjTRN_TENDON),
            "actuator is not a tendon transmission",
        )
        assert_true(
            Int(py=m.actuator_trnid[a][0]) == a,
            "actuator a does not drive tendon a — the apply hook's pairing",
        )

    # Our side: same records, from the runtime parser.
    var env = Env()
    _ = env.reset()
    for t in range(NTENDON):
        var o = t * MODEL_TENDON_SIZE
        assert_true(
            Int(env.mf.tendons.data[o + TENDON_IDX_KIND]) == TENDON_KIND_FIXED,
            "our tendon is not FIXED",
        )
        assert_true(
            Int(env.mf.tendons.data[o + TENDON_IDX_NUM_JOINTS]) == 2,
            "our tendon does not carry two joint wraps",
        )
        for k in range(2):
            assert_true(
                Int(env.mf.tendons.data[o + TENDON_IDX_JOINT_0 + k]) == k,
                "our wrap joint order is not (root_x, root_y)",
            )

    # …and our transmission RECORDS agree, which is what `easy` runs on. The
    # `comptime for` hoist these loops used to need is gone with `_acd`: a
    # tensor takes a runtime index.
    var trn_n = List[Int]()
    var gears = List[Float64]()
    for a in range(NACT):
        var o = a * MODEL_ACTUATOR_SIZE
        trn_n.append(Int(sf.actuators.data[o + ACT_IDX_TRN_N]))
        gears.append(Float64(sf.actuators.data[o + ACT_IDX_GEAR]))
    var trn_coef = List[Float64]()
    for a in range(NACT):
        for k in range(2):
            trn_coef.append(
                Float64(
                    sf.actuators.data[
                        a * MODEL_ACTUATOR_SIZE + ACT_IDX_TRN_COEF_0 + k
                    ]
                )
            )

    for a in range(NACT):
        assert_true(
            trn_n[a] == 2,
            "comptime transmission is not the 2-joint tendon",
        )
        for k in range(2):
            var want = 1.0 if a == k else 0.0
            assert_true(
                abs(trn_coef[a * 2 + k] - want) <= 1e-15,
                "comptime tendon coefs are not the XML's identity mixing",
            )
        assert_true(
            abs(gears[a] - 0.1) <= 1e-15,
            "motor gear is not the default .1",
        )


def test_point_mass_hard_transmission_matches_mujoco() raises:
    """Lockstep under four non-identity mixings, coefs pinned on both sides.

    The mixings are chosen to make the transmission earn its keep: a rotation
    (both DOFs driven by both controls), a near-parallel pair right at the
    rejection boundary (worst-conditioned mixing the task admits), a
    sign-flipped pair, and a swap. If our apply hook dropped a coef, used the
    wrong DOF address, or transposed the matrix, all four diverge on step 1.
    """
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

    # (dir1, dir2) — all unit, all with |dot| <= .9 so each is a mixing the
    # reference could actually have drawn.
    var mixings = [
        [0.8, 0.6, -0.6, 0.8],                    # rotation, orthogonal
        [1.0, 0.0, 0.9, 0.435889894354067],       # |dot| = .9, the boundary
        [-0.28, 0.96, 0.96, 0.28],                # sign-flipped, orthogonal
        [0.0, 1.0, 1.0, 0.0],                     # swap: control k drives the
                                                  # OTHER joint
    ]
    var inits = [
        [0.1, -0.05],
        [-0.2, 0.15],
        [0.005, -0.008],
        [0.25, 0.25],
    ]

    var max_state = 0.0
    var max_obs = 0.0
    var max_r = 0.0
    var min_dist = 1e9
    var max_qvel = 0.0

    for mi in range(len(mixings)):
        var mx = mixings[mi].copy()
        var init = inits[mi].copy()

        mujoco.mj_resetData(model, data)
        # The reference's own line, verbatim in effect: wrap_prm[[0,1]] = dir1,
        # wrap_prm[[2,3]] = dir2.
        for k in range(4):
            model.wrap_prm[k] = mx[k]
        for i in range(NQ):
            data.qpos[i] = init[i]
        mujoco.mj_forward(model, data)

        var env = Env()
        _ = env.reset()  # draws a random mixing…
        _set_our_coefs(env, mx[0], mx[1], mx[2], mx[3])  # …which we replace
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
            for i in range(NV):
                var v = Float64(py=data.qvel[i])
                var dv = abs(v - Float64(env.d.qvel.data[i]))
                if dv > max_state:
                    max_state = dv
                if abs(v) > max_qvel:
                    max_qvel = abs(v)

            var obs = out[0]
            for i in range(NQ):
                var d_o = abs(Float64(py=data.qpos[i]) - Float64(obs.data[i]))
                if d_o > max_obs:
                    max_obs = d_o
            for i in range(NV):
                var d_o = abs(
                    Float64(py=data.qvel[i]) - Float64(obs.data[NQ + i])
                )
                if d_o > max_obs:
                    max_obs = d_o

            # reward = near_target * small_control, from MuJoCo's own state
            var dx = Float64(py=data.geom_xpos[5][0]) - Float64(
                py=data.geom_xpos[6][0]
            )
            var dy = Float64(py=data.geom_xpos[5][1]) - Float64(
                py=data.geom_xpos[6][1]
            )
            var dz = Float64(py=data.geom_xpos[5][2]) - Float64(
                py=data.geom_xpos[6][2]
            )
            var dist = sqrt(dx * dx + dy * dy + dz * dz)
            if dist < min_dist:
                min_dist = dist
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

    print("point_mass-hard vs MuJoCo,", len(mixings), "mixings x", N_STEPS,
          "steps:")
    print("  max |d(state)| =", max_state, " |d(obs)| =", max_obs)
    print("  max |d(reward)| =", max_r)
    print("  closest approach =", min_dist, " max |qvel| =", max_qvel)

    assert_true(max_state <= STATE_TOL, "physics deviated under a mixing")
    assert_true(max_obs <= OBS_TOL, "observation deviated")
    assert_true(max_r <= REWARD_TOL, "reward deviated")
    # Non-vacuity: the mass has to have actually been driven. A hook that
    # silently wrote no force at all would match MuJoCo at 0.0 only if MuJoCo
    # were also motionless, so pin that it was not.
    assert_true(
        max_qvel > 0.05,
        "the rollouts barely moved — the transmission gate is near-vacuous",
    )
    assert_true(
        min_dist < 4.0 * TARGET_SIZE,
        "no rollout approached the target — the reward gate is near-vacuous",
    )


def test_point_mass_hard_randomization_matches_reference() raises:
    """Our sampler's distribution, and that the mixing reaches the physics.

    dm_control's draws are unreproducible, so this gates the four properties
    the reference's loop actually guarantees, plus the one that no property of
    the sampler can cover: whether `apply_actions` uses the result. Driving
    control 0 alone must accelerate the mass along `dir1` — under the comptime
    tables it would go along +x every time, whatever the records say.
    """
    var n_eps = 64
    var worst_norm = 0.0
    var worst_dot = 0.0
    var worst_align = 0.0
    var seen_spread = 0.0
    var first_d1x = 0.0

    var env = Env()
    for ep in range(n_eps):
        _ = env.reset()
        var t0 = T1_TENDON_IDX * MODEL_TENDON_SIZE
        var t1 = T2_TENDON_IDX * MODEL_TENDON_SIZE
        var d1x = Float64(env.mf.tendons.data[t0 + TENDON_IDX_COEF_0 + 0])
        var d1y = Float64(env.mf.tendons.data[t0 + TENDON_IDX_COEF_0 + 1])
        var d2x = Float64(env.mf.tendons.data[t1 + TENDON_IDX_COEF_0 + 0])
        var d2y = Float64(env.mf.tendons.data[t1 + TENDON_IDX_COEF_0 + 1])

        # `dir /= np.linalg.norm(dir)` — both unit.
        var n1 = sqrt(d1x * d1x + d1y * d1y)
        var n2 = sqrt(d2x * d2x + d2y * d2y)
        if abs(n1 - 1.0) > worst_norm:
            worst_norm = abs(n1 - 1.0)
        if abs(n2 - 1.0) > worst_norm:
            worst_norm = abs(n2 - 1.0)

        # `while parallel: ... parallel = abs(dot) > .9`
        var dot = abs(d1x * d2x + d1y * d2y)
        if dot > worst_dot:
            worst_dot = dot

        if ep == 0:
            first_d1x = d1x
        var spread = abs(d1x - first_d1x)
        if spread > seen_spread:
            seen_spread = spread

        # Drive control 0 only, from rest at the origin, and check the mass
        # accelerates along dir1. Free of joint limits and of the target, so
        # the only thing steering it is the transmission.
        var qs: List[Float64] = [0.0, 0.0]
        var vs: List[Float64] = [0.0, 0.0]
        env.set_state(qs, vs)
        var act = Env.ActionType()
        act.data[0] = 1.0
        act.data[1] = 0.0
        for _ in range(5):
            _ = env.step(act)
        var vx = Float64(env.d.qvel.data[0])
        var vy = Float64(env.d.qvel.data[1])
        var vn = sqrt(vx * vx + vy * vy)
        assert_true(vn > 1e-9, "control 0 produced no motion at all")
        # cos between the response and dir1 must be 1.
        var align = abs((vx * d1x + vy * d1y) / vn - 1.0)
        if align > worst_align:
            worst_align = align

    print("point_mass-hard sampler over", n_eps, "episodes:")
    print("  max |‖dir‖ - 1| =", worst_norm, " max |dot(dir1, dir2)| =",
          worst_dot)
    print("  max misalignment of the response with dir1 =", worst_align)
    print("  spread of dir1_x across episodes =", seen_spread)

    assert_true(worst_norm <= 1e-15, "a sampled direction is not unit-norm")
    assert_true(
        worst_dot <= PARALLEL_COS + 1e-15,
        "the |dot| > .9 rejection let a near-parallel pair through",
    )
    assert_true(
        worst_align <= 1e-12,
        "the response does not follow dir1 — apply_actions is ignoring the"
        " randomized coefs (this is `easy` wearing `hard`'s name)",
    )
    assert_true(
        seen_spread > 0.5,
        "dir1 barely varied across episodes — the randomizer is stuck",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
