"""dm_control `humanoid` parity: our envs vs MuJoCo + the reference tasks.

Four tasks (stand, walk, run, run_pure_state) over `suite/humanoid.xml`
verbatim. What this exercises beyond every earlier ported domain:

  - `<freejoint>`. humanoid is the first suite model to use the alias, and an
    unrecognized element is not an error — it just silently yields a model
    with no root joint and nq/nv 7/6 short. The dimension check below is the
    gate on `merge_mjcf`'s normalization.
  - JOINT SPRINGS. Every joint inherits `stiffness="1"` from
    `<default class="body">`, with 5/10/20 on the big joints and 3/6 on the
    ankles. The integrators have always assembled a stiffness term but our Gym
    humanoid zeroes it everywhere, so no test has ever loaded it. The model
    check asserts the parsed values and the airborne rollout is what would
    diverge if the FORCE were wrong.
  - three-deep nested `<default>` classes plus `childclass` on the torso.
  - a 27-DOF chain — nearly triple walker's, and the largest model in the
    port.
  - `extremities()`, whose 12 slots are a rotation into the TORSO frame. Get
    the transpose backwards and the shape is still right and the magnitudes
    still plausible; only a value check catches it.

The rollout starts with the torso lifted clear of the floor, and the numbers
are reported over a CONTACT-FREE PREFIX as well as the whole run — the split
walker's test already uses, for the same reason: the contact solver is the one
component known to disagree with MuJoCo at a level that would swamp
everything else, so the tight gate goes on the interval where it is provably
out of the picture.

The prefix is short here (single-digit steps) and that is a property of the
model, not a choice. humanoid SELF-collides: `condim="1"` geoms, no exclusions
beyond MuJoCo's parent-child filter, and gears of 120 on the hips and 80 on
the knees, so any actuation folds a leg into the butt or swings a hand into
the head within ~8 control steps. Zero action stays clean, but then
`small_control` is constant and the actuator path goes untested. The
compromise is deliberate: gate the prefix, report the rest.

Run with:
    pixi run mojo run -I . tests/dm_control/test_humanoid_vs_dm_control.mojo
"""

from std.math import abs, sin, sqrt, inf
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.humanoid import (
    DMHumanoidStand,
    DMHumanoidWalk,
    DMHumanoidRun,
    DMHumanoidRunPureState,
    DMHumanoidModel,
    HUMANOID_OBS_DIM,
    HUMANOID_PURE_OBS_DIM,
    TORSO_BODY_IDX,
    HEAD_BODY_IDX,
    extremity_body_indices,
    STAND_HEIGHT,
    WALK_SPEED,
    RUN_SPEED,
)
from mojo_rl.physics3d.fields import Model
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_IPOS_X,
    BODY_IDX_IXX,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_STIFFNESS,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_DAMPING,
)
from mojo_rl.physics3d.joint_types import JNT_FREE


comptime EnvStand = DMHumanoidStand[DType.float64]
comptime EnvWalk = DMHumanoidWalk[DType.float64]
comptime EnvRun = DMHumanoidRun[DType.float64]
comptime EnvPure = DMHumanoidRunPureState[DType.float64]

comptime REF_XML: StaticString = (
    "references/dm_control-main/dm_control/suite/humanoid.xml"
)
comptime REF_PATH: StaticString = "references/dm_control-main"

comptime NQ: Int = 28
comptime NV: Int = 27
comptime NBODY: Int = 17
comptime NJOINT: Int = 22
comptime NGEOM: Int = 20
comptime NSITE: Int = 25
comptime NACT: Int = 21
# humanoid.xml timestep .005, _CONTROL_TIMESTEP .025 => 5 substeps.
comptime FRAME_SKIP: Int = 5

comptime MODEL_TOL: Float64 = 1e-14
comptime STATE_TOL: Float64 = 1e-8
comptime OBS_TOL: Float64 = 1e-8
comptime REWARD_TOL: Float64 = 1e-9

comptime AMP: Float64 = 0.6
comptime N_STEPS: Int = 60
# The prefix must be long enough to gate something. MuJoCo first self-contacts
# at step 4-5 for every init below, so this is a floor, not a target.
comptime MIN_SMOOTH_STEPS: Int = 4


def _action_at(step: Int, k: Int) -> Float64:
    return AMP * sin(0.07 * Float64(step) + 0.41 * Float64(k))


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


def _build_model() raises -> Model[
    DType.float64,
    DMHumanoidModel.NV,
    DMHumanoidModel.NBODY,
    DMHumanoidModel.NJOINT,
    DMHumanoidModel.NGEOM,
    DMHumanoidModel.MAX_EQUALITY,
    DMHumanoidModel.MAX_TENDON,
    DMHumanoidModel.NSITE,
    DMHumanoidModel.NEXCLUDE,
    0,
]:
    var ctx = DeviceContext()
    var mf = Model[
        DType.float64,
        DMHumanoidModel.NV,
        DMHumanoidModel.NBODY,
        DMHumanoidModel.NJOINT,
        DMHumanoidModel.NGEOM,
        DMHumanoidModel.MAX_EQUALITY,
        DMHumanoidModel.MAX_TENDON,
        DMHumanoidModel.NSITE,
        DMHumanoidModel.NEXCLUDE,
        0,
    ]()
    DMHumanoidModel.init_fields[DType.float64, 0](ctx, mf)
    return mf^


def test_humanoid_model_matches_mujoco() raises:
    """Dims, inertials, joint ranges, and the passive-force parameters.

    The stiffness assertion is the one that matters most: it is a term no
    previously ported model loads at all, and a zero here would make the
    humanoid limp in a way that still looks like plausible physics.
    """
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path(String(REF_XML))

    # `<freejoint>` gate: without normalization nq/nv come out 7/6 short.
    assert_true(Int(py=m.nq) == DMHumanoidModel.NQ, "nq mismatch")
    assert_true(Int(py=m.nv) == DMHumanoidModel.NV, "nv mismatch")
    assert_true(Int(py=m.nbody) == DMHumanoidModel.NBODY, "nbody mismatch")
    assert_true(Int(py=m.njnt) == DMHumanoidModel.NJOINT, "njnt mismatch")
    assert_true(Int(py=m.ngeom) == DMHumanoidModel.NGEOM, "ngeom mismatch")
    assert_true(Int(py=m.nsite) == DMHumanoidModel.NSITE, "nsite mismatch")
    assert_true(Int(py=m.nu) == DMHumanoidModel.nact, "nu mismatch")

    var mf = _build_model()

    # Joint 0 must be the free root, and the body order must be the tree DFS
    # that our body-index comptimes assume.
    assert_true(
        Int(mf.joints.data[0 * MODEL_JOINT_SIZE + JOINT_IDX_TYPE]) == JNT_FREE,
        "joint 0 is not the free root — did <freejoint> normalization break?",
    )
    var named = [
        ("torso", TORSO_BODY_IDX),
        ("head", HEAD_BODY_IDX),
    ]
    for nb in named:
        var ref_id = Int(
            py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nb[0])
        )
        assert_true(ref_id == nb[1], "body index drifted from MuJoCo's")

    # Inertials.
    var worst_mass = 0.0
    var worst_ipos = 0.0
    var worst_inertia = 0.0
    for b in range(NBODY):
        var dm = abs(
            Float64(mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_MASS])
            - Float64(py=m.body_mass[b])
        )
        if dm > worst_mass:
            worst_mass = dm
        for k in range(3):
            var dp = abs(
                Float64(
                    mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_IPOS_X + k]
                )
                - Float64(py=m.body_ipos[b][k])
            )
            if dp > worst_ipos:
                worst_ipos = dp
        for k in range(3):
            var di = abs(
                Float64(
                    mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_IXX + k]
                )
                - Float64(py=m.body_inertia[b][k])
            )
            if di > worst_inertia:
                worst_inertia = di
    print("humanoid model build:")
    print("  max |d(mass)| =", worst_mass, " |d(ipos)| =", worst_ipos,
          " |d(inertia)| =", worst_inertia)
    assert_true(worst_mass <= MODEL_TOL, "masses differ from MuJoCo")
    assert_true(worst_ipos <= MODEL_TOL, "body CoMs differ from MuJoCo")
    assert_true(worst_inertia <= MODEL_TOL, "inertias differ from MuJoCo")

    # Passive-force parameters, per joint. `dof_*` is indexed by DOF, and the
    # free root occupies DOFs 0..5, so hinge j maps to DOF j + 5.
    var worst_stiff = 0.0
    var worst_arm = 0.0
    var worst_damp = 0.0
    var worst_range = 0.0
    var nonzero_stiffness = 0
    for j in range(1, NJOINT):
        var dof = j + 5
        var ds = abs(
            Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_STIFFNESS])
            - Float64(py=m.jnt_stiffness[j])
        )
        if ds > worst_stiff:
            worst_stiff = ds
        if Float64(py=m.jnt_stiffness[j]) != 0.0:
            nonzero_stiffness += 1
        var da = abs(
            Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_ARMATURE])
            - Float64(py=m.dof_armature[dof])
        )
        if da > worst_arm:
            worst_arm = da
        var dd = abs(
            Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_DAMPING])
            - Float64(py=m.dof_damping[dof])
        )
        if dd > worst_damp:
            worst_damp = dd
        var dlo = abs(
            Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN])
            - Float64(py=m.jnt_range[j][0])
        )
        var dhi = abs(
            Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MAX])
            - Float64(py=m.jnt_range[j][1])
        )
        if dlo > worst_range:
            worst_range = dlo
        if dhi > worst_range:
            worst_range = dhi
    print("  max |d(stiffness)| =", worst_stiff, " over", nonzero_stiffness,
          "springs;  |d(armature)| =", worst_arm,
          " |d(damping)| =", worst_damp)
    print("  max |d(jnt_range)| =", worst_range)
    # 19 of the 21 hinges carry a spring; the two elbows are explicitly
    # `stiffness="0"`. The check is that springs EXIST at all — before the
    # nested-default fix every class-named joint reported 0.
    assert_true(
        nonzero_stiffness >= 15,
        "MuJoCo reports almost no joint springs — wrong model?",
    )
    assert_true(worst_stiff <= MODEL_TOL, "joint stiffness differs")
    assert_true(worst_arm <= MODEL_TOL, "armature differs")
    assert_true(worst_damp <= MODEL_TOL, "damping differs")
    assert_true(
        worst_range <= MODEL_TOL,
        "joint ranges differ — degree->radian conversion missing?",
    )


def test_humanoid_dynamics_vs_dm_control() raises:
    """Physics / observation / reward parity over the contact-free prefix.

    This was print-only while three engine bugs the port surfaced were open.
    All three are fixed (2026-07-30) and it now asserts:

      1. `_vel_body` rotated every joint axis by the PARENT body's quat,
         valid only for a body with one joint and no fixed `quat=`. Fixed by
         walking the running frame MuJoCo builds in `mj_kinematics`; gated
         against `mj_objectVelocity` at 1e-10 in
         tests/physics3d/test_body_velocities_vs_mujoco.mojo. `com_velocity`
         was out by ~0.37 before the state had drifted at all; `|d(obs)|` now
         equals `|d(state)|` exactly, i.e. the obs adds no error of its own.
      2. `_parse_quat` did not normalize, and `lower_waist` carries
         `quat="1.000 0 -.002 0"` (norm 1.000002), scaling every vector that
         quat rotated by |q|^2.
      3. `quat_integrate` was a FIRST-ORDER approximation where MuJoCo's
         `mju_quatIntegrate` is the exact exponential map. The free root's
         quaternion was the largest single state error at every step; fixing
         it took step-1 qpos from 1.1e-8 to 2.0e-12 and these rewards from
         ~2e-6 to ~1e-8.

    The one remaining gap is a joint LIMIT constraint — see the state gate
    below for why it is budgeted rather than tightened.
    """
    var handle = _setup()
    var mujoco = handle[0]
    var model = handle[1]
    var data = handle[2]
    var tol = handle[3]

    var svl_adr = Int(
        py=model.sensor_adr[
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_subtreelinvel"
            )
        ]
    )

    var max_state = 0.0
    var max_obs = 0.0
    var max_obs_pure = 0.0
    var max_r_stand = 0.0
    var max_r_walk = 0.0
    var max_r_run = 0.0
    # `_s` = accumulated only while the reference reports zero contacts.
    var max_state_s = 0.0
    var max_obs_s = 0.0
    var max_obs_pure_s = 0.0
    var max_r_stand_s = 0.0
    var max_r_walk_s = 0.0
    var max_r_run_s = 0.0
    var min_smooth = N_STEPS
    var max_ncon = 0
    var r_stand_lo = 1e9
    var r_stand_hi = -1e9

    # qpos = [x, y, z, qw, qx, qy, qz, 21 joint angles]. z = 3 keeps every
    # geom clear of the floor for the whole rollout (the humanoid is ~1.9 m
    # tall from torso origin to foot). The three orientations are deliberately
    # far apart so `upright` (xmat zz) sweeps from ~1 to negative and the
    # `standing` term sees head heights on both sides of 1.4.
    var quats = [
        [1.0, 0.0, 0.0, 0.0],
        [0.9239, 0.3827, 0.0, 0.0],
        [0.7071, 0.0, 0.7071, 0.0],
    ]
    var joint_seeds = [0.0, 0.25, -0.3]

    for t in range(3):
        var quat = quats[t].copy()
        var seed = joint_seeds[t]

        mujoco.mj_resetData(model, data)
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = 3.0
        for k in range(4):
            data.qpos[3 + k] = quat[k]
        for i in range(7, NQ):
            data.qpos[i] = seed * sin(0.7 * Float64(i))
        mujoco.mj_forward(model, data)

        var qs = List[Float64]()
        var vs = List[Float64]()
        qs.append(0.0)
        qs.append(0.0)
        qs.append(3.0)
        for k in range(4):
            qs.append(quat[k])
        for i in range(7, NQ):
            qs.append(seed * sin(0.7 * Float64(i)))
        for _ in range(NV):
            vs.append(0.0)

        var e_stand = EnvStand()
        _ = e_stand.reset()
        e_stand.set_state(qs, vs)
        var e_walk = EnvWalk()
        _ = e_walk.reset()
        e_walk.set_state(qs, vs)
        var e_run = EnvRun()
        _ = e_run.reset()
        e_run.set_state(qs, vs)
        var e_pure = EnvPure()
        _ = e_pure.reset()
        e_pure.set_state(qs, vs)

        var smooth = True
        var smooth_steps = 0
        for step in range(N_STEPS):
            var a_stand = EnvStand.ActionType()
            var a_walk = EnvWalk.ActionType()
            var a_run = EnvRun.ActionType()
            var a_pure = EnvPure.ActionType()
            for k in range(NACT):
                var a = _action_at(step, k)
                data.ctrl[k] = a
                a_stand.data[k] = a
                a_walk.data[k] = a
                a_run.data[k] = a
                a_pure.data[k] = a
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            var o_stand = e_stand.step(a_stand)
            var o_walk = e_walk.step(a_walk)
            var o_run = e_run.step(a_run)
            var o_pure = e_pure.step(a_pure)

            var ncon = Int(py=data.ncon)
            if ncon > max_ncon:
                max_ncon = ncon
            if ncon > 0:
                smooth = False
            if smooth:
                smooth_steps += 1

            for i in range(NQ):
                var dq = abs(
                    Float64(py=data.qpos[i])
                    - Float64(e_stand.d.qpos.data[i])
                )
                if dq > max_state:
                    max_state = dq
                if smooth and dq > max_state_s:
                    max_state_s = dq
            for i in range(NV):
                var dv = abs(
                    Float64(py=data.qvel[i])
                    - Float64(e_stand.d.qvel.data[i])
                )
                if dv > max_state:
                    max_state = dv
                if smooth and dv > max_state_s:
                    max_state_s = dv

            # ── the reference observation, rebuilt from MuJoCo ──
            var ref_obs = List[Float64]()
            for i in range(7, NQ):  # joint_angles()
                ref_obs.append(Float64(py=data.qpos[i]))
            var head_z = Float64(py=data.xpos[HEAD_BODY_IDX][2])
            ref_obs.append(head_z)  # head_height()

            # extremities(): (limb - torso) . xmat[torso], i.e. R^T v.
            var tp = [
                Float64(py=data.xpos[TORSO_BODY_IDX][0]),
                Float64(py=data.xpos[TORSO_BODY_IDX][1]),
                Float64(py=data.xpos[TORSO_BODY_IDX][2]),
            ]
            var rm = List[Float64]()
            for k in range(9):
                rm.append(Float64(py=data.xmat[TORSO_BODY_IDX][k]))
            var limbs = extremity_body_indices()
            for li in range(len(limbs)):
                var b = limbs[li]
                var v0 = Float64(py=data.xpos[b][0]) - tp[0]
                var v1 = Float64(py=data.xpos[b][1]) - tp[1]
                var v2 = Float64(py=data.xpos[b][2]) - tp[2]
                for c in range(3):
                    ref_obs.append(
                        v0 * rm[0 * 3 + c]
                        + v1 * rm[1 * 3 + c]
                        + v2 * rm[2 * 3 + c]
                    )
            for k in range(6, 9):  # torso_vertical: zx, zy, zz
                ref_obs.append(rm[k])
            var com = [
                Float64(py=data.sensordata[svl_adr + 0]),
                Float64(py=data.sensordata[svl_adr + 1]),
                Float64(py=data.sensordata[svl_adr + 2]),
            ]
            for k in range(3):
                ref_obs.append(com[k])
            for i in range(NV):  # velocity()
                ref_obs.append(Float64(py=data.qvel[i]))

            assert_true(
                len(ref_obs) == HUMANOID_OBS_DIM,
                "the reference observation is not 67 wide",
            )
            var obs = o_stand[0]
            for i in range(HUMANOID_OBS_DIM):
                var d_o = abs(ref_obs[i] - Float64(obs.data[i]))
                if d_o > max_obs:
                    max_obs = d_o
                if smooth and d_o > max_obs_s:
                    max_obs_s = d_o

            # pure_state: qpos then qvel, whole.
            var obs_p = o_pure[0]
            for i in range(NQ):
                var d_p = abs(
                    Float64(py=data.qpos[i]) - Float64(obs_p.data[i])
                )
                if d_p > max_obs_pure:
                    max_obs_pure = d_p
                if smooth and d_p > max_obs_pure_s:
                    max_obs_pure_s = d_p
            for i in range(NV):
                var d_p = abs(
                    Float64(py=data.qvel[i]) - Float64(obs_p.data[NQ + i])
                )
                if d_p > max_obs_pure:
                    max_obs_pure = d_p
                if smooth and d_p > max_obs_pure_s:
                    max_obs_pure_s = d_p

            # ── the reference rewards ──
            var standing = Float64(
                py=tol(
                    head_z,
                    STAND_HEIGHT,
                    Float64(py=Python.evaluate("float('inf')")),
                    STAND_HEIGHT / 4.0,
                    String("gaussian"),
                    0.1,
                )
            )
            var upright = Float64(
                py=tol(
                    rm[8],
                    0.9,
                    Float64(py=Python.evaluate("float('inf')")),
                    1.9,
                    String("linear"),
                    0.0,
                )
            )
            var stand_reward = standing * upright
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
            var small_control = (4.0 + acc / Float64(NACT)) / 5.0

            # stand: dont_move is a MEAN over vx, vy scored separately.
            var dm0 = Float64(
                py=tol(com[0], 0.0, 0.0, 2.0, String("gaussian"), 0.1)
            )
            var dm1 = Float64(
                py=tol(com[1], 0.0, 0.0, 2.0, String("gaussian"), 0.1)
            )
            var r_stand = small_control * stand_reward * (dm0 + dm1) / 2.0
            var d_rs = abs(r_stand - Float64(o_stand[1]))
            if d_rs > max_r_stand:
                max_r_stand = d_rs
            if smooth:
                if d_rs > max_r_stand_s:
                    max_r_stand_s = d_rs
                if r_stand < r_stand_lo:
                    r_stand_lo = r_stand
                if r_stand > r_stand_hi:
                    r_stand_hi = r_stand

            # walk / run: `move` uses the NORM of the horizontal velocity.
            var speed = sqrt(com[0] * com[0] + com[1] * com[1])
            var mv_w = Float64(
                py=tol(
                    speed,
                    WALK_SPEED,
                    Float64(py=Python.evaluate("float('inf')")),
                    WALK_SPEED,
                    String("linear"),
                    0.0,
                )
            )
            var r_walk = small_control * stand_reward * (5.0 * mv_w + 1.0) / 6.0
            var d_rw = abs(r_walk - Float64(o_walk[1]))
            if d_rw > max_r_walk:
                max_r_walk = d_rw
            if smooth and d_rw > max_r_walk_s:
                max_r_walk_s = d_rw

            var mv_r = Float64(
                py=tol(
                    speed,
                    RUN_SPEED,
                    Float64(py=Python.evaluate("float('inf')")),
                    RUN_SPEED,
                    String("linear"),
                    0.0,
                )
            )
            var r_run = small_control * stand_reward * (5.0 * mv_r + 1.0) / 6.0
            var d_rr = abs(r_run - Float64(o_run[1]))
            if d_rr > max_r_run:
                max_r_run = d_rr
            if smooth and d_rr > max_r_run_s:
                max_r_run_s = d_rr

        if smooth_steps < min_smooth:
            min_smooth = smooth_steps

    print("humanoid vs MuJoCo, 3 x", N_STEPS, "steps:")
    print("  contact-free prefix: shortest =", min_smooth, "steps;",
          " reference max ncon over the full run =", max_ncon)
    print("  PREFIX  max |d(state)| =", max_state_s,
          " |d(obs)| =", max_obs_s, " |d(obs_pure)| =", max_obs_pure_s)
    print("  PREFIX  max |d(reward)| stand =", max_r_stand_s,
          " walk =", max_r_walk_s, " run =", max_r_run_s)
    print("  FULL    max |d(state)| =", max_state,
          " |d(obs)| =", max_obs, " |d(obs_pure)| =", max_obs_pure)
    print("  FULL    max |d(reward)| stand =", max_r_stand,
          " walk =", max_r_walk, " run =", max_r_run)
    print("  stand reward range over the prefix =", r_stand_lo, "..",
          r_stand_hi)

    # ── Gates ────────────────────────────────────────────────────────────
    # Rewards are what the tasks actually consume. Over the contact-free
    # prefix they agree to ~1e-11 (was ~1e-8 before the solimp clamp landed;
    # see the state gate below). Bounds are ~100x looser than observed so they
    # track real regressions, not platform noise.
    assert_true(
        max_r_stand_s < 1e-8, "stand reward diverged over the prefix"
    )
    assert_true(max_r_walk_s < 1e-8, "walk reward diverged over the prefix")
    assert_true(max_r_run_s < 1e-8, "run reward diverged over the prefix")

    # A reward that never moves would pass the gates above vacuously.
    assert_true(
        r_stand_hi - r_stand_lo > 0.3,
        "stand reward is degenerate over the prefix — gate is vacuous",
    )

    # RESOLVED 2026-07-30. This bound was 5e-4 and carried a note calling its
    # own tightening "the acceptance test for constraint softness". The
    # softness was real and it was the joint-limit IMPEDANCE, not the solver:
    # MuJoCo clamps both ends of solimp to [mjMINIMP, mjMAXIMP] = [1e-4,
    # 0.9999] BEFORE interpolating (engine_core_constraint.c:1284-1287), and
    # we clamped only dmax while flooring the interpolated `imp` at 1e-6 —
    # 100x below MuJoCo's floor. humanoid's joints carry
    # `solimplimit="0 .99 .01"`, so dmin IS 0 and every shallow limit
    # violation got a force ~100x too soft. Prefix |d(state)| 7.7259e-05 ->
    # 5.66e-08.
    #
    # Why it resisted diagnosis for so long: every probe swept ANT, whose
    # solimplimit is the DEFAULT (dmin=0.9), so the defect was structurally
    # absent from the model under test — ant's limits genuinely did match to
    # 1e-13. It also only bites at SHALLOW penetration (deep violations
    # saturate to dmax, where both engines agree), which is why one humanoid
    # init held 1.7e-9 while the others sat at 6.5e-5.
    #
    # Bound is ~50x the observed value, per this file's convention.
    assert_true(
        max_state_s < 3e-6,
        "humanoid prefix state diverged beyond the joint-limit budget",
    )

    # NOT gated: the FULL run. Once feet contact the floor the trajectories
    # separate outright (|d(state)| ~ 30), which is the contact solver, not
    # this domain.


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
