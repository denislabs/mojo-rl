"""dm_control `quadruped` walk/run parity: our env vs MuJoCo + the reference task.

The engine side of this domain is gated by
`tests/physics3d/test_rne_post_sensors_vs_mujoco.mojo` (rne_post, the
accelerometer and the force/torque sensors, against MuJoCo's own sensordata).
This file gates the TASK on top of it: the sensor ordering the observation
assumes, the observation layout, and the reward.

WHAT IS AND IS NOT COMPARABLE HERE.

* Sensor ORDER is asserted against the reference model's own `sensor_type` /
  `sensor_adr` tables. `physics.imu()` and `physics.force_torque()` build
  their name lists with `np.where(np.isin(...))`, i.e. in sensor-ID order, so
  a `_TOES`-ordered implementation would transpose twelve dims and no
  numerical check on a symmetric pose would notice.
* The OBSERVATION is compared elementwise against MuJoCo's `sensordata` and
  `qpos`/`qvel` at the same state, with the reference's own `arcsinh` and
  layout applied on the Python side.
* The REWARD is compared against `dm_control.utils.rewards.tolerance` —
  importable from the reference tree without dm_control being installed.
* The RESET is not reproducible: `Move.initialize_episode` draws a random
  orientation from its own RandomState. What IS checked is the two properties
  that make the draw correct — unit norm, and a height that leaves the model
  clear of the floor.

CONTACT-FREE STATES ONLY for the numeric comparisons — the loaded case is
gated by test_rne_post_sensors_vs_mujoco's standing test (now at 5e-11 on
every force/torque component), and repeating it here would only duplicate
that pin.

Run: pixi run mojo run -I . tests/dm_control/test_quadruped_vs_dm_control.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs, sqrt, inf
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.quadruped import (
    DMQuadrupedWalk,
    DMQuadrupedRun,
    DMQuadrupedWalkConfig,
    DMQuadrupedRunConfig,
    DMQuadrupedWalkModel,
    QUADRUPED_OBS_DIM,
    QUADRUPED_WALK_SPEED,
    QUADRUPED_RUN_SPEED,
    N_HINGE,
    HINGE_QPOS_0,
    HINGE_DOF_0,
)
from mojo_rl.physics3d.fields import Model
from mojo_rl.physics3d.constants import (
    GEOM_PLANE, GEOM_SPHERE, GEOM_CAPSULE, GEOM_BOX, GEOM_CYLINDER,
    GEOM_MESH, GEOM_ELLIPSOID,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS, BODY_IDX_IXX, BODY_IDX_IYY, BODY_IDX_IZZ,
    BODY_IDX_POS_X, BODY_IDX_QUAT_X, BODY_IDX_PARENT,
    BODY_IDX_IPOS_X, BODY_IDX_IQUAT_X, BODY_IDX_ROOTID,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE, JOINT_IDX_BODY_ID, JOINT_IDX_QPOS_ADR, JOINT_IDX_DOF_ADR,
    JOINT_IDX_POS_X, JOINT_IDX_AXIS_X,
    JOINT_IDX_RANGE_MIN, JOINT_IDX_RANGE_MAX,
    JOINT_IDX_ARMATURE, JOINT_IDX_DAMPING, JOINT_IDX_STIFFNESS,
    JOINT_IDX_SPRINGREF, JOINT_IDX_FRICTIONLOSS,
    JOINT_IDX_SOLREF_LIMIT_0, JOINT_IDX_SOLIMP_LIMIT_0, JOINT_IDX_QPOS0,
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE, GEOM_IDX_BODY, GEOM_IDX_POS_X, GEOM_IDX_QUAT_X,
    GEOM_IDX_RADIUS, GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X, GEOM_IDX_HALF_Y, GEOM_IDX_HALF_Z,
    GEOM_IDX_FRICTION, GEOM_IDX_CONTYPE, GEOM_IDX_CONAFFINITY,
    GEOM_IDX_CONDIM, GEOM_IDX_RBOUND, GEOM_IDX_SOLREF_0, GEOM_IDX_SOLIMP_0,
    GEOM_IDX_MARGIN,
    MODEL_SITE_SIZE,
    SITE_IDX_BODY, SITE_IDX_POS_X, SITE_IDX_TYPE, SITE_IDX_SIZE_0,
    SITE_IDX_QUAT_X,
    MODEL_TENDON_SIZE,
    TENDON_IDX_NUM_JOINTS, TENDON_IDX_JOINT_0, TENDON_IDX_COEF_0,
    TENDON_IDX_INVWEIGHT0, TENDON_IDX_KIND, TENDON_IDX_IS_EQUALITY,
    TENDON_KIND_FIXED,
    TENDON_IDX_SOLREF_0, TENDON_IDX_SOLIMP_0,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_ACTUATOR_SIZE,
    ACT_IDX_GEAR,
    ACT_IDX_KP,
    ACT_IDX_KV,
    ACT_IDX_CTRL_MIN,
    ACT_IDX_CTRL_MAX,
    ACT_IDX_TRN_N,
    ACT_IDX_DYN_TAU,
    ACT_IDX_ACT_ADR,
    ACT_IDX_TRN_DADR_0,
    ACT_IDX_TRN_COEF_0,
)

# ⚠ THE WRAP STRIDE IS A CONSTANT, NOT A LITERAL. These tables are
# `[actuator * MAX_COMPTIME_TENDON_WRAPS + k]`; the cap moved 4 -> 16
# with defect 17 and a hardcoded 4 here silently reads the wrong slot.

comptime REF_PATH: StaticString = "references/dm_control-main"

comptime NQ: Int = 23
comptime NV: Int = 22
comptime NU: Int = 12
comptime NA: Int = 12
comptime NBODY: Int = 18
comptime NJOINT: Int = 17
comptime NGEOM: Int = 20
comptime NSITE: Int = 29
comptime NTEN: Int = 12
comptime NEQ: Int = 4

# mjtSensor: accelerometer 1, gyro 3, force 4, torque 5, velocimeter 2.
comptime SENS_ACC: Int = 1
comptime SENS_VEL: Int = 2
comptime SENS_GYRO: Int = 3
comptime SENS_FORCE: Int = 4
comptime SENS_TORQUE: Int = 5

# ⚠ RE-PINNED 2026-08-03 after the quaternion-normalizer fix (see FLIGHT_TOL
# and friends in tests/physics3d/test_rne_post_sensors_vs_mujoco.mojo).
# `quat_math.mojo` normalized as `1/sqrt(norm_sq + 1e-10)`, leaving every body
# quaternion 5e-11 short of unit, and the whole observation inherited it.
#
# ⚠ THE OBSERVATION DID NOT COLLAPSE TO MACHINE PRECISION LIKE EVERYTHING
# ELSE. It went 2.34e-9 -> 5.52e-10 at dim 54, a factor of 4, where the
# sensors themselves now agree to 4.07e-15 and the reward is bit-exact. So
# dim 54 carries a SECOND residual with a different cause, and this bound is
# set to keep watch on it rather than to certify it. Do not read the pass as
# "the observation matches to 1e-15" — it does not, and that is a live thread.
comptime OBS_TOL: Float64 = 2e-9
# The reward reads the velocimeter. It used to inherit the ~1e-10 FK gap
# (observed 2.0e-12); with that gone it is now BIT-EXACT against the
# reference for both walk and run.
comptime REWARD_TOL: Float64 = 1e-14


def _mj(state_qpos: List[Float64], state_qvel: List[Float64]) raises -> Tuple[
    PythonObject, PythonObject, PythonObject
]:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path("mojo_rl/envs/dm_control/assets/quadruped_walk.xml")
    var dat = mujoco.MjData(m)
    for i in range(NQ):
        dat.qpos[i] = state_qpos[i]
    for i in range(NV):
        dat.qvel[i] = state_qvel[i]
    for i in range(NA):
        dat.act[i] = 0.0
    for i in range(NU):
        dat.ctrl[i] = 0.0
    mujoco.mj_forward(m, dat)
    return (mujoco, m, dat)


def _sens(
    mujoco: PythonObject, m: PythonObject, dat: PythonObject, name: String
) raises -> List[Float64]:
    """One named sensor's slice of `sensordata`.

    Module level, not a nested closure: a `def` inside a test cannot infer a
    capture convention for a `PythonObject`.
    """
    var sid = Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name))
    assert_true(sid >= 0, "no such sensor: " + name)
    var adr = Int(py=m.sensor_adr[sid])
    var dim = Int(py=m.sensor_dim[sid])
    var out = List[Float64]()
    for k in range(dim):
        out.append(Float64(py=dat.sensordata[adr + k]))
    return out^


def test_sensor_order_matches_the_observation_layout() raises:
    """The ordering `quadruped_config` hard-codes IS the reference's.

    `imu` is accelerometer-then-gyro and `force_torque` is all four forces
    then all four torques, each FL, FR, BR, BL — because both come out of a
    sensor-ID sort. Nothing in the XML guarantees that beyond declaration
    order, and a wrong order is a silent transposition.
    """
    print("--- quadruped: sensor ordering ---")
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path("mojo_rl/envs/dm_control/assets/quadruped_walk.xml")

    assert_true(Int(py=m.nq) == NQ, "nq")
    assert_true(Int(py=m.nv) == NV, "nv")
    assert_true(Int(py=m.nu) == NU, "nu")
    assert_true(Int(py=m.na) == NA, "na — the twelve filter activations")

    # imu: the accelerometer must sort BEFORE the gyro.
    var acc_id = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
    )
    var gyro_id = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    )
    assert_true(
        acc_id < gyro_id,
        "imu() sorts by sensor id, so the accelerometer must be declared"
        " first — the config emits accel then gyro",
    )
    assert_true(Int(py=m.sensor_type[acc_id]) == SENS_ACC, "accel type")
    assert_true(Int(py=m.sensor_type[gyro_id]) == SENS_GYRO, "gyro type")

    # force_torque: four forces, then four torques, each in FL FR BR BL.
    var toes = [
        String("toe_front_left"), String("toe_front_right"),
        String("toe_back_right"), String("toe_back_left"),
    ]
    var prev = -1
    for t in toes:
        var sid = Int(
            py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "force_" + t)
        )
        assert_true(Int(py=m.sensor_type[sid]) == SENS_FORCE, "force type")
        assert_true(sid > prev, "force sensors are not in FL FR BR BL order")
        prev = sid
    var first_torque = prev
    for t in toes:
        var sid = Int(
            py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "torque_" + t)
        )
        assert_true(Int(py=m.sensor_type[sid]) == SENS_TORQUE, "torque type")
        assert_true(sid > prev, "torque sensors are not in FL FR BR BL order")
        prev = sid
    assert_true(
        first_torque < prev,
        "every force sensor must sort before every torque sensor",
    )
    print("  PASS: imu = accel,gyro; force_torque = 4 forces then 4 torques")


def _airborne_state() raises -> Tuple[List[Float64], List[Float64]]:
    """Tumbling, well clear of the floor, hinges inside their ranges."""
    var qpos = List[Float64]()
    for _ in range(NQ):
        qpos.append(0.0)
    qpos[2] = 3.0
    qpos[3] = 0.9762960071199334  # 25 deg about y, w first
    qpos[5] = 0.2164396139381029
    for leg in range(4):
        var sign = 1.0 if (leg % 2) == 0 else -1.0
        qpos[HINGE_QPOS_0 + 4 * leg + 0] = 0.06 * sign
        qpos[HINGE_QPOS_0 + 4 * leg + 1] = 0.15 * sign
        qpos[HINGE_QPOS_0 + 4 * leg + 2] = -0.15 * sign

    var qvel = List[Float64]()
    for _ in range(NV):
        qvel.append(0.0)
    qvel[0] = 0.35
    qvel[2] = -0.2
    qvel[3] = 0.65
    qvel[5] = 0.45
    for leg in range(4):
        var sign = 1.0 if (leg % 2) == 0 else -1.0
        qvel[HINGE_DOF_0 + 4 * leg + 0] = 0.25 * sign
        qvel[HINGE_DOF_0 + 4 * leg + 1] = 0.5 * sign
        qvel[HINGE_DOF_0 + 4 * leg + 2] = -0.5 * sign
    return (qpos^, qvel^)


def test_observation_matches_common_observations() raises:
    """All 78 dims, against MuJoCo's sensordata with the reference layout."""
    print("--- quadruped: observation vs _common_observations ---")
    var st = _airborne_state()
    var qpos = st[0].copy()
    var qvel = st[1].copy()

    # FRAME_SKIP = 1 on purpose. `rne_post` runs inside a SUBSTEP, so with the
    # task's real frame_skip of 4 the acceleration-stage sensors would be
    # filled at the fourth substep — three substeps past the state we pinned,
    # and the accelerometer would be compared against the wrong instant. One
    # substep makes "the state the step started from" and "the state MuJoCo's
    # mj_forward saw" the same thing.
    var env = DMQuadrupedWalk[DType.float64](DeviceContext(), 1000, 1)
    _ = env.reset()
    env.set_state(qpos, qvel)
    var a = type_of(env).ActionType()
    _ = env.step(a)
    env.set_state(qpos, qvel)
    var obs = env.get_obs_list()
    assert_true(
        len(obs) == QUADRUPED_OBS_DIM, "observation is not 78 long"
    )

    var mj = _mj(qpos, qvel)
    var mujoco = mj[0]
    var m = mj[1]
    var dat = mj[2]
    assert_true(Int(py=dat.ncon) == 0, "state must be contact-free")

    var np = Python.import_module("numpy")

    # Build the reference observation in the reference's own order.
    var want = List[Float64]()
    for k in range(N_HINGE):
        want.append(Float64(py=dat.qpos[HINGE_QPOS_0 + k]))
    for k in range(N_HINGE):
        want.append(Float64(py=dat.qvel[HINGE_DOF_0 + k]))
    for k in range(NA):
        want.append(Float64(py=dat.act[k]))
    var vel = _sens(mujoco, m, dat, "velocimeter")
    for k in range(3):
        want.append(vel[k])
    want.append(Float64(py=dat.xmat[1][8]))  # torso zz
    var acc = _sens(mujoco, m, dat, "imu_accel")
    var gyr = _sens(mujoco, m, dat, "imu_gyro")
    for k in range(3):
        want.append(acc[k])
    for k in range(3):
        want.append(gyr[k])
    var toes = [
        String("toe_front_left"), String("toe_front_right"),
        String("toe_back_right"), String("toe_back_left"),
    ]
    for t in toes:
        var f = _sens(mujoco, m, dat, "force_" + t)
        for k in range(3):
            want.append(Float64(py=np.arcsinh(f[k])))
    for t in toes:
        var tq = _sens(mujoco, m, dat, "torque_" + t)
        for k in range(3):
            want.append(Float64(py=np.arcsinh(tq[k])))

    assert_true(len(want) == QUADRUPED_OBS_DIM, "reference obs is not 78")

    var worst = Float64(0)
    var worst_i = 0
    for i in range(QUADRUPED_OBS_DIM):
        var e = abs(Float64(obs[i]) - want[i])
        if e > worst:
            worst = e
            worst_i = i
    print("  worst |obs err| =", worst, "at dim", worst_i)
    print("    ours =", Float64(obs[worst_i]), " ref =", want[worst_i])
    assert_true(worst < OBS_TOL, "observation diverges from dm_control")
    print("  PASS: all", QUADRUPED_OBS_DIM, "dims")


def test_reward_matches_move() raises:
    """`Move.get_reward` = `_upright_reward * tolerance(velocimeter_x, ...)`,
    against the reference's own `rewards.tolerance`."""
    print("--- quadruped: reward vs Move.get_reward ---")
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var rw = Python.import_module("dm_control.utils.rewards")
    # `bounds` is a Python tuple; a Mojo Tuple will not convert, so build it
    # with a lambda that takes the two runtime floats.
    var mk_bounds = Python.evaluate("lambda a, b: (a, b)")
    var py_inf = Python.evaluate("float('inf')")

    var st = _airborne_state()
    var qpos = st[0].copy()
    var qvel = st[1].copy()

    var mj = _mj(qpos, qvel)
    var mujoco = mj[0]
    var m = mj[1]
    var dat = mj[2]
    var sid = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "velocimeter")
    )
    var vx = Float64(py=dat.sensordata[Int(py=m.sensor_adr[sid])])
    var zz = Float64(py=dat.xmat[1][8])

    for which in range(2):
        var speed = QUADRUPED_WALK_SPEED if which == 0 else QUADRUPED_RUN_SPEED
        var label = "walk" if which == 0 else "run"

        var upright = Float64(
            py=rw.tolerance(
                zz, mk_bounds(1.0, py_inf),
                sigmoid="linear", margin=2.0, value_at_margin=0.0,
            )
        )
        var move = Float64(
            py=rw.tolerance(
                vx, mk_bounds(speed, py_inf),
                margin=speed, value_at_margin=0.5, sigmoid="linear",
            )
        )
        var want = upright * move

        # The reward hook is a static method on the config, so it can be
        # called directly at a pinned state — no need to infer it from a
        # `step()` that would also advance the physics.
        var got: Float64
        if which == 0:
            var env = DMQuadrupedWalk[DType.float64]()
            _ = env.reset()
            env.set_state(qpos, qvel)
            var res = DMQuadrupedWalkConfig.compute_reward_and_done_cpu(
                env.d, env.mf.bodies.data, env.mf.joints.data,
                env.mf.geoms.data, env.mf.sites.data,
                Scalar[DType.float64](0), List[Float64](), 0, 4,
            )
            got = Float64(res[0])
        else:
            var env = DMQuadrupedRun[DType.float64]()
            _ = env.reset()
            env.set_state(qpos, qvel)
            var res = DMQuadrupedRunConfig.compute_reward_and_done_cpu(
                env.d, env.mf.bodies.data, env.mf.joints.data,
                env.mf.geoms.data, env.mf.sites.data,
                Scalar[DType.float64](0), List[Float64](), 0, 4,
            )
            got = Float64(res[0])

        print("  ", label, ": ours =", got, " ref =", want,
              " (upright", upright, "move", move, ")")
        assert_true(
            abs(got - want) < REWARD_TOL,
            "reward diverges from Move.get_reward",
        )
    print("  PASS: both speeds")


def test_reset_draws_a_unit_quaternion_and_clears_the_floor() raises:
    """`Move.initialize_episode`'s two checkable properties.

    The draw itself is not reproducible across engines (dm_control uses its
    own RandomState), so this asserts what the draw must SATISFY rather than
    what it returns: a unit quaternion, and a height that leaves the model
    clear of the floor. Over many resets it also asserts the orientation is
    not constant, which a broken RNG or a dropped write would make it.
    """
    print("--- quadruped: reset ---")
    var env = DMQuadrupedWalk[DType.float64]()
    var worst_norm = Float64(0)
    var min_z = Float64(1e9)
    var max_z = Float64(-1e9)
    var first_qw = Float64(0)
    var qw_spread = Float64(0)
    for r in range(24):
        _ = env.reset()
        var qw = Float64(env.d.qpos.data[3])
        var qx = Float64(env.d.qpos.data[4])
        var qy = Float64(env.d.qpos.data[5])
        var qz = Float64(env.d.qpos.data[6])
        var n = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        worst_norm = max(worst_norm, abs(n - 1.0))
        var z = Float64(env.d.qpos.data[2])
        min_z = min(min_z, z)
        max_z = max(max_z, z)
        if r == 0:
            first_qw = qw
        else:
            qw_spread = max(qw_spread, abs(qw - first_qw))
    print("  worst |‖q‖-1| =", worst_norm,
          "  z in [", min_z, ",", max_z, "]",
          "  qw spread =", qw_spread)
    assert_true(worst_norm < 1e-12, "reset orientation is not a unit quaternion")
    assert_true(min_z > 0.0, "reset left the model at or below the floor")
    assert_true(
        max_z < 2.0, "the height search ran away — 2 m is far past any pose"
    )
    assert_true(
        qw_spread > 1e-3, "the orientation is not being randomized at all"
    )
    print("  PASS")


# ── model parity (task #16) ──────────────────────────────────────────────────
#
# TWO SEPARATE COMPARISONS, and conflating them is how the first bug below
# hid. Layer 1 compiles our XML STRING with MuJoCo and diffs it against the
# reference model built by `quadruped_ref.py`: both sides are MuJoCo, so
# nothing of ours but the XML text is in the loop. Layer 2 diffs our
# `fields.Model` against MuJoCo compiled from that same string, which is
# then a test of OUR PARSER alone.
#
# Layer 1 found one on its first run. `<freejoint name="root"/>` is rewritten
# to `<joint type="free" .../>` by `parser/xml_parser.mojo::_normalize_freejoint`,
# which pinned the passive scalars to zero but not `solimplimit` — so the root
# inherited `"0 .99 .01"` from `<default class="body">` where MuJoCo's
# `mjs_addFreeJoint` ("create free joint without defaults") keeps the global
# `(.9 .95 .001 .5 2)`. Inert here, because a free joint is never `limited` and
# nothing reads a limit's solimp for it — but the same omission covered `ref`,
# which would NOT have been inert, and five other quadruped gates had run over
# it without noticing.

comptime Mdl = DMQuadrupedWalkModel
# Parameterised from the model def, not from the literals above — the two are
# tied together by `test_quadruped_dims_match_the_model_def`, and `init_fields`
# will not accept a `Model` whose parameters it did not compute.
comptime Mod = Model[
    DType.float64, Mdl.NV, Mdl.NBODY, Mdl.NJOINT, Mdl.NGEOM,
    Mdl.MAX_EQUALITY, Mdl.MAX_TENDON, Mdl.NSITE, Mdl.NEXCLUDE, 0
]

# Relative bound on every model constant. These are PARSED, not computed, so
# anything above rounding is a real divergence; `invweight0` is the exception
# and carries its own note.
comptime TOL_MODEL: Float64 = 1e-12
# `invweight0` is COMPUTED from the mass matrix at qpos0, not parsed. It used
# to inherit this model's ~1e-10 forward-kinematics gap and was pinned at 1e-9
# with 5.02e-10 observed; that gap was the `1/sqrt(norm_sq + 1e-10)`
# quaternion normalizer in `kinematics/quat_math.mojo`, fixed 2026-08-03.
# Observed 4.92e-15 now — the mass matrix at qpos0 agrees with MuJoCo's at
# float64 rounding, which is the strongest evidence that the FK chain feeding
# it is exact.
comptime INVWEIGHT_TOL: Float64 = 1e-13


def _build() raises -> Mod:
    var ctx = DeviceContext()
    var mf = Mod()
    Mdl.init_fields[DType.float64, 0](ctx, mf)
    return mf^


def _mj_from_our_xml() raises -> PythonObject:
    """MuJoCo compiled from OUR XML — valid as a reference for layers 2+ only
    because `test_quadruped_xml_compiles_to_the_reference_model` proves that
    string IS the reference model."""
    var mujoco = Python.import_module("mujoco")
    return mujoco.MjModel.from_xml_path("mojo_rl/envs/dm_control/assets/quadruped_walk.xml")


def _mj_geom_type(ours: Int) -> Int:
    """Our `GEOM_*` code -> MuJoCo's `mjtGeom`.

    The two enums are NOT the same numbering — MuJoCo interleaves HFIELD at 1
    and orders ellipsoid/cylinder/box differently. Site types share `mjtGeom`,
    so this maps both.
    """
    if ours == GEOM_PLANE:
        return 0
    if ours == GEOM_SPHERE:
        return 2
    if ours == GEOM_CAPSULE:
        return 3
    if ours == GEOM_ELLIPSOID:
        return 4
    if ours == GEOM_CYLINDER:
        return 5
    if ours == GEOM_BOX:
        return 6
    if ours == GEOM_MESH:
        return 7
    return -1


def _rel(ours: Float64, want: Float64) -> Float64:
    return abs(ours - want) / (1e-15 + abs(want))


def test_quadruped_xml_compiles_to_the_reference_model() raises:
    """Layer 1: our XML vs `dm_control`'s own `quadruped.make_model()`.

    97 mjModel tables plus the counts, the `<option>` block and the element
    ORDER of all seven named object types, for BOTH tasks. Exhaustive on
    purpose — a hand-picked subset is what let `jnt_solimp` stay wrong.

    The two tasks differ only in the floor plane's half-extent (walk 20*.5,
    run 20*5), which a plane ignores dynamically, so `run` is checked here or
    nowhere.
    """
    print("--- quadruped: XML vs the reference model ---")
    var sys = Python.import_module("sys")
    sys.path.insert(0, "tests/dm_control")
    var builder = Python.import_module("quadruped_ref")

    var ntables = Int(py=builder.n_tables_compared())
    assert_true(
        ntables >= 97,
        "quadruped_ref._TABLES shrank to " + String(ntables) + " — entries"
        " were deleted rather than the mismatch fixed",
    )

    for run in range(2):
        var xml = (
            String("mojo_rl/envs/dm_control/assets/quadruped_run.xml")
            if run == 1
            else String("mojo_rl/envs/dm_control/assets/quadruped_walk.xml")
        )
        var bad = builder.compare_xml_to_reference(xml, run == 1)
        var n = Int(py=Python.import_module("builtins").len(bad))
        var tag = String("run") if run == 1 else String("walk")
        if n > 0:
            for i in range(n):
                print("   ", tag, String(bad[i]))
        assert_true(
            n == 0,
            tag + ": our XML does not compile to the reference model",
        )
        print("  PASS:", tag, "—", ntables, "tables, counts, option, order")


def test_quadruped_dims_match_the_model_def() raises:
    """The literals this file indexes with ARE the model's.

    Every loop below runs to a hardcoded count; if one of them drifted from
    `DMQuadrupedWalkModel` the comparisons would silently cover a prefix.
    """
    assert_true(Mdl.NQ == NQ, "NQ")
    assert_true(Mdl.NV == NV, "NV")
    assert_true(Mdl.NBODY == NBODY, "NBODY")
    assert_true(Mdl.NJOINT == NJOINT, "NJOINT")
    assert_true(Mdl.NGEOM == NGEOM, "NGEOM")
    assert_true(Mdl.NSITE == NSITE, "NSITE")
    assert_true(Mdl.MAX_TENDON == NTEN, "NTENDON")
    assert_true(Mdl.nact == NU, "NACT")
    assert_true(Mdl.NA == NA, "NA — the twelve filter activations")


def test_quadruped_body_and_joint_constants_match_mujoco() raises:
    """Layer 2, part 1: `bodies` and `joints` in our `fields.Model`.

    Body ORDER is asserted by `body_parentid`/`body_rootid` rather than
    assumed — an index-wise comparison of masses says nothing if the two
    engines number the bodies differently.

    ⚠ QUATERNIONS ARE STORED (x, y, z, w) HERE and (w, x, y, z) in MuJoCo, for
    bodies, inertia frames, geoms and sites alike. A comparison that forgets to
    reorder passes on any model whose frames are all identity — which is most
    of a quadruped, since only the legs carry a rotation.
    """
    print("--- quadruped: body + joint constants ---")
    var mj = _mj_from_our_xml()
    var mf = _build()

    var parent = mj.body_parentid.tolist()
    var rootid = mj.body_rootid.tolist()
    var mass = mj.body_mass.tolist()
    var inertia = mj.body_inertia.tolist()
    var bpos = mj.body_pos.tolist()
    var bquat = mj.body_quat.tolist()
    var ipos = mj.body_ipos.tolist()
    var iquat = mj.body_iquat.tolist()

    var worst = Float64(0)
    for b in range(NBODY):
        var o = b * MODEL_BODY_SIZE
        assert_true(
            Int(mf.bodies.data[o + BODY_IDX_PARENT]) == Int(py=parent[b]),
            String("body_parentid mismatch on body ") + String(b),
        )
        assert_true(
            Int(mf.bodies.data[o + BODY_IDX_ROOTID]) == Int(py=rootid[b]),
            String("body_rootid mismatch on body ") + String(b),
        )
        worst = max(
            worst,
            _rel(Float64(mf.bodies.data[o + BODY_IDX_MASS]),
                 Float64(py=mass[b])),
        )
        assert_true(
            _rel(Float64(mf.bodies.data[o + BODY_IDX_MASS]),
                 Float64(py=mass[b])) <= TOL_MODEL,
            String("body_mass mismatch on body ") + String(b),
        )
        for k in range(3):
            var ours_i = Float64(mf.bodies.data[o + BODY_IDX_IXX + k])
            worst = max(worst, _rel(ours_i, Float64(py=inertia[b][k])))
            assert_true(
                _rel(ours_i, Float64(py=inertia[b][k])) <= TOL_MODEL,
                String("body_inertia mismatch on body ") + String(b),
            )
            worst = max(
                worst,
                abs(Float64(mf.bodies.data[o + BODY_IDX_POS_X + k])
                    - Float64(py=bpos[b][k])),
            )
            assert_true(
                abs(Float64(mf.bodies.data[o + BODY_IDX_POS_X + k])
                    - Float64(py=bpos[b][k])) <= TOL_MODEL,
                String("body_pos mismatch on body ") + String(b),
            )
            assert_true(
                abs(Float64(mf.bodies.data[o + BODY_IDX_IPOS_X + k])
                    - Float64(py=ipos[b][k])) <= TOL_MODEL,
                String("body_ipos mismatch on body ") + String(b),
            )
        # (x, y, z, w) here, (w, x, y, z) there.
        for k in range(4):
            var mjk = (k + 1) % 4
            assert_true(
                abs(Float64(mf.bodies.data[o + BODY_IDX_QUAT_X + k])
                    - Float64(py=bquat[b][mjk])) <= TOL_MODEL,
                String("body_quat mismatch on body ") + String(b),
            )
            assert_true(
                abs(Float64(mf.bodies.data[o + BODY_IDX_IQUAT_X + k])
                    - Float64(py=iquat[b][mjk])) <= TOL_MODEL,
                String("body_iquat mismatch on body ") + String(b),
            )

    var jtype = mj.jnt_type.tolist()
    var jbody = mj.jnt_bodyid.tolist()
    var jqadr = mj.jnt_qposadr.tolist()
    var jdadr = mj.jnt_dofadr.tolist()
    var jpos = mj.jnt_pos.tolist()
    var jaxis = mj.jnt_axis.tolist()
    var jrange = mj.jnt_range.tolist()
    var jlimited = mj.jnt_limited.tolist()
    var jsolref = mj.jnt_solref.tolist()
    var jsolimp = mj.jnt_solimp.tolist()
    var jstiff = mj.jnt_stiffness.tolist()
    var qpos0 = mj.qpos0.tolist()
    var qspring = mj.qpos_spring.tolist()
    var darm = mj.dof_armature.tolist()
    var ddamp = mj.dof_damping.tolist()
    var dfric = mj.dof_frictionloss.tolist()

    # Our joint TYPE enum is our own; only free/hinge appear here, so map the
    # two rather than pretend to a general translation.
    var n_limited = 0
    for j in range(NJOINT):
        var o = j * MODEL_JOINT_SIZE
        var ours_t = Int(mf.joints.data[o + JOINT_IDX_TYPE])
        var mj_t = Int(py=jtype[j])
        # mjJNT_FREE = 0, mjJNT_HINGE = 3.
        var want_free = mj_t == 0
        assert_true(
            (ours_t == 0) == want_free,
            String("jnt_type mismatch on joint ") + String(j),
        )
        assert_true(
            Int(mf.joints.data[o + JOINT_IDX_BODY_ID]) == Int(py=jbody[j]),
            String("jnt_bodyid mismatch on joint ") + String(j),
        )
        assert_true(
            Int(mf.joints.data[o + JOINT_IDX_QPOS_ADR]) == Int(py=jqadr[j]),
            String("jnt_qposadr mismatch on joint ") + String(j),
        )
        assert_true(
            Int(mf.joints.data[o + JOINT_IDX_DOF_ADR]) == Int(py=jdadr[j]),
            String("jnt_dofadr mismatch on joint ") + String(j),
        )
        for k in range(3):
            assert_true(
                abs(Float64(mf.joints.data[o + JOINT_IDX_POS_X + k])
                    - Float64(py=jpos[j][k])) <= TOL_MODEL,
                String("jnt_pos mismatch on joint ") + String(j),
            )
            assert_true(
                abs(Float64(mf.joints.data[o + JOINT_IDX_AXIS_X + k])
                    - Float64(py=jaxis[j][k])) <= TOL_MODEL,
                String("jnt_axis mismatch on joint ") + String(j),
            )
        # ⚠ THE TWO ENGINES SPELL "unlimited" DIFFERENTLY. MuJoCo zeroes
        # `jnt_range` for an unlimited joint; we store ±1e10, which is the
        # sentinel `constraints/limits.mojo:99` tests (`rmin < -1e9 or
        # rmax > 1e9` -> no row). Comparing the numbers directly would call
        # the free root a mismatch, and widening the tolerance to swallow it
        # would stop checking the sixteen hinges that matter.
        var ours_lo = Float64(mf.joints.data[o + JOINT_IDX_RANGE_MIN])
        var ours_hi = Float64(mf.joints.data[o + JOINT_IDX_RANGE_MAX])
        if Int(py=jlimited[j]) == 1:
            n_limited += 1
            assert_true(
                abs(ours_lo - Float64(py=jrange[j][0])) <= TOL_MODEL
                and abs(ours_hi - Float64(py=jrange[j][1])) <= TOL_MODEL,
                String("jnt_range mismatch on joint ") + String(j),
            )
        else:
            assert_true(
                ours_lo < -1e9 and ours_hi > 1e9,
                String("joint ") + String(j) + " is unlimited in MuJoCo but"
                " our range is inside the ±1e9 sentinel, so we would build a"
                " limit row MuJoCo does not have",
            )
        assert_true(
            abs(Float64(mf.joints.data[o + JOINT_IDX_STIFFNESS])
                - Float64(py=jstiff[j])) <= TOL_MODEL,
            String("jnt_stiffness mismatch on joint ") + String(j),
        )
        # `springref` is MuJoCo's `qpos_spring` at this joint's qpos address,
        # and `qpos0` its `ref`; both are scalars for a hinge and meaningless
        # for the free root, whose 7 qpos entries are the body pose.
        var qa = Int(py=jqadr[j])
        if mj_t != 0:
            assert_true(
                abs(Float64(mf.joints.data[o + JOINT_IDX_SPRINGREF])
                    - Float64(py=qspring[qa])) <= TOL_MODEL,
                String("jnt springref mismatch on joint ") + String(j),
            )
            assert_true(
                abs(Float64(mf.joints.data[o + JOINT_IDX_QPOS0])
                    - Float64(py=qpos0[qa])) <= TOL_MODEL,
                String("qpos0 mismatch on joint ") + String(j),
            )
        # Limit impedance. The free root is where the `<freejoint>` expansion
        # bug lived, so it is compared like every other joint rather than
        # skipped for being inert.
        for k in range(2):
            assert_true(
                abs(Float64(mf.joints.data[o + JOINT_IDX_SOLREF_LIMIT_0 + k])
                    - Float64(py=jsolref[j][k])) <= TOL_MODEL,
                String("jnt_solref mismatch on joint ") + String(j),
            )
        for k in range(5):
            assert_true(
                abs(Float64(mf.joints.data[o + JOINT_IDX_SOLIMP_LIMIT_0 + k])
                    - Float64(py=jsolimp[j][k])) <= TOL_MODEL,
                String("jnt_solimp mismatch on joint ") + String(j)
                + " element " + String(k),
            )
        # armature / damping / frictionloss are PER-DOF in MuJoCo and per-JOINT
        # for us. A hinge has one dof; the free root has six, all zero.
        var da = Int(py=jdadr[j])
        assert_true(
            abs(Float64(mf.joints.data[o + JOINT_IDX_ARMATURE])
                - Float64(py=darm[da])) <= TOL_MODEL,
            String("dof_armature mismatch on joint ") + String(j),
        )
        assert_true(
            abs(Float64(mf.joints.data[o + JOINT_IDX_DAMPING])
                - Float64(py=ddamp[da])) <= TOL_MODEL,
            String("dof_damping mismatch on joint ") + String(j),
        )
        assert_true(
            abs(Float64(mf.joints.data[o + JOINT_IDX_FRICTIONLOSS])
                - Float64(py=dfric[da])) <= TOL_MODEL,
            String("dof_frictionloss mismatch on joint ") + String(j),
        )
    assert_true(
        n_limited == N_HINGE,
        String("only ") + String(n_limited) + " joints are limited — the"
        " range comparison above has gone vacuous",
    )
    print("  worst body rel err =", worst, " limited joints =", n_limited)


def test_quadruped_geom_and_site_constants_match_mujoco() raises:
    """Layer 2, part 2: `geoms` and `sites`.

    quadruped is the widest type mix in the tree — PLANE, SPHERE, CAPSULE,
    CYLINDER and ELLIPSOID geoms, and SPHERE / CAPSULE / BOX site zones — so
    the size mapping is exercised per type rather than assumed. `rbound` is
    compared as well as the sizes it is derived from, because it is the number
    the broad phase actually reads and it is computed, not parsed.
    """
    print("--- quadruped: geom + site constants ---")
    var mj = _mj_from_our_xml()
    var mf = _build()

    var gtype = mj.geom_type.tolist()
    var gbody = mj.geom_bodyid.tolist()
    var gpos = mj.geom_pos.tolist()
    var gquat = mj.geom_quat.tolist()
    var gsize = mj.geom_size.tolist()
    var gfric = mj.geom_friction.tolist()
    var gct = mj.geom_contype.tolist()
    var gca = mj.geom_conaffinity.tolist()
    var gdim = mj.geom_condim.tolist()
    var grb = mj.geom_rbound.tolist()
    var gsolref = mj.geom_solref.tolist()
    var gsolimp = mj.geom_solimp.tolist()
    var gmargin = mj.geom_margin.tolist()

    var n_plane = 0
    var n_sphere = 0
    var n_capsule = 0
    var n_cylinder = 0
    var n_ellipsoid = 0
    var worst_size = Float64(0)

    for g in range(NGEOM):
        var o = g * MODEL_GEOM_SIZE
        var ours_t = Int(mf.geoms.data[o + GEOM_IDX_TYPE])
        assert_true(
            _mj_geom_type(ours_t) == Int(py=gtype[g]),
            String("geom_type mismatch on geom ") + String(g),
        )
        assert_true(
            Int(mf.geoms.data[o + GEOM_IDX_BODY]) == Int(py=gbody[g]),
            String("geom_bodyid mismatch on geom ") + String(g),
        )
        for k in range(3):
            assert_true(
                abs(Float64(mf.geoms.data[o + GEOM_IDX_POS_X + k])
                    - Float64(py=gpos[g][k])) <= TOL_MODEL,
                String("geom_pos mismatch on geom ") + String(g),
            )
        for k in range(4):
            var mjk = (k + 1) % 4
            assert_true(
                abs(Float64(mf.geoms.data[o + GEOM_IDX_QUAT_X + k])
                    - Float64(py=gquat[g][mjk])) <= TOL_MODEL,
                String("geom_quat mismatch on geom ") + String(g),
            )

        # Sizes, per type. `size` means different slots for each.
        var s0 = Float64(py=gsize[g][0])
        var s1 = Float64(py=gsize[g][1])
        var s2 = Float64(py=gsize[g][2])
        var r = Float64(mf.geoms.data[o + GEOM_IDX_RADIUS])
        var hl = Float64(mf.geoms.data[o + GEOM_IDX_HALF_LENGTH])
        var hx = Float64(mf.geoms.data[o + GEOM_IDX_HALF_X])
        var hy = Float64(mf.geoms.data[o + GEOM_IDX_HALF_Y])
        var hz = Float64(mf.geoms.data[o + GEOM_IDX_HALF_Z])
        if ours_t == GEOM_PLANE:
            n_plane += 1
            worst_size = max(worst_size, max(abs(hx - s0), abs(hy - s1)))
            assert_true(
                abs(hx - s0) <= TOL_MODEL and abs(hy - s1) <= TOL_MODEL,
                String("plane half-extent mismatch on geom ") + String(g),
            )
        elif ours_t == GEOM_SPHERE:
            n_sphere += 1
            worst_size = max(worst_size, abs(r - s0))
            assert_true(
                abs(r - s0) <= TOL_MODEL,
                String("sphere radius mismatch on geom ") + String(g),
            )
        elif ours_t == GEOM_CAPSULE or ours_t == GEOM_CYLINDER:
            if ours_t == GEOM_CAPSULE:
                n_capsule += 1
            else:
                n_cylinder += 1
            worst_size = max(worst_size, max(abs(r - s0), abs(hl - s1)))
            assert_true(
                abs(r - s0) <= TOL_MODEL and abs(hl - s1) <= TOL_MODEL,
                String("capsule/cylinder size mismatch on geom ") + String(g),
            )
        elif ours_t == GEOM_ELLIPSOID:
            n_ellipsoid += 1
            worst_size = max(
                worst_size,
                max(abs(hx - s0), max(abs(hy - s1), abs(hz - s2))),
            )
            assert_true(
                abs(hx - s0) <= TOL_MODEL and abs(hy - s1) <= TOL_MODEL
                and abs(hz - s2) <= TOL_MODEL,
                String("ellipsoid semi-axis mismatch on geom ") + String(g),
            )
        else:
            assert_true(
                False,
                String("unhandled geom type on geom ") + String(g)
                + " — this test's size mapping is incomplete",
            )

        # ⚠ SAME SENTINEL SPLIT AS `jnt_range`. MuJoCo writes `rbound = 0` for
        # a plane and reads that as INFINITE (`mjc_...` never bounds a plane);
        # we write 1e10 and mean the same thing. Zero on one of ours would be
        # a bounding sphere of nothing — the opposite of what MuJoCo's 0 says
        # — so the two are asserted separately rather than reconciled.
        if ours_t == GEOM_PLANE:
            assert_true(
                Float64(py=grb[g]) == 0.0,
                "MuJoCo stopped using rbound=0 for a plane",
            )
            assert_true(
                Float64(mf.geoms.data[o + GEOM_IDX_RBOUND]) > 1e9,
                String("plane geom ") + String(g) + " has a FINITE rbound —"
                " the broad phase would cull contacts against the floor",
            )
        else:
            assert_true(
                _rel(Float64(mf.geoms.data[o + GEOM_IDX_RBOUND]),
                     Float64(py=grb[g])) <= TOL_MODEL,
                String("geom_rbound mismatch on geom ") + String(g),
            )
        assert_true(
            abs(Float64(mf.geoms.data[o + GEOM_IDX_FRICTION])
                - Float64(py=gfric[g][0])) <= TOL_MODEL,
            String("geom_friction mismatch on geom ") + String(g),
        )
        assert_true(
            Int(mf.geoms.data[o + GEOM_IDX_CONTYPE]) == Int(py=gct[g])
            and Int(mf.geoms.data[o + GEOM_IDX_CONAFFINITY])
                == Int(py=gca[g])
            and Int(mf.geoms.data[o + GEOM_IDX_CONDIM]) == Int(py=gdim[g]),
            String("geom contype/conaffinity/condim mismatch on geom ")
            + String(g),
        )
        for k in range(2):
            assert_true(
                abs(Float64(mf.geoms.data[o + GEOM_IDX_SOLREF_0 + k])
                    - Float64(py=gsolref[g][k])) <= TOL_MODEL,
                String("geom_solref mismatch on geom ") + String(g),
            )
        for k in range(5):
            assert_true(
                abs(Float64(mf.geoms.data[o + GEOM_IDX_SOLIMP_0 + k])
                    - Float64(py=gsolimp[g][k])) <= TOL_MODEL,
                String("geom_solimp mismatch on geom ") + String(g),
            )
        assert_true(
            abs(Float64(mf.geoms.data[o + GEOM_IDX_MARGIN])
                - Float64(py=gmargin[g])) <= TOL_MODEL,
            String("geom_margin mismatch on geom ") + String(g),
        )

    # Non-vacuity: five geom types really are in play. If a future strip left
    # only capsules, the per-type branches above would go untested in silence.
    assert_true(
        n_plane == 1 and n_sphere == 4 and n_capsule == 12
        and n_cylinder == 2 and n_ellipsoid == 1,
        "the geom type mix changed — this test no longer covers what its"
        " docstring claims",
    )

    var stype = mj.site_type.tolist()
    var sbody = mj.site_bodyid.tolist()
    var spos = mj.site_pos.tolist()
    var squat = mj.site_quat.tolist()
    var ssize = mj.site_size.tolist()
    var n_site_box = 0
    var n_site_capsule = 0
    for s in range(NSITE):
        var o = s * MODEL_SITE_SIZE
        var ours_t = Int(mf.sites.data[o + SITE_IDX_TYPE])
        assert_true(
            _mj_geom_type(ours_t) == Int(py=stype[s]),
            String("site_type mismatch on site ") + String(s),
        )
        if ours_t == GEOM_BOX:
            n_site_box += 1
        elif ours_t == GEOM_CAPSULE:
            n_site_capsule += 1
        assert_true(
            Int(mf.sites.data[o + SITE_IDX_BODY]) == Int(py=sbody[s]),
            String("site_bodyid mismatch on site ") + String(s),
        )
        for k in range(3):
            assert_true(
                abs(Float64(mf.sites.data[o + SITE_IDX_POS_X + k])
                    - Float64(py=spos[s][k])) <= TOL_MODEL,
                String("site_pos mismatch on site ") + String(s),
            )
        # ⚠ ONLY THE SLOTS THE TYPE USES. MuJoCo pads `site_size`'s unused
        # entries with the .005 default from `<default class="site">`, while we
        # leave them at 0 (or, for some sites, replicate size[0]). Comparing
        # all three would call a correct sphere a mismatch; comparing only
        # size[0] for every type would stop checking the capsule zones'
        # half-length, which is the number the touch sensor's ray tests.
        var n_slots = 1
        if ours_t == GEOM_CAPSULE or ours_t == GEOM_CYLINDER:
            n_slots = 2
        elif ours_t == GEOM_BOX or ours_t == GEOM_ELLIPSOID:
            n_slots = 3
        for k in range(n_slots):
            assert_true(
                abs(Float64(mf.sites.data[o + SITE_IDX_SIZE_0 + k])
                    - Float64(py=ssize[s][k])) <= TOL_MODEL,
                String("site_size mismatch on site ") + String(s)
                + " slot " + String(k),
            )
        for k in range(4):
            var mjk = (k + 1) % 4
            assert_true(
                abs(Float64(mf.sites.data[o + SITE_IDX_QUAT_X + k])
                    - Float64(py=squat[s][mjk])) <= TOL_MODEL,
                String("site_quat mismatch on site ") + String(s),
            )
    assert_true(
        n_site_box > 0 and n_site_capsule > 0,
        "the site zone mix lost its box or capsule — the touch-sensor zone"
        " types are no longer covered here",
    )
    print("  worst size abs err =", worst_size,
          " geoms: plane", n_plane, "sphere", n_sphere, "capsule", n_capsule,
          "cylinder", n_cylinder, "ellipsoid", n_ellipsoid)


def test_quadruped_invweight0_matches_mujoco() raises:
    """`body_invweight0` / `dof_invweight0` / `tendon_invweight0`.

    Run for every newly ported model since bug 20, because both bug 20 and bug
    26 were silent multipliers living exactly here. quadruped puts all three in
    play: contacts read `body_invweight0`, the sixteen hinge limits read
    `dof_invweight0`, and the four `coupling_*` equalities read
    `tendon_invweight0`.

    ⚠ THESE ARE COMPUTED, NOT PARSED — `mj_setConst` builds them from the mass
    matrix at qpos0 — so they inherit this model's forward-kinematics gap
    rather than rounding. That gap is pinned as `FK_TOL = 5e-10` in
    tests/physics3d/test_rne_post_sensors_vs_mujoco.mojo and is the reason the
    bound here is not `TOL_MODEL`.
    """
    print("--- quadruped: invweight0 ---")
    var mj = _mj_from_our_xml()
    var mf = _build()

    var biw = mj.body_invweight0.tolist()
    var diw = mj.dof_invweight0.tolist()
    var tiw = mj.tendon_invweight0.tolist()
    var worst = Float64(0)
    for b in range(NBODY):
        for k in range(2):
            var rel = _rel(Float64(mf.body_invweight0.data[2 * b + k]),
                           Float64(py=biw[b][k]))
            worst = max(worst, rel)
            assert_true(
                rel <= INVWEIGHT_TOL,
                String("body_invweight0 mismatch on body ") + String(b),
            )
    for i in range(NV):
        var rel = _rel(Float64(mf.dof_invweight0.data[i]),
                       Float64(py=diw[i]))
        worst = max(worst, rel)
        assert_true(
            rel <= INVWEIGHT_TOL,
            String("dof_invweight0 mismatch on dof ") + String(i),
        )
    for t in range(NTEN):
        var rel = _rel(
            Float64(mf.tendons.data[t * MODEL_TENDON_SIZE
                                    + TENDON_IDX_INVWEIGHT0]),
            Float64(py=tiw[t]),
        )
        worst = max(worst, rel)
        assert_true(
            rel <= INVWEIGHT_TOL,
            String("tendon_invweight0 mismatch on tendon ") + String(t),
        )
    print("  worst invweight0 rel err =", worst)


def test_quadruped_tendon_and_equality_constants_match_mujoco() raises:
    """The twelve `<fixed>` tendons and the four `<equality><tendon>` rows.

    Every tendon here is FIXED (a joint-coefficient sum), so each of ours must
    reproduce MuJoCo's `wrap_objid`/`wrap_prm` slice — the joint ids and their
    coefficients, in order. The first four also carry an equality; the other
    eight are actuator TRANSMISSIONS and must NOT, because `_tendon_env` turns
    every `IS_EQUALITY` record into a bilateral constraint and would weld the
    legs of any model that over-set it.
    """
    print("--- quadruped: tendons + equalities ---")
    var mj = _mj_from_our_xml()
    var mf = _build()

    var tadr = mj.tendon_adr.tolist()
    var tnum = mj.tendon_num.tolist()
    var wtype = mj.wrap_type.tolist()
    var wobj = mj.wrap_objid.tolist()
    var wprm = mj.wrap_prm.tolist()

    var n_eq_flagged = 0
    for t in range(NTEN):
        var o = t * MODEL_TENDON_SIZE
        assert_true(
            Int(mf.tendons.data[o + TENDON_IDX_KIND]) == TENDON_KIND_FIXED,
            String("tendon ") + String(t) + " is not FIXED",
        )
        var n = Int(mf.tendons.data[o + TENDON_IDX_NUM_JOINTS])
        assert_true(
            n == Int(py=tnum[t]),
            String("tendon ") + String(t) + " joint count "
            + String(n) + " != MuJoCo " + String(Int(py=tnum[t])),
        )
        var adr = Int(py=tadr[t])
        for k in range(n):
            # mjWRAP_JOINT = 1.
            assert_true(
                Int(py=wtype[adr + k]) == 1,
                String("tendon ") + String(t) + " wrap " + String(k)
                + " is not a joint wrap",
            )
            # Both sides store the JOINT ID here; the consumers
            # (`dynamics/invweight.mojo:472`, `constraints/equality_tendon.mojo`)
            # look up `JOINT_IDX_DOF_ADR` themselves. The two are NOT
            # interchangeable on this model — joint 2 sits at dof 7 — so a
            # slot holding the wrong one of them is visible here.
            assert_true(
                Int(mf.tendons.data[o + TENDON_IDX_JOINT_0 + k])
                == Int(py=wobj[adr + k]),
                String("tendon ") + String(t) + " joint " + String(k)
                + " mismatch",
            )
            assert_true(
                abs(Float64(mf.tendons.data[o + TENDON_IDX_COEF_0 + k])
                    - Float64(py=wprm[adr + k])) <= TOL_MODEL,
                String("tendon ") + String(t) + " coef " + String(k)
                + " mismatch",
            )
        if Int(mf.tendons.data[o + TENDON_IDX_IS_EQUALITY]) == 1:
            n_eq_flagged += 1

    assert_true(
        n_eq_flagged == NEQ,
        String("IS_EQUALITY is set on ") + String(n_eq_flagged)
        + " tendons but MuJoCo declares " + String(NEQ)
        + " — an over-set flag welds legs that should swing freely",
    )

    # ⚠ A `<equality><tendon>` DOES NOT PRODUCE AN `equality` RECORD FOR US.
    # `Model.equality` holds connect/weld rows; a tendon equality is carried
    # entirely by the tendon it constrains — the `IS_EQUALITY` flag plus the
    # solref/solimp pair at TENDON_IDX_SOLREF_0..SOLIMP_4, which is a DIFFERENT
    # pair from the limit one at 29..35. So `MAX_EQUALITY` is legitimately 0
    # here even though MuJoCo reports `neq = 4`, and indexing `mf.equality`
    # would read a one-element placeholder slab.
    assert_true(
        Mdl.MAX_EQUALITY == 0,
        "quadruped grew a connect/weld equality — the impedance comparison"
        " below only covers the tendon-carried ones",
    )
    var etype = mj.eq_type.tolist()
    var eobj1 = mj.eq_obj1id.tolist()
    var esolref = mj.eq_solref.tolist()
    var esolimp = mj.eq_solimp.tolist()
    for e in range(NEQ):
        # mjEQ_TENDON = 3.
        assert_true(
            Int(py=etype[e]) == 3,
            String("reference equality ") + String(e)
            + " is no longer a tendon equality",
        )
        var t = Int(py=eobj1[e])
        var o = t * MODEL_TENDON_SIZE
        assert_true(
            Int(mf.tendons.data[o + TENDON_IDX_IS_EQUALITY]) == 1,
            String("MuJoCo constrains tendon ") + String(t)
            + " but our record does not carry IS_EQUALITY",
        )
        for k in range(2):
            assert_true(
                abs(Float64(mf.tendons.data[o + TENDON_IDX_SOLREF_0 + k])
                    - Float64(py=esolref[e][k])) <= TOL_MODEL,
                String("eq_solref mismatch on tendon ") + String(t),
            )
        for k in range(5):
            assert_true(
                abs(Float64(mf.tendons.data[o + TENDON_IDX_SOLIMP_0 + k])
                    - Float64(py=esolimp[e][k])) <= TOL_MODEL,
                String("eq_solimp mismatch on tendon ") + String(t),
            )
    # Non-vacuity: `class="coupling"` overrides both, so neither may be left at
    # MuJoCo's default — reading the default back would mean the class was
    # dropped and the constraint is softer than the reference's.
    assert_true(
        abs(Float64(py=esolref[0][0]) - 0.005) < 1e-15
        and abs(Float64(py=esolimp[0][0]) - 0.95) < 1e-15,
        "the `coupling` default class no longer supplies solref/solimp — this"
        " comparison has stopped exercising a non-default impedance",
    )
    print("  PASS:", NTEN, "fixed tendons,", NEQ, "tendon equalities")


def test_quadruped_actuator_constants_match_mujoco() raises:
    """The twelve `<general>` position servos with `dyntype="filter"`.

    Actuators do NOT live in `fields.Model` — they are compiled into
    `SpecFields` by `build_spec_fields`, a separate code path from
    the runtime one every other test here exercises. So this is the only gate
    on that half for quadruped, and the numbers it reads are the ones
    `apply_actions` multiplies:

        force = gear * (gainprm[0] * (act - length))      biasprm = (0, -kp, -kv)
        act_dot = (ctrl - act) / dynprm[0]                mjDYN_FILTER

    Four actuators drive a joint directly (`mjTRN_JOINT`) and eight drive a
    fixed tendon (`mjTRN_TENDON`); both reduce to our (dof address,
    coefficient) triples, so the transmission is checked as those triples
    rather than as a type code.
    """
    print("--- quadruped: actuators ---")
    var mj = _mj_from_our_xml()
    # ⚠ MATERIALIZE `_acd` ONCE. `Mdl._acd` is a COMPTIME value, so every
    # `acd.field[i]` in a runtime expression re-materializes the whole
    # `ComptimeActData` struct. In a function this size that produced GARBAGE
    # — `motor_kp[0]` read back as -3.04e-314 while the identical expression
    # in a ten-line probe returned 1000.0 — so the failure looks like a model
    # bug and is not one. One explicit `materialize` gives a real local with a
    # normal lifetime and the reads are stable.
    var sf = Mdl.make_spec_fields[DType.float64]()

    var trntype = mj.actuator_trntype.tolist()
    var trnid = mj.actuator_trnid.tolist()
    var dyntype = mj.actuator_dyntype.tolist()
    var dynprm = mj.actuator_dynprm.tolist()
    var gainprm = mj.actuator_gainprm.tolist()
    var biasprm = mj.actuator_biasprm.tolist()
    var gear = mj.actuator_gear.tolist()
    var ctrlrange = mj.actuator_ctrlrange.tolist()
    var actadr = mj.actuator_actadr.tolist()
    var jdadr = mj.jnt_dofadr.tolist()
    var tadr = mj.tendon_adr.tolist()
    var tnum = mj.tendon_num.tolist()
    var wobj = mj.wrap_objid.tolist()
    var wprm = mj.wrap_prm.tolist()

    var n_joint_trn = 0
    var n_tendon_trn = 0
    for a in range(NU):
        assert_true(
            abs(Float64(sf.actuators.data[(a) * MODEL_ACTUATOR_SIZE + ACT_IDX_GEAR]) - Float64(py=gear[a][0]))
            <= TOL_MODEL,
            String("actuator_gear mismatch on actuator ") + String(a),
        )
        assert_true(
            abs(Float64(sf.actuators.data[(a) * MODEL_ACTUATOR_SIZE + ACT_IDX_KP]) - Float64(py=gainprm[a][0]))
            <= TOL_MODEL,
            String("gainprm[0] (kp) mismatch on actuator ") + String(a),
        )
        # biasprm = (0, -kp, -kv) for a position servo; ours stores kv.
        assert_true(
            abs(Float64(sf.actuators.data[(a) * MODEL_ACTUATOR_SIZE + ACT_IDX_KV]) + Float64(py=biasprm[a][2]))
            <= TOL_MODEL,
            String("biasprm[2] (kv) mismatch on actuator ") + String(a),
        )
        var bias1 = Float64(py=biasprm[a][1])
        var kp = Float64(sf.actuators.data[(a) * MODEL_ACTUATOR_SIZE + ACT_IDX_KP])
        assert_true(
            abs(bias1 + kp) <= TOL_MODEL,
            String("actuator ") + String(a) + " is not a position servo:"
            " biasprm[1] = " + String(bias1) + " but -gainprm[0] = "
            + String(-kp) + " — the law we simulate is wrong",
        )
        # mjDYN_FILTER = 2 (mjmodel.h:244). NOT 3 — that is FILTEREXACT, which
        # integrates the same ODE exactly instead of with the timestep, so
        # accepting either would accept a different activation trajectory.
        assert_true(
            Int(py=dyntype[a]) == 2,
            String("actuator ") + String(a) + " is dyntype "
            + String(Int(py=dyntype[a])) + ", not filter (2)",
        )
        assert_true(
            abs(Float64(sf.actuators.data[(a) * MODEL_ACTUATOR_SIZE + ACT_IDX_DYN_TAU]) - Float64(py=dynprm[a][0]))
            <= TOL_MODEL,
            String("dynprm[0] (filter tau) mismatch on actuator ")
            + String(a),
        )
        assert_true(
            Int(sf.actuators.data[(a) * MODEL_ACTUATOR_SIZE + ACT_IDX_ACT_ADR]) == Int(py=actadr[a]),
            String("actuator_actadr mismatch on actuator ") + String(a),
        )
        assert_true(
            abs(Float64(sf.actuators.data[(a) * MODEL_ACTUATOR_SIZE + ACT_IDX_CTRL_MIN]) - Float64(py=ctrlrange[a][0]))
            <= TOL_MODEL
            and abs(Float64(sf.actuators.data[(a) * MODEL_ACTUATOR_SIZE + ACT_IDX_CTRL_MAX]) - Float64(py=ctrlrange[a][1]))
                <= TOL_MODEL,
            String("actuator_ctrlrange mismatch on actuator ") + String(a),
        )

        # Transmission, as (dof address, coefficient) triples.
        var tt = Int(py=trntype[a])
        var n = Int(sf.actuators.data[(a) * MODEL_ACTUATOR_SIZE + ACT_IDX_TRN_N])
        if tt == 0:  # mjTRN_JOINT
            n_joint_trn += 1
            assert_true(
                n == 1,
                String("actuator ") + String(a) + " drives a joint but has "
                + String(n) + " transmission triples",
            )
            assert_true(
                Int(sf.actuators.data[(a) * MODEL_ACTUATOR_SIZE + ACT_IDX_TRN_DADR_0 + (0)]) == Int(py=jdadr[Int(py=trnid[a][0])]),
                String("joint transmission dof mismatch on actuator ")
                + String(a),
            )
            assert_true(
                abs(Float64(sf.actuators.data[(a) * MODEL_ACTUATOR_SIZE + ACT_IDX_TRN_COEF_0 + (0)]) - 1.0) <= TOL_MODEL,
                String("joint transmission coef is not 1 on actuator ")
                + String(a),
            )
        elif tt == 3:  # mjTRN_TENDON
            n_tendon_trn += 1
            var t = Int(py=trnid[a][0])
            var adr = Int(py=tadr[t])
            assert_true(
                n == Int(py=tnum[t]),
                String("actuator ") + String(a) + " has " + String(n)
                + " triples for a tendon of " + String(Int(py=tnum[t])),
            )
            for k in range(n):
                assert_true(
                    Int(sf.actuators.data[(a) * MODEL_ACTUATOR_SIZE + ACT_IDX_TRN_DADR_0 + (k)])
                    == Int(py=jdadr[Int(py=wobj[adr + k])]),
                    String("tendon transmission dof ") + String(k)
                    + " mismatch on actuator " + String(a),
                )
                assert_true(
                    abs(Float64(sf.actuators.data[(a) * MODEL_ACTUATOR_SIZE + ACT_IDX_TRN_COEF_0 + (k)])
                        - Float64(py=wprm[adr + k])) <= TOL_MODEL,
                    String("tendon transmission coef ") + String(k)
                    + " mismatch on actuator " + String(a),
                )
        else:
            assert_true(
                False,
                String("actuator ") + String(a) + " uses transmission type "
                + String(tt) + ", which this test does not map",
            )

    assert_true(
        Mdl.NA == NA,
        "the comptime parser and MuJoCo disagree on `na`",
    )
    assert_true(
        n_joint_trn == 4 and n_tendon_trn == 8,
        "the transmission mix changed — this test no longer covers both",
    )
    print("  PASS: 12 filter servos,", n_joint_trn, "joint /",
          n_tendon_trn, "tendon transmissions")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
