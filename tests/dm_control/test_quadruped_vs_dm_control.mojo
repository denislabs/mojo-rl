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
from std.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.quadruped import (
    DMQuadrupedWalk,
    DMQuadrupedRun,
    DMQuadrupedWalkConfig,
    DMQuadrupedRunConfig,
    dm_quadruped_walk_xml,
    QUADRUPED_OBS_DIM,
    QUADRUPED_WALK_SPEED,
    QUADRUPED_RUN_SPEED,
    N_HINGE,
    HINGE_QPOS_0,
    HINGE_DOF_0,
)

comptime REF_PATH: StaticString = "references/dm_control-main"

comptime NQ: Int = 23
comptime NV: Int = 22
comptime NU: Int = 12
comptime NA: Int = 12

# mjtSensor: accelerometer 1, gyro 3, force 4, torque 5, velocimeter 2.
comptime SENS_ACC: Int = 1
comptime SENS_VEL: Int = 2
comptime SENS_GYRO: Int = 3
comptime SENS_FORCE: Int = 4
comptime SENS_TORQUE: Int = 5

comptime OBS_TOL: Float64 = 1e-8
# The reward reads the velocimeter, so it inherits the ~1e-10 forward-
# kinematics gap between the two engines on this model (pinned as FK_TOL in
# tests/physics3d/test_rne_post_sensors_vs_mujoco.mojo). Observed 2.0e-12.
comptime REWARD_TOL: Float64 = 1e-10


def _mj(state_qpos: List[Float64], state_qvel: List[Float64]) raises -> Tuple[
    PythonObject, PythonObject, PythonObject
]:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(String(dm_quadruped_walk_xml))
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
    var m = mujoco.MjModel.from_xml_string(String(dm_quadruped_walk_xml))

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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
