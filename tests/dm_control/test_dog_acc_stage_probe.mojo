"""Where does dog's ACCELERATION stage diverge — `cacc`, or the site sensor?

`test_dog_observation_matches_dm_control` reports 1.96 of error across the 15
acceleration-stage dims (accelerometer + four force sensors) on a reference
magnitude of 17.4, while the other 208 dims agree to 2.66e-15 and the rollout
integrates to 1.23e-13. So the dynamics are right and something in the
acceleration stage is not.

THREE HYPOTHESES ARE ALREADY DEAD, all by measurement:

  * the test's protocol — fixed the position/velocity half and left this
    untouched;
  * sampling the wrong substep — the accelerometer takes -6.3859 / -1.704 /
    0.7163 across dog's three substeps, so a substep offset would read 7.10 or
    2.42, not 1.96;
  * a stale `cacc` combined with a refreshed `cvel` (our `_fields_vel()` moves
    `cvel` to the post-integration value after the substep loop, and
    `mj_objectAcceleration` adds `omega x v` from it) — that reproduces 0.82,
    the right axis and sign but under half the error.

So this file stops hypothesising and reads our own numbers.

THE DESIGN. One substep, and the comparison split so a failure names its half:

  * `cacc` per body against MuJoCo's, at the SAME pre-integration state. This
    is `compute_rne_post` on its own. The FIRST body that diverges localises
    the bug in the tree walk; a matching `cacc` exonerates it entirely.
  * the accelerometer against MuJoCo's, same state. If `cacc` matches and this
    does not, the defect is in the site-sensor evaluation — the `cvel` mixing
    above being the leading candidate.

⚠ FRAME_SKIP = 1 ON PURPOSE, VIA THE THIRD CONSTRUCTOR ARGUMENT.
`Phyics3dEnv.__init__` is `(ctx, max_steps, frame_skip)` and passing 1 silently
overrides `CONFIG.FRAME_SKIP` — which cost a debugging round once (§14) and is
exactly what is wanted here. With one substep our `cacc` belongs to the pinned
state, so it is directly comparable to `mj_forward` at that state and the
substep question cannot confound the answer.

⚠ THE POSE IS THE OBSERVATION TEST'S, and contact-free by construction — the
same assert that file makes. With contacts live this would compare contact
solvers instead of the acceleration stage.

Run with:
    pixi run mojo run -I . tests/dm_control/test_dog_acc_stage_probe.mojo
"""

from std.math import abs, sin, cos
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from std.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.dog import (
    DMDogStand,
    DMDogStandWalkModel,
    dm_dog_stand_walk_xml,
)

comptime DTYPE = DType.float64
comptime M = DMDogStandWalkModel
comptime NQ = M.NQ
comptime NV = M.NV
comptime NBODY = M.NBODY
comptime NACT = M.nact
comptime TEST_PATH = "tests/dm_control"

comptime HINGE_QPOS_0 = 7
comptime HINGE_DOF_0 = 6
comptime N_HINGE = 73

# Both sides run the same float64 arithmetic from the same state over ONE
# substep, so anything real is far above this.
comptime TOL: Float64 = 1e-9


def _ref() raises -> Tuple[PythonObject, PythonObject, PythonObject]:
    var sys = Python.import_module("sys")
    sys.path.insert(0, TEST_PATH)
    var mujoco = Python.import_module("mujoco")
    var builder = Python.import_module("dog_ref")
    var m = builder.model()
    return (mujoco, m, mujoco.MjData(m))


def _pinned_pose(qpos0: List[Float64]) -> Tuple[List[Float64], List[Float64], List[Float64]]:
    """The observation test's pose, transcribed so both sides start identical."""
    var qpos = List[Float64]()
    var qvel = List[Float64]()
    var act = List[Float64]()
    for i in range(NQ):
        qpos.append(qpos0[i])
    for _ in range(NV):
        qvel.append(0.0)
    for i in range(NACT):
        act.append(0.3 * sin(0.9 * Float64(i) + 0.2))

    qpos[0] = 0.3
    qpos[1] = -0.2
    qpos[2] = 2.0
    var hr = 0.9
    var hy = 0.45
    qpos[3] = cos(hy) * cos(hr)
    qpos[4] = cos(hy) * sin(hr)
    qpos[5] = sin(hy) * sin(hr)
    qpos[6] = sin(hy) * cos(hr)
    for k in range(N_HINGE):
        qpos[HINGE_QPOS_0 + k] = (
            qpos0[HINGE_QPOS_0 + k] + 0.008 * sin(0.7 * Float64(k) + 0.3)
        )
        qvel[HINGE_DOF_0 + k] = 0.23 * cos(0.5 * Float64(k) + 1.1)
    qvel[0] = 0.7
    qvel[1] = -0.4
    qvel[2] = 0.25
    qvel[3] = 0.3
    qvel[4] = -0.55
    qvel[5] = 0.8
    return (qpos^, qvel^, act^)


def test_dog_cacc_matches_mujoco() raises:
    """`compute_rne_post`'s output, body by body, over ONE substep."""
    print("--- dog: cacc vs MuJoCo (one substep, contact-free) ---")
    var h = _ref()
    var mujoco = h[0]
    var m = h[1]
    var dat = h[2]

    # ⚠ THIRD ARGUMENT = 1 — see the module docstring.
    var env = DMDogStand[DTYPE](DeviceContext(), 1000, 1)
    _ = env.reset()

    var qpos0 = List[Float64]()
    for i in range(NQ):
        qpos0.append(Float64(env.d.qpos.data[i]))
    var st = _pinned_pose(qpos0)
    var qpos = st[0].copy()
    var qvel = st[1].copy()
    var act = st[2].copy()

    # MuJoCo at the pinned state: `cacc` here is the PRE-integration one, which
    # is what a single substep of ours produces.
    for i in range(NQ):
        dat.qpos[i] = qpos[i]
    for i in range(NV):
        dat.qvel[i] = qvel[i]
    for i in range(NACT):
        dat.act[i] = act[i]
        dat.ctrl[i] = act[i]
    mujoco.mj_forward(m, dat)
    print("  MuJoCo ncon at the pose:", Int(py=dat.ncon), " (must be 0)")
    assert_true(
        Int(py=dat.ncon) == 0,
        "the pinned pose has contacts — this would compare contact solvers"
        " rather than the acceleration stage",
    )

    var qv = List[Scalar[DTYPE]]()
    var qp = List[Scalar[DTYPE]]()
    for i in range(NQ):
        qp.append(Scalar[DTYPE](qpos[i]))
    for i in range(NV):
        qv.append(Scalar[DTYPE](qvel[i]))
    env.set_state(qp, qv)
    for i in range(NACT):
        env.act[i] = Scalar[DTYPE](act[i])
    var a = type_of(env).ActionType()
    for i in range(NACT):
        a.data[i] = Scalar[DTYPE](act[i])
    _ = env.step(a)

    # NON-VACUITY: a zero cacc would match trivially. dog is under gravity with
    # `act` driving it, so the reference is large.
    var ref_mag = 0.0
    for b in range(NBODY):
        for k in range(6):
            var v = abs(Float64(py=dat.cacc[b][k]))
            if v > ref_mag:
                ref_mag = v
    print("  MuJoCo |cacc|_inf =", ref_mag, " (must be >> 0)")
    assert_true(
        ref_mag > 1.0,
        "MuJoCo's cacc is ~zero at this pose, so a match would prove nothing",
    )

    var worst = 0.0
    var worst_b = -1
    var worst_k = -1
    var first_bad = -1
    for b in range(NBODY):
        for k in range(6):
            var ours = Float64(env.d.cacc.data[b * 6 + k])
            var want = Float64(py=dat.cacc[b][k])
            var e = abs(ours - want)
            if e > TOL and first_bad < 0:
                first_bad = b
            if e > worst:
                worst = e
                worst_b = b
                worst_k = k

    print("  max |d(cacc)| =", worst, " at body", worst_b, "component", worst_k)
    print("  FIRST body to diverge:", first_bad, " (-1 = none)")
    if worst_b >= 0:
        print(
            "      ours", Float64(env.d.cacc.data[worst_b * 6 + worst_k]),
            " MuJoCo", Float64(py=dat.cacc[worst_b][worst_k]),
        )

    assert_true(
        worst <= TOL,
        "our `cacc` differs from MuJoCo's over a single substep — the defect is"
        " in `compute_rne_post`'s tree walk, and the FIRST diverging body above"
        " localises it. If instead this passes, the acceleration-stage error is"
        " downstream, in the site-sensor evaluation.",
    )


def test_dog_accelerometer_matches_mujoco() raises:
    """The whole chain: `cacc` + `cvel` + the site transform, one substep."""
    print("--- dog: accelerometer vs MuJoCo (one substep) ---")
    var h = _ref()
    var mujoco = h[0]
    var m = h[1]
    var dat = h[2]

    var env = DMDogStand[DTYPE](DeviceContext(), 1000, 1)
    _ = env.reset()

    var qpos0 = List[Float64]()
    for i in range(NQ):
        qpos0.append(Float64(env.d.qpos.data[i]))
    var st = _pinned_pose(qpos0)
    var qpos = st[0].copy()
    var qvel = st[1].copy()
    var act = st[2].copy()

    for i in range(NQ):
        dat.qpos[i] = qpos[i]
    for i in range(NV):
        dat.qvel[i] = qvel[i]
    for i in range(NACT):
        dat.act[i] = act[i]
        dat.ctrl[i] = act[i]
    mujoco.mj_forward(m, dat)

    var qv = List[Scalar[DTYPE]]()
    var qp = List[Scalar[DTYPE]]()
    for i in range(NQ):
        qp.append(Scalar[DTYPE](qpos[i]))
    for i in range(NV):
        qv.append(Scalar[DTYPE](qvel[i]))
    env.set_state(qp, qv)
    for i in range(NACT):
        env.act[i] = Scalar[DTYPE](act[i])
    var a = type_of(env).ActionType()
    for i in range(NACT):
        a.data[i] = Scalar[DTYPE](act[i])
    _ = env.step(a)
    var obs = env.get_obs_list()

    # dims 160..162 of the 223-long observation are the accelerometer.
    var sadr = Int(
        py=m.sensor_adr[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer")
        ]
    )
    var worst = 0.0
    var mag = 0.0
    for k in range(3):
        var ours = Float64(obs[160 + k])
        var want = Float64(py=dat.sensordata[sadr + k])
        var e = abs(ours - want)
        print("    axis", k, ": ours", ours, " MuJoCo", want, " |d| =", e)
        if e > worst:
            worst = e
        if abs(want) > mag:
            mag = abs(want)

    print("  max |d(accelerometer)| =", worst, "  reference |.|_inf =", mag)
    assert_true(
        mag > 1.0,
        "the reference accelerometer is ~zero, so this would pass whatever we"
        " reported",
    )
    assert_true(
        worst <= TOL,
        "our accelerometer differs from MuJoCo's over a single substep. If"
        " `test_dog_cacc_matches_mujoco` PASSED, the defect is in the site"
        " sensor rather than in `cacc` — most likely `cvel`, which"
        " `_fields_vel()` refreshes to the post-integration value after the"
        " substep loop while `cacc` stays pre-integration, and which enters the"
        " linear acceleration through `omega x v`.",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
