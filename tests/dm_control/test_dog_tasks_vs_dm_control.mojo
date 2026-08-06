"""dm_control `dog` task parity — observation and reward, at a pinned pose.

Split out of `test_dog_vs_dm_control.mojo` because that file already costs
~25 minutes to build: the dog model def is cheap (14.8 s) but instantiating
`init_fields` and a full `Phyics3dEnv` over 62 bodies / 74 joints / 128 geoms
is not, and two files that each build one env beat one file that builds two.

WHAT THIS GATES, AND WHAT IT DELIBERATELY DOES NOT

It gates the ACCESSORS and the REWARD ALGEBRA at a state both engines are
given explicitly. It does NOT gate a rollout, and it cannot yet: dog.xml sets
`noslip_iterations="4"` and `mj_solNoSlip` is not implemented, which moves
MuJoCo's own trajectory by 2.9e-2 of qvel on the first contacting step. The
model defs therefore pass `allow_missing_noslip=True` and a rollout gate is
owed once the pass lands. Saying that here rather than shipping a rollout
comparison with a loose tolerance is the point — an inherited tolerance is a
placeholder, and nine orders of slack means the gate cannot fail.

THE `act` PROBLEM, and why the test writes it on both sides

38 of the 223 observation numbers ARE `data.act`, and every dog actuator is
`dyntype="filter"` whose force is `gainprm[0] * act` — so `act` is both an
observation and a driver of the dynamics. `set_state` carries qpos and qvel
only. The test therefore copies OUR `env.act` into MuJoCo's `dat.act` before
`mj_forward`, which isolates the accessors from the activation integration.
The integration itself is gated by the actuator constants in the model test.

⚠ THE POSE MUST BE ASYMMETRIC IN THE TORSO FRAME. `torso_com_velocity` is
`v.dot(xmat['torso'])`, i.e. `Rᵀ v` — and at `R == I`, `R` and `Rᵀ` agree, so
an upright dog passes whether or not the transpose is right. The pose below
carries a roll and a yaw for exactly that reason, and the test asserts the
frame's asymmetry before comparing anything.

Run with:
    pixi run mojo run -I . tests/dm_control/test_dog_tasks_vs_dm_control.mojo
"""

from std.math import abs, cos, sin, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from std.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.dog import (
    DMDogStand,
    DMDogWalk,
    DMDogStandWalkModel,
    dm_dog_stand_walk_xml,
    DOG_OBS_DIM,
    DOG_N_HINGE,
    DOG_HINGE_QPOS_0,
    DOG_HINGE_DOF_0,
    DOG_WALK_SPEED,
    DOG_FRAME_SKIP,
)

comptime TEST_PATH = "tests/dm_control"
comptime NQ = DMDogStandWalkModel.NQ
comptime NV = DMDogStandWalkModel.NV
comptime NACT = DMDogStandWalkModel.nact

# Both sides run the same float64 arithmetic on the same state; the only
# spread is summation order inside the sensor reductions.
comptime OBS_TOL = 1e-11
# Measured 1.2e-12 absolute on a reward of 0.044 (~2.8e-11 relative). That is
# FK round-off propagated through `tolerance`'s `exp`, not a formula
# difference — the six factors agree to every printed digit. Budgeted two
# orders above the observation.
comptime REWARD_TOL = 1e-9

# Rollout gate. `N_SETTLE` is MuJoCo-side only — it just produces a loaded
# starting pose.
# ⚠ CONSTRUCT THE ENV WITHOUT A THIRD ARGUMENT. `Phyics3dEnv.__init__` is
# `(ctx, max_steps, frame_skip)`, so `(ctx, 1000, 1)` sets FRAME_SKIP TO 1 —
# one substep per `step()` where dm_control takes three. The rollout gate
# caught it as a 0.209 divergence in `act` and 67 in qvel on the first
# contacting step; nothing about the message said "frame skip". Omitting the
# argument takes `CONFIG.FRAME_SKIP`, which is the 3 dog wants.
comptime N_SETTLE: Int = 400
comptime N_ROLL: Int = 30

# ⚠ ALL THREE BUDGETS ARE PROVISIONAL UNTIL MEASURED, and are deliberately
# NOT inherited from another domain — an inherited tolerance is a placeholder,
# and nine orders of slack means the gate cannot fail. They are set from the
# first run's reported numbers and tightened.
comptime ACT_TOL = 1e-12
comptime SMOOTH_TOL = 1e-9
comptime CONTACT_TOL = 1e-6


def _action_at(step: Int, k: Int) -> Float64:
    """A deterministic, non-repeating drive. Amplitude 0.4 rather than 1.0:
    dog's actuators are `gainprm="0.02"` force sources, and saturating all 38
    makes the dog thrash off its feet within a few steps, which would empty
    the contact phase this gate exists to measure."""
    return 0.4 * sin(0.17 * Float64(step) + 0.41 * Float64(k))


def _ref_module() raises -> PythonObject:
    var sys = Python.import_module("sys")
    sys.path.insert(0, TEST_PATH)
    return Python.import_module("dog_ref")


def _mj() raises -> Tuple[PythonObject, PythonObject, PythonObject]:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[dm_dog_stand_walk_xml]())
    return (mujoco, m, mujoco.MjData(m))


def _pose(qpos0: List[Scalar[DType.float64]], z: Float64) raises -> Tuple[
    List[Float64], List[Float64], List[Float64]
]:
    """A deterministic, CONTACT-FREE, tilted pose — plus the activations.

    Returns (qpos, qvel, act). Every constant below was chosen by measuring
    against MuJoCo, not by taste:

      * `z` is a PARAMETER because the two callers want opposite things. The
        REWARD tests need it low (0.30) so the height factors stay off their
        plateau; the OBSERVATION test needs it high (2.0) so the dog is still
        contact-free AFTER the step — at 0.30 it touches within one control
        step, and then the touch/force dims would compare two engines' contact
        SOLVERS rather than their accessors.

      * `z = 0.30` and a 1.8 rad roll. At the obvious pose (upright, z = 2)
        FIVE of the six Stand reward factors saturate at exactly 1.0 and the
        reward is 0.9 no matter what the formula does — a wrong `upright` or a
        wrong height bound passes. Here the factors are
        `[0.914, 0.772, 0.411, 0.414, 0.414, 0.9]`: the two heights and all
        THREE uprights vary, and the three uprights are DISTINCT, which is
        what makes "upright is three factors, not one" a testable claim.

      * The hinge perturbation is `qpos0 + 0.008*sin(...)`, and 0.008 is an
        upper bound, not a preference. dog has 120 collision primitives and
        SELF-COLLIDES: measured `ncon` against amplitude is
        0.008 -> 0, 0.01 -> 1, 0.02 -> 2, 0.11 -> 9, at ANY height (raising
        the root does not help — these are the dog's own legs, not the floor).
        A pose with contacts would compare contact SOLVERS here, which is not
        what this file is for and is blocked on noslip anyway.

      * Perturbing from `qpos0` rather than from zero. At `qpos0` every hinge
        is 0, so all 73 `joint_angles` would be zero ON BOTH SIDES and that
        block of the observation would be gated vacuously.

      * `act` is set explicitly. Left alone it is zero after a reset and a
        zero-action step, which would make 38 of the 223 dims another vacuous
        block — and `act` is not decoration here, it IS the actuator force.
    """
    var qpos = List[Float64]()
    var qvel = List[Float64]()
    var act = List[Float64]()
    for i in range(NQ):
        qpos.append(Float64(qpos0[i]))
    for _ in range(NV):
        qvel.append(0.0)
    for i in range(NACT):
        act.append(0.3 * sin(0.9 * Float64(i) + 0.2))

    qpos[0] = 0.3
    qpos[1] = -0.2
    qpos[2] = z

    # Roll 1.8 rad about x composed with yaw 0.9 about z (half-angles below),
    # so the torso frame is genuinely off-axis and R != R^T.
    var hr = 0.9
    var hy = 0.45
    var cr = cos(hr)
    var sr = sin(hr)
    var cy = cos(hy)
    var sy = sin(hy)
    qpos[3] = cy * cr
    qpos[4] = cy * sr
    qpos[5] = sy * sr
    qpos[6] = sy * cr

    for k in range(DOG_N_HINGE):
        qpos[DOG_HINGE_QPOS_0 + k] = (
            Float64(qpos0[DOG_HINGE_QPOS_0 + k])
            + 0.008 * sin(0.7 * Float64(k) + 0.3)
        )
        qvel[DOG_HINGE_DOF_0 + k] = 0.23 * cos(0.5 * Float64(k) + 1.1)

    # Root twist, all six non-zero so velocimeter, gyro and subtreelinvel are
    # each exercised.
    qvel[0] = 0.7
    qvel[1] = -0.4
    qvel[2] = 0.25
    qvel[3] = 0.3
    qvel[4] = -0.55
    qvel[5] = 0.8
    return (qpos^, qvel^, act^)


def test_dog_observation_matches_dm_control() raises:
    """All 223 numbers, against `dog.py::get_observation_components`."""
    print("--- dog: observation vs dm_control ---")
    var env = DMDogStand[DType.float64](DeviceContext(), 1000)
    _ = env.reset()

    # `qpos0` is read back from the model rather than transcribed: 73 hinge
    # rest values is not something to retype, and layers 1 and 2 already gate
    # that our `qpos0` is MuJoCo's.
    var qpos0 = List[Scalar[DType.float64]]()
    for i in range(NQ):
        qpos0.append(env.d.qpos.data[i])

    var st = _pose(qpos0, 2.0)
    var qpos = st[0].copy()
    var qvel = st[1].copy()
    var act = st[2].copy()

    # ⚠ SET `act` BEFORE THE STEP, AND COMPARE AT THE POST-STEP STATE.
    # The first draft pinned qpos/qvel, stepped, re-pinned, and only THEN wrote
    # `env.act` — so our `cacc` came from a step in which `act` was still zero
    # while MuJoCo's `mj_forward` used the new activations. The accelerometer
    # is derived from `cacc`, and the mismatch showed up as a 597.7 error on
    # observation dim 162 while every kinematic dim was exact. `act` is not
    # inert state: it IS the actuator force.
    env.set_state(qpos, qvel)
    for i in range(NACT):
        env.act[i] = Scalar[DType.float64](act[i])
    var a = type_of(env).ActionType()
    for i in range(NACT):
        a.data[i] = Scalar[DType.float64](act[i])
    _ = env.step(a)
    var obs = env.get_obs_list()
    assert_true(len(obs) == DOG_OBS_DIM, "dog observation is not 223 long")

    # Both sides are now compared at OUR post-step state, so this gates the
    # accessors rather than one substep of integration.
    var post_q = List[Float64]()
    var post_v = List[Float64]()
    var post_a = List[Float64]()
    for i in range(NQ):
        post_q.append(Float64(env.d.qpos.data[i]))
    for i in range(NV):
        post_v.append(Float64(env.d.qvel.data[i]))
    for i in range(NACT):
        post_a.append(Float64(env.act[i]))

    var h = _mj()
    var mujoco = h[0]
    var m = h[1]
    var dat = h[2]
    for i in range(NQ):
        dat.qpos[i] = post_q[i]
    for i in range(NV):
        dat.qvel[i] = post_v[i]
    for i in range(NACT):
        dat.act[i] = post_a[i]
    mujoco.mj_forward(m, dat)
    assert_true(
        Int(py=dat.ncon) == 0,
        "the pinned pose must be contact-free, or this compares contact"
        " SOLVERS rather than sensors — dog self-collides above a hinge"
        " perturbation of ~0.008 rad regardless of height",
    )

    # Non-vacuity for the torso-frame transpose: at R == I a wrong transpose
    # is invisible.
    var np = Python.import_module("numpy")
    var tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    var R = np.array(dat.xmat[tb]).reshape(3, 3)
    var asym = Float64(py=np.abs(np.subtract(R, R.T)).max())
    print("  torso frame asymmetry (must be >> 0):", asym)
    assert_true(
        asym > 0.1,
        "the pose's torso frame is near-symmetric, so `v.dot(R)` and `R v`"
        " agree and the transpose is untested",
    )

    var refmod = _ref_module()
    var builtins = Python.import_module("builtins")
    var want = refmod.observation(m, dat)
    assert_true(
        Int(py=builtins.len(want)) == DOG_OBS_DIM,
        "the reference observation is not 223 long",
    )

    # ⚠ THE ACCELEROMETER CANNOT BE GATED BY PINNING A POST-STEP STATE, and
    # dims 160..162 are excluded here and gated separately below.
    #
    # Everything else in the observation is a POSITION- or VELOCITY-stage
    # quantity, so recomputing it with `mj_forward` at our post-step state is
    # exactly what dm_control would report. The accelerometer is the one
    # ACCELERATION-stage entry, and dm_control never recomputes that stage
    # after a step: `Physics.step()` ends with `mj_step1`, which refreshes the
    # pos/vel stages only, so the accelerometer a task observes belongs to the
    # state BEFORE the step. Measured on this model (scratch probe):
    #
    #     dm_control post-step accel  ==  PRE-step accel      (diff 0.0)
    #     fresh mj_forward at post-step state                  differs by 23.3
    #     velocimeter / gyro under both protocols              differ by 0.0
    #
    # Our engine agrees with dm_control — `RNE_POST` runs inside the substep,
    # off that substep's `qacc` — so the 4.516 this file used to report at dim
    # 162 was the REFERENCE being wrong, not the sensor. A post-step state
    # simply does not contain the pre-step acceleration; no amount of
    # tolerance fixes that, and widening OBS_TOL would have buried a correct
    # sensor under a broken comparison.
    # ⚠ IT IS NOT JUST THE ACCELEROMETER. Every ACCELERATION-stage sensor has
    # this property, and dog has two blocks of them — `dog_config` names both:
    # "accelerometer + the four force sensors need `mj_rnePostConstraint`".
    # The first pass of this fix excluded only dims 160..162 and the failure
    # moved straight to dim 169, `foot_forces[0]`, at 2.062. Fixing a symptom
    # one index at a time is how you end up with a tolerance instead of a gate.
    #
    #   160..162  accelerometer      (acceleration stage)
    #   169..180  foot/hand force x4 (acceleration stage)
    #
    # `touch` (181..184) is also acceleration-stage but stays with the pinned
    # comparison: it is identically zero on both sides at this contact-free
    # pose, which the block comment below already records as ungated here.
    comptime ACC_0 = 160
    comptime ACC_N = 3
    comptime FRC_0 = 169
    comptime FRC_N = 12
    var worst = 0.0
    var worst_i = -1
    for i in range(DOG_OBS_DIM):
        if i >= ACC_0 and i < ACC_0 + ACC_N:
            continue
        if i >= FRC_0 and i < FRC_0 + FRC_N:
            continue
        var d = abs(Float64(obs[i]) - Float64(py=want[i]))
        if d > worst:
            worst = d
            worst_i = i
    print(
        "  worst |err| over the 208 position/velocity-stage dims =",
        worst, "at", worst_i,
    )
    assert_true(
        worst <= OBS_TOL,
        "dog observation differs from dm_control on a position- or"
        " velocity-stage dim — the acceleration-stage sensors are excluded"
        " here and gated separately below, so this is a real accessor defect",
    )

    # --- the accelerometer, against dm_control's OWN stepping ---------------
    # A second MjData is driven from the SAME pre-step state through the same
    # FRAME_SKIP substeps and finished with `mj_step1`, which is precisely
    # `Physics._step_with_up_to_date_position_velocity`. Its `sensordata` is
    # then the accelerometer dm_control would hand the task.
    #
    # ⚠ This half DOES depend on the integration agreeing, unlike the pinned
    # comparison above — that is unavoidable, because the quantity is defined
    # by a step rather than by a state. It is a fair gate here: the pose is
    # contact-free by the assert above, and the rollout gate measures the
    # driven contacting step at 1.23e-13.
    var dat2 = mujoco.MjData(m)
    for i in range(NQ):
        dat2.qpos[i] = qpos[i]
    for i in range(NV):
        dat2.qvel[i] = qvel[i]
    for i in range(NACT):
        dat2.act[i] = act[i]
        dat2.ctrl[i] = act[i]
    mujoco.mj_forward(m, dat2)
    for _ in range(DOG_FRAME_SKIP):
        mujoco.mj_step(m, dat2)
    mujoco.mj_step1(m, dat2)

    var acc_ref = refmod.observation(m, dat2)
    var worst_acc = 0.0
    var worst_acc_i = -1
    var acc_mag = 0.0
    for k in range(ACC_N + FRC_N):
        var i = (ACC_0 + k) if k < ACC_N else (FRC_0 + k - ACC_N)
        var d = abs(Float64(obs[i]) - Float64(py=acc_ref[i]))
        if d > worst_acc:
            worst_acc = d
            worst_acc_i = i
        var v = abs(Float64(py=acc_ref[i]))
        if v > acc_mag:
            acc_mag = v
    print(
        "  acceleration-stage (accel + 4 force sensors) vs dm_control"
        " stepping: worst |err| =", worst_acc, "at", worst_acc_i,
    )

    # NON-VACUITY: an all-zero acceleration block would pass any tolerance.
    # dog is under gravity with `act` driving it, so this is far from zero.
    print("    reference |acc-stage|_inf =", acc_mag, " (must be >> 0)")
    assert_true(
        acc_mag > 1.0,
        "the reference acceleration-stage block is ~zero, so this comparison"
        " would pass whatever our sensors reported",
    )
    assert_true(
        worst_acc <= OBS_TOL,
        "an acceleration-stage sensor differs from dm_control's own stepping —"
        " suspect `RNE_POST`/`cacc`/`cfrc_int` or the substep at which they are"
        " sampled, NOT the pinned-state accessors above",
    )

    # ⚠ NON-VACUITY, PER BLOCK. A whole block of zeros would sit inside any
    # tolerance if the reference read the same zeros. Measured at this pose:
    # 219 of 223 are non-zero. The four that are not are the TOUCH sensors,
    # which are zero because the pose is contact-free by construction — so
    # `touch` is the one block this file cannot gate, and it is named here
    # rather than hidden in an aggregate count.
    var nonzero = 0
    for i in range(DOG_OBS_DIM):
        if Float64(obs[i]) != 0.0:
            nonzero += 1
    print("  non-zero observation dims:", nonzero, "/", DOG_OBS_DIM,
          " (the 4 touch sensors are 0 at a contact-free pose)")
    assert_true(
        nonzero >= 215,
        "a block of the observation is not being filled",
    )
    # The hinge block specifically — it is 73 of the 223 and would be all
    # zeros if the pose were left at qpos0.
    var hinge_nonzero = 0
    for k in range(DOG_N_HINGE):
        if Float64(obs[k]) != 0.0:
            hinge_nonzero += 1
    assert_true(
        hinge_nonzero == DOG_N_HINGE,
        "the joint_angles block is not fully populated — perturbing from"
        " qpos0 is what keeps those 73 dims from being gated against zero",
    )


def _reward_reference(
    env_act: List[Float64],
    post_q: List[Float64],
    post_v: List[Float64],
) raises -> Tuple[PythonObject, PythonObject, PythonObject]:
    """MuJoCo forwarded at the POST-step state, ready for the ref factors."""
    var h = _mj()
    var mujoco = h[0]
    var m = h[1]
    var dat = h[2]
    for i in range(NQ):
        dat.qpos[i] = post_q[i]
    for i in range(NV):
        dat.qvel[i] = post_v[i]
    for i in range(NACT):
        dat.act[i] = env_act[i]
    mujoco.mj_forward(m, dat)
    return (mujoco, m, dat)


def test_dog_stand_reward_matches_dm_control() raises:
    """`Stand.get_reward` = the product of SIX factors, upright counted thrice.
    """
    print("--- dog: Stand reward vs dm_control ---")
    var env = DMDogStand[DType.float64](DeviceContext(), 1000)
    _ = env.reset()
    var qpos0 = List[Scalar[DType.float64]]()
    for i in range(NQ):
        qpos0.append(env.d.qpos.data[i])
    var st = _pose(qpos0, 0.30)
    var qpos = st[0].copy()
    var qvel = st[1].copy()
    var act = st[2].copy()

    env.set_state(qpos, qvel)
    for i in range(NACT):
        env.act[i] = Scalar[DType.float64](act[i])
    var a = type_of(env).ActionType()
    var res = env.step(a)
    var ours = Float64(res[1])

    # ⚠ `step` ADVANCES THE PHYSICS BEFORE COMPUTING THE REWARD, so the
    # reference must be built from the POST-step state, not from the pinned
    # pose. Comparing against `mj_forward` at the pinned pose is off by one
    # substep and reads as a formula error.
    var post_q = List[Float64]()
    var post_v = List[Float64]()
    var post_a = List[Float64]()
    for i in range(NQ):
        post_q.append(Float64(env.d.qpos.data[i]))
    for i in range(NV):
        post_v.append(Float64(env.d.qvel.data[i]))
    for i in range(NACT):
        post_a.append(Float64(env.act[i]))

    var h = _reward_reference(post_a, post_q, post_v)
    var m = h[1]
    var dat = h[2]

    var refmod = _ref_module()
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")
    var factors = refmod.stand_reward_factors(m, dat)
    assert_true(
        Int(py=builtins.len(factors)) == 6,
        "Stand's reward is SIX factors — torso, pelvis, THREE uprights"
        " (skull/torso/pelvis) and touch. Five means upright collapsed to one"
        " body, which is too generous by two factors and otherwise invisible.",
    )
    var want = Float64(py=np.prod(factors))
    print("  factors:", String(py=builtins.str(np.round(factors, 6))))
    print("  ours", ours, " ref", want)
    assert_true(
        abs(ours - want) <= REWARD_TOL,
        "Stand reward differs from dm_control",
    )

    # ⚠ NON-VACUITY. At the obvious pose (upright, high) five of these six are
    # exactly 1.0 and the product is 0.9 whatever the formula does. This pose
    # was chosen so they vary — assert that they still do, or the gate has
    # quietly stopped testing the reward.
    var n_saturated = 0
    for k in range(6):
        if Float64(py=factors[k]) >= 0.999999:
            n_saturated += 1
    print("  saturated factors:", n_saturated, "/ 6")
    assert_true(
        n_saturated <= 1,
        "the reward factors have saturated at 1.0 — this pose no longer"
        " discriminates between a right and a wrong reward formula",
    )
    # And the three uprights must be DISTINCT, or 'upright is three factors'
    # is indistinguishable from 'upright is one factor cubed'.
    var u0 = Float64(py=factors[2])
    var u1 = Float64(py=factors[3])
    var u2 = Float64(py=factors[4])
    assert_true(
        abs(u0 - u1) > 1e-6 or abs(u1 - u2) > 1e-6,
        "the skull/torso/pelvis uprights are identical at this pose, so a"
        " single-body upright cubed would pass too",
    )


def test_dog_move_reward_matches_dm_control() raises:
    """`Move.get_reward` = Stand's six times a seventh, forward-speed factor."""
    print("--- dog: Move(walk) reward vs dm_control ---")
    var env = DMDogWalk[DType.float64](DeviceContext(), 1000)
    _ = env.reset()
    var qpos0 = List[Scalar[DType.float64]]()
    for i in range(NQ):
        qpos0.append(env.d.qpos.data[i])
    var st = _pose(qpos0, 0.30)
    var qpos = st[0].copy()
    var qvel = st[1].copy()
    var act = st[2].copy()

    env.set_state(qpos, qvel)
    for i in range(NACT):
        env.act[i] = Scalar[DType.float64](act[i])
    var a = type_of(env).ActionType()
    var res = env.step(a)
    var ours = Float64(res[1])

    var post_q = List[Float64]()
    var post_v = List[Float64]()
    var post_a = List[Float64]()
    for i in range(NQ):
        post_q.append(Float64(env.d.qpos.data[i]))
    for i in range(NV):
        post_v.append(Float64(env.d.qvel.data[i]))
    for i in range(NACT):
        post_a.append(Float64(env.act[i]))

    var h = _reward_reference(post_a, post_q, post_v)
    var m = h[1]
    var dat = h[2]

    var refmod = _ref_module()
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")
    var factors = refmod.move_reward_factors(m, dat, DOG_WALK_SPEED)
    assert_true(
        Int(py=builtins.len(factors)) == 7,
        "Move's reward is Stand's six factors plus forward — `Move` subclasses"
        " `Stand` and its get_reward_factors calls super()",
    )
    var want = Float64(py=np.prod(factors))
    print("  factors:", String(py=builtins.str(np.round(factors, 6))))
    print("  ours", ours, " ref", want)
    assert_true(
        abs(ours - want) <= REWARD_TOL,
        "Move reward differs from dm_control",
    )

    # The seventh factor must not sit at its floor, or Move's reward is
    # indistinguishable from Stand's times a constant.
    var fwd = Float64(py=factors[6])
    print("  forward factor:", fwd, " (floor is 0.2 = (4*0+1)/5)")
    assert_true(
        fwd > 0.21,
        "the forward factor is at its floor — this pose has no forward speed,"
        " so the Move-specific half of the reward is untested",
    )


def test_dog_rollout_matches_mujoco() raises:
    """A CONTACTING rollout against MuJoCo — the gate `noslip` actually needs.

    Every other dog gate pins constants or reads accessors at a pinned pose.
    This one steps the physics, and it is the only place `mj_solNoSlip` is
    exercised for what it computes rather than merely for running: the pass
    moves nothing on a converged single-contact solve (measured 8.9e-16 in
    `tests/physics3d/test_noslip_vs_mujoco.mojo`), and dog is the model where
    it is worth 2.9e-2 of qvel on the first contacting step.

    ⚠ THE ROLLOUT MUST HAVE CONTACTS, which is the opposite of what the
    observation test wants. The dog is dropped and settled by MuJoCo first, so
    it starts loaded on its feet; a contact-free rollout here would gate the
    smooth dynamics twice and `noslip` not at all.

    ⚠ `act` IS PART OF THE STATE. All 38 actuators are `dyntype="filter"` and
    their force is `gainprm[0] * act`, so a rollout that synced only qpos/qvel
    would diverge through the activation even with identical `ctrl`. Both sides
    start from a reset (act all zero) and are driven with the same `ctrl`, so
    the filters integrate together — and any disagreement in that integration
    shows up here, which is the point.

    The contact-free prefix and the contacting phase are reported SEPARATELY.
    They are different claims: the prefix budgets smooth dynamics, the rest
    budgets the contact solver plus noslip, and averaging them would hide
    whichever is worse.
    """
    print("--- dog: contacting rollout vs MuJoCo ---")
    var env = DMDogStand[DType.float64](DeviceContext(), 1000)
    _ = env.reset()

    var h = _mj()
    var mujoco = h[0]
    var m = h[1]
    var dat = h[2]

    # Let MuJoCo settle the dog onto its feet from the compiler's qpos0, then
    # hand that state to us. Settling on OUR side instead would make the
    # comparison depend on the very solver under test.
    mujoco.mj_resetData(m, dat)
    for _ in range(N_SETTLE):
        mujoco.mj_step(m, dat)
    for k in range(NACT):
        dat.ctrl[k] = 0.0
        dat.act[k] = 0.0
    mujoco.mj_forward(m, dat)

    var q0 = List[Float64]()
    var v0 = List[Float64]()
    for i in range(NQ):
        q0.append(Float64(py=dat.qpos[i]))
    for i in range(NV):
        v0.append(Float64(py=dat.qvel[i]))
    env.set_state(q0, v0)
    for k in range(NACT):
        env.act[k] = Scalar[DType.float64](0)

    var worst_q_free = 0.0
    var worst_v_free = 0.0
    var worst_q_con = 0.0
    var worst_v_con = 0.0
    var worst_act = 0.0
    var contact_steps = 0
    var max_ncon = 0
    var first_q = 0.0
    var first_v = 0.0
    var seen_contact = False

    for step in range(N_ROLL):
        var a = type_of(env).ActionType()
        for k in range(NACT):
            var u = _action_at(step, k)
            dat.ctrl[k] = u
            a.data[k] = Scalar[DType.float64](u)
        for _ in range(DOG_FRAME_SKIP):
            mujoco.mj_step(m, dat)
        mujoco.mj_forward(m, dat)
        _ = env.step(a)

        var ncon = Int(py=dat.ncon)
        if ncon > max_ncon:
            max_ncon = ncon
        if ncon > 0:
            contact_steps += 1

        # ⚠ THE FIRST CONTACTING STEP IS THE GATE; the accumulated 30-step
        # number is NOT. With 14 simultaneous contacts the rollout is chaotic,
        # so any solver difference at all grows without bound — measured
        # |d(qvel)| 67 by step 30 — and budgeting that would either be
        # meaningless or force a tolerance so loose it could not fail. It is
        # also the horizon noslip's own claim is stated at: 2.9e-2 of qvel on
        # the FIRST contacting step, which is the number this has to resolve.
        if ncon > 0 and not seen_contact:
            seen_contact = True
            for i in range(NQ):
                var dq0 = abs(
                    Float64(py=dat.qpos[i]) - Float64(env.d.qpos.data[i])
                )
                if dq0 > first_q:
                    first_q = dq0
            for i in range(NV):
                var dv0 = abs(
                    Float64(py=dat.qvel[i]) - Float64(env.d.qvel.data[i])
                )
                if dv0 > first_v:
                    first_v = dv0

        for i in range(NQ):
            var dq = abs(
                Float64(py=dat.qpos[i]) - Float64(env.d.qpos.data[i])
            )
            if ncon > 0:
                if dq > worst_q_con:
                    worst_q_con = dq
            elif dq > worst_q_free:
                worst_q_free = dq
        for i in range(NV):
            var dv = abs(
                Float64(py=dat.qvel[i]) - Float64(env.d.qvel.data[i])
            )
            if ncon > 0:
                if dv > worst_v_con:
                    worst_v_con = dv
            elif dv > worst_v_free:
                worst_v_free = dv
        for k in range(NACT):
            var da = abs(Float64(py=dat.act[k]) - Float64(env.act[k]))
            if da > worst_act:
                worst_act = da

    print("  contacts on", contact_steps, "/", N_ROLL,
          "steps, max ncon", max_ncon)
    print("  FIRST contacting step: |d(qpos)|", first_q, " |d(qvel)|", first_v)
    print("  contact-free : |d(qpos)|", worst_q_free,
          " |d(qvel)|", worst_v_free)
    print("  contacting   : |d(qpos)|", worst_q_con,
          " |d(qvel)|", worst_v_con)
    print("  actuator act : |d(act)| ", worst_act)

    # ⚠ NON-VACUITY FIRST. A rollout that floated free, or one where every
    # actuator sat at zero, would post beautiful numbers and gate nothing.
    assert_true(
        contact_steps > N_ROLL // 2,
        "the dog is airborne for most of the rollout — this gates the smooth"
        " dynamics, not the contact solver, and noslip is untouched",
    )
    assert_true(
        max_ncon >= 4,
        "fewer than four contacts at any point — dog stands on four feet, so"
        " this is not a loaded pose",
    )

    # `act` is the cheapest thing to get wrong and the easiest to check: it is
    # a first-order filter driven by identical `ctrl` on both sides.
    assert_true(
        worst_act <= ACT_TOL,
        "the actuator activation diverged — dyntype=filter integration does"
        " not match MuJoCo, which corrupts every force before the solver even"
        " runs",
    )
    assert_true(
        worst_q_free <= SMOOTH_TOL and worst_v_free <= SMOOTH_TOL,
        "the contact-FREE part of the rollout diverged — that is smooth"
        " dynamics, and it should be near round-off",
    )
    assert_true(
        seen_contact,
        "no contacting step was reached — the gate measured nothing",
    )
    assert_true(
        first_q <= CONTACT_TOL and first_v <= CONTACT_TOL,
        "the FIRST contacting step diverged — this is the horizon at which"
        " noslip's own effect is stated (2.9e-2 of qvel), so a miss here means"
        " the contact solve or mj_solNoSlip is wrong, not that chaos grew",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
