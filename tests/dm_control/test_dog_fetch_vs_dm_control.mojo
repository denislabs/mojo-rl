"""dm_control `dog fetch` parity — model, observation and reward (Phase 5).

`Fetch` subclasses `Stand`, so this file gates only what fetch ADDS: the ball
body and its free joint, the target geom, and the two reward factors. Stand's
223 observation dims and six factors are gated by
`test_dog_tasks_vs_dm_control.mojo` and are not restated here — but the
observation comparison below covers all 232, because a base dim that broke
under fetch's model would otherwise go unseen.

⚠ THE `tennis_ball` MATERIAL IS A LABELLED DEVIATION. dog.xml dresses the ball
with a PNG cube-map texture; a ported XML carries no asset bundle, so
`dog_fetch_xml.mojo` substitutes a flat colour. That changes `mat_rgba` and the
ball's `geom_rgba` and NOTHING else — no table that feeds the dynamics reads a
material. The model comparison therefore exempts those two columns EXPLICITLY.
An exemption is a claim: if it ever covers a third column, this file should
fail rather than widen.

⚠ THE REWARD IS GATED AT TWO POSES, ON PURPOSE. `Fetch.get_reward_factors`
contains a discontinuity:

    if ball_to_target_distance < 2 * target_radius:  reach_ball = 1

Measured on this model, `2 * target_radius = 0.2`, and the two branches are far
apart — `reach_ball` is 0.3788 at 0.50 m and exactly 1.0 at 0.04 m. A fixture
that only samples one side would pass while the other branch was unimplemented,
which is the shape of dead test this project keeps finding.

⚠⚠ AND BOTH POSES ARE AIRBORNE. THIS IS THE FIX FOR DEFECT 20, NOT A WEAKENING.

The two branches were originally sampled with the dog ON THE FLOOR, and both
rows were red — by 1.3e-5 and 1.1e-2. Neither was an engine defect. Stand's
sixth factor is TOUCH, a sum of contact normal forces, and at the dog's reset
pose that sum is NUMERICALLY INDETERMINATE: the palms sit at ~zero clearance,
so the contact set and the redundant-contact force split bifurcate on rounding.
Measured on the REFERENCE ALONE, sweeping the root yaw over 25 samples at that
pose. Yaw is an exact symmetry here — the floor is a `plane` (geom_type 0), so
infinite and homogeneous; gravity is vertical; the free joint rotates the dog
about its own root; and the ball, which is placed in WORLD coordinates and so
does NOT rotate with the dog, is out of contact at every sampled yaw (checked,
not assumed). So none of these numbers may move, and they do:

    touch sum   32.6396 .. 40.2162      spread 22.6% of the mean
    ncon        18 .. 25
    palm_L/palm_R   10.05/10.05  ->  18.76/8.46   (no ball contact at any yaw)

Our own residual there is 0.13%, two orders INSIDE MuJoCo's spread. Pressing the
root down to force firmer contact makes it worse, not better (49%), so there is
no nearby well-conditioned floor fixture to move to. See
`test_dog_fetch_touch_probe.mojo` for the staged measurement.

Lifted 3 m the dog touches nothing, touch is 0 on BOTH sides and its factor is
the 0.9 floor, so what remains is exactly what this file exists to gate. And
`ball_to_target_distance` — the quantity the discontinuity keys on — is a world
-frame distance between the ball and the target geom, independent of the dog
entirely, so lifting the dog costs the branch coverage nothing.

It makes the waiver row STRICTLY MORE discriminating, in fact. `reach_ball` is
built from the ball-to-MOUTH distance, which airborne is ~3 m, so dropping the
`reach_ball = 1` waiver would give 1/7 rather than the 0.3788 it gives on the
floor — the failure it gates is now 6x larger.

⚠ CONSEQUENCE, AND IT IS A REAL COVERAGE HOLE: dog's touch factor is gated
NOWHERE. `test_dog_tasks_vs_dm_control` runs at a contact-free pose and asserts
as much, so its touch factor is the 0.9 floor too. `touch_sphere_site` itself is
gated on hopper (aggregate, for the same underlying reason — see that file's
docstring), finger and manipulator. What is unpinned is dog's touch
specifically, and it cannot be pinned tightly at any pose this model reaches.

⚠ THE FIXTURE IS STILL A RANDOM DRAW. `DMDogFetchConfig.custom_reset_cpu` takes
the root yaw from the GLOBAL RNG and each row builds a fresh env, so the rows
are different yaws and adding one reshuffles the others. That is harmless here
BECAUSE the rows are airborne — with no contacts the reward is smooth in the
draw — and it is exactly what made the floor rows brittle.

⚠ THE HEAD FRAME IS A SITE FRAME AND THE ROTATION IS WORLD -> SITE. The
reference writes `v.dot(head_frame)`, which under numpy's row-vector convention
is `R^T v`. Transposing it is the single most plausible way to get `ball_state`
and `target_position` wrong, and no norm-based check would notice — so the
comparison is per-component.

Run with:
    pixi run mojo run -I . tests/dm_control/test_dog_fetch_vs_dm_control.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.dog import (
    DMDogFetch,
    DMDogFetchModel,
    DOG_FETCH_OBS_DIM,
)
from mojo_rl.envs.dm_control.dog.dog_xml import DOG_FRAME_SKIP
from mojo_rl.envs.dm_control.dog.dog_fetch_xml import (
    FETCH_BALL_QPOS_0,
    FETCH_BALL_DOF_0,
    FETCH_TARGET_RADIUS,
)

comptime DTYPE = DType.float64
comptime M = DMDogFetchModel
comptime NQ = M.NQ
comptime NV = M.NV
comptime NACT = M.nact
comptime TEST_PATH = "tests/dm_control"

comptime OBS_TOL: Float64 = 1e-11
comptime REWARD_TOL: Float64 = 1e-11


def _ref() raises -> Tuple[PythonObject, PythonObject, PythonObject, PythonObject]:
    var sys = Python.import_module("sys")
    sys.path.insert(0, TEST_PATH)
    var mujoco = Python.import_module("mujoco")
    var refmod = Python.import_module("dog_ref")
    var m = refmod.model(10, False)  # floor_size=10, remove_ball=False
    return (mujoco, m, mujoco.MjData(m), refmod)


def test_dog_fetch_model_matches_dm_control() raises:
    """Counts and the per-table diff, with the material deviation named."""
    print("--- dog fetch: model vs dm_control ---")
    var h = _ref()
    var m = h[1]

    print(
        "  ours nbody", M.NBODY, " njnt", M.NJOINT, " nq", M.NQ,
        " nv", M.NV, " ngeom", M.NGEOM, " nsite", M.NSITE,
    )
    print(
        "  mj   nbody", Int(py=m.nbody), " njnt", Int(py=m.njnt),
        " nq", Int(py=m.nq), " nv", Int(py=m.nv),
        " ngeom", Int(py=m.ngeom), " nsite", Int(py=m.nsite),
    )
    assert_true(M.NBODY == Int(py=m.nbody), "nbody differs")
    assert_true(M.NJOINT == Int(py=m.njnt), "njnt differs")
    assert_true(M.NQ == Int(py=m.nq), "nq differs")
    assert_true(M.NV == Int(py=m.nv), "nv differs")
    assert_true(M.NGEOM == Int(py=m.ngeom), "ngeom differs")
    assert_true(M.NSITE == Int(py=m.nsite), "nsite differs")
    assert_true(M.nact == Int(py=m.nu), "nact differs")

    # NON-VACUITY: fetch must actually differ from stand, or this file is
    # gating the stand model under another name.
    assert_true(
        Int(py=m.nq) == 87 and Int(py=m.ngeom) == 134,
        "this is not the fetch model — stand is nq 80 / ngeom 128, and fetch"
        " must carry the ball's free joint and the ball/target/wall geoms",
    )


def _pose_and_step(
    mut env: DMDogFetch[DTYPE], ball_x: Float64, root_dz: Float64 = 0.0
) raises -> List[Float64]:
    """Reset, place the ball at `(ball_x, 0, 0.05)` moving, then STEP once.

    Returns the POST-step `[qpos, qvel, act]`, because that is the state the
    reference must be pinned to. There is no `get_reward()` on the env — the
    reward comes out of `step`, exactly as `test_dog_tasks_vs_dm_control` does
    it, so the comparison has to be at the post-step state either way.

    ⚠ THE BALL MUST BE MOVING. With a zero ball velocity the second half of
    `ball_state` is three zeros and a wrong velocity transport — the very thing
    `point_velocity_world` was extracted for — would pass.
    """
    _ = env.reset()
    var q = List[Float64]()
    for i in range(NQ):
        q.append(Float64(env.d.qpos.data[i]))
    # `root_dz > 0` lifts the dog clear of the floor, which zeroes the TOUCH
    # sensors on both sides and so removes Stand's touch factor from the
    # comparison — the only way to gate fetch's OWN two factors on their own.
    q[2] = q[2] + root_dz
    q[FETCH_BALL_QPOS_0 + 0] = ball_x
    q[FETCH_BALL_QPOS_0 + 1] = 0.0
    q[FETCH_BALL_QPOS_0 + 2] = 0.05
    q[FETCH_BALL_QPOS_0 + 3] = 1.0
    q[FETCH_BALL_QPOS_0 + 4] = 0.0
    q[FETCH_BALL_QPOS_0 + 5] = 0.0
    q[FETCH_BALL_QPOS_0 + 6] = 0.0

    var v = List[Float64]()
    for _ in range(NV):
        v.append(0.0)
    v[FETCH_BALL_DOF_0 + 0] = 0.7
    v[FETCH_BALL_DOF_0 + 1] = -0.3
    v[FETCH_BALL_DOF_0 + 2] = 1.1

    var pre = List[Float64]()
    for i in range(NQ):
        pre.append(q[i])
    for i in range(NV):
        pre.append(v[i])

    env.set_state(q, v)
    for k in range(NACT):
        env.act[k] = Scalar[DTYPE](0)
    var a = type_of(env).ActionType()
    for k in range(NACT):
        a.data[k] = Scalar[DTYPE](0)
    var res = env.step(a)

    var out = List[Float64]()
    for i in range(NQ):
        out.append(Float64(env.d.qpos.data[i]))
    for i in range(NV):
        out.append(Float64(env.d.qvel.data[i]))
    for k in range(NACT):
        out.append(Float64(env.act[k]))
    out.append(Float64(res[1]))          # the reward this step produced
    # ...then the PRE-step state, for the lockstep reference below.
    for i in range(NQ + NV):
        out.append(pre[i])
    return out^


def _pin_reference(
    mujoco: PythonObject, m: PythonObject, dat: PythonObject,
    qv: List[Float64],
) raises:
    for i in range(NQ):
        dat.qpos[i] = qv[i]
    for i in range(NV):
        dat.qvel[i] = qv[NQ + i]
    for k in range(NACT):
        dat.act[k] = qv[NQ + NV + k]
        dat.ctrl[k] = 0.0
    mujoco.mj_forward(m, dat)


def test_dog_fetch_observation_matches_dm_control() raises:
    """All 232 dims, at a pose where the ball is off-axis and moving."""
    print("--- dog fetch: observation vs dm_control ---")
    var h = _ref()
    var mujoco = h[0]
    var m = h[1]
    var dat = h[2]
    var refmod = h[3]

    var env = DMDogFetch[DTYPE](DeviceContext(), 1000)
    var qv = _pose_and_step(env, 0.5)
    var obs = env.get_obs_list()
    assert_true(
        len(obs) == DOG_FETCH_OBS_DIM,
        "observation is not 232 long",
    )
    _pin_reference(mujoco, m, dat, qv)
    var want = refmod.fetch_observation(m, dat)

    # ⚠ THE 15 ACCELERATION-STAGE DIMS ARE EXCLUDED, and not for convenience.
    # `mj_forward` at a post-step state recomputes the acceleration stage
    # FRESH, which is a quantity dm_control never reports: its `Physics.step()`
    # ends with `mj_step1`, refreshing only the position and velocity stages.
    # That was defect 19. Those dims — accelerometer 160..162 and the four
    # force sensors 169..180 — are gated against dm_control's OWN stepping in
    # `test_dog_tasks_vs_dm_control.mojo`, on the same sensors and the same
    # code path. Comparing them here would re-fail a fixed defect against a
    # reference that cannot produce the right number.
    comptime ACC_0 = 160
    comptime ACC_N = 3
    comptime FRC_0 = 169
    comptime FRC_N = 12
    # ⚠ AND TOUCH (181..184), WHICH IS ALSO ACCELERATION-STAGE — it sums
    # contact normal forces. `test_dog_tasks_vs_dm_control` leaves touch in the
    # pinned comparison and is right to: ITS pose is contact-free by assertion,
    # so touch is identically zero on both sides. Fetch's dog is standing on
    # the floor, so touch is live here and the pinned reference is wrong for it
    # by exactly the same defect-19 timing. Measured 4.36 at dim 182 before
    # this exclusion.
    comptime TCH_0 = 181
    comptime TCH_N = 4
    var worst = 0.0
    var worst_i = -1
    var n_cmp = 0
    for i in range(DOG_FETCH_OBS_DIM):
        if i >= ACC_0 and i < ACC_0 + ACC_N:
            continue
        if i >= FRC_0 and i < FRC_0 + FRC_N:
            continue
        if i >= TCH_0 and i < TCH_0 + TCH_N:
            continue
        n_cmp += 1
        var e = abs(Float64(obs[i]) - Float64(py=want[i]))
        if e > worst:
            worst = e
            worst_i = i
    print("  worst |err| over", n_cmp, "pos/vel-stage dims =", worst,
          "at", worst_i)
    assert_true(
        n_cmp == DOG_FETCH_OBS_DIM - ACC_N - FRC_N - TCH_N,
        "the acceleration-stage exclusion is the wrong width",
    )

    # The 9 fetch-only dims reported separately: a base-observation regression
    # and a fetch bug are different problems and should not be averaged.
    var worst_new = 0.0
    var worst_new_i = -1
    for i in range(223, DOG_FETCH_OBS_DIM):
        var e = abs(Float64(obs[i]) - Float64(py=want[i]))
        if e > worst_new:
            worst_new = e
            worst_new_i = i
    print("  worst over the 9 fetch dims (223..231) =", worst_new,
          "at", worst_new_i)

    # NON-VACUITY: ball_state must be far from zero, or a wrong head-frame
    # rotation would pass. The ball sits 0.5 m away and moves at ~1.3 m/s.
    var mag = 0.0
    for i in range(223, DOG_FETCH_OBS_DIM):
        var v = abs(Float64(py=want[i]))
        if v > mag:
            mag = v
    print("  |fetch dims|_inf =", mag, " (must be >> 0)")
    assert_true(
        mag > 0.1,
        "the ball/target observation is ~zero at this pose, so a transposed"
        " head frame or a dropped velocity term would pass unnoticed",
    )
    assert_true(worst <= OBS_TOL, "dog fetch observation differs")


def _reward_at(
    ball_x: Float64, label: String, root_dz: Float64 = 0.0
) raises -> Tuple[Float64, Float64, Bool]:
    """Our reward and the reference's at a ball placed `ball_x` from origin."""
    var h = _ref()
    var mujoco = h[0]
    var m = h[1]
    var dat = h[2]
    var refmod = h[3]

    var env = DMDogFetch[DTYPE](DeviceContext(), 1000)
    var qv = _pose_and_step(env, ball_x, root_dz)
    var ours = qv[NQ + NV + NACT]

    # ⚠ DRIVEN, NOT PINNED. Stand's sixth factor is TOUCH, which is
    # acceleration-stage: `mj_forward` at a post-step state recomputes it
    # fresh, which dm_control never does (defect 19). With the dog standing on
    # the floor that is worth 8e-3 of reward. So the reference runs the same
    # FRAME_SKIP substeps from the same start and finishes with `mj_step1`,
    # which is exactly `Physics._step_with_up_to_date_position_velocity`.
    comptime PRE = NQ + NV + NACT + 1
    for i in range(NQ):
        dat.qpos[i] = qv[PRE + i]
    for i in range(NV):
        dat.qvel[i] = qv[PRE + NQ + i]
    for k in range(NACT):
        dat.act[k] = 0.0
        dat.ctrl[k] = 0.0
    mujoco.mj_forward(m, dat)
    for _ in range(DOG_FRAME_SKIP):
        mujoco.mj_step(m, dat)
    mujoco.mj_step1(m, dat)

    var np = Python.import_module("numpy")
    var factors = refmod.fetch_reward_factors(m, dat)
    var want = Float64(py=np.prod(factors))
    var dist = Float64(py=refmod.ball_to_target_distance(m, dat))
    var active = dist < 2.0 * FETCH_TARGET_RADIUS

    print(
        "  ", label, ": ball_to_target", dist,
        " discontinuity active", active,
    )
    print(
        "      factors", String(py=np.round(factors, 6).__str__()),
    )
    print("      ours", ours, " ref", want)
    return (ours, want, active)


def test_dog_fetch_reward_matches_dm_control() raises:
    """The 8-factor product, on BOTH sides of the reward's discontinuity."""
    print("--- dog fetch: reward vs dm_control ---")

    # ⚠ BOTH ROWS LIFTED 3 m. On the floor, Stand's sixth factor is TOUCH, a
    # sum of contact normal forces, and any difference in it multiplies straight
    # through the product and masks `reach_ball`/`fetch_ball` entirely — worse,
    # at this model's reset pose that sum is indeterminate to 22.6% in MuJoCo
    # ITSELF under a symmetry of the fixture, so a floor row cannot fail for a
    # reason worth acting on. Airborne, touch is 0 on both sides and its factor
    # is the 0.9 floor, so what remains IS fetch. Header, defect 20.
    var far = _reward_at(0.5, String("airborne, far from target"), 3.0)
    var near = _reward_at(
        0.03, String("airborne, inside 2*target_radius"), 3.0
    )

    # NON-VACUITY: lifting the dog must not have zeroed what is being compared.
    # `reach_ball`/`fetch_ball` are built from the ball and target positions in
    # the HEAD frame, and the head moved 3 m — so assert the rewards are still
    # ordinary numbers, not both collapsed to 0 or both saturated at 1.
    assert_true(
        far[1] > 1e-6 and far[1] < 1.0 - 1e-6,
        "the reference reward is 0 or 1 with the dog airborne, so fetch's two"
        " factors are saturated and this row gates nothing",
    )

    # The fixture must straddle the branch, or one side is never gated.
    assert_true(
        (not far[2]) and near[2],
        "the two poses do not straddle `ball_to_target < 2*target_radius` —"
        " one branch of the reward is untested and this file would pass with"
        " the waiver unimplemented",
    )
    # And the branch must MATTER, or straddling it proves nothing.
    assert_true(
        abs(far[1] - near[1]) > 1e-3,
        "the reference reward is the same on both sides of the"
        " discontinuity, so crossing it gates nothing here",
    )

    assert_true(
        abs(far[0] - far[1]) <= REWARD_TOL,
        "dog fetch reward differs from dm_control far from the target, with the"
        " dog airborne — so Stand's touch factor is the 0.9 floor on both sides"
        " and this is `reach_ball`/`fetch_ball`, fetch's own arithmetic, not a"
        " contact-force difference",
    )
    assert_true(
        abs(near[0] - near[1]) <= REWARD_TOL,
        "dog fetch reward differs from dm_control INSIDE"
        " `2*target_radius` — suspect the `reach_ball = 1` waiver, which the"
        " reference applies AFTER the (6x+1)/7 rescaling rather than before",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
