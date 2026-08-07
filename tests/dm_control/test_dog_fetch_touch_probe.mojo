"""dog fetch floor-pose residual — a STAGED probe, not a gate.

`test_dog_fetch_vs_dm_control::test_dog_fetch_reward_matches_dm_control` is
exact with the dog airborne and red on both floor poses. Run 1 of this probe
(3 stages x 4 poses) produced two results, and the SECOND one invalidates the
first as a comparison:

  * at one fixture our step tracked MuJoCo — |d(qpos)| 3.4e-4, touch sum
    32.9998 vs 32.9580 — and at another it did not, at all: |d(qvel)| 2.49,
    ncon 16 vs 21, and BOTH FRONT PALMS reading 0.0 against MuJoCo's 18.41
    and 8.47.
  * but the reference's own touch differed between those two fixtures
    (39.87 vs 32.96) although the only intended difference was where the ball
    sat, and the ball never touches the dog at either place. So the fixtures
    were not the same dog.

⚠⚠ THE GATE'S FIXTURE IS RANDOM. `DMDogFetchConfig.custom_reset_cpu` draws
from the GLOBAL RNG (yaw, three root velocities, the ball throw), so every
`DMDogFetch` construction in a process gets a different draw. The reward gate
builds a fresh env per pose, which makes its three rows three different dogs —
and makes the residual it reports a function of the draw, not of the pose.
Reproducing 1.3e-5 or 3e-2 is then luck.

So this run pins the fixture instead of accepting one: BOTH sides start from a
state MuJoCo produced, and no `env.reset()` result is used.

    A. the model's default pose      mj_resetData — the configuration the
                                     random reset perturbs, and one that is
                                     NOT settled: the dog is dropped into it
    B. settled, 400 MuJoCo steps     the physically meaningful contacting
                                     pose, and the one `test_dog_step_probe`
                                     already reaches 8.69e-13 from

Both are reproducible in Python (scratchpad/ball_clearance.py drives A), so
any residual can be chased on the reference side without a Mojo rebuild.

WHAT IT MEASURED (2026-08-07, both fixtures pinned)

  A. default pose   ncon 24 vs 24, the SAME four body pairs against the floor,
                    normal forces to 0.2% (3.2127 vs 3.2125, 5.0823 vs 5.0705),
                    touch sum 32.9998 vs 32.9579 = 0.13%, heights to 1e-8.
                    State drifts |d(qpos)| 3.4e-4 over the control step, and the
                    contact DISTANCES carry it — ours -3.618e-4 where MuJoCo has
                    -5.477e-4, i.e. the same force at a different penetration,
                    which is the position difference and not a stiffness one.
                    ⇒ nothing here is an engine defect at the 22.6% scale of the
                    fixture's own indeterminacy. Defect 20 closes on this row.

  B. settled 400    ⚠ ncon 10 vs 23, |d(qvel)| 2.25. NOT a rounding difference:
                    we are missing whole rows, including BOTH `(55,49)` and
                    `(61,49)` dim-1 self-contacts and most of the condim-6
                    rows on bodies 52/58, while carrying two 1.5 cm-deep
                    penetrations MuJoCo does not have. Tracked separately — it
                    is a contact-set question, not a sensor one, and this file
                    is not the instrument for it.

Stages, in the order the engine computes them:

    1. the post-step STATE   |d(qpos)|, |d(qvel)| vs a lockstep reference
    2. the CONTACT SET       every contact, both sides, by body pair
    3. the TOUCH SENSORS     per-site, ours vs `sensordata`
    4. the OTHER factors     torso/pelvis height

Stage 1 comes first on purpose: a touch sum computed at a different pose is
not a sensor defect, and stage 2 before any force comparison because a solver
fed a different set of rows is not a solver bug.

WHY TOUCH HAS NEVER BEEN GATED ON DOG BEFORE

`test_dog_tasks_vs_dm_control` runs at a CONTACT-FREE pose — it asserts as much
("the 4 touch sensors are 0 at a contact-free pose") and Stand's touch factor
there is the 0.9 floor on both sides. So dog's Stand/Move reward parity, which
is exact, says nothing about touch. Hopper's is deliberately AGGREGATE (its
docstring explains why). Fetch's floor poses are the first tight comparison of
`touch_sphere_site` anywhere in the tree.

Run with:
    pixi run mojo run -I . tests/dm_control/test_dog_fetch_touch_probe.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from std.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.dog import (
    DMDogFetch,
    DMDogFetchModel,
    DOG_SITE_PALM_L,
    DOG_SITE_PALM_R,
    DOG_SITE_SOLE_L,
    DOG_SITE_SOLE_R,
    DOG_BODY_WEIGHT,
    DOG_TORSO_BODY_IDX,
    DOG_PELVIS_BODY_IDX,
)
from mojo_rl.envs.dm_control.dog.dog_xml import DOG_FRAME_SKIP
from mojo_rl.envs.dm_control.dog.dog_fetch_xml import (
    FETCH_BALL_QPOS_0,
    FETCH_BALL_DOF_0,
    FETCH_BALL_BODY_IDX,
)
from mojo_rl.physics3d.sensors.touch import touch_sphere_site
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_DIST,
    CONTACT_IDX_FORCE_N,
    CONTACT_IDX_CONDIM,
    META_IDX_NUM_CONTACTS,
)

comptime DTYPE = DType.float64
comptime M = DMDogFetchModel
comptime NQ = M.NQ
comptime NV = M.NV
comptime NACT = M.nact
comptime TEST_PATH = "tests/dm_control"


def _touch_sites() -> List[Int]:
    """The four `<touch>` sites, in `_touch_sum`'s order.

    A runtime `List`, not a `comptime InlineArray` — the loop below indexes it
    with a runtime `k`, and a comptime aggregate subscripted that way is one of
    this codebase's recurring compile failures.
    """
    var s = List[Int]()
    s.append(DOG_SITE_PALM_L)
    s.append(DOG_SITE_PALM_R)
    s.append(DOG_SITE_SOLE_L)
    s.append(DOG_SITE_SOLE_R)
    return s^


def _touch_names() -> List[String]:
    var s = List[String]()
    s.append(String("palm_L"))
    s.append(String("palm_R"))
    s.append(String("sole_L"))
    s.append(String("sole_R"))
    return s^


def _ref() raises -> Tuple[
    PythonObject, PythonObject, PythonObject, PythonObject
]:
    var sys = Python.import_module("sys")
    sys.path.insert(0, TEST_PATH)
    var mujoco = Python.import_module("mujoco")
    var refmod = Python.import_module("dog_ref")
    var m = refmod.model(10, False)  # floor_size=10, remove_ball=False
    return (mujoco, m, mujoco.MjData(m), refmod)


def _stage(n_settle: Int, ball_x: Float64, label: String) raises:
    """One pinned fixture through all four stages.

    `n_settle` MuJoCo steps from `mj_resetData` define the pre-state — 0 is the
    model's default pose. The ball is then placed at `(ball_x, 0, 0.05)` and
    thrown, and every dog velocity is zeroed, so the fixture is a pure function
    of `(n_settle, ball_x)` on BOTH sides.
    """
    print()
    print("=== ", label, " (settle", n_settle, " ball_x", ball_x, ") ===")

    var h = _ref()
    var mujoco = h[0]
    var m = h[1]
    var dat = h[2]
    var refmod = h[3]
    var np = Python.import_module("numpy")

    # --- the pinned pre-state, built by MuJoCo -----------------------------
    mujoco.mj_resetData(m, dat)
    for _ in range(n_settle):
        mujoco.mj_step(m, dat)
    for k in range(NACT):
        dat.act[k] = 0.0
        dat.ctrl[k] = 0.0
    var pre = List[Float64]()
    for i in range(NQ):
        pre.append(Float64(py=dat.qpos[i]))
    for _ in range(NV):
        pre.append(0.0)
    pre[FETCH_BALL_QPOS_0 + 0] = ball_x
    pre[FETCH_BALL_QPOS_0 + 1] = 0.0
    pre[FETCH_BALL_QPOS_0 + 2] = 0.05
    pre[FETCH_BALL_QPOS_0 + 3] = 1.0
    pre[FETCH_BALL_QPOS_0 + 4] = 0.0
    pre[FETCH_BALL_QPOS_0 + 5] = 0.0
    pre[FETCH_BALL_QPOS_0 + 6] = 0.0
    pre[NQ + FETCH_BALL_DOF_0 + 0] = 0.7
    pre[NQ + FETCH_BALL_DOF_0 + 1] = -0.3
    pre[NQ + FETCH_BALL_DOF_0 + 2] = 1.1

    # --- our side ----------------------------------------------------------
    var env = DMDogFetch[DTYPE](DeviceContext(), 1000)
    _ = env.reset()          # discarded: its draw is random, see the header
    var q = List[Float64]()
    var v = List[Float64]()
    for i in range(NQ):
        q.append(pre[i])
    for i in range(NV):
        v.append(pre[NQ + i])
    env.set_state(q, v)
    for k in range(NACT):
        env.act[k] = Scalar[DTYPE](0)
    var a = type_of(env).ActionType()
    for k in range(NACT):
        a.data[k] = Scalar[DTYPE](0)
    _ = env.step(a)

    # --- the reference, driven as `Physics.step()` drives it ---------------
    for i in range(NQ):
        dat.qpos[i] = pre[i]
    for i in range(NV):
        dat.qvel[i] = pre[NQ + i]
    for k in range(NACT):
        dat.act[k] = 0.0
        dat.ctrl[k] = 0.0
    mujoco.mj_forward(m, dat)
    for _ in range(DOG_FRAME_SKIP):
        mujoco.mj_step(m, dat)
    mujoco.mj_step1(m, dat)

    # --- 1. the post-step state -------------------------------------------
    var wq = 0.0
    var wq_at = -1
    for i in range(NQ):
        var e = abs(Float64(env.d.qpos.data[i]) - Float64(py=dat.qpos[i]))
        if e > wq:
            wq = e
            wq_at = i
    var wv = 0.0
    var wv_at = -1
    for i in range(NV):
        var e = abs(Float64(env.d.qvel.data[i]) - Float64(py=dat.qvel[i]))
        if e > wv:
            wv = e
            wv_at = i
    print("  1. state   |d(qpos)| =", wq, "at", wq_at,
          "   |d(qvel)| =", wv, "at", wv_at)

    # --- 2. the contact set ------------------------------------------------
    var nc = Int(Float64(env.d.meta.data[META_IDX_NUM_CONTACTS]))
    var ref_nc = Int(py=dat.ncon)
    print("  2. contacts   ours ncon", nc, "   MuJoCo ncon", ref_nc)
    print("     ours:  (body_a, body_b)  dist  fn  dim")
    for c in range(nc):
        var o = c * CONTACT_SIZE
        print(
            "       (",
            Int(Float64(env.d.contacts.data[o + CONTACT_IDX_BODY_A])), ",",
            Int(Float64(env.d.contacts.data[o + CONTACT_IDX_BODY_B])), ")",
            Float64(env.d.contacts.data[o + CONTACT_IDX_DIST]),
            Float64(env.d.contacts.data[o + CONTACT_IDX_FORCE_N]),
            Int(Float64(env.d.contacts.data[o + CONTACT_IDX_CONDIM])),
        )
    print("     MuJoCo:  (body1, body2)  dist  fn  dim")
    var buf = np.zeros(6)
    for c in range(ref_nc):
        var con = dat.contact[c]
        mujoco.mj_contactForce(m, dat, c, buf)
        print(
            "       (",
            Int(py=m.geom_bodyid[Int(py=con.geom1)]), ",",
            Int(py=m.geom_bodyid[Int(py=con.geom2)]), ")",
            Float64(py=con.dist), Float64(py=buf[0]), Int(py=con.dim),
        )
    print("     ball body is", FETCH_BALL_BODY_IDX)

    # --- 3. the touch sensors ----------------------------------------------
    var sites = _touch_sites()
    var names = _touch_names()
    var ours_sum = 0.0
    var ref_sum = 0.0
    for k in range(4):
        var mine = touch_sphere_site[DTYPE](
            env.d, env.mf.sites.data, sites[k], 1.0
        )
        var theirs = Float64(
            py=np.sum(refmod._named_sensor(m, dat, names[k]))
        )
        ours_sum += mine
        ref_sum += theirs
        print(
            "  3. touch  ", names[k], "  ours", mine, " MuJoCo", theirs,
            "  |d|", abs(mine - theirs),
        )
    print(
        "     sum  ours", ours_sum, " MuJoCo", ref_sum,
        "   body weight", DOG_BODY_WEIGHT,
    )

    # --- 4. the other stand factors ----------------------------------------
    print(
        "  4. heights  torso ours",
        Float64(env.d.xpos.data[DOG_TORSO_BODY_IDX * 3 + 2]),
        " MuJoCo", Float64(py=dat.xpos[DOG_TORSO_BODY_IDX][2]),
        "   pelvis ours",
        Float64(env.d.xpos.data[DOG_PELVIS_BODY_IDX * 3 + 2]),
        " MuJoCo", Float64(py=dat.xpos[DOG_PELVIS_BODY_IDX][2]),
    )


def test_dog_fetch_touch_probe() raises:
    """Two PINNED fixtures — the default pose and a settled one."""
    print("--- dog fetch: touch / contact residual probe ---")
    _stage(0, 0.5, String("A. model default pose, ball far"))
    _stage(400, 0.5, String("B. settled 400 steps, ball far"))

    assert_true(True, "probe")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
