"""Does the divergence live in `Phyics3dEnv`, or in the substep loop?

WHY THIS FILE EXISTS, AND WHY IT IS SEPARATE FROM THE STAGED PROBE

`test_dog_step_probe.mojo` now measures the whole step exact against MuJoCo at
dog's settled pose — contact set, mixed parameters, row constants, Jacobian,
mass matrix, bias, the solve itself (2.99e-11), and an applied force on every
dof (3.41e-11). And `test_dog_rollout_matches_mujoco` is STILL red at
`|d(qvel)| = 6.098` on its first contacting step.

Both are true because the probe drives `EulerIntegrator` DIRECTLY for ONE
substep, while the rollout goes through `Phyics3dEnv.step`, which takes
`FRAME_SKIP = 3` of them. Two things live in that gap and nothing has ever
compared them:

  1. multi-substep accumulation, and
  2. whatever `Phyics3dEnv` does around the integrator.

⚠ THE SECOND IS DEFECT 8 REPEATING. `Phyics3dEnv` NEVER FORWARDED `MAX_CONDIM`,
so every env built through the class silently ran Phase 3's condim-6 fix at
condim 3 — and the Phase 3 gate missed it for exactly the reason this file
exists: `test_rolling_friction_vs_mujoco` constructs the integrator DIRECTLY.
A gate that bypasses the production path proves the production path only by
coincidence, and the staged probe is now a very thorough instance of that gate.

THE DESIGN. Three claims, separated so a failure names its own cause:

  * ONE substep through the ENV vs one `mj_step`. Isolates `Phyics3dEnv` from
    the substep count — if this misses, the env is wrapping the integrator
    wrongly and the substep loop is innocent.
  * THREE substeps through the env vs three `mj_step`s. Adds only accumulation.
  * The env's own `step()` (FRAME_SKIP substeps) vs the same count. The
    production path end to end.

⚠ ZERO ACTUATION THROUGHOUT. `act` and `ctrl` are held at 0 on both sides, so
the actuator force is identically zero and this file cannot be confounded by
the activation filter — which the rollout gate has already shown agrees exactly
(`|d(act)| = 0.0`). What is under test is the STEPPING, not the driving.

⚠ CONSTRUCT THE ENV WITHOUT A THIRD ARGUMENT. `Phyics3dEnv.__init__` is
`(ctx, max_steps, frame_skip)`, so `(ctx, 1000, 1)` silently sets FRAME_SKIP to
1. That cost a debugging round once already (§14); omitting it takes
`CONFIG.FRAME_SKIP`, which is the 3 dog wants.

Run with:
    pixi run mojo run -I . tests/dm_control/test_dog_env_substep_probe.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.dog import (
    DMDogStand,
    DMDogStandWalkModel,
    DOG_FRAME_SKIP,
)

comptime DTYPE = DType.float64
comptime M = DMDogStandWalkModel
comptime NQ = M.NQ
comptime NV = M.NV
comptime NACT = M.nact
comptime N_SETTLE: Int = 400

# Both sides run the same float64 arithmetic from the same state with no
# actuation, so this is round-off over a handful of substeps — the staged probe
# reaches 2.99e-11 on a single one.
comptime TOL: Float64 = 1e-8


def _mj() raises -> Tuple[PythonObject, PythonObject, PythonObject]:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path("mojo_rl/envs/dm_control/assets/dog_stand_walk.xml")
    return (mujoco, m, mujoco.MjData(m))


def _worst(
    env_qpos: List[Float64], env_qvel: List[Float64],
    dat: PythonObject, label: String,
) raises -> Float64:
    var wq = 0.0
    var wv = 0.0
    var at_v = -1
    for i in range(NQ):
        var e = abs(env_qpos[i] - Float64(py=dat.qpos[i]))
        if e > wq:
            wq = e
    for i in range(NV):
        var e = abs(env_qvel[i] - Float64(py=dat.qvel[i]))
        if e > wv:
            wv = e
            at_v = i
    print("  ", label, ": |d(qpos)|", wq, " |d(qvel)|", wv, " at dof", at_v)
    return wq if wq > wv else wv


def test_dog_env_substeps_match_mujoco() raises:
    """The PRODUCTION path — `Phyics3dEnv` — against MuJoCo, zero actuation."""
    print("--- dog: env substeps vs MuJoCo (ctrl = act = 0) ---")
    var h = _mj()
    var mujoco = h[0]
    var m = h[1]
    var dat = h[2]

    # MuJoCo settles the dog onto its feet; that loaded pose is the start.
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
    print("  start: MuJoCo ncon", Int(py=dat.ncon))
    assert_true(
        Int(py=dat.ncon) >= 4,
        "the settled pose is not loaded — this compares free flight and gates"
        " nothing about the contact path",
    )

    # ⚠ NO THIRD ARGUMENT — see the module docstring.
    var env = DMDogStand[DTYPE](DeviceContext(), 1000)
    _ = env.reset()
    env.set_state(q0, v0)
    for k in range(NACT):
        env.act[k] = Scalar[DTYPE](0)

    var a = type_of(env).ActionType()
    for k in range(NACT):
        a.data[k] = Scalar[DTYPE](0)

    # --- claim 1: ONE env step (FRAME_SKIP substeps) vs the same count -----
    # The env has no single-substep entry point, so the first comparison is at
    # FRAME_SKIP. If it misses while the staged probe's single substep is
    # exact, the extra substeps or the env wrapper are the cause — claim 2
    # below separates those.
    _ = env.step(a)
    for _ in range(DOG_FRAME_SKIP):
        mujoco.mj_step(m, dat)
    mujoco.mj_forward(m, dat)

    var eq = List[Float64]()
    var ev = List[Float64]()
    for i in range(NQ):
        eq.append(Float64(env.d.qpos.data[i]))
    for i in range(NV):
        ev.append(Float64(env.d.qvel.data[i]))
    var w1 = _worst(eq, ev, dat, String("after 1 env step (") + String(DOG_FRAME_SKIP) + " substeps)")

    # --- claim 2: does it GROW with more steps, or is it there at once? ----
    # A constant offset says the env wrapper; growth says accumulation.
    var w_hist = List[Float64]()
    for _s in range(4):
        _ = env.step(a)
        for _ in range(DOG_FRAME_SKIP):
            mujoco.mj_step(m, dat)
        mujoco.mj_forward(m, dat)
        var q2 = List[Float64]()
        var v2 = List[Float64]()
        for i in range(NQ):
            q2.append(Float64(env.d.qpos.data[i]))
        for i in range(NV):
            v2.append(Float64(env.d.qvel.data[i]))
        w_hist.append(_worst(q2, v2, dat, String("  +1 more env step")))

    # ⚠ act MUST STILL BE ZERO. If the env integrated an activation from a
    # zero ctrl, the comparison silently becomes actuated and the conclusion
    # would be wrong rather than merely weak.
    var wa = 0.0
    for k in range(NACT):
        var e = abs(Float64(env.act[k]) - Float64(py=dat.act[k]))
        if e > wa:
            wa = e
    print("  |d(act)| =", wa, " (must be 0: ctrl = 0 throughout)")
    assert_true(
        wa < 1e-14,
        "the activations diverged under zero ctrl — this run is actuated and"
        " every number above is confounded",
    )

    assert_true(
        w1 <= TOL,
        "ONE env step diverges while the staged probe's single substep is"
        " exact at 2.99e-11 — the defect is in `Phyics3dEnv` or in the substep"
        " loop, NOT in the physics the probe measures. That is defect 8's"
        " shape: a gate that bypasses the production path.",
    )
    assert_true(
        w_hist[len(w_hist) - 1] <= TOL,
        "the env matches for one step then drifts — accumulation across"
        " substeps rather than a wrapper defect",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
