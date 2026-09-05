"""One physics step, timed, on the single-env CPU path.

What is measured is ONE INTEGRATOR STEP — `apply_actions` (MuJoCo recomputes
`qfrc_actuator` inside every `mj_step`, so it belongs in the step) followed by
`integ.step["cpu"]`. Nothing from the env layer: no observation, no reward, no
hook. That is the same quantity `mj_step` is, and the MuJoCo twin
(`benchmarks/physics3d_cpu_vs_mujoco.py`) drives the same XML through the
same protocol:

    reset to qpos0, qvel = 0, ctrl = 0.1 on every actuator
    WARMUP steps untimed
    a short counting loop for the work counters (contacts), NOT inside the
        timed region -- reading a counter per step is a cost the other side
        does not pay
    STEPS steps timed with one clock read on each side of the loop

⚠ THE INTEGRATOR IS A PARAMETER, AND IT MUST MATCH THE XML. MuJoCo takes it
from `<option integrator>` and defaults to Euler. Our env configs pick their
own (`So101ParkProbeConfig` inherits "rk4" for a scene whose XML says
nothing), so a step that trusted the config would run four RK4 stages against
MuJoCo's one. The caller says which; both integrators live on the env.

⚠ THE MODEL'S `NUM_CONTACTS` IS THE EQUIVALENCE CHECK. Two engines stepping
the same scene from the same state must see the same contact count on
average; a row whose counts disagree is two different problems, not one
engine faster than the other. The table prints both.

⚠ `Phyics3dEnv` is the production facade (`CRBA_TREEWALK=True`, `MAX_CONDIM`
and `NOSLIP_ITER` forwarded from the model). Building `Model`/`Data`/an
integrator by hand here would be measuring a configuration nothing ships.
"""

from std.time import perf_counter_ns

from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.physics3d.gpu.constants import META_IDX_NUM_CONTACTS
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.envs.phyics3d_env import Phyics3dEnv


comptime CTRL: Float64 = 0.1
comptime COUNT_STEPS = 2000


def fmt(v: Float64, places: Int = 3) -> String:
    var mul = 1.0
    for _ in range(places):
        mul *= 10.0
    var scaled = Int(v * mul + (0.5 if v >= 0 else -0.5))
    var whole = scaled // Int(mul)
    var frac = scaled % Int(mul)
    if frac < 0:
        frac = -frac
    var f = String(frac)
    while f.byte_length() < places:
        f = "0" + f
    return String(whole) + "." + f


@always_inline
def _one_step[
    MODEL: ModelDefLike, CONFIG: Phyics3dEnvConfig, DT: DType, EULER: Bool
](
    mut env: Phyics3dEnv[MODEL, CONFIG, DT, False],
    actions: List[Float64],
) raises:
    MODEL.apply_actions[NORMALIZED=False](env.sf, env.d, actions, env.act)
    comptime if EULER:
        env.integ_euler.step["cpu"](env.d, env.mf)
    else:
        env.integ_rk4.step["cpu"](env.d, env.mf)


def bench[
    MODEL: ModelDefLike, CONFIG: Phyics3dEnvConfig, DT: DType, EULER: Bool
](name: String, warmup: Int, steps: Int, rounds: Int = 1) raises:
    """`rounds` > 1 repeats the WHOLE protocol from qpos0 and reports the MIN.

    ⚠ A ROUND IS A RESET, NOT A CONTINUATION. The park scenes drop their props
    from z = 50 m and the first one lands at step 1596; a horizon past that
    measures a contact scene, and at k >= 6 it saturates `MAX_CONTACTS` on our
    side while MuJoCo keeps counting -- two different problems. Short rounds
    from the same state keep every round on the same problem, and enough of
    them give `sample` a process long enough to profile.
    """
    var env = Phyics3dEnv[MODEL, CONFIG, DT, False]()
    _ = env.reset()
    var actions = List[Float64]()
    for _ in range(MODEL.ACTION_DIM):
        actions.append(CTRL)

    var best_us = Float64(1e30)
    var ncon_sum = 0.0
    var n_count = min(steps, COUNT_STEPS)
    for _ in range(rounds):
        # The twin does `mj_resetData`: qpos0, zero velocity, zero warmstart.
        for i in range(MODEL.NQ):
            env.d.qpos.data[i] = env.sf.qpos0.data[i]
        for i in range(MODEL.NV):
            env.d.qvel.data[i] = Scalar[DT](0)
            env.d.qacc_warmstart.data[i] = Scalar[DT](0)

        for _ in range(warmup):
            _one_step[MODEL, CONFIG, DT, EULER](env, actions)

        ncon_sum = 0.0
        for _ in range(n_count):
            _one_step[MODEL, CONFIG, DT, EULER](env, actions)
            ncon_sum += Float64(env.d.meta.data[META_IDX_NUM_CONTACTS])

        var t0 = perf_counter_ns()
        for _ in range(steps):
            _one_step[MODEL, CONFIG, DT, EULER](env, actions)
        var t1 = perf_counter_ns()
        var us = Float64(t1 - t0) / 1000.0 / Float64(steps)
        if us < best_us:
            best_us = us
    var us = best_us

    # ⚠ A CHECKSUM OF THE FINAL STATE, so that a change meant to be bit-exact
    # can be checked as one across every model in one sweep: two builds that
    # print the same `qsum` walked the same trajectory to the last bit.
    var qsum = Float64(0)
    for i in range(MODEL.NQ):
        qsum += Float64(env.d.qpos.data[i]) * Float64(i + 1)
    var dtype_name = "f32" if DT == DType.float32 else "f64"
    var integ = "euler" if EULER else "rk4"
    print(
        "RESULT side=ours model=" + name + " dtype=" + dtype_name
        + " integ=" + integ
        + " nq=" + String(MODEL.NQ) + " nv=" + String(MODEL.NV)
        + " us_per_step=" + fmt(us, 4)
        + " ncon_mean=" + fmt(ncon_sum / Float64(n_count), 3)
        + " steps=" + String(steps) + " rounds=" + String(rounds)
        + " qpos0=" + String(Float64(env.d.qpos.data[0]))
        + " qpos1=" + String(Float64(env.d.qpos.data[1]))
        + " qsum=" + String(qsum)
    )
