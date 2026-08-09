"""Pong CPU-vs-GPU physics parity — does the GPU env step match the CPU env?

The batched GPU training path feeds the agent observations/rewards from the
GPU Pong kernel (`step_kernel_gpu`); the converging CPU path uses the CPU
`_step_impl`. These are two separate physics implementations. If they diverge,
the GPU agent trains on subtly-wrong dynamics → could explain the uniform
collapse.

Method: reset a CPU env, upload its exact 12-D state to a 1-env GPU buffer
(bypassing the GPU reset so both start identical), then step both with the
same action sequence. Pong physics is deterministic EXCEPT on scoring (ball
reset uses RNG — CPU random() vs GPU Philox), so we compare until the first
score/done. A divergence BEFORE any score = GPU physics bug.

Run:
    pixi run -e apple mojo run -I . tests/arcade_games/test_pong_cpu_gpu_parity.mojo
"""

from mojo_rl.envs.arcade_games.pong import PongEnv
from max.gpu.host import DeviceContext
from std.testing import assert_true
from mojo_rl.core.fmt import fit

comptime dt = DType.float32
comptime SS = PongEnv[DType.float64].STATE_SIZE  # 12
comptime OD = PongEnv[DType.float64].OBS_DIM  # 6
comptime CpuPong = PongEnv[dt, 0.0]  # HIT_REWARD=0 (matches training)
comptime GpuPong = PongEnv[DType.float64, 0.0]  # static GPU kernels, HIT_REWARD=0


def _absf(x: Scalar[dt]) -> Scalar[dt]:
    return x if x >= 0 else -x


def main() raises:
    print("=== Pong CPU vs GPU physics parity ===")
    var ctx = DeviceContext()

    var cenv = CpuPong()
    _ = cenv.reset()

    var states = ctx.enqueue_create_buffer[dt](SS)
    var actions = ctx.enqueue_create_buffer[dt](1)
    var rewards = ctx.enqueue_create_buffer[dt](1)
    var dones = ctx.enqueue_create_buffer[dt](1)
    var terminated = ctx.enqueue_create_buffer[dt](1)
    var obs = ctx.enqueue_create_buffer[dt](OD)
    var hstate = ctx.enqueue_create_host_buffer[dt](SS)
    var hobs = ctx.enqueue_create_host_buffer[dt](OD)
    var hrew = ctx.enqueue_create_host_buffer[dt](1)
    var hdone = ctx.enqueue_create_host_buffer[dt](1)
    var hact = ctx.enqueue_create_host_buffer[dt](1)

    # Seed the GPU env from the CPU env's exact reset state.
    for i in range(SS):
        hstate[i] = cenv.state[i]
    ctx.enqueue_copy(states, hstate)
    ctx.synchronize()

    print("initial state (both):")
    for i in range(SS):
        print("  [", i, "]", cenv.state[i])
    print()
    print("step | act | cpu_rew | gpu_rew | max_obs_diff | state_diff")
    print("-" * 64)

    var max_obs_diff = Scalar[dt](0.0)
    var first_diverge = -1
    var compared = 0
    for step in range(200):
        # Cycle NOOP / UP / DOWN to exercise paddle physics too.
        var action = (step // 7) % 3
        hact[0] = Scalar[dt](action)
        ctx.enqueue_copy(actions, hact)
        GpuPong.step_kernel_gpu[1, SS, OD](
            ctx, states, actions, rewards, dones, terminated, obs,
            rng_seed=UInt64(step),
        )
        ctx.synchronize()
        ctx.enqueue_copy(hobs, obs)
        ctx.enqueue_copy(hrew, rewards)
        ctx.enqueue_copy(hdone, dones)
        ctx.enqueue_copy(hstate, states)
        ctx.synchronize()

        var cres = cenv.step_obs(action)
        var cobs = cres[0].copy()
        var crew = cres[1]
        var cdone = cres[2]

        var od = Scalar[dt](0.0)
        for d in range(OD):
            var diff = _absf(cobs[d] - hobs[d])
            if diff > od:
                od = diff
        if od > max_obs_diff:
            max_obs_diff = od

        # State-space diff (12-D) for a tighter check.
        var sd = Scalar[dt](0.0)
        for i in range(SS):
            var diff = _absf(cenv.state[i] - hstate[i])
            if diff > sd:
                sd = diff

        compared += 1
        var scored = (
            crew != Scalar[dt](0.0)
            or hrew[0] != Scalar[dt](0.0)
            or cdone
            or hdone[0] > Scalar[dt](0.5)
        )
        # A score resets the ball with independent RNG (CPU random vs GPU
        # Philox) → obs legitimately diverges AFTER it. Stop comparing there;
        # only a PRE-score obs mismatch indicates a physics bug.
        if scored:
            print("--- first score/done at step", step, "(RNG diverges after) ---")
            break
        if step < 30 or od > Scalar[dt](1e-3):
            print(
                fit(String(step), 4), " | ", action, " | ",
                fit(String(crew), 7), " | ", fit(String(hrew[0]), 7),
                " | ", fit(String(od), 11), " | ", fit(String(sd), 11),
            )
        if od > Scalar[dt](1e-3) and first_diverge < 0:
            first_diverge = step

    print("-" * 64)
    print("compared steps:", compared, " max_obs_diff(pre-score):", max_obs_diff)
    if first_diverge >= 0:
        print(">>> PHYSICS DIVERGES at step", first_diverge, "BEFORE any score.")
        print(">>> The GPU Pong env does NOT match CPU → env physics bug.")
    else:
        print(">>> CPU and GPU physics MATCH up to first score → env physics OK.")
    assert_true(
        first_diverge < 0,
        "GPU Pong physics diverged from CPU before any score",
    )
    print("ALL PASSED")
