"""EfficientZero V2 (continuous) — Pendulum DIAGNOSTIC run.

Same as `ezv2_pendulum_training.mojo` but, every DUMP_EVERY train calls,
downloads `gpu.value_target_full_buf` (BATCH * (K+1) raw scalar value
targets, the input to two-hot encoding) and `gpu.batch_mcts_val_buf`
(BATCH * (K+1) raw stored MCTS root values fetched from the buffer at
sampling time) and prints summary stats + a coarse histogram.

Goal: diagnose why L_V was flat at log(BINS) ≈ 3.93 nats across the full
30k-step run on 2026-05-09 (continuous EZ-V2 first convergence attempt).
That signature points to value targets being effectively uniform — this
diagnostic distinguishes:

    (a) targets are uniform garbage      → both buffers have wide spread
                                            with mean≈0 and dist looks
                                            like noise
    (b) targets are stuck at a constant  → very narrow std, e.g. all -20
        (e.g. v_min clipping)              → mass piles up at a single
                                            histogram bin
    (c) targets are correct but the      → reasonable spread (e.g. mean
        network can't fit                  ≈ -1100 for Pendulum), but
                                            L_V never drops anyway

Run:
    pixi run -e apple mojo run -I . examples/pendulum/ezv2_pendulum_diagnostic.mojo
"""

from std.random import seed
from std.gpu.host import DeviceContext
from std.math import sqrt
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2ContinuousMLPConfig,
    EZV2GPUStateBase,
    GenericEZV2ContinuousAgent,
)
from mojo_rl.envs.pendulum import PendulumEnv
from mojo_rl.nn.constants import dtype


def _is_finite(x: Float64) -> Bool:
    if x != x:
        return False
    if x > 1.0e300 or x < -1.0e300:
        return False
    return True


# Compute simple stats + 8-bucket histogram over [lo, hi].
def _dump_stats(
    label: String,
    host_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    lo: Float64,
    hi: Float64,
):
    var mn = Float64(1e30)
    var mx = Float64(-1e30)
    var s = Float64(0.0)
    var s2 = Float64(0.0)
    var n_finite = 0
    for i in range(n):
        var v = Float64(host_ptr[i])
        if v != v:
            continue
        if v > 1e300 or v < -1e300:
            continue
        if v < mn:
            mn = v
        if v > mx:
            mx = v
        s += v
        s2 += v * v
        n_finite += 1
    var mean = s / Float64(n_finite) if n_finite > 0 else 0.0
    var var_ = s2 / Float64(n_finite) - mean * mean if n_finite > 0 else 0.0
    if var_ < 0.0:
        var_ = 0.0
    var std = sqrt(var_)

    # 8-bucket histogram.
    var hist = InlineArray[Int, 8](fill=0)
    var below = 0
    var above = 0
    var width = (hi - lo) / 8.0
    if width <= 0.0:
        width = 1.0
    for i in range(n):
        var v = Float64(host_ptr[i])
        if v != v:
            continue
        if v > 1e300 or v < -1e300:
            continue
        if v < lo:
            below += 1
            continue
        if v >= hi:
            above += 1
            continue
        var b = Int((v - lo) / width)
        if b < 0:
            b = 0
        if b >= 8:
            b = 7
        hist[b] += 1

    print(
        "    [", label, "]",
        " n=", n_finite, "/", n,
        " min=", mn, " max=", mx,
        " mean=", mean, " std=", std,
    )
    print(
        "      hist[", lo, "..", hi, "] below=", below,
        " bins=[", hist[0], hist[1], hist[2], hist[3],
        hist[4], hist[5], hist[6], hist[7], "] above=", above,
    )


def main() raises:
    print("=" * 64)
    print("    EZ-V2 Pendulum — DIAGNOSTIC run (value-target inspection)")
    print("=" * 64)

    comptime Config = EZV2ContinuousMLPConfig[
        OBS=3,
        ACT_DIM=1,
        LATENT=64,
        HIDDEN=64,
        PROJ=128,
        PRED_BOTTLENECK=64,
        BINS=51,
        BS=64,
        K_UNROLL=5,
        N_TD=5,
        SIMS=32,
        NODES=128,
        K_ROOT=8,
        K_NON_ROOT=4,
        MAX_ACTION=2.0,
        MIN_STD=0.1,
        STD_MAGNIFICATION=3.0,
    ]

    comptime NUM_ENV_STEPS = 30_000
    comptime MAX_STEPS_PER_EPISODE = 200
    comptime TRAIN_INTERVAL = 4
    comptime TARGET_SYNC_INTERVAL = 200
    comptime LOG_EVERY_EPISODES = 5
    comptime DUMP_EVERY = 500  # train calls between buffer dumps
    comptime BUF_LEN = Config.batch_size * (Config.unroll_steps + 1)

    seed(2026)
    var env = PendulumEnv[dtype]()
    var agent = GenericEZV2ContinuousAgent[Config](
        gamma=0.99,
        v_min=-20.0,
        v_max=2.0,
        temperature=1.0,
        temperature_decay_steps=NUM_ENV_STEPS // 2,
        max_grad_norm=5.0,
    )
    var ctx = DeviceContext()
    var gpu = EZV2GPUStateBase[Config](ctx)
    gpu.upload_from(agent.state, ctx)
    ctx.synchronize()

    var env_step = 0
    var episode = 0
    var train_calls = 0
    var ep_returns = List[Float64]()
    var any_nan_loss = False
    var last_L_R = Float64(0.0)
    var last_L_P = Float64(0.0)
    var last_L_V = Float64(0.0)
    var last_L_G = Float64(0.0)
    var best_recent_mean = Float64(-1e9)

    print()
    print("Starting DIAGNOSTIC training...")
    print("    NUM_ENV_STEPS:", NUM_ENV_STEPS)
    print("    BS:", Config.batch_size, " K_UNROLL:", Config.unroll_steps)
    print("    BUF_LEN (BS*(K+1)):", BUF_LEN)
    print("    DUMP_EVERY (train calls):", DUMP_EVERY)
    print()

    while env_step < NUM_ENV_STEPS:
        var obs = env.reset_obs_list()
        var ep_reward = Float64(0.0)
        for _step in range(MAX_STEPS_PER_EPISODE):
            if env_step >= NUM_ENV_STEPS:
                break
            var sel = agent.select_action(obs, training=True)
            var action = sel[0].copy()
            var root_value = sel[1]

            var step_result = env.step_continuous_vec(action)
            var next_obs = step_result[0].copy()
            var reward = Float64(step_result[1])
            var done = step_result[2]
            ep_reward += reward
            agent.store_transition(obs, action, reward, root_value, done)
            env_step += 1
            obs = next_obs^

            if env_step % TRAIN_INTERVAL == 0 and agent.state.is_ready():
                var t = agent.train_step_gpu(gpu, ctx)
                last_L_R = t[1]
                last_L_P = t[2]
                last_L_V = t[3]
                last_L_G = t[4]
                if not _is_finite(t[0]):
                    any_nan_loss = True
                train_calls += 1

                # ── Diagnostic dump ─────────────────────────────────────
                # Right after train_step_gpu returns, the value_target
                # buffers still hold the targets used for the last loss
                # computation. Download both and print stats.
                if train_calls % DUMP_EVERY == 0:
                    print()
                    print("=== train_call", train_calls, " env_step", env_step, " ===")
                    print(
                        "    last L_R=", last_L_R,
                        " L_P=", last_L_P,
                        " L_V=", last_L_V,
                        " L_G=", last_L_G,
                    )
                    ctx.enqueue_copy(
                        gpu.value_target_full_host, gpu.value_target_full_buf
                    )
                    ctx.enqueue_copy(
                        gpu.batch_mcts_val_host, gpu.batch_mcts_val_buf
                    )
                    ctx.synchronize()
                    _dump_stats(
                        "value_target_full_buf",
                        gpu.value_target_full_host.unsafe_ptr(),
                        BUF_LEN,
                        -20.0,
                        2.0,
                    )
                    _dump_stats(
                        "batch_mcts_val_buf",
                        gpu.batch_mcts_val_host.unsafe_ptr(),
                        BUF_LEN,
                        -20.0,
                        2.0,
                    )
                    print()

                if train_calls % TARGET_SYNC_INTERVAL == 0:
                    agent.update_target_networks(tau=1.0)
                gpu.download_to(agent.state, ctx)
                ctx.synchronize()

            if done:
                break

        agent.decay_temperature()
        ep_returns.append(ep_reward)
        episode += 1

        if episode % LOG_EVERY_EPISODES == 0 or env_step >= NUM_ENV_STEPS:
            var window = 10 if len(ep_returns) > 10 else len(ep_returns)
            var sum_r = Float64(0.0)
            for k in range(window):
                sum_r += ep_returns[len(ep_returns) - 1 - k]
            var recent_mean = sum_r / Float64(window)
            if recent_mean > best_recent_mean:
                best_recent_mean = recent_mean
            print(
                "[ep ", episode, "][step ", env_step, "]",
                " ep_reward=", ep_reward,
                " mean10=", recent_mean,
                " best10=", best_recent_mean,
                " L=(", last_L_R, " ", last_L_P, " ", last_L_V, " ", last_L_G, ")",
                " T=", agent.temperature,
            )

    print()
    print("=" * 64)
    print("    DIAGNOSTIC training complete")
    print("=" * 64)
    print("    episodes      :", episode)
    print("    env_steps     :", env_step)
    print("    train_calls   :", train_calls)
    print("    best mean10   :", best_recent_mean)
    print("    any_nan_loss  :", any_nan_loss)
