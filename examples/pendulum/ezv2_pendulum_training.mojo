"""EfficientZero V2 (continuous) — Pendulum training example.

Drives `GenericEZV2ContinuousAgent` on the native Mojo Pendulum env
(3-D obs, 1-D continuous action in [-2.0, 2.0], reward
`-(θ² + 0.1·θ_dot² + 0.001·torque²)`).

Phase 3.4 success criterion (paper App. G + EFFICIENTZERO_V2_PLAN.md):
    mean episode return ≥ -200 over the last 10 episodes within 30k env
    steps.

This script is also useful as a sanity smoke at smaller `NUM_ENV_STEPS`
(e.g. 5_000) — it should at least not NaN and the value head should
trend down.

Run:
    pixi run mojo run -I . examples/pendulum/ezv2_pendulum_training.mojo
"""

from std.random import seed
from std.gpu.host import DeviceContext
from std.math import abs as fabs
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


def main() raises:
    print("=" * 64)
    print("    EfficientZero V2 — Pendulum continuous-action training")
    print("=" * 64)

    # ── Config ────────────────────────────────────────────────────────────
    # Pendulum: obs_dim=3, action_dim=1, torque ∈ [-2, 2], reward in
    # roughly [-16, 0] per step.
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
    comptime TRAIN_INTERVAL = 4  # train every N env steps
    comptime TARGET_SYNC_INTERVAL = 200  # train steps
    comptime LOG_EVERY_EPISODES = 5
    comptime EPISODE_REWARD_TARGET = Float64(-200.0)

    seed(2026)
    var env = PendulumEnv[dtype]()
    var agent = GenericEZV2ContinuousAgent[Config](
        gamma=0.99,
        v_min=-50.0,
        v_max=2.0,
        temperature=1.0,
        temperature_decay_steps=NUM_ENV_STEPS // 2,
        max_grad_norm=5.0,
    )
    var ctx = DeviceContext()
    var gpu = EZV2GPUStateBase[Config](ctx)
    gpu.upload_from(agent.state, ctx)
    ctx.synchronize()

    # ── Training loop ─────────────────────────────────────────────────────
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
    print("Starting training...")
    print("    NUM_ENV_STEPS:", NUM_ENV_STEPS)
    print("    BS:", Config.batch_size, " K_UNROLL:", Config.unroll_steps)
    print(
        "    SIMS:",
        Config.num_simulations,
        " K_ROOT:",
        Config.num_root_candidates,
    )
    print("    MAX_ACTION:", 2.0, " MIN_STD:", 0.1)
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

            # Train every TRAIN_INTERVAL env steps once buffer is ready.
            if env_step % TRAIN_INTERVAL == 0 and agent.state.is_ready():
                var t = agent.train_step_gpu(gpu, ctx)
                last_L_total = t[0]
                last_L_R = t[1]
                last_L_P = t[2]
                last_L_V = t[3]
                last_L_G = t[4]
                if not _is_finite(last_L_total):
                    any_nan_loss = True
                train_calls += 1
                # Polyak target sync at coarse interval.
                if train_calls % TARGET_SYNC_INTERVAL == 0:
                    agent.update_target_networks(tau=1.0)
                # Pull GPU weights back into CPU state so the host CPU
                # MCTS in `select_action` sees fresh networks at the
                # next env step.
                gpu.download_to(agent.state, ctx)
                ctx.synchronize()

            if done:
                break

        agent.decay_temperature()
        ep_returns.append(ep_reward)
        episode += 1

        # Periodic logging.
        if episode % LOG_EVERY_EPISODES == 0 or env_step >= NUM_ENV_STEPS:
            var window = 10 if len(ep_returns) > 10 else len(ep_returns)
            var sum_r = Float64(0.0)
            for k in range(window):
                sum_r += ep_returns[len(ep_returns) - 1 - k]
            var recent_mean = sum_r / Float64(window)
            if recent_mean > best_recent_mean:
                best_recent_mean = recent_mean
            print(
                "[ep ",
                episode,
                "][step ",
                env_step,
                "]",
                " ep_reward=",
                ep_reward,
                " mean10=",
                recent_mean,
                " best10=",
                best_recent_mean,
                " L=(",
                last_L_R,
                " ",
                last_L_P,
                " ",
                last_L_V,
                " ",
                last_L_G,
                ")",
                " T=",
                agent.temperature,
            )

    print()
    print("=" * 64)
    print("    Training complete")
    print("=" * 64)
    print("    episodes      :", episode)
    print("    env_steps     :", env_step)
    print("    train_calls   :", train_calls)
    print("    best mean10   :", best_recent_mean)
    print("    target ≥ -200 :", best_recent_mean >= EPISODE_REWARD_TARGET)
    print("    any_nan_loss  :", any_nan_loss)
