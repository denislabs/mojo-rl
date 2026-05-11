"""EfficientZero V2 (continuous) — Pendulum training with N parallel envs.

Mirrors `examples/acrobot/acrobot_ezv2_gpu_multienv.mojo` (which solved the
single-env stuck-mean-490 problem on Acrobot Phase A) for the continuous
agent. The single-env Pendulum run on 2026-05-09 plateaued at mean10 ≈
-1100 even after the train_step_core layout fix landed; per the Acrobot
memory note, a 4× buffer-diversity boost from N_ENVS=4 closed most of the
gap on a similarly sparse-reward env. Same hypothesis here.

Differences from `ezv2_pendulum_training.mojo` (single-env baseline):
    - N_ENVS = 4 with per-env episode tracking + per-env reset
    - temperature_decay_steps very high (keeps T=1.0 throughout — exploration
      stays on; the single-env run hit T=0 by ep 75 / 15k env steps which
      is well before the value head had anything to exploit)
    - START_TRANSITIONS = 2000 random-action warmup (pre-populates buffer
      with diverse exploration before MCTS-driven acting begins)
    - BS=128 (matches Acrobot multi-env), N_TD=5

Run:
    pixi run -e apple mojo run -I . examples/pendulum/ezv2_pendulum_training_multienv.mojo
"""

from std.memory import UnsafePointer, alloc
from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
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


def _mean(xs: List[Float64]) -> Float64:
    if len(xs) == 0:
        return Float64(0.0)
    var s = Float64(0.0)
    for x in xs:
        s += x
    return s / Float64(len(xs))


def main() raises:
    print("=" * 72)
    print("    EZ-V2 Pendulum — multi-env (Phase A) continuous training")
    print("=" * 72)

    # Total env transitions across ALL envs combined (apples-to-apples vs
    # the single-env run's NUM_ENV_STEPS).
    comptime NUM_ENV_STEPS = 30_000
    comptime N_ENVS = 4
    comptime MAX_STEPS_PER_EPISODE = 200

    # Train every BATCH_TRAIN_INTERVAL env-step batches. With N_ENVS=4 and
    # BATCH_TRAIN_INTERVAL=1, train every 4 transitions — same training-
    # frequency-per-transition as single-env's TRAIN_INTERVAL=4.
    comptime BATCH_TRAIN_INTERVAL = 1
    comptime START_TRANSITIONS = 2_000
    comptime SYNC_INTERVAL = 50  # train_steps between GPU→CPU weight sync
    comptime TARGET_SYNC_INTERVAL = 200
    comptime LOG_EVERY_BATCHES = NUM_ENV_STEPS // (N_ENVS * 50)
    comptime EPISODE_REWARD_TARGET = Float64(-200.0)

    # Continuous EZ-V2 config: same network shapes as the single-env
    # baseline, larger BS to absorb the N_ENVS×4 diversity. Visit
    # temperature is kept ≈ constant by setting decay_steps very high
    # (the policy improves through MCTS visits, not by aggressive
    # temperature annealing).
    comptime Config = EZV2ContinuousMLPConfig[
        OBS=3,
        ACT_DIM=1,
        LATENT=64,
        HIDDEN=64,
        PROJ=128,
        PRED_BOTTLENECK=64,
        BINS=51,
        BS=128,
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

    seed(2026)
    var agent = GenericEZV2ContinuousAgent[Config](
        gamma=0.99,
        v_min=-20.0,
        v_max=2.0,
        temperature=1.0,
        temperature_decay_steps=10_000_000,
        max_grad_norm=5.0,
        n_envs=N_ENVS,
    )
    var ctx = DeviceContext()
    var gpu = EZV2GPUStateBase[Config](ctx)
    gpu.upload_from(agent.state, ctx)
    ctx.synchronize()

    # N parallel envs. PendulumEnv is small + Copyable-friendly via
    # default-init, but we follow the Acrobot template (heap-allocated
    # pointers in a List) for symmetry — keeps the per-env state
    # bookkeeping consistent.
    var envs = List[UnsafePointer[PendulumEnv[dtype], MutAnyOrigin]]()
    var obs_list = List[List[Scalar[dtype]]]()
    var ep_returns_per_env = List[Float64]()
    var ep_steps_per_env = List[Int]()
    for _ in range(N_ENVS):
        var p = alloc[PendulumEnv[dtype]](1)
        p.init_pointee_move(PendulumEnv[dtype]())
        envs.append(p)
        ep_returns_per_env.append(Float64(0.0))
        ep_steps_per_env.append(0)
    for env_id in range(N_ENVS):
        obs_list.append(envs[env_id][].reset_obs_list())

    print()
    print("--- Run config ---")
    print("    N_ENVS                =", N_ENVS)
    print("    NUM_ENV_STEPS         =", NUM_ENV_STEPS, "(across all envs)")
    print("    BATCH_TRAIN_INTERVAL  =", BATCH_TRAIN_INTERVAL)
    print("    START_TRANSITIONS     =", START_TRANSITIONS, "(random warmup)")
    print("    SYNC_INTERVAL         =", SYNC_INTERVAL, "train_steps")
    print("    TARGET_SYNC_INTERVAL  =", TARGET_SYNC_INTERVAL)
    print(
        "    BS=", Config.batch_size,
        " K_UNROLL=", Config.unroll_steps,
        " N_TD=", Config.td_steps,
    )
    print(
        "    SIMS=", Config.num_simulations,
        " K_ROOT=", Config.num_root_candidates,
    )
    print("    MAX_ACTION=", 2.0, " MIN_STD=", 0.1)
    print(
        "    γ=", agent.gamma,
        " v_min=", agent.v_min, " v_max=", agent.v_max,
        " T_decay_steps=", agent.temperature_decay_steps,
    )
    print()

    # ── Training loop ─────────────────────────────────────────────────────
    var ep_returns = List[Float64]()
    var num_train_calls = 0
    var num_gpu_syncs = 0
    var any_nan_loss = False
    var last_L_R = Float64(0.0)
    var last_L_P = Float64(0.0)
    var last_L_V = Float64(0.0)
    var last_L_G = Float64(0.0)
    var best_recent_mean = Float64(-1e9)
    var total_transitions = 0
    var num_batches = NUM_ENV_STEPS // N_ENVS

    var t0 = perf_counter_ns()

    for batch in range(num_batches):
        for env_id in range(N_ENVS):
            var action_vec = List[Scalar[dtype]](capacity=Config.action_dim)
            var root_value = Float64(0.0)
            if total_transitions < START_TRANSITIONS:
                # Uniform-random action in [-MAX_ACTION, +MAX_ACTION].
                for _ in range(Config.action_dim):
                    action_vec.append(
                        Scalar[dtype](
                            random_float64(-1.0, 1.0) * 2.0
                        )
                    )
            else:
                var sel = agent.select_action(
                    obs_list[env_id], training=True
                )
                action_vec = sel[0].copy()
                root_value = sel[1]

            var step_result = envs[env_id][].step_continuous_vec(action_vec)
            var next_obs = step_result[0].copy()
            var reward = Float64(step_result[1])
            var done = step_result[2]

            ep_steps_per_env[env_id] += 1
            var truncated = ep_steps_per_env[env_id] >= MAX_STEPS_PER_EPISODE
            var done_or_trunc = done or truncated

            agent.store_transition(
                obs_list[env_id],
                action_vec,
                reward,
                root_value,
                done_or_trunc,
                env_id=env_id,
            )
            ep_returns_per_env[env_id] += reward
            total_transitions += 1

            if done_or_trunc:
                ep_returns.append(ep_returns_per_env[env_id])
                ep_returns_per_env[env_id] = Float64(0.0)
                ep_steps_per_env[env_id] = 0
                obs_list[env_id] = envs[env_id][].reset_obs_list()
            else:
                obs_list[env_id] = next_obs^

        if (
            agent.state.is_ready()
            and total_transitions >= START_TRANSITIONS
            and (batch + 1) % BATCH_TRAIN_INTERVAL == 0
        ):
            var t = agent.train_step_gpu(gpu, ctx)
            num_train_calls += 1
            last_L_R = t[1]
            last_L_P = t[2]
            last_L_V = t[3]
            last_L_G = t[4]
            if not _is_finite(t[0]):
                any_nan_loss = True

            if num_train_calls % SYNC_INTERVAL == 0:
                gpu.download_to(agent.state, ctx)
                ctx.synchronize()
                num_gpu_syncs += 1

            if num_train_calls % TARGET_SYNC_INTERVAL == 0:
                agent.update_target_networks(tau=1.0)

        if (batch + 1) % LOG_EVERY_BATCHES == 0:
            var t_now = perf_counter_ns()
            var wall_s = Float64(t_now - t0) / 1.0e9
            var n_eps = len(ep_returns)
            var window = 10 if n_eps > 10 else n_eps
            var recent = List[Float64]()
            var start = (n_eps - window) if n_eps > window else 0
            for i in range(start, n_eps):
                recent.append(ep_returns[i])
            var rmean = _mean(recent)
            if rmean > best_recent_mean and len(recent) >= window:
                best_recent_mean = rmean
            print(
                "[batch ", batch + 1,
                " step ", total_transitions,
                " eps=", n_eps,
                " train=", num_train_calls,
                " wall=", wall_s, "s",
                "] mean10=", rmean,
                "  best10=", best_recent_mean,
                "  L=(", last_L_R, " ", last_L_P, " ",
                last_L_V, " ", last_L_G, ")",
            )

    print()
    print("=" * 72)
    print("    Multi-env training complete")
    print("=" * 72)
    print("    total_transitions :", total_transitions)
    print("    episodes_finished :", len(ep_returns))
    print("    train_calls       :", num_train_calls)
    print("    gpu_syncs         :", num_gpu_syncs)
    print("    best mean10       :", best_recent_mean)
    print("    target ≥ -200     :", best_recent_mean >= EPISODE_REWARD_TARGET)
    print("    any_nan_loss      :", any_nan_loss)

    for env_id in range(N_ENVS):
        envs[env_id].destroy_pointee()
        envs[env_id].free()
