"""EZ-V2 Pendulum multi-env — experiment: ENT_WEIGHT = 0.

Same as `ezv2_pendulum_training_multienv.mojo` but with the entropy
bonus on the squashed-Gaussian policy disabled. Tests the hypothesis
that the policy was collapsing to μ ≈ 0 (pendulum dangling) because
the entropy bonus dominates the NLL term once `a* ≈ μ` from MCTS, and
forces σ → MIN_STD = 0.1.

Compare directly with `logs/ezv2_pendulum_multienv_v1.log` (default
ENT_WEIGHT = 5e-3, identical otherwise).
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
    print("    EZ-V2 Pendulum — multi-env, ENT_WEIGHT=0 experiment")
    print("=" * 72)

    comptime NUM_ENV_STEPS = 30_000
    comptime N_ENVS = 4
    comptime MAX_STEPS_PER_EPISODE = 200
    comptime BATCH_TRAIN_INTERVAL = 1
    comptime START_TRANSITIONS = 2_000
    comptime SYNC_INTERVAL = 50
    comptime TARGET_SYNC_INTERVAL = 200
    comptime LOG_EVERY_BATCHES = NUM_ENV_STEPS // (N_ENVS * 50)
    comptime EPISODE_REWARD_TARGET = Float64(-200.0)

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
        ENT_WEIGHT=0.0,            # ← experiment knob
        MAX_ACTION=2.0,
        MIN_STD=0.1,
        STD_MAGNIFICATION=3.0,
    ]

    seed(2026)
    var agent = GenericEZV2ContinuousAgent[Config](
        gamma=0.99,
        v_min=-50.0,
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
    print("    ENT_WEIGHT            =", Config.entropy_weight, "(disabled)")
    print(
        "    BS=", Config.batch_size,
        " K_UNROLL=", Config.unroll_steps,
        " N_TD=", Config.td_steps,
    )
    print(
        "    SIMS=", Config.num_simulations,
        " K_ROOT=", Config.num_root_candidates,
    )
    print()

    var ep_returns = List[Float64]()
    var num_train_calls = 0
    var num_gpu_syncs = 0
    var any_nan_loss = False
    var last_L_R = Float64(0.0)
    var last_L_P = Float64(0.0)
    var last_L_V = Float64(0.0)
    var last_L_G = Float64(0.0)
    # FIX: only update best_recent_mean when window is filled (otherwise
    # the empty-recent _mean() returns 0.0 and clobbers the -1e9 init).
    var best_recent_mean = Float64(-1e9)
    var total_transitions = 0
    var num_batches = NUM_ENV_STEPS // N_ENVS

    var t0 = perf_counter_ns()

    for batch in range(num_batches):
        for env_id in range(N_ENVS):
            var action_vec = List[Scalar[dtype]](capacity=Config.action_dim)
            var root_value = Float64(0.0)
            if total_transitions < START_TRANSITIONS:
                for _ in range(Config.action_dim):
                    action_vec.append(
                        Scalar[dtype](random_float64(-1.0, 1.0) * 2.0)
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
            # Only update best once the window is full — otherwise the
            # rmean=0 of an empty list overwrites the -1e9 sentinel.
            if len(recent) >= 10 and rmean > best_recent_mean:
                best_recent_mean = rmean
            var best_display = best_recent_mean if best_recent_mean > -1e8 else 0.0
            print(
                "[batch ", batch + 1,
                " step ", total_transitions,
                " eps=", n_eps,
                " train=", num_train_calls,
                " wall=", wall_s, "s",
                "] mean10=", rmean,
                "  best10=", best_display,
                "  L=(", last_L_R, " ", last_L_P, " ",
                last_L_V, " ", last_L_G, ")",
            )

    print()
    print("=" * 72)
    print("    Training complete (ENT_WEIGHT=0)")
    print("=" * 72)
    print("    total_transitions :", total_transitions)
    print("    episodes_finished :", len(ep_returns))
    print("    train_calls       :", num_train_calls)
    print("    gpu_syncs         :", num_gpu_syncs)
    print(
        "    best mean10       :",
        best_recent_mean if best_recent_mean > -1e8 else 0.0,
    )
    print(
        "    target ≥ -200     :",
        best_recent_mean >= EPISODE_REWARD_TARGET and best_recent_mean > -1e8,
    )
    print("    any_nan_loss      :", any_nan_loss)

    for env_id in range(N_ENVS):
        envs[env_id].destroy_pointee()
        envs[env_id].free()
