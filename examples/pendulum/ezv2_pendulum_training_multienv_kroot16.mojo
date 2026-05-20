"""EZ-V2 Pendulum multi-env — experiment: K_ROOT = 16 (vs default 8).

Same as `ezv2_pendulum_training_multienv.mojo` but doubles the number of
root candidates sampled at the start of each MCTS search. Tests the
hypothesis that with only 8 candidates around the (poorly-trained)
prior μ, MCTS can't differentiate good actions and `a*` is dominated by
prior noise — so the squashed-Gaussian NLL toward `a*` provides no real
learning signal.

K_NON_ROOT also doubled to 8 to keep the per-non-root expansion ratio
the same. Compare against the K_ROOT=8 baseline in
`logs/ezv2_pendulum_multienv_v1.log`.
"""

from std.memory import UnsafePointer, alloc
from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2ContinuousMLPShallowConfig,
    EZV2GPUStateBase,
    GenericEZV2ContinuousAgent,
    VALUE_TARGET_SARSA,
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
    print("    EZ-V2 Pendulum — multi-env, K_ROOT=16 experiment")
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

    comptime Config = EZV2ContinuousMLPShallowConfig[
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
        K_ROOT=16,  # ← experiment knob (was 8)
        K_NON_ROOT=8,  # ← also doubled
        MAX_ACTION=2.0,
        MIN_STD=0.5,  # ← exploration fix (was 0.1)
        STD_MAGNIFICATION=3.0,
        ENT_WEIGHT=0.05,  # ← exploration fix (was 0.005, 10×)
        # Use bootstrapped TD value targets (reference's choice for DMC
        # state envs). SVE-based SEARCH targets caused value-head
        # overestimation feedback — see
        # `docs/EZV2_CONTINUOUS_PHASE3_POSTMORTEM.md`.
        VALUE_TARGET_MODE=VALUE_TARGET_SARSA,
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
    # Phase 3: mirror CPU target nets onto GPU for MIXED/SARSA boot-V.
    gpu.upload_targets_from(agent.state, ctx)
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
    print(
        "    K_ROOT                =",
        Config.num_root_candidates,
        "(experiment)",
    )
    print(
        "    BS=",
        Config.batch_size,
        " K_UNROLL=",
        Config.unroll_steps,
        " N_TD=",
        Config.td_steps,
    )
    print(
        "    SIMS=",
        Config.num_simulations,
        " ENT_WEIGHT=",
        Config.entropy_weight,
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
    var best_recent_mean = Float64(-1e9)
    var total_transitions = 0
    var num_batches = NUM_ENV_STEPS // N_ENVS

    var t0 = perf_counter_ns()

    for batch in range(num_batches):
        for env_id in range(N_ENVS):
            var action_vec = List[Scalar[dtype]](capacity=Config.action_dim)
            var root_value = Float64(0.0)
            comptime K_ROOT_C = Config.num_root_candidates
            var sampled_actions_vec = List[Scalar[dtype]](
                capacity=K_ROOT_C * Config.action_dim
            )
            var improved_policy_vec = List[Scalar[dtype]](capacity=K_ROOT_C)
            if total_transitions < START_TRANSITIONS:
                for _ in range(Config.action_dim):
                    action_vec.append(
                        Scalar[dtype](random_float64(-1.0, 1.0) * 2.0)
                    )
                # Random-action warmup: one-hot improved policy on the
                # chosen action so the full-π kernel reduces to simple-
                # best NLL on it (zero candidate variance is fine for
                # warmup — value head is the dominant signal there).
                for _ in range(K_ROOT_C * Config.action_dim):
                    sampled_actions_vec.append(Scalar[dtype](0.0))
                for _ in range(K_ROOT_C):
                    improved_policy_vec.append(Scalar[dtype](0.0))
                for d in range(Config.action_dim):
                    sampled_actions_vec[d] = action_vec[d]
                improved_policy_vec[0] = Scalar[dtype](1.0)
            else:
                var sel = agent.select_action(obs_list[env_id], training=True)
                action_vec = sel[0].copy()
                root_value = sel[1]
                sampled_actions_vec = sel[2].copy()
                improved_policy_vec = sel[3].copy()
                # Diagnostic probe: dump root stats every ~3000 batches
                # for env_id=0 so we can see whether MCTS Q-values
                # differentiate candidates or stay uniform.
                if env_id == 0 and (batch + 1) % 3000 == 0:
                    agent.inspect_root(tag=String("batch=") + String(batch + 1))

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
                sampled_actions_vec,
                improved_policy_vec,
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
                # Phase 3: mirror fresh CPU targets onto GPU.
                gpu.upload_targets_from(agent.state, ctx)

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
            if len(recent) >= 10 and rmean > best_recent_mean:
                best_recent_mean = rmean
            var best_display = (
                best_recent_mean if best_recent_mean > -1e8 else 0.0
            )
            print(
                "[batch ",
                batch + 1,
                " step ",
                total_transitions,
                " eps=",
                n_eps,
                " train=",
                num_train_calls,
                " wall=",
                wall_s,
                "s",
                "] mean10=",
                rmean,
                "  best10=",
                best_display,
                "  L=(",
                last_L_R,
                " ",
                last_L_P,
                " ",
                last_L_V,
                " ",
                last_L_G,
                ")",
            )

    print()
    print("=" * 72)
    print("    Training complete (K_ROOT=16)")
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
