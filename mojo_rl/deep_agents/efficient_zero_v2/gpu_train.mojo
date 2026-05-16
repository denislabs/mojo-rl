"""Unified full-GPU training driver for EfficientZero V2 (discrete actions).

`run_ezv2_train_gpu[Env, Config, N_ENVS, NUM_ENV_STEPS](...)` packages
Steps 1-4 of `docs/EZV2_FULL_GPU_PLAN.md`:

  • GPU env step + selective reset (Step 1)
  • GPU Gumbel-search MCTS for action selection (Step 2)
  • Multi-env rollout (Step 3, N_ENVS in parallel)
  • GPU-resident replay buffer mirror (Step 4, bulk-uploaded at sync
    intervals — read-only for `train_step_gpu` which still samples
    from CPU; Step 5's GPU sampling kernels live in
    `gpu_sampling.mojo` for opt-in integration in a follow-up)

Returns an `EZV2TrainStats` struct with end-of-run aggregates.

The driver is generic over any `GPUDiscreteEnv`; CartPole is the
reference testbed.
"""

from std.math import abs, exp, log
from std.random import random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.core.env_traits import GPUDiscreteEnv
from mojo_rl.deep_agents.efficient_zero_v2.configs import EZV2DiscreteConfig
from mojo_rl.deep_agents.efficient_zero_v2.efficient_zero_v2 import (
    GenericEfficientZeroV2Agent,
)
from mojo_rl.deep_agents.efficient_zero_v2.state import EZV2GPUStateBase
from mojo_rl.deep_agents.efficient_zero_v2.gpu_mcts import EZV2GPUMCTSState
from mojo_rl.deep_agents.efficient_zero_v2.gpu_replay import (
    EZV2GPUReplayBuffer,
)
from mojo_rl.deep_agents.efficient_zero_v2.strategies import compute_sve
from mojo_rl.nn.constants import dtype


# ═════════════════════════════════════════════════════════════════════════
# Output stats
# ═════════════════════════════════════════════════════════════════════════


struct EZV2TrainStats(Movable):
    """End-of-run aggregates returned by `run_ezv2_train_gpu`."""

    var wall_time_s: Float64
    var total_env_steps: Int
    var num_train_calls: Int
    var num_gpu_syncs: Int
    var num_buffer_uploads: Int
    var num_episodes: Int
    var best_episode_return: Float64
    var any_nan_loss: Bool
    var ep_returns: List[Float64]

    var last_L_R: Float64
    var last_L_P: Float64
    var last_L_V: Float64
    var last_L_G: Float64

    # Diagnostics (added 2026-05-14 to track SimSiam-collapse / MCTS-no-
    # signal / unreachable-reward failure modes that hyperparam tuning
    # alone wasn't isolating on HalfCheetah). Updated by the driver:
    #   • `last_*` mirrors the most recent value (last train step / search
    #     / env step).
    #   • `*_sum` + `*_count` (or *_n for the windowed counter) carry the
    #     running aggregate within a log window so we can print the mean.
    var last_z_var: Float64
    var z_var_sum: Float64
    var z_var_n: Int

    var last_v_pred_var: Float64
    var v_pred_var_sum: Float64
    var v_pred_var_n: Int

    var last_mcts_visit_entropy: Float64
    var mcts_visit_entropy_sum: Float64
    var mcts_visit_entropy_n: Int

    # Running max of any per-step reward observed across all envs. NOT
    # reset per log window — the question this answers ("did the agent
    # ever visit a high-reward state?") is monotonic.
    var buf_reward_max: Float64

    def __init__(out self):
        self.wall_time_s = 0.0
        self.total_env_steps = 0
        self.num_train_calls = 0
        self.num_gpu_syncs = 0
        self.num_buffer_uploads = 0
        self.num_episodes = 0
        # Init very-negative so envs with non-positive rewards (e.g.
        # Acrobot at −1/step) update `best` on the first finished episode.
        self.best_episode_return = Float64(-1.0e308)
        self.any_nan_loss = False
        self.ep_returns = List[Float64]()
        self.last_L_R = 0.0
        self.last_L_P = 0.0
        self.last_L_V = 0.0
        self.last_L_G = 0.0
        self.last_z_var = 0.0
        self.z_var_sum = 0.0
        self.z_var_n = 0
        self.last_v_pred_var = 0.0
        self.v_pred_var_sum = 0.0
        self.v_pred_var_n = 0
        self.last_mcts_visit_entropy = 0.0
        self.mcts_visit_entropy_sum = 0.0
        self.mcts_visit_entropy_n = 0
        self.buf_reward_max = Float64(-1.0e308)

    def __init__(out self, *, deinit take: Self):
        self.wall_time_s = take.wall_time_s
        self.total_env_steps = take.total_env_steps
        self.num_train_calls = take.num_train_calls
        self.num_gpu_syncs = take.num_gpu_syncs
        self.num_buffer_uploads = take.num_buffer_uploads
        self.num_episodes = take.num_episodes
        self.best_episode_return = take.best_episode_return
        self.any_nan_loss = take.any_nan_loss
        self.ep_returns = take.ep_returns^
        self.last_L_R = take.last_L_R
        self.last_L_P = take.last_L_P
        self.last_L_V = take.last_L_V
        self.last_L_G = take.last_L_G
        self.last_z_var = take.last_z_var
        self.z_var_sum = take.z_var_sum
        self.z_var_n = take.z_var_n
        self.last_v_pred_var = take.last_v_pred_var
        self.v_pred_var_sum = take.v_pred_var_sum
        self.v_pred_var_n = take.v_pred_var_n
        self.last_mcts_visit_entropy = take.last_mcts_visit_entropy
        self.mcts_visit_entropy_sum = take.mcts_visit_entropy_sum
        self.mcts_visit_entropy_n = take.mcts_visit_entropy_n
        self.buf_reward_max = take.buf_reward_max


def _is_finite(x: Float64) -> Bool:
    if x != x:
        return False
    if x > 1.0e300 or x < -1.0e300:
        return False
    return True


def _mean(xs: List[Float64]) -> Float64:
    if len(xs) == 0:
        return 0.0
    var s = Float64(0.0)
    for i in range(len(xs)):
        s += xs[i]
    return s / Float64(len(xs))


# ═════════════════════════════════════════════════════════════════════════
# Driver
# ═════════════════════════════════════════════════════════════════════════


def run_ezv2_train_gpu[
    Env: GPUDiscreteEnv,
    Config: EZV2DiscreteConfig,
    N_ENVS: Int,
    NUM_ENV_STEPS: Int,
](
    mut agent: GenericEfficientZeroV2Agent[Config],
    ctx: DeviceContext,
    *,
    train_interval: Int = 4,
    sync_interval: Int = 50,
    target_sync_interval: Int = 200,
    reanalyze_interval: Int = 200,
    reanalyze_samples: Int = 32,
    reanalyze_warmup: Int = 1000,
    warmup_random_steps: Int = 0,
    log_every: Int = 2_000,
    rng_seed_base: UInt64 = UInt64(2026),
    use_gpu_sampling: Bool = False,
    verbose: Bool = True,
) raises -> EZV2TrainStats:
    """Drive EfficientZero V2 training fully on GPU (env + MCTS) with a
    CPU mirror for reanalyze + replay buffer ground truth.

    Parameters:
        Env: A `GPUDiscreteEnv` providing `step_kernel_gpu`,
            `reset_kernel_gpu`, `selective_reset_kernel_gpu`,
            `extract_obs_kernel_gpu`, and `init_step_workspace_gpu`.
        Config: An `EZV2DiscreteConfig` matching the agent's config.
        N_ENVS: Parallel envs (≥ 1). Phase A semantics: train every
            `train_interval` env-batches, not env-steps.
        NUM_ENV_STEPS: Total env-step transitions across all envs.

    Args:
        agent: The pre-constructed agent (must already have
            `n_envs == N_ENVS`).
        ctx: The GPU device context.
        train_interval: Train every Nth env-batch.
        sync_interval: GPU → CPU network + buffer sync every Nth train.
        target_sync_interval: Hard-copy target nets every Nth train.
        reanalyze_interval: Reanalyze every Nth train post-warmup.
        reanalyze_samples: How many windows per reanalyze call.
        reanalyze_warmup: Skip reanalyze until this many train calls.
        warmup_random_steps: Skip GPU MCTS for the first N env-step
            transitions and pick uniform-random actions instead. The
            train_step is also skipped during this window. Matches the
            paper's `start_transitions` for sparse-reward envs (e.g.
            Acrobot at 2000). Set to 0 to start MCTS from step 0.
        log_every: Print a log line every Nth env-step.
        rng_seed_base: Initial seed for env reset RNG.
        use_gpu_sampling: When `True`, the train step samples its batch
            via `ezv2_gpu_sample_and_gather` directly from the GPU replay
            mirror instead of the CPU host loop (legacy default). The
            `gpu_replay` mirror is uploaded each `sync_interval`
            train_steps as before, so the GPU sees up to that many
            train_steps of stale priorities/transitions between syncs —
            tighten `sync_interval` (or both env-step counts × N_ENVS)
            for fresher data. Phase 3d (2026-05-13): SARSA / MIXED now
            supported via GPU target-net forward + decode (Phase 3a-c);
            the old SEARCH-only restriction is gone.
        verbose: Print progress / config / summary.

    Returns:
        `EZV2TrainStats` with run aggregates.
    """
    comptime ACT = Config.action_dim
    comptime OBS = Config.obs_dim
    comptime LATENT = Config.latent_dim
    comptime BINS = Config.num_bins
    comptime SIMS = Config.num_simulations
    comptime NODES = Config.max_nodes
    comptime MAX_K = Config.num_root_candidates
    comptime CAP = Config.buffer_capacity

    comptime STATE_SIZE = Env.STATE_SIZE
    comptime OBS_DIM = Env.OBS_DIM

    if verbose:
        print()
        print("=== run_ezv2_train_gpu ===")
        print("    NUM_ENV_STEPS         =", NUM_ENV_STEPS)
        print("    N_ENVS                =", N_ENVS)
        print("    train_interval        =", train_interval, "(per env-batch)")
        print("    sync_interval         =", sync_interval, "train_steps")
        print("    target_sync_interval  =", target_sync_interval, "train_steps")
        print("    reanalyze_interval    =", reanalyze_interval, "train_steps")
        print("    reanalyze_warmup      =", reanalyze_warmup, "train_steps")
        print("    Config: LATENT=", LATENT, " BINS=", BINS,
              " BS=", Config.batch_size,
              " K_UNROLL=", Config.unroll_steps,
              " SIMS=", SIMS, " K_GUMBEL=", MAX_K)
        print()

    # ─── Allocate GPU state + initial upload ─────────────────────────────
    var gpu = EZV2GPUStateBase[Config](ctx)
    gpu.upload_from(agent.state, ctx)
    # Phase 3: mirror CPU target nets onto GPU for the MIXED/SARSA boot-V
    # forward.
    gpu.upload_targets_from(agent.state, ctx)
    ctx.synchronize()

    # ─── GPU env buffers ─────────────────────────────────────────────────
    var states_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * STATE_SIZE)
    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS_DIM)
    var actions_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var dones_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var terminated_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)

    var host_obs = ctx.enqueue_create_host_buffer[dtype](N_ENVS * OBS_DIM)
    var host_action = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var host_reward = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var host_done = ctx.enqueue_create_host_buffer[dtype](N_ENVS)

    # ─── Env step workspace (no-op for envs with STEP_WS_SHARED == 0) ───
    comptime ws_size_total = (
        Env.STEP_WS_SHARED + N_ENVS * Env.STEP_WS_PER_ENV
    )
    comptime ws_alloc = ws_size_total if ws_size_total > 0 else 1
    var env_workspace = ctx.enqueue_create_buffer[dtype](ws_alloc)
    if Env.STEP_WS_SHARED + Env.STEP_WS_PER_ENV > 0:
        Env.init_step_workspace_gpu[N_ENVS](ctx, env_workspace)

    # ─── GPU MCTS state + workspace ─────────────────────────────────────
    var mcts_gpu = EZV2GPUMCTSState[
        N_ENVS, NODES, ACT, LATENT, BINS, MAX_K
    ](ctx)
    comptime WS_R = Config.RepModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_D = Config.DynModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_P = Config.PredModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime MAX_WS_AB = WS_R if WS_R > WS_D else WS_D
    comptime MAX_WS = MAX_WS_AB if MAX_WS_AB > WS_P else WS_P
    comptime MCTS_WS_TOTAL = N_ENVS * MAX_WS if MAX_WS > 0 else 1
    var mcts_workspace = ctx.enqueue_create_buffer[dtype](MCTS_WS_TOTAL)

    var host_policies = ctx.enqueue_create_host_buffer[dtype](N_ENVS * ACT)
    var host_visits = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * ACT
    )
    var host_total_value = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * ACT
    )

    # ─── GPU-resident replay buffer mirror ──────────────────────────────
    var gpu_replay = EZV2GPUReplayBuffer[CAP, OBS, ACT, MAX_K](ctx)
    ctx.synchronize()

    # ─── Initial reset ──────────────────────────────────────────────────
    Env.reset_kernel_gpu[N_ENVS, STATE_SIZE](
        ctx, states_buf, rng_seed=rng_seed_base
    )
    Env.extract_obs_kernel_gpu[N_ENVS, STATE_SIZE, OBS_DIM](
        ctx, states_buf, obs_buf
    )
    ctx.synchronize()

    if verbose:
        print("    GPU state + env + MCTS + replay buffer ready")
        print()

    # ─── Per-env CPU obs view ────────────────────────────────────────────
    ctx.enqueue_copy(host_obs.unsafe_ptr(), obs_buf)
    ctx.synchronize()
    var obs_per_env = List[List[Scalar[dtype]]]()
    for e in range(N_ENVS):
        var lst = List[Scalar[dtype]]()
        for d in range(OBS_DIM):
            lst.append(host_obs[e * OBS_DIM + d])
        obs_per_env.append(lst^)

    var ep_return_per_env = List[Float64]()
    for _ in range(N_ENVS):
        ep_return_per_env.append(Float64(0.0))

    var stats = EZV2TrainStats()
    var step_seed: UInt64 = rng_seed_base + 1
    var mcts_seed: UInt32 = UInt32(0)
    var sample_seed: UInt32 = UInt32(0)

    var t0 = perf_counter_ns()

    while stats.total_env_steps < NUM_ENV_STEPS:
        var in_warmup = stats.total_env_steps < warmup_random_steps

        var actions_per_env = List[Int]()
        var policies_per_env = List[InlineArray[Float64, ACT]]()
        var root_value_per_env = List[Float64]()

        if in_warmup:
            # Random-action warmup (paper `start_transitions`). Uniform
            # 1/ACT policy target, root_value=0 — same as the hybrid
            # multi-env demo. No GPU MCTS launches at all during this
            # window, so the buffer fills cheaply with random rollouts.
            for e in range(N_ENVS):
                var rand_a = Int(
                    random_float64() * Float64(ACT)
                )
                if rand_a >= ACT:
                    rand_a = ACT - 1
                var policy = InlineArray[Float64, ACT](uninitialized=True)
                for a in range(ACT):
                    policy[a] = 1.0 / Float64(ACT)
                policies_per_env.append(policy)
                root_value_per_env.append(Float64(0.0))
                actions_per_env.append(rand_a)
                host_action[e] = Scalar[dtype](Float64(rand_a))
        else:
            # ── 1. GPU MCTS — batched over N_ENVS ──────────────────────
            gpu.mcts_search[N_ENVS, NODES, MAX_K, SIMS](
                ctx,
                mcts_gpu,
                obs_buf,
                mcts_workspace,
                v_min=agent.v_min,
                v_max=agent.v_max,
                gamma=agent.gamma,
                rng_seed=mcts_seed,
                apply_legal=False,
                k_actual=MAX_K,
            )
            mcts_seed += UInt32(1)

            ctx.enqueue_copy(
                host_policies.unsafe_ptr(), mcts_gpu.policies_out
            )
            ctx.enqueue_copy(
                host_visits.unsafe_ptr(), mcts_gpu.visit_count
            )
            ctx.enqueue_copy(
                host_total_value.unsafe_ptr(), mcts_gpu.total_value
            )
            ctx.synchronize()

            # ── 2. Per-env action sampling + policy/value extract ─────
            for e in range(N_ENVS):
                var root_off = e * NODES * ACT
                var sum_value = Float64(0.0)
                var sum_visits = 0
                for a in range(ACT):
                    sum_value += Float64(host_total_value[root_off + a])
                    sum_visits += Int(Float64(host_visits[root_off + a]))
                root_value_per_env.append(
                    compute_sve(sum_value, sum_visits)
                )

                var policy = InlineArray[Float64, ACT](uninitialized=True)
                var pol_off = e * ACT
                for a in range(ACT):
                    policy[a] = Float64(host_policies[pol_off + a])
                policies_per_env.append(policy)

                var action: Int
                if agent.temperature < 0.01:
                    action = 0
                    var best = policy[0]
                    for a in range(1, ACT):
                        if policy[a] > best:
                            best = policy[a]
                            action = a
                else:
                    var temp_policy = InlineArray[Float64, ACT](
                        uninitialized=True
                    )
                    var inv_t = 1.0 / agent.temperature
                    var sum_p = Float64(0.0)
                    for a in range(ACT):
                        if policy[a] > 0.0:
                            temp_policy[a] = exp(inv_t * log(policy[a]))
                        else:
                            temp_policy[a] = Float64(0.0)
                        sum_p += temp_policy[a]
                    if sum_p > 0.0:
                        for a in range(ACT):
                            temp_policy[a] /= sum_p
                    else:
                        for a in range(ACT):
                            temp_policy[a] = 1.0 / Float64(ACT)
                    var u = random_float64(0.0, 1.0)
                    var cumsum = Float64(0.0)
                    action = ACT - 1
                    for a in range(ACT):
                        cumsum += temp_policy[a]
                        if u <= cumsum:
                            action = a
                            break
                actions_per_env.append(action)
                host_action[e] = Scalar[dtype](Float64(action))

        ctx.enqueue_copy(actions_buf, host_action.unsafe_ptr())

        # ── 3. GPU env step (batched over N_ENVS) ───────────────────────
        Env.step_kernel_gpu[N_ENVS, STATE_SIZE, OBS_DIM](
            ctx,
            states_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            terminated_buf,
            obs_buf,
            rng_seed=step_seed,
            workspace_ptr=env_workspace.unsafe_ptr(),
        )

        ctx.enqueue_copy(host_reward.unsafe_ptr(), rewards_buf)
        ctx.enqueue_copy(host_done.unsafe_ptr(), dones_buf)
        ctx.synchronize()

        # ── 4. Store transitions on CPU (replay buffer ground truth) ───
        var any_done = False
        for e in range(N_ENVS):
            var reward = Float64(host_reward[e])
            var done = host_done[e] > Scalar[dtype](0.5)

            agent.store_transition(
                obs_per_env[e],
                actions_per_env[e],
                reward,
                policies_per_env[e],
                root_value_per_env[e],
                done,
                env_id=e,
            )
            ep_return_per_env[e] += reward
            stats.total_env_steps += 1

            if done:
                stats.ep_returns.append(ep_return_per_env[e])
                if ep_return_per_env[e] > stats.best_episode_return:
                    stats.best_episode_return = ep_return_per_env[e]
                ep_return_per_env[e] = Float64(0.0)
                any_done = True

        # ── 5. Selective reset for done envs + re-extract obs ───────────
        if any_done:
            step_seed += 1
            Env.selective_reset_kernel_gpu[N_ENVS, STATE_SIZE](
                ctx, states_buf, dones_buf, rng_seed=step_seed,
                workspace_ptr=env_workspace.unsafe_ptr(),
            )
            Env.extract_obs_kernel_gpu[N_ENVS, STATE_SIZE, OBS_DIM](
                ctx, states_buf, obs_buf
            )

        ctx.enqueue_copy(host_obs.unsafe_ptr(), obs_buf)
        ctx.synchronize()
        for e in range(N_ENVS):
            obs_per_env[e].clear()
            for d in range(OBS_DIM):
                obs_per_env[e].append(host_obs[e * OBS_DIM + d])

        # ── 6. GPU train step (host or GPU sampling, dispatched at runtime)
        if (
            agent.state.is_ready()
            and stats.total_env_steps >= warmup_random_steps
            and (stats.total_env_steps // N_ENVS) % train_interval == 0
        ):
            # Pre-train-step gpu_replay sync — required for GPU sampling
            # so the device mirror sees the latest CPU transitions and
            # priorities. Triggered at train_step=0 and every
            # `sync_interval` train calls. For the host-sampling path
            # this block is a no-op (gpu_replay is synced post-train
            # below, matching the legacy behavior).
            if use_gpu_sampling:
                var first_train = stats.num_train_calls == 0
                var sync_now = (
                    first_train
                    or stats.num_train_calls % sync_interval == 0
                )
                if sync_now:
                    gpu_replay.upload_from_cpu(agent.state, ctx)
                    gpu_replay.max_priority = agent.max_priority
                    ctx.synchronize()
                    stats.num_buffer_uploads += 1

            var L_total: Float64
            var L_R: Float64
            var L_P: Float64
            var L_V: Float64
            var L_G: Float64
            if use_gpu_sampling:
                var t = agent.train_step_gpu_with_replay(
                    gpu, gpu_replay, ctx, sample_seed
                )
                L_total = t[0]
                L_R = t[1]
                L_P = t[2]
                L_V = t[3]
                L_G = t[4]
                sample_seed += UInt32(1)
            else:
                var t = agent.train_step_gpu(gpu, ctx)
                L_total = t[0]
                L_R = t[1]
                L_P = t[2]
                L_V = t[3]
                L_G = t[4]

            stats.num_train_calls += 1
            stats.last_L_R = L_R
            stats.last_L_P = L_P
            stats.last_L_V = L_V
            stats.last_L_G = L_G
            if not _is_finite(L_total):
                stats.any_nan_loss = True

            # Sync GPU networks → CPU mirror, plus buffer mirror to GPU.
            # The buffer mirror sync is suppressed for the GPU-sampling
            # path because it already happened pre-train-step above —
            # double-syncing would clobber any GPU-side priority writes
            # we plan to land later (item 5 in `EZV2_FULL_GPU_PLAN.md`).
            if stats.num_train_calls % sync_interval == 0:
                gpu.download_to(agent.state, ctx)
                ctx.synchronize()
                stats.num_gpu_syncs += 1

                if not use_gpu_sampling:
                    gpu_replay.upload_from_cpu(agent.state, ctx)
                    gpu_replay.max_priority = agent.max_priority
                    ctx.synchronize()
                    stats.num_buffer_uploads += 1

            # Hard-sync target ← online + reanalyze on CPU.
            if stats.num_train_calls % target_sync_interval == 0:
                agent.update_target_networks(tau=1.0)
                # Phase 3: mirror fresh CPU targets onto GPU.
                gpu.upload_targets_from(agent.state, ctx)
            if (
                stats.num_train_calls >= reanalyze_warmup
                and stats.num_train_calls % reanalyze_interval == 0
            ):
                _ = agent.reanalyze(num_samples=reanalyze_samples)

        step_seed += 1

        # ── 7. Logging ──────────────────────────────────────────────────
        if verbose and stats.total_env_steps % log_every == 0 and (
            stats.total_env_steps != 0
        ):
            var t_now = perf_counter_ns()
            var wall_s = Float64(t_now - t0) / 1.0e9
            var window = 30
            var n_eps = len(stats.ep_returns)
            var recent = List[Float64]()
            var start_i = (
                n_eps - window if n_eps > window else 0
            )
            for i in range(start_i, n_eps):
                recent.append(stats.ep_returns[i])
            print(
                "[step ", stats.total_env_steps,
                " ep=", n_eps,
                " train=", stats.num_train_calls,
                " syncs=", stats.num_gpu_syncs,
                " wall=", wall_s, "s",
                "] recent_mean=", _mean(recent),
                "  best=", stats.best_episode_return,
                "  L=(R", stats.last_L_R,
                ", P", stats.last_L_P,
                ", V", stats.last_L_V,
                ", G", stats.last_L_G, ")",
            )

    var t_end = perf_counter_ns()
    stats.wall_time_s = Float64(t_end - t0) / 1.0e9
    stats.num_episodes = len(stats.ep_returns)

    # ─── Final GPU → CPU sync ────────────────────────────────────────────
    gpu.download_to(agent.state, ctx)
    ctx.synchronize()
    stats.num_gpu_syncs += 1

    if verbose:
        print()
        print("=== run_ezv2_train_gpu summary ===")
        print("    wall time             =", stats.wall_time_s, "s")
        print("    env steps             =", stats.total_env_steps)
        print("    train_step_gpu calls  =", stats.num_train_calls)
        print("    GPU→CPU syncs         =", stats.num_gpu_syncs)
        print("    buffer uploads        =", stats.num_buffer_uploads)
        print("    episodes finished     =", stats.num_episodes)
        print("    best episode return   =", stats.best_episode_return)
        print("    any NaN loss          =", stats.any_nan_loss)
        print("    final loss components:")
        print("        L_R =", stats.last_L_R)
        print("        L_P =", stats.last_L_P)
        print("        L_V =", stats.last_L_V)
        print("        L_G =", stats.last_L_G)

    return stats^
