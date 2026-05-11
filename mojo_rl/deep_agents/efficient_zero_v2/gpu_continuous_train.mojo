"""Unified full-GPU training driver for EfficientZero V2 (continuous actions).

Continuous-action sibling of `gpu_train.mojo`. Same loop shape, same
sync cadence, same stats struct — adapted for:

  • `GPUContinuousEnv` env-step kernels (action buffer is
    `[N_ENVS × ACTION_DIM]` floats, not discrete indices).
  • `run_sampled_gumbel_search_gpu` for MCTS (sampled-Gumbel over K
    continuous candidates per env, paper App. A).
  • `GenericEZV2ContinuousAgent.store_transition` 7-arg path (action
    vector + K-candidate full-π targets).

Action-selection outputs are downloaded each step from
`EZV2GPUSampledMCTSState`:
  • `chosen_actions[N_ENVS × ACT_DIM]` → played action.
  • root slice of `visit_count` + `total_value` → SVE backup.
  • root slice of `actions` (per-candidate vectors) → K-candidate full-π
    target actions.
  • `root_visits[N_ENVS × K_ROOT]` → K-candidate full-π weights.

Train step uses the host-sampling `train_step_gpu` path. The continuous
agent's GPU-sampling variant (`train_step_gpu_with_replay`) is
SEARCH-only today; configs using SARSA / MIXED (HalfCheetah) must keep
`use_gpu_sampling=False`. A runtime check enforces this.

Reanalyze is dispatched on the agent's continuous reanalyze method
(host-side, target nets), matching the discrete driver's cadence.

This file does NOT depend on any continuous-side GPU replay extension
because the host-sampling path reads its targets from `agent.state`
directly. If/when GPU sampling lands for continuous (full-π in
particular), `EZV2GPUReplayBuffer` will need `mcts_sampled_actions` +
`mcts_improved_policy` fields and `gpu_replay.upload_from_cpu` will need
to mirror them.
"""

from std.math import exp, log
from std.random import random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.core.env_traits import GPUContinuousEnv
from mojo_rl.deep_agents.efficient_zero_v2.configs import (
    EZV2DiscreteConfig,
    VALUE_TARGET_SEARCH,
)
from mojo_rl.deep_agents.efficient_zero_v2.continuous_agent import (
    GenericEZV2ContinuousAgent,
)
from mojo_rl.deep_agents.efficient_zero_v2.state import EZV2GPUStateBase
from mojo_rl.deep_agents.efficient_zero_v2.gpu_mcts_sampled import (
    EZV2GPUSampledMCTSState,
    run_sampled_gumbel_search_gpu,
)
from mojo_rl.deep_agents.efficient_zero_v2.gpu_replay import (
    EZV2GPUReplayBuffer,
)
from mojo_rl.deep_agents.efficient_zero_v2.gpu_train import EZV2TrainStats
from mojo_rl.deep_agents.efficient_zero_v2.strategies import compute_sve
from mojo_rl.nn.constants import dtype


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


def run_ezv2_continuous_train_gpu[
    Env: GPUContinuousEnv,
    Config: EZV2DiscreteConfig,
    N_ENVS: Int,
    NUM_ENV_STEPS: Int,
](
    mut agent: GenericEZV2ContinuousAgent[Config],
    ctx: DeviceContext,
    *,
    train_interval: Int = 1,
    sync_interval: Int = 50,
    target_sync_interval: Int = 200,
    reanalyze_interval: Int = 200,
    reanalyze_samples: Int = 32,
    reanalyze_warmup: Int = 1000,
    warmup_random_steps: Int = 2_000,
    max_steps_per_episode: Int = 1_000,
    log_every: Int = 2_000,
    rng_seed_base: UInt64 = UInt64(2026),
    use_gpu_sampling: Bool = False,
    # PUCT / Q-normalization constants for the GPU MCTS. Defaults match
    # the converging CPU-side `SampledGumbelMCTS` in `continuous_agent.mojo:219`
    # (`c_scale=1.0` overrides the GPU MCTS signature's `0.1` default,
    # which gave a 10× weaker Q signal in candidate scoring and was the
    # root cause of the Pendulum GPU-driver regression observed
    # 2026-05-12 — see git log).
    mcts_c_visit: Float64 = 50.0,
    mcts_c_scale: Float64 = 1.0,
    verbose: Bool = True,
) raises -> EZV2TrainStats:
    """Drive EZ-V2 continuous training fully on GPU (env + MCTS) with a
    CPU mirror for replay-buffer ground truth + reanalyze.

    Parameters:
        Env: `GPUContinuousEnv` providing `step_kernel_gpu`,
            `reset_kernel_gpu`, `selective_reset_kernel_gpu`,
            `extract_obs_kernel_gpu`, and `init_step_workspace_gpu`.
        Config: `EZV2DiscreteConfig` (trait name is historical — works
            for `EZV2ContinuousMLPConfig` too via the `ActSpace` dispatch).
            `Config.ActSpace.IS_CONTINUOUS` must be True.
        N_ENVS: Parallel envs (≥ 1). Trains every `train_interval`
            env-batches (not env-steps).
        NUM_ENV_STEPS: Total env-step transitions across all envs.

    Args:
        agent: Pre-constructed `GenericEZV2ContinuousAgent` (its
            `n_envs` must equal `N_ENVS`).
        ctx: GPU device context.
        train_interval: Train every Nth env-batch.
        sync_interval: GPU → CPU network sync every Nth train.
        target_sync_interval: Hard-copy target nets every Nth train.
        reanalyze_interval: Reanalyze every Nth train post-warmup.
        reanalyze_samples: How many windows per reanalyze call.
        reanalyze_warmup: Skip reanalyze until this many train calls.
        warmup_random_steps: Uniform-random actions for the first N
            env-step transitions (sparse-reward warmup). No GPU MCTS
            launches and no train step during this window.
        max_steps_per_episode: Truncation horizon per episode. Continuous
            envs (HalfCheetah etc.) generally use 1000.
        log_every: Print a log line every Nth env-step.
        rng_seed_base: Initial seed for env reset RNG.
        use_gpu_sampling: When True, the train step samples its batch
            via the GPU replay path. The continuous agent's variant only
            supports `VALUE_TARGET_SEARCH` today; passing True with a
            SARSA / MIXED config raises a runtime error. SARSA configs
            (HalfCheetah default) must keep this False.
        verbose: Print progress / config / summary.

    Returns:
        `EZV2TrainStats` with run aggregates.
    """
    comptime ACT_DIM = Config.action_dim
    comptime OBS = Config.obs_dim
    comptime LATENT = Config.latent_dim
    comptime BINS = Config.num_bins
    comptime SIMS = Config.num_simulations
    comptime NODES = Config.max_nodes
    comptime K_ROOT = Config.num_root_candidates
    # Paper App. A: K_NON_ROOT = K_ROOT // 2 (matches the agent's CPU MCTS
    # default in `continuous_agent.mojo:140-145`). Floor to 1.
    comptime K_NON_ROOT = (
        Config.num_root_candidates // 2
        if Config.num_root_candidates // 2 >= 1
        else 1
    )
    comptime CAP = 50000  # matches `EZV2DiscreteCPUState`'s default _CAP

    comptime STATE_SIZE = Env.STATE_SIZE
    comptime OBS_DIM = Env.OBS_DIM
    comptime ACTION_DIM = Env.ACTION_DIM

    comptime MAX_ACTION_F = Config.ActSpace.MAX_ACTION
    comptime MIN_STD_F = Config.ActSpace.MIN_STD
    comptime STD_MAG_F = Config.ActSpace.STD_MAGNIFICATION

    # Compile-time sanity checks.
    comptime if not Config.ActSpace.IS_CONTINUOUS:
        comptime assert False, (
            "run_ezv2_continuous_train_gpu requires a continuous Config"
            " (Config.ActSpace.IS_CONTINUOUS must be True). Use"
            " run_ezv2_train_gpu for discrete configs."
        )
    comptime if Config.action_dim != Env.ACTION_DIM:
        comptime assert False, (
            "Config.action_dim does not match Env.ACTION_DIM. Both must"
            " agree on the continuous-action vector width."
        )
    comptime if Config.obs_dim != Env.OBS_DIM:
        comptime assert False, (
            "Config.obs_dim does not match Env.OBS_DIM."
        )

    # Runtime sanity: GPU sampling for continuous is SEARCH-mode-only.
    if use_gpu_sampling and Config.value_target_mode != VALUE_TARGET_SEARCH:
        raise Error(
            "use_gpu_sampling=True is only supported with"
            " VALUE_TARGET_MODE=VALUE_TARGET_SEARCH for continuous EZ-V2."
            " SARSA / MIXED require a GPU target-net forward that isn't"
            " ported yet; use train_step_gpu (use_gpu_sampling=False)."
        )

    if verbose:
        print()
        print("=== run_ezv2_continuous_train_gpu ===")
        print("    NUM_ENV_STEPS         =", NUM_ENV_STEPS)
        print("    N_ENVS                =", N_ENVS)
        print("    train_interval        =", train_interval, "(per env-batch)")
        print("    sync_interval         =", sync_interval, "train_steps")
        print("    target_sync_interval  =", target_sync_interval, "train_steps")
        print("    reanalyze_interval    =", reanalyze_interval, "train_steps")
        print("    reanalyze_warmup      =", reanalyze_warmup, "train_steps")
        print("    warmup_random_steps   =", warmup_random_steps)
        print("    max_steps_per_episode =", max_steps_per_episode)
        print(
            "    Config: OBS=", OBS, " ACT_DIM=", ACT_DIM,
            " LATENT=", LATENT, " BINS=", BINS,
        )
        print(
            "            BS=", Config.batch_size,
            " K_UNROLL=", Config.unroll_steps,
            " SIMS=", SIMS, " K_ROOT=", K_ROOT, " K_NON_ROOT=", K_NON_ROOT,
        )
        print(
            "            MAX_ACTION=", MAX_ACTION_F,
            " MIN_STD=", MIN_STD_F,
            " STD_MAG=", STD_MAG_F,
        )
        print("    use_gpu_sampling      =", use_gpu_sampling)
        print()

    # ─── Allocate GPU state + initial upload ─────────────────────────────
    var gpu = EZV2GPUStateBase[Config](ctx)
    gpu.upload_from(agent.state, ctx)
    ctx.synchronize()

    # ─── GPU env buffers ─────────────────────────────────────────────────
    # Continuous: actions_buf is [N_ENVS * ACTION_DIM], one row per env.
    var states_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * STATE_SIZE)
    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS_DIM)
    var actions_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * ACTION_DIM)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var dones_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var terminated_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)

    var host_obs = ctx.enqueue_create_host_buffer[dtype](N_ENVS * OBS_DIM)
    var host_action = ctx.enqueue_create_host_buffer[dtype](N_ENVS * ACTION_DIM)
    var host_reward = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var host_done = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    # `host_terminated` distinguishes natural termination from time-limit
    # truncation. The env's `terminated_buf` is 1.0 only on real terminal
    # states; `dones_buf` combines (terminated OR truncated). We need
    # both to thread the correct bootstrap mask into the replay buffer
    # (terminations field) — without it, every truncation kills V_next
    # in the N-step target and V is systematically biased.
    var host_terminated = ctx.enqueue_create_host_buffer[dtype](N_ENVS)

    # ─── Env step workspace (no-op for envs with STEP_WS_SHARED == 0) ───
    comptime ws_size_total = (
        Env.STEP_WS_SHARED + N_ENVS * Env.STEP_WS_PER_ENV
    )
    comptime ws_alloc = ws_size_total if ws_size_total > 0 else 1
    var env_workspace = ctx.enqueue_create_buffer[dtype](ws_alloc)
    if Env.STEP_WS_SHARED + Env.STEP_WS_PER_ENV > 0:
        Env.init_step_workspace_gpu[N_ENVS](ctx, env_workspace)

    # ─── GPU MCTS state + workspace ─────────────────────────────────────
    var mcts_gpu = EZV2GPUSampledMCTSState[
        N_ENVS, NODES, ACT_DIM, LATENT, BINS, K_ROOT, K_NON_ROOT
    ](ctx)
    comptime WS_R = Config.RepModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_D = Config.DynModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_P = Config.PredModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime MAX_WS_AB = WS_R if WS_R > WS_D else WS_D
    comptime MAX_WS = MAX_WS_AB if MAX_WS_AB > WS_P else WS_P
    comptime MCTS_WS_TOTAL = N_ENVS * MAX_WS if MAX_WS > 0 else 1
    var mcts_workspace = ctx.enqueue_create_buffer[dtype](MCTS_WS_TOTAL)

    # Host mirrors for MCTS outputs we need each step.
    #   - chosen_actions: per-env action vector played in env.
    #   - root_visits:    K_ROOT improved-policy weights per env (sum=1).
    #   - visit_count[root]: per-candidate visit counts for SVE backup.
    #   - total_value[root]: per-candidate Σ-of-backed-up-values for SVE.
    #   - actions[root]: per-candidate action vectors for full-π targets.
    # The visit/total/action arrays span all MAX_NODES but we only read
    # the node-0 (root) slice each step. Bulk-download is simpler than
    # offset-aware host slicing.
    var host_chosen = ctx.enqueue_create_host_buffer[dtype](N_ENVS * ACT_DIM)
    var host_root_visits = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * K_ROOT
    )
    var host_node_visit = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * K_ROOT
    )
    var host_node_total_value = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * K_ROOT
    )
    var host_node_actions = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * K_ROOT * ACT_DIM
    )

    # ─── GPU-resident replay buffer mirror ──────────────────────────────
    # Only used by the GPU-sampling train path. For the default host-
    # sampling path it's allocated but never read.
    var gpu_replay = EZV2GPUReplayBuffer[CAP, OBS, ACT_DIM](ctx)
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

    # ─── Per-env CPU obs view (refreshed each step) ──────────────────────
    ctx.enqueue_copy(host_obs.unsafe_ptr(), obs_buf)
    ctx.synchronize()
    var obs_per_env = List[List[Scalar[dtype]]]()
    for e in range(N_ENVS):
        var lst = List[Scalar[dtype]]()
        for d in range(OBS_DIM):
            lst.append(host_obs[e * OBS_DIM + d])
        obs_per_env.append(lst^)

    var ep_return_per_env = List[Float64]()
    var ep_steps_per_env = List[Int]()
    for _ in range(N_ENVS):
        ep_return_per_env.append(Float64(0.0))
        ep_steps_per_env.append(0)

    var stats = EZV2TrainStats()
    var step_seed: UInt64 = rng_seed_base + 1
    var mcts_seed: UInt32 = UInt32(0)
    var sample_seed: UInt32 = UInt32(0)

    var t0 = perf_counter_ns()

    while stats.total_env_steps < NUM_ENV_STEPS:
        var in_warmup = stats.total_env_steps < warmup_random_steps

        var actions_per_env = List[List[Scalar[dtype]]]()
        var sampled_actions_per_env = List[List[Scalar[dtype]]]()
        var improved_policy_per_env = List[List[Scalar[dtype]]]()
        var root_value_per_env = List[Float64]()

        if in_warmup:
            # Uniform [-MAX_ACTION, +MAX_ACTION] per dim. K-candidate slots
            # carry a one-hot at the played action so the simple-best
            # loss path (ACT_DIM>1) and the full-π path (ACT_DIM==1) both
            # reduce to chosen-action NLL when the agent eventually trains
            # on these warmup transitions. root_value=0 — neutral target.
            for e in range(N_ENVS):
                var rand_act = List[Scalar[dtype]](capacity=ACT_DIM)
                for _ in range(ACT_DIM):
                    rand_act.append(
                        Scalar[dtype](
                            random_float64(-MAX_ACTION_F, MAX_ACTION_F)
                        )
                    )

                var sampled = List[Scalar[dtype]](
                    capacity=K_ROOT * ACT_DIM
                )
                var improved = List[Scalar[dtype]](capacity=K_ROOT)
                for _ in range(K_ROOT * ACT_DIM):
                    sampled.append(Scalar[dtype](0.0))
                for _ in range(K_ROOT):
                    improved.append(Scalar[dtype](0.0))
                for d in range(ACT_DIM):
                    sampled[d] = rand_act[d]
                improved[0] = Scalar[dtype](1.0)

                for d in range(ACT_DIM):
                    host_action[e * ACT_DIM + d] = rand_act[d]

                actions_per_env.append(rand_act^)
                sampled_actions_per_env.append(sampled^)
                improved_policy_per_env.append(improved^)
                root_value_per_env.append(Float64(0.0))
        else:
            # ── 1. GPU MCTS — batched across all envs ─────────────────
            run_sampled_gumbel_search_gpu[
                N_ENVS,
                NODES,
                ACT_DIM,
                LATENT,
                BINS,
                K_ROOT,
                K_NON_ROOT,
                SIMS,
                Config.RepModel,
                Config.DynModel,
                Config.PredModel,
                Config.OptType,
                Config.OptType,
                Config.OptType,
            ](
                ctx,
                mcts_gpu,
                obs_buf,
                gpu.representation,
                gpu.dynamics,
                gpu.prediction,
                mcts_workspace,
                v_min=agent.v_min,
                v_max=agent.v_max,
                max_action=MAX_ACTION_F,
                min_std=MIN_STD_F,
                std_magnification=STD_MAG_F,
                c_visit=mcts_c_visit,
                c_scale=mcts_c_scale,
                gamma=agent.gamma,
                deterministic=(agent.temperature < 0.01),
                rng_seed=mcts_seed,
            )
            mcts_seed += UInt32(1)

            ctx.enqueue_copy(host_chosen.unsafe_ptr(), mcts_gpu.chosen_actions)
            ctx.enqueue_copy(
                host_root_visits.unsafe_ptr(), mcts_gpu.root_visits
            )
            ctx.enqueue_copy(host_node_visit.unsafe_ptr(), mcts_gpu.visit_count)
            ctx.enqueue_copy(
                host_node_total_value.unsafe_ptr(), mcts_gpu.total_value
            )
            ctx.enqueue_copy(host_node_actions.unsafe_ptr(), mcts_gpu.actions)
            ctx.synchronize()

            # ── 2. Per-env extract: SVE, chosen action, K-candidates ──
            for e in range(N_ENVS):
                # Root candidate visits + values live at node 0:
                #   visit_count[e * NODES * K_ROOT + 0 .. + K_ROOT]
                # K_PAD = K_ROOT by construction.
                var root_slot_off = e * NODES * K_ROOT
                var sum_value = Float64(0.0)
                var sum_visits = 0
                for i in range(K_ROOT):
                    sum_value += Float64(host_node_total_value[root_slot_off + i])
                    sum_visits += Int(
                        Float64(host_node_visit[root_slot_off + i])
                    )
                root_value_per_env.append(compute_sve(sum_value, sum_visits))

                # Chosen action (the GPU search picked this per the
                # `deterministic` flag we passed — soft sample by visits
                # if training, argmax if eval).
                var action = List[Scalar[dtype]](capacity=ACT_DIM)
                for d in range(ACT_DIM):
                    action.append(host_chosen[e * ACT_DIM + d])
                for d in range(ACT_DIM):
                    host_action[e * ACT_DIM + d] = action[d]

                # K root candidate action vectors. Offset:
                #   actions[e * NODES * K_ROOT * ACT_DIM + 0 .. + K_ROOT*ACT_DIM]
                var act_off = e * NODES * K_ROOT * ACT_DIM
                var sampled = List[Scalar[dtype]](
                    capacity=K_ROOT * ACT_DIM
                )
                for j in range(K_ROOT * ACT_DIM):
                    sampled.append(host_node_actions[act_off + j])

                # Improved-policy weights (sum=1 over K candidates).
                var iv_off = e * K_ROOT
                var improved = List[Scalar[dtype]](capacity=K_ROOT)
                for i in range(K_ROOT):
                    improved.append(host_root_visits[iv_off + i])

                actions_per_env.append(action^)
                sampled_actions_per_env.append(sampled^)
                improved_policy_per_env.append(improved^)

        ctx.enqueue_copy(actions_buf, host_action.unsafe_ptr())

        # ── 3. GPU env step (batched across all envs) ───────────────────
        Env.step_kernel_gpu[N_ENVS, STATE_SIZE, OBS_DIM, ACTION_DIM](
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
        ctx.enqueue_copy(host_terminated.unsafe_ptr(), terminated_buf)
        ctx.synchronize()

        # ── 4. Store transitions on CPU (replay buffer ground truth) ───
        # We mark truncated envs done-or-trunc for the replay flush but
        # also force the GPU `dones_buf` to 1 so the next selective-reset
        # picks them up. Native-done envs already have dones_buf=1 from
        # the env kernel.
        var any_done_or_trunc = False
        var any_truncation = False
        for e in range(N_ENVS):
            var reward = Float64(host_reward[e])
            var native_done = host_done[e] > Scalar[dtype](0.5)
            var natively_terminated = (
                host_terminated[e] > Scalar[dtype](0.5)
            )
            ep_steps_per_env[e] += 1
            var truncated = ep_steps_per_env[e] >= max_steps_per_episode
            var done_or_trunc = native_done or truncated
            # `truly_terminated` is True only on a real terminal state
            # (env's `terminated_buf=1`). Truncation (step-count clamp
            # or env-internal truncation that only sets `dones_buf=1`)
            # leaves it False so the N-step TD target keeps γ^n·V_next.
            # See SequenceReplayBuffer.add_with_termination's docstring.
            var truly_terminated = natively_terminated

            agent.store_transition(
                obs_per_env[e],
                actions_per_env[e],
                reward,
                root_value_per_env[e],
                sampled_actions_per_env[e],
                improved_policy_per_env[e],
                done_or_trunc,
                env_id=e,
                terminated=truly_terminated,
            )
            ep_return_per_env[e] += reward
            stats.total_env_steps += 1

            if done_or_trunc:
                stats.ep_returns.append(ep_return_per_env[e])
                if ep_return_per_env[e] > stats.best_episode_return:
                    stats.best_episode_return = ep_return_per_env[e]
                ep_return_per_env[e] = Float64(0.0)
                ep_steps_per_env[e] = 0
                any_done_or_trunc = True

            if truncated and not native_done:
                # Force GPU dones_buf[e] = 1 so selective_reset
                # actually resets this env on the next dispatch.
                host_done[e] = Scalar[dtype](1.0)
                any_truncation = True

        # ── 5. Selective reset for done/truncated envs + re-extract obs ─
        if any_done_or_trunc:
            if any_truncation:
                # Sync the truncation-corrected mask back to GPU.
                ctx.enqueue_copy(dones_buf, host_done.unsafe_ptr())
            step_seed += 1
            Env.selective_reset_kernel_gpu[N_ENVS, STATE_SIZE](
                ctx,
                states_buf,
                dones_buf,
                rng_seed=step_seed,
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

        # ── 6. GPU train step ───────────────────────────────────────────
        if (
            agent.state.is_ready()
            and stats.total_env_steps >= warmup_random_steps
            and (stats.total_env_steps // N_ENVS) % train_interval == 0
        ):
            # Pre-train gpu_replay sync if using GPU sampling. The host-
            # sampling path skips this — `train_step_gpu` reads from
            # `agent.state` directly.
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
            # `train_step_gpu_with_replay` for continuous comptime-
            # asserts SEARCH-only (see continuous_agent.mojo:1092). Mojo
            # walks both branches of a runtime `if` at compile time, so
            # the call must be elided at the *comptime* level when the
            # config is SARSA / MIXED. The earlier runtime guard catches
            # `use_gpu_sampling=True` with a non-SEARCH config; this
            # `comptime if` keeps the SARSA build path callable.
            comptime SEARCH_MODE = (
                Config.value_target_mode == VALUE_TARGET_SEARCH
            )
            comptime if SEARCH_MODE:
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
            else:
                # Non-SEARCH configs: CPU sampling only (the runtime
                # guard at the top of the function rejects
                # `use_gpu_sampling=True` here).
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

            # GPU → CPU network sync + optional buffer mirror.
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
            if (
                stats.num_train_calls >= reanalyze_warmup
                and stats.num_train_calls % reanalyze_interval == 0
            ):
                _ = agent.reanalyze(num_samples=reanalyze_samples)

        step_seed += 1

        # ── 7. Logging ──────────────────────────────────────────────────
        if (
            verbose
            and stats.total_env_steps % log_every == 0
            and stats.total_env_steps != 0
        ):
            var t_now = perf_counter_ns()
            var wall_s = Float64(t_now - t0) / 1.0e9
            var window = 30
            var n_eps = len(stats.ep_returns)
            var recent = List[Float64]()
            var start_i = n_eps - window if n_eps > window else 0
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
        print("=== run_ezv2_continuous_train_gpu summary ===")
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
