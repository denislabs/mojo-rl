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

from std.collections.optional import Optional
from std.math import exp, log
from std.memory import UnsafePointer
from std.random import random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.core.env_traits import GPUContinuousEnv
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.core.obs_norm import ObsNormStats
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
from mojo_rl.nn.training.scheduler import Scheduler, ConstantSchedule


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
    # Phase 4 (2026-05-13): reanalyze samples is comptime so the GPU MCTS
    # state can be allocated up front. The runtime `reanalyze_samples`
    # kwarg is gone — pass via this template parameter instead.
    REANALYZE_SAMPLES: Int = 32,
    L: Logger = NoOpLogger,
    # LR scheduler applied once per train_step. Default `ConstantSchedule`
    # = no schedule. Override e.g. `LinearWarmupSchedule[WARMUP_EPOCHS=1000]`
    # to match the reference `dmc_state.yaml: lr_warm_up=0.01` (= 1% of
    # `training_steps=100000` = 1000 train-step linear warmup, then flat).
    # The scheduler reads (train_step, total_train_steps) where the latter
    # is computed from `NUM_ENV_STEPS * train_steps_per_iter / N_ENVS` at
    # call time. The lr_scale is broadcast to all 5 networks (rep / dyn /
    # pred / projector / predictor) via `GPUNetworkState.set_lr_scale`.
    SCHEDULER: Scheduler = ConstantSchedule,
](
    mut agent: GenericEZV2ContinuousAgent[Config],
    ctx: DeviceContext,
    *,
    train_interval: Int = 1,
    # Number of training steps to run per training-interval firing.
    # Defaults to `N_ENVS` → UTD = 1.0 (one grad step per env transition),
    # matching the DMC reference `dmc_state.yaml` (training_steps=100000,
    # total_transitions=100000). Override with a smaller value to under-
    # train (faster wall clock) or a larger value for higher UTD.
    train_steps_per_iter: Int = N_ENVS,
    sync_interval: Int = 50,
    target_sync_interval: Int = 200,
    reanalyze_interval: Int = 200,
    reanalyze_warmup: Int = 1000,
    warmup_random_steps: Int = 2_000,
    max_steps_per_episode: Int = 1_000,
    log_every: Int = 2_000,
    rng_seed_base: UInt64 = UInt64(2026),
    use_gpu_sampling: Bool = False,
    # PUCT / Q-normalization constants for the GPU MCTS. Defaults
    # match the reference EZ-V2 DMC config (`dmc_state.yaml`:
    # c_visit=50, c_scale=0.1). Project previously used c_scale=1.0;
    # tuned down to 0.1 on 2026-05-16 after `inspect_root_gpu` showed
    # the Q-dominated regime was preventing improved-policy commitment
    # (sigma_scale=54 vs log_prior range ~8 → noisy single-rollout Q
    # spreads jerk the target around). See `continuous_agent.mojo:243`
    # for the matching CPU-side change.
    mcts_c_visit: Float64 = 50.0,
    mcts_c_scale: Float64 = 0.1,
    # Hybrid diagnostic: when False, replace the GPU MCTS with the
    # agent's CPU `SampledGumbelMCTS` (one search per env, sequential).
    # Useful for isolating GPU-MCTS bugs from env/training-driver bugs
    # — if a config converges with `use_gpu_mcts=False` but not True,
    # the issue is in `gpu_mcts_sampled.mojo`. Defaults to True (GPU
    # MCTS, the production path); pass False only for debugging since
    # the CPU MCTS is much slower per env-step than the batched GPU
    # search.
    use_gpu_mcts: Bool = True,
    obs_norm: Bool = False,
    logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
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
        REANALYZE_SAMPLES: Number of samples to draw from the replay
            buffer for reanalyze.
        L: Logger type.
        SCHEDULER: LR-scaler applied once per train step. Default
            `ConstantSchedule` is a no-op. Pass `LinearWarmupSchedule[N]`
            for reference-parity warmup. The schedule is broadcast to all
            5 networks (rep / dyn / pred / projector / predictor).

    Args:
        agent: Pre-constructed `GenericEZV2ContinuousAgent` (its
            `n_envs` must equal `N_ENVS`).
        ctx: GPU device context.
        train_interval: Train every Nth env-batch.
        train_steps_per_iter: Number of training steps to run per training-interval firing.
        sync_interval: GPU → CPU network sync every Nth train.
        target_sync_interval: Hard-copy target nets every Nth train.
        reanalyze_interval: Reanalyze every Nth train post-warmup.
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
        mcts_c_visit: PUCT / Q-normalization constant for the GPU MCTS.
        mcts_c_scale: PUCT / Q-normalization constant for the GPU MCTS.
        use_gpu_mcts: When True, use the GPU MCTS.
        obs_norm: When True, normalize the per-step `obs_buf` in place
            using a running mean / variance tracker (CleanRL VecNormalize
            semantics). Stats update from every env step; replay stores
            the normalized obs. Off by default to preserve every existing
            script's baseline. Enable on envs with non-zero-mean / wide-
            scale obs (HalfCheetah, Humanoid, Ant, etc.).
        logger: Logger for logging progress.
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
        Config.num_root_candidates // 2 if Config.num_root_candidates // 2
        >= 1 else 1
    )
    comptime CAP = 50000  # matches `EZV2DiscreteCPUState`'s default _CAP

    comptime STATE_SIZE = Env.STATE_SIZE
    comptime OBS_DIM = Env.OBS_DIM
    comptime ACTION_DIM = Env.ACTION_DIM

    comptime MAX_ACTION_F = Config.ActSpace.MAX_ACTION
    comptime MIN_STD_F = Config.ActSpace.MIN_STD
    comptime STD_MAG_F = Config.ActSpace.STD_MAGNIFICATION
    comptime SOFT_CLAMP_F = Config.ActSpace.SOFT_CLAMP
    comptime INIT_STD_F = Config.ActSpace.INIT_STD

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
        comptime assert False, "Config.obs_dim does not match Env.OBS_DIM."

    # Phase 3d (2026-05-13): `use_gpu_sampling=True` now supports
    # SARSA/MIXED via the GPU target-net forward + decode added in
    # Phase 3b/3c. The old SEARCH-only assertion has been dropped.

    if verbose:
        print()
        print("=== run_ezv2_continuous_train_gpu ===")
        print("    NUM_ENV_STEPS         =", NUM_ENV_STEPS)
        print("    N_ENVS                =", N_ENVS)
        print("    train_interval        =", train_interval, "(per env-batch)")
        print("    sync_interval         =", sync_interval, "train_steps")
        print(
            "    target_sync_interval  =", target_sync_interval, "train_steps"
        )
        print("    reanalyze_interval    =", reanalyze_interval, "train_steps")
        print(
            "    reanalyze_samples     =", REANALYZE_SAMPLES, "(comptime, GPU)"
        )
        print("    reanalyze_warmup      =", reanalyze_warmup, "train_steps")
        print("    warmup_random_steps   =", warmup_random_steps)
        print("    max_steps_per_episode =", max_steps_per_episode)
        print(
            "    Config: OBS=",
            OBS,
            " ACT_DIM=",
            ACT_DIM,
            " LATENT=",
            LATENT,
            " BINS=",
            BINS,
        )
        print(
            "            BS=",
            Config.batch_size,
            " K_UNROLL=",
            Config.unroll_steps,
            " SIMS=",
            SIMS,
            " K_ROOT=",
            K_ROOT,
            " K_NON_ROOT=",
            K_NON_ROOT,
        )
        print(
            "            MAX_ACTION=",
            MAX_ACTION_F,
            " MIN_STD=",
            MIN_STD_F,
            " STD_MAG=",
            STD_MAG_F,
        )
        print("    use_gpu_sampling      =", use_gpu_sampling)
        print()

    # ─── Allocate GPU state + initial upload ─────────────────────────────
    var gpu = EZV2GPUStateBase[Config](ctx)
    gpu.upload_from(agent.state, ctx)
    # Phase 3: mirror CPU target nets onto GPU so MIXED/SARSA boot-V
    # forwards can run on device. Targets were hard-synced from CPU
    # online in `GenericEZV2ContinuousAgent.__init__`.
    gpu.upload_targets_from(agent.state, ctx)
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
    comptime ws_size_total = (Env.STEP_WS_SHARED + N_ENVS * Env.STEP_WS_PER_ENV)
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
    # Additional buffers needed to reconstruct CPU's smooth
    # `_improved_policy_at(0)` formula on host (paper Eq. 4):
    #   π_improved(i) = softmax(log_prior + sigma_q)
    # where sigma_q[i] = (c_visit + max_visit) · c_scale · normalize(Q_i).
    # The GPU's `root_visits` output is the *normalized visit count
    # distribution* — semantically different from the agent's intended
    # full-π target (the smooth softmax form). Without matching CPU's
    # formula here, the GPU agent's policy loss fits the SH visit-
    # distribution instead, which is sparse and concentrated on SH
    # survivors and prevents learning. See mcts_sampled.mojo:695-726.
    var host_log_prior = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES * K_ROOT
    )
    var host_node_value = ctx.enqueue_create_host_buffer[dtype](N_ENVS * NODES)
    var host_node_total_visits = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * NODES
    )
    var host_min_q = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var host_max_q = ctx.enqueue_create_host_buffer[dtype](N_ENVS)

    # ─── GPU reanalyze state (Phase 4, 2026-05-13) ──────────────────────
    # Dedicated GPU MCTS state + obs staging + workspace + host mirrors
    # sized for `REANALYZE_SAMPLES`. The reanalyze GPU search runs against
    # the *target* networks (`gpu.representation_target` etc.) — same
    # semantics as `agent.reanalyze`, but batched across all samples in
    # one GPU launch instead of N sequential CPU MCTS searches.
    var reanalyze_mcts_gpu = EZV2GPUSampledMCTSState[
        REANALYZE_SAMPLES, NODES, ACT_DIM, LATENT, BINS, K_ROOT, K_NON_ROOT
    ](ctx)
    var reanalyze_obs_buf = ctx.enqueue_create_buffer[dtype](
        REANALYZE_SAMPLES * OBS
    )
    comptime REANALYZE_WS_TOTAL = (
        REANALYZE_SAMPLES * MAX_WS if MAX_WS > 0 else 1
    )
    var reanalyze_workspace = ctx.enqueue_create_buffer[dtype](
        REANALYZE_WS_TOTAL
    )
    var host_reanalyze_obs = ctx.enqueue_create_host_buffer[dtype](
        REANALYZE_SAMPLES * OBS
    )
    var host_reanalyze_chosen = ctx.enqueue_create_host_buffer[dtype](
        REANALYZE_SAMPLES * ACT_DIM
    )
    var host_reanalyze_node_visit = ctx.enqueue_create_host_buffer[dtype](
        REANALYZE_SAMPLES * NODES * K_ROOT
    )
    var host_reanalyze_node_total_value = ctx.enqueue_create_host_buffer[dtype](
        REANALYZE_SAMPLES * NODES * K_ROOT
    )
    var host_reanalyze_node_actions = ctx.enqueue_create_host_buffer[dtype](
        REANALYZE_SAMPLES * NODES * K_ROOT * ACT_DIM
    )
    var host_reanalyze_log_prior = ctx.enqueue_create_host_buffer[dtype](
        REANALYZE_SAMPLES * NODES * K_ROOT
    )
    var host_reanalyze_node_value = ctx.enqueue_create_host_buffer[dtype](
        REANALYZE_SAMPLES * NODES
    )
    var host_reanalyze_node_total_visits = ctx.enqueue_create_host_buffer[
        dtype
    ](REANALYZE_SAMPLES * NODES)
    var host_reanalyze_min_q = ctx.enqueue_create_host_buffer[dtype](
        REANALYZE_SAMPLES
    )
    var host_reanalyze_max_q = ctx.enqueue_create_host_buffer[dtype](
        REANALYZE_SAMPLES
    )
    var reanalyze_seed: UInt32 = UInt32(0)

    # ─── GPU-resident replay buffer mirror ──────────────────────────────
    # Only used by the GPU-sampling train path. For the default host-
    # sampling path it's allocated but never read.
    var gpu_replay = EZV2GPUReplayBuffer[CAP, OBS, ACT_DIM, K_ROOT](ctx)

    # ─── Running obs normalization (opt-in) ──────────────────────────────
    # Always allocated so the local `obs_norm_stats` variable has a fixed
    # type; the per-step update/apply calls are guarded by `obs_norm`.
    # Memory cost when `obs_norm=False`: 2·OBS_DIM·sizeof(gpu_dtype) +
    # 8 bytes — negligible vs the rest of the GPU state.
    var obs_norm_stats = ObsNormStats[OBS_DIM](ctx)
    ctx.synchronize()

    # ─── Initial reset ──────────────────────────────────────────────────
    Env.reset_kernel_gpu[N_ENVS, STATE_SIZE](
        ctx, states_buf, rng_seed=rng_seed_base
    )
    Env.extract_obs_kernel_gpu[N_ENVS, STATE_SIZE, OBS_DIM](
        ctx, states_buf, obs_buf
    )
    if obs_norm:
        obs_norm_stats.update_and_apply[N_ENVS](ctx, obs_buf)
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

    # Total train steps the scheduler is parameterised over. Matches the
    # iteration count produced by `NUM_ENV_STEPS` env-steps under the
    # current `train_steps_per_iter` / `N_ENVS` ratio (UTD≈train_steps_
    # per_iter/N_ENVS). Default reference DMC config has UTD=1 →
    # total = NUM_ENV_STEPS.
    var total_train_steps: Int = NUM_ENV_STEPS * train_steps_per_iter // N_ENVS
    # Host shadow of the device `lr_scale` slot — avoids re-launching the
    # 1-thread write kernel when the scheduler returns the same value
    # (e.g. `ConstantSchedule` returns 1.0 every step).
    var last_lr_scale: Float64 = -1.0

    # ─── Rolling diagnostics accumulators (since last log flush) ─────────
    # Action stats give visibility into policy saturation / collapse —
    # critical for diagnosing HC convergence failures where π_σ shrinks
    # to MIN_STD and actions pin at ±MAX_ACTION. SVE mean tracks MCTS-
    # backed value; improved-policy entropy tracks how concentrated MCTS
    # search has become on a few candidates.
    var diag_abs_action_sum: Float64 = 0.0
    var diag_action_count: Int = 0
    var diag_action_saturated: Int = 0
    var diag_sve_sum: Float64 = 0.0
    var diag_sve_count: Int = 0
    var diag_improved_entropy_sum: Float64 = 0.0
    var diag_improved_entropy_count: Int = 0
    # Loss accumulators (mean across train steps since last log)
    var diag_loss_total_sum: Float64 = 0.0
    var diag_loss_R_sum: Float64 = 0.0
    var diag_loss_P_sum: Float64 = 0.0
    var diag_loss_V_sum: Float64 = 0.0
    var diag_loss_G_sum: Float64 = 0.0
    var diag_loss_count: Int = 0

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

                var sampled = List[Scalar[dtype]](capacity=K_ROOT * ACT_DIM)
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
        elif not use_gpu_mcts:
            # ── Hybrid diagnostic path: CPU MCTS per env ──────────────
            # Uses `agent.select_action()` against the agent's CPU
            # networks. Those networks lag the on-device weights by up
            # to `sync_interval` train steps — same staleness as the
            # converging CPU-stepping baseline driver. The host already
            # has `obs_per_env[e]` from the previous step's GPU obs
            # download, so no extra download is needed.
            for e in range(N_ENVS):
                var sel = agent.select_action(obs_per_env[e], training=True)
                var action = sel[0].copy()
                var root_value = sel[1]
                var sampled = sel[2].copy()
                var improved = sel[3].copy()
                # Diagnostic probe: dump root stats every 12000 env steps
                # for env_id=0 — mirrors the kroot16 baseline's per-3000-
                # batch inspect_root. Lets us check whether MCTS Q-values
                # in the driver path concentrate on a candidate or stay
                # uniform vs the converging CPU-stepping baseline.
                if (
                    e == 0
                    and stats.total_env_steps > 0
                    and stats.total_env_steps % 12000 == 0
                ):
                    agent.inspect_root(
                        tag=String("env_step=") + String(stats.total_env_steps)
                    )
                for d in range(ACT_DIM):
                    host_action[e * ACT_DIM + d] = action[d]
                actions_per_env.append(action^)
                sampled_actions_per_env.append(sampled^)
                improved_policy_per_env.append(improved^)
                root_value_per_env.append(root_value)
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
                Config.ActSpace.N_POLICY_AT_ROOT,
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
                reward_min=agent.reward_min,
                reward_max=agent.reward_max,
                max_action=MAX_ACTION_F,
                min_std=MIN_STD_F,
                std_magnification=STD_MAG_F,
                soft_clamp=SOFT_CLAMP_F,
                init_std=INIT_STD_F,
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
            ctx.enqueue_copy(host_log_prior.unsafe_ptr(), mcts_gpu.log_prior)
            ctx.enqueue_copy(host_node_value.unsafe_ptr(), mcts_gpu.node_value)
            ctx.enqueue_copy(
                host_node_total_visits.unsafe_ptr(), mcts_gpu.total_visits
            )
            ctx.enqueue_copy(host_min_q.unsafe_ptr(), mcts_gpu.min_q)
            ctx.enqueue_copy(host_max_q.unsafe_ptr(), mcts_gpu.max_q)
            ctx.synchronize()

            # MCTS root-visit entropy: average over envs of
            # `−Σᵢ p[i] log p[i]` where p is the normalized visit
            # distribution at the root. Uniform = ln(K_ROOT) (≈ 2.77 for
            # K_ROOT=16) → MCTS found no Q-value signal in this state.
            # Near zero ⇒ MCTS concentrated all visits on one candidate
            # (either healthy convergence or visit-policy collapse).
            var entropy_sum = Float64(0.0)
            for e in range(N_ENVS):
                var ent = Float64(0.0)
                var k_off = e * K_ROOT
                for i in range(K_ROOT):
                    var p = Float64(host_root_visits[k_off + i])
                    if p > 1.0e-12:
                        ent -= p * log(p)
                entropy_sum += ent
            var mean_entropy = entropy_sum / Float64(N_ENVS)
            if _is_finite(mean_entropy):
                stats.last_mcts_visit_entropy = mean_entropy
                stats.mcts_visit_entropy_sum += mean_entropy
                stats.mcts_visit_entropy_n += 1

            # ── 2. Per-env extract: SVE, chosen action, K-candidates ──
            for e in range(N_ENVS):
                # Root candidate visits + values live at node 0:
                #   visit_count[e * NODES * K_ROOT + 0 .. + K_ROOT]
                # K_PAD = K_ROOT by construction.
                var root_slot_off = e * NODES * K_ROOT
                var sum_value = Float64(0.0)
                var sum_visits = 0
                for i in range(K_ROOT):
                    sum_value += Float64(
                        host_node_total_value[root_slot_off + i]
                    )
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
                var sampled = List[Scalar[dtype]](capacity=K_ROOT * ACT_DIM)
                for j in range(K_ROOT * ACT_DIM):
                    sampled.append(host_node_actions[act_off + j])

                # Improved-policy weights — match CPU
                # `_improved_policy_at(0)`: softmax(log_prior + sigma_q),
                # not the visit-count distribution. Required for the
                # full-π policy loss target to match what the agent
                # expects (mcts_sampled.mojo:695-726).
                var nk_base = e * NODES * K_ROOT  # root = node 0
                var ns_idx = e * NODES
                var v_self = Float64(host_node_value[ns_idx])
                var n_total = Float64(host_node_total_visits[ns_idx])
                var mn = Float64(host_min_q[e])
                var mx = Float64(host_max_q[e])

                # v_mix: visit-weighted blend of root V and the K
                # candidates' visited mean_Q's, with log_prior weights.
                var visited_logp_max = Float64(-1.0e18)
                var any_visited = False
                for i in range(K_ROOT):
                    var nva_f = Float64(host_node_visit[nk_base + i])
                    if nva_f > 0.5:
                        var lp_f = Float64(host_log_prior[nk_base + i])
                        if lp_f > visited_logp_max:
                            visited_logp_max = lp_f
                        any_visited = True
                var v_mix = v_self
                if any_visited:
                    var sum_w = Float64(0.0)
                    var weighted_q = Float64(0.0)
                    for i in range(K_ROOT):
                        var nva_f2 = Float64(host_node_visit[nk_base + i])
                        if nva_f2 > 0.5:
                            var lp_f2 = Float64(host_log_prior[nk_base + i])
                            var w = exp(lp_f2 - visited_logp_max)
                            sum_w += w
                            var qa = (
                                Float64(host_node_total_value[nk_base + i])
                                / nva_f2
                            )
                            weighted_q += w * qa
                    if sum_w > 1.0e-12:
                        v_mix = (v_self + n_total * (weighted_q / sum_w)) / (
                            1.0 + n_total
                        )

                # σ scale: (c_visit + max_visit_at_root) · c_scale.
                var max_visit = Float64(0.0)
                for i in range(K_ROOT):
                    var nva_f3 = Float64(host_node_visit[nk_base + i])
                    if nva_f3 > max_visit:
                        max_visit = nva_f3
                var sigma_scale_ = (mcts_c_visit + max_visit) * mcts_c_scale
                var q_range = mx - mn

                # z[i] = log_prior + sigma_scale · normalize(Q_i)
                var z = List[Float64](capacity=K_ROOT)
                var max_z = Float64(-1.0e18)
                for i in range(K_ROOT):
                    var nva_f4 = Float64(host_node_visit[nk_base + i])
                    var qa2: Float64
                    if nva_f4 > 0.5:
                        qa2 = (
                            Float64(host_node_total_value[nk_base + i]) / nva_f4
                        )
                    else:
                        qa2 = v_mix
                    var qn: Float64
                    if q_range > 1.0e-8:
                        qn = (qa2 - mn) / q_range
                    else:
                        qn = qa2
                    var zi = (
                        Float64(host_log_prior[nk_base + i]) + sigma_scale_ * qn
                    )
                    z.append(zi)
                    if zi > max_z:
                        max_z = zi

                # softmax with uniform fallback when sum underflows
                # (matches CPU mcts_sampled.mojo:720-723).
                var improved = List[Scalar[dtype]](capacity=K_ROOT)
                var sum_exp = Float64(0.0)
                var raw_probs = List[Float64](capacity=K_ROOT)
                for i in range(K_ROOT):
                    var ev = exp(z[i] - max_z)
                    raw_probs.append(ev)
                    sum_exp += ev
                if sum_exp <= 1.0e-12:
                    var inv_k = 1.0 / Float64(K_ROOT)
                    for _ in range(K_ROOT):
                        improved.append(Scalar[dtype](inv_k))
                else:
                    for i in range(K_ROOT):
                        improved.append(Scalar[dtype](raw_probs[i] / sum_exp))

                # ── Diagnostic: inspect_root for env 0 every 12000 env
                # steps. Mirrors the CPU `agent.inspect_root` print format
                # so GPU vs CPU MCTS Q/visit distributions are directly
                # comparable. Added 2026-05-16 while debugging GPU MCTS
                # uniform-visit collapse (visit_H ≈ log(K_ROOT) → no
                # candidate preferred, improved-policy targets degenerate).
                if (
                    e == 0
                    and stats.total_env_steps > 0
                    and stats.total_env_steps % 12000 == 0
                ):
                    var H_dbg = Float64(0.0)
                    var max_pi_dbg = Float64(0.0)
                    for i in range(K_ROOT):
                        var p_dbg = Float64(improved[i])
                        if p_dbg > 1.0e-12:
                            H_dbg -= p_dbg * log(p_dbg)
                        if p_dbg > max_pi_dbg:
                            max_pi_dbg = p_dbg
                    var H_unif_dbg = log(Float64(K_ROOT))
                    print(
                        "[inspect_root_gpu env_step=",
                        stats.total_env_steps,
                        " env=0 ] total_visits=",
                        sum_visits,
                        " value_estimate=",
                        root_value_per_env[e],
                    )
                    # min_q / max_q / q_range are the load-bearing scalars
                    # for the host improved-policy reconstruction. If
                    # q_range degenerates (near-zero or huge), every
                    # candidate's normalized Q collapses to a constant and
                    # `improved` reduces to softmax(log_prior). Print
                    # alongside v_mix (visit-weighted Q blend used as the
                    # unvisited-Q fallback) to distinguish:
                    #   • Q-value collapse (mean_v identical, q_range tiny)
                    #   • min_q runaway (q_range huge, mean_v spread small)
                    #   • healthy spread (mean_v differentiated, q_range
                    #     matches max(mean_v) − min(mean_v))
                    print(
                        "       min_q=",
                        mn,
                        " max_q=",
                        mx,
                        " q_range=",
                        mx - mn,
                        " v_self=",
                        v_self,
                        " v_mix=",
                        v_mix,
                        " sigma_scale=",
                        sigma_scale_,
                    )
                    print(
                        "       pi entropy =",
                        H_dbg,
                        "/ log(K)=",
                        H_unif_dbg,
                        "  (ratio=",
                        H_dbg / H_unif_dbg,
                        ")  max_pi=",
                        max_pi_dbg,
                    )
                    for i in range(K_ROOT):
                        var visits_i = Float64(host_node_visit[nk_base + i])
                        var mean_v_i: Float64
                        if visits_i > 0.5:
                            mean_v_i = (
                                Float64(host_node_total_value[nk_base + i])
                                / visits_i
                            )
                        else:
                            mean_v_i = Float64(0.0)
                        var a_str = String("a=[")
                        for d in range(ACT_DIM):
                            a_str += String(sampled[i * ACT_DIM + d])
                            if d + 1 < ACT_DIM:
                                a_str += String(",")
                        a_str += String("]")
                        print(
                            "       i=",
                            i,
                            a_str,
                            " log_prior=",
                            host_log_prior[nk_base + i],
                            " visits=",
                            Int(visits_i),
                            " mean_v=",
                            mean_v_i,
                            " pi=",
                            improved[i],
                        )

                actions_per_env.append(action^)
                sampled_actions_per_env.append(sampled^)
                improved_policy_per_env.append(improved^)

        # Diagnostic accumulators (action saturation, SVE, MCTS entropy).
        # These are cheap host-side scans — same data we just used to
        # build the action / target tensors. Saturation threshold 0.95 ×
        # MAX_ACTION matches CleanRL convention.
        var sat_thresh = 0.95 * MAX_ACTION_F
        for e in range(N_ENVS):
            for d in range(ACT_DIM):
                var a_val = Float64(actions_per_env[e][d])
                var a_abs = a_val if a_val >= 0.0 else -a_val
                diag_abs_action_sum += a_abs
                diag_action_count += 1
                if a_abs >= Float64(sat_thresh):
                    diag_action_saturated += 1
            diag_sve_sum += root_value_per_env[e]
            diag_sve_count += 1
            # Entropy H(π_improved) = -Σ p log p over K_ROOT — only
            # meaningful outside the warmup branch where π_improved is a
            # one-hot. Skip warmup entries.
            if not in_warmup:
                var ent = Float64(0.0)
                for i in range(K_ROOT):
                    var p = Float64(improved_policy_per_env[e][i])
                    if p > 1.0e-12:
                        ent -= p * log(p)
                diag_improved_entropy_sum += ent
                diag_improved_entropy_count += 1

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
        if obs_norm:
            obs_norm_stats.update_and_apply[N_ENVS](ctx, obs_buf)

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
            # Diagnostic: track the max per-step reward we've ever seen
            # across all envs. Answers "did the agent ever visit a high-
            # reward state?" — if buf_reward_max stays ≤ ~0.5 over 100k
            # steps, it's an exploration failure, not a learning failure.
            if reward > stats.buf_reward_max:
                stats.buf_reward_max = reward
            var native_done = host_done[e] > Scalar[dtype](0.5)
            var natively_terminated = host_terminated[e] > Scalar[dtype](0.5)
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
            if obs_norm:
                # Apply without updating stats: extract_obs_kernel_gpu
                # rewrites ALL N_ENVS rows from raw state (not just the
                # reset rows), so a full update_and_apply here would
                # double-count the non-reset envs' already-counted post-
                # step obs. Reset-row obs end up in stats indirectly when
                # the next step's update_and_apply fires.
                obs_norm_stats.apply_only[N_ENVS](ctx, obs_buf)

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
            # UTD loop — run `train_steps_per_iter` gradient updates per
            # training-interval firing. Each iteration of this loop is a
            # full train step with its own batch sample, sync/target/
            # reanalyze gating based on `num_train_calls`. Default 1
            # preserves UTD=1/N_ENVS legacy behavior; set to N_ENVS to
            # match reference UTD=1.0.
            for _ in range(train_steps_per_iter):
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

                # LR-scheduler hook (host, runs once per train step).
                # `lr_scale_at(num_train_calls, total_train_steps)` returns a
                # multiplier in [0, 1]; broadcast to every optimizer-bearing
                # network's `opt_global_state` slot. `ConstantSchedule` is
                # a no-op; only re-write the device slot when the scale
                # actually changed (skips 5 device launches per step on the
                # constant default).
                var lr_scale = SCHEDULER.lr_scale_at(
                    stats.num_train_calls, total_train_steps
                )
                if lr_scale != last_lr_scale:
                    gpu.representation.set_lr_scale(lr_scale, ctx)
                    gpu.dynamics.set_lr_scale(lr_scale, ctx)
                    gpu.prediction.set_lr_scale(lr_scale, ctx)
                    gpu.projector.set_lr_scale(lr_scale, ctx)
                    gpu.predictor.set_lr_scale(lr_scale, ctx)
                    last_lr_scale = lr_scale

                var L_total: Float64
                var L_R: Float64
                var L_P: Float64
                var L_V: Float64
                var L_G: Float64
                # Phase 3d (2026-05-13): SEARCH-only gate dropped — the
                # GPU target-net forward + V-target decode (Phase 3b+3c)
                # now back `train_step_gpu_with_replay` for all modes, so
                # `use_gpu_sampling=True` works under SARSA/MIXED too.
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

                if _is_finite(L_total):
                    diag_loss_total_sum += L_total
                if _is_finite(L_R):
                    diag_loss_R_sum += L_R
                if _is_finite(L_P):
                    diag_loss_P_sum += L_P
                if _is_finite(L_V):
                    diag_loss_V_sum += L_V
                if _is_finite(L_G):
                    diag_loss_G_sum += L_G
                diag_loss_count += 1

                # Diagnostics — `train_step_core.mojo` section 4.5 wrote
                # these scalars to device + section 9 already downloaded
                # them on the same `ctx.synchronize()`. Cheap host read.
                var z_var_v = Float64(gpu.z_var_host[0])
                var v_pred_var_v = Float64(gpu.v_pred_var_host[0])
                if _is_finite(z_var_v):
                    stats.last_z_var = z_var_v
                    stats.z_var_sum += z_var_v
                    stats.z_var_n += 1
                if _is_finite(v_pred_var_v):
                    stats.last_v_pred_var = v_pred_var_v
                    stats.v_pred_var_sum += v_pred_var_v
                    stats.v_pred_var_n += 1

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
                    # Phase 3: mirror fresh CPU targets onto GPU.
                    gpu.upload_targets_from(agent.state, ctx)
                if (
                    stats.num_train_calls >= reanalyze_warmup
                    and stats.num_train_calls % reanalyze_interval == 0
                    and agent.state.is_ready()
                ):
                    # ── GPU reanalyze (Phase 4) ─────────────────────────
                    # Replaces the CPU `agent.reanalyze` per-sample MCTS
                    # loop with a single GPU MCTS launch over
                    # REANALYZE_SAMPLES sampled buffer windows. Same
                    # writeback semantics as the CPU path: overwrites
                    # `mcts_policies` (simple-best chosen action),
                    # `mcts_sampled_actions` (K root candidate vectors),
                    # `mcts_improved_policy` (full-π weights),
                    # `mcts_values` (SVE), `step_at_write` (freshness).
                    var buf_size = agent.state.buffer.size
                    if buf_size > 0:
                        var buf_ptr = agent.state.buffer.ptr
                        var oldest = (buf_ptr - buf_size + CAP) % CAP
                        var sampled_idx = List[Int](capacity=REANALYZE_SAMPLES)
                        for s in range(REANALYZE_SAMPLES):
                            var rand_offset = Int(
                                random_float64() * Float64(buf_size)
                            )
                            if rand_offset >= buf_size:
                                rand_offset = buf_size - 1
                            if rand_offset < 0:
                                rand_offset = 0
                            var idx = (oldest + rand_offset) % CAP
                            sampled_idx.append(idx)
                            for d in range(OBS):
                                host_reanalyze_obs[
                                    s * OBS + d
                                ] = agent.state.buffer.obs[idx * OBS + d]

                        ctx.enqueue_copy(
                            reanalyze_obs_buf,
                            host_reanalyze_obs.unsafe_ptr(),
                        )

                        # GPU MCTS against TARGET networks.
                        # `deterministic=True` matches the CPU
                        # reanalyze: argmax-visit pick for the simple-
                        # best chosen-action target.
                        run_sampled_gumbel_search_gpu[
                            REANALYZE_SAMPLES,
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
                            Config.ActSpace.N_POLICY_AT_ROOT,
                        ](
                            ctx,
                            reanalyze_mcts_gpu,
                            reanalyze_obs_buf,
                            gpu.representation_target,
                            gpu.dynamics_target,
                            gpu.prediction_target,
                            reanalyze_workspace,
                            v_min=agent.v_min,
                            v_max=agent.v_max,
                            reward_min=agent.reward_min,
                            reward_max=agent.reward_max,
                            max_action=MAX_ACTION_F,
                            min_std=MIN_STD_F,
                            std_magnification=STD_MAG_F,
                            soft_clamp=SOFT_CLAMP_F,
                            init_std=INIT_STD_F,
                            c_visit=mcts_c_visit,
                            c_scale=mcts_c_scale,
                            gamma=agent.gamma,
                            deterministic=True,
                            rng_seed=reanalyze_seed,
                        )
                        reanalyze_seed += UInt32(1)

                        ctx.enqueue_copy(
                            host_reanalyze_chosen.unsafe_ptr(),
                            reanalyze_mcts_gpu.chosen_actions,
                        )
                        ctx.enqueue_copy(
                            host_reanalyze_node_visit.unsafe_ptr(),
                            reanalyze_mcts_gpu.visit_count,
                        )
                        ctx.enqueue_copy(
                            host_reanalyze_node_total_value.unsafe_ptr(),
                            reanalyze_mcts_gpu.total_value,
                        )
                        ctx.enqueue_copy(
                            host_reanalyze_node_actions.unsafe_ptr(),
                            reanalyze_mcts_gpu.actions,
                        )
                        ctx.enqueue_copy(
                            host_reanalyze_log_prior.unsafe_ptr(),
                            reanalyze_mcts_gpu.log_prior,
                        )
                        ctx.enqueue_copy(
                            host_reanalyze_node_value.unsafe_ptr(),
                            reanalyze_mcts_gpu.node_value,
                        )
                        ctx.enqueue_copy(
                            host_reanalyze_node_total_visits.unsafe_ptr(),
                            reanalyze_mcts_gpu.total_visits,
                        )
                        ctx.enqueue_copy(
                            host_reanalyze_min_q.unsafe_ptr(),
                            reanalyze_mcts_gpu.min_q,
                        )
                        ctx.enqueue_copy(
                            host_reanalyze_max_q.unsafe_ptr(),
                            reanalyze_mcts_gpu.max_q,
                        )
                        ctx.synchronize()

                        # ── Per-sample extract + writeback ─────────────
                        # Same math as the acting extract block (lines
                        # 513-643): SVE from root-slot visits/values,
                        # improved-policy = softmax(log_prior + σ·Q).
                        for s in range(REANALYZE_SAMPLES):
                            var idx = sampled_idx[s]
                            var root_slot_off = s * NODES * K_ROOT
                            var ns_idx = s * NODES

                            # SVE: Σ total_value(root, i) / Σ visits(root, i)
                            var sum_value = Float64(0.0)
                            var sum_visits = 0
                            for i in range(K_ROOT):
                                sum_value += Float64(
                                    host_reanalyze_node_total_value[
                                        root_slot_off + i
                                    ]
                                )
                                sum_visits += Int(
                                    Float64(
                                        host_reanalyze_node_visit[
                                            root_slot_off + i
                                        ]
                                    )
                                )
                            var sve = compute_sve(sum_value, sum_visits)

                            # Improved-policy softmax — same recipe as
                            # acting (mcts_sampled.mojo:695-726).
                            var v_self = Float64(
                                host_reanalyze_node_value[ns_idx]
                            )
                            var n_total = Float64(
                                host_reanalyze_node_total_visits[ns_idx]
                            )
                            var mn = Float64(host_reanalyze_min_q[s])
                            var mx = Float64(host_reanalyze_max_q[s])

                            var visited_logp_max = Float64(-1.0e18)
                            var any_visited = False
                            for i in range(K_ROOT):
                                var nva = Float64(
                                    host_reanalyze_node_visit[root_slot_off + i]
                                )
                                if nva > 0.5:
                                    var lp = Float64(
                                        host_reanalyze_log_prior[
                                            root_slot_off + i
                                        ]
                                    )
                                    if lp > visited_logp_max:
                                        visited_logp_max = lp
                                    any_visited = True
                            var v_mix = v_self
                            if any_visited:
                                var sum_w = Float64(0.0)
                                var weighted_q = Float64(0.0)
                                for i in range(K_ROOT):
                                    var nva2 = Float64(
                                        host_reanalyze_node_visit[
                                            root_slot_off + i
                                        ]
                                    )
                                    if nva2 > 0.5:
                                        var lp2 = Float64(
                                            host_reanalyze_log_prior[
                                                root_slot_off + i
                                            ]
                                        )
                                        var w = exp(lp2 - visited_logp_max)
                                        sum_w += w
                                        var qa = (
                                            Float64(
                                                host_reanalyze_node_total_value[
                                                    root_slot_off + i
                                                ]
                                            )
                                            / nva2
                                        )
                                        weighted_q += w * qa
                                if sum_w > 1.0e-12:
                                    v_mix = (
                                        v_self + n_total * (weighted_q / sum_w)
                                    ) / (1.0 + n_total)

                            var max_visit = Float64(0.0)
                            for i in range(K_ROOT):
                                var nva3 = Float64(
                                    host_reanalyze_node_visit[root_slot_off + i]
                                )
                                if nva3 > max_visit:
                                    max_visit = nva3
                            var sigma_scale_ = (
                                mcts_c_visit + max_visit
                            ) * mcts_c_scale
                            var q_range = mx - mn

                            var z = List[Float64](capacity=K_ROOT)
                            var max_z = Float64(-1.0e18)
                            for i in range(K_ROOT):
                                var nva4 = Float64(
                                    host_reanalyze_node_visit[root_slot_off + i]
                                )
                                var qa2: Float64
                                if nva4 > 0.5:
                                    qa2 = (
                                        Float64(
                                            host_reanalyze_node_total_value[
                                                root_slot_off + i
                                            ]
                                        )
                                        / nva4
                                    )
                                else:
                                    qa2 = v_mix
                                var qn: Float64
                                if q_range > 1.0e-8:
                                    qn = (qa2 - mn) / q_range
                                else:
                                    qn = qa2
                                var zi = (
                                    Float64(
                                        host_reanalyze_log_prior[
                                            root_slot_off + i
                                        ]
                                    )
                                    + sigma_scale_ * qn
                                )
                                z.append(zi)
                                if zi > max_z:
                                    max_z = zi

                            var sum_exp = Float64(0.0)
                            var raw_probs = List[Float64](capacity=K_ROOT)
                            for i in range(K_ROOT):
                                var ev = exp(z[i] - max_z)
                                raw_probs.append(ev)
                                sum_exp += ev

                            # Writeback — chosen action, K candidates,
                            # improved-policy, SVE, freshness stamp.
                            for d in range(ACT_DIM):
                                agent.state.mcts_policies[
                                    idx * ACT_DIM + d
                                ] = host_reanalyze_chosen[s * ACT_DIM + d]
                            var act_off = s * NODES * K_ROOT * ACT_DIM
                            for j in range(K_ROOT * ACT_DIM):
                                agent.state.mcts_sampled_actions[
                                    idx * K_ROOT * ACT_DIM + j
                                ] = host_reanalyze_node_actions[act_off + j]
                            if sum_exp <= 1.0e-12:
                                var inv_k = Scalar[dtype](1.0 / Float64(K_ROOT))
                                for i in range(K_ROOT):
                                    agent.state.mcts_improved_policy[
                                        idx * K_ROOT + i
                                    ] = inv_k
                            else:
                                for i in range(K_ROOT):
                                    agent.state.mcts_improved_policy[
                                        idx * K_ROOT + i
                                    ] = Scalar[dtype](raw_probs[i] / sum_exp)
                            agent.state.mcts_values[idx] = Scalar[dtype](sve)
                            agent.state.step_at_write[idx] = Scalar[
                                DType.uint32
                            ](agent.train_step_count)

        step_seed += 1

        # ── 7. Logging ──────────────────────────────────────────────────
        var logger_active = Bool(logger) and logger.value()[].is_active()
        if (
            (verbose or logger_active)
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
            var recent_mean = _mean(recent)

            # Rolling diagnostics — mean over this interval.
            var mean_abs_action = Float64(0.0)
            if diag_action_count > 0:
                mean_abs_action = diag_abs_action_sum / Float64(
                    diag_action_count
                )
            var frac_saturated = Float64(0.0)
            if diag_action_count > 0:
                frac_saturated = Float64(diag_action_saturated) / Float64(
                    diag_action_count
                )
            var mean_sve = Float64(0.0)
            if diag_sve_count > 0:
                mean_sve = diag_sve_sum / Float64(diag_sve_count)
            var mean_improved_entropy = Float64(0.0)
            if diag_improved_entropy_count > 0:
                mean_improved_entropy = diag_improved_entropy_sum / Float64(
                    diag_improved_entropy_count
                )
            var mean_L_total = Float64(0.0)
            var mean_L_R = Float64(0.0)
            var mean_L_P = Float64(0.0)
            var mean_L_V = Float64(0.0)
            var mean_L_G = Float64(0.0)
            if diag_loss_count > 0:
                var inv = 1.0 / Float64(diag_loss_count)
                mean_L_total = diag_loss_total_sum * inv
                mean_L_R = diag_loss_R_sum * inv
                mean_L_P = diag_loss_P_sum * inv
                mean_L_V = diag_loss_V_sum * inv
                mean_L_G = diag_loss_G_sum * inv

            # Diagnostics windowed means (2026-05-14: SimSiam collapse /
            # value-head collapse / MCTS no-signal hunt). All four answer
            # different "is the bug here?" questions, so we always log
            # them, even when the underlying counter never advanced
            # (default 0 is itself diagnostic).
            var mean_z_var = Float64(0.0)
            if stats.z_var_n > 0:
                mean_z_var = stats.z_var_sum / Float64(stats.z_var_n)
            var mean_v_pred_var = Float64(0.0)
            if stats.v_pred_var_n > 0:
                mean_v_pred_var = stats.v_pred_var_sum / Float64(
                    stats.v_pred_var_n
                )
            var mean_visit_entropy = Float64(0.0)
            if stats.mcts_visit_entropy_n > 0:
                mean_visit_entropy = stats.mcts_visit_entropy_sum / Float64(
                    stats.mcts_visit_entropy_n
                )

            if verbose:
                print(
                    "[step ",
                    stats.total_env_steps,
                    " ep=",
                    n_eps,
                    " train=",
                    stats.num_train_calls,
                    " syncs=",
                    stats.num_gpu_syncs,
                    " wall=",
                    wall_s,
                    "s",
                    "] recent_mean=",
                    recent_mean,
                    "  best=",
                    stats.best_episode_return,
                    "  L=(R",
                    stats.last_L_R,
                    ", P",
                    stats.last_L_P,
                    ", V",
                    stats.last_L_V,
                    ", G",
                    stats.last_L_G,
                    ")",
                    "  diag=(z_var",
                    mean_z_var,
                    ", v_pred_var",
                    mean_v_pred_var,
                    ", visit_H",
                    mean_visit_entropy,
                    ", r_max",
                    stats.buf_reward_max,
                    ")",
                )

            if logger_active:
                var step = stats.total_env_steps

                # ── Episode Reward group ──────────────────────────────
                # Skip while no episode has finished — `best_episode_return`
                # is still the -1e308 sentinel and `recent_mean` is 0.0,
                # both of which crush the chart's y-axis scale.
                if n_eps > 0:
                    var last_ep_reward = recent[len(recent) - 1]
                    logger.value()[].log_scalar("avg_reward", recent_mean, step)
                    logger.value()[].log_scalar(
                        "episode_reward", last_ep_reward, step
                    )
                    logger.value()[].log_scalar(
                        "best_reward", stats.best_episode_return, step
                    )
                # ── Training Progress group ───────────────────────────
                logger.value()[].log_scalar("episodes", Float64(n_eps), step)
                logger.value()[].log_scalar(
                    "train_steps", Float64(stats.num_train_calls), step
                )
                # ── Loss group / specialized loss groups ──────────────
                # Rolling means over the interval — smoother than the
                # last-call snapshot used in the print line.
                if diag_loss_count > 0:
                    logger.value()[].log_scalar("loss", mean_L_total, step)
                    # World Model Losses group
                    logger.value()[].log_scalar("reward_loss", mean_L_R, step)
                    # SimSiam consistency between predicted z and the
                    # target-encoder z fits the "obs prediction" slot
                    # semantically (Dreamer/MuZero analogue).
                    logger.value()[].log_scalar("obs_loss", mean_L_G, step)
                    # Critic Loss group
                    logger.value()[].log_scalar("value_loss", mean_L_V, step)
                    # Policy Loss group
                    logger.value()[].log_scalar("policy_loss", mean_L_P, step)
                # ── Entropy group ─────────────────────────────────────
                # Improved-policy entropy = entropy of the MCTS-derived
                # target distribution over K root candidates.
                logger.value()[].log_scalar(
                    "entropy", mean_improved_entropy, step
                )
                # ── Exploration group ─────────────────────────────────
                # EZ-V2's `temperature` plays the same MCTS-sampling
                # role as ε-greedy's `explore_rate` for DQN.
                logger.value()[].log_scalar(
                    "explore_rate", agent.temperature, step
                )
                # ── TD Targets / Value Head group ─────────────────────
                # MCTS-derived SVE is the value target the V head fits
                # against — natural fit for the TD-targets group.
                logger.value()[].log_scalar("value_target_mean", mean_sve, step)
                # ── Extra (no exact KNOWN_GROUPS match) ───────────────
                # `best_reward` is logged with the Episode Reward group
                # above (gated on n_eps > 0). `action_*` track policy
                # saturation (HC's MIN_STD collapse symptom).
                # `buffer_size`, `gpu_syncs`, `buffer_uploads`, `wall_s`
                # are runtime telemetry.
                logger.value()[].log_scalar(
                    "action_abs_mean", mean_abs_action, step
                )
                logger.value()[].log_scalar(
                    "action_saturated_frac", frac_saturated, step
                )
                logger.value()[].log_scalar(
                    "buffer_size",
                    Float64(agent.state.buffer.size),
                    step,
                )
                logger.value()[].log_scalar(
                    "gpu_syncs", Float64(stats.num_gpu_syncs), step
                )
                logger.value()[].log_scalar(
                    "buffer_uploads",
                    Float64(stats.num_buffer_uploads),
                    step,
                )
                logger.value()[].log_scalar("wall_s", wall_s, step)

                # ── Collapse-hunt diagnostics (2026-05-14) ────────────
                # Decisive metrics for the four dominant failure modes
                # I haven't been able to rule out via hyperparam tuning.
                # See `train_step_core.mojo:4.5` and the MCTS-entropy /
                # buf_reward_max hooks earlier in this driver.
                logger.value()[].log_scalar("z_var", mean_z_var, step)
                logger.value()[].log_scalar("v_pred_var", mean_v_pred_var, step)
                logger.value()[].log_scalar(
                    "mcts_visit_entropy", mean_visit_entropy, step
                )
                logger.value()[].log_scalar(
                    "buf_reward_max", stats.buf_reward_max, step
                )

            # Reset interval accumulators.
            diag_abs_action_sum = 0.0
            diag_action_count = 0
            diag_action_saturated = 0
            diag_sve_sum = 0.0
            diag_sve_count = 0
            diag_improved_entropy_sum = 0.0
            diag_improved_entropy_count = 0
            diag_loss_total_sum = 0.0
            diag_loss_R_sum = 0.0
            diag_loss_P_sum = 0.0
            diag_loss_V_sum = 0.0
            diag_loss_G_sum = 0.0
            diag_loss_count = 0
            # Collapse-hunt diagnostics: reset the per-window aggregates
            # (kept on `stats` since these are per-train-step / per-MCTS,
            # not per-env-step like the others above). `buf_reward_max`
            # is intentionally NOT reset — it's a monotonic high-water
            # mark across the entire run.
            stats.z_var_sum = 0.0
            stats.z_var_n = 0
            stats.v_pred_var_sum = 0.0
            stats.v_pred_var_n = 0
            stats.mcts_visit_entropy_sum = 0.0
            stats.mcts_visit_entropy_n = 0

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

    if Bool(logger) and logger.value()[].is_active():
        var window = 30
        var n_eps = len(stats.ep_returns)
        var recent = List[Float64]()
        var start_i = n_eps - window if n_eps > window else 0
        for i in range(start_i, n_eps):
            recent.append(stats.ep_returns[i])
        logger.value()[].log_scalar(
            "avg_reward", _mean(recent), stats.total_env_steps
        )
        logger.value()[].log_scalar(
            "best_reward",
            stats.best_episode_return,
            stats.total_env_steps,
        )
        logger.value()[].log_scalar(
            "wall_s", stats.wall_time_s, stats.total_env_steps
        )
        logger.value()[].flush()

    return stats^
