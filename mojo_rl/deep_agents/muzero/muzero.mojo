"""MuZero Agent — Model-based RL with learned model and MCTS planning.

Learns three networks from environment interaction:
  - Representation h(o) -> s^0: Encode observation to hidden state
  - Dynamics g(s, a) -> (r, s'): Predict next hidden state and reward
  - Prediction f(s) -> (p, v): Predict policy and value from any hidden state

Uses MCTS with the learned model for action selection. Trains via K-step
unrolled forward/backward through all three networks with policy, value,
and reward cross-entropy losses.

Reference: Schrittwieser et al., 2020 — Mastering Atari, Go, Chess and
Shogi by Planning with a Learned Model (Nature)
"""

from std.math import exp, log, sqrt
from std.random import random_float64
from std.time import perf_counter_ns
from std.memory import alloc, memset
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, LinearMish, Sequential, Parallel
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.training import Network, NetworkState, GPUNetworkState
from mojo_rl.nn.training.scheduler import OneCycleSchedule
from mojo_rl.deep_agents.core.utils import (
    print_progress_bar,
    clear_progress_bar,
)
from mojo_rl.nn.checkpoint import (
    write_checkpoint_header,
    write_metadata_section,
    save_checkpoint_file,
    read_checkpoint_file,
    parse_checkpoint_header,
    read_metadata_section,
    get_metadata_value,
)
from mojo_rl.core import (
    TrainingMetrics,
    BoxDiscreteActionEnv,
    GPUDiscreteEnv,
    GPUTwoPlayerDiscreteEnv,
    TwoPlayerDiscreteEnv,
)
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.deep_agents.core.kernels import (
    accumulate_rewards_kernel,
    increment_steps_kernel,
    log_and_reset_completed_kernel,
    uniform_random_discrete_actions_kernel,
    uniform_random_legal_actions_kernel,
)
from mojo_rl.deep_agents.core.replay.sequence_replay_buffer import (
    SequenceReplayBuffer,
)
from .configs import MuZeroConfig, MuZeroMLPConfig
from .evaluators import Evaluator, GPUEvaluator, RandomOpponent
from .state import MuZeroCPUState, MuZeroGPUState
from mojo_rl.planners.tree_search.mcts_gpu import (
    TPB,
    MAX_DEPTH,
)
from mojo_rl.planners.tree_search import GenericGPUMCTS, GenericCPUMCTS, NoNoise
from mojo_rl.planners.tree_search.mcts_gpu_gumbel_orchestrator import (
    GumbelGPUMCTS,
)
from .gpu_trait_adapters import (
    MuZeroRepGPU,
    MuZeroDynGPU,
    MuZeroPredGPU,
    MuZeroRepCPU,
    MuZeroDynCPU,
    MuZeroPredCPU,
)
from .utils import (
    scalar_transform,
    inverse_scalar_transform,
    encode_categorical,
    cross_entropy_with_softmax,
    softmax_inplace,
)
from .kernels import (
    scale_hidden_kernel,
    ce_policy_grad_kernel,
    ce_value_grad_kernel,
    ce_reward_grad_kernel,
    two_hot_encode_kernel,
    build_dyn_input_kernel,
    extract_hidden_kernel,
    extract_hidden_grad_kernel,
    add_scaled_kernel,
    scale_kernel,
    copy_kernel,
    set_hidden_grad_for_dyn_kernel,
    store_mcts_targets_kernel,
    sample_seq_with_targets_kernel,
    sample_seq_with_targets_priority_kernel,
    nstep_value_targets_kernel,
    scalar_transform_kernel,
    to_play_from_episode_step_kernel,
    decode_value_dist_kernel,
    action_histogram_kernel,
    scalar_to_onehot_actions_kernel,
    action_switch_kernel,
)


# =============================================================================
# MuZero Agent
# =============================================================================


struct GenericMuZeroAgent[
    Config: MuZeroConfig, n_envs: Int = 64, L: Logger = NoOpLogger
](Movable):
    """MuZero agent for discrete action environments.

    Combines learned representation/dynamics/prediction networks with
    MCTS planning for action selection. Trains via K-step unrolled
    cross-entropy losses on policy, value, and reward targets.

    Parameters:
        Config: MuZeroConfig trait providing all dimensions, network types,
                and training hyperparameters.
        n_envs: Number of parallel GPU environments (default: 64).
        L: Logger type for diagnostic logging (default: NoOpLogger — zero overhead).
    """

    # ── State type alias ──────────────────────────────────────────────────
    comptime StateType = MuZeroCPUState[
        Self.Config, Self.Config.buffer_capacity
    ]

    # ── Network type shortcuts ────────────────────────────────────────────
    comptime RepNet = Network[Self.Config.RepModel, Self.Config.OptType]
    comptime DynNet = Network[Self.Config.DynModel, Self.Config.OptType]
    comptime PredNet = Network[Self.Config.PredModel, Self.Config.OptType]

    # ── Core state ────────────────────────────────────────────────────────
    var state: Self.StateType

    # MCTS is no longer a struct field — every CPU call site (self-play,
    # eval, reanalyze) goes through ``GenericCPUMCTS`` via
    # ``_mcts_search_visits_cpu``, which constructs a fresh planner per
    # search. GPU MCTS state still lives in ``MuZeroGPUState``.

    # Hyperparameters
    var gamma: Float64
    var weight_decay: Float64
    var v_min: Float64
    var v_max: Float64
    var temperature: Float64
    var temperature_decay_steps: Int
    var max_grad_norm: Float64
    # Polyak τ for prediction/representation target networks (E4); used
    # only when use_reanalyze is True. τ=1.0 → hard sync each step (no
    # decoupling); τ=0.0 → frozen target. Typical: 0.005–0.01.
    var target_tau: Float64
    # Prioritized Experience Replay (Phase H13 / muzero-general PER):
    # priority = (|TD-error| + per_eps)^per_alpha, sampled with probability
    # ∝ priority, IS-corrected with weight w = (N·P)^(-beta) / max_w. Beta
    # is linearly annealed from per_beta_init → 1.0 over training steps
    # (passed in via _per_beta(progress) at sample time).
    var per_alpha: Float64
    var per_beta_init: Float64
    var per_eps: Float64

    # Step counters
    var total_steps: Int
    var train_step_count: Int

    # Logger (NoOpLogger = zero-overhead default)
    var logger: Optional[UnsafePointer[Self.L, MutAnyOrigin]]
    var diag_every: Int

    # Episode data storage for MCTS targets
    var _episode_obs: List[List[Scalar[dtype]]]
    var _episode_actions: List[Int]
    var _episode_rewards: List[Float64]
    var _episode_policies: List[InlineArray[Float64, Self.Config.action_dim]]
    var _episode_values: List[Float64]
    # Per-step player-to-move (0 for single-player; 0/1 for two-player).
    # Used for muzero-general-style sign flipping during n-step value
    # target computation (replay_buffer.py:242-259).
    var _episode_to_play: List[Int]

    # ══════════════════════════════════════════════════════════════════════
    # Constructors
    # ══════════════════════════════════════════════════════════════════════

    def __init__(
        out self,
        gamma: Float64 = 0.997,
        weight_decay: Float64 = 1e-4,
        v_min: Float64 = -50.0,
        v_max: Float64 = 50.0,
        temperature: Float64 = 1.0,
        temperature_decay_steps: Int = 100000,
        max_grad_norm: Float64 = 10.0,
        target_tau: Float64 = 0.01,
        pred_head_input_dim: Int = 0,
        per_alpha: Float64 = 0.5,
        per_beta_init: Float64 = 0.4,
        per_eps: Float64 = 1e-6,
    ):
        """Initialize MuZero agent with all networks and MCTS engine.

        Args:
            gamma: Discount factor (default: 0.997).
            weight_decay: L2 regularization coefficient (default: 1e-4).
            v_min: Minimum value support for categorical encoding.
            v_max: Maximum value support for categorical encoding.
            temperature: Initial action selection temperature (default: 1.0).
            temperature_decay_steps: Steps to decay temperature to 0.
            max_grad_norm: Maximum gradient norm for clipping (default: 10.0).
            target_tau: Target network update rate (default: 0.01).
            pred_head_input_dim: Input dimension for the prediction head (default: 0).
            per_alpha: Prioritized experience replay alpha (default: 0.5).
            per_beta_init: Prioritized experience replay beta initial value (default: 0.4).
            per_eps: Prioritized experience replay epsilon (default: 1e-6).
        """
        self.state = Self.StateType()
        # Zero pred policy + value head Linear params (W and b) to fix the
        # "untrained MCTS amplifies init bias" failure mode. Kaiming init
        # produces logits ~±2.5 (softmax 97/3) and decoded value ~-450 on
        # CartPole; zero-init the heads gives uniform softmax and value 0.
        # The hidden LinearMish layer keeps Kaiming so input differentiation
        # works once pred is trained. See docs/MUZERO_AUDIT.md Bug-A/B
        # (2026-05-04 init-state diagnostic).
        if pred_head_input_dim > 0:
            comptime ACT_C = Self.Config.action_dim
            comptime BINS_C = Self.Config.num_bins
            var head_size = pred_head_input_dim * (ACT_C + BINS_C) + (
                ACT_C + BINS_C
            )
            comptime PRED_PS = Self.Config.PredModel.PARAM_SIZE
            var start = PRED_PS - head_size
            var pred_params = self.state.prediction.params
            for i in range(start, PRED_PS):
                (pred_params + i)[] = Scalar[dtype](0)
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.v_min = v_min
        self.v_max = v_max
        self.temperature = temperature
        self.temperature_decay_steps = temperature_decay_steps
        self.max_grad_norm = max_grad_norm
        self.target_tau = target_tau
        self.per_alpha = per_alpha
        self.per_beta_init = per_beta_init
        self.per_eps = per_eps
        self.total_steps = 0
        self.train_step_count = 0
        self.logger = None
        self.diag_every = 0
        self._episode_obs = List[List[Scalar[dtype]]]()
        self._episode_actions = List[Int]()
        self._episode_rewards = List[Float64]()
        self._episode_policies = List[
            InlineArray[Float64, Self.Config.action_dim]
        ]()
        self._episode_values = List[Float64]()
        self._episode_to_play = List[Int]()

    def __init__(out self, *, deinit take: Self):
        """Move constructor — transfer ownership of all fields."""
        self.state = take.state^
        self.gamma = take.gamma
        self.weight_decay = take.weight_decay
        self.v_min = take.v_min
        self.v_max = take.v_max
        self.temperature = take.temperature
        self.temperature_decay_steps = take.temperature_decay_steps
        self.max_grad_norm = take.max_grad_norm
        self.target_tau = take.target_tau
        self.per_alpha = take.per_alpha
        self.per_beta_init = take.per_beta_init
        self.per_eps = take.per_eps
        self.total_steps = take.total_steps
        self.train_step_count = take.train_step_count
        self.logger = take.logger
        self.diag_every = take.diag_every
        self._episode_obs = take._episode_obs^
        self._episode_actions = take._episode_actions^
        self._episode_rewards = take._episode_rewards^
        self._episode_policies = take._episode_policies^
        self._episode_values = take._episode_values^
        self._episode_to_play = take._episode_to_play^

    # ══════════════════════════════════════════════════════════════════════
    # Episode Management
    # ══════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════
    # Prioritized Experience Replay (Phase H13)
    # ══════════════════════════════════════════════════════════════════════

    @always_inline
    def _per_beta(self, progress: Float64) -> Float64:
        """Linearly anneal IS-correction beta from per_beta_init → 1.0.

        Args:
            progress: Training progress in [0, 1] (e.g. train_step/num_steps).
        """
        var p = progress
        if p < 0.0:
            p = 0.0
        if p > 1.0:
            p = 1.0
        return self.per_beta_init + (1.0 - self.per_beta_init) * p

    def _per_record_new_transitions[
        N_ENVS_P: Int = 64,
        PER_ENV_CAP_P: Int = 1000,
    ](
        mut self,
        mut gpu: MuZeroGPUState[Self.Config, N_ENVS_P, PER_ENV_CAP_P],
        slot: Int,
    ):
        """Set tree priority = max_priority for each env's new transition.

        Called immediately after gpu.replay.store_with_termination(). All
        envs step in lockstep so they share the same write slot. New
        transitions get the max priority seen so far so they're guaranteed
        to be sampled at least once before being prioritized down.
        """
        comptime PER_ENV_CAP = PER_ENV_CAP_P
        var p = gpu.per_max_priority
        for e in range(N_ENVS_P):
            gpu.per_tree.update(e * PER_ENV_CAP + slot, p)

    def _per_sample_indices[
        N_ENVS_P: Int = 64,
        PER_ENV_CAP_P: Int = 1000,
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu: MuZeroGPUState[Self.Config, N_ENVS_P, PER_ENV_CAP_P],
        progress: Float64,
    ) raises:
        """Stratified sum-tree sampling — fills (env, start, weight) host
        buffers and uploads to GPU for the priority sample kernel.
        """
        comptime BATCH = Self.Config.batch_size
        comptime PER_ENV_CAP = PER_ENV_CAP_P

        # IS-weight formula from muzero-general (replay_buffer.py:113-118):
        #   w_b_raw = 1 / (N · P_b)        # beta=1 (no annealing)
        #   w_b = w_b_raw / max(w_b_raw over batch)
        # Normalize by MAX OVER THE BATCH, not max-over-tree. This is
        # algebraically simpler and avoids needing tree-wide min_priority
        # (which is broken in our SumTree because we use update() not add()
        # so its `size` field stays 0 and its min_priority falls back to 1.0).
        # Note: muzero-general uses beta=1 always; we ignore the annealing
        # progress kwarg to match. _per_beta is kept for completeness.
        _ = progress  # currently unused; beta=1 fixed
        var total = Float64(gpu.per_tree.total_sum())
        # n = number of filled (env, slot) positions in the circular buffer.
        var n_filled = gpu.replay.size * N_ENVS_P
        var n = Float64(n_filled if n_filled > 0 else 1)

        if total <= 0.0:
            # Empty/uniform tree — fall back to round-robin sample positions
            # near the head of the buffer with weight 1.0. Should only fire
            # before the first transition is stored.
            for b in range(BATCH):
                gpu.per_sampled_envs_host[b] = Int32(b % N_ENVS_P)
                gpu.per_sampled_starts_host[b] = Int32(0)
                gpu.per_is_weights_host[b] = Scalar[dtype](1.0)
        else:
            var segment = total / Float64(BATCH)
            # Pass 1: sample, compute raw weights w_b = 1/(N*P_b)
            var raw_w = List[Float64](capacity=BATCH)
            for b in range(BATCH):
                var lo = segment * Float64(b)
                var hi = segment * Float64(b + 1)
                var target = lo + random_float64() * (hi - lo)
                var idx = gpu.per_tree.sample(Scalar[dtype](target))
                var env_idx = idx // PER_ENV_CAP
                var slot = idx % PER_ENV_CAP
                gpu.per_sampled_envs_host[b] = Int32(env_idx)
                gpu.per_sampled_starts_host[b] = Int32(slot)
                var p_b = Float64(gpu.per_tree.get(idx))
                if p_b <= 0.0:
                    p_b = total / n  # fallback to uniform prob slot
                var prob = p_b / total
                raw_w.append(1.0 / (n * prob))
            # Pass 2: normalize by max within batch
            var max_raw = raw_w[0]
            for b in range(1, BATCH):
                if raw_w[b] > max_raw:
                    max_raw = raw_w[b]
            if max_raw <= 0.0:
                max_raw = 1.0
            for b in range(BATCH):
                gpu.per_is_weights_host[b] = Scalar[dtype](raw_w[b] / max_raw)

        ctx.enqueue_copy(gpu.per_sampled_envs_buf, gpu.per_sampled_envs_host)
        ctx.enqueue_copy(
            gpu.per_sampled_starts_buf, gpu.per_sampled_starts_host
        )
        ctx.enqueue_copy(gpu.per_is_weights_buf, gpu.per_is_weights_host)

    def _per_update_priorities[
        N_ENVS_P: Int = 64,
        PER_ENV_CAP_P: Int = 1000,
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu: MuZeroGPUState[Self.Config, N_ENVS_P, PER_ENV_CAP_P],
    ) raises:
        """Recompute per-sample priorities from value-head TD error and
        update the sum-tree. Called after the K-step backward.

        Priority = (|h(pred_value_k=0) − value_target_k=0| + ε)^α
        where pred is in raw scalar space (from reanalyze) and value_target
        is in encoded h-space, so we apply h() to pred before subtracting.
        Mirrors muzero-general's `priorities = |target - pred|^α` (their pred
        is already in scalar space; we apply h to bring both sides to the
        same representation).
        """
        comptime BATCH = Self.Config.batch_size
        comptime PER_ENV_CAP = PER_ENV_CAP_P
        var EPS_H = Float64(0.001)

        # Download value targets (encoded) and predicted values (raw scalar).
        ctx.enqueue_copy(gpu.value_targets_host, gpu.value_targets_buf)
        ctx.enqueue_copy(gpu.batch_values_host, gpu.batch_values_buf)
        ctx.synchronize()

        var max_p_seen = Float64(gpu.per_max_priority)
        for b in range(BATCH):
            var env_idx = Int(gpu.per_sampled_envs_host[b])
            var slot = Int(gpu.per_sampled_starts_host[b])
            # Reference (muzero-general/trainer.py:204) computes priority in
            # RAW scalar space: |pred_value_scalar - target_value_scalar|^α.
            # Our `value_targets_host` is in encoded h-space (post
            # scalar_transform), and `batch_values_host` is already raw
            # scalar (decode_value_dist → inverse h). Apply h⁻¹ to target
            # to bring both to raw space and match reference.
            var target_h = Float64(gpu.value_targets_host[0 * BATCH + b])
            # Inverse h: x = sign(y) * ((((1+4·eps·(|y|+1+eps))^0.5 - 1)
            # /(2·eps))^2 - 1). Same closed form as utils.inverse_scalar_transform.
            var sgn_t = 1.0 if target_h >= 0.0 else -1.0
            var abs_t = target_h if target_h >= 0.0 else -target_h
            var inner = sqrt(1.0 + 4.0 * EPS_H * (abs_t + 1.0 + EPS_H))
            var f = (inner - 1.0) / (2.0 * EPS_H)
            var target_raw = sgn_t * (f * f - 1.0)
            var pred_raw = Float64(gpu.batch_values_host[0 * BATCH + b])
            var diff = target_raw - pred_raw
            if diff < 0.0:
                diff = -diff
            var priority = (diff + self.per_eps) ** self.per_alpha
            if priority > max_p_seen:
                max_p_seen = priority
            gpu.per_tree.update(
                env_idx * PER_ENV_CAP + slot, Scalar[dtype](priority)
            )
        gpu.per_max_priority = Scalar[dtype](max_p_seen)

    # ══════════════════════════════════════════════════════════════════════
    # Diagnostics
    # ══════════════════════════════════════════════════════════════════════

    def _net_param_l2[
        N_ENVS_P: Int = 64,
        PER_ENV_CAP_P: Int = 1000,
    ](
        self,
        ctx: DeviceContext,
        mut gpu: MuZeroGPUState[Self.Config, N_ENVS_P, PER_ENV_CAP_P],
    ) raises -> Tuple[Float64, Float64, Float64]:
        """Compute L2 norm of each network's params (rep, dyn, pred).

        Lightweight diagnostic — DMAs all three params to host and
        computes norms there. Caller pays one synchronize. Use to
        verify training is actually changing weights (norms should
        evolve over training; values stuck = optimizer or gradient
        path is broken).
        """
        ctx.enqueue_copy(
            gpu.representation.params_host, gpu.representation.params_buf
        )
        ctx.enqueue_copy(gpu.dynamics.params_host, gpu.dynamics.params_buf)
        ctx.enqueue_copy(gpu.prediction.params_host, gpu.prediction.params_buf)
        ctx.synchronize()

        var rep_n = Float64(0.0)
        for i in range(Self.Config.RepModel.PARAM_SIZE):
            var v = Float64(gpu.representation.params_host[i])
            rep_n += v * v
        var dyn_n = Float64(0.0)
        for i in range(Self.Config.DynModel.PARAM_SIZE):
            var v = Float64(gpu.dynamics.params_host[i])
            dyn_n += v * v
        var pred_n = Float64(0.0)
        for i in range(Self.Config.PredModel.PARAM_SIZE):
            var v = Float64(gpu.prediction.params_host[i])
            pred_n += v * v

        from std.math import sqrt as _sqrt

        return (_sqrt(rep_n), _sqrt(dyn_n), _sqrt(pred_n))

    def _global_clip_grad_norm[
        N_ENVS_P: Int = 64,
        PER_ENV_CAP_P: Int = 1000,
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu: MuZeroGPUState[Self.Config, N_ENVS_P, PER_ENV_CAP_P],
    ) raises:
        """Per-network L2 norm grad clip. CPU roundtrip.

        We deliberately clip each network independently rather than
        jointly: rep+dyn naturally have larger gradient norms (the
        unrolled chain accumulates K reward+value backprops into them),
        and a JOINT clip would be dominated by their norms — pred's
        smaller grads then get scaled to the noise floor and pred stops
        learning. Per-network clipping matches each net's own scale.

        Mirrors the CPU `_clip_gradients` semantics conceptually but
        applied per-network. Total ~30KB DMA per train step.
        """
        comptime REP_PS = Self.Config.RepModel.PARAM_SIZE
        comptime DYN_PS = Self.Config.DynModel.PARAM_SIZE
        comptime PRED_PS = Self.Config.PredModel.PARAM_SIZE

        # DMA grads to host
        ctx.enqueue_copy(gpu.rep_grads_host, gpu.representation.grads_buf)
        ctx.enqueue_copy(gpu.dyn_grads_host, gpu.dynamics.grads_buf)
        ctx.enqueue_copy(gpu.pred_grads_host, gpu.prediction.grads_buf)
        ctx.synchronize()

        # Per-network norms
        var rep_sq = Float64(0.0)
        for i in range(REP_PS):
            var v = Float64(gpu.rep_grads_host[i])
            rep_sq += v * v
        var dyn_sq = Float64(0.0)
        for i in range(DYN_PS):
            var v = Float64(gpu.dyn_grads_host[i])
            dyn_sq += v * v
        var pred_sq = Float64(0.0)
        for i in range(PRED_PS):
            var v = Float64(gpu.pred_grads_host[i])
            pred_sq += v * v

        var rep_norm = sqrt(rep_sq)
        var dyn_norm = sqrt(dyn_sq)
        var pred_norm = sqrt(pred_sq)
        var thresh = self.max_grad_norm

        var rep_dirty = False
        var dyn_dirty = False
        var pred_dirty = False

        if rep_norm > thresh:
            var s = Scalar[dtype](thresh / rep_norm)
            for i in range(REP_PS):
                gpu.rep_grads_host[i] = gpu.rep_grads_host[i] * s
            rep_dirty = True
        if dyn_norm > thresh:
            var s = Scalar[dtype](thresh / dyn_norm)
            for i in range(DYN_PS):
                gpu.dyn_grads_host[i] = gpu.dyn_grads_host[i] * s
            dyn_dirty = True
        if pred_norm > thresh:
            var s = Scalar[dtype](thresh / pred_norm)
            for i in range(PRED_PS):
                gpu.pred_grads_host[i] = gpu.pred_grads_host[i] * s
            pred_dirty = True

        if rep_dirty:
            ctx.enqueue_copy(gpu.representation.grads_buf, gpu.rep_grads_host)
        if dyn_dirty:
            ctx.enqueue_copy(gpu.dynamics.grads_buf, gpu.dyn_grads_host)
        if pred_dirty:
            ctx.enqueue_copy(gpu.prediction.grads_buf, gpu.pred_grads_host)

    def reset_episode(mut self):
        """Reset episode buffers for a new episode."""
        self._episode_obs.clear()
        self._episode_actions.clear()
        self._episode_rewards.clear()
        self._episode_policies.clear()
        self._episode_values.clear()
        self._episode_to_play.clear()

    def store_transition(
        mut self,
        obs: List[Scalar[dtype]],
        action: Int,
        reward: Float64,
        policy: InlineArray[Float64, Self.Config.action_dim],
        value: Float64,
        done: Bool,
        to_play: Int = 0,
    ):
        """Store a transition with MCTS policy/value targets.

        Stores to the episode buffer. When the episode ends (done=True),
        flushes the entire episode to the replay buffer with MCTS targets.

        Args:
            obs: Current observation.
            action: Action taken (discrete index).
            reward: Reward received.
            policy: MCTS visit count policy.
            value: MCTS root value.
            done: Whether the episode ended.
            to_play: Player to move at this step (0 for single-player envs;
                0/1 for two-player turn-taking games). Used to sign-flip
                bootstrap value and rewards during n-step target
                computation in two-player games.
        """
        self._episode_obs.append(obs.copy())
        self._episode_actions.append(action)
        self._episode_rewards.append(reward)
        self._episode_policies.append(policy)
        self._episode_values.append(value)
        self._episode_to_play.append(to_play)

        if done:
            self._flush_episode()

    def _flush_episode(mut self):
        """Flush episode data to the replay buffer with MCTS targets.

        Writes ``Config.Aug.NUM_SYMMETRIES`` augmented copies of the
        complete episode. Each augmented copy uses ONE symmetry index
        applied uniformly to every (obs, action, mcts_policy) in the
        episode — this preserves the K-step unroll sequence semantics:
        sampling a sequence starting at step t inside any augmented
        copy gives a consistent ``(s_t, a_t, s_{t+1}, …)`` chain in a
        single symmetry frame, which is what the dynamics network is
        trained to predict.

        Per-step scalars that are spatially invariant (reward, root
        value, to_play) are written identically across all symmetries.

        For configs with ``Aug = IdentityAugmenter`` (NUM_SYMMETRIES=1)
        this is exactly the pre-augmentation behavior — one episode
        flush per game, no extra cost.
        """
        comptime OBS = Self.Config.obs_dim
        comptime ACT = Self.Config.action_dim
        comptime NUM_SYM: Int = Self.Config.Aug.NUM_SYMMETRIES

        var ep_len = len(self._episode_obs)

        # Per-step augmentation scratch — pointers reused across all
        # (step, sym) pairs. Allocate once for the whole episode.
        var src_obs: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](OBS)
        var dst_obs: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](OBS)
        var src_pol: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](ACT)
        var dst_pol: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](ACT)
        var src_act_oh: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](ACT)
        var dst_act_oh: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](ACT)

        for sym in range(NUM_SYM):
            for t in range(ep_len):
                # ── Build raw (unaugmented) obs / policy / action buffers
                for i in range(OBS):
                    if i < len(self._episode_obs[t]):
                        src_obs[i] = Scalar[dtype](self._episode_obs[t][i])
                    else:
                        src_obs[i] = Scalar[dtype](0.0)

                for a in range(ACT):
                    src_pol[a] = Scalar[dtype](
                        self._episode_policies[t][a]
                    )
                    src_act_oh[a] = Scalar[dtype](0.0)
                src_act_oh[self._episode_actions[t]] = Scalar[dtype](1.0)

                # ── Apply symmetry to obs, policy, one-hot action
                # ``augment_policy`` treats ACT as a flat spatial grid;
                # for board games ACT == cells, so the same transform
                # permutes one-hot actions correctly. For non-board
                # configs Aug is IdentityAugmenter and these are no-ops.
                Self.Config.Aug.augment_obs[OBS](src_obs, sym, dst_obs)
                Self.Config.Aug.augment_policy[ACT](src_pol, sym, dst_pol)
                Self.Config.Aug.augment_policy[ACT](
                    src_act_oh, sym, dst_act_oh
                )

                # ── Copy into the InlineArray shapes the buffer expects
                var obs_arr = InlineArray[
                    Scalar[DType.float32], Self.Config.obs_dim
                ](uninitialized=True)
                for i in range(OBS):
                    obs_arr[i] = Scalar[DType.float32](dst_obs[i])

                var act_arr = InlineArray[
                    Scalar[DType.float32], Self.Config.action_dim
                ](uninitialized=True)
                for a in range(ACT):
                    act_arr[a] = Scalar[DType.float32](dst_act_oh[a])

                # done=True on the last step of every augmented copy so
                # each copy is its own sequence (own episode_id from
                # SequenceReplayBuffer.add → increments current_episode
                # at each flush). Crucial for ``_is_valid_sequence_start``
                # to refuse cross-symmetry sequences.
                var is_done = t == ep_len - 1

                self.state.buffer.add(
                    obs_arr,
                    act_arr,
                    Scalar[DType.float32](self._episode_rewards[t]),
                    is_done,
                )

                # Store MCTS targets at the buffer write position.
                # Policy target uses the augmented policy. Value /
                # to_play are spatially invariant — same value for
                # every symmetry of the same step.
                var buf_idx = (
                    self.state.buffer.ptr - 1 + Self.StateType._CAP
                ) % Self.StateType._CAP
                for a in range(ACT):
                    self.state.mcts_policies[
                        buf_idx * ACT + a
                    ] = dst_pol[a]
                self.state.mcts_values[buf_idx] = Scalar[dtype](
                    self._episode_values[t]
                )
                self.state.mcts_to_play[buf_idx] = Scalar[DType.uint8](
                    self._episode_to_play[t]
                )

        src_obs.free()
        dst_obs.free()
        src_pol.free()
        dst_pol.free()
        src_act_oh.free()
        dst_act_oh.free()

        self.reset_episode()

    # ══════════════════════════════════════════════════════════════════════
    # CPU MCTS (delegates to shared planner)
    # ══════════════════════════════════════════════════════════════════════

    def _mcts_search_visits_cpu(
        mut self,
        obs: List[Scalar[dtype]],
        legal_mask: List[Bool],
        mut visits_out: UnsafePointer[Int, MutAnyOrigin],
        add_noise: Bool = True,
    ) raises -> Float64:
        """Run MuZero MCTS from ``obs`` via ``GenericCPUMCTS``.

        Writes per-action integer visit counts into ``visits_out``
        (length ``Config.action_dim``) and returns the visit-weighted
        root Q (the agent's V(root) estimate, consumed by the n-step
        bootstrap stored in the replay buffer).

        Adapters: ``MuZeroRepCPU`` (representation net), ``MuZeroDynCPU``
        (dynamics net + categorical reward decode), ``MuZeroPredCPU``
        (prediction net + softmax policy + categorical value decode).
        Search strategies come from ``Config.PUCT`` / ``Config.Noise`` /
        ``Config.Players`` — for board games these are
        ``AlphaGoPUCT`` / ``DirichletNoise`` / ``SelfPlay``.

        Mirrors the AZ CPU helper's signature so the self-play loop can
        switch between agents with minimal call-site churn.
        """
        comptime ACT = Self.Config.action_dim
        comptime OBS = Self.Config.obs_dim
        comptime LATENT = Self.Config.latent_dim
        comptime BINS = Self.Config.num_bins
        comptime SIMS = Self.Config.num_simulations
        comptime MAX_N = Self.Config.max_nodes
        comptime RepModel = Self.Config.RepModel
        comptime DynModel = Self.Config.DynModel
        comptime PredModel = Self.Config.PredModel
        comptime OptType = Self.Config.OptType

        for a in range(ACT):
            visits_out[a] = 0

        # Build adapters. They alias the agent's NetworkState param /
        # model-state pointers — no copies. The search path is
        # forwards-only, so this is read-only sharing.
        var rep = MuZeroRepCPU[OBS, LATENT, RepModel, OptType](
            params=self.state.representation.params,
            model_state=self.state.representation.model_state,
        )
        var dyn = MuZeroDynCPU[LATENT, ACT, BINS, DynModel, OptType](
            params=self.state.dynamics.params,
            model_state=self.state.dynamics.model_state,
            v_min=self.v_min,
            v_max=self.v_max,
        )
        var pred = MuZeroPredCPU[LATENT, ACT, BINS, PredModel, OptType](
            params=self.state.prediction.params,
            model_state=self.state.prediction.model_state,
            v_min=self.v_min,
            v_max=self.v_max,
        )

        # Widen obs to Float64 for the shared planner's contract.
        var obs_f64 = List[Float64](length=OBS, fill=Float64(0.0))
        for i in range(OBS):
            if i < len(obs):
                obs_f64[i] = Float64(obs[i])

        var mcts = GenericCPUMCTS[
            ACT,
            LATENT,
            SIMS,
            MAX_N,
            Self.Config.PUCT,
            Self.Config.Noise,
            Self.Config.Players,
            Self.Config.batch_sims,
            Self.Config.virtual_loss,
        ](gamma=self.gamma)

        var policy = mcts.search[
            MuZeroRepCPU[OBS, LATENT, RepModel, OptType],
            MuZeroDynCPU[LATENT, ACT, BINS, DynModel, OptType],
            MuZeroPredCPU[LATENT, ACT, BINS, PredModel, OptType],
        ](
            rep,
            dyn,
            pred,
            obs_f64,
            add_noise=add_noise,
            legal_mask=legal_mask,
        )

        # Visit-weighted root Q — the agent's V(root) estimate.
        # ``select_action`` / ``reanalyze`` / ``train_selfplay_cpu``
        # all consume this as ``mcts_values[t]`` for the n-step
        # bootstrap stored in the replay buffer.
        var root_value = mcts.root_value()

        # Copy integer visit counts directly out of the planner's root
        # node. The previous ``Int(policy[a] * SIMS + 0.5)`` round-trip
        # introduced rounding bias (e.g., policy=[0.5, 0.5] with SIMS=25
        # mapped to [13, 13] summing to 26 instead of 25), and lost
        # bit-parity with the legacy MCTS path that consumers like the
        # temperature sampler in ``select_action`` were calibrated for.
        # Reading directly preserves the exact integer counts MCTS
        # backed up during the search.
        if len(mcts.nodes) > 0:
            for a in range(ACT):
                visits_out[a] = mcts.nodes[0].visit_count[a]
        else:
            for a in range(ACT):
                visits_out[a] = 0
        _ = policy  # consumed only for root_value() above

        return root_value

    # ══════════════════════════════════════════════════════════════════════
    # Action Selection
    # ══════════════════════════════════════════════════════════════════════

    def select_action(
        mut self,
        obs: List[Scalar[dtype]],
        training: Bool = True,
        legal_mask: List[Bool] = List[Bool](),
    ) raises -> Tuple[
        Int, InlineArray[Float64, Self.Config.action_dim], Float64
    ]:
        """Select an action using MCTS with the learned model.

        Routes through the shared ``GenericCPUMCTS`` planner via
        ``_mcts_search_visits_cpu``. Returns the canonical MuZero tuple
        ``(action, mcts_policy, root_value)``.

        Args:
            obs: Current observation [obs_dim].
            training: If True, blend root prior with Dirichlet noise AND
                sample with temperature for the first move; if False,
                no noise, argmax visits.
            legal_mask: Optional length-``action_dim`` legal-action
                mask. Empty means all legal.

        Returns:
            ``(action, normalized_visit_policy, root_value)``.
        """
        comptime ACT = Self.Config.action_dim

        # Scratch — same shape as the train-loop helper.
        var visits: UnsafePointer[Int, MutAnyOrigin] = alloc[Int](ACT)
        var root_value = self._mcts_search_visits_cpu(
            obs, legal_mask, visits, add_noise=training
        )

        # Normalize visits → policy (the public API contract).
        var total: Int = 0
        for a in range(ACT):
            total += visits[a]

        var policy = InlineArray[Float64, ACT](uninitialized=True)
        if total > 0:
            for a in range(ACT):
                policy[a] = Float64(visits[a]) / Float64(total)
        else:
            for a in range(ACT):
                policy[a] = 1.0 / Float64(ACT)

        # Sample action.
        var action: Int = 0
        if not training or self.temperature < 0.01:
            # Greedy argmax over visit counts (stable: discriminates ties
            # by index, same as the legacy MCTS did).
            var best_v: Int = -1
            for a in range(ACT):
                if visits[a] > best_v:
                    best_v = visits[a]
                    action = a
        else:
            # Temperature sampling on visit^{1/T}.
            var temp_policy = InlineArray[Float64, ACT](uninitialized=True)
            var inv_temp = 1.0 / self.temperature
            var sum_p = Float64(0.0)
            for a in range(ACT):
                var count = Float64(visits[a])
                if count > 0.0:
                    temp_policy[a] = exp(inv_temp * log(count))
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

        visits.free()
        return (action, policy, root_value)

    # ══════════════════════════════════════════════════════════════════════
    # MuZero Reanalyze
    # ══════════════════════════════════════════════════════════════════════

    def reanalyze(mut self, num_positions: Int = 32) raises:
        """Re-run MCTS on old observations using the latest networks.

        Updates ``mcts_policies`` and ``mcts_values`` in the replay
        buffer with fresh visit-count distributions + root values from
        the current networks. Routes through ``GenericCPUMCTS`` via
        ``_mcts_search_visits_cpu`` for parity with the self-play
        collection path.

        No Dirichlet noise — reanalyze should produce a deterministic
        re-evaluation of stored positions, not exploration.

        Reference: Schrittwieser et al., 2021 — Online and Offline RL
        by Planning with a Learned World Model.

        Args:
            num_positions: Number of random positions to reanalyze.
        """
        comptime CAPACITY = Self.StateType._CAP
        comptime ACT = Self.Config.action_dim
        comptime OBS = Self.Config.obs_dim

        if self.state.buffer.len() < 10:
            return

        var visits: UnsafePointer[Int, MutAnyOrigin] = alloc[Int](ACT)
        var obs = List[Scalar[dtype]](capacity=OBS)
        for _ in range(OBS):
            obs.append(Scalar[dtype](0.0))

        for _ in range(num_positions):
            # Sample a random position from the buffer.
            var pos = (
                Int(random_float64() * Float64(self.state.buffer.size))
                % self.state.buffer.size
            )
            var actual_idx = (
                self.state.buffer.ptr - self.state.buffer.size + pos
            ) % CAPACITY
            if actual_idx < 0:
                actual_idx += CAPACITY

            # Pull obs out of the buffer into the scratch list.
            for i in range(OBS):
                obs[i] = Scalar[dtype](
                    self.state.buffer.obs[actual_idx * OBS + i]
                )

            # No legal mask available for stored positions (we don't
            # cache the env state in the SequenceReplayBuffer). Pass
            # empty mask → planner treats all actions as legal, same
            # behavior as the legacy reanalyze.
            var root_value = self._mcts_search_visits_cpu(
                obs, List[Bool](), visits, add_noise=False
            )

            # Normalize visits → policy and write back to targets.
            var total: Int = 0
            for a in range(ACT):
                total += visits[a]
            if total > 0:
                for a in range(ACT):
                    self.state.mcts_policies[
                        actual_idx * ACT + a
                    ] = Scalar[dtype](Float64(visits[a]) / Float64(total))
            else:
                var inv = Scalar[dtype](1.0 / Float64(ACT))
                for a in range(ACT):
                    self.state.mcts_policies[actual_idx * ACT + a] = inv

            self.state.mcts_values[actual_idx] = Scalar[dtype](root_value)

        visits.free()

    # ══════════════════════════════════════════════════════════════════════
    # Training (K-Step Unrolled Forward/Backward)
    # ══════════════════════════════════════════════════════════════════════

    def update(mut self, use_reanalyze: Bool = False) raises -> Float64:
        """Run one training step with K-step unrolled forward/backward.

        1. Sample batch of positions from replay buffer
        2. Compute n-step value targets and reward targets
        3. Forward: h(obs) -> s^0, then K steps of f(s^k) and g(s^k, a^k)
        4. Backward: propagate gradients through unrolled chain
        5. Optimizer step on all three networks

        Returns:
            Total training loss.
        """
        comptime BATCH = Self.Config.batch_size
        comptime K = Self.Config.unroll_steps
        comptime LATENT = Self.Config.latent_dim
        comptime ACT = Self.Config.action_dim
        comptime BINS = Self.Config.num_bins
        comptime OBS = Self.Config.obs_dim

        # ── Step 0 (optional): Reanalyze old positions ────────────────
        if use_reanalyze:
            self.reanalyze(num_positions=Self.Config.batch_size // 4)

        # ── Step 1+2: Sample batch with MCTS targets + n-step returns ──
        self.state.sample_batch_with_targets(self.gamma)

        # Apply scalar transform to value and reward targets
        for k in range(K + 1):
            for b in range(BATCH):
                var val = Float64(self.state._value_targets[k * BATCH + b])
                self.state._value_targets[k * BATCH + b] = Scalar[dtype](
                    scalar_transform(val)
                )
        for k in range(K):
            for b in range(BATCH):
                var rew = Float64(self.state._reward_targets[k * BATCH + b])
                self.state._reward_targets[k * BATCH + b] = Scalar[dtype](
                    scalar_transform(rew)
                )

        # ── Step 3: Forward pass (K-step unroll) ─────────────────────
        var total_loss = self._forward_and_compute_loss()

        # ── Step 4: Backward pass ────────────────────────────────────
        self._backward()

        # ── Step 5: Gradient clipping + Optimizer step ─────────────────
        self._clip_gradients()
        self.state.representation.optimizer_step()
        self.state.dynamics.optimizer_step()
        self.state.prediction.optimizer_step()

        self.train_step_count += 1
        return total_loss

    def _log_cpu_diag(mut self) raises -> None:
        """Emit the standard CPU MuZero dashboard keys.

        Reads from the scratch buffers ``update()`` just populated
        (still live until the next ``sample_batch_with_targets``).
        Caller is responsible for gating by ``self.logger`` /
        ``self.diag_every`` — this method unconditionally logs.

        Keys (KNOWN_GROUPS-aligned with TTT MuZero so curves overlay):
          policy_ce / entropy / value_mse / value_mean /
          value_target_mean / loss / target_entropy / target_max_prob /
          policy_ce_minus_target_entropy / param_norm / grad_param_norm.
        """
        comptime BATCH = Self.Config.batch_size
        comptime ACT = Self.Config.action_dim
        comptime BINS = Self.Config.num_bins
        comptime PRED_OUT = Self.StateType.PRED_OUT
        comptime REP_PS = Self.StateType.RepModel.PARAM_SIZE
        comptime DYN_PS = Self.StateType.DynModel.PARAM_SIZE
        comptime PRED_PS = Self.StateType.PredModel.PARAM_SIZE

        var ts = self.train_step_count

        var sum_p_ce: Float64 = 0.0
        var sum_p_ent: Float64 = 0.0
        var sum_tgt_ent: Float64 = 0.0
        var sum_tgt_max: Float64 = 0.0
        var sum_v_mse: Float64 = 0.0
        var sum_v_mean: Float64 = 0.0
        var sum_v_tgt: Float64 = 0.0

        var v_step = (
            self.v_max - self.v_min
        ) / Float64(BINS - 1) if BINS > 1 else Float64(1.0)

        for b in range(BATCH):
            # Softmax over policy logits at k=0
            var pol_off = b * PRED_OUT
            var max_l = Float64(self.state._pred_outputs[pol_off])
            for a in range(1, ACT):
                var lv = Float64(self.state._pred_outputs[pol_off + a])
                if lv > max_l:
                    max_l = lv
            var sum_e: Float64 = 0.0
            for a in range(ACT):
                sum_e += exp(
                    Float64(self.state._pred_outputs[pol_off + a]) - max_l
                )

            var tmax: Float64 = 0.0
            for a in range(ACT):
                var prob = (
                    exp(Float64(self.state._pred_outputs[pol_off + a]) - max_l)
                    / sum_e
                )
                var target = Float64(self.state._batch_policies[b * ACT + a])
                if target > 1e-8 and prob > 1e-8:
                    sum_p_ce -= target * log(prob)
                if prob > 1e-8:
                    sum_p_ent -= prob * log(prob)
                if target > 1e-8:
                    sum_tgt_ent -= target * log(target)
                if target > tmax:
                    tmax = target
            sum_tgt_max += tmax

            # Decoded value vs scalar target.
            var val_off = b * PRED_OUT + ACT
            var max_v = Float64(self.state._pred_outputs[val_off])
            for i in range(1, BINS):
                var v = Float64(self.state._pred_outputs[val_off + i])
                if v > max_v:
                    max_v = v
            var sum_e_v: Float64 = 0.0
            for i in range(BINS):
                sum_e_v += exp(
                    Float64(self.state._pred_outputs[val_off + i]) - max_v
                )
            var pred_v: Float64 = 0.0
            for i in range(BINS):
                var probv = exp(
                    Float64(self.state._pred_outputs[val_off + i]) - max_v
                ) / sum_e_v
                var bin_center = self.v_min + Float64(i) * v_step
                pred_v += probv * bin_center
            var tgt_v = Float64(self.state._value_targets[b])
            var diff = pred_v - tgt_v
            sum_v_mse += diff * diff
            sum_v_mean += pred_v
            sum_v_tgt += tgt_v

        var n = Float64(BATCH)
        var policy_ce = sum_p_ce / n
        var policy_entropy = sum_p_ent / n
        var value_mse = sum_v_mse / n
        var value_mean = sum_v_mean / n
        var value_target_mean = sum_v_tgt / n
        var tgt_entropy = sum_tgt_ent / n
        var tgt_max = sum_tgt_max / n

        # Clamp NaN / explosions so dashboard doesn't silently drop.
        if policy_ce != policy_ce or policy_ce > 1e10:
            policy_ce = 0.0
        if value_mse != value_mse or value_mse > 1e10:
            value_mse = 0.0

        self.logger.value()[].log_scalar("policy_ce", policy_ce, ts)
        self.logger.value()[].log_scalar("entropy", policy_entropy, ts)
        self.logger.value()[].log_scalar("value_mse", value_mse, ts)
        self.logger.value()[].log_scalar("value_mean", value_mean, ts)
        self.logger.value()[].log_scalar(
            "value_target_mean", value_target_mean, ts
        )
        self.logger.value()[].log_scalar("loss", policy_ce + value_mse, ts)
        self.logger.value()[].log_scalar("target_entropy", tgt_entropy, ts)
        self.logger.value()[].log_scalar("target_max_prob", tgt_max, ts)
        self.logger.value()[].log_scalar(
            "policy_ce_minus_target_entropy", policy_ce - tgt_entropy, ts
        )

        # Param + grad norms (joint across all 3 networks — matches
        # the spirit of the global-L2 grad clip).
        var param_norm: Float64 = 0.0
        for i in range(REP_PS):
            var p = Float64(self.state.representation.params[i])
            if p == p:
                param_norm += p * p
        for i in range(DYN_PS):
            var p = Float64(self.state.dynamics.params[i])
            if p == p:
                param_norm += p * p
        for i in range(PRED_PS):
            var p = Float64(self.state.prediction.params[i])
            if p == p:
                param_norm += p * p
        self.logger.value()[].log_scalar(
            "param_norm",
            sqrt(param_norm) if param_norm == param_norm else 0.0,
            ts,
        )

        var grad_norm: Float64 = 0.0
        for i in range(REP_PS):
            var g = Float64(self.state.representation.grads[i])
            if g == g:
                grad_norm += g * g
        for i in range(DYN_PS):
            var g = Float64(self.state.dynamics.grads[i])
            if g == g:
                grad_norm += g * g
        for i in range(PRED_PS):
            var g = Float64(self.state.prediction.grads[i])
            if g == g:
                grad_norm += g * g
        self.logger.value()[].log_scalar(
            "grad_param_norm",
            sqrt(grad_norm) if grad_norm == grad_norm else 0.0,
            ts,
        )

    def _forward_and_compute_loss(mut self) -> Float64:
        """K-step unrolled forward pass with loss computation.

        s^0 = h(obs_t)                  # representation
        for k = 0..K:
            p^k, v^k = f(s^k)           # prediction
            if k < K:
                s^{k+1}, r^{k+1} = g(s^k, a_{t+k})  # dynamics

        Computes cross-entropy losses for policy, value, and reward.

        Returns:
            Total loss (sum of all components).
        """
        comptime BATCH = Self.Config.batch_size
        comptime K = Self.Config.unroll_steps
        comptime LATENT = Self.Config.latent_dim
        comptime ACT = Self.Config.action_dim
        comptime BINS = Self.Config.num_bins
        comptime OBS = Self.Config.obs_dim
        comptime PRED_OUT = Self.StateType.PRED_OUT
        comptime DYN_IN = Self.StateType.DYN_IN
        comptime DYN_OUT = Self.StateType.DYN_OUT

        # Use Model's own dimensions for LayoutTensor compatibility
        comptime REP_IN_DIM = Self.StateType.RepModel.IN_DIM
        comptime REP_OUT_DIM = Self.StateType.RepModel.OUT_DIM
        comptime REP_CS = Self.StateType.RepModel.CACHE_SIZE
        comptime DYN_IN_DIM = Self.StateType.DynModel.IN_DIM
        comptime DYN_OUT_DIM = Self.StateType.DynModel.OUT_DIM
        comptime DYN_CS = Self.StateType.DynModel.CACHE_SIZE
        comptime PRED_IN_DIM = Self.StateType.PredModel.IN_DIM
        comptime PRED_OUT_DIM = Self.StateType.PredModel.OUT_DIM
        comptime PRED_CS = Self.StateType.PredModel.CACHE_SIZE

        var total_loss = Float64(0.0)
        var inv_k = 1.0 / Float64(K + 1)  # Scale loss by 1/(K+1)

        # ── Representation: h(obs_0) -> s^0 ──────────────────────────
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, REP_IN_DIM), MutAnyOrigin
        ](self.state._batch_obs)

        var h0_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, REP_OUT_DIM), MutAnyOrigin
        ](self.state._hidden_states)

        var rep_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, REP_CS), MutAnyOrigin
        ](self.state._rep_cache)

        Self.RepNet.forward_with_cache[BATCH](
            obs_t,
            h0_t,
            self.state.representation.params_view(),
            self.state.representation.model_state_view(),
            rep_cache_t,
        )

        # Scale hidden state
        self._scale_batch_hidden(0)

        # ── K-step unroll ────────────────────────────────────────────
        for k in range(K + 1):
            # Prediction: f(s^k) -> (policy_logits, value_logits)
            var hk_offset = k * BATCH * LATENT
            var hk_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, PRED_IN_DIM), MutAnyOrigin
            ](self.state._hidden_states + hk_offset)

            var pred_offset = k * BATCH * PRED_OUT
            var pred_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, PRED_OUT_DIM), MutAnyOrigin
            ](self.state._pred_outputs + pred_offset)

            var pred_cache_offset = k * BATCH * PRED_CS
            var pred_cache_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, PRED_CS), MutAnyOrigin
            ](self.state._pred_caches + pred_cache_offset)

            Self.PredNet.forward_with_cache[BATCH](
                hk_t,
                pred_t,
                self.state.prediction.params_view(),
                self.state.prediction.model_state_view(),
                pred_cache_t,
            )

            # Compute policy and value loss for this step
            for b in range(BATCH):
                # Policy loss: CE(predicted_logits, mcts_policy)
                # batch_policies layout: [t, b, a] = t*BATCH*ACT + b*ACT + a
                var policy_logits = alloc[Float64](ACT)
                var policy_target = alloc[Float64](ACT)
                var pol_base = k * BATCH * ACT + b * ACT
                for a in range(ACT):
                    policy_logits[a] = Float64(
                        (
                            self.state._pred_outputs
                            + pred_offset
                            + b * PRED_OUT
                            + a
                        )[]
                    )
                    policy_target[a] = Float64(
                        self.state._batch_policies[pol_base + a]
                    )

                var policy_loss = cross_entropy_with_softmax[ACT](
                    policy_logits, policy_target
                )
                total_loss += policy_loss * inv_k

                policy_logits.free()
                policy_target.free()

                # Value loss: CE(predicted_value_logits, two_hot(target))
                var value_logits = alloc[Float64](BINS)
                var value_target = alloc[Float64](BINS)
                for i in range(BINS):
                    value_logits[i] = Float64(
                        (
                            self.state._pred_outputs
                            + pred_offset
                            + b * PRED_OUT
                            + ACT
                            + i
                        )[]
                    )
                    value_target[i] = Float64(0.0)

                # Encode target as two-hot
                var target_val = Float64(
                    self.state._value_targets[k * BATCH + b]
                )
                encode_categorical[BINS](
                    target_val, self.v_min, self.v_max, value_target
                )

                var value_loss = cross_entropy_with_softmax[BINS](
                    value_logits, value_target
                )
                total_loss += value_loss * inv_k

                value_logits.free()
                value_target.free()

            # Dynamics: g(s^k, a_{t+k}) -> (s^{k+1}, r^{k+1})
            if k < K:
                # Build dynamics input: [hidden_state || one_hot_action]
                var dyn_input_ptr = alloc[Scalar[dtype]](BATCH * DYN_IN)
                memset(dyn_input_ptr, 0, BATCH * DYN_IN)

                for b in range(BATCH):
                    # Copy hidden state
                    for i in range(LATENT):
                        dyn_input_ptr[b * DYN_IN + i] = (
                            self.state._hidden_states
                            + hk_offset
                            + b * LATENT
                            + i
                        )[]
                    # Copy one-hot action from batch_actions (TIME-MAJOR:
                    # [k, b, a] = k*BATCH*ACT + b*ACT + a)
                    for a in range(ACT):
                        dyn_input_ptr[b * DYN_IN + LATENT + a] = (
                            self.state._batch_actions
                            + k * BATCH * ACT
                            + b * ACT
                            + a
                        )[]

                var dyn_in_t = LayoutTensor[
                    dtype, Layout.row_major(BATCH, DYN_IN_DIM), MutAnyOrigin
                ](dyn_input_ptr)

                var next_offset = (k + 1) * BATCH * LATENT
                var dyn_out_ptr = alloc[Scalar[dtype]](BATCH * DYN_OUT)
                memset(dyn_out_ptr, 0, BATCH * DYN_OUT)
                var dyn_out_t = LayoutTensor[
                    dtype, Layout.row_major(BATCH, DYN_OUT_DIM), MutAnyOrigin
                ](dyn_out_ptr)

                var dyn_cache_offset = k * BATCH * DYN_CS
                var dyn_cache_t = LayoutTensor[
                    dtype, Layout.row_major(BATCH, DYN_CS), MutAnyOrigin
                ](self.state._dyn_caches + dyn_cache_offset)

                Self.DynNet.forward_with_cache[BATCH](
                    dyn_in_t,
                    dyn_out_t,
                    self.state.dynamics.params_view(),
                    self.state.dynamics.model_state_view(),
                    dyn_cache_t,
                )

                # Extract next hidden state and reward logits
                for b in range(BATCH):
                    for i in range(LATENT):
                        (
                            self.state._hidden_states
                            + next_offset
                            + b * LATENT
                            + i
                        )[] = (dyn_out_ptr + b * DYN_OUT + i)[]
                    for i in range(BINS):
                        (
                            self.state._dyn_reward_logits
                            + k * BATCH * BINS
                            + b * BINS
                            + i
                        )[] = (dyn_out_ptr + b * DYN_OUT + LATENT + i)[]

                # Scale next hidden state
                self._scale_batch_hidden(k + 1)

                # Reward loss: CE(reward_logits, two_hot(reward_target))
                for b in range(BATCH):
                    var rew_logits = alloc[Float64](BINS)
                    var rew_target = alloc[Float64](BINS)
                    for i in range(BINS):
                        rew_logits[i] = Float64(
                            (
                                self.state._dyn_reward_logits
                                + k * BATCH * BINS
                                + b * BINS
                                + i
                            )[]
                        )
                        rew_target[i] = Float64(0.0)

                    var target_rew = Float64(
                        self.state._reward_targets[k * BATCH + b]
                    )
                    encode_categorical[BINS](
                        target_rew, self.v_min, self.v_max, rew_target
                    )

                    var rew_loss = cross_entropy_with_softmax[BINS](
                        rew_logits, rew_target
                    )
                    total_loss += rew_loss * inv_k

                    rew_logits.free()
                    rew_target.free()

                dyn_input_ptr.free()
                dyn_out_ptr.free()

        return total_loss / Float64(BATCH)

    def _backward(mut self):
        """K-step unrolled backward pass through all three networks.

        Gradients flow: prediction -> dynamics (xK) -> representation.
        Dynamics gradients are scaled by 1/K to prevent gradient explosion.
        """
        comptime BATCH = Self.Config.batch_size
        comptime K = Self.Config.unroll_steps
        comptime LATENT = Self.Config.latent_dim
        comptime ACT = Self.Config.action_dim
        comptime BINS = Self.Config.num_bins
        comptime OBS = Self.Config.obs_dim
        comptime PRED_OUT = Self.StateType.PRED_OUT
        comptime DYN_IN = Self.StateType.DYN_IN
        comptime DYN_OUT = Self.StateType.DYN_OUT

        # Use Model's own dimensions for LayoutTensor compatibility
        comptime REP_IN_DIM = Self.StateType.RepModel.IN_DIM
        comptime REP_OUT_DIM = Self.StateType.RepModel.OUT_DIM
        comptime REP_CS = Self.StateType.RepModel.CACHE_SIZE
        comptime DYN_IN_DIM = Self.StateType.DynModel.IN_DIM
        comptime DYN_OUT_DIM = Self.StateType.DynModel.OUT_DIM
        comptime DYN_CS = Self.StateType.DynModel.CACHE_SIZE
        comptime PRED_IN_DIM = Self.StateType.PredModel.IN_DIM
        comptime PRED_OUT_DIM = Self.StateType.PredModel.OUT_DIM
        comptime PRED_CS = Self.StateType.PredModel.CACHE_SIZE

        var inv_k = 1.0 / Float64(K + 1)
        var inv_batch = 1.0 / Float64(BATCH)

        # Zero all gradients
        self.state.representation.zero_grads()
        self.state.dynamics.zero_grads()
        self.state.prediction.zero_grads()

        # Gradient carry for hidden state (accumulated through K steps)
        memset(self.state._grad_hidden, 0, BATCH * LATENT)

        # Process steps in REVERSE for BPTT
        for _ri in range(K + 1):
            var k = K - _ri

            # ── Prediction backward at step k ────────────────────────
            # Compute grad of prediction output: softmax(logits) - target
            memset(self.state._grad_pred_out, 0, BATCH * PRED_OUT)

            for b in range(BATCH):
                var pred_offset = k * BATCH * PRED_OUT + b * PRED_OUT

                # Policy gradient: (softmax(p) - target) / batch * inv_k
                # batch_policies layout: [t, b, a] = t*BATCH*ACT + b*ACT + a
                var policy_logits = alloc[Float64](ACT)
                var pol_base = k * BATCH * ACT + b * ACT
                for a in range(ACT):
                    policy_logits[a] = Float64(
                        (self.state._pred_outputs + pred_offset + a)[]
                    )
                # Softmax
                var max_p = policy_logits[0]
                for a in range(1, ACT):
                    if policy_logits[a] > max_p:
                        max_p = policy_logits[a]
                var sum_exp_p = Float64(0.0)
                for a in range(ACT):
                    policy_logits[a] = exp(policy_logits[a] - max_p)
                    sum_exp_p += policy_logits[a]
                for a in range(ACT):
                    var prob = policy_logits[a] / sum_exp_p
                    var target = Float64(
                        self.state._batch_policies[pol_base + a]
                    )
                    self.state._grad_pred_out[b * PRED_OUT + a] = Scalar[dtype](
                        (prob - target) * inv_k * inv_batch
                    )
                policy_logits.free()

                # Value gradient: (softmax(v) - target) / batch * inv_k
                var value_logits = alloc[Float64](BINS)
                var value_target = alloc[Float64](BINS)
                for i in range(BINS):
                    value_logits[i] = Float64(
                        (self.state._pred_outputs + pred_offset + ACT + i)[]
                    )
                    value_target[i] = Float64(0.0)
                var target_val = Float64(
                    self.state._value_targets[k * BATCH + b]
                )
                encode_categorical[BINS](
                    target_val, self.v_min, self.v_max, value_target
                )
                # Softmax
                var max_v = value_logits[0]
                for i in range(1, BINS):
                    if value_logits[i] > max_v:
                        max_v = value_logits[i]
                var sum_exp_v = Float64(0.0)
                for i in range(BINS):
                    value_logits[i] = exp(value_logits[i] - max_v)
                    sum_exp_v += value_logits[i]
                for i in range(BINS):
                    var prob = value_logits[i] / sum_exp_v
                    self.state._grad_pred_out[b * PRED_OUT + ACT + i] = Scalar[
                        dtype
                    ]((prob - value_target[i]) * inv_k * inv_batch)
                value_logits.free()
                value_target.free()

            # Prediction backward: grad_pred_out -> grad_hidden (accumulated)
            var pred_cache_offset = k * BATCH * PRED_CS
            var pred_cache_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, PRED_CS), MutAnyOrigin
            ](self.state._pred_caches + pred_cache_offset)

            var grad_pred_out_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, PRED_OUT_DIM), MutAnyOrigin
            ](self.state._grad_pred_out)

            # Temporary gradient for prediction input (hidden state)
            var grad_pred_in_ptr = alloc[Scalar[dtype]](BATCH * LATENT)
            memset(grad_pred_in_ptr, 0, BATCH * LATENT)
            var grad_pred_in_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, PRED_IN_DIM), MutAnyOrigin
            ](grad_pred_in_ptr)

            var pred_grads = self.state.prediction.grads_view()
            Self.PredNet.backward[BATCH](
                grad_pred_out_t,
                grad_pred_in_t,
                self.state.prediction.params_view(),
                self.state.prediction.model_state_view(),
                pred_cache_t,
                pred_grads,
            )

            # Accumulate into hidden state gradient
            for i in range(BATCH * LATENT):
                self.state._grad_hidden[i] = (
                    self.state._grad_hidden[i] + grad_pred_in_ptr[i]
                )

            grad_pred_in_ptr.free()

            # ── Dynamics backward at step k (if k > 0) ──────────────
            if k > 0:
                var dk = k - 1  # Dynamics step that produced hidden state k

                # Compute reward gradient for dynamics step dk
                memset(self.state._grad_dyn_out, 0, BATCH * DYN_OUT)

                # Hidden state gradient portion
                for b in range(BATCH):
                    for i in range(LATENT):
                        # Scale by 0.5 for dual consumers (prediction + next dynamics)
                        self.state._grad_dyn_out[
                            b * DYN_OUT + i
                        ] = self.state._grad_hidden[b * LATENT + i] * Scalar[
                            dtype
                        ](
                            0.5
                        )

                # Reward gradient portion
                for b in range(BATCH):
                    var rew_logits = alloc[Float64](BINS)
                    var rew_target = alloc[Float64](BINS)
                    for i in range(BINS):
                        rew_logits[i] = Float64(
                            (
                                self.state._dyn_reward_logits
                                + dk * BATCH * BINS
                                + b * BINS
                                + i
                            )[]
                        )
                        rew_target[i] = Float64(0.0)
                    var target_rew = Float64(
                        self.state._reward_targets[dk * BATCH + b]
                    )
                    encode_categorical[BINS](
                        target_rew, self.v_min, self.v_max, rew_target
                    )
                    # Softmax gradient
                    var max_r = rew_logits[0]
                    for i in range(1, BINS):
                        if rew_logits[i] > max_r:
                            max_r = rew_logits[i]
                    var sum_exp_r = Float64(0.0)
                    for i in range(BINS):
                        rew_logits[i] = exp(rew_logits[i] - max_r)
                        sum_exp_r += rew_logits[i]
                    for i in range(BINS):
                        var prob = rew_logits[i] / sum_exp_r
                        self.state._grad_dyn_out[
                            b * DYN_OUT + LATENT + i
                        ] = Scalar[dtype](
                            (prob - rew_target[i]) * inv_k * inv_batch
                        )
                    rew_logits.free()
                    rew_target.free()

                # Dynamics backward
                var dyn_cache_offset = dk * BATCH * DYN_CS
                var dyn_cache_t = LayoutTensor[
                    dtype, Layout.row_major(BATCH, DYN_CS), MutAnyOrigin
                ](self.state._dyn_caches + dyn_cache_offset)

                var grad_dyn_out_t = LayoutTensor[
                    dtype, Layout.row_major(BATCH, DYN_OUT_DIM), MutAnyOrigin
                ](self.state._grad_dyn_out)

                memset(self.state._grad_dyn_in, 0, BATCH * DYN_IN)
                var grad_dyn_in_t = LayoutTensor[
                    dtype, Layout.row_major(BATCH, DYN_IN_DIM), MutAnyOrigin
                ](self.state._grad_dyn_in)

                # Scale dynamics gradients by 1/K
                var dyn_scale = Scalar[dtype](1.0 / Float64(K))
                for i in range(BATCH * DYN_OUT):
                    self.state._grad_dyn_out[i] = (
                        self.state._grad_dyn_out[i] * dyn_scale
                    )

                var dyn_grads = self.state.dynamics.grads_view()
                Self.DynNet.backward[BATCH](
                    grad_dyn_out_t,
                    grad_dyn_in_t,
                    self.state.dynamics.params_view(),
                    self.state.dynamics.model_state_view(),
                    dyn_cache_t,
                    dyn_grads,
                )

                # Extract hidden state gradient from dynamics input gradient
                # (first LATENT elements) -> becomes the new grad_hidden for step k-1
                memset(self.state._grad_hidden, 0, BATCH * LATENT)
                for b in range(BATCH):
                    for i in range(LATENT):
                        self.state._grad_hidden[
                            b * LATENT + i
                        ] = self.state._grad_dyn_in[b * DYN_IN + i]

        # ── Representation backward ──────────────────────────────────
        var grad_hidden_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, REP_OUT_DIM), MutAnyOrigin
        ](self.state._grad_hidden)

        memset(self.state._grad_rep_in, 0, BATCH * OBS)
        var grad_rep_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, REP_IN_DIM), MutAnyOrigin
        ](self.state._grad_rep_in)

        var rep_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, REP_CS), MutAnyOrigin
        ](self.state._rep_cache)

        var rep_grads = self.state.representation.grads_view()
        Self.RepNet.backward[BATCH](
            grad_hidden_t,
            grad_rep_in_t,
            self.state.representation.params_view(),
            self.state.representation.model_state_view(),
            rep_cache_t,
            rep_grads,
        )

    def _clip_gradients(mut self):
        """Clip gradients by global norm across all three networks.

        Computes the total gradient norm across representation, dynamics,
        and prediction networks, then scales all gradients if the norm
        exceeds max_grad_norm.
        """
        # Compute total gradient norm squared
        var total_norm_sq = Float64(0.0)

        comptime REP_PS = Self.StateType.RepModel.PARAM_SIZE
        comptime DYN_PS = Self.StateType.DynModel.PARAM_SIZE
        comptime PRED_PS = Self.StateType.PredModel.PARAM_SIZE

        var rep_grads = self.state.representation.grads
        for i in range(REP_PS):
            var g = Float64(rep_grads[i])
            total_norm_sq += g * g

        var dyn_grads = self.state.dynamics.grads
        for i in range(DYN_PS):
            var g = Float64(dyn_grads[i])
            total_norm_sq += g * g

        var pred_grads = self.state.prediction.grads
        for i in range(PRED_PS):
            var g = Float64(pred_grads[i])
            total_norm_sq += g * g

        var total_norm = sqrt(total_norm_sq)

        # Clip if norm exceeds threshold
        if total_norm > self.max_grad_norm:
            var scale = Scalar[dtype](self.max_grad_norm / total_norm)
            for i in range(REP_PS):
                rep_grads[i] = rep_grads[i] * scale
            for i in range(DYN_PS):
                dyn_grads[i] = dyn_grads[i] * scale
            for i in range(PRED_PS):
                pred_grads[i] = pred_grads[i] * scale

    def _scale_batch_hidden(mut self, step: Int):
        """Min-max scale hidden states to [0, 1] for a given unroll step.

        Args:
            step: Unroll step index (0 = from representation, 1..K from dynamics).
        """
        comptime BATCH = Self.Config.batch_size
        comptime LATENT = Self.Config.latent_dim

        var offset = step * BATCH * LATENT
        for b in range(BATCH):
            var b_offset = offset + b * LATENT
            var min_val = Float64((self.state._hidden_states + b_offset)[0])
            var max_val = min_val
            for i in range(1, LATENT):
                var v = Float64((self.state._hidden_states + b_offset + i)[])
                if v < min_val:
                    min_val = v
                if v > max_val:
                    max_val = v

            var delta = max_val - min_val
            if delta > 1e-8:
                for i in range(LATENT):
                    var v = Float64(
                        (self.state._hidden_states + b_offset + i)[]
                    )
                    (self.state._hidden_states + b_offset + i)[] = Scalar[
                        dtype
                    ]((v - min_val) / delta)

    # ══════════════════════════════════════════════════════════════════════
    # Evaluation against opponents
    # ══════════════════════════════════════════════════════════════════════

    def select_action_policy_only(
        mut self,
        obs: List[Scalar[dtype]],
        legal_mask: List[Bool],
    ) -> Int:
        """Select action using raw prediction network policy (no MCTS).

        For AlphaZero mode (TrueGameRules): MCTS on CPU uses dynamics
        network which may be untrained. This method bypasses MCTS and
        uses the prediction network's policy head directly.

        1. Forward prediction network: obs → (policy_logits, value)
        2. Mask illegal actions (set to -inf)
        3. Argmax over remaining legal actions

        Args:
            obs: Canonical observation.
            legal_mask: Legal action mask from environment.

        Returns:
            Best legal action index.
        """
        comptime B: Int = 1
        comptime OBS = Self.Config.obs_dim
        comptime ACT = Self.Config.action_dim
        comptime PRED_IN = Self.StateType.PredModel.IN_DIM
        comptime PRED_OUT_DIM = Self.StateType.PredModel.OUT_DIM

        # Prepare obs tensor
        var obs_ptr = alloc[Scalar[dtype]](OBS)
        for i in range(OBS):
            if i < len(obs):
                obs_ptr[i] = obs[i]
            else:
                obs_ptr[i] = Scalar[dtype](0.0)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(B, PRED_IN), MutAnyOrigin
        ](obs_ptr)

        # Forward prediction network
        var pred_ptr = alloc[Scalar[dtype]](PRED_OUT_DIM)
        memset(pred_ptr, 0, PRED_OUT_DIM)
        var pred_t = LayoutTensor[
            dtype, Layout.row_major(B, PRED_OUT_DIM), MutAnyOrigin
        ](pred_ptr)

        Self.PredNet.forward[B](
            obs_t,
            pred_t,
            self.state.prediction.params_view(),
            self.state.prediction.model_state_view(),
        )

        # Argmax over legal actions (first ACT elements are policy logits)
        var best_action = -1
        var best_logit = Float64(-1e18)
        for a in range(ACT):
            if a < len(legal_mask) and legal_mask[a]:
                var logit = Float64(rebind[Scalar[dtype]](pred_t[0, a]))
                if logit > best_logit:
                    best_logit = logit
                    best_action = a

        obs_ptr.free()
        pred_ptr.free()

        if best_action < 0:
            # Fallback: first legal action
            for a in range(ACT):
                if a < len(legal_mask) and legal_mask[a]:
                    return a
        return best_action

    def evaluate_against[
        E: TwoPlayerDiscreteEnv,
        EvalType: Evaluator,
    ](
        mut self,
        mut env: E,
        mut evaluator: EvalType,
        num_games: Int = 50,
    ) raises -> Tuple[Int, Int, Int]:
        """Play the agent against an evaluator opponent.

        Alternates who plays first (half as P0, half as P1).
        Evaluator tracks its own game state via observe_action().

        Args:
            env: Environment instance (reset for each game).
            evaluator: The opponent strategy.
            num_games: Total games (half as first, half as second).

        Returns:
            (wins, draws, losses) from agent's perspective.
        """
        comptime ACT = Self.Config.action_dim
        comptime OBS = Self.Config.obs_dim

        var wins = 0
        var draws = 0
        var losses = 0

        for game_idx in range(num_games):
            var agent_is_p0 = game_idx < num_games // 2

            _ = env.reset()
            evaluator.reset()

            while env.game_result() == 0:
                var player = env.current_player()
                var is_agent_turn = (player == 0 and agent_is_p0) or (
                    player == 1 and not agent_is_p0
                )

                var legal = env.legal_action_mask()
                var action: Int

                if is_agent_turn:
                    # Agent plays
                    var obs = List[Scalar[dtype]](capacity=OBS)
                    var obs_raw = env.get_obs_list()
                    for i in range(OBS):
                        if i < len(obs_raw):
                            obs.append(Scalar[dtype](obs_raw[i]))
                        else:
                            obs.append(Scalar[dtype](0.0))

                    # Dispatch: policy-only for AlphaZero, MCTS for MuZero
                    comptime USE_RULES = Self.Config.Search.NEEDS_GAME_STATE
                    if USE_RULES:
                        # AlphaZero: use raw policy network (no MCTS on CPU)
                        action = self.select_action_policy_only(obs, legal)
                    else:
                        # MuZero: use MCTS with learned dynamics
                        var result = self.select_action(obs, training=False)
                        action = result[0]

                    if action < 0 or action >= ACT or not legal[action]:
                        action = -1
                        for a in range(ACT):
                            if legal[a]:
                                action = a
                                break
                else:
                    # Evaluator plays
                    action = evaluator.select_action(legal, ACT)
                    if action < 0 or action >= ACT or not legal[action]:
                        action = -1
                        for a in range(ACT):
                            if legal[a]:
                                action = a
                                break

                if action >= 0:
                    # Both agent and evaluator observe the action
                    evaluator.observe_action(action, player)
                    _ = env.step(env.action_from_index(action))

            var result = env.game_result()
            if result == 1:
                if agent_is_p0:
                    wins += 1
                else:
                    losses += 1
            elif result == 2:
                if agent_is_p0:
                    losses += 1
                else:
                    wins += 1
            else:
                draws += 1

        return (wins, draws, losses)

    def print_eval[
        E: TwoPlayerDiscreteEnv,
        EvalType: Evaluator,
    ](mut self, mut env: E, mut evaluator: EvalType, num_games: Int = 50,):
        """Evaluate and print results against one evaluator."""
        var r = self.evaluate_against[E, EvalType](env, evaluator, num_games)
        print(
            "  vs",
            evaluator.name(),
            "| W:",
            r[0],
            "D:",
            r[1],
            "L:",
            r[2],
            "| Win%:",
            r[0] * 100 // num_games,
            "Draw%:",
            r[1] * 100 // num_games,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Checkpointing
    # ══════════════════════════════════════════════════════════════════════

    def save_checkpoint(self, path: String) raises:
        """Save all three networks + metadata to a checkpoint file.

        Saves representation, dynamics, and prediction network params +
        optimizer states, plus training counters.

        Args:
            path: File path for the checkpoint.
        """
        comptime REP_PS = Self.StateType.RepModel.PARAM_SIZE
        comptime DYN_PS = Self.StateType.DynModel.PARAM_SIZE
        comptime PRED_PS = Self.StateType.PredModel.PARAM_SIZE
        comptime TOTAL_PS = REP_PS + DYN_PS + PRED_PS
        comptime STATE_PER = Self.Config.OptType.STATE_PER_PARAM
        comptime TOTAL_SS = TOTAL_PS * STATE_PER

        var content = write_checkpoint_header(
            "muzero_agent", TOTAL_PS, TOTAL_SS
        )
        content += self.state.representation.write_sections("rep_")
        content += self.state.dynamics.write_sections("dyn_")
        content += self.state.prediction.write_sections("pred_")

        var metadata = List[String]()
        metadata.append("config_name=" + Self.Config.NAME)
        metadata.append("train_step_count=" + String(self.train_step_count))
        metadata.append("total_steps=" + String(self.total_steps))
        metadata.append("gamma=" + String(self.gamma))
        metadata.append("v_min=" + String(self.v_min))
        metadata.append("v_max=" + String(self.v_max))
        metadata.append("temperature=" + String(self.temperature))
        content += write_metadata_section(metadata)

        save_checkpoint_file(path, content)

    def load_checkpoint(mut self, path: String) raises:
        """Load all three networks + metadata from a checkpoint file.

        Args:
            path: File path to load from.
        """
        var content = read_checkpoint_file(path)
        _ = parse_checkpoint_header(content)

        self.state.representation.read_sections(content, "rep_")
        self.state.dynamics.read_sections(content, "dyn_")
        self.state.prediction.read_sections(content, "pred_")

        var metadata = read_metadata_section(content)

        var steps_str = get_metadata_value(metadata, "train_step_count")
        if steps_str.byte_length() > 0:
            self.train_step_count = Int(atol(steps_str))

        var total_str = get_metadata_value(metadata, "total_steps")
        if total_str.byte_length() > 0:
            self.total_steps = Int(atol(total_str))

        var gamma_str = get_metadata_value(metadata, "gamma")
        if gamma_str.byte_length() > 0:
            self.gamma = atof(gamma_str)

        var vmin_str = get_metadata_value(metadata, "v_min")
        if vmin_str.byte_length() > 0:
            self.v_min = atof(vmin_str)

        var vmax_str = get_metadata_value(metadata, "v_max")
        if vmax_str.byte_length() > 0:
            self.v_max = atof(vmax_str)

    # ══════════════════════════════════════════════════════════════════════
    # Main Training Loop
    # ══════════════════════════════════════════════════════════════════════

    def train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        total_timesteps: Int = 500000,
        train_every: Int = 1,
        seed_episodes: Int = 10,
        print_every: Int = 10,
        use_reanalyze: Bool = False,
        warmup_steps: Int = 1000,
        logger: Optional[UnsafePointer[Self.L, MutAnyOrigin]] = None,
        diag_every: Int = 500,
        log_episode_every: Int = 10,
    ) raises -> TrainingMetrics:
        """Train MuZero on a discrete action environment.

        Alternates between self-play (with MCTS) and training from replay.
        Optionally uses Reanalyze to refresh old MCTS targets with the
        latest networks for improved sample efficiency.

        Diagnostic ``logger`` emits the same dashboard keys as
        ``train_selfplay_cpu`` (policy_ce / value_mse / value_mean /
        value_target_mean / loss / entropy / target_max_prob /
        target_entropy / param_norm / grad_param_norm) every
        ``diag_every`` SGD steps, plus per-episode ``episode_reward``
        and ``episode_length`` curves on a step-count timeline.

        Args:
            env: Environment implementing BoxDiscreteActionEnv.
            total_timesteps: Total environment steps (default: 500K).
            train_every: Steps between training updates (default: 1).
            seed_episodes: Random exploration episodes (default: 10).
            print_every: Episodes between progress prints (default: 10).
            use_reanalyze: Enable MuZero Reanalyze (default: False).
            warmup_steps: Random exploration steps before training (default: 1000).
            logger: Optional ``RemoteLogger`` for dashboard curves.
            diag_every: SGD steps between dashboard dumps (default: 500).
            log_episode_every: Episodes between dashboard dumps of the
                per-episode samples (``episode_reward`` /
                ``episode_length`` / ``temperature``). Each emitted
                point is the most-recent episode's value — matches the
                printed line exactly. Default: 10.

        Returns:
            TrainingMetrics with episode rewards and statistics.
        """
        # Wire up the logger so the diag block in ``update()`` fires.
        self.logger = logger
        self.diag_every = diag_every

        var metrics = TrainingMetrics(algorithm_name="MuZero")
        var episode_reward = Float64(0.0)
        var episode_steps = 0
        var episode_count = 0
        var total_env_steps = 0

        # ── Seed with random episodes ────────────────────────────────
        _ = env.reset()
        for _ in range(seed_episodes):
            self.reset_episode()
            var done = False
            while not done:
                var obs = _to_dtype_list(env.get_obs_list())
                var action = Int(
                    random_float64(0.0, Float64(Self.Config.action_dim))
                )
                if action >= Self.Config.action_dim:
                    action = Self.Config.action_dim - 1

                var result = env.step_obs(action)
                var reward = Float64(result[1])
                done = result[2]

                # Store with uniform policy (random exploration)
                var uniform_policy = InlineArray[
                    Float64, Self.Config.action_dim
                ](uninitialized=True)
                for a in range(Self.Config.action_dim):
                    uniform_policy[a] = 1.0 / Float64(Self.Config.action_dim)

                self.store_transition(
                    obs, action, reward, uniform_policy, Float64(0.0), done
                )
                total_env_steps += 1
                if done:
                    _ = env.reset()

        # ── Main training loop ───────────────────────────────────────
        _ = env.reset()
        self.reset_episode()

        for step in range(total_timesteps):
            var obs = _to_dtype_list(env.get_obs_list())

            # Select action
            var action: Int
            var policy: InlineArray[Float64, Self.Config.action_dim]
            var root_value: Float64

            if total_env_steps < warmup_steps:
                action = Int(
                    random_float64(0.0, Float64(Self.Config.action_dim))
                )
                if action >= Self.Config.action_dim:
                    action = Self.Config.action_dim - 1
                policy = InlineArray[Float64, Self.Config.action_dim](
                    uninitialized=True
                )
                for a in range(Self.Config.action_dim):
                    policy[a] = 1.0 / Float64(Self.Config.action_dim)
                root_value = Float64(0.0)
            else:
                var result = self.select_action(obs, training=True)
                action = result[0]
                policy = result[1]
                root_value = result[2]

            # Environment step
            var env_result = env.step_obs(action)
            var reward = Float64(env_result[1])
            var done = Bool(env_result[2])

            self.store_transition(obs, action, reward, policy, root_value, done)
            episode_reward += reward
            episode_steps += 1
            total_env_steps += 1
            self.total_steps += 1

            if done:
                episode_count += 1
                metrics.log_episode[dtype](
                    episode_count,
                    Scalar[dtype](episode_reward),
                    episode_steps,
                    0.0,
                )

                # Per-episode dashboard curves — log every
                # ``log_episode_every`` episodes the most-recent episode's
                # values (point sample). Matches the printed line exactly
                # so dashboard and console stay in sync.
                if (
                    self.logger
                    and log_episode_every > 0
                    and episode_count % log_episode_every == 0
                ):
                    try:
                        self.logger.value()[].log_scalar(
                            "episode_reward",
                            episode_reward,
                            self.total_steps,
                        )
                        self.logger.value()[].log_scalar(
                            "episode_length",
                            Float64(episode_steps),
                            self.total_steps,
                        )
                        self.logger.value()[].log_scalar(
                            "temperature",
                            self.temperature,
                            self.total_steps,
                        )
                    except:
                        pass

                if episode_count % print_every == 0:
                    clear_progress_bar()
                    print(
                        "Episode "
                        + String(episode_count)
                        + " | Reward: "
                        + String(Int(episode_reward))
                        + " | Steps: "
                        + String(episode_steps)
                        + " | Train: "
                        + String(self.train_step_count)
                        + " | Buffer: "
                        + String(self.state.buffer.len())
                        + " | Temp: "
                        + String(self.temperature)
                    )

                episode_reward = 0.0
                episode_steps = 0
                _ = env.reset()
                self.reset_episode()

            # Train
            if step % train_every == 0 and self.state.is_ready():
                _ = self.update(use_reanalyze=use_reanalyze)

                # Dashboard diagnostics (gated by logger + diag_every).
                if Bool(self.logger) and (
                    self.diag_every <= 0
                    or self.train_step_count % self.diag_every == 0
                ):
                    try:
                        self._log_cpu_diag()
                    except:
                        pass

            # Temperature schedule — muzero-general step schedule
            # (games/cartpole.py:86-99). Floor 0.25, never fully greedy:
            #   frac < 0.5  → 1.0   (full exploration)
            #   frac < 0.75 → 0.5   (moderate)
            #   else        → 0.25  (low but non-zero)
            # Linear-decay-to-0.01 pre-fix collapsed to greedy at ~40% of
            # training horizon → no exploration once policy committed →
            # MuZero CartPole stuck at reward 7-8. See Bug E follow-up
            # (2026-05-04 temperature audit).
            if self.temperature_decay_steps > 0:
                var _frac = Float64(self.total_steps) / Float64(
                    self.temperature_decay_steps
                )
                if _frac < 0.5:
                    self.temperature = 1.0
                elif _frac < 0.75:
                    self.temperature = 0.5
                else:
                    self.temperature = 0.25

            # Progress bar
            if step % 100 == 0:
                print_progress_bar(
                    step,
                    total_timesteps,
                    self.train_step_count,
                    "MuZero",
                )

        return metrics

    # ══════════════════════════════════════════════════════════════════════
    # Arena (new-vs-best via shared MCTS) — accept-or-revert each iter
    # ══════════════════════════════════════════════════════════════════════

    def arena_compare[
        E: TwoPlayerDiscreteEnv,
        OriginR: MutOrigin,
        OriginD: MutOrigin,
        OriginP: MutOrigin,
    ](
        mut self,
        mut env: E,
        mut prev_rep: UnsafePointer[Scalar[dtype], OriginR],
        mut prev_dyn: UnsafePointer[Scalar[dtype], OriginD],
        mut prev_pred: UnsafePointer[Scalar[dtype], OriginP],
        num_games: Int = 40,
        threshold: Float64 = 0.55,
    ) raises -> Bool:
        """Play current ("new") model vs ``prev_*`` ("best") model.

        Each game alternates which side plays which model. Action
        selection routes through ``_mcts_search_visits_cpu`` (the same
        shared planner self-play uses) with ``add_noise=False`` and
        argmax over visit counts — matches the production eval path so
        arena scores reflect actual playing strength.

        Scoring (Elo-style, draws excluded — mirrors AZ
        ``alphazero.mojo:arena_compare`` exactly):
            decisive = new_wins + old_wins
            accept   = new_wins / decisive ≥ threshold

        On reject, this method restores all three networks' params
        in-place from ``prev_*``. Optimizer state + step counter are
        the caller's responsibility (the self-play loop snapshots them
        alongside ``prev_*`` and restores on reject).

        Args:
            env: TTT/Connect4-style two-player env, reset each game.
            prev_rep / prev_dyn / prev_pred: previous-best param
                snapshots. Lengths must match
                ``Self.Config.{Rep,Dyn,Pred}Model.PARAM_SIZE``.
            num_games: Games played (half as P0, half as P1).
            threshold: Score fraction required to accept new.

        Returns:
            True if new model accepted, False if reverted.
        """
        comptime ACT = Self.Config.action_dim
        comptime OBS = Self.Config.obs_dim
        comptime REP_PS = Self.Config.RepModel.PARAM_SIZE
        comptime DYN_PS = Self.Config.DynModel.PARAM_SIZE
        comptime PRED_PS = Self.Config.PredModel.PARAM_SIZE

        # Snapshot the "new" params so we can swap freely during play
        # and decide accept/revert at the end without losing them.
        var new_rep: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](REP_PS)
        var new_dyn: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](DYN_PS)
        var new_pred: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](PRED_PS)
        for i in range(REP_PS):
            new_rep[i] = self.state.representation.params[i]
        for i in range(DYN_PS):
            new_dyn[i] = self.state.dynamics.params[i]
        for i in range(PRED_PS):
            new_pred[i] = self.state.prediction.params[i]

        var visits: UnsafePointer[Int, MutAnyOrigin] = alloc[Int](ACT)
        var new_wins = 0
        var draws = 0

        for game_idx in range(num_games):
            var new_is_p0 = game_idx < num_games // 2
            _ = env.reset()

            while env.game_result() == 0:
                var player = env.current_player()
                var is_new = (player == 0 and new_is_p0) or (
                    player == 1 and not new_is_p0
                )

                # Hot-swap all three networks' params to whichever
                # snapshot is on move. This is cheap relative to the
                # MCTS forward pass (TTT param footprint is small).
                if is_new:
                    for i in range(REP_PS):
                        self.state.representation.params[i] = new_rep[i]
                    for i in range(DYN_PS):
                        self.state.dynamics.params[i] = new_dyn[i]
                    for i in range(PRED_PS):
                        self.state.prediction.params[i] = new_pred[i]
                else:
                    for i in range(REP_PS):
                        self.state.representation.params[i] = prev_rep[i]
                    for i in range(DYN_PS):
                        self.state.dynamics.params[i] = prev_dyn[i]
                    for i in range(PRED_PS):
                        self.state.prediction.params[i] = prev_pred[i]

                var legal = env.legal_action_mask()
                var obs_raw = env.get_obs_list()
                var obs = List[Scalar[dtype]](capacity=OBS)
                for i in range(OBS):
                    if i < len(obs_raw):
                        obs.append(Scalar[dtype](obs_raw[i]))
                    else:
                        obs.append(Scalar[dtype](0.0))

                # MCTS with no noise → argmax visit counts.
                _ = self._mcts_search_visits_cpu(
                    obs, legal, visits, add_noise=False
                )
                var action: Int = -1
                var best_v: Int = -1
                for a in range(ACT):
                    if visits[a] > best_v and a < len(legal) and legal[a]:
                        best_v = visits[a]
                        action = a
                if action < 0:
                    for a in range(ACT):
                        if a < len(legal) and legal[a]:
                            action = a
                            break
                _ = env.step(env.action_from_index(action))

            var result = env.game_result()
            if result == 3:
                draws += 1
            elif result == 1:
                if new_is_p0:
                    new_wins += 1
            elif result == 2:
                if not new_is_p0:
                    new_wins += 1

        # Restore "new" params so the agent ends in the "new" state
        # if accepted (caller may still revert on the outside).
        for i in range(REP_PS):
            self.state.representation.params[i] = new_rep[i]
        for i in range(DYN_PS):
            self.state.dynamics.params[i] = new_dyn[i]
        for i in range(PRED_PS):
            self.state.prediction.params[i] = new_pred[i]

        new_rep.free()
        new_dyn.free()
        new_pred.free()
        visits.free()

        # Elo-style scoring: draws excluded from the denominator.
        var old_wins = num_games - new_wins - draws
        var decisive = new_wins + old_wins
        var accepted: Bool
        if decisive == 0:
            accepted = False
        else:
            var win_rate = Float64(new_wins) / Float64(decisive)
            accepted = win_rate >= threshold

        if not accepted:
            for i in range(REP_PS):
                self.state.representation.params[i] = prev_rep[i]
            for i in range(DYN_PS):
                self.state.dynamics.params[i] = prev_dyn[i]
            for i in range(PRED_PS):
                self.state.prediction.params[i] = prev_pred[i]

        return accepted

    # ══════════════════════════════════════════════════════════════════════
    # Self-Play CPU Training (board games — MLP/small configs)
    # ══════════════════════════════════════════════════════════════════════

    def train_selfplay_cpu[
        E: TwoPlayerDiscreteEnv,
        EvalType: Evaluator,
        EvalType2: Evaluator,
        temp_threshold: Int = 5,
    ](
        mut self,
        mut env: E,
        mut evaluator: EvalType,
        mut evaluator2: EvalType2,
        num_iters: Int = 25,
        steps_per_iter: Int = 500,
        train_epochs: Int = 2,
        warmup_iters: Int = 1,
        do_eval: Bool = True,
        do_eval2: Bool = False,
        eval_games: Int = 20,
        do_arena: Bool = False,
        arena_threshold: Float64 = 0.55,
        arena_games: Int = 20,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "muzero.ckpt",
        use_reanalyze: Bool = False,
        reanalyze_warmup: Int = 500,
        reanalyze_interval: Int = 50,
        logger: Optional[UnsafePointer[Self.L, MutAnyOrigin]] = None,
        diag_every: Int = 500,
        verbose: Bool = True,
        use_one_cycle: Bool = False,
        max_episode_length: Int = 64,
    ) raises -> TrainingMetrics:
        """CPU self-play training — signature mirrors ``train_selfplay_gpu``.

        Per iteration:
          1. Collect self-play data via ``_mcts_search_visits_cpu`` until
             ~``steps_per_iter`` env-steps have been played. First
             ``warmup_iters`` iterations use uniform-random legal actions
             with a uniform policy target — fills the buffer cheaply.
             Episodes are flushed to the SequenceReplayBuffer (with the
             parallel ``mcts_policies`` / ``mcts_values`` / ``mcts_to_play``
             targets) via the existing ``store_transition`` helper.
          2. Run ``train_epochs * (buf_size // batch_size)`` ``update()``
             steps. Each ``update()`` does the K-step unrolled forward,
             backward, global-L2 grad clip, and Adam step on all three
             networks. Optional 1cycle LR scaling across the iteration.
          3. ``EvalType`` evaluation (do_eval) + optional ``EvalType2``
             evaluation (do_eval2, e.g. a Minimax oracle for TTT).
          4. Periodic checkpoint via ``save_checkpoint``.

        Diagnostic ``logger`` emits CPU↔GPU MuZero-parity keys
        (policy_ce / value_mse / value_mean / value_target_mean / loss
        / entropy / target_max_prob / target_entropy / param_norm /
        grad_param_norm) every ``diag_every`` SGD steps. Aligns with the
        AZ Phase D dashboard convention so MuZero CPU vs MuZero GPU
        curves can be overlaid for the same kind of bug hunt.

        Parameters:
            E: Two-player discrete env (TicTacToe / ConnectFour CPU).
            EvalType: Evaluator opponent (e.g. RandomOpponent).
            EvalType2: Second evaluator (e.g. MinimaxTicTacToe).
            temp_threshold: Use temp>0 for first N moves, argmax after.
                Comptime so the value is fixed across the iter (matches
                ``train_selfplay_gpu``).

        Args:
            env, evaluator, evaluator2: Standard CPU eval interface.
            num_iters: Outer self-play / train iteration count.
            steps_per_iter: Env-steps to collect per iteration.
            train_epochs: Epochs over the current replay window per iter.
            warmup_iters: Random-play iterations before MCTS starts.
            do_eval / do_eval2: Toggle each evaluator.
            eval_games: Games per evaluator (split half-as-P0, half-as-P1).
            checkpoint_every: Save every N iters (0 disables).
            checkpoint_path: Path for the periodic checkpoint.
            use_reanalyze: If True, ``update()`` refreshes a fraction of
                replay positions through fresh MCTS before each batch —
                gated by ``reanalyze_warmup`` and ``reanalyze_interval``
                (see below).
            reanalyze_warmup: Number of train steps before reanalyze
                first fires. Default 500: the network needs to be at
                least somewhat trained before its re-projection of old
                targets is more informative than the stale-but-grounded
                stored targets. Mirrors EZv2 (gpu_train.mojo:182).
            reanalyze_interval: Refresh every Nth train step after
                warmup. Default 50. Reanalyzing on EVERY step (the
                naive use_reanalyze=True semantic) amplifies any
                network bias by re-projecting every target through the
                same freshly-biased model — confirmed on the
                2026-05-21 MuZero TTT diagnostic.
            logger: Optional ``RemoteLogger`` for dashboard curves.
            diag_every: SGD steps between dashboard dumps.
            verbose: Per-iter summary print.
            use_one_cycle: 1cycle LR schedule across each iter's SGD
                pass — mirrors the AZ CPU loop.
            max_episode_length: Cap on per-episode step count (safety
                net; the env's own done flag normally terminates first).
        """
        comptime ACT = Self.Config.action_dim
        comptime OBS = Self.Config.obs_dim
        comptime BATCH = Self.Config.batch_size
        comptime BINS = Self.Config.num_bins
        comptime PRED_OUT = Self.Config.PRED_OUT
        comptime REP_PS = Self.StateType.RepModel.PARAM_SIZE
        comptime DYN_PS = Self.StateType.DynModel.PARAM_SIZE
        comptime PRED_PS = Self.StateType.PredModel.PARAM_SIZE
        comptime TEMP_THRESH = temp_threshold

        # Wire up the logger so the diag block can fire.
        self.logger = logger
        self.diag_every = diag_every

        # Per-step scratch.
        var visits: UnsafePointer[Int, MutAnyOrigin] = alloc[Int](ACT)

        # ── Arena: best-model tracking (mirrors AZ CPU loop) ─────────
        # Snapshots of params + optimizer state + step counters for all
        # three networks. After each iter's training we run an arena
        # match (new-vs-best); accept → snapshot becomes the new params,
        # reject → revert params + opt state + step counter to the
        # snapshot. Preserves Adam's m/v moments correctly across
        # accept/reject decisions so a rejected run doesn't poison the
        # next iter's update with future-iter momentum.
        # (REP_PS / DYN_PS / PRED_PS already declared above for the
        # param-norm diag block; reuse them here.)
        comptime STATE_PER = Self.Config.OptType.STATE_PER_PARAM
        comptime REP_OS = REP_PS * STATE_PER
        comptime DYN_OS = DYN_PS * STATE_PER
        comptime PRED_OS = PRED_PS * STATE_PER

        var best_rep_params: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](REP_PS)
        var best_dyn_params: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](DYN_PS)
        var best_pred_params: UnsafePointer[
            Scalar[dtype], MutAnyOrigin
        ] = alloc[Scalar[dtype]](PRED_PS)
        var best_rep_opt: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](REP_OS if REP_OS > 0 else 1)
        var best_dyn_opt: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](DYN_OS if DYN_OS > 0 else 1)
        var best_pred_opt: UnsafePointer[Scalar[dtype], MutAnyOrigin] = alloc[
            Scalar[dtype]
        ](PRED_OS if PRED_OS > 0 else 1)
        var best_rep_step = self.state.representation.step_num
        var best_dyn_step = self.state.dynamics.step_num
        var best_pred_step = self.state.prediction.step_num
        for i in range(REP_PS):
            best_rep_params[i] = self.state.representation.params[i]
        for i in range(DYN_PS):
            best_dyn_params[i] = self.state.dynamics.params[i]
        for i in range(PRED_PS):
            best_pred_params[i] = self.state.prediction.params[i]
        for i in range(REP_OS):
            best_rep_opt[i] = self.state.representation.optimizer_state[i]
        for i in range(DYN_OS):
            best_dyn_opt[i] = self.state.dynamics.optimizer_state[i]
        for i in range(PRED_OS):
            best_pred_opt[i] = self.state.prediction.optimizer_state[i]
        var arena_accepts = 0
        var arena_rejects = 0

        var metrics = TrainingMetrics(algorithm_name="MuZero-CPU")

        for iter in range(num_iters):
            var use_mcts = iter >= warmup_iters

            var p0_wins = 0
            var p1_wins = 0
            var iter_draws = 0
            var games_completed = 0
            var iter_step_count = 0

            # ── 1. Collect self-play data ──────────────────────────
            while iter_step_count < steps_per_iter:
                _ = env.reset()
                self.reset_episode()
                var step_count = 0

                while (
                    env.game_result() == 0 and step_count < max_episode_length
                ):
                    var legal = env.legal_action_mask()
                    var obs_raw = env.get_obs_list()
                    var obs = List[Scalar[dtype]](capacity=OBS)
                    for i in range(OBS):
                        if i < len(obs_raw):
                            obs.append(Scalar[dtype](obs_raw[i]))
                        else:
                            obs.append(Scalar[dtype](0.0))

                    var player = env.current_player()

                    var chosen_action: Int = -1
                    var root_value: Float64 = 0.0
                    var policy = InlineArray[Float64, ACT](uninitialized=True)
                    for a in range(ACT):
                        policy[a] = 0.0

                    if use_mcts:
                        root_value = self._mcts_search_visits_cpu(
                            obs, legal, visits, add_noise=True
                        )

                        var total = 0
                        for a in range(ACT):
                            total += visits[a]

                        if total > 0:
                            for a in range(ACT):
                                policy[a] = Float64(visits[a]) / Float64(total)
                        else:
                            var n_leg = 0
                            for a in range(ACT):
                                if a < len(legal) and legal[a]:
                                    n_leg += 1
                            if n_leg > 0:
                                for a in range(ACT):
                                    if a < len(legal) and legal[a]:
                                        policy[a] = 1.0 / Float64(n_leg)

                        if (
                            step_count < TEMP_THRESH
                            and self.temperature > 0.01
                            and total > 0
                        ):
                            # Temperature-1 sampling over visit policy
                            # (rough proxy for muzero-general's temp=1
                            # over the first ``temp_threshold`` moves).
                            var rnd = random_float64()
                            var acc: Float64 = 0.0
                            for a in range(ACT):
                                acc += policy[a]
                                if rnd <= acc:
                                    chosen_action = a
                                    break
                            if chosen_action < 0:
                                for a in range(ACT):
                                    if a < len(legal) and legal[a]:
                                        chosen_action = a
                                        break
                        else:
                            var best_v = -1
                            for a in range(ACT):
                                if visits[a] > best_v:
                                    best_v = visits[a]
                                    chosen_action = a
                            if chosen_action < 0 or not (
                                chosen_action < len(legal)
                                and legal[chosen_action]
                            ):
                                for a in range(ACT):
                                    if a < len(legal) and legal[a]:
                                        chosen_action = a
                                        break
                    else:
                        # Warmup — uniform-random legal action with
                        # uniform-over-legal policy target. Matches the
                        # AZ CPU warmup so the buffer fills with
                        # outcome-grounded data (vs AZ Phase D Bug 3
                        # where GPU warmup stored nothing).
                        var n_leg = 0
                        for a in range(ACT):
                            if a < len(legal) and legal[a]:
                                n_leg += 1
                        if n_leg == 0:
                            break
                        for a in range(ACT):
                            if a < len(legal) and legal[a]:
                                policy[a] = 1.0 / Float64(n_leg)
                        var pick = Int(random_float64() * Float64(n_leg))
                        if pick >= n_leg:
                            pick = n_leg - 1
                        var c = 0
                        for a in range(ACT):
                            if a < len(legal) and legal[a]:
                                if c == pick:
                                    chosen_action = a
                                    break
                                c += 1

                    # Step env, capture reward + done.
                    var step_result = env.step(
                        env.action_from_index(chosen_action)
                    )
                    var reward = Float64(step_result[1])
                    var done = Bool(step_result[2])

                    self.store_transition(
                        obs,
                        chosen_action,
                        reward,
                        policy,
                        root_value,
                        done,
                        to_play=player,
                    )

                    step_count += 1

                # Game outcome stats.
                var result = env.game_result()
                games_completed += 1
                if result == 1:
                    p0_wins += 1
                elif result == 2:
                    p1_wins += 1
                elif result == 3:
                    iter_draws += 1
                self.total_steps += step_count
                iter_step_count += step_count

            # ── 2. Train ───────────────────────────────────────────
            var avg_loss_iter: Float64 = 0.0
            var num_train_steps_this_iter: Int = 0
            if use_mcts and self.state.is_ready():
                var steps_per_epoch = self.state.buffer.len() // BATCH
                if steps_per_epoch < 1:
                    steps_per_epoch = 1
                var num_train_steps = train_epochs * steps_per_epoch
                num_train_steps_this_iter = num_train_steps

                var sum_loss: Float64 = 0.0

                for s_idx in range(num_train_steps):
                    if use_one_cycle:
                        var sc = OneCycleSchedule[].lr_scale_at(
                            s_idx, num_train_steps
                        )
                        self.state.representation.set_lr_scale(sc)
                        self.state.dynamics.set_lr_scale(sc)
                        self.state.prediction.set_lr_scale(sc)

                    # Reanalyze schedule (mirrors EZv2's pattern at
                    # `efficient_zero_v2/gpu_train.mojo:664`): hold off
                    # until ``reanalyze_warmup`` train steps have elapsed
                    # — so the network has learned something worth
                    # projecting old positions through — then fire only
                    # every ``reanalyze_interval`` train steps so each
                    # refresh sees a meaningfully-updated network. The
                    # 2026-05-21 diagnostic showed that reanalyzing on
                    # EVERY step actively amplified a small perspective
                    # bias because every old target got re-projected
                    # through a freshly-biased network with no slack.
                    var step_idx_global = self.train_step_count
                    var do_reanalyze = (
                        use_reanalyze
                        and step_idx_global >= reanalyze_warmup
                        and (
                            reanalyze_interval <= 1
                            or step_idx_global % reanalyze_interval == 0
                        )
                    )
                    var loss = self.update(use_reanalyze=do_reanalyze)
                    sum_loss += loss

                    # ── Diagnostic logging (after update()) ────────
                    # Emit the same dashboard keys MuZero GPU does so
                    # CPU↔GPU curves overlay cleanly. We compute them
                    # from the k=0 slice of the scratch buffers that
                    # ``update()`` has just populated (these are still
                    # live until the next ``sample_batch_with_targets``).
                    if Bool(self.logger) and (
                        diag_every <= 0
                        or self.train_step_count % diag_every == 0
                    ):
                        try:
                            self._log_cpu_diag()
                        except:
                            pass

                avg_loss_iter = sum_loss / Float64(num_train_steps)
                # Reset LR scale after the iteration.
                self.state.representation.set_lr_scale(1.0)
                self.state.dynamics.set_lr_scale(1.0)
                self.state.prediction.set_lr_scale(1.0)

            # ── 3a. Eval vs evaluator ───────────────────────────────
            var eval_w = 0
            var eval_d = 0
            var eval_l = 0
            if do_eval and eval_games > 0:
                var r = self.evaluate_against[E, EvalType](
                    env, evaluator, eval_games
                )
                eval_w = r[0]
                eval_d = r[1]
                eval_l = r[2]

            # ── 3b. Eval vs evaluator2 ──────────────────────────────
            var eval2_w = 0
            var eval2_d = 0
            var eval2_l = 0
            if do_eval2 and eval_games > 0:
                var r2 = self.evaluate_against[E, EvalType2](
                    env, evaluator2, eval_games
                )
                eval2_w = r2[0]
                eval2_d = r2[1]
                eval2_l = r2[2]

            # ── 4. Arena vs best-so-far ─────────────────────────────
            # Only runs after warmup ends. Skipped during the random
            # warmup iter since the buffer is filled with uniform-policy
            # data and the network hasn't been updated.
            var arena_msg = String("")
            if do_arena and use_mcts:
                # Snapshot current ("new") state in case we accept.
                var new_rep: UnsafePointer[
                    Scalar[dtype], MutAnyOrigin
                ] = alloc[Scalar[dtype]](REP_PS)
                var new_dyn: UnsafePointer[
                    Scalar[dtype], MutAnyOrigin
                ] = alloc[Scalar[dtype]](DYN_PS)
                var new_pred: UnsafePointer[
                    Scalar[dtype], MutAnyOrigin
                ] = alloc[Scalar[dtype]](PRED_PS)
                var new_rep_opt: UnsafePointer[
                    Scalar[dtype], MutAnyOrigin
                ] = alloc[Scalar[dtype]](REP_OS if REP_OS > 0 else 1)
                var new_dyn_opt: UnsafePointer[
                    Scalar[dtype], MutAnyOrigin
                ] = alloc[Scalar[dtype]](DYN_OS if DYN_OS > 0 else 1)
                var new_pred_opt: UnsafePointer[
                    Scalar[dtype], MutAnyOrigin
                ] = alloc[Scalar[dtype]](PRED_OS if PRED_OS > 0 else 1)
                for i in range(REP_PS):
                    new_rep[i] = self.state.representation.params[i]
                for i in range(DYN_PS):
                    new_dyn[i] = self.state.dynamics.params[i]
                for i in range(PRED_PS):
                    new_pred[i] = self.state.prediction.params[i]
                for i in range(REP_OS):
                    new_rep_opt[i] = self.state.representation.optimizer_state[
                        i
                    ]
                for i in range(DYN_OS):
                    new_dyn_opt[i] = self.state.dynamics.optimizer_state[i]
                for i in range(PRED_OS):
                    new_pred_opt[i] = self.state.prediction.optimizer_state[
                        i
                    ]
                var new_rep_step = self.state.representation.step_num
                var new_dyn_step = self.state.dynamics.step_num
                var new_pred_step = self.state.prediction.step_num

                var accepted = self.arena_compare[
                    E, MutAnyOrigin, MutAnyOrigin, MutAnyOrigin
                ](
                    env,
                    best_rep_params,
                    best_dyn_params,
                    best_pred_params,
                    arena_games,
                    arena_threshold,
                )
                if accepted:
                    # New params already in place; swap snapshot.
                    for i in range(REP_PS):
                        best_rep_params[i] = new_rep[i]
                    for i in range(DYN_PS):
                        best_dyn_params[i] = new_dyn[i]
                    for i in range(PRED_PS):
                        best_pred_params[i] = new_pred[i]
                    for i in range(REP_OS):
                        best_rep_opt[i] = new_rep_opt[i]
                    for i in range(DYN_OS):
                        best_dyn_opt[i] = new_dyn_opt[i]
                    for i in range(PRED_OS):
                        best_pred_opt[i] = new_pred_opt[i]
                    best_rep_step = new_rep_step
                    best_dyn_step = new_dyn_step
                    best_pred_step = new_pred_step
                    arena_accepts += 1
                    arena_msg = (
                        " | ARENA ACCEPT ("
                        + String(arena_accepts)
                        + "/"
                        + String(arena_accepts + arena_rejects)
                        + ")"
                    )
                else:
                    # ``arena_compare`` reverted params already; mirror
                    # AZ by also reverting optimizer state + step
                    # counters so Adam's m/v moments don't carry
                    # "future-iter" momentum into the next iter under
                    # the restored params.
                    for i in range(REP_OS):
                        self.state.representation.optimizer_state[
                            i
                        ] = best_rep_opt[i]
                    for i in range(DYN_OS):
                        self.state.dynamics.optimizer_state[
                            i
                        ] = best_dyn_opt[i]
                    for i in range(PRED_OS):
                        self.state.prediction.optimizer_state[
                            i
                        ] = best_pred_opt[i]
                    self.state.representation.step_num = best_rep_step
                    self.state.dynamics.step_num = best_dyn_step
                    self.state.prediction.step_num = best_pred_step
                    arena_rejects += 1
                    arena_msg = (
                        " | arena reject ("
                        + String(arena_accepts)
                        + "/"
                        + String(arena_accepts + arena_rejects)
                        + ")"
                    )

                new_rep.free()
                new_dyn.free()
                new_pred.free()
                new_rep_opt.free()
                new_dyn_opt.free()
                new_pred_opt.free()

            # ── 5. Checkpoint ───────────────────────────────────────
            if checkpoint_every > 0 and (iter + 1) % checkpoint_every == 0:
                self.save_checkpoint(checkpoint_path)

            if verbose:
                var eval_part = String("")
                if do_eval and eval_games > 0:
                    eval_part = (
                        " | vs "
                        + evaluator.name()
                        + " W/D/L="
                        + String(eval_w)
                        + "/"
                        + String(eval_d)
                        + "/"
                        + String(eval_l)
                    )
                var eval2_part = String("")
                if do_eval2 and eval_games > 0:
                    eval2_part = (
                        " | vs "
                        + evaluator2.name()
                        + " W/D/L="
                        + String(eval2_w)
                        + "/"
                        + String(eval2_d)
                        + "/"
                        + String(eval2_l)
                    )
                print(
                    "[iter ",
                    iter + 1,
                    "/",
                    num_iters,
                    "] steps=",
                    iter_step_count,
                    " games=",
                    games_completed,
                    " buf=",
                    self.state.buffer.len(),
                    " P0w/P1w/Dr=",
                    p0_wins,
                    "/",
                    p1_wins,
                    "/",
                    iter_draws,
                    " loss=",
                    avg_loss_iter,
                    " train+=",
                    num_train_steps_this_iter,
                    eval_part,
                    eval2_part,
                    arena_msg,
                    sep="",
                )

        if do_arena:
            print(
                "Arena: accepted",
                arena_accepts,
                "/ rejected",
                arena_rejects,
            )
        if checkpoint_every > 0:
            self.save_checkpoint(checkpoint_path)

        # Detach logger before returning.
        self.logger = None

        visits.free()
        best_rep_params.free()
        best_dyn_params.free()
        best_pred_params.free()
        best_rep_opt.free()
        best_dyn_opt.free()
        best_pred_opt.free()

        return metrics^

    # ══════════════════════════════════════════════════════════════════════
    # GPU Training
    # ══════════════════════════════════════════════════════════════════════

    def update_gpu[
        N_ENVS_P: Int = 64,
        PER_ENV_CAP_P: Int = 1000,
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu: MuZeroGPUState[Self.Config, N_ENVS_P, PER_ENV_CAP_P],
        use_reanalyze: Bool = False,
        use_per: Bool = False,
        per_progress: Float64 = 0.0,
    ) raises -> Float64:
        """Run one GPU-accelerated training step.

        CPU handles: batch sampling, MCTS target extraction, n-step returns,
                     scalar transform, data upload to GPU.
        GPU handles: K-step unrolled forward/backward, gradient computation,
                     optimizer step.

        Args:
            ctx: GPU device context.
            gpu: GPU state with network states and scratch buffers.
            use_reanalyze: Whether to reanalyze old positions.
            use_per: Whether to use prioritized experience replay.
            per_progress: Progress of the prioritized experience replay.

        Returns:
            Total training loss.
        """
        comptime BATCH = Self.Config.batch_size
        comptime K = Self.Config.unroll_steps
        comptime LATENT = Self.Config.latent_dim
        comptime ACT = Self.Config.action_dim
        comptime BINS = Self.Config.num_bins
        comptime OBS = Self.Config.obs_dim
        comptime PRED_OUT = Self.StateType.PRED_OUT
        comptime DYN_IN = Self.StateType.DYN_IN
        comptime DYN_OUT = Self.StateType.DYN_OUT

        # Model dimensions for LayoutTensor compatibility
        comptime REP_IN_DIM = Self.StateType.RepModel.IN_DIM
        comptime REP_OUT_DIM = Self.StateType.RepModel.OUT_DIM
        comptime REP_CS = Self.StateType.RepModel.CACHE_SIZE
        comptime DYN_IN_DIM = Self.StateType.DynModel.IN_DIM
        comptime DYN_OUT_DIM = Self.StateType.DynModel.OUT_DIM
        comptime DYN_CS = Self.StateType.DynModel.CACHE_SIZE
        comptime PRED_IN_DIM = Self.StateType.PredModel.IN_DIM
        comptime PRED_OUT_DIM = Self.StateType.PredModel.OUT_DIM
        comptime PRED_CS = Self.StateType.PredModel.CACHE_SIZE

        comptime PER_ENV_CAP = PER_ENV_CAP_P
        comptime N_ENVS_GPU = N_ENVS_P
        comptime N_TD = Self.Config.td_steps

        # NOTE: a CPU-side `self.reanalyze(num_positions=BATCH//4)` was
        # called here previously when use_reanalyze=True. Removed
        # (2026-05-04) because it operates on the CPU replay buffer
        # (`self.state.buffer`) which `train_gpu`/`train_selfplay_gpu`
        # never populate — they use `gpu.replay`. The actual GPU
        # reanalyze runs further down (per-timestep RepNet+PredNet
        # forward on target nets, decode_value_dist_kernel, value-buf
        # overwrite) when `use_reanalyze=True` is passed through.

        # ── Step 1: GPU-native sampling (sequences + MCTS targets) ───
        # Sample directly from GPU replay buffer + parallel MCTS buffers
        comptime BATCH_BLOCKS_S = (BATCH + TPB - 1) // TPB

        # Prepare LayoutTensors for the sampling kernel
        var buf_obs_t = LayoutTensor[
            dtype,
            Layout.row_major(N_ENVS_GPU * PER_ENV_CAP * OBS),
            MutAnyOrigin,
        ](gpu.replay.obs_buf.unsafe_ptr())
        var buf_act_t = LayoutTensor[
            dtype,
            Layout.row_major(N_ENVS_GPU * PER_ENV_CAP * ACT),
            MutAnyOrigin,
        ](gpu.replay.actions_buf.unsafe_ptr())
        var buf_rew_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS_GPU * PER_ENV_CAP), MutAnyOrigin
        ](gpu.replay.rewards_buf.unsafe_ptr())
        var buf_done_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS_GPU * PER_ENV_CAP), MutAnyOrigin
        ](gpu.replay.dones_buf.unsafe_ptr())
        # buf_terminations: terminated-only (excludes truncation). Sample
        # kernel uses this for the OUTPUT batch_dones so n-step bootstrap
        # is preserved on time-limit truncation. Boundary check still uses
        # buf_done_t. See gpu_sequence_replay_buffer.mojo:1-44.
        var buf_term_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS_GPU * PER_ENV_CAP), MutAnyOrigin
        ](gpu.replay.terminations_buf.unsafe_ptr())
        var buf_pol_t = LayoutTensor[
            dtype,
            Layout.row_major(N_ENVS_GPU * PER_ENV_CAP * ACT),
            MutAnyOrigin,
        ](gpu.mcts_policy_buf.unsafe_ptr())
        var buf_val_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS_GPU * PER_ENV_CAP), MutAnyOrigin
        ](gpu.mcts_value_buf.unsafe_ptr())
        var buf_tp_t = LayoutTensor[
            DType.uint8,
            Layout.row_major(N_ENVS_GPU * PER_ENV_CAP),
            MutAnyOrigin,
        ](gpu.mcts_to_play_buf.unsafe_ptr())

        # All batch tensors are TIME-MAJOR. Window K+N_TD+1 timesteps for
        # full-window data (obs/policies/values/to_play), K+N_TD for
        # per-transition (rewards/dones), K for actions.
        comptime WIN_FULL = K + N_TD + 1
        comptime WIN_TRN = K + N_TD

        var b_obs_t = LayoutTensor[
            dtype, Layout.row_major(WIN_FULL * BATCH * OBS), MutAnyOrigin
        ](gpu.batch_obs_buf.unsafe_ptr())
        var b_act_t = LayoutTensor[
            dtype, Layout.row_major(K * BATCH * ACT), MutAnyOrigin
        ](gpu.batch_actions_buf.unsafe_ptr())
        var b_rew_t = LayoutTensor[
            dtype, Layout.row_major(WIN_TRN * BATCH), MutAnyOrigin
        ](gpu.batch_rewards_buf.unsafe_ptr())
        var b_done_t = LayoutTensor[
            dtype, Layout.row_major(WIN_TRN * BATCH), MutAnyOrigin
        ](gpu.batch_dones_buf.unsafe_ptr())
        var b_pol_t = LayoutTensor[
            dtype, Layout.row_major(WIN_FULL * BATCH * ACT), MutAnyOrigin
        ](gpu.batch_policies_buf.unsafe_ptr())
        var b_val_t = LayoutTensor[
            dtype, Layout.row_major(WIN_FULL * BATCH), MutAnyOrigin
        ](gpu.batch_values_buf.unsafe_ptr())
        var b_tp_t = LayoutTensor[
            DType.uint8, Layout.row_major(WIN_FULL * BATCH), MutAnyOrigin
        ](gpu.batch_to_play_buf.unsafe_ptr())

        var buf_size_s = Scalar[DType.int32](gpu.replay.size)
        var buf_wptr_s = Scalar[DType.int32](gpu.replay.write_idx)
        var sample_seed = Scalar[DType.uint32](
            UInt32(self.train_step_count * 7 + 1)
        )

        @parameter
        @always_inline
        def sample_wrapper(
            bo: LayoutTensor[
                dtype,
                Layout.row_major(N_ENVS_GPU * PER_ENV_CAP * OBS),
                MutAnyOrigin,
            ],
            ba: LayoutTensor[
                dtype,
                Layout.row_major(N_ENVS_GPU * PER_ENV_CAP * ACT),
                MutAnyOrigin,
            ],
            br: LayoutTensor[
                dtype, Layout.row_major(N_ENVS_GPU * PER_ENV_CAP), MutAnyOrigin
            ],
            bd: LayoutTensor[
                dtype, Layout.row_major(N_ENVS_GPU * PER_ENV_CAP), MutAnyOrigin
            ],
            bd_term: LayoutTensor[
                dtype, Layout.row_major(N_ENVS_GPU * PER_ENV_CAP), MutAnyOrigin
            ],
            bp: LayoutTensor[
                dtype,
                Layout.row_major(N_ENVS_GPU * PER_ENV_CAP * ACT),
                MutAnyOrigin,
            ],
            bv: LayoutTensor[
                dtype, Layout.row_major(N_ENVS_GPU * PER_ENV_CAP), MutAnyOrigin
            ],
            btp: LayoutTensor[
                DType.uint8,
                Layout.row_major(N_ENVS_GPU * PER_ENV_CAP),
                MutAnyOrigin,
            ],
            oo: LayoutTensor[
                dtype, Layout.row_major(WIN_FULL * BATCH * OBS), MutAnyOrigin
            ],
            oa: LayoutTensor[
                dtype, Layout.row_major(K * BATCH * ACT), MutAnyOrigin
            ],
            orw: LayoutTensor[
                dtype, Layout.row_major(WIN_TRN * BATCH), MutAnyOrigin
            ],
            od: LayoutTensor[
                dtype, Layout.row_major(WIN_TRN * BATCH), MutAnyOrigin
            ],
            op: LayoutTensor[
                dtype, Layout.row_major(WIN_FULL * BATCH * ACT), MutAnyOrigin
            ],
            ov: LayoutTensor[
                dtype, Layout.row_major(WIN_FULL * BATCH), MutAnyOrigin
            ],
            otp: LayoutTensor[
                DType.uint8, Layout.row_major(WIN_FULL * BATCH), MutAnyOrigin
            ],
            bsz: Scalar[DType.int32],
            bwi: Scalar[DType.int32],
            seed: Scalar[DType.uint32],
        ):
            sample_seq_with_targets_kernel[
                BATCH, K, N_TD, N_ENVS_GPU, PER_ENV_CAP, OBS, ACT, dtype
            ](
                bo,
                ba,
                br,
                bd,
                bd_term,
                bp,
                bv,
                btp,
                oo,
                oa,
                orw,
                od,
                op,
                ov,
                otp,
                bsz,
                bwi,
                seed,
            )

        if use_per:
            # CPU sum-tree sampling → fills (env, start, weight) host buffers
            # then uploads to GPU. The priority kernel gathers using the
            # pre-sampled indices instead of running internal RNG.
            self._per_sample_indices[N_ENVS_P, PER_ENV_CAP_P](
                ctx, gpu, per_progress
            )
            var per_envs_t = LayoutTensor[
                DType.int32, Layout.row_major(BATCH), MutAnyOrigin
            ](gpu.per_sampled_envs_buf.unsafe_ptr())
            var per_starts_t = LayoutTensor[
                DType.int32, Layout.row_major(BATCH), MutAnyOrigin
            ](gpu.per_sampled_starts_buf.unsafe_ptr())

            @parameter
            @always_inline
            def per_sample_wrapper(
                bo: LayoutTensor[
                    dtype,
                    Layout.row_major(N_ENVS_GPU * PER_ENV_CAP * OBS),
                    MutAnyOrigin,
                ],
                ba: LayoutTensor[
                    dtype,
                    Layout.row_major(N_ENVS_GPU * PER_ENV_CAP * ACT),
                    MutAnyOrigin,
                ],
                br: LayoutTensor[
                    dtype,
                    Layout.row_major(N_ENVS_GPU * PER_ENV_CAP),
                    MutAnyOrigin,
                ],
                bd: LayoutTensor[
                    dtype,
                    Layout.row_major(N_ENVS_GPU * PER_ENV_CAP),
                    MutAnyOrigin,
                ],
                bd_term: LayoutTensor[
                    dtype,
                    Layout.row_major(N_ENVS_GPU * PER_ENV_CAP),
                    MutAnyOrigin,
                ],
                bp: LayoutTensor[
                    dtype,
                    Layout.row_major(N_ENVS_GPU * PER_ENV_CAP * ACT),
                    MutAnyOrigin,
                ],
                bv: LayoutTensor[
                    dtype,
                    Layout.row_major(N_ENVS_GPU * PER_ENV_CAP),
                    MutAnyOrigin,
                ],
                btp: LayoutTensor[
                    DType.uint8,
                    Layout.row_major(N_ENVS_GPU * PER_ENV_CAP),
                    MutAnyOrigin,
                ],
                oo: LayoutTensor[
                    dtype,
                    Layout.row_major(WIN_FULL * BATCH * OBS),
                    MutAnyOrigin,
                ],
                oa: LayoutTensor[
                    dtype, Layout.row_major(K * BATCH * ACT), MutAnyOrigin
                ],
                orw: LayoutTensor[
                    dtype, Layout.row_major(WIN_TRN * BATCH), MutAnyOrigin
                ],
                od: LayoutTensor[
                    dtype, Layout.row_major(WIN_TRN * BATCH), MutAnyOrigin
                ],
                op: LayoutTensor[
                    dtype,
                    Layout.row_major(WIN_FULL * BATCH * ACT),
                    MutAnyOrigin,
                ],
                ov: LayoutTensor[
                    dtype, Layout.row_major(WIN_FULL * BATCH), MutAnyOrigin
                ],
                otp: LayoutTensor[
                    DType.uint8,
                    Layout.row_major(WIN_FULL * BATCH),
                    MutAnyOrigin,
                ],
                pe: LayoutTensor[
                    DType.int32, Layout.row_major(BATCH), MutAnyOrigin
                ],
                ps: LayoutTensor[
                    DType.int32, Layout.row_major(BATCH), MutAnyOrigin
                ],
            ):
                sample_seq_with_targets_priority_kernel[
                    BATCH, K, N_TD, N_ENVS_GPU, PER_ENV_CAP, OBS, ACT, dtype
                ](
                    bo,
                    ba,
                    br,
                    bd,
                    bd_term,
                    bp,
                    bv,
                    btp,
                    oo,
                    oa,
                    orw,
                    od,
                    op,
                    ov,
                    otp,
                    pe,
                    ps,
                )

            ctx.enqueue_function[per_sample_wrapper](
                buf_obs_t,
                buf_act_t,
                buf_rew_t,
                buf_done_t,
                buf_term_t,
                buf_pol_t,
                buf_val_t,
                buf_tp_t,
                b_obs_t,
                b_act_t,
                b_rew_t,
                b_done_t,
                b_pol_t,
                b_val_t,
                b_tp_t,
                per_envs_t,
                per_starts_t,
                grid_dim=(BATCH_BLOCKS_S,),
                block_dim=(TPB,),
            )
        else:
            ctx.enqueue_function[sample_wrapper](
                buf_obs_t,
                buf_act_t,
                buf_rew_t,
                buf_done_t,
                buf_term_t,
                buf_pol_t,
                buf_val_t,
                buf_tp_t,
                b_obs_t,
                b_act_t,
                b_rew_t,
                b_done_t,
                b_pol_t,
                b_val_t,
                b_tp_t,
                buf_size_s,
                buf_wptr_s,
                sample_seed,
                grid_dim=(BATCH_BLOCKS_S,),
                block_dim=(TPB,),
            )

        # ── Step 1.5: GPU reanalyze (use_last_model_value) ───────────
        # When enabled, refresh batch_values with fresh predictions from
        # the current network. The n-step kernel then bootstraps with
        # up-to-date values rather than stale stored MCTS values.
        # Reference: muzero-general/replay_buffer.py:237-238 (see also
        # the asynchronous Reanalyse worker at .py:306-374).
        #
        # Chunked one timestep at a time so the existing workspace buffer
        # (sized for BATCH samples) stays valid; we reuse pred_out_buf
        # for the chunk's prediction output and write the decoded scalar
        # straight into batch_values at the matching time-major offset.
        if use_reanalyze:
            comptime BATCH_BLOCKS_RA = (BATCH + TPB - 1) // TPB
            comptime run_ra_scale = scale_hidden_kernel[BATCH, LATENT, dtype]
            comptime run_ra_dec = decode_value_dist_kernel[
                BATCH, BINS, PRED_OUT_DIM, ACT, dtype
            ]
            for t in range(WIN_FULL):
                var obs_off = t * BATCH * OBS
                var ra_obs_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, REP_IN_DIM),
                    MutAnyOrigin,
                ](gpu.batch_obs_buf.unsafe_ptr() + obs_off)
                var ra_h_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, REP_OUT_DIM),
                    MutAnyOrigin,
                ](gpu.reanalyze_hidden_buf.unsafe_ptr())
                # Use TARGET nets (E4) so the bootstrap value tracks a
                # slowly-updating snapshot of the online weights, not the
                # current step's mid-update weights.
                Self.RepNet.forward_gpu[BATCH](
                    ctx,
                    ra_obs_t,
                    ra_h_t,
                    gpu.representation_target.params_view(),
                    gpu.representation_target.model_state_view(),
                    gpu.workspace_buf,
                )

                var ra_h_1d = LayoutTensor[
                    dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
                ](gpu.reanalyze_hidden_buf.unsafe_ptr())
                ctx.enqueue_function[run_ra_scale](
                    ra_h_1d,
                    grid_dim=(BATCH_BLOCKS_RA,),
                    block_dim=(TPB,),
                )

                # Re-view the hidden as PRED input (same memory; the
                # type system needs matching IN_DIM literal).
                var ra_h_pred_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, PRED_IN_DIM),
                    MutAnyOrigin,
                ](gpu.reanalyze_hidden_buf.unsafe_ptr())
                var ra_p_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, PRED_OUT_DIM),
                    MutAnyOrigin,
                ](gpu.pred_out_buf.unsafe_ptr())
                Self.PredNet.forward_gpu[BATCH](
                    ctx,
                    ra_h_pred_t,
                    ra_p_t,
                    gpu.prediction_target.params_view(),
                    gpu.prediction_target.model_state_view(),
                    gpu.workspace_buf,
                )

                # Decode value distribution → scalar in untransformed
                # space; write directly into batch_values[t, :].
                var ra_p_1d = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH * PRED_OUT_DIM),
                    MutAnyOrigin,
                ](gpu.pred_out_buf.unsafe_ptr())
                var bval_slot = LayoutTensor[
                    dtype, Layout.row_major(BATCH), MutAnyOrigin
                ](gpu.batch_values_buf.unsafe_ptr() + t * BATCH)
                ctx.enqueue_function[run_ra_dec](
                    ra_p_1d,
                    bval_slot,
                    Scalar[dtype](self.v_min),
                    Scalar[dtype](self.v_max),
                    Scalar[dtype](0.001),
                    grid_dim=(BATCH_BLOCKS_RA,),
                    block_dim=(TPB,),
                )

        # ── Step 2: GPU n-step targets + scalar transform ────────────
        comptime TARGET_EL = BATCH * (K + 1)
        comptime TARGET_BLOCKS = (TARGET_EL + TPB - 1) // TPB

        # N-step value/reward targets (on GPU)
        var val_tgt_t = LayoutTensor[
            dtype, Layout.row_major((K + 1) * BATCH), MutAnyOrigin
        ](gpu.value_targets_buf.unsafe_ptr())
        var rew_tgt_t = LayoutTensor[
            dtype, Layout.row_major(K * BATCH), MutAnyOrigin
        ](gpu.reward_targets_buf.unsafe_ptr())

        comptime BACKUP_TYPE = Self.Config.Backup.BACKUP_TYPE

        @parameter
        @always_inline
        def nstep_wrapper(
            vt: LayoutTensor[
                dtype, Layout.row_major((K + 1) * BATCH), MutAnyOrigin
            ],
            rt: LayoutTensor[dtype, Layout.row_major(K * BATCH), MutAnyOrigin],
            brew: LayoutTensor[
                dtype, Layout.row_major(WIN_TRN * BATCH), MutAnyOrigin
            ],
            bdn: LayoutTensor[
                dtype, Layout.row_major(WIN_TRN * BATCH), MutAnyOrigin
            ],
            bval: LayoutTensor[
                dtype, Layout.row_major(WIN_FULL * BATCH), MutAnyOrigin
            ],
            btp: LayoutTensor[
                DType.uint8,
                Layout.row_major(WIN_FULL * BATCH),
                MutAnyOrigin,
            ],
            g: Scalar[dtype],
        ):
            nstep_value_targets_kernel[BATCH, K, N_TD, dtype, BACKUP_TYPE](
                vt, rt, brew, bdn, bval, btp, g
            )

        # b_val_t still holds raw MCTS values from sampling, used as bootstrap
        # b_rew_t holds raw rewards from sampling
        # b_tp_t holds the per-step player-to-move stream (0 for single-player)
        ctx.enqueue_function[nstep_wrapper](
            val_tgt_t,
            rew_tgt_t,
            b_rew_t,
            b_done_t,
            b_val_t,
            b_tp_t,
            Scalar[dtype](self.gamma),
            grid_dim=(TARGET_BLOCKS,),
            block_dim=(TPB,),
        )

        # Scalar transform value targets
        comptime VAL_TGT_SIZE = (K + 1) * BATCH
        comptime VAL_TGT_BLOCKS = (VAL_TGT_SIZE + TPB - 1) // TPB
        comptime run_st_val = scalar_transform_kernel[VAL_TGT_SIZE, dtype]
        ctx.enqueue_function[run_st_val](
            val_tgt_t,
            Scalar[dtype](0.001),
            grid_dim=(VAL_TGT_BLOCKS,),
            block_dim=(TPB,),
        )

        # Scalar transform reward targets
        comptime REW_TGT_SIZE = K * BATCH
        comptime REW_TGT_BLOCKS = (REW_TGT_SIZE + TPB - 1) // TPB
        comptime run_st_rew = scalar_transform_kernel[REW_TGT_SIZE, dtype]
        ctx.enqueue_function[run_st_rew](
            rew_tgt_t,
            Scalar[dtype](0.001),
            grid_dim=(REW_TGT_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── Step 3: GPU forward pass (K-step unroll) ─────────────────

        # Zero all gradients
        gpu.representation.zero_grads(ctx)
        gpu.dynamics.zero_grads(ctx)
        gpu.prediction.zero_grads(ctx)

        # Representation: h(obs_0) -> s^0
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, REP_IN_DIM), MutAnyOrigin
        ](gpu.batch_obs_buf.unsafe_ptr())

        var h0_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, REP_OUT_DIM), MutAnyOrigin
        ](gpu.hidden_buf.unsafe_ptr())

        var rep_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, REP_CS), MutAnyOrigin
        ](gpu.rep_cache_buf.unsafe_ptr())

        Self.RepNet.forward_gpu_with_cache[BATCH](
            ctx,
            obs_t,
            h0_t,
            gpu.representation.params_view(),
            gpu.representation.model_state_view(),
            rep_cache_t,
            gpu.workspace_buf,
        )

        # Scale hidden state on GPU
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB
        comptime run_scale = scale_hidden_kernel[BATCH, LATENT, dtype]
        var hidden_1d = LayoutTensor[
            dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
        ](gpu.hidden_buf.unsafe_ptr())
        ctx.enqueue_function[run_scale](
            hidden_1d,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # K-step unroll: prediction + dynamics
        for k in range(K + 1):
            # Prediction: f(s^k) -> (policy_logits, value_logits)
            var hk_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, PRED_IN_DIM), MutAnyOrigin
            ](gpu.hidden_buf.unsafe_ptr() + k * BATCH * LATENT)

            var pred_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, PRED_OUT_DIM), MutAnyOrigin
            ](gpu.pred_out_buf.unsafe_ptr())

            var pred_cache_off = k * BATCH * PRED_CS
            var pred_cache_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, PRED_CS), MutAnyOrigin
            ](gpu.pred_cache_buf.unsafe_ptr() + pred_cache_off)

            Self.PredNet.forward_gpu_with_cache[BATCH](
                ctx,
                hk_t,
                pred_t,
                gpu.prediction.params_view(),
                gpu.prediction.model_state_view(),
                pred_cache_t,
                gpu.workspace_buf,
            )

            # Store pred outputs for backward (copy to persistent location)
            # We'll recompute during backward instead to save memory

            # Dynamics: g(s^k, a_{t+k}) -> (s^{k+1}, r^{k+1})
            if k < K:
                # Build dynamics input on GPU
                var actions_k_t = LayoutTensor[
                    dtype, Layout.row_major(BATCH * ACT), MutAnyOrigin
                ](gpu.batch_actions_buf.unsafe_ptr() + k * BATCH * ACT)

                comptime DYN_EL = BATCH * DYN_IN
                comptime DYN_BLOCKS = (DYN_EL + TPB - 1) // TPB
                comptime run_build = build_dyn_input_kernel[
                    BATCH, LATENT, ACT, DYN_IN, dtype
                ]
                var dyn_in_1d = LayoutTensor[
                    dtype, Layout.row_major(BATCH * DYN_IN), MutAnyOrigin
                ](gpu.dyn_input_buf.unsafe_ptr())
                var hk_1d = LayoutTensor[
                    dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
                ](gpu.hidden_buf.unsafe_ptr() + k * BATCH * LATENT)
                ctx.enqueue_function[run_build](
                    dyn_in_1d,
                    hk_1d,
                    actions_k_t,
                    grid_dim=(DYN_BLOCKS,),
                    block_dim=(TPB,),
                )

                var dyn_in_t = LayoutTensor[
                    dtype, Layout.row_major(BATCH, DYN_IN_DIM), MutAnyOrigin
                ](gpu.dyn_input_buf.unsafe_ptr())

                var dyn_out_t = LayoutTensor[
                    dtype, Layout.row_major(BATCH, DYN_OUT_DIM), MutAnyOrigin
                ](gpu.dyn_output_buf.unsafe_ptr())

                var dyn_cache_off = k * BATCH * DYN_CS
                var dyn_cache_t = LayoutTensor[
                    dtype, Layout.row_major(BATCH, DYN_CS), MutAnyOrigin
                ](gpu.dyn_cache_buf.unsafe_ptr() + dyn_cache_off)

                Self.DynNet.forward_gpu_with_cache[BATCH](
                    ctx,
                    dyn_in_t,
                    dyn_out_t,
                    gpu.dynamics.params_view(),
                    gpu.dynamics.model_state_view(),
                    dyn_cache_t,
                    gpu.workspace_buf,
                )

                # Extract hidden state from dynamics output
                var next_hidden = LayoutTensor[
                    dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
                ](gpu.hidden_buf.unsafe_ptr() + (k + 1) * BATCH * LATENT)
                var dyn_out_1d = LayoutTensor[
                    dtype, Layout.row_major(BATCH * DYN_OUT), MutAnyOrigin
                ](gpu.dyn_output_buf.unsafe_ptr())

                comptime EXTRACT_EL = BATCH * LATENT
                comptime EXTRACT_BLOCKS = (EXTRACT_EL + TPB - 1) // TPB
                comptime run_extract = extract_hidden_kernel[
                    BATCH, LATENT, DYN_OUT, dtype
                ]
                ctx.enqueue_function[run_extract](
                    next_hidden,
                    dyn_out_1d,
                    grid_dim=(EXTRACT_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Scale next hidden state
                var next_h_scale = LayoutTensor[
                    dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
                ](gpu.hidden_buf.unsafe_ptr() + (k + 1) * BATCH * LATENT)
                ctx.enqueue_function[run_scale](
                    next_h_scale,
                    grid_dim=(BATCH_BLOCKS,),
                    block_dim=(TPB,),
                )

        # ── Step 4: GPU backward pass ────────────────────────────────
        var inv_k_s = Scalar[dtype](1.0 / Float64(K + 1) / Float64(BATCH))

        # PER IS-weights view (constant 1.0 when use_per=False — caller fills
        # gpu.per_is_weights_buf with 1s; under use_per=True caller writes
        # the sum-tree-derived weights). Passed into the 3 CE grad kernels
        # to scale per-sample gradient contribution.
        var is_weights_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu.per_is_weights_buf.unsafe_ptr())

        # Zero hidden gradient carry
        ctx.enqueue_memset(gpu.grad_hidden_buf, 0)

        # Process steps in REVERSE
        for _ri in range(K + 1):
            var k = K - _ri

            # Re-run prediction forward to get outputs for gradient computation
            var hk_bwd = LayoutTensor[
                dtype, Layout.row_major(BATCH, PRED_IN_DIM), MutAnyOrigin
            ](gpu.hidden_buf.unsafe_ptr() + k * BATCH * LATENT)

            var pred_bwd = LayoutTensor[
                dtype, Layout.row_major(BATCH, PRED_OUT_DIM), MutAnyOrigin
            ](gpu.pred_out_buf.unsafe_ptr())

            Self.PredNet.forward_gpu[BATCH](
                ctx,
                hk_bwd,
                pred_bwd,
                gpu.prediction.params_view(),
                gpu.prediction.model_state_view(),
                gpu.workspace_buf,
            )

            # Compute policy CE gradient on GPU
            var pol_off = k * BATCH * ACT
            var policy_targets_k = LayoutTensor[
                dtype, Layout.row_major(BATCH * ACT), MutAnyOrigin
            ](gpu.batch_policies_buf.unsafe_ptr() + pol_off)

            ctx.enqueue_memset(gpu.grad_pred_out_buf, 0)
            var grad_pred_1d = LayoutTensor[
                dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
            ](gpu.grad_pred_out_buf.unsafe_ptr())
            var pred_1d = LayoutTensor[
                dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
            ](gpu.pred_out_buf.unsafe_ptr())

            comptime run_pol_grad = ce_policy_grad_kernel[
                BATCH, ACT, PRED_OUT, dtype
            ]
            ctx.enqueue_function[run_pol_grad](
                grad_pred_1d,
                pred_1d,
                policy_targets_k,
                is_weights_t,
                inv_k_s,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # Value CE gradient — need two-hot encoded targets
            var val_targets_k = LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ](gpu.value_targets_buf.unsafe_ptr() + k * BATCH)
            var val_dist = LayoutTensor[
                dtype, Layout.row_major(BATCH * BINS), MutAnyOrigin
            ](gpu.value_target_dist_buf.unsafe_ptr())

            comptime run_twohot = two_hot_encode_kernel[BATCH, BINS, dtype]
            ctx.enqueue_function[run_twohot](
                val_dist,
                val_targets_k,
                Scalar[dtype](self.v_min),
                Scalar[dtype](self.v_max),
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            comptime run_val_grad = ce_value_grad_kernel[
                BATCH, BINS, ACT, PRED_OUT, dtype
            ]
            ctx.enqueue_function[run_val_grad](
                grad_pred_1d,
                pred_1d,
                val_dist,
                is_weights_t,
                inv_k_s,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            # Prediction backward: grad_pred_out -> grad_pred_in
            var pred_cache_bwd_off = k * BATCH * PRED_CS
            var pred_cache_bwd = LayoutTensor[
                dtype, Layout.row_major(BATCH, PRED_CS), MutAnyOrigin
            ](gpu.pred_cache_buf.unsafe_ptr() + pred_cache_bwd_off)

            var grad_pred_out_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, PRED_OUT_DIM), MutAnyOrigin
            ](gpu.grad_pred_out_buf.unsafe_ptr())

            ctx.enqueue_memset(gpu.grad_pred_in_buf, 0)
            var grad_pred_in_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, PRED_IN_DIM), MutAnyOrigin
            ](gpu.grad_pred_in_buf.unsafe_ptr())

            var pred_grads = gpu.prediction.grads_view()
            Self.PredNet.backward_gpu[BATCH](
                ctx,
                grad_pred_out_t,
                grad_pred_in_t,
                gpu.prediction.params_view(),
                gpu.prediction.model_state_view(),
                pred_cache_bwd,
                pred_grads,
                gpu.workspace_buf,
            )

            # Accumulate pred gradient into hidden carry
            comptime HIDDEN_EL = BATCH * LATENT
            comptime HIDDEN_BLOCKS = (HIDDEN_EL + TPB - 1) // TPB
            comptime run_add = add_scaled_kernel[HIDDEN_EL, dtype]
            var grad_hidden_1d = LayoutTensor[
                dtype, Layout.row_major(HIDDEN_EL), MutAnyOrigin
            ](gpu.grad_hidden_buf.unsafe_ptr())
            var grad_pred_in_1d = LayoutTensor[
                dtype, Layout.row_major(HIDDEN_EL), MutAnyOrigin
            ](gpu.grad_pred_in_buf.unsafe_ptr())
            ctx.enqueue_function[run_add](
                grad_hidden_1d,
                grad_pred_in_1d,
                Scalar[dtype](1.0),
                grid_dim=(HIDDEN_BLOCKS,),
                block_dim=(TPB,),
            )

            # Dynamics backward (if k > 0)
            if k > 0:
                var dk = k - 1

                # Set hidden gradient into dynamics output gradient (scaled by 0.5)
                ctx.enqueue_memset(gpu.grad_dyn_out_buf, 0)
                comptime run_set_hgrad = set_hidden_grad_for_dyn_kernel[
                    BATCH, LATENT, DYN_OUT, dtype
                ]
                var grad_dyn_1d = LayoutTensor[
                    dtype, Layout.row_major(BATCH * DYN_OUT), MutAnyOrigin
                ](gpu.grad_dyn_out_buf.unsafe_ptr())
                ctx.enqueue_function[run_set_hgrad](
                    grad_dyn_1d,
                    grad_hidden_1d,
                    Scalar[dtype](0.5),
                    grid_dim=(HIDDEN_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Reward CE gradient
                var rew_targets_k = LayoutTensor[
                    dtype, Layout.row_major(BATCH), MutAnyOrigin
                ](gpu.reward_targets_buf.unsafe_ptr() + dk * BATCH)
                var rew_dist = LayoutTensor[
                    dtype, Layout.row_major(BATCH * BINS), MutAnyOrigin
                ](gpu.reward_target_dist_buf.unsafe_ptr())
                ctx.enqueue_function[run_twohot](
                    rew_dist,
                    rew_targets_k,
                    Scalar[dtype](self.v_min),
                    Scalar[dtype](self.v_max),
                    grid_dim=(BATCH_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Re-run dynamics forward to get output for reward grad
                var dyn_in_bwd = LayoutTensor[
                    dtype, Layout.row_major(BATCH, DYN_IN_DIM), MutAnyOrigin
                ](gpu.dyn_input_buf.unsafe_ptr())

                # Rebuild dynamics input
                var hk_bwd_1d = LayoutTensor[
                    dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
                ](gpu.hidden_buf.unsafe_ptr() + dk * BATCH * LATENT)
                var act_bwd_1d = LayoutTensor[
                    dtype, Layout.row_major(BATCH * ACT), MutAnyOrigin
                ](gpu.batch_actions_buf.unsafe_ptr() + dk * BATCH * ACT)
                comptime run_build2 = build_dyn_input_kernel[
                    BATCH, LATENT, ACT, DYN_IN, dtype
                ]
                var dyn_in_1d_bwd = LayoutTensor[
                    dtype, Layout.row_major(BATCH * DYN_IN), MutAnyOrigin
                ](gpu.dyn_input_buf.unsafe_ptr())
                comptime DYN_BUILD_BLOCKS = (BATCH * DYN_IN + TPB - 1) // TPB
                ctx.enqueue_function[run_build2](
                    dyn_in_1d_bwd,
                    hk_bwd_1d,
                    act_bwd_1d,
                    grid_dim=(DYN_BUILD_BLOCKS,),
                    block_dim=(TPB,),
                )

                var dyn_out_bwd = LayoutTensor[
                    dtype, Layout.row_major(BATCH, DYN_OUT_DIM), MutAnyOrigin
                ](gpu.dyn_output_buf.unsafe_ptr())
                Self.DynNet.forward_gpu[BATCH](
                    ctx,
                    dyn_in_bwd,
                    dyn_out_bwd,
                    gpu.dynamics.params_view(),
                    gpu.dynamics.model_state_view(),
                    gpu.workspace_buf,
                )

                # Reward gradient
                var dyn_out_1d_bwd = LayoutTensor[
                    dtype, Layout.row_major(BATCH * DYN_OUT), MutAnyOrigin
                ](gpu.dyn_output_buf.unsafe_ptr())
                comptime run_rew_grad = ce_reward_grad_kernel[
                    BATCH, BINS, DYN_OUT, LATENT, dtype
                ]
                ctx.enqueue_function[run_rew_grad](
                    grad_dyn_1d,
                    dyn_out_1d_bwd,
                    rew_dist,
                    is_weights_t,
                    inv_k_s,
                    grid_dim=(BATCH_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Scale dynamics gradients by 1/K
                comptime DYN_OUT_EL = BATCH * DYN_OUT
                comptime DYN_OUT_BLOCKS = (DYN_OUT_EL + TPB - 1) // TPB
                comptime run_dyn_scale = scale_kernel[DYN_OUT_EL, dtype]
                ctx.enqueue_function[run_dyn_scale](
                    grad_dyn_1d,
                    Scalar[dtype](1.0 / Float64(K)),
                    grid_dim=(DYN_OUT_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Dynamics backward
                var dyn_cache_bwd_off = dk * BATCH * DYN_CS
                var dyn_cache_bwd = LayoutTensor[
                    dtype, Layout.row_major(BATCH, DYN_CS), MutAnyOrigin
                ](gpu.dyn_cache_buf.unsafe_ptr() + dyn_cache_bwd_off)

                var grad_dyn_out_t = LayoutTensor[
                    dtype, Layout.row_major(BATCH, DYN_OUT_DIM), MutAnyOrigin
                ](gpu.grad_dyn_out_buf.unsafe_ptr())

                ctx.enqueue_memset(gpu.grad_dyn_in_buf, 0)
                var grad_dyn_in_t = LayoutTensor[
                    dtype, Layout.row_major(BATCH, DYN_IN_DIM), MutAnyOrigin
                ](gpu.grad_dyn_in_buf.unsafe_ptr())

                var dyn_grads = gpu.dynamics.grads_view()
                Self.DynNet.backward_gpu[BATCH](
                    ctx,
                    grad_dyn_out_t,
                    grad_dyn_in_t,
                    gpu.dynamics.params_view(),
                    gpu.dynamics.model_state_view(),
                    dyn_cache_bwd,
                    dyn_grads,
                    gpu.workspace_buf,
                )

                # Extract hidden grad from dynamics input grad -> new carry
                ctx.enqueue_memset(gpu.grad_hidden_buf, 0)
                comptime run_extract_hgrad = extract_hidden_grad_kernel[
                    BATCH, LATENT, DYN_IN, dtype
                ]
                var grad_dyn_in_1d = LayoutTensor[
                    dtype, Layout.row_major(BATCH * DYN_IN), MutAnyOrigin
                ](gpu.grad_dyn_in_buf.unsafe_ptr())
                ctx.enqueue_function[run_extract_hgrad](
                    grad_hidden_1d,
                    grad_dyn_in_1d,
                    grid_dim=(HIDDEN_BLOCKS,),
                    block_dim=(TPB,),
                )

        # Representation backward
        var grad_rep_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, REP_OUT_DIM), MutAnyOrigin
        ](gpu.grad_hidden_buf.unsafe_ptr())

        ctx.enqueue_memset(gpu.grad_rep_in_buf, 0)
        var grad_rep_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, REP_IN_DIM), MutAnyOrigin
        ](gpu.grad_rep_in_buf.unsafe_ptr())

        var rep_cache_bwd = LayoutTensor[
            dtype, Layout.row_major(BATCH, REP_CS), MutAnyOrigin
        ](gpu.rep_cache_buf.unsafe_ptr())

        var rep_grads = gpu.representation.grads_view()
        Self.RepNet.backward_gpu[BATCH](
            ctx,
            grad_rep_out_t,
            grad_rep_in_t,
            gpu.representation.params_view(),
            gpu.representation.model_state_view(),
            rep_cache_bwd,
            rep_grads,
            gpu.workspace_buf,
        )

        # ── Step 4.4: Loss diagnostics (logger) ──────────────────────
        # At end of backward loop, pred_out_buf holds predictions at
        # k=0 (last backward iter recomputes pred forward there). Copy
        # one batch's worth of pred + targets (policy, value at k=0)
        # to host and compute the CE losses + entropy + value mean.
        # Reward loss is omitted — would need dyn forward re-run at
        # k=0 since dyn_output_buf is reused across K dyn steps.
        # Cost: 3 small DMAs + 1 sync per diag step (gated by
        # diag_every; default 0 = log every update).
        if Bool(self.logger) and (
            self.diag_every <= 0 or self.train_step_count % self.diag_every == 0
        ):
            # Copy whole K+1-step target buffers to pre-allocated host
            # buffers; the loss calc below reads only the k=0 slice
            # (first BATCH*ACT / BATCH*BINS entries). Buffers live on
            # gpu state so they aren't re-created in this hot loop —
            # repeated host buffer creation triggered "not enough data
            # in src" on NVIDIA under high call frequency.
            ctx.enqueue_copy(gpu.diag_pred_host, gpu.pred_out_buf)
            ctx.enqueue_copy(gpu.diag_pol_host, gpu.batch_policies_buf)
            # value_targets_buf is the per-step h-transformed scalar
            # target ((K+1)*BATCH); diag computes MSE against decoded
            # predicted scalar instead of TwoHot CE (the encoding
            # buffer `value_target_dist_buf` is recycled per-step
            # inside the backward loop and would not survive to here).
            ctx.enqueue_copy(gpu.diag_val_host, gpu.value_targets_buf)
            ctx.synchronize()

            var step = self.train_step_count
            var sum_policy_ce: Float64 = 0.0
            var sum_policy_entropy: Float64 = 0.0
            var sum_value_ce: Float64 = 0.0
            var sum_value_mean: Float64 = 0.0
            var sum_value_target: Float64 = 0.0

            for b in range(BATCH):
                var pred_off = b * PRED_OUT
                # Policy softmax + CE + entropy (numerically stable)
                var max_p: Float64 = -1e18
                for a in range(ACT):
                    var l = Float64(gpu.diag_pred_host[pred_off + a])
                    if l > max_p:
                        max_p = l
                var sum_e_p: Float64 = 0.0
                for a in range(ACT):
                    sum_e_p += exp(
                        Float64(gpu.diag_pred_host[pred_off + a]) - max_p
                    )
                for a in range(ACT):
                    var prob = (
                        exp(Float64(gpu.diag_pred_host[pred_off + a]) - max_p)
                        / sum_e_p
                    )
                    var tgt = Float64(gpu.diag_pol_host[b * ACT + a])
                    if tgt > 1e-8 and prob > 1e-8:
                        sum_policy_ce -= tgt * log(prob)
                    if prob > 1e-8:
                        sum_policy_entropy -= prob * log(prob)

                # Value softmax → decode to scalar; loss = MSE vs
                # scalar target at k=0 (value_targets_buf[0..BATCH]).
                var val_off = pred_off + ACT
                var max_v: Float64 = -1e18
                for i in range(BINS):
                    var l = Float64(gpu.diag_pred_host[val_off + i])
                    if l > max_v:
                        max_v = l
                var sum_e_v: Float64 = 0.0
                for i in range(BINS):
                    sum_e_v += exp(
                        Float64(gpu.diag_pred_host[val_off + i]) - max_v
                    )
                var v_step = (self.v_max - self.v_min) / Float64(
                    BINS - 1
                ) if BINS > 1 else 1.0
                var pred_val_scalar: Float64 = 0.0
                for i in range(BINS):
                    var prob_v = (
                        exp(Float64(gpu.diag_pred_host[val_off + i]) - max_v)
                        / sum_e_v
                    )
                    var bin_center = self.v_min + Float64(i) * v_step
                    pred_val_scalar += prob_v * bin_center
                var tgt_scalar = Float64(gpu.diag_val_host[b])
                var diff = pred_val_scalar - tgt_scalar
                sum_value_ce += diff * diff  # MSE; reuses sum_value_ce slot
                sum_value_mean += pred_val_scalar
                sum_value_target += tgt_scalar

            var n = Float64(BATCH)
            var policy_ce = sum_policy_ce / n
            var policy_entropy = sum_policy_entropy / n
            var value_ce = sum_value_ce / n
            var value_mean = sum_value_mean / n
            var value_target_mean = sum_value_target / n

            # Clamp NaN/inf so logger doesn't silently drop
            if policy_ce != policy_ce or policy_ce > 1e10:
                policy_ce = 0.0
            if value_ce != value_ce or value_ce > 1e10:
                value_ce = 0.0
            if value_mean != value_mean:
                value_mean = 0.0
            if value_target_mean != value_target_mean:
                value_target_mean = 0.0

            # KNOWN_GROUPS alignment: drop "loss/" prefix on names that
            # already match canonical group metrics.
            # - policy_ce       → Policy Cross-Entropy group
            # - entropy         → Entropy group
            # - value_mse,
            #   value_mean,
            #   value_target_mean → Value Head group
            # - loss            → Loss group (total = CE + MSE)
            self.logger.value()[].log_scalar("policy_ce", policy_ce, step)
            self.logger.value()[].log_scalar("entropy", policy_entropy, step)
            self.logger.value()[].log_scalar("value_mse", value_ce, step)
            self.logger.value()[].log_scalar("value_mean", value_mean, step)
            self.logger.value()[].log_scalar(
                "value_target_mean", value_target_mean, step
            )
            self.logger.value()[].log_scalar(
                "loss", policy_ce + value_ce, step
            )

        # ── Step 4.5: Global-L2 gradient clipping ────────────────────
        # Mirrors the CPU `_clip_gradients` exactly: compute joint L2
        # norm across rep+dyn+pred grads; if it exceeds max_grad_norm,
        # scale all three by max_grad_norm / total_norm. Implemented
        # via host roundtrip (DMA grads down, reduce on CPU, DMA scaled
        # grads back up). The roundtrip is small (~30KB for typical
        # configs) and matches the CPU semantics.
        #
        # Without this the K-step unrolled dynamics chain compounds the
        # reward/value gradient and rep+dyn param norms balloon ~60%
        # in the first 1024 steps (verified with the param-norm diag).
        if self.max_grad_norm > 0.0:
            self._global_clip_grad_norm[N_ENVS_P, PER_ENV_CAP_P](ctx, gpu)

        # ── Step 5: GPU optimizer step ───────────────────────────────
        gpu.representation.optimizer_step(ctx)
        gpu.dynamics.optimizer_step(ctx)
        gpu.prediction.optimizer_step(ctx)

        # ── Step 6: Polyak soft-update of target nets (E4) ──────────
        # Slowly track the online networks. Active only when reanalyze
        # is on (the only consumer of target params); skip otherwise to
        # avoid wasted GPU work.
        if use_reanalyze and self.target_tau > 0.0:
            gpu.representation_target.soft_update_from_gpu(
                gpu.representation, self.target_tau, ctx
            )
            gpu.prediction_target.soft_update_from_gpu(
                gpu.prediction, self.target_tau, ctx
            )

        # ── Step 7: PER priority refresh (Phase H13) ────────────────
        # Recompute priority = (|target - pred|)^α for each sampled
        # position and update the sum-tree. Cheap: BATCH host-side calc
        # + ~BATCH×log(tree) tree updates. Most accurate when reanalyze
        # is on (batch_values_buf holds fresh predictions); without
        # reanalyze it holds the stored MCTS root value at collection
        # time, which is still meaningful but staler.
        if use_per:
            self._per_update_priorities[N_ENVS_P, PER_ENV_CAP_P](ctx, gpu)

        self.train_step_count += 1
        return Float64(0.0)  # Loss computation on GPU would require readback


    def train_gpu[
        E: GPUDiscreteEnv
    ](
        mut self,
        ctx: DeviceContext,
        num_steps: Int = 500000,
        warmup_steps: Int = 1000,
        gradient_steps: Int = 0,
        print_every: Int = 50000,
        use_reanalyze: Bool = False,
        sync_every: Int = 5000,
        lr_decay_rate: Float64 = 1.0,
        lr_decay_steps: Int = 1000,
        use_per: Bool = False,
    ) raises -> TrainingMetrics:
        """Train MuZero with GPU environment stepping and GPU training.

        GPU handles: Self.n_envs parallel env stepping, episode tracking,
                     K-step unrolled forward/backward training.
        CPU handles: MCTS planning (tree search), replay buffer with
                     MCTS targets, batch sampling.

        Each iteration:
          1. GPU: step Self.n_envs environments in parallel
          2. GPU→CPU: download obs/rewards/dones for MCTS + replay
          3. CPU: MCTS action selection per environment (batched network eval)
          4. CPU→GPU: upload actions
          5. CPU: store transitions with MCTS targets in replay buffer
          6. GPU: K-step training (forward/backward/optimizer)
          7. Periodic GPU→CPU weight sync for MCTS inference

        Parameters:
            E: GPU environment type implementing GPUDiscreteEnv.

        Args:
            ctx: GPU device context.
            num_steps: Total env transitions across all envs.
            warmup_steps: Random transitions before training starts.
            gradient_steps: Training steps per collection (0 = 1).
            print_every: Print interval in transitions.
            use_reanalyze: Enable MuZero Reanalyze.
            sync_every: GPU→CPU weight sync interval in transitions.
            lr_decay_rate: Learning rate decay rate.
            lr_decay_steps: Learning rate decay steps.
            use_per: Whether to use prioritized experience replay.

        Returns:
            TrainingMetrics with episode rewards.
        """
        comptime ACT = Self.Config.action_dim
        comptime OBS = Self.Config.obs_dim
        comptime ENV_BLOCKS = (Self.n_envs + TPB - 1) // TPB

        # Default to one grad step per env step (averaged across n_envs)
        # — matches MuZero paper's UTD-like cadence and is ~n_envs× the
        # old default of 1 grad step per env-step batch. The previous
        # default did roughly 1/n_envs the training of an AZ-comparable
        # setup, leaving CartPole stuck at random-policy reward.
        var grad_steps = gradient_steps if gradient_steps > 0 else Self.n_envs

        # ── Create GPU state with correct Self.n_envs ─────────────────────
        # PER_ENV_CAP derived from Config.buffer_capacity (Bug G, 2026-05-05):
        # MuZeroGPUState's PER_ENV_CAP defaulted to 1000, so with n_envs=32
        # the actual buffer capped at 32k transitions regardless of
        # Config.buffer_capacity. For CAP=50000 + 32 envs, ~18k env steps
        # never made it into the training distribution. Now ceil-divides so
        # n_envs × PER_ENV_CAP ≥ buffer_capacity.
        comptime _PER_ENV_CAP = (
            Self.Config.buffer_capacity + Self.n_envs - 1
        ) // Self.n_envs
        comptime LocalGPUState = MuZeroGPUState[
            Self.Config, Self.n_envs, _PER_ENV_CAP
        ]
        var gpu = LocalGPUState(ctx)
        gpu.upload_from(self.state, ctx)

        # Apply LR decay state from persistent `self.train_step_count`.
        # Each `train_gpu` call builds a fresh GPU state whose lr_scale
        # defaults to 1.0; without this, multi-phase training (where
        # `train_gpu` is called multiple times) resets to the full LR at
        # the start of each phase, defeating the decay schedule. The
        # value here matches what the in-loop print-boundary code below
        # would compute, so a single-phase run is unaffected.
        if lr_decay_rate < 1.0 and self.train_step_count > 0:
            var initial_lr_scale = exp(
                log(lr_decay_rate)
                * Float64(self.train_step_count)
                / Float64(lr_decay_steps)
            )
            gpu.representation.set_lr_scale(initial_lr_scale, ctx)
            gpu.dynamics.set_lr_scale(initial_lr_scale, ctx)
            gpu.prediction.set_lr_scale(initial_lr_scale, ctx)

        # ── Allocate GPU environment buffers ─────────────────────────
        var states_buf = ctx.enqueue_create_buffer[dtype](
            Self.n_envs * E.STATE_SIZE
        )
        var obs_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs * OBS)
        var prev_obs_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs * OBS)
        var actions_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        # One-hot action buffer for replay storage — the GPU replay store
        # kernel reads ACTION_DIM=ACT floats per env, so it expects a
        # one-hot view. ``actions_buf`` itself stays scalar because the
        # env.step kernel reads indices, not one-hot.
        var actions_onehot_buf = ctx.enqueue_create_buffer[dtype](
            Self.n_envs * ACT
        )
        var rewards_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var dones_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var terminated_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)

        # Legal mask buffer — for single-player envs all actions are legal,
        # so just fill with 1.0 once. Required by extract_actions_temp_kernel
        # (Bug E: argmax tiebreak deterministically picks action 0 → MuZero
        # commits to one action when MCTS visits are tied at init; switching
        # to temperature-weighted sampling restores stochastic exploration).
        var legal_masks_buf = ctx.enqueue_create_buffer[dtype](
            Self.n_envs * ACT
        )
        legal_masks_buf.enqueue_fill(Scalar[dtype](1.0))

        # Episode tracking on GPU
        var episode_rewards_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var episode_steps_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var gpu_reward_sum_buf = ctx.enqueue_create_buffer[dtype](1)
        var gpu_episode_count_buf = ctx.enqueue_create_buffer[dtype](1)

        # Workspace for env stepping
        comptime WS_SIZE = E.STEP_WS_SHARED + Self.n_envs * E.STEP_WS_PER_ENV
        var workspace_buf = ctx.enqueue_create_buffer[dtype](
            WS_SIZE if WS_SIZE > 0 else 1
        )

        # Host buffers for episode stats (periodic sync only)
        var reward_sum_host = ctx.enqueue_create_host_buffer[dtype](1)
        var episode_count_host = ctx.enqueue_create_host_buffer[dtype](1)

        # Action histogram — counts how many times each action was actually
        # executed in the env between print boundaries. Diagnostic for the
        # "MuZero produces sub-random episode lengths" failure mode: if
        # diagnose-time visit policy is ~uniform but executed actions are
        # heavily biased (or temporally correlated within an episode), the
        # bug is in the MCTS-visits-to-action sampling path, not in the
        # representation network.
        var action_hist_buf = ctx.enqueue_create_buffer[dtype](ACT)
        action_hist_buf.enqueue_fill(Scalar[dtype](0.0))
        var action_hist_host = ctx.enqueue_create_host_buffer[dtype](ACT)

        # Switch-rate counter — number of times action[t] != action[t-1] per
        # env, summed across the print interval. Random uniform = 50% switch
        # rate. Sub-50% = temporal correlation (sticky actions); near-0% =
        # per-env episode-long action commitment. Used jointly with the
        # marginal histogram to distinguish failure modes.
        var prev_actions_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        prev_actions_buf.enqueue_fill(Scalar[dtype](0.0))
        var switch_count_buf = ctx.enqueue_create_buffer[dtype](1)
        switch_count_buf.enqueue_fill(Scalar[dtype](0.0))
        var switch_count_host = ctx.enqueue_create_host_buffer[dtype](1)

        # GPU MCTS orchestrator (owns its own tree state).
        comptime LATENT = Self.Config.latent_dim
        comptime BINS = Self.Config.num_bins
        comptime MAX_NODES = 64
        comptime NUM_SIMS = Self.Config.num_simulations
        # Cap by ACT to forbid leaf duplication; floor to a divisor of
        # NUM_SIMS (orchestrator asserts divisibility).
        comptime _REQ_BSIMS_TG = Self.Config.batch_sims
        comptime _CAP_BSIMS_TG = (
            _REQ_BSIMS_TG if _REQ_BSIMS_TG < ACT else ACT
        )
        comptime MCTS_BATCH_SIMS = (
            _CAP_BSIMS_TG if NUM_SIMS % _CAP_BSIMS_TG == 0
            else (
                4 if (NUM_SIMS % 4 == 0 and _CAP_BSIMS_TG >= 4)
                else (
                    2 if (NUM_SIMS % 2 == 0 and _CAP_BSIMS_TG >= 2)
                    else 1
                )
            )
        )
        var generic_planner = GenericGPUMCTS[
            Self.n_envs, ACT, LATENT, BINS, MAX_NODES, NUM_SIMS,
            MCTS_BATCH_SIMS,
            Self.Config.PUCT,
            Self.Config.Noise,
            Self.Config.Players,
        ](ctx, gamma=self.gamma, v_min=self.v_min, v_max=self.v_max)
        # Gumbel-MuZero orchestrator — only used when
        # ``Config.PolicyMode.IS_GUMBEL`` is True. Allocated
        # unconditionally to keep the loop body free of comptime-if
        # around state declarations; the EZV2GPUMCTSState overhead
        # when unused is ~few MB vs the multi-MB NN buffers.
        var gumbel_planner = GumbelGPUMCTS[
            Self.n_envs, ACT, LATENT, BINS, MAX_NODES,
            Self.Config.PolicyMode.MAX_K, NUM_SIMS,
            Self.Config.Players,
        ](
            ctx, gamma=self.gamma, v_min=self.v_min, v_max=self.v_max,
            gumbel_scale=1.0,
        )

        # Network workspace for GPU MCTS forward calls
        comptime RepModel = Self.StateType.RepModel
        comptime DynModel = Self.StateType.DynModel
        comptime PredModel = Self.StateType.PredModel
        comptime OptType = Self.StateType.OptType
        comptime RepNet = Network[RepModel, OptType]
        comptime DynNet = Network[DynModel, OptType]
        comptime PredNet = Network[PredModel, OptType]
        comptime WS_R = RepModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_D = DynModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_P = PredModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime MAX_WS_1 = WS_R if WS_R > WS_D else WS_D
        comptime MAX_WS_2 = MAX_WS_1 if MAX_WS_1 > WS_P else WS_P
        comptime MCTS_BATCH_SZ = 8  # Must match MCTS_BATCH_SIMS in sim loop
        comptime MCTS_WS_SIZE = Self.n_envs * MCTS_BATCH_SZ * MAX_WS_2 if MAX_WS_2 > 0 else 1
        var mcts_workspace = ctx.enqueue_create_buffer[dtype](MCTS_WS_SIZE)

        comptime REP_IN_DIM = RepModel.IN_DIM
        comptime REP_OUT_DIM = RepModel.OUT_DIM
        comptime DYN_IN_DIM = DynModel.IN_DIM
        comptime DYN_OUT_DIM = DynModel.OUT_DIM
        comptime PRED_IN_DIM = PredModel.IN_DIM
        comptime PRED_OUT_DIM = PredModel.OUT_DIM
        comptime PRED_OUT = Self.StateType.PRED_OUT
        comptime DYN_IN = Self.StateType.DYN_IN
        comptime DYN_OUT = Self.StateType.DYN_OUT

        # ── Initialize environments on GPU ───────────────────────────
        E.reset_kernel_gpu[Self.n_envs, E.STATE_SIZE](
            ctx, states_buf, rng_seed=42
        )
        actions_buf.enqueue_fill(Scalar[dtype](0.0))
        # Extract obs from the just-reset state (no physics step). The
        # previous "step + re-reset + step" double-tap was the original
        # source of Bug F's per-step double-step (the loop re-extract
        # copy-pasted the no-op pattern), and even at init it advanced
        # all envs by one phantom action-0 step before training started.
        E.extract_obs_kernel_gpu[Self.n_envs, E.STATE_SIZE, OBS](
            ctx, states_buf, obs_buf
        )

        # Initialize workspace
        E.init_step_workspace_gpu[Self.n_envs](ctx, workspace_buf)

        # Zero episode accumulators
        episode_rewards_buf.enqueue_fill(Scalar[dtype](0.0))
        episode_steps_buf.enqueue_fill(Scalar[dtype](0.0))
        gpu_reward_sum_buf.enqueue_fill(Scalar[dtype](0.0))
        gpu_episode_count_buf.enqueue_fill(Scalar[dtype](0.0))

        ctx.synchronize()

        # ── Main training loop ───────────────────────────────────────
        var metrics = TrainingMetrics(algorithm_name="MuZero-GPU")
        var total_steps = 0
        var episode_count = 0
        var gpu_train_count = 0
        var next_print = print_every
        var next_sync = sync_every

        # Param-norm tracking for the print diagnostic. Initialized to
        # the params right after upload so the first print shows the
        # delta over the first interval (not delta-from-zero).
        var init_norms = self._net_param_l2(ctx, gpu)
        var last_rep_n = init_norms[0]
        var last_dyn_n = init_norms[1]
        var last_pred_n = init_norms[2]

        comptime PER_ENV_CAP = LocalGPUState.PER_ENV_CAP

        while total_steps < num_steps:
            # ── 1. Save obs in GPU replay before stepping ────────────
            gpu.replay.save_obs(ctx, obs_buf)

            # ── 2. GPU MCTS action selection (zero CPU sync) ─────────
            if total_steps < warmup_steps:
                # Random actions during warmup
                comptime run_warmup = uniform_random_discrete_actions_kernel[
                    dtype, Self.n_envs, ACT
                ]
                var warmup_acts_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](actions_buf.unsafe_ptr())
                ctx.enqueue_function[run_warmup](
                    warmup_acts_t,
                    Scalar[DType.uint32](UInt32(total_steps)),
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
                # Set uniform policies in staging buf for replay
                gpu.mcts_step_policy_buf.enqueue_fill(
                    Scalar[dtype](1.0 / Float64(ACT))
                )
                gpu.mcts_step_value_buf.enqueue_fill(Scalar[dtype](0.0))
            else:
                # ── Full GPU MCTS ────────────────────────────────────
                # Adapters are the same for PUCT and Gumbel — they're
                # just network wrappers. The dispatch is on which
                # orchestrator (``generic_planner`` vs ``gumbel_planner``)
                # consumes them.
                var rep_a = MuZeroRepGPU[
                    OBS, LATENT, RepModel, OptType
                ](
                    params=gpu.representation.params_buf.unsafe_ptr(),
                    model_state=gpu.representation.model_state_buf.unsafe_ptr(),
                    workspace=mcts_workspace,
                )
                var dyn_a = MuZeroDynGPU[
                    ACT, LATENT, BINS, DynModel, OptType
                ](
                    params=gpu.dynamics.params_buf.unsafe_ptr(),
                    model_state=gpu.dynamics.model_state_buf.unsafe_ptr(),
                    workspace=mcts_workspace,
                )
                var pred_a = MuZeroPredGPU[
                    ACT, LATENT, BINS, PredModel, OptType
                ](
                    params=gpu.prediction.params_buf.unsafe_ptr(),
                    model_state=gpu.prediction.model_state_buf.unsafe_ptr(),
                    workspace=mcts_workspace,
                )

                var obs_t_new = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, OBS),
                    MutAnyOrigin,
                ](obs_buf.unsafe_ptr())
                var ep_t_new = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](episode_steps_buf.unsafe_ptr())
                var lm_t_new = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin
                ](legal_masks_buf.unsafe_ptr())

                comptime if Self.Config.PolicyMode.IS_GUMBEL:
                    # Gumbel-MuZero: Gumbel-Top-k + Sequential Halving +
                    # σ(Q) − N/(1+ΣN) interior selection. Legal mask
                    # disabled for single-player envs (all-ones).
                    gumbel_planner.search_gpu[
                        MuZeroRepGPU[OBS, LATENT, RepModel, OptType],
                        MuZeroDynGPU[
                            ACT, LATENT, BINS, DynModel, OptType
                        ],
                        MuZeroPredGPU[
                            ACT, LATENT, BINS, PredModel, OptType
                        ],
                    ](
                        ctx,
                        rep_a,
                        dyn_a,
                        pred_a,
                        obs_t_new,
                        apply_legal=False,
                        rng_seed=UInt32(total_steps),
                    )
                    # Gumbel-argmax (mctx convention): adds independent
                    # Gumbel noise per env to log(π̂) then argmaxes.
                    # Sampling from π̂ would collapse all envs onto the
                    # same action because the improved policy is heavily
                    # peaked (σ_scale × Q_diff dominates the softmax).
                    gumbel_planner.extract_actions_gumbel(
                        ctx,
                        rng_seed=UInt32(total_steps),
                        gumbel_scale=1.0,
                    )
                    ctx.enqueue_copy(
                        actions_buf, gumbel_planner.actions_out
                    )
                    ctx.enqueue_copy(
                        gpu.mcts_step_policy_buf,
                        gumbel_planner.state.policies_out,
                    )
                    ctx.enqueue_copy(
                        gpu.mcts_step_value_buf,
                        gumbel_planner.root_value_out,
                    )
                else:
                    # Vanilla MuZero PUCT — current production path.
                    generic_planner.search_gpu[
                        MuZeroRepGPU[OBS, LATENT, RepModel, OptType],
                        MuZeroDynGPU[
                            ACT, LATENT, BINS, DynModel, OptType
                        ],
                        MuZeroPredGPU[
                            ACT, LATENT, BINS, PredModel, OptType
                        ],
                    ](
                        ctx,
                        rep_a,
                        dyn_a,
                        pred_a,
                        obs_t_new,
                        rng_seed=UInt32(total_steps),
                    )
                    generic_planner.extract_actions_temp[TEMP_THRESHOLD=0](
                        ctx,
                        ep_t_new,
                        lm_t_new,
                        rng_seed=UInt32(total_steps),
                        temp_min=self.temperature,
                    )
                    ctx.enqueue_copy(
                        actions_buf, generic_planner.actions_out
                    )
                    ctx.enqueue_copy(
                        gpu.mcts_step_policy_buf,
                        generic_planner.policies_out,
                    )
                    ctx.enqueue_copy(
                        gpu.mcts_step_value_buf,
                        generic_planner.root_value_out,
                    )

            # ── 2.99. Action telemetry — tally per-action selections +
            # per-env switches (small kernels, only single-threaded over
            # N_ENVS so cost is trivial). The accumulators are reset in
            # the print block after each window.
            var act_hist_in_t = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ](actions_buf.unsafe_ptr())
            var act_hist_out_t = LayoutTensor[
                dtype, Layout.row_major(ACT), MutAnyOrigin
            ](action_hist_buf.unsafe_ptr())
            comptime run_act_hist = action_histogram_kernel[
                Self.n_envs, ACT, dtype
            ]
            ctx.enqueue_function[run_act_hist](
                act_hist_in_t,
                act_hist_out_t,
                grid_dim=(1,),
                block_dim=(1,),
            )

            var prev_act_t = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ](prev_actions_buf.unsafe_ptr())
            var sw_count_t = LayoutTensor[
                dtype, Layout.row_major(1), MutAnyOrigin
            ](switch_count_buf.unsafe_ptr())
            comptime run_act_switch = action_switch_kernel[
                Self.n_envs, dtype
            ]
            ctx.enqueue_function[run_act_switch](
                act_hist_in_t,
                prev_act_t,
                sw_count_t,
                grid_dim=(1,),
                block_dim=(1,),
            )

            # ── 3. GPU environment step ──────────────────────────────
            E.step_kernel_gpu[Self.n_envs, E.STATE_SIZE, OBS](
                ctx,
                states_buf,
                actions_buf,
                rewards_buf,
                dones_buf,
                terminated_buf,
                obs_buf,
                rng_seed=UInt64(total_steps),
                workspace_ptr=workspace_buf.unsafe_ptr(),
            )

            # ── 4. Store transitions + MCTS targets in GPU replay ────
            # dones_buf (term|trunc) drives sequence-rejection in sampling;
            # terminated_buf (term-only) is the bootstrap mask returned in
            # batch_dones — the n-step TD target uses `(1 - terminated) *
            # V(s_{t+n})` so truncation does NOT zero the bootstrap.
            #
            # Convert ``actions_buf`` (scalar [N_ENVS]) → ``actions_onehot_buf``
            # ([N_ENVS × ACT]) before storing. The replay store kernel
            # expects ACTION_DIM consecutive floats per env per slot, and
            # ACTION_DIM = ACT for the MuZero replay (because the dyn
            # network's action input is one-hot). Without this conversion
            # the store would read out-of-bounds scalar actions of other
            # envs as if they were one-hot bits → dyn sees garbage →
            # action-blind collapse (see ``docs/MUZERO_AUDIT.md``
            # Phase-K action-encoding fix).
            var actions_scalar_t = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ](actions_buf.unsafe_ptr())
            var actions_onehot_t = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin
            ](actions_onehot_buf.unsafe_ptr())
            comptime ONEHOT_TOTAL = Self.n_envs * ACT
            comptime ONEHOT_BLOCKS = (ONEHOT_TOTAL + TPB - 1) // TPB
            comptime run_onehot = scalar_to_onehot_actions_kernel[
                Self.n_envs, ACT, dtype
            ]
            ctx.enqueue_function[run_onehot](
                actions_onehot_t,
                actions_scalar_t,
                grid_dim=(ONEHOT_BLOCKS,),
                block_dim=(TPB,),
            )

            var pre_store_slot = gpu.replay.write_idx
            gpu.replay.store_with_termination(
                ctx, actions_onehot_buf, rewards_buf, dones_buf, terminated_buf
            )
            # PER hook (Phase H13): assign max_priority to the slot just
            # written for every env, so new transitions get sampled at
            # least once before being prioritized down. No-op when use_per
            # is False — `_per_record_new_transitions` only updates the
            # tree, which is unused unless _per_sample_indices is called.
            if use_per:
                self._per_record_new_transitions[Self.n_envs, PER_ENV_CAP](
                    gpu, pre_store_slot
                )

            # Store MCTS targets (policies from extract_actions, values from staging)
            var mcts_pol_in_t = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin
            ](gpu.mcts_step_policy_buf.unsafe_ptr())
            var mcts_val_in_t = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ](gpu.mcts_step_value_buf.unsafe_ptr())
            var mcts_tp_in_t = LayoutTensor[
                DType.uint8, Layout.row_major(Self.n_envs), MutAnyOrigin
            ](gpu.mcts_step_to_play_buf.unsafe_ptr())
            var mcts_pol_buf_t = LayoutTensor[
                dtype,
                Layout.row_major(Self.n_envs * PER_ENV_CAP * ACT),
                MutAnyOrigin,
            ](gpu.mcts_policy_buf.unsafe_ptr())
            var mcts_val_buf_t = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs * PER_ENV_CAP), MutAnyOrigin
            ](gpu.mcts_value_buf.unsafe_ptr())
            var mcts_tp_buf_t = LayoutTensor[
                DType.uint8,
                Layout.row_major(Self.n_envs * PER_ENV_CAP),
                MutAnyOrigin,
            ](gpu.mcts_to_play_buf.unsafe_ptr())
            var prev_write_idx = Scalar[DType.int32](
                (gpu.replay.write_idx - 1 + PER_ENV_CAP) % PER_ENV_CAP
            )

            @parameter
            @always_inline
            def store_targets_wrapper(
                pi: LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin
                ],
                vi: LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ],
                ti: LayoutTensor[
                    DType.uint8,
                    Layout.row_major(Self.n_envs),
                    MutAnyOrigin,
                ],
                pb: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * PER_ENV_CAP * ACT),
                    MutAnyOrigin,
                ],
                vb: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * PER_ENV_CAP),
                    MutAnyOrigin,
                ],
                tb: LayoutTensor[
                    DType.uint8,
                    Layout.row_major(Self.n_envs * PER_ENV_CAP),
                    MutAnyOrigin,
                ],
                widx: Scalar[DType.int32],
            ):
                store_mcts_targets_kernel[Self.n_envs, PER_ENV_CAP, ACT, dtype](
                    pi, vi, ti, pb, vb, tb, widx
                )

            ctx.enqueue_function[store_targets_wrapper](
                mcts_pol_in_t,
                mcts_val_in_t,
                mcts_tp_in_t,
                mcts_pol_buf_t,
                mcts_val_buf_t,
                mcts_tp_buf_t,
                prev_write_idx,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )

            # ── 5. GPU episode tracking ──────────────────────────────
            var rewards_t = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ](rewards_buf.unsafe_ptr())
            var dones_t = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ](dones_buf.unsafe_ptr())
            var ep_rew_t = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ](episode_rewards_buf.unsafe_ptr())
            var ep_steps_t = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ](episode_steps_buf.unsafe_ptr())

            comptime run_accum = accumulate_rewards_kernel[dtype, Self.n_envs]
            ctx.enqueue_function[run_accum](
                ep_rew_t,
                rewards_t,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
            comptime run_incr = increment_steps_kernel[dtype, Self.n_envs]
            ctx.enqueue_function[run_incr](
                ep_steps_t,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )

            var rew_sum_t = LayoutTensor[
                dtype, Layout.row_major(1), MutAnyOrigin
            ](gpu_reward_sum_buf.unsafe_ptr())
            var ep_count_t = LayoutTensor[
                dtype, Layout.row_major(1), MutAnyOrigin
            ](gpu_episode_count_buf.unsafe_ptr())
            comptime run_log = log_and_reset_completed_kernel[
                dtype, Self.n_envs
            ]
            ctx.enqueue_function[run_log](
                dones_t,
                ep_rew_t,
                ep_steps_t,
                rew_sum_t,
                ep_count_t,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )

            # ── 8. Selective reset of done envs (GPU) ────────────────
            E.selective_reset_kernel_gpu[Self.n_envs, E.STATE_SIZE](
                ctx,
                states_buf,
                dones_buf,
                rng_seed=UInt64(total_steps),
            )
            # Re-extract obs after reset. Bug F (2026-05-05): the previous
            # `step_kernel_gpu` call here applied actions_buf a SECOND time,
            # silently advancing physics by an extra step per outer iteration
            # while overwriting rewards_buf/dones_buf without accumulating
            # them. Net effect: env ran at 2× speed but rewards counted only
            # half of steps → MuZero CartPole reported ~10-step episodes
            # vs the random-policy baseline of ~22 steps, masking the actual
            # learning signal. Replaced with read-only extract_obs that
            # populates obs_buf from the just-reset state without touching
            # physics or counters.
            E.extract_obs_kernel_gpu[Self.n_envs, E.STATE_SIZE, OBS](
                ctx, states_buf, obs_buf
            )

            total_steps += Self.n_envs
            self.total_steps += Self.n_envs

            # ── 9. GPU training (sampling + targets + training all on GPU) ──
            if (
                total_steps >= warmup_steps
                and gpu.replay.is_ready[Self.Config.unroll_steps + 1]()
            ):
                var per_progress = (
                    Float64(total_steps) / Float64(num_steps)
                ) if num_steps > 0 else Float64(0.0)
                for _ in range(grad_steps):
                    _ = self.update_gpu(
                        ctx,
                        gpu,
                        use_reanalyze=use_reanalyze,
                        use_per=use_per,
                        per_progress=per_progress,
                    )
                gpu_train_count += grad_steps

                # No periodic CPU sync needed — MCTS runs on GPU
                # with the same GPU network states used for training
                _ = next_sync  # Unused

            # Temperature schedule — muzero-general step schedule
            # (games/cartpole.py:86-99). Floor 0.25, never fully greedy:
            #   frac < 0.5  → 1.0   (full exploration)
            #   frac < 0.75 → 0.5   (moderate)
            #   else        → 0.25  (low but non-zero)
            # Linear-decay-to-0.01 pre-fix collapsed to greedy at ~40% of
            # training horizon → no exploration once policy committed →
            # MuZero CartPole stuck at reward 7-8. See Bug E follow-up
            # (2026-05-04 temperature audit).
            if self.temperature_decay_steps > 0:
                var _frac = Float64(self.total_steps) / Float64(
                    self.temperature_decay_steps
                )
                if _frac < 0.5:
                    self.temperature = 1.0
                elif _frac < 0.75:
                    self.temperature = 0.5
                else:
                    self.temperature = 0.25

            # ── 11. Progress reporting ───────────────────────────────
            if total_steps >= next_print:
                # Sync episode stats from GPU
                ctx.enqueue_copy(reward_sum_host, gpu_reward_sum_buf)
                ctx.enqueue_copy(episode_count_host, gpu_episode_count_buf)
                ctx.enqueue_copy(action_hist_host, action_hist_buf)
                ctx.enqueue_copy(switch_count_host, switch_count_buf)
                ctx.synchronize()

                var total_reward = Float64(reward_sum_host[0])
                var total_eps = Int(Float64(episode_count_host[0]))
                var avg_reward = total_reward / Float64(
                    total_eps
                ) if total_eps > 0 else Float64(0.0)

                # Diagnostic: per-network param L2 norms — verifies
                # training is actually moving the weights. Compare
                # against the previous print to see param-norm delta.
                var norms = self._net_param_l2(ctx, gpu)
                var rep_n = norms[0]
                var dyn_n = norms[1]
                var pred_n = norms[2]
                var d_rep = rep_n - last_rep_n
                var d_dyn = dyn_n - last_dyn_n
                var d_pred = pred_n - last_pred_n
                last_rep_n = rep_n
                last_dyn_n = dyn_n
                last_pred_n = pred_n

                # Replay size = per-env count × n_envs (true total).
                var replay_total = gpu.replay.size * Self.n_envs

                clear_progress_bar()
                print(
                    "Steps: "
                    + String(total_steps)
                    + " | Episodes: "
                    + String(total_eps)
                    + " | Avg Reward: "
                    + String(Int(avg_reward))
                    + " | Train: "
                    + String(self.train_step_count)
                    + " | Replay: "
                    + String(replay_total)
                    + " | |W| rep/dyn/pred: "
                    + String(rep_n)
                    + "/"
                    + String(dyn_n)
                    + "/"
                    + String(pred_n)
                    + " (Δ "
                    + String(d_rep)
                    + "/"
                    + String(d_dyn)
                    + "/"
                    + String(d_pred)
                    + ")"
                )

                # Action histogram across this print window — counts how
                # many times each action was actually executed in the env.
                var act_line = String("    actions: [")
                var hist_total = Float64(0.0)
                for ai in range(ACT):
                    hist_total += Float64(action_hist_host[ai])
                for ai in range(ACT):
                    var c = Float64(action_hist_host[ai])
                    var pct = (
                        c / hist_total * 100.0
                    ) if hist_total > 0 else Float64(0.0)
                    act_line += (
                        String("a") + String(ai) + String("=") + String(Int(c))
                    )
                    act_line += String(" (") + String(Int(pct)) + String("%)")
                    if ai < ACT - 1:
                        act_line += String(", ")
                act_line += String("]  total=") + String(Int(hist_total))
                var switches = Float64(switch_count_host[0])
                var switch_pct = (
                    switches / hist_total * 100.0
                ) if hist_total > 0 else Float64(0.0)
                act_line += String("  switches=") + String(Int(switches))
                act_line += (
                    String(" (") + String(Int(switch_pct)) + String("%)")
                )
                print(act_line)
                action_hist_buf.enqueue_fill(Scalar[dtype](0.0))
                switch_count_buf.enqueue_fill(Scalar[dtype](0.0))

                # ── MCTS probe: download a sample env's actions /
                # improved policy / root value to verify the planner is
                # actually producing sensible outputs. Synchronize first
                # so we observe the latest values, not stale ones from
                # before the previous round of MCTS calls flushed.
                ctx.synchronize()
                comptime PROBE_ENVS = 4 if Self.n_envs >= 4 else Self.n_envs
                var probe_acts = ctx.enqueue_create_host_buffer[dtype](
                    PROBE_ENVS
                )
                var probe_pol = ctx.enqueue_create_host_buffer[dtype](
                    PROBE_ENVS * ACT
                )
                var probe_val = ctx.enqueue_create_host_buffer[dtype](
                    PROBE_ENVS
                )
                # Probe the agent-side buffers that downstream training
                # consumes — these are the copies of generic_planner /
                # gumbel_planner outputs.
                ctx.enqueue_copy(probe_acts, actions_buf)
                ctx.enqueue_copy(probe_pol, gpu.mcts_step_policy_buf)
                ctx.enqueue_copy(probe_val, gpu.mcts_step_value_buf)
                ctx.synchronize()

                var probe_line = String("    MCTS probe (envs 0..")
                probe_line += String(PROBE_ENVS - 1) + String("): ")
                for pe in range(PROBE_ENVS):
                    probe_line += String("env") + String(pe) + String("[a=")
                    probe_line += String(Int(Float64(probe_acts[pe])))
                    probe_line += String(" V=")
                    var v_f = Float64(probe_val[pe])
                    probe_line += String(
                        Float64(Int(v_f * 100.0)) / 100.0
                    )
                    probe_line += String(" π=[")
                    for a in range(ACT):
                        if a > 0:
                            probe_line += String(",")
                        var p = Float64(probe_pol[pe * ACT + a])
                        probe_line += String(
                            Float64(Int(p * 1000.0)) / 1000.0
                        )
                    probe_line += String("]] ")
                print(probe_line)

                # ── Replay-dump probe: download env-0's last few slots
                # and print the stored (action, reward, done, π, V) tuples.
                # If the agent-side MCTS probe above looks reasonable but
                # what landed in replay is misaligned/garbled, the bug is
                # in store_with_termination + store_mcts_targets timing
                # rather than upstream. Reads env 0 only — each per-env
                # slice is independent in the [N_ENVS, PER_ENV_CAP, ...]
                # layout. WINDOW slots ending at the most recent write.
                comptime DUMP_WINDOW = 4
                var dump_w = gpu.replay.write_idx
                var dump_start = (
                    dump_w - DUMP_WINDOW + PER_ENV_CAP
                ) % PER_ENV_CAP

                # Download env 0's full per-env slices (cheap — single
                # PER_ENV_CAP-sized chunks). For ACTION_DIM=1 (discrete),
                # the action buffer stores one float per slot.
                var dump_obs = ctx.enqueue_create_host_buffer[dtype](
                    PER_ENV_CAP * OBS
                )
                var dump_act = ctx.enqueue_create_host_buffer[dtype](
                    PER_ENV_CAP
                )
                var dump_rew = ctx.enqueue_create_host_buffer[dtype](
                    PER_ENV_CAP
                )
                var dump_done = ctx.enqueue_create_host_buffer[dtype](
                    PER_ENV_CAP
                )
                var dump_pol = ctx.enqueue_create_host_buffer[dtype](
                    PER_ENV_CAP * ACT
                )
                var dump_val = ctx.enqueue_create_host_buffer[dtype](
                    PER_ENV_CAP
                )

                # Layouts are [N_ENVS, PER_ENV_CAP, *]; env 0 occupies
                # the leading slab of size PER_ENV_CAP * (field stride).
                var obs_e0_ptr = gpu.replay.obs_buf.create_sub_buffer[
                    dtype
                ](0, PER_ENV_CAP * OBS)
                var act_e0_ptr = gpu.replay.actions_buf.create_sub_buffer[
                    dtype
                ](0, PER_ENV_CAP)
                var rew_e0_ptr = gpu.replay.rewards_buf.create_sub_buffer[
                    dtype
                ](0, PER_ENV_CAP)
                var done_e0_ptr = gpu.replay.dones_buf.create_sub_buffer[
                    dtype
                ](0, PER_ENV_CAP)
                var pol_e0_ptr = gpu.mcts_policy_buf.create_sub_buffer[
                    dtype
                ](0, PER_ENV_CAP * ACT)
                var val_e0_ptr = gpu.mcts_value_buf.create_sub_buffer[
                    dtype
                ](0, PER_ENV_CAP)
                ctx.enqueue_copy(dump_obs, obs_e0_ptr)
                ctx.enqueue_copy(dump_act, act_e0_ptr)
                ctx.enqueue_copy(dump_rew, rew_e0_ptr)
                ctx.enqueue_copy(dump_done, done_e0_ptr)
                ctx.enqueue_copy(dump_pol, pol_e0_ptr)
                ctx.enqueue_copy(dump_val, val_e0_ptr)
                ctx.synchronize()

                print(
                    "    Replay dump (env 0, slots "
                    + String(Int(dump_start))
                    + ".."
                    + String(Int((dump_w - 1 + PER_ENV_CAP) % PER_ENV_CAP))
                    + ", write_idx="
                    + String(Int(dump_w))
                    + "):"
                )
                for i in range(DUMP_WINDOW):
                    var s = (dump_start + i) % PER_ENV_CAP
                    var dump_l = String("      slot ")
                    dump_l += String(Int(s)) + String(": obs=[")
                    for d in range(OBS):
                        if d > 0:
                            dump_l += String(",")
                        var o = Float64(dump_obs[s * OBS + d])
                        dump_l += String(
                            Float64(Int(o * 100.0)) / 100.0
                        )
                    dump_l += String("] a=")
                    dump_l += String(Int(Float64(dump_act[s])))
                    dump_l += String(" r=")
                    dump_l += String(
                        Float64(Int(Float64(dump_rew[s]) * 100.0)) / 100.0
                    )
                    dump_l += String(" d=")
                    dump_l += String(Int(Float64(dump_done[s])))
                    dump_l += String(" π=[")
                    for a in range(ACT):
                        if a > 0:
                            dump_l += String(",")
                        var p = Float64(dump_pol[s * ACT + a])
                        dump_l += String(
                            Float64(Int(p * 1000.0)) / 1000.0
                        )
                    dump_l += String("] V=")
                    var v = Float64(dump_val[s])
                    dump_l += String(
                        Float64(Int(v * 100.0)) / 100.0
                    )
                    print(dump_l)

                # ── Training target probe: download the value_targets_buf
                # (in h-transformed space) from the last update_gpu() call
                # and decode back to raw via h⁻¹. This reveals what the
                # value head is actually being trained against:
                #   * If targets ≈ 0 always: value loss has no signal
                #     (broken n-step kernel or bootstrap).
                #   * If targets stuck near a constant (e.g. 5): value
                #     head fits constant, MCTS rollouts return constant Q.
                #   * If targets reflect real returns (~10-22 for CartPole
                #     random play): value learning is OK, bug is elsewhere.
                # Also downloads reward_targets (scalar-transformed; for
                # CartPole real rewards are +1 → h(1)≈0.42).
                comptime BATCH_TGT = Self.Config.batch_size
                comptime K_TGT = Self.Config.unroll_steps
                comptime VT_SIZE = (K_TGT + 1) * BATCH_TGT
                comptime RT_SIZE = K_TGT * BATCH_TGT
                var probe_vt = ctx.enqueue_create_host_buffer[dtype](VT_SIZE)
                var probe_rt = ctx.enqueue_create_host_buffer[dtype](RT_SIZE)
                ctx.enqueue_copy(probe_vt, gpu.value_targets_buf)
                ctx.enqueue_copy(probe_rt, gpu.reward_targets_buf)
                ctx.synchronize()

                # Summary: mean / min / max of value_target at k=0 and k=K
                # (first and last unroll step). Apply h⁻¹ to map back to
                # raw return space for readability.
                def _h_inv(y: Float64) -> Float64:
                    var sgn = 1.0 if y >= 0.0 else -1.0
                    var ay = y if y >= 0.0 else -y
                    var eps = 0.001
                    var inner = (1.0 + 4.0 * eps * (ay + 1.0 + eps)) ** 0.5
                    var f = (inner - 1.0) / (2.0 * eps)
                    return sgn * (f * f - 1.0)

                # k=0 slice (the most directly bootstrapped target).
                var vt_sum = 0.0
                var vt_min = 1e18
                var vt_max = -1e18
                for b in range(BATCH_TGT):
                    var vh = Float64(probe_vt[0 * BATCH_TGT + b])
                    var vraw = _h_inv(vh)
                    vt_sum += vraw
                    if vraw < vt_min:
                        vt_min = vraw
                    if vraw > vt_max:
                        vt_max = vraw
                var vt_mean_k0 = vt_sum / Float64(BATCH_TGT)

                # k=K slice (last unroll target).
                var vt_sum_kK = 0.0
                for b in range(BATCH_TGT):
                    vt_sum_kK += _h_inv(
                        Float64(probe_vt[K_TGT * BATCH_TGT + b])
                    )
                var vt_mean_kK = vt_sum_kK / Float64(BATCH_TGT)

                # Reward target sample (k=0 slice). For CartPole
                # h(1.0) ≈ 0.4142 (since (1-1)+0.001*1=0.001 so
                # f≈(sqrt(1.008)-1)/.002≈2, no wait — h(x)=sgn(x)
                # (sqrt(|x|+1)-1)+eps·x; h(1)=sqrt(2)-1+0.001=0.4152).
                # So we expect raw r=1 → stored ≈ 0.415.
                var rt_sum = 0.0
                for b in range(BATCH_TGT):
                    rt_sum += Float64(probe_rt[0 * BATCH_TGT + b])
                var rt_mean_k0 = rt_sum / Float64(BATCH_TGT)

                var probe_t_line = String("    Train target probe (batch=")
                probe_t_line += String(BATCH_TGT) + String("): ")
                probe_t_line += String("V_target k=0 [mean=")
                probe_t_line += String(
                    Float64(Int(vt_mean_k0 * 100.0)) / 100.0
                )
                probe_t_line += String(", min=")
                probe_t_line += String(
                    Float64(Int(vt_min * 100.0)) / 100.0
                )
                probe_t_line += String(", max=")
                probe_t_line += String(
                    Float64(Int(vt_max * 100.0)) / 100.0
                )
                probe_t_line += String("] V_target k=K [mean=")
                probe_t_line += String(
                    Float64(Int(vt_mean_kK * 100.0)) / 100.0
                )
                probe_t_line += String("] h(R_target) k=0 mean=")
                probe_t_line += String(
                    Float64(Int(rt_mean_k0 * 1000.0)) / 1000.0
                )
                probe_t_line += String(" (real r≈1 → h(r)≈0.41)")
                print(probe_t_line)

                # ── Dynamics action-discrimination probe: feed the
                # same hidden state through ``dyn(s, a=0)`` vs
                # ``dyn(s, a=1)`` and print L2 differences. If both
                # outputs are nearly identical, the dynamics network
                # has collapsed to action-blind — MCTS rollouts return
                # the same Q for both actions, policy improvement
                # halts. PUCT path only (probes ``generic_planner``);
                # extend to Gumbel by switching the source buffer.
                comptime if not Self.Config.PolicyMode.IS_GUMBEL:
                    var hidden_host = ctx.enqueue_create_host_buffer[
                        dtype
                    ](LATENT)
                    var hidden_e0 = (
                        generic_planner.state.hidden_states
                        .create_sub_buffer[dtype](0, LATENT)
                    )
                    ctx.enqueue_copy(hidden_host, hidden_e0)
                    ctx.synchronize()

                    var probe_din_host = (
                        ctx.enqueue_create_host_buffer[dtype](2 * DYN_IN)
                    )
                    for i in range(LATENT):
                        probe_din_host[0 * DYN_IN + i] = hidden_host[i]
                        probe_din_host[1 * DYN_IN + i] = hidden_host[i]
                    for a in range(ACT):
                        probe_din_host[0 * DYN_IN + LATENT + a] = (
                            Scalar[dtype](0.0)
                        )
                        probe_din_host[1 * DYN_IN + LATENT + a] = (
                            Scalar[dtype](0.0)
                        )
                    probe_din_host[0 * DYN_IN + LATENT + 0] = (
                        Scalar[dtype](1.0)
                    )
                    probe_din_host[1 * DYN_IN + LATENT + 1] = (
                        Scalar[dtype](1.0)
                    )

                    var probe_din_dev = ctx.enqueue_create_buffer[dtype](
                        2 * DYN_IN
                    )
                    var probe_dout_dev = ctx.enqueue_create_buffer[dtype](
                        2 * DYN_OUT
                    )
                    var probe_dyn_ws = (
                        ctx.enqueue_create_buffer[dtype](
                            2 * MAX_WS_2 if MAX_WS_2 > 0 else 1
                        )
                    )
                    ctx.enqueue_copy(probe_din_dev, probe_din_host)

                    var probe_din_t = LayoutTensor[
                        dtype,
                        Layout.row_major(2, DYN_IN_DIM),
                        MutAnyOrigin,
                    ](probe_din_dev.unsafe_ptr())
                    var probe_dout_t = LayoutTensor[
                        dtype,
                        Layout.row_major(2, DYN_OUT_DIM),
                        MutAnyOrigin,
                    ](probe_dout_dev.unsafe_ptr())

                    DynNet.forward_gpu[2](
                        ctx,
                        probe_din_t,
                        probe_dout_t,
                        gpu.dynamics.params_view(),
                        gpu.dynamics.model_state_view(),
                        probe_dyn_ws,
                    )

                    var probe_dout_host = (
                        ctx.enqueue_create_host_buffer[dtype](2 * DYN_OUT)
                    )
                    ctx.enqueue_copy(probe_dout_host, probe_dout_dev)
                    ctx.synchronize()

                    # Hidden-prefix L2 distance + norm.
                    var hd_sq = 0.0
                    var h0_sq = 0.0
                    for i in range(LATENT):
                        var x0 = Float64(probe_dout_host[i])
                        var x1 = Float64(
                            probe_dout_host[DYN_OUT + i]
                        )
                        hd_sq += (x0 - x1) * (x0 - x1)
                        h0_sq += x0 * x0
                    var h_diff = hd_sq ** 0.5
                    var h_norm = h0_sq ** 0.5

                    # Reward-bins L2 distance.
                    var rd_sq = 0.0
                    for i in range(BINS):
                        var x0 = Float64(
                            probe_dout_host[LATENT + i]
                        )
                        var x1 = Float64(
                            probe_dout_host[DYN_OUT + LATENT + i]
                        )
                        rd_sq += (x0 - x1) * (x0 - x1)
                    var r_diff = rd_sq ** 0.5

                    var dyn_line = String("    Dyn action-diff: ")
                    dyn_line += String("Δhidden_L2=") + String(
                        Float64(Int(h_diff * 10000.0)) / 10000.0
                    )
                    dyn_line += String(" (relative ")
                    dyn_line += String(
                        Float64(Int(
                            (h_diff / (h_norm + 1e-9)) * 10000.0
                        )) / 10000.0
                    )
                    dyn_line += String(")  Δreward_logits_L2=")
                    dyn_line += String(
                        Float64(Int(r_diff * 10000.0)) / 10000.0
                    )
                    print(dyn_line)

                # Log to metrics
                if total_eps > 0:
                    metrics.log_episode[dtype](
                        total_eps,
                        Scalar[dtype](avg_reward),
                        0,
                        0.0,
                    )

                # Reset GPU accumulators for next interval
                gpu_reward_sum_buf.enqueue_fill(Scalar[dtype](0.0))
                gpu_episode_count_buf.enqueue_fill(Scalar[dtype](0.0))

                # LR decay — match muzero-general's exponential schedule
                # (`trainer.py:275-283` lr = lr_init * decay_rate^(train_step
                # / decay_steps), with cartpole.py defaults rate=0.8,
                # steps=1000). Without this, constant LR + DC drift makes
                # rep weights grow unbounded and post-min-max hidden state
                # saturates uniform across obs. Applied at print boundary
                # (coarse vs reference's per-step but cheap and adequate for
                # ~2000-env-step intervals).
                if lr_decay_rate < 1.0:
                    var lr_scale = exp(
                        log(lr_decay_rate)
                        * Float64(self.train_step_count)
                        / Float64(lr_decay_steps)
                    )
                    gpu.representation.set_lr_scale(lr_scale, ctx)
                    gpu.dynamics.set_lr_scale(lr_scale, ctx)
                    gpu.prediction.set_lr_scale(lr_scale, ctx)

                next_print += print_every

        # Final sync
        gpu.download_to(self.state, ctx)

        return metrics

    # ══════════════════════════════════════════════════════════════════════
    # Self-Play GPU Training (for GPUTwoPlayerDiscreteEnv)
    # ══════════════════════════════════════════════════════════════════════

    def train_selfplay_gpu[
        E: GPUTwoPlayerDiscreteEnv,
        GPUEval: GPUEvaluator = RandomOpponent,
        GPUEval2: GPUEvaluator = RandomOpponent,
        temp_threshold: Int = 15,
    ](
        mut self,
        ctx: DeviceContext,
        num_iters: Int = 25,
        steps_per_iter: Int = 5000,
        train_epochs: Int = 5,
        warmup_iters: Int = 1,
        arena_threshold: Float64 = 0.55,
        do_eval: Bool = True,
        do_eval2: Bool = False,
        do_arena: Bool = False,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "muzero.ckpt",
        use_reanalyze: Bool = False,
        reanalyze_per_iter: Int = 64,
        logger: Optional[UnsafePointer[Self.L, MutAnyOrigin]] = None,
        diag_every: Int = 0,
        use_one_cycle: Bool = False,
    ) raises -> TrainingMetrics:
        """Train MuZero via GPU self-play with batch-then-train loop.

        Each iteration:
          1. Collect self-play data (frozen network, batched GPU MCTS)
          2. Train for train_epochs on GPU replay buffer
          3. GPU evaluation vs GPUEval opponent
          4. GPU arena: new model vs best model (accept/reject)
          5. Checkpoint

        Parameters:
            E: GPU two-player environment (TicTacToe, Chess, Go, etc.).
            GPUEval: GPU evaluator for opponent evaluation.
            GPUEval2: GPU evaluator for opponent evaluation.
            temp_threshold: Use temp=1 for first N moves, then argmax.

        Args:
            ctx: GPU device context.
            num_iters: Total training iterations.
            steps_per_iter: Env transitions per iteration (across n_envs).
            train_epochs: Training epochs per iteration.
            warmup_iters: Random play iterations before MCTS starts.
            arena_threshold: Win rate to accept new model (0.5-1.0).
            do_eval: Whether to evaluate vs GPUEval after each iteration.
            do_eval2: Whether to evaluate vs GPUEval2 after each iteration.
            do_arena: Whether to run arena comparison (new vs best).
            checkpoint_every: Save checkpoint every N iterations (0=off).
            checkpoint_path: Path for checkpoint files.
            use_reanalyze: If True, run CPU-side reanalyze
                (re-run MCTS on stored positions with the latest networks)
                between iterations to refresh stale value/policy targets.
                muzero-general considers this part of the canonical
                stability story; off by default for backwards compatibility.
            reanalyze_per_iter: Number of positions to reanalyze each
                iteration when use_reanalyze=True. Capped by buffer size.
            logger: Optional logger for diagnostic logging.
            diag_every: Print diagnostic information every N iterations.
            use_one_cycle: Whether to use one-cycle learning rate schedule.

        Compile-time parameters:
            temp_threshold: Use temp=1 for first N moves, then argmax.
                Comptime so it can be passed directly to GPU kernels
                (Mojo's GPU enqueue does not accept runtime Int args).

        Returns:
            TrainingMetrics with game statistics.
        """
        comptime ACT = Self.Config.action_dim
        comptime OBS = Self.Config.obs_dim
        comptime LATENT = Self.Config.latent_dim
        comptime BINS = Self.Config.num_bins
        comptime NUM_SIMS = Self.Config.num_simulations
        comptime MAX_NODES = Self.Config.max_nodes
        comptime ENV_BLOCKS = (Self.n_envs + TPB - 1) // TPB
        comptime PER_ENV_CAP = 1000
        # MCTS batched-sims size (orchestrator absorbs round-loop + leaf
        # allocation). Cap by ACT so BATCH_SIMS > ACT can never create
        # duplicate leaf nodes; floor to a divisor of NUM_SIMS so no sims
        # are silently dropped (orchestrator asserts divisibility).
        comptime _REQ_BSIMS_SP = Self.Config.batch_sims
        comptime _CAP_BSIMS_SP = (
            _REQ_BSIMS_SP if _REQ_BSIMS_SP < ACT else ACT
        )
        comptime ORCH_BSIMS = (
            _CAP_BSIMS_SP if NUM_SIMS % _CAP_BSIMS_SP == 0
            else (
                4 if (NUM_SIMS % 4 == 0 and _CAP_BSIMS_SP >= 4)
                else (
                    2 if (NUM_SIMS % 2 == 0 and _CAP_BSIMS_SP >= 2)
                    else 1
                )
            )
        )
        comptime TOTAL_EXPAND = Self.n_envs * ORCH_BSIMS

        comptime RepModel = Self.Config.RepModel
        comptime DynModel = Self.Config.DynModel
        comptime PredModel = Self.Config.PredModel
        comptime OptType = Self.Config.OptType
        comptime RepNet = Network[RepModel, OptType]
        comptime DynNet = Network[DynModel, OptType]
        comptime PredNet = Network[PredModel, OptType]
        comptime PRED_OUT = Self.Config.PRED_OUT
        comptime DYN_IN = Self.Config.DYN_IN
        comptime DYN_OUT = Self.Config.DYN_OUT
        comptime REP_IN_DIM = RepModel.IN_DIM
        comptime REP_OUT_DIM = RepModel.OUT_DIM
        comptime DYN_IN_DIM = DynModel.IN_DIM
        comptime DYN_OUT_DIM = DynModel.OUT_DIM
        comptime PRED_IN_DIM = PredModel.IN_DIM
        comptime PRED_OUT_DIM = PredModel.OUT_DIM
        comptime NEGATE = Self.Config.Players.NEGATE_BACKUP
        comptime GS = E.STATE_SIZE

        # ── Create GPU training state ────────────────────────────
        comptime LocalGPUState = MuZeroGPUState[
            Self.Config, Self.n_envs, PER_ENV_CAP
        ]
        var gpu = LocalGPUState(ctx)
        gpu.upload_from(self.state, ctx)

        # ── GPU MCTS orchestrator ────────────────────────────────
        # Owns its own tree-state buffers (no separate GPUMCTSState).
        # ``PlayerMode.SelfPlay`` baked in via ``Self.Config.Players``
        # → negated backup in the expand+backup kernel.
        var generic_planner = GenericGPUMCTS[
            Self.n_envs, ACT, LATENT, BINS, MAX_NODES, NUM_SIMS,
            ORCH_BSIMS,
            Self.Config.PUCT,
            Self.Config.Noise,
            Self.Config.Players,
        ](ctx, gamma=self.gamma, v_min=self.v_min, v_max=self.v_max)
        # Gumbel-MuZero orchestrator — used when
        # ``Config.PolicyMode.IS_GUMBEL`` is True. Backup negation
        # comes from ``Self.Config.Players`` (= SelfPlay for board
        # games) baked into the kernel choice.
        var gumbel_planner = GumbelGPUMCTS[
            Self.n_envs, ACT, LATENT, BINS, MAX_NODES,
            Self.Config.PolicyMode.MAX_K, NUM_SIMS,
            Self.Config.Players,
        ](
            ctx, gamma=self.gamma, v_min=self.v_min, v_max=self.v_max,
            gumbel_scale=1.0,
        )

        # Network workspace (sized for batched: n_envs * BATCH_SIMS)
        comptime WS_R = RepModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_D = DynModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_P = PredModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime MAX_WS_1 = WS_R if WS_R > WS_D else WS_D
        comptime MAX_WS_2 = MAX_WS_1 if MAX_WS_1 > WS_P else WS_P
        comptime MCTS_WS = TOTAL_EXPAND * MAX_WS_2 if MAX_WS_2 > 0 else 1
        var mcts_workspace = ctx.enqueue_create_buffer[dtype](MCTS_WS)

        # ── GPU environment buffers ──────────────────────────────
        var states_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs * GS)
        var obs_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs * OBS)
        var actions_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var rewards_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var dones_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var terminated_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var legal_masks_buf = ctx.enqueue_create_buffer[dtype](
            Self.n_envs * ACT
        )

        # Episode tracking
        var ep_rew_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var ep_steps_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var rew_sum_buf = ctx.enqueue_create_buffer[dtype](1)
        var ep_count_buf = ctx.enqueue_create_buffer[dtype](1)
        var rew_sum_host = ctx.enqueue_create_host_buffer[dtype](1)
        var ep_count_host = ctx.enqueue_create_host_buffer[dtype](1)

        # ── Initialize games ─────────────────────────────────────
        E.reset_kernel_gpu[Self.n_envs, GS](ctx, states_buf, rng_seed=42)
        E.extract_obs_kernel_gpu[Self.n_envs, GS, OBS](
            ctx,
            states_buf,
            obs_buf,
            legal_masks_buf,
        )
        ep_rew_buf.enqueue_fill(Scalar[dtype](0.0))
        ep_steps_buf.enqueue_fill(Scalar[dtype](0.0))
        rew_sum_buf.enqueue_fill(Scalar[dtype](0.0))
        ep_count_buf.enqueue_fill(Scalar[dtype](0.0))
        ctx.synchronize()

        var metrics = TrainingMetrics(algorithm_name="MuZero-SelfPlay")
        var total_steps = 0

        # Plug logger into self for downstream methods.
        self.logger = logger
        self.diag_every = diag_every

        # Save initial params for arena comparison
        comptime REP_PS = RepModel.PARAM_SIZE
        comptime DYN_PS = DynModel.PARAM_SIZE
        comptime PRED_PS = PredModel.PARAM_SIZE
        comptime TOTAL_PS = REP_PS + DYN_PS + PRED_PS
        var best_params = alloc[Scalar[dtype]](TOTAL_PS)
        for i in range(REP_PS):
            best_params[i] = self.state.representation.params[i]
        for i in range(DYN_PS):
            best_params[REP_PS + i] = self.state.dynamics.params[i]
        for i in range(PRED_PS):
            best_params[REP_PS + DYN_PS + i] = self.state.prediction.params[i]
        var arena_accepts = 0
        var arena_rejects = 0

        # Param-norm diagnostic state. Initialized to the params right
        # after upload so the first iter's print reports delta over
        # iter 1's training, not delta-from-zero-init.
        var init_norms = self._net_param_l2(ctx, gpu)
        var last_rep_n = init_norms[0]
        var last_dyn_n = init_norms[1]
        var last_pred_n = init_norms[2]

        # ══════════════════════════════════════════════════════════
        # Iteration loop (batch-then-train)
        # ══════════════════════════════════════════════════════════
        for iter in range(num_iters):
            var use_mcts = iter >= warmup_iters

            # Re-upload params to GPU (training may have modified them)
            gpu.upload_from(self.state, ctx)

            # Reset episode counters for this iteration
            rew_sum_buf.enqueue_fill(Scalar[dtype](0.0))
            ep_count_buf.enqueue_fill(Scalar[dtype](0.0))
            ctx.synchronize()

            # ── 1. Collect self-play data (frozen network) ───────
            var iter_steps = 0
            while iter_steps < steps_per_iter:
                if use_mcts:
                    # ── Batched GPU MCTS with learned dynamics ───
                    # Adapters are PolicyMode-agnostic.
                    var rep_a_sp = MuZeroRepGPU[
                        OBS, LATENT, RepModel, OptType
                    ](
                        params=gpu.representation.params_buf.unsafe_ptr(),
                        model_state=gpu.representation.model_state_buf.unsafe_ptr(),
                        workspace=mcts_workspace,
                    )
                    var dyn_a_sp = MuZeroDynGPU[
                        ACT, LATENT, BINS, DynModel, OptType
                    ](
                        params=gpu.dynamics.params_buf.unsafe_ptr(),
                        model_state=gpu.dynamics.model_state_buf.unsafe_ptr(),
                        workspace=mcts_workspace,
                    )
                    var pred_a_sp = MuZeroPredGPU[
                        ACT, LATENT, BINS, PredModel, OptType
                    ](
                        params=gpu.prediction.params_buf.unsafe_ptr(),
                        model_state=gpu.prediction.model_state_buf.unsafe_ptr(),
                        workspace=mcts_workspace,
                    )

                    var obs_t_sp = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs, OBS),
                        MutAnyOrigin,
                    ](obs_buf.unsafe_ptr())
                    var lm_t_sp = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin
                    ](legal_masks_buf.unsafe_ptr())
                    var ep_steps_t_sp = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                    ](ep_steps_buf.unsafe_ptr())

                    comptime if Self.Config.PolicyMode.IS_GUMBEL:
                        # Gumbel-MuZero self-play. Caller populates the
                        # legal mask on ``gumbel_planner.state.legal_mask``
                        # — copy from agent's ``legal_masks_buf``.
                        ctx.enqueue_copy(
                            gumbel_planner.state.legal_mask,
                            legal_masks_buf,
                        )
                        gumbel_planner.search_gpu_selfplay[
                            MuZeroRepGPU[
                                OBS, LATENT, RepModel, OptType
                            ],
                            MuZeroDynGPU[
                                ACT, LATENT, BINS, DynModel, OptType
                            ],
                            MuZeroPredGPU[
                                ACT, LATENT, BINS, PredModel, OptType
                            ],
                        ](
                            ctx,
                            rep_a_sp,
                            dyn_a_sp,
                            pred_a_sp,
                            obs_t_sp,
                            rng_seed=UInt32(total_steps + iter_steps),
                        )
                        # Gumbel-argmax (mctx convention).
                        gumbel_planner.extract_actions_gumbel(
                            ctx,
                            rng_seed=UInt32(total_steps + iter_steps),
                            gumbel_scale=1.0,
                        )
                        ctx.enqueue_copy(
                            actions_buf, gumbel_planner.actions_out
                        )
                        ctx.enqueue_copy(
                            gpu.mcts_step_policy_buf,
                            gumbel_planner.state.policies_out,
                        )
                        ctx.enqueue_copy(
                            gpu.mcts_step_value_buf,
                            gumbel_planner.root_value_out,
                        )
                    else:
                        # Vanilla PUCT self-play (existing production path).
                        generic_planner.search_gpu_selfplay[
                            MuZeroRepGPU[
                                OBS, LATENT, RepModel, OptType
                            ],
                            MuZeroDynGPU[
                                ACT, LATENT, BINS, DynModel, OptType
                            ],
                            MuZeroPredGPU[
                                ACT, LATENT, BINS, PredModel, OptType
                            ],
                        ](
                            ctx,
                            rep_a_sp,
                            dyn_a_sp,
                            pred_a_sp,
                            obs_t_sp,
                            lm_t_sp,
                            rng_seed=UInt32(total_steps + iter_steps),
                        )
                        generic_planner.extract_actions_temp[
                            TEMP_THRESHOLD=temp_threshold
                        ](
                            ctx,
                            ep_steps_t_sp,
                            lm_t_sp,
                            rng_seed=UInt32(total_steps + iter_steps),
                            temp_min=Float64(0.0),
                        )
                        ctx.enqueue_copy(
                            actions_buf, generic_planner.actions_out
                        )
                        ctx.enqueue_copy(
                            gpu.mcts_step_policy_buf,
                            generic_planner.policies_out,
                        )
                        ctx.enqueue_copy(
                            gpu.mcts_step_value_buf,
                            generic_planner.root_value_out,
                        )

                else:
                    # Warmup: random legal actions
                    comptime run_warmup = uniform_random_legal_actions_kernel[
                        dtype, Self.n_envs, ACT
                    ]
                    var wa_t = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                    ](actions_buf.unsafe_ptr())
                    var wl_t = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin
                    ](legal_masks_buf.unsafe_ptr())
                    ctx.enqueue_function[run_warmup](
                        wa_t,
                        wl_t,
                        Scalar[DType.uint32](UInt32(total_steps + iter_steps)),
                        grid_dim=(ENV_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    # No MCTS tree during warmup — leave the bootstrap
                    # value at 0. The n-step kernel will treat this as a
                    # zero bootstrap, equivalent to truncated MC return
                    # over the K-step window.
                    gpu.mcts_step_value_buf.enqueue_fill(Scalar[dtype](0.0))

                # ── GPU env step ─────────────────────────────────
                gpu.replay.save_obs(ctx, obs_buf)

                E.step_kernel_gpu[Self.n_envs, GS, OBS](
                    ctx,
                    states_buf,
                    actions_buf,
                    rewards_buf,
                    dones_buf,
                    terminated_buf,
                    obs_buf,
                    legal_masks_buf,
                    rng_seed=UInt64(total_steps + iter_steps),
                )

                # ── Store in GPU replay ──────────────────────────
                # dones_buf (term|trunc) → sequence-rejection in sampling.
                # terminated_buf (term-only) → bootstrap mask in batch_dones
                # so n-step TD targets keep V(s_{t+n}) on truncation.
                gpu.replay.store_with_termination(
                    ctx,
                    actions_buf,
                    rewards_buf,
                    dones_buf,
                    terminated_buf,
                )

                # Compute per-env player-to-move from episode-step counter.
                # ep_steps_buf still holds the step index of the move we
                # just took (incrementing happens after the store).
                # Assumes player 0 starts and players strictly alternate.
                var ep_steps_for_tp = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](ep_steps_buf.unsafe_ptr())
                var step_tp_out = LayoutTensor[
                    DType.uint8, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](gpu.mcts_step_to_play_buf.unsafe_ptr())
                comptime run_to_play = to_play_from_episode_step_kernel[
                    Self.n_envs, dtype
                ]
                ctx.enqueue_function[run_to_play](
                    ep_steps_for_tp,
                    step_tp_out,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Store MCTS targets (policies/values/to-play).
                var mcts_pol_in = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin
                ](gpu.mcts_step_policy_buf.unsafe_ptr())
                var mcts_val_in = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](gpu.mcts_step_value_buf.unsafe_ptr())
                var mcts_tp_in = LayoutTensor[
                    DType.uint8,
                    Layout.row_major(Self.n_envs),
                    MutAnyOrigin,
                ](gpu.mcts_step_to_play_buf.unsafe_ptr())
                var mcts_pol_buf_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * PER_ENV_CAP * ACT),
                    MutAnyOrigin,
                ](gpu.mcts_policy_buf.unsafe_ptr())
                var mcts_val_buf_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * PER_ENV_CAP),
                    MutAnyOrigin,
                ](gpu.mcts_value_buf.unsafe_ptr())
                var mcts_tp_buf_t = LayoutTensor[
                    DType.uint8,
                    Layout.row_major(Self.n_envs * PER_ENV_CAP),
                    MutAnyOrigin,
                ](gpu.mcts_to_play_buf.unsafe_ptr())
                var prev_widx = Scalar[DType.int32](
                    (gpu.replay.write_idx - 1 + PER_ENV_CAP) % PER_ENV_CAP
                )

                @parameter
                @always_inline
                def store_tgt_w(
                    pi: LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin
                    ],
                    vi: LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                    ],
                    ti: LayoutTensor[
                        DType.uint8,
                        Layout.row_major(Self.n_envs),
                        MutAnyOrigin,
                    ],
                    pb: LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs * PER_ENV_CAP * ACT),
                        MutAnyOrigin,
                    ],
                    vb: LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs * PER_ENV_CAP),
                        MutAnyOrigin,
                    ],
                    tb: LayoutTensor[
                        DType.uint8,
                        Layout.row_major(Self.n_envs * PER_ENV_CAP),
                        MutAnyOrigin,
                    ],
                    w: Scalar[DType.int32],
                ):
                    store_mcts_targets_kernel[
                        Self.n_envs, PER_ENV_CAP, ACT, dtype
                    ](pi, vi, ti, pb, vb, tb, w)

                ctx.enqueue_function[store_tgt_w](
                    mcts_pol_in,
                    mcts_val_in,
                    mcts_tp_in,
                    mcts_pol_buf_t,
                    mcts_val_buf_t,
                    mcts_tp_buf_t,
                    prev_widx,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                # ── Episode tracking ─────────────────────────────
                var rewards_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](rewards_buf.unsafe_ptr())
                var dones_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](dones_buf.unsafe_ptr())
                var ep_rew_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](ep_rew_buf.unsafe_ptr())
                var ep_steps_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](ep_steps_buf.unsafe_ptr())
                comptime run_accum = accumulate_rewards_kernel[
                    dtype, Self.n_envs
                ]
                ctx.enqueue_function[run_accum](
                    ep_rew_t,
                    rewards_t,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
                comptime run_incr = increment_steps_kernel[dtype, Self.n_envs]
                ctx.enqueue_function[run_incr](
                    ep_steps_t,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
                var rew_sum_t = LayoutTensor[
                    dtype, Layout.row_major(1), MutAnyOrigin
                ](rew_sum_buf.unsafe_ptr())
                var ep_count_t = LayoutTensor[
                    dtype, Layout.row_major(1), MutAnyOrigin
                ](ep_count_buf.unsafe_ptr())
                comptime run_log = log_and_reset_completed_kernel[
                    dtype, Self.n_envs
                ]
                ctx.enqueue_function[run_log](
                    dones_t,
                    ep_rew_t,
                    ep_steps_t,
                    rew_sum_t,
                    ep_count_t,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                # ── Selective reset + re-extract obs ─────────────
                E.selective_reset_kernel_gpu[Self.n_envs, GS](
                    ctx,
                    states_buf,
                    dones_buf,
                    rng_seed=UInt64(total_steps + iter_steps),
                )
                E.extract_obs_kernel_gpu[Self.n_envs, GS, OBS](
                    ctx,
                    states_buf,
                    obs_buf,
                    legal_masks_buf,
                )

                iter_steps += Self.n_envs
                total_steps += Self.n_envs
                self.total_steps += Self.n_envs

            # ── 2. Print collection stats ────────────────────────
            ctx.enqueue_copy(rew_sum_host, rew_sum_buf)
            ctx.enqueue_copy(ep_count_host, ep_count_buf)
            ctx.synchronize()
            var total_rew = Float64(rew_sum_host[0])
            var total_eps = Int(Float64(ep_count_host[0]))
            var avg_rew = total_rew / Float64(
                total_eps
            ) if total_eps > 0 else Float64(0.0)

            # ── 3. Train on collected data ───────────────────────
            if gpu.replay.is_ready[Self.Config.unroll_steps + 1]():
                # gpu.replay.size is PER-ENV (capped at PER_ENV_CAP).
                # Total buffer across all envs is per-env × N_ENVS;
                # AlphaZero's `num_train_steps = train_epochs * total / BS`
                # was doing N_ENVS× more training than the old per-env
                # formula here. Match AZ's training intensity.
                var total_buf_size = gpu.replay.size * Self.n_envs
                var grad_steps = (
                    train_epochs * total_buf_size // Self.Config.batch_size
                )
                if grad_steps < 1:
                    grad_steps = 1
                # GPU-side reanalyze: when use_reanalyze is True, each
                # update_gpu refreshes batch_values with fresh predictions
                # from the current network before the n-step kernel runs
                # (see muzero.mojo:1820-1900 / "Step 1.5"). This matches
                # muzero-general's use_last_model_value behavior.
                _ = reanalyze_per_iter  # reserved for future async-style reanalyze
                for s_idx in range(grad_steps):
                    if use_one_cycle:
                        # Per-iter 1cycle: ramp up to base LR by 30% of
                        # iter, cosine-anneal to 1% by end. Applied to
                        # all 3 networks so rep/dyn/pred stay in lock
                        # step. Mirrors AlphaZero's pattern at
                        # alphazero.mojo:4404-4408. Mitigates the late-
                        # iter gradient-spike collapse we observed at
                        # iter ~63 (constant LR=1e-3 was too aggressive
                        # near the perfect-play plateau).
                        var sc = OneCycleSchedule[].lr_scale_at(
                            s_idx, grad_steps
                        )
                        gpu.representation.set_lr_scale(sc, ctx)
                        gpu.dynamics.set_lr_scale(sc, ctx)
                        gpu.prediction.set_lr_scale(sc, ctx)
                    _ = self.update_gpu(ctx, gpu, use_reanalyze=use_reanalyze)
                self.train_step_count += grad_steps
                gpu.download_to(self.state, ctx)

            # ── 4. GPU Evaluation vs opponent ────────────────────
            if do_eval and use_mcts:
                var eval_r = self._gpu_eval_muzero[E, GPUEval](ctx, gpu)
                clear_progress_bar()
                print(
                    "  Iter",
                    iter,
                    "| Eval vs",
                    GPUEval.NAME,
                    "| W:",
                    eval_r[0],
                    "D:",
                    eval_r[1],
                    "L:",
                    eval_r[2],
                )
                if Bool(self.logger):
                    # Flat snake_case so each evaluator gets its own
                    # series without nested slashes.
                    self.logger.value()[].log_scalar(
                        "eval_" + GPUEval.NAME + "_wins",
                        Float64(eval_r[0]),
                        iter,
                    )
                    self.logger.value()[].log_scalar(
                        "eval_" + GPUEval.NAME + "_draws",
                        Float64(eval_r[1]),
                        iter,
                    )
                    self.logger.value()[].log_scalar(
                        "eval_" + GPUEval.NAME + "_losses",
                        Float64(eval_r[2]),
                        iter,
                    )

            # ── 4b. Second GPU Evaluation ────────────────────────
            if do_eval2 and use_mcts:
                var eval_r2 = self._gpu_eval_muzero[E, GPUEval2](
                    ctx,
                    gpu,
                )
                print(
                    "    vs",
                    GPUEval2.NAME,
                    ": W",
                    eval_r2[0],
                    "D",
                    eval_r2[1],
                    "L",
                    eval_r2[2],
                )
                if Bool(self.logger):
                    self.logger.value()[].log_scalar(
                        "eval_" + GPUEval2.NAME + "_wins",
                        Float64(eval_r2[0]),
                        iter,
                    )
                    self.logger.value()[].log_scalar(
                        "eval_" + GPUEval2.NAME + "_draws",
                        Float64(eval_r2[1]),
                        iter,
                    )
                    self.logger.value()[].log_scalar(
                        "eval_" + GPUEval2.NAME + "_losses",
                        Float64(eval_r2[2]),
                        iter,
                    )

            # ── 5. Arena comparison (new vs best) ────────────────
            if do_arena and use_mcts:
                var ar = self._arena_compare_muzero[E](
                    ctx,
                    gpu,
                    best_params,
                    threshold=arena_threshold,
                )
                if ar[0]:
                    arena_accepts += 1
                    # Save new best params
                    for i in range(REP_PS):
                        best_params[i] = self.state.representation.params[i]
                    for i in range(DYN_PS):
                        best_params[REP_PS + i] = self.state.dynamics.params[i]
                    for i in range(PRED_PS):
                        best_params[
                            REP_PS + DYN_PS + i
                        ] = self.state.prediction.params[i]
                    # Re-upload accepted params to GPU
                    gpu.upload_from(self.state, ctx)
                else:
                    arena_rejects += 1
                    # Re-upload reverted params to GPU
                    gpu.upload_from(self.state, ctx)
                clear_progress_bar()
                print(
                    "  Arena: new",
                    ar[1],
                    "draw",
                    ar[2],
                    "old",
                    ar[3],
                    "| Accepted" if ar[0] else "| Rejected",
                    "(",
                    arena_accepts,
                    "/",
                    arena_accepts + arena_rejects,
                    ")",
                )
                if Bool(self.logger):
                    self.logger.value()[].log_scalar(
                        "arena_new_wins", Float64(ar[1]), iter
                    )
                    self.logger.value()[].log_scalar(
                        "arena_draws", Float64(ar[2]), iter
                    )
                    self.logger.value()[].log_scalar(
                        "arena_old_wins", Float64(ar[3]), iter
                    )
                    self.logger.value()[].log_scalar(
                        "arena_accepted", 1.0 if ar[0] else 0.0, iter
                    )
                    self.logger.value()[].log_scalar(
                        "arena_accept_rate",
                        Float64(arena_accepts)
                        / Float64(arena_accepts + arena_rejects),
                        iter,
                    )

            # ── 6. Print iteration summary ───────────────────────
            # Param-norm + replay-size diagnostic. Verifies training
            # is actually moving the weights and the buffer is filling
            # at the expected rate.
            var iter_norms = self._net_param_l2(ctx, gpu)
            var d_rep = iter_norms[0] - last_rep_n
            var d_dyn = iter_norms[1] - last_dyn_n
            var d_pred = iter_norms[2] - last_pred_n
            last_rep_n = iter_norms[0]
            last_dyn_n = iter_norms[1]
            last_pred_n = iter_norms[2]
            var replay_total = gpu.replay.size * Self.n_envs

            clear_progress_bar()
            print(
                "Iter",
                iter + 1,
                "/",
                num_iters,
                "| Steps:",
                total_steps,
                "| Games:",
                total_eps,
                "| Avg Rew:",
                Int(avg_rew * 100) if total_eps > 0 else 0,
                "% | Train:",
                self.train_step_count,
                "| Replay:",
                replay_total,
                "| |W| rep/dyn/pred:",
                last_rep_n,
                "/",
                last_dyn_n,
                "/",
                last_pred_n,
                "(Δ",
                d_rep,
                "/",
                d_dyn,
                "/",
                d_pred,
                ")",
            )
            if total_eps > 0:
                metrics.log_episode[dtype](
                    total_eps, Scalar[dtype](avg_rew), 0, 0.0
                )

            # Log iter summary metrics. Param norms are the key signal for
            # spotting weight runaway (we saw NVIDIA explode pre-MinMaxNorm
            # with pred_norm growing 25 → 297 over 6 iters before NaN).
            if Bool(self.logger):
                # KNOWN_GROUPS alignment for iter-summary metrics:
                # - param_norm (Parameter Norm) = sum of rep+dyn+pred
                #   submodel norms. Per-subnet breakdown is kept as flat
                #   metrics so the rep/dyn/pred runaway can still be
                #   diagnosed (e.g. the historical NVIDIA pred_norm bug).
                # - avg_reward (Episode Reward) for 2-player games is
                #   {-1, 0, +1}-valued; trends toward +1 vs the eval pool.
                # - episodes / train_steps (Training Progress) — `games`
                #   == episodes here.
                # - buffer_size (Runtime Throughput) for the replay.
                self.logger.value()[].log_scalar(
                    "param_norm", last_rep_n + last_dyn_n + last_pred_n, iter
                )
                self.logger.value()[].log_scalar(
                    "param_norm_rep", last_rep_n, iter
                )
                self.logger.value()[].log_scalar(
                    "param_norm_dyn", last_dyn_n, iter
                )
                self.logger.value()[].log_scalar(
                    "param_norm_pred", last_pred_n, iter
                )
                self.logger.value()[].log_scalar(
                    "param_norm_d_rep", d_rep, iter
                )
                self.logger.value()[].log_scalar(
                    "param_norm_d_dyn", d_dyn, iter
                )
                self.logger.value()[].log_scalar(
                    "param_norm_d_pred", d_pred, iter
                )
                self.logger.value()[].log_scalar(
                    "avg_reward", avg_rew, iter
                )
                self.logger.value()[].log_scalar(
                    "episodes", Float64(total_eps), iter
                )
                self.logger.value()[].log_scalar(
                    "env_steps", Float64(total_steps), iter
                )
                self.logger.value()[].log_scalar(
                    "buffer_size", Float64(replay_total), iter
                )
                self.logger.value()[].log_scalar(
                    "train_steps",
                    Float64(self.train_step_count),
                    iter,
                )
                self.logger.value()[].flush()

            # ── 7. Checkpoint ────────────────────────────────────
            if checkpoint_every > 0 and (iter + 1) % checkpoint_every == 0:
                self.save_checkpoint(checkpoint_path)
                print("  Saved checkpoint:", checkpoint_path)

        best_params.free()
        gpu.download_to(self.state, ctx)
        return metrics

    # ══════════════════════════════════════════════════════════════════════
    # GPU Evaluation (MuZero)
    # ══════════════════════════════════════════════════════════════════════

    def _gpu_eval_muzero[
        E: GPUTwoPlayerDiscreteEnv,
        Eval: GPUEvaluator,
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu: MuZeroGPUState[Self.Config, Self.n_envs, 1000],
        rng_offset: Int = 0,
    ) raises -> Tuple[Int, Int, Int]:
        """GPU evaluation: agent (MCTS temp=0) vs GPU evaluator.

        Agent plays as P0 (even moves). Opponent plays as P1 (odd moves).
        Uses learned dynamics for MCTS expansion.

        Returns (wins, draws, losses).
        """
        comptime ACT = Self.Config.action_dim
        comptime OBS = Self.Config.obs_dim
        comptime LATENT = Self.Config.latent_dim
        comptime BINS = Self.Config.num_bins
        comptime MAX_NODES = Self.Config.max_nodes
        comptime PRED_OUT = Self.Config.PRED_OUT
        comptime DYN_IN = Self.Config.DYN_IN
        comptime DYN_OUT = Self.Config.DYN_OUT
        comptime GS = E.STATE_SIZE
        comptime ENV_BLOCKS = (Self.n_envs + TPB - 1) // TPB
        comptime NUM_SIMS_EV = Self.Config.num_simulations
        # Same ACT-capped / divisibility-aware sizing as the training
        # path so eval uses sequential MCTS on small-action board games.
        comptime _REQ_BSIMS_EV = Self.Config.batch_sims
        comptime _CAP_BSIMS_EV = (
            _REQ_BSIMS_EV if _REQ_BSIMS_EV < ACT else ACT
        )
        comptime ORCH_BSIMS_EV = (
            _CAP_BSIMS_EV if NUM_SIMS_EV % _CAP_BSIMS_EV == 0
            else (
                4 if (NUM_SIMS_EV % 4 == 0 and _CAP_BSIMS_EV >= 4)
                else (
                    2 if (NUM_SIMS_EV % 2 == 0 and _CAP_BSIMS_EV >= 2)
                    else 1
                )
            )
        )
        comptime TOTAL_EXPAND = Self.n_envs * ORCH_BSIMS_EV

        comptime RepNet = Network[Self.Config.RepModel, Self.Config.OptType]
        comptime DynNet = Network[Self.Config.DynModel, Self.Config.OptType]
        comptime PredNet = Network[Self.Config.PredModel, Self.Config.OptType]
        comptime REP_IN_DIM = Self.Config.RepModel.IN_DIM
        comptime REP_OUT_DIM = Self.Config.RepModel.OUT_DIM
        comptime DYN_IN_DIM = Self.Config.DynModel.IN_DIM
        comptime DYN_OUT_DIM = Self.Config.DynModel.OUT_DIM
        comptime PRED_IN_DIM = Self.Config.PredModel.IN_DIM
        comptime PRED_OUT_DIM = Self.Config.PredModel.OUT_DIM

        comptime WS_R = Self.Config.RepModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_D = Self.Config.DynModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_P = Self.Config.PredModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime MAX_WS_1 = WS_R if WS_R > WS_D else WS_D
        comptime MAX_WS_2 = MAX_WS_1 if MAX_WS_1 > WS_P else WS_P
        comptime MCTS_WS = TOTAL_EXPAND * MAX_WS_2 if MAX_WS_2 > 0 else 1
        var mcts_ws = ctx.enqueue_create_buffer[dtype](MCTS_WS)

        # Env buffers
        var eval_states = ctx.enqueue_create_buffer[dtype](Self.n_envs * GS)
        var eval_obs = ctx.enqueue_create_buffer[dtype](Self.n_envs * OBS)
        var eval_acts = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var eval_rews = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var eval_dones = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var eval_term = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var eval_legal = ctx.enqueue_create_buffer[dtype](Self.n_envs * ACT)
        var eval_dones_host = ctx.enqueue_create_host_buffer[dtype](Self.n_envs)
        var eval_rews_host = ctx.enqueue_create_host_buffer[dtype](Self.n_envs)

        # MCTS orchestrator for evaluation — deterministic (no Dirichlet)
        # since the agent plays its best move in eval. PlayerMode.SelfPlay
        # is baked in via ``Self.Config.Players`` so the expand+backup
        # kernel uses negated backup.
        var eval_planner = GenericGPUMCTS[
            Self.n_envs, ACT, LATENT, BINS, MAX_NODES, NUM_SIMS_EV,
            ORCH_BSIMS_EV,
            Self.Config.PUCT,
            NoNoise,
            Self.Config.Players,
        ](ctx, gamma=self.gamma, v_min=self.v_min, v_max=self.v_max)
        # Gumbel-MuZero eval planner. ``gumbel_scale=0.0`` makes the root
        # action selection deterministic (mctx convention for eval). The
        # ``Self.Config.Players`` comptime param threads SelfPlay
        # negation into the backup kernel.
        var eval_gumbel_planner = GumbelGPUMCTS[
            Self.n_envs, ACT, LATENT, BINS, MAX_NODES,
            Self.Config.PolicyMode.MAX_K, NUM_SIMS_EV,
            Self.Config.Players,
        ](
            ctx, gamma=self.gamma, v_min=self.v_min, v_max=self.v_max,
            gumbel_scale=0.0,
        )

        E.reset_kernel_gpu[Self.n_envs, GS](
            ctx, eval_states, rng_seed=UInt64(rng_offset + 9999)
        )
        E.extract_obs_kernel_gpu[Self.n_envs, GS, OBS](
            ctx, eval_states, eval_obs, eval_legal
        )
        ctx.synchronize()

        var eval_done = InlineArray[Bool, 64](fill=False)
        var eval_result = InlineArray[Int, 64](fill=0)
        var eval_move = 0
        var eval_all_done = False
        comptime MAX_EVAL_MOVES = ACT * ACT

        while not eval_all_done and eval_move < MAX_EVAL_MOVES:
            var agent_turn = eval_move % 2 == 0

            if agent_turn:
                # Agent: GPU MCTS with learned dynamics (temp=0, no noise)
                # Drive the shared ``GenericGPUMCTS`` orchestrator via
                # ``search_gpu_selfplay``. NoNoise is baked into the
                # planner type so the root prior gets pure-network priors
                # (eval is deterministic). Action extraction uses temp=0
                # → greedy argmax over legal actions.
                var rep_a_ev = MuZeroRepGPU[
                    OBS, LATENT, Self.Config.RepModel, Self.Config.OptType
                ](
                    params=gpu.representation.params_buf.unsafe_ptr(),
                    model_state=gpu.representation.model_state_buf.unsafe_ptr(),
                    workspace=mcts_ws,
                )
                var dyn_a_ev = MuZeroDynGPU[
                    ACT, LATENT, BINS, Self.Config.DynModel, Self.Config.OptType
                ](
                    params=gpu.dynamics.params_buf.unsafe_ptr(),
                    model_state=gpu.dynamics.model_state_buf.unsafe_ptr(),
                    workspace=mcts_ws,
                )
                var pred_a_ev = MuZeroPredGPU[
                    ACT, LATENT, BINS, Self.Config.PredModel, Self.Config.OptType
                ](
                    params=gpu.prediction.params_buf.unsafe_ptr(),
                    model_state=gpu.prediction.model_state_buf.unsafe_ptr(),
                    workspace=mcts_ws,
                )

                var obs_t_ev = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, OBS),
                    MutAnyOrigin,
                ](eval_obs.unsafe_ptr())
                var lm_t_ev = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin
                ](eval_legal.unsafe_ptr())

                comptime if Self.Config.PolicyMode.IS_GUMBEL:
                    # Gumbel-MuZero eval: ``gumbel_scale=0`` baked into
                    # ``eval_gumbel_planner`` ctor → no Gumbel perturbation
                    # → deterministic root choice. Legal mask is read
                    # from ``state.legal_mask``; copy from agent's buffer.
                    ctx.enqueue_copy(
                        eval_gumbel_planner.state.legal_mask, eval_legal
                    )
                    eval_gumbel_planner.search_gpu_selfplay[
                        MuZeroRepGPU[
                            OBS, LATENT, Self.Config.RepModel,
                            Self.Config.OptType,
                        ],
                        MuZeroDynGPU[
                            ACT, LATENT, BINS, Self.Config.DynModel,
                            Self.Config.OptType,
                        ],
                        MuZeroPredGPU[
                            ACT, LATENT, BINS, Self.Config.PredModel,
                            Self.Config.OptType,
                        ],
                    ](
                        ctx,
                        rep_a_ev,
                        dyn_a_ev,
                        pred_a_ev,
                        obs_t_ev,
                        rng_seed=UInt32(rng_offset + eval_move),
                    )
                    eval_gumbel_planner.extract_actions_argmax(ctx)
                    ctx.enqueue_copy(
                        eval_acts, eval_gumbel_planner.actions_out
                    )
                else:
                    # Vanilla PUCT eval (existing path).
                    eval_planner.search_gpu_selfplay[
                        MuZeroRepGPU[
                            OBS, LATENT, Self.Config.RepModel,
                            Self.Config.OptType,
                        ],
                        MuZeroDynGPU[
                            ACT, LATENT, BINS, Self.Config.DynModel,
                            Self.Config.OptType,
                        ],
                        MuZeroPredGPU[
                            ACT, LATENT, BINS, Self.Config.PredModel,
                            Self.Config.OptType,
                        ],
                    ](
                        ctx,
                        rep_a_ev,
                        dyn_a_ev,
                        pred_a_ev,
                        obs_t_ev,
                        lm_t_ev,
                        rng_seed=UInt32(rng_offset + eval_move),
                    )

                    # Greedy action — zero ep_steps buffer + temp=0 so
                    # the kernel always picks argmax over legal visits.
                    var zero_ep_steps = ctx.enqueue_create_buffer[dtype](
                        Self.n_envs
                    )
                    zero_ep_steps.enqueue_fill(Scalar[dtype](0.0))
                    var ep_t_ev = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                    ](zero_ep_steps.unsafe_ptr())
                    eval_planner.extract_actions_temp[TEMP_THRESHOLD=0](
                        ctx,
                        ep_t_ev,
                        lm_t_ev,
                        rng_seed=UInt32(rng_offset + eval_move),
                        temp_min=Float64(0.0),
                    )

                    ctx.enqueue_copy(eval_acts, eval_planner.actions_out)
            else:
                # Opponent's turn
                Eval.select_action_gpu[Self.n_envs, ACT, GS](
                    ctx,
                    eval_acts,
                    eval_legal,
                    eval_states,
                    rng_seed=UInt64(rng_offset + eval_move),
                )

            # Env step
            E.step_kernel_gpu[Self.n_envs, GS, OBS](
                ctx,
                eval_states,
                eval_acts,
                eval_rews,
                eval_dones,
                eval_term,
                eval_obs,
                eval_legal,
                rng_seed=UInt64(rng_offset + eval_move + 5000),
            )

            # Check completions
            ctx.enqueue_copy(eval_dones_host, eval_dones)
            ctx.enqueue_copy(eval_rews_host, eval_rews)
            ctx.synchronize()

            eval_all_done = True
            for e in range(Self.n_envs):
                if not eval_done[e] and Float64(eval_dones_host[e]) > 0.5:
                    eval_done[e] = True
                    var rew = Float64(eval_rews_host[e])
                    if rew > 0.5:
                        eval_result[e] = 1 if agent_turn else 2
                    elif rew < -0.5:
                        eval_result[e] = 2 if agent_turn else 1
                    else:
                        eval_result[e] = 3
                if not eval_done[e]:
                    eval_all_done = False

            E.selective_reset_kernel_gpu[Self.n_envs, GS](
                ctx,
                eval_states,
                eval_dones,
                rng_seed=UInt64(rng_offset + eval_move + 7000),
            )
            E.extract_obs_kernel_gpu[Self.n_envs, GS, OBS](
                ctx, eval_states, eval_obs, eval_legal
            )
            eval_move += 1

        var eval_wins = 0
        var eval_draws = 0
        var eval_losses = 0
        for e in range(Self.n_envs):
            if eval_result[e] == 1:
                eval_wins += 1
            elif eval_result[e] == 3:
                eval_draws += 1
            else:
                eval_losses += 1
        return (eval_wins, eval_draws, eval_losses)

    # ══════════════════════════════════════════════════════════════════════
    # Arena Comparison (MuZero)
    # ══════════════════════════════════════════════════════════════════════

    def _arena_compare_muzero[
        E: GPUTwoPlayerDiscreteEnv,
        origin: MutOrigin,
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu: MuZeroGPUState[Self.Config, Self.n_envs, 1000],
        prev_params: UnsafePointer[Scalar[dtype], origin],
        threshold: Float64 = 0.55,
    ) raises -> Tuple[Bool, Int, Int, Int]:
        """Compare current model vs previous best model.

        Plays n_envs games with each side. Returns (accepted, new_wins, draws, old_wins).
        """
        comptime REP_PS = Self.Config.RepModel.PARAM_SIZE
        comptime DYN_PS = Self.Config.DynModel.PARAM_SIZE
        comptime PRED_PS = Self.Config.PredModel.PARAM_SIZE

        # Save current (new) params
        var new_rep = alloc[Scalar[dtype]](REP_PS)
        var new_dyn = alloc[Scalar[dtype]](DYN_PS)
        var new_pred = alloc[Scalar[dtype]](PRED_PS)
        for i in range(REP_PS):
            new_rep[i] = self.state.representation.params[i]
        for i in range(DYN_PS):
            new_dyn[i] = self.state.dynamics.params[i]
        for i in range(PRED_PS):
            new_pred[i] = self.state.prediction.params[i]

        # Phase 1: new model plays (current params already loaded)
        gpu.upload_from(self.state, ctx)
        var r1 = self._gpu_eval_muzero[E, RandomOpponent](
            ctx, gpu, rng_offset=42
        )

        # Phase 2: old model plays
        for i in range(REP_PS):
            self.state.representation.params[i] = prev_params[i]
        for i in range(DYN_PS):
            self.state.dynamics.params[i] = prev_params[REP_PS + i]
        for i in range(PRED_PS):
            self.state.prediction.params[i] = prev_params[REP_PS + DYN_PS + i]
        gpu.upload_from(self.state, ctx)
        var r2 = self._gpu_eval_muzero[E, RandomOpponent](
            ctx, gpu, rng_offset=9999
        )

        # Restore new params
        for i in range(REP_PS):
            self.state.representation.params[i] = new_rep[i]
        for i in range(DYN_PS):
            self.state.dynamics.params[i] = new_dyn[i]
        for i in range(PRED_PS):
            self.state.prediction.params[i] = new_pred[i]

        new_rep.free()
        new_dyn.free()
        new_pred.free()

        # Compare: new wins = new_as_P0 wins + old_as_P0 losses
        var new_wins = r1[0] + r2[2]
        var draws = r1[1] + r2[1]
        var old_wins = r1[2] + r2[0]

        # Elo-style score with draws counting as half-wins. Earlier formula
        # `wins / (wins + losses)` excluded draws and so two strong models
        # (e.g., a perfect-play TicTacToe candidate vs the prior best) could
        # not surpass threshold even when they were strictly equivalent —
        # most games drew. Score = (W + 0.5·D) / (W + D + L) lets a model
        # that ties on score with the prior best (50%) cross any threshold
        # ≤ 0.5 and lets surpluses of half-points from extra wins also count.
        var total = new_wins + draws + old_wins
        var accepted: Bool
        if total == 0:
            accepted = False
        else:
            var score = (Float64(new_wins) + 0.5 * Float64(draws)) / Float64(
                total
            )
            accepted = score >= threshold

        if not accepted:
            # Revert to previous best params
            for i in range(REP_PS):
                self.state.representation.params[i] = prev_params[i]
            for i in range(DYN_PS):
                self.state.dynamics.params[i] = prev_params[REP_PS + i]
            for i in range(PRED_PS):
                self.state.prediction.params[i] = prev_params[
                    REP_PS + DYN_PS + i
                ]

        return (accepted, new_wins, draws, old_wins)


def _to_dtype_list[D: DType](obs: List[Scalar[D]]) -> List[Scalar[dtype]]:
    """Convert observation list to dtype (float32) list.

    Args:
        obs: Observation list from environment.

    Returns:
        List of Scalar[dtype] values.
    """
    var result = List[Scalar[dtype]](capacity=len(obs))
    for i in range(len(obs)):
        result.append(Scalar[dtype](obs[i]))
    return result^


# Migration: Old MuZeroAgent[obs_dim=4, action_dim=2, ...] becomes:
#   GenericMuZeroAgent[MuZeroMLPConfig[4, 2, ...]]()
