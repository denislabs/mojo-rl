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
from .mcts import MCTS, MCTSNode
from .gpu_mcts import (
    GPUMCTSState,
    gpu_mcts_init_root_kernel,
    gpu_mcts_select_kernel,
    gpu_mcts_build_dyn_input_kernel,
    gpu_mcts_expand_kernel,
    gpu_mcts_backup_kernel,
    gpu_mcts_backup_negated_kernel,
    gpu_mcts_extract_actions_kernel,
    gpu_mcts_extract_root_value_kernel,
    gpu_mcts_extract_actions_masked_kernel,
    gpu_mcts_extract_actions_temp_kernel,
    gpu_mcts_apply_legal_mask_kernel,
    gpu_mcts_copy_parent_state_kernel,
    gpu_mcts_store_child_state_kernel,
    gpu_mcts_copy_root_state_kernel,
    gpu_mcts_batched_select_and_build_dyn_kernel,
    gpu_mcts_batched_expand_backup_muzero_kernel,
    gpu_mcts_batched_select_and_copy_kernel,
    gpu_mcts_batched_expand_backup_kernel,
    TPB,
    MAX_DEPTH,
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
    nstep_value_targets_kernel,
    scalar_transform_kernel,
    to_play_from_episode_step_kernel,
    decode_value_dist_kernel,
    action_histogram_kernel,
    action_switch_kernel,
)


# =============================================================================
# MuZero Agent
# =============================================================================


struct GenericMuZeroAgent[Config: MuZeroConfig, n_envs: Int = 64](Movable):
    """MuZero agent for discrete action environments.

    Combines learned representation/dynamics/prediction networks with
    MCTS planning for action selection. Trains via K-step unrolled
    cross-entropy losses on policy, value, and reward targets.

    Parameters:
        Config: MuZeroConfig trait providing all dimensions, network types,
                and training hyperparameters.
        n_envs: Number of parallel GPU environments (default: 64).
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

    # MCTS search engine
    var mcts: MCTS[
        Self.Config.action_dim,
        Self.Config.latent_dim,
        Self.Config.num_bins,
        Self.Config.num_simulations,
    ]

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

    # Step counters
    var total_steps: Int
    var train_step_count: Int

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
        self.mcts = MCTS[
            Self.Config.action_dim,
            Self.Config.latent_dim,
            Self.Config.num_bins,
            Self.Config.num_simulations,
        ](gamma=gamma)
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.v_min = v_min
        self.v_max = v_max
        self.temperature = temperature
        self.temperature_decay_steps = temperature_decay_steps
        self.max_grad_norm = max_grad_norm
        self.target_tau = target_tau
        self.total_steps = 0
        self.train_step_count = 0
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
        self.mcts = take.mcts^
        self.gamma = take.gamma
        self.weight_decay = take.weight_decay
        self.v_min = take.v_min
        self.v_max = take.v_max
        self.temperature = take.temperature
        self.temperature_decay_steps = take.temperature_decay_steps
        self.max_grad_norm = take.max_grad_norm
        self.target_tau = take.target_tau
        self.total_steps = take.total_steps
        self.train_step_count = take.train_step_count
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
        ctx.enqueue_copy(
            gpu.prediction.params_host, gpu.prediction.params_buf
        )
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
            ctx.enqueue_copy(
                gpu.representation.grads_buf, gpu.rep_grads_host
            )
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

        Stores obs/action/reward/done to the SequenceReplayBuffer and
        MCTS policies/values to the parallel target arrays.
        """
        var ep_len = len(self._episode_obs)

        for t in range(ep_len):
            var obs_arr = InlineArray[
                Scalar[DType.float32], Self.Config.obs_dim
            ](uninitialized=True)
            for i in range(Self.Config.obs_dim):
                if i < len(self._episode_obs[t]):
                    obs_arr[i] = Scalar[DType.float32](self._episode_obs[t][i])
                else:
                    obs_arr[i] = Scalar[DType.float32](0.0)

            # Action as one-hot for SequenceReplayBuffer (uses ACTION_DIM float array)
            var act_arr = InlineArray[
                Scalar[DType.float32], Self.Config.action_dim
            ](uninitialized=True)
            for i in range(Self.Config.action_dim):
                act_arr[i] = Scalar[DType.float32](0.0)
            act_arr[self._episode_actions[t]] = Scalar[DType.float32](1.0)

            var is_done = t == ep_len - 1

            self.state.buffer.add(
                obs_arr,
                act_arr,
                Scalar[DType.float32](self._episode_rewards[t]),
                is_done,
            )

            # Store MCTS targets at the buffer write position
            var buf_idx = (
                self.state.buffer.ptr - 1 + Self.StateType._CAP
            ) % Self.StateType._CAP
            for a in range(Self.Config.action_dim):
                self.state.mcts_policies[
                    buf_idx * Self.Config.action_dim + a
                ] = Scalar[dtype](self._episode_policies[t][a])
            self.state.mcts_values[buf_idx] = Scalar[dtype](
                self._episode_values[t]
            )
            self.state.mcts_to_play[buf_idx] = Scalar[DType.uint8](
                self._episode_to_play[t]
            )

        self.reset_episode()

    # ══════════════════════════════════════════════════════════════════════
    # Action Selection
    # ══════════════════════════════════════════════════════════════════════

    def select_action(
        mut self,
        obs: List[Scalar[dtype]],
        training: Bool = True,
        legal_mask: List[Bool] = List[Bool](),
    ) -> Tuple[Int, InlineArray[Float64, Self.Config.action_dim], Float64]:
        """Select an action using batched MCTS with the learned model.

        Uses batched MCTS (virtual loss + batched network forward) for
        ~8x speedup over sequential simulations.

        Args:
            obs: Current observation [obs_dim].
            training: If True, sample with temperature; if False, argmax.
            legal_mask: Optional list of bools, length action_dim. When
                provided, illegal actions are zeroed out of the root prior
                and the action sampler. Default: empty (all actions
                treated as legal — correct for envs without illegal moves).

        Returns:
            Tuple of (action_index, mcts_policy, root_value).
        """
        # Run batched MCTS search (8 simulations per batch)
        var policy = self.mcts.search_batched[
            Self.StateType.RepModel,
            Self.StateType.DynModel,
            Self.StateType.PredModel,
            Self.StateType.OptType,
            Self.StateType.OptType,
            Self.StateType.OptType,
            8,  # BATCH_SIMS
        ](
            obs,
            self.state.representation,
            self.state.dynamics,
            self.state.prediction,
            self.v_min,
            self.v_max,
            add_noise=training,
            legal_mask=legal_mask,
        )

        # Get root value from MCTS (value at root after search)
        var root_value = Float64(0.0)
        if len(self.mcts.nodes) > 0:
            var root = self.mcts.nodes[0]
            for a in range(Self.Config.action_dim):
                if root.visit_count[a] > 0:
                    root_value += policy[a] * root.mean_value(a)

        # Sample action with temperature
        var action: Int
        if not training or self.temperature < 0.01:
            # Argmax
            action = 0
            var best_prob = policy[0]
            for a in range(1, Self.Config.action_dim):
                if policy[a] > best_prob:
                    best_prob = policy[a]
                    action = a
        else:
            # Temperature sampling: pi(a) = N(a)^(1/T) / sum_b N(b)^(1/T)
            var temp_policy = InlineArray[Float64, Self.Config.action_dim](
                uninitialized=True
            )
            var inv_temp = 1.0 / self.temperature
            var sum_p = Float64(0.0)
            for a in range(Self.Config.action_dim):
                # Use visit counts raised to 1/T
                var count = Float64(0.0)
                if len(self.mcts.nodes) > 0:
                    count = Float64(self.mcts.nodes[0].visit_count[a])
                if count > 0.0:
                    # exp((1/T) * ln(count)) for numerical stability
                    temp_policy[a] = exp(inv_temp * log(count))
                else:
                    temp_policy[a] = Float64(0.0)
                sum_p += temp_policy[a]

            if sum_p > 0.0:
                for a in range(Self.Config.action_dim):
                    temp_policy[a] /= sum_p
            else:
                for a in range(Self.Config.action_dim):
                    temp_policy[a] = 1.0 / Float64(Self.Config.action_dim)

            # Multinomial sample
            var u = random_float64(0.0, 1.0)
            var cumsum = Float64(0.0)
            action = Self.Config.action_dim - 1
            for a in range(Self.Config.action_dim):
                cumsum += temp_policy[a]
                if u <= cumsum:
                    action = a
                    break

        return (action, policy, root_value)

    # ══════════════════════════════════════════════════════════════════════
    # MuZero Reanalyze
    # ══════════════════════════════════════════════════════════════════════

    def reanalyze(mut self, num_positions: Int = 32):
        """Re-run MCTS on old observations using the latest networks.

        Updates the stored MCTS policy and value targets in the replay buffer
        with fresh predictions from the current networks. This dramatically
        improves sample efficiency by preventing stale targets from degrading
        the training signal.

        Reference: Schrittwieser et al., 2021 — Online and Offline RL by
        Planning with a Learned World Model

        Args:
            num_positions: Number of random positions to reanalyze (default: 32).
        """
        comptime CAPACITY = Self.StateType._CAP
        comptime ACT = Self.Config.action_dim

        if self.state.buffer.len() < 10:
            return

        for _ in range(num_positions):
            # Sample a random position from the buffer
            var pos = (
                Int(random_float64() * Float64(self.state.buffer.size))
                % self.state.buffer.size
            )
            var actual_idx = (
                self.state.buffer.ptr - self.state.buffer.size + pos
            ) % CAPACITY
            if actual_idx < 0:
                actual_idx += CAPACITY

            # Extract observation at this position
            var obs = List[Scalar[dtype]](capacity=Self.Config.obs_dim)
            for i in range(Self.Config.obs_dim):
                obs.append(
                    Scalar[dtype](
                        self.state.buffer.obs[
                            actual_idx * Self.Config.obs_dim + i
                        ]
                    )
                )

            # Re-run batched MCTS with current networks
            var policy = self.mcts.search_batched[
                Self.StateType.RepModel,
                Self.StateType.DynModel,
                Self.StateType.PredModel,
                Self.StateType.OptType,
                Self.StateType.OptType,
                Self.StateType.OptType,
                8,  # BATCH_SIMS
            ](
                obs,
                self.state.representation,
                self.state.dynamics,
                self.state.prediction,
                self.v_min,
                self.v_max,
                add_noise=False,  # No exploration noise for reanalysis
            )

            # Get updated root value
            var root_value = Float64(0.0)
            if len(self.mcts.nodes) > 0:
                var root = self.mcts.nodes[0]
                for a in range(ACT):
                    if root.visit_count[a] > 0:
                        root_value += policy[a] * root.mean_value(a)

            # Update stored targets
            for a in range(ACT):
                self.state.mcts_policies[actual_idx * ACT + a] = Scalar[dtype](
                    policy[a]
                )
            self.state.mcts_values[actual_idx] = Scalar[dtype](root_value)

    # ══════════════════════════════════════════════════════════════════════
    # Training (K-Step Unrolled Forward/Backward)
    # ══════════════════════════════════════════════════════════════════════

    def update(mut self, use_reanalyze: Bool = False) -> Float64:
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
            obs_t, h0_t, self.state.representation.params_view(), self.state.representation.model_state_view(), rep_cache_t
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
                hk_t, pred_t, self.state.prediction.params_view(), self.state.prediction.model_state_view(), pred_cache_t
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
            obs_t, pred_t, self.state.prediction.params_view(), self.state.prediction.model_state_view()
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
    ) -> Tuple[Int, Int, Int]:
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
    ) -> TrainingMetrics:
        """Train MuZero on a discrete action environment.

        Alternates between self-play (with MCTS) and training from replay.
        Optionally uses Reanalyze to refresh old MCTS targets with the
        latest networks for improved sample efficiency.

        Args:
            env: Environment implementing BoxDiscreteActionEnv.
            total_timesteps: Total environment steps (default: 500K).
            train_every: Steps between training updates (default: 1).
            seed_episodes: Random exploration episodes (default: 10).
            print_every: Episodes between progress prints (default: 10).
            use_reanalyze: Enable MuZero Reanalyze (default: False).
            warmup_steps: Random exploration steps before training (default: 1000).

        Returns:
            TrainingMetrics with episode rewards and statistics.
        """
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
                bo, ba, br, bd, bd_term, bp, bv, btp,
                oo, oa, orw, od, op, ov, otp,
                bsz, bwi, seed,
            )

        ctx.enqueue_function[sample_wrapper, sample_wrapper](
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
                ctx.enqueue_function[run_ra_scale, run_ra_scale](
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
                ctx.enqueue_function[run_ra_dec, run_ra_dec](
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
        ctx.enqueue_function[nstep_wrapper, nstep_wrapper](
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
        ctx.enqueue_function[run_st_val, run_st_val](
            val_tgt_t,
            Scalar[dtype](0.001),
            grid_dim=(VAL_TGT_BLOCKS,),
            block_dim=(TPB,),
        )

        # Scalar transform reward targets
        comptime REW_TGT_SIZE = K * BATCH
        comptime REW_TGT_BLOCKS = (REW_TGT_SIZE + TPB - 1) // TPB
        comptime run_st_rew = scalar_transform_kernel[REW_TGT_SIZE, dtype]
        ctx.enqueue_function[run_st_rew, run_st_rew](
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
        ctx.enqueue_function[run_scale, run_scale](
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
                ctx.enqueue_function[run_build, run_build](
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
                ctx.enqueue_function[run_extract, run_extract](
                    next_hidden,
                    dyn_out_1d,
                    grid_dim=(EXTRACT_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Scale next hidden state
                var next_h_scale = LayoutTensor[
                    dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
                ](gpu.hidden_buf.unsafe_ptr() + (k + 1) * BATCH * LATENT)
                ctx.enqueue_function[run_scale, run_scale](
                    next_h_scale,
                    grid_dim=(BATCH_BLOCKS,),
                    block_dim=(TPB,),
                )

        # ── Step 4: GPU backward pass ────────────────────────────────
        var inv_k_s = Scalar[dtype](1.0 / Float64(K + 1) / Float64(BATCH))

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
            ctx.enqueue_function[run_pol_grad, run_pol_grad](
                grad_pred_1d,
                pred_1d,
                policy_targets_k,
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
            ctx.enqueue_function[run_twohot, run_twohot](
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
            ctx.enqueue_function[run_val_grad, run_val_grad](
                grad_pred_1d,
                pred_1d,
                val_dist,
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
            ctx.enqueue_function[run_add, run_add](
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
                ctx.enqueue_function[run_set_hgrad, run_set_hgrad](
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
                ctx.enqueue_function[run_twohot, run_twohot](
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
                ctx.enqueue_function[run_build2, run_build2](
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
                ctx.enqueue_function[run_rew_grad, run_rew_grad](
                    grad_dyn_1d,
                    dyn_out_1d_bwd,
                    rew_dist,
                    inv_k_s,
                    grid_dim=(BATCH_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Scale dynamics gradients by 1/K
                comptime DYN_OUT_EL = BATCH * DYN_OUT
                comptime DYN_OUT_BLOCKS = (DYN_OUT_EL + TPB - 1) // TPB
                comptime run_dyn_scale = scale_kernel[DYN_OUT_EL, dtype]
                ctx.enqueue_function[run_dyn_scale, run_dyn_scale](
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
                ctx.enqueue_function[run_extract_hgrad, run_extract_hgrad](
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

        self.train_step_count += 1
        return Float64(0.0)  # Loss computation on GPU would require readback

    def diagnose_init_state(
        mut self,
        ctx: DeviceContext,
        obs_values: List[Float64],
        label: String = "",
    ) raises:
        """Run rep+pred forward and one MCTS pass on a canonical obs, dump.

        Used to localize the failure mode for untrained MuZero on CartPole:
          [1] post-rep + post-min-max hidden state for env 0
          [2] pred policy logits + softmax — is the prior already biased?
          [3] pred value distribution → decoded raw scalar
          [4] root prior after init_root_kernel (= softmax(policy) for env 0)
          [5] root visit_count and total_value after NUM_SIMS — does MCTS
              differentiate actions, or commit to a single one?

        Broadcasts `obs_values` to all Self.n_envs envs (so all envs see the
        same obs) and reports env 0's results. Pure forward + MCTS only —
        no training, no env stepping, no replay. Network init is whatever
        the agent's __init__ produced (Xavier or whichever default).
        """
        comptime ACT = Self.Config.action_dim
        comptime OBS = Self.Config.obs_dim
        comptime LATENT = Self.Config.latent_dim
        comptime BINS = Self.Config.num_bins
        comptime PRED_OUT = Self.Config.PRED_OUT
        comptime DYN_IN = Self.Config.DYN_IN
        comptime DYN_OUT = Self.Config.DYN_OUT
        comptime NUM_SIMS = Self.Config.num_simulations
        comptime REP_IN_DIM = Self.RepNet.IN_DIM
        comptime REP_OUT_DIM = Self.RepNet.OUT_DIM
        comptime PRED_IN_DIM = Self.PredNet.IN_DIM
        comptime PRED_OUT_DIM = Self.PredNet.OUT_DIM
        comptime DYN_IN_DIM = Self.DynNet.IN_DIM
        comptime DYN_OUT_DIM = Self.DynNet.OUT_DIM
        comptime MAX_NODES = 64
        comptime MCTS_BATCH_SIMS = 8
        comptime MCTS_TOTAL = Self.n_envs * MCTS_BATCH_SIMS
        comptime MCTS_ROUNDS = NUM_SIMS // MCTS_BATCH_SIMS
        comptime ENV_BLOCKS = (Self.n_envs + TPB - 1) // TPB
        comptime BATCH_BLOCKS = (Self.n_envs + TPB - 1) // TPB
        comptime WS_R = Self.RepNet.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_D = Self.DynNet.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_P = Self.PredNet.WORKSPACE_SIZE_PER_SAMPLE
        comptime MAX_WS = WS_R if WS_R > WS_D else WS_D
        comptime MAX_WS2 = MAX_WS if MAX_WS > WS_P else WS_P
        comptime WS_TOTAL = MCTS_TOTAL * MAX_WS2 if MAX_WS2 > 0 else 1

        comptime LocalGPUState = MuZeroGPUState[Self.Config, Self.n_envs]
        var gpu = LocalGPUState(ctx)
        gpu.upload_from(self.state, ctx)
        var gpu_mcts = GPUMCTSState[
            Self.n_envs, MAX_NODES, ACT, LATENT, BINS
        ](ctx)

        var obs_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs * OBS)
        var workspace_buf = ctx.enqueue_create_buffer[dtype](WS_TOTAL)

        var obs_host = ctx.enqueue_create_host_buffer[dtype](Self.n_envs * OBS)
        for e in range(Self.n_envs):
            for d in range(OBS):
                obs_host[e * OBS + d] = Scalar[dtype](obs_values[d])
        ctx.enqueue_copy(obs_buf, obs_host)
        ctx.synchronize()

        print()
        print("=== diagnose_init_state " + label + " ===")
        var ov_str = String("Input obs (env 0) [")
        for d in range(OBS):
            if d > 0:
                ov_str += ", "
            ov_str += String(obs_values[d])
        ov_str += "]"
        print(ov_str)

        # ── 1. Rep forward → hidden ──────────────────────────────────
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, REP_IN_DIM), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var hidden_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, REP_OUT_DIM), MutAnyOrigin
        ](gpu_mcts.hidden_states.unsafe_ptr())
        Self.RepNet.forward_gpu[Self.n_envs](
            ctx,
            obs_t,
            hidden_t,
            gpu.representation.params_view(),
            gpu.representation.model_state_view(),
            workspace_buf,
        )

        var hidden_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs * LATENT
        )
        ctx.enqueue_copy(hidden_host, gpu_mcts.hidden_states)
        ctx.synchronize()
        var h_min = Float64(hidden_host[0])
        var h_max = h_min
        var h_sum = Float64(0.0)
        for d in range(LATENT):
            var v = Float64(hidden_host[d])
            if v < h_min:
                h_min = v
            if v > h_max:
                h_max = v
            h_sum += v
        print(
            "[1] rep hidden (pre-scale, env 0) min=",
            h_min,
            " max=",
            h_max,
            " mean=",
            h_sum / Float64(LATENT),
        )

        # ── 2. Min-max scale ─────────────────────────────────────────
        comptime run_scale = scale_hidden_kernel[Self.n_envs, LATENT, dtype]
        var hidden_1d = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs * LATENT), MutAnyOrigin
        ](gpu_mcts.hidden_states.unsafe_ptr())
        ctx.enqueue_function[run_scale, run_scale](
            hidden_1d,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.enqueue_copy(hidden_host, gpu_mcts.hidden_states)
        ctx.synchronize()
        var hs_min = Float64(hidden_host[0])
        var hs_max = hs_min
        var hs_sum = Float64(0.0)
        var hs_first = String("[")
        for d in range(LATENT):
            var v = Float64(hidden_host[d])
            if v < hs_min:
                hs_min = v
            if v > hs_max:
                hs_max = v
            hs_sum += v
            if d < 8:
                if d > 0:
                    hs_first += ", "
                hs_first += String(v)
        hs_first += ", ...]"
        print(
            "[2] hidden (post-min-max, env 0) ",
            hs_first,
            " min=",
            hs_min,
            " max=",
            hs_max,
            " mean=",
            hs_sum / Float64(LATENT),
        )

        # ── 3. Pred forward → policy + value_dist ────────────────────
        var pred_in_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, PRED_IN_DIM), MutAnyOrigin
        ](gpu_mcts.hidden_states.unsafe_ptr())
        var pred_out_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, PRED_OUT_DIM), MutAnyOrigin
        ](gpu_mcts.pred_output.unsafe_ptr())
        Self.PredNet.forward_gpu[Self.n_envs](
            ctx,
            pred_in_t,
            pred_out_t,
            gpu.prediction.params_view(),
            gpu.prediction.model_state_view(),
            workspace_buf,
        )
        var pred_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs * PRED_OUT
        )
        ctx.enqueue_copy(pred_host, gpu_mcts.pred_output)
        ctx.synchronize()
        var pol_str = String("[3a] policy logits     [")
        for a in range(ACT):
            if a > 0:
                pol_str += ", "
            pol_str += String(Float64(pred_host[a]))
        pol_str += "]"
        print(pol_str)
        var mx_pol = Float64(pred_host[0])
        for a in range(ACT):
            var v = Float64(pred_host[a])
            if v > mx_pol:
                mx_pol = v
        var sm_sum = Float64(0.0)
        for a in range(ACT):
            sm_sum += exp(Float64(pred_host[a]) - mx_pol)
        var sm_str = String("[3b] softmax(policy)   [")
        for a in range(ACT):
            if a > 0:
                sm_str += ", "
            var p = exp(Float64(pred_host[a]) - mx_pol) / sm_sum
            sm_str += String(p)
        sm_str += "]"
        print(sm_str)
        var mx_val = Float64(pred_host[ACT])
        for b in range(BINS):
            var v = Float64(pred_host[ACT + b])
            if v > mx_val:
                mx_val = v
        var vsum = Float64(0.0)
        for b in range(BINS):
            vsum += exp(Float64(pred_host[ACT + b]) - mx_val)
        var v_min_d = Float64(self.v_min)
        var v_max_d = Float64(self.v_max)
        var step = (v_max_d - v_min_d) / Float64(BINS - 1) if BINS > 1 else 0.0
        var expected_h = Float64(0.0)
        if BINS > 1:
            for b in range(BINS):
                var atom = v_min_d + Float64(b) * step
                var p = exp(Float64(pred_host[ACT + b]) - mx_val) / vsum
                expected_h += p * atom
        else:
            expected_h = Float64(pred_host[ACT])
        var eps_h = 0.001
        var sgn = 1.0 if expected_h >= 0.0 else -1.0
        var ah = expected_h if expected_h >= 0.0 else -expected_h
        var f = (
            sqrt(1.0 + 4.0 * eps_h * (ah + 1.0 + eps_h)) - 1.0
        ) / (2.0 * eps_h)
        var v_decoded = sgn * (f * f - 1.0) if BINS > 1 else expected_h
        print(
            "[3c] value: expected_h=",
            expected_h,
            " decoded_raw=",
            v_decoded,
        )

        # ── 4. Init MCTS root (no Dirichlet) ─────────────────────────
        var vc_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs * MAX_NODES * ACT),
            MutAnyOrigin,
        ](gpu_mcts.visit_count.unsafe_ptr())
        var tv_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs * MAX_NODES * ACT),
            MutAnyOrigin,
        ](gpu_mcts.total_value.unsafe_ptr())
        var pr_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs * MAX_NODES * ACT),
            MutAnyOrigin,
        ](gpu_mcts.prior.unsafe_ptr())
        var rw_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs * MAX_NODES * ACT),
            MutAnyOrigin,
        ](gpu_mcts.reward.unsafe_ptr())
        var ci_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs * MAX_NODES * ACT),
            MutAnyOrigin,
        ](gpu_mcts.child_idx.unsafe_ptr())
        var tvis_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs * MAX_NODES), MutAnyOrigin
        ](gpu_mcts.total_visits.unsafe_ptr())
        var nc_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](gpu_mcts.node_count.unsafe_ptr())
        var po_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs * PRED_OUT), MutAnyOrigin
        ](gpu_mcts.pred_output.unsafe_ptr())
        var miq_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](gpu_mcts.min_q.unsafe_ptr())
        var mxq_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](gpu_mcts.max_q.unsafe_ptr())
        # Bug D fix (2026-05-04): zero tree state before init_root, per
        # the kernel's docstring. AZ does this at alphazero.mojo:2662.
        gpu_mcts.zero_tree(ctx)
        comptime run_init = gpu_mcts_init_root_kernel[
            Self.n_envs, MAX_NODES, ACT, LATENT, PRED_OUT, dtype
        ]
        ctx.enqueue_function[run_init, run_init](
            vc_t,
            tv_t,
            pr_t,
            rw_t,
            ci_t,
            tvis_t,
            nc_t,
            po_t,
            miq_t,
            mxq_t,
            Scalar[dtype](0.0),
            Scalar[DType.uint32](0),
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )
        var pr_full_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs * MAX_NODES * ACT
        )
        ctx.enqueue_copy(pr_full_host, gpu_mcts.prior)
        ctx.synchronize()
        var prior_str = String("[4]  root prior        [")
        for a in range(ACT):
            if a > 0:
                prior_str += ", "
            prior_str += String(Float64(pr_full_host[a]))
        prior_str += "]"
        print(prior_str)

        # ── 5. Run MCTS sims ─────────────────────────────────────────
        var hs_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs * MAX_NODES * LATENT),
            MutAnyOrigin,
        ](gpu_mcts.hidden_states.unsafe_ptr())
        var b_pp = LayoutTensor[
            dtype, Layout.row_major(MCTS_TOTAL), MutAnyOrigin
        ](gpu_mcts.pending_parent.unsafe_ptr())
        var b_pa = LayoutTensor[
            dtype, Layout.row_major(MCTS_TOTAL), MutAnyOrigin
        ](gpu_mcts.pending_action.unsafe_ptr())
        var b_sp = LayoutTensor[
            dtype, Layout.row_major(MCTS_TOTAL * MAX_DEPTH), MutAnyOrigin
        ](gpu_mcts.search_paths.unsafe_ptr())
        var b_ap = LayoutTensor[
            dtype, Layout.row_major(MCTS_TOTAL * MAX_DEPTH), MutAnyOrigin
        ](gpu_mcts.action_paths.unsafe_ptr())
        var b_pl = LayoutTensor[
            dtype, Layout.row_major(MCTS_TOTAL), MutAnyOrigin
        ](gpu_mcts.path_lengths.unsafe_ptr())
        var b_di = LayoutTensor[
            dtype, Layout.row_major(MCTS_TOTAL * DYN_IN), MutAnyOrigin
        ](gpu_mcts.dyn_input.unsafe_ptr())

        for _round in range(MCTS_ROUNDS):
            comptime run_sel_dyn = (
                gpu_mcts_batched_select_and_build_dyn_kernel[
                    Self.n_envs,
                    MAX_NODES,
                    ACT,
                    MCTS_BATCH_SIMS,
                    LATENT,
                    DYN_IN,
                    dtype,
                ]
            )
            ctx.enqueue_function[run_sel_dyn, run_sel_dyn](
                vc_t,
                tv_t,
                pr_t,
                ci_t,
                tvis_t,
                nc_t,
                miq_t,
                mxq_t,
                hs_t,
                b_di,
                b_pp,
                b_pa,
                b_sp,
                b_ap,
                b_pl,
                Scalar[dtype](Self.Config.PUCT.C_BASE),
                Scalar[dtype](Self.Config.PUCT.C_INIT),
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
            var dyn_in_b = LayoutTensor[
                dtype,
                Layout.row_major(MCTS_TOTAL, DYN_IN_DIM),
                MutAnyOrigin,
            ](gpu_mcts.dyn_input.unsafe_ptr())
            var dyn_out_b = LayoutTensor[
                dtype,
                Layout.row_major(MCTS_TOTAL, DYN_OUT_DIM),
                MutAnyOrigin,
            ](gpu_mcts.dyn_output.unsafe_ptr())
            Self.DynNet.forward_gpu[MCTS_TOTAL](
                ctx,
                dyn_in_b,
                dyn_out_b,
                gpu.dynamics.params_view(),
                gpu.dynamics.model_state_view(),
                workspace_buf,
            )
            var pred_in_b = LayoutTensor[
                dtype, Layout.row_major(MCTS_TOTAL * LATENT), MutAnyOrigin
            ](gpu_mcts.pred_input.unsafe_ptr())
            var dyn_out_b_flat = LayoutTensor[
                dtype, Layout.row_major(MCTS_TOTAL * DYN_OUT), MutAnyOrigin
            ](gpu_mcts.dyn_output.unsafe_ptr())
            comptime EXTR_TOTAL = MCTS_TOTAL * LATENT
            comptime EXTR_BLK = (EXTR_TOTAL + TPB - 1) // TPB
            comptime run_extr = extract_hidden_kernel[
                MCTS_TOTAL, LATENT, DYN_OUT, dtype
            ]
            ctx.enqueue_function[run_extr, run_extr](
                pred_in_b,
                dyn_out_b_flat,
                grid_dim=(EXTR_BLK,),
                block_dim=(TPB,),
            )
            var pred_in_net = LayoutTensor[
                dtype,
                Layout.row_major(MCTS_TOTAL, PRED_IN_DIM),
                MutAnyOrigin,
            ](gpu_mcts.pred_input.unsafe_ptr())
            var pred_out_net = LayoutTensor[
                dtype,
                Layout.row_major(MCTS_TOTAL, PRED_OUT_DIM),
                MutAnyOrigin,
            ](gpu_mcts.pred_output.unsafe_ptr())
            Self.PredNet.forward_gpu[MCTS_TOTAL](
                ctx,
                pred_in_net,
                pred_out_net,
                gpu.prediction.params_view(),
                gpu.prediction.model_state_view(),
                workspace_buf,
            )
            var b_do = LayoutTensor[
                dtype, Layout.row_major(MCTS_TOTAL * DYN_OUT), MutAnyOrigin
            ](gpu_mcts.dyn_output.unsafe_ptr())
            var b_po = LayoutTensor[
                dtype, Layout.row_major(MCTS_TOTAL * PRED_OUT), MutAnyOrigin
            ](gpu_mcts.pred_output.unsafe_ptr())
            comptime run_exp_bk = (
                gpu_mcts_batched_expand_backup_muzero_kernel[
                    Self.n_envs,
                    MAX_NODES,
                    ACT,
                    MCTS_BATCH_SIMS,
                    LATENT,
                    PRED_OUT,
                    DYN_OUT,
                    dtype,
                ]
            )
            ctx.enqueue_function[run_exp_bk, run_exp_bk](
                vc_t,
                tv_t,
                pr_t,
                rw_t,
                ci_t,
                tvis_t,
                nc_t,
                miq_t,
                mxq_t,
                hs_t,
                b_pp,
                b_pa,
                b_do,
                b_po,
                b_sp,
                b_ap,
                b_pl,
                Scalar[dtype](self.v_min),
                Scalar[dtype](self.v_max),
                Scalar[dtype](self.gamma),
                Scalar[DType.bool](False),
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )

        # ── 6. Read root visits and total_value ──────────────────────
        var vc_full_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs * MAX_NODES * ACT
        )
        var tv_full_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs * MAX_NODES * ACT
        )
        ctx.enqueue_copy(vc_full_host, gpu_mcts.visit_count)
        ctx.enqueue_copy(tv_full_host, gpu_mcts.total_value)
        ctx.synchronize()
        var vc_str = String("[5a] root visit_count  [")
        var tv_str = String("[5b] root total_value  [")
        var vsum_root = Float64(0.0)
        var tvsum_root = Float64(0.0)
        for a in range(ACT):
            var vc = Float64(vc_full_host[a])
            var tv = Float64(tv_full_host[a])
            if a > 0:
                vc_str += ", "
                tv_str += ", "
            vc_str += String(vc)
            tv_str += String(tv)
            vsum_root += vc
            tvsum_root += tv
        vc_str += "]"
        tv_str += "]"
        print(vc_str)
        print(tv_str)
        var root_v = (
            tvsum_root / vsum_root if vsum_root > 0.5 else 0.0
        )
        print(
            "[5c] root V (Σtv/Σvc)  =",
            root_v,
            " (visit_sum=",
            vsum_root,
            ")",
        )
        var vp_str = String("[5d] visit policy      [")
        for a in range(ACT):
            if a > 0:
                vp_str += ", "
            var p = (
                Float64(vc_full_host[a]) / vsum_root if vsum_root > 0.5 else 0.5
            )
            vp_str += String(p)
        vp_str += "]"
        print(vp_str)

        # ── 7. Extra: tree state for debugging ───────────────────────
        var nc_host = ctx.enqueue_create_host_buffer[dtype](Self.n_envs)
        var tvis_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs * MAX_NODES
        )
        ctx.enqueue_copy(nc_host, gpu_mcts.node_count)
        ctx.enqueue_copy(tvis_host, gpu_mcts.total_visits)
        ctx.synchronize()
        print(
            "[6a] node_count(env 0) =",
            Int(Float64(nc_host[0])),
            " total_visits(root)   =",
            Float64(tvis_host[0]),
        )
        # Visit counts for first 3 child nodes (1, 2, 3) if they exist
        var n_nodes = Int(Float64(nc_host[0]))
        var nodes_to_dump = 3 if n_nodes > 3 else n_nodes - 1
        if nodes_to_dump > 0:
            var child_str = String("[6b] children visit_count: ")
            for nidx in range(1, 1 + nodes_to_dump):
                child_str += "node" + String(nidx) + "=["
                for a in range(ACT):
                    if a > 0:
                        child_str += ","
                    child_str += String(
                        Float64(vc_full_host[nidx * ACT + a])
                    )
                child_str += "] "
            print(child_str)

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
        var grad_steps = (
            gradient_steps if gradient_steps > 0 else Self.n_envs
        )

        # ── Create GPU state with correct Self.n_envs ─────────────────────
        comptime LocalGPUState = MuZeroGPUState[Self.Config, Self.n_envs]
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

        # GPU MCTS state
        comptime LATENT = Self.Config.latent_dim
        comptime BINS = Self.Config.num_bins
        comptime MAX_NODES = 64
        comptime NUM_SIMS = Self.Config.num_simulations
        var gpu_mcts = GPUMCTSState[Self.n_envs, MAX_NODES, ACT, LATENT, BINS](
            ctx
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

        # Extract initial observations
        # Use a step with no-op to get initial obs (reset already randomized state)
        actions_buf.enqueue_fill(Scalar[dtype](0.0))
        E.step_kernel_gpu[Self.n_envs, E.STATE_SIZE, OBS](
            ctx,
            states_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            terminated_buf,
            obs_buf,
            rng_seed=0,
            workspace_ptr=workspace_buf.unsafe_ptr(),
        )
        # Re-reset since step may have changed state
        E.reset_kernel_gpu[Self.n_envs, E.STATE_SIZE](
            ctx, states_buf, rng_seed=123
        )
        E.step_kernel_gpu[Self.n_envs, E.STATE_SIZE, OBS](
            ctx,
            states_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            terminated_buf,
            obs_buf,
            rng_seed=0,
            workspace_ptr=workspace_buf.unsafe_ptr(),
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
                ctx.enqueue_function[run_warmup, run_warmup](
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

                # 2a. Representation forward: obs → root hidden states
                var rep_obs_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, REP_IN_DIM),
                    MutAnyOrigin,
                ](obs_buf.unsafe_ptr())
                var rep_h_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, REP_OUT_DIM),
                    MutAnyOrigin,
                ](gpu_mcts.hidden_states.unsafe_ptr())
                RepNet.forward_gpu[Self.n_envs](
                    ctx,
                    rep_obs_t,
                    rep_h_t,
                    gpu.representation.params_view(),
                    gpu.representation.model_state_view(),
                    mcts_workspace,
                )

                # 2b. Prediction forward: root hidden → policy + value
                var pred_root_in = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, PRED_IN_DIM),
                    MutAnyOrigin,
                ](gpu_mcts.hidden_states.unsafe_ptr())
                var pred_root_out = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, PRED_OUT_DIM),
                    MutAnyOrigin,
                ](gpu_mcts.pred_output.unsafe_ptr())
                PredNet.forward_gpu[Self.n_envs](
                    ctx,
                    pred_root_in,
                    pred_root_out,
                    gpu.prediction.params_view(),
                    gpu.prediction.model_state_view(),
                    mcts_workspace,
                )

                # 2c. Initialize root nodes
                var vc_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                    MutAnyOrigin,
                ](gpu_mcts.visit_count.unsafe_ptr())
                var tv_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                    MutAnyOrigin,
                ](gpu_mcts.total_value.unsafe_ptr())
                var pr_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                    MutAnyOrigin,
                ](gpu_mcts.prior.unsafe_ptr())
                var rw_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                    MutAnyOrigin,
                ](gpu_mcts.reward.unsafe_ptr())
                var ci_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                    MutAnyOrigin,
                ](gpu_mcts.child_idx.unsafe_ptr())
                var tvis_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES),
                    MutAnyOrigin,
                ](gpu_mcts.total_visits.unsafe_ptr())
                var nc_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](gpu_mcts.node_count.unsafe_ptr())
                var po_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * PRED_OUT),
                    MutAnyOrigin,
                ](gpu_mcts.pred_output.unsafe_ptr())
                var miq_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](gpu_mcts.min_q.unsafe_ptr())
                var mxq_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](gpu_mcts.max_q.unsafe_ptr())

                # Bug D fix (2026-05-04): zero tree state before init_root.
                # Without this, the previous env-step's child_idx still
                # points to old expanded children → MCTS descends into
                # stale tree → backup accumulates ~30 visits per sim
                # instead of 1. AZ does the same at alphazero.mojo:2662.
                gpu_mcts.zero_tree(ctx)
                comptime run_init = gpu_mcts_init_root_kernel[
                    Self.n_envs, MAX_NODES, ACT, LATENT, PRED_OUT, dtype
                ]
                ctx.enqueue_function[run_init, run_init](
                    vc_t,
                    tv_t,
                    pr_t,
                    rw_t,
                    ci_t,
                    tvis_t,
                    nc_t,
                    po_t,
                    miq_t,
                    mxq_t,
                    Scalar[dtype](0.25),
                    Scalar[DType.uint32](UInt32(total_steps)),
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                # 2d. Run NUM_SIMS simulation rounds
                # ── Batched simulations (BATCH_SIMS per round) ──
                comptime MCTS_BATCH_SIMS = 8
                comptime MCTS_ROUNDS = NUM_SIMS // MCTS_BATCH_SIMS
                comptime MCTS_TOTAL = Self.n_envs * MCTS_BATCH_SIMS

                var hs_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES * LATENT),
                    MutAnyOrigin,
                ](gpu_mcts.hidden_states.unsafe_ptr())
                var b_pp = LayoutTensor[
                    dtype, Layout.row_major(MCTS_TOTAL), MutAnyOrigin
                ](gpu_mcts.pending_parent.unsafe_ptr())
                var b_pa = LayoutTensor[
                    dtype, Layout.row_major(MCTS_TOTAL), MutAnyOrigin
                ](gpu_mcts.pending_action.unsafe_ptr())
                var b_sp = LayoutTensor[
                    dtype,
                    Layout.row_major(MCTS_TOTAL * MAX_DEPTH),
                    MutAnyOrigin,
                ](gpu_mcts.search_paths.unsafe_ptr())
                var b_ap = LayoutTensor[
                    dtype,
                    Layout.row_major(MCTS_TOTAL * MAX_DEPTH),
                    MutAnyOrigin,
                ](gpu_mcts.action_paths.unsafe_ptr())
                var b_pl = LayoutTensor[
                    dtype, Layout.row_major(MCTS_TOTAL), MutAnyOrigin
                ](gpu_mcts.path_lengths.unsafe_ptr())
                var b_di = LayoutTensor[
                    dtype, Layout.row_major(MCTS_TOTAL * DYN_IN), MutAnyOrigin
                ](gpu_mcts.dyn_input.unsafe_ptr())

                for _round in range(MCTS_ROUNDS):
                    # 1. Fused select + build dynamics input
                    comptime run_sel_dyn = gpu_mcts_batched_select_and_build_dyn_kernel[
                        Self.n_envs,
                        MAX_NODES,
                        ACT,
                        MCTS_BATCH_SIMS,
                        LATENT,
                        DYN_IN,
                        dtype,
                    ]
                    ctx.enqueue_function[run_sel_dyn, run_sel_dyn](
                        vc_t,
                        tv_t,
                        pr_t,
                        ci_t,
                        tvis_t,
                        nc_t,
                        miq_t,
                        mxq_t,
                        hs_t,
                        b_di,
                        b_pp,
                        b_pa,
                        b_sp,
                        b_ap,
                        b_pl,
                        Scalar[dtype](Self.Config.PUCT.C_BASE),
                        Scalar[dtype](Self.Config.PUCT.C_INIT),
                        grid_dim=(ENV_BLOCKS,),
                        block_dim=(TPB,),
                    )

                    # 2. Batched dynamics forward [BATCH = n_envs * BATCH_SIMS]
                    var dyn_in_b = LayoutTensor[
                        dtype,
                        Layout.row_major(MCTS_TOTAL, DYN_IN_DIM),
                        MutAnyOrigin,
                    ](gpu_mcts.dyn_input.unsafe_ptr())
                    var dyn_out_b = LayoutTensor[
                        dtype,
                        Layout.row_major(MCTS_TOTAL, DYN_OUT_DIM),
                        MutAnyOrigin,
                    ](gpu_mcts.dyn_output.unsafe_ptr())
                    DynNet.forward_gpu[MCTS_TOTAL](
                        ctx,
                        dyn_in_b,
                        dyn_out_b,
                        gpu.dynamics.params_view(),
                        gpu.dynamics.model_state_view(),
                        mcts_workspace,
                    )

                    # 3. Extract hidden from dyn → pred input
                    var pred_in_b = LayoutTensor[
                        dtype,
                        Layout.row_major(MCTS_TOTAL * LATENT),
                        MutAnyOrigin,
                    ](gpu_mcts.pred_input.unsafe_ptr())
                    var dyn_out_b_flat = LayoutTensor[
                        dtype,
                        Layout.row_major(MCTS_TOTAL * DYN_OUT),
                        MutAnyOrigin,
                    ](gpu_mcts.dyn_output.unsafe_ptr())
                    comptime EXTR_TOTAL = MCTS_TOTAL * LATENT
                    comptime EXTR_BLK = (EXTR_TOTAL + TPB - 1) // TPB
                    comptime run_extr = extract_hidden_kernel[
                        MCTS_TOTAL, LATENT, DYN_OUT, dtype
                    ]
                    ctx.enqueue_function[run_extr, run_extr](
                        pred_in_b,
                        dyn_out_b_flat,
                        grid_dim=(EXTR_BLK,),
                        block_dim=(TPB,),
                    )

                    # 4. Batched prediction forward [BATCH = n_envs * BATCH_SIMS]
                    var pred_in_net = LayoutTensor[
                        dtype,
                        Layout.row_major(MCTS_TOTAL, PRED_IN_DIM),
                        MutAnyOrigin,
                    ](gpu_mcts.pred_input.unsafe_ptr())
                    var pred_out_net = LayoutTensor[
                        dtype,
                        Layout.row_major(MCTS_TOTAL, PRED_OUT_DIM),
                        MutAnyOrigin,
                    ](gpu_mcts.pred_output.unsafe_ptr())
                    PredNet.forward_gpu[MCTS_TOTAL](
                        ctx,
                        pred_in_net,
                        pred_out_net,
                        gpu.prediction.params_view(),
                        gpu.prediction.model_state_view(),
                        mcts_workspace,
                    )

                    # 5. Fused expand + backup + remove virtual losses
                    var b_do = LayoutTensor[
                        dtype,
                        Layout.row_major(MCTS_TOTAL * DYN_OUT),
                        MutAnyOrigin,
                    ](gpu_mcts.dyn_output.unsafe_ptr())
                    var b_po = LayoutTensor[
                        dtype,
                        Layout.row_major(MCTS_TOTAL * PRED_OUT),
                        MutAnyOrigin,
                    ](gpu_mcts.pred_output.unsafe_ptr())
                    comptime run_exp_bk = gpu_mcts_batched_expand_backup_muzero_kernel[
                        Self.n_envs,
                        MAX_NODES,
                        ACT,
                        MCTS_BATCH_SIMS,
                        LATENT,
                        PRED_OUT,
                        DYN_OUT,
                        dtype,
                    ]
                    ctx.enqueue_function[run_exp_bk, run_exp_bk](
                        vc_t,
                        tv_t,
                        pr_t,
                        rw_t,
                        ci_t,
                        tvis_t,
                        nc_t,
                        miq_t,
                        mxq_t,
                        hs_t,
                        b_pp,
                        b_pa,
                        b_do,
                        b_po,
                        b_sp,
                        b_ap,
                        b_pl,
                        Scalar[dtype](self.v_min),
                        Scalar[dtype](self.v_max),
                        Scalar[dtype](self.gamma),
                        Scalar[DType.bool](False),
                        grid_dim=(ENV_BLOCKS,),
                        block_dim=(TPB,),
                    )

                # 2e. Extract actions + policies from visit counts via
                # temperature-weighted sampling. Bug E (2026-05-04): the
                # previous `gpu_mcts_extract_actions_kernel` did pure
                # argmax — when MCTS visits tie (which happens whenever
                # pred init produces a uniform-ish prior, e.g. post-Bug-A
                # zero-init), argmax deterministically returns action 0
                # every step → CartPole tips deterministically → 5-step
                # episodes regardless of training progress. The temp
                # kernel samples π(a) ∝ N_a^(1/τ) for τ > 0, breaking
                # ties stochastically. τ = self.temperature decays from
                # 1.0 (full exploration) to 0.01 (greedy) over
                # `temperature_decay_steps`, matching muzero-general.
                var act_out_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](actions_buf.unsafe_ptr())
                var pol_out_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin
                ](gpu.mcts_step_policy_buf.unsafe_ptr())
                var lm_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin
                ](legal_masks_buf.unsafe_ptr())
                var ep_steps_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](episode_steps_buf.unsafe_ptr())
                comptime run_act = gpu_mcts_extract_actions_temp_kernel[
                    Self.n_envs, MAX_NODES, ACT, dtype
                ]
                # temp_threshold=0 → use temp_min=self.temperature for
                # every move (no separate "exploration phase" — the global
                # temperature schedule handles annealing). Lifted to
                # comptime per Mojo GPU enqueue_function constraint on
                # runtime Int args (see feedback_gpu_scalar_args memory).
                comptime _temp_thr_zero = 0
                ctx.enqueue_function[run_act, run_act](
                    vc_t,
                    lm_t,
                    ep_steps_t,
                    act_out_t,
                    pol_out_t,
                    _temp_thr_zero,
                    Scalar[DType.uint32](UInt32(total_steps)),
                    Scalar[dtype](self.temperature),
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Tally per-action selection counts for diagnostics.
                var hist_t = LayoutTensor[
                    dtype, Layout.row_major(ACT), MutAnyOrigin
                ](action_hist_buf.unsafe_ptr())
                comptime run_hist = action_histogram_kernel[
                    Self.n_envs, ACT, dtype
                ]
                ctx.enqueue_function[run_hist, run_hist](
                    act_out_t,
                    hist_t,
                    grid_dim=(1,),
                    block_dim=(1,),
                )

                # Switch-rate counter — increments by #{i : action[t,i] != action[t-1,i]}
                # then snapshots actions into prev_actions for the next step.
                var prev_act_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](prev_actions_buf.unsafe_ptr())
                var switch_t = LayoutTensor[
                    dtype, Layout.row_major(1), MutAnyOrigin
                ](switch_count_buf.unsafe_ptr())
                comptime run_switch = action_switch_kernel[
                    Self.n_envs, dtype
                ]
                ctx.enqueue_function[run_switch, run_switch](
                    act_out_t,
                    prev_act_t,
                    switch_t,
                    grid_dim=(1,),
                    block_dim=(1,),
                )

                # Compute root values for MCTS value targets via
                # Σ_a total_value[root,a] / Σ_a visit_count[root,a] — the
                # MCTS-improved value at the root, in raw scalar space (the
                # expand+backup kernel decodes via h⁻¹ post-F1, so total_value
                # is already raw). This feeds the n-step bootstrap V(s_{t+n})
                # in nstep_value_targets_kernel. Replaces the previous stub
                # that filled zero — see docs/MUZERO_AUDIT.md F2.
                var val_out_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](gpu.mcts_step_value_buf.unsafe_ptr())
                comptime run_root_val = gpu_mcts_extract_root_value_kernel[
                    Self.n_envs, MAX_NODES, ACT, dtype
                ]
                ctx.enqueue_function[run_root_val, run_root_val](
                    vc_t,
                    tv_t,
                    val_out_t,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
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
            gpu.replay.store_with_termination(
                ctx, actions_buf, rewards_buf, dones_buf, terminated_buf
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

            ctx.enqueue_function[store_targets_wrapper, store_targets_wrapper](
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
            ctx.enqueue_function[run_accum, run_accum](
                ep_rew_t,
                rewards_t,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
            comptime run_incr = increment_steps_kernel[dtype, Self.n_envs]
            ctx.enqueue_function[run_incr, run_incr](
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
            ctx.enqueue_function[run_log, run_log](
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
            # Re-extract obs after reset
            E.step_kernel_gpu[Self.n_envs, E.STATE_SIZE, OBS](
                ctx,
                states_buf,
                actions_buf,
                rewards_buf,
                dones_buf,
                terminated_buf,
                obs_buf,
                rng_seed=0,
                workspace_ptr=workspace_buf.unsafe_ptr(),
            )

            total_steps += Self.n_envs
            self.total_steps += Self.n_envs

            # ── 9. GPU training (sampling + targets + training all on GPU) ──
            if (
                total_steps >= warmup_steps
                and gpu.replay.is_ready[Self.Config.unroll_steps + 1]()
            ):
                for _ in range(grad_steps):
                    _ = self.update_gpu(ctx, gpu, use_reanalyze=use_reanalyze)
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
                    var pct = (c / hist_total * 100.0) if hist_total > 0 else Float64(0.0)
                    act_line += String("a") + String(ai) + String("=") + String(Int(c))
                    act_line += String(" (") + String(Int(pct)) + String("%)")
                    if ai < ACT - 1:
                        act_line += String(", ")
                act_line += String("]  total=") + String(Int(hist_total))
                var switches = Float64(switch_count_host[0])
                var switch_pct = (switches / hist_total * 100.0) if hist_total > 0 else Float64(0.0)
                act_line += String("  switches=") + String(Int(switches))
                act_line += String(" (") + String(Int(switch_pct)) + String("%)")
                print(act_line)
                action_hist_buf.enqueue_fill(Scalar[dtype](0.0))
                switch_count_buf.enqueue_fill(Scalar[dtype](0.0))

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
        comptime BATCH_SIMS = 8
        comptime NUM_ROUNDS = NUM_SIMS // BATCH_SIMS
        comptime TOTAL_EXPAND = Self.n_envs * BATCH_SIMS

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

        # ── GPU MCTS state ───────────────────────────────────────
        var gpu_mcts = GPUMCTSState[
            Self.n_envs, MAX_NODES, ACT, LATENT, BINS, GS
        ](ctx)

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

                    # Representation: obs → hidden (root)
                    var rep_obs = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs, REP_IN_DIM),
                        MutAnyOrigin,
                    ](obs_buf.unsafe_ptr())
                    var rep_h = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs, REP_OUT_DIM),
                        MutAnyOrigin,
                    ](gpu_mcts.hidden_states.unsafe_ptr())
                    RepNet.forward_gpu[Self.n_envs](
                        ctx,
                        rep_obs,
                        rep_h,
                        gpu.representation.params_view(),
                        gpu.representation.model_state_view(),
                        mcts_workspace,
                    )

                    # Prediction on root hidden state
                    var pred_root_in = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs, PRED_IN_DIM),
                        MutAnyOrigin,
                    ](gpu_mcts.hidden_states.unsafe_ptr())
                    var pred_root_out = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs, PRED_OUT_DIM),
                        MutAnyOrigin,
                    ](gpu_mcts.pred_output.unsafe_ptr())
                    PredNet.forward_gpu[Self.n_envs](
                        ctx,
                        pred_root_in,
                        pred_root_out,
                        gpu.prediction.params_view(),
                        gpu.prediction.model_state_view(),
                        mcts_workspace,
                    )

                    # Init root with Dirichlet noise
                    var vc_t = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                        MutAnyOrigin,
                    ](gpu_mcts.visit_count.unsafe_ptr())
                    var tv_t = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                        MutAnyOrigin,
                    ](gpu_mcts.total_value.unsafe_ptr())
                    var pr_t = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                        MutAnyOrigin,
                    ](gpu_mcts.prior.unsafe_ptr())
                    var rw_t = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                        MutAnyOrigin,
                    ](gpu_mcts.reward.unsafe_ptr())
                    var ci_t = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                        MutAnyOrigin,
                    ](gpu_mcts.child_idx.unsafe_ptr())
                    var tvis_t = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs * MAX_NODES),
                        MutAnyOrigin,
                    ](gpu_mcts.total_visits.unsafe_ptr())
                    var nc_t = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                    ](gpu_mcts.node_count.unsafe_ptr())
                    var po_t = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs * PRED_OUT),
                        MutAnyOrigin,
                    ](gpu_mcts.pred_output.unsafe_ptr())
                    var miq_t = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                    ](gpu_mcts.min_q.unsafe_ptr())
                    var mxq_t = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                    ](gpu_mcts.max_q.unsafe_ptr())

                    # Bug D fix (2026-05-04): zero tree state before
                    # init_root. AZ does the same at alphazero.mojo:2662.
                    gpu_mcts.zero_tree(ctx)
                    comptime run_init = gpu_mcts_init_root_kernel[
                        Self.n_envs, MAX_NODES, ACT, LATENT, PRED_OUT, dtype
                    ]
                    ctx.enqueue_function[run_init, run_init](
                        vc_t,
                        tv_t,
                        pr_t,
                        rw_t,
                        ci_t,
                        tvis_t,
                        nc_t,
                        po_t,
                        miq_t,
                        mxq_t,
                        Scalar[dtype](Self.Config.Noise.NOISE_FRACTION),
                        Scalar[DType.uint32](UInt32(total_steps + iter_steps)),
                        grid_dim=(ENV_BLOCKS,),
                        block_dim=(TPB,),
                    )

                    # Apply legal mask to root prior
                    var lm_t = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin
                    ](legal_masks_buf.unsafe_ptr())
                    comptime run_mask = gpu_mcts_apply_legal_mask_kernel[
                        Self.n_envs, MAX_NODES, ACT, dtype
                    ]
                    ctx.enqueue_function[run_mask, run_mask](
                        pr_t,
                        lm_t,
                        grid_dim=(ENV_BLOCKS,),
                        block_dim=(TPB,),
                    )

                    # ── Batched MCTS simulations (BATCH_SIMS per round) ──
                    var hs_t = LayoutTensor[
                        dtype,
                        Layout.row_major(Self.n_envs * MAX_NODES * LATENT),
                        MutAnyOrigin,
                    ](gpu_mcts.hidden_states.unsafe_ptr())

                    # Batched buffers [N_ENVS * BATCH_SIMS * ...]
                    var b_pp = LayoutTensor[
                        dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin
                    ](gpu_mcts.pending_parent.unsafe_ptr())
                    var b_pa = LayoutTensor[
                        dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin
                    ](gpu_mcts.pending_action.unsafe_ptr())
                    var b_sp = LayoutTensor[
                        dtype,
                        Layout.row_major(TOTAL_EXPAND * MAX_DEPTH),
                        MutAnyOrigin,
                    ](gpu_mcts.search_paths.unsafe_ptr())
                    var b_ap = LayoutTensor[
                        dtype,
                        Layout.row_major(TOTAL_EXPAND * MAX_DEPTH),
                        MutAnyOrigin,
                    ](gpu_mcts.action_paths.unsafe_ptr())
                    var b_pl = LayoutTensor[
                        dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin
                    ](gpu_mcts.path_lengths.unsafe_ptr())

                    for _r in range(NUM_ROUNDS):
                        # Select BATCH_SIMS leaves + build dynamics input
                        var b_dyn_in = LayoutTensor[
                            dtype,
                            Layout.row_major(TOTAL_EXPAND * DYN_IN),
                            MutAnyOrigin,
                        ](gpu_mcts.dyn_input.unsafe_ptr())
                        comptime run_sel_dyn = gpu_mcts_batched_select_and_build_dyn_kernel[
                            Self.n_envs,
                            MAX_NODES,
                            ACT,
                            BATCH_SIMS,
                            LATENT,
                            DYN_IN,
                            dtype,
                        ]
                        ctx.enqueue_function[run_sel_dyn, run_sel_dyn](
                            vc_t,
                            tv_t,
                            pr_t,
                            ci_t,
                            tvis_t,
                            nc_t,
                            miq_t,
                            mxq_t,
                            hs_t,
                            b_dyn_in,
                            b_pp,
                            b_pa,
                            b_sp,
                            b_ap,
                            b_pl,
                            Scalar[dtype](Self.Config.PUCT.C_BASE),
                            Scalar[dtype](Self.Config.PUCT.C_INIT),
                            grid_dim=(ENV_BLOCKS,),
                            block_dim=(TPB,),
                        )

                        # Dynamics forward on batched leaves
                        var dyn_in_t = LayoutTensor[
                            dtype,
                            Layout.row_major(TOTAL_EXPAND, DYN_IN_DIM),
                            MutAnyOrigin,
                        ](gpu_mcts.dyn_input.unsafe_ptr())
                        var dyn_out_t = LayoutTensor[
                            dtype,
                            Layout.row_major(TOTAL_EXPAND, DYN_OUT_DIM),
                            MutAnyOrigin,
                        ](gpu_mcts.dyn_output.unsafe_ptr())
                        DynNet.forward_gpu[TOTAL_EXPAND](
                            ctx,
                            dyn_in_t,
                            dyn_out_t,
                            gpu.dynamics.params_view(),
                            gpu.dynamics.model_state_view(),
                            mcts_workspace,
                        )

                        # Prediction forward on new hidden states
                        # (extract hidden from dyn_output first, then predict)
                        var pred_in_flat = LayoutTensor[
                            dtype,
                            Layout.row_major(TOTAL_EXPAND * LATENT),
                            MutAnyOrigin,
                        ](gpu_mcts.pred_input.unsafe_ptr())
                        var dyn_out_flat = LayoutTensor[
                            dtype,
                            Layout.row_major(TOTAL_EXPAND * DYN_OUT),
                            MutAnyOrigin,
                        ](gpu_mcts.dyn_output.unsafe_ptr())
                        comptime EXTR_BLK = (
                            TOTAL_EXPAND * LATENT + TPB - 1
                        ) // TPB
                        comptime run_extr = extract_hidden_kernel[
                            TOTAL_EXPAND, LATENT, DYN_OUT, dtype
                        ]
                        ctx.enqueue_function[run_extr, run_extr](
                            pred_in_flat,
                            dyn_out_flat,
                            grid_dim=(EXTR_BLK,),
                            block_dim=(TPB,),
                        )

                        var pred_in_t = LayoutTensor[
                            dtype,
                            Layout.row_major(TOTAL_EXPAND, PRED_IN_DIM),
                            MutAnyOrigin,
                        ](gpu_mcts.pred_input.unsafe_ptr())
                        var pred_out_t = LayoutTensor[
                            dtype,
                            Layout.row_major(TOTAL_EXPAND, PRED_OUT_DIM),
                            MutAnyOrigin,
                        ](gpu_mcts.pred_output.unsafe_ptr())
                        PredNet.forward_gpu[TOTAL_EXPAND](
                            ctx,
                            pred_in_t,
                            pred_out_t,
                            gpu.prediction.params_view(),
                            gpu.prediction.model_state_view(),
                            mcts_workspace,
                        )

                        # Expand + backup
                        var b_dyn_out = LayoutTensor[
                            dtype,
                            Layout.row_major(TOTAL_EXPAND * DYN_OUT),
                            MutAnyOrigin,
                        ](gpu_mcts.dyn_output.unsafe_ptr())
                        var b_pred_out = LayoutTensor[
                            dtype,
                            Layout.row_major(TOTAL_EXPAND * PRED_OUT),
                            MutAnyOrigin,
                        ](gpu_mcts.pred_output.unsafe_ptr())
                        comptime run_exp_bk = gpu_mcts_batched_expand_backup_muzero_kernel[
                            Self.n_envs,
                            MAX_NODES,
                            ACT,
                            BATCH_SIMS,
                            LATENT,
                            PRED_OUT,
                            DYN_OUT,
                            dtype,
                        ]
                        ctx.enqueue_function[run_exp_bk, run_exp_bk](
                            vc_t,
                            tv_t,
                            pr_t,
                            rw_t,
                            ci_t,
                            tvis_t,
                            nc_t,
                            miq_t,
                            mxq_t,
                            hs_t,
                            b_pp,
                            b_pa,
                            b_dyn_out,
                            b_pred_out,
                            b_sp,
                            b_ap,
                            b_pl,
                            Scalar[dtype](self.v_min),
                            Scalar[dtype](self.v_max),
                            Scalar[dtype](self.gamma),
                            Scalar[DType.bool](NEGATE),
                            grid_dim=(ENV_BLOCKS,),
                            block_dim=(TPB,),
                        )

                    # Extract actions with temperature annealing
                    var act_out = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                    ](actions_buf.unsafe_ptr())
                    var pol_out = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin
                    ](gpu.mcts_step_policy_buf.unsafe_ptr())

                    # Temperature-annealed action extraction
                    # Uses per-env ep_steps to decide temp=1 or temp=0
                    var ep_steps_t = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                    ](ep_steps_buf.unsafe_ptr())
                    comptime run_act_temp = gpu_mcts_extract_actions_temp_kernel[
                        Self.n_envs, MAX_NODES, ACT, dtype
                    ]
                    comptime _temp_thr = temp_threshold
                    ctx.enqueue_function[run_act_temp, run_act_temp](
                        vc_t,
                        lm_t,
                        ep_steps_t,
                        act_out,
                        pol_out,
                        _temp_thr,
                        Scalar[DType.uint32](UInt32(total_steps + iter_steps)),
                        Scalar[dtype](0.0),
                        grid_dim=(ENV_BLOCKS,),
                        block_dim=(TPB,),
                    )

                    # Compute root values for MCTS value targets via
                    # Σ_a total_value[root,a] / Σ_a visit_count[root,a].
                    # Feeds the n-step bootstrap V(s_{t+n}). Replaces the
                    # previous stub that filled zero — see Phase F2 of
                    # docs/MUZERO_AUDIT.md and docs/AUTODIFF_GRAD_ACCUMULATION.md.
                    var sp_val_out_t = LayoutTensor[
                        dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                    ](gpu.mcts_step_value_buf.unsafe_ptr())
                    comptime sp_run_root_val = (
                        gpu_mcts_extract_root_value_kernel[
                            Self.n_envs, MAX_NODES, ACT, dtype
                        ]
                    )
                    ctx.enqueue_function[
                        sp_run_root_val, sp_run_root_val
                    ](
                        vc_t,
                        tv_t,
                        sp_val_out_t,
                        grid_dim=(ENV_BLOCKS,),
                        block_dim=(TPB,),
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
                    ctx.enqueue_function[run_warmup, run_warmup](
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
                ctx.enqueue_function[run_to_play, run_to_play](
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

                ctx.enqueue_function[store_tgt_w, store_tgt_w](
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
                ctx.enqueue_function[run_accum, run_accum](
                    ep_rew_t,
                    rewards_t,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
                comptime run_incr = increment_steps_kernel[dtype, Self.n_envs]
                ctx.enqueue_function[run_incr, run_incr](
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
                ctx.enqueue_function[run_log, run_log](
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
                for _ in range(grad_steps):
                    _ = self.update_gpu(
                        ctx, gpu, use_reanalyze=use_reanalyze
                    )
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
        comptime BATCH_SIMS = 8
        comptime NUM_ROUNDS = Self.Config.num_simulations // BATCH_SIMS
        comptime TOTAL_EXPAND = Self.n_envs * BATCH_SIMS
        comptime NEGATE = Self.Config.Players.NEGATE_BACKUP

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

        # MCTS state for evaluation
        var eval_mcts = GPUMCTSState[
            Self.n_envs, MAX_NODES, ACT, LATENT, BINS, GS
        ](ctx)

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

                # Representation: obs → hidden
                var rep_obs = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, REP_IN_DIM),
                    MutAnyOrigin,
                ](eval_obs.unsafe_ptr())
                var rep_h = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, REP_OUT_DIM),
                    MutAnyOrigin,
                ](eval_mcts.hidden_states.unsafe_ptr())
                RepNet.forward_gpu[Self.n_envs](
                    ctx,
                    rep_obs,
                    rep_h,
                    gpu.representation.params_view(),
                    gpu.representation.model_state_view(),
                    mcts_ws,
                )

                # Prediction on root
                var pred_root_in = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, PRED_IN_DIM),
                    MutAnyOrigin,
                ](eval_mcts.hidden_states.unsafe_ptr())
                var pred_root_out = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, PRED_OUT_DIM),
                    MutAnyOrigin,
                ](eval_mcts.pred_output.unsafe_ptr())
                PredNet.forward_gpu[Self.n_envs](
                    ctx,
                    pred_root_in,
                    pred_root_out,
                    gpu.prediction.params_view(),
                    gpu.prediction.model_state_view(),
                    mcts_ws,
                )

                # Init root (no noise)
                var e_vc = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                    MutAnyOrigin,
                ](eval_mcts.visit_count.unsafe_ptr())
                var e_tv = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                    MutAnyOrigin,
                ](eval_mcts.total_value.unsafe_ptr())
                var e_pr = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                    MutAnyOrigin,
                ](eval_mcts.prior.unsafe_ptr())
                var e_rw = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                    MutAnyOrigin,
                ](eval_mcts.reward.unsafe_ptr())
                var e_ci = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES * ACT),
                    MutAnyOrigin,
                ](eval_mcts.child_idx.unsafe_ptr())
                var e_tvis = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES),
                    MutAnyOrigin,
                ](eval_mcts.total_visits.unsafe_ptr())
                var e_nc = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](eval_mcts.node_count.unsafe_ptr())
                var e_po = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * PRED_OUT),
                    MutAnyOrigin,
                ](eval_mcts.pred_output.unsafe_ptr())
                var e_miq = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](eval_mcts.min_q.unsafe_ptr())
                var e_mxq = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](eval_mcts.max_q.unsafe_ptr())
                var e_lm = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin
                ](eval_legal.unsafe_ptr())
                var e_hs = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs * MAX_NODES * LATENT),
                    MutAnyOrigin,
                ](eval_mcts.hidden_states.unsafe_ptr())

                # Bug D fix (2026-05-04): zero tree state before init_root.
                eval_mcts.zero_tree(ctx)
                comptime e_run_init = gpu_mcts_init_root_kernel[
                    Self.n_envs, MAX_NODES, ACT, LATENT, PRED_OUT, dtype
                ]
                ctx.enqueue_function[e_run_init, e_run_init](
                    e_vc,
                    e_tv,
                    e_pr,
                    e_rw,
                    e_ci,
                    e_tvis,
                    e_nc,
                    e_po,
                    e_miq,
                    e_mxq,
                    Scalar[dtype](0.0),  # No noise
                    Scalar[DType.uint32](UInt32(rng_offset + eval_move)),
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
                comptime e_run_mask = gpu_mcts_apply_legal_mask_kernel[
                    Self.n_envs, MAX_NODES, ACT, dtype
                ]
                ctx.enqueue_function[e_run_mask, e_run_mask](
                    e_pr,
                    e_lm,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Batched MCTS simulations
                var e_b_pp = LayoutTensor[
                    dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin
                ](eval_mcts.pending_parent.unsafe_ptr())
                var e_b_pa = LayoutTensor[
                    dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin
                ](eval_mcts.pending_action.unsafe_ptr())
                var e_b_sp = LayoutTensor[
                    dtype,
                    Layout.row_major(TOTAL_EXPAND * MAX_DEPTH),
                    MutAnyOrigin,
                ](eval_mcts.search_paths.unsafe_ptr())
                var e_b_ap = LayoutTensor[
                    dtype,
                    Layout.row_major(TOTAL_EXPAND * MAX_DEPTH),
                    MutAnyOrigin,
                ](eval_mcts.action_paths.unsafe_ptr())
                var e_b_pl = LayoutTensor[
                    dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin
                ](eval_mcts.path_lengths.unsafe_ptr())

                for _r in range(NUM_ROUNDS):
                    var e_b_dyn_in = LayoutTensor[
                        dtype,
                        Layout.row_major(TOTAL_EXPAND * DYN_IN),
                        MutAnyOrigin,
                    ](eval_mcts.dyn_input.unsafe_ptr())
                    comptime e_run_sel = gpu_mcts_batched_select_and_build_dyn_kernel[
                        Self.n_envs,
                        MAX_NODES,
                        ACT,
                        BATCH_SIMS,
                        LATENT,
                        DYN_IN,
                        dtype,
                    ]
                    ctx.enqueue_function[e_run_sel, e_run_sel](
                        e_vc,
                        e_tv,
                        e_pr,
                        e_ci,
                        e_tvis,
                        e_nc,
                        e_miq,
                        e_mxq,
                        e_hs,
                        e_b_dyn_in,
                        e_b_pp,
                        e_b_pa,
                        e_b_sp,
                        e_b_ap,
                        e_b_pl,
                        Scalar[dtype](Self.Config.PUCT.C_BASE),
                        Scalar[dtype](Self.Config.PUCT.C_INIT),
                        grid_dim=(ENV_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    var e_dyn_in_t = LayoutTensor[
                        dtype,
                        Layout.row_major(TOTAL_EXPAND, DYN_IN_DIM),
                        MutAnyOrigin,
                    ](eval_mcts.dyn_input.unsafe_ptr())
                    var e_dyn_out_t = LayoutTensor[
                        dtype,
                        Layout.row_major(TOTAL_EXPAND, DYN_OUT_DIM),
                        MutAnyOrigin,
                    ](eval_mcts.dyn_output.unsafe_ptr())
                    DynNet.forward_gpu[TOTAL_EXPAND](
                        ctx,
                        e_dyn_in_t,
                        e_dyn_out_t,
                        gpu.dynamics.params_view(),
                        gpu.dynamics.model_state_view(),
                        mcts_ws,
                    )

                    var e_pi_flat = LayoutTensor[
                        dtype,
                        Layout.row_major(TOTAL_EXPAND * LATENT),
                        MutAnyOrigin,
                    ](eval_mcts.pred_input.unsafe_ptr())
                    var e_do_flat = LayoutTensor[
                        dtype,
                        Layout.row_major(TOTAL_EXPAND * DYN_OUT),
                        MutAnyOrigin,
                    ](eval_mcts.dyn_output.unsafe_ptr())
                    comptime E_EXTR_BLK = (
                        TOTAL_EXPAND * LATENT + TPB - 1
                    ) // TPB
                    comptime e_run_extr = extract_hidden_kernel[
                        TOTAL_EXPAND, LATENT, DYN_OUT, dtype
                    ]
                    ctx.enqueue_function[e_run_extr, e_run_extr](
                        e_pi_flat,
                        e_do_flat,
                        grid_dim=(E_EXTR_BLK,),
                        block_dim=(TPB,),
                    )

                    var e_pred_in = LayoutTensor[
                        dtype,
                        Layout.row_major(TOTAL_EXPAND, PRED_IN_DIM),
                        MutAnyOrigin,
                    ](eval_mcts.pred_input.unsafe_ptr())
                    var e_pred_out = LayoutTensor[
                        dtype,
                        Layout.row_major(TOTAL_EXPAND, PRED_OUT_DIM),
                        MutAnyOrigin,
                    ](eval_mcts.pred_output.unsafe_ptr())
                    PredNet.forward_gpu[TOTAL_EXPAND](
                        ctx,
                        e_pred_in,
                        e_pred_out,
                        gpu.prediction.params_view(),
                        gpu.prediction.model_state_view(),
                        mcts_ws,
                    )

                    var e_b_dyn_out = LayoutTensor[
                        dtype,
                        Layout.row_major(TOTAL_EXPAND * DYN_OUT),
                        MutAnyOrigin,
                    ](eval_mcts.dyn_output.unsafe_ptr())
                    var e_b_pred_out = LayoutTensor[
                        dtype,
                        Layout.row_major(TOTAL_EXPAND * PRED_OUT),
                        MutAnyOrigin,
                    ](eval_mcts.pred_output.unsafe_ptr())
                    comptime e_run_exp = gpu_mcts_batched_expand_backup_muzero_kernel[
                        Self.n_envs,
                        MAX_NODES,
                        ACT,
                        BATCH_SIMS,
                        LATENT,
                        PRED_OUT,
                        DYN_OUT,
                        dtype,
                    ]
                    ctx.enqueue_function[e_run_exp, e_run_exp](
                        e_vc,
                        e_tv,
                        e_pr,
                        e_rw,
                        e_ci,
                        e_tvis,
                        e_nc,
                        e_miq,
                        e_mxq,
                        e_hs,
                        e_b_pp,
                        e_b_pa,
                        e_b_dyn_out,
                        e_b_pred_out,
                        e_b_sp,
                        e_b_ap,
                        e_b_pl,
                        Scalar[dtype](self.v_min),
                        Scalar[dtype](self.v_max),
                        Scalar[dtype](self.gamma),
                        Scalar[DType.bool](NEGATE),
                        grid_dim=(ENV_BLOCKS,),
                        block_dim=(TPB,),
                    )

                # Greedy action (temp=0)
                var e_act = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](eval_acts.unsafe_ptr())
                var e_pol = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs * ACT), MutAnyOrigin
                ](eval_mcts.pred_output.unsafe_ptr())
                comptime e_run_act = gpu_mcts_extract_actions_masked_kernel[
                    Self.n_envs, MAX_NODES, ACT, dtype
                ]
                ctx.enqueue_function[e_run_act, e_run_act](
                    e_vc,
                    e_lm,
                    e_act,
                    e_pol,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
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

        var decisive = new_wins + old_wins
        var accepted: Bool
        if decisive == 0:
            accepted = False
        else:
            var win_rate = Float64(new_wins) / Float64(decisive)
            accepted = win_rate >= threshold

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
