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
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, LinearMish, Sequential, Parallel
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.training import Network, NetworkState
from mojo_rl.deep_agents.core.utils import (
    print_progress_bar,
    clear_progress_bar,
)
from mojo_rl.core import (
    TrainingMetrics,
    BoxDiscreteActionEnv,
)
from mojo_rl.deep_agents.core.replay.sequence_replay_buffer import (
    SequenceReplayBuffer,
)
from .state import MuZeroCPUState
from .mcts import MCTS, MCTSNode
from .utils import (
    scalar_transform,
    inverse_scalar_transform,
    encode_categorical,
    cross_entropy_with_softmax,
    softmax_inplace,
)


# =============================================================================
# MuZero Agent
# =============================================================================


struct MuZeroAgent[
    obs_dim: Int,
    action_dim: Int,
    latent_dim: Int = 256,
    hidden_dim: Int = 256,
    num_bins: Int = 101,
    num_simulations: Int = 50,
    unroll_steps: Int = 5,
    td_steps: Int = 10,
    batch_size: Int = 128,
    buffer_capacity: Int = 100000,
    lr: Float64 = 1e-3,
](Movable):
    """MuZero agent for discrete action environments.

    Combines learned representation/dynamics/prediction networks with
    MCTS planning for action selection. Trains via K-step unrolled
    cross-entropy losses on policy, value, and reward targets.

    Parameters:
        obs_dim: Observation space dimension.
        action_dim: Number of discrete actions.
        latent_dim: Hidden state dimension (default: 256).
        hidden_dim: Network hidden width (default: 256).
        num_bins: Categorical bins for value/reward (default: 101).
        num_simulations: MCTS simulations per move (default: 50).
        unroll_steps: K-step unroll depth for training (default: 5).
        td_steps: N-step return horizon (default: 10).
        batch_size: Training batch size (default: 128).
        buffer_capacity: Replay buffer capacity (default: 100K).
        lr: Learning rate for all networks (default: 1e-3).
    """

    # ── Derived compile-time constants ────────────────────────────────────
    comptime K: Int = Self.unroll_steps
    comptime N: Int = Self.td_steps

    # ── State type alias ──────────────────────────────────────────────────
    comptime StateType = MuZeroCPUState[
        Self.obs_dim,
        Self.action_dim,
        Self.latent_dim,
        Self.hidden_dim,
        Self.num_bins,
        LR=Self.lr,
        BUFFER_CAPACITY=Self.buffer_capacity,
        BATCH_SIZE=Self.batch_size,
        UNROLL_STEPS=Self.unroll_steps,
        TD_STEPS=Self.td_steps,
    ]

    # ── Network type shortcuts (from state) ───────────────────────────────
    comptime RepNet = Network[Self.StateType.RepModel, Self.StateType.OptType]
    comptime DynNet = Network[Self.StateType.DynModel, Self.StateType.OptType]
    comptime PredNet = Network[Self.StateType.PredModel, Self.StateType.OptType]

    # ── Core state ────────────────────────────────────────────────────────
    var state: Self.StateType

    # MCTS search engine
    var mcts: MCTS[Self.action_dim, Self.latent_dim, Self.num_bins, Self.num_simulations]

    # Hyperparameters
    var gamma: Float64
    var weight_decay: Float64
    var v_min: Float64
    var v_max: Float64
    var temperature: Float64
    var temperature_decay_steps: Int

    # Step counters
    var total_steps: Int
    var train_step_count: Int
    var warmup_steps: Int

    # Episode data storage for MCTS targets
    var _episode_obs: List[List[Scalar[dtype]]]
    var _episode_actions: List[Int]
    var _episode_rewards: List[Float64]
    var _episode_policies: List[InlineArray[Float64, Self.action_dim]]
    var _episode_values: List[Float64]

    # ══════════════════════════════════════════════════════════════════════
    # Constructors
    # ══════════════════════════════════════════════════════════════════════

    fn __init__(
        out self,
        gamma: Float64 = 0.997,
        weight_decay: Float64 = 1e-4,
        v_min: Float64 = -50.0,
        v_max: Float64 = 50.0,
        temperature: Float64 = 1.0,
        temperature_decay_steps: Int = 100000,
        warmup_steps: Int = 1000,
    ):
        """Initialize MuZero agent with all networks and MCTS engine.

        Args:
            gamma: Discount factor (default: 0.997).
            weight_decay: L2 regularization coefficient (default: 1e-4).
            v_min: Minimum value support for categorical encoding.
            v_max: Maximum value support for categorical encoding.
            temperature: Initial action selection temperature (default: 1.0).
            temperature_decay_steps: Steps to decay temperature to 0.
            warmup_steps: Random exploration steps before training.
        """
        self.state = Self.StateType()
        self.mcts = MCTS[
            Self.action_dim, Self.latent_dim, Self.num_bins, Self.num_simulations
        ](gamma=gamma)
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.v_min = v_min
        self.v_max = v_max
        self.temperature = temperature
        self.temperature_decay_steps = temperature_decay_steps
        self.total_steps = 0
        self.train_step_count = 0
        self.warmup_steps = warmup_steps
        self._episode_obs = List[List[Scalar[dtype]]]()
        self._episode_actions = List[Int]()
        self._episode_rewards = List[Float64]()
        self._episode_policies = List[InlineArray[Float64, Self.action_dim]]()
        self._episode_values = List[Float64]()

    fn __init__(out self, *, deinit take: Self):
        """Move constructor — transfer ownership of all fields."""
        self.state = take.state^
        self.mcts = take.mcts^
        self.gamma = take.gamma
        self.weight_decay = take.weight_decay
        self.v_min = take.v_min
        self.v_max = take.v_max
        self.temperature = take.temperature
        self.temperature_decay_steps = take.temperature_decay_steps
        self.total_steps = take.total_steps
        self.train_step_count = take.train_step_count
        self.warmup_steps = take.warmup_steps
        self._episode_obs = take._episode_obs^
        self._episode_actions = take._episode_actions^
        self._episode_rewards = take._episode_rewards^
        self._episode_policies = take._episode_policies^
        self._episode_values = take._episode_values^

    # ══════════════════════════════════════════════════════════════════════
    # Episode Management
    # ══════════════════════════════════════════════════════════════════════

    fn reset_episode(mut self):
        """Reset episode buffers for a new episode."""
        self._episode_obs.clear()
        self._episode_actions.clear()
        self._episode_rewards.clear()
        self._episode_policies.clear()
        self._episode_values.clear()

    fn store_transition(
        mut self,
        obs: List[Scalar[dtype]],
        action: Int,
        reward: Float64,
        policy: InlineArray[Float64, Self.action_dim],
        value: Float64,
        done: Bool,
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
        """
        self._episode_obs.append(obs.copy())
        self._episode_actions.append(action)
        self._episode_rewards.append(reward)
        self._episode_policies.append(policy)
        self._episode_values.append(value)

        if done:
            self._flush_episode()

    fn _flush_episode(mut self):
        """Flush episode data to the replay buffer with MCTS targets.

        Stores obs/action/reward/done to the SequenceReplayBuffer and
        MCTS policies/values to the parallel target arrays.
        """
        var ep_len = len(self._episode_obs)

        for t in range(ep_len):
            var obs_arr = InlineArray[Scalar[DType.float32], Self.obs_dim](
                uninitialized=True
            )
            for i in range(Self.obs_dim):
                if i < len(self._episode_obs[t]):
                    obs_arr[i] = Scalar[DType.float32](self._episode_obs[t][i])
                else:
                    obs_arr[i] = Scalar[DType.float32](0.0)

            # Action as one-hot for SequenceReplayBuffer (uses ACTION_DIM float array)
            var act_arr = InlineArray[Scalar[DType.float32], Self.action_dim](
                uninitialized=True
            )
            for i in range(Self.action_dim):
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
            var buf_idx = (self.state.buffer.ptr - 1 + Self.buffer_capacity) % Self.buffer_capacity
            for a in range(Self.action_dim):
                self.state.mcts_policies[buf_idx * Self.action_dim + a] = Scalar[
                    dtype
                ](self._episode_policies[t][a])
            self.state.mcts_values[buf_idx] = Scalar[dtype](
                self._episode_values[t]
            )

        self.reset_episode()

    # ══════════════════════════════════════════════════════════════════════
    # Action Selection
    # ══════════════════════════════════════════════════════════════════════

    fn select_action(
        mut self,
        obs: List[Scalar[dtype]],
        training: Bool = True,
    ) -> Tuple[Int, InlineArray[Float64, Self.action_dim], Float64]:
        """Select an action using MCTS with the learned model.

        1. Run MCTS from the current observation
        2. Get visit count policy
        3. Sample action from policy with temperature (or argmax if not training)

        Args:
            obs: Current observation [obs_dim].
            training: If True, sample with temperature; if False, argmax.

        Returns:
            Tuple of (action_index, mcts_policy, root_value).
        """
        # Run MCTS search
        var policy = self.mcts.search[
            Self.StateType.RepModel,
            Self.StateType.DynModel,
            Self.StateType.PredModel,
            Self.StateType.OptType,
            Self.StateType.OptType,
            Self.StateType.OptType,
        ](
            obs,
            self.state.representation,
            self.state.dynamics,
            self.state.prediction,
            self.v_min,
            self.v_max,
            add_noise=training,
        )

        # Get root value from MCTS (value at root after search)
        var root_value = Float64(0.0)
        if len(self.mcts.nodes) > 0:
            var root = self.mcts.nodes[0]
            for a in range(Self.action_dim):
                if root.visit_count[a] > 0:
                    root_value += policy[a] * root.mean_value(a)

        # Sample action with temperature
        var action: Int
        if not training or self.temperature < 0.01:
            # Argmax
            action = 0
            var best_prob = policy[0]
            for a in range(1, Self.action_dim):
                if policy[a] > best_prob:
                    best_prob = policy[a]
                    action = a
        else:
            # Temperature sampling: pi(a) = N(a)^(1/T) / sum_b N(b)^(1/T)
            var temp_policy = InlineArray[Float64, Self.action_dim](
                uninitialized=True
            )
            var inv_temp = 1.0 / self.temperature
            var sum_p = Float64(0.0)
            for a in range(Self.action_dim):
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
                for a in range(Self.action_dim):
                    temp_policy[a] /= sum_p
            else:
                for a in range(Self.action_dim):
                    temp_policy[a] = 1.0 / Float64(Self.action_dim)

            # Multinomial sample
            var u = random_float64(0.0, 1.0)
            var cumsum = Float64(0.0)
            action = Self.action_dim - 1
            for a in range(Self.action_dim):
                cumsum += temp_policy[a]
                if u <= cumsum:
                    action = a
                    break

        return (action, policy, root_value)

    # ══════════════════════════════════════════════════════════════════════
    # Training (K-Step Unrolled Forward/Backward)
    # ══════════════════════════════════════════════════════════════════════

    fn update(mut self) -> Float64:
        """Run one training step with K-step unrolled forward/backward.

        1. Sample batch of positions from replay buffer
        2. Compute n-step value targets and reward targets
        3. Forward: h(obs) -> s^0, then K steps of f(s^k) and g(s^k, a^k)
        4. Backward: propagate gradients through unrolled chain
        5. Optimizer step on all three networks

        Returns:
            Total training loss.
        """
        comptime BATCH = Self.batch_size
        comptime K = Self.K
        comptime LATENT = Self.latent_dim
        comptime ACT = Self.action_dim
        comptime BINS = Self.num_bins
        comptime OBS = Self.obs_dim

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

        # ── Step 5: Optimizer step ───────────────────────────────────
        self.state.representation.optimizer_step()
        self.state.dynamics.optimizer_step()
        self.state.prediction.optimizer_step()

        self.train_step_count += 1
        return total_loss

    fn _forward_and_compute_loss(mut self) -> Float64:
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
        comptime BATCH = Self.batch_size
        comptime K = Self.K
        comptime LATENT = Self.latent_dim
        comptime ACT = Self.action_dim
        comptime BINS = Self.num_bins
        comptime OBS = Self.obs_dim
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
            obs_t, h0_t, self.state.representation.params_view(), rep_cache_t
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
                hk_t, pred_t, self.state.prediction.params_view(), pred_cache_t
            )

            # Compute policy and value loss for this step
            for b in range(BATCH):
                # Policy loss: CE(predicted_logits, mcts_policy)
                var policy_logits = alloc[Float64](ACT)
                var policy_target = alloc[Float64](ACT)
                var pol_base = b * (K + 1) * ACT + k * ACT
                for a in range(ACT):
                    policy_logits[a] = Float64(
                        (self.state._pred_outputs + pred_offset + b * PRED_OUT + a)[]
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
                            self.state._hidden_states + hk_offset + b * LATENT + i
                        )[]
                    # Copy one-hot action from batch_actions
                    for a in range(ACT):
                        dyn_input_ptr[b * DYN_IN + LATENT + a] = (
                            self.state._batch_actions + b * K * ACT + k * ACT + a
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
                    dyn_cache_t,
                )

                # Extract next hidden state and reward logits
                for b in range(BATCH):
                    for i in range(LATENT):
                        (self.state._hidden_states + next_offset + b * LATENT + i)[] = (
                            dyn_out_ptr + b * DYN_OUT + i
                        )[]
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

    fn _backward(mut self):
        """K-step unrolled backward pass through all three networks.

        Gradients flow: prediction -> dynamics (xK) -> representation.
        Dynamics gradients are scaled by 1/K to prevent gradient explosion.
        """
        comptime BATCH = Self.batch_size
        comptime K = Self.K
        comptime LATENT = Self.latent_dim
        comptime ACT = Self.action_dim
        comptime BINS = Self.num_bins
        comptime OBS = Self.obs_dim
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
                var policy_logits = alloc[Float64](ACT)
                var pol_base = b * (K + 1) * ACT + k * ACT
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
                var target_val = Float64(self.state._value_targets[k * BATCH + b])
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
                        self.state._grad_dyn_out[b * DYN_OUT + i] = (
                            self.state._grad_hidden[b * LATENT + i]
                            * Scalar[dtype](0.5)
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
                    dyn_cache_t,
                    dyn_grads,
                )

                # Extract hidden state gradient from dynamics input gradient
                # (first LATENT elements) -> becomes the new grad_hidden for step k-1
                memset(self.state._grad_hidden, 0, BATCH * LATENT)
                for b in range(BATCH):
                    for i in range(LATENT):
                        self.state._grad_hidden[b * LATENT + i] = (
                            self.state._grad_dyn_in[b * DYN_IN + i]
                        )

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
            rep_cache_t,
            rep_grads,
        )

    fn _scale_batch_hidden(mut self, step: Int):
        """Min-max scale hidden states to [0, 1] for a given unroll step.

        Args:
            step: Unroll step index (0 = from representation, 1..K from dynamics).
        """
        comptime BATCH = Self.batch_size
        comptime LATENT = Self.latent_dim

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
                    var v = Float64((self.state._hidden_states + b_offset + i)[])
                    (self.state._hidden_states + b_offset + i)[] = Scalar[dtype](
                        (v - min_val) / delta
                    )

    # ══════════════════════════════════════════════════════════════════════
    # Main Training Loop
    # ══════════════════════════════════════════════════════════════════════

    fn train[E: BoxDiscreteActionEnv](
        mut self,
        mut env: E,
        total_timesteps: Int = 500000,
        train_every: Int = 1,
        seed_episodes: Int = 10,
        print_every: Int = 10,
    ) -> TrainingMetrics:
        """Train MuZero on a discrete action environment.

        Alternates between self-play (with MCTS) and training from replay.

        Args:
            env: Environment implementing BoxDiscreteActionEnv.
            total_timesteps: Total environment steps (default: 500K).
            train_every: Steps between training updates (default: 1).
            seed_episodes: Random exploration episodes (default: 10).
            print_every: Episodes between progress prints (default: 10).

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
                var action = Int(random_float64(0.0, Float64(Self.action_dim)))
                if action >= Self.action_dim:
                    action = Self.action_dim - 1

                var result = env.step_obs(action)
                var reward = Float64(result[1])
                done = result[2]

                # Store with uniform policy (random exploration)
                var uniform_policy = InlineArray[Float64, Self.action_dim](
                    uninitialized=True
                )
                for a in range(Self.action_dim):
                    uniform_policy[a] = 1.0 / Float64(Self.action_dim)

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
            var policy: InlineArray[Float64, Self.action_dim]
            var root_value: Float64

            if total_env_steps < self.warmup_steps:
                action = Int(random_float64(0.0, Float64(Self.action_dim)))
                if action >= Self.action_dim:
                    action = Self.action_dim - 1
                policy = InlineArray[Float64, Self.action_dim](
                    uninitialized=True
                )
                for a in range(Self.action_dim):
                    policy[a] = 1.0 / Float64(Self.action_dim)
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
                _ = self.update()

            # Decay temperature
            if self.temperature_decay_steps > 0:
                self.temperature = 1.0 - Float64(self.total_steps) / Float64(
                    self.temperature_decay_steps
                )
                if self.temperature < 0.01:
                    self.temperature = 0.01

            # Progress bar
            if step % 100 == 0:
                print_progress_bar(
                    step,
                    total_timesteps,
                    self.train_step_count,
                    "MuZero",
                )

        return metrics


# =============================================================================
# Helpers
# =============================================================================


fn _to_dtype_list[D: DType](obs: List[Scalar[D]]) -> List[Scalar[dtype]]:
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
