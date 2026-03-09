"""Deep Dueling DQN Agent using the new trait-based deep learning architecture.

This Dueling DQN implementation uses:
- NetworkState for heap-allocated params + grads + optimizer state
- Network (all-static) for stateless forward/backward ops via LayoutTensor
- Parallel combinator for Value/Advantage stream split
- ReplayBuffer from nn.replay for experience replay
- compile-time lr (Adam LR baked in at compile time)

Dueling Architecture (unified model with Parallel):
- Shared backbone: obs -> h1 (ReLU) -> h2 (ReLU)
- Parallel split:
  - Value stream: h2 -> stream_hidden (ReLU) -> V(s) [scalar]
  - Advantage stream: h2 -> stream_hidden (ReLU) -> A(s,a) [num_actions]
- Output: [V(s), A(s,a_1), ..., A(s,a_n)]
- Q(s,a) = V(s) + (A(s,a) - mean(A))

Features:
- Works with any BoxDiscreteActionEnv (continuous obs, discrete actions)
- Epsilon-greedy exploration with decay
- Target network with soft updates
- Double DQN support (online selects, target evaluates)

Usage:
    from deep_agents.dueling_dqn import DuelingDQNAgent
    from envs import LunarLanderEnv

    var env = LunarLanderEnv()
    var agent = DuelingDQNAgent[8, 4, 128, 64, 100000, 64]()

    var metrics = agent.train(env, num_episodes=500)

Reference: Wang et al. "Dueling Network Architectures for Deep RL" (2016)
"""

from std.math import exp
from std.random import random_float64, seed

from layout import Layout, LayoutTensor

from nn.constants import dtype, TILE, TPB
from nn.model import Linear, LinearReLU, Sequential, Parallel, Model
from nn.optimizer import Adam, Optimizer
from nn.initializer import Kaiming
from nn.training import Network, NetworkState
from deep_agents.core.replay import HeapReplayBuffer
from deep_agents.core import (
    fill_inline,
    obs_to_inline,
    OffPolicyDiscreteState,
    OffPolicyDiscreteAgent,
    run_offpolicy_discrete_train,
    run_offpolicy_discrete_eval,
    Checkpointable,
)
from core import TrainingMetrics, BoxDiscreteActionEnv, RenderableEnv


# =============================================================================
# DuelingDQNCPUState — CPU buffer container for Dueling DQN
# =============================================================================


struct DuelingDQNCPUState[
    DuelingModel: Model,
    Opt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    batch_size: Int,
](Movable, OffPolicyDiscreteState):
    """CPU-resident state for Dueling DQN training.

    Holds online/target network states and replay buffer.

    Parameters:
        DuelingModel: Unified dueling model type (backbone + Parallel[V, A]).
        Opt: Optimizer type.
        buffer_capacity: Replay buffer capacity.
        obs_dim: Observation space dimension.
        batch_size: Training batch size.
    """

    comptime BUFFER_DTYPE = dtype

    var online: NetworkState[Self.DuelingModel, Self.Opt]
    var target: NetworkState[Self.DuelingModel, Self.Opt]
    var buffer: HeapReplayBuffer[Self.buffer_capacity, Self.obs_dim, 1, dtype]

    fn __init__(out self):
        """Allocate and Kaiming-initialize online; copy online → target."""
        self.online = NetworkState[Self.DuelingModel, Self.Opt]()
        self.online.initialize[Kaiming]()
        self.target = NetworkState[Self.DuelingModel, Self.Opt](
            copy=self.online
        )
        self.buffer = HeapReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, dtype
        ]()

    fn __init__(out self, *, deinit take: Self):
        self.online = take.online^
        self.target = take.target^
        self.buffer = take.buffer^

    fn store[
        dtype: DType
    ](
        mut self,
        obs: List[Scalar[dtype]],
        action: Int,
        reward: Float64,
        next_obs: List[Scalar[dtype]],
        done: Bool,
    ) -> None:
        """Push one discrete transition into the replay buffer."""
        comptime BUFFER_DTYPE = Self.BUFFER_DTYPE
        var obs_arr = InlineArray[Scalar[BUFFER_DTYPE], Self.obs_dim](
            uninitialized=True
        )
        var next_arr = InlineArray[Scalar[BUFFER_DTYPE], Self.obs_dim](
            uninitialized=True
        )
        for i in range(Self.obs_dim):
            obs_arr[i] = Scalar[BUFFER_DTYPE](Float64(obs[i]))
            next_arr[i] = Scalar[BUFFER_DTYPE](Float64(next_obs[i]))
        var action_arr = InlineArray[Scalar[BUFFER_DTYPE], 1](
            fill=Scalar[BUFFER_DTYPE](action)
        )
        self.buffer.add(
            obs_arr,
            action_arr,
            Scalar[BUFFER_DTYPE](reward),
            next_arr,
            done,
        )

    fn is_ready(self) -> Bool:
        return self.buffer.is_ready[Self.batch_size]()


# =============================================================================
# Deep Dueling DQN Agent
# =============================================================================


struct DuelingDQNAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 128,
    stream_hidden_dim: Int = 64,
    buffer_capacity: Int = 100000,
    batch_size: Int = 64,
    double_dqn: Bool = True,
    lr: Float64 = 0.0005,
](OffPolicyDiscreteAgent & Checkpointable):
    """Deep Dueling DQN Agent using NetworkState architecture.

    Dueling DQN separates the Q-network into two streams:
    - Value stream V(s): Estimates how good is this state
    - Advantage stream A(s,a): Estimates relative action advantages

    Final Q-values: Q(s,a) = V(s) + (A(s,a) - mean(A))

    Parameters:
        obs_dim: Dimension of observation space.
        num_actions: Number of discrete actions.
        hidden_dim: Hidden layer size for shared backbone.
        stream_hidden_dim: Hidden layer size for value/advantage streams.
        buffer_capacity: Replay buffer capacity.
        batch_size: Training batch size.
        double_dqn: If True, use Double DQN target computation.
        lr: Adam learning rate — compile-time (default: 0.0005).
    """

    # Convenience aliases
    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.num_actions
    comptime HIDDEN = Self.hidden_dim
    comptime STREAM_H = Self.stream_hidden_dim
    comptime BATCH = Self.batch_size

    # =========================================================================
    # Unified Dueling Model
    # =========================================================================

    # Backbone + Parallel[Value stream, Advantage stream]
    # Output: [V(s), A(s,a_1), ..., A(s,a_n)] per row
    comptime DuelingModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Sequential[LinearReLU[Self.HIDDEN, Self.STREAM_H], Linear[Self.STREAM_H, 1]],
            Sequential[LinearReLU[Self.HIDDEN, Self.STREAM_H], Linear[Self.STREAM_H, Self.ACTIONS]],
        ],
    ]
    comptime DUELING_OUT = Self.DuelingModel.OUT_DIM  # = 1 + ACTIONS
    comptime DuelingNet = Network[Self.DuelingModel, Adam[Self.lr]]

    # CPU state type (online + target + replay buffer)
    comptime CPUStateType = DuelingDQNCPUState[
        Self.DuelingModel,
        Adam[Self.lr],
        Self.buffer_capacity,
        Self.obs_dim,
        Self.batch_size,
    ]

    # CPU state: persistent for evaluate() and checkpointing
    var state: Self.CPUStateType

    # Hyperparameters
    var gamma: Float64
    var tau: Float64
    var epsilon: Float64
    var epsilon_min: Float64
    var epsilon_decay: Float64

    # Training state
    var total_steps: Int
    var train_step_count: Int

    # Auto-checkpoint settings
    var checkpoint_every: Int
    var checkpoint_path: String

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        epsilon: Float64 = 1.0,
        epsilon_min: Float64 = 0.01,
        epsilon_decay: Float64 = 0.995,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ):
        self.state = Self.CPUStateType()
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0
        self.train_step_count = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

    # =========================================================================
    # Dueling Forward Helper
    # =========================================================================

    fn _dueling_forward_inline[
        BATCH_N: Int
    ](
        self,
        cpu_state: Self.CPUStateType,
        obs: InlineArray[Scalar[dtype], BATCH_N * Self.OBS],
        mut q_values: InlineArray[Scalar[dtype], BATCH_N * Self.ACTIONS],
        use_target: Bool = False,
    ):
        """Forward pass: Q(s,a) = V(s) + (A(s,a) - mean(A)).

        Single forward through unified model, then post-process output.
        Output layout from Parallel: [V(s), A(s,a_1), ..., A(s,a_n)] per row.
        """
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_N, Self.OBS), MutAnyOrigin
        ](obs.unsafe_ptr())
        var out_arr = InlineArray[Scalar[dtype], BATCH_N * Self.DUELING_OUT](
            uninitialized=True
        )
        var out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_N, Self.DUELING_OUT), MutAnyOrigin
        ](out_arr.unsafe_ptr())

        if use_target:
            var p = cpu_state.target.params_view()
            Self.DuelingNet.forward[BATCH_N](obs_t, out_t, p)
        else:
            var p = cpu_state.online.params_view()
            Self.DuelingNet.forward[BATCH_N](obs_t, out_t, p)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A))
        for b in range(BATCH_N):
            var v_s = out_arr[b * Self.DUELING_OUT]  # First element: V(s)
            var mean_adv: Scalar[dtype] = 0.0
            for a in range(Self.ACTIONS):
                mean_adv += out_arr[b * Self.DUELING_OUT + 1 + a]
            mean_adv /= Scalar[dtype](Self.ACTIONS)

            for a in range(Self.ACTIONS):
                var adv = out_arr[b * Self.DUELING_OUT + 1 + a]
                q_values[b * Self.ACTIONS + a] = v_s + (adv - mean_adv)

    # =========================================================================
    # Action Selection
    # =========================================================================

    fn _select_action_inline(
        self,
        cpu_state: Self.CPUStateType,
        obs: InlineArray[Scalar[dtype], Self.OBS],
        greedy: Bool = False,
    ) -> Int:
        if not greedy and random_float64() < self.epsilon:
            return Int(random_float64() * Float64(Self.ACTIONS)) % Self.ACTIONS

        var q_arr = InlineArray[Scalar[dtype], Self.ACTIONS](uninitialized=True)
        var obs_batch = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)
        for i in range(Self.OBS):
            obs_batch[i] = obs[i]
        self._dueling_forward_inline[1](
            cpu_state, obs_batch, q_arr, use_target=False
        )

        var best_action = 0
        var best_q = q_arr[0]
        for a in range(1, Self.ACTIONS):
            if q_arr[a] > best_q:
                best_q = q_arr[a]
                best_action = a
        return best_action

    # =========================================================================
    # CPU Training Step
    # =========================================================================

    fn do_cpu_train_step(mut self, mut cpu_state: Self.CPUStateType) -> Float64:
        if not cpu_state.buffer.is_ready[Self.BATCH]():
            return 0.0

        # --- Phase 1: Sample batch ---
        var batch_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var batch_act1 = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var batch_rewards = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var batch_next_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var batch_dones = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )

        var batch_act_flat = InlineArray[Scalar[dtype], Self.BATCH * 1](
            uninitialized=True
        )
        cpu_state.buffer.sample[Self.BATCH](
            batch_obs,
            batch_act_flat,
            batch_rewards,
            batch_next_obs,
            batch_dones,
        )
        for i in range(Self.BATCH):
            batch_act1[i] = batch_act_flat[i]

        # --- Phase 2: Compute TD targets ---
        var max_next_q = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )

        comptime if Self.double_dqn:
            var online_next_q = InlineArray[
                Scalar[dtype], Self.BATCH * Self.ACTIONS
            ](uninitialized=True)
            var target_next_q = InlineArray[
                Scalar[dtype], Self.BATCH * Self.ACTIONS
            ](uninitialized=True)
            self._dueling_forward_inline[Self.BATCH](
                cpu_state, batch_next_obs, online_next_q, use_target=False
            )
            self._dueling_forward_inline[Self.BATCH](
                cpu_state, batch_next_obs, target_next_q, use_target=True
            )
            for b in range(Self.BATCH):
                var best_action = 0
                var best_online_q = online_next_q[b * Self.ACTIONS]
                for a in range(1, Self.ACTIONS):
                    var q = online_next_q[b * Self.ACTIONS + a]
                    if q > best_online_q:
                        best_online_q = q
                        best_action = a
                max_next_q[b] = target_next_q[b * Self.ACTIONS + best_action]
        else:
            var next_q = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
                uninitialized=True
            )
            self._dueling_forward_inline[Self.BATCH](
                cpu_state, batch_next_obs, next_q, use_target=True
            )
            for b in range(Self.BATCH):
                var best_q = next_q[b * Self.ACTIONS]
                for a in range(1, Self.ACTIONS):
                    var q = next_q[b * Self.ACTIONS + a]
                    if q > best_q:
                        best_q = q
                max_next_q[b] = best_q

        var targets = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        for b in range(Self.BATCH):
            var done_mask = Scalar[dtype](1.0) - batch_dones[b]
            targets[b] = (
                batch_rewards[b]
                + Scalar[dtype](self.gamma) * max_next_q[b] * done_mask
            )

        # --- Phase 3: Forward with cache (online) ---
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](batch_obs.unsafe_ptr())

        var out_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.DUELING_OUT
        ](uninitialized=True)
        var out_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.DUELING_OUT), MutAnyOrigin
        ](out_arr.unsafe_ptr())
        var cache_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.DuelingModel.CACHE_SIZE
        ](uninitialized=True)
        var cache_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.DuelingModel.CACHE_SIZE),
            MutAnyOrigin,
        ](cache_arr.unsafe_ptr())
        var p = cpu_state.online.params_view()
        Self.DuelingNet.forward_with_cache[Self.BATCH](
            obs_t, out_t, p, cache_t
        )

        # Compute Q-values: Q(s,a) = V(s) + (A(s,a) - mean(A))
        var q_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        for b in range(Self.BATCH):
            var v_s = out_arr[b * Self.DUELING_OUT]
            var mean_adv: Scalar[dtype] = 0.0
            for a in range(Self.ACTIONS):
                mean_adv += out_arr[b * Self.DUELING_OUT + 1 + a]
            mean_adv /= Scalar[dtype](Self.ACTIONS)
            for a in range(Self.ACTIONS):
                var idx = b * Self.ACTIONS + a
                q_arr[idx] = v_s + (out_arr[b * Self.DUELING_OUT + 1 + a] - mean_adv)

        # --- Phase 4: Compute loss and gradients ---
        var loss: Float64 = 0.0
        var dq_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            fill=Scalar[dtype](0.0)
        )

        for b in range(Self.BATCH):
            var action_idx = Int(batch_act1[b])
            var q_idx = b * Self.ACTIONS + action_idx
            var td_error = q_arr[q_idx] - targets[b]
            loss += Float64(td_error * td_error)
            dq_arr[q_idx] = (
                Scalar[dtype](2.0) * td_error / Scalar[dtype](Self.BATCH)
            )
        loss /= Float64(Self.BATCH)

        # --- Phase 5: Backward through unified dueling network ---
        # Convert dQ gradients to dueling output gradients:
        # dV = sum(dQ_j), dA_i = dQ_i - (1/n)*sum(dQ_j)
        var dout_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.DUELING_OUT
        ](fill=Scalar[dtype](0.0))
        var one_over_n = Scalar[dtype](1.0) / Scalar[dtype](Self.ACTIONS)

        for b in range(Self.BATCH):
            var sum_dq: Scalar[dtype] = 0.0
            for a in range(Self.ACTIONS):
                sum_dq += dq_arr[b * Self.ACTIONS + a]
            # dV = sum(dQ)
            dout_arr[b * Self.DUELING_OUT] = sum_dq
            # dA_i = dQ_i - (1/n)*sum(dQ)
            for a in range(Self.ACTIONS):
                dout_arr[b * Self.DUELING_OUT + 1 + a] = (
                    dq_arr[b * Self.ACTIONS + a] - one_over_n * sum_dq
                )

        var dout_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.DUELING_OUT), MutAnyOrigin
        ](dout_arr.unsafe_ptr())
        var dobs_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var dobs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](dobs_arr.unsafe_ptr())

        var g = cpu_state.online.grads_view()
        cpu_state.online.zero_grads()
        Self.DuelingNet.backward[Self.BATCH](
            dout_t, dobs_t, p, cache_t, g
        )
        cpu_state.online.optimizer_step()

        # --- Phase 6: Soft update target network ---
        cpu_state.target.soft_update_from(cpu_state.online, self.tau)

        self.train_step_count += 1

        return loss

    fn decay_epsilon(mut self):
        """Decay exploration rate (call once per episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        return self.epsilon

    fn get_train_steps(self) -> Int:
        return self.train_step_count

    # =========================================================================
    # OffPolicyDiscreteAgent trait conformance
    # =========================================================================

    fn make_cpu_state(self) -> Self.CPUStateType:
        return Self.CPUStateType()

    fn select_action[
        dt: DType
    ](mut self, mut cpu_state: Self.CPUStateType, obs: List[Scalar[dt]]) -> Int:
        var obs_inline = obs_to_inline[Self.OBS, dt](obs)
        return self._select_action_inline(cpu_state, obs_inline, greedy=False)

    fn store_transition[
        dt: DType
    ](
        mut self,
        mut cpu_state: Self.CPUStateType,
        obs: List[Scalar[dt]],
        action: Int,
        reward: Float64,
        next_obs: List[Scalar[dt]],
        done: Bool,
    ) -> None:
        cpu_state.store[dt](obs, action, reward, next_obs, done)

    fn decay_explore(mut self) -> None:
        self.decay_epsilon()

    fn get_explore_rate(self) -> Float64:
        return self.epsilon

    fn random_action(self) -> Int:
        return Int(random_float64() * Float64(Self.ACTIONS)) % Self.ACTIONS

    fn select_greedy_action(
        self, cpu_state: Self.CPUStateType, obs: List[Float64]
    ) -> Int:
        var obs_inline = obs_to_inline[Self.OBS, DType.float64](obs)
        return self._select_action_inline(cpu_state, obs_inline, greedy=True)

    # =========================================================================
    # High-level CPU Training and Evaluation
    # =========================================================================

    fn train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 1000,
        warmup_steps: Int = 1000,
        train_every: Int = 1,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) raises -> TrainingMetrics:
        var cpu_state = Self.CPUStateType()
        var ckpt_every = self.checkpoint_every
        var ckpt_path = self.checkpoint_path
        var metrics = run_offpolicy_discrete_train(
            self,
            cpu_state,
            env,
            num_episodes,
            max_steps_per_episode,
            warmup_steps,
            train_every,
            ckpt_every,
            ckpt_path,
            verbose,
            print_every,
            environment_name,
            "Deep Dueling DQN",
        )
        self.state = cpu_state^
        return metrics

    fn evaluate[
        E: BoxDiscreteActionEnv & RenderableEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps: Int = 1000,
        verbose: Bool = False,
        render: Bool = False,
        frame_delay_ms: Int = 16,
        greedy: Bool = True,
    ) raises -> Float64:
        var metrics = run_offpolicy_discrete_eval(
            self,
            self.state,
            env,
            num_episodes=num_episodes,
            max_steps=max_steps,
            verbose=verbose,
            render=render,
            frame_delay_ms=frame_delay_ms,
        )
        return metrics.mean_reward()

    # =========================================================================
    # Checkpoint Save / Load
    # =========================================================================

    fn save_checkpoint(self, filepath: String) raises:
        from nn.checkpoint import (
            write_checkpoint_header,
            write_metadata_section,
            save_checkpoint_file,
        )

        var param_size = Self.DuelingModel.PARAM_SIZE
        var content = write_checkpoint_header(
            "dueling_dqn", param_size, param_size
        )
        content += self.state.online.write_sections("online_")
        content += self.state.target.write_sections("target_")
        var metadata = List[String]()
        metadata.append("epsilon=" + String(self.epsilon))
        metadata.append("total_steps=" + String(self.total_steps))
        content += write_metadata_section(metadata)
        save_checkpoint_file(filepath, content)

    fn load_checkpoint(mut self, filepath: String) raises:
        from nn.checkpoint import (
            read_checkpoint_file,
            read_metadata_section,
            get_metadata_value,
        )

        var content = read_checkpoint_file(filepath)
        self.state.online.read_sections(content, "online_")
        self.state.target.read_sections(content, "target_")
        var metadata = read_metadata_section(content)
        var eps_str = get_metadata_value(metadata, "epsilon")
        if len(eps_str) > 0:
            self.epsilon = Float64(atof(eps_str))
