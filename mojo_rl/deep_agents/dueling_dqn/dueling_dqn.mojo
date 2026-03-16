"""Deep Dueling DQN Agent using the new trait-based deep learning architecture.

This Dueling DQN implementation uses:
- NetworkState for heap-allocated params + grads + optimizer state
- Network (all-static) for stateless forward/backward ops via LayoutTensor
- Parallel combinator for Value/Advantage stream split
- ReplayBuffer from mojo_rl.nn.replay for experience replay
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
- GPU support via GPUOffPolicyAgent trait + run_offpolicy_discrete_train_gpu

Usage:
    from mojo_rl.deep_agents.dueling_dqn import DuelingDQNAgent
    from mojo_rl.envs import LunarLanderEnv

    var env = LunarLanderEnv()
    var agent = DuelingDQNAgent[8, 4, 128, 64, 100000, 64]()

    # CPU Training
    var metrics = agent.train(env, num_episodes=500)

    # GPU Training (step-based, using shared off-policy GPU loop)
    var ctx = DeviceContext()
    var metrics_gpu = agent.train_gpu[LunarLanderGPUEnv](ctx, num_steps=100000)

Reference: Wang et al. "Dueling Network Architectures for Deep RL" (2016)
"""

from std.math import exp
from std.random import random_float64, seed

from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TILE, TPB
from mojo_rl.nn.model import Linear, LinearReLU, Sequential, Parallel, Model
from mojo_rl.nn.optimizer import Adam, Optimizer
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.training import Network, NetworkState, GPUNetworkState
from mojo_rl.deep_agents.core.replay import HeapReplayBuffer, GPUReplayBuffer
from mojo_rl.deep_agents.core import (
    fill_inline,
    obs_to_inline,
    OffPolicyDiscreteState,
    OffPolicyDiscreteAgent,
    GPUOffPolicyState,
    GPUOffPolicyAgent,
    run_offpolicy_discrete_train,
    run_offpolicy_discrete_eval,
    run_offpolicy_discrete_train_gpu,
    Checkpointable,
)
from mojo_rl.deep_agents.core.perf_timer import PerfTimer
from mojo_rl.nn.model.model import PerfTimerPtr
from mojo_rl.core import (
    TrainingMetrics,
    BoxDiscreteActionEnv,
    GPUDiscreteEnv,
    RenderableEnv,
)
from mojo_rl.core.logger import Logger, NoOpLogger
from .state import DuelingDQNGPUState
from .kernels import (
    dueling_combine_kernel,
    dueling_grad_kernel,
    dqn_td_target_kernel,
    dqn_double_td_target_kernel,
)


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
        self.online.initialize[Kaiming[]]()
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
    n_envs: Int = 1024,
    double_dqn: Bool = True,
    lr: Float64 = 0.0005,
    profile: Int = 0,
    L: Logger = NoOpLogger,
](OffPolicyDiscreteAgent & GPUOffPolicyAgent & Checkpointable):
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
        n_envs: Number of parallel environments for GPU training (default: 1024).
        double_dqn: If True, use Double DQN target computation.
        lr: Adam learning rate — compile-time (default: 0.0005).
        profile: Level of profiling (0: none, 1: L2, 2: L3, 3: L4).
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
            Sequential[
                LinearReLU[Self.HIDDEN, Self.STREAM_H], Linear[Self.STREAM_H, 1]
            ],
            Sequential[
                LinearReLU[Self.HIDDEN, Self.STREAM_H],
                Linear[Self.STREAM_H, Self.ACTIONS],
            ],
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

    # GPUOffPolicyAgent required compile-time constants
    comptime OBS_DIM: Int = Self.OBS
    comptime ACTION_DIM: Int = 1  # discrete action stored as float scalar index
    comptime BUFFER_CAPACITY: Int = Self.buffer_capacity
    comptime MAX_N_ENVS: Int = Self.n_envs
    comptime GPUStateType = DuelingDQNGPUState[
        Self.DuelingModel,
        Adam[Self.lr],
        Self.buffer_capacity,
        Self.obs_dim,
        Self.num_actions,
        Self.DUELING_OUT,
        Self.batch_size,
        Self.n_envs,
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

    # Level-2 profiler: sub-phases of do_gpu_train_step
    var train_timer: PerfTimer[Self.profile >= 1]

    # Level-3 profiler: per-layer timing (base slot indices into train_timer)
    var online_fwd_base: Int
    var target_fwd_base: Int
    var online_bwd_base: Int

    # Auto-checkpoint settings
    var checkpoint_every: Int
    var checkpoint_path: String

    # Optional metrics logger
    var logger: UnsafePointer[Self.L, MutAnyOrigin]
    var diag_every: Int

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
        self.train_timer = PerfTimer[Self.profile >= 1]()
        self.online_fwd_base = 0
        self.target_fwd_base = 0
        self.online_bwd_base = 0
        comptime if Self.profile >= 2:
            _ = self.train_timer.add_slot("sample_batch")  # 0
            _ = self.train_timer.add_slot("online_forward")  # 1
            _ = self.train_timer.add_slot("target_forward")  # 2
            _ = self.train_timer.add_slot("td_targets")  # 3
            _ = self.train_timer.add_slot("grad_kernel")  # 4
            _ = self.train_timer.add_slot("dueling_grad")  # 5
            _ = self.train_timer.add_slot("backward_update")  # 6
        comptime if Self.profile >= 3:
            # L3 slots as children of L2 sub-phases
            # slot 1 = online_forward, 2 = target_forward, 6 = backward_update
            self.online_fwd_base = Self.DuelingModel.register_forward_slots(
                self.train_timer, parent=1
            )
            self.target_fwd_base = Self.DuelingModel.register_forward_slots(
                self.train_timer, parent=2
            )
            self.online_bwd_base = Self.DuelingModel.register_backward_slots(
                self.train_timer, parent=6
            )
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        self.diag_every = 0

    fn _perf_ptr(mut self) -> PerfTimerPtr:
        """Return opaque timer pointer for L3 profiling (null when profile < 3).
        """
        comptime if Self.profile >= 3:
            return UnsafePointer(to=self.train_timer).bitcast[NoneType]()
        else:
            return PerfTimerPtr(unsafe_from_address=0)

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
        ](rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](obs.unsafe_ptr()))
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

        var out_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.DUELING_OUT](
            uninitialized=True
        )
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
        Self.DuelingNet.forward_with_cache[Self.BATCH](obs_t, out_t, p, cache_t)

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
                q_arr[idx] = v_s + (
                    out_arr[b * Self.DUELING_OUT + 1 + a] - mean_adv
                )

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
        Self.DuelingNet.backward[Self.BATCH](dout_t, dobs_t, p, cache_t, g)
        cpu_state.online.optimizer_step()

        # --- Phase 6: Soft update target network ---
        cpu_state.target.soft_update_from(cpu_state.online, self.tau)

        # Log Dueling DQN diagnostics
        if self.logger and (
            self.diag_every <= 0 or self.train_step_count % self.diag_every == 0
        ):
            try:
                var step = self.train_step_count
                self.logger[].log_scalar("loss", loss, step)

                # Q-value stats
                var q_min = Float64(q_arr[0])
                var q_max = Float64(q_arr[0])
                var q_sum: Float64 = 0.0
                for i in range(Self.BATCH * Self.ACTIONS):
                    var v = Float64(q_arr[i])
                    q_sum += v
                    if v < q_min:
                        q_min = v
                    if v > q_max:
                        q_max = v
                self.logger[].log_scalar(
                    "q_mean", q_sum / Float64(Self.BATCH * Self.ACTIONS), step
                )
                self.logger[].log_scalar("q_min", q_min, step)
                self.logger[].log_scalar("q_max", q_max, step)

                # TD target stats
                var tgt_sum: Float64 = 0.0
                for i in range(Self.BATCH):
                    tgt_sum += Float64(targets[i])
                self.logger[].log_scalar(
                    "td_target_mean", tgt_sum / Float64(Self.BATCH), step
                )
            except:
                pass

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
    # GPUOffPolicyAgent trait conformance
    # =========================================================================

    fn make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        """Allocate all GPU buffers for Dueling DQN training.

        Does NOT upload CPU weights — call upload_to_gpu after this.
        """
        return Self.GPUStateType(ctx)

    fn upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Upload CPU network weights to GPU online and target networks."""
        gpu_state.online.upload_from(self.state.online, ctx)
        gpu_state.target.upload_from(self.state.target, ctx)

    fn download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Download trained GPU weights back to CPU network states."""
        gpu_state.online.download_to(self.state.online, ctx)
        gpu_state.target.download_to(self.state.target, ctx)

    fn select_actions_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Forward dueling Q-network on GPU for N_ENVS environments + epsilon-greedy.

        Steps:
        1. Forward through dueling model → raw output [V, A1..An]
        2. Combine V+A into Q values via dueling_combine_kernel
        3. Epsilon-greedy argmax selection

        Args:
            ctx: GPU device context.
            gpu_state: GPU state with dueling network and inference scratch buffers.
            obs_buf: Observations [N_ENVS * obs_dim].
            actions_buf: Output actions [N_ENVS] (float scalar indices).
        """
        comptime ENVS_BLOCKS = (N_ENVS + TPB - 1) // TPB

        # Step 1: Forward through dueling model
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var dueling_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.DUELING_OUT), MutAnyOrigin
        ](gpu_state.env_dueling_buf.unsafe_ptr())
        var p = gpu_state.online.params_view()
        Self.DuelingNet.forward_gpu[N_ENVS](
            ctx, obs_t, dueling_t, p, gpu_state.inf_ws
        )

        # Step 2: Combine V+A into Q values
        var q_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.env_q_buf.unsafe_ptr())

        @always_inline
        fn combine_wrapper(
            qv: LayoutTensor[
                dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
            ],
            dout: LayoutTensor[
                dtype, Layout.row_major(N_ENVS, Self.DUELING_OUT), MutAnyOrigin
            ],
        ):
            dueling_combine_kernel[
                dtype, N_ENVS, Self.ACTIONS, Self.DUELING_OUT
            ](qv, dout)

        ctx.enqueue_function[combine_wrapper, combine_wrapper](
            q_t,
            dueling_t,
            grid_dim=(ENVS_BLOCKS,),
            block_dim=(TPB,),
        )

        # Step 3: Epsilon-greedy argmax
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var epsilon_s = Scalar[dtype](self.epsilon)
        var seed_s = Scalar[DType.uint32](
            UInt32(self.get_total_steps() * 2654435761)
        )

        @always_inline
        fn argmax_wrapper(
            eps: Scalar[dtype],
            q_vals: LayoutTensor[
                dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
            ],
            acts: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
            base_seed: Scalar[DType.uint32],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= N_ENVS:
                return

            from std.random.philox import Random as PhiloxRandom

            var rng = PhiloxRandom(
                seed=UInt64(base_seed) * UInt64(N_ENVS) + UInt64(b), offset=0
            )
            var rand_vals = rng.step_uniform()
            var rand_val = Scalar[dtype](rand_vals[0])

            if rand_val < eps:
                var rand_vals2 = rng.step_uniform()
                acts[b] = Scalar[dtype](
                    Int(
                        Scalar[dtype](rand_vals2[0])
                        * Scalar[dtype](Self.ACTIONS)
                    )
                )
                return

            var best_q = q_vals[b, 0]
            var best_action = 0
            for a in range(1, Self.ACTIONS):
                var qv = q_vals[b, a]
                if qv > best_q:
                    best_q = qv
                    best_action = a

            acts[b] = Scalar[dtype](best_action)

        ctx.enqueue_function[argmax_wrapper, argmax_wrapper](
            epsilon_s,
            q_t,
            actions_t,
            seed_s,
            grid_dim=(ENVS_BLOCKS,),
            block_dim=(TPB,),
        )

    fn do_gpu_train_step(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """One Dueling DQN training step on GPU.

        Phases:
        1. Sample batch from GPU replay buffer
        2. Online forward with cache → dueling output → combine to Q values
        3. Target forward → dueling output → combine to next Q values
        4. (Double DQN) Online forward on next obs → combine to online next Q
        5. Compute TD targets (reuse DQN kernels)
        6. Compute dQ gradient (MSE masked to taken action)
        7. Transform dQ → dueling output gradient via dueling_grad_kernel
        8. Backward through dueling model + optimizer step

        Args:
            ctx: GPU device context.
            gpu_state: GPU state with replay buffer, networks, and scratch buffers.
        """
        comptime BATCH = Self.batch_size
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB

        # ---- Phase 1: Sample batch ----
        comptime if Self.profile >= 2:
            self.train_timer.sync_and_mark(ctx)
        gpu_state.buffer.sample[BATCH](
            ctx,
            UInt32(self.train_step_count * (BATCH + 1)),
            gpu_state.s_obs,
            gpu_state.s_act,
            gpu_state.s_rew,
            gpu_state.s_nobs,
            gpu_state.s_done,
            gpu_state.s_idx,
        )
        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(0, ctx)
            self.train_timer.mark()

        # LayoutTensor views for sampled batch
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.s_obs.unsafe_ptr())
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.s_nobs.unsafe_ptr())

        # Dueling output tensors
        var dueling_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.DUELING_OUT), MutAnyOrigin
        ](gpu_state.dueling_out_buf.unsafe_ptr())
        var dueling_next_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.DUELING_OUT), MutAnyOrigin
        ](gpu_state.dueling_next_out.unsafe_ptr())

        # Q-value tensors (post-combination)
        var q_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.q_values.unsafe_ptr())
        var next_q_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.next_q_values.unsafe_ptr())

        var cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.DuelingModel.CACHE_SIZE),
            MutAnyOrigin,
        ](gpu_state.cache.unsafe_ptr())
        var targets_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.targets.unsafe_ptr())
        var rewards_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_rew.unsafe_ptr())
        var dones_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_done.unsafe_ptr())
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.s_act.unsafe_ptr())

        # ---- Phase 2: Online forward with cache → combine ----
        var p_online = gpu_state.online.params_view()
        var p_target = gpu_state.target.params_view()
        Self.DuelingNet.forward_gpu_with_cache[BATCH](
            ctx,
            obs_t,
            dueling_out_t,
            p_online,
            cache_t,
            gpu_state.train_ws,
            perf=self._perf_ptr(),
            perf_slot=self.online_fwd_base,
        )

        # Combine V+A → Q values
        @always_inline
        fn combine_online(
            qv: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
            ],
            dout: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.DUELING_OUT), MutAnyOrigin
            ],
        ):
            dueling_combine_kernel[
                dtype, BATCH, Self.ACTIONS, Self.DUELING_OUT
            ](qv, dout)

        ctx.enqueue_function[combine_online, combine_online](
            q_t,
            dueling_out_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )
        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(1, ctx)
            self.train_timer.mark()

        # ---- Phase 3: Target forward → combine ----
        Self.DuelingNet.forward_gpu[BATCH](
            ctx,
            next_obs_t,
            dueling_next_t,
            p_target,
            gpu_state.train_ws,
            perf=self._perf_ptr(),
            perf_slot=self.target_fwd_base,
        )

        @always_inline
        fn combine_target(
            qv: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
            ],
            dout: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.DUELING_OUT), MutAnyOrigin
            ],
        ):
            dueling_combine_kernel[
                dtype, BATCH, Self.ACTIONS, Self.DUELING_OUT
            ](qv, dout)

        ctx.enqueue_function[combine_target, combine_target](
            next_q_t,
            dueling_next_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )
        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(2, ctx)
            self.train_timer.mark()

        # ---- Phase 4: Compute TD targets ----
        var gamma_s = Scalar[dtype](self.gamma)

        comptime if Self.double_dqn:
            # Forward online on next_obs → combine to online_next_q
            var online_dueling_next_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.DUELING_OUT), MutAnyOrigin
            ](gpu_state.online_dueling_next_out.unsafe_ptr())
            Self.DuelingNet.forward_gpu[BATCH](
                ctx,
                next_obs_t,
                online_dueling_next_t,
                p_online,
                gpu_state.train_ws,
            )

            var online_next_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
            ](gpu_state.online_next_q.unsafe_ptr())

            @always_inline
            fn combine_online_next(
                qv: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
                ],
                dout: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.DUELING_OUT),
                    MutAnyOrigin,
                ],
            ):
                dueling_combine_kernel[
                    dtype, BATCH, Self.ACTIONS, Self.DUELING_OUT
                ](qv, dout)

            ctx.enqueue_function[combine_online_next, combine_online_next](
                online_next_t,
                online_dueling_next_t,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

            @always_inline
            fn double_td_wrapper(
                tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
                onq: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
                ],
                tnq: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
                ],
                rew: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
                don: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
                g: Scalar[dtype],
            ):
                dqn_double_td_target_kernel[dtype, BATCH, Self.ACTIONS](
                    tgt, onq, tnq, rew, don, g
                )

            ctx.enqueue_function[double_td_wrapper, double_td_wrapper](
                targets_t,
                online_next_t,
                next_q_t,
                rewards_t,
                dones_t,
                gamma_s,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
        else:

            @always_inline
            fn td_wrapper(
                tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
                nq: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
                ],
                rew: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
                don: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
                g: Scalar[dtype],
            ):
                dqn_td_target_kernel[dtype, BATCH, Self.ACTIONS](
                    tgt, nq, rew, don, g
                )

            ctx.enqueue_function[td_wrapper, td_wrapper](
                targets_t,
                next_q_t,
                rewards_t,
                dones_t,
                gamma_s,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(3, ctx)
            self.train_timer.mark()

        # ---- Phase 5: Gradient kernel (masked MSE grad on Q values) ----
        # Reuse next_q_values buffer for dQ grad (no longer needed after TD targets)
        var dq_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.next_q_values.unsafe_ptr())

        @always_inline
        fn grad_wrapper(
            grd: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
            ],
            qv: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
            ],
            tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            act: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return
            var action = Int(act[b])
            var q_pred = qv[b, action]
            var td_error = q_pred - tgt[b]
            for a in range(Self.ACTIONS):
                if a == action:
                    grd[b, a] = (
                        Scalar[dtype](2.0) * td_error / Scalar[dtype](BATCH)
                    )
                else:
                    grd[b, a] = Scalar[dtype](0.0)

        ctx.enqueue_function[grad_wrapper, grad_wrapper](
            dq_t,
            q_t,
            targets_t,
            actions_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )
        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(4, ctx)
            self.train_timer.mark()

        # ---- Phase 6: Transform dQ → dueling output gradient ----
        var dueling_grad_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.DUELING_OUT), MutAnyOrigin
        ](gpu_state.grad_output.unsafe_ptr())

        @always_inline
        fn dueling_grad_wrapper(
            dg: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.DUELING_OUT), MutAnyOrigin
            ],
            dq: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
            ],
        ):
            dueling_grad_kernel[dtype, BATCH, Self.ACTIONS, Self.DUELING_OUT](
                dg, dq
            )

        ctx.enqueue_function[dueling_grad_wrapper, dueling_grad_wrapper](
            dueling_grad_t,
            dq_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )
        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(5, ctx)
            self.train_timer.mark()

        # ---- Phase 7: Backward + optimizer step ----
        var grad_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.grad_input.unsafe_ptr())
        var g = gpu_state.online.grads_view()
        gpu_state.online.zero_grads(ctx)
        Self.DuelingNet.backward_gpu[BATCH](
            ctx,
            dueling_grad_t,
            grad_in_t,
            p_online,
            cache_t,
            g,
            gpu_state.train_ws,
            perf=self._perf_ptr(),
            perf_slot=self.online_bwd_base,
        )
        gpu_state.online.optimizer_step(ctx)
        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(6, ctx)

        self.train_step_count += 1

    fn get_action_scale(self) -> Float64:
        return 1.0  # Discrete actions don't use action_scale

    fn get_total_steps(self) -> Int:
        return self.train_step_count

    fn set_total_steps(mut self, steps: Int):
        self.train_step_count = steps

    fn decay_explore_gpu(mut self, total_steps: Int, num_steps: Int):
        """Linear epsilon schedule matching CleanRL:
        epsilon = max(end_e, start_e + (end_e - start_e) * t / duration).
        Exploration fraction = 0.5 (decay over first half of training).
        """
        var duration = Float64(num_steps) * 0.5  # exploration_fraction = 0.5
        var slope = (self.epsilon_min - 1.0) / duration
        self.epsilon = max(
            self.epsilon_min,
            slope * Float64(total_steps) + 1.0,
        )

    fn soft_update_targets_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """Soft-update target dueling network on GPU: theta_t <- tau*theta + (1-tau)*theta_t.
        """
        gpu_state.target.soft_update_from_gpu(gpu_state.online, self.tau, ctx)

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
        logger: UnsafePointer[Self.L, MutAnyOrigin] = UnsafePointer[
            Self.L, MutAnyOrigin
        ](),
        diag_every: Int = 0,
    ) raises -> TrainingMetrics:
        self.logger = logger
        self.diag_every = diag_every
        var cpu_state = Self.CPUStateType()
        var ckpt_every = self.checkpoint_every
        var ckpt_path = self.checkpoint_path
        var metrics = run_offpolicy_discrete_train[E, Self, L](
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
            logger,
        )
        self.state = cpu_state^
        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
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
    # GPU Training — delegates to shared run_offpolicy_discrete_train_gpu
    # =========================================================================

    fn train_gpu[
        E: GPUDiscreteEnv,
    ](
        mut self,
        ctx: DeviceContext,
        num_steps: Int,
        warmup_steps: Int = 1000,
        gradient_steps: Int = 0,
        sync_every: Int = 5000,
        verbose: Bool = False,
        print_every: Int = 50_000,
        environment_name: String = "Environment",
        logger: UnsafePointer[Self.L, MutAnyOrigin] = UnsafePointer[
            Self.L, MutAnyOrigin
        ](),
        diag_every: Int = 100,
    ) raises -> TrainingMetrics:
        """Train on GPU using the shared off-policy discrete GPU loop.

        GPU state (networks, replay buffer, scratch buffers) is created
        locally for the duration of training and freed when the method returns.
        After this call self.state.online / target hold the trained GPU weights,
        so evaluate() works immediately.

        All step-based parameters are in total env transitions (n_envs per
        loop iteration), matching on-policy convention.

        Parameters:
            E: GPU environment type implementing GPUDiscreteEnv.

        Args:
            ctx: GPU device context.
            num_steps: Total env transitions across all parallel envs.
            warmup_steps: Transitions before training starts (default: 1000).
            gradient_steps: Training steps per env collection iteration.
                0 (default) = n_envs for 1:1 replay ratio.
            sync_every: GPU->CPU sync interval in transitions (default: 5000).
            verbose: Print progress (default: False).
            print_every: Print interval in transitions (default: 50000).
            environment_name: Name for metrics labeling.

        Returns:
            TrainingMetrics with episode-level statistics.
        """
        self.logger = logger
        self.diag_every = diag_every
        var algo_name = String(
            "Dueling DQN (GPU)" if not Self.double_dqn else "Double Dueling DQN (GPU)"
        )

        # Create profiling timer with L1 slots (off-policy phases)
        var timer = PerfTimer[Self.profile >= 1]()
        _ = timer.add_slot("copy_prev_obs")
        _ = timer.add_slot("select_actions")
        _ = timer.add_slot("env_step")
        _ = timer.add_slot("buffer_store")
        _ = timer.add_slot("episode_tracking")
        _ = timer.add_slot("reset")
        _ = timer.add_slot("train_step")
        _ = timer.add_slot("gpu_cpu_sync")

        var metrics = run_offpolicy_discrete_train_gpu[
            E, Self, Self.profile, L
        ](
            self,
            ctx,
            num_steps,
            timer,
            warmup_steps=warmup_steps,
            gradient_steps=gradient_steps,
            sync_every=sync_every,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            algorithm_name=algo_name,
            logger=logger,
        )

        # Merge L2 sub-phases as children of train_step (slot 6)
        comptime if Self.profile >= 2:
            timer.merge_children(6, self.train_timer)

        comptime if Self.profile >= 1:
            timer.print_report(algo_name + " Profile")

        self.logger = UnsafePointer[Self.L, MutAnyOrigin]()
        return metrics^

    # =========================================================================
    # Checkpoint Save / Load
    # =========================================================================

    fn save_checkpoint(self, filepath: String) raises:
        from mojo_rl.nn.checkpoint import (
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
        from mojo_rl.nn.checkpoint import (
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
