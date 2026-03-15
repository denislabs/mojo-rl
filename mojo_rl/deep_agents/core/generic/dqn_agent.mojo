"""Generic DQN agent parameterized by DiscreteOffPolicyConfig.

Supports standard DQN and Double DQN via Config.DOUBLE_DQN flag.
"""

from std.math import exp
from std.random import random_float64
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Model, Linear, LinearReLU, Sequential
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.nn.training import Network, NetworkState, NetworkPair
from mojo_rl.nn.initializer import Xavier

from mojo_rl.deep_agents.core import (
    OffPolicyDiscreteState,
    OffPolicyDiscreteAgent,
    Checkpointable,
)
from mojo_rl.deep_agents.core.utils import obs_to_inline
from mojo_rl.deep_agents.core.replay import HeapReplayBuffer
from mojo_rl.core import TrainingMetrics, BoxDiscreteActionEnv


# =============================================================================
# DiscreteOffPolicyConfig trait
# =============================================================================


trait DiscreteOffPolicyConfig:
    """Compile-time config for DQN family agents."""

    comptime obs_dim: Int
    comptime num_actions: Int
    comptime batch_size: Int
    comptime buffer_capacity: Int
    comptime QModel: Model
    comptime QOpt: Optimizer
    comptime DOUBLE_DQN: Bool


# =============================================================================
# DQNConfig
# =============================================================================


struct DQNConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 120,
    HIDDEN2: Int = 84,
    CAP: Int = 10000,
    BS: Int = 128,
    lr: Float64 = 2.5e-4,
    double: Bool = False,
](DiscreteOffPolicyConfig):
    """DQN / Double DQN config."""

    comptime obs_dim: Int = Self.OBS
    comptime num_actions: Int = Self.ACT
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    comptime QModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN2],
        Linear[Self.HIDDEN2, Self.ACT],
    ]
    comptime QOpt = Adam[Self.lr]
    comptime DOUBLE_DQN: Bool = Self.double


# =============================================================================
# DQN CPU State
# =============================================================================


struct DQNCPUStateGeneric[
    QModel: Model,
    QOpt: Optimizer,
    buffer_capacity: Int,
    obs_dim: Int,
    batch_size: Int,
](Movable, OffPolicyDiscreteState):
    """CPU state for DQN: online + target Q-networks + replay buffer."""

    comptime BUFFER_DTYPE = dtype

    var online: NetworkState[Self.QModel, Self.QOpt]
    var target: NetworkState[Self.QModel, Self.QOpt]
    var buffer: HeapReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, 1, dtype
    ]

    fn __init__(out self):
        self.online = NetworkState[Self.QModel, Self.QOpt]()
        self.online.initialize[Xavier[]]()
        self.target = NetworkState[Self.QModel, Self.QOpt]()
        self.target.initialize[Xavier[]]()
        # Copy online → target
        self.target.copy_params_from(self.online)
        self.buffer = HeapReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, dtype
        ]()

    fn store[
        d: DType
    ](
        mut self,
        obs: List[Scalar[d]],
        action: Int,
        reward: Float64,
        next_obs: List[Scalar[d]],
        done: Bool,
    ) -> None:
        var obs_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.obs_dim](
            uninitialized=True
        )
        var next_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], Self.obs_dim](
            uninitialized=True
        )
        for i in range(Self.obs_dim):
            obs_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(obs[i]))
            next_arr[i] = Scalar[Self.BUFFER_DTYPE](Float64(next_obs[i]))
        var act_arr = InlineArray[Scalar[Self.BUFFER_DTYPE], 1](
            uninitialized=True
        )
        act_arr[0] = Scalar[Self.BUFFER_DTYPE](action)
        self.buffer.add(
            obs_arr,
            act_arr,
            Scalar[Self.BUFFER_DTYPE](reward),
            next_arr,
            done,
        )

    fn is_ready(self) -> Bool:
        return self.buffer.is_ready[Self.batch_size]()


# =============================================================================
# GenericDQNAgent[Config: DiscreteOffPolicyConfig]
# =============================================================================


struct GenericDQNAgent[
    Config: DiscreteOffPolicyConfig,
](OffPolicyDiscreteAgent & Checkpointable):
    """Generic DQN agent. Supports standard and double DQN via Config."""

    comptime OBS: Int = Self.Config.QModel.IN_DIM
    comptime ACTIONS: Int = Self.Config.QModel.OUT_DIM
    comptime BATCH: Int = Self.Config.batch_size
    comptime Q_CS: Int = Self.Config.QModel.CACHE_SIZE
    comptime QNet = Network[Self.Config.QModel, Self.Config.QOpt]

    comptime CPUStateType = DQNCPUStateGeneric[
        Self.Config.QModel,
        Self.Config.QOpt,
        Self.Config.buffer_capacity,
        Self.Config.QModel.IN_DIM,
        Self.Config.batch_size,
    ]

    var gamma: Float64
    var tau: Float64
    var epsilon: Float64
    var epsilon_min: Float64
    var epsilon_decay: Float64
    var target_update_freq: Int
    var train_step_count: Int
    var _target_update_ctr: Int
    var checkpoint_every: Int
    var checkpoint_path: String

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 1.0,
        epsilon: Float64 = 1.0,
        epsilon_min: Float64 = 0.05,
        epsilon_decay: Float64 = 0.995,
        target_update_freq: Int = 500,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ):
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.train_step_count = 0
        self._target_update_ctr = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

    fn make_cpu_state(self) -> Self.CPUStateType:
        return Self.CPUStateType()

    fn select_action[
        d: DType
    ](
        mut self, mut cpu_state: Self.CPUStateType, obs: List[Scalar[d]]
    ) -> Int:
        # Epsilon-greedy
        if random_float64() < self.epsilon:
            return Int(random_float64() * Float64(Self.ACTIONS))

        var obs_arr = obs_to_inline[Self.OBS, d](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var q_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var q_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](q_arr.unsafe_ptr())
        var p = cpu_state.online.params_view()
        Self.QNet.forward[1](obs_t, q_t, p)

        var best = 0
        var best_q = q_arr[0]
        for a in range(1, Self.ACTIONS):
            if q_arr[a] > best_q:
                best_q = q_arr[a]
                best = a
        return best

    fn store_transition[
        d: DType
    ](
        mut self,
        mut cpu_state: Self.CPUStateType,
        obs: List[Scalar[d]],
        action: Int,
        reward: Float64,
        next_obs: List[Scalar[d]],
        done: Bool,
    ) -> None:
        cpu_state.store[d](obs, action, reward, next_obs, done)

    fn do_cpu_train_step(
        mut self, mut cpu_state: Self.CPUStateType
    ) -> Float64:
        if not cpu_state.buffer.is_ready[Self.BATCH]():
            return 0.0

        # Sample batch
        var b_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var b_act1 = InlineArray[Scalar[dtype], Self.BATCH * 1](
            uninitialized=True
        )
        var b_rew = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var b_next = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var b_done = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        cpu_state.buffer.sample[Self.BATCH](
            b_obs, b_act1, b_rew, b_next, b_done
        )

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](b_obs.unsafe_ptr())
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](b_next.unsafe_ptr())

        # Online forward with cache
        var q_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        var q_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](q_arr.unsafe_ptr())
        var cache_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.Q_CS
        ](uninitialized=True)
        var cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.Q_CS), MutAnyOrigin
        ](cache_arr.unsafe_ptr())
        var p_online = cpu_state.online.params_view()
        Self.QNet.forward_with_cache[Self.BATCH](
            obs_t, q_t, p_online, cache_t
        )

        # Target forward
        var next_q_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        var next_q_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](next_q_arr.unsafe_ptr())
        var p_target = cpu_state.target.params_view()
        Self.QNet.forward[Self.BATCH](next_obs_t, next_q_t, p_target)

        # TD targets
        var targets = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        comptime if Self.Config.DOUBLE_DQN:
            # Double DQN: online selects, target evaluates
            var online_next_arr = InlineArray[
                Scalar[dtype], Self.BATCH * Self.ACTIONS
            ](uninitialized=True)
            var online_next_t = LayoutTensor[
                dtype,
                Layout.row_major(Self.BATCH, Self.ACTIONS),
                MutAnyOrigin,
            ](online_next_arr.unsafe_ptr())
            Self.QNet.forward[Self.BATCH](
                next_obs_t, online_next_t, p_online
            )
            for b in range(Self.BATCH):
                var best_a = 0
                var best_q = online_next_arr[b * Self.ACTIONS]
                for a in range(1, Self.ACTIONS):
                    var q = online_next_arr[b * Self.ACTIONS + a]
                    if q > best_q:
                        best_q = q
                        best_a = a
                var nq = next_q_arr[b * Self.ACTIONS + best_a]
                var dm = Scalar[dtype](1.0) - b_done[b]
                targets[b] = b_rew[b] + Scalar[dtype](self.gamma) * nq * dm

        comptime if not Self.Config.DOUBLE_DQN:
            for b in range(Self.BATCH):
                var max_nq = next_q_arr[b * Self.ACTIONS]
                for a in range(1, Self.ACTIONS):
                    var q = next_q_arr[b * Self.ACTIONS + a]
                    if q > max_nq:
                        max_nq = q
                var dm = Scalar[dtype](1.0) - b_done[b]
                targets[b] = (
                    b_rew[b] + Scalar[dtype](self.gamma) * max_nq * dm
                )

        # Gradient (MSE, masked to taken action)
        var grad_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        var total_loss: Float64 = 0.0
        for b in range(Self.BATCH):
            var action = Int(b_act1[b])
            var q_pred = q_arr[b * Self.ACTIONS + action]
            var td_err = q_pred - targets[b]
            total_loss += Float64(td_err * td_err)
            for a in range(Self.ACTIONS):
                if a == action:
                    grad_arr[b * Self.ACTIONS + a] = (
                        Scalar[dtype](2.0)
                        * td_err
                        / Scalar[dtype](Self.BATCH)
                    )
                else:
                    grad_arr[b * Self.ACTIONS + a] = Scalar[dtype](0.0)
        total_loss /= Float64(Self.BATCH)

        # Backward + optimizer step
        var grad_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](grad_arr.unsafe_ptr())
        var d_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var d_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](d_obs.unsafe_ptr())
        var g = cpu_state.online.grads_view()
        cpu_state.online.zero_grads()
        Self.QNet.backward[Self.BATCH](
            grad_t, d_obs_t, p_online, cache_t, g
        )
        cpu_state.online.optimizer_step()

        # Target update (hard or soft)
        self._target_update_ctr += 1
        if self._target_update_ctr >= self.target_update_freq:
            self._target_update_ctr = 0
            if self.tau >= 1.0:
                # Hard update
                cpu_state.target.copy_params_from(cpu_state.online)
            else:
                cpu_state.target.soft_update_from(cpu_state.online, self.tau)

        self.train_step_count += 1
        return total_loss

    fn decay_explore(mut self) -> None:
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    fn get_explore_rate(self) -> Float64:
        return self.epsilon

    fn random_action(self) -> Int:
        return Int(random_float64() * Float64(Self.ACTIONS))

    fn select_greedy_action(
        self, cpu_state: Self.CPUStateType, obs: List[Float64]
    ) -> Int:
        var obs_arr = obs_to_inline[Self.OBS, DType.float64](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var q_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var q_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](q_arr.unsafe_ptr())
        var p = cpu_state.online.params_view()
        Self.QNet.forward[1](obs_t, q_t, p)
        var best = 0
        var best_q = q_arr[0]
        for a in range(1, Self.ACTIONS):
            if q_arr[a] > best_q:
                best_q = q_arr[a]
                best = a
        return best

    # Checkpointable
    fn save_checkpoint(self, path: String) raises -> None:
        pass

    fn load_checkpoint(mut self, path: String) raises -> None:
        pass

    # Convenience
    fn train[
        E: BoxDiscreteActionEnv
    ](
        mut self, mut env: E, num_episodes: Int = 300
    ) raises -> TrainingMetrics:
        from mojo_rl.deep_agents.core.offpolicy_train import (
            run_offpolicy_discrete_train,
        )

        var cpu_state = self.make_cpu_state()
        var ckpt_path = String(self.checkpoint_path)
        return run_offpolicy_discrete_train(
            self,
            cpu_state,
            env,
            num_episodes,
            checkpoint_every=self.checkpoint_every,
            checkpoint_path=ckpt_path,
        )
