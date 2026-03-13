"""Deep TD3 Agent using the new trait-based deep learning architecture.

This TD3 (Twin Delayed Deep Deterministic Policy Gradient) implementation uses:
- NetworkState for heap-allocated params + grads + optimizer state
- Network (all-static) for stateless forward/backward ops via LayoutTensor
- Sequential composition for actor and critic networks
- Tanh output activation for bounded actions
- ReplayBuffer from mojo_rl.nn.replay for experience replay
- OffPolicyAgent trait for shared CPU training loop
- GPUOffPolicyAgent trait for shared GPU training loop

TD3 improves upon DDPG with three key innovations:
1. Twin Q-networks: Use two critics and take min(Q1, Q2) to reduce overestimation
2. Delayed policy updates: Update actor less frequently than critics
3. Target policy smoothing: Add clipped noise to target actions

Features:
- Works with any BoxContinuousActionEnv (continuous obs, continuous actions)
- Deterministic policy with Gaussian exploration noise
- Twin Q-networks to reduce overestimation bias
- Target networks for both actor and critics with soft updates
- Delayed actor updates (every policy_delay critic updates)
- Target policy smoothing with clipped noise
- lr is a compile-time parameter (Adam LR baked in at compile time)
- Checkpoint via NetworkState.write_sections / read_sections
- Unified CPU+GPU agent — same struct for both training modes

Usage:
    from mojo_rl.deep_agents.td3 import DeepTD3Agent
    from mojo_rl.envs import PendulumEnv

    var env = PendulumEnv()
    var agent = DeepTD3Agent[3, 1, 256, 100000, 64]()

    # CPU Training
    var metrics = agent.train(env, num_episodes=300)

    # GPU Training
    var ctx = DeviceContext()
    var metrics = agent.train_gpu[PendulumGPUEnv](ctx, num_steps=100000)

Reference: Fujimoto et al., "Addressing Function Approximation Error in
Actor-Critic Methods" (2018)
"""

from std.math import exp, sqrt
from std.random import random_float64, seed

from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TILE, TPB
from mojo_rl.nn import Model, Dense, DenseTanh, DenseReLU, Sequential
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.nn.initializer import Kaiming, Xavier
from mojo_rl.nn.training import (
    Network,
    NetworkState,
    GPUNetworkState,
    NetworkPair,
    GPUNetworkPair,
)
from mojo_rl.deep_agents.core import (
    obs_to_inline,
    concat_obs_action_batch,
    deterministic_select_action,
    greedy_continuous_action,
    store_continuous_transition,
    random_continuous_action,
    OffPolicyAgent,
    OffPolicyContinuousAgent,
    GPUOffPolicyState,
    GPUOffPolicyAgent,
    run_offpolicy_continuous_train,
    run_offpolicy_continuous_eval,
    run_offpolicy_continuous_train_gpu,
    Checkpointable,
)
from mojo_rl.deep_agents.core.replay import HeapReplayBuffer, GPUReplayBuffer
from mojo_rl.deep_agents.core.perf_timer import PerfTimer
from mojo_rl.nn.model.model import PerfTimerPtr
from mojo_rl.nn.gpu.random import gaussian_noise
from mojo_rl.deep_agents.core.kernels import (
    concat_obs_action_kernel,
    ddpg_exploration_kernel,
    td_mse_grad_kernel,
    actor_grad_from_critic_kernel,
    td_target_min_twin_kernel,
)
from .kernels import add_gaussian_noise_kernel
from std.gpu.host import DeviceContext, DeviceBuffer
from mojo_rl.nn.checkpoint import (
    write_checkpoint_header,
    write_metadata_section,
    parse_checkpoint_header,
    read_checkpoint_file,
    read_metadata_section,
    get_metadata_value,
    save_checkpoint_file,
)
from mojo_rl.core import (
    TrainingMetrics,
    BoxContinuousActionEnv,
    GPUContinuousEnv,
)
from mojo_rl.core.logger import LoggerPtr, _log
from .state import TD3CPUState, TD3GPUState

# =============================================================================
# Deep TD3 Agent
# =============================================================================


struct DeepTD3Agent[
    obs_dim: Int,
    action_dim: Int,
    hidden_dim: Int = 256,
    buffer_capacity: Int = 100000,
    batch_size: Int = 64,
    actor_lr: Float64 = 0.001,
    critic_lr: Float64 = 0.001,
    max_n_envs: Int = 64,
    profile: Int = 0,
](OffPolicyContinuousAgent & GPUOffPolicyAgent & Checkpointable):
    """Deep Twin Delayed DDPG agent — unified CPU + GPU.

    TD3 improves upon DDPG by addressing function approximation error through
    three key techniques: twin Q-networks, delayed policy updates, and target
    policy smoothing.

    Key features:
    - Deterministic policy (actor outputs action directly, not distribution)
    - Twin Q-networks to reduce overestimation bias (min of Q1, Q2)
    - Target networks for both actor and critics with soft updates
    - Delayed actor updates (every policy_delay critic updates)
    - Target policy smoothing (clipped Gaussian noise on target actions)
    - GPU training via GPUOffPolicyAgent trait + TD3GPUState

    Parameters:
        obs_dim: Dimension of observation space.
        action_dim: Dimension of action space.
        hidden_dim: Hidden layer size (default: 256).
        buffer_capacity: Replay buffer capacity (default: 100000).
        batch_size: Training batch size (default: 64).
        actor_lr: Actor Adam learning rate — compile-time (default: 0.001).
        critic_lr: Critic Adam learning rate — compile-time (default: 0.001).
        max_n_envs: Max parallel environments for GPU training (default: 64).
        profile: Level of profiling (0: none, 1: L2, 2: L3, 3: L4).
    """

    # Convenience compile-time aliases
    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.action_dim
    comptime HIDDEN = Self.hidden_dim
    comptime BATCH = Self.batch_size

    # Critic input dimension: obs + action concatenated
    comptime CRITIC_IN = Self.OBS + Self.ACTIONS

    # Actor: obs → hidden (ReLU) → hidden (ReLU) → action (Tanh)
    comptime ActorModel = Sequential[
        DenseReLU[Self.OBS, Self.HIDDEN],
        DenseReLU[Self.HIDDEN, Self.HIDDEN],
        DenseTanh[Self.HIDDEN, Self.ACTIONS],
    ]
    comptime ActorNet = Network[Self.ActorModel, Adam[Self.actor_lr]]

    # Critic: (obs ‖ action) → hidden (ReLU) → hidden (ReLU) → Q-value
    # Both critics share the same model architecture
    comptime CriticModel = Sequential[
        DenseReLU[Self.CRITIC_IN, Self.HIDDEN],
        DenseReLU[Self.HIDDEN, Self.HIDDEN],
        Dense[Self.HIDDEN, 1],
    ]
    comptime CriticNet = Network[Self.CriticModel, Adam[Self.critic_lr]]

    # GPUOffPolicyAgent required compile-time constants
    comptime OBS_DIM: Int = Self.OBS
    comptime ACTION_DIM: Int = Self.ACTIONS
    comptime BUFFER_CAPACITY: Int = Self.buffer_capacity
    comptime MAX_N_ENVS: Int = Self.max_n_envs
    comptime GPUStateType = TD3GPUState[
        Self.ActorModel,
        Adam[Self.actor_lr],
        Self.CriticModel,
        Adam[Self.critic_lr],
        Self.buffer_capacity,
        Self.obs_dim,
        Self.action_dim,
        Self.batch_size,
        Self.max_n_envs,
    ]
    comptime CPUStateType = TD3CPUState[
        Self.ActorModel,
        Adam[Self.actor_lr],
        Self.CriticModel,
        Adam[Self.critic_lr],
        Self.buffer_capacity,
        Self.obs_dim,
        Self.action_dim,
        Self.batch_size,
    ]

    # CPU state: networks + replay buffer + pre-allocated scratch buffers
    var state: Self.CPUStateType

    # Hyperparameters
    var gamma: Float64
    var tau: Float64
    var action_scale: Float64
    var noise_std: Float64
    var noise_std_min: Float64
    var noise_decay: Float64

    # TD3-specific hyperparameters
    var policy_delay: Int
    var target_noise_std: Float64
    var target_noise_clip: Float64

    # Training state
    var total_steps: Int
    var train_step_count: Int
    var update_count: Int

    # Level-2 profiler: sub-phases of do_gpu_train_step
    var train_timer: PerfTimer[Self.profile >= 1]

    # Level-3 profiler: per-layer timing (base slot indices into train_timer)
    # td_targets (L2 slot 1): actor target fwd, critic1 target fwd, critic2 target fwd
    var actor_target_fwd_base: Int
    var critic1_target_fwd_base: Int
    var critic2_target_fwd_base: Int
    # critic1_update (L2 slot 2): critic1 online fwd + bwd
    var critic1_fwd_base: Int
    var critic1_bwd_base: Int
    # critic2_update (L2 slot 3): critic2 online fwd + bwd
    var critic2_fwd_base: Int
    var critic2_bwd_base: Int
    # actor_update (L2 slot 4): actor fwd, critic1-for-PG fwd + bwd, actor bwd
    var actor_fwd_base: Int
    var critic1_pg_fwd_base: Int
    var critic1_pg_bwd_base: Int
    var actor_bwd_base: Int

    # Auto-checkpoint settings
    var checkpoint_every: Int
    var checkpoint_path: String

    # Optional metrics logger
    var logger: LoggerPtr
    var diag_every: Int

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        action_scale: Float64 = 1.0,
        noise_std: Float64 = 0.1,
        noise_std_min: Float64 = 0.01,
        noise_decay: Float64 = 0.995,
        policy_delay: Int = 2,
        target_noise_std: Float64 = 0.2,
        target_noise_clip: Float64 = 0.5,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ):
        """Initialize Deep TD3 agent.

        Args:
            gamma: Discount factor (default: 0.99).
            tau: Soft update coefficient (default: 0.005).
            action_scale: Action scaling factor (default: 1.0).
            noise_std: Initial exploration noise std (default: 0.1).
            noise_std_min: Minimum exploration noise std (default: 0.01).
            noise_decay: Noise decay per episode (default: 0.995).
            policy_delay: Update actor every N critic updates (default: 2).
            target_noise_std: Target policy smoothing noise std (default: 0.2).
            target_noise_clip: Clip range for target noise (default: 0.5).
            checkpoint_every: Save checkpoint every N episodes (0 to disable).
            checkpoint_path: Path to save checkpoints.
        """
        self.state = Self.CPUStateType()

        self.gamma = gamma
        self.tau = tau
        self.action_scale = action_scale
        self.noise_std = noise_std
        self.noise_std_min = noise_std_min
        self.noise_decay = noise_decay
        self.policy_delay = policy_delay
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.total_steps = 0
        self.train_step_count = 0
        self.update_count = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.train_timer = PerfTimer[Self.profile >= 1]()
        self.actor_target_fwd_base = 0
        self.critic1_target_fwd_base = 0
        self.critic2_target_fwd_base = 0
        self.critic1_fwd_base = 0
        self.critic1_bwd_base = 0
        self.critic2_fwd_base = 0
        self.critic2_bwd_base = 0
        self.actor_fwd_base = 0
        self.critic1_pg_fwd_base = 0
        self.critic1_pg_bwd_base = 0
        self.actor_bwd_base = 0
        comptime if Self.profile >= 2:
            _ = self.train_timer.add_slot("sample_batch")  # 0
            _ = self.train_timer.add_slot("td_targets")  # 1
            _ = self.train_timer.add_slot("critic1_update")  # 2
            _ = self.train_timer.add_slot("critic2_update")  # 3
            _ = self.train_timer.add_slot("actor_update")  # 4
        comptime if Self.profile >= 3:
            # td_targets (L2 slot 1): actor target fwd, critic1/2 target fwd
            self.actor_target_fwd_base = Self.ActorModel.register_forward_slots(
                self.train_timer, parent=1
            )
            self.critic1_target_fwd_base = (
                Self.CriticModel.register_forward_slots(
                    self.train_timer, parent=1
                )
            )
            self.critic2_target_fwd_base = (
                Self.CriticModel.register_forward_slots(
                    self.train_timer, parent=1
                )
            )
            # critic1_update (L2 slot 2): critic1 online fwd + bwd
            self.critic1_fwd_base = Self.CriticModel.register_forward_slots(
                self.train_timer, parent=2
            )
            self.critic1_bwd_base = Self.CriticModel.register_backward_slots(
                self.train_timer, parent=2
            )
            # critic2_update (L2 slot 3): critic2 online fwd + bwd
            self.critic2_fwd_base = Self.CriticModel.register_forward_slots(
                self.train_timer, parent=3
            )
            self.critic2_bwd_base = Self.CriticModel.register_backward_slots(
                self.train_timer, parent=3
            )
            # actor_update (L2 slot 4): actor fwd, critic1-for-PG fwd+bwd, actor bwd
            self.actor_fwd_base = Self.ActorModel.register_forward_slots(
                self.train_timer, parent=4
            )
            self.critic1_pg_fwd_base = Self.CriticModel.register_forward_slots(
                self.train_timer, parent=4
            )
            self.critic1_pg_bwd_base = Self.CriticModel.register_backward_slots(
                self.train_timer, parent=4
            )
            self.actor_bwd_base = Self.ActorModel.register_backward_slots(
                self.train_timer, parent=4
            )
        self.logger = LoggerPtr()
        self.diag_every = 0

    fn _perf_ptr(mut self) -> PerfTimerPtr:
        """Return opaque timer pointer for L3 profiling (null when profile < 3).
        """
        comptime if Self.profile >= 3:
            return UnsafePointer(to=self.train_timer).bitcast[NoneType]()
        else:
            return PerfTimerPtr(unsafe_from_address=0)

    # =========================================================================
    # OffPolicyContinuousAgent trait — required methods (CPU training)
    # =========================================================================

    fn make_cpu_state(self) -> Self.CPUStateType:
        """Allocate a fresh CPUStateType (called once before training)."""
        return Self.CPUStateType()

    fn select_action[
        DTYPE: DType
    ](
        mut self, mut cpu_state: Self.CPUStateType, obs: List[Scalar[DTYPE]]
    ) -> List[Scalar[DTYPE]]:
        """Select action with Gaussian exploration noise (training)."""
        return deterministic_select_action[
            DTYPE, Self.ActorModel, Adam[Self.actor_lr]
        ](cpu_state.actor.online, obs, self.action_scale, self.noise_std)

    fn store_transition[
        DTYPE: DType
    ](
        mut self,
        mut cpu_state: Self.CPUStateType,
        obs: List[Scalar[DTYPE]],
        action: List[Scalar[DTYPE]],
        reward: Float64,
        next_obs: List[Scalar[DTYPE]],
        done: Bool,
    ) -> None:
        """Store transition in the replay buffer."""
        var normalized_action = List[Scalar[DTYPE]](capacity=len(action))
        for i in range(len(action)):
            normalized_action.append(
                Scalar[DTYPE](Float64(action[i]) / self.action_scale)
            )
        cpu_state.store[DTYPE](obs, normalized_action, reward, next_obs, done)
        self.total_steps += 1

    fn do_cpu_train_step(mut self, mut cpu_state: Self.CPUStateType) -> Float64:
        """Perform one TD3 gradient update step.

        Returns:
            Average critic loss value.
        """
        return self.train_step(cpu_state)

    fn decay_explore(mut self) -> None:
        """Decay exploration noise (call once per episode)."""
        self.noise_std *= self.noise_decay
        if self.noise_std < self.noise_std_min:
            self.noise_std = self.noise_std_min

    fn get_explore_rate(self) -> Float64:
        """Return current exploration noise std."""
        return self.noise_std

    fn random_action[DTYPE: DType](self) -> List[Scalar[DTYPE]]:
        """Return a uniformly random action in [-action_scale, action_scale]."""
        return random_continuous_action[DTYPE](
            Self.action_dim, self.action_scale
        )

    fn select_greedy_action(
        self, cpu_state: Self.CPUStateType, obs: List[Float64]
    ) -> List[Float64]:
        """Select action using deterministic policy (no exploration noise)."""
        return greedy_continuous_action[Self.ActorModel, Adam[Self.actor_lr]](
            cpu_state.actor.online, obs, self.action_scale
        )

    # =========================================================================
    # Core TD3 CPU Training Step
    # =========================================================================

    fn train_step(mut self, mut cpu_state: Self.CPUStateType) -> Float64:
        """Perform one TD3 training step.

        TD3 update procedure:
        1. Always update both critics using TD error with min(Q1, Q2) target
        2. Every policy_delay updates:
           - Update actor using policy gradient from Q1
           - Soft update all target networks

        Returns:
            Average critic loss, or 0.0 if buffer not ready.
        """
        if not cpu_state.buffer.is_ready[Self.BATCH]():
            return 0.0

        self.update_count += 1

        # =================================================================
        # Phase 1: Sample batch
        # These 5 must remain local InlineArrays — ReplayBuffer.sample takes mut InlineArray
        # =================================================================
        var batch_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var batch_act = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        var batch_rew = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var batch_next = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var batch_done = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )

        cpu_state.buffer.sample[Self.BATCH](
            batch_obs, batch_act, batch_rew, batch_next, batch_done
        )

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](batch_obs.unsafe_ptr())
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](batch_next.unsafe_ptr())
        var act_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](batch_act.unsafe_ptr())

        # =================================================================
        # Phase 2: Compute TD targets with target policy smoothing
        # y = r + γ * min(Q1_t, Q2_t)(s', µ_t(s') + clip_noise) * (1 − done)
        # =================================================================

        # next_actions = actor_target(next_obs) — stored in pre-allocated _next_act
        var next_act_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](cpu_state._next_act.unsafe_ptr())
        var p_actor_target = cpu_state.actor.target.params_view()
        Self.ActorNet.forward[Self.BATCH](
            next_obs_t, next_act_t, p_actor_target
        )

        # TD3 Innovation #3: target policy smoothing with clipped noise
        for b in range(Self.BATCH):
            for i in range(Self.ACTIONS):
                var idx = b * Self.ACTIONS + i
                var noise = gaussian_noise() * self.target_noise_std
                if noise > self.target_noise_clip:
                    noise = self.target_noise_clip
                elif noise < -self.target_noise_clip:
                    noise = -self.target_noise_clip
                var noisy_a = Float64(cpu_state._next_act[idx]) + noise
                if noisy_a > 1.0:
                    noisy_a = 1.0
                elif noisy_a < -1.0:
                    noisy_a = -1.0
                cpu_state._next_act[idx] = Scalar[dtype](noisy_a)

        # Build next critic input: concat(next_obs, next_act)
        var next_ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](cpu_state._next_ci.unsafe_ptr())
        concat_obs_action_batch[Self.OBS, Self.ACTIONS, Self.BATCH](
            next_ci_t, next_obs_t, next_act_t
        )

        # Forward both target critics
        var nq1_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](cpu_state._nq1.unsafe_ptr())
        var nq2_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](cpu_state._nq2.unsafe_ptr())

        var p_c1t = cpu_state.critic1.target.params_view()
        var p_c2t = cpu_state.critic2.target.params_view()
        Self.CriticNet.forward[Self.BATCH](next_ci_t, nq1_t, p_c1t)
        Self.CriticNet.forward[Self.BATCH](next_ci_t, nq2_t, p_c2t)

        # TD3 Innovation #1: take min(Q1, Q2) for TD targets
        for b in range(Self.BATCH):
            var q1 = Float64(cpu_state._nq1[b])
            var q2 = Float64(cpu_state._nq2[b])
            if q1 != q1:
                q1 = 0.0
            if q2 != q2:
                q2 = 0.0
            var min_q = q1 if q1 < q2 else q2
            var done_mask = 1.0 - Float64(batch_done[b])
            var tgt = Float64(batch_rew[b]) + self.gamma * min_q * done_mask
            if tgt != tgt:
                tgt = 0.0
            elif tgt > 1000.0:
                tgt = 1000.0
            elif tgt < -1000.0:
                tgt = -1000.0
            cpu_state._targets[b] = Scalar[dtype](tgt)

        # =================================================================
        # Phase 3: Update Both Critics with the same targets
        # =================================================================

        # Build critic input: concat(obs, act)
        var ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](cpu_state._ci.unsafe_ptr())
        concat_obs_action_batch[Self.OBS, Self.ACTIONS, Self.BATCH](
            ci_t, obs_t, act_t
        )

        # --- Update Critic 1 ---
        var q1_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](cpu_state._q1_out.unsafe_ptr())
        var c1_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.CriticModel.CACHE_SIZE),
            MutAnyOrigin,
        ](cpu_state._q1_cache.unsafe_ptr())

        var p_c1 = cpu_state.critic1.params_view()
        Self.CriticNet.forward_with_cache[Self.BATCH](
            ci_t, q1_t, p_c1, c1_cache_t
        )

        var q1_grad_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](cpu_state._q_grad.unsafe_ptr())
        var critic1_loss: Float64 = 0.0
        for b in range(Self.BATCH):
            var td_err = cpu_state._q1_out[b] - cpu_state._targets[b]
            critic1_loss += Float64(td_err * td_err)
            cpu_state._q_grad[b] = (
                Scalar[dtype](2.0) * td_err / Scalar[dtype](Self.BATCH)
            )
        critic1_loss /= Float64(Self.BATCH)

        var d_c1_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](cpu_state._d_ci.unsafe_ptr())

        var g_c1 = cpu_state.critic1.grads_view()
        cpu_state.critic1.zero_grads()
        Self.CriticNet.backward[Self.BATCH](
            q1_grad_t, d_c1_t, p_c1, c1_cache_t, g_c1
        )
        cpu_state.critic1.optimizer_step()

        # --- Update Critic 2 ---
        var q2_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](cpu_state._q2_out.unsafe_ptr())
        var c2_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.CriticModel.CACHE_SIZE),
            MutAnyOrigin,
        ](cpu_state._q2_cache.unsafe_ptr())

        var p_c2 = cpu_state.critic2.params_view()
        Self.CriticNet.forward_with_cache[Self.BATCH](
            ci_t, q2_t, p_c2, c2_cache_t
        )

        var q2_grad_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](cpu_state._q_grad.unsafe_ptr())
        var critic2_loss: Float64 = 0.0
        for b in range(Self.BATCH):
            var td_err = cpu_state._q2_out[b] - cpu_state._targets[b]
            critic2_loss += Float64(td_err * td_err)
            cpu_state._q_grad[b] = (
                Scalar[dtype](2.0) * td_err / Scalar[dtype](Self.BATCH)
            )
        critic2_loss /= Float64(Self.BATCH)

        var d_c2_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](cpu_state._d_ci.unsafe_ptr())

        var g_c2 = cpu_state.critic2.grads_view()
        cpu_state.critic2.zero_grads()
        Self.CriticNet.backward[Self.BATCH](
            q2_grad_t, d_c2_t, p_c2, c2_cache_t, g_c2
        )
        cpu_state.critic2.optimizer_step()

        var avg_critic_loss = (critic1_loss + critic2_loss) / 2.0

        # Log TD3 diagnostics
        if self.logger and (
            self.diag_every <= 0
            or self.train_step_count % self.diag_every == 0
        ):
            try:
                var step = self.train_step_count
                _log(self.logger, "loss", avg_critic_loss, step)
                _log(self.logger, "critic1_loss", critic1_loss, step)
                _log(self.logger, "critic2_loss", critic2_loss, step)

                # Q1 stats
                var q1_sum: Float64 = 0.0
                for i in range(Self.BATCH):
                    q1_sum += Float64(cpu_state._q1_out[i])
                _log(self.logger, "q1_mean", q1_sum / Float64(Self.BATCH), step)

                # Q2 stats
                var q2_sum: Float64 = 0.0
                for i in range(Self.BATCH):
                    q2_sum += Float64(cpu_state._q2_out[i])
                _log(self.logger, "q2_mean", q2_sum / Float64(Self.BATCH), step)

                # TD target stats
                var tgt_sum: Float64 = 0.0
                for i in range(Self.BATCH):
                    tgt_sum += Float64(cpu_state._targets[i])
                _log(self.logger, "td_target_mean", tgt_sum / Float64(Self.BATCH), step)
            except:
                pass

        # =================================================================
        # Phase 4: Delayed Actor Update (TD3 Innovation #2)
        # Only update actor and all targets every policy_delay critic steps
        # =================================================================
        if self.update_count % self.policy_delay == 0:
            # actor_actions = actor_online(obs) with cache
            var actor_act_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
            ](cpu_state._actor_act.unsafe_ptr())
            var actor_cache_t = LayoutTensor[
                dtype,
                Layout.row_major(Self.BATCH, Self.ActorModel.CACHE_SIZE),
                MutAnyOrigin,
            ](cpu_state._actor_cache.unsafe_ptr())

            var p_actor = cpu_state.actor.params_view()
            Self.ActorNet.forward_with_cache[Self.BATCH](
                obs_t, actor_act_t, p_actor, actor_cache_t
            )

            # Build actor critic input: concat(obs, actor_act)
            var new_ci_t = LayoutTensor[
                dtype,
                Layout.row_major(Self.BATCH, Self.CRITIC_IN),
                MutAnyOrigin,
            ](cpu_state._new_ci.unsafe_ptr())
            concat_obs_action_batch[Self.OBS, Self.ACTIONS, Self.BATCH](
                new_ci_t, obs_t, actor_act_t
            )

            var new_q_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
            ](cpu_state._new_q.unsafe_ptr())
            var new_c1_cache_t = LayoutTensor[
                dtype,
                Layout.row_major(Self.BATCH, Self.CriticModel.CACHE_SIZE),
                MutAnyOrigin,
            ](cpu_state._q1_cache.unsafe_ptr())

            # TD3 uses Q1 for the actor gradient
            Self.CriticNet.forward_with_cache[Self.BATCH](
                new_ci_t, new_q_t, p_c1, new_c1_cache_t
            )

            # Gradient ascent: -1/BATCH
            var dq_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
            ](cpu_state._dq.unsafe_ptr())
            for b in range(Self.BATCH):
                cpu_state._dq[b] = Scalar[dtype](-1.0 / Float64(Self.BATCH))

            var d_new_ci_t = LayoutTensor[
                dtype,
                Layout.row_major(Self.BATCH, Self.CRITIC_IN),
                MutAnyOrigin,
            ](cpu_state._d_new_ci.unsafe_ptr())

            # Backward through critic1 to get dQ/d(actions) — do NOT update critic
            cpu_state.critic1.zero_grads()
            Self.CriticNet.backward[Self.BATCH](
                dq_t, d_new_ci_t, p_c1, new_c1_cache_t, g_c1
            )
            # Intentionally NOT calling critic1_online.optimizer_step() here

            # Extract action gradients
            var d_act_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
            ](cpu_state._d_act.unsafe_ptr())
            for b in range(Self.BATCH):
                for i in range(Self.ACTIONS):
                    cpu_state._d_act[
                        b * Self.ACTIONS + i
                    ] = cpu_state._d_new_ci[b * Self.CRITIC_IN + Self.OBS + i]

            var d_obs_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
            ](cpu_state._d_obs.unsafe_ptr())

            var g_actor = cpu_state.actor.grads_view()
            cpu_state.actor.zero_grads()
            Self.ActorNet.backward[Self.BATCH](
                d_act_t, d_obs_t, p_actor, actor_cache_t, g_actor
            )
            cpu_state.actor.optimizer_step()

            # Soft update all target networks
            cpu_state.actor.soft_update(self.tau)
            cpu_state.critic1.soft_update(self.tau)
            cpu_state.critic2.soft_update(self.tau)

        self.train_step_count += 1
        return avg_critic_loss

    # =========================================================================
    # High-level CPU training loop (delegates to shared off-policy runner)
    # =========================================================================

    fn train[
        E: BoxContinuousActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 200,
        warmup_steps: Int = 1000,
        train_every: Int = 1,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
        logger: LoggerPtr = LoggerPtr(),
        diag_every: Int = 0,
    ) raises -> TrainingMetrics:
        """Train the TD3 agent on a continuous action environment.

        Args:
            env: Environment implementing BoxContinuousActionEnv.
            num_episodes: Number of training episodes.
            max_steps_per_episode: Maximum steps per episode (default: 200).
            warmup_steps: Random steps to pre-fill replay buffer (default: 1000).
            train_every: Train every N steps (default: 1).
            verbose: Print progress (default: False).
            print_every: Print every N episodes if verbose (default: 10).
            environment_name: Name for metrics labeling.
            logger: Optional metrics logger pointer.
            diag_every: Log diagnostics every N train steps (0 = every step).

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        self.logger = logger
        self.diag_every = diag_every
        var cpu_state = Self.CPUStateType()
        var checkpoint_path = self.checkpoint_path
        var checkpoint_every = self.checkpoint_every
        var metrics = run_offpolicy_continuous_train(
            self,
            cpu_state,
            env,
            num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            warmup_steps=warmup_steps,
            train_every=train_every,
            verbose=verbose,
            print_every=print_every,
            checkpoint_every=checkpoint_every,
            checkpoint_path=checkpoint_path,
            algorithm_name="TD3 (CPU)",
            logger=logger,
        )
        self.state = cpu_state^
        self.logger = LoggerPtr()
        return metrics

    # =========================================================================
    # Evaluation (deterministic policy, no noise)
    # =========================================================================

    fn evaluate[
        E: BoxContinuousActionEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps: Int = 200,
        verbose: Bool = False,
    ) -> Float64:
        """Evaluate the agent using the deterministic policy (no noise).

        Delegates to run_offpolicy_continuous_eval (uses select_greedy_action_list).

        Args:
            env: Environment to evaluate on.
            num_episodes: Number of evaluation episodes (default: 10).
            max_steps: Maximum steps per episode (default: 200).
            verbose: Print per-episode results (default: False).

        Returns:
            Average reward across evaluation episodes.
        """
        return run_offpolicy_continuous_eval(
            self,
            self.state,
            env,
            num_episodes=num_episodes,
            max_steps=max_steps,
            verbose=verbose,
            algorithm_name="Deep TD3",
        ).mean_reward()

    # =========================================================================
    # GPUOffPolicyAgent trait — required methods
    # =========================================================================

    fn make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        """Allocate all GPU buffers for TD3 training.

        Does NOT upload CPU weights — call upload_to_gpu after this.
        """
        return Self.GPUStateType(ctx)

    fn upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Upload CPU network states and replay buffer to GPU."""
        gpu_state.actor.upload_from(self.state.actor, ctx)
        gpu_state.critic1.upload_from(self.state.critic1, ctx)
        gpu_state.critic2.upload_from(self.state.critic2, ctx)
        gpu_state.buffer.upload_from(self.state.buffer, ctx)

    fn download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Download trained GPU weights back to CPU network states."""
        gpu_state.actor.download_to(self.state.actor, ctx)
        gpu_state.critic1.download_to(self.state.critic1, ctx)
        gpu_state.critic2.download_to(self.state.critic2, ctx)

    fn select_actions_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Forward actor on GPU for N_ENVS environments + add exploration noise.
        """
        comptime BLOCKS = (N_ENVS * Self.ACTIONS + TPB - 1) // TPB

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var raw_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.raw_act.unsafe_ptr())
        var act_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        var p = gpu_state.actor.online.params_view()
        Self.ActorNet.forward_gpu[N_ENVS](
            ctx, obs_t, raw_t, p, gpu_state.inf_ws
        )

        var noise_std_s = Scalar[dtype](self.noise_std)
        var scale_s = Scalar[dtype](self.action_scale)
        # Kernel uses N_ENVS*ACTION_DIM seeds; total_steps increments by N_ENVS,
        # so multiply by ACTION_DIM to avoid overlap between consecutive calls.
        var rng_seed_s = Scalar[DType.uint32](
            UInt32(self.total_steps) * UInt32(Self.ACTIONS)
        )

        @always_inline
        fn exploration_wrapper(
            out_t: LayoutTensor[
                dtype,
                Layout.row_major(N_ENVS, Self.ACTIONS),
                MutAnyOrigin,
            ],
            raw_in: LayoutTensor[
                dtype,
                Layout.row_major(N_ENVS, Self.ACTIONS),
                MutAnyOrigin,
            ],
            ns: Scalar[dtype],
            sc: Scalar[dtype],
            rng_seed: Scalar[DType.uint32],
        ):
            ddpg_exploration_kernel[dtype, N_ENVS, Self.ACTIONS](
                out_t, raw_in, ns, sc, rng_seed
            )

        ctx.enqueue_function[exploration_wrapper, exploration_wrapper](
            act_t,
            raw_t,
            noise_std_s,
            scale_s,
            rng_seed_s,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    fn do_gpu_train_step(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """One TD3 training step on GPU.

        Always updates both critics.
        Every policy_delay steps: also updates actor.
        Uses self for hyperparams (gamma, tau, policy_delay, etc.)
        and gpu_state for all device buffers.
        """
        comptime BATCH = Self.BATCH
        comptime OBS = Self.OBS
        comptime ACTIONS = Self.ACTIONS
        comptime CRITIC_IN = Self.CRITIC_IN
        comptime CRITIC_CS = Self.CriticModel.CACHE_SIZE
        comptime ACTOR_CS = Self.ActorModel.CACHE_SIZE
        comptime TPB256 = 256
        comptime ELEM_BLOCKS = (BATCH * CRITIC_IN + TPB256 - 1) // TPB256
        comptime BATCH_BLOCKS = (BATCH + TPB256 - 1) // TPB256
        comptime ACT_BLOCKS = (BATCH * ACTIONS + TPB256 - 1) // TPB256

        self.update_count += 1

        # ----- Phase 1: Sample batch -----
        comptime if Self.profile >= 2:
            self.train_timer.sync_and_mark(ctx)
        # Kernel uses BATCH seeds [seed, seed+BATCH-1]; stride must be >= BATCH
        gpu_state.buffer.sample[BATCH](
            ctx,
            rng_seed=UInt32(self.update_count) * UInt32(BATCH + 1),
            sampled_obs=gpu_state.s_obs,
            sampled_actions=gpu_state.s_act,
            sampled_rewards=gpu_state.s_rew,
            sampled_next_obs=gpu_state.s_nobs,
            sampled_dones=gpu_state.s_done,
            indices=gpu_state.s_idx,
        )
        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(0, ctx)
            self.train_timer.mark()

        # LayoutTensor views on sampled data
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
        ](gpu_state.s_obs.unsafe_ptr())
        var nobs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
        ](gpu_state.s_nobs.unsafe_ptr())
        var act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](gpu_state.s_act.unsafe_ptr())
        var rew_t = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
            gpu_state.s_rew.unsafe_ptr()
        )
        var done_t = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
            gpu_state.s_done.unsafe_ptr()
        )

        # LayoutTensor views on target/twin scratch
        var next_act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](gpu_state.next_act.unsafe_ptr())
        var noisy_next_act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](gpu_state.noisy_next_act.unsafe_ptr())
        var next_ci_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
        ](gpu_state.next_ci.unsafe_ptr())
        var nq1_t = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
            gpu_state.nq1.unsafe_ptr()
        )
        var nq2_t = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
            gpu_state.nq2.unsafe_ptr()
        )
        var nq1_2d_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](gpu_state.nq1.unsafe_ptr())
        var nq2_2d_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](gpu_state.nq2.unsafe_ptr())
        var targets_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.targets.unsafe_ptr())

        var p_actor_t = gpu_state.actor.target.params_view()
        var p_c1t = gpu_state.critic1.target.params_view()
        var p_c2t = gpu_state.critic2.target.params_view()
        var p_actor = gpu_state.actor.online.params_view()
        var p_c1 = gpu_state.critic1.online.params_view()
        var p_c2 = gpu_state.critic2.online.params_view()

        # ----- Phase 2: Actor target → next_act -----
        Self.ActorNet.forward_gpu[BATCH](
            ctx,
            nobs_t,
            next_act_t,
            p_actor_t,
            gpu_state.actor_ws,
            perf=self._perf_ptr(),
            perf_slot=self.actor_target_fwd_base,
        )

        # ----- Phase 3: Target policy smoothing (TD3 Innovation #3) -----
        var tns_s = Scalar[dtype](self.target_noise_std)
        var tnc_s = Scalar[dtype](self.target_noise_clip)
        var act_min_s = Scalar[dtype](-self.action_scale)
        var act_max_s = Scalar[dtype](self.action_scale)
        # Kernel uses BATCH*ACTIONS seeds; stride must be >= BATCH*ACTIONS to avoid collision.
        var noise_seed_s = Scalar[DType.uint32](
            UInt32(self.update_count) * UInt32(BATCH * ACTIONS + 1)
        )

        @always_inline
        fn smooth_noise(
            noisy: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            clean: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            ns: Scalar[dtype],
            nc: Scalar[dtype],
            amin: Scalar[dtype],
            amax: Scalar[dtype],
            rng_seed: Scalar[DType.uint32],
        ):
            add_gaussian_noise_kernel[dtype, BATCH, ACTIONS](
                noisy, clean, ns, nc, amin, amax, rng_seed
            )

        ctx.enqueue_function[smooth_noise, smooth_noise](
            noisy_next_act_t,
            next_act_t,
            tns_s,
            tnc_s,
            act_min_s,
            act_max_s,
            noise_seed_s,
            grid_dim=(ACT_BLOCKS,),
            block_dim=(TPB256,),
        )

        # ----- Phase 4: Concat(next_obs, noisy_next_act) → next_ci -----
        @always_inline
        fn concat_next(
            d: LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
            ],
            o: LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin],
            a: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
        ):
            concat_obs_action_kernel[dtype, BATCH, OBS, ACTIONS](d, o, a)

        ctx.enqueue_function[concat_next, concat_next](
            next_ci_t,
            nobs_t,
            noisy_next_act_t,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB256,),
        )

        # ----- Phase 5: Both critics target forward -----
        Self.CriticNet.forward_gpu[BATCH](
            ctx,
            next_ci_t,
            nq1_2d_t,
            p_c1t,
            gpu_state.critic1_ws,
            perf=self._perf_ptr(),
            perf_slot=self.critic1_target_fwd_base,
        )
        Self.CriticNet.forward_gpu[BATCH](
            ctx,
            next_ci_t,
            nq2_2d_t,
            p_c2t,
            gpu_state.critic2_ws,
            perf=self._perf_ptr(),
            perf_slot=self.critic2_target_fwd_base,
        )

        # ----- Phase 6: TD targets (TD3 Innovation #1: min(Q1, Q2)) -----
        var gamma_s = Scalar[dtype](self.gamma)
        var zero_s = Scalar[dtype](0.0)
        var log_probs_dummy = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.targets.unsafe_ptr())

        @always_inline
        fn td3_targets(
            tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            r: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            q1: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            q2: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            d: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            lp: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            g: Scalar[dtype],
            a: Scalar[dtype],
        ):
            td_target_min_twin_kernel[dtype, BATCH, False](
                tgt, r, q1, q2, d, lp, g, a
            )

        ctx.enqueue_function[td3_targets, td3_targets](
            targets_t,
            rew_t,
            nq1_t,
            nq2_t,
            done_t,
            log_probs_dummy,
            gamma_s,
            zero_s,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB256,),
        )

        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(1, ctx)
            self.train_timer.mark()

        # ----- Phase 7: Concat(obs, actions) → ci -----
        var ci_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
        ](gpu_state.ci.unsafe_ptr())

        @always_inline
        fn concat_ci(
            d: LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
            ],
            o: LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin],
            a: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
        ):
            concat_obs_action_kernel[dtype, BATCH, OBS, ACTIONS](d, o, a)

        ctx.enqueue_function[concat_ci, concat_ci](
            ci_t,
            obs_t,
            act_t,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB256,),
        )

        # ----- Phase 8: Critic1 forward + MSE grad + backward + optim -----
        var q1_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](gpu_state.q1_out.unsafe_ptr())
        var q1_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_CS), MutAnyOrigin
        ](gpu_state.q1_cache.unsafe_ptr())
        var q1_grad_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](gpu_state.q1_grad.unsafe_ptr())
        var d_ci1_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
        ](gpu_state.d_ci1.unsafe_ptr())

        Self.CriticNet.forward_gpu_with_cache[BATCH](
            ctx,
            ci_t,
            q1_t,
            p_c1,
            q1_cache_t,
            gpu_state.critic1_ws,
            perf=self._perf_ptr(),
            perf_slot=self.critic1_fwd_base,
        )

        @always_inline
        fn mse_grad1(
            qg: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
            q: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
            tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        ):
            td_mse_grad_kernel[dtype, BATCH](qg, q, tgt)

        ctx.enqueue_function[mse_grad1, mse_grad1](
            q1_grad_t,
            q1_t,
            targets_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB256,),
        )

        var g_c1 = gpu_state.critic1.online.grads_view()
        gpu_state.critic1.online.zero_grads(ctx)
        Self.CriticNet.backward_gpu[BATCH](
            ctx,
            q1_grad_t,
            d_ci1_t,
            p_c1,
            q1_cache_t,
            g_c1,
            gpu_state.critic1_ws,
            perf=self._perf_ptr(),
            perf_slot=self.critic1_bwd_base,
        )
        gpu_state.critic1.online.optimizer_step(ctx)
        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(2, ctx)
            self.train_timer.mark()

        # ----- Phase 9: Critic2 forward + MSE grad + backward + optim -----
        var q2_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](gpu_state.q2_out.unsafe_ptr())
        var q2_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_CS), MutAnyOrigin
        ](gpu_state.q2_cache.unsafe_ptr())
        var q2_grad_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](gpu_state.q2_grad.unsafe_ptr())
        var d_ci2_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
        ](gpu_state.d_ci2.unsafe_ptr())

        Self.CriticNet.forward_gpu_with_cache[BATCH](
            ctx,
            ci_t,
            q2_t,
            p_c2,
            q2_cache_t,
            gpu_state.critic2_ws,
            perf=self._perf_ptr(),
            perf_slot=self.critic2_fwd_base,
        )

        @always_inline
        fn mse_grad2(
            qg: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
            q: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
            tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        ):
            td_mse_grad_kernel[dtype, BATCH](qg, q, tgt)

        ctx.enqueue_function[mse_grad2, mse_grad2](
            q2_grad_t,
            q2_t,
            targets_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB256,),
        )

        var g_c2 = gpu_state.critic2.online.grads_view()
        gpu_state.critic2.online.zero_grads(ctx)
        Self.CriticNet.backward_gpu[BATCH](
            ctx,
            q2_grad_t,
            d_ci2_t,
            p_c2,
            q2_cache_t,
            g_c2,
            gpu_state.critic2_ws,
            perf=self._perf_ptr(),
            perf_slot=self.critic2_bwd_base,
        )
        gpu_state.critic2.online.optimizer_step(ctx)
        comptime if Self.profile >= 2:
            self.train_timer.sync_and_accumulate(3, ctx)
            self.train_timer.mark()

        # ----- Phase 10+: Delayed actor update (TD3 Innovation #2) -----
        if self.update_count % self.policy_delay == 0:
            var actor_act_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ](gpu_state.actor_act.unsafe_ptr())
            var new_ci_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
            ](gpu_state.new_ci.unsafe_ptr())
            var new_q_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ](gpu_state.new_q.unsafe_ptr())
            var new_q_cache_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_CS), MutAnyOrigin
            ](gpu_state.new_q_cache.unsafe_ptr())
            var actor_cache_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTOR_CS), MutAnyOrigin
            ](gpu_state.actor_cache.unsafe_ptr())
            var dq_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ](gpu_state.dq.unsafe_ptr())
            var d_new_ci_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
            ](gpu_state.d_new_ci.unsafe_ptr())
            var d_act_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ](gpu_state.d_act.unsafe_ptr())
            var d_obs_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
            ](gpu_state.d_obs.unsafe_ptr())

            # Actor forward with cache
            Self.ActorNet.forward_gpu_with_cache[BATCH](
                ctx,
                obs_t,
                actor_act_t,
                p_actor,
                actor_cache_t,
                gpu_state.actor_ws,
                perf=self._perf_ptr(),
                perf_slot=self.actor_fwd_base,
            )

            # Concat(obs, actor_actions) → new_ci
            @always_inline
            fn concat_new_ci(
                d: LayoutTensor[
                    dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
                ],
                o: LayoutTensor[
                    dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
                ],
                a: LayoutTensor[
                    dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
                ],
            ):
                concat_obs_action_kernel[dtype, BATCH, OBS, ACTIONS](d, o, a)

            ctx.enqueue_function[concat_new_ci, concat_new_ci](
                new_ci_t,
                obs_t,
                actor_act_t,
                grid_dim=(ELEM_BLOCKS,),
                block_dim=(TPB256,),
            )

            # Critic1 forward for policy gradient (use critic1 per TD3 paper)
            Self.CriticNet.forward_gpu_with_cache[BATCH](
                ctx,
                new_ci_t,
                new_q_t,
                p_c1,
                new_q_cache_t,
                gpu_state.critic1_ws,
                perf=self._perf_ptr(),
                perf_slot=self.critic1_pg_fwd_base,
            )

            # Critic1 backward for action gradients (no optimizer step)
            var g_c1_pg = gpu_state.critic1.online.grads_view()
            gpu_state.critic1.online.zero_grads(ctx)
            Self.CriticNet.backward_gpu[BATCH](
                ctx,
                dq_t,
                d_new_ci_t,
                p_c1,
                new_q_cache_t,
                g_c1_pg,
                gpu_state.critic1_ws,
                perf=self._perf_ptr(),
                perf_slot=self.critic1_pg_bwd_base,
            )

            # Extract action gradients from critic input gradient
            @always_inline
            fn extract_act_grad(
                da: LayoutTensor[
                    dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
                ],
                dnc: LayoutTensor[
                    dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
                ],
            ):
                actor_grad_from_critic_kernel[dtype, BATCH, OBS, ACTIONS](
                    da, dnc
                )

            ctx.enqueue_function[extract_act_grad, extract_act_grad](
                d_act_t,
                d_new_ci_t,
                grid_dim=(ACT_BLOCKS,),
                block_dim=(TPB256,),
            )

            # Actor backward + optimizer step
            var g_actor = gpu_state.actor.online.grads_view()
            gpu_state.actor.online.zero_grads(ctx)
            Self.ActorNet.backward_gpu[BATCH](
                ctx,
                d_act_t,
                d_obs_t,
                p_actor,
                actor_cache_t,
                g_actor,
                gpu_state.actor_ws,
                perf=self._perf_ptr(),
                perf_slot=self.actor_bwd_base,
            )
            gpu_state.actor.online.optimizer_step(ctx)
            comptime if Self.profile >= 2:
                self.train_timer.sync_and_accumulate(4, ctx)

    fn get_action_scale(self) -> Float64:
        return self.action_scale

    fn get_total_steps(self) -> Int:
        return self.total_steps

    fn set_total_steps(mut self, steps: Int):
        self.total_steps = steps

    fn decay_explore_gpu(mut self, total_steps: Int, num_steps: Int):
        pass  # TD3 uses deterministic policy + Gaussian noise, no epsilon

    fn soft_update_targets_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """Soft-update target networks on GPU (only when actor was updated).

        TD3 delays target updates to match actor updates: only every
        policy_delay critic steps. This method checks update_count to decide.
        """
        if self.update_count % self.policy_delay == 0:
            gpu_state.actor.soft_update(self.tau, ctx)
            gpu_state.critic1.soft_update(self.tau, ctx)
            gpu_state.critic2.soft_update(self.tau, ctx)

    # =========================================================================
    # GPU training — delegates to shared run_offpolicy_continuous_train_gpu
    # =========================================================================

    fn train_gpu[
        E: GPUContinuousEnv,
    ](
        mut self,
        ctx: DeviceContext,
        num_steps: Int,
        warmup_steps: Int = 1000,
        gradient_steps: Int = 0,
        sync_every: Int = 5000,
        verbose: Bool = False,
        print_every: Int = 50_000,
        logger: LoggerPtr = LoggerPtr(),
        diag_every: Int = 100,
    ) raises -> TrainingMetrics:
        """Train on GPU using the shared off-policy GPU loop.

        GPU state (networks, replay buffer, scratch buffers) is created
        locally for the duration of training and freed when the method returns.
        After this call self.actor / critic1 / critic2 (online and target) hold
        the trained GPU weights (synced by download_from_gpu).

        All step-based parameters are in total env transitions (n_envs per
        loop iteration), matching on-policy convention.

        Parameters:
            E: GPU environment type implementing GPUContinuousEnv.

        Args:
            ctx: GPU device context.
            num_steps: Total env transitions across all parallel envs.
            warmup_steps: Transitions before training starts (default: 1000).
            gradient_steps: Training steps per env collection iteration.
                0 (default) = n_envs for 1:1 replay ratio.
            sync_every: GPU→CPU sync interval in transitions (default: 5000).
            verbose: Print progress (default: False).
            print_every: Print interval in transitions (default: 50000).

        Returns:
            TrainingMetrics with episode-level statistics.
        """
        self.logger = logger
        self.diag_every = diag_every
        var checkpoint_path = self.checkpoint_path
        var checkpoint_every = self.checkpoint_every
        var timer = PerfTimer[Self.profile >= 1]()
        _ = timer.add_slot("copy_prev_obs")
        _ = timer.add_slot("select_actions")
        _ = timer.add_slot("env_step")
        _ = timer.add_slot("buffer_store")
        _ = timer.add_slot("episode_tracking")
        _ = timer.add_slot("reset")
        _ = timer.add_slot("train_step")
        _ = timer.add_slot("gpu_cpu_sync")
        var metrics = run_offpolicy_continuous_train_gpu[E, Self, Self.profile](
            self,
            ctx,
            num_steps,
            timer,
            warmup_steps=warmup_steps,
            gradient_steps=gradient_steps,
            sync_every=sync_every,
            verbose=verbose,
            print_every=print_every,
            algorithm_name="TD3 (GPU)",
            environment_name=E.NAME,
            checkpoint_every=checkpoint_every,
            checkpoint_path=checkpoint_path,
            logger=logger,
        )

        # Merge L2 sub-phases as children of train_step (slot 6)
        comptime if Self.profile >= 2:
            timer.merge_children(6, self.train_timer)

        comptime if Self.profile >= 1:
            timer.print_report("TD3 (GPU) Profile")
        self.logger = LoggerPtr()
        return metrics^

    # =========================================================================
    # Checkpoint Save / Load
    # =========================================================================

    fn save_checkpoint(self, filepath: String) raises:
        """Save agent state to a checkpoint file.

        Saves all six NetworkState objects plus runtime hyperparameters.
        The replay buffer is NOT saved.

        Args:
            filepath: Destination path (e.g. "td3_agent.ckpt").
        """
        comptime ACTOR_PARAM_SIZE = Self.ActorNet.PARAM_SIZE
        comptime CRITIC_PARAM_SIZE = Self.CriticNet.PARAM_SIZE
        comptime ACTOR_STATE_SIZE = ACTOR_PARAM_SIZE * Adam[
            Self.actor_lr
        ].STATE_PER_PARAM
        comptime CRITIC_STATE_SIZE = CRITIC_PARAM_SIZE * Adam[
            Self.critic_lr
        ].STATE_PER_PARAM

        var content = write_checkpoint_header(
            "td3_agent",
            ACTOR_PARAM_SIZE + 2 * CRITIC_PARAM_SIZE,
            ACTOR_STATE_SIZE + 2 * CRITIC_STATE_SIZE,
        )
        content += self.state.actor.write_sections("actor_")
        content += self.state.critic1.write_sections("critic1_")
        content += self.state.critic2.write_sections("critic2_")

        var metadata = List[String]()
        metadata.append("gamma=" + String(self.gamma))
        metadata.append("tau=" + String(self.tau))
        metadata.append("actor_lr=" + String(Self.actor_lr))
        metadata.append("critic_lr=" + String(Self.critic_lr))
        metadata.append("action_scale=" + String(self.action_scale))
        metadata.append("noise_std=" + String(self.noise_std))
        metadata.append("noise_std_min=" + String(self.noise_std_min))
        metadata.append("noise_decay=" + String(self.noise_decay))
        metadata.append("policy_delay=" + String(self.policy_delay))
        metadata.append("target_noise_std=" + String(self.target_noise_std))
        metadata.append("target_noise_clip=" + String(self.target_noise_clip))
        metadata.append("total_steps=" + String(self.total_steps))
        metadata.append("train_step_count=" + String(self.train_step_count))
        metadata.append("update_count=" + String(self.update_count))
        content += write_metadata_section(metadata)

        save_checkpoint_file(filepath, content)

    fn load_checkpoint(mut self, filepath: String) raises:
        """Load agent state from a checkpoint file.

        Args:
            filepath: Path to the checkpoint file.
        """
        var content = read_checkpoint_file(filepath)

        self.state.actor.read_sections(content, "actor_")
        self.state.critic1.read_sections(content, "critic1_")
        self.state.critic2.read_sections(content, "critic2_")

        var metadata = read_metadata_section(content)

        var gamma_str = get_metadata_value(metadata, "gamma")
        if len(gamma_str) > 0:
            self.gamma = atof(gamma_str)

        var tau_str = get_metadata_value(metadata, "tau")
        if len(tau_str) > 0:
            self.tau = atof(tau_str)

        var action_scale_str = get_metadata_value(metadata, "action_scale")
        if len(action_scale_str) > 0:
            self.action_scale = atof(action_scale_str)

        var noise_std_str = get_metadata_value(metadata, "noise_std")
        if len(noise_std_str) > 0:
            self.noise_std = atof(noise_std_str)

        var noise_std_min_str = get_metadata_value(metadata, "noise_std_min")
        if len(noise_std_min_str) > 0:
            self.noise_std_min = atof(noise_std_min_str)

        var noise_decay_str = get_metadata_value(metadata, "noise_decay")
        if len(noise_decay_str) > 0:
            self.noise_decay = atof(noise_decay_str)

        var policy_delay_str = get_metadata_value(metadata, "policy_delay")
        if len(policy_delay_str) > 0:
            self.policy_delay = Int(atol(policy_delay_str))

        var tns_str = get_metadata_value(metadata, "target_noise_std")
        if len(tns_str) > 0:
            self.target_noise_std = atof(tns_str)

        var tnc_str = get_metadata_value(metadata, "target_noise_clip")
        if len(tnc_str) > 0:
            self.target_noise_clip = atof(tnc_str)

        var total_steps_str = get_metadata_value(metadata, "total_steps")
        if len(total_steps_str) > 0:
            self.total_steps = Int(atol(total_steps_str))

        var train_step_str = get_metadata_value(metadata, "train_step_count")
        if len(train_step_str) > 0:
            self.train_step_count = Int(atol(train_step_str))

        var update_count_str = get_metadata_value(metadata, "update_count")
        if len(update_count_str) > 0:
            self.update_count = Int(atol(update_count_str))
