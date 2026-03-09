"""Deep SAC Agent using the new trait-based deep learning architecture.

This SAC (Soft Actor-Critic) implementation uses:
- NetworkState for heap-allocated params + grads + optimizer state
- Network (all-static) for stateless forward/backward ops via LayoutTensor
- Sequential composition with Linear output layer (state-dependent log_std)
- ReplayBuffer from nn.replay for experience replay
- OffPolicyAgent trait for shared training loop

Features:
- Works with any BoxContinuousActionEnv (continuous obs, continuous actions)
- Stochastic Gaussian policy for better exploration (reparameterization trick)
- Twin Q-networks to reduce overestimation bias
- Automatic entropy temperature (alpha) tuning
- Maximum entropy RL objective: maximize reward + alpha * entropy
- Target networks with soft updates (critics only, no target actor)
- lr is a compile-time parameter (Adam LR baked in at compile time)
- Checkpoint via NetworkState.write_sections / read_sections

Usage:
    from deep_agents.sac import DeepSACAgent
    from envs import PendulumEnv

    var env = PendulumEnv()
    var agent = DeepSACAgent[3, 1, 256, 100000, 64]()

    # CPU Training
    var metrics = agent.train(env, num_episodes=300)

Reference: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy
Deep Reinforcement Learning with a Stochastic Actor" (2018)
"""

from std.math import exp, log, sqrt
from layout import Layout, LayoutTensor

from nn.constants import dtype, TILE, TPB
from nn.model import Model, Linear, LinearReLU, Sequential
from nn.model.stochastic_actor import (
    rsample,
    rsample_with_cache,
    rsample_backward,
    sample_action,
    get_deterministic_action,
)
from nn.optimizer import Optimizer, Adam
from nn.initializer import Kaiming
from nn.training import (
    Network,
    NetworkState,
    NetworkPair,
    GPUNetworkState,
    GPUNetworkPair,
)
from .state import SACCPUState, SACGPUState
from deep_agents.core import (
    obs_to_inline,
    concat_obs_action_batch,
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
from deep_agents.core.replay import HeapReplayBuffer, GPUReplayBuffer

from nn.gpu.random import gaussian_noise
from deep_agents.core.kernels import (
    concat_obs_action_kernel,
    td_mse_grad_kernel,
    actor_grad_from_critic_kernel,
    td_target_min_twin_kernel,
)
from .kernels import (
    sac_rsample_with_cache_kernel,
    sac_rsample_bwd_kernel,
    sac_sample_actions_kernel,
    min_q_dq_kernel,
    add_ci_grads_kernel,
)
from std.gpu.host import DeviceContext, DeviceBuffer
from nn.checkpoint import (
    write_checkpoint_header,
    write_metadata_section,
    parse_checkpoint_header,
    read_checkpoint_file,
    read_metadata_section,
    get_metadata_value,
    set_metadata_value_float,
    set_metadata_value_int,
    set_metadata_value_bool,
    save_checkpoint_file,
)
from core import (
    TrainingMetrics,
    BoxContinuousActionEnv,
    GPUContinuousEnv,
)


# =============================================================================
# Deep SAC Agent
# =============================================================================


struct DeepSACAgent[
    obs_dim: Int,
    action_dim: Int,
    hidden_dim: Int = 256,
    buffer_capacity: Int = 100000,
    batch_size: Int = 64,
    actor_lr: Float64 = 0.0003,
    critic_lr: Float64 = 0.0003,
    max_n_envs: Int = 64,
](OffPolicyContinuousAgent & GPUOffPolicyAgent & Checkpointable):
    """Deep Soft Actor-Critic agent using the new trait-based architecture.

    SAC is an off-policy actor-critic algorithm based on the maximum entropy
    reinforcement learning framework. It maximizes both expected reward and
    entropy, leading to more robust exploration and better sample efficiency.

    Key features:
    - Stochastic Gaussian policy (learns mean and log_std)
    - Twin Q-networks to reduce overestimation bias (like TD3)
    - No target actor (uses current policy for next-state actions)
    - Automatic entropy coefficient (alpha) tuning
    - Soft target updates for critic networks only
    - lr is compile-time (Adam LR baked in at compile time)

    Parameters:
        obs_dim: Dimension of observation space.
        action_dim: Dimension of action space.
        hidden_dim: Hidden layer size (default: 256).
        buffer_capacity: Replay buffer capacity (default: 100000).
        batch_size: Training batch size (default: 64).
        actor_lr: Actor Adam learning rate — compile-time (default: 0.0003).
        critic_lr: Critic Adam learning rate — compile-time (default: 0.0003).
        max_n_envs: Maximum number of environments (default: 64).
    """

    # Convenience compile-time aliases
    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.action_dim
    comptime HIDDEN = Self.hidden_dim
    comptime BATCH = Self.batch_size

    # Actor output: mean + log_std (state-dependent)
    comptime ACTOR_OUT = Self.ACTIONS * 2

    # Critic input dimension: obs + action concatenated
    comptime CRITIC_IN = Self.OBS + Self.ACTIONS

    # Actor: obs → hidden (ReLU) → hidden (ReLU) → Linear (mean + log_std)
    # Linear[HIDDEN, ACTIONS*2] gives state-dependent log_std (SAC requirement).
    # StochasticActor uses state-independent log_std (PPO design), which is wrong
    # for SAC where different states need different exploration levels.
    comptime ActorModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, Self.ACTIONS * 2],
    ]
    comptime ActorNet = Network[Self.ActorModel, Adam[Self.actor_lr]]

    # Critic: (obs ‖ action) → hidden (ReLU) → hidden (ReLU) → Q-value
    comptime CriticModel = Sequential[
        LinearReLU[Self.CRITIC_IN, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]
    comptime CriticNet = Network[Self.CriticModel, Adam[Self.critic_lr]]

    comptime CPUStateType = SACCPUState[
        Self.ActorModel,
        Adam[Self.actor_lr],
        Self.CriticModel,
        Adam[Self.critic_lr],
        Self.buffer_capacity,
        Self.obs_dim,
        Self.action_dim,
        Self.batch_size,
    ]

    # GPU compile-time aliases
    comptime OBS_DIM = Self.obs_dim
    comptime ACTION_DIM = Self.action_dim
    comptime BUFFER_CAPACITY = Self.buffer_capacity
    comptime MAX_N_ENVS = Self.max_n_envs

    comptime GPUStateType = SACGPUState[
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

    # CPU state: actor + twin critics + replay buffer + pre-allocated scratch
    var state: Self.CPUStateType

    # Hyperparameters
    var gamma: Float64
    var tau: Float64
    var action_scale: Float64

    # Entropy tuning
    var alpha: Float64
    var log_alpha: Float64
    var target_entropy: Float64
    var alpha_lr: Float64
    var auto_alpha: Bool

    # Adam state for alpha optimizer (scalar Adam, matching CleanRL)
    var alpha_adam_m: Float64  # First moment estimate
    var alpha_adam_v: Float64  # Second moment estimate
    var alpha_adam_t: Int  # Timestep counter

    # Policy delay (update actor + alpha every N critic updates, like TD3)
    var policy_delay: Int

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
        action_scale: Float64 = 1.0,
        alpha: Float64 = 0.2,
        auto_alpha: Bool = True,
        alpha_lr: Float64 = 0.0003,
        target_entropy: Float64 = -1.0,
        policy_delay: Int = 1,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ):
        """Initialize Deep SAC agent.

        Args:
            gamma: Discount factor (default: 0.99).
            tau: Soft update rate for target networks (default: 0.005).
            action_scale: Action scaling factor (default: 1.0).
            alpha: Initial entropy coefficient (default: 0.2).
            auto_alpha: Automatically tune alpha (default: True).
            alpha_lr: Alpha learning rate (default: 0.0003).
            target_entropy: Target entropy, typically -action_dim (default: -1.0).
            policy_delay: Update actor/alpha every N critic updates (default: 1).
            checkpoint_every: Save checkpoint every N episodes (0 to disable).
            checkpoint_path: Path to save checkpoints.
        """
        self.state = Self.CPUStateType()

        self.gamma = gamma
        self.tau = tau
        self.action_scale = action_scale
        self.alpha = alpha
        self.log_alpha = log(alpha)
        self.target_entropy = target_entropy
        self.alpha_lr = alpha_lr
        self.auto_alpha = auto_alpha
        self.alpha_adam_m = 0.0
        self.alpha_adam_v = 0.0
        self.alpha_adam_t = 0
        self.policy_delay = policy_delay
        self.total_steps = 0
        self.train_step_count = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

    # =========================================================================
    # OffPolicyContinuousAgent trait — required methods
    # =========================================================================

    fn make_cpu_state(self) -> Self.CPUStateType:
        """Allocate a fresh CPUStateType (called once before training)."""
        return Self.CPUStateType()

    fn select_action[
        DTYPE: DType
    ](
        mut self, mut cpu_state: Self.CPUStateType, obs: List[Scalar[DTYPE]]
    ) -> List[Scalar[DTYPE]]:
        """Select action using the stochastic policy (with reparameterization).

        SAC uses the inherently stochastic policy for exploration — no external
        noise is needed.

        Args:
            cpu_state: CPU state containing actor network.
            obs: Observation as List[Float64].

        Returns:
            Action list of length action_dim, scaled by action_scale.
        """
        var obs_arr = obs_to_inline[Self.OBS, DTYPE](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var out_arr = InlineArray[Scalar[dtype], Self.ACTOR_OUT](
            uninitialized=True
        )
        var out_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTOR_OUT), MutAnyOrigin
        ](out_arr.unsafe_ptr())

        var p = cpu_state.actor.params_view()
        Self.ActorNet.forward[1](obs_t, out_t, p)

        # Clamp mean and log_std
        var mean_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var log_std_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        for i in range(Self.ACTIONS):
            var m = Float64(out_arr[i])
            var raw_ls = Float64(out_arr[Self.ACTIONS + i])
            if m != m:
                m = 0.0
            elif m > 10.0:
                m = 10.0
            elif m < -10.0:
                m = -10.0
            if raw_ls != raw_ls:
                raw_ls = 0.0
            # Tanh scaling for log_std (CleanRL-style)
            from std.math import tanh as f64_tanh

            var ls = -5.0 + 0.5 * 7.0 * (f64_tanh(raw_ls) + 1.0)
            mean_arr[i] = Scalar[dtype](m)
            log_std_arr[i] = Scalar[dtype](ls)

        var mean_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](mean_arr.unsafe_ptr())
        var log_std_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](log_std_arr.unsafe_ptr())

        # Sample with reparameterization
        var noise_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        for i in range(Self.ACTIONS):
            noise_arr[i] = Scalar[dtype](gaussian_noise())
        var noise_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](noise_arr.unsafe_ptr())

        var act_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var act_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](act_arr.unsafe_ptr())
        var lp_arr = InlineArray[Scalar[dtype], 1](uninitialized=True)
        var lp_t = LayoutTensor[dtype, Layout.row_major(1, 1), MutAnyOrigin](
            lp_arr.unsafe_ptr()
        )

        sample_action[1, Self.ACTIONS](mean_t, log_std_t, noise_t, act_t)

        var result = List[Scalar[DTYPE]](capacity=Self.action_dim)
        for i in range(Self.action_dim):
            result.append(
                Scalar[DTYPE](Float64(act_arr[i]) * self.action_scale)
            )
        return result^

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
        """Store transition in the replay buffer.

        Actions are stored unscaled (divided by action_scale).
        """
        var normalized_action = List[Scalar[DTYPE]](capacity=len(action))
        for i in range(len(action)):
            normalized_action.append(
                Scalar[DTYPE](Float64(action[i]) / self.action_scale)
            )
        cpu_state.store[DTYPE](obs, normalized_action, reward, next_obs, done)
        self.total_steps += 1

    fn do_cpu_train_step(mut self, mut cpu_state: Self.CPUStateType) -> Float64:
        """Perform one SAC gradient update step.

        Returns:
            Average critic loss value.
        """
        return self.train_step(cpu_state)

    fn decay_explore(mut self) -> None:
        """SAC uses stochastic policy for exploration — no noise to decay.

        Alpha auto-tuning is handled inside train_step.
        """
        pass

    fn get_explore_rate(self) -> Float64:
        """Return current entropy coefficient alpha as exploration measure."""
        return self.alpha

    fn random_action[DTYPE: DType](self) -> List[Scalar[DTYPE]]:
        """Return a uniformly random action in [-action_scale, action_scale]."""
        return random_continuous_action[DTYPE](
            Self.action_dim, self.action_scale
        )

    fn select_greedy_action(
        self, cpu_state: Self.CPUStateType, obs: List[Float64]
    ) -> List[Float64]:
        """Select action using deterministic mean policy (no reparameterization noise).

        Used for evaluation. Applies tanh(mean) as the deterministic action
        instead of sampling from the Gaussian distribution.

        Args:
            cpu_state: CPU state containing actor network.
            obs: Observation as List[Float64].

        Returns:
            Deterministic action list of length action_dim,
            clipped to [-action_scale, action_scale].
        """
        var obs_arr = obs_to_inline[Self.OBS, DType.float64](obs)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var out_arr = InlineArray[Scalar[dtype], Self.ACTOR_OUT](
            uninitialized=True
        )
        var out_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTOR_OUT), MutAnyOrigin
        ](out_arr.unsafe_ptr())

        var p = cpu_state.actor.params_view()
        Self.ActorNet.forward[1](obs_t, out_t, p)

        # Extract mean (first ACTIONS elements of actor output)
        var mean_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var mean_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](mean_arr.unsafe_ptr())
        for i in range(Self.ACTIONS):
            mean_arr[i] = out_arr[i]

        var act_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var act_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](act_arr.unsafe_ptr())
        get_deterministic_action[1, Self.ACTIONS](mean_t, act_t)

        var result = List[Float64](capacity=Self.action_dim)
        for i in range(Self.action_dim):
            var a = Float64(act_arr[i]) * self.action_scale
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            result.append(Float64(a))
        return result^

    # =========================================================================
    # Core SAC Training Step
    # =========================================================================

    fn train_step(mut self, mut cpu_state: Self.CPUStateType) -> Float64:
        """Perform one SAC training step.

        Updates:
        1. Both critics using TD error with min(Q1, Q2) + entropy targets
        2. Actor using policy gradient (maximize Q - alpha * log_pi)
        3. Alpha (if auto_alpha=True)
        4. Soft update target critics

        Returns:
            Average critic loss, or 0.0 if buffer not ready.
        """
        if not cpu_state.buffer.is_ready[Self.BATCH]():
            return 0.0

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
        # Phase 2: Compute TD targets
        # y = r + γ * (min(Q1_t, Q2_t)(s', a') - α * log_π(a'|s')) * (1 - done)
        # where a' ~ π(·|s') (current actor, reparameterization)
        # =================================================================

        # Forward actor on next_obs to get next mean + log_std
        var next_out_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTOR_OUT), MutAnyOrigin
        ](cpu_state._next_out.unsafe_ptr())
        var p_actor = cpu_state.actor.params_view()
        Self.ActorNet.forward[Self.BATCH](next_obs_t, next_out_t, p_actor)

        # Extract and clamp mean + log_std for next states
        # These remain local — small (BATCH*ACTIONS), needed only within this phase
        var next_mean_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        var next_ls_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        for b in range(Self.BATCH):
            for a in range(Self.ACTIONS):
                var m = Float64(cpu_state._next_out[b * Self.ACTOR_OUT + a])
                var raw_ls = Float64(
                    cpu_state._next_out[b * Self.ACTOR_OUT + Self.ACTIONS + a]
                )
                if m != m:
                    m = 0.0
                elif m > 10.0:
                    m = 10.0
                elif m < -10.0:
                    m = -10.0
                if raw_ls != raw_ls:
                    raw_ls = 0.0
                # Tanh scaling for log_std (CleanRL-style)
                from std.math import tanh as f64_tanh

                var ls = -5.0 + 0.5 * 7.0 * (f64_tanh(raw_ls) + 1.0)
                next_mean_arr[b * Self.ACTIONS + a] = Scalar[dtype](m)
                next_ls_arr[b * Self.ACTIONS + a] = Scalar[dtype](ls)

        # Sample next actions + log_probs — noise is local (temporary per step)
        var next_noise_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        for i in range(Self.BATCH * Self.ACTIONS):
            next_noise_arr[i] = Scalar[dtype](gaussian_noise())

        var next_act_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](cpu_state._next_act.unsafe_ptr())
        var next_lp_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](cpu_state._next_log_pi.unsafe_ptr())

        var next_mean_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](next_mean_arr.unsafe_ptr())
        var next_ls_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](next_ls_arr.unsafe_ptr())
        var next_noise_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](next_noise_arr.unsafe_ptr())

        rsample[Self.BATCH, Self.ACTIONS](
            next_mean_t, next_ls_t, next_noise_t, next_act_t, next_lp_t
        )

        # Guard NaN/inf in log_probs
        for b in range(Self.BATCH):
            var lp = Float64(cpu_state._next_log_pi[b])
            if lp != lp or lp > 100.0 or lp < -100.0:
                cpu_state._next_log_pi[b] = Scalar[dtype](-1.0)

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

        # TD targets: r + γ * (min(Q1,Q2) - α * log_π) * (1 - done)
        for b in range(Self.BATCH):
            var q1 = Float64(cpu_state._nq1[b])
            var q2 = Float64(cpu_state._nq2[b])
            if q1 != q1:
                q1 = 0.0
            if q2 != q2:
                q2 = 0.0
            var min_q = q1 if q1 < q2 else q2
            var lp = Float64(cpu_state._next_log_pi[b])
            var done_mask = 1.0 - Float64(batch_done[b])
            var tgt = (
                Float64(batch_rew[b])
                + self.gamma * (min_q - self.alpha * lp) * done_mask
            )
            if tgt != tgt:
                tgt = 0.0
            elif tgt > 1000.0:
                tgt = 1000.0
            elif tgt < -1000.0:
                tgt = -1000.0
            cpu_state._targets[b] = Scalar[dtype](tgt)

        # =================================================================
        # Phase 3: Update Both Critics
        # =================================================================

        # Build critic input: concat(obs, act)
        var ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](cpu_state._ci.unsafe_ptr())
        concat_obs_action_batch[Self.OBS, Self.ACTIONS, Self.BATCH](
            ci_t, obs_t, act_t
        )

        # --- Critic 1 ---
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

        # --- Critic 2 ---
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

        # =================================================================
        # Phase 4: Update Actor
        # J_π = E[α * log_π(a|s) - Q(s, a)]  →  minimize (gradient descent)
        # =================================================================

        # Step 1: Forward actor with cache → mean + log_std
        var actor_out_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTOR_OUT), MutAnyOrigin
        ](cpu_state._curr_out.unsafe_ptr())
        var actor_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.ActorModel.CACHE_SIZE),
            MutAnyOrigin,
        ](cpu_state._actor_cache.unsafe_ptr())

        Self.ActorNet.forward_with_cache[Self.BATCH](
            obs_t, actor_out_t, p_actor, actor_cache_t
        )

        # Extract and clamp mean + log_std — remain local (temporary per step)
        var curr_mean_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        var curr_ls_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        for b in range(Self.BATCH):
            for a in range(Self.ACTIONS):
                var m = Float64(cpu_state._curr_out[b * Self.ACTOR_OUT + a])
                var raw_ls = Float64(
                    cpu_state._curr_out[b * Self.ACTOR_OUT + Self.ACTIONS + a]
                )
                if m != m:
                    m = 0.0
                elif m > 10.0:
                    m = 10.0
                elif m < -10.0:
                    m = -10.0
                if raw_ls != raw_ls:
                    raw_ls = 0.0
                # Tanh scaling for log_std (CleanRL-style)
                from std.math import tanh as f64_tanh

                var ls = -5.0 + 0.5 * 7.0 * (f64_tanh(raw_ls) + 1.0)
                curr_mean_arr[b * Self.ACTIONS + a] = Scalar[dtype](m)
                curr_ls_arr[b * Self.ACTIONS + a] = Scalar[dtype](ls)

        # Step 2: rsample_with_cache → sampled_actions, log_probs, z_cache
        # noise and z_cache are local (temporary per step)
        var curr_noise_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        for i in range(Self.BATCH * Self.ACTIONS):
            curr_noise_arr[i] = Scalar[dtype](gaussian_noise())

        var curr_act_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](cpu_state._curr_act.unsafe_ptr())
        var curr_lp_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](cpu_state._curr_log_pi.unsafe_ptr())
        var z_cache_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )

        var curr_mean_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](curr_mean_arr.unsafe_ptr())
        var curr_ls_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](curr_ls_arr.unsafe_ptr())
        var curr_noise_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](curr_noise_arr.unsafe_ptr())
        var z_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](z_cache_arr.unsafe_ptr())

        rsample_with_cache[Self.BATCH, Self.ACTIONS](
            curr_mean_t,
            curr_ls_t,
            curr_noise_t,
            curr_act_t,
            curr_lp_t,
            z_cache_t,
        )

        # Guard NaN/inf in log_probs
        for b in range(Self.BATCH):
            var lp = Float64(cpu_state._curr_log_pi[b])
            if lp != lp or lp > 100.0 or lp < -100.0:
                cpu_state._curr_log_pi[b] = Scalar[dtype](-1.0)

        # =================================================================
        # Phase 4: Delayed Actor Update (every policy_delay critic steps)
        # =================================================================
        if self.train_step_count % self.policy_delay == 0:
            # Step 3: Build critic input with sampled actions: concat(obs, curr_act)
            var new_ci_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
            ](cpu_state._new_ci.unsafe_ptr())
            concat_obs_action_batch[Self.OBS, Self.ACTIONS, Self.BATCH](
                new_ci_t, obs_t, curr_act_t
            )

            # Step 4: Forward critic1 with cache (need for actor backward)
            var new_q_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
            ](cpu_state._new_q1.unsafe_ptr())
            var new_c1_cache_t = LayoutTensor[
                dtype,
                Layout.row_major(Self.BATCH, Self.CriticModel.CACHE_SIZE),
                MutAnyOrigin,
            ](cpu_state._new_c1_cache.unsafe_ptr())

            Self.CriticNet.forward_with_cache[Self.BATCH](
                new_ci_t, new_q_t, p_c1, new_c1_cache_t
            )

            # Step 5: Backward through critic1 to get dQ/da (-1/BATCH per sample)
            var dq_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
            ](cpu_state._q_grad.unsafe_ptr())
            for b in range(Self.BATCH):
                cpu_state._q_grad[b] = Scalar[dtype](-1.0 / Float64(Self.BATCH))

            var d_new_ci_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
            ](cpu_state._d_ci.unsafe_ptr())

            # Backward through critic to get action gradient — don't update critic
            cpu_state.critic1.zero_grads()
            Self.CriticNet.backward[Self.BATCH](
                dq_t, d_new_ci_t, p_c1, new_c1_cache_t, g_c1
            )
            # Intentionally NOT calling critic1.optimizer_step() here

            # Extract action gradients from d_new_ci (last ACTIONS columns per row)
            var grad_act_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
            ](cpu_state._grad_act.unsafe_ptr())
            for b in range(Self.BATCH):
                for a in range(Self.ACTIONS):
                    cpu_state._grad_act[b * Self.ACTIONS + a] = cpu_state._d_ci[
                        b * Self.CRITIC_IN + Self.OBS + a
                    ]

            # Step 6: Entropy gradient: α/BATCH per sample (we're minimizing α*log_π)
            # grad_lp is local (small, BATCH elements, only used within this phase)
            var grad_lp_arr = InlineArray[Scalar[dtype], Self.BATCH](
                uninitialized=True
            )
            for b in range(Self.BATCH):
                grad_lp_arr[b] = Scalar[dtype](self.alpha / Float64(Self.BATCH))

            # Step 7: Backward through reparameterization → grad_mean, grad_log_std
            var grad_mean_arr = InlineArray[
                Scalar[dtype], Self.BATCH * Self.ACTIONS
            ](uninitialized=True)
            var grad_ls_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
                uninitialized=True
            )

            var grad_lp_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
            ](grad_lp_arr.unsafe_ptr())
            var grad_mean_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
            ](grad_mean_arr.unsafe_ptr())
            var grad_ls_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
            ](grad_ls_arr.unsafe_ptr())

            rsample_backward[Self.BATCH, Self.ACTIONS](
                grad_act_t,
                grad_lp_t,
                curr_act_t,
                curr_ls_t,
                curr_noise_t,
                grad_mean_t,
                grad_ls_t,
            )

            # Step 8: Build actor_grad = concat(grad_mean, grad_log_std)
            var actor_grad_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.ACTOR_OUT), MutAnyOrigin
            ](cpu_state._actor_grad_arr.unsafe_ptr())
            for b in range(Self.BATCH):
                for a in range(Self.ACTIONS):
                    cpu_state._actor_grad_arr[
                        b * Self.ACTOR_OUT + a
                    ] = grad_mean_arr[b * Self.ACTIONS + a]
                    cpu_state._actor_grad_arr[
                        b * Self.ACTOR_OUT + Self.ACTIONS + a
                    ] = grad_ls_arr[b * Self.ACTIONS + a]

            # Step 9: Backward through actor network
            var d_obs_t = LayoutTensor[
                dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
            ](cpu_state._d_obs.unsafe_ptr())

            var g_actor = cpu_state.actor.grads_view()
            cpu_state.actor.zero_grads()
            Self.ActorNet.backward[Self.BATCH](
                actor_grad_t, d_obs_t, p_actor, actor_cache_t, g_actor
            )
            cpu_state.actor.optimizer_step()

            # =============================================================
            # Phase 5: Update Alpha (if auto_alpha)
            # =============================================================
            if self.auto_alpha:
                var mean_lp: Float64 = 0.0
                for b in range(Self.BATCH):
                    mean_lp += Float64(cpu_state._curr_log_pi[b])
                mean_lp /= Float64(Self.BATCH)

                # CleanRL: alpha_loss = (-exp(log_alpha) * (log_pi + target_entropy)).mean()
                # ∂loss/∂log_alpha = -alpha * mean(log_pi + target_entropy)
                var grad = -self.alpha * (mean_lp + self.target_entropy)

                # Adam update for log_alpha (matches CleanRL's a_optimizer)
                self.alpha_adam_t += 1
                var beta1: Float64 = 0.9
                var beta2: Float64 = 0.999
                var eps: Float64 = 1e-8
                self.alpha_adam_m = beta1 * self.alpha_adam_m + (1.0 - beta1) * grad
                self.alpha_adam_v = beta2 * self.alpha_adam_v + (1.0 - beta2) * grad * grad
                var m_hat = self.alpha_adam_m / (1.0 - beta1 ** Float64(self.alpha_adam_t))
                var v_hat = self.alpha_adam_v / (1.0 - beta2 ** Float64(self.alpha_adam_t))
                self.log_alpha -= self.alpha_lr * m_hat / (sqrt(v_hat) + eps)
                self.alpha = exp(self.log_alpha)

        # =================================================================
        # Phase 6: Soft Update Target Critics (every step, not delayed)
        # =================================================================
        cpu_state.critic1.soft_update(self.tau)
        cpu_state.critic2.soft_update(self.tau)

        self.train_step_count += 1
        return avg_critic_loss

    # =========================================================================
    # GPUOffPolicyAgent trait — required methods
    # =========================================================================

    fn make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        """Allocate all GPU buffers for SAC training.

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
        """Forward SAC actor on GPU for N_ENVS environments + reparameterized sampling.
        """
        comptime BLOCKS = (N_ENVS + TPB - 1) // TPB

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var inf_out_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTOR_OUT), MutAnyOrigin
        ](gpu_state.inf_out.unsafe_ptr())
        var act_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        var p = gpu_state.actor.params_view()
        Self.ActorNet.forward_gpu[N_ENVS](
            ctx, obs_t, inf_out_t, p, gpu_state.inf_ws
        )

        var scale_s = Scalar[dtype](self.action_scale)
        var log_std_min_s = Scalar[dtype](-5.0)
        var log_std_max_s = Scalar[dtype](2.0)
        # Kernel uses N_ENVS*ACTION_DIM seeds; total_steps increments by N_ENVS,
        # so multiply by ACTION_DIM to avoid overlap between consecutive calls.
        var rng_seed_s = Scalar[DType.uint32](
            UInt32(self.total_steps) * UInt32(Self.ACTIONS)
        )

        @always_inline
        fn sample_actions(
            acts: LayoutTensor[
                dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
            ],
            ao: LayoutTensor[
                dtype,
                Layout.row_major(N_ENVS, Self.ACTIONS + Self.ACTIONS),
                MutAnyOrigin,
            ],
            sc: Scalar[dtype],
            lsmin: Scalar[dtype],
            lsmax: Scalar[dtype],
            rng_seed: Scalar[DType.uint32],
        ):
            sac_sample_actions_kernel[dtype, N_ENVS, Self.ACTIONS](
                acts, ao, sc, lsmin, lsmax, rng_seed
            )

        ctx.enqueue_function[sample_actions, sample_actions](
            act_t,
            inf_out_t,
            scale_s,
            log_std_min_s,
            log_std_max_s,
            rng_seed_s,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    fn do_gpu_train_step(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """One SAC training step on GPU.

        Updates both critics, the actor via reparameterization, and alpha (if auto_alpha).
        Unlike TD3, SAC has no target actor and always updates the actor.
        Alpha update requires a CPU-GPU sync to compute the entropy gradient.
        """
        comptime BATCH = Self.BATCH
        comptime OBS = Self.OBS
        comptime ACTIONS = Self.ACTIONS
        comptime ACTOR_OUT = Self.ACTOR_OUT
        comptime CRITIC_IN = Self.CRITIC_IN
        comptime CRITIC_CS = Self.CriticModel.CACHE_SIZE
        comptime ACTOR_CS = Self.ActorModel.CACHE_SIZE
        comptime TPB256 = 256
        comptime ELEM_BLOCKS = (BATCH * CRITIC_IN + TPB256 - 1) // TPB256
        comptime BATCH_BLOCKS = (BATCH + TPB256 - 1) // TPB256
        comptime ACT_BLOCKS = (BATCH * ACTIONS + TPB256 - 1) // TPB256

        self.train_step_count += 1

        # ----- Phase 1: Sample batch -----
        # Kernel uses BATCH seeds [seed, seed+BATCH-1]; stride must be >= BATCH
        gpu_state.buffer.sample[BATCH](
            ctx,
            rng_seed=UInt32(self.train_step_count) * UInt32(BATCH + 1),
            sampled_obs=gpu_state.s_obs,
            sampled_actions=gpu_state.s_act,
            sampled_rewards=gpu_state.s_rew,
            sampled_next_obs=gpu_state.s_nobs,
            sampled_dones=gpu_state.s_done,
            indices=gpu_state.s_idx,
        )

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

        var p_actor = gpu_state.actor.params_view()
        var p_c1 = gpu_state.critic1.online.params_view()
        var p_c2 = gpu_state.critic2.online.params_view()
        var p_c1t = gpu_state.critic1.target.params_view()
        var p_c2t = gpu_state.critic2.target.params_view()

        var log_std_min_s = Scalar[dtype](-5.0)
        var log_std_max_s = Scalar[dtype](2.0)
        # Each rsample kernel uses BATCH*ACTIONS seeds [base, base+BATCH*ACTIONS-1].
        # Stride between next-state and curr-state must be >= BATCH*ACTIONS to avoid
        # seed collision (same fix as PPO Bug 2 in MEMORY.md).
        var seed_stride = UInt32(BATCH * ACTIONS + 1)
        var next_rng_seed_s = Scalar[DType.uint32](
            UInt32(self.train_step_count) * seed_stride * 2
        )
        var curr_rng_seed_s = Scalar[DType.uint32](
            UInt32(self.train_step_count) * seed_stride * 2 + seed_stride
        )

        # ----- Phase 2: Actor forward on next_obs → next_actor_out -----
        # SAC uses CURRENT actor (no target actor)
        var next_actor_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTOR_OUT), MutAnyOrigin
        ](gpu_state.next_actor_out.unsafe_ptr())
        Self.ActorNet.forward_gpu[BATCH](
            ctx, nobs_t, next_actor_out_t, p_actor, gpu_state.actor_ws
        )

        # ----- Phase 3: sac_rsample next actions + log_probs -----
        var next_act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](gpu_state.next_act.unsafe_ptr())
        var next_lp_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.next_lp.unsafe_ptr())
        var eps_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
        ](gpu_state.eps_cache.unsafe_ptr())

        @always_inline
        fn next_rsample(
            acts: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            lp: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            eps: LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ],
            ao: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, ACTIONS + ACTIONS),
                MutAnyOrigin,
            ],
            lsmin: Scalar[dtype],
            lsmax: Scalar[dtype],
            rng_seed: Scalar[DType.uint32],
        ):
            sac_rsample_with_cache_kernel[dtype, BATCH, ACTIONS](
                acts, lp, eps, ao, lsmin, lsmax, rng_seed
            )

        ctx.enqueue_function[next_rsample, next_rsample](
            next_act_t,
            next_lp_t,
            eps_cache_t,
            next_actor_out_t,
            log_std_min_s,
            log_std_max_s,
            next_rng_seed_s,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB256,),
        )

        # ----- Phase 4: Concat(next_obs, next_act) → next_ci -----
        var next_ci_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
        ](gpu_state.next_ci.unsafe_ptr())

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
            next_act_t,
            grid_dim=(ELEM_BLOCKS,),
            block_dim=(TPB256,),
        )

        # ----- Phase 5: Both critic targets forward -----
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

        Self.CriticNet.forward_gpu[BATCH](
            ctx, next_ci_t, nq1_2d_t, p_c1t, gpu_state.critic1_ws
        )
        Self.CriticNet.forward_gpu[BATCH](
            ctx, next_ci_t, nq2_2d_t, p_c2t, gpu_state.critic2_ws
        )

        # ----- Phase 6: SAC TD targets with entropy bonus -----
        var targets_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](gpu_state.targets.unsafe_ptr())
        var gamma_s = Scalar[dtype](self.gamma)
        var alpha_s = Scalar[dtype](self.alpha)

        @always_inline
        fn sac_targets(
            tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            r: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            q1: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            q2: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            d: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            lp: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            g: Scalar[dtype],
            a: Scalar[dtype],
        ):
            td_target_min_twin_kernel[dtype, BATCH, True](
                tgt, r, q1, q2, d, lp, g, a
            )

        ctx.enqueue_function[sac_targets, sac_targets](
            targets_t,
            rew_t,
            nq1_t,
            nq2_t,
            done_t,
            next_lp_t,
            gamma_s,
            alpha_s,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB256,),
        )

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
            ctx, ci_t, q1_t, p_c1, q1_cache_t, gpu_state.critic1_ws
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
        )
        gpu_state.critic1.online.optimizer_step(ctx)

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
            ctx, ci_t, q2_t, p_c2, q2_cache_t, gpu_state.critic2_ws
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
        )
        gpu_state.critic2.online.optimizer_step(ctx)

        # ----- Phase 10: Delayed actor update (every policy_delay critic steps) -----
        # CleanRL uses policy_frequency=2: update actor + alpha only every 2 critic updates
        if self.train_step_count % self.policy_delay == 0:

            # 10a: Actor forward with cache on sampled obs → actor_out
            var actor_out_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTOR_OUT), MutAnyOrigin
            ](gpu_state.actor_out.unsafe_ptr())
            var actor_cache_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTOR_CS), MutAnyOrigin
            ](gpu_state.actor_cache.unsafe_ptr())

            Self.ActorNet.forward_gpu_with_cache[BATCH](
                ctx, obs_t, actor_out_t, p_actor, actor_cache_t, gpu_state.actor_ws
            )

            # 10b: sac_rsample with cache → curr_act, curr_lp, eps_cache (for backward)
            var curr_act_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ](gpu_state.curr_act.unsafe_ptr())
            var curr_lp_t = LayoutTensor[
                dtype, Layout.row_major(BATCH), MutAnyOrigin
            ](gpu_state.curr_lp.unsafe_ptr())

            @always_inline
            fn curr_rsample(
                acts: LayoutTensor[
                    dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
                ],
                lp: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
                eps: LayoutTensor[
                    dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
                ],
                ao: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, ACTIONS + ACTIONS),
                    MutAnyOrigin,
                ],
                lsmin: Scalar[dtype],
                lsmax: Scalar[dtype],
                rng_seed: Scalar[DType.uint32],
            ):
                sac_rsample_with_cache_kernel[dtype, BATCH, ACTIONS](
                    acts, lp, eps, ao, lsmin, lsmax, rng_seed
                )

            ctx.enqueue_function[curr_rsample, curr_rsample](
                curr_act_t,
                curr_lp_t,
                eps_cache_t,
                actor_out_t,
                log_std_min_s,
                log_std_max_s,
                curr_rng_seed_s,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB256,),
            )

            # 10c: Concat(obs, curr_act) → new_ci
            var new_ci_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
            ](gpu_state.new_ci.unsafe_ptr())

            @always_inline
            fn concat_new_ci(
                d: LayoutTensor[
                    dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
                ],
                o: LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin],
                a: LayoutTensor[
                    dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
                ],
            ):
                concat_obs_action_kernel[dtype, BATCH, OBS, ACTIONS](d, o, a)

            ctx.enqueue_function[concat_new_ci, concat_new_ci](
                new_ci_t,
                obs_t,
                curr_act_t,
                grid_dim=(ELEM_BLOCKS,),
                block_dim=(TPB256,),
            )

            # 10d: Both critics forward with cache for min(Q1, Q2) policy gradient
            var new_q_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ](gpu_state.new_q.unsafe_ptr())
            var new_q_cache_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_CS), MutAnyOrigin
            ](gpu_state.new_q_cache.unsafe_ptr())

            Self.CriticNet.forward_gpu_with_cache[BATCH](
                ctx, new_ci_t, new_q_t, p_c1, new_q_cache_t, gpu_state.critic1_ws
            )

            # Q2 forward on policy actions (reuse q2_out/q2_cache — Phase 9 is done)
            var new_q2_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ](gpu_state.q2_out.unsafe_ptr())
            var new_q2_cache_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_CS), MutAnyOrigin
            ](gpu_state.q2_cache.unsafe_ptr())

            Self.CriticNet.forward_gpu_with_cache[BATCH](
                ctx, new_ci_t, new_q2_t, p_c2, new_q2_cache_t, gpu_state.critic2_ws
            )

            # 10d2: min(Q1, Q2) mask → dq1 goes to dq, dq2 goes to q2_grad
            var dq_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ](gpu_state.dq.unsafe_ptr())
            var dq2_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ](gpu_state.q2_grad.unsafe_ptr())

            @always_inline
            fn min_q_mask(
                dq1: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
                dq2: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
                q1: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
                q2: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
            ):
                min_q_dq_kernel[dtype, BATCH](dq1, dq2, q1, q2)

            ctx.enqueue_function[min_q_mask, min_q_mask](
                dq_t,
                dq2_t,
                new_q_t,
                new_q2_t,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB256,),
            )

            # 10e: Backward Q1 with masked dq → d_new_ci
            var d_new_ci_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
            ](gpu_state.d_new_ci.unsafe_ptr())

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
            )
            # Intentionally NO optimizer_step here

            # 10e2: Backward Q2 with masked dq2 → d_ci2 (reuse buffer)
            var d_ci2_pg_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
            ](gpu_state.d_ci2.unsafe_ptr())

            var g_c2_pg = gpu_state.critic2.online.grads_view()
            gpu_state.critic2.online.zero_grads(ctx)
            Self.CriticNet.backward_gpu[BATCH](
                ctx,
                dq2_t,
                d_ci2_pg_t,
                p_c2,
                new_q2_cache_t,
                g_c2_pg,
                gpu_state.critic2_ws,
            )
            # Intentionally NO optimizer_step here

            # 10e3: d_new_ci += d_ci2 (combine gradients from both critics)
            @always_inline
            fn add_grads(
                dst: LayoutTensor[
                    dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
                ],
                src: LayoutTensor[
                    dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
                ],
            ):
                add_ci_grads_kernel[dtype, BATCH, CRITIC_IN](dst, src)

            ctx.enqueue_function[add_grads, add_grads](
                d_new_ci_t,
                d_ci2_pg_t,
                grid_dim=(ELEM_BLOCKS,),
                block_dim=(TPB256,),
            )

            # 10f: Extract action gradients from d_new_ci
            var grad_act_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
            ](gpu_state.grad_act.unsafe_ptr())

            @always_inline
            fn extract_act_grad(
                da: LayoutTensor[
                    dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
                ],
                dnc: LayoutTensor[
                    dtype, Layout.row_major(BATCH, CRITIC_IN), MutAnyOrigin
                ],
            ):
                actor_grad_from_critic_kernel[dtype, BATCH, OBS, ACTIONS](da, dnc)

            ctx.enqueue_function[extract_act_grad, extract_act_grad](
                grad_act_t,
                d_new_ci_t,
                grid_dim=(ACT_BLOCKS,),
                block_dim=(TPB256,),
            )

            # 10g: Backward through reparameterization → actor_grad [BATCH, ACTOR_OUT]
            var actor_grad_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, ACTOR_OUT), MutAnyOrigin
            ](gpu_state.actor_grad.unsafe_ptr())
            var alpha_per_sample = Scalar[dtype](self.alpha / Float64(BATCH))

            @always_inline
            fn rsample_bwd(
                agrad: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, ACTIONS + ACTIONS),
                    MutAnyOrigin,
                ],
                ga: LayoutTensor[
                    dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
                ],
                aps: Scalar[dtype],
                ca: LayoutTensor[
                    dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
                ],
                eps: LayoutTensor[
                    dtype, Layout.row_major(BATCH, ACTIONS), MutAnyOrigin
                ],
                ao: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, ACTIONS + ACTIONS),
                    MutAnyOrigin,
                ],
                lsmin: Scalar[dtype],
                lsmax: Scalar[dtype],
            ):
                sac_rsample_bwd_kernel[dtype, BATCH, ACTIONS](
                    agrad, ga, aps, ca, eps, ao, lsmin, lsmax
                )

            ctx.enqueue_function[rsample_bwd, rsample_bwd](
                actor_grad_t,
                grad_act_t,
                alpha_per_sample,
                curr_act_t,
                eps_cache_t,
                actor_out_t,
                log_std_min_s,
                log_std_max_s,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB256,),
            )

            # 10h: Actor backward + optimizer step
            var d_obs_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
            ](gpu_state.d_obs.unsafe_ptr())

            var g_actor = gpu_state.actor.grads_view()
            gpu_state.actor.zero_grads(ctx)
            Self.ActorNet.backward_gpu[BATCH](
                ctx,
                actor_grad_t,
                d_obs_t,
                p_actor,
                actor_cache_t,
                g_actor,
                gpu_state.actor_ws,
            )
            gpu_state.actor.optimizer_step(ctx)

            # ----- Phase 11: Alpha update via CPU-GPU sync -----
            if self.auto_alpha:
                ctx.synchronize()
                var lp_host = ctx.enqueue_create_host_buffer[dtype](BATCH)
                ctx.enqueue_copy(lp_host, gpu_state.curr_lp)
                ctx.synchronize()

                var mean_lp: Float64 = 0.0
                for b in range(BATCH):
                    mean_lp += Float64(lp_host[b])
                mean_lp /= Float64(BATCH)

                # CleanRL: alpha_loss = (-exp(log_alpha) * (log_pi + target_entropy)).mean()
                # ∂loss/∂log_alpha = -alpha * mean(log_pi + target_entropy)
                var grad = -self.alpha * (mean_lp + self.target_entropy)

                # Adam update for log_alpha (matches CleanRL's a_optimizer)
                self.alpha_adam_t += 1
                var beta1: Float64 = 0.9
                var beta2: Float64 = 0.999
                var eps: Float64 = 1e-8
                self.alpha_adam_m = beta1 * self.alpha_adam_m + (1.0 - beta1) * grad
                self.alpha_adam_v = beta2 * self.alpha_adam_v + (1.0 - beta2) * grad * grad
                var m_hat = self.alpha_adam_m / (1.0 - beta1 ** Float64(self.alpha_adam_t))
                var v_hat = self.alpha_adam_v / (1.0 - beta2 ** Float64(self.alpha_adam_t))
                self.log_alpha -= self.alpha_lr * m_hat / (sqrt(v_hat) + eps)
                self.alpha = exp(self.log_alpha)

    fn get_action_scale(self) -> Float64:
        return self.action_scale

    fn get_total_steps(self) -> Int:
        return self.total_steps

    fn set_total_steps(mut self, steps: Int):
        self.total_steps = steps

    fn soft_update_targets_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
    ) raises -> None:
        """Soft-update target critic networks on GPU.

        SAC has NO target actor — only critic targets are updated.
        Target networks are updated every step (CleanRL target_network_frequency=1).
        """
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
        environment_name: String = "Environment",
    ) raises -> TrainingMetrics:
        """Train on GPU using the shared off-policy GPU loop.

        GPU state (networks, replay buffer, scratch buffers) is created
        locally for the duration of training and freed when the method returns.
        After this call self.state.actor / critic1 / critic2 hold the trained
        GPU weights (synced by download_from_gpu).

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
            environment_name: Name for metrics labeling.

        Returns:
            TrainingMetrics with episode-level statistics.
        """
        var checkpoint_path = self.checkpoint_path
        var checkpoint_every = self.checkpoint_every
        return run_offpolicy_continuous_train_gpu[E, Self](
            self,
            ctx,
            num_steps,
            warmup_steps=warmup_steps,
            gradient_steps=gradient_steps,
            sync_every=sync_every,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            algorithm_name="SAC (GPU)",
            checkpoint_every=checkpoint_every,
            checkpoint_path=checkpoint_path,
        )

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
    ) raises -> TrainingMetrics:
        """Train the SAC agent on a continuous action environment.

        Delegates to run_offpolicy_continuous_train which handles warmup,
        episode loop, and metric logging.

        Args:
            env: Environment implementing BoxContinuousActionEnv.
            num_episodes: Number of training episodes.
            max_steps_per_episode: Maximum steps per episode (default: 200).
            warmup_steps: Random steps to pre-fill replay buffer (default: 1000).
            train_every: Train every N steps (default: 1).
            verbose: Print progress (default: False).
            print_every: Print every N episodes if verbose (default: 10).
            environment_name: Name for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
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
            environment_name=environment_name,
            algorithm_name="SAC (CPU)",
            checkpoint_every=checkpoint_every,
            checkpoint_path=checkpoint_path,
        )
        self.state = cpu_state^
        return metrics

    # =========================================================================
    # Evaluation (deterministic policy: use mean action)
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
        """Evaluate the agent using the deterministic mean action (no sampling).

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
            algorithm_name="SAC",
        ).mean_reward()

    # =========================================================================
    # Checkpoint Save / Load
    # =========================================================================

    fn save_checkpoint(self, filepath: String) raises:
        """Save agent state to a checkpoint file.

        Saves actor, critic1 (online+target), and critic2 (online+target)
        params and optimizer states, plus hyperparameters.
        The replay buffer is NOT saved.

        Args:
            filepath: Destination path (e.g. "sac_agent.ckpt").
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
            "sac_agent",
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
        metadata.append("alpha=" + String(self.alpha))
        metadata.append("log_alpha=" + String(self.log_alpha))
        metadata.append("target_entropy=" + String(self.target_entropy))
        metadata.append("alpha_lr=" + String(self.alpha_lr))
        metadata.append("auto_alpha=" + String(Int(self.auto_alpha)))
        metadata.append("policy_delay=" + String(self.policy_delay))
        metadata.append("total_steps=" + String(self.total_steps))
        metadata.append("train_step_count=" + String(self.train_step_count))
        metadata.append("alpha_adam_m=" + String(self.alpha_adam_m))
        metadata.append("alpha_adam_v=" + String(self.alpha_adam_v))
        metadata.append("alpha_adam_t=" + String(self.alpha_adam_t))
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

        set_metadata_value_float(metadata, "gamma", self.gamma)
        set_metadata_value_float(metadata, "tau", self.tau)
        set_metadata_value_float(metadata, "action_scale", self.action_scale)
        set_metadata_value_float(metadata, "alpha", self.alpha)
        set_metadata_value_float(metadata, "log_alpha", self.log_alpha)
        set_metadata_value_float(
            metadata, "target_entropy", self.target_entropy
        )
        set_metadata_value_float(metadata, "alpha_lr", self.alpha_lr)
        set_metadata_value_bool(metadata, "auto_alpha", self.auto_alpha)
        set_metadata_value_int(metadata, "policy_delay", self.policy_delay)
        set_metadata_value_int(metadata, "total_steps", self.total_steps)
        set_metadata_value_int(
            metadata, "train_step_count", self.train_step_count
        )
        set_metadata_value_float(
            metadata, "alpha_adam_m", self.alpha_adam_m
        )
        set_metadata_value_float(
            metadata, "alpha_adam_v", self.alpha_adam_v
        )
        set_metadata_value_int(
            metadata, "alpha_adam_t", self.alpha_adam_t
        )
