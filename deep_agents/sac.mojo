"""Deep SAC Agent using the new trait-based deep learning architecture.

This SAC (Soft Actor-Critic) implementation uses:
- NetworkState for heap-allocated params + grads + optimizer state
- Network (all-static) for stateless forward/backward ops via LayoutTensor
- Sequential composition with StochasticActor output layer
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

from math import exp, log, sqrt
from layout import Layout, LayoutTensor

from nn.constants import dtype, TILE, TPB
from nn.model import Model, Linear, LinearReLU, Sequential, StochasticActor
from nn.model.stochastic_actor import (
    rsample,
    rsample_with_cache,
    rsample_backward,
    sample_action,
    get_deterministic_action,
)
from nn.optimizer import Optimizer, Adam
from nn.initializer import Kaiming
from nn.training import Network, NetworkState, NetworkPair
from nn.utils import obs_to_inline, concat_obs_action_batch
from nn.replay import ReplayBuffer
from deep_agents.offpolicy_helpers import (
    store_continuous_transition,
    random_continuous_action,
)
from nn.gpu.random import gaussian_noise
from nn.checkpoint import (
    write_checkpoint_header,
    write_metadata_section,
    parse_checkpoint_header,
    read_checkpoint_file,
    read_metadata_section,
    get_metadata_value,
    save_checkpoint_file,
)
from core import (
    TrainingMetrics,
    BoxContinuousActionEnv,
    OffPolicyAgent,
    run_offpolicy_continuous_train,
    run_offpolicy_continuous_eval,
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
](OffPolicyAgent):
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
    """

    # Convenience compile-time aliases
    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.action_dim
    comptime HIDDEN = Self.hidden_dim
    comptime BATCH = Self.batch_size

    # StochasticActor outputs mean + log_std
    comptime ACTOR_OUT = Self.ACTIONS * 2

    # Critic input dimension: obs + action concatenated
    comptime CRITIC_IN = Self.OBS + Self.ACTIONS

    # Actor: obs → hidden (ReLU) → hidden (ReLU) → StochasticActor (mean + log_std)
    comptime ActorModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        StochasticActor[Self.HIDDEN, Self.ACTIONS],
    ]
    comptime ActorNet = Network[Self.ActorModel, Adam[Self.actor_lr]]

    # Critic: (obs ‖ action) → hidden (ReLU) → hidden (ReLU) → Q-value
    comptime CriticModel = Sequential[
        LinearReLU[Self.CRITIC_IN, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]
    comptime CriticNet = Network[Self.CriticModel, Adam[Self.critic_lr]]

    # Network states (heap-allocated)
    # Note: SAC has NO target actor — only target critics
    var actor: NetworkState[Self.ActorModel, Adam[Self.actor_lr]]
    var critic1: NetworkPair[Self.CriticModel, Adam[Self.critic_lr]]
    var critic2: NetworkPair[Self.CriticModel, Adam[Self.critic_lr]]

    # Replay buffer
    var buffer: ReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
    ]

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

    # Training state
    var total_steps: Int
    var train_step_count: Int

    # Auto-checkpoint settings
    var checkpoint_every: Int
    var checkpoint_path: String

    # Pre-allocated train_step scratch (heap, avoids per-call stack allocation)
    var _batch_obs: List[Scalar[dtype]]
    var _batch_act: List[Scalar[dtype]]
    var _batch_rew: List[Scalar[dtype]]
    var _batch_next: List[Scalar[dtype]]
    var _batch_done: List[Scalar[dtype]]
    var _next_out: List[Scalar[dtype]]        # BATCH * ACTOR_OUT (2*ACTIONS)
    var _next_act: List[Scalar[dtype]]        # BATCH * ACTIONS
    var _next_log_pi: List[Scalar[dtype]]     # BATCH * 1
    var _next_ci: List[Scalar[dtype]]         # BATCH * CRITIC_IN
    var _nq1: List[Scalar[dtype]]             # BATCH * 1
    var _nq2: List[Scalar[dtype]]             # BATCH * 1
    var _targets: List[Scalar[dtype]]         # BATCH
    var _ci: List[Scalar[dtype]]              # BATCH * CRITIC_IN
    var _q1_out: List[Scalar[dtype]]          # BATCH * 1
    var _q2_out: List[Scalar[dtype]]          # BATCH * 1
    var _q1_cache: List[Scalar[dtype]]        # BATCH * CriticModel.CACHE_SIZE
    var _q2_cache: List[Scalar[dtype]]        # BATCH * CriticModel.CACHE_SIZE
    var _q_grad: List[Scalar[dtype]]          # BATCH * 1 (reused for q1_grad and q2_grad)
    var _d_ci: List[Scalar[dtype]]            # BATCH * CRITIC_IN (reused for d_c1 and d_c2)
    var _curr_out: List[Scalar[dtype]]        # BATCH * ACTOR_OUT
    var _curr_act: List[Scalar[dtype]]        # BATCH * ACTIONS
    var _curr_log_pi: List[Scalar[dtype]]     # BATCH * 1
    var _actor_cache: List[Scalar[dtype]]     # BATCH * ActorModel.CACHE_SIZE
    var _new_ci: List[Scalar[dtype]]          # BATCH * CRITIC_IN
    var _new_q1: List[Scalar[dtype]]          # BATCH * 1
    var _new_c1_cache: List[Scalar[dtype]]    # BATCH * CriticModel.CACHE_SIZE
    var _actor_grad_arr: List[Scalar[dtype]]  # BATCH * ACTOR_OUT
    var _d_new_ci: List[Scalar[dtype]]        # BATCH * CRITIC_IN
    var _grad_act: List[Scalar[dtype]]        # BATCH * ACTIONS
    var _d_obs: List[Scalar[dtype]]           # BATCH * OBS

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        action_scale: Float64 = 1.0,
        alpha: Float64 = 0.2,
        auto_alpha: Bool = True,
        alpha_lr: Float64 = 0.0003,
        target_entropy: Float64 = -1.0,
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
            checkpoint_every: Save checkpoint every N episodes (0 to disable).
            checkpoint_path: Path to save checkpoints.
        """
        self.actor = NetworkState[Self.ActorModel, Adam[Self.actor_lr]]()
        self.actor.initialize[Kaiming]()

        self.critic1 = NetworkPair[Self.CriticModel, Adam[Self.critic_lr]]()
        self.critic1.initialize[Kaiming]()
        self.critic2 = NetworkPair[Self.CriticModel, Adam[Self.critic_lr]]()
        self.critic2.initialize[Kaiming]()

        self.buffer = ReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
        ]()

        self.gamma = gamma
        self.tau = tau
        self.action_scale = action_scale
        self.alpha = alpha
        self.log_alpha = log(alpha)
        self.target_entropy = target_entropy
        self.alpha_lr = alpha_lr
        self.auto_alpha = auto_alpha
        self.total_steps = 0
        self.train_step_count = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

        self._batch_obs = List[Scalar[dtype]](capacity=Self.BATCH * Self.OBS)
        self._batch_act = List[Scalar[dtype]](capacity=Self.BATCH * Self.ACTIONS)
        self._batch_rew = List[Scalar[dtype]](capacity=Self.BATCH)
        self._batch_next = List[Scalar[dtype]](capacity=Self.BATCH * Self.OBS)
        self._batch_done = List[Scalar[dtype]](capacity=Self.BATCH)
        self._next_out = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.ACTOR_OUT
        )
        self._next_act = List[Scalar[dtype]](capacity=Self.BATCH * Self.ACTIONS)
        self._next_log_pi = List[Scalar[dtype]](capacity=Self.BATCH)
        self._next_ci = List[Scalar[dtype]](capacity=Self.BATCH * Self.CRITIC_IN)
        self._nq1 = List[Scalar[dtype]](capacity=Self.BATCH)
        self._nq2 = List[Scalar[dtype]](capacity=Self.BATCH)
        self._targets = List[Scalar[dtype]](capacity=Self.BATCH)
        self._ci = List[Scalar[dtype]](capacity=Self.BATCH * Self.CRITIC_IN)
        self._q1_out = List[Scalar[dtype]](capacity=Self.BATCH)
        self._q2_out = List[Scalar[dtype]](capacity=Self.BATCH)
        self._q1_cache = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.CriticModel.CACHE_SIZE
        )
        self._q2_cache = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.CriticModel.CACHE_SIZE
        )
        self._q_grad = List[Scalar[dtype]](capacity=Self.BATCH)
        self._d_ci = List[Scalar[dtype]](capacity=Self.BATCH * Self.CRITIC_IN)
        self._curr_out = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.ACTOR_OUT
        )
        self._curr_act = List[Scalar[dtype]](capacity=Self.BATCH * Self.ACTIONS)
        self._curr_log_pi = List[Scalar[dtype]](capacity=Self.BATCH)
        self._actor_cache = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.ActorModel.CACHE_SIZE
        )
        self._new_ci = List[Scalar[dtype]](capacity=Self.BATCH * Self.CRITIC_IN)
        self._new_q1 = List[Scalar[dtype]](capacity=Self.BATCH)
        self._new_c1_cache = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.CriticModel.CACHE_SIZE
        )
        self._actor_grad_arr = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.ACTOR_OUT
        )
        self._d_new_ci = List[Scalar[dtype]](
            capacity=Self.BATCH * Self.CRITIC_IN
        )
        self._grad_act = List[Scalar[dtype]](capacity=Self.BATCH * Self.ACTIONS)
        self._d_obs = List[Scalar[dtype]](capacity=Self.BATCH * Self.OBS)
        for _ in range(Self.BATCH * Self.OBS):
            self._batch_obs.append(Scalar[dtype](0))
            self._batch_next.append(Scalar[dtype](0))
            self._d_obs.append(Scalar[dtype](0))
        for _ in range(Self.BATCH * Self.ACTIONS):
            self._batch_act.append(Scalar[dtype](0))
            self._next_act.append(Scalar[dtype](0))
            self._curr_act.append(Scalar[dtype](0))
            self._grad_act.append(Scalar[dtype](0))
        for _ in range(Self.BATCH):
            self._batch_rew.append(Scalar[dtype](0))
            self._batch_done.append(Scalar[dtype](0))
            self._next_log_pi.append(Scalar[dtype](0))
            self._nq1.append(Scalar[dtype](0))
            self._nq2.append(Scalar[dtype](0))
            self._targets.append(Scalar[dtype](0))
            self._q1_out.append(Scalar[dtype](0))
            self._q2_out.append(Scalar[dtype](0))
            self._q_grad.append(Scalar[dtype](0))
            self._curr_log_pi.append(Scalar[dtype](0))
            self._new_q1.append(Scalar[dtype](0))
        for _ in range(Self.BATCH * Self.ACTOR_OUT):
            self._next_out.append(Scalar[dtype](0))
            self._curr_out.append(Scalar[dtype](0))
            self._actor_grad_arr.append(Scalar[dtype](0))
        for _ in range(Self.BATCH * Self.CRITIC_IN):
            self._next_ci.append(Scalar[dtype](0))
            self._ci.append(Scalar[dtype](0))
            self._d_ci.append(Scalar[dtype](0))
            self._new_ci.append(Scalar[dtype](0))
            self._d_new_ci.append(Scalar[dtype](0))
        for _ in range(Self.BATCH * Self.CriticModel.CACHE_SIZE):
            self._q1_cache.append(Scalar[dtype](0))
            self._q2_cache.append(Scalar[dtype](0))
            self._new_c1_cache.append(Scalar[dtype](0))
        for _ in range(Self.BATCH * Self.ActorModel.CACHE_SIZE):
            self._actor_cache.append(Scalar[dtype](0))

    # =========================================================================
    # OffPolicyAgent trait — required methods
    # =========================================================================

    fn select_action_list(
        mut self, obs: List[Float64]
    ) -> List[Float64]:
        """Select action using the stochastic policy (with reparameterization).

        SAC uses the inherently stochastic policy for exploration — no external
        noise is needed.

        Args:
            obs: Observation as List[Float64].

        Returns:
            Action list of length action_dim, scaled by action_scale.
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

        var p = self.actor.params_view()
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
            var ls = Float64(out_arr[Self.ACTIONS + i])
            if m != m:
                m = 0.0
            elif m > 10.0:
                m = 10.0
            elif m < -10.0:
                m = -10.0
            if ls != ls:
                ls = -1.0
            elif ls > 2.0:
                ls = 2.0
            elif ls < -5.0:
                ls = -5.0
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
        var lp_t = LayoutTensor[
            dtype, Layout.row_major(1, 1), MutAnyOrigin
        ](lp_arr.unsafe_ptr())

        sample_action[1, Self.ACTIONS](mean_t, log_std_t, noise_t, act_t)

        var result = List[Float64](capacity=Self.action_dim)
        for i in range(Self.action_dim):
            result.append(Float64(act_arr[i]) * self.action_scale)
        return result^

    fn store_list_transition(
        mut self,
        obs: List[Float64],
        action: List[Float64],
        reward: Float64,
        next_obs: List[Float64],
        done: Bool,
    ) -> None:
        """Store transition in the replay buffer.

        Actions are stored unscaled (divided by action_scale).
        """
        store_continuous_transition[Self.OBS, Self.ACTIONS, Self.buffer_capacity](
            self.buffer,
            obs,
            action,
            reward,
            next_obs,
            done,
            self.action_scale,
            self.total_steps,
        )

    fn is_ready(self) -> Bool:
        """Return True if buffer has enough samples."""
        return self.buffer.is_ready[Self.BATCH]()

    fn do_train_step(mut self) -> Float64:
        """Perform one SAC gradient update step.

        Returns:
            Average critic loss value.
        """
        return self.train_step()

    fn decay_explore(mut self) -> None:
        """SAC uses stochastic policy for exploration — no noise to decay.

        Alpha auto-tuning is handled inside train_step.
        """
        pass

    fn get_explore_rate(self) -> Float64:
        """Return current entropy coefficient alpha as exploration measure."""
        return self.alpha

    fn random_action_list(self) -> List[Float64]:
        """Return a uniformly random action in [-action_scale, action_scale]."""
        return random_continuous_action(Self.action_dim, self.action_scale)

    fn select_greedy_action_list(
        self, obs: List[Float64]
    ) -> List[Float64]:
        """Select action using deterministic mean policy (no reparameterization noise).

        Used for evaluation. Applies tanh(mean) as the deterministic action
        instead of sampling from the Gaussian distribution.

        Args:
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

        var p = self.actor.params_view()
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

    fn train_step(mut self) -> Float64:
        """Perform one SAC training step.

        Updates:
        1. Both critics using TD error with min(Q1, Q2) + entropy targets
        2. Actor using policy gradient (maximize Q - alpha * log_pi)
        3. Alpha (if auto_alpha=True)
        4. Soft update target critics

        Returns:
            Average critic loss, or 0.0 if buffer not ready.
        """
        if not self.buffer.is_ready[Self.BATCH]():
            return 0.0

        # =================================================================
        # Phase 1: Sample batch
        # These 5 must remain local InlineArrays — ReplayBuffer.sample takes mut InlineArray
        # =================================================================
        var batch_obs = InlineArray[
            Scalar[dtype], Self.BATCH * Self.OBS
        ](uninitialized=True)
        var batch_act = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        var batch_rew = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var batch_next = InlineArray[
            Scalar[dtype], Self.BATCH * Self.OBS
        ](uninitialized=True)
        var batch_done = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )

        self.buffer.sample[Self.BATCH](
            batch_obs, batch_act, batch_rew, batch_next, batch_done
        )

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](batch_obs.unsafe_ptr())
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](batch_next.unsafe_ptr())

        # =================================================================
        # Phase 2: Compute TD targets
        # y = r + γ * (min(Q1_t, Q2_t)(s', a') - α * log_π(a'|s')) * (1 - done)
        # where a' ~ π(·|s') (current actor, reparameterization)
        # =================================================================

        # Forward actor on next_obs to get next mean + log_std
        var next_out_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTOR_OUT), MutAnyOrigin
        ](self._next_out.unsafe_ptr())
        var p_actor = self.actor.params_view()
        Self.ActorNet.forward[Self.BATCH](next_obs_t, next_out_t, p_actor)

        # Extract and clamp mean + log_std for next states
        # These remain local — small (BATCH*ACTIONS), needed only within this phase
        var next_mean_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        var next_ls_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        for b in range(Self.BATCH):
            for a in range(Self.ACTIONS):
                var m = Float64(self._next_out[b * Self.ACTOR_OUT + a])
                var ls = Float64(
                    self._next_out[b * Self.ACTOR_OUT + Self.ACTIONS + a]
                )
                if m != m:
                    m = 0.0
                elif m > 10.0:
                    m = 10.0
                elif m < -10.0:
                    m = -10.0
                if ls != ls:
                    ls = -1.0
                elif ls > 2.0:
                    ls = 2.0
                elif ls < -5.0:
                    ls = -5.0
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
        ](self._next_act.unsafe_ptr())
        var next_lp_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._next_log_pi.unsafe_ptr())

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
            var lp = Float64(self._next_log_pi[b])
            if lp != lp or lp > 100.0 or lp < -100.0:
                self._next_log_pi[b] = Scalar[dtype](-1.0)

        # Build next critic input: concat(batch_next, _next_act) via manual loop
        var next_ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](self._next_ci.unsafe_ptr())
        for b in range(Self.BATCH):
            for i in range(Self.OBS):
                self._next_ci[b * Self.CRITIC_IN + i] = batch_next[
                    b * Self.OBS + i
                ]
            for i in range(Self.ACTIONS):
                self._next_ci[b * Self.CRITIC_IN + Self.OBS + i] = (
                    self._next_act[b * Self.ACTIONS + i]
                )

        # Forward both target critics
        var nq1_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._nq1.unsafe_ptr())
        var nq2_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._nq2.unsafe_ptr())

        var p_c1t = self.critic1.target.params_view()
        var p_c2t = self.critic2.target.params_view()
        Self.CriticNet.forward[Self.BATCH](next_ci_t, nq1_t, p_c1t)
        Self.CriticNet.forward[Self.BATCH](next_ci_t, nq2_t, p_c2t)

        # TD targets: r + γ * (min(Q1,Q2) - α * log_π) * (1 - done)
        for b in range(Self.BATCH):
            var q1 = Float64(self._nq1[b])
            var q2 = Float64(self._nq2[b])
            if q1 != q1:
                q1 = 0.0
            if q2 != q2:
                q2 = 0.0
            var min_q = q1 if q1 < q2 else q2
            var lp = Float64(self._next_log_pi[b])
            var done_mask = 1.0 - Float64(batch_done[b])
            var tgt = Float64(batch_rew[b]) + self.gamma * (
                min_q - self.alpha * lp
            ) * done_mask
            if tgt != tgt:
                tgt = 0.0
            elif tgt > 1000.0:
                tgt = 1000.0
            elif tgt < -1000.0:
                tgt = -1000.0
            self._targets[b] = Scalar[dtype](tgt)

        # =================================================================
        # Phase 3: Update Both Critics
        # =================================================================

        # Build critic input: concat(batch_obs, batch_act) via manual loop
        var ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](self._ci.unsafe_ptr())
        for b in range(Self.BATCH):
            for i in range(Self.OBS):
                self._ci[b * Self.CRITIC_IN + i] = batch_obs[b * Self.OBS + i]
            for i in range(Self.ACTIONS):
                self._ci[b * Self.CRITIC_IN + Self.OBS + i] = batch_act[
                    b * Self.ACTIONS + i
                ]

        # --- Critic 1 ---
        var q1_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._q1_out.unsafe_ptr())
        var c1_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.CriticModel.CACHE_SIZE),
            MutAnyOrigin,
        ](self._q1_cache.unsafe_ptr())

        var p_c1 = self.critic1.params_view()
        Self.CriticNet.forward_with_cache[Self.BATCH](
            ci_t, q1_t, p_c1, c1_cache_t
        )

        var q1_grad_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._q_grad.unsafe_ptr())
        var critic1_loss: Float64 = 0.0
        for b in range(Self.BATCH):
            var td_err = self._q1_out[b] - self._targets[b]
            critic1_loss += Float64(td_err * td_err)
            self._q_grad[b] = (
                Scalar[dtype](2.0) * td_err / Scalar[dtype](Self.BATCH)
            )
        critic1_loss /= Float64(Self.BATCH)

        var d_c1_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](self._d_ci.unsafe_ptr())

        var g_c1 = self.critic1.grads_view()
        self.critic1.zero_grads()
        Self.CriticNet.backward[Self.BATCH](
            q1_grad_t, d_c1_t, p_c1, c1_cache_t, g_c1
        )
        self.critic1.optimizer_step()

        # --- Critic 2 ---
        var q2_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._q2_out.unsafe_ptr())
        var c2_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.CriticModel.CACHE_SIZE),
            MutAnyOrigin,
        ](self._q2_cache.unsafe_ptr())

        var p_c2 = self.critic2.params_view()
        Self.CriticNet.forward_with_cache[Self.BATCH](
            ci_t, q2_t, p_c2, c2_cache_t
        )

        var q2_grad_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._q_grad.unsafe_ptr())
        var critic2_loss: Float64 = 0.0
        for b in range(Self.BATCH):
            var td_err = self._q2_out[b] - self._targets[b]
            critic2_loss += Float64(td_err * td_err)
            self._q_grad[b] = (
                Scalar[dtype](2.0) * td_err / Scalar[dtype](Self.BATCH)
            )
        critic2_loss /= Float64(Self.BATCH)

        var d_c2_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](self._d_ci.unsafe_ptr())

        var g_c2 = self.critic2.grads_view()
        self.critic2.zero_grads()
        Self.CriticNet.backward[Self.BATCH](
            q2_grad_t, d_c2_t, p_c2, c2_cache_t, g_c2
        )
        self.critic2.optimizer_step()

        var avg_critic_loss = (critic1_loss + critic2_loss) / 2.0

        # =================================================================
        # Phase 4: Update Actor
        # J_π = E[α * log_π(a|s) - Q(s, a)]  →  minimize (gradient descent)
        # =================================================================

        # Step 1: Forward actor with cache → mean + log_std
        var actor_out_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTOR_OUT), MutAnyOrigin
        ](self._curr_out.unsafe_ptr())
        var actor_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.ActorModel.CACHE_SIZE),
            MutAnyOrigin,
        ](self._actor_cache.unsafe_ptr())

        Self.ActorNet.forward_with_cache[Self.BATCH](
            obs_t, actor_out_t, p_actor, actor_cache_t
        )

        # Extract and clamp mean + log_std — remain local (temporary per step)
        var curr_mean_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        var curr_ls_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        for b in range(Self.BATCH):
            for a in range(Self.ACTIONS):
                var m = Float64(self._curr_out[b * Self.ACTOR_OUT + a])
                var ls = Float64(
                    self._curr_out[b * Self.ACTOR_OUT + Self.ACTIONS + a]
                )
                if m != m:
                    m = 0.0
                elif m > 10.0:
                    m = 10.0
                elif m < -10.0:
                    m = -10.0
                if ls != ls:
                    ls = -1.0
                elif ls > 2.0:
                    ls = 2.0
                elif ls < -5.0:
                    ls = -5.0
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
        ](self._curr_act.unsafe_ptr())
        var curr_lp_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._curr_log_pi.unsafe_ptr())
        var z_cache_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)

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
            var lp = Float64(self._curr_log_pi[b])
            if lp != lp or lp > 100.0 or lp < -100.0:
                self._curr_log_pi[b] = Scalar[dtype](-1.0)

        # Step 3: Build critic input with sampled actions: concat(batch_obs, _curr_act) via manual loop
        var new_ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](self._new_ci.unsafe_ptr())
        for b in range(Self.BATCH):
            for i in range(Self.OBS):
                self._new_ci[b * Self.CRITIC_IN + i] = batch_obs[
                    b * Self.OBS + i
                ]
            for i in range(Self.ACTIONS):
                self._new_ci[b * Self.CRITIC_IN + Self.OBS + i] = (
                    self._curr_act[b * Self.ACTIONS + i]
                )

        # Step 4: Forward critic1 with cache (need for actor backward)
        var new_q_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._new_q1.unsafe_ptr())
        var new_c1_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.CriticModel.CACHE_SIZE),
            MutAnyOrigin,
        ](self._new_c1_cache.unsafe_ptr())

        Self.CriticNet.forward_with_cache[Self.BATCH](
            new_ci_t, new_q_t, p_c1, new_c1_cache_t
        )

        # Step 5: Backward through critic1 to get dQ/da (-1/BATCH per sample)
        var dq_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](self._q_grad.unsafe_ptr())
        for b in range(Self.BATCH):
            self._q_grad[b] = Scalar[dtype](-1.0 / Float64(Self.BATCH))

        var d_new_ci_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.CRITIC_IN), MutAnyOrigin
        ](self._d_new_ci.unsafe_ptr())

        # Backward through critic to get action gradient — don't update critic
        self.critic1.zero_grads()
        Self.CriticNet.backward[Self.BATCH](
            dq_t, d_new_ci_t, p_c1, new_c1_cache_t, g_c1
        )
        # Intentionally NOT calling critic1.optimizer_step() here

        # Extract action gradients from d_new_ci (last ACTIONS columns per row)
        var grad_act_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](self._grad_act.unsafe_ptr())
        for b in range(Self.BATCH):
            for a in range(Self.ACTIONS):
                self._grad_act[b * Self.ACTIONS + a] = self._d_new_ci[
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
        # These remain local (small, used only for rsample_backward call below)
        var grad_mean_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        var grad_ls_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)

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
        # Actor output layout: [mean (ACTIONS) | log_std (ACTIONS)]
        var actor_grad_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTOR_OUT), MutAnyOrigin
        ](self._actor_grad_arr.unsafe_ptr())
        for b in range(Self.BATCH):
            for a in range(Self.ACTIONS):
                self._actor_grad_arr[b * Self.ACTOR_OUT + a] = grad_mean_arr[
                    b * Self.ACTIONS + a
                ]
                self._actor_grad_arr[
                    b * Self.ACTOR_OUT + Self.ACTIONS + a
                ] = grad_ls_arr[b * Self.ACTIONS + a]

        # Step 9: Backward through actor network
        var d_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](self._d_obs.unsafe_ptr())

        var g_actor = self.actor.grads_view()
        self.actor.zero_grads()
        Self.ActorNet.backward[Self.BATCH](
            actor_grad_t, d_obs_t, p_actor, actor_cache_t, g_actor
        )
        self.actor.optimizer_step()

        # =================================================================
        # Phase 5: Update Alpha (if auto_alpha)
        # =================================================================
        if self.auto_alpha:
            # J(α) = E[α * (log_π + target_entropy)]
            # Gradient descent on log_α: log_α -= α_lr * mean(log_π + target_entropy)
            var alpha_grad: Float64 = 0.0
            for b in range(Self.BATCH):
                alpha_grad += Float64(self._curr_log_pi[b]) + self.target_entropy
            alpha_grad /= Float64(Self.BATCH)

            self.log_alpha -= self.alpha_lr * alpha_grad
            if self.log_alpha < -5.0:
                self.log_alpha = -5.0
            elif self.log_alpha > 2.0:
                self.log_alpha = 2.0
            self.alpha = exp(self.log_alpha)

        # =================================================================
        # Phase 6: Soft Update Target Critics
        # =================================================================
        self.critic1.soft_update(self.tau)
        self.critic2.soft_update(self.tau)

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
    ) -> TrainingMetrics:
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
        return run_offpolicy_continuous_train(
            self,
            env,
            num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            warmup_steps=warmup_steps,
            train_every=train_every,
            verbose=verbose,
            print_every=print_every,
            environment_name=environment_name,
            algorithm_name="Deep SAC",
        )

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
            env,
            num_episodes=num_episodes,
            max_steps=max_steps,
            verbose=verbose,
            algorithm_name="Deep SAC",
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
        comptime ACTOR_STATE_SIZE = ACTOR_PARAM_SIZE * Adam[Self.actor_lr].STATE_PER_PARAM
        comptime CRITIC_STATE_SIZE = CRITIC_PARAM_SIZE * Adam[Self.critic_lr].STATE_PER_PARAM

        var content = write_checkpoint_header(
            "sac_agent",
            ACTOR_PARAM_SIZE + 2 * CRITIC_PARAM_SIZE,
            ACTOR_STATE_SIZE + 2 * CRITIC_STATE_SIZE,
        )
        content += self.actor.write_sections("actor_")
        content += self.critic1.write_sections("critic1_")
        content += self.critic2.write_sections("critic2_")

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
        metadata.append("total_steps=" + String(self.total_steps))
        metadata.append("train_step_count=" + String(self.train_step_count))
        content += write_metadata_section(metadata)

        save_checkpoint_file(filepath, content)

    fn load_checkpoint(mut self, filepath: String) raises:
        """Load agent state from a checkpoint file.

        Args:
            filepath: Path to the checkpoint file.
        """
        var content = read_checkpoint_file(filepath)

        self.actor.read_sections(content, "actor_")
        self.critic1.read_sections(content, "critic1_")
        self.critic2.read_sections(content, "critic2_")

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

        var alpha_str = get_metadata_value(metadata, "alpha")
        if len(alpha_str) > 0:
            self.alpha = atof(alpha_str)

        var log_alpha_str = get_metadata_value(metadata, "log_alpha")
        if len(log_alpha_str) > 0:
            self.log_alpha = atof(log_alpha_str)

        var te_str = get_metadata_value(metadata, "target_entropy")
        if len(te_str) > 0:
            self.target_entropy = atof(te_str)

        var alpha_lr_str = get_metadata_value(metadata, "alpha_lr")
        if len(alpha_lr_str) > 0:
            self.alpha_lr = atof(alpha_lr_str)

        var auto_alpha_str = get_metadata_value(metadata, "auto_alpha")
        if len(auto_alpha_str) > 0:
            self.auto_alpha = Int(atol(auto_alpha_str)) != 0

        var total_steps_str = get_metadata_value(metadata, "total_steps")
        if len(total_steps_str) > 0:
            self.total_steps = Int(atol(total_steps_str))

        var train_step_str = get_metadata_value(metadata, "train_step_count")
        if len(train_step_str) > 0:
            self.train_step_count = Int(atol(train_step_str))
