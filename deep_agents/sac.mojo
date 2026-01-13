"""Deep SAC Agent using the new trait-based deep learning architecture.

This SAC (Soft Actor-Critic) implementation uses:
- Network wrapper from deep_rl.training for stateless model + params management
- seq() composition for building actor and critic networks
- StochasticActor for Gaussian policy with reparameterization trick
- ReplayBuffer from deep_rl.replay for experience replay

Features:
- Works with any BoxContinuousActionEnv (continuous obs, continuous actions)
- Stochastic Gaussian policy for better exploration
- Twin Q-networks to reduce overestimation bias
- Automatic entropy temperature (alpha) tuning
- Maximum entropy RL objective: maximize reward + alpha * entropy
- Target networks with soft updates (critics only, no target actor)

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
from random import random_float64, seed

from layout import Layout, LayoutTensor

from deep_rl.constants import dtype, TILE, TPB
from deep_rl.model import Linear, ReLU, seq, StochasticActor
from deep_rl.model.stochastic_actor import (
    rsample,
    sample_action,
    get_deterministic_action,
)
from deep_rl.optimizer import Adam
from deep_rl.initializer import Kaiming
from deep_rl.training import Network
from deep_rl.replay import ReplayBuffer
from core import TrainingMetrics, BoxContinuousActionEnv


# =============================================================================
# Helper Functions
# =============================================================================


fn _gaussian_noise() -> Float64:
    """Generate standard Gaussian noise using Box-Muller transform."""
    var u1 = random_float64()
    var u2 = random_float64()
    # Avoid log(0)
    if u1 < 1e-10:
        u1 = 1e-10
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2)


fn cos(x: Float64) -> Float64:
    """Compute cosine using Taylor series."""
    var x2 = x * x
    var x4 = x2 * x2
    var x6 = x4 * x2
    var x8 = x4 * x4
    return 1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0 + x8 / 40320.0


# =============================================================================
# Deep SAC Agent
# =============================================================================


struct DeepSACAgent[
    obs_dim: Int,
    action_dim: Int,
    hidden_dim: Int = 256,
    buffer_capacity: Int = 100000,
    batch_size: Int = 64,
]:
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

    Parameters:
        obs_dim: Dimension of observation space.
        action_dim: Dimension of action space.
        hidden_dim: Hidden layer size (default: 256).
        buffer_capacity: Replay buffer capacity (default: 100000).
        batch_size: Training batch size (default: 64).
    """

    # Convenience aliases
    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.action_dim
    comptime HIDDEN = Self.hidden_dim
    comptime BATCH = Self.batch_size

    # Actor input/output dimensions
    # Actor: obs -> hidden -> hidden -> StochasticActor output (mean + log_std)
    comptime ACTOR_OUT = Self.ACTIONS * 2  # mean + log_std

    # Critic input dimension: obs + action concatenated
    comptime CRITIC_IN = Self.OBS + Self.ACTIONS

    # Cache sizes for networks
    # Actor: Linear[obs, h] + ReLU[h] + Linear[h, h] + ReLU[h] + StochasticActor[h, action]
    # StochasticActor caches input (HIDDEN)
    comptime ACTOR_CACHE_SIZE: Int = Self.OBS + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN

    # Critic: Linear[critic_in, h] + ReLU[h] + Linear[h, h] + ReLU[h] + Linear[h, 1]
    comptime CRITIC_CACHE_SIZE: Int = Self.CRITIC_IN + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN

    # Actor network: obs -> hidden (ReLU) -> hidden (ReLU) -> StochasticActor
    var actor: Network[
        type_of(
            seq(
                Linear[Self.OBS, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                StochasticActor[Self.HIDDEN, Self.ACTIONS](),
            )
        ),
        Adam,
        Kaiming,
    ]

    # Critic 1: (obs, action) -> hidden (ReLU) -> hidden (ReLU) -> Q-value
    var critic1: Network[
        type_of(
            seq(
                Linear[Self.CRITIC_IN, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, 1](),
            )
        ),
        Adam,
        Kaiming,
    ]

    # Critic 2 (twin)
    var critic2: Network[
        type_of(
            seq(
                Linear[Self.CRITIC_IN, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, 1](),
            )
        ),
        Adam,
        Kaiming,
    ]

    # Target critics (no target actor in SAC)
    var critic1_target: Network[
        type_of(
            seq(
                Linear[Self.CRITIC_IN, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, 1](),
            )
        ),
        Adam,
        Kaiming,
    ]

    var critic2_target: Network[
        type_of(
            seq(
                Linear[Self.CRITIC_IN, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, 1](),
            )
        ),
        Adam,
        Kaiming,
    ]

    # Replay buffer
    var buffer: ReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
    ]

    # Hyperparameters
    var gamma: Float64  # Discount factor
    var tau: Float64  # Soft update rate
    var actor_lr: Float64  # Actor learning rate
    var critic_lr: Float64  # Critic learning rate
    var action_scale: Float64  # Action scaling factor

    # Entropy tuning
    var alpha: Float64  # Entropy coefficient
    var log_alpha: Float64  # Log of alpha (for gradient stability)
    var target_entropy: Float64  # Target entropy
    var alpha_lr: Float64  # Alpha learning rate
    var auto_alpha: Bool  # Whether to automatically tune alpha

    # Training state
    var total_steps: Int
    var train_step_count: Int

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        actor_lr: Float64 = 0.0003,
        critic_lr: Float64 = 0.0003,
        action_scale: Float64 = 1.0,
        alpha: Float64 = 0.2,
        auto_alpha: Bool = True,
        alpha_lr: Float64 = 0.0003,
        target_entropy: Float64 = -1.0,  # Typically -dim(action_space)
    ):
        """Initialize Deep SAC agent.

        Args:
            gamma: Discount factor (default: 0.99).
            tau: Soft update rate for target networks (default: 0.005).
            actor_lr: Actor learning rate (default: 0.0003).
            critic_lr: Critic learning rate (default: 0.0003).
            action_scale: Action scaling factor (default: 1.0).
            alpha: Initial entropy coefficient (default: 0.2).
            auto_alpha: Whether to automatically tune alpha (default: True).
            alpha_lr: Alpha learning rate (default: 0.0003).
            target_entropy: Target entropy, typically -dim(action_space) (default: -1.0).
        """
        # Build actor model
        var actor_model = seq(
            Linear[Self.OBS, Self.HIDDEN](),
            ReLU[Self.HIDDEN](),
            Linear[Self.HIDDEN, Self.HIDDEN](),
            ReLU[Self.HIDDEN](),
            StochasticActor[Self.HIDDEN, Self.ACTIONS](),
        )

        # Build critic model
        var critic_model = seq(
            Linear[Self.CRITIC_IN, Self.HIDDEN](),
            ReLU[Self.HIDDEN](),
            Linear[Self.HIDDEN, Self.HIDDEN](),
            ReLU[Self.HIDDEN](),
            Linear[Self.HIDDEN, 1](),
        )

        # Initialize networks
        self.actor = Network(actor_model, Adam(lr=actor_lr), Kaiming())

        self.critic1 = Network(critic_model, Adam(lr=critic_lr), Kaiming())
        self.critic2 = Network(critic_model, Adam(lr=critic_lr), Kaiming())
        self.critic1_target = Network(
            critic_model, Adam(lr=critic_lr), Kaiming()
        )
        self.critic2_target = Network(
            critic_model, Adam(lr=critic_lr), Kaiming()
        )

        # Initialize target networks with same weights as online networks
        self.critic1_target.copy_params_from(self.critic1)
        self.critic2_target.copy_params_from(self.critic2)

        # Initialize replay buffer
        self.buffer = ReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, Self.action_dim, dtype
        ]()

        # Store hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.action_scale = action_scale

        # Entropy tuning
        self.alpha = alpha
        self.log_alpha = log(alpha)
        self.target_entropy = target_entropy
        self.alpha_lr = alpha_lr
        self.auto_alpha = auto_alpha

        # Training state
        self.total_steps = 0
        self.train_step_count = 0

    fn select_action(
        self,
        obs: SIMD[DType.float64, Self.obs_dim],
        deterministic: Bool = False,
    ) -> InlineArray[Float64, Self.action_dim]:
        """Select action using the stochastic policy.

        Args:
            obs: Current observation.
            deterministic: If True, use mean action (for evaluation).

        Returns:
            Action array.
        """
        # Convert obs to input array
        var obs_input = InlineArray[Scalar[dtype], Self.obs_dim](
            uninitialized=True
        )
        for i in range(Self.obs_dim):
            obs_input[i] = Scalar[dtype](obs[i])

        # Forward pass through actor (batch_size=1)
        var actor_output = InlineArray[Scalar[dtype], Self.ACTOR_OUT](
            uninitialized=True
        )
        self.actor.forward[1](obs_input, actor_output)

        # Extract mean and log_std (first ACTIONS elements are mean, next ACTIONS are log_std)
        var mean_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var log_std_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        for i in range(Self.ACTIONS):
            var mean_val = Float64(actor_output[i])
            var log_std_val = Float64(actor_output[Self.ACTIONS + i])

            # Clamp for numerical stability
            if mean_val != mean_val:  # NaN
                mean_val = 0.0
            elif mean_val > 10.0:
                mean_val = 10.0
            elif mean_val < -10.0:
                mean_val = -10.0

            if log_std_val != log_std_val:  # NaN
                log_std_val = -1.0
            elif log_std_val > 2.0:
                log_std_val = 2.0
            elif log_std_val < -5.0:
                log_std_val = -5.0

            mean_arr[i] = Scalar[dtype](mean_val)
            log_std_arr[i] = Scalar[dtype](log_std_val)

        var mean = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](mean_arr.unsafe_ptr())
        var log_std = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](log_std_arr.unsafe_ptr())

        var action_result = InlineArray[Float64, Self.action_dim](
            uninitialized=True
        )

        if deterministic:
            # Use mean action (no randomness)
            var action_tensor = InlineArray[Scalar[dtype], Self.ACTIONS](
                uninitialized=True
            )
            var action_layout = LayoutTensor[
                dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
            ](action_tensor.unsafe_ptr())
            get_deterministic_action[1, Self.ACTIONS](mean, action_layout)

            for i in range(Self.action_dim):
                action_result[i] = Float64(action_tensor[i]) * self.action_scale
        else:
            # Sample with reparameterization
            var noise = InlineArray[Scalar[dtype], Self.ACTIONS](
                uninitialized=True
            )
            for i in range(Self.ACTIONS):
                noise[i] = Scalar[dtype](_gaussian_noise())

            var noise_tensor = LayoutTensor[
                dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
            ](noise.unsafe_ptr())
            var action_tensor = InlineArray[Scalar[dtype], Self.ACTIONS](
                uninitialized=True
            )
            var action_layout = LayoutTensor[
                dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
            ](action_tensor.unsafe_ptr())

            sample_action[1, Self.ACTIONS](
                mean, log_std, noise_tensor, action_layout
            )

            for i in range(Self.action_dim):
                action_result[i] = Float64(action_tensor[i]) * self.action_scale

        return action_result

    fn store_transition(
        mut self,
        obs: SIMD[DType.float64, Self.obs_dim],
        action: InlineArray[Float64, Self.action_dim],
        reward: Float64,
        next_obs: SIMD[DType.float64, Self.obs_dim],
        done: Bool,
    ):
        """Store transition in replay buffer.

        Args:
            obs: Current observation.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation.
            done: Whether episode ended.
        """
        var obs_arr = InlineArray[Scalar[dtype], Self.obs_dim](
            uninitialized=True
        )
        var next_obs_arr = InlineArray[Scalar[dtype], Self.obs_dim](
            uninitialized=True
        )
        for i in range(Self.obs_dim):
            obs_arr[i] = Scalar[dtype](obs[i])
            next_obs_arr[i] = Scalar[dtype](next_obs[i])

        var action_arr = InlineArray[Scalar[dtype], Self.action_dim](
            uninitialized=True
        )
        for i in range(Self.action_dim):
            # Store unscaled action (divide by action_scale)
            action_arr[i] = Scalar[dtype](action[i] / self.action_scale)

        self.buffer.add(
            obs_arr, action_arr, Scalar[dtype](reward), next_obs_arr, done
        )
        self.total_steps += 1

    fn train_step(mut self) -> Float64:
        """Perform one SAC training step.

        Updates critics, actor, and optionally alpha.

        Returns:
            Critic loss value (average of Q1 and Q2 losses).
        """
        # Check if buffer has enough samples
        if not self.buffer.is_ready[Self.batch_size]():
            return 0.0

        # =====================================================================
        # Sample batch from buffer
        # =====================================================================
        var batch_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var batch_actions = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        var batch_rewards = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var batch_next_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var batch_dones = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )

        self.buffer.sample[Self.batch_size](
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones
        )

        # =====================================================================
        # Phase 1: Compute TD targets (with entropy bonus)
        # =====================================================================

        # Forward actor on next_obs to get next actions and log_probs
        var next_actor_output = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTOR_OUT
        ](uninitialized=True)
        self.actor.forward[Self.BATCH](batch_next_obs, next_actor_output)

        # Extract mean and log_std for next states (with clamping for stability)
        var next_mean = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        var next_log_std = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        for b in range(Self.BATCH):
            for a in range(Self.ACTIONS):
                var mean_val = Float64(
                    next_actor_output[b * Self.ACTOR_OUT + a]
                )
                var log_std_val = Float64(
                    next_actor_output[b * Self.ACTOR_OUT + Self.ACTIONS + a]
                )

                # Clamp values for numerical stability
                if mean_val != mean_val:  # NaN check
                    mean_val = 0.0
                elif mean_val > 10.0:
                    mean_val = 10.0
                elif mean_val < -10.0:
                    mean_val = -10.0

                if log_std_val != log_std_val:  # NaN check
                    log_std_val = -1.0
                elif log_std_val > 2.0:
                    log_std_val = 2.0
                elif log_std_val < -5.0:
                    log_std_val = -5.0

                next_mean[b * Self.ACTIONS + a] = Scalar[dtype](mean_val)
                next_log_std[b * Self.ACTIONS + a] = Scalar[dtype](log_std_val)

        # Sample next actions with log_prob using reparameterization
        var next_noise = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        for i in range(Self.BATCH * Self.ACTIONS):
            next_noise[i] = Scalar[dtype](_gaussian_noise())

        var next_sampled_actions = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        var next_log_probs = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )

        # Create layout tensors for rsample
        var next_mean_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](next_mean.unsafe_ptr())
        var next_log_std_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](next_log_std.unsafe_ptr())
        var next_noise_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](next_noise.unsafe_ptr())
        var next_action_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](next_sampled_actions.unsafe_ptr())
        var next_log_prob_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](next_log_probs.unsafe_ptr())

        rsample[Self.BATCH, Self.ACTIONS](
            next_mean_tensor,
            next_log_std_tensor,
            next_noise_tensor,
            next_action_tensor,
            next_log_prob_tensor,
        )

        # Guard against NaN/inf in log_probs
        for b in range(Self.BATCH):
            var lp = Float64(next_log_probs[b])
            if (
                lp != lp or lp > 100.0 or lp < -100.0
            ):  # NaN check or extreme values
                next_log_probs[b] = Scalar[dtype](
                    -1.0
                )  # Default reasonable value

        # Build critic input for next state: (next_obs, next_action)
        var next_critic_input = InlineArray[
            Scalar[dtype], Self.BATCH * Self.CRITIC_IN
        ](uninitialized=True)
        for b in range(Self.BATCH):
            for i in range(Self.OBS):
                next_critic_input[b * Self.CRITIC_IN + i] = batch_next_obs[
                    b * Self.OBS + i
                ]
            for i in range(Self.ACTIONS):
                next_critic_input[
                    b * Self.CRITIC_IN + Self.OBS + i
                ] = next_sampled_actions[b * Self.ACTIONS + i]

        # Forward target critics
        var next_q1 = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        var next_q2 = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        self.critic1_target.forward[Self.BATCH](next_critic_input, next_q1)
        self.critic2_target.forward[Self.BATCH](next_critic_input, next_q2)

        # Compute TD targets: r + gamma * (min(Q1, Q2) - alpha * log_prob) * (1 - done)
        var targets = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        for b in range(Self.BATCH):
            var q1_val = Float64(next_q1[b])
            var q2_val = Float64(next_q2[b])

            # Guard against NaN in Q-values
            if q1_val != q1_val:
                q1_val = 0.0
            if q2_val != q2_val:
                q2_val = 0.0

            var min_q = q1_val if q1_val < q2_val else q2_val
            var log_prob_val = Float64(next_log_probs[b])
            var entropy_bonus = self.alpha * log_prob_val
            var done_mask = 1.0 - Float64(batch_dones[b])
            var target = (
                Float64(batch_rewards[b])
                + self.gamma * (min_q - entropy_bonus) * done_mask
            )

            # Guard against extreme target values
            if target != target:  # NaN
                target = 0.0
            elif target > 1000.0:
                target = 1000.0
            elif target < -1000.0:
                target = -1000.0

            targets[b] = Scalar[dtype](target)

        # =====================================================================
        # Phase 2: Update Critics
        # =====================================================================

        # Build critic input for current state: (obs, action)
        var critic_input = InlineArray[
            Scalar[dtype], Self.BATCH * Self.CRITIC_IN
        ](uninitialized=True)
        for b in range(Self.BATCH):
            for i in range(Self.OBS):
                critic_input[b * Self.CRITIC_IN + i] = batch_obs[
                    b * Self.OBS + i
                ]
            for i in range(Self.ACTIONS):
                critic_input[b * Self.CRITIC_IN + Self.OBS + i] = batch_actions[
                    b * Self.ACTIONS + i
                ]

        # Forward critics with cache
        var q1_values = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var q2_values = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var critic1_cache = InlineArray[
            Scalar[dtype], Self.BATCH * Self.CRITIC_CACHE_SIZE
        ](uninitialized=True)
        var critic2_cache = InlineArray[
            Scalar[dtype], Self.BATCH * Self.CRITIC_CACHE_SIZE
        ](uninitialized=True)

        self.critic1.forward_with_cache[Self.BATCH](
            critic_input, q1_values, critic1_cache
        )
        self.critic2.forward_with_cache[Self.BATCH](
            critic_input, q2_values, critic2_cache
        )

        # Compute critic loss gradients (MSE)
        var q1_grad = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        var q2_grad = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        var total_critic_loss: Float64 = 0.0

        for b in range(Self.BATCH):
            var td_error1 = q1_values[b] - targets[b]
            var td_error2 = q2_values[b] - targets[b]

            total_critic_loss += Float64(
                td_error1 * td_error1 + td_error2 * td_error2
            )

            # Gradient: d/dQ (Q - target)^2 = 2 * (Q - target) / batch_size
            q1_grad[b] = (
                Scalar[dtype](2.0) * td_error1 / Scalar[dtype](Self.BATCH)
            )
            q2_grad[b] = (
                Scalar[dtype](2.0) * td_error2 / Scalar[dtype](Self.BATCH)
            )

        total_critic_loss /= Float64(
            2 * Self.BATCH
        )  # Average over both critics

        # Backward pass for critics
        var critic1_grad_input = InlineArray[
            Scalar[dtype], Self.BATCH * Self.CRITIC_IN
        ](uninitialized=True)
        var critic2_grad_input = InlineArray[
            Scalar[dtype], Self.BATCH * Self.CRITIC_IN
        ](uninitialized=True)

        self.critic1.zero_grads()
        self.critic1.backward[Self.BATCH](
            q1_grad, critic1_grad_input, critic1_cache
        )
        self.critic1.update()

        self.critic2.zero_grads()
        self.critic2.backward[Self.BATCH](
            q2_grad, critic2_grad_input, critic2_cache
        )
        self.critic2.update()

        # =====================================================================
        # Phase 3: Update Actor
        # =====================================================================

        # Forward actor with cache to get new actions
        var actor_output = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTOR_OUT
        ](uninitialized=True)
        var actor_cache = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTOR_CACHE_SIZE
        ](uninitialized=True)
        self.actor.forward_with_cache[Self.BATCH](
            batch_obs, actor_output, actor_cache
        )

        # Extract mean and log_std (with clamping for stability)
        var curr_mean = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        var curr_log_std = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        for b in range(Self.BATCH):
            for a in range(Self.ACTIONS):
                var mean_val = Float64(actor_output[b * Self.ACTOR_OUT + a])
                var log_std_val = Float64(
                    actor_output[b * Self.ACTOR_OUT + Self.ACTIONS + a]
                )

                # Clamp values for numerical stability
                if mean_val != mean_val:  # NaN check
                    mean_val = 0.0
                elif mean_val > 10.0:
                    mean_val = 10.0
                elif mean_val < -10.0:
                    mean_val = -10.0

                if log_std_val != log_std_val:  # NaN check
                    log_std_val = -1.0
                elif log_std_val > 2.0:
                    log_std_val = 2.0
                elif log_std_val < -5.0:
                    log_std_val = -5.0

                curr_mean[b * Self.ACTIONS + a] = Scalar[dtype](mean_val)
                curr_log_std[b * Self.ACTIONS + a] = Scalar[dtype](log_std_val)

        # Sample actions with log_prob
        var curr_noise = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        for i in range(Self.BATCH * Self.ACTIONS):
            curr_noise[i] = Scalar[dtype](_gaussian_noise())

        var curr_sampled_actions = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        var curr_log_probs = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )

        var curr_mean_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](curr_mean.unsafe_ptr())
        var curr_log_std_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](curr_log_std.unsafe_ptr())
        var curr_noise_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](curr_noise.unsafe_ptr())
        var curr_action_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](curr_sampled_actions.unsafe_ptr())
        var curr_log_prob_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](curr_log_probs.unsafe_ptr())

        rsample[Self.BATCH, Self.ACTIONS](
            curr_mean_tensor,
            curr_log_std_tensor,
            curr_noise_tensor,
            curr_action_tensor,
            curr_log_prob_tensor,
        )

        # Guard against NaN/inf in log_probs
        for b in range(Self.BATCH):
            var lp = Float64(curr_log_probs[b])
            if lp != lp or lp > 100.0 or lp < -100.0:
                curr_log_probs[b] = Scalar[dtype](-1.0)

        # Build critic input with new actions: (obs, new_action)
        var new_critic_input = InlineArray[
            Scalar[dtype], Self.BATCH * Self.CRITIC_IN
        ](uninitialized=True)
        for b in range(Self.BATCH):
            for i in range(Self.OBS):
                new_critic_input[b * Self.CRITIC_IN + i] = batch_obs[
                    b * Self.OBS + i
                ]
            for i in range(Self.ACTIONS):
                new_critic_input[
                    b * Self.CRITIC_IN + Self.OBS + i
                ] = curr_sampled_actions[b * Self.ACTIONS + i]

        # Forward critic1 to get Q-values for actor update
        var q1_new = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        self.critic1.forward[Self.BATCH](new_critic_input, q1_new)

        # Actor loss: J_π = E[α * log_π(a|s) - Q(s, a)]
        # We want to minimize this (maximize Q, maximize entropy)
        #
        # Using a simplified policy gradient approach:
        # - Gradient for mean: proportional to (α - Q) * direction_to_improve
        # - Gradient for log_std: encourage exploration when Q is low
        #
        # This is an approximation that avoids complex backprop through critic.

        var actor_grad = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTOR_OUT
        ](uninitialized=True)

        # Initialize to zero
        for i in range(Self.BATCH * Self.ACTOR_OUT):
            actor_grad[i] = Scalar[dtype](0.0)

        # Compute mean Q and log_prob for normalization
        var mean_q: Float64 = 0.0
        var mean_log_prob: Float64 = 0.0
        for b in range(Self.BATCH):
            mean_q += Float64(q1_new[b])
            mean_log_prob += Float64(curr_log_probs[b])
        mean_q /= Float64(Self.BATCH)
        mean_log_prob /= Float64(Self.BATCH)

        # For each sample, compute gradient contribution
        for b in range(Self.BATCH):
            var log_prob = Float64(curr_log_probs[b])
            var q_val = Float64(q1_new[b])

            # Advantage: how much better is this Q than average?
            var advantage = q_val - mean_q

            # Normalize advantage for stability
            var adv_scale = 1.0 / (abs(advantage) + 1.0)

            for a in range(Self.ACTIONS):
                var action_val = Float64(
                    curr_sampled_actions[b * Self.ACTIONS + a]
                )
                var mean_val = Float64(curr_mean[b * Self.ACTIONS + a])
                var log_std_val = Float64(curr_log_std[b * Self.ACTIONS + a])
                var std = exp(log_std_val)

                # Mean gradient: move mean towards actions with high Q
                # Using policy gradient: gradient = advantage * (action - mean) / std^2
                var mean_grad = (
                    advantage
                    * adv_scale
                    * (action_val - mean_val)
                    / (std * std + 0.01)
                )

                # Add entropy gradient: encourage higher entropy when alpha is high
                # d(log_prob)/d(mean) ≈ -(action - mean) / std^2
                var entropy_grad = (
                    -self.alpha * (action_val - mean_val) / (std * std + 0.01)
                )

                # Combined: we want to maximize Q and maximize entropy
                # So we DESCEND on: -Q_grad + alpha * entropy_loss_grad
                actor_grad[b * Self.ACTOR_OUT + a] = Scalar[dtype](
                    (-mean_grad + entropy_grad * 0.1) / Float64(Self.BATCH)
                )

                # Log_std gradient:
                # - Higher log_std = higher entropy (good for exploration)
                # - d(entropy)/d(log_std) = 1 (approximately, for Gaussian)
                # - But we need to balance with Q-value
                var log_std_grad = -self.alpha * 0.1  # Encourage exploration
                if advantage < 0:
                    # If this action was bad, increase exploration
                    log_std_grad += 0.01

                actor_grad[b * Self.ACTOR_OUT + Self.ACTIONS + a] = Scalar[
                    dtype
                ](log_std_grad / Float64(Self.BATCH))

        # Backward pass for actor
        var actor_grad_input = InlineArray[
            Scalar[dtype], Self.BATCH * Self.OBS
        ](uninitialized=True)

        self.actor.zero_grads()
        self.actor.backward[Self.BATCH](
            actor_grad, actor_grad_input, actor_cache
        )
        self.actor.update()

        # =====================================================================
        # Phase 4: Update Alpha (if auto_alpha)
        # =====================================================================

        if self.auto_alpha:
            # Alpha loss: J(α) = E[α * (log_π + target_entropy)]
            # Gradient: dJ/dα = E[log_π + target_entropy]
            # We minimize, so: log_alpha -= lr * gradient
            # If log_π + target_entropy < 0: gradient < 0, log_alpha increases (more exploration)
            # If log_π + target_entropy > 0: gradient > 0, log_alpha decreases (less exploration)
            var alpha_grad: Float64 = 0.0
            for b in range(Self.BATCH):
                alpha_grad += Float64(curr_log_probs[b]) + self.target_entropy
            alpha_grad /= Float64(Self.BATCH)

            # Update log_alpha (gradient descent to minimize loss)
            self.log_alpha -= self.alpha_lr * alpha_grad

            # Clamp log_alpha to reasonable range [-5, 2] for stability
            if self.log_alpha < -5.0:
                self.log_alpha = -5.0
            elif self.log_alpha > 2.0:
                self.log_alpha = 2.0

            # Update alpha
            self.alpha = exp(self.log_alpha)

        # =====================================================================
        # Phase 5: Soft Update Target Networks
        # =====================================================================

        self.critic1_target.soft_update_from(self.critic1, self.tau)
        self.critic2_target.soft_update_from(self.critic2, self.tau)

        self.train_step_count += 1

        return total_critic_loss

    fn _list_to_simd(
        self, obs_list: List[Float64]
    ) -> SIMD[DType.float64, Self.obs_dim]:
        """Convert List[Float64] to SIMD for internal use."""
        var obs = SIMD[DType.float64, Self.obs_dim]()
        for i in range(Self.obs_dim):
            if i < len(obs_list):
                obs[i] = obs_list[i]
            else:
                obs[i] = 0.0
        return obs

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

        Args:
            env: The environment to train on (must implement BoxContinuousActionEnv).
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode (default: 200).
            warmup_steps: Number of random steps to fill replay buffer (default: 1000).
            train_every: Train every N steps (default: 1).
            verbose: Whether to print progress (default: False).
            print_every: Print progress every N episodes if verbose (default: 10).
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="Deep SAC",
            environment_name=environment_name,
        )

        # =====================================================================
        # Warmup: fill replay buffer with random actions
        # =====================================================================
        var warmup_obs = self._list_to_simd(env.reset_obs_list())
        var warmup_count = 0

        while warmup_count < warmup_steps:
            # Random action in [-action_scale, action_scale]
            var action = InlineArray[Float64, Self.action_dim](
                uninitialized=True
            )
            for i in range(Self.action_dim):
                action[i] = (random_float64() * 2.0 - 1.0) * self.action_scale

            # Step environment (use first action dimension for 1D environments)
            var step_action = action[0]
            var result = env.step_continuous(step_action)
            # result is Tuple[List[Float64], Float64, Bool] = (obs_list, reward, done)
            var reward = result[1]
            var done = result[2]

            var next_obs = self._list_to_simd(env.get_obs_list())
            self.store_transition(warmup_obs, action, reward, next_obs, done)

            warmup_obs = next_obs
            warmup_count += 1

            if done:
                warmup_obs = self._list_to_simd(env.reset_obs_list())

        if verbose:
            print("Warmup complete:", warmup_count, "transitions collected")

        # =====================================================================
        # Training loop
        # =====================================================================
        var total_train_steps = 0

        for episode in range(num_episodes):
            var obs = self._list_to_simd(env.reset_obs_list())
            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for step in range(max_steps_per_episode):
                # Select action using stochastic policy
                var action = self.select_action(obs, deterministic=False)

                # Step environment
                var step_action = action[0]
                var result = env.step_continuous(step_action)
                # result is Tuple[List[Float64], Float64, Bool] = (obs_list, reward, done)
                var reward = result[1]
                var done = result[2]

                var next_obs = self._list_to_simd(env.get_obs_list())

                # Store transition
                self.store_transition(obs, action, reward, next_obs, done)

                # Train every N steps
                if total_train_steps % train_every == 0:
                    _ = self.train_step()

                episode_reward += reward
                obs = next_obs
                total_train_steps += 1
                episode_steps += 1

                if done:
                    break

            # Log metrics
            metrics.log_episode(
                episode, episode_reward, episode_steps, self.alpha
            )

            # Print progress
            if verbose and (episode + 1) % print_every == 0:
                var avg_reward = metrics.mean_reward_last_n(print_every)
                print(
                    "Episode",
                    episode + 1,
                    "| Avg reward:",
                    String(avg_reward)[:7],
                    "| Alpha:",
                    String(self.alpha)[:5],
                    "| Steps:",
                    total_train_steps,
                )

        return metrics^

    fn evaluate[
        E: BoxContinuousActionEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps: Int = 200,
        render: Bool = False,
    ) -> Float64:
        """Evaluate the agent using deterministic policy.

        Args:
            env: The environment to evaluate on.
            num_episodes: Number of evaluation episodes (default: 10).
            max_steps: Maximum steps per episode (default: 200).
            render: Whether to render the environment (default: False).

        Returns:
            Average reward over evaluation episodes.
        """
        var total_reward: Float64 = 0.0

        for episode in range(num_episodes):
            var obs = self._list_to_simd(env.reset_obs_list())
            var episode_reward: Float64 = 0.0

            for step in range(max_steps):
                # Deterministic action
                var action = self.select_action(obs, deterministic=True)

                # Step environment
                var step_action = action[0]
                var result = env.step_continuous(step_action)
                # result is Tuple[List[Float64], Float64, Bool] = (obs_list, reward, done)
                var reward = result[1]
                var done = result[2]

                if render:
                    env.render()

                episode_reward += reward
                obs = self._list_to_simd(env.get_obs_list())

                if done:
                    break

            total_reward += episode_reward

        return total_reward / Float64(num_episodes)

    fn get_alpha(self) -> Float64:
        """Get current entropy coefficient."""
        return self.alpha

    fn get_train_steps(self) -> Int:
        """Get total training steps performed."""
        return self.train_step_count
