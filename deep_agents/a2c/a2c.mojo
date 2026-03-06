"""Deep A2C (Advantage Actor-Critic) Agent using the new trait-based architecture.

This A2C implementation uses:
- NetworkState for heap-allocated params + grads + optimizer state
- Network (all-static) for stateless forward/backward ops via LayoutTensor
- Sequential composition for actor and critic networks
- GAE for advantage estimation

Key features:
- Works with any BoxDiscreteActionEnv (continuous obs, discrete actions)
- On-policy learning with rollout collection
- Softmax policy for discrete action spaces
- Entropy bonus for exploration
- Flexible n-step returns

Architecture:
- Actor: obs -> hidden (ReLU) -> hidden (ReLU) -> num_actions (Softmax)
- Critic: obs -> hidden (ReLU) -> hidden (ReLU) -> 1 (value)

Usage:
    from deep_agents.a2c import DeepA2CAgent
    from envs import CartPoleNative

    var env = CartPoleNative()
    var agent = DeepA2CAgent[4, 2, 128]()

    var metrics = agent.train(env, num_episodes=1000)

Note: actor_lr and critic_lr are compile-time parameters (they parameterize the
Adam optimizer type). gamma, gae_lambda, entropy_coef, etc. remain runtime fields.

Reference: Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016)
"""

from std.math import exp, log
from std.random import random_float64, seed

from layout import Layout, LayoutTensor

from nn.constants import dtype, TILE, TPB
from nn.model import Linear, ReLU, Sequential
from nn.optimizer import Adam
from nn.initializer import Xavier
from nn.training import Network, NetworkState
from nn.checkpoint import (
    save_checkpoint_file,
    read_checkpoint_file,
    write_checkpoint_header,
    write_metadata_section,
    read_metadata_section,
    get_metadata_value,
)
from core import (
    TrainingMetrics,
    BoxDiscreteActionEnv,
    BoxContinuousActionEnv,
    RenderableEnv,
)
from core.utils.gae import compute_gae_inline
from deep_agents.core import (
    OnPolicyAgent,
    run_onpolicy_discrete_train,
    Checkpointable,
)
from core.utils.softmax import (
    softmax_inline,
    sample_from_probs_inline,
    argmax_probs_inline,
)
from core.utils.normalization import normalize_inline


# =============================================================================
# Deep A2C Agent
# =============================================================================


struct DeepA2CAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 128,
    rollout_len: Int = 128,
    actor_lr: Float64 = 0.0003,
    critic_lr: Float64 = 0.001,
](OnPolicyAgent & Checkpointable):
    """Deep Advantage Actor-Critic Agent using the new stateless architecture.

    Uses separate actor and critic NetworkStates with heap-allocated params.
    Network (all-static) provides forward/backward ops via LayoutTensor views.

    Parameters:
        obs_dim: Dimension of observation space.
        num_actions: Number of discrete actions.
        hidden_dim: Hidden layer size (default: 128).
        rollout_len: Number of steps per rollout before update (default: 128).
        actor_lr: Actor Adam learning rate — compile-time (default: 0.0003).
        critic_lr: Critic Adam learning rate — compile-time (default: 0.001).
    """

    # Convenience aliases
    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.num_actions
    comptime HIDDEN = Self.hidden_dim
    comptime ROLLOUT = Self.rollout_len

    # OnPolicyAgent trait requirement
    comptime ROLLOUT_LEN: Int = Self.rollout_len

    # Stateless model descriptions (no stored weights)
    comptime ActorModel = Sequential[
        Linear[Self.OBS, Self.HIDDEN],
        ReLU[Self.HIDDEN],
        Linear[Self.HIDDEN, Self.HIDDEN],
        ReLU[Self.HIDDEN],
        Linear[Self.HIDDEN, Self.ACTIONS],
    ]
    comptime ActorNetwork = Network[Self.ActorModel, Adam[Self.actor_lr]]
    comptime CriticModel = Sequential[
        Linear[Self.OBS, Self.HIDDEN],
        ReLU[Self.HIDDEN],
        Linear[Self.HIDDEN, Self.HIDDEN],
        ReLU[Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]
    comptime CriticNetwork = Network[Self.CriticModel, Adam[Self.critic_lr]]

    # Network states: heap-allocated params + grads + optimizer state
    var actor: NetworkState[Self.ActorModel, Adam[Self.actor_lr]]
    var critic: NetworkState[Self.CriticModel, Adam[Self.critic_lr]]

    # Hyperparameters
    var gamma: Float64
    var gae_lambda: Float64
    var entropy_coef: Float64
    var value_loss_coef: Float64
    var max_grad_norm: Float64

    # Rollout buffers
    var buffer_obs: InlineArray[Scalar[dtype], Self.ROLLOUT * Self.OBS]
    var buffer_actions: InlineArray[Int, Self.ROLLOUT]
    var buffer_rewards: InlineArray[Scalar[dtype], Self.ROLLOUT]
    var buffer_values: InlineArray[Scalar[dtype], Self.ROLLOUT]
    var buffer_log_probs: InlineArray[Scalar[dtype], Self.ROLLOUT]
    var buffer_dones: InlineArray[Bool, Self.ROLLOUT]
    var buffer_idx: Int

    # OnPolicyAgent state: advantages/returns computed in compute_advantages()
    var _advantages: InlineArray[Scalar[dtype], Self.ROLLOUT]
    var _returns: InlineArray[Scalar[dtype], Self.ROLLOUT]
    # Current observation carried across rollout boundaries
    var _current_obs: InlineArray[Scalar[dtype], Self.OBS]
    var _env_initialized: Bool

    # Training state
    var train_step_count: Int

    # Checkpointing
    var checkpoint_every: Int
    var checkpoint_path: String

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        gae_lambda: Float64 = 0.95,
        entropy_coef: Float64 = 0.01,
        value_loss_coef: Float64 = 0.5,
        max_grad_norm: Float64 = 0.5,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ):
        """Initialize Deep A2C agent.

        Args:
            gamma: Discount factor (default: 0.99).
            gae_lambda: GAE lambda parameter (default: 0.95).
            entropy_coef: Entropy bonus coefficient (default: 0.01).
            value_loss_coef: Value loss coefficient (default: 0.5).
            max_grad_norm: Max gradient norm for clipping (default: 0.5).
            checkpoint_every: Save checkpoint every N episodes (0 = disabled).
            checkpoint_path: Base path for checkpoints (saves .actor/.critic/.meta).
        """
        # Initialize network states with Xavier initialization
        self.actor = NetworkState[Self.ActorModel, Adam[Self.actor_lr]]()
        self.actor.initialize[Xavier]()
        self.critic = NetworkState[Self.CriticModel, Adam[Self.critic_lr]]()
        self.critic.initialize[Xavier]()

        # Store hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        # Initialize rollout buffers
        self.buffer_obs = InlineArray[Scalar[dtype], Self.ROLLOUT * Self.OBS](
            fill=0
        )
        self.buffer_actions = InlineArray[Int, Self.ROLLOUT](fill=0)
        self.buffer_rewards = InlineArray[Scalar[dtype], Self.ROLLOUT](fill=0)
        self.buffer_values = InlineArray[Scalar[dtype], Self.ROLLOUT](fill=0)
        self.buffer_log_probs = InlineArray[Scalar[dtype], Self.ROLLOUT](fill=0)
        self.buffer_dones = InlineArray[Bool, Self.ROLLOUT](fill=False)
        self.buffer_idx = 0

        # OnPolicyAgent state
        self._advantages = InlineArray[Scalar[dtype], Self.ROLLOUT](fill=0)
        self._returns = InlineArray[Scalar[dtype], Self.ROLLOUT](fill=0)
        self._current_obs = InlineArray[Scalar[dtype], Self.OBS](fill=0)
        self._env_initialized = False

        # Training state
        self.train_step_count = 0

        # Checkpointing
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

    fn select_action(
        self,
        obs: InlineArray[Scalar[dtype], Self.OBS],
        training: Bool = True,
    ) -> Tuple[Int, Scalar[dtype], Scalar[dtype]]:
        """Select action from policy and compute log probability and value.

        Args:
            obs: Current observation.
            training: If True, sample action; else use greedy.

        Returns:
            Tuple of (action, log_prob, value).
        """
        # Create LayoutTensor view over obs (no copy — shared pointer)
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs.unsafe_ptr())

        # Actor forward (no cache — inference only)
        var logits_data = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var logits_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](logits_data.unsafe_ptr())
        var actor_params = self.actor.params_view()
        Self.ActorNetwork.forward[1](obs_t, logits_t, actor_params)

        # Copy logits to InlineArray for softmax
        var logits = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        for i in range(Self.ACTIONS):
            logits[i] = rebind[Scalar[dtype]](logits_t[0, i])

        # Compute softmax probabilities
        var probs = softmax_inline[dtype, Self.ACTIONS](logits)

        # Critic forward
        var value_data = InlineArray[Scalar[dtype], 1](uninitialized=True)
        var value_t = LayoutTensor[dtype, Layout.row_major(1, 1), MutAnyOrigin](
            value_data.unsafe_ptr()
        )
        var critic_params = self.critic.params_view()
        Self.CriticNetwork.forward[1](obs_t, value_t, critic_params)
        var value = rebind[Scalar[dtype]](value_t[0, 0])

        # Sample or greedy action
        var action: Int
        if training:
            action = sample_from_probs_inline[dtype, Self.ACTIONS](probs)
        else:
            action = argmax_probs_inline[dtype, Self.ACTIONS](probs)

        var log_prob = log(probs[action] + Scalar[dtype](1e-8))

        return (action, log_prob, value)

    fn store_transition(
        mut self,
        obs: InlineArray[Scalar[dtype], Self.OBS],
        action: Int,
        reward: Float64,
        log_prob: Scalar[dtype],
        value: Scalar[dtype],
        done: Bool,
    ):
        """Store transition in rollout buffer."""
        for i in range(Self.OBS):
            self.buffer_obs[self.buffer_idx * Self.OBS + i] = obs[i]

        self.buffer_actions[self.buffer_idx] = action
        self.buffer_rewards[self.buffer_idx] = Scalar[dtype](reward)
        self.buffer_log_probs[self.buffer_idx] = log_prob
        self.buffer_values[self.buffer_idx] = value
        self.buffer_dones[self.buffer_idx] = done

        self.buffer_idx += 1

    fn update(
        mut self,
        next_obs: InlineArray[Scalar[dtype], Self.OBS],
    ) -> Float64:
        """Update actor and critic using collected rollout.

        Args:
            next_obs: Next observation for bootstrapping.

        Returns:
            Total loss value.
        """
        if self.buffer_idx == 0:
            return 0.0

        # Bootstrap value for next_obs
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](next_obs.unsafe_ptr())
        var next_value_data = InlineArray[Scalar[dtype], 1](uninitialized=True)
        var next_value_t = LayoutTensor[
            dtype, Layout.row_major(1, 1), MutAnyOrigin
        ](next_value_data.unsafe_ptr())
        var critic_params = self.critic.params_view()
        Self.CriticNetwork.forward[1](next_obs_t, next_value_t, critic_params)
        var next_value = rebind[Scalar[dtype]](next_value_t[0, 0])

        # Compute GAE advantages and returns
        var advantages = InlineArray[Scalar[dtype], Self.ROLLOUT](fill=0)
        var returns = InlineArray[Scalar[dtype], Self.ROLLOUT](fill=0)

        compute_gae_inline[dtype, Self.ROLLOUT](
            self.buffer_rewards,
            self.buffer_values,
            next_value,
            self.buffer_dones,
            self.gamma,
            self.gae_lambda,
            self.buffer_idx,
            advantages,
            returns,
        )

        # Normalize advantages
        if self.buffer_idx > 1:
            normalize_inline[dtype, Self.ROLLOUT](self.buffer_idx, advantages)

        # =====================================================================
        # Update loop over rollout
        # =====================================================================

        var total_policy_loss = Scalar[dtype](0.0)
        var total_value_loss = Scalar[dtype](0.0)
        var total_entropy = Scalar[dtype](0.0)

        for t in range(self.buffer_idx):
            # Reconstruct obs for this timestep
            var obs = InlineArray[Scalar[dtype], Self.OBS](fill=0)
            for i in range(Self.OBS):
                obs[i] = self.buffer_obs[t * Self.OBS + i]

            var action = self.buffer_actions[t]
            var advantage = advantages[t]
            var return_t = returns[t]

            # Create obs LayoutTensor view (shared pointer into obs)
            var obs_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
            ](obs.unsafe_ptr())

            # =================================================================
            # Actor forward with cache
            # =================================================================
            var logits_data = InlineArray[Scalar[dtype], Self.ACTIONS](
                uninitialized=True
            )
            var logits_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
            ](logits_data.unsafe_ptr())
            var actor_cache = InlineArray[
                Scalar[dtype], Self.ActorNetwork.CACHE_SIZE
            ](uninitialized=True)
            var actor_cache_t = LayoutTensor[
                dtype,
                Layout.row_major(1, Self.ActorNetwork.CACHE_SIZE),
                MutAnyOrigin,
            ](actor_cache.unsafe_ptr())
            var actor_params = self.actor.params_view()
            Self.ActorNetwork.forward_with_cache[1](
                obs_t, logits_t, actor_params, actor_cache_t
            )

            var logits = InlineArray[Scalar[dtype], Self.ACTIONS](
                uninitialized=True
            )
            for i in range(Self.ACTIONS):
                logits[i] = rebind[Scalar[dtype]](logits_t[0, i])

            var probs = softmax_inline[dtype, Self.ACTIONS](logits)
            var new_log_prob = log(probs[action] + Scalar[dtype](1e-8))

            # Policy loss: -log_prob * advantage
            var policy_loss = -new_log_prob * advantage
            total_policy_loss += policy_loss

            # Entropy bonus: H = -Σ π(a) log π(a)
            var entropy = Scalar[dtype](0.0)
            for a in range(Self.ACTIONS):
                if probs[a] > Scalar[dtype](1e-8):
                    entropy -= probs[a] * log(probs[a])
            total_entropy += entropy

            # Actor gradient: d(-log_prob * advantage - entropy_coef * entropy) / d(logits)
            var d_logits = InlineArray[Scalar[dtype], Self.ACTIONS](fill=0)
            for a in range(Self.ACTIONS):
                var d_log_prob: Scalar[dtype]
                if a == action:
                    d_log_prob = Scalar[dtype](1.0) - probs[a]
                else:
                    d_log_prob = -probs[a]

                var d_entropy = -probs[a] * (
                    Scalar[dtype](1.0) + log(probs[a] + Scalar[dtype](1e-8))
                )

                d_logits[a] = (
                    -advantage * d_log_prob
                    - Scalar[dtype](self.entropy_coef) * d_entropy
                )

            # Actor backward + optimizer step
            var d_logits_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
            ](d_logits.unsafe_ptr())
            var actor_grad_in = InlineArray[Scalar[dtype], Self.OBS](fill=0)
            var actor_grad_in_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
            ](actor_grad_in.unsafe_ptr())
            self.actor.zero_grads()
            var actor_grads = self.actor.grads_view()
            Self.ActorNetwork.backward[1](
                d_logits_t,
                actor_grad_in_t,
                actor_params,
                actor_cache_t,
                actor_grads,
            )
            self.actor.optimizer_step()

            # =================================================================
            # Critic forward with cache
            # =================================================================
            var value_data = InlineArray[Scalar[dtype], 1](uninitialized=True)
            var value_t = LayoutTensor[
                dtype, Layout.row_major(1, 1), MutAnyOrigin
            ](value_data.unsafe_ptr())
            var critic_cache = InlineArray[
                Scalar[dtype], Self.CriticNetwork.CACHE_SIZE
            ](uninitialized=True)
            var critic_cache_t = LayoutTensor[
                dtype,
                Layout.row_major(1, Self.CriticNetwork.CACHE_SIZE),
                MutAnyOrigin,
            ](critic_cache.unsafe_ptr())
            var new_critic_params = self.critic.params_view()
            Self.CriticNetwork.forward_with_cache[1](
                obs_t, value_t, new_critic_params, critic_cache_t
            )
            var value = rebind[Scalar[dtype]](value_t[0, 0])

            # Value loss: (return - value)^2
            var value_loss = (return_t - value) * (return_t - value)
            total_value_loss += value_loss

            # Critic gradient: d(value_loss) / d(value) = 2 * (value - return)
            var d_value = InlineArray[Scalar[dtype], 1](fill=0)
            d_value[0] = (
                Scalar[dtype](2.0)
                * Scalar[dtype](self.value_loss_coef)
                * (value - return_t)
            )
            var d_value_t = LayoutTensor[
                dtype, Layout.row_major(1, 1), MutAnyOrigin
            ](d_value.unsafe_ptr())
            var critic_grad_in = InlineArray[Scalar[dtype], Self.OBS](fill=0)
            var critic_grad_in_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
            ](critic_grad_in.unsafe_ptr())
            self.critic.zero_grads()
            var critic_grads = self.critic.grads_view()
            Self.CriticNetwork.backward[1](
                d_value_t,
                critic_grad_in_t,
                new_critic_params,
                critic_cache_t,
                critic_grads,
            )
            self.critic.optimizer_step()

        # Clear buffer
        self.buffer_idx = 0
        self.train_step_count += 1

        # Return average loss
        var n = Scalar[dtype](Self.ROLLOUT)
        var total_loss = (
            total_policy_loss / n
            + Scalar[dtype](self.value_loss_coef) * total_value_loss / n
            - Scalar[dtype](self.entropy_coef) * total_entropy / n
        )
        return Float64(total_loss)

    fn _list_to_inline[
        T: DType
    ](self, obs_list: List[Scalar[T]]) -> InlineArray[Scalar[dtype], Self.OBS]:
        """Convert List[Scalar[T]] to InlineArray."""
        var obs = InlineArray[Scalar[dtype], Self.OBS](fill=Scalar[dtype](0))
        for i in range(Self.OBS):
            if i < len(obs_list):
                obs[i] = Scalar[dtype](obs_list[i])
        return obs^

    # =========================================================================
    # OnPolicyAgent trait conformance
    # =========================================================================

    fn collect_rollout[E: BoxDiscreteActionEnv](mut self, mut env: E) -> None:
        """Collect exactly ROLLOUT_LEN steps from the environment.

        Handles episode resets internally. Stores observations, actions,
        rewards, log_probs, values, and dones in internal rollout buffers.
        After collecting, self._current_obs holds the last observation
        (used as bootstrap value in compute_advantages).

        Args:
            env: Discrete-action environment.
        """
        if not self._env_initialized:
            self._current_obs = self._list_to_inline(env.reset_obs_list())
            self._env_initialized = True

        self.buffer_idx = 0

        for _ in range(Self.ROLLOUT):
            var action_result = self.select_action(
                self._current_obs, training=True
            )
            var action = action_result[0]
            var log_prob = action_result[1]
            var value = action_result[2]

            var result = env.step_obs(action)
            var reward = result[1]
            var done = result[2]

            # Copy to local to avoid aliasing self while calling mut self method
            var obs_copy = InlineArray[Scalar[dtype], Self.OBS](
                uninitialized=True
            )
            for _i in range(Self.OBS):
                obs_copy[_i] = self._current_obs[_i]
            self.store_transition(
                obs_copy,
                action,
                Float64(reward),
                log_prob,
                value,
                done,
            )

            if done:
                self._current_obs = self._list_to_inline(env.reset_obs_list())
            else:
                self._current_obs = self._list_to_inline(result[0])

    fn collect_rollout_continuous[
        E: BoxContinuousActionEnv
    ](mut self, mut env: E) -> None:
        """Not implemented: A2C is discrete-only."""
        pass

    fn compute_advantages(mut self) -> None:
        """Compute GAE advantages and returns using self._current_obs for bootstrap.

        Called after collect_rollout(), before update_epochs().
        Fills self._advantages and self._returns in-place.
        Normalizes advantages if buffer has more than one step.
        """
        if self.buffer_idx == 0:
            return

        # Bootstrap value from the observation after the last rollout step
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](self._current_obs.unsafe_ptr())
        var next_value_data = InlineArray[Scalar[dtype], 1](uninitialized=True)
        var next_value_t = LayoutTensor[
            dtype, Layout.row_major(1, 1), MutAnyOrigin
        ](next_value_data.unsafe_ptr())
        var critic_params = self.critic.params_view()
        Self.CriticNetwork.forward[1](next_obs_t, next_value_t, critic_params)
        var next_value = rebind[Scalar[dtype]](next_value_t[0, 0])

        compute_gae_inline[dtype, Self.ROLLOUT](
            self.buffer_rewards,
            self.buffer_values,
            next_value,
            self.buffer_dones,
            self.gamma,
            self.gae_lambda,
            self.buffer_idx,
            self._advantages,
            self._returns,
        )

        if self.buffer_idx > 1:
            normalize_inline[dtype, Self.ROLLOUT](
                self.buffer_idx, self._advantages
            )

    fn update_epochs(mut self) -> Float64:
        """Update actor and critic using computed advantages and returns.

        Must be called after compute_advantages(). Iterates over the
        collected rollout, updates actor with policy gradient + entropy,
        updates critic with value loss, then clears the buffer.

        Returns:
            Mean total loss across the rollout.
        """
        if self.buffer_idx == 0:
            return 0.0

        var total_policy_loss = Scalar[dtype](0.0)
        var total_value_loss = Scalar[dtype](0.0)
        var total_entropy = Scalar[dtype](0.0)

        for t in range(self.buffer_idx):
            var obs = InlineArray[Scalar[dtype], Self.OBS](fill=0)
            for i in range(Self.OBS):
                obs[i] = self.buffer_obs[t * Self.OBS + i]

            var action = self.buffer_actions[t]
            var advantage = self._advantages[t]
            var return_t = self._returns[t]

            var obs_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
            ](obs.unsafe_ptr())

            # Actor forward with cache
            var logits_data = InlineArray[Scalar[dtype], Self.ACTIONS](
                uninitialized=True
            )
            var logits_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
            ](logits_data.unsafe_ptr())
            var actor_cache = InlineArray[
                Scalar[dtype], Self.ActorNetwork.CACHE_SIZE
            ](uninitialized=True)
            var actor_cache_t = LayoutTensor[
                dtype,
                Layout.row_major(1, Self.ActorNetwork.CACHE_SIZE),
                MutAnyOrigin,
            ](actor_cache.unsafe_ptr())
            var actor_params = self.actor.params_view()
            Self.ActorNetwork.forward_with_cache[1](
                obs_t, logits_t, actor_params, actor_cache_t
            )

            var logits = InlineArray[Scalar[dtype], Self.ACTIONS](
                uninitialized=True
            )
            for i in range(Self.ACTIONS):
                logits[i] = rebind[Scalar[dtype]](logits_t[0, i])

            var probs = softmax_inline[dtype, Self.ACTIONS](logits)
            var new_log_prob = log(probs[action] + Scalar[dtype](1e-8))

            var policy_loss = -new_log_prob * advantage
            total_policy_loss += policy_loss

            var entropy = Scalar[dtype](0.0)
            for a in range(Self.ACTIONS):
                if probs[a] > Scalar[dtype](1e-8):
                    entropy -= probs[a] * log(probs[a])
            total_entropy += entropy

            var d_logits = InlineArray[Scalar[dtype], Self.ACTIONS](fill=0)
            for a in range(Self.ACTIONS):
                var d_log_prob: Scalar[dtype]
                if a == action:
                    d_log_prob = Scalar[dtype](1.0) - probs[a]
                else:
                    d_log_prob = -probs[a]

                var d_entropy = -probs[a] * (
                    Scalar[dtype](1.0) + log(probs[a] + Scalar[dtype](1e-8))
                )

                d_logits[a] = (
                    -advantage * d_log_prob
                    - Scalar[dtype](self.entropy_coef) * d_entropy
                )

            var d_logits_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
            ](d_logits.unsafe_ptr())
            var actor_grad_in = InlineArray[Scalar[dtype], Self.OBS](fill=0)
            var actor_grad_in_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
            ](actor_grad_in.unsafe_ptr())
            self.actor.zero_grads()
            var actor_grads = self.actor.grads_view()
            Self.ActorNetwork.backward[1](
                d_logits_t,
                actor_grad_in_t,
                actor_params,
                actor_cache_t,
                actor_grads,
            )
            self.actor.optimizer_step()

            # Critic forward with cache
            var value_data = InlineArray[Scalar[dtype], 1](uninitialized=True)
            var value_t = LayoutTensor[
                dtype, Layout.row_major(1, 1), MutAnyOrigin
            ](value_data.unsafe_ptr())
            var critic_cache = InlineArray[
                Scalar[dtype], Self.CriticNetwork.CACHE_SIZE
            ](uninitialized=True)
            var critic_cache_t = LayoutTensor[
                dtype,
                Layout.row_major(1, Self.CriticNetwork.CACHE_SIZE),
                MutAnyOrigin,
            ](critic_cache.unsafe_ptr())
            var new_critic_params = self.critic.params_view()
            Self.CriticNetwork.forward_with_cache[1](
                obs_t, value_t, new_critic_params, critic_cache_t
            )
            var value = rebind[Scalar[dtype]](value_t[0, 0])

            var value_loss = (return_t - value) * (return_t - value)
            total_value_loss += value_loss

            var d_value = InlineArray[Scalar[dtype], 1](fill=0)
            d_value[0] = (
                Scalar[dtype](2.0)
                * Scalar[dtype](self.value_loss_coef)
                * (value - return_t)
            )
            var d_value_t = LayoutTensor[
                dtype, Layout.row_major(1, 1), MutAnyOrigin
            ](d_value.unsafe_ptr())
            var critic_grad_in = InlineArray[Scalar[dtype], Self.OBS](fill=0)
            var critic_grad_in_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
            ](critic_grad_in.unsafe_ptr())
            self.critic.zero_grads()
            var critic_grads = self.critic.grads_view()
            Self.CriticNetwork.backward[1](
                d_value_t,
                critic_grad_in_t,
                new_critic_params,
                critic_cache_t,
                critic_grads,
            )
            self.critic.optimizer_step()

        self.buffer_idx = 0
        self.train_step_count += 1

        var n = Scalar[dtype](Self.ROLLOUT)
        var total_loss = (
            total_policy_loss / n
            + Scalar[dtype](self.value_loss_coef) * total_value_loss / n
            - Scalar[dtype](self.entropy_coef) * total_entropy / n
        )
        return Float64(total_loss)

    fn select_greedy_action_list(self, obs: List[Float64]) -> List[Float64]:
        """Select greedy action for evaluation (no sampling).

        Args:
            obs: Current observation as List[Float64].

        Returns:
            List with one element: the greedy action index as Float64.
        """
        var obs_inline = self._list_to_inline(obs)
        var action_result = self.select_action(obs_inline, training=False)
        var result = List[Float64]()
        result.append(Float64(action_result[0]))
        return result^

    fn get_explore_rate(self) -> Float64:
        """Return entropy coefficient as exploration rate proxy."""
        return self.entropy_coef

    fn train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        num_updates: Int,
        max_steps_per_episode: Int = 1000,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the A2C agent on a discrete action environment.

        Args:
            env: The environment to train on.
            num_updates: Number of collect→advantage→update cycles to run.
            max_steps_per_episode: Unused (kept for API compatibility).
            verbose: Whether to print progress.
            print_every: Print progress every N updates if verbose.
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var checkpoint_path = self.checkpoint_path
        var checkpoint_every = self.checkpoint_every
        return run_onpolicy_discrete_train(
            self,
            env,
            num_updates,
            checkpoint_path,
            checkpoint_every,
            verbose,
            print_every,
            environment_name,
            "Deep A2C",
        )

    fn save_checkpoint(self, path: String) raises:
        """Save agent state to a single checkpoint file.

        File layout:
            # mojo-rl checkpoint v1
            # type: deep_a2c_agent
            actor_params:       (float values)
            actor_optimizer_state:
            critic_params:
            critic_optimizer_state:
            metadata:           (key=value pairs)

        Args:
            path: Path for the checkpoint file.
        """
        var content = write_checkpoint_header("deep_a2c_agent", 0, 0)
        content += self.actor.write_sections("actor_")
        content += self.critic.write_sections("critic_")

        var metadata = List[String]()
        metadata.append("actor_step_num=" + String(self.actor.step_num))
        metadata.append("critic_step_num=" + String(self.critic.step_num))
        metadata.append("gamma=" + String(self.gamma))
        metadata.append("gae_lambda=" + String(self.gae_lambda))
        metadata.append("entropy_coef=" + String(self.entropy_coef))
        metadata.append("value_loss_coef=" + String(self.value_loss_coef))
        metadata.append("max_grad_norm=" + String(self.max_grad_norm))
        metadata.append("train_step_count=" + String(self.train_step_count))
        content += write_metadata_section(metadata)
        save_checkpoint_file(path, content)

    fn load_checkpoint(mut self, path: String) raises:
        """Load agent state from a single checkpoint file.

        Args:
            path: Path used when saving.
        """
        var content = read_checkpoint_file(path)
        self.actor.read_sections(content, "actor_")
        self.critic.read_sections(content, "critic_")

        var metadata = read_metadata_section(content)
        var v = get_metadata_value(metadata, "actor_step_num")
        if len(v) > 0:
            self.actor.step_num = Int(atof(v))
        v = get_metadata_value(metadata, "critic_step_num")
        if len(v) > 0:
            self.critic.step_num = Int(atof(v))
        v = get_metadata_value(metadata, "gamma")
        if len(v) > 0:
            self.gamma = Float64(atof(v))
        v = get_metadata_value(metadata, "gae_lambda")
        if len(v) > 0:
            self.gae_lambda = Float64(atof(v))
        v = get_metadata_value(metadata, "entropy_coef")
        if len(v) > 0:
            self.entropy_coef = Float64(atof(v))
        v = get_metadata_value(metadata, "value_loss_coef")
        if len(v) > 0:
            self.value_loss_coef = Float64(atof(v))
        v = get_metadata_value(metadata, "max_grad_norm")
        if len(v) > 0:
            self.max_grad_norm = Float64(atof(v))
        v = get_metadata_value(metadata, "train_step_count")
        if len(v) > 0:
            self.train_step_count = Int(atof(v))

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
    ) raises -> Float64:
        """Evaluate the agent using greedy policy.

        Args:
            env: The environment to evaluate on.
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
            verbose: Whether to print per-episode results.
            render: Whether to render the environment (default: False).
            frame_delay_ms: Delay between frames in milliseconds (default: 16).

        Returns:
            Average reward over evaluation episodes.
        """
        var total_reward: Float64 = 0.0
        var quit_requested = False

        if render:
            _ = env.init_renderer()

        for episode in range(num_episodes):
            if quit_requested:
                break

            var obs_list = env.reset_obs_list()
            var obs = self._list_to_inline(obs_list)
            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for _ in range(max_steps):
                # Greedy action
                var action_result = self.select_action(obs, training=False)
                var action = action_result[0]

                # Step environment
                var result = env.step_obs(action)
                var next_obs_list = result[0].copy()
                var reward = result[1]
                var done = result[2]

                if render:
                    env.render_frame()
                    env.renderer_delay(frame_delay_ms)
                    if env.check_renderer_quit():
                        quit_requested = True
                        break

                episode_reward += Float64(reward)
                obs = self._list_to_inline(next_obs_list)
                episode_steps += 1

                if done:
                    break

            total_reward += episode_reward

            if verbose:
                print(
                    "Eval Episode",
                    episode + 1,
                    "| Reward:",
                    String(episode_reward)[:10],
                    "| Steps:",
                    episode_steps,
                )

        if render:
            env.close_renderer()

        return total_reward / Float64(num_episodes)
