"""Deep DDPG Agent using the new trait-based deep learning architecture.

This DDPG (Deep Deterministic Policy Gradient) implementation uses:
- Network wrapper from deep_rl.training for stateless model + params management
- seq() composition for building actor and critic networks
- Tanh output activation for bounded actions
- ReplayBuffer from deep_rl.replay for experience replay

Features:
- Works with any BoxContinuousActionEnv (continuous obs, continuous actions)
- Deterministic policy with Gaussian exploration noise
- Target networks with soft updates
- Single critic network (unlike TD3/SAC which use twin critics)

Usage:
    from deep_agents.ddpg import DeepDDPGAgent
    from envs import PendulumEnv

    var env = PendulumEnv()
    var agent = DeepDDPGAgent[3, 1, 256, 100000, 64]()

    # CPU Training
    var metrics = agent.train(env, num_episodes=300)

Reference: Lillicrap et al., "Continuous control with deep reinforcement learning" (2015)
"""

from math import exp, log, sqrt
from random import random_float64, seed

from layout import Layout, LayoutTensor

from deep_rl.constants import dtype, TILE, TPB
from deep_rl.model import Linear, ReLU, Tanh, seq
from deep_rl.optimizer import Adam
from deep_rl.initializer import Kaiming, Xavier
from deep_rl.training import Network
from deep_rl.replay import ReplayBuffer
from deep_rl.gpu.random import gaussian_noise
from core import TrainingMetrics, BoxContinuousActionEnv


# =============================================================================
# Deep DDPG Agent
# =============================================================================


struct DeepDDPGAgent[
    obs_dim: Int,
    action_dim: Int,
    hidden_dim: Int = 256,
    buffer_capacity: Int = 100000,
    batch_size: Int = 64,
]:
    """Deep Deterministic Policy Gradient agent using the new trait-based architecture.

    DDPG is an off-policy actor-critic algorithm that uses a deterministic policy
    with additive exploration noise for continuous action spaces.

    Key features:
    - Deterministic policy (actor outputs action directly, not distribution)
    - Single Q-network critic (unlike TD3/SAC which use twin critics)
    - Target networks for both actor and critic with soft updates
    - Gaussian exploration noise with decay

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

    # Critic input dimension: obs + action concatenated
    comptime CRITIC_IN = Self.OBS + Self.ACTIONS

    # Cache sizes for networks
    # Actor: Linear[obs, h] + ReLU[h] + Linear[h, h] + ReLU[h] + Linear[h, action] + Tanh[action]
    # Cache: OBS + HIDDEN + HIDDEN + HIDDEN + HIDDEN + ACTIONS
    comptime ACTOR_CACHE_SIZE: Int = (
        Self.OBS
        + Self.HIDDEN
        + Self.HIDDEN
        + Self.HIDDEN
        + Self.HIDDEN
        + Self.ACTIONS
    )

    # Critic: Linear[critic_in, h] + ReLU[h] + Linear[h, h] + ReLU[h] + Linear[h, 1]
    # Cache: CRITIC_IN + HIDDEN + HIDDEN + HIDDEN + HIDDEN
    comptime CRITIC_CACHE_SIZE: Int = (
        Self.CRITIC_IN + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN
    )

    # Actor network: obs -> hidden (ReLU) -> hidden (ReLU) -> action (Tanh)
    # Deterministic policy with tanh-bounded output
    var actor: Network[
        type_of(
            seq(
                Linear[Self.OBS, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.ACTIONS](),
                Tanh[Self.ACTIONS](),
            )
        ),
        Adam,
        Xavier,  # Xavier is good for tanh activation
    ]

    # Target actor network
    var actor_target: Network[
        type_of(
            seq(
                Linear[Self.OBS, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.ACTIONS](),
                Tanh[Self.ACTIONS](),
            )
        ),
        Adam,
        Xavier,
    ]

    # Critic network: (obs, action) -> hidden (ReLU) -> hidden (ReLU) -> Q-value
    var critic: Network[
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
        Kaiming,  # Kaiming is good for ReLU
    ]

    # Target critic network
    var critic_target: Network[
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
    var noise_std: Float64  # Exploration noise standard deviation
    var noise_std_min: Float64  # Minimum noise after decay
    var noise_decay: Float64  # Noise decay rate per episode

    # Training state
    var total_steps: Int
    var train_step_count: Int

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        actor_lr: Float64 = 0.001,
        critic_lr: Float64 = 0.001,
        action_scale: Float64 = 1.0,
        noise_std: Float64 = 0.1,
        noise_std_min: Float64 = 0.01,
        noise_decay: Float64 = 0.995,
    ):
        """Initialize Deep DDPG agent.

        Args:
            gamma: Discount factor (default: 0.99).
            tau: Soft update coefficient (default: 0.005).
            actor_lr: Actor learning rate (default: 0.001).
            critic_lr: Critic learning rate (default: 0.001).
            action_scale: Action scaling factor (default: 1.0).
            noise_std: Initial exploration noise std (default: 0.1).
            noise_std_min: Minimum exploration noise std (default: 0.01).
            noise_decay: Noise decay per episode (default: 0.995).
        """
        # Build actor model
        var actor_model = seq(
            Linear[Self.OBS, Self.HIDDEN](),
            ReLU[Self.HIDDEN](),
            Linear[Self.HIDDEN, Self.HIDDEN](),
            ReLU[Self.HIDDEN](),
            Linear[Self.HIDDEN, Self.ACTIONS](),
            Tanh[Self.ACTIONS](),
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
        self.actor = Network(actor_model, Adam(lr=actor_lr), Xavier())
        self.actor_target = Network(actor_model, Adam(lr=actor_lr), Xavier())
        self.critic = Network(critic_model, Adam(lr=critic_lr), Kaiming())
        self.critic_target = Network(
            critic_model, Adam(lr=critic_lr), Kaiming()
        )

        # Initialize target networks with same weights as online networks
        self.actor_target.copy_params_from(self.actor)
        self.critic_target.copy_params_from(self.critic)

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
        self.noise_std = noise_std
        self.noise_std_min = noise_std_min
        self.noise_decay = noise_decay

        # Training state
        self.total_steps = 0
        self.train_step_count = 0

    fn select_action(
        self,
        obs: SIMD[DType.float64, Self.obs_dim],
        add_noise: Bool = True,
    ) -> InlineArray[Float64, Self.action_dim]:
        """Select action using the deterministic policy with optional exploration noise.

        Args:
            obs: Current observation.
            add_noise: If True, add Gaussian exploration noise.

        Returns:
            Action array, scaled by action_scale.
        """
        # Convert obs to input array
        var obs_input = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)
        for i in range(Self.OBS):
            obs_input[i] = Scalar[dtype](obs[i])

        # Forward pass through actor (batch_size=1)
        var actor_output = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        self.actor.forward[1](obs_input, actor_output)

        # Extract action and optionally add noise
        var action_result = InlineArray[Float64, Self.action_dim](
            uninitialized=True
        )
        for i in range(Self.action_dim):
            # Actor output is already tanh-bounded to [-1, 1], scale by action_scale
            var a = Float64(actor_output[i]) * self.action_scale

            if add_noise:
                a += self.noise_std * self.action_scale * gaussian_noise()

            # Clip to action bounds
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale

            action_result[i] = a

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

        Note: Actions are stored unscaled (divided by action_scale) for consistency.
        """
        var obs_arr = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)
        var next_obs_arr = InlineArray[Scalar[dtype], Self.OBS](
            uninitialized=True
        )
        for i in range(Self.OBS):
            obs_arr[i] = Scalar[dtype](obs[i])
            next_obs_arr[i] = Scalar[dtype](next_obs[i])

        var action_arr = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        for i in range(Self.ACTIONS):
            # Store unscaled action (divide by action_scale)
            action_arr[i] = Scalar[dtype](action[i] / self.action_scale)

        self.buffer.add(
            obs_arr, action_arr, Scalar[dtype](reward), next_obs_arr, done
        )
        self.total_steps += 1

    fn train_step(mut self) -> Float64:
        """Perform one DDPG training step.

        Updates critic, then actor, then performs soft updates on targets.

        Returns:
            Critic loss value.
        """
        # Check if buffer has enough samples
        if not self.buffer.is_ready[Self.BATCH]():
            return 0.0

        # =====================================================================
        # Phase 1: Sample batch from buffer
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

        self.buffer.sample[Self.BATCH](
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones
        )

        # =====================================================================
        # Phase 2: Compute TD targets
        # y = r + gamma * Q_target(s', mu_target(s')) * (1 - done)
        # =====================================================================

        # Get next actions from target actor
        var next_actions = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        self.actor_target.forward[Self.BATCH](batch_next_obs, next_actions)

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
                ] = next_actions[b * Self.ACTIONS + i]

        # Forward target critic to get Q-values for next state
        var next_q_values = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        self.critic_target.forward[Self.BATCH](next_critic_input, next_q_values)

        # Compute TD targets: y = r + gamma * Q_target(s', mu_target(s')) * (1 - done)
        var targets = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        for b in range(Self.BATCH):
            var q_val = Float64(next_q_values[b])

            # Guard against NaN
            if q_val != q_val:
                q_val = 0.0

            var done_mask = 1.0 - Float64(batch_dones[b])
            var target = (
                Float64(batch_rewards[b]) + self.gamma * q_val * done_mask
            )

            # Clamp for numerical stability
            if target != target:
                target = 0.0
            elif target > 1000.0:
                target = 1000.0
            elif target < -1000.0:
                target = -1000.0

            targets[b] = Scalar[dtype](target)

        # =====================================================================
        # Phase 3: Update Critic
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

        # Forward critic with cache
        var q_values = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var critic_cache = InlineArray[
            Scalar[dtype], Self.BATCH * Self.CRITIC_CACHE_SIZE
        ](uninitialized=True)
        self.critic.forward_with_cache[Self.BATCH](
            critic_input, q_values, critic_cache
        )

        # Compute critic loss (MSE) and gradients
        var q_grad = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        var critic_loss: Float64 = 0.0

        for b in range(Self.BATCH):
            var td_error = q_values[b] - targets[b]
            critic_loss += Float64(td_error * td_error)

            # Gradient: d/dQ (Q - target)^2 = 2 * (Q - target) / batch_size
            q_grad[b] = (
                Scalar[dtype](2.0) * td_error / Scalar[dtype](Self.BATCH)
            )

        critic_loss /= Float64(Self.BATCH)

        # Backward pass for critic
        var critic_grad_input = InlineArray[
            Scalar[dtype], Self.BATCH * Self.CRITIC_IN
        ](uninitialized=True)

        self.critic.zero_grads()
        self.critic.backward[Self.BATCH](
            q_grad, critic_grad_input, critic_cache
        )
        self.critic.update()

        # =====================================================================
        # Phase 4: Update Actor
        # Policy gradient: maximize Q(s, mu(s)) -> minimize -Q(s, mu(s))
        # =====================================================================

        # Forward actor with cache to get current policy actions
        var actor_actions = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTIONS
        ](uninitialized=True)
        var actor_cache = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ACTOR_CACHE_SIZE
        ](uninitialized=True)
        self.actor.forward_with_cache[Self.BATCH](
            batch_obs, actor_actions, actor_cache
        )

        # Build critic input with actor's current actions
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
                ] = actor_actions[b * Self.ACTIONS + i]

        # Forward critic with cache (need cache for backward through critic to get action gradients)
        var new_q_values = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var new_critic_cache = InlineArray[
            Scalar[dtype], Self.BATCH * Self.CRITIC_CACHE_SIZE
        ](uninitialized=True)
        self.critic.forward_with_cache[Self.BATCH](
            new_critic_input, new_q_values, new_critic_cache
        )

        # Actor gradient: maximize Q -> dQ/daction = -1/batch_size (gradient ascent)
        var dq_actor = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        for b in range(Self.BATCH):
            dq_actor[b] = Scalar[dtype](-1.0 / Float64(Self.BATCH))

        # Backward through critic to get dQ/daction
        # Note: We don't want to update critic params here, just get action gradients
        var d_critic_input = InlineArray[
            Scalar[dtype], Self.BATCH * Self.CRITIC_IN
        ](uninitialized=True)

        # Zero critic grads before backward (we won't update critic, just need gradients)
        self.critic.zero_grads()
        self.critic.backward[Self.BATCH](
            dq_actor, d_critic_input, new_critic_cache
        )
        # Don't call self.critic.update() - we only want action gradients

        # Extract action gradients from d_critic_input (last ACTIONS elements per sample)
        var d_actions = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        for b in range(Self.BATCH):
            for i in range(Self.ACTIONS):
                d_actions[b * Self.ACTIONS + i] = d_critic_input[
                    b * Self.CRITIC_IN + Self.OBS + i
                ]

        # Backward through actor with action gradients
        var actor_grad_input = InlineArray[
            Scalar[dtype], Self.BATCH * Self.OBS
        ](uninitialized=True)

        self.actor.zero_grads()
        self.actor.backward[Self.BATCH](
            d_actions, actor_grad_input, actor_cache
        )
        self.actor.update()

        # =====================================================================
        # Phase 5: Soft Update Target Networks
        # =====================================================================

        self.actor_target.soft_update_from(self.actor, self.tau)
        self.critic_target.soft_update_from(self.critic, self.tau)

        self.train_step_count += 1

        return critic_loss

    fn decay_noise(mut self):
        """Decay exploration noise (call once per episode)."""
        self.noise_std *= self.noise_decay
        if self.noise_std < self.noise_std_min:
            self.noise_std = self.noise_std_min

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

    fn get_noise_std(self) -> Float64:
        """Get current exploration noise standard deviation."""
        return self.noise_std

    fn get_train_steps(self) -> Int:
        """Get total training steps performed."""
        return self.train_step_count

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
        """Train the DDPG agent on a continuous action environment.

        Args:
            env: The environment to train on (must implement BoxContinuousActionEnv).
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            warmup_steps: Number of random steps to fill replay buffer.
            train_every: Train every N steps.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes if verbose.
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="Deep DDPG",
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
            var action_list = List[Float64](capacity=Self.action_dim)
            for i in range(Self.action_dim):
                action[i] = (random_float64() * 2.0 - 1.0) * self.action_scale
                action_list.append(action[i])

            # Step environment with full action vector
            var result = env.step_continuous_vec(action_list^)
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
                # Select action with exploration noise
                var action = self.select_action(obs, add_noise=True)

                # Convert action to List for step_continuous_vec
                var action_list = List[Float64](capacity=Self.action_dim)
                for i in range(Self.action_dim):
                    action_list.append(action[i])

                # Step environment with full action vector
                var result = env.step_continuous_vec(action_list^)
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

            # Decay exploration noise
            self.decay_noise()

            # Log metrics
            metrics.log_episode(
                episode, episode_reward, episode_steps, self.noise_std
            )

            # Print progress
            if verbose and (episode + 1) % print_every == 0:
                var avg_reward = metrics.mean_reward_last_n(print_every)
                print(
                    "Episode",
                    episode + 1,
                    "| Avg reward:",
                    String(avg_reward)[:7],
                    "| Noise:",
                    String(self.noise_std)[:5],
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
        verbose: Bool = False,
        render: Bool = False,
    ) -> Float64:
        """Evaluate the agent using deterministic policy (no noise).

        Args:
            env: The environment to evaluate on.
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
            verbose: Whether to print per-episode results.
            render: Whether to render the environment.

        Returns:
            Average reward over evaluation episodes.
        """
        var total_reward: Float64 = 0.0

        for episode in range(num_episodes):
            var obs = self._list_to_simd(env.reset_obs_list())
            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for step in range(max_steps):
                # Deterministic action (no noise)
                var action = self.select_action(obs, add_noise=False)

                # Convert action to List for step_continuous_vec
                var action_list = List[Float64](capacity=Self.action_dim)
                for i in range(Self.action_dim):
                    action_list.append(action[i])

                # Step environment with full action vector
                var result = env.step_continuous_vec(action_list^)
                var reward = result[1]
                var done = result[2]

                if render:
                    env.render()

                episode_reward += reward
                obs = self._list_to_simd(env.get_obs_list())
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

        return total_reward / Float64(num_episodes)
