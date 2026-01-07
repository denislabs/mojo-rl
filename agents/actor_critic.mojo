"""Actor-Critic Agents for Policy Gradient Learning.

Actor-Critic methods combine policy gradient (actor) with value function
approximation (critic). The critic provides a baseline for variance reduction
and enables online (TD) updates instead of waiting for episode completion.

Variants implemented:
1. ActorCriticAgent - Basic one-step Actor-Critic (A2C-style)
2. ActorCriticLambdaAgent - Actor-Critic with eligibility traces

Both use tile coding for function approximation:
- Actor: Softmax policy π(a|s) over tile-coded features
- Critic: Linear value function V(s) over tile-coded features

Actor update (policy gradient):
    θ += α_actor * δ * ∇log π(a|s)

Critic update (TD learning):
    w += α_critic * δ * ∇V(s)

where δ = r + γ*V(s') - V(s) is the TD error.

References:
- Sutton & Barto, Chapter 13.5: "Actor-Critic Methods"
- Mnih et al. (2016): "Asynchronous Methods for Deep RL" (A3C/A2C)

Example usage:
    from core.tile_coding import make_cartpole_tile_coding
    from agents.actor_critic import ActorCriticAgent

    var tc = make_cartpole_tile_coding(num_tilings=8, tiles_per_dim=8)
    var agent = ActorCriticAgent(
        tile_coding=tc,
        num_actions=2,
        actor_lr=0.001,
        critic_lr=0.01,
    )

    # Training loop (online updates!)
    var tiles = tc.get_tiles_simd4(obs)
    var action = agent.select_action(tiles)
    # ... environment step ...
    var next_tiles = tc.get_tiles_simd4(next_obs)
    agent.update(tiles, action, reward, next_tiles, done)
"""

from math import exp, log
from random import random_float64
from core.tile_coding import TileCoding
from core import BoxDiscreteActionEnv, TrainingMetrics


struct ActorCriticAgent(Copyable, Movable, ImplicitlyCopyable):
    """One-step Actor-Critic with tile coding function approximation.

    Performs online TD(0) updates - no need to wait for episode completion.

    Actor: Softmax policy parameterized by θ
    Critic: Linear value function parameterized by w
    """

    # Actor (policy) parameters: θ[action][tile]
    var theta: List[List[Float64]]

    # Critic (value) parameters: w[tile]
    var critic_weights: List[Float64]

    var num_actions: Int
    var num_tiles: Int
    var num_tilings: Int
    var actor_lr: Float64
    var critic_lr: Float64
    var discount_factor: Float64
    var entropy_coef: Float64  # Entropy bonus for exploration

    fn __init__(
        out self,
        tile_coding: TileCoding,
        num_actions: Int,
        actor_lr: Float64 = 0.001,
        critic_lr: Float64 = 0.01,
        discount_factor: Float64 = 0.99,
        entropy_coef: Float64 = 0.0,
        init_value: Float64 = 0.0,
    ):
        """Initialize Actor-Critic agent.

        Args:
            tile_coding: TileCoding instance defining the feature representation
            num_actions: Number of discrete actions
            actor_lr: Actor (policy) learning rate (default 0.001)
            critic_lr: Critic (value) learning rate (default 0.01)
            discount_factor: Discount factor γ (default 0.99)
            entropy_coef: Entropy bonus coefficient (default 0.0)
            init_value: Initial parameter value (default 0.0)
        """
        self.num_actions = num_actions
        self.num_tiles = tile_coding.get_num_tiles()
        self.num_tilings = tile_coding.get_num_tilings()
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.entropy_coef = entropy_coef

        # Initialize actor parameters (policy)
        self.theta = List[List[Float64]]()
        for a in range(num_actions):
            var action_params = List[Float64]()
            for _ in range(self.num_tiles):
                action_params.append(init_value)
            self.theta.append(action_params^)

        # Initialize critic parameters (value function)
        self.critic_weights = List[Float64]()
        for _ in range(self.num_tiles):
            self.critic_weights.append(init_value)

    fn __copyinit__(out self, existing: Self):
        self.num_actions = existing.num_actions
        self.num_tiles = existing.num_tiles
        self.num_tilings = existing.num_tilings
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.entropy_coef = existing.entropy_coef
        self.theta = List[List[Float64]]()
        for a in range(existing.num_actions):
            var action_params = List[Float64]()
            for t in range(existing.num_tiles):
                action_params.append(existing.theta[a][t])
            self.theta.append(action_params^)
        self.critic_weights = List[Float64]()
        for t in range(existing.num_tiles):
            self.critic_weights.append(existing.critic_weights[t])

    fn __moveinit__(out self, deinit existing: Self):
        self.num_actions = existing.num_actions
        self.num_tiles = existing.num_tiles
        self.num_tilings = existing.num_tilings
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.entropy_coef = existing.entropy_coef
        self.theta = existing.theta^
        self.critic_weights = existing.critic_weights^

    fn _get_action_preferences(self, tiles: List[Int]) -> List[Float64]:
        """Compute action preferences (logits).

        h(s, a) = θ_a · φ(s)
        """
        var preferences = List[Float64]()
        for a in range(self.num_actions):
            var pref: Float64 = 0.0
            for i in range(len(tiles)):
                pref += self.theta[a][tiles[i]]
            preferences.append(pref)
        return preferences^

    fn _softmax(self, preferences: List[Float64]) -> List[Float64]:
        """Compute softmax probabilities (numerically stable)."""
        var max_pref = preferences[0]
        for i in range(1, len(preferences)):
            if preferences[i] > max_pref:
                max_pref = preferences[i]

        var exp_prefs = List[Float64]()
        var sum_exp: Float64 = 0.0
        for i in range(len(preferences)):
            var e = exp(preferences[i] - max_pref)
            exp_prefs.append(e)
            sum_exp += e

        var probs = List[Float64]()
        for i in range(len(exp_prefs)):
            probs.append(exp_prefs[i] / sum_exp)
        return probs^

    fn get_action_probs(self, tiles: List[Int]) -> List[Float64]:
        """Get action probabilities π(·|s).

        Args:
            tiles: Active tile indices

        Returns:
            Probability distribution over actions
        """
        var prefs = self._get_action_preferences(tiles)
        return self._softmax(prefs^)

    fn get_value(self, tiles: List[Int]) -> Float64:
        """Get state value estimate V(s).

        Args:
            tiles: Active tile indices

        Returns:
            Estimated state value
        """
        var value: Float64 = 0.0
        for i in range(len(tiles)):
            value += self.critic_weights[tiles[i]]
        return value

    fn select_action(self, tiles: List[Int]) -> Int:
        """Sample action from policy π(a|s).

        Args:
            tiles: Active tile indices

        Returns:
            Sampled action index
        """
        var probs = self.get_action_probs(tiles)

        var rand = random_float64()
        var cumsum: Float64 = 0.0
        for a in range(self.num_actions):
            cumsum += probs[a]
            if rand < cumsum:
                return a
        return self.num_actions - 1

    fn get_best_action(self, tiles: List[Int]) -> Int:
        """Get greedy action (highest probability).

        Args:
            tiles: Active tile indices

        Returns:
            Action with highest probability
        """
        var probs = self.get_action_probs(tiles)
        var best_action = 0
        var best_prob = probs[0]
        for a in range(1, self.num_actions):
            if probs[a] > best_prob:
                best_prob = probs[a]
                best_action = a
        return best_action

    fn update(
        mut self,
        tiles: List[Int],
        action: Int,
        reward: Float64,
        next_tiles: List[Int],
        done: Bool,
    ):
        """Update actor and critic using one-step TD error.

        δ = r + γ*V(s') - V(s)
        Critic: w += α_c * δ * ∇V(s)
        Actor:  θ += α_a * δ * ∇log π(a|s)

        Args:
            tiles: Active tiles for current state
            action: Action taken
            reward: Reward received
            next_tiles: Active tiles for next state
            done: Whether episode terminated
        """
        # Compute TD error (advantage estimate)
        var current_value = self.get_value(tiles)
        var next_value: Float64 = 0.0
        if not done:
            next_value = self.get_value(next_tiles)
        var td_error = reward + self.discount_factor * next_value - current_value

        # Update critic (value function)
        var critic_step = self.critic_lr / Float64(self.num_tilings)
        for i in range(len(tiles)):
            self.critic_weights[tiles[i]] += critic_step * td_error

        # Update actor (policy)
        var actor_step = self.actor_lr / Float64(self.num_tilings)
        var probs = self.get_action_probs(tiles)

        for a in range(self.num_actions):
            # Policy gradient for softmax
            var grad: Float64
            if a == action:
                grad = 1.0 - probs[a]
            else:
                grad = -probs[a]

            # Optional entropy bonus
            var entropy_grad: Float64 = 0.0
            if self.entropy_coef > 0.0 and probs[a] > 1e-10:
                # Gradient of entropy w.r.t. θ_a
                entropy_grad = -probs[a] * (1.0 + log(probs[a])) * probs[a] * (1.0 - probs[a])

            var total_grad = td_error * grad + self.entropy_coef * entropy_grad

            for i in range(len(tiles)):
                self.theta[a][tiles[i]] += actor_step * total_grad

    fn reset(mut self):
        """Reset for new episode (no-op for basic actor-critic)."""
        pass

    fn get_policy_entropy(self, tiles: List[Int]) -> Float64:
        """Compute policy entropy at given state.

        Args:
            tiles: Active tile indices

        Returns:
            Policy entropy (in nats)
        """
        var probs = self.get_action_probs(tiles)
        var entropy: Float64 = 0.0
        for a in range(self.num_actions):
            if probs[a] > 1e-10:
                entropy -= probs[a] * log(probs[a])
        return entropy

    fn train[E: BoxDiscreteActionEnv](
        mut self,
        mut env: E,
        tile_coding: TileCoding,
        num_episodes: Int,
        max_steps_per_episode: Int = 500,
        verbose: Bool = False,
        print_every: Int = 100,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the agent on a continuous-state environment.

        Args:
            env: The classic control environment to train on.
            tile_coding: TileCoding instance for feature extraction.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes (if verbose).
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="Actor-Critic",
            environment_name=environment_name,
        )

        for episode in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var obs = _list_to_simd4(obs_list)
            var total_reward: Float64 = 0.0
            var steps = 0

            for _ in range(max_steps_per_episode):
                var tiles = tile_coding.get_tiles_simd4(obs)
                var action = self.select_action(tiles)

                var result = env.step_obs(action)
                var next_obs_list = result[0]
                var next_obs = _list_to_simd4(next_obs_list)
                var reward = result[1]
                var done = result[2]

                var next_tiles = tile_coding.get_tiles_simd4(next_obs)
                self.update(tiles, action, reward, next_tiles, done)

                total_reward += reward
                steps += 1
                obs = next_obs

                if done:
                    break

            metrics.log_episode(episode, total_reward, steps, 0.0)

            if verbose and (episode + 1) % print_every == 0:
                metrics.print_progress(episode, window=100)

        return metrics^

    fn evaluate[E: BoxDiscreteActionEnv](
        self,
        mut env: E,
        tile_coding: TileCoding,
        num_episodes: Int = 10,
        max_steps: Int = 500,
        render: Bool = False,
    ) -> Float64:
        """Evaluate the agent on the environment.

        Args:
            env: The classic control environment to evaluate on.
            tile_coding: TileCoding instance for feature extraction.
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
            render: Whether to render the environment.

        Returns:
            Average reward across episodes.
        """
        var total_reward: Float64 = 0.0

        for _ in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var obs = _list_to_simd4(obs_list)
            var episode_reward: Float64 = 0.0

            for _ in range(max_steps):
                if render:
                    env.render()

                var tiles = tile_coding.get_tiles_simd4(obs)
                var action = self.get_best_action(tiles)

                var result = env.step_obs(action)
                var next_obs_list = result[0]
                var next_obs = _list_to_simd4(next_obs_list)
                var reward = result[1]
                var done = result[2]

                episode_reward += reward
                obs = next_obs

                if done:
                    break

            total_reward += episode_reward

        return total_reward / Float64(num_episodes)


struct ActorCriticLambdaAgent(Copyable, Movable, ImplicitlyCopyable):
    """Actor-Critic with eligibility traces for both actor and critic.

    Uses eligibility traces for more efficient credit assignment:
    - Critic uses TD(λ) for value learning
    - Actor uses eligibility traces scaled by TD error

    This provides a middle ground between one-step TD and Monte Carlo.
    """

    # Actor parameters and traces
    var theta: List[List[Float64]]
    var actor_traces: List[List[Float64]]

    # Critic parameters and traces
    var critic_weights: List[Float64]
    var critic_traces: List[Float64]

    var num_actions: Int
    var num_tiles: Int
    var num_tilings: Int
    var actor_lr: Float64
    var critic_lr: Float64
    var discount_factor: Float64
    var lambda_: Float64  # Trace decay parameter
    var entropy_coef: Float64

    fn __init__(
        out self,
        tile_coding: TileCoding,
        num_actions: Int,
        actor_lr: Float64 = 0.001,
        critic_lr: Float64 = 0.01,
        discount_factor: Float64 = 0.99,
        lambda_: Float64 = 0.9,
        entropy_coef: Float64 = 0.0,
        init_value: Float64 = 0.0,
    ):
        """Initialize Actor-Critic(λ) agent.

        Args:
            tile_coding: TileCoding instance
            num_actions: Number of discrete actions
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            discount_factor: Discount factor γ
            lambda_: Eligibility trace decay (0=TD(0), 1=MC)
            entropy_coef: Entropy bonus coefficient
            init_value: Initial parameter value
        """
        self.num_actions = num_actions
        self.num_tiles = tile_coding.get_num_tiles()
        self.num_tilings = tile_coding.get_num_tilings()
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.lambda_ = lambda_
        self.entropy_coef = entropy_coef

        # Initialize actor parameters
        self.theta = List[List[Float64]]()
        self.actor_traces = List[List[Float64]]()
        for a in range(num_actions):
            var action_params = List[Float64]()
            var action_traces = List[Float64]()
            for _ in range(self.num_tiles):
                action_params.append(init_value)
                action_traces.append(0.0)
            self.theta.append(action_params^)
            self.actor_traces.append(action_traces^)

        # Initialize critic parameters
        self.critic_weights = List[Float64]()
        self.critic_traces = List[Float64]()
        for _ in range(self.num_tiles):
            self.critic_weights.append(init_value)
            self.critic_traces.append(0.0)

    fn __copyinit__(out self, existing: Self):
        self.num_actions = existing.num_actions
        self.num_tiles = existing.num_tiles
        self.num_tilings = existing.num_tilings
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.lambda_ = existing.lambda_
        self.entropy_coef = existing.entropy_coef
        self.theta = List[List[Float64]]()
        self.actor_traces = List[List[Float64]]()
        for a in range(existing.num_actions):
            var action_params = List[Float64]()
            var action_traces = List[Float64]()
            for t in range(existing.num_tiles):
                action_params.append(existing.theta[a][t])
                action_traces.append(0.0)
            self.theta.append(action_params^)
            self.actor_traces.append(action_traces^)
        self.critic_weights = List[Float64]()
        self.critic_traces = List[Float64]()
        for t in range(existing.num_tiles):
            self.critic_weights.append(existing.critic_weights[t])
            self.critic_traces.append(0.0)

    fn __moveinit__(out self, deinit existing: Self):
        self.num_actions = existing.num_actions
        self.num_tiles = existing.num_tiles
        self.num_tilings = existing.num_tilings
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.lambda_ = existing.lambda_
        self.entropy_coef = existing.entropy_coef
        self.theta = existing.theta^
        self.actor_traces = existing.actor_traces^
        self.critic_weights = existing.critic_weights^
        self.critic_traces = existing.critic_traces^

    fn _get_action_preferences(self, tiles: List[Int]) -> List[Float64]:
        """Compute action preferences."""
        var preferences = List[Float64]()
        for a in range(self.num_actions):
            var pref: Float64 = 0.0
            for i in range(len(tiles)):
                pref += self.theta[a][tiles[i]]
            preferences.append(pref)
        return preferences^

    fn _softmax(self, preferences: List[Float64]) -> List[Float64]:
        """Compute softmax probabilities."""
        var max_pref = preferences[0]
        for i in range(1, len(preferences)):
            if preferences[i] > max_pref:
                max_pref = preferences[i]

        var exp_prefs = List[Float64]()
        var sum_exp: Float64 = 0.0
        for i in range(len(preferences)):
            var e = exp(preferences[i] - max_pref)
            exp_prefs.append(e)
            sum_exp += e

        var probs = List[Float64]()
        for i in range(len(exp_prefs)):
            probs.append(exp_prefs[i] / sum_exp)
        return probs^

    fn get_action_probs(self, tiles: List[Int]) -> List[Float64]:
        """Get action probabilities."""
        var prefs = self._get_action_preferences(tiles)
        return self._softmax(prefs^)

    fn get_value(self, tiles: List[Int]) -> Float64:
        """Get state value estimate."""
        var value: Float64 = 0.0
        for i in range(len(tiles)):
            value += self.critic_weights[tiles[i]]
        return value

    fn select_action(self, tiles: List[Int]) -> Int:
        """Sample action from policy."""
        var probs = self.get_action_probs(tiles)
        var rand = random_float64()
        var cumsum: Float64 = 0.0
        for a in range(self.num_actions):
            cumsum += probs[a]
            if rand < cumsum:
                return a
        return self.num_actions - 1

    fn get_best_action(self, tiles: List[Int]) -> Int:
        """Get greedy action."""
        var probs = self.get_action_probs(tiles)
        var best_action = 0
        var best_prob = probs[0]
        for a in range(1, self.num_actions):
            if probs[a] > best_prob:
                best_prob = probs[a]
                best_action = a
        return best_action

    fn update(
        mut self,
        tiles: List[Int],
        action: Int,
        reward: Float64,
        next_tiles: List[Int],
        done: Bool,
    ):
        """Update actor and critic with eligibility traces.

        Eligibility trace update:
        e *= γλ
        e += ∇ (for active features)
        θ += α * δ * e
        w += α * δ * e

        Args:
            tiles: Active tiles for current state
            action: Action taken
            reward: Reward received
            next_tiles: Active tiles for next state
            done: Whether episode terminated
        """
        # Compute TD error
        var current_value = self.get_value(tiles)
        var next_value: Float64 = 0.0
        if not done:
            next_value = self.get_value(next_tiles)
        var td_error = reward + self.discount_factor * next_value - current_value

        var probs = self.get_action_probs(tiles)

        # Decay traces
        var decay = self.discount_factor * self.lambda_
        for a in range(self.num_actions):
            for t in range(self.num_tiles):
                self.actor_traces[a][t] *= decay
        for t in range(self.num_tiles):
            self.critic_traces[t] *= decay

        # Update traces for active tiles
        # Critic: accumulating traces
        for i in range(len(tiles)):
            self.critic_traces[tiles[i]] += 1.0

        # Actor: policy gradient traces
        for a in range(self.num_actions):
            var grad: Float64
            if a == action:
                grad = 1.0 - probs[a]
            else:
                grad = -probs[a]

            for i in range(len(tiles)):
                self.actor_traces[a][tiles[i]] += grad

        # Update parameters using traces
        var critic_step = self.critic_lr / Float64(self.num_tilings)
        var actor_step = self.actor_lr / Float64(self.num_tilings)

        # Update critic
        for t in range(self.num_tiles):
            if self.critic_traces[t] != 0.0:
                self.critic_weights[t] += critic_step * td_error * self.critic_traces[t]

        # Update actor
        for a in range(self.num_actions):
            for t in range(self.num_tiles):
                if self.actor_traces[a][t] != 0.0:
                    var update = actor_step * td_error * self.actor_traces[a][t]

                    # Optional entropy bonus
                    if self.entropy_coef > 0.0 and probs[a] > 1e-10:
                        var entropy_grad = -probs[a] * (1.0 + log(probs[a])) * probs[a] * (1.0 - probs[a])
                        update += actor_step * self.entropy_coef * entropy_grad

                    self.theta[a][t] += update

        # Clear traces if episode ended
        if done:
            self._reset_traces()

    fn _reset_traces(mut self):
        """Reset all eligibility traces to zero."""
        for a in range(self.num_actions):
            for t in range(self.num_tiles):
                self.actor_traces[a][t] = 0.0
        for t in range(self.num_tiles):
            self.critic_traces[t] = 0.0

    fn reset(mut self):
        """Reset traces for new episode."""
        self._reset_traces()

    fn get_policy_entropy(self, tiles: List[Int]) -> Float64:
        """Compute policy entropy."""
        var probs = self.get_action_probs(tiles)
        var entropy: Float64 = 0.0
        for a in range(self.num_actions):
            if probs[a] > 1e-10:
                entropy -= probs[a] * log(probs[a])
        return entropy

    fn train[E: BoxDiscreteActionEnv](
        mut self,
        mut env: E,
        tile_coding: TileCoding,
        num_episodes: Int,
        max_steps_per_episode: Int = 500,
        verbose: Bool = False,
        print_every: Int = 100,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the agent on a continuous-state environment using Actor-Critic(λ).

        Args:
            env: The classic control environment to train on.
            tile_coding: TileCoding instance for feature extraction.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes (if verbose).
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="Actor-Critic(λ)",
            environment_name=environment_name,
        )

        for episode in range(num_episodes):
            self.reset()  # Reset eligibility traces
            var obs_list = env.reset_obs_list()
            var obs = _list_to_simd4(obs_list)
            var total_reward: Float64 = 0.0
            var steps = 0

            for _ in range(max_steps_per_episode):
                var tiles = tile_coding.get_tiles_simd4(obs)
                var action = self.select_action(tiles)

                var result = env.step_obs(action)
                var next_obs_list = result[0]
                var next_obs = _list_to_simd4(next_obs_list)
                var reward = result[1]
                var done = result[2]

                var next_tiles = tile_coding.get_tiles_simd4(next_obs)
                self.update(tiles, action, reward, next_tiles, done)

                total_reward += reward
                steps += 1
                obs = next_obs

                if done:
                    break

            metrics.log_episode(episode, total_reward, steps, 0.0)

            if verbose and (episode + 1) % print_every == 0:
                metrics.print_progress(episode, window=100)

        return metrics^

    fn evaluate[E: BoxDiscreteActionEnv](
        self,
        mut env: E,
        tile_coding: TileCoding,
        num_episodes: Int = 10,
        max_steps: Int = 500,
        render: Bool = False,
    ) -> Float64:
        """Evaluate the agent on the environment.

        Args:
            env: The classic control environment to evaluate on.
            tile_coding: TileCoding instance for feature extraction.
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
            render: Whether to render the environment.

        Returns:
            Average reward across episodes.
        """
        var total_reward: Float64 = 0.0

        for _ in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var obs = _list_to_simd4(obs_list)
            var episode_reward: Float64 = 0.0

            for _ in range(max_steps):
                if render:
                    env.render()

                var tiles = tile_coding.get_tiles_simd4(obs)
                var action = self.get_best_action(tiles)

                var result = env.step_obs(action)
                var next_obs_list = result[0]
                var next_obs = _list_to_simd4(next_obs_list)
                var reward = result[1]
                var done = result[2]

                episode_reward += reward
                obs = next_obs

                if done:
                    break

            total_reward += episode_reward

        return total_reward / Float64(num_episodes)


struct A2CAgent(Copyable, Movable, ImplicitlyCopyable):
    """Advantage Actor-Critic (A2C) with n-step returns.

    Accumulates transitions over n steps before updating, using
    n-step returns for lower variance advantage estimates.

    A(s, a) = R_n - V(s)

    where R_n = r_0 + γr_1 + ... + γ^(n-1)r_{n-1} + γ^n V(s_n)
    """

    # Actor and critic parameters
    var theta: List[List[Float64]]
    var critic_weights: List[Float64]

    var num_actions: Int
    var num_tiles: Int
    var num_tilings: Int
    var actor_lr: Float64
    var critic_lr: Float64
    var discount_factor: Float64
    var n_steps: Int
    var entropy_coef: Float64

    # N-step buffer
    var buffer_tiles: List[List[Int]]
    var buffer_actions: List[Int]
    var buffer_rewards: List[Float64]

    fn __init__(
        out self,
        tile_coding: TileCoding,
        num_actions: Int,
        actor_lr: Float64 = 0.001,
        critic_lr: Float64 = 0.01,
        discount_factor: Float64 = 0.99,
        n_steps: Int = 5,
        entropy_coef: Float64 = 0.01,
        init_value: Float64 = 0.0,
    ):
        """Initialize A2C agent.

        Args:
            tile_coding: TileCoding instance
            num_actions: Number of discrete actions
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            discount_factor: Discount factor γ
            n_steps: Number of steps for n-step returns
            entropy_coef: Entropy bonus coefficient
            init_value: Initial parameter value
        """
        self.num_actions = num_actions
        self.num_tiles = tile_coding.get_num_tiles()
        self.num_tilings = tile_coding.get_num_tilings()
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef

        # Initialize actor
        self.theta = List[List[Float64]]()
        for a in range(num_actions):
            var action_params = List[Float64]()
            for _ in range(self.num_tiles):
                action_params.append(init_value)
            self.theta.append(action_params^)

        # Initialize critic
        self.critic_weights = List[Float64]()
        for _ in range(self.num_tiles):
            self.critic_weights.append(init_value)

        # Initialize n-step buffer
        self.buffer_tiles = List[List[Int]]()
        self.buffer_actions = List[Int]()
        self.buffer_rewards = List[Float64]()

    fn __copyinit__(out self, existing: Self):
        self.num_actions = existing.num_actions
        self.num_tiles = existing.num_tiles
        self.num_tilings = existing.num_tilings
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.n_steps = existing.n_steps
        self.entropy_coef = existing.entropy_coef
        self.theta = List[List[Float64]]()
        for a in range(existing.num_actions):
            var action_params = List[Float64]()
            for t in range(existing.num_tiles):
                action_params.append(existing.theta[a][t])
            self.theta.append(action_params^)
        self.critic_weights = List[Float64]()
        for t in range(existing.num_tiles):
            self.critic_weights.append(existing.critic_weights[t])
        self.buffer_tiles = List[List[Int]]()
        self.buffer_actions = List[Int]()
        self.buffer_rewards = List[Float64]()

    fn __moveinit__(out self, deinit existing: Self):
        self.num_actions = existing.num_actions
        self.num_tiles = existing.num_tiles
        self.num_tilings = existing.num_tilings
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.n_steps = existing.n_steps
        self.entropy_coef = existing.entropy_coef
        self.theta = existing.theta^
        self.critic_weights = existing.critic_weights^
        self.buffer_tiles = existing.buffer_tiles^
        self.buffer_actions = existing.buffer_actions^
        self.buffer_rewards = existing.buffer_rewards^

    fn _get_action_preferences(self, tiles: List[Int]) -> List[Float64]:
        """Compute action preferences."""
        var preferences = List[Float64]()
        for a in range(self.num_actions):
            var pref: Float64 = 0.0
            for i in range(len(tiles)):
                pref += self.theta[a][tiles[i]]
            preferences.append(pref)
        return preferences^

    fn _softmax(self, preferences: List[Float64]) -> List[Float64]:
        """Compute softmax probabilities."""
        var max_pref = preferences[0]
        for i in range(1, len(preferences)):
            if preferences[i] > max_pref:
                max_pref = preferences[i]

        var exp_prefs = List[Float64]()
        var sum_exp: Float64 = 0.0
        for i in range(len(preferences)):
            var e = exp(preferences[i] - max_pref)
            exp_prefs.append(e)
            sum_exp += e

        var probs = List[Float64]()
        for i in range(len(exp_prefs)):
            probs.append(exp_prefs[i] / sum_exp)
        return probs^

    fn get_action_probs(self, tiles: List[Int]) -> List[Float64]:
        """Get action probabilities."""
        var prefs = self._get_action_preferences(tiles)
        return self._softmax(prefs^)

    fn get_value(self, tiles: List[Int]) -> Float64:
        """Get state value estimate."""
        var value: Float64 = 0.0
        for i in range(len(tiles)):
            value += self.critic_weights[tiles[i]]
        return value

    fn _get_value_idx(self, buffer_idx: Int) -> Float64:
        """Get state value for buffer step by index."""
        var value: Float64 = 0.0
        var num_tiles = len(self.buffer_tiles[buffer_idx])
        for i in range(num_tiles):
            var tile_idx = self.buffer_tiles[buffer_idx][i]
            value += self.critic_weights[tile_idx]
        return value

    fn _get_action_probs_idx(self, buffer_idx: Int) -> List[Float64]:
        """Get action probabilities for buffer step by index."""
        var preferences = List[Float64]()
        var num_tiles = len(self.buffer_tiles[buffer_idx])
        for a in range(self.num_actions):
            var pref: Float64 = 0.0
            for i in range(num_tiles):
                var tile_idx = self.buffer_tiles[buffer_idx][i]
                pref += self.theta[a][tile_idx]
            preferences.append(pref)
        return self._softmax(preferences^)

    fn select_action(self, tiles: List[Int]) -> Int:
        """Sample action from policy."""
        var probs = self.get_action_probs(tiles)
        var rand = random_float64()
        var cumsum: Float64 = 0.0
        for a in range(self.num_actions):
            cumsum += probs[a]
            if rand < cumsum:
                return a
        return self.num_actions - 1

    fn get_best_action(self, tiles: List[Int]) -> Int:
        """Get greedy action."""
        var probs = self.get_action_probs(tiles)
        var best_action = 0
        var best_prob = probs[0]
        for a in range(1, self.num_actions):
            if probs[a] > best_prob:
                best_prob = probs[a]
                best_action = a
        return best_action

    fn store_transition(
        mut self,
        tiles: List[Int],
        action: Int,
        reward: Float64,
    ):
        """Store transition in n-step buffer.

        Args:
            tiles: Active tiles for state.
            action: Action taken.
            reward: Reward received.
        """
        var tiles_copy = List[Int]()
        for i in range(len(tiles)):
            tiles_copy.append(tiles[i])

        self.buffer_tiles.append(tiles_copy^)
        self.buffer_actions.append(action)
        self.buffer_rewards.append(reward)

    fn update(
        mut self,
        next_tiles: List[Int],
        done: Bool,
    ):
        """Update when buffer is full or episode ends.

        Args:
            next_tiles: Tiles for next state (for bootstrap)
            done: Whether episode terminated
        """
        var buffer_len = len(self.buffer_tiles)

        # Check if we should update
        if buffer_len < self.n_steps and not done:
            return

        if buffer_len == 0:
            return

        # Compute n-step returns
        var returns = List[Float64]()
        for _ in range(buffer_len):
            returns.append(0.0)

        # Bootstrap value
        var bootstrap: Float64 = 0.0
        if not done:
            bootstrap = self.get_value(next_tiles)

        # Compute returns backwards
        var g = bootstrap
        for t in range(buffer_len - 1, -1, -1):
            g = self.buffer_rewards[t] + self.discount_factor * g
            returns[t] = g

        # Update for each transition
        var actor_step = self.actor_lr / Float64(self.num_tilings)
        var critic_step = self.critic_lr / Float64(self.num_tilings)

        for t in range(buffer_len):
            var action = self.buffer_actions[t]
            var g_t = returns[t]
            var num_tiles_t = len(self.buffer_tiles[t])

            var current_value = self._get_value_idx(t)
            var advantage = g_t - current_value

            # Update critic
            for i in range(num_tiles_t):
                var tile_idx = self.buffer_tiles[t][i]
                self.critic_weights[tile_idx] += critic_step * advantage

            # Update actor
            var probs = self._get_action_probs_idx(t)
            for a in range(self.num_actions):
                var grad: Float64
                if a == action:
                    grad = 1.0 - probs[a]
                else:
                    grad = -probs[a]

                # Entropy bonus
                var entropy_grad: Float64 = 0.0
                if self.entropy_coef > 0.0 and probs[a] > 1e-10:
                    entropy_grad = -probs[a] * (1.0 + log(probs[a])) * probs[a] * (1.0 - probs[a])

                var total_grad = advantage * grad + self.entropy_coef * entropy_grad

                for i in range(num_tiles_t):
                    var tile_idx = self.buffer_tiles[t][i]
                    self.theta[a][tile_idx] += actor_step * total_grad

        # Clear buffer
        self.buffer_tiles.clear()
        self.buffer_actions.clear()
        self.buffer_rewards.clear()

    fn reset(mut self):
        """Reset buffer for new episode."""
        self.buffer_tiles.clear()
        self.buffer_actions.clear()
        self.buffer_rewards.clear()

    fn get_policy_entropy(self, tiles: List[Int]) -> Float64:
        """Compute policy entropy."""
        var probs = self.get_action_probs(tiles)
        var entropy: Float64 = 0.0
        for a in range(self.num_actions):
            if probs[a] > 1e-10:
                entropy -= probs[a] * log(probs[a])
        return entropy

    fn train[E: BoxDiscreteActionEnv](
        mut self,
        mut env: E,
        tile_coding: TileCoding,
        num_episodes: Int,
        max_steps_per_episode: Int = 500,
        verbose: Bool = False,
        print_every: Int = 100,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the agent on a continuous-state environment using A2C.

        Args:
            env: The classic control environment to train on.
            tile_coding: TileCoding instance for feature extraction.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes (if verbose).
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="A2C",
            environment_name=environment_name,
        )

        for episode in range(num_episodes):
            self.reset()  # Clear buffer
            var obs_list = env.reset_obs_list()
            var obs = _list_to_simd4(obs_list)
            var total_reward: Float64 = 0.0
            var steps = 0

            for _ in range(max_steps_per_episode):
                var tiles = tile_coding.get_tiles_simd4(obs)
                var action = self.select_action(tiles)

                var result = env.step_obs(action)
                var next_obs_list = result[0]
                var next_obs = _list_to_simd4(next_obs_list)
                var reward = result[1]
                var done = result[2]

                self.store_transition(tiles, action, reward)

                var next_tiles = tile_coding.get_tiles_simd4(next_obs)
                self.update(next_tiles, done)

                total_reward += reward
                steps += 1
                obs = next_obs

                if done:
                    break

            metrics.log_episode(episode, total_reward, steps, 0.0)

            if verbose and (episode + 1) % print_every == 0:
                metrics.print_progress(episode, window=100)

        return metrics^

    fn evaluate[E: BoxDiscreteActionEnv](
        self,
        mut env: E,
        tile_coding: TileCoding,
        num_episodes: Int = 10,
        max_steps: Int = 500,
        render: Bool = False,
    ) -> Float64:
        """Evaluate the agent on the environment.

        Args:
            env: The classic control environment to evaluate on.
            tile_coding: TileCoding instance for feature extraction.
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
            render: Whether to render the environment.

        Returns:
            Average reward across episodes.
        """
        var total_reward: Float64 = 0.0

        for _ in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var obs = _list_to_simd4(obs_list)
            var episode_reward: Float64 = 0.0

            for _ in range(max_steps):
                if render:
                    env.render()

                var tiles = tile_coding.get_tiles_simd4(obs)
                var action = self.get_best_action(tiles)

                var result = env.step_obs(action)
                var next_obs_list = result[0]
                var next_obs = _list_to_simd4(next_obs_list)
                var reward = result[1]
                var done = result[2]

                episode_reward += reward
                obs = next_obs

                if done:
                    break

            total_reward += episode_reward

        return total_reward / Float64(num_episodes)


# ============================================================================
# Helper Functions
# ============================================================================


fn _list_to_simd4(obs: List[Float64]) -> SIMD[DType.float64, 4]:
    """Convert a List[Float64] to SIMD[DType.float64, 4].

    Pads with zeros if the list has fewer than 4 elements.
    """
    var result = SIMD[DType.float64, 4](0.0)
    var n = min(len(obs), 4)
    for i in range(n):
        result[i] = obs[i]
    return result
