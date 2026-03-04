"""Shared off-policy training infrastructure.

Provides the OffPolicyAgent trait and two free training loop functions that
eliminate the boilerplate warmup + episode loop duplicated across DQN, DQN+PER,
DuelingDQN, DDPG, TD3, and SAC agents.

The OffPolicyAgent trait is action-space agnostic: discrete and continuous
agents both expose List[Float64] for actions.  For discrete agents
each List has exactly one element (the chosen action index); for continuous
agents the List length equals action_dim.

Usage:
    struct MyAgent[...](OffPolicyAgent):
        fn select_action_list(mut self, obs: List[Float64]) -> List[Float64]: ...
        fn store_list_transition(mut self, ...) -> None: ...
        fn is_ready(self) -> Bool: ...
        fn do_train_step(mut self) -> Float64: ...
        fn decay_explore(mut self) -> None: ...
        fn get_explore_rate(self) -> Float64: ...
        fn random_action_list(self) -> List[Float64]: ...

    var metrics = run_offpolicy_discrete_train(agent, env, num_episodes=500)
    var metrics = run_offpolicy_continuous_train(agent, env, num_episodes=500)
"""

from math import exp
from random import random_float64, seed

from .metrics import TrainingMetrics
from .env_traits import BoxDiscreteActionEnv, BoxContinuousActionEnv


# =============================================================================
# OffPolicyAgent Trait
# =============================================================================


trait OffPolicyAgent:
    """Common interface for all off-policy deep RL agents.

    Agents implement this trait so that run_offpolicy_discrete_train and
    run_offpolicy_continuous_train can drive any agent without knowing
    agent-specific internals.

    Action representation:
        Discrete agents: List with one element = action index as Float64.
        Continuous agents: List with action_dim elements = raw action values.
    """

    fn select_action_list(
        mut self, obs: List[Float64]
    ) -> List[Float64]:
        """Select an action given the current observation.

        Applies epsilon-greedy / noise internally.

        Args:
            obs: Current observation as List[Float64].

        Returns:
            Action list (length 1 for discrete, action_dim for continuous).
        """
        ...

    fn store_list_transition(
        mut self,
        obs: List[Float64],
        action: List[Float64],
        reward: Float64,
        next_obs: List[Float64],
        done: Bool,
    ) -> None:
        """Store a transition in the replay buffer.

        Args:
            obs: Current observation.
            action: Action taken (see trait docstring for encoding).
            reward: Scalar reward received.
            next_obs: Next observation.
            done: Whether the episode ended.
        """
        ...

    fn is_ready(self) -> Bool:
        """Return True if the buffer holds enough samples to train."""
        ...

    fn do_train_step(mut self) -> Float64:
        """Perform one gradient update step.

        Returns:
            Loss value (may be 0.0 if not ready).
        """
        ...

    fn decay_explore(mut self) -> None:
        """Decay exploration rate (epsilon, noise_std, etc.)."""
        ...

    fn get_explore_rate(self) -> Float64:
        """Return current exploration rate (for logging)."""
        ...

    fn random_action_list(self) -> List[Float64]:
        """Return a uniformly random action (used during warmup).

        Returns:
            Random action list using same encoding as select_action_list.
        """
        ...

    fn select_greedy_action_list(
        self, obs: List[Float64]
    ) -> List[Float64]:
        """Select action without exploration noise (for evaluation).

        DQN: pure argmax (epsilon=0). DDPG/TD3: actor forward, no Gaussian
        noise. SAC: deterministic mean (no reparameterization sampling).

        Args:
            obs: Current observation as List[Float64].

        Returns:
            Greedy action list (length 1 for discrete, action_dim for continuous).
        """
        ...


# =============================================================================
# Shared Training Loop — Discrete Actions
# =============================================================================


fn run_offpolicy_discrete_train[
    E: BoxDiscreteActionEnv, A: OffPolicyAgent
](
    mut agent: A,
    mut env: E,
    num_episodes: Int,
    max_steps_per_episode: Int = 500,
    warmup_steps: Int = 1000,
    train_every: Int = 4,
    verbose: Bool = False,
    print_every: Int = 10,
    environment_name: String = "Environment",
    algorithm_name: String = "OffPolicy",
) -> TrainingMetrics:
    """Warmup + episode training loop for discrete-action off-policy agents.

    Shared implementation used by DQN, DQN+PER, and DuelingDQN to eliminate
    the duplicated warmup/train loop.

    Parameters:
        E: Environment type implementing BoxDiscreteActionEnv.
        A: Agent type implementing OffPolicyAgent.

    Args:
        agent: Off-policy agent (updated in-place).
        env: Discrete-action environment.
        num_episodes: Number of training episodes.
        max_steps_per_episode: Maximum steps per episode (default: 500).
        warmup_steps: Random exploration steps before training (default: 1000).
        train_every: Call do_train_step every N steps (default: 4).
        verbose: Print progress (default: False).
        print_every: Print every N episodes if verbose (default: 10).
        environment_name: Name for metrics labeling.
        algorithm_name: Name for metrics labeling.

    Returns:
        TrainingMetrics with per-episode rewards and statistics.
    """
    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )

    # --- Warmup: fill buffer with random transitions ---
    var warmup_obs = env.reset_obs_list()
    var warmup_count = 0
    while warmup_count < warmup_steps:
        var action = agent.random_action_list()
        var action_int = Int(Float64(action[0]))
        var result = env.step_obs(action_int)
        var next_obs = result[0].copy()
        var reward = Float64(result[1])
        var done = result[2]
        agent.store_list_transition(warmup_obs, action, reward, next_obs, done)
        warmup_count += 1
        if done:
            warmup_obs = env.reset_obs_list()
        else:
            warmup_obs = next_obs^

    # --- Training loop ---
    var total_steps = 0
    for episode in range(num_episodes):
        var obs = env.reset_obs_list()
        var episode_reward: Float64 = 0.0
        var episode_steps = 0

        for _ in range(max_steps_per_episode):
            var action = agent.select_action_list(obs)
            var action_int = Int(Float64(action[0]))
            var result = env.step_obs(action_int)
            var next_obs = result[0].copy()
            var reward = Float64(result[1])
            var done = result[2]

            agent.store_list_transition(obs, action, reward, next_obs, done)

            if agent.is_ready() and total_steps % train_every == 0:
                _ = agent.do_train_step()

            episode_reward += reward
            total_steps += 1
            episode_steps += 1
            obs = next_obs^

            if done:
                break

        agent.decay_explore()
        metrics.log_episode(
            episode,
            Scalar[DType.float64](episode_reward),
            episode_steps,
            agent.get_explore_rate(),
        )

        if verbose and (episode + 1) % print_every == 0:
            var avg_reward = metrics.mean_reward_last_n(print_every)
            print(
                "Episode "
                + String(episode + 1)
                + " | Avg reward: "
                + String(avg_reward)[:7]
                + " | Explore: "
                + String(agent.get_explore_rate())[:5]
                + " | Steps: "
                + String(total_steps)
            )

    return metrics^


# =============================================================================
# Shared Training Loop — Continuous Actions
# =============================================================================


fn run_offpolicy_continuous_train[
    E: BoxContinuousActionEnv, A: OffPolicyAgent
](
    mut agent: A,
    mut env: E,
    num_episodes: Int,
    max_steps_per_episode: Int = 1000,
    warmup_steps: Int = 1000,
    train_every: Int = 1,
    verbose: Bool = False,
    print_every: Int = 10,
    environment_name: String = "Environment",
    algorithm_name: String = "OffPolicy",
) -> TrainingMetrics:
    """Warmup + episode training loop for continuous-action off-policy agents.

    Shared implementation used by DDPG, TD3, and SAC.

    Parameters:
        E: Environment type implementing BoxContinuousActionEnv.
        A: Agent type implementing OffPolicyAgent.

    Args:
        agent: Off-policy agent (updated in-place).
        env: Continuous-action environment.
        num_episodes: Number of training episodes.
        max_steps_per_episode: Maximum steps per episode (default: 1000).
        warmup_steps: Random exploration steps before training (default: 1000).
        train_every: Call do_train_step every N steps (default: 1).
        verbose: Print progress (default: False).
        print_every: Print every N episodes if verbose (default: 10).
        environment_name: Name for metrics labeling.
        algorithm_name: Name for metrics labeling.

    Returns:
        TrainingMetrics with per-episode rewards and statistics.
    """
    var metrics = TrainingMetrics(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
    )

    # --- Warmup: fill buffer with random transitions ---
    var warmup_obs_raw = env.reset_obs_list()
    var warmup_obs = List[Float64]()
    for i in range(len(warmup_obs_raw)):
        warmup_obs.append(Float64(warmup_obs_raw[i]))

    var warmup_count = 0
    while warmup_count < warmup_steps:
        var action = agent.random_action_list()
        var result = env.step_continuous_vec(action)
        var next_obs = List[Float64]()
        for i in range(len(result[0])):
            next_obs.append(Float64(result[0][i]))
        var reward = Float64(result[1])
        var done = result[2]
        agent.store_list_transition(warmup_obs, action, reward, next_obs, done)
        warmup_count += 1
        if done:
            var reset_raw = env.reset_obs_list()
            warmup_obs = List[Float64]()
            for i in range(len(reset_raw)):
                warmup_obs.append(Float64(reset_raw[i]))
        else:
            warmup_obs = next_obs^

    # --- Training loop ---
    var total_steps = 0
    for episode in range(num_episodes):
        var obs_raw = env.reset_obs_list()
        var obs = List[Float64]()
        for i in range(len(obs_raw)):
            obs.append(Float64(obs_raw[i]))

        var episode_reward: Float64 = 0.0
        var episode_steps = 0

        for _ in range(max_steps_per_episode):
            var action = agent.select_action_list(obs)
            var result = env.step_continuous_vec(action)
            var next_obs = List[Float64]()
            for i in range(len(result[0])):
                next_obs.append(Float64(result[0][i]))
            var reward = Float64(result[1])
            var done = result[2]

            agent.store_list_transition(obs, action, reward, next_obs, done)

            if agent.is_ready() and total_steps % train_every == 0:
                _ = agent.do_train_step()

            episode_reward += reward
            total_steps += 1
            episode_steps += 1
            obs = next_obs^

            if done:
                break

        agent.decay_explore()
        metrics.log_episode(
            episode,
            Scalar[DType.float64](episode_reward),
            episode_steps,
            agent.get_explore_rate(),
        )

        if verbose and (episode + 1) % print_every == 0:
            var avg_reward = metrics.mean_reward_last_n(print_every)
            print(
                "Episode "
                + String(episode + 1)
                + " | Avg reward: "
                + String(avg_reward)[:7]
                + " | Explore: "
                + String(agent.get_explore_rate())[:5]
                + " | Steps: "
                + String(total_steps)
            )

    return metrics^
