"""Generic training functions for tabular RL agents."""

from .env import Env
from .state import State
from .tabular_agent import TabularAgent
from .metrics import TrainingMetrics


trait DiscreteEnv(Env):
    """Environment with discrete states that can be indexed.

    This extends Env with a state_to_index method for tabular methods.
    """

    fn state_to_index(self, state: Self.StateType) -> Int:
        """Convert a state to an index for tabular methods."""
        ...

    fn action_from_index(self, action_idx: Int) -> Self.ActionType:
        """Create an action from an index."""
        ...


fn train_tabular[
    E: DiscreteEnv, A: TabularAgent
](
    mut env: E,
    mut agent: A,
    num_episodes: Int,
    max_steps_per_episode: Int = 100,
    verbose: Bool = False,
) -> List[Float64]:
    """Train a tabular RL agent on any discrete environment.

    Args:
        env: The discrete environment to train on.
        agent: The tabular agent to train.
        num_episodes: Number of episodes to train.
        max_steps_per_episode: Maximum steps per episode.
        verbose: Whether to print progress.

    Returns:
        List of episode rewards.
    """
    var episode_rewards = List[Float64]()

    for episode in range(num_episodes):
        var state = env.reset()
        var total_reward: Float64 = 0.0

        for _ in range(max_steps_per_episode):
            var state_idx = env.state_to_index(state)
            var action_idx = agent.select_action(state_idx)
            var action = env.action_from_index(action_idx)

            var result = env.step(action)
            var next_state = result[0]
            var reward = result[1]
            var done = result[2]

            var next_state_idx = env.state_to_index(next_state)
            agent.update(state_idx, action_idx, reward, next_state_idx, done)

            total_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if verbose and (episode + 1) % 100 == 0:
            var avg_reward: Float64 = 0.0
            var start_idx = max(0, len(episode_rewards) - 100)
            for i in range(start_idx, len(episode_rewards)):
                avg_reward += episode_rewards[i]
            avg_reward /= Float64(len(episode_rewards) - start_idx)
            print(
                "Episode",
                episode + 1,
                "| Avg Reward (last 100):",
                avg_reward,
                "| Epsilon:",
                agent.get_epsilon(),
            )

    return episode_rewards^


fn evaluate_tabular[
    E: DiscreteEnv, A: TabularAgent
](
    mut env: E,
    agent: A,
    num_episodes: Int = 10,
    render: Bool = False,
) -> Float64:
    """Evaluate a trained tabular agent.

    Args:
        env: The discrete environment to evaluate on.
        agent: The trained agent.
        num_episodes: Number of evaluation episodes.
        render: Whether to render the environment.

    Returns:
        Average reward across episodes.
    """
    var total_reward: Float64 = 0.0

    for episode in range(num_episodes):
        var state = env.reset()
        var episode_reward: Float64 = 0.0

        if render:
            print("=== Evaluation Episode", episode + 1, "===")
            env.render()

        for step in range(100):
            var state_idx = env.state_to_index(state)
            var action_idx = agent.get_best_action(state_idx)
            var action = env.action_from_index(action_idx)

            var result = env.step(action)
            var next_state = result[0]
            var reward = result[1]
            var done = result[2]

            episode_reward += reward
            state = next_state

            if render:
                print("Action:", action_idx, "-> Reward:", reward)
                env.render()

            if done:
                if render:
                    print("Goal reached in", step + 1, "steps!")
                break

        total_reward += episode_reward

    return total_reward / Float64(num_episodes)


fn train_tabular_with_metrics[
    E: DiscreteEnv, A: TabularAgent
](
    mut env: E,
    mut agent: A,
    num_episodes: Int,
    max_steps_per_episode: Int = 100,
    verbose: Bool = False,
    print_every: Int = 100,
    algorithm_name: String = "Unknown",
    environment_name: String = "Unknown",
) -> TrainingMetrics:
    """Train a tabular RL agent and collect detailed metrics.

    This function is similar to train_tabular but returns a TrainingMetrics
    object instead of just rewards. The metrics can be exported to CSV for
    visualization with tools like matplotlib or pandas.

    Args:
        env: The discrete environment to train on.
        agent: The tabular agent to train.
        num_episodes: Number of episodes to train.
        max_steps_per_episode: Maximum steps per episode.
        verbose: Whether to print progress.
        print_every: Print progress every N episodes (if verbose).
        algorithm_name: Name of the algorithm (for metrics labeling).
        environment_name: Name of the environment (for metrics labeling).

    Returns:
        TrainingMetrics object with all episode data.
    """
    var metrics = TrainingMetrics(algorithm_name, environment_name)

    for episode in range(num_episodes):
        var state = env.reset()
        var total_reward: Float64 = 0.0
        var steps = 0

        for step in range(max_steps_per_episode):
            var state_idx = env.state_to_index(state)
            var action_idx = agent.select_action(state_idx)
            var action = env.action_from_index(action_idx)

            var result = env.step(action)
            var next_state = result[0]
            var reward = result[1]
            var done = result[2]

            var next_state_idx = env.state_to_index(next_state)
            agent.update(state_idx, action_idx, reward, next_state_idx, done)

            total_reward += reward
            steps = step + 1
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        var epsilon = agent.get_epsilon()
        metrics.log_episode(episode, total_reward, steps, epsilon)

        if verbose and (episode + 1) % print_every == 0:
            metrics.print_progress(episode, window=print_every)

    return metrics^
