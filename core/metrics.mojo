"""Logging and metrics infrastructure for tracking training progress.

This module provides structures and functions for collecting training metrics
and exporting them to CSV format for visualization with tools like matplotlib,
pandas, or any spreadsheet application.
"""

from math import sqrt


struct EpisodeMetrics(Copyable, Movable, ImplicitlyCopyable):
    """Metrics collected for a single episode."""

    var episode: Int
    var total_reward: Float64
    var steps: Int
    var epsilon: Float64

    fn __init__(out self, episode: Int, total_reward: Float64, steps: Int, epsilon: Float64):
        self.episode = episode
        self.total_reward = total_reward
        self.steps = steps
        self.epsilon = epsilon

    fn __copyinit__(out self, existing: Self):
        self.episode = existing.episode
        self.total_reward = existing.total_reward
        self.steps = existing.steps
        self.epsilon = existing.epsilon

    fn __moveinit__(out self, deinit existing: Self):
        self.episode = existing.episode
        self.total_reward = existing.total_reward
        self.steps = existing.steps
        self.epsilon = existing.epsilon


struct TrainingMetrics(Movable):
    """Collects and manages training metrics across episodes.

    This struct accumulates episode-level metrics during training and provides
    methods for computing statistics and exporting data for visualization.

    Example:
        var metrics = TrainingMetrics()
        # During training loop:
        metrics.log_episode(episode, reward, steps, epsilon)
        # After training:
        metrics.to_csv("training_log.csv")
    """

    var episodes: List[EpisodeMetrics]
    var algorithm_name: String
    var environment_name: String

    fn __init__(out self, algorithm_name: String = "Unknown", environment_name: String = "Unknown"):
        """Initialize training metrics collector.

        Args:
            algorithm_name: Name of the algorithm being trained.
            environment_name: Name of the environment.
        """
        self.episodes = List[EpisodeMetrics]()
        self.algorithm_name = algorithm_name
        self.environment_name = environment_name

    fn __moveinit__(out self, deinit existing: Self):
        self.episodes = existing.episodes^
        self.algorithm_name = existing.algorithm_name^
        self.environment_name = existing.environment_name^

    fn log_episode(mut self, episode: Int, total_reward: Float64, steps: Int, epsilon: Float64):
        """Log metrics for a completed episode.

        Args:
            episode: Episode number (0-indexed).
            total_reward: Total reward accumulated in the episode.
            steps: Number of steps taken in the episode.
            epsilon: Current epsilon value at end of episode.
        """
        self.episodes.append(EpisodeMetrics(episode, total_reward, steps, epsilon))

    fn num_episodes(self) -> Int:
        """Return the number of logged episodes."""
        return len(self.episodes)

    fn get_rewards(self) -> List[Float64]:
        """Return a list of all episode rewards."""
        var rewards = List[Float64]()
        for i in range(len(self.episodes)):
            rewards.append(self.episodes[i].total_reward)
        return rewards^

    fn get_steps(self) -> List[Int]:
        """Return a list of all episode step counts."""
        var steps = List[Int]()
        for i in range(len(self.episodes)):
            steps.append(self.episodes[i].steps)
        return steps^

    fn mean_reward(self) -> Float64:
        """Compute mean reward across all episodes."""
        if len(self.episodes) == 0:
            return 0.0
        var total: Float64 = 0.0
        for i in range(len(self.episodes)):
            total += self.episodes[i].total_reward
        return total / Float64(len(self.episodes))

    fn std_reward(self) -> Float64:
        """Compute standard deviation of rewards across all episodes."""
        if len(self.episodes) == 0:
            return 0.0
        var mean = self.mean_reward()
        var sum_sq: Float64 = 0.0
        for i in range(len(self.episodes)):
            var diff = self.episodes[i].total_reward - mean
            sum_sq += diff * diff
        return sqrt(sum_sq / Float64(len(self.episodes)))

    fn mean_steps(self) -> Float64:
        """Compute mean steps per episode."""
        if len(self.episodes) == 0:
            return 0.0
        var total: Float64 = 0.0
        for i in range(len(self.episodes)):
            total += Float64(self.episodes[i].steps)
        return total / Float64(len(self.episodes))

    fn max_reward(self) -> Float64:
        """Return maximum reward achieved."""
        if len(self.episodes) == 0:
            return 0.0
        var max_r = self.episodes[0].total_reward
        for i in range(1, len(self.episodes)):
            if self.episodes[i].total_reward > max_r:
                max_r = self.episodes[i].total_reward
        return max_r

    fn min_reward(self) -> Float64:
        """Return minimum reward achieved."""
        if len(self.episodes) == 0:
            return 0.0
        var min_r = self.episodes[0].total_reward
        for i in range(1, len(self.episodes)):
            if self.episodes[i].total_reward < min_r:
                min_r = self.episodes[i].total_reward
        return min_r

    fn moving_average(self, window: Int = 100) -> List[Float64]:
        """Compute moving average of rewards.

        Args:
            window: Window size for moving average.

        Returns:
            List of moving average values (one per episode).
        """
        var result = List[Float64]()
        for i in range(len(self.episodes)):
            var start_idx = max(0, i - window + 1)
            var sum_val: Float64 = 0.0
            var count = 0
            for j in range(start_idx, i + 1):
                sum_val += self.episodes[j].total_reward
                count += 1
            result.append(sum_val / Float64(count))
        return result^

    fn to_csv(self, filepath: String) raises:
        """Export metrics to a CSV file for visualization.

        The CSV file will contain columns:
        - episode: Episode number
        - reward: Total episode reward
        - steps: Steps taken in episode
        - epsilon: Epsilon value at end of episode
        - moving_avg_100: 100-episode moving average of reward

        Args:
            filepath: Path to write the CSV file.
        """
        var moving_avg = self.moving_average(100)

        # Build CSV content
        var content = String("episode,reward,steps,epsilon,moving_avg_100\n")

        for i in range(len(self.episodes)):
            var ep = self.episodes[i]
            content += String(ep.episode) + ","
            content += String(ep.total_reward) + ","
            content += String(ep.steps) + ","
            content += String(ep.epsilon) + ","
            content += String(moving_avg[i]) + "\n"

        # Write to file
        with open(filepath, "w") as f:
            f.write(content)

    fn print_summary(self):
        """Print a summary of training metrics to stdout."""
        print("=" * 50)
        print("Training Summary")
        print("=" * 50)
        print("Algorithm:", self.algorithm_name)
        print("Environment:", self.environment_name)
        print("Episodes:", self.num_episodes())
        print("-" * 50)
        print("Reward Statistics:")
        print("  Mean:    ", self.mean_reward())
        print("  Std:     ", self.std_reward())
        print("  Min:     ", self.min_reward())
        print("  Max:     ", self.max_reward())
        print("  Mean steps:", self.mean_steps())
        print("=" * 50)

    fn print_progress(self, episode: Int, window: Int = 100):
        """Print progress update during training.

        Args:
            episode: Current episode number.
            window: Window size for moving average calculation.
        """
        if len(self.episodes) == 0:
            return

        var ep = self.episodes[len(self.episodes) - 1]

        # Calculate moving average
        var start_idx = max(0, len(self.episodes) - window)
        var sum_val: Float64 = 0.0
        for i in range(start_idx, len(self.episodes)):
            sum_val += self.episodes[i].total_reward
        var avg = sum_val / Float64(len(self.episodes) - start_idx)

        print(
            "Episode", episode + 1,
            "| Reward:", ep.total_reward,
            "| Steps:", ep.steps,
            "| Avg(" + String(window) + "):", avg,
            "| Eps:", ep.epsilon
        )


fn compute_success_rate(rewards: List[Float64], threshold: Float64) -> Float64:
    """Compute the fraction of episodes with reward >= threshold.

    Args:
        rewards: List of episode rewards.
        threshold: Success threshold.

    Returns:
        Fraction of episodes that achieved reward >= threshold.
    """
    if len(rewards) == 0:
        return 0.0
    var count = 0
    for i in range(len(rewards)):
        if rewards[i] >= threshold:
            count += 1
    return Float64(count) / Float64(len(rewards))


fn compute_convergence_episode(rewards: List[Float64], target: Float64, window: Int = 100, threshold: Float64 = 0.95) -> Int:
    """Find the first episode where moving average reaches target.

    Args:
        rewards: List of episode rewards.
        target: Target reward value.
        window: Window size for moving average.
        threshold: Fraction of target that counts as "converged".

    Returns:
        Episode number where convergence occurred, or -1 if not converged.
    """
    var target_value = target * threshold

    for i in range(len(rewards)):
        var start_idx = max(0, i - window + 1)
        var sum_val: Float64 = 0.0
        var count = 0
        for j in range(start_idx, i + 1):
            sum_val += rewards[j]
            count += 1
        var avg = sum_val / Float64(count)
        if avg >= target_value:
            return i
    return -1
