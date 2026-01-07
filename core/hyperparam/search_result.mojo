"""Search result storage and aggregation for hyperparameter optimization.

This module provides structs for storing individual trial results and
aggregating them for analysis and export.
"""

from core.metrics import TrainingMetrics, compute_convergence_episode


struct TrialResult(Copyable, Movable, ImplicitlyCopyable):
    """Result from a single hyperparameter trial.

    Computes and stores various metrics from training to enable
    comparison and selection of best hyperparameters.
    """

    var trial_id: Int
    var hyperparams_str: String

    # Primary metrics
    var mean_reward: Float64
    var max_reward: Float64
    var min_reward: Float64
    var std_reward: Float64
    var final_reward: Float64

    # Convergence metrics
    var convergence_episode: Int
    var convergence_target: Float64

    # Efficiency metrics
    var mean_steps: Float64
    var total_steps: Int
    var training_episodes: Int

    fn __init__(
        out self,
        trial_id: Int,
        hyperparams_str: String,
        metrics: TrainingMetrics,
        convergence_target: Float64 = 0.0,
        final_window: Int = 100,
    ):
        """Compute trial result from training metrics.

        Args:
            trial_id: Unique identifier for this trial.
            hyperparams_str: CSV-formatted hyperparameter values.
            metrics: TrainingMetrics from agent training.
            convergence_target: Target reward for convergence calculation.
            final_window: Window size for final reward calculation.
        """
        self.trial_id = trial_id
        self.hyperparams_str = hyperparams_str

        # Basic metrics
        self.mean_reward = metrics.mean_reward()
        self.max_reward = metrics.max_reward()
        self.min_reward = metrics.min_reward()
        self.std_reward = metrics.std_reward()
        self.mean_steps = metrics.mean_steps()
        self.training_episodes = metrics.num_episodes()

        # Compute final reward (average of last N episodes)
        var rewards = metrics.get_rewards()
        var num_rewards = len(rewards)
        if num_rewards > 0:
            var start_idx = num_rewards - final_window
            if start_idx < 0:
                start_idx = 0
            var final_sum: Float64 = 0.0
            var count = 0
            for i in range(start_idx, num_rewards):
                final_sum += rewards[i]
                count += 1
            self.final_reward = final_sum / Float64(count)
        else:
            self.final_reward = 0.0

        # Compute total steps
        var steps = metrics.get_steps()
        self.total_steps = 0
        for i in range(len(steps)):
            self.total_steps += steps[i]

        # Compute convergence
        self.convergence_target = convergence_target
        if convergence_target > 0.0 and num_rewards > 0:
            self.convergence_episode = compute_convergence_episode(
                rewards, convergence_target, window=100, threshold=0.95
            )
        else:
            self.convergence_episode = -1

    fn __copyinit__(out self, existing: Self):
        self.trial_id = existing.trial_id
        self.hyperparams_str = existing.hyperparams_str
        self.mean_reward = existing.mean_reward
        self.max_reward = existing.max_reward
        self.min_reward = existing.min_reward
        self.std_reward = existing.std_reward
        self.final_reward = existing.final_reward
        self.convergence_episode = existing.convergence_episode
        self.convergence_target = existing.convergence_target
        self.mean_steps = existing.mean_steps
        self.total_steps = existing.total_steps
        self.training_episodes = existing.training_episodes

    fn __moveinit__(out self, deinit existing: Self):
        self.trial_id = existing.trial_id
        self.hyperparams_str = existing.hyperparams_str^
        self.mean_reward = existing.mean_reward
        self.max_reward = existing.max_reward
        self.min_reward = existing.min_reward
        self.std_reward = existing.std_reward
        self.final_reward = existing.final_reward
        self.convergence_episode = existing.convergence_episode
        self.convergence_target = existing.convergence_target
        self.mean_steps = existing.mean_steps
        self.total_steps = existing.total_steps
        self.training_episodes = existing.training_episodes

    @staticmethod
    fn csv_header() -> String:
        """Return CSV header for trial metrics."""
        return "trial_id,mean_reward,max_reward,min_reward,std_reward,final_reward,convergence_episode,mean_steps,total_steps,training_episodes"

    fn to_csv_row(self) -> String:
        """Return CSV row for trial metrics."""
        return (
            String(self.trial_id)
            + ","
            + String(self.mean_reward)
            + ","
            + String(self.max_reward)
            + ","
            + String(self.min_reward)
            + ","
            + String(self.std_reward)
            + ","
            + String(self.final_reward)
            + ","
            + String(self.convergence_episode)
            + ","
            + String(self.mean_steps)
            + ","
            + String(self.total_steps)
            + ","
            + String(self.training_episodes)
        )


struct SearchResults(Copyable, Movable):
    """Aggregated results from hyperparameter search.

    Stores all trial results and provides methods for finding
    best configurations and exporting results.
    """

    var algorithm_name: String
    var environment_name: String
    var search_type: String
    var trials: List[TrialResult]
    var hyperparam_header: String

    fn __init__(
        out self,
        algorithm_name: String,
        environment_name: String,
        search_type: String,
        hyperparam_header: String,
    ):
        """Initialize search results container.

        Args:
            algorithm_name: Name of the algorithm being tuned.
            environment_name: Name of the environment.
            search_type: Type of search ("grid" or "random").
            hyperparam_header: CSV header for hyperparameters.
        """
        self.algorithm_name = algorithm_name
        self.environment_name = environment_name
        self.search_type = search_type
        self.trials = List[TrialResult]()
        self.hyperparam_header = hyperparam_header

    fn __copyinit__(out self, existing: Self):
        self.algorithm_name = existing.algorithm_name
        self.environment_name = existing.environment_name
        self.search_type = existing.search_type
        self.trials = List[TrialResult]()
        for i in range(len(existing.trials)):
            self.trials.append(existing.trials[i])
        self.hyperparam_header = existing.hyperparam_header

    fn __moveinit__(out self, deinit existing: Self):
        self.algorithm_name = existing.algorithm_name^
        self.environment_name = existing.environment_name^
        self.search_type = existing.search_type^
        self.trials = existing.trials^
        self.hyperparam_header = existing.hyperparam_header^

    fn add_trial(mut self, var trial: TrialResult):
        """Add a trial result to the collection."""
        self.trials.append(trial^)

    fn num_trials(self) -> Int:
        """Return the number of trials."""
        return len(self.trials)

    fn get_best_by_mean_reward(self) -> TrialResult:
        """Return trial with highest mean reward.

        Returns:
            TrialResult with the best mean reward.
        """
        if len(self.trials) == 0:
            # Return empty result if no trials
            return TrialResult(
                trial_id=-1,
                hyperparams_str="",
                metrics=TrainingMetrics(),
            )

        var best_idx = 0
        var best_val = self.trials[0].mean_reward
        for i in range(1, len(self.trials)):
            if self.trials[i].mean_reward > best_val:
                best_val = self.trials[i].mean_reward
                best_idx = i
        return self.trials[best_idx]

    fn get_best_by_final_reward(self) -> TrialResult:
        """Return trial with highest final reward.

        Final reward is the average of the last N episodes,
        which often better reflects the converged performance.

        Returns:
            TrialResult with the best final reward.
        """
        if len(self.trials) == 0:
            return TrialResult(
                trial_id=-1,
                hyperparams_str="",
                metrics=TrainingMetrics(),
            )

        var best_idx = 0
        var best_val = self.trials[0].final_reward
        for i in range(1, len(self.trials)):
            if self.trials[i].final_reward > best_val:
                best_val = self.trials[i].final_reward
                best_idx = i
        return self.trials[best_idx]

    fn get_best_by_max_reward(self) -> TrialResult:
        """Return trial with highest max reward.

        Returns:
            TrialResult with the best max reward achieved.
        """
        if len(self.trials) == 0:
            return TrialResult(
                trial_id=-1,
                hyperparams_str="",
                metrics=TrainingMetrics(),
            )

        var best_idx = 0
        var best_val = self.trials[0].max_reward
        for i in range(1, len(self.trials)):
            if self.trials[i].max_reward > best_val:
                best_val = self.trials[i].max_reward
                best_idx = i
        return self.trials[best_idx]

    fn get_best_by_convergence(self) -> TrialResult:
        """Return trial with fastest convergence.

        Finds the trial that converged to the target reward
        in the fewest episodes. If no trial converged, returns
        the trial with best mean reward.

        Returns:
            TrialResult with fastest convergence or best mean reward.
        """
        if len(self.trials) == 0:
            return TrialResult(
                trial_id=-1,
                hyperparams_str="",
                metrics=TrainingMetrics(),
            )

        var best_idx = -1
        var best_ep = -1
        for i in range(len(self.trials)):
            var ep = self.trials[i].convergence_episode
            if ep > 0:  # Only consider converged trials
                if best_idx == -1 or ep < best_ep:
                    best_ep = ep
                    best_idx = i

        if best_idx == -1:
            # No trial converged, return best by mean reward
            return self.get_best_by_mean_reward()
        return self.trials[best_idx]

    fn get_trial(self, idx: Int) -> TrialResult:
        """Get trial result by index."""
        return self.trials[idx]

    fn to_csv(self, filepath: String) raises:
        """Export all results to CSV.

        The CSV file contains all hyperparameters and metrics
        for each trial, with one trial per row.

        Args:
            filepath: Path to write the CSV file.
        """
        var header = self.hyperparam_header + "," + TrialResult.csv_header()
        var content = header + "\n"

        for i in range(len(self.trials)):
            content += self.trials[i].hyperparams_str + ","
            content += self.trials[i].to_csv_row() + "\n"

        with open(filepath, "w") as f:
            f.write(content)

    fn print_summary(self):
        """Print search summary to stdout."""
        print("=" * 60)
        print("Hyperparameter Search Results")
        print("=" * 60)
        print("Algorithm:", self.algorithm_name)
        print("Environment:", self.environment_name)
        print("Search type:", self.search_type)
        print("Total trials:", len(self.trials))
        print("-" * 60)

        if len(self.trials) == 0:
            print("No trials completed.")
            print("=" * 60)
            return

        var best_mean = self.get_best_by_mean_reward()
        print("Best by mean reward:")
        print("  Trial", best_mean.trial_id, "- Mean:", best_mean.mean_reward)
        print("  Params:", best_mean.hyperparams_str)

        var best_final = self.get_best_by_final_reward()
        print("Best by final reward:")
        print("  Trial", best_final.trial_id, "- Final:", best_final.final_reward)
        print("  Params:", best_final.hyperparams_str)

        var best_conv = self.get_best_by_convergence()
        if best_conv.convergence_episode > 0:
            print("Best by convergence speed:")
            print(
                "  Trial",
                best_conv.trial_id,
                "- Converged at episode:",
                best_conv.convergence_episode,
            )
            print("  Params:", best_conv.hyperparams_str)
        else:
            print("Best by convergence: No trials converged to target")

        print("=" * 60)

    fn print_all_trials(self):
        """Print all trial results in a table format."""
        print("-" * 80)
        print(
            "Trial | Mean Reward | Final Reward | Max Reward | Convergence | Steps"
        )
        print("-" * 80)

        for i in range(len(self.trials)):
            var t = self.trials[i]
            var conv_str: String
            if t.convergence_episode > 0:
                conv_str = String(t.convergence_episode)
            else:
                conv_str = "N/A"
            print(
                t.trial_id,
                "   |",
                t.mean_reward,
                "|",
                t.final_reward,
                "|",
                t.max_reward,
                "|",
                conv_str,
                "|",
                t.mean_steps,
            )
        print("-" * 80)

    fn get_statistics(self) -> Tuple[Float64, Float64, Float64, Float64]:
        """Get statistics across all trials.

        Returns:
            Tuple of (mean of means, std of means, best mean, worst mean).
        """
        if len(self.trials) == 0:
            return (0.0, 0.0, 0.0, 0.0)

        var sum_val: Float64 = 0.0
        var best_val = self.trials[0].mean_reward
        var worst_val = self.trials[0].mean_reward

        for i in range(len(self.trials)):
            var r = self.trials[i].mean_reward
            sum_val += r
            if r > best_val:
                best_val = r
            if r < worst_val:
                worst_val = r

        var mean_val = sum_val / Float64(len(self.trials))

        # Compute std
        var sum_sq: Float64 = 0.0
        for i in range(len(self.trials)):
            var diff = self.trials[i].mean_reward - mean_val
            sum_sq += diff * diff
        var std_val = (sum_sq / Float64(len(self.trials))) ** 0.5

        return (mean_val, std_val, best_val, worst_val)
