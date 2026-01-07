"""Random search implementation for hyperparameter optimization.

Random search samples hyperparameter configurations randomly from the
search space. This is often more efficient than grid search for
high-dimensional parameter spaces (Bergstra & Bengio, 2012).
"""

from core import TrainingMetrics
from core.hyperparam.param_space import TabularHyperparams, TabularParamSpace
from core.hyperparam.search_result import TrialResult, SearchResults


fn create_random_search_results(
    algorithm_name: String,
    environment_name: String,
    hyperparam_header: String,
) -> SearchResults:
    """Create a SearchResults container for random search.

    Args:
        algorithm_name: Name of the algorithm.
        environment_name: Name of the environment.
        hyperparam_header: CSV header for hyperparameters.

    Returns:
        Empty SearchResults container.
    """
    return SearchResults(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
        search_type="random",
        hyperparam_header=hyperparam_header,
    )


fn run_random_trial(
    trial_id: Int,
    hyperparams: TabularHyperparams,
    metrics: TrainingMetrics,
    convergence_target: Float64 = 0.0,
) -> TrialResult:
    """Create a TrialResult from a completed training run.

    Args:
        trial_id: Unique identifier for this trial.
        hyperparams: Hyperparameters used for this trial.
        metrics: Training metrics from the completed run.
        convergence_target: Target reward for convergence calculation.

    Returns:
        TrialResult containing all computed metrics.
    """
    return TrialResult(
        trial_id=trial_id,
        hyperparams_str=hyperparams.to_csv_row(),
        metrics=metrics,
        convergence_target=convergence_target,
    )


fn print_random_progress(
    trial_id: Int,
    num_trials: Int,
    hyperparams: TabularHyperparams,
    mean_reward: Float64,
):
    """Print progress during random search.

    Args:
        trial_id: Current trial number (0-indexed).
        num_trials: Total number of trials.
        hyperparams: Current hyperparameters.
        mean_reward: Mean reward from this trial.
    """
    print(
        "Trial",
        trial_id + 1,
        "/",
        num_trials,
        "| lr:",
        hyperparams.learning_rate,
        "| gamma:",
        hyperparams.discount_factor,
        "| eps:",
        hyperparams.epsilon,
        "| Mean reward:",
        mean_reward,
    )


fn print_random_search_header(num_trials: Int, num_episodes: Int, max_steps: Int):
    """Print header information for random search.

    Args:
        num_trials: Number of random configurations to try.
        num_episodes: Episodes per trial.
        max_steps: Max steps per episode.
    """
    print("=" * 60)
    print("Random Hyperparameter Search")
    print("=" * 60)
    print("Number of trials:", num_trials)
    print("Episodes per trial:", num_episodes)
    print("Max steps per episode:", max_steps)
    print("-" * 60)
