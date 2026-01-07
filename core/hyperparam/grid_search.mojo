"""Grid search implementation for hyperparameter optimization.

Grid search exhaustively evaluates all combinations of hyperparameter
values in the search space. This is effective for smaller spaces but
grows exponentially with the number of parameters.
"""

from core import DiscreteEnv, TrainingMetrics
from core.hyperparam.param_space import TabularHyperparams, TabularParamSpace
from core.hyperparam.search_result import TrialResult, SearchResults


fn grid_search_tabular(
    param_space: TabularParamSpace,
    num_states: Int,
    num_actions: Int,
    num_episodes: Int,
    max_steps: Int,
    algorithm_name: String,
    environment_name: String,
    convergence_target: Float64 = 0.0,
    verbose: Bool = False,
    print_every: Int = 1,
    # Agent training function - we pass a closure-like approach via callbacks
) -> SearchResults:
    """Grid search over tabular agent hyperparameters.

    This function provides the framework for grid search. Since Mojo doesn't
    support passing agent constructors directly, the actual implementation
    should be done in example files where the agent type is known.

    Args:
        param_space: Parameter space to search over.
        num_states: Number of states in the environment.
        num_actions: Number of actions in the environment.
        num_episodes: Episodes per trial.
        max_steps: Max steps per episode.
        algorithm_name: Name of the algorithm for logging.
        environment_name: Name of the environment for logging.
        convergence_target: Target reward for convergence metric.
        verbose: Print progress.
        print_every: Print every N trials.

    Returns:
        SearchResults with all trial outcomes.
    """
    var results = SearchResults(
        algorithm_name=algorithm_name,
        environment_name=environment_name,
        search_type="grid",
        hyperparam_header=TabularHyperparams().to_csv_header(),
    )

    var grid_size = param_space.get_grid_size()
    if verbose:
        print("Grid search:", grid_size, "configurations")
        print("Parameters per trial:", num_episodes, "episodes x", max_steps, "max steps")

    # Note: Actual training loop must be implemented in the calling code
    # since Mojo doesn't support passing agent constructors as parameters

    return results^


fn create_grid_search_results(
    algorithm_name: String,
    environment_name: String,
    hyperparam_header: String,
) -> SearchResults:
    """Create a SearchResults container for grid search.

    This is a helper function to create the results container
    before running the search loop.

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
        search_type="grid",
        hyperparam_header=hyperparam_header,
    )


fn run_grid_trial(
    trial_id: Int,
    hyperparams: TabularHyperparams,
    metrics: TrainingMetrics,
    convergence_target: Float64 = 0.0,
) -> TrialResult:
    """Create a TrialResult from a completed training run.

    This helper function wraps the metrics from a training run
    into a TrialResult for storage in SearchResults.

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


fn print_grid_progress(
    trial_id: Int,
    grid_size: Int,
    hyperparams: TabularHyperparams,
    mean_reward: Float64,
):
    """Print progress during grid search.

    Args:
        trial_id: Current trial number (0-indexed).
        grid_size: Total number of trials.
        hyperparams: Current hyperparameters.
        mean_reward: Mean reward from this trial.
    """
    print(
        "Trial",
        trial_id + 1,
        "/",
        grid_size,
        "| lr:",
        hyperparams.learning_rate,
        "| gamma:",
        hyperparams.discount_factor,
        "| eps:",
        hyperparams.epsilon,
        "| Mean reward:",
        mean_reward,
    )
