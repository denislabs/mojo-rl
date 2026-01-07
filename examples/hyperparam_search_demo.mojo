"""Hyperparameter search demonstration for tabular agents.

This example demonstrates both grid search and random search for
hyperparameter optimization on Q-Learning with GridWorld.

Run with:
    pixi run mojo run examples/hyperparam_search_demo.mojo
"""

from envs.gridworld import GridWorldEnv
from agents.qlearning import QLearningAgent
from core.hyperparam.param_space import TabularParamSpace, TabularHyperparams
from core.hyperparam.search_result import SearchResults, TrialResult
from core.hyperparam.agent_factories import make_qlearning_agent


fn run_grid_search(
    num_episodes: Int = 300,
    max_steps: Int = 100,
    convergence_target: Float64 = 3.0,
    verbose: Bool = True,
) raises -> SearchResults:
    """Run grid search over Q-Learning hyperparameters on GridWorld.

    Args:
        num_episodes: Training episodes per trial.
        max_steps: Maximum steps per episode.
        convergence_target: Target reward for convergence metric.
        verbose: Print progress during search.

    Returns:
        SearchResults containing all trial outcomes.
    """
    # Create search space with reduced grid for faster demo
    var param_space = TabularParamSpace()
    # Reduce grid size for demo (3x3x2x2x2 = 72 combinations)
    param_space.learning_rate.num_values = 3
    param_space.discount_factor.num_values = 3
    param_space.epsilon.num_values = 2
    param_space.epsilon_decay.num_values = 2
    param_space.epsilon_min.num_values = 2

    var grid_size = param_space.get_grid_size()

    # Create results container
    var results = SearchResults(
        algorithm_name="Q-Learning",
        environment_name="GridWorld",
        search_type="grid",
        hyperparam_header=TabularHyperparams().to_csv_header(),
    )

    print("=" * 60)
    print("Grid Search: Q-Learning on GridWorld")
    print("=" * 60)
    print("Grid size:", grid_size, "configurations")
    print("Episodes per trial:", num_episodes)
    print("-" * 60)

    # Run grid search
    for trial_id in range(grid_size):
        var params = param_space.get_grid_config(trial_id)

        # Create fresh environment and agent
        var env = GridWorldEnv(width=5, height=5)
        var agent = make_qlearning_agent(
            env.num_states(), env.num_actions(), params
        )

        # Train
        var metrics = agent.train(
            env,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps,
            verbose=False,
        )

        # Record results
        var trial = TrialResult(
            trial_id=trial_id,
            hyperparams_str=params.to_csv_row(),
            metrics=metrics,
            convergence_target=convergence_target,
        )

        if verbose:
            print(
                "Trial",
                trial_id + 1,
                "/",
                grid_size,
                "| lr:",
                params.learning_rate,
                "| gamma:",
                params.discount_factor,
                "| Mean:",
                trial.mean_reward,
            )

        results.add_trial(trial^)

    return results^


fn run_random_search(
    num_trials: Int = 20,
    num_episodes: Int = 300,
    max_steps: Int = 100,
    convergence_target: Float64 = 3.0,
    verbose: Bool = True,
) raises -> SearchResults:
    """Run random search over Q-Learning hyperparameters on GridWorld.

    Args:
        num_trials: Number of random configurations to try.
        num_episodes: Training episodes per trial.
        max_steps: Maximum steps per episode.
        convergence_target: Target reward for convergence metric.
        verbose: Print progress during search.

    Returns:
        SearchResults containing all trial outcomes.
    """
    # Create search space
    var param_space = TabularParamSpace()

    # Create results container
    var results = SearchResults(
        algorithm_name="Q-Learning",
        environment_name="GridWorld",
        search_type="random",
        hyperparam_header=TabularHyperparams().to_csv_header(),
    )

    print("=" * 60)
    print("Random Search: Q-Learning on GridWorld")
    print("=" * 60)
    print("Number of trials:", num_trials)
    print("Episodes per trial:", num_episodes)
    print("-" * 60)

    # Run random search
    for trial_id in range(num_trials):
        var params = param_space.sample_random()

        # Create fresh environment and agent
        var env = GridWorldEnv(width=5, height=5)
        var agent = make_qlearning_agent(
            env.num_states(), env.num_actions(), params
        )

        # Train
        var metrics = agent.train(
            env,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps,
            verbose=False,
        )

        # Record results
        var trial = TrialResult(
            trial_id=trial_id,
            hyperparams_str=params.to_csv_row(),
            metrics=metrics,
            convergence_target=convergence_target,
        )

        if verbose:
            print(
                "Trial",
                trial_id + 1,
                "/",
                num_trials,
                "| lr:",
                params.learning_rate,
                "| gamma:",
                params.discount_factor,
                "| Mean:",
                trial.mean_reward,
            )

        results.add_trial(trial^)

    return results^


fn main() raises:
    """Run hyperparameter search demonstration."""
    print("")
    print("=" * 60)
    print("Hyperparameter Search Demo")
    print("=" * 60)
    print("")

    # Run a smaller grid search for demo
    print("Running Grid Search (reduced size for demo)...")
    print("")
    var grid_results = run_grid_search(
        num_episodes=200,
        max_steps=100,
        convergence_target=3.0,
        verbose=True,
    )

    print("")
    grid_results.print_summary()

    # Export to CSV
    grid_results.to_csv("grid_search_results.csv")
    print("Results saved to: grid_search_results.csv")

    print("")
    print("-" * 60)
    print("")

    # Run random search
    print("Running Random Search...")
    print("")
    var random_results = run_random_search(
        num_trials=15,
        num_episodes=200,
        max_steps=100,
        convergence_target=3.0,
        verbose=True,
    )

    print("")
    random_results.print_summary()

    # Export to CSV
    random_results.to_csv("random_search_results.csv")
    print("Results saved to: random_search_results.csv")

    # Compare best results
    print("")
    print("=" * 60)
    print("Comparison: Grid vs Random Search")
    print("=" * 60)

    var grid_best = grid_results.get_best_by_mean_reward()
    var random_best = random_results.get_best_by_mean_reward()

    print("Grid search best mean reward:", grid_best.mean_reward)
    print("  Hyperparameters:", grid_best.hyperparams_str)
    print("")
    print("Random search best mean reward:", random_best.mean_reward)
    print("  Hyperparameters:", random_best.hyperparams_str)
    print("")

    if grid_best.mean_reward > random_best.mean_reward:
        print("Winner: Grid Search")
    elif random_best.mean_reward > grid_best.mean_reward:
        print("Winner: Random Search")
    else:
        print("Tie!")

    print("")
    print("Demo complete!")
