"""Hyperparameter search module for mojo-rl.

This module provides infrastructure for hyperparameter optimization
including grid search and random search over RL agent configurations.

Usage:
    from core.hyperparam import TabularParamSpace, TabularHyperparams
    from core.hyperparam import SearchResults, TrialResult
    from core.hyperparam import make_qlearning_agent

    # Define search space
    var param_space = TabularParamSpace()

    # For grid search: iterate over all configurations
    for i in range(param_space.get_grid_size()):
        var params = param_space.get_grid_config(i)
        var agent = make_qlearning_agent(num_states, num_actions, params)
        # ... train and record results

    # For random search: sample N configurations
    for i in range(num_trials):
        var params = param_space.sample_random()
        var agent = make_qlearning_agent(num_states, num_actions, params)
        # ... train and record results
"""

# Parameter space definitions
from .param_space import (
    FloatParam,
    IntParam,
    BoolParam,
    TabularHyperparams,
    TabularParamSpace,
    NStepHyperparams,
    NStepParamSpace,
    LambdaHyperparams,
    LambdaParamSpace,
    ModelBasedHyperparams,
    ModelBasedParamSpace,
    ReplayHyperparams,
    ReplayParamSpace,
    PolicyGradientHyperparams,
    PolicyGradientParamSpace,
    PPOHyperparams,
    PPOParamSpace,
    ContinuousHyperparams,
    ContinuousParamSpace,
)

# Search results
from .search_result import (
    TrialResult,
    SearchResults,
)

# Grid search utilities
from .grid_search import (
    create_grid_search_results,
    run_grid_trial,
    print_grid_progress,
)

# Random search utilities
from .random_search import (
    create_random_search_results,
    run_random_trial,
    print_random_progress,
    print_random_search_header,
)

# Agent factories
from .agent_factories import (
    make_qlearning_agent,
    make_sarsa_agent,
    make_expected_sarsa_agent,
    make_double_qlearning_agent,
    make_monte_carlo_agent,
    make_nstep_sarsa_agent,
    make_sarsa_lambda_agent,
    make_dyna_q_agent,
)
