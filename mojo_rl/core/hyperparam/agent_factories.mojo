"""Agent factory functions for hyperparameter search.

This module provides factory functions that create agents with specific
hyperparameter configurations. Since Mojo doesn't support passing
constructors as parameters, these factories are used in the search
examples to create agents from hyperparameter structs.
"""

from agents.qlearning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.expected_sarsa import ExpectedSARSAAgent
from agents.double_qlearning import DoubleQLearningAgent
from agents.monte_carlo import MonteCarloAgent
from agents.nstep_sarsa import NStepSARSAAgent
from agents.sarsa_lambda import SARSALambdaAgent
from agents.dyna_q import DynaQAgent

from core.hyperparam.param_space import (
    TabularHyperparams,
    NStepHyperparams,
    LambdaHyperparams,
    ModelBasedHyperparams,
)


# ============================================================================
# Tabular Agent Factories
# ============================================================================


fn make_qlearning_agent(
    num_states: Int,
    num_actions: Int,
    params: TabularHyperparams,
) -> QLearningAgent:
    """Create a Q-Learning agent with specified hyperparameters.

    Args:
        num_states: Number of discrete states.
        num_actions: Number of discrete actions.
        params: Hyperparameters for the agent.

    Returns:
        Configured QLearningAgent.
    """
    return QLearningAgent(
        num_states=num_states,
        num_actions=num_actions,
        learning_rate=params.learning_rate,
        discount_factor=params.discount_factor,
        epsilon=params.epsilon,
        epsilon_decay=params.epsilon_decay,
        epsilon_min=params.epsilon_min,
    )


fn make_sarsa_agent(
    num_states: Int,
    num_actions: Int,
    params: TabularHyperparams,
) -> SARSAAgent:
    """Create a SARSA agent with specified hyperparameters.

    Args:
        num_states: Number of discrete states.
        num_actions: Number of discrete actions.
        params: Hyperparameters for the agent.

    Returns:
        Configured SARSAAgent.
    """
    return SARSAAgent(
        num_states=num_states,
        num_actions=num_actions,
        learning_rate=params.learning_rate,
        discount_factor=params.discount_factor,
        epsilon=params.epsilon,
        epsilon_decay=params.epsilon_decay,
        epsilon_min=params.epsilon_min,
    )


fn make_expected_sarsa_agent(
    num_states: Int,
    num_actions: Int,
    params: TabularHyperparams,
) -> ExpectedSARSAAgent:
    """Create an Expected SARSA agent with specified hyperparameters.

    Args:
        num_states: Number of discrete states.
        num_actions: Number of discrete actions.
        params: Hyperparameters for the agent.

    Returns:
        Configured ExpectedSARSAAgent.
    """
    return ExpectedSARSAAgent(
        num_states=num_states,
        num_actions=num_actions,
        learning_rate=params.learning_rate,
        discount_factor=params.discount_factor,
        epsilon=params.epsilon,
        epsilon_decay=params.epsilon_decay,
        epsilon_min=params.epsilon_min,
    )


fn make_double_qlearning_agent(
    num_states: Int,
    num_actions: Int,
    params: TabularHyperparams,
) -> DoubleQLearningAgent:
    """Create a Double Q-Learning agent with specified hyperparameters.

    Args:
        num_states: Number of discrete states.
        num_actions: Number of discrete actions.
        params: Hyperparameters for the agent.

    Returns:
        Configured DoubleQLearningAgent.
    """
    return DoubleQLearningAgent(
        num_states=num_states,
        num_actions=num_actions,
        learning_rate=params.learning_rate,
        discount_factor=params.discount_factor,
        epsilon=params.epsilon,
        epsilon_decay=params.epsilon_decay,
        epsilon_min=params.epsilon_min,
    )


fn make_monte_carlo_agent(
    num_states: Int,
    num_actions: Int,
    params: TabularHyperparams,
) -> MonteCarloAgent:
    """Create a Monte Carlo agent with specified hyperparameters.

    Args:
        num_states: Number of discrete states.
        num_actions: Number of discrete actions.
        params: Hyperparameters for the agent.

    Returns:
        Configured MonteCarloAgent.
    """
    return MonteCarloAgent(
        num_states=num_states,
        num_actions=num_actions,
        discount_factor=params.discount_factor,
        epsilon=params.epsilon,
        epsilon_decay=params.epsilon_decay,
        epsilon_min=params.epsilon_min,
    )


# ============================================================================
# N-Step Agent Factories
# ============================================================================


fn make_nstep_sarsa_agent(
    num_states: Int,
    num_actions: Int,
    params: NStepHyperparams,
) -> NStepSARSAAgent:
    """Create an N-step SARSA agent with specified hyperparameters.

    Args:
        num_states: Number of discrete states.
        num_actions: Number of discrete actions.
        params: Hyperparameters including n (number of steps).

    Returns:
        Configured NStepSARSAAgent.
    """
    return NStepSARSAAgent(
        num_states=num_states,
        num_actions=num_actions,
        n=params.n,
        learning_rate=params.learning_rate,
        discount_factor=params.discount_factor,
        epsilon=params.epsilon,
        epsilon_decay=params.epsilon_decay,
        epsilon_min=params.epsilon_min,
    )


# ============================================================================
# Lambda (Eligibility Trace) Agent Factories
# ============================================================================


fn make_sarsa_lambda_agent(
    num_states: Int,
    num_actions: Int,
    params: LambdaHyperparams,
) -> SARSALambdaAgent:
    """Create a SARSA(lambda) agent with specified hyperparameters.

    Args:
        num_states: Number of discrete states.
        num_actions: Number of discrete actions.
        params: Hyperparameters including lambda (trace decay).

    Returns:
        Configured SARSALambdaAgent.
    """
    return SARSALambdaAgent(
        num_states=num_states,
        num_actions=num_actions,
        lambda_=params.lambda_,
        learning_rate=params.learning_rate,
        discount_factor=params.discount_factor,
        epsilon=params.epsilon,
        epsilon_decay=params.epsilon_decay,
        epsilon_min=params.epsilon_min,
    )


# ============================================================================
# Model-Based Agent Factories
# ============================================================================


fn make_dyna_q_agent(
    num_states: Int,
    num_actions: Int,
    params: ModelBasedHyperparams,
) -> DynaQAgent:
    """Create a Dyna-Q agent with specified hyperparameters.

    Args:
        num_states: Number of discrete states.
        num_actions: Number of discrete actions.
        params: Hyperparameters including n_planning.

    Returns:
        Configured DynaQAgent.
    """
    return DynaQAgent(
        num_states=num_states,
        num_actions=num_actions,
        n_planning=params.n_planning,
        learning_rate=params.learning_rate,
        discount_factor=params.discount_factor,
        epsilon=params.epsilon,
        epsilon_decay=params.epsilon_decay,
        epsilon_min=params.epsilon_min,
    )
