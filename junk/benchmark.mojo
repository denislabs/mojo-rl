"""Benchmark all RL algorithms on GridWorld."""

from core import evaluate_tabular
from envs import GridWorld
from agents import (
    QLearningAgent,
    SARSAAgent,
    ExpectedSARSAAgent,
    MonteCarloAgent,
    DoubleQLearningAgent,
)


fn main():
    print("=" * 60)
    print("    Comparing RL Algorithms on GridWorld (5x5)")
    print("=" * 60)
    print("")

    var num_episodes = 500

    # Q-Learning
    print(">>> Q-Learning")
    var env1 = GridWorld(width=5, height=5)
    var q_agent = QLearningAgent(env1.num_states(), env1.num_actions())
    var metrics = q_agent.train(env1, num_episodes=num_episodes, verbose=False)
    var q_eval = evaluate_tabular(env1, q_agent, num_episodes=10)
    print("  Evaluation avg reward:", q_eval)

    # SARSA
    print(">>> SARSA")
    var env2 = GridWorld(width=5, height=5)
    var sarsa_agent, _ = SARSAAgent.train(
        env2, num_episodes=num_episodes, verbose=False
    )
    var sarsa_eval = evaluate_tabular(env2, sarsa_agent, num_episodes=10)
    print("  Evaluation avg reward:", sarsa_eval)

    # Expected SARSA
    print(">>> Expected SARSA")
    var env2b = GridWorld(width=5, height=5)
    var exp_sarsa_agent, _ = ExpectedSARSAAgent.train(
        env2b, num_episodes=num_episodes, verbose=False
    )
    var exp_sarsa_eval = evaluate_tabular(
        env2b, exp_sarsa_agent, num_episodes=10
    )
    print("  Evaluation avg reward:", exp_sarsa_eval)

    # Monte Carlo
    print(">>> Monte Carlo")
    var env3 = GridWorld(width=5, height=5)
    var mc_agent, _ = MonteCarloAgent.train(
        env3, num_episodes=num_episodes, verbose=False
    )
    var mc_eval = evaluate_tabular(env3, mc_agent, num_episodes=10)
    print("  Evaluation avg reward:", mc_eval)

    # Double Q-Learning
    print(">>> Double Q-Learning")
    var env4 = GridWorld(width=5, height=5)
    var dq_agent, _ = DoubleQLearningAgent.train(
        env4, num_episodes=num_episodes, verbose=False
    )
    var dq_eval = evaluate_tabular(env4, dq_agent, num_episodes=10)
    print("  Evaluation avg reward:", dq_eval)

    # Summary
    print("")
    print("=" * 60)
    print("    Summary (optimal = 3.0)")
    print("=" * 60)
    print("  Q-Learning:        ", q_eval)
    print("  SARSA:             ", sarsa_eval)
    print("  Expected SARSA:    ", exp_sarsa_eval)
    print("  Monte Carlo:       ", mc_eval)
    print("  Double Q-Learning: ", dq_eval)
