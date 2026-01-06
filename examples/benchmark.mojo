"""Benchmark all RL algorithms on GridWorld."""

from core import evaluate_tabular
from envs import GridWorldEnv
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
    var env1 = GridWorldEnv(width=5, height=5)
    var q_agent = QLearningAgent(env1.num_states(), env1.num_actions())
    var _ = q_agent.train(env1, num_episodes=num_episodes, verbose=False)
    var q_eval = q_agent.evaluate(env1, num_episodes=10)
    print("  Evaluation avg reward:", q_eval)

    # SARSA
    print(">>> SARSA")
    var env2 = GridWorldEnv(width=5, height=5)
    var sarsa_agent = SARSAAgent(env2.num_states(), env2.num_actions())
    var _ = sarsa_agent.train(env2, num_episodes=num_episodes, verbose=False)
    var sarsa_eval = sarsa_agent.evaluate(env2, num_episodes=10)
    print("  Evaluation avg reward:", sarsa_eval)

    # Expected SARSA
    print(">>> Expected SARSA")
    var env2b = GridWorldEnv(width=5, height=5)
    var exp_sarsa_agent = ExpectedSARSAAgent(
        env2b.num_states(), env2b.num_actions()
    )
    var _ = exp_sarsa_agent.train(
        env2b, num_episodes=num_episodes, verbose=False
    )
    var exp_sarsa_eval = exp_sarsa_agent.evaluate(env2b, num_episodes=10)
    print("  Evaluation avg reward:", exp_sarsa_eval)

    # Monte Carlo
    print(">>> Monte Carlo")
    var env3 = GridWorldEnv(width=5, height=5)
    var mc_agent = MonteCarloAgent(env3.num_states(), env3.num_actions())
    var _get_action_probs_idx = mc_agent.train(
        env3, num_episodes=num_episodes, verbose=False
    )
    var mc_eval = mc_agent.evaluate(env3, num_episodes=10)
    print("  Evaluation avg reward:", mc_eval)

    # Double Q-Learning
    print(">>> Double Q-Learning")
    var env4 = GridWorldEnv(width=5, height=5)
    var dq_agent = DoubleQLearningAgent(env4.num_states(), env4.num_actions())
    var _ = dq_agent.train(env4, num_episodes=num_episodes, verbose=False)
    var dq_eval = dq_agent.evaluate(env4, num_episodes=10)
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
