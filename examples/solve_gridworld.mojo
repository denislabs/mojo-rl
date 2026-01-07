"""Solve GridWorld - Demonstrating tabular RL algorithms.

GridWorld is a simple 5x5 navigation task where the agent starts at (0,0)
and must reach the goal at (4,4). This is a good testbed for tabular methods.

Best algorithms for GridWorld:
1. Q-Learning: Fast convergence, finds optimal policy quickly
2. SARSA(λ): More sample efficient with eligibility traces
3. Double Q-Learning: Reduces overestimation (overkill for simple GridWorld)

This example demonstrates solving GridWorld using multiple algorithms
and comparing their performance.

Run with:
    pixi run mojo run examples/solve_gridworld.mojo
"""

from envs import GridWorldEnv
from agents import QLearningAgent, SARSAAgent, SARSALambdaAgent, DoubleQLearningAgent


fn main() raises:
    print("=" * 60)
    print("    Solving GridWorld - Tabular RL Comparison")
    print("=" * 60)
    print("")
    print("Environment: 5x5 GridWorld")
    print("Start: (0, 0) | Goal: (4, 4)")
    print("Reward: -1 per step, +10 for reaching goal")
    print("Optimal solution: ~8 steps (diagonal path)")
    print("")

    var num_episodes = 500
    var max_steps = 100

    # ========================================================================
    # Algorithm 1: Q-Learning
    # ========================================================================
    print("-" * 60)
    print("Algorithm 1: Q-Learning")
    print("-" * 60)

    var env_q = GridWorldEnv(width=5, height=5)
    var agent_q = QLearningAgent(
        num_states=env_q.num_states(),
        num_actions=env_q.num_actions(),
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    var metrics_q = agent_q.train(
        env_q,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
        print_every=100,
        environment_name="GridWorld",
    )

    var eval_q = agent_q.evaluate(env_q, num_episodes=10)
    print("Q-Learning Evaluation avg reward:", eval_q)
    print("")

    # ========================================================================
    # Algorithm 2: SARSA
    # ========================================================================
    print("-" * 60)
    print("Algorithm 2: SARSA")
    print("-" * 60)

    var env_s = GridWorldEnv(width=5, height=5)
    var agent_s = SARSAAgent(
        num_states=env_s.num_states(),
        num_actions=env_s.num_actions(),
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    var metrics_s = agent_s.train(
        env_s,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
        print_every=100,
        environment_name="GridWorld",
    )

    var eval_s = agent_s.evaluate(env_s, num_episodes=10)
    print("SARSA Evaluation avg reward:", eval_s)
    print("")

    # ========================================================================
    # Algorithm 3: SARSA(λ) - Most sample efficient
    # ========================================================================
    print("-" * 60)
    print("Algorithm 3: SARSA(lambda) - Eligibility Traces")
    print("-" * 60)

    var env_sl = GridWorldEnv(width=5, height=5)
    var agent_sl = SARSALambdaAgent(
        num_states=env_sl.num_states(),
        num_actions=env_sl.num_actions(),
        learning_rate=0.1,
        discount_factor=0.99,
        lambda_=0.9,  # Eligibility trace decay
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    var metrics_sl = agent_sl.train(
        env_sl,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
        print_every=100,
        environment_name="GridWorld",
    )

    var eval_sl = agent_sl.evaluate(env_sl, num_episodes=10)
    print("SARSA(lambda) Evaluation avg reward:", eval_sl)
    print("")

    # ========================================================================
    # Results Summary
    # ========================================================================
    print("=" * 60)
    print("    Results Summary")
    print("=" * 60)
    print("")
    print("Algorithm         | Mean Reward | Max Reward | Eval Reward")
    print("-" * 60)
    print(
        "Q-Learning        |",
        String(metrics_q.mean_reward())[:8],
        "  |",
        String(metrics_q.max_reward())[:8],
        " |",
        String(eval_q)[:8],
    )
    print(
        "SARSA             |",
        String(metrics_s.mean_reward())[:8],
        "  |",
        String(metrics_s.max_reward())[:8],
        " |",
        String(eval_s)[:8],
    )
    print(
        "SARSA(lambda)     |",
        String(metrics_sl.mean_reward())[:8],
        "  |",
        String(metrics_sl.max_reward())[:8],
        " |",
        String(eval_sl)[:8],
    )
    print("")

    # Show learned policy with best agent
    print("-" * 60)
    print("Demonstrating learned policy (Q-Learning):")
    print("-" * 60)
    _ = env_q.reset()
    env_q.render()

    var episode_reward: Float64 = 0.0
    var steps = 0
    for _ in range(20):
        var state = env_q.get_state()
        var state_idx = env_q.state_to_index(state)
        var action_idx = agent_q.get_best_action(state_idx)
        var action = env_q.action_from_index(action_idx)

        var result = env_q.step(action)
        episode_reward += result[1]
        steps += 1

        var action_names = List[String]()
        action_names.append("UP")
        action_names.append("RIGHT")
        action_names.append("DOWN")
        action_names.append("LEFT")

        print("Action:", action_names[action_idx])
        env_q.render()

        if result[2]:
            print("Goal reached in", steps, "steps!")
            print("Total reward:", episode_reward)
            break

    print("")
    print("Optimal path is 8 steps (right 4, up 4) with reward = -8 + 10 = 2")
    print("=" * 60)
