"""Solve Acrobot - Tabular RL with state discretization.

Acrobot is a two-link pendulum where only the second joint is actuated.
The goal is to swing the free end above a target height by applying
torque at the middle joint.

State: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot]
Actions: 0 (-1 torque), 1 (0 torque), 2 (+1 torque)

Challenges:
1. Continuous 6D state space - requires discretization or function approximation
2. Sparse reward: -1 per step until goal
3. Requires coordinated swinging motions

This example uses the discretized state representation (built into the
AcrobotEnv) with tabular Q-learning. While this works for moderate
discretization levels, function approximation (tile coding) would be
more efficient for finer resolution.

Run with:
    pixi run mojo run examples/solve_acrobot.mojo

Requires SDL2 for visualization: brew install sdl2 sdl2_ttf
"""

from envs import AcrobotEnv
from agents import QLearningAgent, SARSALambdaAgent


fn main() raises:
    print("=" * 60)
    print("    Solving Acrobot - Tabular RL with Discretization")
    print("=" * 60)
    print("")
    print("Environment: Acrobot (two-link pendulum)")
    print("State: 6D continuous, discretized into bins")
    print("Actions: 3 discrete [-1, 0, +1] torque")
    print("Goal: Swing free end above target height")
    print("Reward: -1 per step (minimize steps to reach goal)")
    print("")

    # Use 6 bins per dimension -> 6^6 = 46,656 states
    var num_bins = 6
    var num_states = AcrobotEnv.get_num_states(num_bins)
    print("Discretization: ", num_bins, "bins per dimension")
    print("Total discrete states:", num_states)
    print("")

    # Training parameters
    var num_episodes = 500
    var max_steps = 500

    # ========================================================================
    # Algorithm 1: Q-Learning with discretized states
    # ========================================================================
    print("-" * 60)
    print("Algorithm 1: Q-Learning (Discretized)")
    print("-" * 60)
    print("Using tabular Q-learning with", num_states, "states")
    print("")

    var env_q = AcrobotEnv(num_bins=num_bins)
    var agent_q = QLearningAgent(
        num_states=num_states,
        num_actions=env_q.num_actions(),
        learning_rate=0.1,
        discount_factor=1.0,  # Undiscounted for shortest path
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
    )

    var metrics_q = agent_q.train(
        env_q,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
        print_every=100,
        environment_name="Acrobot",
    )

    var eval_q = agent_q.evaluate(env_q, num_episodes=10)
    print("")
    print("Q-Learning evaluation avg reward:", String(eval_q)[:8])
    print("(Lower is better - fewer steps to goal)")
    print("")

    # ========================================================================
    # Algorithm 2: SARSA(lambda) with discretized states
    # ========================================================================
    print("-" * 60)
    print("Algorithm 2: SARSA(lambda) (Discretized)")
    print("-" * 60)
    print("SARSA(lambda) with eligibility traces for faster credit assignment.")
    print("")

    var env_sl = AcrobotEnv(num_bins=num_bins)
    var agent_sl = SARSALambdaAgent(
        num_states=num_states,
        num_actions=env_sl.num_actions(),
        learning_rate=0.1,
        discount_factor=1.0,
        lambda_=0.9,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
    )

    var metrics_sl = agent_sl.train(
        env_sl,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
        print_every=100,
        environment_name="Acrobot",
    )

    var eval_sl = agent_sl.evaluate(env_sl, num_episodes=10)
    print("")
    print("SARSA(lambda) evaluation avg reward:", String(eval_sl)[:8])
    print("")

    # ========================================================================
    # Results Summary
    # ========================================================================
    print("=" * 60)
    print("    Results Summary")
    print("=" * 60)
    print("")
    print("Algorithm       | Mean Reward | Max Reward | Eval Reward")
    print("-" * 60)

    # Max reward is negative, so best steps = -max_reward
    var best_steps_q = -metrics_q.max_reward()
    var best_steps_sl = -metrics_sl.max_reward()

    print(
        "Q-Learning      |",
        String(metrics_q.mean_reward())[:8],
        "   |",
        String(metrics_q.max_reward())[:8],
        "  |",
        String(eval_q)[:8],
    )
    print(
        "SARSA(lambda)   |",
        String(metrics_sl.mean_reward())[:8],
        "   |",
        String(metrics_sl.max_reward())[:8],
        "  |",
        String(eval_sl)[:8],
    )
    print("")
    print("Best Q-Learning episode:", Int(best_steps_q), "steps")
    print("Best SARSA(lambda) episode:", Int(best_steps_sl), "steps")
    print("")
    print("Good performance: ~100-200 steps to reach goal")
    print("With coarse discretization, ~200-300 steps is acceptable")
    print("")

    # ========================================================================
    # Visual Demo
    # ========================================================================
    print("-" * 60)
    print("Visual Demo - Watch the trained agent!")
    print("-" * 60)
    print("Using Q-Learning agent with SDL2 rendering.")
    print("Close the window when done watching.")
    print("")

    # Run visual demo with the trained agent
    for episode in range(3):
        _ = env_q.reset()
        var total_reward: Float64 = 0.0
        var steps = 0

        print("Demo episode", episode + 1)

        for step in range(max_steps):
            var state = env_q.get_state()
            var state_idx = env_q.state_to_index(state)
            var action_idx = agent_q.get_best_action(state_idx)
            var action = env_q.action_from_index(action_idx)

            var result = env_q.step(action)
            total_reward += result[1]
            steps += 1

            # Render every frame
            env_q.render()

            if result[2]:
                if total_reward > -max_steps:
                    print("  Goal reached in", steps, "steps!")
                else:
                    print("  Timeout after", steps, "steps")
                break

    env_q.close()

    print("")
    print("Demo complete!")
    print("")
    print("Note: For better performance on Acrobot, consider using")
    print("tile coding (function approximation) which provides smoother")
    print("generalization across the continuous state space.")
    print("=" * 60)
