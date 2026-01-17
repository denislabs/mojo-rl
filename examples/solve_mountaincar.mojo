"""Solve MountainCar - Function Approximation with Tile Coding.

MountainCar is a challenging sparse reward problem where a car must build
momentum by swinging back and forth to reach the top of a hill.

State: [position, velocity] (2D continuous)
  - Position: [-1.2, 0.6]
  - Velocity: [-0.07, 0.07]
Actions: 0 (push left), 1 (no push), 2 (push right)

The car starts at the bottom of a valley and cannot directly climb
to the goal. It must learn to build momentum by going back and forth.

Reward: -1 per step (minimize steps to reach goal)
Goal: Reach position >= 0.5 (flag on the right hill)

Key challenges:
1. Sparse reward: Only -1 per step until goal
2. Counter-intuitive solution: Must go backwards first
3. Continuous state space: Requires function approximation

Best algorithms for MountainCar:
1. Tiled Q-Learning: Works well with high learning rate
2. Tiled SARSA(lambda): Faster credit assignment

Run with:
    pixi run mojo run examples/solve_mountaincar.mojo

Requires SDL2 for visualization: brew install sdl2 sdl2_ttf
"""

from envs import MountainCarEnv
from agents.tiled_qlearning import TiledQLearningAgent, TiledSARSALambdaAgent


fn main() raises:
    print("=" * 60)
    print("    Solving MountainCar - Tile-Coded Function Approximation")
    print("=" * 60)
    print("")
    print("Environment: MountainCar-v0")
    print("State: 2D continuous [position, velocity]")
    print("Actions: 3 discrete [push left, no push, push right]")
    print("Goal: Reach position >= 0.5 (flag on right hill)")
    print("Reward: -1 per step (minimize steps)")
    print("Best possible: ~100 steps")
    print("")

    # Create tile coding for MountainCar's 2D state space
    var tc = MountainCarEnv.make_tile_coding(num_tilings=8, tiles_per_dim=8)
    print("Tile coding configuration:")
    print("  Tilings:", tc.get_num_tilings())
    print("  Tiles per dim: 8")
    print("  Total tiles:", tc.get_num_tiles())
    print("")

    # ========================================================================
    # Algorithm 1: Tiled Q-Learning
    # ========================================================================
    print("-" * 60)
    print("Algorithm 1: Tiled Q-Learning")
    print("-" * 60)

    var env_q = MountainCarEnv[DType.float64]()
    var agent_q = TiledQLearningAgent(
        tile_coding=tc,
        num_actions=env_q.num_actions(),
        learning_rate=0.5,  # High learning rate works well
        discount_factor=1.0,  # Undiscounted for shortest path
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
    )

    var metrics_q = agent_q.train(
        env_q,
        tc,
        num_episodes=3000,
        max_steps_per_episode=200,
        verbose=True,
    )

    print("")
    print("Tiled Q-Learning results:")
    print("  Mean reward:", String(metrics_q.mean_reward())[:8])
    print("  Max reward:", String(metrics_q.max_reward())[:8])
    print("  Best steps:", Int(-metrics_q.max_reward()))
    print("")

    # ========================================================================
    # Algorithm 2: Tiled SARSA(lambda)
    # ========================================================================
    print("-" * 60)
    print("Algorithm 2: Tiled SARSA(lambda)")
    print("-" * 60)
    print("SARSA(lambda) with eligibility traces.")
    print("")

    # Create fresh tile coding
    var tc_sl = MountainCarEnv.make_tile_coding(num_tilings=8, tiles_per_dim=8)

    var env_sl = MountainCarEnv[DType.float64]()
    var agent_sl = TiledSARSALambdaAgent(
        tile_coding=tc_sl,
        num_actions=env_sl.num_actions(),
        learning_rate=0.5,
        discount_factor=1.0,
        lambda_=0.9,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
    )

    var metrics_sl = agent_sl.train(
        env_sl,
        tc_sl,
        num_episodes=3000,
        max_steps_per_episode=200,
        verbose=True,
    )

    print("")
    print("Tiled SARSA(lambda) results:")
    print("  Mean reward:", String(metrics_sl.mean_reward())[:8])
    print("  Max reward:", String(metrics_sl.max_reward())[:8])
    print("  Best steps:", Int(-metrics_sl.max_reward()))
    print("")

    # ========================================================================
    # Results Summary
    # ========================================================================
    print("=" * 60)
    print("    Results Summary")
    print("=" * 60)
    print("")
    print("Algorithm          | Mean Reward | Best Steps")
    print("-" * 60)
    print(
        "Tiled Q-Learning   |",
        String(metrics_q.mean_reward())[:8],
        "   |",
        Int(-metrics_q.max_reward()),
    )
    print(
        "Tiled SARSA(lambda)|",
        String(metrics_sl.mean_reward())[:8],
        "   |",
        Int(-metrics_sl.max_reward()),
    )
    print("")
    print("Good: ~110-130 steps | Excellent: <100 steps")
    print("")

    # ========================================================================
    # Evaluation
    # ========================================================================
    print("-" * 60)
    print("Evaluation (no exploration):")
    print("-" * 60)

    var eval_q = agent_q.evaluate(env_q, tc, num_episodes=20, render=False)
    var eval_sl = agent_sl.evaluate(env_sl, tc_sl, num_episodes=20, render=False)

    print("Tiled Q-Learning avg steps:", Int(-eval_q))
    print("Tiled SARSA(lambda) avg steps:", Int(-eval_sl))
    print("")

    # ========================================================================
    # Visual Demo
    # ========================================================================
    print("-" * 60)
    print("Visual Demo - Watch the trained agent!")
    print("-" * 60)
    print("Using Q-Learning agent with SDL2 rendering.")
    print("Watch the car swing back and forth to build momentum.")
    print("Close the window when done watching.")
    print("")

    # Run visual demo
    _ = agent_q.evaluate(env_q, tc, num_episodes=3, render=True)

    env_q.close()

    print("")
    print("Demo complete!")
    print("=" * 60)
