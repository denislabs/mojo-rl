"""Solve CartPole - Function Approximation with Tile Coding.

CartPole is the classic control task where a pole is balanced on a cart.
The agent must keep the pole upright by moving the cart left or right.

State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity] (4D)
Actions: 0 (push left), 1 (push right)

Episode terminates when:
- Pole angle > +/- 12 degrees
- Cart position > +/- 2.4
- Episode length > 500 steps (success!)

This example uses tile coding for function approximation, which provides:
1. Smooth generalization between nearby states
2. Better sample efficiency through feature sharing
3. Multiple tilings for robustness

Best algorithms for CartPole:
1. Tiled Q-Learning: Simple and effective
2. Tiled SARSA(lambda): Faster credit assignment with traces
3. PPO: Policy gradient with clipping (more complex)

Run with:
    pixi run mojo run examples/solve_cartpole.mojo

Requires SDL2 for visualization: brew install sdl2 sdl2_ttf
"""

from mojo_rl.envs import CartPoleEnv
from mojo_rl.agents.tiled_qlearning import (
    TiledQLearningAgent,
    TiledSARSALambdaAgent,
)
from mojo_rl.render import Renderer2D


fn main() raises:
    print("=" * 60)
    print("    Solving CartPole - Tile-Coded Function Approximation")
    print("=" * 60)
    print("")
    print("Environment: CartPole-v1")
    print("State: 4D continuous [x, x_dot, theta, theta_dot]")
    print("Actions: 2 discrete [push left, push right]")
    print("Goal: Keep pole balanced for 500 steps")
    print("Solved: Average reward >= 475 over 100 episodes")
    print("")

    # Create tile coding for CartPole's 4D state space
    var tc = CartPoleEnv[DType.float64].make_tile_coding(
        num_tilings=8, tiles_per_dim=8
    )
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

    var env_q = CartPoleEnv[DType.float64]()
    var agent_q = TiledQLearningAgent(
        tile_coding=tc,
        num_actions=env_q.num_actions(),
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    var metrics_q = agent_q.train(
        env_q,
        tc,
        num_episodes=2000,
        max_steps_per_episode=500,
        verbose=True,
    )

    print("")
    print("Tiled Q-Learning results:")
    print("  Mean reward:", String(metrics_q.mean_reward())[:8])
    print("  Max reward:", String(metrics_q.max_reward())[:8])
    print("")

    # ========================================================================
    # Algorithm 2: Tiled SARSA(lambda)
    # ========================================================================
    print("-" * 60)
    print("Algorithm 2: Tiled SARSA(lambda)")
    print("-" * 60)
    print("SARSA(lambda) with eligibility traces for faster learning.")
    print("")

    # Create fresh tile coding
    var tc_sl = CartPoleEnv[DType.float64].make_tile_coding(
        num_tilings=8, tiles_per_dim=8
    )

    var env_sl = CartPoleEnv[DType.float64]()
    var agent_sl = TiledSARSALambdaAgent(
        tile_coding=tc_sl,
        num_actions=env_sl.num_actions(),
        learning_rate=0.1,
        discount_factor=0.99,
        lambda_=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    var metrics_sl = agent_sl.train(
        env_sl,
        tc_sl,
        num_episodes=2000,
        max_steps_per_episode=500,
        verbose=True,
    )

    print("")
    print("Tiled SARSA(lambda) results:")
    print("  Mean reward:", String(metrics_sl.mean_reward())[:8])
    print("  Max reward:", String(metrics_sl.max_reward())[:8])
    print("")

    # ========================================================================
    # Results Summary
    # ========================================================================
    print("=" * 60)
    print("    Results Summary")
    print("=" * 60)
    print("")
    print("Algorithm          | Mean Reward | Max Reward")
    print("-" * 60)
    print(
        "Tiled Q-Learning   |",
        String(metrics_q.mean_reward())[:8],
        "   |",
        String(metrics_q.max_reward())[:8],
    )
    print(
        "Tiled SARSA(lambda)|",
        String(metrics_sl.mean_reward())[:8],
        "   |",
        String(metrics_sl.max_reward())[:8],
    )
    print("")
    print("Solved: Max reward = 500 (balanced for full episode)")
    print("")

    # ========================================================================
    # Evaluation
    # ========================================================================
    print("-" * 60)
    print("Evaluation (no exploration):")
    print("-" * 60)

    var eval_q = agent_q.evaluate(env_q, tc, num_episodes=100, render=False)
    var eval_sl = agent_sl.evaluate(
        env_sl, tc_sl, num_episodes=100, render=False
    )

    print("Tiled Q-Learning avg reward:", String(eval_q)[:8])
    print("Tiled SARSA(lambda) avg reward:", String(eval_sl)[:8])
    print("")

    if eval_q >= 475 or eval_sl >= 475:
        print("SUCCESS: CartPole solved!")
    else:
        print("Training complete. Consider more episodes for better results.")
    print("")

    # ========================================================================
    # Visual Demo + GIF Recording
    # ========================================================================
    print("-" * 60)
    print("Visual Demo - Recording trained agent to GIF")
    print("-" * 60)
    print("Using Q-Learning agent with SDL2 rendering.")
    print("")

    # Create renderer and start GIF recording
    var renderer = Renderer2D(width=600, height=400, fps=30)
    _ = renderer.init_display()
    renderer.start_recording("cartpole_demo.gif", fps=30)

    # Run 1 episode with recording
    var obs_raw = env_q.reset_obs_list()
    var obs_f64 = List[Float64](capacity=len(obs_raw))
    for i in range(len(obs_raw)):
        obs_f64.append(Float64(obs_raw[i]))

    for step in range(500):
        var tiles = tc.get_tiles(obs_f64)
        var action = agent_q.get_best_action(tiles)
        var result = env_q.step_obs(action)

        env_q.render(renderer)

        if result[2]:  # done
            break

        obs_f64.clear()
        for i in range(len(result[0])):
            obs_f64.append(Float64(result[0][i]))

    # Stop recording and clean up
    renderer.stop_recording()
    renderer.close()
    env_q.close()

    print("Saved: cartpole_demo.gif")
    print("")
    print("Demo complete!")
    print("=" * 60)
