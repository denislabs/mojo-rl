"""Demo script for Gymnasium environment wrappers with built-in training agents.

This demonstrates training RL agents on Gymnasium environments:
- Classic Control: MountainCar, Acrobot (using TiledQLearningAgent)
- Box2D: LunarLander (using TiledQLearningAgent)
- Toy Text: FrozenLake, Taxi, CliffWalking (using QLearningAgent)

Continuous action environments (Pendulum, BipedalWalker, MuJoCo) are shown
with simple heuristic policies since they require policy gradient methods.

Run with: pixi run mojo run examples/gymnasium_demo.mojo
"""

from python import Python
from core.tile_coding import TileCoding
from agents.tiled_qlearning import TiledQLearningAgent
from agents.qlearning import QLearningAgent
from envs.gymnasium import (
    GymMountainCarEnv,
    GymPendulumEnv,
    GymAcrobotEnv,
    GymLunarLanderEnv,
    GymBipedalWalkerEnv,
    GymCarRacingEnv,
    GymFrozenLakeEnv,
    GymTaxiEnv,
    GymBlackjackEnv,
    GymBlackjackAction,
    GymCliffWalkingEnv,
    GymnasiumEnv,
)


fn train_mountain_car() raises:
    """Train Q-learning agent on MountainCar-v0."""
    print("\n" + "=" * 60)
    print("MountainCar-v0 Training (TiledQLearning)")
    print("=" * 60)
    print("Goal: Drive the underpowered car to reach the flag")
    print("Actions: 0=left, 1=none, 2=right")
    print("")

    # Create environment
    var env = GymMountainCarEnv()

    # Create tile coding for 2D continuous state
    var tc = GymMountainCarEnv.make_tile_coding(
        num_tilings=8,
        tiles_per_dim=8,
    )

    # Create tiled Q-learning agent
    var agent = TiledQLearningAgent(
        tile_coding=tc,
        num_actions=3,  # left, none, right
        learning_rate=0.2,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    # Train the agent
    var metrics = agent.train(
        env,
        tc,
        num_episodes=500,
        max_steps_per_episode=200,
        verbose=True,
        print_every=100,
        environment_name="MountainCar-v0",
    )

    print("")
    print("Training Complete!")
    print("Mean reward:", metrics.mean_reward())
    print("Max reward:", metrics.max_reward())

    # Evaluate
    var eval_reward = agent.evaluate(env, tc, num_episodes=5)
    print("Evaluation avg reward:", Int(eval_reward))

    if eval_reward > -200:
        print("SUCCESS: MountainCar solved!")
    else:
        print("Training complete. MountainCar is challenging - consider more episodes.")

    env.close()


fn train_acrobot() raises:
    """Train Q-learning agent on Acrobot-v1."""
    print("\n" + "=" * 60)
    print("Acrobot-v1 Training (TiledQLearning)")
    print("=" * 60)
    print("Goal: Swing the tip of the double pendulum above the base")
    print("Actions: 0=-1 torque, 1=0 torque, 2=+1 torque")
    print("")

    # Create environment
    var env = GymAcrobotEnv()

    # Create tile coding for 4D continuous state (first 4 dims of 6D state)
    # Acrobot state: (cos1, sin1, cos2, sin2, vel1, vel2), using first 4
    var state_low = List[Float64]()
    state_low.append(-1.0)  # cos1
    state_low.append(-1.0)  # sin1
    state_low.append(-1.0)  # cos2
    state_low.append(-1.0)  # sin2

    var state_high = List[Float64]()
    state_high.append(1.0)
    state_high.append(1.0)
    state_high.append(1.0)
    state_high.append(1.0)

    var tiles_per_dim = List[Int]()
    tiles_per_dim.append(6)
    tiles_per_dim.append(6)
    tiles_per_dim.append(6)
    tiles_per_dim.append(6)

    var tc = TileCoding(
        num_tilings=8,
        tiles_per_dim=tiles_per_dim^,
        state_low=state_low^,
        state_high=state_high^,
    )

    # Create tiled Q-learning agent
    var agent = TiledQLearningAgent(
        tile_coding=tc,
        num_actions=3,
        learning_rate=0.2,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    # Train the agent
    var metrics = agent.train(
        env,
        tc,
        num_episodes=500,
        max_steps_per_episode=500,
        verbose=True,
        print_every=100,
        environment_name="Acrobot-v1",
    )

    print("")
    print("Training Complete!")
    print("Mean reward:", metrics.mean_reward())
    print("Max reward:", metrics.max_reward())

    # Evaluate
    var eval_reward = agent.evaluate(env, tc, num_episodes=5)
    print("Evaluation avg reward:", Int(eval_reward))

    env.close()


fn demo_pendulum() raises:
    """Demo Pendulum-v1 with heuristic policy (continuous action space)."""
    print("\n" + "=" * 60)
    print("Pendulum-v1 Demo (Heuristic Controller)")
    print("=" * 60)
    print("Goal: Swing up the pendulum and balance it upright")
    print("Action: Continuous torque in [-2, 2]")
    print("Note: Requires policy gradient methods for proper training")
    print("")

    var env = GymPendulumEnv()
    _ = env.reset()
    var obs = env.get_obs()  # SIMD[DType.float64, 4]

    print(
        "Initial state: cos="
        + String(obs[0])
        + ", sin="
        + String(obs[1])
        + ", vel="
        + String(obs[2])
    )

    var total_reward: Float64 = 0.0
    var steps = 0

    # Simple proportional controller
    while not env.done and steps < 200:
        var angle = obs[2]  # Angular velocity
        var torque = -2.0 * obs[1] - 0.5 * angle  # Simple P controller
        if torque > 2.0:
            torque = 2.0
        elif torque < -2.0:
            torque = -2.0

        var result = env.step_continuous(torque)
        obs = result[0]
        var reward = result[1]
        total_reward += reward
        steps += 1

    print("Episode finished in " + String(steps) + " steps")
    print("Total reward: " + String(total_reward))
    env.close()


fn train_lunar_lander() raises:
    """Train Q-learning agent on LunarLander-v3."""
    print("\n" + "=" * 60)
    print("LunarLander-v3 Training (TiledQLearning)")
    print("=" * 60)
    print("Goal: Land the spacecraft safely on the landing pad")
    print("Actions: 0=nothing, 1=left, 2=main, 3=right")
    print("")

    # Create environment
    var env = GymLunarLanderEnv()

    # Create tile coding for 8D continuous state
    # Using 4D subset: [x, y, vx, vy] for trait conformance
    var state_low = List[Float64]()
    state_low.append(-1.5)  # x
    state_low.append(-1.5)  # y
    state_low.append(-5.0)  # vx
    state_low.append(-5.0)  # vy

    var state_high = List[Float64]()
    state_high.append(1.5)
    state_high.append(1.5)
    state_high.append(5.0)
    state_high.append(5.0)

    var tiles_per_dim = List[Int]()
    tiles_per_dim.append(8)
    tiles_per_dim.append(8)
    tiles_per_dim.append(8)
    tiles_per_dim.append(8)

    var tc = TileCoding(
        num_tilings=8,
        tiles_per_dim=tiles_per_dim^,
        state_low=state_low^,
        state_high=state_high^,
    )

    # Create tiled Q-learning agent
    var agent = TiledQLearningAgent(
        tile_coding=tc,
        num_actions=4,
        learning_rate=0.15,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    # Train the agent
    var metrics = agent.train(
        env,
        tc,
        num_episodes=500,
        max_steps_per_episode=1000,
        verbose=True,
        print_every=100,
        environment_name="LunarLander-v3",
    )

    print("")
    print("Training Complete!")
    print("Mean reward:", metrics.mean_reward())
    print("Max reward:", metrics.max_reward())

    # Evaluate
    var eval_reward = agent.evaluate(env, tc, num_episodes=5)
    print("Evaluation avg reward:", Int(eval_reward))

    if eval_reward >= 200:
        print("SUCCESS: LunarLander solved!")
    else:
        print("Training complete. Consider more episodes for better performance.")

    env.close()


fn train_frozenlake() raises:
    """Train Q-learning agent on FrozenLake-v1."""
    print("\n" + "=" * 60)
    print("FrozenLake-v1 Training (Q-Learning)")
    print("=" * 60)
    print("Goal: Navigate from S to G without falling in holes")
    print("Actions: 0=left, 1=down, 2=right, 3=up")
    print("")

    # Create environment (non-slippery for easier learning)
    var env = GymFrozenLakeEnv(map_name="4x4", is_slippery=False)

    print("Num states: " + String(env.num_states()))
    print("Num actions: " + String(env.num_actions()))

    # Create Q-learning agent
    var agent = QLearningAgent(
        num_states=env.num_states(),
        num_actions=env.num_actions(),
        learning_rate=0.8,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
    )

    # Train the agent
    var metrics = agent.train(
        env,
        num_episodes=1000,
        max_steps_per_episode=100,
        verbose=True,
        print_every=200,
        environment_name="FrozenLake-v1",
    )

    print("")
    print("Training Complete!")
    print("Mean reward:", metrics.mean_reward())
    print("Max reward:", metrics.max_reward())

    # Evaluate
    var eval_reward = agent.evaluate(env, num_episodes=10)
    print("Evaluation avg reward:", eval_reward)

    if eval_reward >= 0.8:
        print("SUCCESS: FrozenLake solved!")
    else:
        print("Training complete.")

    env.close()


fn train_taxi() raises:
    """Train Q-learning agent on Taxi-v3."""
    print("\n" + "=" * 60)
    print("Taxi-v3 Training (Q-Learning)")
    print("=" * 60)
    print("Goal: Pick up passenger and drop off at destination")
    print("Actions: 0=south, 1=north, 2=east, 3=west, 4=pickup, 5=dropoff")
    print("")

    # Create environment
    var env = GymTaxiEnv()

    print("Num states: " + String(env.num_states()))
    print("Num actions: " + String(env.num_actions()))

    # Create Q-learning agent
    var agent = QLearningAgent(
        num_states=env.num_states(),
        num_actions=env.num_actions(),
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    # Train the agent
    var metrics = agent.train(
        env,
        num_episodes=2000,
        max_steps_per_episode=200,
        verbose=True,
        print_every=400,
        environment_name="Taxi-v3",
    )

    print("")
    print("Training Complete!")
    print("Mean reward:", metrics.mean_reward())
    print("Max reward:", metrics.max_reward())

    # Evaluate
    var eval_reward = agent.evaluate(env, num_episodes=10)
    print("Evaluation avg reward:", eval_reward)

    if eval_reward >= 8:
        print("SUCCESS: Taxi solved!")
    else:
        print("Training complete.")

    env.close()


fn demo_blackjack() raises:
    """Demo Blackjack-v1 with simple strategy."""
    print("\n" + "=" * 60)
    print("Blackjack-v1 Demo (Simple Strategy)")
    print("=" * 60)
    print("Goal: Beat the dealer without going over 21")
    print("Actions: 0=stick, 1=hit")
    print("Note: Has special 3D discrete state - using heuristic policy")
    print("")

    var env = GymBlackjackEnv()
    var wins = 0
    var losses = 0
    var draws = 0

    for game in range(10):
        _ = env.reset()
        var player_sum = env.player_sum
        var total_reward: Float64 = 0.0

        # Simple strategy: hit until 17
        while not env.done:
            var action: GymBlackjackAction
            if player_sum < 17:
                action = GymBlackjackAction(action=1)  # hit
            else:
                action = GymBlackjackAction(action=0)  # stick

            var result = env.step(action)
            player_sum = env.player_sum
            var reward = result[1]
            total_reward += reward

        if total_reward > 0:
            wins += 1
        elif total_reward < 0:
            losses += 1
        else:
            draws += 1

    print("Results over 10 games:")
    print("  Wins: " + String(wins))
    print("  Losses: " + String(losses))
    print("  Draws: " + String(draws))
    env.close()


fn train_cliffwalking() raises:
    """Train Q-learning agent on CliffWalking-v0."""
    print("\n" + "=" * 60)
    print("CliffWalking-v0 Training (Q-Learning)")
    print("=" * 60)
    print("Goal: Navigate to goal without falling off cliff")
    print("Actions: 0=up, 1=right, 2=down, 3=left")
    print("")

    # Create environment
    var env = GymCliffWalkingEnv()

    print("Num states: " + String(env.num_states()))
    print("Num actions: " + String(env.num_actions()))

    # Create Q-learning agent
    var agent = QLearningAgent(
        num_states=env.num_states(),
        num_actions=env.num_actions(),
        learning_rate=0.5,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
    )

    # Train the agent
    var metrics = agent.train(
        env,
        num_episodes=500,
        max_steps_per_episode=100,
        verbose=True,
        print_every=100,
        environment_name="CliffWalking-v0",
    )

    print("")
    print("Training Complete!")
    print("Mean reward:", metrics.mean_reward())
    print("Max reward:", metrics.max_reward())

    # Evaluate
    var eval_reward = agent.evaluate(env, num_episodes=10)
    print("Evaluation avg reward:", eval_reward)

    # Optimal path gives -13 reward (13 steps)
    if eval_reward >= -15:
        print("SUCCESS: CliffWalking near-optimal!")
    else:
        print("Training complete.")

    env.close()


fn main() raises:
    print("=" * 60)
    print("Gymnasium Environment Training Demo")
    print("=" * 60)
    print("")
    print("This demo trains RL agents on various Gymnasium environments.")
    print("Note: Box2D envs require: pip install gymnasium[box2d]")
    print("")

    # Classic Control - Continuous state spaces
    print("\n" + "=" * 60)
    print("CLASSIC CONTROL (TiledQLearning)")
    print("=" * 60)
    train_mountain_car()
    train_acrobot()
    demo_pendulum()  # Continuous action - heuristic only

    # Toy Text - Discrete state spaces
    print("\n" + "=" * 60)
    print("TOY TEXT (Tabular Q-Learning)")
    print("=" * 60)
    train_frozenlake()
    train_taxi()
    demo_blackjack()  # Special state structure - heuristic
    train_cliffwalking()

    # Box2D - Requires additional installation
    print("\n" + "=" * 60)
    print("BOX2D (requires gymnasium[box2d])")
    print("=" * 60)
    try:
        train_lunar_lander()
    except e:
        print("LunarLander not available: " + String(e))
        print("Install with: pip install gymnasium[box2d]")

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
