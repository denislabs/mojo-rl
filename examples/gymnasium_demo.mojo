"""Demo script for all Gymnasium environment wrappers.

This demonstrates:
- Classic Control: MountainCar, Pendulum, Acrobot
- Box2D: LunarLander, BipedalWalker (CarRacing requires image processing)
- Toy Text: FrozenLake, Taxi, Blackjack, CliffWalking
- MuJoCo: HalfCheetah (requires mujoco installation)

Run with: mojo run gymnasium_demo.mojo
"""

from python import Python
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
    GymCliffWalkingEnv,
    GymnasiumEnv,
)


fn demo_mountain_car() raises:
    """Demo MountainCar-v0: Drive up a mountain."""
    print("\n" + "=" * 50)
    print("MountainCar-v0 Demo")
    print("=" * 50)
    print("Goal: Drive the underpowered car to reach the flag")
    print("Actions: 0=left, 1=none, 2=right")
    print("")

    var env = GymMountainCarEnv()
    var obs = env.reset()

    print("Initial state: pos=" + String(obs[0]) + ", vel=" + String(obs[1]))

    var total_reward: Float64 = 0.0
    var steps = 0

    # Simple policy: accelerate in direction of velocity
    while not env.done and steps < 200:
        var action: Int
        if obs[1] > 0:
            action = 2  # Push right
        else:
            action = 0  # Push left

        var result = env.step(action)
        obs = result[0]
        var reward = result[1]
        total_reward += reward
        steps += 1

    print("Episode finished in " + String(steps) + " steps")
    print("Total reward: " + String(total_reward))
    print("Final position: " + String(obs[0]) + " (goal: >= 0.5)")
    env.close()


fn demo_pendulum() raises:
    """Demo Pendulum-v1: Swing up and balance."""
    print("\n" + "=" * 50)
    print("Pendulum-v1 Demo")
    print("=" * 50)
    print("Goal: Swing up the pendulum and balance it upright")
    print("Action: Continuous torque in [-2, 2]")
    print("")

    var env = GymPendulumEnv()
    var obs = env.reset()

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
        # Simple control: apply torque proportional to angle
        var angle = obs[2]  # Angular velocity
        var torque = -2.0 * obs[1] - 0.5 * angle  # Simple P controller
        if torque > 2.0:
            torque = 2.0
        elif torque < -2.0:
            torque = -2.0

        var result = env.step(torque)
        obs = result[0]
        var reward = result[1]
        total_reward += reward
        steps += 1

    print("Episode finished in " + String(steps) + " steps")
    print("Total reward: " + String(total_reward))
    env.close()


fn demo_acrobot() raises:
    """Demo Acrobot-v1: Swing up a double pendulum."""
    print("\n" + "=" * 50)
    print("Acrobot-v1 Demo")
    print("=" * 50)
    print("Goal: Swing the tip of the double pendulum above the base")
    print("Actions: 0=-1 torque, 1=0 torque, 2=+1 torque")
    print("")

    var env = GymAcrobotEnv()
    var obs = env.reset()

    print("Observation dim: " + String(env.obs_dim()))

    var total_reward: Float64 = 0.0
    var steps = 0

    # Random policy (Acrobot is hard without learning)
    from random import random_si64

    while not env.done and steps < 500:
        var action = Int(random_si64(0, 2))

        var result = env.step(action)
        obs = result[0]
        var reward = result[1]
        total_reward += reward
        steps += 1

    print("Episode finished in " + String(steps) + " steps")
    print("Total reward: " + String(total_reward))
    env.close()


fn demo_lunar_lander() raises:
    """Demo LunarLander-v3: Land a spacecraft."""
    print("\n" + "=" * 50)
    print("LunarLander-v3 Demo")
    print("=" * 50)
    print("Goal: Land the spacecraft safely on the landing pad")
    print("Actions: 0=nothing, 1=left, 2=main, 3=right")
    print("")

    var env = GymLunarLanderEnv()
    var obs = env.reset()

    print(
        "Observation: [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]"
    )
    print("Initial y-position: " + String(obs[1]))

    var total_reward: Float64 = 0.0
    var steps = 0

    # Simple heuristic policy
    while not env.done and steps < 1000:
        var action: Int = 0
        var x = obs[0]
        var y = obs[1]
        var vx = obs[2]
        var vy = obs[3]
        var angle = obs[4]

        # Fire main engine if falling too fast
        if vy < -0.1:
            action = 2
        # Fire side thrusters to stay centered
        elif x > 0.1 or angle > 0.1:
            action = 1
        elif x < -0.1 or angle < -0.1:
            action = 3

        var result = env.step_discrete(action)
        obs = result[0]
        var reward = result[1]
        total_reward += reward
        steps += 1

    print("Episode finished in " + String(steps) + " steps")
    print("Total reward: " + String(total_reward))
    print("Final y-position: " + String(obs[1]))
    env.close()


fn demo_frozenlake() raises:
    """Demo FrozenLake-v1 (Gymnasium wrapper)."""
    print("\n" + "=" * 50)
    print("FrozenLake-v1 Demo (Gymnasium)")
    print("=" * 50)
    print("Goal: Navigate from S to G without falling in holes")
    print("Actions: 0=left, 1=down, 2=right, 3=up")
    print("")

    var env = GymFrozenLakeEnv(map_name="4x4", is_slippery=False)
    var state = env.reset()

    print("Initial state: " + String(state))
    print("Num states: " + String(env.num_states()))

    var total_reward: Float64 = 0.0
    var steps = 0

    # Fixed sequence to reach goal (without slippery ice)
    var actions = List[Int]()
    actions.append(2)  # right
    actions.append(2)  # right
    actions.append(1)  # down
    actions.append(1)  # down
    actions.append(1)  # down
    actions.append(2)  # right

    var idx = 0
    while not env.done and steps < 20:
        var action: Int
        if idx < len(actions):
            action = actions[idx]
            idx += 1
        else:
            action = 1  # down

        var result = env.step(action)
        state = result[0]
        var reward = result[1]
        total_reward += reward
        steps += 1
        print(
            "Step "
            + String(steps)
            + ": action="
            + String(action)
            + ", state="
            + String(state)
            + ", reward="
            + String(reward)
        )

    print("Episode finished in " + String(steps) + " steps")
    print("Total reward: " + String(total_reward))
    env.close()


fn demo_taxi() raises:
    """Demo Taxi-v3 (Gymnasium wrapper)."""
    print("\n" + "=" * 50)
    print("Taxi-v3 Demo (Gymnasium)")
    print("=" * 50)
    print("Goal: Pick up passenger and drop off at destination")
    print("Actions: 0=south, 1=north, 2=east, 3=west, 4=pickup, 5=dropoff")
    print("")

    var env = GymTaxiEnv()
    var state = env.reset()

    print("Initial state index: " + String(state))
    print("Num states: " + String(env.num_states()))

    var total_reward: Float64 = 0.0
    var steps = 0

    # Random exploration
    from random import random_si64

    while not env.done and steps < 50:
        var action = Int(random_si64(0, 5))

        var result = env.step(action)
        state = result[0]
        var reward = result[1]
        total_reward += reward
        steps += 1

    print("Episode finished in " + String(steps) + " steps")
    print("Total reward: " + String(total_reward))
    env.close()


fn demo_blackjack() raises:
    """Demo Blackjack-v1."""
    print("\n" + "=" * 50)
    print("Blackjack-v1 Demo")
    print("=" * 50)
    print("Goal: Beat the dealer without going over 21")
    print("Actions: 0=stick, 1=hit")
    print("")

    var env = GymBlackjackEnv()
    var obs = env.reset()
    var player_sum = obs[0]
    var dealer_card = obs[1]
    var usable_ace = obs[2]

    print("Player sum: " + String(player_sum))
    print("Dealer showing: " + String(dealer_card))
    print("Usable ace: " + String(usable_ace))

    var total_reward: Float64 = 0.0

    # Simple strategy: hit until 17
    while not env.done:
        var action: Int
        if player_sum < 17:
            action = 1  # hit
            print("Action: Hit")
        else:
            action = 0  # stick
            print("Action: Stick")

        var result = env.step(action)
        player_sum = result[0]
        dealer_card = result[1]
        usable_ace = result[2]
        var reward = result[3]
        total_reward += reward

        if not env.done:
            print("Player sum: " + String(player_sum))

    print("Game over! Reward: " + String(total_reward))
    if total_reward > 0:
        print("You won!")
    elif total_reward < 0:
        print("You lost.")
    else:
        print("It's a draw.")
    env.close()


fn demo_cliffwalking() raises:
    """Demo CliffWalking-v0 (Gymnasium wrapper)."""
    print("\n" + "=" * 50)
    print("CliffWalking-v0 Demo (Gymnasium)")
    print("=" * 50)
    print("Goal: Navigate to goal without falling off cliff")
    print("Actions: 0=up, 1=right, 2=down, 3=left")
    print("")

    var env = GymCliffWalkingEnv()
    var state = env.reset()

    print("Initial state: " + String(state))
    print("Num states: " + String(env.num_states()))

    var total_reward: Float64 = 0.0
    var steps = 0

    # Safe path: go up, then right, then down
    var actions = List[Int]()
    # Go up 3 times
    for _ in range(3):
        actions.append(0)
    # Go right 11 times
    for _ in range(11):
        actions.append(1)
    # Go down 3 times
    for _ in range(3):
        actions.append(2)

    for i in range(len(actions)):
        if env.done:
            break
        var action = actions[i]
        var result = env.step(action)
        state = result[0]
        var reward = result[1]
        total_reward += reward
        steps += 1

    print("Episode finished in " + String(steps) + " steps")
    print("Total reward: " + String(total_reward))
    env.close()


fn demo_generic_wrapper() raises:
    """Demo the generic GymnasiumEnv wrapper."""
    print("\n" + "=" * 50)
    print("Generic Gymnasium Wrapper Demo")
    print("=" * 50)
    print("Using CartPole-v1 via generic wrapper")
    print("")

    var env = GymnasiumEnv("CartPole-v1")
    print(env.get_info())

    var obs = env.reset()
    var obs_list = env.get_obs_as_list(obs)

    print(
        "Initial observation: "
        + String(obs_list[0])
        + ", "
        + String(obs_list[1])
        + ", "
        + String(obs_list[2])
        + ", "
        + String(obs_list[3])
    )

    var steps = 0
    while not env.done and steps < 100:
        var result = env.step_discrete(0 if obs_list[2] < 0 else 1)
        obs = result[0]
        obs_list = env.get_obs_as_list(obs)
        steps += 1

    print("Episode finished in " + String(steps) + " steps")
    print("Total reward: " + String(env.episode_reward))
    env.close()


fn main() raises:
    print("=" * 50)
    print("Gymnasium Environment Wrappers Demo")
    print("=" * 50)
    print("")
    print("This demo shows all available Gymnasium wrappers.")
    print("Note: Box2D and MuJoCo envs require additional packages:")
    print("  - Box2D: pip install gymnasium[box2d]")
    print("  - MuJoCo: pip install gymnasium[mujoco]")
    print("")

    # Classic Control
    demo_mountain_car()
    demo_pendulum()
    demo_acrobot()

    # Toy Text
    demo_frozenlake()
    demo_taxi()
    demo_blackjack()
    demo_cliffwalking()

    # Generic wrapper
    demo_generic_wrapper()

    # Box2D (may fail if box2d not installed)
    print("\n" + "=" * 50)
    print("Box2D Environments (requires gymnasium[box2d])")
    print("=" * 50)
    try:
        demo_lunar_lander()
    except e:
        print("LunarLander not available: " + String(e))

    print("\n" + "=" * 50)
    print("All demos completed!")
    print("=" * 50)
