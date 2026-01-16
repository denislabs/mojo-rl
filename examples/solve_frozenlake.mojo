"""Solve FrozenLake - Handling stochastic environments.

FrozenLake is a 4x4 grid where the agent must navigate from start (S) to goal (G)
while avoiding holes (H). The ice is SLIPPERY - the agent only moves in the
intended direction 1/3 of the time, making this a challenging stochastic problem.

Grid layout:
    S F F F     S = Start (position 0)
    F H F H     F = Frozen (safe)
    F F F H     H = Hole (terminal, reward=0)
    H F F G     G = Goal (terminal, reward=1)

Key challenges:
1. Sparse reward: Only +1 for reaching goal, 0 otherwise
2. Stochastic transitions: Agent slips 2/3 of the time
3. Irreversible failures: Falling in a hole ends the episode

Best algorithms for FrozenLake:
1. Q-Learning with slow epsilon decay (needs extensive exploration)
2. Double Q-Learning (reduces overestimation in noisy environment)

This example also demonstrates non-slippery mode for comparison.

Run with:
    pixi run mojo run examples/solve_frozenlake.mojo
"""

from envs import FrozenLakeEnv
from agents import QLearningAgent, DoubleQLearningAgent
from render import RendererBase


fn train_and_evaluate(
    mut env: FrozenLakeEnv,
    mut agent: QLearningAgent,
    num_episodes: Int,
    max_steps: Int,
    algorithm_name: String,
) raises -> Tuple[Float64, Float64]:
    """Train agent and return (training mean reward, evaluation success rate)."""
    var metrics = agent.train(
        env,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
        print_every=1000,
        environment_name="FrozenLake",
    )

    # Evaluate with multiple episodes to get success rate
    var successes = 0
    var eval_episodes = 100
    for _ in range(eval_episodes):
        var state = env.reset()
        for _ in range(max_steps):
            var state_idx = env.state_to_index(state)
            var action_idx = agent.get_best_action(state_idx)
            var action = env.action_from_index(action_idx)
            var result = env.step(action)
            state = result[0]
            if result[2]:  # done
                if result[1] > 0.5:  # reached goal (reward = 1)
                    successes += 1
                break

    var success_rate = Float64(successes) / Float64(eval_episodes) * 100.0
    return (metrics.mean_reward(), success_rate)


fn main() raises:
    print("=" * 60)
    print("    Solving FrozenLake - Stochastic Environment")
    print("=" * 60)
    print("")
    print("Environment: 4x4 FrozenLake")
    print("Start: top-left | Goal: bottom-right")
    print("Reward: +1 for goal, 0 otherwise (sparse!)")
    print("")
    print("Challenge: Ice is SLIPPERY - agent moves in intended")
    print("direction only 1/3 of the time!")
    print("")

    # Parameters tuned for slippery environment
    var num_episodes = 10000  # Need many episodes for sparse rewards
    var max_steps = 100

    # ========================================================================
    # Part 1: Non-slippery FrozenLake (deterministic baseline)
    # ========================================================================
    print("-" * 60)
    print("Part 1: Non-Slippery FrozenLake (deterministic)")
    print("-" * 60)
    print("This is the easy version - agent always moves as intended.")
    print("")

    var env_easy = FrozenLakeEnv(size=4, is_slippery=False)
    var agent_easy = QLearningAgent(
        num_states=env_easy.num_states(),
        num_actions=env_easy.num_actions(),
        learning_rate=0.8,  # Higher learning rate for sparse rewards
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,  # Slower decay to explore more
        epsilon_min=0.01,
    )

    var result_easy = train_and_evaluate(
        env_easy, agent_easy, 2000, max_steps, "Q-Learning (non-slippery)"
    )
    print("")
    print("Non-slippery success rate:", String(result_easy[1])[:5], "%")
    print("")

    # ========================================================================
    # Part 2: Slippery FrozenLake with Q-Learning
    # ========================================================================
    print("-" * 60)
    print("Part 2: Slippery FrozenLake with Q-Learning")
    print("-" * 60)
    print("The hard version - agent slips 2/3 of the time!")
    print("Training for", num_episodes, "episodes...")
    print("")

    var env_q = FrozenLakeEnv(size=4, is_slippery=True)
    var agent_q = QLearningAgent(
        num_states=env_q.num_states(),
        num_actions=env_q.num_actions(),
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,  # Slow decay for exploration
        epsilon_min=0.05,  # Keep some exploration
    )

    var result_q = train_and_evaluate(
        env_q, agent_q, num_episodes, max_steps, "Q-Learning (slippery)"
    )
    print("")
    print("Q-Learning (slippery) success rate:", String(result_q[1])[:5], "%")
    print("")

    # ========================================================================
    # Part 3: Slippery FrozenLake with Double Q-Learning
    # ========================================================================
    print("-" * 60)
    print("Part 3: Slippery FrozenLake with Double Q-Learning")
    print("-" * 60)
    print("Double Q-Learning reduces overestimation in noisy environments.")
    print("Training for", num_episodes, "episodes...")
    print("")

    var env_dq = FrozenLakeEnv(size=4, is_slippery=True)
    var agent_dq = DoubleQLearningAgent(
        num_states=env_dq.num_states(),
        num_actions=env_dq.num_actions(),
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.05,
    )

    var metrics_dq = agent_dq.train(
        env_dq,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
        print_every=1000,
        environment_name="FrozenLake",
    )

    # Evaluate Double Q-Learning
    var successes_dq = 0
    var eval_episodes = 100
    for _ in range(eval_episodes):
        var state = env_dq.reset()
        for _ in range(max_steps):
            var state_idx = env_dq.state_to_index(state)
            var action_idx = agent_dq.get_best_action(state_idx)
            var action = env_dq.action_from_index(action_idx)
            var result = env_dq.step(action)
            state = result[0]
            if result[2]:
                if result[1] > 0.5:
                    successes_dq += 1
                break

    var success_rate_dq = Float64(successes_dq) / Float64(eval_episodes) * 100.0
    print("")
    print("Double Q-Learning success rate:", String(success_rate_dq)[:5], "%")
    print("")

    # ========================================================================
    # Results Summary
    # ========================================================================
    print("=" * 60)
    print("    Results Summary")
    print("=" * 60)
    print("")
    print("Environment          | Success Rate")
    print("-" * 60)
    print("Non-slippery (easy)  |", String(result_easy[1])[:5], "%")
    print("Q-Learning (slippery)|", String(result_q[1])[:5], "%")
    print("Double Q (slippery)  |", String(success_rate_dq)[:5], "%")
    print("")
    print("Note: Slippery FrozenLake is genuinely hard!")
    print("A 60-70% success rate is considered good for this environment.")
    print("")

    # ========================================================================
    # Demonstrate learned policy
    # ========================================================================
    print("-" * 60)
    print("Demonstrating learned policy (Q-Learning on slippery):")
    print("-" * 60)
    print("")

    var action_names = List[String]()
    action_names.append("LEFT ")
    action_names.append("DOWN ")
    action_names.append("RIGHT")
    action_names.append("UP   ")

    # Show policy map
    print("Learned policy (intended actions):")
    for row in range(4):
        var line = String("  ")
        for col in range(4):
            var pos = row * 4 + col
            # Check if hole or goal
            var is_hole = (pos == 5 or pos == 7 or pos == 11 or pos == 12)
            var is_goal = pos == 15

            if is_hole:
                line += " H   "
            elif is_goal:
                line += " G   "
            else:
                var best_action = agent_q.get_best_action(pos)
                line += action_names[best_action] + " "
        print(line)
    print("")

    # Run a demo episode
    print("Demo episode:")
    var demo_env = FrozenLakeEnv(size=4, is_slippery=True)
    var renderer = RendererBase()
    _ = demo_env.reset()
    demo_env.render(renderer)

    for step in range(20):
        var state = demo_env.get_state()
        var state_idx = demo_env.state_to_index(state)
        var action_idx = agent_q.get_best_action(state_idx)
        var action = demo_env.action_from_index(action_idx)

        print("Step", step + 1, "- Intended:", action_names[action_idx])
        var result = demo_env.step(action)
        demo_env.render(renderer)

        if result[2]:
            if result[1] > 0.5:
                print("SUCCESS! Reached the goal!")
            else:
                print("Fell in a hole. (The ice is slippery!)")
            break

    print("")
    print("=" * 60)
