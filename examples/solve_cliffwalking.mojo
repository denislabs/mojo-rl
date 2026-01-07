"""Solve CliffWalking - SARSA vs Q-Learning comparison.

CliffWalking is a classic example demonstrating the difference between
on-policy (SARSA) and off-policy (Q-Learning) methods.

Grid layout (4x12):
    . . . . . . . . . . . .
    . . . . . . . . . . . .
    . . . . . . . . . . . .
    S C C C C C C C C C C G

    S = Start (bottom-left)
    G = Goal (bottom-right)
    C = Cliff (falling resets to start with -100 reward)
    . = Safe cells (-1 reward per step)

Key insight:
- Q-Learning learns the OPTIMAL path (along the cliff edge) but frequently
  falls during training due to epsilon-greedy exploration.
- SARSA learns a SAFER path (going up and around) because it accounts for
  its own exploratory actions when learning.

This makes CliffWalking a famous example of why SARSA can outperform
Q-Learning during training, even though Q-Learning finds the optimal policy.

Run with:
    pixi run mojo run examples/solve_cliffwalking.mojo
"""

from envs import CliffWalkingEnv
from agents import QLearningAgent, SARSAAgent, SARSALambdaAgent


fn main() raises:
    print("=" * 60)
    print("    Solving CliffWalking - SARSA vs Q-Learning")
    print("=" * 60)
    print("")
    print("Environment: 4x12 CliffWalking")
    print("Start: bottom-left | Goal: bottom-right")
    print("Cliff: bottom row (except start/goal)")
    print("Reward: -1 per step, -100 for falling off cliff")
    print("")
    print("Optimal path: Walk along cliff edge (13 steps, reward = -13)")
    print("Safe path: Go up, across, down (17 steps, reward = -17)")
    print("")

    var num_episodes = 500
    var max_steps = 200

    # ========================================================================
    # Algorithm 1: Q-Learning (Off-policy)
    # ========================================================================
    print("-" * 60)
    print("Algorithm 1: Q-Learning (Off-policy)")
    print("-" * 60)
    print("Q-Learning learns the optimal policy but exploration near")
    print("the cliff causes frequent falls during training.")
    print("")

    var env_q = CliffWalkingEnv(width=12, height=4)
    var agent_q = QLearningAgent(
        num_states=env_q.num_states(),
        num_actions=env_q.num_actions(),
        learning_rate=0.5,
        discount_factor=1.0,  # Undiscounted for shortest path
        epsilon=0.1,  # Fixed epsilon for fair comparison
        epsilon_decay=1.0,  # No decay
        epsilon_min=0.1,
    )

    var metrics_q = agent_q.train(
        env_q,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
        print_every=100,
        environment_name="CliffWalking",
    )

    var eval_q = agent_q.evaluate(env_q, num_episodes=10)
    print("")
    print("Q-Learning training mean reward:", String(metrics_q.mean_reward())[:8])
    print("Q-Learning evaluation reward:", String(eval_q)[:8])
    print("")

    # ========================================================================
    # Algorithm 2: SARSA (On-policy)
    # ========================================================================
    print("-" * 60)
    print("Algorithm 2: SARSA (On-policy)")
    print("-" * 60)
    print("SARSA learns a safer policy because it considers its own")
    print("exploratory actions when updating values.")
    print("")

    var env_s = CliffWalkingEnv(width=12, height=4)
    var agent_s = SARSAAgent(
        num_states=env_s.num_states(),
        num_actions=env_s.num_actions(),
        learning_rate=0.5,
        discount_factor=1.0,
        epsilon=0.1,
        epsilon_decay=1.0,
        epsilon_min=0.1,
    )

    var metrics_s = agent_s.train(
        env_s,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
        print_every=100,
        environment_name="CliffWalking",
    )

    var eval_s = agent_s.evaluate(env_s, num_episodes=10)
    print("")
    print("SARSA training mean reward:", String(metrics_s.mean_reward())[:8])
    print("SARSA evaluation reward:", String(eval_s)[:8])
    print("")

    # ========================================================================
    # Algorithm 3: SARSA(lambda) with eligibility traces
    # ========================================================================
    print("-" * 60)
    print("Algorithm 3: SARSA(lambda) - Eligibility Traces")
    print("-" * 60)
    print("SARSA(lambda) propagates rewards faster using traces.")
    print("")

    var env_sl = CliffWalkingEnv(width=12, height=4)
    var agent_sl = SARSALambdaAgent(
        num_states=env_sl.num_states(),
        num_actions=env_sl.num_actions(),
        learning_rate=0.5,
        discount_factor=1.0,
        lambda_=0.9,
        epsilon=0.1,
        epsilon_decay=1.0,
        epsilon_min=0.1,
    )

    var metrics_sl = agent_sl.train(
        env_sl,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
        print_every=100,
        environment_name="CliffWalking",
    )

    var eval_sl = agent_sl.evaluate(env_sl, num_episodes=10)
    print("")
    print("SARSA(lambda) training mean reward:", String(metrics_sl.mean_reward())[:8])
    print("SARSA(lambda) evaluation reward:", String(eval_sl)[:8])
    print("")

    # ========================================================================
    # Results Summary
    # ========================================================================
    print("=" * 60)
    print("    Results Summary")
    print("=" * 60)
    print("")
    print("Algorithm     | Training Mean | Eval Reward | Policy Type")
    print("-" * 60)
    print(
        "Q-Learning    |",
        String(metrics_q.mean_reward())[:8],
        "    |",
        String(eval_q)[:8],
        "   | Optimal (risky)",
    )
    print(
        "SARSA         |",
        String(metrics_s.mean_reward())[:8],
        "    |",
        String(eval_s)[:8],
        "   | Safe (suboptimal)",
    )
    print(
        "SARSA(lambda) |",
        String(metrics_sl.mean_reward())[:8],
        "    |",
        String(eval_sl)[:8],
        "   | Safe (suboptimal)",
    )
    print("")
    print("Key observation:")
    print("- Q-Learning has LOWER training reward (more cliff falls)")
    print("- But Q-Learning has HIGHER eval reward (optimal policy)")
    print("- SARSA's training reward is more stable (safer exploration)")
    print("")

    # ========================================================================
    # Visualize learned policies
    # ========================================================================
    print("-" * 60)
    print("Learned Policies Visualization")
    print("-" * 60)

    var action_chars = List[String]()
    action_chars.append("^")  # up
    action_chars.append(">")  # right
    action_chars.append("v")  # down
    action_chars.append("<")  # left

    print("")
    print("Q-Learning policy (greedy, no exploration):")
    for y in range(3, -1, -1):
        var line = String("  ")
        for x in range(12):
            if y == 0 and x == 0:
                line += "S "
            elif y == 0 and x == 11:
                line += "G "
            elif y == 0 and x > 0 and x < 11:
                line += "C "
            else:
                var state_idx = y * 12 + x
                var best_action = agent_q.get_best_action(state_idx)
                line += action_chars[best_action] + " "
        print(line)

    print("")
    print("SARSA policy (greedy, no exploration):")
    for y in range(3, -1, -1):
        var line = String("  ")
        for x in range(12):
            if y == 0 and x == 0:
                line += "S "
            elif y == 0 and x == 11:
                line += "G "
            elif y == 0 and x > 0 and x < 11:
                line += "C "
            else:
                var state_idx = y * 12 + x
                var best_action = agent_s.get_best_action(state_idx)
                line += action_chars[best_action] + " "
        print(line)

    print("")

    # ========================================================================
    # Demo: Run both policies
    # ========================================================================
    print("-" * 60)
    print("Demo: Running Q-Learning policy")
    print("-" * 60)

    var demo_env = CliffWalkingEnv(width=12, height=4)
    _ = demo_env.reset()
    demo_env.render()

    var action_names = List[String]()
    action_names.append("UP")
    action_names.append("RIGHT")
    action_names.append("DOWN")
    action_names.append("LEFT")

    var total_reward_q: Float64 = 0.0
    for step in range(30):
        var state = demo_env.get_state()
        var state_idx = demo_env.state_to_index(state)
        var action_idx = agent_q.get_best_action(state_idx)
        var action = demo_env.action_from_index(action_idx)

        var result = demo_env.step(action)
        total_reward_q += result[1]

        if result[2]:
            print("Goal reached in", step + 1, "steps!")
            print("Total reward:", total_reward_q)
            demo_env.render()
            break

    print("")
    print("Note: Q-Learning finds the optimal path along the cliff edge.")
    print("With epsilon=0 (greedy), it's safe. During training, falls happen!")
    print("")
    print("=" * 60)
