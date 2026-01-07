"""Solve Taxi - Larger state space with pickup/dropoff actions.

Taxi is a 5x5 grid environment where the agent must:
1. Navigate to the passenger's location
2. Pick up the passenger
3. Navigate to the destination
4. Drop off the passenger

Grid layout:
    +---------+
    |R: | : :G|
    | : | : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+

State space: 5 * 5 * 5 * 4 = 500 states
  - Taxi position: 25 cells
  - Passenger location: 4 locations + in taxi = 5
  - Destination: 4 locations

Actions: 6 (south, north, east, west, pickup, dropoff)

Rewards:
  - -1 per step
  - +20 for successful dropoff
  - -10 for illegal pickup/dropoff

Best algorithms for Taxi:
1. Q-Learning: Works well with proper exploration
2. Double Q-Learning: Reduces overestimation
3. Dyna-Q: Model-based planning speeds up learning

Run with:
    pixi run mojo run examples/solve_taxi.mojo
"""

from envs import TaxiEnv
from agents import QLearningAgent, DoubleQLearningAgent, DynaQAgent


fn main() raises:
    print("=" * 60)
    print("    Solving Taxi - Pickup/Dropoff Task")
    print("=" * 60)
    print("")
    print("Environment: 5x5 Taxi")
    print("State space: 500 states (position x passenger x destination)")
    print("Actions: 6 (move + pickup + dropoff)")
    print("Reward: +20 dropoff, -10 illegal action, -1 per step")
    print("")
    print("Task: Pick up passenger at R, drop off at G")
    print("Optimal: ~10-12 steps for this specific task")
    print("")

    var num_episodes = 2000
    var max_steps = 200

    # ========================================================================
    # Algorithm 1: Q-Learning
    # ========================================================================
    print("-" * 60)
    print("Algorithm 1: Q-Learning")
    print("-" * 60)

    var env_q = TaxiEnv()
    var agent_q = QLearningAgent(
        num_states=env_q.num_states(),
        num_actions=env_q.num_actions(),
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.01,
    )

    var metrics_q = agent_q.train(
        env_q,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
        print_every=500,
        environment_name="Taxi",
    )

    var eval_q = agent_q.evaluate(env_q, num_episodes=20)
    print("Q-Learning evaluation reward:", String(eval_q)[:8])
    print("")

    # ========================================================================
    # Algorithm 2: Double Q-Learning
    # ========================================================================
    print("-" * 60)
    print("Algorithm 2: Double Q-Learning")
    print("-" * 60)
    print("Reduces overestimation with two Q-tables.")
    print("")

    var env_dq = TaxiEnv()
    var agent_dq = DoubleQLearningAgent(
        num_states=env_dq.num_states(),
        num_actions=env_dq.num_actions(),
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.01,
    )

    var metrics_dq = agent_dq.train(
        env_dq,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
        print_every=500,
        environment_name="Taxi",
    )

    var eval_dq = agent_dq.evaluate(env_dq, num_episodes=20)
    print("Double Q-Learning evaluation reward:", String(eval_dq)[:8])
    print("")

    # ========================================================================
    # Algorithm 3: Dyna-Q (Model-based)
    # ========================================================================
    print("-" * 60)
    print("Algorithm 3: Dyna-Q (Model-based Planning)")
    print("-" * 60)
    print("Uses 10 planning steps per real step for faster learning.")
    print("")

    var env_dyna = TaxiEnv()
    var agent_dyna = DynaQAgent(
        num_states=env_dyna.num_states(),
        num_actions=env_dyna.num_actions(),
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.01,
        n_planning=10,  # 10 planning steps per real step
    )

    var metrics_dyna = agent_dyna.train(
        env_dyna,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=True,
        print_every=500,
        environment_name="Taxi",
    )

    var eval_dyna = agent_dyna.evaluate(env_dyna, num_episodes=20)
    print("Dyna-Q evaluation reward:", String(eval_dyna)[:8])
    print("")

    # ========================================================================
    # Results Summary
    # ========================================================================
    print("=" * 60)
    print("    Results Summary")
    print("=" * 60)
    print("")
    print("Algorithm       | Training Mean | Max Reward | Eval Reward")
    print("-" * 60)
    print(
        "Q-Learning      |",
        String(metrics_q.mean_reward())[:8],
        "     |",
        String(metrics_q.max_reward())[:8],
        "  |",
        String(eval_q)[:8],
    )
    print(
        "Double Q-Learn  |",
        String(metrics_dq.mean_reward())[:8],
        "     |",
        String(metrics_dq.max_reward())[:8],
        "  |",
        String(eval_dq)[:8],
    )
    print(
        "Dyna-Q          |",
        String(metrics_dyna.mean_reward())[:8],
        "     |",
        String(metrics_dyna.max_reward())[:8],
        "  |",
        String(eval_dyna)[:8],
    )
    print("")
    print("Good performance: Eval reward around +8 to +10")
    print("(Task takes ~10 steps: -10 + 20 = +10 optimal)")
    print("")

    # ========================================================================
    # Demo: Watch the trained agent
    # ========================================================================
    print("-" * 60)
    print("Demo: Watching Q-Learning agent solve the task")
    print("-" * 60)
    print("")

    var action_names = List[String]()
    action_names.append("SOUTH ")
    action_names.append("NORTH ")
    action_names.append("EAST  ")
    action_names.append("WEST  ")
    action_names.append("PICKUP")
    action_names.append("DROPOFF")

    var demo_env = TaxiEnv()
    _ = demo_env.reset()

    print("Initial state:")
    demo_env.render()

    var total_reward: Float64 = 0.0
    for step in range(30):
        var state = demo_env.get_state()
        var state_idx = demo_env.state_to_index(state)
        var action_idx = agent_q.get_best_action(state_idx)
        var action = demo_env.action_from_index(action_idx)

        print("Step", step + 1, "- Action:", action_names[action_idx])

        var result = demo_env.step(action)
        total_reward += result[1]

        demo_env.render()

        if result[2]:
            print("Task completed!")
            print("Total reward:", total_reward)
            print("Steps taken:", step + 1)
            break

    print("")
    print("Legend:")
    print("  t = empty taxi")
    print("  T = taxi with passenger")
    print("  R/G/Y/B = pickup/dropoff locations")
    print("=" * 60)
