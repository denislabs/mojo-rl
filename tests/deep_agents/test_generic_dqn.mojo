"""Test: GenericDQNAgent with DQNConfig/DoubleDQNConfig/DuelingDQNConfig on CartPole."""

from std.random import seed

from mojo_rl.deep_agents.core.agents import (
    GenericDQNAgent,
    DQNConfig,
    DoubleDQNConfig,
    DuelingDQNConfig,
)
from mojo_rl.envs import CartPoleEnv


def main() raises:
    print("=== Generic DQN Test (CPU) ===\n")

    # Test 1: Standard DQN
    print("1. GenericDQNAgent[DQNConfig] (200 episodes)...")
    seed(42)
    var dqn = GenericDQNAgent[DQNConfig[4, 2, 120, 84, 1000, 32]]()
    var env1 = CartPoleEnv[DType.float64]()
    var m1 = dqn.train(env1, num_episodes=200)
    print(
        "   steps:",
        dqn.train_step_count,
        " epsilon:",
        dqn.epsilon,
        " last-20 avg:",
        m1.mean_reward_last_n(20),
    )

    # Test 2: Double DQN
    print("\n2. GenericDQNAgent[DoubleDQNConfig] (200 episodes)...")
    seed(42)
    var ddqn = GenericDQNAgent[DoubleDQNConfig[4, 2, 120, 84, 1000, 32]]()
    var env2 = CartPoleEnv[DType.float64]()
    var m2 = ddqn.train(env2, num_episodes=200)
    print(
        "   steps:",
        ddqn.train_step_count,
        " epsilon:",
        ddqn.epsilon,
        " last-20 avg:",
        m2.mean_reward_last_n(20),
    )

    # Test 3: Dueling DQN (CPU)
    print("\n3. GenericDQNAgent[DuelingDQNConfig] (5000 episodes)...")
    seed(42)
    var dueling = GenericDQNAgent[DuelingDQNConfig[4, 2, 120, 84, 1000, 32]]()
    var env3 = CartPoleEnv[DType.float64]()
    var m3 = dueling.train(env3, num_episodes=5000)
    print(
        "   steps:",
        dueling.train_step_count,
        " epsilon:",
        dueling.epsilon,
        " last-20 avg:",
        m3.mean_reward_last_n(20),
    )

    # Checks
    print("\n4. Validation...")
    if dqn.train_step_count > 0:
        print(
            "   OK: Standard DQN trained ("
            + String(dqn.train_step_count)
            + " steps)"
        )
    else:
        print("   FAIL: Standard DQN did not train")

    if ddqn.train_step_count > 0:
        print(
            "   OK: Double DQN trained ("
            + String(ddqn.train_step_count)
            + " steps)"
        )
    else:
        print("   FAIL: Double DQN did not train")

    if dueling.train_step_count > 0:
        print(
            "   OK: Dueling DQN trained ("
            + String(dueling.train_step_count)
            + " steps)"
        )
    else:
        print("   FAIL: Dueling DQN did not train")

    print("\n=== Done ===")
