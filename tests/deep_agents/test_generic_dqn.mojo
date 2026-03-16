"""Test: GenericDQNAgent with DQNConfig on CartPole."""

from std.random import seed

from mojo_rl.deep_agents.core.generic import GenericDQNAgent, DQNConfig, DoubleDQNConfig
from mojo_rl.envs import CartPoleEnv


fn main() raises:
    print("=== Generic DQN Test ===\n")

    # Test 1: Standard DQN
    print("1. GenericDQNAgent[DQNConfig] (10 episodes)...")
    seed(42)
    var dqn = GenericDQNAgent[DQNConfig[4, 2, 120, 84, 1000, 32]]()
    var env1 = CartPoleEnv[DType.float64]()
    var m1 = dqn.train(env1, num_episodes=10)
    print("   steps:", dqn.train_step_count, " epsilon:", dqn.epsilon)

    # Test 2: Double DQN
    print("\n2. GenericDQNAgent[DoubleDQNConfig] (10 episodes)...")
    seed(42)
    var ddqn = GenericDQNAgent[DoubleDQNConfig[4, 2, 120, 84, 1000, 32]]()
    var env2 = CartPoleEnv[DType.float64]()
    var m2 = ddqn.train(env2, num_episodes=10)
    print("   steps:", ddqn.train_step_count, " epsilon:", ddqn.epsilon)

    # Checks
    print("\n3. Validation...")
    if dqn.train_step_count > 0:
        print("   OK: Standard DQN trained (" + String(dqn.train_step_count) + " steps)")
    else:
        print("   FAIL: Standard DQN did not train")

    if ddqn.train_step_count > 0:
        print("   OK: Double DQN trained (" + String(ddqn.train_step_count) + " steps)")
    else:
        print("   FAIL: Double DQN did not train")

    print("\n=== Done ===")
