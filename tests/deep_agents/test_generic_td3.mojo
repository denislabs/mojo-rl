"""Test: GenericOffPolicyAgent with TD3Config trains on PendulumEnv."""

from std.random import seed

from mojo_rl.deep_agents.core.agents import (
    GenericOffPolicyAgent,
    DDPGConfig,
    TD3Config,
)
from mojo_rl.deep_agents.core.agents import DeepTD3Agent
from mojo_rl.envs.pendulum import PendulumEnv


def main() raises:
    print("=== Generic TD3 Test ===\n")

    # Test 1: DDPG still works after refactor
    print("1. DDPG (5 episodes)...")
    seed(42)
    var ddpg = GenericOffPolicyAgent[DDPGConfig[3, 1, 64, 1000, 32]](
        action_scale=2.0
    )
    var env1 = PendulumEnv[DType.float64]()
    var m1 = ddpg.train(env1, num_episodes=5)
    print("   steps:", ddpg.train_step_count, " OK")

    # Test 2: TD3 trains
    print("\n2. TD3 Generic (5 episodes)...")
    seed(42)
    var td3 = GenericOffPolicyAgent[TD3Config[3, 1, 64, 1000, 32]](
        action_scale=2.0
    )
    var env2 = PendulumEnv[DType.float64]()
    var m2 = td3.train(env2, num_episodes=5)
    print("   steps:", td3.train_step_count, " OK")

    # Test 3: Old TD3 for comparison
    print("\n3. Old DeepTD3Agent (5 episodes)...")
    seed(42)
    var old_td3 = DeepTD3Agent[3, 1, 64, 1000, 32](action_scale=2.0)
    var env3 = PendulumEnv[DType.float64]()
    var m3 = old_td3.train(env3, num_episodes=5)
    print("   steps:", old_td3.train_step_count, " OK")

    # Comparison
    print("\n4. Comparison...")
    if td3.train_step_count > 0:
        print("   OK: Generic TD3 trained")
    else:
        print("   FAIL: Generic TD3 did not train")

    if td3.train_step_count == old_td3.train_step_count:
        print("   OK: Same step count (" + String(td3.train_step_count) + ")")
    else:
        print(
            "   WARN: Step count differs (generic="
            + String(td3.train_step_count)
            + " old="
            + String(old_td3.train_step_count)
            + ")"
        )

    print("\n=== Done ===")
