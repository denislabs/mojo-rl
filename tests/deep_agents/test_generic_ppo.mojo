"""Test: GenericOnPolicyAgent with PPOConfig and A2CConfig on CartPole."""

from std.random import seed

from mojo_rl.deep_agents.core.generic import (
    GenericOnPolicyAgent,
    PPOConfig,
    A2CConfig,
)
from mojo_rl.deep_agents.ppo import DeepPPOAgent
from mojo_rl.envs import CartPoleEnv


fn main() raises:
    print("=== Generic On-Policy Agent Test ===\n")

    # Test 1: PPO
    print("1. GenericOnPolicyAgent[PPOConfig] (20 updates)...")
    seed(42)
    var ppo = GenericOnPolicyAgent[PPOConfig[4, 2, 64, 128]]()
    var env1 = CartPoleEnv[DType.float64]()
    var m1 = ppo.train(env1, num_updates=20)
    print(
        "   updates:", ppo.train_step_count,
        " entries:", len(m1.episodes),
    )

    # Test 2: A2C
    print("\n2. GenericOnPolicyAgent[A2CConfig] (20 updates)...")
    seed(42)
    var a2c = GenericOnPolicyAgent[A2CConfig[4, 2, 128, 128]]()
    var env2 = CartPoleEnv[DType.float64]()
    var m2 = a2c.train(env2, num_updates=20)
    print(
        "   updates:", a2c.train_step_count,
        " entries:", len(m2.episodes),
    )

    # Test 3: Old PPO for comparison
    print("\n3. DeepPPOAgent (20 updates)...")
    seed(42)
    var old_ppo = DeepPPOAgent[4, 2, 64, 128]()
    var env3 = CartPoleEnv[DType.float64]()
    var m3 = old_ppo.train(env3, num_episodes=20)
    print(
        "   updates:", old_ppo.train_step_count,
    )

    # Comparison
    print("\n4. Comparison...")
    if ppo.train_step_count > 0:
        print("   OK: Generic PPO trained (" + String(ppo.train_step_count) + " updates)")
    else:
        print("   FAIL: Generic PPO did not train")

    if a2c.train_step_count > 0:
        print("   OK: Generic A2C trained (" + String(a2c.train_step_count) + " updates)")
    else:
        print("   FAIL: Generic A2C did not train")

    if ppo.train_step_count == old_ppo.train_step_count:
        print("   OK: Same update count as old PPO")
    else:
        print(
            "   WARN: Update count differs (generic="
            + String(ppo.train_step_count)
            + " old="
            + String(old_ppo.train_step_count)
            + ")"
        )

    print("\n=== Done ===")
