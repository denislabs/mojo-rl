"""Test: GenericOffPolicyAgent[SACConfig] trains on PendulumEnv."""

from std.random import seed

from mojo_rl.deep_agents.core.agents import GenericOffPolicyAgent, SACConfig
from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.pendulum import PendulumEnv


def main() raises:
    print("=== Generic SAC Test ===\n")

    # Test 1: Unified SAC agent
    print("1. GenericOffPolicyAgent[SACConfig] (5 episodes)...")
    seed(42)
    var sac = GenericOffPolicyAgent[SACConfig[3, 1, 64, 1000, 32]](
        action_scale=2.0
    )
    var env1 = PendulumEnv[DType.float64]()
    var m1 = sac.train(env1, num_episodes=5)
    print("   steps:", sac.train_step_count, " alpha:", sac.alpha)

    # Test 2: Old SAC for comparison
    print("\n2. DeepSACAgent (5 episodes)...")
    seed(42)
    var old_sac = DeepSACAgent[3, 1, 64, 1000, 32](
        action_scale=2.0,
    )
    var env2 = PendulumEnv[DType.float64]()
    var m2 = old_sac.train(env2, num_episodes=5)
    print("   steps:", old_sac.train_step_count, " alpha:", old_sac.alpha)

    # Comparison
    print("\n3. Comparison...")
    if sac.train_step_count > 0:
        print(
            "   OK: Generic SAC trained ("
            + String(sac.train_step_count)
            + " steps)"
        )
    else:
        print("   FAIL: Generic SAC did not train")

    if sac.train_step_count == old_sac.train_step_count:
        print("   OK: Same step count")
    else:
        print(
            "   WARN: Step count differs (generic="
            + String(sac.train_step_count)
            + " old="
            + String(old_sac.train_step_count)
            + ")"
        )

    print("\n=== Done ===")
