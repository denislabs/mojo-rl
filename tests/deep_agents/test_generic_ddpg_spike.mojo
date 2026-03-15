"""Test: GenericOffPolicyAgent[Config] compiles with OffPolicyConfig trait."""

from mojo_rl.deep_agents.core.generic import (
    GenericOffPolicyAgent,
    DDPGConfig,
    TD3Config,
)


fn main():
    print("=== Generic Off-Policy Agent with Config Trait ===")

    # Test 1: DDPG
    print("\n1. GenericOffPolicyAgent[DDPGConfig[3, 1, 64, 1000, 32]]...")
    var ddpg = GenericOffPolicyAgent[DDPGConfig[3, 1, 64, 1000, 32]]()
    print("   OBS:", ddpg.OBS, "ACTIONS:", ddpg.ACTIONS)
    print("   OK")

    # Test 2: TD3 (same agent struct, different config)
    print("\n2. GenericOffPolicyAgent[TD3Config[3, 1, 64, 1000, 32]]...")
    var td3 = GenericOffPolicyAgent[TD3Config[3, 1, 64, 1000, 32]]()
    print("   OBS:", td3.OBS, "ACTIONS:", td3.ACTIONS)
    print("   NUM_CRITICS:", td3.Config.NUM_CRITICS)
    print("   OK")

    # Test 3: Create state + workspace views
    print("\n3. Creating DDPG state...")
    var state = ddpg.make_cpu_state()
    print("   OK")

    # Test 4: Exploration
    print("\n4. Exploration rate:", ddpg.get_explore_rate())
    ddpg.decay_explore()
    print("   After decay:", ddpg.get_explore_rate())

    print("\n=== All tests passed! ===")
