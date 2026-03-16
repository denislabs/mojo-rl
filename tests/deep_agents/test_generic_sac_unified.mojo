"""Test: SAC through the unified GenericOffPolicyAgent (not GenericSACAgent).

Validates that the composable strategy approach works for SAC —
the same GenericOffPolicyAgent that handles DDPG/TD3 now also handles SAC
via Config strategies (ReparamTarget, EntropicTwinQTarget, MaxEntLoss, etc.)
"""

from std.random import seed

from mojo_rl.deep_agents.core.generic import (
    GenericOffPolicyAgent,
    GenericSACAgent,
    SACConfig,
)
from mojo_rl.envs.pendulum import PendulumEnv


fn main() raises:
    print("=== Unified SAC via GenericOffPolicyAgent[SACConfig] ===\n")

    # 1. Unified agent (new path — GenericOffPolicyAgent with SACConfig)
    print("1. GenericOffPolicyAgent[SACConfig] (5 episodes)...")
    seed(42)
    var unified = GenericOffPolicyAgent[SACConfig[3, 1, 64, 1000, 32]](
        action_scale=2.0
    )
    var env1 = PendulumEnv[DType.float64]()
    var m1 = unified.train(env1, num_episodes=5)
    print(
        "   steps:", unified.train_step_count,
        " alpha:", unified.alpha,
    )

    # 2. Old GenericSACAgent for comparison
    print("\n2. GenericSACAgent[SACConfig] (5 episodes)...")
    seed(42)
    var old_sac = GenericSACAgent[SACConfig[3, 1, 64, 1000, 32]](
        action_scale=2.0, target_entropy=-1.0
    )
    var env2 = PendulumEnv[DType.float64]()
    var m2 = old_sac.train(env2, num_episodes=5)
    print(
        "   steps:", old_sac.train_step_count,
        " alpha:", old_sac.alpha,
    )

    # 3. Validation
    print("\n3. Comparison...")
    if unified.train_step_count == old_sac.train_step_count:
        print("   OK: Same step count (", unified.train_step_count, ")")
    else:
        print(
            "   WARN: Different step counts:",
            unified.train_step_count, "vs", old_sac.train_step_count,
        )

    if unified.train_step_count > 0:
        print("   OK: Unified SAC agent trained successfully")
    else:
        print("   FAIL: Unified SAC agent did not train")

    print("\n=== Done ===")
