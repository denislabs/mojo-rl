"""Test: GenericOffPolicyAgent[DDPGConfig] trains on PendulumEnv."""

from std.random import seed

from mojo_rl.deep_agents.core.generic.offpolicy_agent import (
    GenericOffPolicyAgent,
)
from mojo_rl.deep_agents.core.generic.offpolicy_config import DDPGConfig
from mojo_rl.envs.pendulum import PendulumEnv


fn main() raises:
    print("=== Generic DDPG Training Test ===\n")

    seed(42)
    var agent = GenericOffPolicyAgent[DDPGConfig[3, 1, 64, 1000, 32]](
        action_scale=2.0
    )
    var env = PendulumEnv[DType.float64]()
    var metrics = agent.train(env, num_episodes=5)
    print(
        "  train_steps =",
        agent.train_step_count,
        " episodes =",
        len(metrics.episodes),
    )

    if agent.train_step_count > 0:
        print("  OK: Generic DDPG agent trained successfully")
    else:
        print("  FAIL: agent did not train")

    print("\n=== Done ===")
