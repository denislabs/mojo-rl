"""Test: GenericOffPolicyAgent[DDPGConfig] GPU training on PendulumV2."""

from std.random import seed
from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import GenericOffPolicyAgent, DDPGConfig
from mojo_rl.envs.pendulum import PendulumV2


def main() raises:
    print("=== Generic DDPG GPU Test ===\n")

    seed(42)
    var agent = GenericOffPolicyAgent[DDPGConfig[3, 1, 64, 1000, 32]](
        action_scale=2.0
    )

    var ctx = DeviceContext()

    # Train on GPU
    print("Training on GPU (10000 steps)...")
    var metrics = agent.train_gpu[PendulumV2[DType.float32]](
        ctx, num_steps=10000, warmup_steps=1000
    )
    print("  total_steps:", agent.total_steps)
    print("  episodes:", len(metrics.episodes))

    if agent.total_steps >= 10000:
        print("  OK: GPU training completed")
    else:
        print("  FAIL: GPU training did not complete")

    print("\n=== Done ===")
