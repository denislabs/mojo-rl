"""Test: GenericOffPolicyAgent[TD3Config] GPU training."""

from std.random import seed
from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.generic import GenericOffPolicyAgent, TD3Config
from mojo_rl.envs.pendulum import PendulumV2


fn main() raises:
    print("=== Generic TD3 GPU Test ===\n")

    seed(42)
    var agent = GenericOffPolicyAgent[TD3Config[3, 1, 64, 1000, 32]](
        action_scale=2.0
    )

    var ctx = DeviceContext()
    print("Training TD3 on GPU (10000 steps)...")
    var metrics = agent.train_gpu[PendulumV2[DType.float32]](
        ctx, num_steps=10000, warmup_steps=1000
    )
    print("  total_steps:", agent.total_steps)

    if agent.total_steps >= 10000:
        print("  OK: TD3 GPU training completed")
    else:
        print("  FAIL: TD3 GPU training did not complete")

    print("\n=== Done ===")
