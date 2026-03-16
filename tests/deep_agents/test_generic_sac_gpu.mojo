"""Test: SAC GPU training via unified GenericOffPolicyAgent."""
from std.random import seed
from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.core.generic import GenericOffPolicyAgent, SACConfig
from mojo_rl.envs.pendulum import PendulumV2

fn main() raises:
    print("=== Generic SAC GPU Test ===")
    seed(42)
    var agent = GenericOffPolicyAgent[SACConfig[3, 1, 64, 1000, 32]](
        action_scale=2.0
    )
    var ctx = DeviceContext()
    var metrics = agent.train_gpu[PendulumV2[DType.float32]](
        ctx, num_steps=10000, warmup_steps=1000
    )
    print("  total_steps:", agent.total_steps)
    print("  alpha:", agent.alpha)
    if agent.total_steps >= 10000:
        print("  OK: SAC GPU training completed")
    else:
        print("  FAIL")
    print("=== Done ===")
