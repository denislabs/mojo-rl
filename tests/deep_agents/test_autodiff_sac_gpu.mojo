"""Test AutodiffSAC GPU training on Pendulum.

Verifies that the GPU autodiff path works end-to-end.

Run with:
    pixi run -e apple mojo run -I . tests/deep_agents/test_autodiff_sac_gpu.mojo
"""

from std.random import seed
from std.memory import UnsafePointer
from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import (
    GenericOffPolicyAgent,
    AutodiffSACConfig,
)
from mojo_rl.deep_agents.core import run_offpolicy_continuous_train_gpu
from mojo_rl.envs.pendulum import PendulumV2
from mojo_rl.nn.constants import dtype
from mojo_rl.core.logger import NoOpLogger


def main() raises:
    seed(42)
    print("=== AutodiffSAC GPU Training Test ===")
    print()

    # Pendulum V2 (GPU): obs_dim=3, action_dim=1
    comptime OBS = 3
    comptime ACT = 1
    comptime Config = AutodiffSACConfig[OBS, ACT, 64, 10000, 32]

    print("Config:", Config.NAME)
    print("  ActorLoss: AutodiffMaxEntLoss (GPU autodiff graph)")
    print()

    with DeviceContext() as ctx:
        var agent = GenericOffPolicyAgent[Config, max_n_envs=4](
            gamma=0.99,
            tau=0.005,
            action_scale=2.0,
            auto_alpha=True,
            alpha=0.2,
        )

        print("Training AutodiffSAC on Pendulum GPU (5000 steps)...")

        var metrics = agent.train_gpu[PendulumV2[dtype]](
            ctx,
            num_steps=5000,
            warmup_steps=500,
            verbose=True,
            print_every=1000,
        )

        print()
        var n_eps = len(metrics.episodes)
        print("Episodes completed:", n_eps)

        if n_eps > 0:
            print("Final avg reward:", metrics.mean_reward_last_n(5))
            print("[PASS] AutodiffSAC GPU completed without errors!")
        else:
            print("[FAIL] No episodes completed")
