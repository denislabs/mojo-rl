"""Diagnostic DreamerV3 GPU training on Pendulum — no logger."""

from std.gpu.host import DeviceContext
from mojo_rl.nn.constants import dtype
from mojo_rl.envs.pendulum import PendulumV2
from mojo_rl.deep_agents.dreamer_v3 import DreamerV3Agent


comptime N_ENVS = 8


fn main() raises:
    print("=" * 60)
    print("DreamerV3 GPU Diagnostic — Pendulum")
    print("=" * 60)

    comptime OBS = 3
    comptime ACT = 1

    var agent = DreamerV3Agent[
        obs_dim=OBS,
        action_dim=ACT,
        deter_dim=128,
        hidden=64,
        stoch_dim=8,
        classes=8,
        units=64,
        num_bins=65,
        blocks=2,
        batch_size=8,
        batch_length=16,
        imagine_horizon=8,
        buffer_capacity=50000,
    ](warmup_steps=500)

    var ctx = DeviceContext()

    var metrics = agent.train_gpu[PendulumV2[dtype], n_envs=N_ENVS](
        ctx,
        num_episodes=500,
        sync_every=100,
        verbose=True,
        print_every=5000,
        diag_every=500,
    )

    print("=" * 60)
    print("Done.")
    print("=" * 60)
