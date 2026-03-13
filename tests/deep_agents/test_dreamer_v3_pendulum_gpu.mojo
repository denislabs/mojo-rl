"""Test DreamerV3 GPU training on Pendulum environment."""

from std.gpu.host import DeviceContext
from mojo_rl.nn.constants import dtype
from mojo_rl.envs.pendulum import PendulumV2
from mojo_rl.deep_agents.dreamer_v3 import DreamerV3Agent, run_dreamer_v3_training_gpu


fn main() raises:
    print("=" * 60)
    print("DreamerV3 GPU Training — Pendulum")
    print("=" * 60)

    # Pendulum: obs_dim=3 (cos, sin, theta_dot), action_dim=1 (torque)
    comptime OBS = 3
    comptime ACT = 1

    var env = PendulumV2[dtype]()
    var agent = DreamerV3Agent[
        obs_dim = OBS,
        action_dim = ACT,
        deter_dim = 128,
        hidden = 64,
        stoch_dim = 8,
        classes = 8,
        units = 64,
        num_bins = 65,
        blocks = 2,
        batch_size = 8,
        batch_length = 16,
        imagine_horizon = 8,
        buffer_capacity = 50000,
    ](warmup_steps=500)

    var ctx = DeviceContext()

    run_dreamer_v3_training_gpu[
        PendulumV2[dtype],
        obs_dim = OBS,
        action_dim = ACT,
        deter_dim = 128,
        hidden = 64,
        stoch_dim = 8,
        classes = 8,
        units = 64,
        num_bins = 65,
        blocks = 2,
        batch_size = 8,
        batch_length = 16,
        imagine_horizon = 8,
        buffer_capacity = 50000,
    ](
        env,
        agent,
        ctx,
        total_timesteps = 10000,
        train_every = 5,
        seed_episodes = 3,
        print_every = 5,
        sync_every = 50,
    )

    print("=" * 60)
    print("Done.")
    print("=" * 60)
