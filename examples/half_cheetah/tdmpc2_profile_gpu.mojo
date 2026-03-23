"""TD-MPC2 GPU profiling script — short run for nsys/ncu.

Runs just 10 episodes with warmup_steps=1000 so MPPI kicks in quickly.
Use with:
    nsys profile --trace=cuda -o tdmpc2_profile \
      pixi run -e nvidia mojo run -I . examples/half_cheetah/tdmpc2_profile_gpu.mojo

    nsys stats tdmpc2_profile.nsys-rep
    nsys stats --report cuda_gpu_kern_sum tdmpc2_profile.nsys-rep
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.tdmpc2 import TDMPC2Agent
from mojo_rl.envs.half_cheetah import (
    HalfCheetah,
    HalfCheetahConfig,
)


# HalfCheetah: 17D observation, 6D continuous action
comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM  # 6

# Network architecture (TD-MPC2 5M config)
comptime LATENT_DIM = 512
comptime MLP_DIM = 512
comptime ENC_DIM = 256
comptime NUM_BINS = 101
comptime NUM_Q = 5
comptime HORIZON = 3
comptime NUM_SAMPLES = 512
comptime NUM_PI_TRAJS = 24
comptime NUM_ITERATIONS = 6
comptime BATCH_SIZE = 256
comptime BUFFER_CAPACITY = 100_000
comptime V_MIN = -10.0
comptime V_MAX = 10.0
comptime N_ENVS = 32

# Short run for profiling
comptime NUM_EPISODES = 10

comptime dtype = DType.float32


def main() raises:
    seed(42)
    print("TD-MPC2 GPU Profiling (short run)")
    print("Episodes:", NUM_EPISODES, "| Warmup: 1000 steps")
    print()

    with DeviceContext() as ctx:
        var agent = TDMPC2Agent[
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            latent_dim=LATENT_DIM,
            mlp_dim=MLP_DIM,
            enc_dim=ENC_DIM,
            num_bins=NUM_BINS,
            num_q=NUM_Q,
            horizon=HORIZON,
            batch_size=BATCH_SIZE,
            buffer_capacity=BUFFER_CAPACITY,
            num_samples=NUM_SAMPLES,
            num_pi_trajs=NUM_PI_TRAJS,
            num_iterations=NUM_ITERATIONS,
            v_min=V_MIN,
            v_max=V_MAX,
        ](
            gamma=0.99,
            rho=0.5,
            tau=0.01,
            consistency_coef=20.0,
            reward_coef=0.1,
            value_coef=0.1,
            terminal_coef=1.0,
            entropy_coef=1e-4,
            temperature=0.5,
            action_scale=1.0,
            warmup_steps=1_000,  # Short warmup so MPPI starts quickly
            wm_lr=3e-4,
            enc_lr_scale=0.3,
            pi_lr=3e-4,
            diag_every=50,
        )

        var start_time = perf_counter_ns()

        var metrics = agent.train_gpu[
            HalfCheetah[dtype, TERMINATE_ON_UNHEALTHY=False],
            n_envs=N_ENVS,
        ](
            ctx,
            num_episodes=NUM_EPISODES,
            verbose=True,
        )

        var elapsed_s = Float64(perf_counter_ns() - start_time) / 1e9
        print()
        print("Profiling run done in " + String(elapsed_s)[:6] + "s")
        print("Episodes:", NUM_EPISODES)
        print(
            "Final avg reward (last 5): "
            + String(metrics.mean_reward_last_n(5))[:8]
        )

    print(">>> profiling main() done <<<")
