"""SAC CPU-vs-GPU target-parity control (HalfCheetah, deep_agents).

Companion control for `mbpo_hc_nn_parity.mojo`. That harness showed the
MBPO SAC-side (mean_q + entropy temperature α) diverges between CPU and GPU
while the dynamics ensemble tracks. But MBPO's CPU and GPU runs also consume
DIFFERENT synthetic data streams (different RNG → different rollout
transitions), so that split could be data-driven rather than a GPU
SAC-kernel bug.

This harness removes the model entirely: plain SAC, same SAC backbone
(identical LayerNorm critic + target_entropy=-3 + arch as the MBPO parity),
trained on 100% REAL data via the SAME single-env driver (`train_single`).
Now CPU and GPU differ ONLY in RNG (replay-sample order + rsample noise) and
draw from the SAME real-buffer distribution. So:

  * CPU and GPU α/mean_q TRACK  → the shared SAC GPU blocks are correct;
    MBPO's split is synthetic-DATA driven → dig into GPU rollout/elite
    data QUALITY, not the SAC kernels.

  * CPU and GPU α/mean_q DIVERGE → a shared SAC GPU-block bug (device-α
    update / actor log-prob reduction / target-Q entropy term). Fix once,
    benefits SAC + MBPO + every off-policy agent.

Same axis convention as the live dashboards: the diag bundle (mean_q,
critic_loss, alpha, …) is keyed on cumulative TRAIN steps; avg_reward /
episodes on env-steps.

Run:
    pixi run -e apple  mojo run -I . examples/half_cheetah/sac_hc_nn_parity.mojo
    pixi run -e nvidia mojo run -I . examples/half_cheetah/sac_hc_nn_parity.mojo

Compare:  /tmp/sac_parity_cpu.csv  vs  /tmp/sac_parity_gpu.csv
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.core.logger import CsvLogger
from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.primitives.layer_norm import LayerNorm
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.sac import SACAgent
from mojo_rl.deep_agents.training.blocks import (
    UniformSampleCpuStep,
    UniformSampleGpuStep,
)
from mojo_rl.envs.half_cheetah import HalfCheetah, HalfCheetahConfig


comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACT_DIM = HalfCheetahConfig.ACTION_DIM  #  6
comptime HIDDEN = 256
comptime BATCH = 128
comptime REPLAY_CAPACITY = 200_000

# Match the MBPO parity scale so the two diagnostics overlay.
comptime NUM_STEPS = 6_000
comptime WARMUP = 1_000
comptime DIAG_EVERY = 250
comptime PRINT_EVERY = 1_000

comptime CPU_CSV = "/tmp/sac_parity_cpu.csv"
comptime GPU_CSV = "/tmp/sac_parity_gpu.csv"


comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
]
# Same LayerNorm critic as the MBPO parity harness — isolates the SAC
# sub-update so the only difference vs MBPO is "no model / 100% real data".
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN],
    LayerNorm[HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    LayerNorm[HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]


def main() raises:
    print("=" * 72)
    print("SAC CPU-vs-GPU target parity (control for MBPO) — HalfCheetah")
    print("  BATCH =", BATCH, " NUM_STEPS =", NUM_STEPS, " WARMUP =", WARMUP)
    print("  target_entropy = -3.0 (matches MBPO parity)")
    print("=" * 72)

    # ─── CPU run ─────────────────────────────────────────────────────────
    seed(42)
    print("\n[1/2] CPU run  →", CPU_CSV)
    var cpu_logger = CsvLogger(file_path=CPU_CSV, buffer_size=64)
    var cpu_logger_ptr = UnsafePointer(to=cpu_logger)
    var cpu_agent = SACAgent[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
        ActorNet, CriticNet,
    ](
        actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
        gamma=0.99, tau=0.005, action_scale=1.0, init_alpha=0.2,
        target_entropy=-3.0, learning_starts=WARMUP,
        window_size=100, initial_episode_fill=0.0,
    )
    var cpu_env = HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False]()
    var t0 = perf_counter_ns()
    _ = cpu_agent.train_single[
        HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False], L=CsvLogger,
    ](
        cpu_env, NUM_STEPS,
        print_every=PRINT_EVERY, verbose=True,
        logger=cpu_logger_ptr, diag_every=DIAG_EVERY,
    )
    var cpu_s = Float64(perf_counter_ns() - t0) / 1e9
    cpu_logger.close()
    _ = cpu_logger
    var cpu_ret = cpu_agent.mean_return()
    print("  CPU done:  mean_return =", cpu_ret, " elapsed =", cpu_s, "s")

    # ─── GPU run ─────────────────────────────────────────────────────────
    seed(42)
    print("\n[2/2] GPU run  →", GPU_CSV)
    with DeviceContext() as ctx:
        var gpu_logger = CsvLogger(file_path=GPU_CSV, buffer_size=64)
        var gpu_logger_ptr = UnsafePointer(to=gpu_logger)
        var gpu_agent = SACAgent[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
            ActorNet, CriticNet,
        ](
            ctx=ctx,
            actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
            gamma=0.99, tau=0.005, action_scale=1.0, init_alpha=0.2,
            target_entropy=-3.0, learning_starts=WARMUP,
            window_size=100, initial_episode_fill=0.0,
        )
        var gpu_env = HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False]()
        var t1 = perf_counter_ns()
        _ = gpu_agent.train_single[
            HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False], L=CsvLogger,
        ](
            gpu_env, NUM_STEPS,
            print_every=PRINT_EVERY, verbose=True,
            logger=gpu_logger_ptr, diag_every=DIAG_EVERY,
        )
        var gpu_s = Float64(perf_counter_ns() - t1) / 1e9
        gpu_logger.close()
        _ = gpu_logger
        var gpu_ret = gpu_agent.mean_return()
        print("  GPU done:  mean_return =", gpu_ret, " elapsed =", gpu_s, "s")

        print("\n" + "=" * 72)
        print("SAC PARITY (final mean episode return):")
        print("  CPU =", cpu_ret, "  GPU =", gpu_ret)
        print("-" * 72)
        print("TRACK   → SAC GPU blocks OK; MBPO split is synthetic-DATA driven.")
        print("DIVERGE → shared SAC GPU-block bug (device-α / log-prob / target).")
        print("CSVs:", CPU_CSV, "|", GPU_CSV)
        print("=" * 72)
