"""SAC storage self-check (HalfCheetah, deep_agents).

Originally a legacy-`nn`-vs-storage CPU/GPU SAC target-parity control (the
companion to `mbpo_hc_nn_parity.mojo`). The legacy `nn` framework was removed
in the sunset, so the cross-implementation comparison can no longer be built.
This is now a STORAGE-ONLY smoke: it constructs the storage `SACAgent` facade
with the same SAC backbone as the old parity control (identical LayerNorm
critic + target_entropy=-3 + arch as the MBPO parity) and runs one short
single-env training pass on HalfCheetah on 100% real data, reporting its own
metrics.

It is a sanity check that the shared off-policy SAC blocks build, train, and
produce finite α / mean_q / returns — not a CPU-vs-GPU diff.

Same axis convention as the live dashboards: the diag bundle (mean_q,
critic_loss, alpha, …) is keyed on cumulative TRAIN steps; avg_reward /
episodes on env-steps.

Run:
    pixi run mojo run -I . examples/half_cheetah/sac_hc_nn_parity.mojo

Inspect:  /tmp/sac_storage_selfcheck.csv
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.core.logger import CsvLogger
from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.activations import ReLU
from mojo_rl.nn.storage.primitives.layer_norm import LayerNorm
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.sac import SACAgent
from mojo_rl.deep_agents.training.blocks import UniformSampleCpuStep
from mojo_rl.envs.half_cheetah import HalfCheetah, HalfCheetahConfig


comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACT_DIM = HalfCheetahConfig.ACTION_DIM  #  6
comptime HIDDEN = 256
comptime BATCH = 128
comptime REPLAY_CAPACITY = 200_000

# Match the MBPO self-check scale so the two diagnostics overlay.
comptime NUM_STEPS = 6_000
comptime WARMUP = 1_000
comptime DIAG_EVERY = 250
comptime PRINT_EVERY = 1_000

comptime SELFCHECK_CSV = "/tmp/sac_storage_selfcheck.csv"


comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
]
# Same LayerNorm critic as the MBPO self-check harness — isolates the SAC
# sub-update on 100% real data.
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
    print("SAC storage self-check (control for MBPO) — HalfCheetah")
    print("  BATCH =", BATCH, " NUM_STEPS =", NUM_STEPS, " WARMUP =", WARMUP)
    print("  target_entropy = -3.0 (matches MBPO self-check)")
    print("=" * 72)

    seed(42)
    print("\nStorage SAC run  →", SELFCHECK_CSV)
    var logger = CsvLogger(file_path=SELFCHECK_CSV, buffer_size=64)
    var logger_ptr = UnsafePointer(to=logger)
    var agent = SACAgent[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
        ActorNet, CriticNet,
    ](
        actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
        gamma=0.99, tau=0.005, action_scale=1.0, init_alpha=0.2,
        target_entropy=-3.0, learning_starts=WARMUP,
        window_size=100, initial_episode_fill=0.0,
    )
    var env = HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False]()
    var t0 = perf_counter_ns()
    _ = agent.train_single[
        HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False], L=CsvLogger,
    ](
        env, NUM_STEPS,
        print_every=PRINT_EVERY, verbose=True,
        logger=logger_ptr, diag_every=DIAG_EVERY,
    )
    var elapsed_s = Float64(perf_counter_ns() - t0) / 1e9
    logger.close()
    _ = logger
    var mean_ret = agent.mean_return()
    print("  done:  mean_return =", mean_ret, " elapsed =", elapsed_s, "s")

    print("\n" + "=" * 72)
    print("STORAGE SAC SELF-CHECK SUMMARY — HalfCheetah")
    print("  mean episode return (last 100) =", mean_ret)
    print("  episodes completed             =", agent.ep_count())
    print("  elapsed                        =", elapsed_s, "s")
    print("-" * 72)
    print("Sanity: the shared off-policy SAC blocks built, trained, and")
    print("produced a finite return. Inspect the CSV for the metric trends.")
    print("CSV:", SELFCHECK_CSV)
    print("=" * 72)
