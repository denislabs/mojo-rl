"""Benchmark batched MCTS with larger networks where speedup is more significant."""

from std.time import perf_counter_ns
from mojo_rl.deep_agents.muzero.state import MuZeroCPUState
from mojo_rl.deep_agents.muzero.mcts import MCTS
from mojo_rl.nn.constants import dtype


def main():
    print("=== Batched MCTS Benchmark (Large Networks) ===")

    # Larger network: 256-dim latent, 256-dim hidden (realistic MuZero size)
    comptime OBS = 8
    comptime ACT = 4
    comptime LATENT = 256
    comptime BINS = 51
    comptime SIMS = 50

    comptime StateType = MuZeroCPUState[
        OBS, ACT, LATENT_DIM=LATENT, HIDDEN_DIM=256, NUM_BINS=BINS
    ]

    print("Network: LATENT=256, HIDDEN=256, BINS=51, ACT=4")
    print("RepModel params:", StateType.RepModel.PARAM_SIZE)
    print("DynModel params:", StateType.DynModel.PARAM_SIZE)
    print("PredModel params:", StateType.PredModel.PARAM_SIZE)

    var state = StateType()

    var obs = List[Scalar[dtype]](capacity=OBS)
    for i in range(OBS):
        obs.append(Scalar[dtype](0.1 * Float64(i + 1)))

    comptime NUM_RUNS = 5

    # Unbatched
    var mcts1 = MCTS[ACT, LATENT, BINS, SIMS](gamma=0.99)
    var t0 = perf_counter_ns()
    for _ in range(NUM_RUNS):
        _ = mcts1.search[
            StateType.RepModel,
            StateType.DynModel,
            StateType.PredModel,
            StateType.OptType,
            StateType.OptType,
            StateType.OptType,
        ](
            obs,
            state.representation,
            state.dynamics,
            state.prediction,
            -10.0,
            10.0,
            add_noise=False,
        )
    var t1 = perf_counter_ns()
    var unbatched_ms = Float64(t1 - t0) / 1e6

    # Batched B=8
    var mcts2 = MCTS[ACT, LATENT, BINS, SIMS](gamma=0.99)
    var t2 = perf_counter_ns()
    for _ in range(NUM_RUNS):
        _ = mcts2.search_batched[
            StateType.RepModel,
            StateType.DynModel,
            StateType.PredModel,
            StateType.OptType,
            StateType.OptType,
            StateType.OptType,
            8,
        ](
            obs,
            state.representation,
            state.dynamics,
            state.prediction,
            -10.0,
            10.0,
            add_noise=False,
        )
    var t3 = perf_counter_ns()
    var batched8_ms = Float64(t3 - t2) / 1e6

    # Batched B=16
    var mcts3 = MCTS[ACT, LATENT, BINS, SIMS](gamma=0.99)
    var t4 = perf_counter_ns()
    for _ in range(NUM_RUNS):
        _ = mcts3.search_batched[
            StateType.RepModel,
            StateType.DynModel,
            StateType.PredModel,
            StateType.OptType,
            StateType.OptType,
            StateType.OptType,
            16,
        ](
            obs,
            state.representation,
            state.dynamics,
            state.prediction,
            -10.0,
            10.0,
            add_noise=False,
        )
    var t5 = perf_counter_ns()
    var batched16_ms = Float64(t5 - t4) / 1e6

    print("\nResults (", NUM_RUNS, "runs x", SIMS, "sims):")
    print(
        "  Unbatched: ",
        unbatched_ms,
        "ms (",
        unbatched_ms / Float64(NUM_RUNS),
        "ms/search)",
    )
    print(
        "  Batched B=8: ",
        batched8_ms,
        "ms (",
        batched8_ms / Float64(NUM_RUNS),
        "ms/search) →",
        unbatched_ms / batched8_ms,
        "x",
    )
    print(
        "  Batched B=16:",
        batched16_ms,
        "ms (",
        batched16_ms / Float64(NUM_RUNS),
        "ms/search) →",
        unbatched_ms / batched16_ms,
        "x",
    )

    print("=== Done ===")
