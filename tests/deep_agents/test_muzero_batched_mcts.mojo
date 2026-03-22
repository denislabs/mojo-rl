"""Test batched MCTS vs unbatched — correctness and speedup."""

from std.time import perf_counter_ns
from mojo_rl.deep_agents.muzero.state import MuZeroCPUState
from mojo_rl.deep_agents.muzero.mcts import MCTS
from mojo_rl.nn.constants import dtype


fn main():
    print("=== Batched MCTS Test ===")

    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 64
    comptime BINS = 21
    comptime SIMS = 50  # More simulations to see batching benefit

    comptime StateType = MuZeroCPUState[
        OBS, ACT, LATENT_DIM=LATENT, HIDDEN_DIM=64, NUM_BINS=BINS
    ]

    var state = StateType()

    # Build observation
    var obs = List[Scalar[dtype]](capacity=OBS)
    for i in range(OBS):
        obs.append(Scalar[dtype](0.1 * Float64(i + 1)))

    # ── Test 1: Unbatched (original) ──────────────────────────────────
    var mcts_unbatched = MCTS[ACT, LATENT, BINS, SIMS](gamma=0.99)

    var t0 = perf_counter_ns()
    comptime NUM_RUNS = 10
    for _ in range(NUM_RUNS):
        _ = mcts_unbatched.search[
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

    print("Unbatched:", NUM_RUNS, "searches x", SIMS, "sims =", unbatched_ms, "ms")
    print("  Per search:", unbatched_ms / Float64(NUM_RUNS), "ms")

    # ── Test 2: Batched (BATCH_SIMS=8) ────────────────────────────────
    var mcts_batched = MCTS[ACT, LATENT, BINS, SIMS](gamma=0.99)

    var t2 = perf_counter_ns()
    for _ in range(NUM_RUNS):
        _ = mcts_batched.search_batched[
            StateType.RepModel,
            StateType.DynModel,
            StateType.PredModel,
            StateType.OptType,
            StateType.OptType,
            StateType.OptType,
            8,  # BATCH_SIMS
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
    var batched_ms = Float64(t3 - t2) / 1e6

    print("Batched (B=8):", NUM_RUNS, "searches x", SIMS, "sims =", batched_ms, "ms")
    print("  Per search:", batched_ms / Float64(NUM_RUNS), "ms")

    var speedup = unbatched_ms / batched_ms
    print("Speedup:", speedup, "x")

    # ── Correctness check ─────────────────────────────────────────────
    var policy_u = mcts_unbatched.search[
        StateType.RepModel,
        StateType.DynModel,
        StateType.PredModel,
        StateType.OptType,
        StateType.OptType,
        StateType.OptType,
    ](obs, state.representation, state.dynamics, state.prediction, -10.0, 10.0, add_noise=False)

    var policy_b = mcts_batched.search_batched[
        StateType.RepModel,
        StateType.DynModel,
        StateType.PredModel,
        StateType.OptType,
        StateType.OptType,
        StateType.OptType,
        8,
    ](obs, state.representation, state.dynamics, state.prediction, -10.0, 10.0, add_noise=False)

    print("\nUnbatched policy:", policy_u[0], policy_u[1])
    print("Batched policy:  ", policy_b[0], policy_b[1])

    # Both should produce valid distributions
    var sum_u = Float64(0.0)
    var sum_b = Float64(0.0)
    for a in range(ACT):
        sum_u += policy_u[a]
        sum_b += policy_b[a]

    if sum_u > 0.99 and sum_u < 1.01:
        print("PASS: unbatched policy valid")
    else:
        print("FAIL: unbatched sum =", sum_u)

    if sum_b > 0.99 and sum_b < 1.01:
        print("PASS: batched policy valid")
    else:
        print("FAIL: batched sum =", sum_b)

    # Node counts should be reasonable
    print("Unbatched nodes:", len(mcts_unbatched.nodes))
    print("Batched nodes:", len(mcts_batched.nodes))

    print("=== Done ===")
