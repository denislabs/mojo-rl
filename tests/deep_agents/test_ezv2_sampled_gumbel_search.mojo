"""Structural tests for the CPU sampled-Gumbel MCTS (Phase 3.2.2).

Mirrors the discrete `test_ezv2_gumbel_search.mojo` invariants on a fresh
untrained network triple. There is no training here — we only verify that
the sampled-search algorithm:

    (a) runs to completion without crashing or overflowing the tree,
    (b) visit budget is consumed at the root (Σ visit_count = NUM_SIMULATIONS),
    (c) the chosen action vector lies inside (−MAX_ACTION, MAX_ACTION) per dim,
    (d) the visit distribution is a valid probability distribution,
    (e) loss/value head outputs are finite (no NaN),
    (f) the search is deterministic given a fixed RNG seed,
    (g) some tree expansion happens (tree size > 1).

Convergence behaviour against a known-Gaussian fixture is deferred — the
GPU-agreement test (3.2.4) is the next gate after this.
"""

from std.random import seed
from std.math import abs
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import (
    Linear,
    LinearReLU,
    Sequential,
)
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.training import NetworkState
from mojo_rl.deep_agents.efficient_zero_v2.mcts_sampled import (
    SampledGumbelMCTS,
)


def main() raises:
    print("=== EZ-V2 sampled-Gumbel CPU search — structural tests ===")
    var passed = 0
    var total = 0

    comptime OBS = 4
    comptime ACT_DIM = 2
    comptime LATENT = 32
    comptime HIDDEN = 32
    comptime BINS = 21
    comptime SIMS = 16
    comptime K_ROOT = 8
    comptime K_NON_ROOT = 4
    comptime NODES = 64
    comptime MAX_ACTION = 1.0
    comptime MIN_STD = 0.1
    comptime STD_MAGNIFICATION = 3.0

    # Networks built by hand — no continuous config struct yet.
    comptime RepModel = Sequential[
        LinearReLU[OBS, HIDDEN], Linear[HIDDEN, LATENT]
    ]
    comptime DynModel = Sequential[
        LinearReLU[LATENT + ACT_DIM, HIDDEN],
        Linear[HIDDEN, LATENT + BINS],
    ]
    comptime PredModel = Sequential[
        LinearReLU[LATENT, HIDDEN],
        Linear[HIDDEN, 2 * ACT_DIM + BINS],
    ]
    comptime Opt = Adam[]

    seed(42)
    var rep_state = NetworkState[RepModel, Opt]()
    rep_state.initialize()
    var dyn_state = NetworkState[DynModel, Opt]()
    dyn_state.initialize()
    var pred_state = NetworkState[PredModel, Opt]()
    pred_state.initialize()

    var obs = List[Scalar[dtype]](capacity=OBS)
    for i in range(OBS):
        obs.append(Scalar[dtype](0.1 * Float64(i + 1)))

    # ── Test 1: run search ──────────────────────────────────────────────
    var mcts = SampledGumbelMCTS[
        ACT_DIM=ACT_DIM,
        LATENT_DIM=LATENT,
        NUM_BINS=BINS,
        NUM_SIMULATIONS=SIMS,
        K_ROOT=K_ROOT,
        K_NON_ROOT=K_NON_ROOT,
        MAX_NODES=NODES,
        MAX_ACTION=MAX_ACTION,
        MIN_STD=MIN_STD,
        STD_MAGNIFICATION=STD_MAGNIFICATION,
    ](gamma=0.99)
    var result = mcts.search(
        obs, rep_state, dyn_state, pred_state, -10.0, 10.0, False
    )
    var chosen = result[0]
    var visits = result[1]
    var root_value = result[2]

    var root_visits = 0
    for i in range(K_ROOT):
        root_visits += mcts.nodes[0].visit_count[i]
    var tree_size = len(mcts.nodes)
    print("    root_visits =", root_visits)
    print("    tree_size   =", tree_size)
    print("    chosen      =", chosen[0], chosen[1])
    print("    root_value  =", root_value)

    total += 1
    if root_visits == SIMS:
        print("PASS: simulation budget consumed exactly (", SIMS, ")")
        passed += 1
    else:
        print("FAIL: budget mismatch — root visits =", root_visits)

    total += 1
    var chosen_in_range = True
    for d in range(ACT_DIM):
        if chosen[d] >= MAX_ACTION or chosen[d] <= -MAX_ACTION:
            chosen_in_range = False
    if chosen_in_range:
        print("PASS: chosen action in (-MAX, MAX) per dim")
        passed += 1
    else:
        print("FAIL: chosen action out of range")

    total += 1
    var sum_v = 0.0
    var min_v = 1e18
    var max_v = -1e18
    for i in range(K_ROOT):
        sum_v += visits[i]
        if visits[i] < min_v:
            min_v = visits[i]
        if visits[i] > max_v:
            max_v = visits[i]
    if sum_v > 0.999 and sum_v < 1.001 and min_v >= -1e-9 and max_v <= 1.0 + 1e-9:
        print("PASS: visit distribution is a valid probability distribution (sum=", sum_v, ")")
        passed += 1
    else:
        print(
            "FAIL: bad visit distribution — sum =", sum_v,
            "min =", min_v, "max =", max_v,
        )

    total += 1
    var any_nan = False
    if root_value != root_value:
        any_nan = True
    for d in range(ACT_DIM):
        if chosen[d] != chosen[d]:
            any_nan = True
    if not any_nan:
        print("PASS: chosen action + root value are finite")
        passed += 1
    else:
        print("FAIL: NaN detected in chosen action or root value")

    total += 1
    if tree_size > 1 and tree_size <= NODES:
        print("PASS: tree expanded (size=", tree_size, ")")
        passed += 1
    else:
        print("FAIL: tree size out of expected range")

    # ── Test 2: determinism wrt RNG seed ────────────────────────────────
    seed(7)
    var mcts2 = SampledGumbelMCTS[
        ACT_DIM=ACT_DIM,
        LATENT_DIM=LATENT,
        NUM_BINS=BINS,
        NUM_SIMULATIONS=SIMS,
        K_ROOT=K_ROOT,
        K_NON_ROOT=K_NON_ROOT,
        MAX_NODES=NODES,
        MAX_ACTION=MAX_ACTION,
        MIN_STD=MIN_STD,
        STD_MAGNIFICATION=STD_MAGNIFICATION,
    ](gamma=0.99)
    var r2 = mcts2.search(
        obs, rep_state, dyn_state, pred_state, -10.0, 10.0, True
    )
    seed(7)
    var mcts3 = SampledGumbelMCTS[
        ACT_DIM=ACT_DIM,
        LATENT_DIM=LATENT,
        NUM_BINS=BINS,
        NUM_SIMULATIONS=SIMS,
        K_ROOT=K_ROOT,
        K_NON_ROOT=K_NON_ROOT,
        MAX_NODES=NODES,
        MAX_ACTION=MAX_ACTION,
        MIN_STD=MIN_STD,
        STD_MAGNIFICATION=STD_MAGNIFICATION,
    ](gamma=0.99)
    var r3 = mcts3.search(
        obs, rep_state, dyn_state, pred_state, -10.0, 10.0, True
    )
    var max_action_diff = 0.0
    for d in range(ACT_DIM):
        var df = r2[0][d] - r3[0][d]
        if df < 0:
            df = -df
        if df > max_action_diff:
            max_action_diff = df
    var max_visit_diff = 0.0
    for i in range(K_ROOT):
        var df = r2[1][i] - r3[1][i]
        if df < 0:
            df = -df
        if df > max_visit_diff:
            max_visit_diff = df

    total += 1
    if max_action_diff < 1e-9 and max_visit_diff < 1e-9:
        print("PASS: search deterministic under fixed RNG seed")
        passed += 1
    else:
        print(
            "FAIL: non-deterministic — action diff =", max_action_diff,
            " visit diff =", max_visit_diff,
        )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
