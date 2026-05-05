"""Phase-1 unit tests for EfficientZero V2 Gumbel-search MCTS.

Verifies the search machinery in isolation against a fresh, untrained MuZero
network triple. There is no training here — only that the algorithm:
    (a) runs to completion without crashing or overflowing the tree,
    (b) returns a valid probability distribution over the action space,
    (c) consumes the simulation budget (root visit count == NUM_SIMULATIONS),
    (d) restricts root expansion to ≤ K sampled candidates,
    (e) honours a legal-action mask,
    (f) is deterministic given a fixed RNG seed.

End-to-end "Gumbel beats PUCT at low simulation count" lives in the separate
TTT A/B test (test_ezv2_gumbel_vs_puct_ttt.mojo).
"""

from std.random import seed
from mojo_rl.deep_agents.muzero.state import MuZeroCPUState
from mojo_rl.deep_agents.muzero.configs import MuZeroConfig, MuZeroMLPConfig
from mojo_rl.deep_agents.efficient_zero_v2.mcts import GumbelMCTS
from mojo_rl.nn.constants import dtype


# Wrapping `search()` in a Config-parameterized helper makes Mojo's
# parameter inference reliable: from inside this fn, `Config.RepModel` is the
# same comptime alias the state's NetworkState was built from, so the trait
# bound on `search`'s `RepModel: Model` resolves cleanly.
def _run_gumbel[
    Config: MuZeroConfig,
    SIMS: Int,
    K: Int,
    NODES: Int,
](
    obs: List[Scalar[dtype]],
    state: MuZeroCPUState[Config, _CAP=128],
    legal_mask: List[Bool] = List[Bool](),
) -> InlineArray[Float64, Config.action_dim]:
    var mcts = GumbelMCTS[
        ACTION_DIM=Config.action_dim,
        LATENT_DIM=Config.latent_dim,
        NUM_BINS=Config.num_bins,
        NUM_SIMULATIONS=SIMS,
        NUM_ROOT_CANDIDATES=K,
        MAX_NODES=NODES,
    ](gamma=0.99)
    var policy = mcts.search(
        obs,
        state.representation,
        state.dynamics,
        state.prediction,
        -10.0,
        10.0,
        legal_mask,
    )
    var root_visits = 0
    var distinct_actions = 0
    var illegal_visits = 0
    for a in range(Config.action_dim):
        var v = mcts.nodes[0].visit_count[a]
        root_visits += v
        if v > 0:
            distinct_actions += 1
        if len(legal_mask) == Config.action_dim and not legal_mask[a]:
            illegal_visits += v
    print("    [helper] root_visits =", root_visits)
    print("    [helper] distinct_actions =", distinct_actions)
    print("    [helper] illegal_visits =", illegal_visits)
    print("    [helper] tree size =", len(mcts.nodes))
    return policy


def _run_gumbel_with_stats[
    Config: MuZeroConfig,
    SIMS: Int,
    K: Int,
    NODES: Int,
](
    obs: List[Scalar[dtype]],
    state: MuZeroCPUState[Config, _CAP=128],
    legal_mask: List[Bool],
    mut root_visits_out: Int,
    mut distinct_actions_out: Int,
    mut illegal_visits_out: Int,
    mut tree_size_out: Int,
) -> InlineArray[Float64, Config.action_dim]:
    var mcts = GumbelMCTS[
        ACTION_DIM=Config.action_dim,
        LATENT_DIM=Config.latent_dim,
        NUM_BINS=Config.num_bins,
        NUM_SIMULATIONS=SIMS,
        NUM_ROOT_CANDIDATES=K,
        MAX_NODES=NODES,
    ](gamma=0.99)
    var policy = mcts.search(
        obs,
        state.representation,
        state.dynamics,
        state.prediction,
        -10.0,
        10.0,
        legal_mask,
    )
    var root_visits = 0
    var distinct_actions = 0
    var illegal_visits = 0
    for a in range(Config.action_dim):
        var v = mcts.nodes[0].visit_count[a]
        root_visits += v
        if v > 0:
            distinct_actions += 1
        if len(legal_mask) == Config.action_dim and not legal_mask[a]:
            illegal_visits += v
    root_visits_out = root_visits
    distinct_actions_out = distinct_actions
    illegal_visits_out = illegal_visits
    tree_size_out = len(mcts.nodes)
    return policy


def main():
    print("=== EfficientZero V2 Gumbel Search — Phase 1 unit tests ===")

    comptime OBS = 4
    comptime ACT = 4
    comptime LATENT = 32
    comptime HIDDEN = 32
    comptime BINS = 21
    comptime SIMS = 16
    comptime K = 4
    comptime NODES = 64

    comptime Config = MuZeroMLPConfig[
        OBS=OBS,
        ACT=ACT,
        LATENT=LATENT,
        HIDDEN=HIDDEN,
        BINS=BINS,
        BS=8,
        SIMS=SIMS,
        NODES=NODES,
    ]

    seed(42)
    var state = MuZeroCPUState[Config, _CAP=128]()

    var obs = List[Scalar[dtype]](capacity=OBS)
    for i in range(OBS):
        obs.append(Scalar[dtype](0.1 * Float64(i + 1)))

    var passed = 0
    var total = 0

    # ── Tests (a)+(b)+(c)+(d) — single search, structural invariants ────
    var rv = 0
    var da = 0
    var iv = 0
    var ts = 0
    var policy = _run_gumbel_with_stats[Config, SIMS, K, NODES](
        obs, state, List[Bool](), rv, da, iv, ts
    )
    var sum_p = Float64(0.0)
    var min_p = Float64(1e18)
    var max_p = Float64(-1e18)
    for a in range(ACT):
        sum_p += policy[a]
        if policy[a] < min_p:
            min_p = policy[a]
        if policy[a] > max_p:
            max_p = policy[a]
        print("  policy[", a, "] =", policy[a])
    print("  Σpolicy =", sum_p)

    total += 1
    if sum_p > 0.999 and sum_p < 1.001:
        print("PASS: improved policy sums to 1")
        passed += 1
    else:
        print("FAIL: improved policy sums to", sum_p)

    total += 1
    if min_p >= -1e-6 and max_p <= 1.0 + 1e-6:
        print("PASS: all probabilities in [0, 1]")
        passed += 1
    else:
        print("FAIL: probability out of range — min=", min_p, "max=", max_p)

    total += 1
    if rv == SIMS:
        print("PASS: simulation budget consumed exactly (", SIMS, ")")
        passed += 1
    else:
        print("FAIL: budget mismatch — root visits =", rv, "expected", SIMS)

    total += 1
    if da <= K:
        print("PASS: root expansion limited to K =", K, "(visited", da, ")")
        passed += 1
    else:
        print("FAIL: more than K root actions visited — got", da)

    # ── Test (e) — legal mask honoured ───────────────────────────────────
    seed(123)
    var legal = List[Bool](capacity=ACT)
    legal.append(True)
    legal.append(False)  # action 1 illegal
    legal.append(True)
    legal.append(True)
    var rv2 = 0
    var da2 = 0
    var iv2 = 0
    var ts2 = 0
    var policy2 = _run_gumbel_with_stats[Config, SIMS, K, NODES](
        obs, state, legal, rv2, da2, iv2, ts2
    )
    print(
        "  legal-mask policy =",
        policy2[0],
        policy2[1],
        policy2[2],
        policy2[3],
    )
    total += 1
    if policy2[1] < 1e-6 and iv2 == 0:
        print("PASS: illegal action gets zero probability and zero visits")
        passed += 1
    else:
        print(
            "FAIL: illegal action leaked — p=",
            policy2[1],
            " visits=",
            iv2,
        )

    # ── Test (f) — determinism wrt RNG seed ──────────────────────────────
    seed(7)
    var p3 = _run_gumbel[Config, SIMS, K, NODES](obs, state, List[Bool]())
    seed(7)
    var p4 = _run_gumbel[Config, SIMS, K, NODES](obs, state, List[Bool]())
    var max_diff = Float64(0.0)
    for a in range(ACT):
        var d = p3[a] - p4[a]
        if d < 0:
            d = -d
        if d > max_diff:
            max_diff = d
    print("  Max |p3 − p4| under same seed =", max_diff)
    total += 1
    if max_diff < 1e-6:
        print("PASS: search is deterministic given fixed RNG seed")
        passed += 1
    else:
        print("FAIL: search is non-deterministic — max diff =", max_diff)

    print("=== Result:", passed, "/", total, "tests passed ===")
