"""Phase-1 A/B test: Gumbel search vs MuZero PUCT on a TicTacToe-sized
9-action MLP network.

We're not training here — both algorithms run against the *same* freshly
initialized representation/dynamics/prediction triple. The point is to verify
that:

  (1) Gumbel runs end-to-end on a non-trivial action space (9 actions, larger
      than the bandit smoke test) without crashing.
  (2) Gumbel expands ≤ K root actions while PUCT expands all 9, confirming
      the "K-only fan-out" property of Gumbel-Top-k sampling.
  (3) Gumbel respects a TTT-style legal mask (occupied cells masked).
  (4) Both algorithms produce valid distributions over legal actions on the
      same starting position.

The paper's "Gumbel n=8 ≥ Sample MCTS n=50" win-rate claim requires a
pre-trained network and self-play arena, which is deferred until Phase 2 land
training.
"""

from std.random import seed
from mojo_rl.deep_agents.muzero.state import MuZeroCPUState
from mojo_rl.deep_agents.muzero.configs import MuZeroConfig, MuZeroTicTacToeConfig
from mojo_rl.deep_agents.muzero.mcts import MCTS
from mojo_rl.deep_agents.efficient_zero_v2.mcts import GumbelMCTS
from mojo_rl.nn.constants import dtype


def _gumbel_search[
    Config: MuZeroConfig,
    SIMS: Int,
    K: Int,
    NODES: Int,
](
    obs: List[Scalar[dtype]],
    state: MuZeroCPUState[Config, _CAP=128],
    legal_mask: List[Bool],
    mut visits_out: InlineArray[Int, Config.action_dim],
) -> InlineArray[Float64, Config.action_dim]:
    var mcts = GumbelMCTS[
        ACTION_DIM=Config.action_dim,
        LATENT_DIM=Config.latent_dim,
        NUM_BINS=Config.num_bins,
        NUM_SIMULATIONS=SIMS,
        NUM_ROOT_CANDIDATES=K,
        MAX_NODES=NODES,
    ](gamma=0.997)
    var policy = mcts.search(
        obs,
        state.representation,
        state.dynamics,
        state.prediction,
        -10.0,
        10.0,
        legal_mask,
    )
    for a in range(Config.action_dim):
        visits_out[a] = mcts.nodes[0].visit_count[a]
    return policy


def _puct_search[
    Config: MuZeroConfig,
    SIMS: Int,
    NODES: Int,
](
    obs: List[Scalar[dtype]],
    state: MuZeroCPUState[Config, _CAP=128],
    legal_mask: List[Bool],
    mut visits_out: InlineArray[Int, Config.action_dim],
) -> InlineArray[Float64, Config.action_dim]:
    var mcts = MCTS[
        Config.action_dim,
        Config.latent_dim,
        Config.num_bins,
        SIMS,
        MAX_NODES=NODES,
    ](gamma=0.997)
    var policy = mcts.search(
        obs,
        state.representation,
        state.dynamics,
        state.prediction,
        -10.0,
        10.0,
        add_noise=False,  # deterministic for A/B comparison
        legal_mask=legal_mask,
    )
    for a in range(Config.action_dim):
        visits_out[a] = mcts.nodes[0].visit_count[a]
    return policy


def main():
    print("=== Phase 1 A/B: Gumbel vs PUCT on a TTT-sized 9-action MLP ===")

    # TTT config: obs=27 (3 planes × 9 cells), 9 actions.
    comptime Config = MuZeroTicTacToeConfig[
        LATENT=64,
        HIDDEN=64,
        BINS=51,
        SIMS=32,
        NODES=128,
    ]
    comptime ACT = Config.action_dim
    comptime SIMS = 32
    comptime K = 8
    comptime NODES = 128

    seed(2026)
    var state = MuZeroCPUState[Config, _CAP=128]()

    # Empty TTT board: planes 0/1 = own/opp pieces (all zero), plane 2 = legal
    # (all 1.0) — for full-board legality.
    var obs = List[Scalar[dtype]](capacity=Config.obs_dim)
    for _ in range(18):
        obs.append(Scalar[dtype](0.0))
    for _ in range(9):
        obs.append(Scalar[dtype](1.0))

    var legal_full = List[Bool](capacity=ACT)
    for _ in range(ACT):
        legal_full.append(True)

    # Partial board: corner-occupied scenario, 8 legal actions.
    var legal_partial = List[Bool](capacity=ACT)
    for a in range(ACT):
        legal_partial.append(a != 0)  # cell 0 occupied

    var passed = 0
    var total = 0

    # ── Run on empty board ───────────────────────────────────────────────
    print()
    print("--- Empty board (9 legal actions) ---")
    seed(42)
    var gv = InlineArray[Int, ACT](uninitialized=True)
    for a in range(ACT):
        gv[a] = 0
    var g_policy = _gumbel_search[Config, SIMS, K, NODES](
        obs, state, legal_full, gv
    )

    seed(42)
    var pv = InlineArray[Int, ACT](uninitialized=True)
    for a in range(ACT):
        pv[a] = 0
    var p_policy = _puct_search[Config, SIMS, NODES](
        obs, state, legal_full, pv
    )

    print("  Gumbel visits:")
    for a in range(ACT):
        print(
            "    a=", a, "v=", gv[a], "p=", g_policy[a]
        )
    print("  PUCT visits:")
    for a in range(ACT):
        print(
            "    a=", a, "v=", pv[a], "p=", p_policy[a]
        )

    # Check (2): Gumbel never visits more than K distinct root actions.
    var g_distinct = 0
    for a in range(ACT):
        if gv[a] > 0:
            g_distinct += 1

    total += 1
    if g_distinct <= K:
        print("PASS: Gumbel expanded ≤ K =", K, "actions (got", g_distinct, ")")
        passed += 1
    else:
        print("FAIL: Gumbel expanded more than K — got", g_distinct)

    # Check (4): both produce valid distributions.
    var g_sum = Float64(0.0)
    var p_sum = Float64(0.0)
    for a in range(ACT):
        g_sum += g_policy[a]
        p_sum += p_policy[a]
    total += 1
    if g_sum > 0.999 and g_sum < 1.001:
        print("PASS: Gumbel policy sums to 1 (sum=", g_sum, ")")
        passed += 1
    else:
        print("FAIL: Gumbel sum=", g_sum)
    total += 1
    if p_sum > 0.999 and p_sum < 1.001:
        print("PASS: PUCT policy sums to 1 (sum=", p_sum, ")")
        passed += 1
    else:
        print("FAIL: PUCT sum=", p_sum)

    # ── K=4 (strict K < ACT) — confirms Gumbel's K-only fan-out ──────────
    print()
    print("--- K=4 strictly < ACT=9 ---")
    seed(99)
    var gv_k4 = InlineArray[Int, ACT](uninitialized=True)
    for a in range(ACT):
        gv_k4[a] = 0
    var _ = _gumbel_search[Config, SIMS, 4, NODES](
        obs, state, legal_full, gv_k4
    )
    var k4_distinct = 0
    var k4_zero = 0
    for a in range(ACT):
        if gv_k4[a] > 0:
            k4_distinct += 1
        else:
            k4_zero += 1
    print("  K=4 distinct visited =", k4_distinct, " zero-visit =", k4_zero)
    total += 1
    if k4_distinct <= 4 and k4_zero >= ACT - 4:
        print(
            "PASS: K=4 → Gumbel only visits 4 actions; ", k4_zero, "stayed at 0"
        )
        passed += 1
    else:
        print(
            "FAIL: K=4 fan-out broken — distinct=",
            k4_distinct,
            " zero-visit=",
            k4_zero,
        )

    # ── Run on partial board (cell 0 occupied) ───────────────────────────
    print()
    print("--- Partial board (8 legal, cell 0 occupied) ---")
    seed(99)
    var gv2 = InlineArray[Int, ACT](uninitialized=True)
    for a in range(ACT):
        gv2[a] = 0
    var g_policy2 = _gumbel_search[Config, SIMS, K, NODES](
        obs, state, legal_partial, gv2
    )

    print("  Gumbel visits w/ illegal mask:")
    var g_illegal_visits = gv2[0]
    for a in range(ACT):
        print("    a=", a, "v=", gv2[a], "p=", g_policy2[a])

    total += 1
    if g_illegal_visits == 0 and g_policy2[0] < 1e-6:
        print("PASS: Gumbel never visits illegal cell 0 and gives it 0 mass")
        passed += 1
    else:
        print(
            "FAIL: illegal-cell leak — visits=",
            g_illegal_visits,
            " p=",
            g_policy2[0],
        )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
