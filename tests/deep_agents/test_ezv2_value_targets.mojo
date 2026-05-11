"""Phase-2 Step 1 unit tests — value target + policy loss formulas.

Direct numerical fixtures for the EZ-V2 strategy traits and their helper
functions. The expected values are derived by hand-evaluating the formulas
in the paper / docstrings.

Coverage:
    1. compute_sve — visit-weighted root mean, including degenerate cases.
    2. compute_multistep_td — non-terminal, terminal, and short-trajectory
       (replay-window edge-effect) cases.
    3. SVETarget / MultiStepTDTarget compute() return the right input.
    4. MixedValueTarget — boundary + midpoint behaviour, including the
       symmetric "swapped thresholds" case.
    5. FullCrossEntropy — matches hand-computed log-softmax expectation.
    6. SimpleBestAction — equals −log_softmax(logits)[a*].
    7. Integration: run GumbelMCTS on a small fixture and verify the
       output of compute_sve(Σtotal_value, Σvisits) matches the
       visit-weighted Q at the root, end-to-end.
"""

from std.math import exp, log
from std.random import seed
from mojo_rl.deep_agents.muzero.state import MuZeroCPUState
from mojo_rl.deep_agents.muzero.configs import MuZeroConfig, MuZeroMLPConfig
from mojo_rl.deep_agents.efficient_zero_v2 import (
    SVETarget,
    MultiStepTDTarget,
    MixedValueTarget,
    FullCrossEntropy,
    SimpleBestAction,
    compute_sve,
    compute_multistep_td,
    GumbelMCTS,
)
from mojo_rl.nn.constants import dtype


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _close(actual: Float64, expected: Float64, tol: Float64 = 1e-9) -> Bool:
    return _abs(actual - expected) < tol


def _expect(
    cond: Bool,
    label: String,
    mut passed: Int,
    mut total: Int,
):
    total += 1
    if cond:
        print("PASS:", label)
        passed += 1
    else:
        print("FAIL:", label)


def _run_mcts_for_sve[
    Config: MuZeroConfig,
    SIMS: Int,
    K: Int,
    NODES: Int,
](
    obs: List[Scalar[dtype]],
    state: MuZeroCPUState[Config, _CAP=128],
    mut total_value_sum_out: Float64,
    mut total_visits_out: Int,
    mut visit_weighted_q_out: Float64,
):
    """Run CPU GumbelMCTS and read out (Σtotal_value, Σvisits, mean Q)."""
    var mcts = GumbelMCTS[
        ACTION_DIM=Config.action_dim,
        LATENT_DIM=Config.latent_dim,
        NUM_BINS=Config.num_bins,
        NUM_SIMULATIONS=SIMS,
        NUM_ROOT_CANDIDATES=K,
        MAX_NODES=NODES,
    ](gamma=0.997)
    _ = mcts.search(
        obs,
        state.representation,
        state.dynamics,
        state.prediction,
        -10.0,
        10.0,
        List[Bool](),
    )
    var sum_v = Float64(0.0)
    var sum_n = 0
    var weighted = Float64(0.0)
    for a in range(Config.action_dim):
        var v = mcts.nodes[0].total_value[a]
        var n = mcts.nodes[0].visit_count[a]
        sum_v += v
        sum_n += n
        if n > 0:
            weighted += Float64(n) * (v / Float64(n))  # = v
    total_value_sum_out = sum_v
    total_visits_out = sum_n
    if sum_n > 0:
        visit_weighted_q_out = weighted / Float64(sum_n)
    else:
        visit_weighted_q_out = Float64(0.0)


def main():
    print("=== EZ-V2 Phase 2 / Step 1 — value target + policy loss tests ===")
    var passed = 0
    var total = 0

    # ── 1. compute_sve ───────────────────────────────────────────────────
    print()
    print("--- compute_sve ---")
    # 4 actions, visits=[3, 5, 1, 7], total_value=[3, 10, -2, 7]
    # Σ visits = 16, Σ value = 18 → SVE = 1.125.
    _expect(
        _close(compute_sve(18.0, 16), 1.125),
        "SVE(18.0 / 16) = 1.125",
        passed,
        total,
    )
    _expect(
        compute_sve(0.0, 0) == 0.0,
        "SVE returns 0 when no visits",
        passed,
        total,
    )
    _expect(
        _close(compute_sve(-3.5, 7), -0.5),
        "SVE handles negative returns",
        passed,
        total,
    )

    # ── 2. compute_multistep_td ─────────────────────────────────────────
    print()
    print("--- compute_multistep_td ---")
    var rewards3 = InlineArray[Float64, 3](uninitialized=True)
    rewards3[0] = 1.0
    rewards3[1] = 2.0
    rewards3[2] = 3.0
    # Non-terminal full window:
    #   1.0 + 0.5·2.0 + 0.25·3.0 + 0.125·4.0 = 1+1+0.75+0.5 = 3.25
    _expect(
        _close(
            compute_multistep_td[3](rewards3, 3, False, 0.5, 4.0),
            3.25,
        ),
        "n-step TD: full window with bootstrap",
        passed,
        total,
    )
    # Terminal full window (no bootstrap):
    #   1+1+0.75 = 2.75
    _expect(
        _close(
            compute_multistep_td[3](rewards3, 3, True, 0.5, 4.0),
            2.75,
        ),
        "n-step TD: terminal drops bootstrap",
        passed,
        total,
    )
    # Replay-edge effect: only 2 of the 3 reward slots are real; bootstrap
    # is still applied at γ^2 = 0.25:
    #   1 + 0.5·2 + 0.25·4 = 1 + 1 + 1 = 3
    _expect(
        _close(
            compute_multistep_td[3](rewards3, 2, False, 0.5, 4.0),
            3.0,
        ),
        "n-step TD: short trajectory keeps bootstrap at γ^valid",
        passed,
        total,
    )
    # gamma=0 collapses to immediate reward:
    _expect(
        _close(
            compute_multistep_td[3](rewards3, 3, False, 0.0, 99.0),
            1.0,
        ),
        "n-step TD: gamma=0 collapses to r_0",
        passed,
        total,
    )

    # ── 3. SVETarget / MultiStepTDTarget select correctly ───────────────
    print()
    print("--- SVETarget / MultiStepTDTarget compute() ---")
    _expect(
        SVETarget.compute(2.0, 5.0, 0) == 2.0,
        "SVETarget.compute returns sve",
        passed,
        total,
    )
    _expect(
        MultiStepTDTarget.compute(2.0, 5.0, 0) == 5.0,
        "MultiStepTDTarget.compute returns td",
        passed,
        total,
    )

    # ── 4. MixedValueTarget — boundaries + midpoint ─────────────────────
    print()
    print("--- MixedValueTarget[10, 20] ---")
    # age ≤ 10  → SVE
    _expect(
        MixedValueTarget[10, 20].compute(2.0, 5.0, 5) == 2.0,
        "age below T_FRESH → pure SVE",
        passed,
        total,
    )
    _expect(
        MixedValueTarget[10, 20].compute(2.0, 5.0, 10) == 2.0,
        "age = T_FRESH → SVE (lower boundary inclusive)",
        passed,
        total,
    )
    # age ≥ 20  → TD
    _expect(
        MixedValueTarget[10, 20].compute(2.0, 5.0, 25) == 5.0,
        "age above T_STALE → pure TD",
        passed,
        total,
    )
    _expect(
        MixedValueTarget[10, 20].compute(2.0, 5.0, 20) == 5.0,
        "age = T_STALE → TD (upper boundary inclusive)",
        passed,
        total,
    )
    # midpoint
    # blend = (15 - 10) / (20 - 10) = 0.5
    # 0.5 · 2 + 0.5 · 5 = 3.5
    _expect(
        _close(MixedValueTarget[10, 20].compute(2.0, 5.0, 15), 3.5),
        "midpoint blends 50/50",
        passed,
        total,
    )
    # quarter point
    _expect(
        _close(MixedValueTarget[10, 20].compute(2.0, 5.0, 12), 2.6),
        "age=12: 0.8·sve + 0.2·td = 2.6",
        passed,
        total,
    )
    # Swapped thresholds — algorithm normalizes internally, midpoint same.
    _expect(
        _close(MixedValueTarget[20, 10].compute(2.0, 5.0, 15), 3.5),
        "swapped thresholds give symmetric midpoint",
        passed,
        total,
    )

    # ── 5. FullCrossEntropy ─────────────────────────────────────────────
    print()
    print("--- FullCrossEntropy ---")
    # logits = [1, 2, 3, 4]; max=4
    # exp(x-max) = [e^-3, e^-2, e^-1, 1]  ≈ [0.04979, 0.13534, 0.36788, 1.0]
    # sum_e = 1.55301
    # log_sum = log(1.55301) + 4 ≈ 0.44019 + 4 = 4.44019
    # log_softmax(logits) = [-3.44019, -2.44019, -1.44019, -0.44019]
    # target = [0, 0.5, 0.5, 0]
    # loss = -(0.5·-2.44019 + 0.5·-1.44019) = 1.94019
    var logits4 = InlineArray[Float64, 4](uninitialized=True)
    logits4[0] = 1.0
    logits4[1] = 2.0
    logits4[2] = 3.0
    logits4[3] = 4.0
    var target4 = InlineArray[Float64, 4](uninitialized=True)
    target4[0] = 0.0
    target4[1] = 0.5
    target4[2] = 0.5
    target4[3] = 0.0
    _expect(
        _close(
            FullCrossEntropy.compute[4](logits4, target4, 0),
            1.9401896985611953,
            tol=1e-6,
        ),
        "FullCE matches hand-computed value",
        passed,
        total,
    )
    # Sanity: cross-entropy with itself's own softmax = entropy ≥ 0.
    var soft4 = InlineArray[Float64, 4](uninitialized=True)
    var max_l = logits4[0]
    for i in range(1, 4):
        if logits4[i] > max_l:
            max_l = logits4[i]
    var sum_e = Float64(0.0)
    for i in range(4):
        sum_e += exp(logits4[i] - max_l)
    for i in range(4):
        soft4[i] = exp(logits4[i] - max_l) / sum_e
    var ce_self = FullCrossEntropy.compute[4](logits4, soft4, 0)
    # Entropy(softmax([1, 2, 3, 4])):
    #   p ≈ [0.0321, 0.0871, 0.2369, 0.6439]
    #   H = -Σ p log p ≈ 0.9474
    _expect(
        _close(ce_self, 0.9474, tol=1e-3),
        "CE(softmax(x), x) ≈ entropy(softmax([1,2,3,4])) ≈ 0.9474",
        passed,
        total,
    )

    # ── 6. SimpleBestAction ─────────────────────────────────────────────
    print()
    print("--- SimpleBestAction ---")
    # -log_softmax(logits)[2] = -(-1.44019) = 1.44019
    _expect(
        _close(
            SimpleBestAction.compute[4](logits4, target4, 2),
            1.4401896985611953,
            tol=1e-6,
        ),
        "SimpleBestAction = -log_softmax(logits)[a*]",
        passed,
        total,
    )
    # Different action gives different loss; check that argmax-of-logits
    # gives the smallest loss.
    var loss_best = SimpleBestAction.compute[4](logits4, target4, 3)
    var loss_worst = SimpleBestAction.compute[4](logits4, target4, 0)
    _expect(
        loss_best < loss_worst,
        "SimpleBestAction smaller for the highest-logit action",
        passed,
        total,
    )

    # ── 7. Integration: GumbelMCTS root SVE ─────────────────────────────
    print()
    print("--- Integration: GumbelMCTS → compute_sve ---")
    comptime OBS = 4
    comptime ACT = 4
    comptime LATENT = 32
    comptime BINS = 21
    comptime SIMS = 16
    comptime K = 4
    comptime NODES = 64

    comptime Config = MuZeroMLPConfig[
        OBS=OBS,
        ACT=ACT,
        LATENT=LATENT,
        HIDDEN=32,
        BINS=BINS,
        BS=8,
        SIMS=SIMS,
        NODES=NODES,
    ]
    seed(2026)
    var state = MuZeroCPUState[Config, _CAP=128]()

    var obs = List[Scalar[dtype]](capacity=OBS)
    for i in range(OBS):
        obs.append(Scalar[dtype](0.1 * Float64(i + 1)))

    var sum_value = Float64(0.0)
    var sum_visits = 0
    var weighted_q = Float64(0.0)
    _run_mcts_for_sve[Config, SIMS, K, NODES](
        obs, state, sum_value, sum_visits, weighted_q
    )
    var sve_via_helper = compute_sve(sum_value, sum_visits)
    print("    Σtotal_value =", sum_value)
    print("    Σvisits      =", sum_visits)
    print("    SVE (helper) =", sve_via_helper)
    print("    SVE (direct) =", weighted_q)
    _expect(
        _close(sve_via_helper, weighted_q, tol=1e-9),
        "compute_sve(Σtv, Σv) == direct visit-weighted Q at the root",
        passed,
        total,
    )
    _expect(
        sum_visits == SIMS,
        "MCTS budget consumed (sanity check upstream)",
        passed,
        total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
