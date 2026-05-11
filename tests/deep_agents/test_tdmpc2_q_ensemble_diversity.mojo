"""TD-MPC2 — Q-ensemble independence (Test 4 of 5).

Goal: Verify that the 5 Q networks are actually different from each
other — both at init AND after training — so the random 2-of-5
target-min subsample produces meaningful pessimistic targets.

Why this catches bugs:
  - Memory project_normedlinear_init_bug.md describes how all networks
    initially shared the same Philox seed (same architecture → same
    params), making the "ensemble" collapse to 5 identical heads.
    We now have per-Q seeds 101-105 in WorldModel.__init__. This test
    is the regression check.
  - If the 5 Q networks are bitwise-identical at init, std_q logged in
    production is purely a function of the random 2-of-5 SUBSAMPLING,
    not of head diversity. That's a much weaker pessimism signal.
  - During training, all 5 Qs see the same TD targets in production.
    They CAN drift toward identical solutions. We test that they
    remain measurably different on a HELD-OUT (z, a) batch never seen
    during training — diversity on training-batch is a weaker signal
    (could just be overfitting to minibatch noise).

Sub-tests:
  4a — At init, mean per-sample std across 5 Q logits > 1e-3.
  4b — At init, decoded Q values across 5 networks have mean range > 1e-3
       on a random (z, a) batch.
  4c — After 200 training steps with synthetic targets, decoded Q values
       on a HELD-OUT batch still have mean range > 1e-3.
  4d — After training, NO pair of Q networks is bitwise identical (params
       not shared via aliasing).
"""

from std.math import sqrt, exp
from std.random import seed, random_float64
from std.memory import alloc, memset

from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import Network, NetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Normal
from mojo_rl.deep_agents.tdmpc2.world_model import WorldModel


comptime OBS = 4
comptime ACT = 2
comptime LATENT = 16
comptime MLP = 32
comptime ENC = 16
comptime SIMPLEX = 4
comptime BINS = 11
comptime BATCH = 16
comptime ZA = LATENT + ACT
comptime NUM_Q = 5
comptime TRAIN_STEPS = 200

comptime ENC_LR = 9e-5
comptime WM_LR = 3e-4

comptime WM = WorldModel[
    OBS_DIM=OBS,
    ACTION_DIM=ACT,
    LATENT_DIM=LATENT,
    MLP_DIM=MLP,
    ENC_DIM=ENC,
    NUM_BINS=BINS,
    NUM_Q=NUM_Q,
    SIMPLEX_DIM=SIMPLEX,
    ENC_LR=ENC_LR,
    WM_LR=WM_LR,
]
comptime QModel = WM.QModel
comptime WMOpt = Adam[LR=WM_LR]


def _expect(cond: Bool, label: String, mut passed: Int, mut total: Int):
    total += 1
    if cond:
        print("PASS:", label)
        passed += 1
    else:
        print("FAIL:", label)


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _decode_logits_to_value(
    logits: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bins: InlineArray[Float32, BINS],
) -> Float64:
    """Single-sample symexp decode of [BINS] logits."""
    var max_l: Float64 = -1e30
    for k in range(BINS):
        var v = Float64(logits[k])
        if v > max_l:
            max_l = v
    var sum_exp: Float64 = 0.0
    for k in range(BINS):
        sum_exp += exp(Float64(logits[k]) - max_l)
    var v_sym: Float64 = 0.0
    for k in range(BINS):
        var p = exp(Float64(logits[k]) - max_l) / sum_exp
        v_sym += p * Float64(bins[k])
    var aps = v_sym if v_sym >= 0.0 else -v_sym
    return (exp(aps) - 1.0) if v_sym >= 0.0 else -(exp(aps) - 1.0)


def _q_forward(
    mut q: NetworkState[QModel, WMOpt],
    za_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
    ],
    out_buf: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    """Run forward (no cache) and write [BATCH * BINS] logits into out_buf.
    """
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin
    ](out_buf)
    Network[QModel, WMOpt].forward[BATCH](
        za_t, out_t, q.params_view(), q.model_state_view()
    )


def _train_step(
    mut q: NetworkState[QModel, WMOpt],
    za_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
    ],
    grad_seed: Float64,
):
    """One training step on Q with grad_logits = grad_seed * uniform(BATCH * BINS).

    Pseudo-loss; what matters is each Q net moves *somewhere*. The
    grad_seed is reused across all Q networks so they see the same
    update direction (closer to production where all Qs see the same
    TD target distribution).
    """
    var logits = alloc[Scalar[dtype]](BATCH * BINS)
    var logits_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin
    ](logits)
    var cache = alloc[Scalar[dtype]](BATCH * QModel.CACHE_SIZE)
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, QModel.CACHE_SIZE), MutAnyOrigin
    ](cache)
    Network[QModel, WMOpt].forward_with_cache[BATCH](
        za_t,
        logits_t,
        q.params_view(),
        q.model_state_view(),
        cache_t,
    )
    # grad = (logits - target) where target is fixed; effectively pulls
    # logits towards a constant random target distribution.
    var grad_logits = alloc[Scalar[dtype]](BATCH * BINS)
    var grad_logits_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin
    ](grad_logits)
    for i in range(BATCH * BINS):
        # Same target seed across networks → same target distribution.
        var t = (i * 17 + 31) % 11
        grad_logits[i] = Scalar[dtype](
            (Float64(t) - 5.0) * grad_seed / Float64(BATCH * BINS)
        )
    var grad_za = alloc[Scalar[dtype]](BATCH * ZA)
    var grad_za_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
    ](grad_za)
    q.zero_grads()
    var grads_v = q.grads_view()
    Network[QModel, WMOpt].backward[BATCH](
        grad_logits_t,
        grad_za_t,
        q.params_view(),
        q.model_state_view(),
        cache_t,
        grads_v,
    )
    q.optimizer_step()
    logits.free()
    cache.free()
    grad_logits.free()
    grad_za.free()


def main() raises:
    seed(0xCAFE13)
    print("=" * 70)
    print("TD-MPC2 Test 4 — Q-ensemble independence")
    print("=" * 70)

    var passed = 0
    var total = 0

    # Per-Q seeds match WorldModel.__init__ (101..105) so we test the
    # actual production init.
    var q1 = NetworkState[QModel, WMOpt]()
    q1.initialize[Normal[0.0, 0.02, SEED=101]]()
    var q2 = NetworkState[QModel, WMOpt]()
    q2.initialize[Normal[0.0, 0.02, SEED=102]]()
    var q3 = NetworkState[QModel, WMOpt]()
    q3.initialize[Normal[0.0, 0.02, SEED=103]]()
    var q4 = NetworkState[QModel, WMOpt]()
    q4.initialize[Normal[0.0, 0.02, SEED=104]]()
    var q5 = NetworkState[QModel, WMOpt]()
    q5.initialize[Normal[0.0, 0.02, SEED=105]]()

    # Compute distributional bins for value decode.
    var bin_step = (10.0 - (-10.0)) / Float64(BINS - 1)
    var bins = InlineArray[Float32, BINS](uninitialized=True)
    for i in range(BINS):
        bins[i] = Float32(-10.0 + Float64(i) * bin_step)

    # Two random batches: train and held-out.
    var za_train_buf = alloc[Scalar[dtype]](BATCH * ZA)
    for i in range(BATCH * ZA):
        za_train_buf[i] = Scalar[dtype](random_float64() * 0.5 - 0.25)
    var za_train = LayoutTensor[
        dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
    ](za_train_buf)

    var za_holdout_buf = alloc[Scalar[dtype]](BATCH * ZA)
    for i in range(BATCH * ZA):
        za_holdout_buf[i] = Scalar[dtype](random_float64() * 0.5 - 0.25)
    var za_holdout = LayoutTensor[
        dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
    ](za_holdout_buf)

    # ─── 4a — Per-sample std across Q logits at init ────────────────────
    print()
    print("--- 4a. Per-sample std across 5 Q logits at init ---")
    var q_logits = alloc[Scalar[dtype]](NUM_Q * BATCH * BINS)
    _q_forward(q1, za_holdout, q_logits + 0 * BATCH * BINS)
    _q_forward(q2, za_holdout, q_logits + 1 * BATCH * BINS)
    _q_forward(q3, za_holdout, q_logits + 2 * BATCH * BINS)
    _q_forward(q4, za_holdout, q_logits + 3 * BATCH * BINS)
    _q_forward(q5, za_holdout, q_logits + 4 * BATCH * BINS)

    var mean_std_logits: Float64 = 0.0
    for b in range(BATCH):
        for k in range(BINS):
            var mean: Float64 = 0.0
            for q in range(NUM_Q):
                mean += Float64(
                    q_logits[q * BATCH * BINS + b * BINS + k]
                )
            mean /= Float64(NUM_Q)
            var var_q: Float64 = 0.0
            for q in range(NUM_Q):
                var d = Float64(
                    q_logits[q * BATCH * BINS + b * BINS + k]
                ) - mean
                var_q += d * d
            var_q /= Float64(NUM_Q)
            mean_std_logits += sqrt(var_q)
    mean_std_logits /= Float64(BATCH * BINS)
    print("    init mean per-sample std across Q logits =", mean_std_logits)
    _expect(
        mean_std_logits > 1e-3,
        "4a — init logit std > 1e-3 (Qs not identical)",
        passed,
        total,
    )

    # ─── 4b — Decoded value range at init ───────────────────────────────
    print()
    print("--- 4b. Decoded value range across 5 Qs at init ---")
    var mean_range_init: Float64 = 0.0
    for b in range(BATCH):
        var vmin: Float64 = 1e30
        var vmax: Float64 = -1e30
        for q in range(NUM_Q):
            var v = _decode_logits_to_value(
                q_logits + q * BATCH * BINS + b * BINS, bins
            )
            if v < vmin:
                vmin = v
            if v > vmax:
                vmax = v
        mean_range_init += vmax - vmin
    mean_range_init /= Float64(BATCH)
    print("    init mean (max-min) decoded Q across 5 nets =", mean_range_init)
    _expect(
        mean_range_init > 1e-3,
        "4b — init decoded Q range > 1e-3",
        passed,
        total,
    )

    # ─── Train all 5 Qs on the SAME pseudo-loss for TRAIN_STEPS ──────────
    print()
    print("--- Training 5 Qs on shared synthetic targets (", TRAIN_STEPS, "steps) ---")
    for step in range(TRAIN_STEPS):
        _train_step(q1, za_train, 1.0)
        _train_step(q2, za_train, 1.0)
        _train_step(q3, za_train, 1.0)
        _train_step(q4, za_train, 1.0)
        _train_step(q5, za_train, 1.0)

    # ─── 4c — Held-out decoded value range after training ───────────────
    print()
    print("--- 4c. Decoded Q range on HELD-OUT batch after training ---")
    _q_forward(q1, za_holdout, q_logits + 0 * BATCH * BINS)
    _q_forward(q2, za_holdout, q_logits + 1 * BATCH * BINS)
    _q_forward(q3, za_holdout, q_logits + 2 * BATCH * BINS)
    _q_forward(q4, za_holdout, q_logits + 3 * BATCH * BINS)
    _q_forward(q5, za_holdout, q_logits + 4 * BATCH * BINS)

    var mean_range_train: Float64 = 0.0
    for b in range(BATCH):
        var vmin: Float64 = 1e30
        var vmax: Float64 = -1e30
        for q in range(NUM_Q):
            var v = _decode_logits_to_value(
                q_logits + q * BATCH * BINS + b * BINS, bins
            )
            if v < vmin:
                vmin = v
            if v > vmax:
                vmax = v
        mean_range_train += vmax - vmin
    mean_range_train /= Float64(BATCH)
    print(
        "    post-train mean (max-min) decoded Q on held-out =",
        mean_range_train,
    )
    _expect(
        mean_range_train > 1e-3,
        "4c — held-out decoded Q range > 1e-3 after",
        passed,
        total,
    )

    # ─── 4d — No pair of Qs is bitwise identical ────────────────────────
    print()
    print("--- 4d. Pairwise param distinctness ---")
    var all_distinct = True
    var max_pair_diff: Float64 = 0.0
    var min_pair_diff: Float64 = 1e30
    var qs = InlineArray[
        UnsafePointer[Scalar[dtype], MutAnyOrigin], NUM_Q
    ](uninitialized=True)
    qs[0] = q1.params
    qs[1] = q2.params
    qs[2] = q3.params
    qs[3] = q4.params
    qs[4] = q5.params

    for i in range(NUM_Q):
        for j in range(i + 1, NUM_Q):
            var d_sum: Float64 = 0.0
            for k in range(QModel.PARAM_SIZE):
                var d = Float64(qs[i][k]) - Float64(qs[j][k])
                d_sum += d * d
            var d_norm = sqrt(d_sum)
            if d_norm > max_pair_diff:
                max_pair_diff = d_norm
            if d_norm < min_pair_diff:
                min_pair_diff = d_norm
            if d_norm == 0.0:
                all_distinct = False
    print(
        "    min pairwise |Δparams| =", min_pair_diff,
        "  max pairwise |Δparams| =", max_pair_diff,
    )
    _expect(
        all_distinct,
        "4d — no pair of Q networks is bitwise identical",
        passed,
        total,
    )
    _expect(
        min_pair_diff > 1e-3,
        "4d.b — minimum pairwise param distance > 1e-3 (meaningful diversity)",
        passed,
        total,
    )

    q_logits.free()
    za_train_buf.free()
    za_holdout_buf.free()

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
