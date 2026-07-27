"""Phase R.1 algebraic unit test for `redq_ensemble_target_cpu`.

Four checks, all on hand-set inputs (no models, no RNG):
  (a) MODE=MIN at N=4, N_MIN=2, subset=[1, 3]: combined = min of rows 1, 3.
  (b) MODE=AVE at N=4: combined = mean over all 4 rows.
  (c) Terminal mask: where term[b]=1, y[b] must equal r[b] exactly.
  (d) SAC-equivalence: MODE=MIN N=2 N_MIN=2 subset=[0,1] is bit-for-bit
      the SAC TwinCritic target formula `r + (1-term)·γ·(min(q1,q2) − α·lp)`.

Verifies the math is correct independently of any actor/critic/RNG
plumbing — the integration smokes (block-level) gate that the block
calls the kernel correctly; this test gates the kernel itself.
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.redq.kernels import (
    redq_ensemble_target_cpu,
    REDQ_TARGET_MIN,
    REDQ_TARGET_AVE,
)
from mojo_rl.nn.core.tensor import Tensor


# ─────────────────────────────────────────────────────────────────────
# Helper: build a [N, BATCH] q_next buffer with explicit per-(n, b) values.
# ─────────────────────────────────────────────────────────────────────


def _alloc_q_next[N: Int, BATCH: Int]() raises -> Tensor:
    """Allocate an N*BATCH zero-filled list. Caller fills entries."""
    return Tensor.alloc(N * BATCH)


def _list_int_2(a: Int, b: Int) raises -> List[Int]:
    var out = List[Int](length=2, fill=0)
    out[0] = a
    out[1] = b
    return out^


def _list_int_1(a: Int) raises -> List[Int]:
    var out = List[Int](length=1, fill=0)
    out[0] = a
    return out^


# ─────────────────────────────────────────────────────────────────────
# (a) MODE=MIN N=4 N_MIN=2 subset=[1, 3]
# ─────────────────────────────────────────────────────────────────────


def test_mode_min_subset_picks() raises:
    print("--- (a) MODE=MIN N=4 N_MIN=2 subset=[1,3] ---")
    comptime N = 4
    comptime BATCH = 3
    comptime N_MIN = 2

    var q_buf = _alloc_q_next[N, BATCH]()
    # Lay out distinct values per (n, b). Row-major: q[n, b] at index n*BATCH+b.
    # Pick rows so the min of rows 1 and 3 is unambiguous AND differs
    # from min over all four rows (verifies subset selection actually
    # honors subset_idxs, not just "min over everything").
    #
    # n=0: [10, 11, 12]   (would dominate min if subset_idxs were ignored)
    # n=1: [ 5,  4,  3]   (selected — small)
    # n=2: [-2, -3, -4]   (NOT selected — would distort if read)
    # n=3: [ 7,  8,  9]   (selected — larger than n=1)
    var row_vals = List[Float64](length=N * BATCH, fill=0.0)
    # n=0: 10, 11, 12
    row_vals[0 * BATCH + 0] = 10.0
    row_vals[0 * BATCH + 1] = 11.0
    row_vals[0 * BATCH + 2] = 12.0
    # n=1: 5, 4, 3
    row_vals[1 * BATCH + 0] = 5.0
    row_vals[1 * BATCH + 1] = 4.0
    row_vals[1 * BATCH + 2] = 3.0
    # n=2: -2, -3, -4
    row_vals[2 * BATCH + 0] = -2.0
    row_vals[2 * BATCH + 1] = -3.0
    row_vals[2 * BATCH + 2] = -4.0
    # n=3: 7, 8, 9
    row_vals[3 * BATCH + 0] = 7.0
    row_vals[3 * BATCH + 1] = 8.0
    row_vals[3 * BATCH + 2] = 9.0
    for k in range(N * BATCH):
        q_buf.data[k] = Scalar[DT](row_vals[k])

    var subset = _list_int_2(1, 3)
    var rewards = Tensor.alloc(BATCH)
    var terms   = Tensor.alloc(BATCH)
    var lps     = Tensor.alloc(BATCH)
    var y       = Tensor.alloc(BATCH)

    redq_ensemble_target_cpu[N, N_MIN, REDQ_TARGET_MIN, BATCH](
        rewards,
        q_buf,
        terms,
        lps,
        subset,
        Scalar[DT](1.0),  # γ = 1
        Scalar[DT](0.0),  # α = 0 → soft_v = combined
        y,
    )
    # With α=0, γ=1, r=0, term=0: y[b] = min(rows[1][b], rows[3][b]).
    # Row 1 is always smaller (5<7, 4<8, 3<9) → expected = [5, 4, 3].
    for b in range(BATCH):
        var expected = Scalar[DT](row_vals[1 * BATCH + b])
        print("  b=", b, " y =", y.data[b], " expected =", expected)
        assert_true(
            y.data[b] == expected,
            "y[b] must equal min(row1[b], row3[b]) = row1[b]",
        )


# ─────────────────────────────────────────────────────────────────────
# (b) MODE=AVE N=4
# ─────────────────────────────────────────────────────────────────────


def test_mode_ave_all_rows() raises:
    print("--- (b) MODE=AVE N=4 ---")
    comptime N = 4
    comptime BATCH = 2
    comptime N_MIN = 1  # ignored

    var q_buf = _alloc_q_next[N, BATCH]()
    # Per-batch values designed so the AVE is non-trivial.
    # b=0: rows=[1, 2, 3, 4] → mean = 2.5
    # b=1: rows=[-1, 1, -1, 1] → mean = 0.0
    q_buf.data[0 * BATCH + 0] = Scalar[DT](1.0)
    q_buf.data[0 * BATCH + 1] = Scalar[DT](-1.0)
    q_buf.data[1 * BATCH + 0] = Scalar[DT](2.0)
    q_buf.data[1 * BATCH + 1] = Scalar[DT](1.0)
    q_buf.data[2 * BATCH + 0] = Scalar[DT](3.0)
    q_buf.data[2 * BATCH + 1] = Scalar[DT](-1.0)
    q_buf.data[3 * BATCH + 0] = Scalar[DT](4.0)
    q_buf.data[3 * BATCH + 1] = Scalar[DT](1.0)

    var subset = _list_int_1(0)  # ignored
    var rewards = Tensor.alloc(BATCH)
    var terms   = Tensor.alloc(BATCH)
    var lps     = Tensor.alloc(BATCH)
    var y       = Tensor.alloc(BATCH)
    redq_ensemble_target_cpu[N, N_MIN, REDQ_TARGET_AVE, BATCH](
        rewards,
        q_buf,
        terms,
        lps,
        subset,
        Scalar[DT](1.0), Scalar[DT](0.0),
        y,
    )
    var expected_0 = Scalar[DT]((1.0 + 2.0 + 3.0 + 4.0) / 4.0)
    var expected_1 = Scalar[DT]((-1.0 + 1.0 + -1.0 + 1.0) / 4.0)
    print("  b=0 y =", y.data[0], " expected =", expected_0)
    print("  b=1 y =", y.data[1], " expected =", expected_1)
    assert_true(y.data[0] == expected_0, "AVE-mode b=0")
    assert_true(y.data[1] == expected_1, "AVE-mode b=1")


# ─────────────────────────────────────────────────────────────────────
# (c) Terminal mask — term=1 drops bootstrap → y == r
# ─────────────────────────────────────────────────────────────────────


def test_terminal_mask_drops_bootstrap() raises:
    print("--- (c) Terminal mask drops bootstrap ---")
    comptime N = 2
    comptime BATCH = 4
    comptime N_MIN = 2

    var q_buf = _alloc_q_next[N, BATCH]()
    # Big arbitrary Q values — must NOT leak into y on terminated samples.
    for n in range(N):
        for b in range(BATCH):
            q_buf.data[n * BATCH + b] = Scalar[DT](100.0 * Float64(n + 1) + Float64(b))

    var subset = _list_int_2(0, 1)
    var rewards = Tensor.alloc(BATCH)
    rewards.data[0] = Scalar[DT](-1.0); rewards.data[1] = Scalar[DT](2.0)
    rewards.data[2] = Scalar[DT](-3.0); rewards.data[3] = Scalar[DT](7.5)
    # b=0, b=2 → term=1 (real termination); b=1, b=3 → term=0.
    var terms = Tensor.alloc(BATCH)
    terms.data[0] = Scalar[DT](1.0); terms.data[1] = Scalar[DT](0.0)
    terms.data[2] = Scalar[DT](1.0); terms.data[3] = Scalar[DT](0.0)
    var lps = Tensor.alloc(BATCH)
    lps.data[0] = Scalar[DT](-0.5); lps.data[1] = Scalar[DT](0.3)
    lps.data[2] = Scalar[DT](-0.8); lps.data[3] = Scalar[DT](1.2)
    var y = Tensor.alloc(BATCH)
    var gamma = Scalar[DT](0.99)
    var alpha = Scalar[DT](0.2)
    redq_ensemble_target_cpu[N, N_MIN, REDQ_TARGET_MIN, BATCH](
        rewards,
        q_buf,
        terms,
        lps,
        subset,
        gamma, alpha,
        y,
    )
    # Terminated samples: y must equal r exactly (no bootstrap leak).
    print("  b=0 (term=1) y =", y.data[0], " expected =", rewards.data[0])
    print("  b=2 (term=1) y =", y.data[2], " expected =", rewards.data[2])
    assert_true(y.data[0] == rewards.data[0], "term=1 ⇒ y == r at b=0")
    assert_true(y.data[2] == rewards.data[2], "term=1 ⇒ y == r at b=2")
    # Non-terminated: y = r + γ · (min(q0,q1) − α·lp). At BATCH index b
    # the min over rows 0/1 = row 0 (smaller for every b).
    for b in range(BATCH):
        if terms.data[b] == Scalar[DT](0.0):
            var combined = q_buf.data[0 * BATCH + b]  # row 0 wins min
            var expected = rewards.data[b] + gamma * (combined - alpha * lps.data[b])
            print("  b=", b, " (term=0) y =", y.data[b], " expected =", expected)
            assert_true(y.data[b] == expected, "term=0 path")


# ─────────────────────────────────────────────────────────────────────
# (d) SAC-equivalence at N=2 N_MIN=2 subset=[0,1] MODE=MIN
# ─────────────────────────────────────────────────────────────────────


def test_sac_equivalence_n2_min2() raises:
    print("--- (d) SAC-equivalence: N=2 N_MIN=2 MODE=MIN ---")
    comptime N = 2
    comptime BATCH = 5
    comptime N_MIN = 2

    var q_buf = _alloc_q_next[N, BATCH]()
    # Q1 (row 0) and Q2 (row 1) per batch — alternate which one is smaller
    # so min isn't always "row 0".
    var q1 = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0.0))
    q1[0] = Scalar[DT](-0.5); q1[1] = Scalar[DT](1.2); q1[2] = Scalar[DT](-3.0)
    q1[3] = Scalar[DT](0.1);  q1[4] = Scalar[DT](2.5)
    var q2 = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0.0))
    q2[0] = Scalar[DT](-1.0); q2[1] = Scalar[DT](1.5); q2[2] = Scalar[DT](-2.5)
    q2[3] = Scalar[DT](-0.3); q2[4] = Scalar[DT](2.0)
    for b in range(BATCH):
        q_buf.data[0 * BATCH + b] = q1[b]
        q_buf.data[1 * BATCH + b] = q2[b]

    var subset = _list_int_2(0, 1)
    var rewards = Tensor.alloc(BATCH)
    rewards.data[0] = Scalar[DT](-1.0); rewards.data[1] = Scalar[DT](0.5)
    rewards.data[2] = Scalar[DT](2.0);  rewards.data[3] = Scalar[DT](-0.2)
    rewards.data[4] = Scalar[DT](0.0)
    var terms = Tensor.alloc(BATCH)
    terms.data[2] = Scalar[DT](1.0)  # b=2 is terminated; others stay 0
    var lps = Tensor.alloc(BATCH)
    lps.data[0] = Scalar[DT](-0.3); lps.data[1] = Scalar[DT](0.8); lps.data[2] = Scalar[DT](-1.5)
    lps.data[3] = Scalar[DT](0.2);  lps.data[4] = Scalar[DT](-0.5)
    var y = Tensor.alloc(BATCH)
    var gamma = Scalar[DT](0.97)
    var alpha = Scalar[DT](0.15)
    redq_ensemble_target_cpu[N, N_MIN, REDQ_TARGET_MIN, BATCH](
        rewards,
        q_buf,
        terms,
        lps,
        subset,
        gamma, alpha,
        y,
    )
    # SAC formula: y.data[b] = r[b] + (1-term[b]) · γ · (min(q1[b], q2[b]) − α·lp[b])
    var max_dev: Float64 = 0.0
    for b in range(BATCH):
        var mn = q1[b] if q1[b] < q2[b] else q2[b]
        var nonterm = Scalar[DT](1.0) - terms.data[b]
        var expected = rewards.data[b] + nonterm * gamma * (mn - alpha * lps.data[b])
        var d = Float64(y.data[b]) - Float64(expected)
        if d < 0.0:
            d = -d
        if d > max_dev:
            max_dev = d
        print("  b=", b, " y =", y.data[b], " SAC-expected =", expected)
    print("  max |y_redq - y_sac| =", max_dev)
    assert_true(max_dev == 0.0, "REDQ N=2 M=2 MIN must be bit-identical to SAC TwinCritic target")


def main() raises:
    test_mode_min_subset_picks()
    test_mode_ave_all_rows()
    test_terminal_mask_drops_bootstrap()
    test_sac_equivalence_n2_min2()
    print("PASS — redq_ensemble_target_cpu kernel passes all 4 unit checks.")
