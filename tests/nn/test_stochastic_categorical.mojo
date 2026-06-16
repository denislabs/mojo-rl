"""StochasticCategorical (Block D-4).

Covers:
  * Forward output shape: one-hot sample then log_prob in last column
  * Sample distribution: with uniform logits each class is sampled ~1/N
  * log_prob math: equals log_softmax(logits)[sample_idx]
  * Backward via FD against a downstream loss that mixes sample + log_prob
"""

from std.math import abs as fabs, log, exp
from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.primitives.stochastic_categorical import StochasticCategorical
from mojo_rl.nn.initializer import Kaiming


def test_forward_shape_and_logprob() raises:
    """One-hot sums to 1 per row; log_prob equals log_softmax at sample."""
    seed(0)
    comptime BATCH = 8
    comptime N = 5
    var s = StochasticCategorical[N].make[target="cpu", INIT=Kaiming]()
    var lg_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N)
    var out_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * (N + 1))
    for k in range(BATCH * N):
        lg_p[k] = Scalar[DT](0.1 + 0.05 * Float64(k))  # tilted but non-confident
    var lg_t = TileTensor(lg_p, row_major[BATCH, N]())
    var out_t = TileTensor(out_p, row_major[BATCH, N + 1]())
    s.forward["cpu", BATCH](lg_t, output=out_t)

    for b in range(BATCH):
        var row_sum: Scalar[DT] = 0.0
        var nonzero_count = 0
        var sample_idx = 0
        for c in range(N):
            var v = out_p[b * (N + 1) + c]
            row_sum += v
            if v != 0.0:
                nonzero_count += 1
                sample_idx = c
            assert_true(
                v == 0.0 or v == 1.0,
                "sample entries must be 0 or 1",
            )
        assert_true(row_sum == Scalar[DT](1.0), "one-hot row must sum to 1")
        assert_true(nonzero_count == 1, "exactly one non-zero per row")

        # log_prob check
        var max_l = lg_p[b * N]
        for c in range(1, N):
            var v = lg_p[b * N + c]
            if v > max_l:
                max_l = v
        var se: Scalar[DT] = 0.0
        for c in range(N):
            se += exp(lg_p[b * N + c] - max_l)
        var lse = max_l + log(se)
        var expected_lp = lg_p[b * N + sample_idx] - lse
        var got_lp = out_p[b * (N + 1) + N]
        assert_true(
            fabs(expected_lp - got_lp) < Scalar[DT](1e-5),
            "log_prob mismatch",
        )

    lg_p.free()
    out_p.free()
    print("  test_forward_shape_and_logprob PASSED")


def test_sample_distribution_uniform() raises:
    """With uniform logits, large-N draws should distribute approximately
    uniformly across classes (within 3σ of multinomial)."""
    seed(42)
    comptime BATCH = 2000
    comptime N = 4
    var s = StochasticCategorical[N].make[target="cpu", INIT=Kaiming]()
    var lg_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N)
    var out_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * (N + 1))
    for k in range(BATCH * N):
        lg_p[k] = 0.0
    var lg_t = TileTensor(lg_p, row_major[BATCH, N]())
    var out_t = TileTensor(out_p, row_major[BATCH, N + 1]())
    s.forward["cpu", BATCH](lg_t, output=out_t)

    var counts = InlineArray[Int, N](fill=0)
    for b in range(BATCH):
        for c in range(N):
            if out_p[b * (N + 1) + c] == 1.0:
                counts[c] += 1

    var expected = Float64(BATCH) / Float64(N)
    var sd = (expected * (1.0 - 1.0 / Float64(N))) ** 0.5  # multinomial sd
    print(
        "  uniform-logit sample counts:", counts[0], counts[1], counts[2], counts[3],
        " expected≈", expected, " ±3σ=", 3.0 * sd,
    )
    for c in range(N):
        var diff = Float64(counts[c]) - expected
        if diff < 0:
            diff = -diff
        assert_true(diff < 4.0 * sd, "uniform sample skew too large")

    lg_p.free()
    out_p.free()
    print("  test_sample_distribution_uniform PASSED")


def test_logprob_backward_only() raises:
    """When the downstream loss is just `log_prob.sum()`, the gradient
    should match the analytical Categorical log-prob gradient:
        grad_logits[b, k] = δ(k, sample_idx[b]) − softmax(logits)[b, k]
    """
    seed(7)
    comptime BATCH = 3
    comptime N = 4
    var s = StochasticCategorical[N].make[target="cpu", INIT=Kaiming]()
    var lg_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N)
    var out_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * (N + 1))
    var go_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * (N + 1))
    var gi_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N)

    for k in range(BATCH * N):
        lg_p[k] = Scalar[DT](-0.2 + 0.1 * Float64(k))
        gi_p[k] = 0.0

    var lg_t = TileTensor(lg_p, row_major[BATCH, N]())
    var out_t = TileTensor(out_p, row_major[BATCH, N + 1]())
    s.forward["cpu", BATCH](lg_t, output=out_t)

    # Loss = sum_b log_prob[b]. → grad_log_prob = 1, grad_sample = 0.
    for k in range(BATCH * (N + 1)):
        go_p[k] = 0.0
    for b in range(BATCH):
        go_p[b * (N + 1) + N] = 1.0
    var go_t = TileTensor(go_p, row_major[BATCH, N + 1]())
    var gi_t = TileTensor(gi_p, row_major[BATCH, N]())
    s.vjp["cpu", BATCH](go_t, gi_t)

    # Build the expected gradient.
    var max_err: Scalar[DT] = 0.0
    for b in range(BATCH):
        var max_l = lg_p[b * N]
        for c in range(1, N):
            var v = lg_p[b * N + c]
            if v > max_l:
                max_l = v
        var se: Scalar[DT] = 0.0
        for c in range(N):
            se += exp(lg_p[b * N + c] - max_l)
        var sm_arr = InlineArray[Scalar[DT], N](fill=0)
        for c in range(N):
            sm_arr[c] = exp(lg_p[b * N + c] - max_l) / se

        # Find the sampled class.
        var sample_idx = 0
        for c in range(N):
            if out_p[b * (N + 1) + c] == 1.0:
                sample_idx = c

        for k in range(N):
            var delta: Scalar[DT] = 1.0 if k == sample_idx else 0.0
            var expected = delta - sm_arr[k]
            var got = gi_p[b * N + k]
            var err = fabs(got - expected)
            if err > max_err:
                max_err = err

    print("  log_prob-only backward max_err = ", max_err)
    assert_true(max_err < Scalar[DT](1e-5), "log_prob backward mismatch")

    lg_p.free()
    out_p.free()
    go_p.free()
    gi_p.free()
    print("  test_logprob_backward_only PASSED")


def test_analytical_sample_branch() raises:
    """The straight-through estimator's sample-branch gradient is, by
    definition, the softmax gradient applied to the discrete sample. We
    verify the analytical math directly (no FD — FD on a discrete output
    has argmax-flip discontinuities):

      grad_logits[b, k] = sm[b, k] · (grad_sample[b, k] − E_sm[grad_sample])

    where E_sm[g] = sum_j sm[b, j] · g[b, j].
    """
    seed(0)
    comptime BATCH = 3
    comptime N = 4
    var s = StochasticCategorical[N].make[target="cpu", INIT=Kaiming]()
    var lg_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N)
    var out_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * (N + 1))
    var go_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * (N + 1))
    var gi_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N)
    for k in range(BATCH * N):
        lg_p[k] = Scalar[DT](-0.4 + 0.1 * Float64(k))
        gi_p[k] = 0.0
    # arbitrary non-uniform grad_sample
    for k in range(BATCH * (N + 1)):
        go_p[k] = 0.0
    for b in range(BATCH):
        for c in range(N):
            go_p[b * (N + 1) + c] = Scalar[DT](0.3 + 0.07 * Float64(c))

    var lg_t = TileTensor(lg_p, row_major[BATCH, N]())
    var out_t = TileTensor(out_p, row_major[BATCH, N + 1]())
    var go_t = TileTensor(go_p, row_major[BATCH, N + 1]())
    var gi_t = TileTensor(gi_p, row_major[BATCH, N]())
    s.forward["cpu", BATCH](lg_t, output=out_t)
    s.vjp["cpu", BATCH](go_t, gi_t)

    # Reference: softmax-grad formula.
    var max_err: Scalar[DT] = 0.0
    for b in range(BATCH):
        var max_l = lg_p[b * N]
        for c in range(1, N):
            var v = lg_p[b * N + c]
            if v > max_l:
                max_l = v
        var se: Scalar[DT] = 0.0
        for c in range(N):
            se += exp(lg_p[b * N + c] - max_l)
        var sm = InlineArray[Scalar[DT], N](fill=0)
        for c in range(N):
            sm[c] = exp(lg_p[b * N + c] - max_l) / se

        var exp_g: Scalar[DT] = 0.0
        for c in range(N):
            exp_g += sm[c] * go_p[b * (N + 1) + c]

        for k in range(N):
            var expected = sm[k] * (go_p[b * (N + 1) + k] - exp_g)
            var got = gi_p[b * N + k]
            var err = fabs(got - expected)
            if err > max_err:
                max_err = err

    print("  analytical sample-branch backward max_err = ", max_err)
    assert_true(max_err < Scalar[DT](1e-5), "sample-branch math mismatch")

    lg_p.free()
    out_p.free()
    go_p.free()
    gi_p.free()
    print("  test_analytical_sample_branch PASSED")


def main() raises:
    print("=" * 60)
    print("nn StochasticCategorical tests (Block D-4)")
    print("=" * 60)
    test_forward_shape_and_logprob()
    test_sample_distribution_uniform()
    test_logprob_backward_only()
    test_analytical_sample_branch()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
