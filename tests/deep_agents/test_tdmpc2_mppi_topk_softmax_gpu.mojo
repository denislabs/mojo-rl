"""TD-MPC2 — MPPI top-K elite softmax kernel correctness (Test 7).

Regression test for Bug 3 (audit doc): production was computing softmax
weights over ALL 536 MPPI samples instead of selecting top-`num_elites=64`
first. Reference TD-MPC2 (`tdmpc2.py:186`) uses `torch.topk` to filter to
elites before the softmax.

The fix added rank-counting inside `mppi_softmax_weights_kernel`. This
test pins that contract on a known-input synthetic case where the elites
are obvious.

Sub-tests:
  7a — exactly NUM_ELITES samples have nonzero weight, the rest are 0.
  7b — the nonzero weights correspond to the top-NUM_ELITES samples by
       return value (no false positives).
  7c — nonzero weights match exp(temp*(v - max)) / sum, normalized.
  7d — when NUM_ELITES == TOTAL_SAMPLES (no filter), all weights are
       nonzero (kernel doesn't accidentally zero anything).
  7e — degenerate case: all returns identical (ties). All NUM_ELITES
       elites should be selected by the index tiebreak (first K).
"""

from std.math import exp, sqrt
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.deep_agents.tdmpc2.kernels import mppi_softmax_weights_kernel


comptime N_ENVS = 1
comptime TOTAL_SAMPLES = 100  # smaller than production; easier to reason about
comptime NUM_ELITES = 16


def _expect(cond: Bool, label: String, mut passed: Int, mut total: Int):
    total += 1
    if cond:
        print("PASS:", label)
        passed += 1
    else:
        print("FAIL:", label)


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def main() raises:
    print("=" * 70)
    print("TD-MPC2 Test 7 — MPPI top-K elite softmax weights")
    print("=" * 70)

    var passed = 0
    var total = 0

    with DeviceContext() as ctx:
        comptime softmax_topk = mppi_softmax_weights_kernel[
            dtype, N_ENVS, TOTAL_SAMPLES, NUM_ELITES, TPB
        ]

        var returns_host = ctx.enqueue_create_host_buffer[dtype](TOTAL_SAMPLES)
        var weights_host = ctx.enqueue_create_host_buffer[dtype](TOTAL_SAMPLES)
        var returns_dev = ctx.enqueue_create_buffer[dtype](TOTAL_SAMPLES)
        var weights_dev = ctx.enqueue_create_buffer[dtype](TOTAL_SAMPLES)

        var returns_t = LayoutTensor[
            dtype, Layout.row_major(TOTAL_SAMPLES), MutAnyOrigin
        ](returns_dev.unsafe_ptr())
        var weights_t = LayoutTensor[
            dtype, Layout.row_major(TOTAL_SAMPLES), MutAnyOrigin
        ](weights_dev.unsafe_ptr())

        # ─── 7a/7b/7c — distinct returns, top-NUM_ELITES selected ────────
        # returns[s] = s gives a clean monotonic input where the top-K are
        # samples [TOTAL_SAMPLES-K .. TOTAL_SAMPLES-1]. Easy to reason about.
        print()
        print("--- 7a/b/c. Distinct returns: top-K elites selected ---")

        for s in range(TOTAL_SAMPLES):
            returns_host[s] = Scalar[dtype](Float64(s))
        ctx.enqueue_copy(returns_dev, returns_host)

        var temperature = Scalar[dtype](0.5)
        ctx.enqueue_function[softmax_topk, softmax_topk](
            returns_t, weights_t, temperature,
            grid_dim=(N_ENVS,), block_dim=(TPB,),
        )
        ctx.enqueue_copy(weights_host, weights_dev)
        ctx.synchronize()

        # 7a: exactly NUM_ELITES samples have nonzero weight.
        var nonzero_count = 0
        for s in range(TOTAL_SAMPLES):
            if Float64(weights_host[s]) > 1e-12:
                nonzero_count += 1
        print(
            "    nonzero count =", nonzero_count,
            "(expected", NUM_ELITES, ")",
        )
        _expect(
            nonzero_count == NUM_ELITES,
            "7a — exactly NUM_ELITES samples have nonzero weight",
            passed, total,
        )

        # 7b: the nonzero ones are the top-K by value (samples 84..99 here).
        var bottom_nonzero = 0
        var top_zero = 0
        comptime FIRST_ELITE = TOTAL_SAMPLES - NUM_ELITES
        for s in range(TOTAL_SAMPLES):
            var w = Float64(weights_host[s])
            if s < FIRST_ELITE and w > 1e-12:
                bottom_nonzero += 1
            if s >= FIRST_ELITE and w <= 1e-12:
                top_zero += 1
        _expect(
            bottom_nonzero == 0,
            "7b — no bottom-(N-K) sample has nonzero weight",
            passed, total,
        )
        _expect(
            top_zero == 0,
            "7b — all top-K samples have nonzero weight",
            passed, total,
        )

        # 7c: nonzero weights match the analytic softmax over elites.
        # Reference: w[s] = exp(temp * (v_s - max_v)) / sum_elite_exp.
        var max_v: Float64 = -1e30
        for s in range(FIRST_ELITE, TOTAL_SAMPLES):
            var v = Float64(returns_host[s])
            if v > max_v:
                max_v = v
        var sum_exp: Float64 = 0.0
        for s in range(FIRST_ELITE, TOTAL_SAMPLES):
            sum_exp += exp(0.5 * (Float64(returns_host[s]) - max_v))
        var max_rel: Float64 = 0.0
        var sum_w: Float64 = 0.0
        for s in range(FIRST_ELITE, TOTAL_SAMPLES):
            var got = Float64(weights_host[s])
            sum_w += got
            var expected = exp(0.5 * (Float64(returns_host[s]) - max_v)) / sum_exp
            var rel = _abs(got - expected) / (_abs(expected) + 1e-12)
            if rel > max_rel:
                max_rel = rel
        print("    max relative error vs analytic softmax =", max_rel)
        print("    sum of nonzero weights =", sum_w, " (expected ~1.0)")
        _expect(
            max_rel < 1e-4,
            "7c — nonzero weights match analytic elite softmax (rel < 1e-4)",
            passed, total,
        )
        _expect(
            _abs(sum_w - 1.0) < 1e-4,
            "7c — weights normalized: sum = 1",
            passed, total,
        )

        # ─── 7d — degenerate: NUM_ELITES == TOTAL_SAMPLES (no filter) ────
        # Compile a fresh kernel alias with NUM_ELITES = TOTAL_SAMPLES so
        # rank < NUM_ELITES is always true. Should produce all-nonzero
        # weights matching standard softmax.
        print()
        print("--- 7d. NUM_ELITES == TOTAL_SAMPLES → no filter, all nonzero ---")
        comptime softmax_no_filter = mppi_softmax_weights_kernel[
            dtype, N_ENVS, TOTAL_SAMPLES, TOTAL_SAMPLES, TPB
        ]
        ctx.enqueue_function[softmax_no_filter, softmax_no_filter](
            returns_t, weights_t, temperature,
            grid_dim=(N_ENVS,), block_dim=(TPB,),
        )
        ctx.enqueue_copy(weights_host, weights_dev)
        ctx.synchronize()
        var n_nonzero_full = 0
        var sum_w_full: Float64 = 0.0
        for s in range(TOTAL_SAMPLES):
            var w = Float64(weights_host[s])
            sum_w_full += w
            if w > 1e-30:
                n_nonzero_full += 1
        print(
            "    nonzero count =", n_nonzero_full,
            "(expected", TOTAL_SAMPLES, "), sum =", sum_w_full,
        )
        _expect(
            n_nonzero_full == TOTAL_SAMPLES,
            "7d — when NUM_ELITES = TOTAL, all weights nonzero",
            passed, total,
        )
        _expect(
            _abs(sum_w_full - 1.0) < 1e-4,
            "7d — weights normalized when no filter",
            passed, total,
        )

        # ─── 7e — degenerate: all returns identical (ties) ───────────────
        # All v_s equal → rank = #{k : v_k > v_s OR (v_k == v_s AND k < s)}
        # = #{k : k < s} (the index tiebreak). So sample s has rank s.
        # First NUM_ELITES samples (rank 0..K-1) are elites.
        print()
        print("--- 7e. Tied returns: index tiebreak picks first NUM_ELITES ---")
        for s in range(TOTAL_SAMPLES):
            returns_host[s] = Scalar[dtype](3.7)  # all the same
        ctx.enqueue_copy(returns_dev, returns_host)
        ctx.enqueue_function[softmax_topk, softmax_topk](
            returns_t, weights_t, temperature,
            grid_dim=(N_ENVS,), block_dim=(TPB,),
        )
        ctx.enqueue_copy(weights_host, weights_dev)
        ctx.synchronize()

        var first_k_nonzero = 0
        var rest_zero = 0
        for s in range(NUM_ELITES):
            if Float64(weights_host[s]) > 1e-12:
                first_k_nonzero += 1
        for s in range(NUM_ELITES, TOTAL_SAMPLES):
            if Float64(weights_host[s]) <= 1e-12:
                rest_zero += 1
        print(
            "    first NUM_ELITES nonzero =", first_k_nonzero,
            ", rest zero =", rest_zero,
        )
        _expect(
            first_k_nonzero == NUM_ELITES,
            "7e — first NUM_ELITES samples are elite by index tiebreak",
            passed, total,
        )
        _expect(
            rest_zero == TOTAL_SAMPLES - NUM_ELITES,
            "7e — remaining samples are non-elite (weight 0)",
            passed, total,
        )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
