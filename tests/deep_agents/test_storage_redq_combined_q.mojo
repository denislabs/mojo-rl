"""REDQ combined_Q reduction isolation gate (storage kernels, CPU).

Builds a stacked [N, BATCH] Q Tensor with KNOWN values, runs the storage
`redq_ensemble_target_cpu` in BOTH modes (MIN over a fixed subset, AVE over all
N) with α=0, γ=1, term=0, r=0 so that

    y[b] == combined_Q[b]   (= min over subset / mean over all N)

and compares to a hand oracle. Then a second pass with non-trivial α/γ/r/term
checks the full Bellman fold. Assert < 1e-5.

Run: pixi run mojo run -I . tests/deep_agents/test_storage_redq_combined_q.mojo
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.deep_agents.redq.kernels import (
    redq_ensemble_target_cpu,
    REDQ_TARGET_MIN,
    REDQ_TARGET_AVE,
)


comptime N = 4
comptime N_MIN = 2
comptime BATCH = 5


def main() raises:
    print("=" * 60)
    print("REDQ combined_Q reduction isolation gate (CPU)")
    print("=" * 60)

    # Stacked q_next[N, BATCH]: q[n, b] = (n + 1) * 10 + b  (distinct, known).
    var q = Tensor.alloc(N * BATCH)
    for n in range(N):
        for b in range(BATCH):
            q.data[n * BATCH + b] = Scalar[DT]((n + 1) * 10 + b)

    var r = Tensor.alloc(BATCH)
    var terms = Tensor.alloc(BATCH)
    var lp = Tensor.alloc(BATCH)
    var y = Tensor.alloc(BATCH)
    for b in range(BATCH):
        r.data[b] = Scalar[DT](0.0)
        terms.data[b] = Scalar[DT](0.0)
        lp.data[b] = Scalar[DT](0.0)

    var max_err = Scalar[DT](0.0)
    var ok = True

    # ── Pass 1a: MIN over subset {2, 0} (idxs into N), α=0 γ=1 → y == min. ──
    var subset = List[Int](length=N_MIN, fill=0)
    subset[0] = 2
    subset[1] = 0
    redq_ensemble_target_cpu[N, N_MIN, REDQ_TARGET_MIN, BATCH](
        r, q, terms, lp, subset, Scalar[DT](1.0), Scalar[DT](0.0), y
    )
    for b in range(BATCH):
        var a = q.data[2 * BATCH + b]
        var c = q.data[0 * BATCH + b]
        var mn = a if a < c else c
        var err = abs(y.data[b] - mn)
        if err > max_err:
            max_err = err
        if err > Scalar[DT](1e-5):
            ok = False
    print("  MIN-subset max err:", max_err)

    # ── Pass 1b: AVE over all N, α=0 γ=1 → y == mean. ──
    redq_ensemble_target_cpu[N, N_MIN, REDQ_TARGET_AVE, BATCH](
        r, q, terms, lp, subset, Scalar[DT](1.0), Scalar[DT](0.0), y
    )
    for b in range(BATCH):
        var acc = Scalar[DT](0.0)
        for n in range(N):
            acc += q.data[n * BATCH + b]
        var mean = acc / Scalar[DT](N)
        var err = abs(y.data[b] - mean)
        if err > max_err:
            max_err = err
        if err > Scalar[DT](1e-5):
            ok = False
    print("  AVE max err       :", max_err)

    # ── Pass 2: full Bellman fold, MIN-subset, α=0.3 γ=0.99 r,term varied. ──
    var gamma = Scalar[DT](0.99)
    var alpha = Scalar[DT](0.3)
    for b in range(BATCH):
        r.data[b] = Scalar[DT](b) * 0.5 - 1.0
        lp.data[b] = Scalar[DT](b) * 0.1 - 0.2
        terms.data[b] = Scalar[DT](1.0) if (b == 2) else Scalar[DT](0.0)
    redq_ensemble_target_cpu[N, N_MIN, REDQ_TARGET_MIN, BATCH](
        r, q, terms, lp, subset, gamma, alpha, y
    )
    for b in range(BATCH):
        var a = q.data[2 * BATCH + b]
        var c = q.data[0 * BATCH + b]
        var mn = a if a < c else c
        var nonterm = Scalar[DT](1.0) - terms.data[b]
        var y_ref = r.data[b] + nonterm * gamma * (mn - alpha * lp.data[b])
        var err = abs(y.data[b] - y_ref)
        if err > max_err:
            max_err = err
        if err > Scalar[DT](1e-5):
            ok = False
    print("  Bellman-fold max err:", max_err)

    assert_true(ok, "combined_Q MIN/AVE + Bellman fold match oracle (< 1e-5)")
    print("REDQ COMBINED_Q ISOLATION OK")
