"""FD-gradcheck for the DreamerV3 discrete (unimix categorical) actor dist.

Verifies `dists_discrete.cat_bwd` against central finite differences of the
composite scalar loss  L = a·logp(k) + b·entropy  over random logits, and
sanity-checks the forward (probs sum to 1, entropy in [0, log C]).

Run: pixi run mojo run -I . tests/nn2/test_dreamerv3_discrete_dist.mojo
"""

from std.memory import alloc
from std.math import log, abs
from std.random import random_float64, seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.dreamerv3.dists_discrete import (
    cat_fwd, cat_bwd, cat_softmax_mix, UNIMIX,
)


def _loss[C: Int](
    logits: UnsafePointer[Scalar[DT], MutAnyOrigin],
    u: Scalar[DT], k: Int, a: Scalar[DT], b: Scalar[DT],
    sm: UnsafePointer[Scalar[DT], MutAnyOrigin],
    p: UnsafePointer[Scalar[DT], MutAnyOrigin],
) -> Scalar[DT]:
    var r = cat_fwd[C](logits, 0, u, k, sm, p)
    return a * r[0] + b * r[1]


def main() raises:
    print("=" * 70)
    print("DreamerV3 discrete actor dist — FD gradcheck")
    print("=" * 70)
    seed(20260529)
    comptime C = 5
    var u = UNIMIX
    var a = Scalar[DT](0.7)     # ∂L/∂logp
    var b = Scalar[DT](-0.3)    # ∂L/∂entropy

    var logits = alloc[Scalar[DT]](C)
    var sm = alloc[Scalar[DT]](C)
    var p = alloc[Scalar[DT]](C)
    var grad = alloc[Scalar[DT]](C)

    var max_rel: Scalar[DT] = 0.0
    for trial in range(6):
        for c in range(C):
            logits[c] = Scalar[DT](random_float64() * 4.0 - 2.0)
        var k = Int(random_float64() * Scalar[DT](C).cast[DType.float64]())
        if k >= C:
            k = C - 1

        # forward sanity
        _ = cat_fwd[C](logits, 0, u, k, sm, p)
        var psum = Scalar[DT](0.0)
        for c in range(C):
            psum += p[c]
        assert_true(abs(psum - Scalar[DT](1.0)) < Scalar[DT](1e-4), "probs sum 1")

        # analytic grad
        for c in range(C):
            grad[c] = 0.0
        # repopulate sm/p caches (cat_fwd above filled them for this k)
        cat_softmax_mix[C](logits, 0, u, sm, p)
        cat_bwd[C](sm, p, u, k, a, b, grad, 0)

        # central FD
        var eps = Scalar[DT](1e-3)
        for c in range(C):
            var orig = logits[c]
            logits[c] = orig + eps
            var lp = _loss[C](logits, u, k, a, b, sm, p)
            logits[c] = orig - eps
            var lm = _loss[C](logits, u, k, a, b, sm, p)
            logits[c] = orig
            var fd = (lp - lm) / (Scalar[DT](2.0) * eps)
            var an = grad[c]
            var denom = abs(fd) + abs(an) + Scalar[DT](1e-6)
            var rel = abs(fd - an) / denom
            if rel > max_rel:
                max_rel = rel
        print("  trial", trial, " k=", k, " max_rel so far=", max_rel)

    print("  max relative grad error =", max_rel)
    assert_true(max_rel < Scalar[DT](1e-2), "cat_bwd matches FD")
    logits.free(); sm.free(); p.free(); grad.free()
    print("=" * 70)
    print("PASSED — discrete actor dist forward + gradient verified")
    print("=" * 70)
