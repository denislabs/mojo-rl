"""EZv2 SimSiam consistency loss — analytic gradient vs finite differences.

Validates `efficient_zero_v2/loss_ops.mojo::consistency_loss_and_grad`: the
returned online gradient ``cos·p_i/‖p‖² − t_i/(‖p‖·‖t‖)`` must match a central
finite-difference of ``−cos(p, sg(t))`` at every component of ``p``. The target
``t`` is held constant (stop-grad), so finite differences are taken on ``p``
only. This is the plan's validation step #1 for Phase C (gradcheck the new head
before training) — the consistency loss is the one genuinely-new objective EZv2
adds over MuZero.

Run:
    pixi run mojo run -I . tests/deep_agents2/test_ezv2_consistency_gradcheck.mojo
"""

from std.memory import alloc
from std.testing import assert_true, assert_almost_equal

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.efficient_zero_v2.loss_ops import (
    consistency_loss_and_grad,
)


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _loss_only[
    BATCH: Int, DIM: Int,
](
    p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    t: UnsafePointer[Scalar[DT], MutAnyOrigin],
) -> Scalar[DT]:
    var junk = _alloc(BATCH * DIM)
    var l = consistency_loss_and_grad[BATCH, DIM](
        p, t, Scalar[DT](1.0), junk
    )
    junk.free()
    return l


def main() raises:
    comptime BATCH = 3
    comptime DIM = 6

    var p = _alloc(BATCH * DIM)
    var t = _alloc(BATCH * DIM)

    # Arbitrary non-degenerate online + target vectors (different directions so
    # cos is strictly inside (−1, 1) and the gradient is well-conditioned).
    for b in range(BATCH):
        for i in range(DIM):
            var idx = b * DIM + i
            p[idx] = Scalar[DT](0.4) * Scalar[DT]((i + b) % 5) - Scalar[DT](0.7)
            t[idx] = Scalar[DT](0.3) * Scalar[DT]((i * 2 + 1) % 4) + Scalar[DT](
                0.15
            ) * Scalar[DT](b)

    var grad = _alloc(BATCH * DIM)
    var l0 = consistency_loss_and_grad[BATCH, DIM](
        p, t, Scalar[DT](1.0), grad
    )
    assert_true(l0 == l0 and l0 < 1e30, "loss non-finite")
    # −cos sums into [−BATCH, BATCH].
    assert_true(
        l0 >= Scalar[DT](-BATCH) - Scalar[DT](1e-4)
        and l0 <= Scalar[DT](BATCH) + Scalar[DT](1e-4),
        "loss out of [−B, B]",
    )

    # Central finite differences on each component of p (t is detached).
    var eps = Scalar[DT](1e-3)
    var max_err = Scalar[DT](0.0)
    for j in range(BATCH * DIM):
        var saved = p[j]
        p[j] = saved + eps
        var lp = _loss_only[BATCH, DIM](p, t)
        p[j] = saved - eps
        var lm = _loss_only[BATCH, DIM](p, t)
        p[j] = saved
        var fd = (lp - lm) / (Scalar[DT](2.0) * eps)
        var err = grad[j] - fd
        if err < Scalar[DT](0.0):
            err = -err
        if err > max_err:
            max_err = err
        assert_almost_equal(
            grad[j], fd, atol=2e-3, rtol=2e-2,
            msg=String("grad mismatch at component ") + String(j),
        )
    print("max |analytic - finite-diff| =", max_err)

    # grad_scale must scale the gradient linearly.
    var grad2 = _alloc(BATCH * DIM)
    _ = consistency_loss_and_grad[BATCH, DIM](
        p, t, Scalar[DT](0.25), grad2
    )
    for j in range(BATCH * DIM):
        assert_almost_equal(
            grad2[j], grad[j] * Scalar[DT](0.25), atol=1e-6, rtol=1e-5,
            msg="grad_scale not applied linearly",
        )
    print("grad_scale linearity: OK")

    p.free(); t.free(); grad.free(); grad2.free()
    print("EZv2 consistency gradcheck: OK")
