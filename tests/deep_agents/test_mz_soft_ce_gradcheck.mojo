"""MuZero soft-CE loss — analytic gradient vs finite differences, pure CPU.

Validates `muzero/loss_ops.mojo::soft_ce_loss_and_grad`: the returned gradient
(softmax(z) − q) must match a central finite-difference of the loss at every
logit. This is the plan's validation step #1 (gradcheck every head before
training) — the soft-CE is the shared math behind the policy, value, and reward
heads, so one gradcheck covers all three.

Run:
    pixi run mojo run -I . tests/deep_agents/test_mz_soft_ce_gradcheck.mojo
"""

from std.memory import alloc
from std.testing import assert_true, assert_almost_equal

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.muzero.loss_ops import soft_ce_loss_and_grad


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _loss_only[
    BATCH: Int, NBINS: Int,
](
    logits: UnsafePointer[Scalar[DT], MutAnyOrigin],
    target: UnsafePointer[Scalar[DT], MutAnyOrigin],
) -> Scalar[DT]:
    var junk = _alloc(BATCH * NBINS)
    var l = soft_ce_loss_and_grad[BATCH, NBINS](
        logits, target, Scalar[DT](1.0), junk
    )
    junk.free()
    return l


def main() raises:
    comptime BATCH = 3
    comptime NBINS = 7

    var logits = _alloc(BATCH * NBINS)
    var target = _alloc(BATCH * NBINS)

    # Arbitrary logits.
    for i in range(BATCH * NBINS):
        logits[i] = Scalar[DT](0.37) * Scalar[DT](i % 5) - Scalar[DT](0.8)
    # Soft targets: a normalized positive distribution per row (not one-hot, to
    # exercise the full soft-CE, whose loss floors at the target entropy).
    for b in range(BATCH):
        var s = Scalar[DT](0.0)
        for i in range(NBINS):
            var w = Scalar[DT](1.0) + Scalar[DT]((b + i) % 4)
            target[b * NBINS + i] = w
            s += w
        for i in range(NBINS):
            target[b * NBINS + i] = target[b * NBINS + i] / s

    var grad = _alloc(BATCH * NBINS)
    var l0 = soft_ce_loss_and_grad[BATCH, NBINS](
        logits, target, Scalar[DT](1.0), grad
    )
    assert_true(l0 == l0 and l0 < 1e30, "loss non-finite")

    # Central finite differences on each logit.
    var eps = Scalar[DT](1e-3)
    var max_err = Scalar[DT](0.0)
    for j in range(BATCH * NBINS):
        var saved = logits[j]
        logits[j] = saved + eps
        var lp = _loss_only[BATCH, NBINS](logits, target)
        logits[j] = saved - eps
        var lm = _loss_only[BATCH, NBINS](logits, target)
        logits[j] = saved
        var fd = (lp - lm) / (Scalar[DT](2.0) * eps)
        var err = grad[j] - fd
        if err < Scalar[DT](0.0):
            err = -err
        if err > max_err:
            max_err = err
        assert_almost_equal(
            grad[j], fd, atol=2e-3, rtol=2e-2,
            msg=String("grad mismatch at logit ") + String(j),
        )
    print("max |analytic - finite-diff| =", max_err)

    # grad_scale must scale the gradient linearly.
    var grad2 = _alloc(BATCH * NBINS)
    _ = soft_ce_loss_and_grad[BATCH, NBINS](
        logits, target, Scalar[DT](0.25), grad2
    )
    for j in range(BATCH * NBINS):
        assert_almost_equal(
            grad2[j], grad[j] * Scalar[DT](0.25), atol=1e-6, rtol=1e-5,
            msg="grad_scale not applied linearly",
        )
    print("grad_scale linearity: OK")

    logits.free(); target.free(); grad.free(); grad2.free()
    print("MuZero soft-CE gradcheck: OK")
