"""EZv2 continuous policy loss — analytic gradient vs finite differences.

Validates `efficient_zero_v2/loss_ops_continuous.mojo::
continuous_policy_loss_and_grad`: the analytic gradients wrt the raw mean/std
logits (through the soft-clamp tanh + softplus parameterization) must match
central finite differences of the squashed-Gaussian NLL + Gaussian-entropy
objective. The target action is held constant (detached), so finite differences
are taken on ``μ_raw`` / ``σ_raw`` only. Plan validation step #1 for the
continuous head.

Run:
    pixi run mojo run -I . tests/deep_agents/test_ezv2_continuous_policy_gradcheck.mojo
"""

from std.memory import alloc
from std.testing import assert_true, assert_almost_equal

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.efficient_zero_v2.loss_ops_continuous import (
    continuous_policy_loss_and_grad,
)


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _loss_only[
    BATCH: Int, ACT_DIM: Int,
](
    musig: UnsafePointer[Scalar[DT], MutAnyOrigin],
    tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],
) -> Scalar[DT]:
    var junk = _alloc(BATCH * 2 * ACT_DIM)
    var l = continuous_policy_loss_and_grad[BATCH, ACT_DIM](
        musig, tgt, Scalar[DT](1.0), junk,
    )
    junk.free()
    return l


def main() raises:
    comptime BATCH = 3
    comptime ACT_DIM = 2

    var musig = _alloc(BATCH * 2 * ACT_DIM)
    var tgt = _alloc(BATCH * ACT_DIM)

    # arbitrary μ_raw / σ_raw and in-range target actions (|a| < max_action=1).
    for b in range(BATCH):
        for d in range(ACT_DIM):
            musig[b * 2 * ACT_DIM + d] = (
                Scalar[DT](0.5) * Scalar[DT]((b + d) % 4) - Scalar[DT](0.6)
            )  # μ_raw
            musig[b * 2 * ACT_DIM + ACT_DIM + d] = (
                Scalar[DT](0.3) * Scalar[DT]((b * 2 + d) % 3) - Scalar[DT](0.2)
            )  # σ_raw
            tgt[b * ACT_DIM + d] = (
                Scalar[DT](0.35) * Scalar[DT]((d + 1) % 3) - Scalar[DT](0.4)
            )

    var grad = _alloc(BATCH * 2 * ACT_DIM)
    var l0 = continuous_policy_loss_and_grad[BATCH, ACT_DIM](
        musig, tgt, Scalar[DT](1.0), grad,
    )
    assert_true(l0 == l0 and l0 < 1e30 and l0 > -1e30, "loss non-finite")

    var eps = Scalar[DT](1e-3)
    var max_err = Scalar[DT](0.0)
    for j in range(BATCH * 2 * ACT_DIM):
        var saved = musig[j]
        musig[j] = saved + eps
        var lp = _loss_only[BATCH, ACT_DIM](musig, tgt)
        musig[j] = saved - eps
        var lm = _loss_only[BATCH, ACT_DIM](musig, tgt)
        musig[j] = saved
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

    # grad_scale linearity.
    var grad2 = _alloc(BATCH * 2 * ACT_DIM)
    _ = continuous_policy_loss_and_grad[BATCH, ACT_DIM](
        musig, tgt, Scalar[DT](0.25), grad2,
    )
    for j in range(BATCH * 2 * ACT_DIM):
        assert_almost_equal(
            grad2[j], grad[j] * Scalar[DT](0.25), atol=1e-6, rtol=1e-5,
            msg="grad_scale not applied linearly",
        )
    print("grad_scale linearity: OK")

    musig.free(); tgt.free(); grad.free(); grad2.free()
    print("EZv2 continuous policy gradcheck: OK")
