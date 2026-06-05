"""EfficientZeroV2 loss primitives — SimSiam temporal-consistency.

The genuinely-new EZv2 objective over MuZero is the **negative cosine
similarity** between the online predictor output and the stop-gradient target
projection (Chen & He, "Exploring Simple Siamese Representation Learning", 2021;
EZv2 paper Eq. 3 λ_G term):

    cos(p, t) = (p · t) / (‖p‖ · ‖t‖)
    L_G       = − cos(p, sg(t))                    (stop-grad on the target t)
    ∂(−cos)/∂p_i = cos · p_i / ‖p‖²  −  t_i / (‖p‖ · ‖t‖)

Only the **online** branch ``p = h_pred(g_proj(z_k))`` receives a gradient; the
**target** branch ``t = g_proj(h(obs_k))`` is detached (no grad written, the
unroll never backprops it). The analytic gradient above is what makes the
manual reverse-scan in ``blocks.mojo`` trustworthy — it is gradchecked against
finite differences in ``tests/deep_agents2/test_ezv2_consistency_gradcheck.mojo``.

``grad_scale`` folds the caller's per-unroll-step weight ``1/K`` (consistency is
summed over the ``k = 1..K`` dynamics steps, not ``K+1`` — there is no
consistency at the root), the ``1/BATCH`` mean, and the loss coefficient
``λ_G`` into the gradient in one place, exactly like ``soft_ce_loss_and_grad``.
"""

from std.math import sqrt

from mojo_rl.nn2.constants import DT


comptime _COS_EPS: Scalar[DT] = Scalar[DT](1e-12)


def consistency_loss_and_grad[
    BATCH: Int, DIM: Int,
](
    p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    t: UnsafePointer[Scalar[DT], MutAnyOrigin],
    grad_scale: Scalar[DT],
    mut grad_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
) -> Scalar[DT]:
    """Negative-cosine SimSiam consistency over ``BATCH`` rows of ``DIM``.

    ``p`` is the online predictor output, ``t`` the stop-grad target projection.
    Returns the **summed** loss ``Σ_b −cos(p_b, t_b)`` and writes the analytic
    online gradient ``grad_p[b,i] = grad_scale · (cos·p_i/‖p‖² − t_i/(‖p‖·‖t‖))``.
    No gradient is produced for ``t`` (it is detached).
    """
    var total = Scalar[DT](0.0)
    for b in range(BATCH):
        var base = b * DIM
        var sum_pp = Scalar[DT](0.0)
        var sum_tt = Scalar[DT](0.0)
        var dot = Scalar[DT](0.0)
        for i in range(DIM):
            var pi = p[base + i]
            var ti = t[base + i]
            sum_pp += pi * pi
            sum_tt += ti * ti
            dot += pi * ti
        var np = sqrt(sum_pp + _COS_EPS)
        var nt = sqrt(sum_tt + _COS_EPS)
        var cos = dot / (np * nt)
        total += -cos
        # ∂(−cos)/∂p_i = cos·p_i/‖p‖² − t_i/(‖p‖·‖t‖)
        var inv_np2 = Scalar[DT](1.0) / (np * np)
        var inv_npnt = Scalar[DT](1.0) / (np * nt)
        for i in range(DIM):
            var gi = cos * p[base + i] * inv_np2 - t[base + i] * inv_npnt
            grad_p[base + i] = grad_scale * gi
    return total
