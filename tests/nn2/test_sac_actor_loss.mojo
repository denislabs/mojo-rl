"""SAC actor backward CPU tests — Phase 7.3.

Covers:
  - squashed_gaussian_sample matches hand-computed values on a 2x1 case.
  - sac_actor_backward zeros grad_log_std cells when log_std is clamped.
  - sac_actor_backward FD gradcheck: fix z + grad_action + alpha +
    action_scale; verify analytical grad_actor_output matches numerical
    gradients of the loss with respect to actor_output entries.

Loss for gradcheck (replicates the form sac_actor_backward expects to
have inverted analytically):

    L = mean_b( alpha * log_prob_b - sum_j grad_action[b, j] * action[b, j] )

This is the chain when grad_action plays the role of `-d_min_q / d_a`
arriving from the critic backwards — i.e. we use a linear "virtual loss"
in the action that, summed with the entropy term, reproduces the SAC
actor loss math. The analytical grad from sac_actor_backward should
match d L / d actor_output[b, k] for k ∈ [0, 2*ACT).
"""

from std.math import abs as fabs, exp as fexp, log as flog, tanh as ftanh
from std.memory import alloc
from std.testing import assert_almost_equal, assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.loss.sac_actor_loss import (
    squashed_gaussian_sample,
    sac_actor_backward,
    sac_actor_loss_value,
    LOG_STD_MIN,
    LOG_STD_MAX,
    EPS_TANH_CORR,
    LOG_2PI,
)


def _scalar_log_prob(
    mu: Scalar[DT], ls: Scalar[DT], z: Scalar[DT], c: Scalar[DT]
) -> Scalar[DT]:
    """Hand re-implements squashed-Gaussian log_prob for one (mu, ls, z)."""
    var ls_c = ls
    if ls_c < LOG_STD_MIN:
        ls_c = LOG_STD_MIN
    elif ls_c > LOG_STD_MAX:
        ls_c = LOG_STD_MAX
    var std = fexp(ls_c)
    var pre = mu + std * z
    var y = ftanh(pre)
    var one_minus_y2 = Scalar[DT](1.0) - y * y
    var corr = c * one_minus_y2 + EPS_TANH_CORR
    return (
        Scalar[DT](-0.5) * z * z
        - ls_c
        - Scalar[DT](0.5) * LOG_2PI
        - flog(corr)
    )


def _scalar_action(
    mu: Scalar[DT], ls: Scalar[DT], z: Scalar[DT], c: Scalar[DT]
) -> Scalar[DT]:
    var ls_c = ls
    if ls_c < LOG_STD_MIN:
        ls_c = LOG_STD_MIN
    elif ls_c > LOG_STD_MAX:
        ls_c = LOG_STD_MAX
    var std = fexp(ls_c)
    var pre = mu + std * z
    return c * ftanh(pre)


def test_squashed_sample_hand_check() raises:
    """One (B=2, ACT=1) case, hand-compute action + log_prob."""
    var ao = alloc[Scalar[DT]](2 * 2)  # 2*ACT=2
    var z = alloc[Scalar[DT]](2 * 1)
    var act = alloc[Scalar[DT]](2 * 1)
    var lp = alloc[Scalar[DT]](2)

    # b=0: mu=0.5, ls=-0.5, z=0.3, c=2.0
    ao[0] = 0.5; ao[1] = -0.5
    z[0] = 0.3
    # b=1: mu=-0.7, ls=0.2, z=-1.1
    ao[2] = -0.7; ao[3] = 0.2
    z[1] = -1.1

    var c: Scalar[DT] = 2.0
    var ao_t = TileTensor(ao, row_major[2, 2]())
    var z_t = TileTensor(z, row_major[2, 1]())
    var act_t = TileTensor(act, row_major[2, 1]())
    var lp_t = TileTensor(lp, row_major[2]())
    squashed_gaussian_sample[1, 2](ao_t, z_t, c, act_t, lp_t)

    var a0_expected = _scalar_action(0.5, -0.5, 0.3, c)
    var lp0_expected = _scalar_log_prob(0.5, -0.5, 0.3, c)
    var a1_expected = _scalar_action(-0.7, 0.2, -1.1, c)
    var lp1_expected = _scalar_log_prob(-0.7, 0.2, -1.1, c)
    assert_almost_equal(act[0], a0_expected, atol=1e-6)
    assert_almost_equal(lp[0], lp0_expected, atol=1e-6)
    assert_almost_equal(act[1], a1_expected, atol=1e-6)
    assert_almost_equal(lp[1], lp1_expected, atol=1e-6)

    print("  test_squashed_sample_hand_check PASSED")

    ao.free(); z.free(); act.free(); lp.free()


def test_grad_log_std_zeroed_when_clamped() raises:
    """If actor_output[b, ACT+j] is outside [-5, 2], grad_log_std cell = 0."""
    var ao = alloc[Scalar[DT]](2 * 2)
    var z = alloc[Scalar[DT]](2 * 1)
    var ga = alloc[Scalar[DT]](2 * 1)
    var gao = alloc[Scalar[DT]](2 * 2)

    # b=0: log_std = -10.0 (clamped low) — gls cell must be 0
    ao[0] = 0.5; ao[1] = -10.0
    # b=1: log_std = +5.0 (clamped high) — gls cell must be 0
    ao[2] = -0.7; ao[3] = 5.0
    z[0] = 0.3; z[1] = -1.1
    ga[0] = 0.4; ga[1] = -0.2

    var c: Scalar[DT] = 2.0
    var alpha: Scalar[DT] = 0.2
    var ao_t = TileTensor(ao, row_major[2, 2]())
    var z_t = TileTensor(z, row_major[2, 1]())
    var ga_t = TileTensor(ga, row_major[2, 1]())
    var gao_t = TileTensor(gao, row_major[2, 2]())
    sac_actor_backward[1, 2](ao_t, z_t, ga_t, alpha, c, gao_t)

    # log_std cells = 0
    assert_almost_equal(gao[1], 0.0, atol=1e-8)
    assert_almost_equal(gao[3], 0.0, atol=1e-8)
    # mu cells nonzero (in general)
    assert_true(fabs(gao[0]) > 1e-9 or fabs(gao[2]) > 1e-9,
                "expected some grad on mu side")

    print("  test_grad_log_std_zeroed_when_clamped PASSED")

    ao.free(); z.free(); ga.free(); gao.free()


def test_fd_gradcheck() raises:
    """FD gradcheck on sac_actor_backward.

    Define virtual loss
        L(actor_output) = mean_b( alpha * log_prob_b
                                  - sum_j grad_action[b, j] * action[b, j] )
                        = alpha * mean(log_prob)
                          - mean( sum_j grad_action[b, j] * action[b, j] )

    Then d L / d actor_output[b, k] should match the value returned by
    sac_actor_backward (where grad_action is the analytical d_L/d_a term
    seen by the helper).

    Sign of the action term: in SAC, grad_action passed to the helper IS
    d_L/d_a (already including −1/BATCH and min-mask). So in our virtual
    loss, the +sum_j grad_action·action term has the same chain. We
    encode that as -sum_j (-grad_action[b, j])·action[b, j] ≡
    +sum_j grad_action·action — i.e. caller's `grad_action` IS d_L/d_a.

    L_b = α · log_prob_b + Σ_j grad_action[b, j] · action[b, j]
    L   = mean_b L_b
    """
    comptime BATCH = 3
    comptime ACT = 2
    var ao = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var z = alloc[Scalar[DT]](BATCH * ACT)
    var ga = alloc[Scalar[DT]](BATCH * ACT)
    var gao = alloc[Scalar[DT]](BATCH * 2 * ACT)

    # Hand-picked values that exercise mid-range tanh + non-zero z.
    # b=0 (mu0, mu1, ls0, ls1), b=1, b=2.
    ao[0]  =  0.3; ao[1]  = -0.6; ao[2]  = -0.5; ao[3]  =  0.1
    ao[4]  =  0.8; ao[5]  =  0.0; ao[6]  = -1.0; ao[7]  = -0.3
    ao[8]  = -0.4; ao[9]  =  0.7; ao[10] = -0.2; ao[11] = -0.8

    z[0] =  0.5; z[1] = -0.3
    z[2] = -1.1; z[3] =  0.2
    z[4] =  0.9; z[5] = -0.7

    ga[0] =  0.4;  ga[1] = -0.15
    ga[2] = -0.25; ga[3] =  0.5
    ga[4] =  0.1;  ga[5] = -0.3

    var c: Scalar[DT] = 2.0
    var alpha: Scalar[DT] = 0.18

    # Analytical
    var ao_t = TileTensor(ao, row_major[BATCH, 2 * ACT]())
    var z_t = TileTensor(z, row_major[BATCH, ACT]())
    var ga_t = TileTensor(ga, row_major[BATCH, ACT]())
    var gao_t = TileTensor(gao, row_major[BATCH, 2 * ACT]())
    sac_actor_backward[ACT, BATCH](ao_t, z_t, ga_t, alpha, c, gao_t)

    # Numerical: probe each actor_output entry.
    var eps: Scalar[DT] = 1e-3
    var act_scratch = alloc[Scalar[DT]](BATCH * ACT)
    var lp_scratch = alloc[Scalar[DT]](BATCH)
    var act_t = TileTensor(act_scratch, row_major[BATCH, ACT]())
    var lp_t = TileTensor(lp_scratch, row_major[BATCH]())

    var max_rel: Scalar[DT] = 0.0
    var max_abs: Scalar[DT] = 0.0
    for idx in range(BATCH * 2 * ACT):
        var orig = ao[idx]
        # +eps
        ao[idx] = orig + eps
        squashed_gaussian_sample[ACT, BATCH](ao_t, z_t, c, act_t, lp_t)
        var lp_plus: Scalar[DT] = 0.0
        for b in range(BATCH):
            # Entropy term is (alpha / BATCH) per sample (mean over batch).
            # Action term uses grad_action as-is — it already encodes
            # the /BATCH baked in by the production critic backward.
            lp_plus += (alpha / Scalar[DT](BATCH)) * lp_scratch[b]
            for j in range(ACT):
                lp_plus += ga[b * ACT + j] * act_scratch[b * ACT + j]
        # -eps
        ao[idx] = orig - eps
        squashed_gaussian_sample[ACT, BATCH](ao_t, z_t, c, act_t, lp_t)
        var lp_minus: Scalar[DT] = 0.0
        for b in range(BATCH):
            lp_minus += (alpha / Scalar[DT](BATCH)) * lp_scratch[b]
            for j in range(ACT):
                lp_minus += ga[b * ACT + j] * act_scratch[b * ACT + j]
        # Restore
        ao[idx] = orig
        var num_grad = (lp_plus - lp_minus) / (Scalar[DT](2.0) * eps)
        var ana_grad = gao[idx]
        var abs_err = fabs(num_grad - ana_grad)
        var denom = fabs(num_grad) + fabs(ana_grad) + Scalar[DT](1e-6)
        var rel_err = abs_err / denom
        if rel_err > max_rel:
            max_rel = rel_err
        if abs_err > max_abs:
            max_abs = abs_err

    print(
        "  test_fd_gradcheck: max_abs=", max_abs, "  max_rel=", max_rel
    )
    # 1e-2 tolerance is consistent with the 3-layer FD precision floor on
    # FP32 (see feedback_fd_eps_deep_chains memory).
    assert_true(max_rel < 1e-2, "FD gradcheck rel-err too high")
    print("  test_fd_gradcheck PASSED")

    ao.free(); z.free(); ga.free(); gao.free()
    act_scratch.free(); lp_scratch.free()


def main() raises:
    print("=" * 70)
    print("SAC actor backward CPU tests (Phase 7.3)")
    print("=" * 70)
    test_squashed_sample_hand_check()
    test_grad_log_std_zeroed_when_clamped()
    test_fd_gradcheck()
    print("All SAC actor loss tests PASSED.")
