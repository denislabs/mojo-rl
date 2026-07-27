"""Dreamer 4 imagination-RL losses — λ-returns + value TD + PMPO (Phase 4).

    pixi run mojo run -I . tests/nn/test_dreamer4_imag_rl_loss.mojo

Validates the three net-new Phase-4 loss pieces in isolation (no transformer):
  1. `lambda_returns` matches a hand-computed recurrence on a tiny example.
  2. `value_td_loss` overfits: SGD on value logits drives the twohot CE down and
     the twohot prediction toward the λ-return targets.
  3. `pmpo_policy_loss_backward` matches finite-difference gradients of the
     forward loss (the max-likelihood terms AND the reverse-KL prior).
  4. PMPO behavior: gradient descent on the policy logits raises the log-prob of
     the sampled action at POSITIVE-advantage states and lowers it at NEGATIVE-
     advantage states — the sign-of-advantage objective working as intended.
"""

from std.memory import alloc
from std.math import abs, log

from std.testing import assert_true, assert_almost_equal

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.dreamer4.imag_rl_loss import (
    lambda_returns,
    value_td_loss_cpu,
    value_td_loss_backward,
    pmpo_policy_loss_cpu,
    pmpo_policy_loss_backward,
)
from mojo_rl.deep_agents.dreamerv3.twohot import symexp_twohot_bins, twohot_pred
from mojo_rl.deep_agents.dreamerv3.dists_discrete import cat_fwd, UNIMIX


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _logp(
    logits: UnsafePointer[Scalar[DT], MutAnyOrigin], base: Int, C: Int, k: Int
) -> Float64:
    var sm = _alloc(C)
    var pp = _alloc(C)
    var r = cat_fwd[3](logits, base, UNIMIX, k, sm, pp)  # C==3 here
    return Float64(r[0])


def test_lambda_returns() raises:
    print("-- λ-returns recurrence")
    comptime B = 1
    comptime H = 4
    var rew = _alloc(B * H)
    var val = _alloc(B * H)
    var con = _alloc(B * H)
    var ret = _alloc(B * (H - 1))
    # rewards/values along a 4-step rollout; γ folded into con.
    var rews: InlineArray[Float64, 4] = [0.0, 1.0, 2.0, 3.0]
    var vals: InlineArray[Float64, 4] = [0.5, 0.6, 0.7, 0.8]
    comptime GAMMA = 0.997
    comptime LAM = 0.95
    for t in range(H):
        rew[t] = Scalar[DT](rews[t])
        val[t] = Scalar[DT](vals[t])
        con[t] = Scalar[DT](GAMMA)            # no termination

    lambda_returns[B, H](rew, val, con, Scalar[DT](LAM), ret)

    # hand recurrence ("arriving" convention — rew/con at t+1; heads trained on
    # the shifted arriving reward, so out_rew[t+1] is the reward for out_act[t]):
    #   R_{H-1}=v_{H-1};  R_t = r_{t+1} + live·[(1-λ)v_{t+1} + λ R_{t+1}]
    var rn = vals[3]
    var expected: InlineArray[Float64, 3] = [0.0, 0.0, 0.0]
    var t2 = H - 2
    while t2 >= 0:
        var live = GAMMA
        var interm = rews[t2 + 1] + (1.0 - LAM) * live * vals[t2 + 1]
        var cur = interm + live * LAM * rn
        expected[t2] = cur
        rn = cur
        t2 -= 1
    for t in range(H - 1):
        assert_almost_equal(
            Float64(ret[t]), expected[t], atol=1e-6,
            msg="λ-return mismatch",
        )
    print("   ret =", Float64(ret[0]), Float64(ret[1]), Float64(ret[2]), "OK")


def test_value_overfit() raises:
    print("-- value TD loss overfit")
    comptime B = 2
    comptime H = 3
    comptime BINS = 41
    comptime HM1 = H - 1
    var bins = _alloc(BINS)
    symexp_twohot_bins[BINS](bins, lo=Scalar[DT](-9.0))

    var vlogits = _alloc(B * H * BINS)
    for i in range(B * H * BINS):
        vlogits[i] = Scalar[DT](0.0)
    var ret = _alloc(B * HM1)
    var targets: InlineArray[Float64, 4] = [1.5, -0.7, 3.0, 0.2]
    for i in range(B * HM1):
        ret[i] = Scalar[DT](targets[i])

    var loss = _alloc(B * HM1)
    var grad = _alloc(B * H * BINS)
    var d_loss = _alloc(B * HM1)
    for i in range(B * HM1):
        d_loss[i] = Scalar[DT](1.0)

    var first = Float64(0.0)
    var last = Float64(0.0)
    for step in range(400):
        value_td_loss_cpu[B, H, BINS](vlogits, bins, ret, loss)
        var tot = Float64(0.0)
        for i in range(B * HM1):
            tot += Float64(loss[i])
        if step == 0:
            first = tot
        last = tot
        value_td_loss_backward[B, H, BINS](vlogits, bins, ret, d_loss, grad)
        for i in range(B * H * BINS):
            vlogits[i] -= Scalar[DT](0.5) * grad[i]

    print("   value CE", first, "->", last)
    # twohot CE floors at the two-hot target entropy (never 0); the real signal
    # is the predicted MEAN matching the targets (checked below).
    assert_true(last < 0.2 * first, "value loss must collapse to the twohot floor")
    # predictions recover the targets (twohot CE floors at target entropy, but
    # the predicted MEAN should match closely)
    for b in range(B):
        for t in range(HM1):
            var pred = Float64(twohot_pred[BINS](vlogits, (b * H + t) * BINS, bins))
            assert_almost_equal(
                pred, targets[b * HM1 + t], atol=0.1,
                msg="value pred should match λ-return",
            )
    print("   value predictions match targets OK")


def test_pmpo_gradcheck() raises:
    print("-- PMPO FD gradcheck")
    comptime B = 2
    comptime H = 3
    comptime NACT = 3
    comptime HM1 = H - 1
    comptime ALPHA = Scalar[DT](0.5)
    comptime BETA = Scalar[DT](0.3)

    var plog = _alloc(B * H * NACT)
    var prior = _alloc(B * H * NACT)
    var actions = _alloc(B * H)
    var adv = _alloc(B * HM1)
    # deterministic, distinct logits / prior / actions / advantages
    for i in range(B * H * NACT):
        plog[i] = Scalar[DT](0.2 * Float64(((i * 7) % 5) - 2))
        prior[i] = Scalar[DT](0.15 * Float64(((i * 3) % 4) - 1))
    var acts: InlineArray[Int, 6] = [0, 1, 2, 1, 0, 2]
    for i in range(B * H):
        actions[i] = Scalar[DT](Float64(acts[i]))
    var advs: InlineArray[Float64, 4] = [1.2, -0.5, 0.0, -2.0]  # mix of signs
    for i in range(B * HM1):
        adv[i] = Scalar[DT](advs[i])

    var grad = _alloc(B * H * NACT)
    pmpo_policy_loss_backward[B, H, NACT](
        plog, prior, actions, adv, ALPHA, BETA, Scalar[DT](1.0), grad
    )

    var eps = 1e-3
    var max_err = Float64(0.0)
    for i in range(B * H * NACT):
        var saved = plog[i]
        plog[i] = saved + Scalar[DT](eps)
        var lp = pmpo_policy_loss_cpu[B, H, NACT](
            plog, prior, actions, adv, ALPHA, BETA
        )
        plog[i] = saved - Scalar[DT](eps)
        var lm = pmpo_policy_loss_cpu[B, H, NACT](
            plog, prior, actions, adv, ALPHA, BETA
        )
        plog[i] = saved
        var fd = (lp - lm) / (2.0 * eps)
        var err = abs(fd - Float64(grad[i]))
        if err > max_err:
            max_err = err
    print("   max |FD − analytic| =", max_err)
    assert_true(max_err < 5e-3, "PMPO backward must match finite differences")


def test_pmpo_behavior() raises:
    print("-- PMPO sign-of-advantage behavior")
    comptime B = 2
    comptime H = 2          # one advantaged state per sequence
    comptime NACT = 3
    comptime HM1 = H - 1
    comptime ALPHA = Scalar[DT](0.5)
    comptime BETA = Scalar[DT](0.0)   # isolate the ML term (no prior pull)

    var plog = _alloc(B * H * NACT)
    var prior = _alloc(B * H * NACT)
    var actions = _alloc(B * H)
    var adv = _alloc(B * HM1)
    for i in range(B * H * NACT):
        plog[i] = Scalar[DT](0.0)
        prior[i] = Scalar[DT](0.0)
    # seq 0: action 0 at a POSITIVE-advantage state ⇒ logp(0) should RISE
    # seq 1: action 2 at a NEGATIVE-advantage state ⇒ logp(2) should FALL
    actions[0 * H + 0] = Scalar[DT](0.0)
    actions[1 * H + 0] = Scalar[DT](2.0)
    adv[0 * HM1 + 0] = Scalar[DT](1.0)
    adv[1 * HM1 + 0] = Scalar[DT](-1.0)

    var lp_pos_0 = _logp(plog, (0 * H + 0) * NACT, NACT, 0)
    var lp_neg_0 = _logp(plog, (1 * H + 0) * NACT, NACT, 2)

    var grad = _alloc(B * H * NACT)
    for step in range(50):
        pmpo_policy_loss_backward[B, H, NACT](
            plog, prior, actions, adv, ALPHA, BETA, Scalar[DT](1.0), grad
        )
        for i in range(B * H * NACT):
            plog[i] -= Scalar[DT](0.2) * grad[i]

    var lp_pos_1 = _logp(plog, (0 * H + 0) * NACT, NACT, 0)
    var lp_neg_1 = _logp(plog, (1 * H + 0) * NACT, NACT, 2)
    print("   positive-adv logp:", lp_pos_0, "->", lp_pos_1)
    print("   negative-adv logp:", lp_neg_0, "->", lp_neg_1)
    assert_true(lp_pos_1 > lp_pos_0 + 0.1, "positive-advantage action up-weighted")
    assert_true(lp_neg_1 < lp_neg_0 - 0.1, "negative-advantage action down-weighted")


def main() raises:
    print("=" * 70)
    print("Dreamer 4 imagination-RL losses (Phase 4)")
    print("=" * 70)
    test_lambda_returns()
    test_value_overfit()
    test_pmpo_gradcheck()
    test_pmpo_behavior()
    print("=" * 70)
    print("ALL PASSED — λ-returns + value TD + PMPO (eq. 10 + 11)")
    print("=" * 70)
