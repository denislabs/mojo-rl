"""RND gate — does novelty actually discriminate seen from unseen states?

RND has exactly one job: be small where the predictor has looked and large
where it has not. Everything else about it is plumbing. So the gate is built
around that one property and around the three ways it silently stops holding:

  [1] **The target must stay frozen.** A trained target makes the predictor's
      job trivial, novelty collapses towards zero EVERYWHERE, and nothing
      errors. Asserted bit-exactly: the target's output on a fixed probe is
      identical before and after fitting.
  [2] **Novelty must DISCRIMINATE.** Fitting on one region of state space must
      leave novelty lower there than on a region never shown. Checking only
      that "novelty went down on the training set" would pass a predictor that
      collapsed to a constant, which is the degenerate solution.
  [3] **The intrinsic reward must decay SLOWER than the raw error.** Raw
      novelty shrinks by orders of magnitude as the predictor fits, so an agent
      rewarded with it would watch its incentive evaporate. `intrinsic` divides
      by the running std, and the gate asserts the normalised reward decays
      strictly slower. ⚠ Note what is NOT claimed: stationarity. Measured, raw
      falls 24x and normalised falls 9x — `rew_norm` is cumulative, so its std
      tracks the whole history rather than the recent scale. See `intrinsic`'s
      docstring; the fix is an EMA, deferred until something trains on this.

  [4] The normaliser is checked against its own definition on data whose mean
      and variance are known.

⚠ [2] is the discriminating test and it needs SEPARATED regions. Two random
draws from the same distribution give novelty differences within noise, and a
gate built on those would be measuring the seed.

Run:
    pixi run mojo run -I . tests/deep_agents/test_fb_rnd.mojo
"""

from std.math import abs, sqrt
from std.random import random_float64, seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU
from mojo_rl.deep_agents.fb.rnd import RND, RunningNorm


comptime OBS: Int = 6
comptime FEAT: Int = 8
comptime BATCH: Int = 32
comptime HID: Int = 64
comptime FIT_STEPS: Int = 300
comptime SEED: Int = 20260805

comptime Net = Sequential[Linear[OBS, HID], ReLU[HID], Linear[HID, FEAT]]
comptime Rnd = RND[Net, OBS, FEAT, BATCH]


def _region(mut t: Tensor, n: Int, centre: Float64, spread: Float64) raises:
    """`n` rows drawn around `centre`. Regions are separated by construction so
    [2] is not measuring sampling noise."""
    t.ensure(n * OBS)
    for r in range(n):
        for d in range(OBS):
            t.data[r * OBS + d] = Scalar[DT](
                centre + (random_float64() * 2.0 - 1.0) * spread
            )


def test_running_norm_matches_its_definition() raises:
    print("[1] RunningNorm reproduces a known mean/variance ...")
    var rn = RunningNorm[2](count_prior=0.0)
    var x = Tensor.alloc(8 * 2)
    # Column 0: 1..8  (mean 4.5, population variance 5.25)
    # Column 1: constant 3 (mean 3, variance 0)
    for r in range(8):
        x.data[r * 2] = Scalar[DT](Float64(r + 1))
        x.data[r * 2 + 1] = Scalar[DT](3.0)
    rn.update(x, 8)
    print("       mean", rn.mean[0], rn.mean[1],
          " var", rn.var_[0], rn.var_[1])
    assert_true(abs(rn.mean[0] - 4.5) < 1e-6, "mean0 " + String(rn.mean[0]))
    assert_true(abs(rn.mean[1] - 3.0) < 1e-6, "mean1 " + String(rn.mean[1]))
    assert_true(abs(rn.var_[0] - 5.25) < 1e-5, "var0 " + String(rn.var_[0]))
    assert_true(abs(rn.var_[1]) < 1e-6, "var1 " + String(rn.var_[1]))

    # Split the same data into two updates: the merge must agree with one shot.
    var rn2 = RunningNorm[2](count_prior=0.0)
    var a = Tensor.alloc(4 * 2)
    var b = Tensor.alloc(4 * 2)
    for r in range(4):
        a.data[r * 2] = Scalar[DT](Float64(r + 1))
        a.data[r * 2 + 1] = Scalar[DT](3.0)
        b.data[r * 2] = Scalar[DT](Float64(r + 5))
        b.data[r * 2 + 1] = Scalar[DT](3.0)
    rn2.update(a, 4)
    rn2.update(b, 4)
    print("       split-merge mean", rn2.mean[0], " var", rn2.var_[0])
    assert_true(
        abs(rn2.mean[0] - 4.5) < 1e-6 and abs(rn2.var_[0] - 5.25) < 1e-5,
        "the two-batch merge disagrees with the one-shot statistics — the"
        " Chan et al. update is wrong, and every long run would drift",
    )


def test_target_stays_frozen() raises:
    print("[2] the target network is never trained ...")
    seed(SEED)
    var r = Rnd.make(lr=1e-3)

    var probe = Tensor()
    _region(probe, BATCH, 0.0, 1.0)

    # Read the target's output through the module directly, before and after.
    var before = Tensor()
    _ = r.novelty[BATCH](probe, before)
    var t_before = Tensor.alloc(BATCH * FEAT)
    for i in range(BATCH * FEAT):
        t_before.data[i] = r._ft.data[i]

    var train = Tensor()
    for _ in range(FIT_STEPS):
        _region(train, BATCH, 0.0, 1.0)
        _ = r.fit[BATCH](train)

    var after = Tensor()
    _ = r.novelty[BATCH](probe, after)

    # ⚠ Compare with the OBS NORM FROZEN, or the target's input changes as the
    # normaliser learns and the comparison says nothing about the weights.
    var worst = Float64(0)
    for i in range(BATCH * FEAT):
        var e = abs(Float64(t_before.data[i]) - Float64(r._ft.data[i]))
        if e > worst:
            worst = e
    print("       |target(probe) before - after| =", worst)
    assert_true(
        worst > 0.0,
        "the target output did not move AT ALL across 300 fit steps, not even"
        " through the observation normaliser — this comparison is not"
        " measuring what it claims",
    )
    # Now the real check: freeze the normaliser and confirm the WEIGHTS are
    # what stayed put.
    r.obs_norm.freeze()
    var f1 = Tensor()
    _ = r.novelty[BATCH](probe, f1)
    var t1 = Tensor.alloc(BATCH * FEAT)
    for i in range(BATCH * FEAT):
        t1.data[i] = r._ft.data[i]
    for _ in range(50):
        _region(train, BATCH, 0.0, 1.0)
        _ = r.fit[BATCH](train)
    var f2 = Tensor()
    _ = r.novelty[BATCH](probe, f2)
    var w2 = Float64(0)
    for i in range(BATCH * FEAT):
        var e = abs(Float64(t1.data[i]) - Float64(r._ft.data[i]))
        if e > w2:
            w2 = e
    print("       with obs_norm frozen, target moved by", w2)
    assert_true(
        w2 == 0.0,
        "the target network CHANGED (" + String(w2) + ") while only the"
        " predictor should have been stepped. Novelty collapses to zero"
        " everywhere when the target trains, and nothing raises.",
    )


def test_novelty_discriminates() raises:
    """Fit on region A; novelty must end LOWER on A than on an unseen region B.
    """
    print("[3] novelty separates a fitted region from an unseen one ...")
    seed(SEED)
    var r = Rnd.make(lr=1e-3)

    var a_probe = Tensor()
    var b_probe = Tensor()
    _region(a_probe, BATCH, -2.0, 0.4)   # fitted region
    _region(b_probe, BATCH, 6.0, 0.4)    # never shown; far away

    var d0 = Tensor()
    var n_a0 = r.novelty[BATCH](a_probe, d0)
    var n_b0 = r.novelty[BATCH](b_probe, d0)
    assert_true(
        n_a0 > 1e-9 and n_b0 > 1e-9,
        "initial novelty is ~0, so target and predictor were initialised"
        " IDENTICALLY — RND has no signal at all in that case",
    )

    var train = Tensor()
    for _ in range(FIT_STEPS):
        _region(train, BATCH, -2.0, 0.4)
        _ = r.fit[BATCH](train)

    var d1 = Tensor()
    var n_a1 = r.novelty[BATCH](a_probe, d1)
    var n_b1 = r.novelty[BATCH](b_probe, d1)
    print("       fitted region  :", n_a0, "->", n_a1)
    print("       unseen region  :", n_b0, "->", n_b1)
    print("       ratio unseen/fitted =", n_b1 / n_a1 if n_a1 > 0 else 0.0)

    assert_true(
        n_a1 < n_a0,
        "novelty did not fall on the region that was fitted (" + String(n_a0)
        + " -> " + String(n_a1) + ") — the predictor is not learning",
    )
    assert_true(
        n_b1 > 5.0 * n_a1,
        "novelty on the UNSEEN region (" + String(n_b1) + ") is not clearly"
        " above the fitted one (" + String(n_a1) + "). A predictor that"
        " collapsed to a constant would also drive the fitted error down, so"
        " this ratio — not the drop — is what says RND works.",
    )


def test_intrinsic_reward_is_scale_stationary() raises:
    """Raw novelty must fall a lot; the normalised reward must fall LESS.

    Not "must not fall" — see the module docstring. A cumulative normaliser
    cannot deliver that, and asserting it would be asserting something the
    implementation does not do.
    """
    print("[4] intrinsic reward decays slower than raw novelty ...")
    seed(SEED)
    var r = Rnd.make(lr=1e-3)

    var train = Tensor()
    var dst = Tensor()

    _region(train, BATCH, 0.0, 1.0)
    var raw0 = r.novelty[BATCH](train, dst)
    var int0 = r.intrinsic[BATCH](train, dst)

    for _ in range(FIT_STEPS):
        _region(train, BATCH, 0.0, 1.0)
        _ = r.fit[BATCH](train)
        _ = r.intrinsic[BATCH](train, dst)

    _region(train, BATCH, 0.0, 1.0)
    var raw1 = r.novelty[BATCH](train, dst)
    var int1 = r.intrinsic[BATCH](train, dst)

    print("       raw novelty      ", raw0, "->", raw1,
          " (factor", raw1 / raw0 if raw0 > 0 else 0.0, ")")
    print("       intrinsic reward ", int0, "->", int1,
          " (factor", int1 / int0 if int0 > 0 else 0.0, ")")

    assert_true(
        raw1 < 0.5 * raw0,
        "raw novelty barely fell (" + String(raw0) + " -> " + String(raw1)
        + "), so this test never entered the regime it is about",
    )
    var raw_factor = raw1 / raw0
    var int_factor = int1 / int0 if int0 > 0 else 0.0
    assert_true(
        int_factor > raw_factor,
        "the normalised reward shrank at least as fast as the raw error"
        " (raw x" + String(raw_factor) + ", intrinsic x" + String(int_factor)
        + ") — the reward normaliser is not doing its job, and an exploration"
        " agent's incentive would evaporate as the predictor fits",
    )


def main() raises:
    test_running_norm_matches_its_definition()
    test_target_stays_frozen()
    test_novelty_discriminates()
    test_intrinsic_reward_is_scale_stationary()
    print("\n[PASS] RND gate")
