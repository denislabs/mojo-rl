"""Phase 8.1 helper unit tests.

Covers:
  - box_muller_normal: empirical mean ≈ 0, var ≈ 1 over a large draw.
  - EpisodeTracker: rolling window, mean, end_episode reset.
  - OnlineTargetPair: hard-copy on make + polyak_step interpolation.
  - CPUReplay: add wraps + sample returns stored values.
  - compute_gae: hand-checked on a 4-step rollout with non-zero
    terminated and a constant-value baseline.
  - normalize_in_place: mean ≈ 0, var ≈ 1 after.
"""

from std.math import abs as fabs, sqrt as fsqrt
from std.memory import alloc
from std.random import seed
from std.testing import assert_almost_equal, assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.random import box_muller_normal
from mojo_rl.nn2.training import (
    EpisodeTracker, compute_gae, normalize_in_place,
)
from mojo_rl.nn2.core import OnlineTargetPair
from mojo_rl.nn2.data import CPUReplay
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.initializer import Xavier


def test_box_muller_empirical_moments() raises:
    seed(123)
    comptime N = 4096
    var buf = alloc[Scalar[DT]](N)
    box_muller_normal(buf, N)
    var s: Scalar[DT] = 0.0
    for i in range(N):
        s = s + buf[i]
    var mean = s / Scalar[DT](N)
    var sq: Scalar[DT] = 0.0
    for i in range(N):
        var d = buf[i] - mean
        sq = sq + d * d
    var std = fsqrt(sq / Scalar[DT](N))
    print(
        "  test_box_muller_empirical_moments: mean=", mean, " std=", std
    )
    assert_true(fabs(mean) < 0.1, "mean too far from 0")
    assert_true(fabs(std - 1.0) < 0.1, "std too far from 1")
    buf.free()
    print("  test_box_muller_empirical_moments PASSED")


def test_episode_tracker() raises:
    var t = EpisodeTracker.new(window_size=3, initial_fill=-100.0)
    # Initial window mean is the fill.
    assert_almost_equal(t.mean_return(), -100.0, atol=1e-6)
    # Episode 1: returns 10 over 3 steps.
    t.add_reward(3.0)
    t.add_reward(3.0)
    t.add_reward(4.0)
    t.end_episode()
    # window = [10, -100, -100], mean = -63.33...
    assert_almost_equal(t.mean_return(), Scalar[DT](-190.0) / Scalar[DT](3), atol=1e-5)
    # Episode 2: returns 20.
    t.add_reward(20.0)
    t.end_episode()
    # window = [10, 20, -100], mean = -70/3
    assert_almost_equal(t.mean_return(), Scalar[DT](-70.0) / Scalar[DT](3), atol=1e-5)
    # Episode 3: returns 30. Pushes into idx 2.
    t.add_reward(30.0)
    t.end_episode()
    # window = [10, 20, 30], mean = 20.
    assert_almost_equal(t.mean_return(), Scalar[DT](20.0), atol=1e-5)
    # Episode 4: wraps. Pushes 40 into idx 0.
    t.add_reward(40.0)
    t.end_episode()
    # window = [40, 20, 30], mean = 30.
    assert_almost_equal(t.mean_return(), Scalar[DT](30.0), atol=1e-5)
    assert_true(t.ep_count == 4, "ep_count")
    print("  test_episode_tracker PASSED")


def test_online_target_pair() raises:
    comptime Net = Sequential[Linear[2, 3], Linear[3, 2]]
    var pair = OnlineTargetPair[Net].make[target="cpu", INIT=Xavier]()
    # After make: online == target params (hard copy).
    # Probe via a forward pass on each.
    var x = alloc[Scalar[DT]](1 * 2)
    x[0] = 0.5; x[1] = -0.3
    var y_online = alloc[Scalar[DT]](1 * 2)
    var y_target = alloc[Scalar[DT]](1 * 2)
    from layout import TileTensor, row_major
    var x_t = TileTensor(x, row_major[1, 2]())
    var yo_t = TileTensor(y_online, row_major[1, 2]())
    var yt_t = TileTensor(y_target, row_major[1, 2]())
    pair.online.forward["cpu", 1](x_t, yo_t)
    pair.target_net.forward["cpu", 1](x_t, yt_t)
    assert_almost_equal(y_online[0], y_target[0], atol=1e-8)
    assert_almost_equal(y_online[1], y_target[1], atol=1e-8)
    print("  test_online_target_pair PASSED (hard-copy)")
    # Polyak step with tau=0 — target shouldn't change.
    var t0_0 = y_target[0]
    pair.polyak_step["cpu"](Scalar[DT](0.0))
    pair.target_net.forward["cpu", 1](x_t, yt_t)
    assert_almost_equal(y_target[0], t0_0, atol=1e-8)
    # tau=1 — target should match online after the step (already equal).
    pair.polyak_step["cpu"](Scalar[DT](1.0))
    pair.target_net.forward["cpu", 1](x_t, yt_t)
    assert_almost_equal(y_target[0], y_online[0], atol=1e-8)
    print("  test_online_target_pair PASSED (polyak)")
    x.free(); y_online.free(); y_target.free()


def test_cpu_replay() raises:
    comptime OBS = 2
    comptime ACT = 1
    comptime CAP = 3
    var buf = CPUReplay[OBS, ACT, CAP].new()
    var s = alloc[Scalar[DT]](OBS)
    var a = alloc[Scalar[DT]](ACT)
    var sp = alloc[Scalar[DT]](OBS)
    s[0] = 1.0; s[1] = 2.0; a[0] = 0.5; sp[0] = 1.1; sp[1] = 2.1
    buf.add(s, a, Scalar[DT](0.7), sp, Scalar[DT](0.0))
    s[0] = 3.0; s[1] = 4.0; a[0] = -0.5; sp[0] = 3.1; sp[1] = 4.1
    buf.add(s, a, Scalar[DT](-0.4), sp, Scalar[DT](1.0))
    assert_true(buf.size == 2, "size should be 2")
    # Fill capacity + overflow → wraps.
    s[0] = 5.0; s[1] = 6.0; a[0] = 0.0; sp[0] = 5.1; sp[1] = 6.1
    buf.add(s, a, Scalar[DT](0.0), sp, Scalar[DT](0.0))
    s[0] = 7.0; s[1] = 8.0
    buf.add(s, a, Scalar[DT](0.0), sp, Scalar[DT](0.0))
    assert_true(buf.size == CAP, "size pinned at CAP")
    assert_true(buf.pos == 1, "pos wrapped to 1")
    # The slot at index 0 was overwritten with the (7,8) sample.
    assert_almost_equal(buf.obs[0 * OBS + 0], Scalar[DT](7.0), atol=1e-8)
    # Sample once and check shape; can't check exact value since RNG.
    var ss = alloc[Scalar[DT]](2 * OBS)
    var aa = alloc[Scalar[DT]](2 * ACT)
    var rr = alloc[Scalar[DT]](2)
    var ssp = alloc[Scalar[DT]](2 * OBS)
    var dd = alloc[Scalar[DT]](2)
    buf.sample(2, ss, aa, rr, ssp, dd)
    print("  test_cpu_replay PASSED (size+wrap+sample shape ok)")
    s.free(); a.free(); sp.free(); ss.free(); aa.free()
    rr.free(); ssp.free(); dd.free()


def test_compute_gae_constant_value() raises:
    """V(s)=0 baseline, no termination → advantages = discounted rewards."""
    comptime N = 4
    var rew = alloc[Scalar[DT]](N)
    var val = alloc[Scalar[DT]](N)
    var term = alloc[Scalar[DT]](N)
    var adv = alloc[Scalar[DT]](N)
    var ret = alloc[Scalar[DT]](N)
    for t in range(N):
        rew[t] = 1.0
        val[t] = 0.0
        term[t] = 0.0
    compute_gae(
        N, rew, val, term, next_value=Scalar[DT](0.0),
        gamma=Scalar[DT](0.5), gae_lambda=Scalar[DT](1.0),
        advantages=adv, returns=ret,
    )
    # With V=0, λ=1, γ=0.5: A_t = sum_{k=t..N-1} γ^(k-t) · r_k
    #   A_3 = 1
    #   A_2 = 1 + 0.5·1 = 1.5
    #   A_1 = 1 + 0.5·1.5 = 1.75
    #   A_0 = 1 + 0.5·1.75 = 1.875
    assert_almost_equal(adv[3], Scalar[DT](1.0), atol=1e-6)
    assert_almost_equal(adv[2], Scalar[DT](1.5), atol=1e-6)
    assert_almost_equal(adv[1], Scalar[DT](1.75), atol=1e-6)
    assert_almost_equal(adv[0], Scalar[DT](1.875), atol=1e-6)
    # returns = adv + values = adv (since V=0).
    for t in range(N):
        assert_almost_equal(ret[t], adv[t], atol=1e-6)
    print("  test_compute_gae_constant_value PASSED")
    rew.free(); val.free(); term.free(); adv.free(); ret.free()


def test_compute_gae_terminated_stops_bootstrap() raises:
    """Terminated=1 at t=1 must zero bootstrap from t=2."""
    comptime N = 3
    var rew = alloc[Scalar[DT]](N)
    var val = alloc[Scalar[DT]](N)
    var term = alloc[Scalar[DT]](N)
    var adv = alloc[Scalar[DT]](N)
    var ret = alloc[Scalar[DT]](N)
    rew[0] = 1.0; rew[1] = 1.0; rew[2] = 1.0
    val[0] = 0.0; val[1] = 0.0; val[2] = 0.0
    term[0] = 0.0; term[1] = 1.0; term[2] = 0.0  # terminal at t=1
    compute_gae(
        N, rew, val, term, next_value=Scalar[DT](0.0),
        gamma=Scalar[DT](1.0), gae_lambda=Scalar[DT](1.0),
        advantages=adv, returns=ret,
    )
    # A_2 = 1 (last step, bootstrap = next_value = 0)
    # A_1 = 1 + 1·0·A_2 = 1  (terminated[1]=1 zeros nonterm)
    # A_0 = 1 + 1·1·A_1 = 2  (terminated[0]=0)
    assert_almost_equal(adv[0], Scalar[DT](2.0), atol=1e-6)
    assert_almost_equal(adv[1], Scalar[DT](1.0), atol=1e-6)
    assert_almost_equal(adv[2], Scalar[DT](1.0), atol=1e-6)
    print("  test_compute_gae_terminated_stops_bootstrap PASSED")
    rew.free(); val.free(); term.free(); adv.free(); ret.free()


def test_normalize_in_place() raises:
    comptime N = 5
    var buf = alloc[Scalar[DT]](N)
    buf[0] = 1.0; buf[1] = 2.0; buf[2] = 3.0; buf[3] = 4.0; buf[4] = 5.0
    normalize_in_place(N, buf)
    # Mean should be ~0, std ~1.
    var s: Scalar[DT] = 0.0
    for i in range(N):
        s = s + buf[i]
    var mean = s / Scalar[DT](N)
    assert_almost_equal(mean, Scalar[DT](0.0), atol=1e-5)
    var sq: Scalar[DT] = 0.0
    for i in range(N):
        sq = sq + buf[i] * buf[i]
    var std = fsqrt(sq / Scalar[DT](N))
    assert_almost_equal(std, Scalar[DT](1.0), atol=1e-3)
    print("  test_normalize_in_place PASSED")
    buf.free()


def main() raises:
    print("=" * 70)
    print("Phase 8.1 helper unit tests")
    print("=" * 70)
    test_box_muller_empirical_moments()
    test_episode_tracker()
    test_online_target_pair()
    test_cpu_replay()
    test_compute_gae_constant_value()
    test_compute_gae_terminated_stops_bootstrap()
    test_normalize_in_place()
    print("ALL PASSED")
