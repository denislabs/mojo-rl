"""SequenceReplay tests (Block D-5).

Covers:
  * `record` populates buffer in chronological order and wraps at CAP
  * `sample_batch` returns contiguous T+1 obs / T act-rew-dne windows
  * Output observations match the buffer contents (chronological)
  * Once `size = CAP`, sampling pulls from the ring (origin = pos)
"""

from std.math import abs as fabs
from std.memory import alloc
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.data.sequence_replay import SequenceReplay


def test_basic_record_and_sample() raises:
    """Record CAP/2 steps, sample one length-T window, verify chronological."""
    seed(0)
    comptime OBS = 2
    comptime ACT = 1
    comptime CAP = 16
    comptime B = 4
    comptime T = 4
    var buf = SequenceReplay[OBS, ACT, CAP].new()

    # Record 12 steps. Each obs[k] = (k, -k), act[k] = (k*0.1), rew[k] = k.
    var s_buf: Pointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OBS).as_unsafe_any_origin()
    var a_buf: Pointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](ACT).as_unsafe_any_origin()
    for k in range(12):
        s_buf[0] = Scalar[DT](k)
        s_buf[1] = Scalar[DT](-Float64(k))
        a_buf[0] = Scalar[DT](Float64(k) * 0.1)
        buf.record(s_buf, a_buf, Scalar[DT](k), Scalar[DT](1.0) if (k % 5 == 4) else Scalar[DT](0.0))

    assert_true(buf.size == 12, "size should be 12")
    assert_true(buf.pos == 12, "pos should be 12")

    var obs_out: Pointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * (T + 1) * OBS).as_unsafe_any_origin()
    var act_out: Pointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * T * ACT).as_unsafe_any_origin()
    var rew_out: Pointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * T).as_unsafe_any_origin()
    var dne_out: Pointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * T).as_unsafe_any_origin()

    buf.sample_batch[B, T](obs_out, act_out, rew_out, dne_out)

    # Verify each window is chronological: obs_out[b, k+1] = obs_out[b, k] + (1, -1).
    for b in range(B):
        var first_step = obs_out[b * (T + 1) * OBS + 0]
        print("  window b=", b, " first step =", first_step)
        for k in range(T + 1):
            var got_x = obs_out[b * (T + 1) * OBS + k * OBS + 0]
            var got_y = obs_out[b * (T + 1) * OBS + k * OBS + 1]
            var expected_x = first_step + Scalar[DT](k)
            var expected_y = -expected_x
            assert_true(fabs(got_x - expected_x) < 1e-6, "obs[0] not chronological")
            assert_true(fabs(got_y - expected_y) < 1e-6, "obs[1] not chronological")
        for k in range(T):
            var expected_step = first_step + Scalar[DT](k)
            var expected_a = expected_step * Scalar[DT](0.1)
            var got_a = act_out[b * T * ACT + k]
            assert_true(
                fabs(got_a - expected_a) < 1e-6,
                "action not chronological",
            )
            assert_true(
                fabs(rew_out[b * T + k] - expected_step) < 1e-6,
                "reward not chronological",
            )

    s_buf.free()
    a_buf.free()
    obs_out.free()
    act_out.free()
    rew_out.free()
    dne_out.free()
    print("  test_basic_record_and_sample PASSED")


def test_circular_wrap() raises:
    """Record more than CAP steps and confirm the ring wraps + sampling
    still produces chronological windows (relative to write order)."""
    seed(1)
    comptime OBS = 1
    comptime ACT = 1
    comptime CAP = 8
    comptime B = 1
    comptime T = 3
    var buf = SequenceReplay[OBS, ACT, CAP].new()

    var s_buf: Pointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OBS).as_unsafe_any_origin()
    var a_buf: Pointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](ACT).as_unsafe_any_origin()
    # Record 20 steps into a CAP=8 buffer → wraps twice. The buffer
    # afterwards holds steps 12..19 (the 8 most recent).
    for k in range(20):
        s_buf[0] = Scalar[DT](k)
        a_buf[0] = Scalar[DT](Float64(k))
        buf.record(s_buf, a_buf, Scalar[DT](k), Scalar[DT](0.0))

    assert_true(buf.size == CAP, "buffer should be full")
    assert_true(buf.pos == 20 % CAP, "pos should equal write count mod CAP")

    var obs_out: Pointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * (T + 1) * OBS).as_unsafe_any_origin()
    var act_out: Pointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * T * ACT).as_unsafe_any_origin()
    var rew_out: Pointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * T).as_unsafe_any_origin()
    var dne_out: Pointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * T).as_unsafe_any_origin()

    # Sample many windows; every returned window must be chronological
    # (consecutive integers) AND every value must be in [12, 19].
    for _ in range(50):
        buf.sample_batch[B, T](obs_out, act_out, rew_out, dne_out)
        var first = obs_out[0]
        assert_true(first >= Scalar[DT](12.0), "earliest window slot must be >= 12")
        assert_true(first + Scalar[DT](T) <= Scalar[DT](19.0), "last slot must be <= 19")
        for k in range(T + 1):
            assert_true(
                fabs(obs_out[k] - (first + Scalar[DT](k))) < 1e-6,
                "wrap-window not chronological",
            )

    s_buf.free()
    a_buf.free()
    obs_out.free()
    act_out.free()
    rew_out.free()
    dne_out.free()
    print("  test_circular_wrap PASSED")


def test_can_sample_guard() raises:
    """`can_sample[T]` reports correctly; sample_batch raises when too small."""
    comptime OBS = 1
    comptime ACT = 1
    comptime CAP = 4
    var buf = SequenceReplay[OBS, ACT, CAP].new()
    assert_true(buf.can_sample[T=3]() == False, "empty buffer cannot sample")

    var s_buf: Pointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OBS).as_unsafe_any_origin()
    var a_buf: Pointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](ACT).as_unsafe_any_origin()
    for _ in range(3):
        s_buf[0] = Scalar[DT](0.0)
        a_buf[0] = Scalar[DT](0.0)
        buf.record(s_buf, a_buf, Scalar[DT](0.0), Scalar[DT](0.0))

    assert_true(buf.can_sample[T=2]() == True, "size=3 supports T=2 window (T+1=3 frames)")
    assert_true(buf.can_sample[T=3]() == False, "size=3 NOT enough for T=3 window (needs T+1=4 frames)")
    s_buf.free()
    a_buf.free()
    print("  test_can_sample_guard PASSED")


def main() raises:
    print("=" * 60)
    print("nn SequenceReplay tests (Block D-5)")
    print("=" * 60)
    test_basic_record_and_sample()
    test_circular_wrap()
    test_can_sample_guard()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
