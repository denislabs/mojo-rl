"""Dreamer 4 reward-bearing Pong buffer — roundtrip + sampler alignment.

    pixi run mojo run -I . tests/nn2/test_dreamer4_reward_buffer.mojo

Validates `Dreamer4PongRewardBuffer`:
  1. save → load roundtrips n_frames / actions / dones / rewards exactly.
  2. the window sampler emits one-hot actions, rewards and dones that stay
     ALIGNED with the frames it returns (recovered via a per-frame pixel
     fingerprint), and never bridges an internal episode boundary.
"""

from std.memory import alloc
from std.math import abs
from std.testing import assert_true, assert_equal

from mojo_rl.deep_agents2.dreamer4.pong_reward_buffer import (
    Dreamer4PongRewardBuffer,
)
from mojo_rl.envs.arcade_games.pong.offline_buffer import PONG_FRAME_BYTES


def _make_synthetic(n: Int) -> Dreamer4PongRewardBuffer:
    """n steps; frame[·,0] = step/255 fingerprints the absolute index, action
    = step%3, reward = 0.1·step − 1, done at steps 11 and (n−1)."""
    var buf = Dreamer4PongRewardBuffer(capacity=n, seed=12345)
    for step in range(n):
        var obs = List[Scalar[DType.float32]](capacity=PONG_FRAME_BYTES)
        for _ in range(PONG_FRAME_BYTES):
            obs.append(Scalar[DType.float32](0.0))
        obs[0] = Scalar[DType.float32](Float64(step) / 255.0)
        var done = (step == 11) or (step == n - 1)
        buf.add_step_fp32_list(
            obs, step % 3, done, Scalar[DType.float32](0.1 * Float64(step) - 1.0)
        )
    return buf^


def test_roundtrip() raises:
    print("-- save/load roundtrip")
    comptime N = 24
    var buf = _make_synthetic(N)
    buf.save("/tmp/dreamer4_reward_buffer_test.bin")
    var lo = Dreamer4PongRewardBuffer.load("/tmp/dreamer4_reward_buffer_test.bin")
    assert_equal(lo.n_frames, N, "n_frames roundtrip")
    var max_re = Float64(0.0)
    for step in range(N):
        assert_equal(Int(lo.actions[step]), step % 3, "action roundtrip")
        var dexp = 1 if (step == 11 or step == N - 1) else 0
        assert_equal(Int(lo.dones[step]), dexp, "done roundtrip")
        var rexp = 0.1 * Float64(step) - 1.0
        var e = abs(Float64(lo.rewards[step]) - rexp)
        if e > max_re:
            max_re = e
    print("   max|Δreward| =", max_re)
    assert_true(max_re < 1e-6, "rewards must roundtrip")
    print("   actions/dones/rewards roundtrip OK")


def test_sampler_alignment() raises:
    print("-- window sampler alignment")
    comptime N = 24
    comptime B = 6
    comptime T = 4
    comptime ACT = 3
    var buf = _make_synthetic(N)

    var pix = alloc[Scalar[DType.float32]](B * T * PONG_FRAME_BYTES)
    var act = alloc[Scalar[DType.float32]](B * T * ACT)
    var rew = alloc[Scalar[DType.float32]](B * T)
    var done = alloc[Scalar[DType.float32]](B * T)

    var checked = 0
    for _ in range(8):  # several batches to exercise many starts
        buf.sample_reward_window_batch[B, T, ACT](pix, act, rew, done)
        for b in range(B):
            # recover absolute frame index of each window position from the
            # pixel fingerprint, assert the window is contiguous + in-range,
            # and that action/reward/done match the source arrays at that index.
            var prev = -1
            var crossed_internal = False
            for t in range(T):
                var bt = b * T + t
                var idx = Int(Float64(pix[bt * PONG_FRAME_BYTES]) * 255.0 + 0.5)
                assert_true(idx >= 0 and idx < N, "recovered index in range")
                if t > 0:
                    assert_equal(idx, prev + 1, "window must be contiguous")
                prev = idx
                # action one-hot matches
                var amax = 0
                var av = Float64(act[bt * ACT])
                for k in range(1, ACT):
                    if Float64(act[bt * ACT + k]) > av:
                        av = Float64(act[bt * ACT + k])
                        amax = k
                assert_equal(amax, idx % 3, "sampled action aligned")
                # reward matches
                var rexp = 0.1 * Float64(idx) - 1.0
                assert_true(
                    abs(Float64(rew[bt]) - rexp) < 1e-5, "sampled reward aligned"
                )
                # done matches
                var dexp = 1.0 if (idx == 11 or idx == N - 1) else 0.0
                assert_true(
                    abs(Float64(done[bt]) - dexp) < 1e-6, "sampled done aligned"
                )
                # an internal (non-final) done would mean the window bridged a
                # boundary — forbidden.
                if t < T - 1 and dexp > 0.5:
                    crossed_internal = True
                checked += 1
            assert_true(
                not crossed_internal, "window must not cross episode boundary"
            )
    print("   checked", checked, "window positions — all aligned")
    print("   no window bridged an episode boundary")

    pix.free()
    act.free()
    rew.free()
    done.free()


def main() raises:
    print("=" * 70)
    print("Dreamer 4 reward-bearing Pong buffer")
    print("=" * 70)
    test_roundtrip()
    test_sampler_alignment()
    print("=" * 70)
    print("ALL PASSED — reward buffer roundtrip + aligned sampler")
    print("=" * 70)
