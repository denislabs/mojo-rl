"""Dreamer4FrameBuffer ring-overwrite + window-sampling correctness (CPU gate).

Encodes each step's pixels with its step index, so after overwriting we can
recover, for every sampled window:
  • only the most recent CAP steps are present (ring overwrite dropped old ones),
  • frames within a window are temporally CONSECUTIVE (logical/temporal order
    survives the physical wrap),
  • no window crosses an episode boundary (`done`) in its first T-1 frames,
  • reward / one-hot action stay aligned to the frame's step.

Run: pixi run mojo run -I . tests/nn/test_dreamer4_frame_buffer.mojo
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.dreamer4.frame_buffer import Dreamer4FrameBuffer


def main() raises:
    print("Dreamer4FrameBuffer ring + window sampling gate (CPU)")
    comptime C = 1
    comptime H = 2
    comptime W = 2
    comptime ACT = 3
    comptime CAP = 8
    comptime FRAME = C * H * W
    comptime NSTEPS = 20
    comptime B = 4
    comptime T = 3

    var buf = Dreamer4FrameBuffer[C, H, W, ACT, CAP]()

    # Add NSTEPS distinct steps; obs encodes the step index (step/255 per pixel).
    for step in range(NSTEPS):
        var obs = List[Scalar[DT]](length=FRAME, fill=Scalar[DT](0))
        for i in range(FRAME):
            obs[i] = Scalar[DT](Float64(step) / 255.0)
        var done = (step % 7) == 6           # dones at steps 6, 13, 19
        buf.add_step_fp32_list(obs, step % ACT, done, Scalar[DT](Float64(step)))

    var ok = True
    if buf.count() != CAP:
        print("  FAIL: count =", buf.count(), " expected", CAP)
        ok = False
    # after 20 adds into CAP=8, the buffer holds steps [12, 19].
    comptime OLDEST_STEP = NSTEPS - CAP      # 12

    var pix = List[Scalar[DT]](length=B * T * FRAME, fill=Scalar[DT](0))
    var act = List[Scalar[DT]](length=B * T * ACT, fill=Scalar[DT](0))
    var rew = List[Scalar[DT]](length=B * T, fill=Scalar[DT](0))
    var done_o = List[Scalar[DT]](length=B * T, fill=Scalar[DT](0))

    buf.sample_reward_window_batch[B, T](
        pix.unsafe_ptr(), act.unsafe_ptr(), rew.unsafe_ptr(), done_o.unsafe_ptr()
    )

    for b in range(B):
        var prev_step = -1
        for t in range(T):
            var bt = b * T + t
            # recover step index from the (constant) pixel value
            var step = Int(Float64(pix[bt * FRAME]) * 255.0 + 0.5)
            # only recent CAP steps present
            if step < OLDEST_STEP or step >= NSTEPS:
                print("  FAIL: stale/oob step", step, "(b", b, "t", t, ")")
                ok = False
            # temporally consecutive within the window
            if t > 0 and step != prev_step + 1:
                print("  FAIL: non-consecutive", prev_step, "->", step)
                ok = False
            prev_step = step
            # reward + action aligned to the step
            if Int(Float64(rew[bt]) + 0.5) != step:
                print("  FAIL: reward", rew[bt], "≠ step", step)
                ok = False
            if act[bt * ACT + (step % ACT)] != Scalar[DT](1.0):
                print("  FAIL: action one-hot misaligned at step", step)
                ok = False
            # no `done` allowed in the first T-1 frames of a window
            if t < T - 1 and done_o[bt] != Scalar[DT](0.0):
                print("  FAIL: window crosses done at b", b, "t", t)
                ok = False

    print("  count == CAP, windows recent+consecutive+aligned, done-respected:",
          "OK" if ok else "FAIL")
    assert_true(ok, "Dreamer4FrameBuffer ring + window sampling")
    print("DREAMER4 FRAME BUFFER GATE OK")
