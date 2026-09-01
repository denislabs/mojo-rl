"""Abandoning a decode mid-file must not read as a failure.

Run: pixi run mojo run -I . tests/io/test_video_abandon.mojo   (needs ffmpeg)

`VideoDecoder.close()` excuses a broken output pipe when it stopped reading on
purpose — LeRobot packs several episodes per mp4 and the importer rolls over
once it has the ones it wants. That excuse was written for **signal 13** and
was DEAD CODE: `ffmpeg` calls `signal(SIGPIPE, SIG_IGN)` in `term_init()`, so
it returns `AVERROR(EPIPE)` from `main` and exits **224** instead. Every
abandoned decode was really raising `proc: child exited 224`, hidden only
because each decoder happened to reach EOF before its rollover.

So `test_abandoning_a_decode_is_not_an_error` is the leg that was RED, and the
three around it are what keep the widened excuse from becoming a blanket one:

* `test_a_full_decode_still_checks_the_exit_status` — the ordinary path, and
  the frame count that proves the file was really read.
* `test_a_real_failure_is_still_an_error` — exit 3 is not excused even with
  the flag set. Without this leg, `close()` could ignore every status and
  every other test here would stay green.
* `test_the_excuse_is_gated_on_the_flag` — exit 224 from a child we were
  reading to the end IS an error. `224` is only special because we asked to
  stop early.
"""

from std.os.path import exists

from mojo_rl.io.proc import Pipe
from mojo_rl.io.video.decoder import VideoDecoder, probe_video
from mojo_rl.io.video.encoder import VideoEncoder


comptime W = 64
comptime H = 48
comptime N_FRAMES = 60
"""Long enough that abandoning after one frame leaves ffmpeg with plenty
still to write — a 5-frame file can finish before we let go, and then the
pipe never breaks and the leg passes vacuously."""

comptime PATH = "/tmp/mojo_video_abandon.mp4"


def _make_video() raises:
    """Write the fixture once per run. Returns nothing; raises if ffmpeg did."""
    var frame = List[UInt8](unsafe_uninit_length=W * H * 3)
    var enc = VideoEncoder(String(PATH), W, H, fps=30)
    for f in range(N_FRAMES):
        for i in range(len(frame)):
            frame[i] = UInt8((i + f * 7) % 256)
        enc.add_frame_list(frame)
    var written = enc.close()
    if written != N_FRAMES:
        raise Error(
            "fixture: encoder took " + String(written) + " of "
            + String(N_FRAMES) + " frames"
        )
    if not exists(String(PATH)):
        raise Error("fixture: ffmpeg wrote no file at " + String(PATH))
    var info = probe_video(String(PATH))
    if info.width != W or info.height != H:
        raise Error(
            "fixture: ffprobe reports " + String(info.width) + "x"
            + String(info.height) + ", wanted " + String(W) + "x" + String(H)
        )
    print(
        "  fixture:", N_FRAMES, "frames of", String(W) + "x" + String(H),
        "at", PATH,
    )


def test_a_full_decode_still_checks_the_exit_status() raises:
    var dst = List[UInt8](unsafe_uninit_length=W * H * 3)
    var dptr = dst.unsafe_ptr().unsafe_bitcast[
        Scalar[DType.uint8]
    ]().as_unsafe_any_origin()
    var dec = VideoDecoder(String(PATH))
    var got = 0
    while True:
        if not dec.next_into(dptr):
            break
        got += 1
    dec.close()
    _ = dst^
    if got != N_FRAMES:
        raise Error(
            "full decode returned " + String(got) + " frames, the file holds "
            + String(N_FRAMES)
        )
    print("  full decode:", got, "frames, ffmpeg exited clean")


def test_abandoning_a_decode_is_not_an_error() raises:
    """THE LEG THAT WAS RED. One frame of sixty, then let go."""
    var dst = List[UInt8](unsafe_uninit_length=W * H * 3)
    var dptr = dst.unsafe_ptr().unsafe_bitcast[
        Scalar[DType.uint8]
    ]().as_unsafe_any_origin()
    var dec = VideoDecoder(String(PATH))
    if not dec.next_into(dptr):
        raise Error("abandon: the decoder produced no frames at all")
    _ = dst^
    try:
        dec.close()
    except e:
        raise Error(
            "abandoning after 1 of " + String(N_FRAMES) + " frames raised: "
            + String(e)
        )
    print("  abandoned after 1 of", N_FRAMES, "frames: close() returned")


def test_a_real_failure_is_still_an_error() raises:
    """Exit 3 is not a broken pipe, flag or no flag."""
    var p = Pipe(String("sh -c 'exit 3'"))
    var raised = False
    try:
        _ = p.close(True)
    except:
        raised = True
    if not raised:
        raise Error(
            "close(allow_broken_pipe=True) accepted exit 3 — the excuse is a"
            " blanket one and every other leg here is vacuous"
        )
    print("  exit 3 with the flag set: raised, as it must")


def test_the_excuse_is_gated_on_the_flag() raises:
    """224 is only special for a caller that stopped reading on purpose."""
    var p = Pipe(String("sh -c 'exit 224'"))
    var raised = False
    try:
        _ = p.close(False)
    except:
        raised = True
    if not raised:
        raise Error("close(allow_broken_pipe=False) accepted exit 224")
    print("  exit 224 without the flag: raised, as it must")

    var q = Pipe(String("sh -c 'exit 224'"))
    try:
        _ = q.close(True)
    except e:
        raise Error("close(allow_broken_pipe=True) rejected 224: " + String(e))
    print("  exit 224 with the flag: accepted, as it must")


def main() raises:
    print("=" * 62)
    print("VideoDecoder — abandoning a file is not a failure")
    print("=" * 62)
    _make_video()
    test_a_full_decode_still_checks_the_exit_status()
    test_abandoning_a_decode_is_not_an_error()
    test_a_real_failure_is_still_an_error()
    test_the_excuse_is_gated_on_the_flag()
    print("[PASS] video_abandon")
