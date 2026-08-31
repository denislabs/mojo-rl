"""FramePipeThread — accounting, back-pressure, and a child that dies.

Run: pixi run mojo run -I . tests/io/test_frame_pipe_thread.mojo

⚠ NO `ffmpeg` HERE, DELIBERATELY. The consumer is `cat` and `sh`, so every
case is deterministic and the assertions are on BYTES ON DISK rather than on
what an encoder chose to do. `tests/render/test_video_recorder.mojo` is the leg
that runs the real encoder and decodes the result back.

The three that matter:

* `test_blocking_never_loses_a_frame` — a full ring must WAIT. A dropped frame
  does not leave a gap in a `rawvideo` recording, it makes the video shorter
  and plays everything after it early, with nothing in the file to show for it.
* `test_dropping_is_counted_exactly` — when dropping is asked for, the tally
  must account for every frame: `written + dropped == submitted`, and the file
  must hold exactly `written` frames.
* `test_a_dead_child_does_not_kill_us` — the SIGPIPE guard in `io/proc.mojo`,
  from a worker thread. Without it this test would not fail, it would take the
  whole process down with exit 141.
"""

from std.memory import Pointer
from std.os.path import exists, getsize
from std.time import perf_counter_ns

from mojo_rl.io.video.frame_pipe_thread import (
    MAX_SLOTS,
    MIN_SLOTS,
    FramePipeThread,
    slots_for,
)


comptime FRAME = 128 * 1024
"""Bigger than a 64 KB pipe buffer, so a write to a stalled reader really
blocks instead of being absorbed."""


@always_inline
def _erase(mut lst: List[UInt8]) -> Pointer[UInt8, MutUntrackedOrigin]:
    return rebind[Pointer[UInt8, MutUntrackedOrigin]](
        lst.unsafe_ptr().as_unsafe_any_origin()
    )


def _frame(fill: Int) -> List[UInt8]:
    var f = List[UInt8](unsafe_uninit_length=FRAME)
    for i in range(FRAME):
        f[i] = UInt8((fill + i) & 0xFF)
    return f^


def test_slots_for_stays_in_band() raises:
    var big = slots_for(1920 * 1080 * 4)
    var small = slots_for(64 * 48 * 4)
    var zero = slots_for(0)
    if big < MIN_SLOTS or big > MAX_SLOTS:
        raise Error("1080p asked for " + String(big) + " slots")
    if small != MAX_SLOTS:
        raise Error("a tiny frame did not reach the cap: " + String(small))
    if zero != MIN_SLOTS:
        raise Error("frame_bytes=0 gave " + String(zero))
    if big * 1920 * 1080 * 4 > 96 << 20:
        raise Error(
            "1080p staging is " + String(big * 1920 * 1080 * 4 >> 20) + " MB"
        )
    print(
        "  slots_for: 1080p ->", big, "slots =",
        big * 1920 * 1080 * 4 >> 20, "MB;  64x48 ->", small, "( cap",
        MAX_SLOTS, ");  0 ->", zero,
    )


def test_every_byte_arrives() raises:
    """The happy path, checked against the file rather than a counter."""
    comptime N = 12
    var path = String("/tmp/mojo_frame_pipe_ok.bin")
    var sink = FramePipeThread(String("cat > ") + path, FRAME, slots=4)
    sink.start()
    var accepted = 0
    for k in range(N):
        var f = _frame(k)
        if sink.submit(_erase(f), FRAME, wait_us=-1):
            accepted += 1
        _ = f^
    var written = sink.stop()

    var size = getsize(path) if exists(path) else -1
    if accepted != N or written != N:
        raise Error(
            String(accepted) + " accepted / " + String(written) + " written of "
            + String(N)
        )
    if sink.dropped() != 0 or sink.failed() != 0:
        raise Error(
            String(sink.dropped()) + " dropped, " + String(sink.failed())
            + " failed on a consumer that never stalled"
        )
    if size != N * FRAME:
        raise Error(
            "the file holds " + String(size) + " bytes, expected "
            + String(N * FRAME)
        )
    print(
        "  happy path:", written, "of", N, "frames written,", size,
        "bytes on disk =", N * FRAME, "expected, 0 dropped",
    )


def test_blocking_never_loses_a_frame() raises:
    """A consumer that sleeps first: the ring fills, the producer must wait.

    Non-vacuous by construction — the assertion on elapsed time is what proves
    the producer actually blocked rather than the ring being big enough.
    """
    comptime N = 12
    comptime SLOTS = 4
    var path = String("/tmp/mojo_frame_pipe_block.bin")
    var sink = FramePipeThread(
        String("sleep 1; cat > ") + path, FRAME, slots=SLOTS
    )
    sink.start()
    var t0 = perf_counter_ns()
    var accepted = 0
    for k in range(N):
        var f = _frame(k)
        if sink.submit(_erase(f), FRAME, wait_us=-1):
            accepted += 1
        _ = f^
    var submit_ms = Float64(perf_counter_ns() - t0) / 1e6
    var written = sink.stop()
    var size = getsize(path) if exists(path) else -1

    if accepted != N or written != N or sink.dropped() != 0:
        raise Error(
            "BLOCKING LOST FRAMES: " + String(accepted) + " accepted, "
            + String(written) + " written, " + String(sink.dropped())
            + " dropped of " + String(N)
        )
    if size != N * FRAME:
        raise Error(
            "the file holds " + String(size) + " of " + String(N * FRAME)
            + " bytes"
        )
    if submit_ms < 500.0:
        raise Error(
            "submitting " + String(N) + " frames into a " + String(SLOTS)
            + "-slot ring took only " + String(submit_ms) + " ms — the"
            " consumer was not stalled, so this gate is VACUOUS"
        )
    print(
        "  blocking:", written, "of", N, "frames kept through a 1s stall (",
        size, "bytes ); the producer waited", submit_ms, "ms on a", SLOTS,
        "slot ring",
    )


def test_dropping_is_counted_exactly() raises:
    """`wait_us=0` is allowed to lose frames — but not to lose the count."""
    comptime N = 40
    comptime SLOTS = 4
    var path = String("/tmp/mojo_frame_pipe_drop.bin")
    var sink = FramePipeThread(
        String("sleep 1; cat > ") + path, FRAME, slots=SLOTS
    )
    sink.start()
    var accepted = 0
    for k in range(N):
        var f = _frame(k)
        if sink.submit(_erase(f), FRAME, wait_us=0):
            accepted += 1
        _ = f^
    var written = sink.stop()
    var dropped = sink.dropped()
    var size = getsize(path) if exists(path) else -1

    if dropped == 0:
        raise Error(
            "nothing was dropped into a " + String(SLOTS) + "-slot ring behind"
            " a 1s stall — this gate is VACUOUS. Raise N or lower SLOTS."
        )
    if accepted + dropped != N:
        raise Error(
            "accounting: " + String(accepted) + " accepted + " + String(dropped)
            + " dropped != " + String(N) + " submitted"
        )
    if written != accepted:
        raise Error(
            String(written) + " frames written but " + String(accepted)
            + " accepted — the drain lost some"
        )
    if size != written * FRAME:
        raise Error(
            "the file holds " + String(size) + " bytes for " + String(written)
            + " written frames (" + String(written * FRAME) + " expected)"
        )
    print(
        "  dropping:", accepted, "accepted +", dropped, "dropped =", N,
        "submitted;", written, "written =", size, "bytes exactly",
    )


def test_a_dead_child_does_not_kill_us() raises:
    """The child exits at once; every write then hits a closed pipe.

    ⚠ IF THE SIGPIPE GUARD REGRESSES THIS DOES NOT FAIL — the process dies
    with exit 141 and the suite never prints another line. That is the point.
    """
    comptime N = 20
    var sink = FramePipeThread(String("exit 0"), FRAME, slots=4)
    sink.start()
    for k in range(N):
        var f = _frame(k)
        _ = sink.submit(_erase(f), FRAME, wait_us=-1)
        _ = f^
    _ = sink.stop()

    if sink.failed() == 0:
        raise Error(
            "a child that exited immediately produced 0 failed writes — the"
            " error is being swallowed"
        )
    var reported = False
    try:
        sink.raise_if_broken()
    except:
        reported = True
    if not reported:
        raise Error("raise_if_broken stayed silent about a dead child")
    print(
        "  dead child:", sink.failed(), "failed writes reported, process alive"
        " (no SIGPIPE),", sink.written(), "frames claimed written",
    )


def test_stop_is_idempotent() raises:
    var path = String("/tmp/mojo_frame_pipe_idem.bin")
    var sink = FramePipeThread(String("cat > ") + path, FRAME, slots=4)
    sink.start()
    var f = _frame(1)
    _ = sink.submit(_erase(f), FRAME, wait_us=-1)
    _ = f^
    var a = sink.stop()
    var b = sink.stop()
    if a != 1 or b != 1:
        raise Error("stop() returned " + String(a) + " then " + String(b))
    print("  stop: idempotent, both calls report", b, "frame")


def main() raises:
    print("=" * 62)
    print("FramePipeThread — bytes, back-pressure, and a dead child")
    print("=" * 62)
    test_slots_for_stays_in_band()
    test_every_byte_arrives()
    test_blocking_never_loses_a_frame()
    test_dropping_is_counted_exactly()
    test_a_dead_child_does_not_kill_us()
    test_stop_is_idempotent()
    print("[PASS] frame_pipe_thread")
