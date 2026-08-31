# +--------------------------------------------------------------------------+ #
# | Does a 30 Hz record tick fit? Measure before designing the loop.
# +--------------------------------------------------------------------------+ #
"""Times the pieces of a recording tick against real hardware.

    pixi run mojo run -I . tools/soarm/record_budget.mojo
    pixi run mojo run -I . tools/soarm/record_budget.mojo --devices 0,1 --seconds 10

`docs/SO101_RECORDING_PLAN.md` lists the real-time budget as a risk and says
it "needs measuring before it is designed". This is that measurement, and it
is deliberately the LAST thing written before the record loop rather than the
first thing assumed.

## What it measures, in layers

    A  cameras only          N blocking `VideoCapture.read()` per tick
    B  A + encode            each frame also pushed into an ffmpeg pipe
    C  B + the leader arm    a `sync_read` of 6 servo positions

Each layer is the previous one plus one cost, so the difference between two
layers IS that cost — which is the only way to know whether the camera read,
the encoder or the bus is what blows a 33.3 ms budget.

⚠ **READ-ONLY ON THE ROBOT.** The leader is read with torque OFF and the
follower is never opened. Nothing here moves anything. The recorder that comes
later does move the follower, and that is a separate program with the
teleop guards in it.

⚠ **A MEAN THAT FITS IS NOT A BUDGET THAT FITS.** A control loop is judged on
its WORST tick, because one late tick is one dropped frame in the dataset.
`teleop.mojo` already measured that `usleep` overshoots far enough on macOS to
miss a 20 ms period outright, so both are reported and the worst is the number
that decides the design.
"""

from std.sys import argv
from std.time import perf_counter_ns

from mojo_rl.core.concurrent.thread import sleep_us

from mojo_rl.io.video import VideoEncoder, VideoEncoderThread
from mojo_rl.robot.so101 import SO101Arm, SO101_N
from mojo_rl.utils.fmt import fixed
from mojo_rl.vision.camera_thread import CameraReader
from mojo_rl.vision.opencv import VideoCapture, opencv_shim_available


comptime LEADER_PORT = "/dev/cu.usbmodem5B910455171"
comptime HZ = 30
comptime WARMUP = 15
"""Frames discarded before timing. Consumer webcams take a moment to settle,
and `sdl_camera.mojo`'s own header says so — timing the warm-up measures the
hardware waking up rather than the loop."""


struct Stat(Movable):
    var n: Int
    var total_ms: Float64
    var worst_ms: Float64
    var over: Int

    def __init__(out self):
        self.n = 0
        self.total_ms = 0.0
        self.worst_ms = 0.0
        self.over = 0

    def __init__(out self, *, deinit move: Self):
        self.n = move.n
        self.total_ms = move.total_ms
        self.worst_ms = move.worst_ms
        self.over = move.over

    def add(mut self, ms: Float64, budget_ms: Float64):
        self.n += 1
        self.total_ms += ms
        if ms > self.worst_ms:
            self.worst_ms = ms
        if ms > budget_ms:
            self.over += 1

    def mean(self) -> Float64:
        return self.total_ms / Float64(self.n) if self.n > 0 else 0.0


def _row(label: String, ref s: Stat, budget_ms: Float64) raises:
    var flag = String("")
    if s.worst_ms > budget_ms:
        flag = "   <-- worst tick BLOWS the " + fixed(budget_ms, 1) + " ms budget"
    print(
        "  " + label + "   mean " + fixed(s.mean(), 2) + " ms   worst "
        + fixed(s.worst_ms, 2) + " ms   over budget " + String(s.over) + "/"
        + String(s.n) + flag
    )


def main() raises:
    if not opencv_shim_available():
        raise Error(
            "record_budget: the OpenCV shim is not built — `pixi run"
            " build-opencv`"
        )

    var devices = List[Int]()
    var seconds = 8
    var args = argv()
    for i in range(len(args)):
        if String(args[i]) == "--devices" and i + 1 < len(args):
            var spec = String(args[i + 1])
            var cur = String("")
            for k in range(spec.byte_length()):
                var c = chr(Int(spec.as_bytes()[k]))
                if c == ",":
                    if cur != "":
                        devices.append(Int(cur))
                    cur = String("")
                else:
                    cur += c
            if cur != "":
                devices.append(Int(cur))
        elif String(args[i]) == "--seconds" and i + 1 < len(args):
            seconds = Int(String(args[i + 1]))
    if len(devices) == 0:
        devices.append(0)
        devices.append(1)

    var budget_ms = 1000.0 / Float64(HZ)
    print("=" * 72)
    print(
        "Recording tick budget — " + String(HZ) + " Hz => "
        + fixed(budget_ms, 1) + " ms per tick"
    )
    print("=" * 72)

    # ── open the cameras ──────────────────────────────────────────────
    var caps = List[VideoCapture]()
    for i in range(len(devices)):
        print("opening camera " + String(devices[i]) + " ...")
        var c = VideoCapture.device(devices[i], 640, 480, 30.0)
        # ⚠ READ BACK, DO NOT ECHO. `VideoCapture.device`'s own header says
        # the size and rate are REQUESTS; a camera is free to ignore them and
        # OpenCV reports no error when it does.
        print(
            "  -> " + String(c.width) + "x" + String(c.height) + "  "
            + String(c.channels) + " channels"
        )
        caps.append(c^)

    var n_cam = len(caps)
    var bufs = List[List[UInt8]]()
    for i in range(n_cam):
        bufs.append(List[UInt8](unsafe_uninit_length = caps[i].frame_bytes()))

    print("\nwarming up (" + String(WARMUP) + " frames) ...")
    for _ in range(WARMUP):
        for i in range(n_cam):
            _ = caps[i].read(bufs[i])

    var ticks = HZ * seconds

    # ── layer A: cameras only ─────────────────────────────────────────
    print(
        "\n── A. " + String(n_cam) + " camera read(s) per tick, "
        + String(ticks) + " ticks ──"
    )
    var a_tick = Stat()
    var a_cam = Stat()
    for _ in range(ticks):
        var t0 = perf_counter_ns()
        for i in range(n_cam):
            var c0 = perf_counter_ns()
            if not caps[i].read(bufs[i]):
                raise Error("record_budget: camera " + String(i) + " ended")
            a_cam.add(Float64(perf_counter_ns() - c0) / 1e6, budget_ms)
        a_tick.add(Float64(perf_counter_ns() - t0) / 1e6, budget_ms)
    _row(String("one camera read "), a_cam, budget_ms)
    _row(String("whole tick      "), a_tick, budget_ms)

    # ── layer B: + encode ─────────────────────────────────────────────
    print("\n── B. the same, each frame pushed into an ffmpeg pipe ──")
    var encs = List[VideoEncoder]()
    for i in range(n_cam):
        encs.append(
            VideoEncoder(
                String("/tmp/mojo_rl_budget_") + String(i) + ".mp4",
                caps[i].width,
                caps[i].height,
                HZ,
            )
        )
    var b_tick = Stat()
    var b_enc = Stat()
    for _ in range(ticks):
        var t0 = perf_counter_ns()
        for i in range(n_cam):
            if not caps[i].read(bufs[i]):
                raise Error("record_budget: camera " + String(i) + " ended")
            var e0 = perf_counter_ns()
            # ⚠ BGR from OpenCV, and the encoder wants RGB24. The byte cost of
            # the swap belongs in this measurement, so it is done here rather
            # than assumed away.
            ref buf = bufs[i]
            for p in range(0, len(buf), 3):
                var t = buf[p]
                buf[p] = buf[p + 2]
                buf[p + 2] = t
            encs[i].add_frame_list(buf)
            b_enc.add(Float64(perf_counter_ns() - e0) / 1e6, budget_ms)
        b_tick.add(Float64(perf_counter_ns() - t0) / 1e6, budget_ms)
    _row(String("bgr->rgb + encode"), b_enc, budget_ms)
    _row(String("whole tick       "), b_tick, budget_ms)
    for i in range(n_cam):
        _ = encs[i].close()

    # ── layer C: + the leader arm ─────────────────────────────────────
    print("\n── C. the same, plus a leader `sync_read` (torque OFF) ──")
    var c_tick = Stat()
    var c_bus = Stat()
    var opened = True
    try:
        var leader = SO101Arm(String(LEADER_PORT), max_step_ticks=0)
        leader.bus.timeout_ms = 20
        leader.set_torque(False)
        var pos = InlineArray[Int32, SO101_N](fill=0)
        var dropped = 0
        for _ in range(ticks):
            var t0 = perf_counter_ns()
            for i in range(n_cam):
                if not caps[i].read(bufs[i]):
                    raise Error("record_budget: camera ended")
            var s0 = perf_counter_ns()
            if leader.read_positions(Span(pos)) != SO101_N:
                dropped += 1
            c_bus.add(Float64(perf_counter_ns() - s0) / 1e6, budget_ms)
            c_tick.add(Float64(perf_counter_ns() - t0) / 1e6, budget_ms)
        _row(String("leader sync_read"), c_bus, budget_ms)
        _row(String("whole tick      "), c_tick, budget_ms)
        print("  partial bus reads: " + String(dropped) + "/" + String(ticks))
    except e:
        opened = False
        print("  leader not available (" + String(e) + ") — layers A and B stand")

    for i in range(n_cam):
        caps[i].close()

    # ── layer D: the same, but each camera on its OWN THREAD ──────────
    print("\n── D. threaded cameras + encode + leader, paced at "
          + String(HZ) + " Hz ──")
    var readers = List[CameraReader]()
    for i in range(len(devices)):
        var rd = CameraReader(devices[i], 640, 480, 30.0)
        rd.start()
        readers.append(rd^)
    var d_encs = List[VideoEncoder]()
    for i in range(n_cam):
        d_encs.append(
            VideoEncoder(
                String("/tmp/mojo_rl_budget_d") + String(i) + ".mp4",
                readers[i].width, readers[i].height, HZ,
            )
        )
    var dbufs = List[List[UInt8]]()
    for i in range(n_cam):
        dbufs.append(
            List[UInt8](unsafe_uninit_length = readers[i].frame_bytes())
        )

    # Let the rings fill so the first ticks are not measuring warm-up.
    _ = sleep_us(300000)

    var d_work = Stat()
    var d_starve = 0
    var period_ns = 1_000_000_000 // HZ
    var d_leader = SO101Arm(String(LEADER_PORT), max_step_ticks=0)
    d_leader.bus.timeout_ms = 20
    d_leader.set_torque(False)
    var dpos = InlineArray[Int32, SO101_N](fill=0)

    for _ in range(ticks):
        var t0 = perf_counter_ns()
        for i in range(n_cam):
            if not readers[i].take(dbufs[i]):
                d_starve += 1
                continue
            ref buf = dbufs[i]
            for p in range(0, len(buf), 3):
                var t = buf[p]
                buf[p] = buf[p + 2]
                buf[p + 2] = t
            d_encs[i].add_frame_list(buf)
        _ = d_leader.read_positions(Span(dpos))
        # WORK only — the wait to the next tick is not a cost, it is slack.
        d_work.add(Float64(perf_counter_ns() - t0) / 1e6, budget_ms)
        while perf_counter_ns() < t0 + period_ns:
            pass

    _row(String("work per tick   "), d_work, budget_ms)
    var d_drop = 0
    for i in range(n_cam):
        d_drop += readers[i].dropped()
    print(
        "  starved takes " + String(d_starve) + "/" + String(ticks * n_cam)
        + "   camera drops " + String(d_drop)
    )
    for i in range(n_cam):
        _ = d_encs[i].close()
        readers[i].stop()
    d_leader.set_torque(False)

    # ── layer E: cameras AND encoders threaded ────────────────────────
    print("\n── E. threaded cameras + THREADED encoders + leader, "
          + String(HZ) + " Hz ──")
    var e_readers = List[CameraReader]()
    for i in range(len(devices)):
        var rd = CameraReader(devices[i], 640, 480, 30.0)
        rd.start()
        e_readers.append(rd^)
    var e_encs = List[VideoEncoderThread]()
    for i in range(n_cam):
        var et = VideoEncoderThread(
            String("/tmp/mojo_rl_budget_e") + String(i) + ".mp4",
            e_readers[i].width, e_readers[i].height, HZ,
        )
        et.start()
        e_encs.append(et^)
    var ebufs = List[List[UInt8]]()
    for i in range(n_cam):
        ebufs.append(
            List[UInt8](unsafe_uninit_length = e_readers[i].frame_bytes())
        )
    _ = sleep_us(300000)

    var e_work = Stat()
    var e_starve = 0
    var e_leader = SO101Arm(String(LEADER_PORT), max_step_ticks=0)
    e_leader.bus.timeout_ms = 20
    e_leader.set_torque(False)
    var epos = InlineArray[Int32, SO101_N](fill=0)

    for _ in range(ticks):
        var t0 = perf_counter_ns()
        for i in range(n_cam):
            if not e_readers[i].take(ebufs[i]):
                e_starve += 1
                continue
            ref buf = ebufs[i]
            for p in range(0, len(buf), 3):
                var t = buf[p]
                buf[p] = buf[p + 2]
                buf[p + 2] = t
            _ = e_encs[i].submit(buf)
        _ = e_leader.read_positions(Span(epos))
        e_work.add(Float64(perf_counter_ns() - t0) / 1e6, budget_ms)
        while perf_counter_ns() < t0 + period_ns:
            pass

    _row(String("work per tick   "), e_work, budget_ms)
    var e_cdrop = 0
    var e_edrop = 0
    for i in range(n_cam):
        e_cdrop += e_readers[i].dropped()
        e_edrop += e_encs[i].dropped()
    print(
        "  starved takes " + String(e_starve) + "/" + String(ticks * n_cam)
        + "   camera drops " + String(e_cdrop)
        + "   encoder drops " + String(e_edrop)
    )
    for i in range(n_cam):
        _ = e_encs[i].stop()
        e_readers[i].stop()
    e_leader.set_torque(False)

    # ── the verdict ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    var serial = b_tick.worst_ms
    if opened and c_tick.worst_ms > serial:
        serial = c_tick.worst_ms
    print(
        "serial  cameras: mean " + fixed(b_tick.mean(), 2) + " ms, worst "
        + fixed(serial, 2) + " ms"
    )
    print(
        "threaded cameras: mean " + fixed(d_work.mean(), 2) + " ms, worst "
        + fixed(d_work.worst_ms, 2) + " ms"
    )
    print(
        "  + threaded encoders: mean " + fixed(e_work.mean(), 2)
        + " ms, worst " + fixed(e_work.worst_ms, 2) + " ms"
    )
    if e_work.worst_ms <= budget_ms:
        print(
            "=> FITS: every tick inside " + fixed(budget_ms, 1)
            + " ms with cameras AND encoders threaded."
        )
    else:
        print(
            "=> STILL DOES NOT FIT (" + fixed(e_work.worst_ms, 2)
            + " ms). Do not add another thread on a hunch — attribute the"
            " spike first."
        )
    print("=" * 72)
