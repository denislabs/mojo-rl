# +--------------------------------------------------------------------------+ #
# | VideoRecorder: does what comes back out match what went in?
# +--------------------------------------------------------------------------+ #
"""Gates `mojo_rl/render/video_recorder.mojo` by reading its own output back.

    pixi run mojo run -I . tests/render/test_video_recorder.mojo

Self-contained: it needs `ffmpeg` on PATH and nothing else — no reference
dump, no dataset, no display.

⚠ "A FILE APPEARED" IS NOT A PASS. The recorder feeds a BGRA buffer to a pipe
and something plausible comes out regardless of whether the channel order is
right, the crop offset is right, or the rows are in order — a swapped R and B
produces a perfectly good video of the wrong colour. So every check here reads
the output back through `mojo_rl/io/video` and compares pixels.

The still path carries the pixel assertions because **PNG is lossless**: what
comes back is exactly what was written, so channel order and crop can be
checked at zero tolerance. H.264 cannot be checked that way, so the video path
asserts the things that survive lossy encoding — frame count, dimensions, and
that a mid-recording geometry change is refused rather than sheared.

The test frame is a gradient that differs in all three channels and in both
axes, so a swap, a transpose or a row offset each move a pixel that a flat or
symmetric frame would leave alone.
"""

from std.os.path import exists

from mojo_rl.io.video import VideoDecoder, probe_video
from mojo_rl.nn.core.ptr import mptr
from mojo_rl.render.video_recorder import VideoRecorder


comptime W = 64
comptime H = 48
comptime N_FRAMES = 10


def blue(x: Int, y: Int, k: Int) -> Int:
    return (x * 3 + k * 11) % 256


def green(x: Int, y: Int, k: Int) -> Int:
    return (y * 5 + k * 7) % 256


def red(x: Int, y: Int, k: Int) -> Int:
    return (x + y * 2 + k * 13) % 256


def fill(mut buf: List[UInt8], k: Int):
    for y in range(H):
        for x in range(W):
            var o = (y * W + x) * 4
            buf[o] = UInt8(blue(x, y, k))
            buf[o + 1] = UInt8(green(x, y, k))
            buf[o + 2] = UInt8(red(x, y, k))
            buf[o + 3] = UInt8(255)


def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


def read_rgb(path: String, mut out: List[UInt8]) raises -> Int:
    """Decode a still or a video through `io/video`; returns frames read."""
    var d = VideoDecoder(String(path))
    out.resize(d.frame_bytes, UInt8(0))
    var n = 0
    while d.next_into(mptr(out)):
        n += 1
        if n == 1:
            # Keep frame 0; later frames overwrite, which is fine for the
            # counting checks below.
            pass
    d.close()
    return n


def main() raises:
    print("VideoRecorder -> ffmpeg -> io/video round trip")
    print("")
    var fails = 0
    var buf = List[UInt8](unsafe_uninit_length = W * H * 4)

    # ── still, uncropped: the pixel-exact leg ─────────────────────────
    fill(buf, 3)
    var rec = VideoRecorder()
    var png = String("/tmp/mojo_rec_gate.png")
    rec.save_frame_bgra(Int(mptr(buf)), W, H, png)

    var info = probe_video(png)
    check(
        fails,
        "still: dimensions preserved",
        info.width == W and info.height == H,
        String(info.width) + "x" + String(info.height),
    )

    var got = List[UInt8]()
    var nf = read_rgb(png, got)
    var bad = 0
    var first = String("")
    if len(got) != W * H * 3:
        check(fails, "still: decoded size", False, String(len(got)) + " bytes")
    else:
        for y in range(H):
            for x in range(W):
                var o = (y * W + x) * 3
                var want_r = red(x, y, 3)
                var want_g = green(x, y, 3)
                var want_b = blue(x, y, 3)
                if (
                    Int(got[o]) != want_r
                    or Int(got[o + 1]) != want_g
                    or Int(got[o + 2]) != want_b
                ):
                    bad += 1
                    if first == "":
                        first = (
                            " first at (" + String(x) + "," + String(y)
                            + "): got RGB " + String(Int(got[o])) + ","
                            + String(Int(got[o + 1])) + ","
                            + String(Int(got[o + 2])) + " want "
                            + String(want_r) + "," + String(want_g) + ","
                            + String(want_b)
                        )
        # ⚠ This is the check that catches a BGRA->RGB swap. It is exact
        # because PNG is lossless; any tolerance here would pass a recorder
        # that writes blue as red.
        check(
            fails,
            "still: every pixel exact (BGRA -> RGB order)",
            bad == 0,
            String(W * H) + " pixels compared" if bad == 0
            else String(bad) + " differ." + first,
        )

    # ── still, cropped: does crop_x actually offset? ───────────────────
    comptime CX = 8
    comptime CW = 33
    var cpng = String("/tmp/mojo_rec_gate_crop.png")
    rec.save_frame_bgra(Int(mptr(buf)), W, H, cpng, CX, CW)
    var cinfo = probe_video(cpng)
    check(
        fails,
        "crop: width follows crop_w (odd width kept on a still)",
        cinfo.width == CW and cinfo.height == H,
        String(cinfo.width) + "x" + String(cinfo.height),
    )
    var cgot = List[UInt8]()
    _ = read_rgb(cpng, cgot)
    var cbad = 0
    if len(cgot) == CW * H * 3:
        for y in range(H):
            for x in range(CW):
                var o = (y * CW + x) * 3
                if (
                    Int(cgot[o]) != red(x + CX, y, 3)
                    or Int(cgot[o + 1]) != green(x + CX, y, 3)
                    or Int(cgot[o + 2]) != blue(x + CX, y, 3)
                ):
                    cbad += 1
    else:
        cbad = -1
    check(
        fails,
        "crop: pixels come from column crop_x onward",
        cbad == 0,
        String(CW * H) + " pixels compared" if cbad == 0
        else String(cbad) + " differ",
    )

    # ── video: frame count and geometry survive ───────────────────────
    var mp4 = String("/tmp/mojo_rec_gate.mp4")
    rec.start(mp4, fps=30)
    for k in range(N_FRAMES):
        fill(buf, k)
        rec.add_frame_bgra(Int(mptr(buf)), W, H)
    rec.stop()

    var vgot = List[UInt8]()
    var vn = read_rgb(mp4, vgot)
    check(
        fails,
        "video: every frame encoded",
        vn == N_FRAMES,
        String(vn) + " frames decoded, wrote " + String(N_FRAMES),
    )
    var vinfo = probe_video(mp4)
    check(
        fails,
        "video: dimensions preserved",
        vinfo.width == W and vinfo.height == H,
        String(vinfo.width) + "x" + String(vinfo.height),
    )

    # ── skip ──────────────────────────────────────────────────────────
    var smp4 = String("/tmp/mojo_rec_gate_skip.mp4")
    rec.start(smp4, fps=30, skip=3)
    for k in range(9):
        fill(buf, k)
        rec.add_frame_bgra(Int(mptr(buf)), W, H)
    rec.stop()
    var sgot = List[UInt8]()
    var sn = read_rgb(smp4, sgot)
    check(
        fails,
        "skip=3 encodes every third frame",
        sn == 3,
        String(sn) + " frames from 9 calls",
    )

    # ── odd dimensions must not abort the recording ───────────────────
    # libx264 refuses odd width/height; the recorder rounds DOWN rather than
    # letting the encoder die at the first frame. The old code only did this
    # for an explicit crop_w, so an odd window aborted.
    comptime OW = 63
    comptime OH = 47
    var obuf = List[UInt8](unsafe_uninit_length = OW * OH * 4)
    for i in range(OW * OH * 4):
        obuf[i] = UInt8((i * 7) % 256)
    var ompt = String("/tmp/mojo_rec_gate_odd.mp4")
    rec.start(ompt, fps=30)
    var odd_ok = True
    var odd_err = String("")
    try:
        for _ in range(4):
            rec.add_frame_bgra(Int(mptr(obuf)), OW, OH)
        rec.stop()
    except e:
        odd_ok = False
        odd_err = String(e)
    check(fails, "odd 63x47 window records", odd_ok, odd_err)
    if odd_ok:
        var oinfo = probe_video(ompt)
        check(
            fails,
            "odd window rounded down to even",
            oinfo.width == 62 and oinfo.height == 46,
            String(oinfo.width) + "x" + String(oinfo.height),
        )

    # ── a geometry change mid-recording is refused ────────────────────
    var gmp4 = String("/tmp/mojo_rec_gate_geom.mp4")
    rec.start(gmp4, fps=30)
    fill(buf, 0)
    rec.add_frame_bgra(Int(mptr(buf)), W, H)
    var refused = False
    try:
        rec.add_frame_bgra(Int(mptr(obuf)), OW, OH)
    except e:
        refused = True
    try:
        rec.stop()
    except:
        pass
    check(
        fails,
        "geometry change mid-recording raises",
        refused,
        "a rawvideo pipe has one fixed frame size",
    )

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("video recorder gate failed")
