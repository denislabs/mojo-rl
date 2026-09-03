"""The camera->SigLIP chain, against torch's own `F.interpolate`.

Needs the dump:
  pixi run -e act-ref python tools/vla/dump_smolvla_image_reference.py --out /tmp/vla_img
Run:
  pixi run mojo run -I . tests/deep_agents/smolvla/test_image_preprocess.mojo

Five frames through the reference's exact chain — uint8 HWC, /255, CHW,
`resize_with_pad(512,512,pad_value=0)`, `*2-1` — covering the four ways this
goes silently wrong:

  * `land` 640x480, the SO-101 camera: downscale, 128 blank rows on TOP.
  * `store` 320x240, what our TrajectoryStore holds: the SAME output shape by
    UPSCALING, which exercises the `real < 0` clamp at the top-left edge.
  * `port` 480x640: the LEFT-pad branch, which the centred-padding sibling
    `resize_with_pad_torch` would fail.
  * `exact` 512x512: the early-out — no resample at all.
  * `odd` 700x510: `int()` truncating 373.06 to 373, so a 139-row pad.

⚠ THE GATE THAT MATTERS MOST IS THE LAST ONE. Everything above would also pass
with PIL's bilinear substituted in the ~1e-2 range if the tolerance were loose,
so the file ALSO runs the repo's existing `pil_bilinear_u8` chain through the
same comparison and asserts it is REJECTED. A gate that cannot fail on the one
wrong implementation actually sitting in the tree is not gating the thing it
claims to.
"""

from std.math import abs
from std.testing import assert_true, assert_equal

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.act.refload import RefDump
from mojo_rl.vision.resize_pad import (
    camera_frame_to_siglip,
    resize_with_pad_chw,
    SIGLIP_INPUT,
)
from mojo_rl.vision.preprocess import pil_bilinear_u8
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.deep_agents.smolvla.observation import fill_camera_images

comptime SIZE = SIGLIP_INPUT
comptime N = 3 * SIZE * SIZE
comptime TOL: Float32 = 1.0e-6
"""Tight on purpose: values live in [-1,1], so 1e-6 is ~8 ulp and PIL-vs-torch
is four orders above it. A tolerance that admitted the wrong filter would be
worthless."""


struct Case(Copyable):
    var name: String
    var w: Int
    var h: Int
    var pad_top: Int
    var pad_left: Int

    def __init__(out self, var name: String, w: Int, h: Int, t: Int, l: Int):
        self.name = name^
        self.w = w
        self.h = h
        self.pad_top = t
        self.pad_left = l


def compare(
    want: List[Scalar[DT]], got: List[Float32], what: String
) raises -> Float32:
    """Max |diff| plus the count outside tolerance, both printed.

    ⚠ Rows COMPARED is printed beside rows DIFFERING: "0 mismatches" over an
    empty comparison is this repo's default failure mode."""
    assert_equal(len(want), N, what + ": reference is the wrong size")
    var worst: Float32 = 0.0
    var bad = 0
    for i in range(N):
        var d = abs(Float32(want[i]) - got[i])
        if d > worst:
            worst = d
        if d > TOL:
            bad += 1
    print(
        "      compared", N, " outside", TOL, ":", bad, " max |d|", worst
    )
    assert_equal(bad, 0, what + ": disagrees with torch")
    return worst


def main() raises:
    print("=" * 70)
    print("SmolVLA image preprocessing vs torch")
    print("=" * 70)

    var dump = RefDump(String("/tmp/vla_img"))
    var cases = List[Case]()
    cases.append(Case(String("land"), 640, 480, 128, 0))
    cases.append(Case(String("store"), 320, 240, 128, 0))
    cases.append(Case(String("port"), 480, 640, 0, 128))
    cases.append(Case(String("exact"), 512, 512, 0, 0))
    cases.append(Case(String("odd"), 700, 510, 139, 0))

    var worst_all: Float32 = 0.0
    for ci in range(len(cases)):
        var c = cases[ci].copy()
        print("  [" + String(ci + 1) + "]", c.name, String(c.w) + "x"
              + String(c.h), " pad top", c.pad_top, "left", c.pad_left)

        var src = dump.get(String("in_") + c.name)
        var expect = dump.get(String("out_") + c.name)
        assert_equal(
            len(src), c.w * c.h * 3, c.name + ": input is the wrong size"
        )

        var frame = List[UInt8](unsafe_uninit_length=c.w * c.h * 3)
        for i in range(len(src)):
            frame[i] = UInt8(Int(Float64(src[i])))

        var got = List[Float32]()
        camera_frame_to_siglip(frame, c.w, c.h, False, got, 0, SIZE)
        worst_all = max(worst_all, compare(expect, got, c.name))

        # --- anti-vacuity: the blank band is real, is -1, and is where the
        # reference put it. A chain that produced a uniform image would sail
        # through a max|d| check against its own uniform reference.
        var band = c.pad_top * SIZE + c.pad_left * (SIZE - c.pad_top)
        if band > 0:
            var pad_wrong = 0
            var pad_seen = 0
            for y in range(SIZE):
                for x in range(SIZE):
                    if y < c.pad_top or x < c.pad_left:
                        pad_seen += 1
                        if abs(got[y * SIZE + x] + 1.0) > 1.0e-7:
                            pad_wrong += 1
            print("      pad cells", pad_seen, " not -1:", pad_wrong)
            assert_true(pad_seen > 0, c.name + ": expected a blank band")
            assert_equal(pad_wrong, 0, c.name + ": pad is not -1")

        var picture_var = 0
        for i in range(N):
            if abs(got[i] + 1.0) > 0.05:
                picture_var += 1
        assert_true(
            picture_var > N // 4,
            c.name + ": the image is nearly all pad — the resize did nothing",
        )

    print("  worst across all five cases:", worst_all)

    # ------------------------------------------------------------------
    # [6] The gate can reject the wrong filter. `pil_bilinear_u8` is a real,
    # correct, gated implementation that is simply the WRONG ONE here.
    # ------------------------------------------------------------------
    print("  [6] would PIL's bilinear pass this gate?")
    var c = cases[0].copy()
    var src = dump.get(String("in_land"))
    var expect = dump.get(String("out_land"))
    var frame = List[UInt8](unsafe_uninit_length=c.w * c.h * 3)
    for i in range(len(src)):
        frame[i] = UInt8(Int(Float64(src[i])))

    var pil_hwc = List[UInt8]()
    pil_bilinear_u8(frame, c.w, c.h, 3, pil_hwc, 512, 384)
    var pil = List[Float32](unsafe_uninit_length=N)
    for i in range(N):
        pil[i] = -1.0
    var px = 512 * 384
    for i in range(px):
        var y = i // 512
        var x = i % 512
        var d = (128 + y) * SIZE + x
        for ch in range(3):
            pil[ch * SIZE * SIZE + d] = (
                Float32(Int(pil_hwc[i * 3 + ch])) / 255.0
            ) * 2.0 - 1.0

    var pil_worst: Float32 = 0.0
    var pil_bad = 0
    for i in range(N):
        var dd = abs(Float32(expect[i]) - pil[i])
        if dd > pil_worst:
            pil_worst = dd
        if dd > TOL:
            pil_bad += 1
    print("      PIL chain: outside", TOL, ":", pil_bad, " max |d|", pil_worst)
    assert_true(
        pil_bad > N // 100,
        "PIL's bilinear passed a gate meant to pin torch's — the tolerance is"
        " too loose to see the filter it was written to distinguish",
    )
    print(
        "      REJECTED, as it must be:", pil_bad, "of", N,
        "values differ — the two filters are not interchangeable"
    )

    # ------------------------------------------------------------------
    # [7] The multi-camera driver: block k belongs to camera k.
    # Both blocks are 786,432 floats of the same shape, so an order swap is
    # invisible to every check except this one.
    # ------------------------------------------------------------------
    print("  [7] two cameras -> [N_CAM, 3*512*512], in order")
    var frames = List[List[UInt8]]()
    var widths = List[Int]()
    var heights = List[Int]()
    var names = List[String]()
    names.append(String("land"))
    names.append(String("port"))
    for k in range(2):
        var c2 = cases[0].copy() if k == 0 else cases[2].copy()
        var raw = dump.get(String("in_") + c2.name)
        var f = List[UInt8](unsafe_uninit_length=len(raw))
        for i in range(len(raw)):
            f[i] = UInt8(Int(Float64(raw[i])))
        frames.append(f^)
        widths.append(c2.w)
        heights.append(c2.h)

    var images = Tensor()
    var scratch = List[Float32]()
    fill_camera_images["cpu", 2, SIZE](
        frames, widths, heights, False, images, scratch
    )

    for k in range(2):
        var want = dump.get(String("out_") + names[k])
        var bad = 0
        for i in range(N):
            if abs(Float32(want[i]) - Float32(images.data[k * N + i])) > TOL:
                bad += 1
        print("      block", k, "=", names[k], " compared", N, " wrong", bad)
        assert_equal(
            bad, 0,
            "camera block " + String(k) + " is not " + names[k]
            + " — the blocks are the same size, so only ordering can tell",
        )

    # a dropped frame must RAISE, not become a black image: SmolVLA's
    # `empty_cameras` default is 0, so the reference would raise too.
    var short_frames = List[List[UInt8]]()
    short_frames.append(frames[0].copy())
    var raised = False
    try:
        fill_camera_images["cpu", 2, SIZE](
            short_frames, widths, heights, False, images, scratch
        )
    except:
        raised = True
    assert_true(raised, "a missing camera must raise, not pad with black")
    print("      a missing camera raises")

    print("PASSED — the camera chain reproduces torch's resize_with_pad")
