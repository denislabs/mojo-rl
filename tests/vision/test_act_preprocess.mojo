"""Gate the camera -> ACT tensor path against PIL, bit for bit.

    pixi run build-opencv
    pixi run mojo run -I . tests/vision/test_act_preprocess.mojo

⚠⚠ EXACT EQUALITY, NOT A TOLERANCE, AND FOR A DIFFERENT REASON THAN THE SHIM
GATE. `test_opencv_shim.mojo` can demand bit equality because both sides call
the same dylib. Here nothing is shared: `pil_bilinear_u8` is a REIMPLEMENTATION
of PIL's `libImaging/Resample.c` in Mojo. Exactness is therefore a claim about
having reproduced an algorithm, not about having called one — which is a
stronger statement, and the only one worth making. A tolerance here would mean
"we resize approximately like the thing that built the training set", which is
precisely the gap this code exists to close.

⚠ NO CAMERA, NO DATASET, NO PIL. The source frame and PIL's answers are
committed; `tools/vision/make_resize_fixture.py` regenerates them.
"""

from std.pathlib import Path

from mojo_rl.vision.opencv import imread, opencv_shim_available
from mojo_rl.vision.preprocess import camera_frame_to_chw_rgb, pil_bilinear_u8

comptime FIX = "tests/fixtures/vision/"
comptime SRC_W = 320
comptime SRC_H = 240


def read_bin(path: String) raises -> List[UInt8]:
    with open(path, "r") as f:
        var raw = f.read_bytes()
        var out = List[UInt8](capacity=len(raw))
        for i in range(len(raw)):
            out.append(raw[i])
        return out^


def compare(
    name: String,
    got: List[UInt8],
    want: List[UInt8],
    n: Int,
    mut failures: Int,
) -> Int:
    """Bit compare, reporting what was COMPARED beside what DIFFERED."""
    var differing = 0
    var worst = 0
    var first_bad = -1
    for i in range(n):
        if got[i] != want[i]:
            differing += 1
            var d = Int(got[i]) - Int(want[i]) if got[i] > want[i] else Int(
                want[i]
            ) - Int(got[i])
            if d > worst:
                worst = d
            if first_bad < 0:
                first_bad = i
    if differing != 0:
        print(
            "  FAIL:",
            name,
            "-",
            differing,
            "of",
            n,
            "bytes differ, worst",
            worst,
            "at",
            first_bad,
        )
        failures += 1
    else:
        print("  ", name, "- compared", n, "bytes, 0 differ")
    return n


def main() raises:
    print("=" * 70)
    print("camera -> ACT preprocessing, against PIL BILINEAR")
    print("=" * 70)

    if not opencv_shim_available():
        print("SKIP: shim not built. Run: pixi run build-opencv")
        return
    if not Path(String(FIX) + "act_src_320x240.png").exists():
        print("SKIP: no fixture. Run:")
        print("  pixi run python tools/vision/make_resize_fixture.py")
        return

    var failures = 0
    var compared = 0

    # `imread` hands back BGR; the pinned answers are RGB, so the RGB source is
    # rebuilt here rather than trusting a second decode path.
    var bgr = List[UInt8]()
    var g = imread(String(FIX) + "act_src_320x240.png", bgr)
    if g[0] != SRC_W or g[1] != SRC_H or g[2] != 3:
        print("  FAIL: source is", g[0], "x", g[1], "x", g[2])
        failures += 1
        return
    var rgb = List[UInt8](unsafe_uninit_length=SRC_W * SRC_H * 3)
    for i in range(SRC_W * SRC_H):
        rgb[i * 3] = bgr[i * 3 + 2]
        rgb[i * 3 + 1] = bgr[i * 3 + 1]
        rgb[i * 3 + 2] = bgr[i * 3]

    # ── exact 2x, the ratio the deploy path actually uses ───────────────────
    var out2 = List[UInt8]()
    pil_bilinear_u8(rgb, SRC_W, SRC_H, 3, out2, 160, 120)
    var want2 = read_bin(String(FIX) + "act_resize_160x120.bin")
    compared += compare(
        String("resize 2x   "), out2, want2, 160 * 120 * 3, failures
    )

    # ── a NON-INTEGER ratio, where a scale-dependent bug would hide ─────────
    # ⚠ 2x is the easy case: window bounds land on tidy boundaries and a naive
    # implementation can pass it while being wrong everywhere else.
    var out3 = List[UInt8]()
    pil_bilinear_u8(rgb, SRC_W, SRC_H, 3, out3, 137, 101)
    var want3 = read_bin(String(FIX) + "act_resize_137x101.bin")
    compared += compare(
        String("resize 137x101"), out3, want3, 137 * 101 * 3, failures
    )

    # ── the whole path: BGR HWC capture -> RGB CHW, what the store holds ────
    var chw = List[UInt8]()
    camera_frame_to_chw_rgb(bgr, SRC_W, SRC_H, chw, 160, 120)
    var want_chw = read_bin(String(FIX) + "act_chw_160x120.bin")
    compared += compare(
        String("BGR->RGB CHW"), chw, want_chw, 160 * 120 * 3, failures
    )

    # ── identity, and a flat image ──────────────────────────────────────────
    # ⚠ A FLAT IMAGE THAT CHANGES VALUE MEANS THE COEFFICIENTS DO NOT SUM TO
    # ONE, which would make every comparison above meaningless even if it
    # somehow passed. Cheap, and it fails loudly.
    var flat = List[UInt8](unsafe_uninit_length=64 * 48 * 3)
    for i in range(64 * 48 * 3):
        flat[i] = 137
    var flat_out = List[UInt8]()
    pil_bilinear_u8(flat, 64, 48, 3, flat_out, 32, 24)
    var flat_bad = 0
    for i in range(32 * 24 * 3):
        if flat_out[i] != 137:
            flat_bad += 1
    if flat_bad != 0:
        print("  FAIL: flat image changed in", flat_bad, "of", 32 * 24 * 3)
        failures += 1
    else:
        print("   flat image  - 2304 bytes unchanged at 137")

    print("-" * 70)
    if failures == 0:
        print("PASS —", compared, "bytes matched PIL exactly")
    else:
        print("FAIL —", failures, "checks")
    print("=" * 70)
