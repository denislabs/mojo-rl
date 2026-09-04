# +--------------------------------------------------------------------------+ #
# | The two PIL-bilinear implementations, against each other
# +--------------------------------------------------------------------------+ #
"""Gate that the DEPLOY resize and the IMPORT resize produce the same bytes.

    pixi run mojo run -I . tests/vision/test_resize_deploy_vs_import.mojo

⚠⚠ **THERE ARE TWO IMPLEMENTATIONS OF PIL's BILINEAR IN THIS REPO, AND ONE
TRAIN/DEPLOY BOUNDARY RUNS BETWEEN THEM.**

    mojo_rl/io/image.mojo       `resize_bilinear_pil`   built the dataset
    mojo_rl/vision/preprocess.mojo `pil_bilinear_u8`    feeds the live camera

`act_so101_deploy_real.mojo` asks a policy trained on the first to act on
pixels produced by the second. Both are separately gated against PIL itself,
which is the right oracle and is what makes each one correct — but neither
gate can see the other, and neither runs at the geometry the deployment uses
(640x480 -> 320x240, the SO-101 rig's).

That is the exact shape of `_a_gate_that_shares_its_reference_implementation_is_blind`
turned inside out: two implementations, two gates, and nothing asserting the
one property the robot depends on — that they agree, HERE, on this reduction.
A divergence would not raise and would not fail either existing gate. It would
present as a policy that behaved on the dataset and behaves worse on the arm,
which is the most expensive failure this project knows how to produce.

⚠ EXACT EQUALITY. Both claim to reproduce the same fixed-point arithmetic, so
anything but bit equality means one of them has drifted from it.

⚠ NO CAMERA, NO PIL, NO NETWORK. The committed 640x480 fixture is decoded with
our own PNG reader.

## Leg 4: a SAME-SIZE resize must be the IDENTITY

⚠⚠ **A 26.5 GiB STORE RESTS ON THIS.** `import_lerobot_v3` ALWAYS calls
`resize_bilinear_pil`, with no early-out when the target equals the source. So
importing the SO-101 recording at its native 480x640 — which is what the
SmolVLA path needs, because SmolVLA wants torch's bilinear applied later and
NOT PIL's applied twice — is lossless only if a same-size resize returns its
input byte for byte.

The arithmetic says it must: at scale 1.0 the support is 1.0, the window is
three taps, the triangle puts weight 1 on the centre and 0 on its neighbours,
and the fixed-point accumulator (`1 << 21` then `>> 22`) rounds an integer
straight back to itself. That is a chain of four things being individually
right, which is a description of code that is usually right rather than a
proof, and the cost of being wrong is a 26.5 GiB dataset that is quietly a
blurred copy of the recording.

⚠ The other three legs all DOWNSCALE. Scale 1.0 is the corner they never
reach, and it is the only one the native import uses.
"""

from std.pathlib import Path

from mojo_rl.io.image import resize_bilinear_pil
from mojo_rl.io.png import load_png_file
from mojo_rl.vision.preprocess import pil_bilinear_u8

comptime FIX = "tests/fixtures/vision/marker_640x480.png"
comptime SRC_W = 640
comptime SRC_H = 480
comptime DST_W = 320
comptime DST_H = 240
"""⚠ THE DEPLOYMENT'S OWN GEOMETRY, not a convenient one. 640x480 is what
`meta/info.json` records for `DenisLabs/record-test_*`
(`observation.images.front.shape = [480, 640, 3]`) and 320x240 is
`SO101_IMG_W` x `SO101_IMG_H`. A 2x reduction is also the case where the
filter window grows to four taps, which is the whole reason these files
exist."""


def _uptr(
    mut lst: List[UInt8],
) -> Pointer[Scalar[DType.uint8], MutAnyOrigin]:
    return (
        lst.unsafe_ptr()
        .unsafe_bitcast[Scalar[DType.uint8]]()
        .as_unsafe_any_origin()
    )


def compare(name: String, ref a: List[UInt8], ref b: List[UInt8], n: Int
) raises -> Int:
    """Bit compare, reporting what was COMPARED beside what DIFFERED.

    ⚠ Both numbers, always. "0 differ" over 0 bytes is the default failure of
    a gate like this one and it looks exactly like success.
    """
    var differing = 0
    var worst = 0
    var first = -1
    for i in range(n):
        if a[i] != b[i]:
            differing += 1
            var d = Int(a[i]) - Int(b[i])
            if d < 0:
                d = -d
            if d > worst:
                worst = d
            if first < 0:
                first = i
    print(
        "  " + name + ": compared " + String(n) + " bytes, "
        + String(differing) + " differ"
        + (
            "  (worst " + String(worst) + " at " + String(first) + ")"
            if differing > 0 else ""
        )
    )
    return differing


def main() raises:
    print("=" * 70)
    print("deploy resize vs import resize, at the SO-101 rig's geometry")
    print("=" * 70)

    var failures = 0
    var compared = 0

    # ── leg 1: a real photograph at exactly the deployment's size ──────────
    if not Path(String(FIX)).exists():
        print("SKIP: missing " + String(FIX))
        return
    var img = load_png_file(String(FIX))
    if img.width != SRC_W or img.height != SRC_H or img.channels != 3:
        raise Error(
            "fixture is " + String(img.width) + "x" + String(img.height)
            + "x" + String(img.channels) + ", expected 640x480x3"
        )

    var want = List[UInt8](length=DST_W * DST_H * 3, fill=0)
    var scratch = List[UInt8]()
    resize_bilinear_pil(
        _uptr(img.pixels), SRC_H, SRC_W, _uptr(want), DST_H, DST_W, scratch, 3
    )
    var got = List[UInt8]()
    pil_bilinear_u8(img.pixels, SRC_W, SRC_H, 3, got, DST_W, DST_H)
    failures += compare(String("photo   640x480 -> 320x240"), got, want,
                        DST_W * DST_H * 3)
    compared += DST_W * DST_H * 3

    # ── leg 2: high spatial frequency, where a window bug cannot hide ──────
    # ⚠ A PHOTOGRAPH IS A WEAK ADVERSARY for a resampling filter: it is
    # locally smooth, so a wrong coefficient or an off-by-one window still
    # lands within a byte of the right answer over most of the image. MEASURED,
    # by pinning `support` to 1.0 in `preprocess.mojo` (which is the bug this
    # whole filter family exists to avoid — the window must GROW with the
    # reduction factor) and reading what each leg saw:
    #
    #     photo 640x480 -> 320x240      1,872 of 230,400 bytes differ   0.8%
    #     noise 640x480 -> 320x240    225,724 of 230,400 bytes differ  98.0%
    #
    # Same bug, 120x the signal. The photograph leg would still have failed
    # here, but a subtler drift is exactly the kind that survives 0.8% and
    # this is the leg that would catch it.
    var noise = List[UInt8](unsafe_uninit_length=SRC_W * SRC_H * 3)
    var st = UInt64(0x243F6A8885A308D3)
    for i in range(SRC_W * SRC_H * 3):
        st = st * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        noise[i] = UInt8((st >> 33) & UInt64(0xFF))
    var want2 = List[UInt8](length=DST_W * DST_H * 3, fill=0)
    resize_bilinear_pil(
        _uptr(noise), SRC_H, SRC_W, _uptr(want2), DST_H, DST_W, scratch, 3
    )
    var got2 = List[UInt8]()
    pil_bilinear_u8(noise, SRC_W, SRC_H, 3, got2, DST_W, DST_H)
    failures += compare(String("noise   640x480 -> 320x240"), got2, want2,
                        DST_W * DST_H * 3)
    compared += DST_W * DST_H * 3

    # ── leg 3: a NON-INTEGER ratio ────────────────────────────────────────
    # 2x is the tidy case — window bounds land on clean boundaries. A rig with
    # a different camera would not be so lucky, and `--width`/`--height` exist
    # for exactly that.
    var want3 = List[UInt8](length=213 * 157 * 3, fill=0)
    resize_bilinear_pil(
        _uptr(img.pixels), SRC_H, SRC_W, _uptr(want3), 157, 213, scratch, 3
    )
    var got3 = List[UInt8]()
    pil_bilinear_u8(img.pixels, SRC_W, SRC_H, 3, got3, 213, 157)
    failures += compare(String("photo   640x480 -> 213x157"), got3, want3,
                        213 * 157 * 3)
    compared += 213 * 157 * 3

    # ── leg 4: identity at scale 1.0, BOTH implementations ───────────────
    # ⚠ BOTH, and `resize_bilinear_pil` FIRST — it is the one the importer
    # calls and therefore the one the store depends on. Checking only the
    # deploy-side `pil_bilinear_u8` here would be checking the wrong function;
    # legs 1-3 make them equal on DOWNSCALES, which does not extend to a case
    # they never exercise.
    print("")
    print("  leg 4: same-size resize must be the identity")
    var same_i = List[UInt8](length=SRC_W * SRC_H * 3, fill=0)
    resize_bilinear_pil(
        _uptr(img.pixels), SRC_H, SRC_W, _uptr(same_i), SRC_H, SRC_W,
        scratch, 3
    )
    failures += compare(
        "import  resize_bilinear_pil(640x480 -> 640x480) vs input",
        img.pixels, same_i, SRC_W * SRC_H * 3,
    )
    compared += SRC_W * SRC_H * 3

    var same_d = List[UInt8]()
    pil_bilinear_u8(img.pixels, SRC_W, SRC_H, 3, same_d, SRC_W, SRC_H)
    failures += compare(
        "deploy  pil_bilinear_u8(640x480 -> 640x480)      vs input",
        img.pixels, same_d, SRC_W * SRC_H * 3,
    )
    compared += SRC_W * SRC_H * 3

    print("")
    if failures != 0:
        print(
            "[FAIL] the two PIL-bilinear implementations disagree. The"
            " deployment path and the\n       dataset it trained on are"
            " producing different pixels — read this file's header."
        )
        raise Error("resize implementations disagree")
    print(
        "[PASS] " + String(compared) + " bytes compared across 5"
        " comparisons (3 downscales + 2 identities), 0 differ"
    )
