# +--------------------------------------------------------------------------+ #
# | The training image and the deployment image, from the same physical frame
# +--------------------------------------------------------------------------+ #
"""Two front doors onto one filter, and the layout that differs between them.

    pixi run mojo run -I . \\
        tests/deep_agents/smolvla/test_store_vs_camera_frame.mojo

SmolVLA gets its pixels from two places that must agree:

    training    the store's `images` column — uint8 [N_CAM, 3, H, W], PLANAR,
                RGB, written by `import_lerobot_v3` from ffmpeg's rgb24
    deployment  a camera — uint8 [H, W, 3], INTERLEAVED, and BGR under OpenCV

`camera_frame_to_siglip` takes the second. `store_frame_to_siglip` takes the
first. Handing a planar buffer to the interleaved reader raises nothing and
produces a recognisable image with its colours and geometry scrambled — the
kind of wrong that survives a thumbnail and shows up only as a policy that
behaves on the dataset and worse on the arm.

This is `test_resize_deploy_vs_import.mojo`'s argument one level up. That file
gates the two PIL-bilinear implementations against each other for the ACT
path; this gates the two SigLIP front doors for the VLA path, at the geometry
the SO-101 store actually holds.

⚠ **The comparison is BIT-EXACT.** Both doors run the same
`resize_with_pad_chw` on the same numbers in the same order once the layout is
undone; anything but equality means one of them transposed, dropped a channel,
or applied the R/B swap where it did not belong.

⚠ Leg [2] is the one with teeth. Leg [1] would pass on two implementations
that are both wrong in the same way, and would also pass on a grey frame. The
fixture is a per-channel gradient with a bright marker, so a channel swap, a
transpose and an off-by-one are each visible — and leg [2] asserts that
feeding the store's planar bytes to the CAMERA door gives a DIFFERENT answer,
which is what says the distinction is real rather than decorative.
"""

from std.math import abs
from std.testing import assert_true, assert_equal

from mojo_rl.vision.resize_pad import (
    camera_frame_to_siglip, store_frame_to_siglip, SIGLIP_INPUT,
)

comptime SRC_W = 640
comptime SRC_H = 480
"""⚠ The SO-101 store's own geometry. `meta/info.json` records
`observation.images.front.shape = [480, 640, 3]` and the store was re-imported
at that native size, so this is the reduction the training path really runs —
640x480 into a 512x512 canvas is 512x384 with a 128-row blank band."""
comptime N = 3 * SIGLIP_INPUT * SIGLIP_INPUT
comptime PX = SRC_W * SRC_H


def main() raises:
    print("=" * 70)
    print("store frame vs camera frame -> SigLIP")
    print("=" * 70)
    print("  source", SRC_W, "x", SRC_H, " -> ", SIGLIP_INPUT, "x",
          SIGLIP_INPUT)

    # ── the fixture ──────────────────────────────────────────────────────
    # A per-channel gradient plus a bright marker block. Each channel is a
    # different function of (x, y), so a channel permutation cannot hide; the
    # marker is off-centre, so a transpose or a flip cannot either.
    var planar = List[Scalar[DType.uint8]](unsafe_uninit_length=PX * 3)
    for y in range(SRC_H):
        for x in range(SRC_W):
            var i = y * SRC_W + x
            planar[i] = Scalar[DType.uint8]((x * 255) // (SRC_W - 1))
            planar[PX + i] = Scalar[DType.uint8]((y * 255) // (SRC_H - 1))
            planar[2 * PX + i] = Scalar[DType.uint8](
                ((x + y) * 255) // (SRC_W + SRC_H - 2)
            )
    for y in range(40, 90):
        for x in range(500, 590):
            var i = y * SRC_W + x
            planar[i] = Scalar[DType.uint8](250)
            planar[PX + i] = Scalar[DType.uint8](8)
            planar[2 * PX + i] = Scalar[DType.uint8](130)

    # the same physical frame as a camera would hand it over: interleaved, and
    # BGR, so the camera door's swap_rb has to undo it.
    var interleaved = List[UInt8](unsafe_uninit_length=PX * 3)
    for i in range(PX):
        interleaved[i * 3 + 0] = UInt8(Int(planar[2 * PX + i]))   # B
        interleaved[i * 3 + 1] = UInt8(Int(planar[PX + i]))       # G
        interleaved[i * 3 + 2] = UInt8(Int(planar[i]))            # R

    # ── [1] the two doors agree, bit for bit ─────────────────────────────
    var from_store = List[Float32]()
    store_frame_to_siglip(planar, SRC_W, SRC_H, from_store, 0)
    var from_cam = List[Float32]()
    camera_frame_to_siglip(interleaved, SRC_W, SRC_H, True, from_cam, 0)

    assert_equal(len(from_store), N, "store door wrote the wrong size")
    assert_equal(len(from_cam), N, "camera door wrote the wrong size")
    var diff = 0
    var worst = Float32(0)
    var first = -1
    for i in range(N):
        if from_store[i] != from_cam[i]:
            diff += 1
            if first < 0:
                first = i
        var d = abs(from_store[i] - from_cam[i])
        if d > worst:
            worst = d
    print("  [1] store vs camera: compared", N, " differing", diff,
          " worst", worst, " first at", first)
    assert_true(
        diff == 0,
        "the training and deployment images differ for the same physical"
        " frame — a policy trained on one would be fed the other",
    )

    # ── [2] the layouts are NOT interchangeable ──────────────────────────
    # ⚠ Without this, leg [1] is satisfied by two doors that are the same
    # function, which would be true if the layout distinction were imaginary.
    # Feeding the store's planar bytes to the CAMERA door must be visibly
    # wrong.
    var as_uint = List[UInt8](unsafe_uninit_length=PX * 3)
    for i in range(PX * 3):
        as_uint[i] = UInt8(Int(planar[i]))
    var mixed = List[Float32]()
    camera_frame_to_siglip(as_uint, SRC_W, SRC_H, False, mixed, 0)
    var mdiff = 0
    var mworst = Float32(0)
    for i in range(N):
        var d = abs(mixed[i] - from_store[i])
        if d > 1.0e-6:
            mdiff += 1
        if d > mworst:
            mworst = d
    print("  [2] planar bytes through the CAMERA door: differing", mdiff,
          "of", N, " worst", mworst, " (must be a large fraction)")
    assert_true(
        mdiff > N // 2,
        "reading the store's planar frame as interleaved gives nearly the"
        " same tensor, so leg [1] cannot tell the two layouts apart",
    )

    # ── [3] the blank band is at the TOP, and it is exactly -1 ──────────
    # 640x480 into 512x512 keeps 512x384, leaving 128 blank rows. WHERE they
    # go is the thing to pin: `resize_with_pad` places the image at
    # `(pad_h, pad_w)`, so the band is at the TOP. `lerobot` also ships
    # `resize_with_pad_torch`, which CENTRES — 64 rows at each end — and
    # smolvla imports the first. Both are the right size and only one is the
    # right picture.
    #
    # ⚠ I wrote this leg with the band at the BOTTOM and it failed; the
    # implementation was right and the assumption was mine. It is gated
    # against a torch dump in `test_image_preprocess.mojo`, which is why the
    # code won that argument.
    comptime KEPT_H = (SIGLIP_INPUT * SRC_H) // SRC_W
    comptime PAD_H = SIGLIP_INPUT - KEPT_H
    var band_bad = 0
    var band_n = 0
    var bottom_at_minus1 = 0
    var bottom_n = 0
    for c in range(3):
        for y in range(SIGLIP_INPUT):
            for x in range(SIGLIP_INPUT):
                var v = from_store[c * SIGLIP_INPUT * SIGLIP_INPUT
                                   + y * SIGLIP_INPUT + x]
                if y < PAD_H:
                    band_n += 1
                    if v != Float32(-1.0):
                        band_bad += 1
                else:
                    bottom_n += 1
                    if v == Float32(-1.0):
                        bottom_at_minus1 += 1
    print("  [3] pad rows 0..", PAD_H - 1, " (", band_n, "slots): not -1:",
          band_bad, " | image rows", PAD_H, "..", SIGLIP_INPUT - 1, "(",
          bottom_n, "slots): sitting at -1:", bottom_at_minus1)
    assert_equal(band_n, 3 * PAD_H * SIGLIP_INPUT, "band size")
    assert_true(
        band_bad == 0,
        "the top band is not exactly -1 — either the pad is not at the top"
        " (a CENTRED resize_with_pad_torch would put half of it at the"
        " bottom) or the *2-1 ran BEFORE the pad and 0 stayed 0",
    )
    # ⚠ and the image half must NOT be mostly -1, or leg [3] would pass on a
    # tensor that is blank everywhere.
    assert_true(
        bottom_at_minus1 * 100 < bottom_n,
        "most of the image region is at -1 — the pad swallowed the picture",
    )
    # A CENTRED pad would put PAD_H/2 rows at the bottom. Assert they are not
    # blank, which is what distinguishes the two lerobot functions.
    var tail_blank = 0
    var tail_n = 0
    for c in range(3):
        for y in range(SIGLIP_INPUT - PAD_H // 2, SIGLIP_INPUT):
            for x in range(SIGLIP_INPUT):
                tail_n += 1
                if from_store[c * SIGLIP_INPUT * SIGLIP_INPUT
                              + y * SIGLIP_INPUT + x] == Float32(-1.0):
                    tail_blank += 1
    print("      the rows a CENTRED pad would blank (", tail_n,
          "slots): blank", tail_blank, " (must be few)")
    assert_true(
        tail_blank * 10 < tail_n,
        "the bottom rows are blank too — this is resize_with_pad_torch's"
        " CENTRED pad, not the LEFT/TOP one smolvla imports",
    )

    print()
    print("PASSED — one filter, two doors, and they agree bit for bit")
