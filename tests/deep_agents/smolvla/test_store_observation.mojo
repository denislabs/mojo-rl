# +--------------------------------------------------------------------------+ #
# | A store row and a set of camera frames must fill the SAME observation
# +--------------------------------------------------------------------------+ #
"""`fill_store_images` against `fill_camera_images`, block for block.

    pixi run mojo run -I . \\
        tests/deep_agents/smolvla/test_store_observation.mojo

`test_store_vs_camera_frame.mojo` gates ONE frame through the two SigLIP
doors. This gates the ASSEMBLY around them: N_CAM blocks laid back to back,
in an order that is part of the checkpoint.

⚠ **Camera order is the whole point.** The reference concatenates in
`config.image_features` order, so token block `k` of the prefix belongs to
camera `k` and the fine-tune learned which is which. Swapping two cameras
between the store and the deployment changes nothing observable except the
policy's behaviour — no shape changes, no error, no NaN. Leg [2] asserts the
blocks are in the right order by making each camera's content identifiable,
and leg [3] asserts that a swap is DETECTABLE, which is what says leg [2] is
testing order rather than just size.
"""

from std.math import abs
from std.testing import assert_true, assert_equal

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.vision.resize_pad import SIGLIP_INPUT
from mojo_rl.deep_agents.smolvla.observation import (
    fill_camera_images, fill_store_images,
)

comptime N_CAM = 2
comptime SRC_W = 64
comptime SRC_H = 48
"""Small: this gates the ASSEMBLY, and the pixel arithmetic is gated at the
SO-101's real 640x480 in `test_store_vs_camera_frame.mojo`."""
comptime PX = SRC_W * SRC_H
comptime PER_CAM = 3 * PX
comptime BLOCK = 3 * SIGLIP_INPUT * SIGLIP_INPUT
comptime TOTAL = N_CAM * BLOCK


def main() raises:
    print("=" * 70)
    print("store row vs camera frames -> the observation block")
    print("=" * 70)
    print("  cameras", N_CAM, " source", SRC_W, "x", SRC_H, " block", BLOCK)

    # ── the fixture: each camera has a DIFFERENT, identifiable content ────
    var row = List[Scalar[DType.uint8]](unsafe_uninit_length=N_CAM * PER_CAM)
    for cam in range(N_CAM):
        for y in range(SRC_H):
            for x in range(SRC_W):
                var i = cam * PER_CAM + y * SRC_W + x
                # camera 0 varies along x, camera 1 along y — so a swapped
                # pair is not merely different, it is diagnosable.
                if cam == 0:
                    row[i] = Scalar[DType.uint8]((x * 255) // (SRC_W - 1))
                    row[PX + i] = Scalar[DType.uint8](30)
                    row[2 * PX + i] = Scalar[DType.uint8](200)
                else:
                    row[i] = Scalar[DType.uint8](200)
                    row[PX + i] = Scalar[DType.uint8]((y * 255) // (SRC_H - 1))
                    row[2 * PX + i] = Scalar[DType.uint8](30)

    # the same two cameras as a capture layer would hand them over
    var frames = List[List[UInt8]]()
    var widths = List[Int]()
    var heights = List[Int]()
    for cam in range(N_CAM):
        var f = List[UInt8](unsafe_uninit_length=PER_CAM)
        for i in range(PX):
            f[i * 3 + 0] = UInt8(Int(row[cam * PER_CAM + 2 * PX + i]))  # B
            f[i * 3 + 1] = UInt8(Int(row[cam * PER_CAM + PX + i]))      # G
            f[i * 3 + 2] = UInt8(Int(row[cam * PER_CAM + i]))           # R
        frames.append(f^)
        widths.append(SRC_W)
        heights.append(SRC_H)

    # ── [1] the two fillers agree ────────────────────────────────────────
    var img_s = Tensor.alloc(TOTAL)
    var sc1 = List[Float32]()
    fill_store_images["cpu", N_CAM](row, SRC_W, SRC_H, img_s, sc1, None)
    var img_c = Tensor.alloc(TOTAL)
    var sc2 = List[Float32]()
    fill_camera_images["cpu", N_CAM](
        frames, widths, heights, True, img_c, sc2, None
    )
    var diff = 0
    var worst = Scalar[DT](0)
    for i in range(TOTAL):
        if img_s.data[i] != img_c.data[i]:
            diff += 1
        var d = abs(img_s.data[i] - img_c.data[i])
        if d > worst:
            worst = d
    print("  [1] store vs camera assembly: compared", TOTAL, " differing",
          diff, " worst", worst)
    assert_true(
        diff == 0,
        "the training and deployment observations differ for the same two"
        " physical frames",
    )

    # ── [2] block k really is camera k ───────────────────────────────────
    # Camera 0's red plane varies along x and camera 1's is constant. Read
    # both blocks' first live row and check which is which.
    comptime KEPT_H = (SIGLIP_INPUT * SRC_H) // SRC_W
    comptime PAD_H = SIGLIP_INPUT - KEPT_H
    var v0 = List[Scalar[DT]]()
    var v1 = List[Scalar[DT]]()
    for x in range(SIGLIP_INPUT):
        v0.append(img_s.data[0 * BLOCK + PAD_H * SIGLIP_INPUT + x])
        v1.append(img_s.data[1 * BLOCK + PAD_H * SIGLIP_INPUT + x])
    var span0 = v0[SIGLIP_INPUT - 1] - v0[0]
    var span1 = v1[SIGLIP_INPUT - 1] - v1[0]
    print("  [2] red-plane span across block 0:", span0, " block 1:", span1)
    assert_true(
        span0 > Scalar[DT](1.0),
        "block 0 does not carry camera 0's x-gradient — the blocks are not in"
        " camera order",
    )
    assert_true(
        abs(span1) < Scalar[DT](0.05),
        "block 1 is not camera 1's flat red plane — the blocks are swapped",
    )

    # ── [3] a swap IS detectable ─────────────────────────────────────────
    # ⚠ Without this, leg [2] only says the two blocks differ from each other.
    var swapped = List[Scalar[DType.uint8]](
        unsafe_uninit_length=N_CAM * PER_CAM
    )
    for i in range(PER_CAM):
        swapped[i] = row[PER_CAM + i]
        swapped[PER_CAM + i] = row[i]
    var img_x = Tensor.alloc(TOTAL)
    var sc3 = List[Float32]()
    fill_store_images["cpu", N_CAM](swapped, SRC_W, SRC_H, img_x, sc3, None)
    var sdiff = 0
    for i in range(TOTAL):
        if img_x.data[i] != img_s.data[i]:
            sdiff += 1
    print("  [3] swapping the two cameras changes", sdiff, "of", TOTAL,
          "values")
    assert_true(
        sdiff > TOTAL // 4,
        "swapping the cameras barely changes the observation, so leg [2]"
        " cannot be testing order",
    )

    # ── [4] a short row raises rather than reading past its end ──────────
    var short = List[Scalar[DType.uint8]](
        unsafe_uninit_length=N_CAM * PER_CAM - 1
    )
    for i in range(len(short)):
        short[i] = row[i]
    var raised = False
    var img_r = Tensor.alloc(TOTAL)
    var sc4 = List[Float32]()
    try:
        fill_store_images["cpu", N_CAM](short, SRC_W, SRC_H, img_r, sc4, None)
    except:
        raised = True
    print("  [4] a row one byte short raises:", raised)
    assert_true(
        raised,
        "a short store row was accepted — the camera count and the row width"
        " disagree and nothing said so",
    )

    print()
    print("PASSED — same observation from both sources, in camera order")
