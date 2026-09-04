"""Camera frame -> the exact tensor SigLIP was fed. torch's bilinear, not PIL's.

⚠⚠ **THIS IS A SECOND RESIZE, AND IT IS NOT THE ONE NEXT DOOR.**
`preprocess.mojo` reproduces PIL BILINEAR because that is what converted the
ACT dataset. SmolVLA's `resize_with_pad` calls
`F.interpolate(mode="bilinear", align_corners=False)`, and the two are
different filters, not two spellings of one:

  * PIL scales its filter support with the reduction factor -- downscaling
    640x480 to 512x384 is a **5-tap** window (`support = 1.25`,
    `ksize = ceil(1.25)*2+1`).
  * torch without `antialias=True` is a fixed **2-tap** triangle at every
    scale; below 1x it is point-ish sampling and it aliases on purpose.

Passing one where the other is expected produces a finite, correctly shaped,
wrong image. That is the exact failure `preprocess.mojo`'s header was written
about, so it gets its own implementation and its own gate rather than a flag on
the existing one.

⚠ **`vla_utils.py` ships two functions one suffix apart.**
`resize_with_pad_torch` pads CENTRED (the openpi convention);
`resize_with_pad` pads on the **LEFT and TOP** (smolvla/xvla).
`modeling_smolvla.py` imports the second, so a 640x480 frame carries 128 blank
rows **above** the picture -- not 64 above and 64 below. Every patch token moves
if this is guessed. `tools/vla/dump_smolvla_image_reference.py` pins it, with a
portrait case so the left-pad branch is covered too.

THE REFERENCE CHAIN, in the order `prepare_images` performs it:

    uint8 HWC  ->  float32 CHW / 255      (values in [0, 1])
               ->  resize_with_pad(512, 512, pad_value=0)
               ->  * 2 - 1                (values in [-1, 1])

⚠ The padding is inserted at **0**, i.e. BEFORE the `*2-1`, so padded pixels
end at **-1** and not at 0. Normalising first and padding with 0 afterwards is
a plausible reordering that silently changes what the tower sees behind the
blank band.

Everything here is Float32 because torch's CPU kernel is: the horizontal pass
lands in a float32 intermediate before the vertical pass reads it, and matching
that ordering is what makes the gate exact rather than approximate.
"""

comptime SIGLIP_INPUT: Int = 512
"""SmolVLA's `resize_imgs_with_padding`, both axes."""


struct _Axis(Movable):
    """One axis's 2-tap plan: the low index and the two weights per output."""

    var i0: List[Int]
    var i1: List[Int]
    var w1: List[Float32]

    def __init__(out self, src: Int, dst: Int):
        # ⚠ `align_corners=False` with no explicit `scales`, so torch's
        # `compute_scales_value` is exactly input/output -- NOT
        # (input-1)/(output-1), which is the align_corners=True formula and the
        # one most hand-written bilinear code uses.
        var scale = Float32(src) / Float32(dst)
        self.i0 = List[Int](unsafe_uninit_length=dst)
        self.i1 = List[Int](unsafe_uninit_length=dst)
        self.w1 = List[Float32](unsafe_uninit_length=dst)
        for i in range(dst):
            # `area_pixel_compute_source_index`, cubic=False: the clamp at 0 is
            # part of the reference, and it is what replicates the top/left edge
            # when upscaling.
            var real = scale * (Float32(i) + 0.5) - 0.5
            if real < 0.0:
                real = 0.0
            var lo = Int(Float64(real))
            var lam = real - Float32(lo)
            self.i0[i] = lo
            self.i1[i] = min(lo + 1, src - 1)
            self.w1[i] = lam

    def __init__(out self, *, deinit move: Self):
        self.i0 = move.i0^
        self.i1 = move.i1^
        self.w1 = move.w1^


def torch_bilinear_chw(
    src: List[Float32],
    channels: Int,
    src_h: Int,
    src_w: Int,
    mut dst: List[Float32],
    dst_h: Int,
    dst_w: Int,
) raises:
    """`F.interpolate(mode="bilinear", align_corners=False, antialias=False)`.

    Separable and in that order -- horizontal into a float32 intermediate, then
    vertical over it -- because that is how torch's CPU kernel rounds.
    """
    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        raise Error("torch_bilinear_chw: every extent must be positive")
    if len(src) < channels * src_h * src_w:
        raise Error(
            "torch_bilinear_chw: source holds "
            + String(len(src))
            + " values, needs "
            + String(channels * src_h * src_w)
        )
    var n = channels * dst_h * dst_w
    if len(dst) < n:
        dst.resize(n, 0.0)

    var ax = _Axis(src_w, dst_w)
    var ay = _Axis(src_h, dst_h)
    var tmp = List[Float32](unsafe_uninit_length=src_h * dst_w)

    for c in range(channels):
        var sp = c * src_h * src_w
        for y in range(src_h):
            var row = sp + y * src_w
            var out = y * dst_w
            for x in range(dst_w):
                var w1 = ax.w1[x]
                tmp[out + x] = (
                    src[row + ax.i0[x]] * (1.0 - w1) + src[row + ax.i1[x]] * w1
                )
        var dp = c * dst_h * dst_w
        for y in range(dst_h):
            var r0 = ay.i0[y] * dst_w
            var r1 = ay.i1[y] * dst_w
            var w1 = ay.w1[y]
            var out = dp + y * dst_w
            for x in range(dst_w):
                dst[out + x] = tmp[r0 + x] * (1.0 - w1) + tmp[r1 + x] * w1


def resize_with_pad_chw(
    src: List[Float32],
    channels: Int,
    src_h: Int,
    src_w: Int,
    mut dst: List[Float32],
    dst_h: Int,
    dst_w: Int,
    pad_value: Float32 = 0.0,
) raises:
    """`lerobot.policies.common.vla_utils.resize_with_pad` -- LEFT and TOP.

    Aspect ratio preserved by the larger of the two reduction factors, so the
    resized image fits inside the target on both axes and the remainder is
    blank. `int()` TRUNCATES, as Python's does: a 700x510 frame resizes to
    512x373, not 512x374, and the pad is 139 rows rather than 138.
    """
    var n = channels * dst_h * dst_w
    if len(dst) < n:
        dst.resize(n, 0.0)

    # ⚠ The reference returns the image UNTOUCHED when it already has the
    # target size -- no resample, so no filter rounding either. Dropping the
    # early-out changes the pixels of an already-512x512 camera.
    if src_h == dst_h and src_w == dst_w:
        for i in range(n):
            dst[i] = src[i]
        return

    # Float64, because the reference's `/` is Python's float division and the
    # truncation right after it is sensitive to the last bit.
    var ratio = max(
        Float64(src_w) / Float64(dst_w), Float64(src_h) / Float64(dst_h)
    )
    var rh = Int(Float64(src_h) / ratio)
    var rw = Int(Float64(src_w) / ratio)
    var pad_h = max(0, dst_h - rh)
    var pad_w = max(0, dst_w - rw)

    var small = List[Float32]()
    torch_bilinear_chw(src, channels, src_h, src_w, small, rh, rw)

    for i in range(n):
        dst[i] = pad_value
    for c in range(channels):
        var sp = c * rh * rw
        var dp = c * dst_h * dst_w
        for y in range(rh):
            var srow = sp + y * rw
            var drow = dp + (pad_h + y) * dst_w + pad_w
            for x in range(rw):
                dst[drow + x] = small[srow + x]


def camera_frame_to_siglip(
    frame: List[UInt8],
    src_w: Int,
    src_h: Int,
    swap_rb: Bool,
    mut dst: List[Float32],
    off: Int,
    size: Int = SIGLIP_INPUT,
) raises:
    """One captured frame to the `3*512*512` block SigLIP expects, at `off`.

    `frame` is HWC uint8 as the camera delivers it; `swap_rb` for OpenCV's BGR.
    Writes `[-1, 1]` floats, blank band included.

    ⚠ The R/B swap commutes with everything downstream (the filter is per
    channel and `*2-1` is per element), so doing it here during the HWC->CHW
    transpose is exact and costs nothing -- the same argument
    `camera_frame_to_chw_rgb` makes for doing it after PIL's resize.
    """
    var px = src_w * src_h
    if len(frame) < px * 3:
        raise Error(
            "camera_frame_to_siglip: frame holds "
            + String(len(frame))
            + " bytes, needs "
            + String(px * 3)
        )
    var n = 3 * size * size
    if len(dst) < off + n:
        dst.resize(off + n, 0.0)

    var chw = List[Float32](unsafe_uninit_length=px * 3)
    var r = 2 if swap_rb else 0
    var b = 0 if swap_rb else 2
    for i in range(px):
        chw[i] = Float32(Int(frame[i * 3 + r])) / 255.0
        chw[px + i] = Float32(Int(frame[i * 3 + 1])) / 255.0
        chw[2 * px + i] = Float32(Int(frame[i * 3 + b])) / 255.0

    var padded = List[Float32]()
    resize_with_pad_chw(chw, 3, src_h, src_w, padded, size, size, 0.0)

    # ⚠ AFTER the pad, so the blank band lands at -1. See the header.
    for i in range(n):
        dst[off + i] = padded[i] * 2.0 - 1.0


def store_frame_to_siglip(
    frame: List[Scalar[DType.uint8]],
    src_w: Int,
    src_h: Int,
    mut dst: List[Float32],
    off: Int,
    size: Int = SIGLIP_INPUT,
) raises:
    """One STORED frame to the `3*512*512` block SigLIP expects, at `off`.

    ⚠ **`frame` is CHW, and `camera_frame_to_siglip`'s is HWC.** That is the
    whole reason this exists as a separate entry point rather than a flag: the
    store writes `[3, H, W]` planes (`import_lerobot_v3`) and a camera hands
    over interleaved `[H, W, 3]`. Feeding one to the other does not fail, does
    not raise, and does not even look obviously wrong — a plane read as
    interleaved is a recognisable image with its colours and geometry
    scrambled in a way that survives a thumbnail.
    
    There is no `swap_rb`. The store is RGB by construction: the importer
    decodes with ffmpeg to rgb24. A camera needs the flag because OpenCV
    hands back BGR.

    Everything after the layout — the 2-tap bilinear, the LEFT/TOP pad, the
    `*2-1` applied AFTER the pad so the blank band lands at -1 — is
    `resize_with_pad_chw` and is shared with the camera path. One filter, two
    front doors.
    """
    var px = src_w * src_h
    if len(frame) < px * 3:
        raise Error(
            "store_frame_to_siglip: frame holds "
            + String(len(frame))
            + " bytes, needs "
            + String(px * 3)
        )
    var n = 3 * size * size
    if len(dst) < off + n:
        dst.resize(off + n, 0.0)

    # Already planar: only the uint8 -> [0,1] cast is needed.
    var chw = List[Float32](unsafe_uninit_length=px * 3)
    for i in range(px * 3):
        chw[i] = Float32(Int(frame[i])) / 255.0

    var padded = List[Float32]()
    resize_with_pad_chw(chw, 3, src_h, src_w, padded, size, size, 0.0)

    for i in range(n):
        dst[off + i] = padded[i] * 2.0 - 1.0
