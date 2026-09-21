"""Captured frames -> the `[N_CAM, 3*512*512]` block the prefix driver reads.

`SmolVLAPrefixEmbed.run` takes `images` as N_CAM camera-sized blocks laid back
to back, one batch entry each, because that is how SmolVLA runs them: the tower
sees one image at a time and the connector's 64 tokens per camera are
concatenated afterwards. This assembles that block from whatever the capture
layer hands over, and does nothing else -- the pixel arithmetic all lives in
`vision/resize_pad.mojo`, gated against torch.

⚠ **A MISSING CAMERA IS AN ERROR HERE, NOT A BLACK IMAGE.**
The reference substitutes an all -1 image for absent cameras, but only for the
first `config.empty_cameras` of them, and SmolVLA's default is **0** -- so with
a two-camera policy the reference raises. Filling a dropped frame with black
instead would be finite, correctly shaped, and would quietly hand the policy a
scene it never saw during training. `CameraReader.take_latest` returning 0 must
propagate, which is why the frames arrive here already fetched rather than this
struct owning the readers.

⚠ **Camera ORDER is part of the checkpoint.** The reference iterates
`config.image_features` and concatenates in that order, so token block `k` of
the prefix belongs to camera `k` and the fine-tune learned which is which.
Swapping two cameras between recording and deployment changes nothing
observable except the policy's behaviour.
"""

from std.memory import Pointer, unsafe_memcpy
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.vision.resize_pad import (
    camera_frame_to_siglip, store_frame_to_siglip, SIGLIP_INPUT,
)


def fill_camera_images[
    target: StaticString, N_CAM: Int, SIZE: Int = SIGLIP_INPUT
](
    ref frames: List[List[UInt8]],
    ref widths: List[Int],
    ref heights: List[Int],
    swap_rb: Bool,
    mut images: Tensor,
    mut scratch: List[Float32],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Assemble `images` for `SmolVLAPrefixEmbed.run`, one block per camera.

    `frames` are HWC uint8 as captured; `swap_rb` for OpenCV's BGR. Cameras may
    differ in resolution -- each is resized and padded to `SIZE` on its own, so
    a wide and a square camera end up the same shape by different pad bands.

    `scratch` is carried by the caller so a 50 Hz loop does not allocate three
    megabytes per tick.
    """
    comptime assert N_CAM >= 1, "fill_camera_images: need a camera"
    comptime BLOCK: Int = 3 * SIZE * SIZE
    comptime TOTAL: Int = N_CAM * BLOCK

    if len(frames) != N_CAM or len(widths) != N_CAM or len(heights) != N_CAM:
        raise Error(
            "fill_camera_images: expected "
            + String(N_CAM)
            + " cameras, got "
            + String(len(frames))
            + " frames / "
            + String(len(widths))
            + " widths / "
            + String(len(heights))
            + " heights"
        )
    if len(scratch) < TOTAL:
        scratch.resize(TOTAL, 0.0)

    for cam in range(N_CAM):
        camera_frame_to_siglip(
            frames[cam],
            widths[cam],
            heights[cam],
            swap_rb,
            scratch,
            cam * BLOCK,
            SIZE,
        )

    comptime if target == "cpu":
        images.ensure(TOTAL)
        for i in range(TOTAL):
            images.data[i] = Scalar[DT](scratch[i])
    else:
        # ⚠ `ensure`, NOT `ensure_host`. `ensure_host` allocates the PINNED
        # STAGING buffer that `upload` copies THROUGH; it leaves `data` empty,
        # so writing `data[i]` afterwards indexes an empty list. `upload` calls
        # `ensure_host` itself — the host-visible slab a caller fills is
        # `data`, and `ensure` is what sizes it.
        images.ensure(TOTAL)
        for i in range(TOTAL):
            images.data[i] = Scalar[DT](scratch[i])
        # ⚠ `upload_resident`, not `upload`. `upload` RECREATES the device
        # buffer and synchronises TWICE on every call — by design, it is the
        # resize path — and this runs once per control tick on a 6.3 MB
        # tensor. Measured at 127 ms per tick before this change.
        images.upload_resident(ctx.value())


def fill_siglip_frames[
    target: StaticString, N_CAM: Int, SIZE: Int = SIGLIP_INPUT
](
    ref frames: List[List[UInt8]],
    mut images: Tensor,
    ctx: Optional[DeviceContext] = None,
) raises:
    """`fill_camera_images` for frames the CAMERA THREAD already preprocessed
    (`CameraReader(..., siglip=SIZE)`): each `frames[cam]` is the bytes of
    the `3*SIZE*SIZE` float block, so this is N_CAM copies and an upload —
    no resize on the control thread, which is the whole point.
    """
    comptime assert N_CAM >= 1, "fill_siglip_frames: need a camera"
    comptime BLOCK: Int = 3 * SIZE * SIZE
    comptime TOTAL: Int = N_CAM * BLOCK
    _check_siglip_frames[N_CAM, SIZE](frames)
    images.ensure(TOTAL)
    for cam in range(N_CAM):
        var src = frames[cam].unsafe_ptr().unsafe_bitcast[Float32]()
        for i in range(BLOCK):
            images.data[cam * BLOCK + i] = Scalar[DT](src[unsafe_offset=i])
    comptime if target != "cpu":
        images.upload_resident(ctx.value())


def siglip_frames_into_slot[
    N_CAM: Int, SIZE: Int = SIGLIP_INPUT
](
    ref frames: List[List[UInt8]],
    dst: Pointer[UInt8, MutUntrackedOrigin],
    byte_off: Int,
) raises:
    """The same frames straight into a request slot (the `--threaded` deploy):
    N_CAM memcpys at `dst + byte_off`, back to back in camera order, which is
    exactly the layout `fill_siglip_frames` gives `images`."""
    comptime BLOCK_BYTES: Int = 3 * SIZE * SIZE * 4
    _check_siglip_frames[N_CAM, SIZE](frames)
    for cam in range(N_CAM):
        unsafe_memcpy(
            dest=dst.unsafe_offset(byte_off + cam * BLOCK_BYTES),
            src=rebind[Pointer[UInt8, MutUntrackedOrigin]](
                frames[cam].unsafe_ptr().as_unsafe_any_origin()
            ),
            count=BLOCK_BYTES,
        )


def _check_siglip_frames[N_CAM: Int, SIZE: Int](
    ref frames: List[List[UInt8]]
) raises:
    """A frame of the wrong size here is a camera opened WITHOUT `siglip=`,
    i.e. raw pixels about to be read as floats — refuse, do not truncate."""
    comptime BLOCK_BYTES: Int = 3 * SIZE * SIZE * 4
    if len(frames) != N_CAM:
        raise Error(
            "siglip frames: expected " + String(N_CAM) + " cameras, got "
            + String(len(frames))
        )
    for cam in range(N_CAM):
        if len(frames[cam]) != BLOCK_BYTES:
            raise Error(
                "siglip frames: camera " + String(cam) + " delivered "
                + String(len(frames[cam])) + " bytes, not the "
                + String(BLOCK_BYTES) + " of a " + String(SIZE) + "x"
                + String(SIZE) + " float block — was the reader opened with"
                " siglip=" + String(SIZE) + "?"
            )


def fill_store_images[
    target: StaticString, N_CAM: Int, SIZE: Int = SIGLIP_INPUT
](
    ref row: List[Scalar[DType.uint8]],
    src_w: Int,
    src_h: Int,
    mut images: Tensor,
    mut scratch: List[Float32],
    ctx: Optional[DeviceContext] = None,
) raises:
    """The TRAINING counterpart of `fill_camera_images`, from one store row.

    `row` is the store's `images` column for a single frame: uint8
    `[N_CAM, 3, src_h, src_w]`, PLANAR and RGB. Every camera in a store shares
    one resolution — the column has a single shape — which is the one thing
    that differs from the camera path, where each reader brings its own.

    ⚠ **The same camera ORDER rule applies, and here it is the store's column
    order.** `import_lerobot_v3` writes cameras in `info.cameras` order and
    that is the order the fine-tune learns; a deployment that opens its
    readers in a different order gets a policy that looks trained and behaves
    wrongly, with nothing to compare against at runtime.

    ⚠ There is no `swap_rb`. See `store_frame_to_siglip` — the store is RGB by
    construction.
    """
    comptime assert N_CAM >= 1, "fill_store_images: need a camera"
    comptime BLOCK: Int = 3 * SIZE * SIZE
    comptime TOTAL: Int = N_CAM * BLOCK

    var per_cam = 3 * src_w * src_h
    if len(row) != N_CAM * per_cam:
        raise Error(
            "fill_store_images: row holds "
            + String(len(row))
            + " bytes, expected "
            + String(N_CAM)
            + " cameras x 3 x "
            + String(src_h)
            + " x "
            + String(src_w)
            + " = "
            + String(N_CAM * per_cam)
        )
    if len(scratch) < TOTAL:
        scratch.resize(TOTAL, 0.0)

    var one = List[Scalar[DType.uint8]](unsafe_uninit_length=per_cam)
    for cam in range(N_CAM):
        var base = cam * per_cam
        for i in range(per_cam):
            one[i] = row[base + i]
        store_frame_to_siglip(one, src_w, src_h, scratch, cam * BLOCK, SIZE)

    comptime if target == "cpu":
        images.ensure(TOTAL)
        for i in range(TOTAL):
            images.data[i] = Scalar[DT](scratch[i])
    else:
        images.ensure(TOTAL)
        for i in range(TOTAL):
            images.data[i] = Scalar[DT](scratch[i])
        images.upload_resident(ctx.value())
