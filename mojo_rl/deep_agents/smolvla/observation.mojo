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

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.vision.resize_pad import camera_frame_to_siglip, SIGLIP_INPUT


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
        images.ensure_host(ctx.value(), TOTAL)
        for i in range(TOTAL):
            images.data[i] = Scalar[DT](scratch[i])
        images.upload(ctx.value())
