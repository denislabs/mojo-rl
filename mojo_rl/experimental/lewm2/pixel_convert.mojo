"""uint8 HWC → fp32 CHW/÷255 pixel conversion (LeWM real-data path).

Environments hand frames as `uint8` in HWC (height, width, channel)
layout; the encoder's `PatchEmbed` expects `fp32` CHW normalised to
[0,1]. This is the one bridge between the two. Stored offline data is
4× smaller as uint8, so the real NVIDIA driver uploads uint8 and converts
on-device right before the encoder — `u8_hwc_to_chw_norm[target=...]`
covers both CPU (reference / offline ingest) and GPU (the batched path).

  chw[(c·H + h)·W + w] = hwc[(h·W + w)·C + c] / 255.0           per sample
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ...nn2.constants import DT, TPB


def _u8_hwc_to_chw_kernel[C: Int, H: Int, W: Int, BATCH: Int](
    src: LayoutTensor[DType.uint8, Layout.row_major(BATCH * H * W * C), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(BATCH * C * H * W), MutAnyOrigin],
):
    comptime HWC = H * W * C
    var idx = Int(global_idx.x)
    if idx >= BATCH * HWC:
        return
    # decode flat dst index → (b, c, h, w)
    var b = idx // HWC
    var rem = idx % HWC
    var c = rem // (H * W)
    var hw = rem % (H * W)
    var h = hw // W
    var w = hw % W
    var src_idx = b * HWC + (h * W + w) * C + c
    dst[idx] = src[src_idx].cast[DT]() / Scalar[DT](255.0)


def u8_hwc_to_chw_norm[
    target: StaticString,
    C: Int, H: Int, W: Int, BATCH: Int,
](
    src: UnsafePointer[Scalar[DType.uint8], MutAnyOrigin],
    dst: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Convert `src` (BATCH·H·W·C uint8, HWC) → `dst` (BATCH·C·H·W fp32,
    CHW, ÷255). `ctx` required for target='gpu'."""
    comptime assert target == "cpu" or target == "gpu", (
        "u8_hwc_to_chw_norm: target must be 'cpu' or 'gpu'"
    )
    comptime HWC = H * W * C
    comptime if target == "cpu":
        for b in range(BATCH):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        var d = b * HWC + (c * H + h) * W + w
                        var sidx = b * HWC + (h * W + w) * C + c
                        dst[d] = src[sidx].cast[DT]() / Scalar[DT](255.0)
    else:
        if not ctx:
            raise Error("u8_hwc_to_chw_norm[target='gpu']: ctx required")
        var c = ctx.value()
        var src_lt = LayoutTensor[
            DType.uint8, Layout.row_major(BATCH * HWC), MutAnyOrigin
        ](src)
        var dst_lt = LayoutTensor[
            DT, Layout.row_major(BATCH * HWC), MutAnyOrigin
        ](dst)
        comptime n = BATCH * HWC
        comptime n_blocks = (n + TPB - 1) // TPB
        comptime kernel = _u8_hwc_to_chw_kernel[C, H, W, BATCH]
        c.enqueue_function[kernel](
            src_lt, dst_lt, grid_dim=n_blocks, block_dim=TPB,
        )
