"""GPU pixel preprocessing — resize + frame stack for pixel observations.

All functions are @always_inline for use inside GPU kernels.
"""

from .colors import SCREEN_W, SCREEN_H, OBS_W, OBS_H, FRAME_STACK


@always_inline
fn resize_160x210_to_84x84(
    src: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
):
    """Resize 160×210 grayscale image to 84×84 using box-filter interpolation.

    src: 160*210 = 33600 bytes (input framebuffer)
    dst: 84*84 = 7056 bytes (output resized frame)
    """
    # Scaling ratios
    # src_y = dst_y * 210/84 = dst_y * 2.5
    # src_x = dst_x * 160/84 = dst_x * 1.904...
    for dy in range(OBS_H):
        # Source y range for this output pixel
        var sy0 = dy * SCREEN_H // OBS_H
        var sy1 = (dy + 1) * SCREEN_H // OBS_H
        if sy1 == sy0:
            sy1 = sy0 + 1

        for dx in range(OBS_W):
            # Source x range for this output pixel
            var sx0 = dx * SCREEN_W // OBS_W
            var sx1 = (dx + 1) * SCREEN_W // OBS_W
            if sx1 == sx0:
                sx1 = sx0 + 1

            # Average over the source box
            var total: Int = 0
            var count: Int = 0
            for sy in range(sy0, sy1):
                for sx in range(sx0, sx1):
                    total += Int(src[sy * SCREEN_W + sx])
                    count += 1

            dst[dy * OBS_W + dx] = UInt8(total // count)


@always_inline
fn push_frame_stack[
    WS_FRAME_OFFSET: Int,  # Offset in workspace to frame stack (4*84*84 bytes)
    WS_IDX_OFFSET: Int,  # Offset in workspace to frame_idx (1 float32)
](
    resized: UnsafePointer[UInt8, MutAnyOrigin],
    workspace: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    obs_out: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
):
    """Push a resized 84×84 frame into the 4-slot ring buffer and output stacked obs.

    The frame stack stores 4 consecutive frames in chronological order.
    Output is [frame_t-3, frame_t-2, frame_t-1, frame_t] normalized to [0, 1].

    workspace layout at WS_FRAME_OFFSET:
        [0 .. 4*7056-1] = 4 frames of 84*84 float32
    workspace at WS_IDX_OFFSET:
        [0] = ring buffer write index (0..3)
    """
    comptime FRAME_SIZE = OBS_W * OBS_H  # 7056

    # Get current write index
    var frame_idx = Int(workspace[WS_IDX_OFFSET]) % FRAME_STACK
    var frame_base = WS_FRAME_OFFSET + frame_idx * FRAME_SIZE

    # Write resized frame into the ring buffer slot
    for i in range(FRAME_SIZE):
        workspace[frame_base + i] = Scalar[DType.float32](resized[i]) / 255.0

    # Advance ring index
    workspace[WS_IDX_OFFSET] = Scalar[DType.float32]((frame_idx + 1) % FRAME_STACK)

    # Output chronological stack: oldest first
    for f in range(FRAME_STACK):
        var read_idx = (frame_idx + 1 + f) % FRAME_STACK  # oldest → newest
        var read_base = WS_FRAME_OFFSET + read_idx * FRAME_SIZE
        var out_base = f * FRAME_SIZE
        for i in range(FRAME_SIZE):
            obs_out[out_base + i] = workspace[read_base + i]
