"""Fused ResBlock for Conv2D: y = ReLU(Conv2(ReLU(Conv1(x))) + x).

Two conv layers with a skip connection, all in one Model.
Fuses the skip-add and final ReLU into a single GPU kernel.

Requires in_channels == out_channels and same padding (spatial dims preserved).
"""

from ..constants import dtype, TPB
from ..model.model import Model, PerfTimerPtr, NULL_PERF
from ..initializer import Initializer
from ..autodiff.fused import FusedConv2DActivation, ReLUActivation
from ..autodiff import AutoFused, Conv2D
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.math import ceildiv


# GPU kernel: output[i] = relu(output[i] + skip[i])
def _add_relu_kernel[
    SIZE: Int,
    dtype: DType where dtype.is_floating_point(),
](
    output: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    skip: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return
    var val = rebind[Scalar[dtype]](output[idx]) + rebind[Scalar[dtype]](skip[idx])
    output[idx] = val if val > Scalar[dtype](0.0) else Scalar[dtype](0.0)


# GPU kernel: cache pre-relu, apply relu
def _fwd_cache_add_relu_kernel[
    SIZE: Int,
    dtype: DType where dtype.is_floating_point(),
](
    output: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    skip: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    pre_relu_cache: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return
    var val = rebind[Scalar[dtype]](output[idx]) + rebind[Scalar[dtype]](skip[idx])
    pre_relu_cache[idx] = val
    output[idx] = val if val > Scalar[dtype](0.0) else Scalar[dtype](0.0)


# GPU kernel: backward through add+relu
def _add_relu_backward_kernel[
    SIZE: Int,
    dtype: DType where dtype.is_floating_point(),
](
    grad_out: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    pre_relu_cache: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    grad_skip: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return
    var pre = rebind[Scalar[dtype]](pre_relu_cache[idx])
    var go = rebind[Scalar[dtype]](grad_out[idx])
    var masked = go if pre > Scalar[dtype](0.0) else Scalar[dtype](0.0)
    grad_out[idx] = masked
    grad_skip[idx] = rebind[Scalar[dtype]](grad_skip[idx]) + masked


def _add_kernel[
    SIZE: Int,
    dtype: DType where dtype.is_floating_point(),
](
    a: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return
    a[idx] = rebind[Scalar[dtype]](a[idx]) + rebind[Scalar[dtype]](b[idx])


@fieldwise_init
struct ResBlockConv2D[
    channels: Int,
    kernel_size: Int,
    padding: Int,
    h: Int,
    w: Int,
](Model):
    """Fused ResBlock: y = ReLU(Conv2(ReLU(Conv1(x))) + x)."""

    comptime Conv1 = AutoFused[FusedConv2DActivation[
        Self.channels, Self.channels, Self.kernel_size, 1, Self.padding,
        Self.h, Self.w, ReLUActivation,
    ]]
    comptime Conv2 = AutoFused[Conv2D[
        Self.channels, Self.channels, Self.kernel_size, 1, Self.padding,
        Self.h, Self.w,
    ]]

    comptime DIM: Int = Self.channels * Self.h * Self.w
    comptime IN_DIM: Int = Self.Conv1.IN_DIM
    comptime OUT_DIM: Int = Self.Conv2.OUT_DIM

    comptime CONV1_PS: Int = Self.Conv1.PARAM_SIZE
    comptime CONV2_PS: Int = Self.Conv2.PARAM_SIZE
    comptime PARAM_SIZE: Int = Self.CONV1_PS + Self.CONV2_PS

    comptime CONV1_CS: Int = Self.Conv1.CACHE_SIZE
    comptime CONV2_CS: Int = Self.Conv2.CACHE_SIZE
    comptime CACHE_SIZE: Int = Self.CONV1_CS + Self.CONV2_CS + Self.DIM

    comptime CONV1_WS: Int = Self.Conv1.WORKSPACE_SIZE_PER_SAMPLE
    comptime CONV2_WS: Int = Self.Conv2.WORKSPACE_SIZE_PER_SAMPLE
    comptime MAX_CONV_WS: Int = Self.CONV1_WS if Self.CONV1_WS > Self.CONV2_WS else Self.CONV2_WS
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self.MAX_CONV_WS + Self.DIM

    # ── Initialization ─────────────────────────────────────────────

    @staticmethod
    def initialize_params[
        INIT: Initializer,
    ](
        mut params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
    ):
        var p1 = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](params.ptr)
        var p2 = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](params.ptr + Self.CONV1_PS)
        Self.Conv1.initialize_params[INIT](p1)
        Self.Conv2.initialize_params[INIT](p2)

    # ── CPU Forward (with cache) ───────────────────────────────────

    @staticmethod
    def forward[BATCH: Int](
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        mut output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        mut cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
    ):
        from std.memory import alloc
        var p1 = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](params.ptr)
        var p2 = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](params.ptr + Self.CONV1_PS)
        var c1 = LayoutTensor[dtype, Layout.row_major(BATCH, Self.CONV1_CS), MutAnyOrigin](cache.ptr)
        var c2 = LayoutTensor[dtype, Layout.row_major(BATCH, Self.CONV2_CS), MutAnyOrigin](cache.ptr + BATCH * Self.CONV1_CS)

        var inter = alloc[Scalar[dtype]](BATCH * Self.DIM)
        var inter_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.OUT_DIM), MutAnyOrigin](inter)
        var in_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.IN_DIM), MutAnyOrigin]](input)
        Self.Conv1.forward[BATCH](in_rb, inter_t, p1, c1)

        var inter_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.IN_DIM), MutAnyOrigin]](inter_t)
        var out_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.OUT_DIM), MutAnyOrigin]](output)
        Self.Conv2.forward[BATCH](inter_rb, out_rb, p2, c2)

        # Skip add + ReLU + cache pre-relu
        var pre_off = BATCH * (Self.CONV1_CS + Self.CONV2_CS)
        for i in range(BATCH * Self.DIM):
            var val = output.ptr[i] + input.ptr[i]
            (cache.ptr + pre_off)[i] = val
            output.ptr[i] = val if Float64(val) > 0.0 else Scalar[dtype](0.0)

        inter.free()

    # ── CPU Forward (inference, no cache) ──────────────────────────

    @staticmethod
    def forward[BATCH: Int](
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        mut output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
    ):
        from std.memory import alloc
        var p1 = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](params.ptr)
        var p2 = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](params.ptr + Self.CONV1_PS)

        var inter = alloc[Scalar[dtype]](BATCH * Self.DIM)
        var inter_t = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.OUT_DIM), MutAnyOrigin](inter)
        var in_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.IN_DIM), MutAnyOrigin]](input)
        Self.Conv1.forward[BATCH](in_rb, inter_t, p1)

        var inter_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.IN_DIM), MutAnyOrigin]](inter_t)
        var out_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.OUT_DIM), MutAnyOrigin]](output)
        Self.Conv2.forward[BATCH](inter_rb, out_rb, p2)

        for i in range(BATCH * Self.DIM):
            var val = output.ptr[i] + input.ptr[i]
            output.ptr[i] = val if Float64(val) > 0.0 else Scalar[dtype](0.0)

        inter.free()

    # ── CPU Backward ────────────────────────────────────────────────

    @staticmethod
    def backward[BATCH: Int](
        grad_output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        mut grad_input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
        mut grads: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
    ):
        pass  # CPU backward not implemented — use GPU backward

    # ── GPU Forward (with cache) ─────────────────────────────────

    @staticmethod
    def forward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        mut output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        mut cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        var p1 = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](params.ptr)
        var p2 = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](params.ptr + Self.CONV1_PS)
        var c1_v = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.CACHE_SIZE), MutAnyOrigin](cache.ptr)
        var c2_v = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.CACHE_SIZE), MutAnyOrigin](cache.ptr + BATCH * Self.CONV1_CS)

        # Workspace layout: [conv_workspace (MAX_CONV_WS per sample) | inter (DIM per sample)]
        var conv_ws_size = BATCH * Self.MAX_CONV_WS
        var conv_ws = DeviceBuffer[dtype](ctx, workspace.unsafe_ptr(), conv_ws_size if conv_ws_size > 0 else 1, owning=False)

        var inter_ptr = workspace.unsafe_ptr() + BATCH * Self.MAX_CONV_WS
        var inter_out = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.OUT_DIM), MutAnyOrigin](inter_ptr)
        var in_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.IN_DIM), MutAnyOrigin]](input)
        Self.Conv1.forward_gpu[BATCH](ctx, inter_out, in_rb, p1, c1_v, conv_ws)

        var inter_in = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.IN_DIM), MutAnyOrigin](inter_ptr)
        var out_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.OUT_DIM), MutAnyOrigin]](output)
        Self.Conv2.forward_gpu[BATCH](ctx, out_rb, inter_in, p2, c2_v, conv_ws)

        comptime TOTAL = BATCH * Self.DIM
        comptime BLOCKS = ceildiv(TOTAL, TPB)
        var out_flat = LayoutTensor[dtype, Layout.row_major(TOTAL), MutAnyOrigin](output.ptr)
        var skip_flat = LayoutTensor[dtype, Layout.row_major(TOTAL), MutAnyOrigin](input.ptr)
        var pre_flat = LayoutTensor[dtype, Layout.row_major(TOTAL), MutAnyOrigin](
            cache.ptr + BATCH * (Self.CONV1_CS + Self.CONV2_CS)
        )
        comptime fwd_kernel = _fwd_cache_add_relu_kernel[TOTAL, dtype]
        ctx.enqueue_function[fwd_kernel, fwd_kernel](out_flat, skip_flat, pre_flat, grid_dim=(BLOCKS,), block_dim=(TPB,))

    # ── GPU Forward (inference, no cache) ────────────────────────

    @staticmethod
    def forward_gpu_no_cache[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        mut output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        var p1 = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](params.ptr)
        var p2 = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](params.ptr + Self.CONV1_PS)

        # Workspace layout: [conv_workspace (MAX_CONV_WS per sample) | inter (DIM per sample)]
        var conv_ws_size = BATCH * Self.MAX_CONV_WS
        var conv_ws = DeviceBuffer[dtype](ctx, workspace.unsafe_ptr(), conv_ws_size if conv_ws_size > 0 else 1, owning=False)

        var inter_ptr = workspace.unsafe_ptr() + BATCH * Self.MAX_CONV_WS
        var inter_out = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.OUT_DIM), MutAnyOrigin](inter_ptr)
        var in_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.IN_DIM), MutAnyOrigin]](input)
        Self.Conv1.forward_gpu_no_cache[BATCH](ctx, inter_out, in_rb, p1, conv_ws)

        var inter_in = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.IN_DIM), MutAnyOrigin](inter_ptr)
        var out_rb = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.OUT_DIM), MutAnyOrigin]](output)
        Self.Conv2.forward_gpu_no_cache[BATCH](ctx, out_rb, inter_in, p2, conv_ws)

        comptime TOTAL = BATCH * Self.DIM
        comptime BLOCKS = ceildiv(TOTAL, TPB)
        var out_flat = LayoutTensor[dtype, Layout.row_major(TOTAL), MutAnyOrigin](output.ptr)
        var skip_flat = LayoutTensor[dtype, Layout.row_major(TOTAL), MutAnyOrigin](input.ptr)
        comptime kernel = _add_relu_kernel[TOTAL, dtype]
        ctx.enqueue_function[kernel, kernel](out_flat, skip_flat, grid_dim=(BLOCKS,), block_dim=(TPB,))

    # ── GPU Forward (no cache, on stream) ────────────────────────

    @staticmethod
    def forward_gpu_no_cache_on_stream[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        stream: DeviceStream,
        mut output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        workspace: DeviceBuffer[dtype],
    ) raises:
        # Default: delegate to forward_gpu_no_cache (default stream)
        Self.forward_gpu_no_cache[BATCH](ctx, output, input, params, workspace)

    # ── GPU Backward ───────────────────────────────────────────────

    @staticmethod
    def backward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        mut grad_input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        grad_output: LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin],
        mut grads: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        var p1 = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](params.ptr)
        var p2 = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](params.ptr + Self.CONV1_PS)

        # Workspace layout: [conv_workspace (MAX_CONV_WS per sample) | inter (DIM per sample)]
        var conv_ws_size = BATCH * Self.MAX_CONV_WS
        var conv_ws = DeviceBuffer[dtype](ctx, workspace.unsafe_ptr(), conv_ws_size if conv_ws_size > 0 else 1, owning=False)

        comptime TOTAL = BATCH * Self.DIM
        comptime BLOCKS = ceildiv(TOTAL, TPB)

        # 1. Backward through add+relu: mask grad_output, accumulate skip grad
        var go_flat = LayoutTensor[dtype, Layout.row_major(TOTAL), MutAnyOrigin](grad_output.ptr)
        var pre_flat = LayoutTensor[dtype, Layout.row_major(TOTAL), MutAnyOrigin](
            cache.ptr + BATCH * (Self.CONV1_CS + Self.CONV2_CS)
        )
        var gi_flat = LayoutTensor[dtype, Layout.row_major(TOTAL), MutAnyOrigin](grad_input.ptr)
        comptime bwd_kernel = _add_relu_backward_kernel[TOTAL, dtype]
        ctx.enqueue_function[bwd_kernel, bwd_kernel](go_flat, pre_flat, gi_flat, grid_dim=(BLOCKS,), block_dim=(TPB,))

        # 2. Conv2 backward: grad_output → grad_inter (in workspace inter region)
        var inter_ptr = workspace.unsafe_ptr() + BATCH * Self.MAX_CONV_WS
        var grad_inter = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.IN_DIM), MutAnyOrigin](inter_ptr)
        var go_c2 = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.OUT_DIM), MutAnyOrigin](grad_output.ptr)
        var c2_v = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv2.CACHE_SIZE), MutAnyOrigin](cache.ptr + BATCH * Self.CONV1_CS)
        var g2_v = LayoutTensor[dtype, Layout.row_major(Self.CONV2_PS), MutAnyOrigin](grads.ptr + Self.CONV1_PS)
        Self.Conv2.backward_gpu[BATCH](ctx, grad_inter, go_c2, p2, c2_v, g2_v, conv_ws)

        # 3. Conv1 backward: grad_inter → temp (reuse grad_output buffer)
        var go_c1 = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.OUT_DIM), MutAnyOrigin](inter_ptr)
        var temp_gi = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.IN_DIM), MutAnyOrigin](grad_output.ptr)
        var c1_v = LayoutTensor[dtype, Layout.row_major(BATCH, Self.Conv1.CACHE_SIZE), MutAnyOrigin](cache.ptr)
        var g1_v = LayoutTensor[dtype, Layout.row_major(Self.CONV1_PS), MutAnyOrigin](grads.ptr)
        Self.Conv1.backward_gpu[BATCH](ctx, temp_gi, go_c1, p1, c1_v, g1_v, conv_ws)

        # 4. Add conv1's grad_input to skip grad
        comptime add_k = _add_kernel[TOTAL, dtype]
        ctx.enqueue_function[add_k, add_k](
            gi_flat,
            LayoutTensor[dtype, Layout.row_major(TOTAL), MutAnyOrigin](grad_output.ptr),
            grid_dim=(BLOCKS,), block_dim=(TPB,),
        )
