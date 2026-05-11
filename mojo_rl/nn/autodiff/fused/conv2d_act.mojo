"""Fused Conv2D + Activation op.

FusedConv2DActivation[ic, oc, k, s, p, h, w, ACT] fuses a Conv2D with an
activation into a single forward/backward pass, eliminating one full
read+write of the output tensor per layer.

Forward:  y = act(Conv2D(x))  — activation applied in-place after bias add
Backward: grad through activation first, then Conv2D backward

Cache layout: [im2col (col_size * spatial_out) | act_cache (OUT_DIM)]
"""

from ...constants import (
    dtype,
    TPB,
    MMA_M,
    MMA_N,
    MMA_K,
    MMA_BLOCK_M,
    MMA_BLOCK_N,
    MMA_WARPS_M,
    MMA_WARPS_N,
    MMA_NUM_WARPS,
    MMA_BLOCK_THREADS,
)
from ...autodiff.op import DiffOp, FusedOp, OpID
from .activation import Activation
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.runtime.asyncrt import DeviceContextPtr
from std.gpu.memory import AddressSpace
from std.gpu.primitives import block, lane_id
from std.sys import is_nvidia_gpu, has_nvidia_gpu_accelerator
from std.gpu.compute.mma import mma
from linalg.matmul import matmul as max_matmul
from layout.tile_tensor import lt_to_tt


struct FusedConv2DActivation[
    in_channels: Int,
    out_channels: Int,
    kernel_size: Int,
    stride: Int,
    padding: Int,
    in_h: Int,
    in_w: Int,
    ACT: Activation,
    USE_MAX_KERNELS: Bool = True,
](FusedOp):
    """Fused y = act(Conv2D(x)) in a single operation.

    Params: W (out_channels, col_size) + bias (out_channels) — same as Conv2D
    Cache: im2col buffer + activation cache (pre-act or output per ACT.cache)
    """

    comptime out_h: Int = (
        Self.in_h + 2 * Self.padding - Self.kernel_size
    ) // Self.stride + 1
    comptime out_w: Int = (
        Self.in_w + 2 * Self.padding - Self.kernel_size
    ) // Self.stride + 1
    comptime col_size: Int = Self.in_channels * Self.kernel_size * Self.kernel_size
    comptime spatial_out: Int = Self.out_h * Self.out_w

    comptime OP_ID: Int = Self.ACT.FUSED_CONV_OP_ID
    comptime IN_DIM: Int = Self.in_channels * Self.in_h * Self.in_w
    comptime OUT_DIM: Int = Self.out_channels * Self.out_h * Self.out_w
    comptime PARAM_SIZE: Int = Self.out_channels * Self.col_size + Self.out_channels
    # Cache: im2col (col_size * spatial_out) + activation cache (OUT_DIM)
    comptime CONV_CACHE: Int = Self.col_size * Self.spatial_out
    comptime CACHE_SIZE: Int = Self.CONV_CACHE + Self.OUT_DIM
    comptime FUSED_COUNT: Int = 2
    # Forward: col_flat (CONV_CACHE) + out_temp (OUT_DIM)
    # Backward dW: col_flat (CONV_CACHE)
    # Custom MMA handles W transpose in-kernel, no extra w_t workspace needed
    comptime OP_WORKSPACE_PER_SAMPLE: Int = Self.CONV_CACHE + Self.OUT_DIM

    def __init__(out self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    # =========================================================================
    # CPU eval
    # =========================================================================

    @staticmethod
    def eval[
        BATCH: Int, dtype: DType = DType.float32
    ](
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        for b in range(BATCH):
            # im2col into cache[b, 0..CONV_CACHE-1]
            for oh in range(Self.out_h):
                for ow in range(Self.out_w):
                    var s = oh * Self.out_w + ow
                    for c in range(Self.in_channels):
                        for kh in range(Self.kernel_size):
                            for kw in range(Self.kernel_size):
                                var ih = oh * Self.stride - Self.padding + kh
                                var iw = ow * Self.stride - Self.padding + kw
                                var c_k = (
                                    c * Self.kernel_size * Self.kernel_size
                                    + kh * Self.kernel_size
                                    + kw
                                )
                                var col_idx = s * Self.col_size + c_k
                                if (
                                    ih >= 0
                                    and ih < Self.in_h
                                    and iw >= 0
                                    and iw < Self.in_w
                                ):
                                    var in_idx = (
                                        c * Self.in_h * Self.in_w
                                        + ih * Self.in_w
                                        + iw
                                    )
                                    cache[b, col_idx] = input[b, in_idx]
                                else:
                                    cache[b, col_idx] = 0

            # W @ col + bias, then activation
            for oc in range(Self.out_channels):
                for s in range(Self.spatial_out):
                    var acc: Scalar[dtype] = 0
                    for k in range(Self.col_size):
                        var w_val = rebind[Scalar[dtype]](
                            params[oc * Self.col_size + k]
                        )
                        var c_val = rebind[Scalar[dtype]](
                            cache[b, s * Self.col_size + k]
                        )
                        acc += w_val * c_val
                    # Add bias
                    acc += rebind[Scalar[dtype]](
                        params[Self.out_channels * Self.col_size + oc]
                    )
                    # Fused activation
                    var pre_act = acc
                    var act_out = Self.ACT.forward(pre_act)
                    var out_idx = oc * Self.spatial_out + s
                    output[b, out_idx] = act_out
                    # Cache activation state for backward
                    cache[b, Self.CONV_CACHE + out_idx] = Self.ACT.cache(
                        pre_act, act_out
                    )

    # =========================================================================
    # CPU vjp
    # =========================================================================

    @staticmethod
    def vjp[
        BATCH: Int, dtype: DType = DType.float32
    ](
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        comptime W_SIZE = Self.out_channels * Self.col_size

        for b in range(BATCH):
            # 1. dW += masked_grad @ col.T
            for oc in range(Self.out_channels):
                for k in range(Self.col_size):
                    var acc: Scalar[dtype] = 0
                    for s in range(Self.spatial_out):
                        var out_idx = oc * Self.spatial_out + s
                        var cache_val = rebind[Scalar[dtype]](
                            cache[b, Self.CONV_CACHE + out_idx]
                        )
                        var go_val = rebind[Scalar[dtype]](
                            grad_output[b, out_idx]
                        )
                        var masked_go = Self.ACT.backward(cache_val, go_val)
                        var col_val = rebind[Scalar[dtype]](
                            cache[b, s * Self.col_size + k]
                        )
                        acc += masked_go * col_val
                    var cur = rebind[Scalar[dtype]](
                        grad_params[oc * Self.col_size + k]
                    )
                    grad_params[oc * Self.col_size + k] = cur + acc

            # 2. db += sum(masked_grad, over spatial)
            for oc in range(Self.out_channels):
                var acc: Scalar[dtype] = 0
                for s in range(Self.spatial_out):
                    var out_idx = oc * Self.spatial_out + s
                    var cache_val = rebind[Scalar[dtype]](
                        cache[b, Self.CONV_CACHE + out_idx]
                    )
                    var go_val = rebind[Scalar[dtype]](grad_output[b, out_idx])
                    acc += Self.ACT.backward(cache_val, go_val)
                var cur = rebind[Scalar[dtype]](grad_params[W_SIZE + oc])
                grad_params[W_SIZE + oc] = cur + acc

            # 3. grad_input via col2im of W.T @ masked_grad
            for i in range(Self.IN_DIM):
                grad_input[b, i] = 0

            for oh in range(Self.out_h):
                for ow in range(Self.out_w):
                    var s = oh * Self.out_w + ow
                    for c in range(Self.in_channels):
                        for kh in range(Self.kernel_size):
                            for kw in range(Self.kernel_size):
                                var ih = oh * Self.stride - Self.padding + kh
                                var iw = ow * Self.stride - Self.padding + kw
                                if (
                                    ih >= 0
                                    and ih < Self.in_h
                                    and iw >= 0
                                    and iw < Self.in_w
                                ):
                                    var in_idx = (
                                        c * Self.in_h * Self.in_w
                                        + ih * Self.in_w
                                        + iw
                                    )
                                    var c_k = (
                                        c * Self.kernel_size * Self.kernel_size
                                        + kh * Self.kernel_size
                                        + kw
                                    )
                                    var dcol_val: Scalar[dtype] = 0
                                    for oc in range(Self.out_channels):
                                        var out_idx = oc * Self.spatial_out + s
                                        var cache_val = rebind[Scalar[dtype]](
                                            cache[b, Self.CONV_CACHE + out_idx]
                                        )
                                        var go_val = rebind[Scalar[dtype]](
                                            grad_output[b, out_idx]
                                        )
                                        var masked_go = Self.ACT.backward(
                                            cache_val, go_val
                                        )
                                        var w_val = rebind[Scalar[dtype]](
                                            params[oc * Self.col_size + c_k]
                                        )
                                        dcol_val += w_val * masked_go
                                    var cur = rebind[Scalar[dtype]](
                                        grad_input[b, in_idx]
                                    )
                                    grad_input[b, in_idx] = cur + dcol_val

    # =========================================================================
    # GPU kernels — naive backward_dx (kept — sufficient parallelism)
    # =========================================================================

    @always_inline
    @staticmethod
    def backward_dx_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
    ):
        """GPU backward for grad_input with fused activation gradient.

        Note: cache layout is (BATCH, CACHE_SIZE) where CACHE_SIZE =
        CONV_CACHE + OUT_DIM per batch. The act_cache section has batch
        stride CACHE_SIZE, not OUT_DIM — so a row_major[BATCH, OC, out_h,
        out_w]() view over cache.ptr + CONV_CACHE would be wrong. Access
        with 2D cache[b, CONV_CACHE + flat_idx] instead.
        """
        var W_4d = TileTensor(
            params.ptr,
            row_major[
                Self.out_channels,
                Self.in_channels,
                Self.kernel_size,
                Self.kernel_size,
            ](),
        )
        var grad_out_4d = TileTensor(
            grad_output.ptr,
            row_major[
                BATCH, Self.out_channels, Self.out_h, Self.out_w
            ](),
        )
        var grad_in_4d = TileTensor(
            grad_input.ptr,
            row_major[BATCH, Self.in_channels, Self.in_h, Self.in_w](),
        )

        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        var total = BATCH * Self.IN_DIM
        if idx >= total:
            return

        var b = idx // Self.IN_DIM
        var in_pos = idx % Self.IN_DIM
        var c = in_pos // (Self.in_h * Self.in_w)
        var rem = in_pos % (Self.in_h * Self.in_w)
        var ih = rem // Self.in_w
        var iw = rem % Self.in_w

        var acc: Scalar[dtype] = 0
        for kh in range(Self.kernel_size):
            for kw in range(Self.kernel_size):
                var oh_num = ih + Self.padding - kh
                var ow_num = iw + Self.padding - kw
                if (
                    oh_num >= 0
                    and oh_num % Self.stride == 0
                    and ow_num >= 0
                    and ow_num % Self.stride == 0
                ):
                    var oh = oh_num // Self.stride
                    var ow = ow_num // Self.stride
                    if oh < Self.out_h and ow < Self.out_w:
                        var s = oh * Self.out_w + ow
                        for oc in range(Self.out_channels):
                            var out_idx = oc * Self.spatial_out + s
                            var cache_val = rebind[Scalar[dtype]](
                                cache[b, Self.CONV_CACHE + out_idx]
                            )
                            var go_val = grad_out_4d[b, oc, oh, ow]
                            var masked_go = Self.ACT.backward(
                                cache_val, go_val
                            )
                            acc += W_4d[oc, c, kh, kw] * masked_go

        grad_in_4d[b, c, ih, iw] = acc

    # =========================================================================
    # GPU kernels — tiled forward (Apple 2x2 / NVIDIA MMA)
    # =========================================================================

    @always_inline
    @staticmethod
    def eval_kernel_2x2[
        BATCH: Int, dtype: DType = DType.float32
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Tiled forward: y = act(W @ im2col(input) + bias) with implicit im2col.

        Grid: (ceil(spatial_out/32), ceil(out_channels/32), BATCH)
        Block: (256, 1)
        """
        comptime BT = 32
        comptime SK = 16
        comptime W_SIZE = Self.out_channels * Self.col_size
        comptime KS2 = Self.kernel_size * Self.kernel_size

        # Note: cache has per-batch layout [im2col | act_cache] with stride
        # CACHE_SIZE, not CONV_CACHE. row_major[BATCH, ...] views over sub-
        # sections of cache would have wrong strides. Keep cache as 2D.
        var W = TileTensor(
            params.ptr, row_major[Self.out_channels, Self.col_size]()
        )
        var bias = TileTensor(
            params.ptr + W_SIZE, row_major[Self.out_channels]()
        )
        var input_4d = TileTensor(
            input.ptr,
            row_major[BATCH, Self.in_channels, Self.in_h, Self.in_w](),
        )
        var output_3d = TileTensor(
            output.ptr,
            row_major[BATCH, Self.out_channels, Self.spatial_out](),
        )

        var tid = Int(thread_idx.x)
        var sub_r = tid // 16
        var sub_c = tid % 16
        var block_oc = Int(block_idx.y) * BT
        var block_s = Int(block_idx.x) * BT
        var batch = Int(block_idx.z)

        var a_smem = LayoutTensor[
            dtype,
            Layout.row_major(BT, SK),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()
        var b_smem = LayoutTensor[
            dtype,
            Layout.row_major(SK, BT),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()

        var acc00: Scalar[dtype] = 0
        var acc01: Scalar[dtype] = 0
        var acc10: Scalar[dtype] = 0
        var acc11: Scalar[dtype] = 0

        comptime num_k_tiles = (Self.col_size + SK - 1) // SK

        for k_tile in range(num_k_tiles):
            var k_off = k_tile * SK

            # Load A tile: W[block_oc + r, k_off + c]
            var a_r0 = tid // SK
            var a_c0 = tid % SK
            var a_r1 = (tid + 256) // SK
            var a_c1 = (tid + 256) % SK

            var ga_oc0 = block_oc + a_r0
            var ga_k0 = k_off + a_c0
            if ga_oc0 < Self.out_channels and ga_k0 < Self.col_size:
                a_smem[a_r0, a_c0] = W[ga_oc0, ga_k0]
            else:
                a_smem[a_r0, a_c0] = 0

            var ga_oc1 = block_oc + a_r1
            var ga_k1 = k_off + a_c1
            if ga_oc1 < Self.out_channels and ga_k1 < Self.col_size:
                a_smem[a_r1, a_c1] = W[ga_oc1, ga_k1]
            else:
                a_smem[a_r1, a_c1] = 0

            # Load B tile: im2col[k_off + r, block_s + c] (implicit)
            var b_r0 = tid // BT
            var b_c0 = tid % BT
            var b_r1 = (tid + 256) // BT
            var b_c1 = (tid + 256) % BT

            # Element 0
            var k_idx0 = k_off + b_r0
            var s_idx0 = block_s + b_c0
            var val0: Scalar[dtype] = 0
            if k_idx0 < Self.col_size and s_idx0 < Self.spatial_out:
                var ch0 = k_idx0 // KS2
                var rem_k0 = k_idx0 % KS2
                var kh0 = rem_k0 // Self.kernel_size
                var kw0 = rem_k0 % Self.kernel_size
                var oh0 = s_idx0 // Self.out_w
                var ow0 = s_idx0 % Self.out_w
                var ih0 = oh0 * Self.stride - Self.padding + kh0
                var iw0 = ow0 * Self.stride - Self.padding + kw0
                if (
                    ih0 >= 0
                    and ih0 < Self.in_h
                    and iw0 >= 0
                    and iw0 < Self.in_w
                ):
                    val0 = input_4d[batch, ch0, ih0, iw0]
            b_smem[b_r0, b_c0] = val0
            if (
                Int(block_idx.y) == 0
                and k_idx0 < Self.col_size
                and s_idx0 < Self.spatial_out
            ):
                cache[batch, s_idx0 * Self.col_size + k_idx0] = val0

            # Element 1
            var k_idx1 = k_off + b_r1
            var s_idx1 = block_s + b_c1
            var val1: Scalar[dtype] = 0
            if k_idx1 < Self.col_size and s_idx1 < Self.spatial_out:
                var ch1 = k_idx1 // KS2
                var rem_k1 = k_idx1 % KS2
                var kh1 = rem_k1 // Self.kernel_size
                var kw1 = rem_k1 % Self.kernel_size
                var oh1 = s_idx1 // Self.out_w
                var ow1 = s_idx1 % Self.out_w
                var ih1 = oh1 * Self.stride - Self.padding + kh1
                var iw1 = ow1 * Self.stride - Self.padding + kw1
                if (
                    ih1 >= 0
                    and ih1 < Self.in_h
                    and iw1 >= 0
                    and iw1 < Self.in_w
                ):
                    val1 = input_4d[batch, ch1, ih1, iw1]
            b_smem[b_r1, b_c1] = val1
            if (
                Int(block_idx.y) == 0
                and k_idx1 < Self.col_size
                and s_idx1 < Self.spatial_out
            ):
                cache[batch, s_idx1 * Self.col_size + k_idx1] = val1

            barrier()

            for k in range(SK):
                if k_off + k < Self.col_size:
                    var a0 = rebind[Scalar[dtype]](a_smem[sub_r * 2, k])
                    var a1 = rebind[Scalar[dtype]](a_smem[sub_r * 2 + 1, k])
                    var b0 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2])
                    var b1 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2 + 1])
                    acc00 += a0 * b0
                    acc01 += a0 * b1
                    acc10 += a1 * b0
                    acc11 += a1 * b1

            barrier()

        # Store with bias + fused activation — 3D output + 3D act-cache views
        var oc0 = block_oc + sub_r * 2
        var s0 = block_s + sub_c * 2

        if oc0 < Self.out_channels and s0 < Self.spatial_out:
            var pre_act = acc00 + bias[oc0]
            var act_out = Self.ACT.forward(pre_act)
            var out_idx = oc0 * Self.spatial_out + s0
            output_3d[batch, oc0, s0] = act_out
            cache[batch, Self.CONV_CACHE + out_idx] = Self.ACT.cache(
                pre_act, act_out
            )
        if oc0 < Self.out_channels and s0 + 1 < Self.spatial_out:
            var pre_act = acc01 + bias[oc0]
            var act_out = Self.ACT.forward(pre_act)
            var out_idx = oc0 * Self.spatial_out + s0 + 1
            output_3d[batch, oc0, s0 + 1] = act_out
            cache[batch, Self.CONV_CACHE + out_idx] = Self.ACT.cache(
                pre_act, act_out
            )
        if oc0 + 1 < Self.out_channels and s0 < Self.spatial_out:
            var pre_act = acc10 + bias[oc0 + 1]
            var act_out = Self.ACT.forward(pre_act)
            var out_idx = (oc0 + 1) * Self.spatial_out + s0
            output_3d[batch, oc0 + 1, s0] = act_out
            cache[batch, Self.CONV_CACHE + out_idx] = Self.ACT.cache(
                pre_act, act_out
            )
        if oc0 + 1 < Self.out_channels and s0 + 1 < Self.spatial_out:
            var pre_act = acc11 + bias[oc0 + 1]
            var act_out = Self.ACT.forward(pre_act)
            var out_idx = (oc0 + 1) * Self.spatial_out + s0 + 1
            output_3d[batch, oc0 + 1, s0 + 1] = act_out
            cache[batch, Self.CONV_CACHE + out_idx] = Self.ACT.cache(
                pre_act, act_out
            )

    # =========================================================================
    # GPU kernels — tiled backward dW and db (with fused activation gradient)
    # =========================================================================

    @always_inline
    @staticmethod
    def backward_dW_kernel_2x2[
        BATCH: Int, dtype: DType = DType.float32
    ](
        dW: LayoutTensor[
            dtype,
            Layout.row_major(Self.out_channels, Self.col_size),
            MutAnyOrigin,
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
    ):
        """Tiled backward: dW = sum_b[masked_grad_b @ col_b.T].

        ACT.backward applied when loading grad_output into A tile.
        Grid: (ceil(col_size/32), ceil(out_channels/32))
        Block: (256, 1)

        Note: cache stays 2D — fused layout is [im2col | act_cache] per batch
        with stride CACHE_SIZE. grad_output is fully contiguous → 3D TT view.
        """
        comptime BT = 32
        comptime SK = 16
        comptime K_TOTAL = BATCH * Self.spatial_out

        var grad_out_3d = TileTensor(
            grad_output.ptr,
            row_major[BATCH, Self.out_channels, Self.spatial_out](),
        )

        var tid = Int(thread_idx.x)
        var sub_r = tid // 16
        var sub_c = tid % 16
        var block_oc = Int(block_idx.y) * BT
        var block_k = Int(block_idx.x) * BT

        var a_smem = LayoutTensor[
            dtype,
            Layout.row_major(BT, SK),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()
        var b_smem = LayoutTensor[
            dtype,
            Layout.row_major(SK, BT),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()

        var acc00: Scalar[dtype] = 0
        var acc01: Scalar[dtype] = 0
        var acc10: Scalar[dtype] = 0
        var acc11: Scalar[dtype] = 0

        comptime num_k_tiles = (K_TOTAL + SK - 1) // SK

        for k_tile in range(num_k_tiles):
            var k_off = k_tile * SK

            # Load A: masked grad_output — apply ACT.backward
            var a_r0 = tid // SK
            var a_c0 = tid % SK
            var a_r1 = (tid + 256) // SK
            var a_c1 = (tid + 256) % SK

            var ki0 = k_off + a_c0
            var g_oc0 = block_oc + a_r0
            if g_oc0 < Self.out_channels and ki0 < K_TOTAL:
                var b0 = ki0 // Self.spatial_out
                var s0 = ki0 % Self.spatial_out
                var out_idx0 = g_oc0 * Self.spatial_out + s0
                var go_val = grad_out_3d[b0, g_oc0, s0]
                var cache_val = rebind[Scalar[dtype]](
                    cache[b0, Self.CONV_CACHE + out_idx0]
                )
                a_smem[a_r0, a_c0] = Self.ACT.backward(cache_val, go_val)
            else:
                a_smem[a_r0, a_c0] = 0

            var ki1 = k_off + a_c1
            var g_oc1 = block_oc + a_r1
            if g_oc1 < Self.out_channels and ki1 < K_TOTAL:
                var b1 = ki1 // Self.spatial_out
                var s1 = ki1 % Self.spatial_out
                var out_idx1 = g_oc1 * Self.spatial_out + s1
                var go_val = grad_out_3d[b1, g_oc1, s1]
                var cache_val = rebind[Scalar[dtype]](
                    cache[b1, Self.CONV_CACHE + out_idx1]
                )
                a_smem[a_r1, a_c1] = Self.ACT.backward(cache_val, go_val)
            else:
                a_smem[a_r1, a_c1] = 0

            # Load B: im2col cache (same as plain Conv2D)
            var b_r0 = tid // BT
            var b_c0 = tid % BT
            var b_r1 = (tid + 256) // BT
            var b_c1 = (tid + 256) % BT

            var bki0 = k_off + b_r0
            var gk0 = block_k + b_c0
            if gk0 < Self.col_size and bki0 < K_TOTAL:
                var bb0 = bki0 // Self.spatial_out
                var bs0 = bki0 % Self.spatial_out
                b_smem[b_r0, b_c0] = cache[bb0, bs0 * Self.col_size + gk0]
            else:
                b_smem[b_r0, b_c0] = 0

            var bki1 = k_off + b_r1
            var gk1 = block_k + b_c1
            if gk1 < Self.col_size and bki1 < K_TOTAL:
                var bb1 = bki1 // Self.spatial_out
                var bs1 = bki1 % Self.spatial_out
                b_smem[b_r1, b_c1] = cache[bb1, bs1 * Self.col_size + gk1]
            else:
                b_smem[b_r1, b_c1] = 0

            barrier()

            for k in range(SK):
                if k_off + k < K_TOTAL:
                    var a0 = rebind[Scalar[dtype]](a_smem[sub_r * 2, k])
                    var a1 = rebind[Scalar[dtype]](a_smem[sub_r * 2 + 1, k])
                    var b0v = rebind[Scalar[dtype]](b_smem[k, sub_c * 2])
                    var b1v = rebind[Scalar[dtype]](b_smem[k, sub_c * 2 + 1])
                    acc00 += a0 * b0v
                    acc01 += a0 * b1v
                    acc10 += a1 * b0v
                    acc11 += a1 * b1v

            barrier()

        var oc0 = block_oc + sub_r * 2
        var k0 = block_k + sub_c * 2
        # Accumulate (+=) into dW so multiple backward calls in a single
        # update (MuZero K-step unroll, DreamerV3/TD-MPC2 BPTT) sum
        # gradients instead of overwriting. Caller pre-zeros via zero_grads.
        if oc0 < Self.out_channels and k0 < Self.col_size:
            dW[oc0, k0] = dW[oc0, k0] + acc00
        if oc0 < Self.out_channels and k0 + 1 < Self.col_size:
            dW[oc0, k0 + 1] = dW[oc0, k0 + 1] + acc01
        if oc0 + 1 < Self.out_channels and k0 < Self.col_size:
            dW[oc0 + 1, k0] = dW[oc0 + 1, k0] + acc10
        if oc0 + 1 < Self.out_channels and k0 + 1 < Self.col_size:
            dW[oc0 + 1, k0 + 1] = dW[oc0 + 1, k0 + 1] + acc11

    @always_inline
    @staticmethod
    def backward_db_kernel[
        BATCH: Int, dtype: DType = DType.float32
    ](
        db: LayoutTensor[
            dtype, Layout.row_major(Self.out_channels), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
    ):
        """Formula: db = sum(masked_dy, axis=batch+spatial).

        Parallelized: one thread per (oc, batch) pair, each reduces over
        spatial_out, then block-level reduction across batches via shared mem.

        Grid: (out_channels,)
        Block: (TPB,)  — TPB threads split across BATCH dimension

        Note: cache stays 2D due to fused [im2col | act_cache] per-batch layout.
        """
        var grad_out_3d = TileTensor(
            grad_output.ptr,
            row_major[BATCH, Self.out_channels, Self.spatial_out](),
        )

        var oc = Int(block_idx.x)
        if oc >= Self.out_channels:
            return
        var tid = Int(thread_idx.x)

        # Each thread reduces a chunk of the batch dimension
        var acc: Scalar[dtype] = 0
        for b in range(tid, BATCH, TPB):
            for s in range(Self.spatial_out):
                var out_idx = oc * Self.spatial_out + s
                var go_val = grad_out_3d[b, oc, s]
                var cache_val = rebind[Scalar[dtype]](
                    cache[b, Self.CONV_CACHE + out_idx]
                )
                acc += Self.ACT.backward(cache_val, go_val)

        # Block-level reduction via shared memory
        var smem = LayoutTensor[
            dtype,
            Layout.row_major(TPB),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()
        smem[tid] = acc
        barrier()

        # Tree reduction
        var db_stride = TPB // 2
        while db_stride > 0:
            if tid < db_stride:
                smem[tid] = rebind[Scalar[dtype]](smem[tid]) + rebind[
                    Scalar[dtype]
                ](smem[tid + db_stride])
            barrier()
            db_stride //= 2

        if tid == 0:
            # Accumulate into db (pre-zeroed via zero_grads) so multi-call
            # backward sequences sum bias gradients instead of overwriting.
            db[oc] = db[oc] + smem[0]

    # =========================================================================
    # GPU kernels — MMA matmul (NVIDIA, replaces max_matmul)
    # =========================================================================

    @always_inline
    @staticmethod
    def conv_matmul_fwd_mma[
        K_TOTAL: Int, dtype: DType = DType.float32
    ](
        out_temp: LayoutTensor[
            dtype,
            Layout.row_major(K_TOTAL, Self.out_channels),
            MutAnyOrigin,
        ],
        col_flat: LayoutTensor[
            dtype,
            Layout.row_major(K_TOTAL, Self.col_size),
            MutAnyOrigin,
        ],
        W: LayoutTensor[
            dtype,
            Layout.row_major(Self.out_channels, Self.col_size),
            MutAnyOrigin,
        ],
    ):
        """MMA forward: out_temp = col_flat @ W.T (transpose_b)."""
        comptime if is_nvidia_gpu():
            var tid = Int(thread_idx.x)
            var warp_id = tid // 32
            var warp_m = warp_id // MMA_WARPS_N
            var warp_n = warp_id % MMA_WARPS_N

            var block_row = Int(block_idx.y) * MMA_BLOCK_M
            var block_col = Int(block_idx.x) * MMA_BLOCK_N

            var a_smem = LayoutTensor[
                dtype,
                Layout.row_major(MMA_BLOCK_M, MMA_K),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()
            var b_smem = LayoutTensor[
                dtype,
                Layout.row_major(MMA_K, MMA_BLOCK_N),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()

            var acc = SIMD[DType.float32, 4](0)
            var lid = lane_id()
            var group_id = lid >> 2
            var group_lane = lid % 4

            comptime num_k_tiles = (Self.col_size + MMA_K - 1) // MMA_K

            for k_tile in range(num_k_tiles):
                var k_off = k_tile * MMA_K

                var a_r = tid // MMA_K
                var a_c = tid % MMA_K
                var ga_r = block_row + a_r
                var ga_c = k_off + a_c
                if ga_r < K_TOTAL and ga_c < Self.col_size:
                    a_smem[a_r, a_c] = col_flat[ga_r, ga_c]
                else:
                    a_smem[a_r, a_c] = 0

                var br = tid // MMA_BLOCK_N
                var bc = tid % MMA_BLOCK_N
                var gb_k = k_off + br
                var gb_n = block_col + bc
                if gb_n < Self.out_channels and gb_k < Self.col_size:
                    b_smem[br, bc] = W[gb_n, gb_k]
                else:
                    b_smem[br, bc] = 0

                barrier()

                var warp_row = warp_m * MMA_M
                var a_frag = SIMD[DType.float32, 4](
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id), Int(group_lane)]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[
                            warp_row + Int(group_id) + 8, Int(group_lane)
                        ]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[
                            warp_row + Int(group_id), Int(group_lane) + 4
                        ]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[
                            warp_row + Int(group_id) + 8,
                            Int(group_lane) + 4,
                        ]
                    ),
                )
                var warp_col = warp_n * MMA_N
                var b_frag = SIMD[DType.float32, 2](
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane), warp_col + Int(group_id)]
                    ),
                    rebind[Scalar[DType.float32]](
                        b_smem[
                            Int(group_lane) + 4, warp_col + Int(group_id)
                        ]
                    ),
                )

                mma(acc, a_frag, b_frag, acc)
                barrier()

            var r0 = block_row + warp_m * MMA_M + Int(group_id)
            var r1 = r0 + 8
            var c0 = block_col + warp_n * MMA_N + Int(group_lane * 2)
            var c1 = c0 + 1

            if r0 < K_TOTAL and c0 < Self.out_channels:
                out_temp[r0, c0] = acc[0].cast[dtype]()
            if r0 < K_TOTAL and c1 < Self.out_channels:
                out_temp[r0, c1] = acc[1].cast[dtype]()
            if r1 < K_TOTAL and c0 < Self.out_channels:
                out_temp[r1, c0] = acc[2].cast[dtype]()
            if r1 < K_TOTAL and c1 < Self.out_channels:
                out_temp[r1, c1] = acc[3].cast[dtype]()

    @always_inline
    @staticmethod
    def conv_matmul_dW_mma[
        K_TOTAL: Int, dtype: DType = DType.float32
    ](
        dW: LayoutTensor[
            dtype,
            Layout.row_major(Self.out_channels, Self.col_size),
            MutAnyOrigin,
        ],
        grad_reshaped: LayoutTensor[
            dtype,
            Layout.row_major(Self.out_channels, K_TOTAL),
            MutAnyOrigin,
        ],
        col_flat: LayoutTensor[
            dtype,
            Layout.row_major(K_TOTAL, Self.col_size),
            MutAnyOrigin,
        ],
    ):
        """MMA backward: dW = grad_reshaped @ col_flat."""
        comptime if is_nvidia_gpu():
            var tid = Int(thread_idx.x)
            var warp_id = tid // 32
            var warp_m = warp_id // MMA_WARPS_N
            var warp_n = warp_id % MMA_WARPS_N

            var block_row = Int(block_idx.y) * MMA_BLOCK_M
            var block_col = Int(block_idx.x) * MMA_BLOCK_N

            var a_smem = LayoutTensor[
                dtype,
                Layout.row_major(MMA_BLOCK_M, MMA_K),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()
            var b_smem = LayoutTensor[
                dtype,
                Layout.row_major(MMA_K, MMA_BLOCK_N),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()

            var acc = SIMD[DType.float32, 4](0)
            var lid = lane_id()
            var group_id = lid >> 2
            var group_lane = lid % 4

            comptime num_k_tiles = (K_TOTAL + MMA_K - 1) // MMA_K

            for k_tile in range(num_k_tiles):
                var k_off = k_tile * MMA_K

                var a_r = tid // MMA_K
                var a_c = tid % MMA_K
                var ga_r = block_row + a_r
                var ga_c = k_off + a_c
                if ga_r < Self.out_channels and ga_c < K_TOTAL:
                    a_smem[a_r, a_c] = grad_reshaped[ga_r, ga_c]
                else:
                    a_smem[a_r, a_c] = 0

                var br = tid // MMA_BLOCK_N
                var bc = tid % MMA_BLOCK_N
                var gb_r = k_off + br
                var gb_c = block_col + bc
                if gb_r < K_TOTAL and gb_c < Self.col_size:
                    b_smem[br, bc] = col_flat[gb_r, gb_c]
                else:
                    b_smem[br, bc] = 0

                barrier()

                var warp_row = warp_m * MMA_M
                var a_frag = SIMD[DType.float32, 4](
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id), Int(group_lane)]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[
                            warp_row + Int(group_id) + 8, Int(group_lane)
                        ]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[
                            warp_row + Int(group_id), Int(group_lane) + 4
                        ]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[
                            warp_row + Int(group_id) + 8,
                            Int(group_lane) + 4,
                        ]
                    ),
                )
                var warp_col = warp_n * MMA_N
                var b_frag = SIMD[DType.float32, 2](
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane), warp_col + Int(group_id)]
                    ),
                    rebind[Scalar[DType.float32]](
                        b_smem[
                            Int(group_lane) + 4, warp_col + Int(group_id)
                        ]
                    ),
                )

                mma(acc, a_frag, b_frag, acc)
                barrier()

            var r0 = block_row + warp_m * MMA_M + Int(group_id)
            var r1 = r0 + 8
            var c0 = block_col + warp_n * MMA_N + Int(group_lane * 2)
            var c1 = c0 + 1

            # Accumulate (+=) into dW. Multi-call backward (MuZero K-step
            # unroll, RSSM/world-model BPTT) requires accumulation across
            # calls. Caller pre-zeros grad_params via zero_grads.
            if r0 < Self.out_channels and c0 < Self.col_size:
                dW[r0, c0] = dW[r0, c0] + acc[0].cast[dtype]()
            if r0 < Self.out_channels and c1 < Self.col_size:
                dW[r0, c1] = dW[r0, c1] + acc[1].cast[dtype]()
            if r1 < Self.out_channels and c0 < Self.col_size:
                dW[r1, c0] = dW[r1, c0] + acc[2].cast[dtype]()
            if r1 < Self.out_channels and c1 < Self.col_size:
                dW[r1, c1] = dW[r1, c1] + acc[3].cast[dtype]()

    @always_inline
    @staticmethod
    def conv_matmul_dx_mma[
        K_TOTAL: Int, dtype: DType = DType.float32
    ](
        dcol: LayoutTensor[
            dtype,
            Layout.row_major(Self.col_size, K_TOTAL),
            MutAnyOrigin,
        ],
        W: LayoutTensor[
            dtype,
            Layout.row_major(Self.out_channels, Self.col_size),
            MutAnyOrigin,
        ],
        grad_reshaped: LayoutTensor[
            dtype,
            Layout.row_major(Self.out_channels, K_TOTAL),
            MutAnyOrigin,
        ],
    ):
        """MMA backward: dcol = W.T @ grad_reshaped (transpose_a on W)."""
        comptime if is_nvidia_gpu():
            var tid = Int(thread_idx.x)
            var warp_id = tid // 32
            var warp_m = warp_id // MMA_WARPS_N
            var warp_n = warp_id % MMA_WARPS_N

            var block_row = Int(block_idx.y) * MMA_BLOCK_M
            var block_col = Int(block_idx.x) * MMA_BLOCK_N

            var a_smem = LayoutTensor[
                dtype,
                Layout.row_major(MMA_BLOCK_M, MMA_K),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()
            var b_smem = LayoutTensor[
                dtype,
                Layout.row_major(MMA_K, MMA_BLOCK_N),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()

            var acc = SIMD[DType.float32, 4](0)
            var lid = lane_id()
            var group_id = lid >> 2
            var group_lane = lid % 4

            comptime num_k_tiles = (Self.out_channels + MMA_K - 1) // MMA_K

            for k_tile in range(num_k_tiles):
                var k_off = k_tile * MMA_K

                var a_r = tid // MMA_K
                var a_c = tid % MMA_K
                var ga_m = block_row + a_r
                var ga_k = k_off + a_c
                if ga_m < Self.col_size and ga_k < Self.out_channels:
                    a_smem[a_r, a_c] = W[ga_k, ga_m]
                else:
                    a_smem[a_r, a_c] = 0

                var br = tid // MMA_BLOCK_N
                var bc = tid % MMA_BLOCK_N
                var gb_r = k_off + br
                var gb_c = block_col + bc
                if gb_r < Self.out_channels and gb_c < K_TOTAL:
                    b_smem[br, bc] = grad_reshaped[gb_r, gb_c]
                else:
                    b_smem[br, bc] = 0

                barrier()

                var warp_row = warp_m * MMA_M
                var a_frag = SIMD[DType.float32, 4](
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id), Int(group_lane)]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[
                            warp_row + Int(group_id) + 8, Int(group_lane)
                        ]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[
                            warp_row + Int(group_id), Int(group_lane) + 4
                        ]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[
                            warp_row + Int(group_id) + 8,
                            Int(group_lane) + 4,
                        ]
                    ),
                )
                var warp_col = warp_n * MMA_N
                var b_frag = SIMD[DType.float32, 2](
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane), warp_col + Int(group_id)]
                    ),
                    rebind[Scalar[DType.float32]](
                        b_smem[
                            Int(group_lane) + 4, warp_col + Int(group_id)
                        ]
                    ),
                )

                mma(acc, a_frag, b_frag, acc)
                barrier()

            var r0 = block_row + warp_m * MMA_M + Int(group_id)
            var r1 = r0 + 8
            var c0 = block_col + warp_n * MMA_N + Int(group_lane * 2)
            var c1 = c0 + 1

            if r0 < Self.col_size and c0 < K_TOTAL:
                dcol[r0, c0] = acc[0].cast[dtype]()
            if r0 < Self.col_size and c1 < K_TOTAL:
                dcol[r0, c1] = acc[1].cast[dtype]()
            if r1 < Self.col_size and c0 < K_TOTAL:
                dcol[r1, c0] = acc[2].cast[dtype]()
            if r1 < Self.col_size and c1 < K_TOTAL:
                dcol[r1, c1] = acc[3].cast[dtype]()

    # =========================================================================
    # GPU launchers (tiled forward + tiled dW/db + naive dx)
    # =========================================================================

    @staticmethod
    def eval_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        workspace: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)

        comptime if has_nvidia_gpu_accelerator():
            # NVIDIA: im2col → MMA matmul → transpose + bias + activation
            comptime K_TOTAL = BATCH * Self.spatial_out
            comptime KS2 = Self.kernel_size * Self.kernel_size
            comptime W_SIZE = Self.out_channels * Self.col_size

            # 1. Explicit im2col: input → cache im2col section (s*col_size+k)
            comptime im2col_elems = BATCH * Self.CONV_CACHE
            comptime im2col_blocks = (im2col_elems + TPB - 1) // TPB

            @parameter
            @always_inline
            def im2col_wrapper(
                cache_out: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.CACHE_SIZE),
                    MutAnyOrigin,
                ],
                input: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.IN_DIM),
                    ImmutAnyOrigin,
                ],
            ):
                var input_4d = TileTensor(
                    input.ptr,
                    row_major[
                        BATCH, Self.in_channels, Self.in_h, Self.in_w
                    ](),
                )
                var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                if idx >= im2col_elems:
                    return
                var b = idx // Self.CONV_CACHE
                var pos = idx % Self.CONV_CACHE
                var s = pos // Self.col_size
                var k = pos % Self.col_size
                var oh = s // Self.out_w
                var ow = s % Self.out_w
                var ch = k // KS2
                var rem_k = k % KS2
                var kh = rem_k // Self.kernel_size
                var kw = rem_k % Self.kernel_size
                var ih = oh * Self.stride - Self.padding + kh
                var iw = ow * Self.stride - Self.padding + kw
                var val: Scalar[dtype] = 0
                if ih >= 0 and ih < Self.in_h and iw >= 0 and iw < Self.in_w:
                    val = input_4d[b, ch, ih, iw]
                # cache_out stays 2D: fused layout is [im2col | act_cache]
                # per-batch with stride CACHE_SIZE, not CONV_CACHE
                cache_out[b, pos] = val

            ctx.enqueue_function[im2col_wrapper](
                cache,
                input_immut,
                grid_dim=(im2col_blocks,),
                block_dim=(TPB,),
            )

            # Workspace layout: [col_flat: BATCH*CONV_CACHE | out_temp: BATCH*OUT_DIM | w_t: col_size*OC]
            # 2. Strided copy: cache im2col → col_flat (skip activation gap)
            var col_flat = LayoutTensor[
                dtype,
                Layout.row_major(K_TOTAL, Self.col_size),
                MutAnyOrigin,
            ](workspace)

            comptime col_elems = K_TOTAL * Self.col_size
            comptime col_blocks = (col_elems + TPB - 1) // TPB

            @parameter
            @always_inline
            def copy_col_fwd(
                dst: LayoutTensor[
                    dtype,
                    Layout.row_major(K_TOTAL, Self.col_size),
                    MutAnyOrigin,
                ],
                src: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.CACHE_SIZE),
                    ImmutAnyOrigin,
                ],
            ):
                # dst (col_flat) is contiguous (K_TOTAL=BATCH*spatial_out,
                # col_size), can be viewed 3D. src (cache) stays 2D due to
                # [im2col | act_cache] per-batch layout.
                var dst_3d = TileTensor(
                    dst.ptr,
                    row_major[BATCH, Self.spatial_out, Self.col_size](),
                )
                var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                if idx >= col_elems:
                    return
                var row = idx // Self.col_size
                var k = idx % Self.col_size
                var b = row // Self.spatial_out
                var s = row % Self.spatial_out
                dst_3d[b, s, k] = rebind[Scalar[dtype]](
                    src[b, s * Self.col_size + k]
                )

            var cache_immut_fwd = LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
            ](cache.ptr)

            ctx.enqueue_function[copy_col_fwd](
                col_flat,
                cache_immut_fwd,
                grid_dim=(col_blocks,),
                block_dim=(TPB,),
            )

            # 3. Forward matmul: out_temp (K_TOTAL × OC) = col_flat @ W.T.
            # Same as Conv2D phase 3a — branch between max_matmul[transpose_b]
            # and the custom MMA kernel. The transpose+bias+activation+cache
            # post-kernel runs unchanged for both paths.
            var W_mat = LayoutTensor[
                dtype,
                Layout.row_major(Self.out_channels, Self.col_size),
                MutAnyOrigin,
            ](params.ptr)

            comptime out_temp_ws_offset = BATCH * Self.CONV_CACHE
            var out_temp = LayoutTensor[
                dtype,
                Layout.row_major(K_TOTAL, Self.out_channels),
                MutAnyOrigin,
            ](workspace + out_temp_ws_offset)

            comptime if Self.USE_MAX_KERNELS:
                max_matmul[transpose_b=True, target="gpu"](
                    lt_to_tt(out_temp),
                    lt_to_tt(col_flat),
                    lt_to_tt(W_mat),
                    DeviceContextPtr(ctx),
                )
            else:
                comptime fwd_grid_x = (Self.out_channels + MMA_BLOCK_N - 1) // MMA_BLOCK_N
                comptime fwd_grid_y = (K_TOTAL + MMA_BLOCK_M - 1) // MMA_BLOCK_M

                @parameter
                @always_inline
                def fwd_mm_wrapper(
                    out_temp: LayoutTensor[
                        dtype,
                        Layout.row_major(K_TOTAL, Self.out_channels),
                        MutAnyOrigin,
                    ],
                    col_flat: LayoutTensor[
                        dtype,
                        Layout.row_major(K_TOTAL, Self.col_size),
                        MutAnyOrigin,
                    ],
                    W_mat: LayoutTensor[
                        dtype,
                        Layout.row_major(Self.out_channels, Self.col_size),
                        MutAnyOrigin,
                    ],
                ):
                    Self.conv_matmul_fwd_mma[K_TOTAL, dtype](
                        out_temp, col_flat, W_mat
                    )

                ctx.enqueue_function[fwd_mm_wrapper](
                    out_temp,
                    col_flat,
                    W_mat,
                    grid_dim=(fwd_grid_x, fwd_grid_y),
                    block_dim=(MMA_BLOCK_THREADS, 1),
                )

            # 5. Transpose output + bias + activation + cache act values
            # out_temp[b*S+s, oc] → output[b, oc*S+s] = act(val + bias[oc])
            # Also cache activation state at cache[b, CONV_CACHE + oc*S+s]
            comptime out_elems = BATCH * Self.OUT_DIM
            comptime out_blocks = (out_elems + TPB - 1) // TPB

            @parameter
            @always_inline
            def transpose_output_act_wrapper(
                output: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.OUT_DIM),
                    MutAnyOrigin,
                ],
                out_temp: LayoutTensor[
                    dtype,
                    Layout.row_major(K_TOTAL, Self.out_channels),
                    MutAnyOrigin,
                ],
                params: LayoutTensor[
                    dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
                ],
                cache_out: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.CACHE_SIZE),
                    MutAnyOrigin,
                ],
            ):
                # out_temp (K_TOTAL, OC) is contiguous → 3D view
                # (BATCH, spatial_out, OC). output fully contiguous → 3D.
                # cache_out stays 2D (fused per-batch layout issue).
                var out_temp_3d = TileTensor(
                    out_temp.ptr,
                    row_major[
                        BATCH, Self.spatial_out, Self.out_channels
                    ](),
                )
                var output_3d = TileTensor(
                    output.ptr,
                    row_major[
                        BATCH, Self.out_channels, Self.spatial_out
                    ](),
                )
                var bias_tt = TileTensor(
                    params.ptr + W_SIZE, row_major[Self.out_channels]()
                )
                var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                if idx >= out_elems:
                    return
                var b = idx // Self.OUT_DIM
                var out_pos = idx % Self.OUT_DIM
                var oc = out_pos // Self.spatial_out
                var s = out_pos % Self.spatial_out
                var pre_act = out_temp_3d[b, s, oc] + bias_tt[oc]
                var act_out = Self.ACT.forward(pre_act)
                output_3d[b, oc, s] = act_out
                cache_out[b, Self.CONV_CACHE + out_pos] = Self.ACT.cache(
                    pre_act, act_out
                )

            ctx.enqueue_function[transpose_output_act_wrapper](
                output,
                out_temp,
                params_immut,
                cache,
                grid_dim=(out_blocks,),
                block_dim=(TPB,),
            )
        else:
            # Apple: fused im2col + tiled matmul + activation kernel
            comptime grid_x = (Self.spatial_out + 31) // 32
            comptime grid_y = (Self.out_channels + 31) // 32

            @parameter
            @always_inline
            def wrapper(
                output: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
                ],
                input: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
                ],
                params: LayoutTensor[
                    dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
                ],
                cache: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.CACHE_SIZE),
                    MutAnyOrigin,
                ],
            ):
                Self.eval_kernel_2x2[BATCH, dtype](output, input, params, cache)

            ctx.enqueue_function[wrapper](
                output,
                input_immut,
                params_immut,
                cache,
                grid_dim=(grid_x, grid_y, BATCH),
                block_dim=(MMA_BLOCK_THREADS, 1),
            )

    @staticmethod
    def vjp_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ](cache.ptr)

        var dW = LayoutTensor[
            dtype,
            Layout.row_major(Self.out_channels, Self.col_size),
            MutAnyOrigin,
        ](grad_params.ptr)

        comptime if has_nvidia_gpu_accelerator():
            # ── dW FIRST (before dx) ──
            # dW = masked_grad_reshaped @ col_reshaped
            # masked_grad: (OC, BATCH*S) with ACT.backward applied
            # col_reshaped: (BATCH*S, col_size)
            comptime K_TOTAL = BATCH * Self.spatial_out

            # Workspace: [col_flat: CONV_CACHE*BATCH | grad_reshaped: OUT_DIM*BATCH | ...]
            comptime grad_reshaped_offset = K_TOTAL * Self.col_size
            var grad_reshaped = LayoutTensor[
                dtype,
                Layout.row_major(Self.out_channels, K_TOTAL),
                MutAnyOrigin,
            ](workspace + grad_reshaped_offset)

            # Cache is (batch, s*col_size+k | act_cache), but CACHE_SIZE
            # includes activation cache so we can't reinterpret directly.
            # Use pre-allocated workspace for strided copy.
            var col_flat = LayoutTensor[
                dtype,
                Layout.row_major(K_TOTAL, Self.col_size),
                MutAnyOrigin,
            ](workspace)

            comptime col_elems = K_TOTAL * Self.col_size
            comptime col_blocks = (col_elems + TPB - 1) // TPB

            @parameter
            @always_inline
            def copy_col_wrapper(
                dst: LayoutTensor[
                    dtype,
                    Layout.row_major(K_TOTAL, Self.col_size),
                    MutAnyOrigin,
                ],
                src: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.CACHE_SIZE),
                    ImmutAnyOrigin,
                ],
            ):
                # dst (col_flat in workspace) contiguous → 3D view. src
                # (cache) stays 2D due to fused per-batch layout.
                var dst_3d = TileTensor(
                    dst.ptr,
                    row_major[BATCH, Self.spatial_out, Self.col_size](),
                )
                var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                if idx >= col_elems:
                    return
                var row = idx // Self.col_size
                var k = idx % Self.col_size
                var b = row // Self.spatial_out
                var s = row % Self.spatial_out
                dst_3d[b, s, k] = rebind[Scalar[dtype]](
                    src[b, s * Self.col_size + k]
                )

            ctx.enqueue_function[copy_col_wrapper](
                col_flat,
                cache_immut,
                grid_dim=(col_blocks,),
                block_dim=(TPB,),
            )

            # Transpose + mask grad: apply ACT.backward and reshape
            # src[b, oc*S + s] → dst[oc, b*S + s] with ACT.backward
            comptime grad_elems = Self.out_channels * K_TOTAL
            comptime grad_blocks = (grad_elems + TPB - 1) // TPB

            @parameter
            @always_inline
            def transpose_mask_grad_wrapper(
                dst: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.out_channels, K_TOTAL),
                    MutAnyOrigin,
                ],
                src: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.OUT_DIM),
                    ImmutAnyOrigin,
                ],
                act_cache: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.CACHE_SIZE),
                    ImmutAnyOrigin,
                ],
            ):
                # src (grad_output) contiguous → 3D view. dst (grad_reshaped
                # in workspace) contiguous → 3D view. act_cache stays 2D.
                var src_3d = TileTensor(
                    src.ptr,
                    row_major[
                        BATCH, Self.out_channels, Self.spatial_out
                    ](),
                )
                var dst_3d = TileTensor(
                    dst.ptr,
                    row_major[
                        Self.out_channels, BATCH, Self.spatial_out
                    ](),
                )
                var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                if idx >= grad_elems:
                    return
                var oc = idx // K_TOTAL
                var bs = idx % K_TOTAL
                var b = bs // Self.spatial_out
                var s = bs % Self.spatial_out
                var out_idx = oc * Self.spatial_out + s
                var go_val = src_3d[b, oc, s]
                var cache_val = rebind[Scalar[dtype]](
                    act_cache[b, Self.CONV_CACHE + out_idx]
                )
                dst_3d[oc, b, s] = Self.ACT.backward(cache_val, go_val)

            ctx.enqueue_function[transpose_mask_grad_wrapper](
                grad_reshaped,
                grad_output_immut,
                cache_immut,
                grid_dim=(grad_blocks,),
                block_dim=(TPB,),
            )

            # dW (OC × col_size) = grad_reshaped (OC × K_TOTAL) @ col_flat
            # (K_TOTAL × col_size). Always routed through the MMA kernel:
            # max_matmul has no accumulate mode and would overwrite
            # grad_params on each backward call — broken for multi-call
            # BPTT unrolls (MuZero K-step, RSSM, world-model BPTT). The MMA
            # kernel uses += so multi-call accumulates. Pre-zeroed via
            # zero_grads.
            comptime dW_grid_x_nv = (Self.col_size + MMA_BLOCK_N - 1) // MMA_BLOCK_N
            comptime dW_grid_y_nv = (Self.out_channels + MMA_BLOCK_M - 1) // MMA_BLOCK_M

            @parameter
            @always_inline
            def dW_mm_wrapper(
                dW: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.out_channels, Self.col_size),
                    MutAnyOrigin,
                ],
                grad_reshaped: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.out_channels, K_TOTAL),
                    MutAnyOrigin,
                ],
                col_flat: LayoutTensor[
                    dtype,
                    Layout.row_major(K_TOTAL, Self.col_size),
                    MutAnyOrigin,
                ],
            ):
                Self.conv_matmul_dW_mma[K_TOTAL, dtype](
                    dW, grad_reshaped, col_flat
                )

            ctx.enqueue_function[dW_mm_wrapper](
                dW,
                grad_reshaped,
                col_flat,
                grid_dim=(dW_grid_x_nv, dW_grid_y_nv),
                block_dim=(MMA_BLOCK_THREADS, 1),
            )

            # ── dx: dcol (col_size × K_TOTAL) = W.T (col_size × OC) @
            #   grad_reshaped (OC × K_TOTAL).
            # Custom MMA handles W.T inline. max_matmul has no transpose_a so
            # we materialize W.T into a scratch buffer first (same pattern as
            # Conv2D phase 3a).
            var W_bwd = LayoutTensor[
                dtype,
                Layout.row_major(Self.out_channels, Self.col_size),
                MutAnyOrigin,
            ](params.ptr)

            var dcol = LayoutTensor[
                dtype,
                Layout.row_major(Self.col_size, K_TOTAL),
                MutAnyOrigin,
            ](workspace)

            comptime if Self.USE_MAX_KERNELS:
                var W_T_buf = ctx.enqueue_create_buffer[dtype](
                    Self.col_size * Self.out_channels
                )
                var W_T = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.col_size, Self.out_channels),
                    MutAnyOrigin,
                ](W_T_buf.unsafe_ptr())

                @parameter
                @always_inline
                def transpose_W_wrapper(
                    dst: LayoutTensor[
                        dtype,
                        Layout.row_major(Self.col_size, Self.out_channels),
                        MutAnyOrigin,
                    ],
                    src: LayoutTensor[
                        dtype,
                        Layout.row_major(Self.out_channels, Self.col_size),
                        MutAnyOrigin,
                    ],
                ):
                    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                    if idx < Self.out_channels * Self.col_size:
                        var oc = idx // Self.col_size
                        var ck = idx % Self.col_size
                        dst[ck, oc] = src[oc, ck]

                comptime W_T_blocks = (
                    Self.out_channels * Self.col_size + TPB - 1
                ) // TPB
                ctx.enqueue_function[transpose_W_wrapper](
                    W_T,
                    W_bwd,
                    grid_dim=(W_T_blocks,),
                    block_dim=(TPB,),
                )

                max_matmul[target="gpu"](
                    lt_to_tt(dcol),
                    lt_to_tt(W_T),
                    lt_to_tt(grad_reshaped),
                    DeviceContextPtr(ctx),
                )
            else:
                comptime dx_grid_x_nv = (K_TOTAL + MMA_BLOCK_N - 1) // MMA_BLOCK_N
                comptime dx_grid_y_nv = (Self.col_size + MMA_BLOCK_M - 1) // MMA_BLOCK_M

                @parameter
                @always_inline
                def dx_mm_wrapper(
                    dcol: LayoutTensor[
                        dtype,
                        Layout.row_major(Self.col_size, K_TOTAL),
                        MutAnyOrigin,
                    ],
                    W: LayoutTensor[
                        dtype,
                        Layout.row_major(Self.out_channels, Self.col_size),
                        MutAnyOrigin,
                    ],
                    grad_reshaped: LayoutTensor[
                        dtype,
                        Layout.row_major(Self.out_channels, K_TOTAL),
                        MutAnyOrigin,
                    ],
                ):
                    Self.conv_matmul_dx_mma[K_TOTAL, dtype](
                        dcol, W, grad_reshaped
                    )

                ctx.enqueue_function[dx_mm_wrapper](
                    dcol,
                    W_bwd,
                    grad_reshaped,
                    grid_dim=(dx_grid_x_nv, dx_grid_y_nv),
                    block_dim=(MMA_BLOCK_THREADS, 1),
                )

            # col2im gather: one thread per input element
            var total_dx = BATCH * Self.IN_DIM
            var grid_dx = (total_dx + TPB - 1) // TPB

            @parameter
            @always_inline
            def col2im_gather(
                grad_input: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.IN_DIM),
                    MutAnyOrigin,
                ],
                dcol: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.col_size, K_TOTAL),
                    MutAnyOrigin,
                ],
            ):
                # dcol (col_size, K_TOTAL=BATCH*spatial_out) contiguous → 3D
                # view. grad_input contiguous → 4D view.
                var dcol_3d = TileTensor(
                    dcol.ptr,
                    row_major[
                        Self.col_size, BATCH, Self.spatial_out
                    ](),
                )
                var grad_in_4d = TileTensor(
                    grad_input.ptr,
                    row_major[
                        BATCH, Self.in_channels, Self.in_h, Self.in_w
                    ](),
                )
                var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                if idx >= BATCH * Self.IN_DIM:
                    return
                var b = idx // Self.IN_DIM
                var in_pos = idx % Self.IN_DIM
                var c = in_pos // (Self.in_h * Self.in_w)
                var rem = in_pos % (Self.in_h * Self.in_w)
                var ih = rem // Self.in_w
                var iw = rem % Self.in_w

                var acc: Scalar[dtype] = 0
                for kh in range(Self.kernel_size):
                    for kw in range(Self.kernel_size):
                        var oh_num = ih + Self.padding - kh
                        var ow_num = iw + Self.padding - kw
                        if (
                            oh_num >= 0
                            and oh_num % Self.stride == 0
                            and ow_num >= 0
                            and ow_num % Self.stride == 0
                        ):
                            var oh = oh_num // Self.stride
                            var ow = ow_num // Self.stride
                            if oh < Self.out_h and ow < Self.out_w:
                                var s = oh * Self.out_w + ow
                                var c_k = (
                                    c * Self.kernel_size * Self.kernel_size
                                    + kh * Self.kernel_size
                                    + kw
                                )
                                acc += dcol_3d[c_k, b, s]
                grad_in_4d[b, c, ih, iw] = acc

            ctx.enqueue_function[col2im_gather](
                grad_input,
                dcol,
                grid_dim=(grid_dx,),
                block_dim=(TPB,),
            )
        else:
            # ── Apple path: dx first, then dW ──
            var total_dx = BATCH * Self.IN_DIM
            var grid_dx = (total_dx + TPB - 1) // TPB

            @parameter
            @always_inline
            def dx_wrapper(
                grad_input: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
                ],
                grad_output: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.OUT_DIM),
                    ImmutAnyOrigin,
                ],
                params: LayoutTensor[
                    dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
                ],
                cache: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.CACHE_SIZE),
                    ImmutAnyOrigin,
                ],
            ):
                Self.backward_dx_kernel_impl[BATCH, dtype](
                    grad_input, grad_output, params, cache
                )

            ctx.enqueue_function[dx_wrapper](
                grad_input,
                grad_output_immut,
                params_immut,
                cache_immut,
                grid_dim=(grid_dx,),
                block_dim=(TPB,),
            )

            comptime dW_grid_x = (Self.col_size + 31) // 32
            comptime dW_grid_y = (Self.out_channels + 31) // 32

            @parameter
            @always_inline
            def dW_wrapper(
                dW: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.out_channels, Self.col_size),
                    MutAnyOrigin,
                ],
                cache: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.CACHE_SIZE),
                    ImmutAnyOrigin,
                ],
                grad_output: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.OUT_DIM),
                    ImmutAnyOrigin,
                ],
            ):
                Self.backward_dW_kernel_2x2[BATCH, dtype](dW, cache, grad_output)

            ctx.enqueue_function[dW_wrapper](
                dW,
                cache_immut,
                grad_output_immut,
                grid_dim=(dW_grid_x, dW_grid_y),
                block_dim=(MMA_BLOCK_THREADS, 1),
            )

        # Kernel 3: db (simple reduction with fused activation gradient)
        var db = LayoutTensor[
            dtype, Layout.row_major(Self.out_channels), MutAnyOrigin
        ](grad_params.ptr + Self.out_channels * Self.col_size)

        # Grid: one block per output channel, TPB threads reduce across BATCH
        @parameter
        @always_inline
        def db_wrapper(
            db: LayoutTensor[
                dtype, Layout.row_major(Self.out_channels), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
            ],
        ):
            Self.backward_db_kernel[BATCH, dtype](db, grad_output, cache)

        ctx.enqueue_function[db_wrapper](
            db,
            grad_output_immut,
            cache_immut,
            grid_dim=(Self.out_channels,),
            block_dim=(TPB,),
        )
