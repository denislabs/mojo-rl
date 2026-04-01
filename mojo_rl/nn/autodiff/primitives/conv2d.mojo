from ...constants import (
    dtype,
    TILE,
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
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives import block, lane_id
from std.sys import is_nvidia_gpu, has_nvidia_gpu_accelerator
from std.gpu.compute.mma import mma
from linalg.bmm import batched_matmul as max_matmul
from layout.tile_tensor import lt_to_tt


struct Conv2D[
    in_channels: Int,
    out_channels: Int,
    kernel_size: Int,
    stride: Int,
    padding: Int,
    in_h: Int,
    in_w: Int,
](DiffOp):
    """2D Convolution via im2col reduction to MatMul.

    Input:  (BATCH, in_channels * in_h * in_w) — flattened spatial
    Output: (BATCH, out_channels * out_h * out_w) — flattened spatial

    Parameters: W (out_channels, in_channels * kernel_size * kernel_size) + bias (out_channels)
    Cache: im2col buffer (col_size * out_h * out_w) for backward

    Forward:
        1. im2col: reshape input patches into columns
        2. output = W @ col + bias (broadcast per output spatial position)

    Backward:
        1. dW += grad_output_reshaped @ col.T
        2. db += sum(grad_output_reshaped, axis=spatial)
        3. dcol = W.T @ grad_output_reshaped
        4. col2im: scatter dcol back to grad_input
    """

    comptime out_h: Int = (
        Self.in_h + 2 * Self.padding - Self.kernel_size
    ) // Self.stride + 1
    comptime out_w: Int = (
        Self.in_w + 2 * Self.padding - Self.kernel_size
    ) // Self.stride + 1
    comptime col_size: Int = Self.in_channels * Self.kernel_size * Self.kernel_size
    comptime spatial_out: Int = Self.out_h * Self.out_w

    comptime OP_ID: Int = OpID.CONV2D._value
    comptime IN_DIM: Int = Self.in_channels * Self.in_h * Self.in_w
    comptime OUT_DIM: Int = Self.out_channels * Self.out_h * Self.out_w
    comptime PARAM_SIZE: Int = Self.out_channels * Self.col_size + Self.out_channels
    comptime CACHE_SIZE: Int = Self.col_size * Self.spatial_out
    # Workspace: max(forward out_temp, backward dcol) + w_t
    # Forward: out_temp (OUT_DIM), Backward dx: dcol (CACHE_SIZE = col_size*spatial_out)
    comptime OP_WORKSPACE_PER_SAMPLE: Int = Self.CACHE_SIZE + Self.col_size * Self.out_channels

    def __init__(out self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    # =========================================================================
    # CPU eval / vjp
    # =========================================================================

    @staticmethod
    def eval[
        BATCH: Int
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
        """Forward: direct convolution with im2col cached for backward."""
        # W: (out_channels, col_size) stored flat in params
        # bias: (out_channels) stored after W

        for b in range(BATCH):
            # Build im2col into cache directly
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

            # output[b] = W @ col + bias
            # W: (out_channels, col_size), col: (col_size, spatial_out)
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
                    var bias_val = rebind[Scalar[dtype]](
                        params[Self.out_channels * Self.col_size + oc]
                    )
                    acc += bias_val
                    output[b, oc * Self.spatial_out + s] = acc

    @staticmethod
    def vjp[
        BATCH: Int
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
        """Backward: compute dW, db, and grad_input via col2im."""
        comptime W_SIZE = Self.out_channels * Self.col_size

        for b in range(BATCH):
            # 1. dW += grad_output_reshaped @ col.T
            for oc in range(Self.out_channels):
                for k in range(Self.col_size):
                    var acc: Scalar[dtype] = 0
                    for s in range(Self.spatial_out):
                        var go_val = rebind[Scalar[dtype]](
                            grad_output[b, oc * Self.spatial_out + s]
                        )
                        var col_val = rebind[Scalar[dtype]](
                            cache[b, s * Self.col_size + k]
                        )
                        acc += go_val * col_val
                    var cur = rebind[Scalar[dtype]](
                        grad_params[oc * Self.col_size + k]
                    )
                    grad_params[oc * Self.col_size + k] = cur + acc

            # 2. db += sum(grad_output, over spatial dims)
            for oc in range(Self.out_channels):
                var acc: Scalar[dtype] = 0
                for s in range(Self.spatial_out):
                    acc += rebind[Scalar[dtype]](
                        grad_output[b, oc * Self.spatial_out + s]
                    )
                var cur = rebind[Scalar[dtype]](grad_params[W_SIZE + oc])
                grad_params[W_SIZE + oc] = cur + acc

            # 3. grad_input via col2im of W.T @ grad_output
            # Zero grad_input for this batch first
            for i in range(Self.IN_DIM):
                grad_input[b, i] = 0

            # For each output position, scatter gradient back through kernel
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
                                    # dcol[c_k, s] = sum_oc(W[oc, c_k] * grad_output[b, oc*spatial_out + s])
                                    var dcol_val: Scalar[dtype] = 0
                                    for oc in range(Self.out_channels):
                                        var w_val = rebind[Scalar[dtype]](
                                            params[oc * Self.col_size + c_k]
                                        )
                                        var go_val = rebind[Scalar[dtype]](
                                            grad_output[
                                                b, oc * Self.spatial_out + s
                                            ]
                                        )
                                        dcol_val += w_val * go_val
                                    var cur = rebind[Scalar[dtype]](
                                        grad_input[b, in_idx]
                                    )
                                    grad_input[b, in_idx] = cur + dcol_val

    # =========================================================================
    # GPU kernels — naive (kept for backward_dx which has sufficient parallelism)
    # =========================================================================

    @always_inline
    @staticmethod
    def backward_dx_kernel_impl[
        BATCH: Int
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
    ):
        """GPU backward for grad_input. One thread per (batch, input_element).
        """
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
                        var c_k = (
                            c * Self.kernel_size * Self.kernel_size
                            + kh * Self.kernel_size
                            + kw
                        )
                        for oc in range(Self.out_channels):
                            acc += rebind[Scalar[dtype]](
                                params[oc * Self.col_size + c_k]
                            ) * rebind[Scalar[dtype]](
                                grad_output[b, oc * Self.spatial_out + s]
                            )

        grad_input[b, in_pos] = acc

    # =========================================================================
    # GPU kernels — tiled forward (Apple 2x2 / NVIDIA MMA)
    # =========================================================================

    @always_inline
    @staticmethod
    def eval_kernel_2x2[
        BATCH: Int
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
        """Tiled forward: output = W @ im2col(input) + bias with implicit im2col.

        Matmul: (out_channels, col_size) @ (col_size, spatial_out) per batch.
        Grid: (ceil(spatial_out/32), ceil(out_channels/32), BATCH)
        Block: (256, 1)
        """
        comptime BT = 32
        comptime SK = 16
        comptime W_SIZE = Self.out_channels * Self.col_size
        comptime KS2 = Self.kernel_size * Self.kernel_size

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
                a_smem[a_r0, a_c0] = params[ga_oc0 * Self.col_size + ga_k0]
            else:
                a_smem[a_r0, a_c0] = 0

            var ga_oc1 = block_oc + a_r1
            var ga_k1 = k_off + a_c1
            if ga_oc1 < Self.out_channels and ga_k1 < Self.col_size:
                a_smem[a_r1, a_c1] = params[ga_oc1 * Self.col_size + ga_k1]
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
                    val0 = rebind[Scalar[dtype]](
                        input[
                            batch,
                            ch0 * Self.in_h * Self.in_w + ih0 * Self.in_w + iw0,
                        ]
                    )
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
                    val1 = rebind[Scalar[dtype]](
                        input[
                            batch,
                            ch1 * Self.in_h * Self.in_w + ih1 * Self.in_w + iw1,
                        ]
                    )
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

        # Store with bias
        var oc0 = block_oc + sub_r * 2
        var s0 = block_s + sub_c * 2

        if oc0 < Self.out_channels and s0 < Self.spatial_out:
            output[batch, oc0 * Self.spatial_out + s0] = acc00 + rebind[
                Scalar[dtype]
            ](params[W_SIZE + oc0])
        if oc0 < Self.out_channels and s0 + 1 < Self.spatial_out:
            output[batch, oc0 * Self.spatial_out + s0 + 1] = acc01 + rebind[
                Scalar[dtype]
            ](params[W_SIZE + oc0])
        if oc0 + 1 < Self.out_channels and s0 < Self.spatial_out:
            output[batch, (oc0 + 1) * Self.spatial_out + s0] = acc10 + rebind[
                Scalar[dtype]
            ](params[W_SIZE + oc0 + 1])
        if oc0 + 1 < Self.out_channels and s0 + 1 < Self.spatial_out:
            output[
                batch, (oc0 + 1) * Self.spatial_out + s0 + 1
            ] = acc11 + rebind[Scalar[dtype]](params[W_SIZE + oc0 + 1])

    @always_inline
    @staticmethod
    def eval_kernel_mma[
        BATCH: Int
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
        """MMA forward: output = W @ im2col(input) + bias with tensor cores.

        Grid: (ceil(spatial_out/32), ceil(out_channels/32), BATCH)
        Block: (256, 1)
        """
        comptime if is_nvidia_gpu():
            comptime W_SIZE = Self.out_channels * Self.col_size
            comptime KS2 = Self.kernel_size * Self.kernel_size

            var tid = Int(thread_idx.x)
            var warp_id = tid // 32
            var warp_m = warp_id // MMA_WARPS_N
            var warp_n = warp_id % MMA_WARPS_N

            var block_oc = Int(block_idx.y) * MMA_BLOCK_M
            var block_s = Int(block_idx.x) * MMA_BLOCK_N
            var batch = Int(block_idx.z)

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

                # Load A: W[block_oc + a_r, k_off + a_c]
                var a_r = tid // MMA_K
                var a_c = tid % MMA_K
                var ga_oc = block_oc + a_r
                var ga_k = k_off + a_c
                if ga_oc < Self.out_channels and ga_k < Self.col_size:
                    a_smem[a_r, a_c] = params[ga_oc * Self.col_size + ga_k]
                else:
                    a_smem[a_r, a_c] = 0

                # Load B: im2col[k_off + br, block_s + bc]
                var br = tid // MMA_BLOCK_N
                var bc = tid % MMA_BLOCK_N
                var k_idx = k_off + br
                var s_idx = block_s + bc
                var val: Scalar[dtype] = 0
                if k_idx < Self.col_size and s_idx < Self.spatial_out:
                    var ch = k_idx // KS2
                    var rem_k = k_idx % KS2
                    var kh = rem_k // Self.kernel_size
                    var kw = rem_k % Self.kernel_size
                    var oh = s_idx // Self.out_w
                    var ow = s_idx % Self.out_w
                    var ih = oh * Self.stride - Self.padding + kh
                    var iw = ow * Self.stride - Self.padding + kw
                    if (
                        ih >= 0
                        and ih < Self.in_h
                        and iw >= 0
                        and iw < Self.in_w
                    ):
                        val = rebind[Scalar[dtype]](
                            input[
                                batch,
                                ch * Self.in_h * Self.in_w
                                + ih * Self.in_w
                                + iw,
                            ]
                        )
                b_smem[br, bc] = val
                if (
                    Int(block_idx.y) == 0
                    and k_idx < Self.col_size
                    and s_idx < Self.spatial_out
                ):
                    cache[batch, s_idx * Self.col_size + k_idx] = val

                barrier()

                var warp_row = warp_m * MMA_M
                var a_frag = SIMD[DType.float32, 4](
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id), Int(group_lane)]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id) + 8, Int(group_lane)]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id), Int(group_lane) + 4]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[
                            warp_row + Int(group_id) + 8, Int(group_lane) + 4
                        ]
                    ),
                )
                var warp_col = warp_n * MMA_N
                var b_frag = SIMD[DType.float32, 2](
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane), warp_col + Int(group_id)]
                    ),
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane) + 4, warp_col + Int(group_id)]
                    ),
                )

                mma(acc, a_frag, b_frag, acc)
                barrier()

            # Store with bias
            var r0 = block_oc + warp_m * MMA_M + Int(group_id)
            var r1 = r0 + 8
            var c0 = block_s + warp_n * MMA_N + Int(group_lane * 2)
            var c1 = c0 + 1

            if r0 < Self.out_channels and c0 < Self.spatial_out:
                output[batch, r0 * Self.spatial_out + c0] = rebind[
                    Scalar[dtype]
                ](acc[0]) + rebind[Scalar[dtype]](params[W_SIZE + r0])
            if r0 < Self.out_channels and c1 < Self.spatial_out:
                output[batch, r0 * Self.spatial_out + c1] = rebind[
                    Scalar[dtype]
                ](acc[1]) + rebind[Scalar[dtype]](params[W_SIZE + r0])
            if r1 < Self.out_channels and c0 < Self.spatial_out:
                output[batch, r1 * Self.spatial_out + c0] = rebind[
                    Scalar[dtype]
                ](acc[2]) + rebind[Scalar[dtype]](params[W_SIZE + r1])
            if r1 < Self.out_channels and c1 < Self.spatial_out:
                output[batch, r1 * Self.spatial_out + c1] = rebind[
                    Scalar[dtype]
                ](acc[3]) + rebind[Scalar[dtype]](params[W_SIZE + r1])

    # =========================================================================
    # GPU kernels — tiled backward dW and db
    # =========================================================================

    @always_inline
    @staticmethod
    def backward_dW_kernel_2x2[
        BATCH: Int
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
        """Tiled backward: dW = sum_b[grad_out_b @ col_b.T].

        Matmul: dW(oc, col_k) = A(oc, K) @ B(K, col_k)
        where K = BATCH * spatial_out, A = grad_output reshaped, B = im2col.T.
        Grid: (ceil(col_size/32), ceil(out_channels/32))
        Block: (256, 1)
        """
        comptime BT = 32
        comptime SK = 16
        comptime K_TOTAL = BATCH * Self.spatial_out

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

            # Load A: grad_output reshaped — A[oc, ki]
            # ki = k_off + col, b_idx = ki // spatial_out, s_idx = ki % spatial_out
            # A[oc, ki] = grad_output[b_idx, oc * spatial_out + s_idx]
            var a_r0 = tid // SK
            var a_c0 = tid % SK
            var a_r1 = (tid + 256) // SK
            var a_c1 = (tid + 256) % SK

            var ki0 = k_off + a_c0
            var g_oc0 = block_oc + a_r0
            if g_oc0 < Self.out_channels and ki0 < K_TOTAL:
                var b0 = ki0 // Self.spatial_out
                var s0 = ki0 % Self.spatial_out
                a_smem[a_r0, a_c0] = grad_output[
                    b0, g_oc0 * Self.spatial_out + s0
                ]
            else:
                a_smem[a_r0, a_c0] = 0

            var ki1 = k_off + a_c1
            var g_oc1 = block_oc + a_r1
            if g_oc1 < Self.out_channels and ki1 < K_TOTAL:
                var b1 = ki1 // Self.spatial_out
                var s1 = ki1 % Self.spatial_out
                a_smem[a_r1, a_c1] = grad_output[
                    b1, g_oc1 * Self.spatial_out + s1
                ]
            else:
                a_smem[a_r1, a_c1] = 0

            # Load B: im2col cache — B[ki, col_k]
            # B[ki, col_k] = cache[b_idx, col_k * spatial_out + s_idx]
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
        if oc0 < Self.out_channels and k0 < Self.col_size:
            dW[oc0, k0] = acc00
        if oc0 < Self.out_channels and k0 + 1 < Self.col_size:
            dW[oc0, k0 + 1] = acc01
        if oc0 + 1 < Self.out_channels and k0 < Self.col_size:
            dW[oc0 + 1, k0] = acc10
        if oc0 + 1 < Self.out_channels and k0 + 1 < Self.col_size:
            dW[oc0 + 1, k0 + 1] = acc11

    @always_inline
    @staticmethod
    def backward_dW_kernel_mma[
        BATCH: Int
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
        """MMA backward: dW = sum_b[grad_out_b @ col_b.T] with tensor cores.

        Grid: (ceil(col_size/32), ceil(out_channels/32))
        Block: (256, 1)
        """
        comptime if is_nvidia_gpu():
            comptime K_TOTAL = BATCH * Self.spatial_out

            var tid = Int(thread_idx.x)
            var warp_id = tid // 32
            var warp_m = warp_id // MMA_WARPS_N
            var warp_n = warp_id % MMA_WARPS_N

            var block_oc = Int(block_idx.y) * MMA_BLOCK_M
            var block_k = Int(block_idx.x) * MMA_BLOCK_N

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

                # Load A: grad_output reshaped
                var a_r = tid // MMA_K
                var a_c = tid % MMA_K
                var g_oc = block_oc + a_r
                var ki = k_off + a_c
                if g_oc < Self.out_channels and ki < K_TOTAL:
                    var b_idx = ki // Self.spatial_out
                    var s_idx = ki % Self.spatial_out
                    a_smem[a_r, a_c] = grad_output[
                        b_idx, g_oc * Self.spatial_out + s_idx
                    ]
                else:
                    a_smem[a_r, a_c] = 0

                # Load B: im2col cache
                var br = tid // MMA_BLOCK_N
                var bc = tid % MMA_BLOCK_N
                var bki = k_off + br
                var gk = block_k + bc
                if gk < Self.col_size and bki < K_TOTAL:
                    var b_idx = bki // Self.spatial_out
                    var s_idx = bki % Self.spatial_out
                    b_smem[br, bc] = cache[b_idx, s_idx * Self.col_size + gk]
                else:
                    b_smem[br, bc] = 0

                barrier()

                var warp_row = warp_m * MMA_M
                var a_frag = SIMD[DType.float32, 4](
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id), Int(group_lane)]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id) + 8, Int(group_lane)]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id), Int(group_lane) + 4]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[
                            warp_row + Int(group_id) + 8, Int(group_lane) + 4
                        ]
                    ),
                )
                var warp_col = warp_n * MMA_N
                var b_frag = SIMD[DType.float32, 2](
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane), warp_col + Int(group_id)]
                    ),
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane) + 4, warp_col + Int(group_id)]
                    ),
                )

                mma(acc, a_frag, b_frag, acc)
                barrier()

            var r0 = block_oc + warp_m * MMA_M + Int(group_id)
            var r1 = r0 + 8
            var c0 = block_k + warp_n * MMA_N + Int(group_lane * 2)
            var c1 = c0 + 1

            if r0 < Self.out_channels and c0 < Self.col_size:
                dW[r0, c0] = rebind[Scalar[dtype]](acc[0])
            if r0 < Self.out_channels and c1 < Self.col_size:
                dW[r0, c1] = rebind[Scalar[dtype]](acc[1])
            if r1 < Self.out_channels and c0 < Self.col_size:
                dW[r1, c0] = rebind[Scalar[dtype]](acc[2])
            if r1 < Self.out_channels and c1 < Self.col_size:
                dW[r1, c1] = rebind[Scalar[dtype]](acc[3])

    @always_inline
    @staticmethod
    def backward_db_kernel[
        BATCH: Int
    ](
        db: LayoutTensor[
            dtype, Layout.row_major(Self.out_channels), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
    ):
        """Formula: db = sum(grad_output, axis=batch+spatial).

        Parallelized: one block per oc, TPB threads reduce across BATCH.

        Grid: (out_channels,)
        Block: (TPB,)
        """
        var oc = Int(block_idx.x)
        if oc >= Self.out_channels:
            return
        var tid = Int(thread_idx.x)

        var acc: Scalar[dtype] = 0
        for b in range(tid, BATCH, TPB):
            for s in range(Self.spatial_out):
                acc += rebind[Scalar[dtype]](
                    grad_output[b, oc * Self.spatial_out + s]
                )

        var smem = LayoutTensor[
            dtype,
            Layout.row_major(TPB),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()
        smem[tid] = acc
        barrier()

        var conv_stride = TPB // 2
        while conv_stride > 0:
            if tid < conv_stride:
                smem[tid] = rebind[Scalar[dtype]](smem[tid]) + rebind[
                    Scalar[dtype]
                ](smem[tid + conv_stride])
            barrier()
            conv_stride //= 2

        if tid == 0:
            db[oc] = smem[0]

    # =========================================================================
    # GPU launchers (tiled forward + tiled dW/db + naive dx)
    # =========================================================================

    @staticmethod
    def eval_gpu[
        BATCH: Int
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
            # NVIDIA: im2col → max_matmul → bias add
            comptime K_TOTAL = BATCH * Self.spatial_out
            comptime KS2 = Self.kernel_size * Self.kernel_size

            # 1. Explicit im2col: input → cache (s*col_size+k layout)
            # Memory-bandwidth bound (~21% of peak on 4090). Indirect addressing
            # pattern limits further optimization without algorithmic change.
            comptime im2col_elems = BATCH * Self.CACHE_SIZE
            comptime im2col_blocks = (im2col_elems + TPB - 1) // TPB

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
                var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                if idx >= im2col_elems:
                    return
                var b = idx // Self.CACHE_SIZE
                var pos = idx % Self.CACHE_SIZE
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
                    val = rebind[Scalar[dtype]](
                        input[
                            b, ch * Self.in_h * Self.in_w + ih * Self.in_w + iw
                        ]
                    )
                cache_out[b, pos] = val

            ctx.enqueue_function[im2col_wrapper, im2col_wrapper](
                cache,
                input_immut,
                grid_dim=(im2col_blocks,),
                block_dim=(TPB,),
            )

            # Cache flattens directly to (K_TOTAL, col_size) since CACHE_SIZE == CONV_CACHE
            var col_flat = LayoutTensor[
                dtype, Layout.row_major(K_TOTAL, Self.col_size), MutAnyOrigin
            ](cache.ptr)

            # 2. max_matmul with transpose_b: out = col_flat @ W.T
            #    W is stored as (OC, col_size), transpose_b transposes it internally
            var W_mat = LayoutTensor[
                dtype,
                Layout.row_major(Self.out_channels, Self.col_size),
                MutAnyOrigin,
            ](params.ptr)

            var out_temp = LayoutTensor[
                dtype,
                Layout.row_major(K_TOTAL, Self.out_channels),
                MutAnyOrigin,
            ](workspace)

            max_matmul[target="gpu", transpose_b=True](lt_to_tt(out_temp), lt_to_tt(col_flat), lt_to_tt(W_mat), context=ctx)

            # 4. Transpose output + bias: (K_TOTAL, OC) → (BATCH, OC*S)
            # out_temp[b*S+s, oc] → output[b, oc*S+s] + bias[oc]
            comptime out_elems = BATCH * Self.OUT_DIM
            comptime out_blocks = (out_elems + TPB - 1) // TPB

            var b_ptr = LayoutTensor[
                dtype, Layout.row_major(Self.out_channels), ImmutAnyOrigin
            ](params.ptr + Self.out_channels * Self.col_size)

            @always_inline
            def transpose_output_bias_wrapper(
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
                bias: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.out_channels),
                    ImmutAnyOrigin,
                ],
            ):
                var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                if idx >= out_elems:
                    return
                var b = idx // Self.OUT_DIM
                var out_pos = idx % Self.OUT_DIM
                var oc = out_pos // Self.spatial_out
                var s = out_pos % Self.spatial_out
                output[b, out_pos] = rebind[Scalar[dtype]](
                    out_temp[b * Self.spatial_out + s, oc]
                ) + rebind[Scalar[dtype]](bias[oc])

            ctx.enqueue_function[
                transpose_output_bias_wrapper,
                transpose_output_bias_wrapper,
            ](
                output,
                out_temp,
                b_ptr,
                grid_dim=(out_blocks,),
                block_dim=(TPB,),
            )
        else:
            # Apple: fused im2col + tiled matmul kernel
            comptime grid_x = (Self.spatial_out + 31) // 32
            comptime grid_y = (Self.out_channels + 31) // 32

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
                Self.eval_kernel_2x2[BATCH](output, input, params, cache)

            ctx.enqueue_function[wrapper, wrapper](
                output,
                input_immut,
                params_immut,
                cache,
                grid_dim=(grid_x, grid_y, BATCH),
                block_dim=(MMA_BLOCK_THREADS, 1),
            )

    @staticmethod
    def vjp_gpu[
        BATCH: Int
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
            # ── dW FIRST (before dx) so we can reuse grad_input as workspace ──
            # dW = grad_reshaped @ col_reshaped
            # grad_output (BATCH, OC*S) → grad_reshaped (OC, BATCH*S)
            # cache (BATCH, col_size*S) → col_reshaped (BATCH*S, col_size)
            comptime K_TOTAL = BATCH * Self.spatial_out

            # Reuse grad_input as workspace for grad_reshaped
            # (BATCH * IN_DIM >= OC * K_TOTAL always holds for conv layers)
            var grad_reshaped = LayoutTensor[
                dtype,
                Layout.row_major(Self.out_channels, K_TOTAL),
                MutAnyOrigin,
            ](grad_input.ptr)

            # Cache is now (batch, s*col_size+k) — flattens to (K_TOTAL, col_size)
            var col_flat = LayoutTensor[
                dtype,
                Layout.row_major(K_TOTAL, Self.col_size),
                MutAnyOrigin,
            ](cache.ptr)

            # Transpose grad: (BATCH, OC*S) → (OC, BATCH*S)
            comptime grad_elems = Self.out_channels * K_TOTAL
            comptime grad_blocks = (grad_elems + TPB - 1) // TPB

            @always_inline
            def transpose_grad_wrapper(
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
            ):
                var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                if idx >= grad_elems:
                    return
                var oc = idx // K_TOTAL
                var bs = idx % K_TOTAL
                var b = bs // Self.spatial_out
                var s = bs % Self.spatial_out
                dst[oc, bs] = src[b, oc * Self.spatial_out + s]

            ctx.enqueue_function[
                transpose_grad_wrapper, transpose_grad_wrapper
            ](
                grad_reshaped,
                grad_output_immut,
                grid_dim=(grad_blocks,),
                block_dim=(TPB,),
            )

            # Zero-alloc dW: dW = grad_reshaped @ col_flat
            max_matmul[target="gpu"](lt_to_tt(dW), lt_to_tt(grad_reshaped), lt_to_tt(col_flat), context=ctx)

            # ── dx via matmul + col2im gather ──
            # dcol = W.T @ grad_reshaped, then col2im gather
            # grad_reshaped (OC, K_TOTAL) is still in grad_input.ptr

            # dcol = W.T @ grad_reshaped  →  dcol_t = grad_reshaped.T @ W
            # grad_reshaped is (OC, K_TOTAL) row-major = (K_TOTAL, OC) col-major
            # Reinterpret as (K_TOTAL, OC) row-major via transpose_b on W:
            # dcol_t(K_TOTAL, col_size) = grad_t(K_TOTAL, OC) @ W(OC, col_size)
            # But we can't reinterpret grad without transpose_a (unsupported).
            # So we keep the explicit transpose for backward.
            # TODO: Remove when max_matmul supports transpose_a
            comptime w_t_bwd_offset = BATCH * Self.CACHE_SIZE
            comptime w_elems_bwd = Self.out_channels * Self.col_size
            comptime w_blocks_bwd = (w_elems_bwd + TPB - 1) // TPB
            var w_t_bwd = LayoutTensor[
                dtype,
                Layout.row_major(Self.col_size, Self.out_channels),
                MutAnyOrigin,
            ](workspace + w_t_bwd_offset)

            @always_inline
            def transpose_w_bwd(
                dst: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.col_size, Self.out_channels),
                    MutAnyOrigin,
                ],
                src: LayoutTensor[
                    dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
                ],
            ):
                var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                if idx >= w_elems_bwd:
                    return
                var k = idx // Self.out_channels
                var oc = idx % Self.out_channels
                dst[k, oc] = src[oc * Self.col_size + k]

            ctx.enqueue_function[transpose_w_bwd, transpose_w_bwd](
                w_t_bwd,
                params_immut,
                grid_dim=(w_blocks_bwd,),
                block_dim=(TPB,),
            )

            var dcol = LayoutTensor[
                dtype,
                Layout.row_major(Self.col_size, K_TOTAL),
                MutAnyOrigin,
            ](workspace)

            max_matmul[target="gpu"](lt_to_tt(dcol), lt_to_tt(w_t_bwd), lt_to_tt(grad_reshaped), context=ctx)

            # col2im gather: one thread per input element
            var total_dx = BATCH * Self.IN_DIM
            var grid_dx = (total_dx + TPB - 1) // TPB

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
                                acc += rebind[Scalar[dtype]](
                                    dcol[c_k, b * Self.spatial_out + s]
                                )
                grad_input[b, in_pos] = acc

            ctx.enqueue_function[col2im_gather, col2im_gather](
                grad_input,
                dcol,
                grid_dim=(grid_dx,),
                block_dim=(TPB,),
            )
        else:
            # ── Apple path: dx first, then dW ──
            var total_dx = BATCH * Self.IN_DIM
            var grid_dx = (total_dx + TPB - 1) // TPB

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
            ):
                Self.backward_dx_kernel_impl[BATCH](
                    grad_input, grad_output, params
                )

            ctx.enqueue_function[dx_wrapper, dx_wrapper](
                grad_input,
                grad_output_immut,
                params_immut,
                grid_dim=(grid_dx,),
                block_dim=(TPB,),
            )

            comptime dW_grid_x = (Self.col_size + 31) // 32
            comptime dW_grid_y = (Self.out_channels + 31) // 32

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
                Self.backward_dW_kernel_2x2[BATCH](dW, cache, grad_output)

            ctx.enqueue_function[dW_wrapper, dW_wrapper](
                dW,
                cache_immut,
                grad_output_immut,
                grid_dim=(dW_grid_x, dW_grid_y),
                block_dim=(MMA_BLOCK_THREADS, 1),
            )

        # Kernel 3: db (simple reduction)
        var db = LayoutTensor[
            dtype, Layout.row_major(Self.out_channels), MutAnyOrigin
        ](grad_params.ptr + Self.out_channels * Self.col_size)

        # Grid: one block per output channel, TPB threads reduce across BATCH
        @always_inline
        def db_wrapper(
            db: LayoutTensor[
                dtype, Layout.row_major(Self.out_channels), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            Self.backward_db_kernel[BATCH](db, grad_output)

        ctx.enqueue_function[db_wrapper, db_wrapper](
            db,
            grad_output_immut,
            grid_dim=(Self.out_channels,),
            block_dim=(TPB,),
        )
