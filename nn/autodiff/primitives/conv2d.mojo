from ...constants import dtype, TILE, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace


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

    comptime out_h: Int = (Self.in_h + 2 * Self.padding - Self.kernel_size) // Self.stride + 1
    comptime out_w: Int = (Self.in_w + 2 * Self.padding - Self.kernel_size) // Self.stride + 1
    comptime col_size: Int = Self.in_channels * Self.kernel_size * Self.kernel_size
    comptime spatial_out: Int = Self.out_h * Self.out_w

    comptime OP_ID: Int = OpID.CONV2D._value
    comptime IN_DIM: Int = Self.in_channels * Self.in_h * Self.in_w
    comptime OUT_DIM: Int = Self.out_channels * Self.out_h * Self.out_w
    comptime PARAM_SIZE: Int = Self.out_channels * Self.col_size + Self.out_channels
    comptime CACHE_SIZE: Int = Self.col_size * Self.spatial_out

    fn __init__(out self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

    fn __init__(out self, *, copy: Self):
        pass

    # =========================================================================
    # CPU eval / vjp
    # =========================================================================

    @staticmethod
    fn eval[
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
                                var c_k = c * Self.kernel_size * Self.kernel_size + kh * Self.kernel_size + kw
                                var col_idx = c_k * Self.spatial_out + s
                                if ih >= 0 and ih < Self.in_h and iw >= 0 and iw < Self.in_w:
                                    var in_idx = c * Self.in_h * Self.in_w + ih * Self.in_w + iw
                                    cache[b, col_idx] = input[b, in_idx]
                                else:
                                    cache[b, col_idx] = 0

            # output[b] = W @ col + bias
            # W: (out_channels, col_size), col: (col_size, spatial_out)
            for oc in range(Self.out_channels):
                for s in range(Self.spatial_out):
                    var acc: Scalar[dtype] = 0
                    for k in range(Self.col_size):
                        var w_val = rebind[Scalar[dtype]](params[oc * Self.col_size + k])
                        var c_val = rebind[Scalar[dtype]](cache[b, k * Self.spatial_out + s])
                        acc += w_val * c_val
                    # Add bias
                    var bias_val = rebind[Scalar[dtype]](params[Self.out_channels * Self.col_size + oc])
                    acc += bias_val
                    output[b, oc * Self.spatial_out + s] = acc

    @staticmethod
    fn vjp[
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
                        var go_val = rebind[Scalar[dtype]](grad_output[b, oc * Self.spatial_out + s])
                        var col_val = rebind[Scalar[dtype]](cache[b, k * Self.spatial_out + s])
                        acc += go_val * col_val
                    var cur = rebind[Scalar[dtype]](grad_params[oc * Self.col_size + k])
                    grad_params[oc * Self.col_size + k] = cur + acc

            # 2. db += sum(grad_output, over spatial dims)
            for oc in range(Self.out_channels):
                var acc: Scalar[dtype] = 0
                for s in range(Self.spatial_out):
                    acc += rebind[Scalar[dtype]](grad_output[b, oc * Self.spatial_out + s])
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
                                if ih >= 0 and ih < Self.in_h and iw >= 0 and iw < Self.in_w:
                                    var in_idx = c * Self.in_h * Self.in_w + ih * Self.in_w + iw
                                    var c_k = c * Self.kernel_size * Self.kernel_size + kh * Self.kernel_size + kw
                                    # dcol[c_k, s] = sum_oc(W[oc, c_k] * grad_output[b, oc*spatial_out + s])
                                    var dcol_val: Scalar[dtype] = 0
                                    for oc in range(Self.out_channels):
                                        var w_val = rebind[Scalar[dtype]](params[oc * Self.col_size + c_k])
                                        var go_val = rebind[Scalar[dtype]](grad_output[b, oc * Self.spatial_out + s])
                                        dcol_val += w_val * go_val
                                    var cur = rebind[Scalar[dtype]](grad_input[b, in_idx])
                                    grad_input[b, in_idx] = cur + dcol_val

    # =========================================================================
    # GPU kernels
    # =========================================================================

    @always_inline
    @staticmethod
    fn eval_kernel_impl[
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
        """GPU forward: one thread per (batch, out_channel, spatial_pos)."""
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        var total = BATCH * Self.out_channels * Self.spatial_out
        if idx >= total:
            return

        var b = idx // (Self.out_channels * Self.spatial_out)
        var rem = idx % (Self.out_channels * Self.spatial_out)
        var oc = rem // Self.spatial_out
        var s = rem % Self.spatial_out

        var oh = s // Self.out_w
        var ow = s % Self.out_w

        var acc: Scalar[dtype] = 0
        for c in range(Self.in_channels):
            for kh in range(Self.kernel_size):
                for kw in range(Self.kernel_size):
                    var ih = oh * Self.stride - Self.padding + kh
                    var iw = ow * Self.stride - Self.padding + kw
                    var c_k = c * Self.kernel_size * Self.kernel_size + kh * Self.kernel_size + kw
                    var val: Scalar[dtype] = 0
                    if ih >= 0 and ih < Self.in_h and iw >= 0 and iw < Self.in_w:
                        var in_idx = c * Self.in_h * Self.in_w + ih * Self.in_w + iw
                        val = rebind[Scalar[dtype]](input[b, in_idx])
                    acc += rebind[Scalar[dtype]](params[oc * Self.col_size + c_k]) * val
                    # Cache im2col value
                    var cache_idx = c_k * Self.spatial_out + s
                    cache[b, cache_idx] = val

        acc += rebind[Scalar[dtype]](params[Self.out_channels * Self.col_size + oc])
        output[b, oc * Self.spatial_out + s] = acc

    @always_inline
    @staticmethod
    fn backward_dx_kernel_impl[
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
        """GPU backward for grad_input. One thread per (batch, input_element)."""
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
                if oh_num >= 0 and oh_num % Self.stride == 0 and ow_num >= 0 and ow_num % Self.stride == 0:
                    var oh = oh_num // Self.stride
                    var ow = ow_num // Self.stride
                    if oh < Self.out_h and ow < Self.out_w:
                        var s = oh * Self.out_w + ow
                        var c_k = c * Self.kernel_size * Self.kernel_size + kh * Self.kernel_size + kw
                        for oc in range(Self.out_channels):
                            acc += rebind[Scalar[dtype]](params[oc * Self.col_size + c_k]) * rebind[Scalar[dtype]](grad_output[b, oc * Self.spatial_out + s])

        grad_input[b, in_pos] = acc

    @always_inline
    @staticmethod
    fn backward_dW_kernel_impl[
        BATCH: Int
    ](
        grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
    ):
        """GPU backward for dW and db. One thread per (oc, col_pos)."""
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        var total = Self.out_channels * Self.col_size
        if idx >= total:
            return

        var oc = idx // Self.col_size
        var k = idx % Self.col_size

        var acc: Scalar[dtype] = 0
        for b in range(BATCH):
            for s in range(Self.spatial_out):
                var go_val = rebind[Scalar[dtype]](grad_output[b, oc * Self.spatial_out + s])
                var col_val = rebind[Scalar[dtype]](cache[b, k * Self.spatial_out + s])
                acc += go_val * col_val
        grad_params[oc * Self.col_size + k] = rebind[Scalar[dtype]](grad_params[oc * Self.col_size + k]) + acc

        # db: only compute once per oc (when k == 0)
        if k == 0:
            var bias_acc: Scalar[dtype] = 0
            for b in range(BATCH):
                for s in range(Self.spatial_out):
                    bias_acc += rebind[Scalar[dtype]](grad_output[b, oc * Self.spatial_out + s])
            var bias_idx = Self.out_channels * Self.col_size + oc
            grad_params[bias_idx] = rebind[Scalar[dtype]](grad_params[bias_idx]) + bias_acc

    # =========================================================================
    # GPU launchers
    # =========================================================================

    @staticmethod
    fn eval_gpu[
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
    ) raises:
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)

        var total = BATCH * Self.out_channels * Self.spatial_out
        var grid_x = (total + TPB - 1) // TPB

        @always_inline
        fn wrapper(
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
            Self.eval_kernel_impl[BATCH](output, input, params, cache)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
            params_immut,
            cache,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn vjp_gpu[
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

        # Kernel 1: grad_input
        var total_dx = BATCH * Self.IN_DIM
        var grid_dx = (total_dx + TPB - 1) // TPB

        @always_inline
        fn dx_wrapper(
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
            Self.backward_dx_kernel_impl[BATCH](grad_input, grad_output, params)

        ctx.enqueue_function[dx_wrapper, dx_wrapper](
            grad_input,
            grad_output_immut,
            params_immut,
            grid_dim=(grid_dx,),
            block_dim=(TPB,),
        )

        # Kernel 2: dW and db
        var total_dW = Self.out_channels * Self.col_size
        var grid_dW = (total_dW + TPB - 1) // TPB

        @always_inline
        fn dW_wrapper(
            grad_params: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            Self.backward_dW_kernel_impl[BATCH](grad_params, cache, grad_output)

        ctx.enqueue_function[dW_wrapper, dW_wrapper](
            grad_params,
            cache_immut,
            grad_output_immut,
            grid_dim=(grid_dW,),
            block_dim=(TPB,),
        )
