from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext


struct MaxPool2D[
    channels: Int,
    in_h: Int,
    in_w: Int,
    pool_size: Int,
](DiffOp):
    """Max pooling over spatial dimensions.

    Input:  (BATCH, channels * in_h * in_w)
    Output: (BATCH, channels * out_h * out_w)

    No learnable parameters.
    Cache stores the argmax index within each pool window for backward routing.
    """

    comptime out_h: Int = Self.in_h // Self.pool_size
    comptime out_w: Int = Self.in_w // Self.pool_size
    comptime spatial_out: Int = Self.out_h * Self.out_w

    comptime OP_ID: Int = OpID.MAX_POOL2D._value
    comptime IN_DIM: Int = Self.channels * Self.in_h * Self.in_w
    comptime OUT_DIM: Int = Self.channels * Self.out_h * Self.out_w
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = Self.channels * Self.out_h * Self.out_w

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
        """Forward: max over each pool window, cache argmax flat index."""
        for b in range(BATCH):
            for c in range(Self.channels):
                for oh in range(Self.out_h):
                    for ow in range(Self.out_w):
                        var max_val: Scalar[dtype] = -1e30
                        var max_idx: Int = 0
                        for ph in range(Self.pool_size):
                            for pw in range(Self.pool_size):
                                var ih = oh * Self.pool_size + ph
                                var iw = ow * Self.pool_size + pw
                                var in_idx = c * Self.in_h * Self.in_w + ih * Self.in_w + iw
                                var val = rebind[Scalar[dtype]](input[b, in_idx])
                                if val > max_val:
                                    max_val = val
                                    max_idx = in_idx
                        var out_idx = c * Self.spatial_out + oh * Self.out_w + ow
                        output[b, out_idx] = max_val
                        # Store argmax as float (will cast back to int in backward)
                        cache[b, out_idx] = Scalar[dtype](max_idx)

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
        """Backward: route gradient to argmax position only."""
        # Zero grad_input
        for b in range(BATCH):
            for i in range(Self.IN_DIM):
                grad_input[b, i] = 0

        # Route gradients
        for b in range(BATCH):
            for out_idx in range(Self.OUT_DIM):
                var max_idx = Int(rebind[Scalar[dtype]](cache[b, out_idx]))
                var cur = rebind[Scalar[dtype]](grad_input[b, max_idx])
                grad_input[b, max_idx] = cur + rebind[Scalar[dtype]](grad_output[b, out_idx])

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
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """One thread per output element."""
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        var total = BATCH * Self.OUT_DIM
        if idx >= total:
            return

        var b = idx // Self.OUT_DIM
        var out_pos = idx % Self.OUT_DIM

        var c = out_pos // Self.spatial_out
        var rem = out_pos % Self.spatial_out
        var oh = rem // Self.out_w
        var ow = rem % Self.out_w

        var max_val = Scalar[dtype](-1e30)
        var max_idx: Int = 0
        for ph in range(Self.pool_size):
            for pw in range(Self.pool_size):
                var ih = oh * Self.pool_size + ph
                var iw = ow * Self.pool_size + pw
                var in_idx = c * Self.in_h * Self.in_w + ih * Self.in_w + iw
                var val = rebind[Scalar[dtype]](input[b, in_idx])
                if val > max_val:
                    max_val = val
                    max_idx = in_idx

        output[b, out_pos] = max_val
        cache[b, out_pos] = Scalar[dtype](max_idx)

    @always_inline
    @staticmethod
    fn backward_kernel_impl[
        BATCH: Int
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
    ):
        """One thread per output element, atomically routes gradient to argmax."""
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        var total = BATCH * Self.OUT_DIM
        if idx >= total:
            return

        var b = idx // Self.OUT_DIM
        var out_pos = idx % Self.OUT_DIM
        var max_idx = Int(rebind[Scalar[dtype]](cache[b, out_pos]))

        # Note: potential race condition if two output positions share same argmax.
        # For non-overlapping pools (stride == pool_size), each input maps to at most
        # one output, so no race.
        var cur = rebind[Scalar[dtype]](grad_input[b, max_idx])
        grad_input[b, max_idx] = cur + rebind[Scalar[dtype]](grad_output[b, out_pos])

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

        var total = BATCH * Self.OUT_DIM
        var grid_x = (total + TPB - 1) // TPB

        @always_inline
        fn wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
            ],
        ):
            Self.eval_kernel_impl[BATCH](output, input, cache)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
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
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ](cache.ptr)

        var total = BATCH * Self.OUT_DIM
        var grid_x = (total + TPB - 1) // TPB

        @always_inline
        fn wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
            ],
        ):
            Self.backward_kernel_impl[BATCH](grad_input, grad_output, cache)

        ctx.enqueue_function[wrapper, wrapper](
            grad_input,
            grad_output_immut,
            cache_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )


struct AvgPool2D[
    channels: Int,
    in_h: Int,
    in_w: Int,
    pool_size: Int,
](DiffOp):
    """Average pooling over spatial dimensions.

    Input:  (BATCH, channels * in_h * in_w)
    Output: (BATCH, channels * out_h * out_w)

    No learnable parameters. No cache needed (backward is uniform distribution).
    """

    comptime out_h: Int = Self.in_h // Self.pool_size
    comptime out_w: Int = Self.in_w // Self.pool_size
    comptime spatial_out: Int = Self.out_h * Self.out_w
    comptime pool_area: Int = Self.pool_size * Self.pool_size

    comptime OP_ID: Int = OpID.AVG_POOL2D._value
    comptime IN_DIM: Int = Self.channels * Self.in_h * Self.in_w
    comptime OUT_DIM: Int = Self.channels * Self.out_h * Self.out_w
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = 0

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
        """Forward: average over each pool window."""
        var scale = Scalar[dtype](1.0) / Scalar[dtype](Self.pool_area)
        for b in range(BATCH):
            for c in range(Self.channels):
                for oh in range(Self.out_h):
                    for ow in range(Self.out_w):
                        var acc: Scalar[dtype] = 0
                        for ph in range(Self.pool_size):
                            for pw in range(Self.pool_size):
                                var ih = oh * Self.pool_size + ph
                                var iw = ow * Self.pool_size + pw
                                var in_idx = c * Self.in_h * Self.in_w + ih * Self.in_w + iw
                                acc += rebind[Scalar[dtype]](input[b, in_idx])
                        var out_idx = c * Self.spatial_out + oh * Self.out_w + ow
                        output[b, out_idx] = acc * scale

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
        """Backward: distribute gradient equally across window."""
        var scale = Scalar[dtype](1.0) / Scalar[dtype](Self.pool_area)

        # Zero grad_input
        for b in range(BATCH):
            for i in range(Self.IN_DIM):
                grad_input[b, i] = 0

        for b in range(BATCH):
            for c in range(Self.channels):
                for oh in range(Self.out_h):
                    for ow in range(Self.out_w):
                        var out_idx = c * Self.spatial_out + oh * Self.out_w + ow
                        var g = rebind[Scalar[dtype]](grad_output[b, out_idx]) * scale
                        for ph in range(Self.pool_size):
                            for pw in range(Self.pool_size):
                                var ih = oh * Self.pool_size + ph
                                var iw = ow * Self.pool_size + pw
                                var in_idx = c * Self.in_h * Self.in_w + ih * Self.in_w + iw
                                var cur = rebind[Scalar[dtype]](grad_input[b, in_idx])
                                grad_input[b, in_idx] = cur + g

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
    ):
        """One thread per output element."""
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        var total = BATCH * Self.OUT_DIM
        if idx >= total:
            return

        var b = idx // Self.OUT_DIM
        var out_pos = idx % Self.OUT_DIM

        var c = out_pos // Self.spatial_out
        var rem = out_pos % Self.spatial_out
        var oh = rem // Self.out_w
        var ow = rem % Self.out_w

        var acc: Scalar[dtype] = 0
        for ph in range(Self.pool_size):
            for pw in range(Self.pool_size):
                var ih = oh * Self.pool_size + ph
                var iw = ow * Self.pool_size + pw
                var in_idx = c * Self.in_h * Self.in_w + ih * Self.in_w + iw
                acc += rebind[Scalar[dtype]](input[b, in_idx])

        var scale = Scalar[dtype](1.0) / Scalar[dtype](Self.pool_area)
        output[b, out_pos] = acc * scale

    @always_inline
    @staticmethod
    fn backward_kernel_impl[
        BATCH: Int
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
    ):
        """One thread per input element, find which pool window it belongs to."""
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

        var oh = ih // Self.pool_size
        var ow = iw // Self.pool_size

        var scale = Scalar[dtype](1.0) / Scalar[dtype](Self.pool_area)

        if oh < Self.out_h and ow < Self.out_w:
            var out_idx = c * Self.spatial_out + oh * Self.out_w + ow
            grad_input[b, in_pos] = rebind[Scalar[dtype]](grad_output[b, out_idx]) * scale
        else:
            grad_input[b, in_pos] = 0

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

        var total = BATCH * Self.OUT_DIM
        var grid_x = (total + TPB - 1) // TPB

        @always_inline
        fn wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
        ):
            Self.eval_kernel_impl[BATCH](output, input)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
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
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)

        var total = BATCH * Self.IN_DIM
        var grid_x = (total + TPB - 1) // TPB

        @always_inline
        fn wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            Self.backward_kernel_impl[BATCH](grad_input, grad_output)

        ctx.enqueue_function[wrapper, wrapper](
            grad_input,
            grad_output_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )
