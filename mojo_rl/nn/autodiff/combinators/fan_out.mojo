"""FanOut combinator: runs N copies of the same Model on the same input, concatenates outputs.

Forward:  output = concat(Inner_0(input), Inner_1(input), ..., Inner_{N-1}(input))
Backward: grad_input = sum_i(Inner_i.backward(grad_output_i))
          Each Inner_i has its own params and cache, but shares the same input.

This is a generalization of DualPath — DualPath runs 2 different Models on the
same input. FanOut runs N copies of the SAME Model (with separate params each)
on the same input.
"""

from ...constants import dtype, TPB
from ...model.model import Model, PerfTimerPtr, NULL_PERF
from ...initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream


@fieldwise_init
struct FanOut[Inner: Model, N: Int](Model):
    """Run N independent copies of Inner on the same input, concat outputs.

    IN_DIM = Inner.IN_DIM
    OUT_DIM = N * Inner.OUT_DIM
    PARAM_SIZE = N * Inner.PARAM_SIZE
    CACHE_SIZE = N * Inner.CACHE_SIZE
    """

    comptime IN_DIM: Int = Self.Inner.IN_DIM
    comptime OUT_DIM: Int = Self.N * Self.Inner.OUT_DIM
    comptime PARAM_SIZE: Int = Self.N * Self.Inner.PARAM_SIZE
    comptime CACHE_SIZE: Int = Self.N * Self.Inner.CACHE_SIZE
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = (
        Self.N * Self.Inner.OUT_DIM
        + Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
    )

    # =========================================================================
    # Offset helpers
    # =========================================================================

    @staticmethod
    fn _param_offset[i: Int]() -> Int:
        return i * Self.Inner.PARAM_SIZE

    @staticmethod
    fn _cache_offset[i: Int]() -> Int:
        return i * Self.Inner.CACHE_SIZE

    @staticmethod
    fn _out_offset[i: Int]() -> Int:
        return i * Self.Inner.OUT_DIM

    # =========================================================================
    # Initialization
    # =========================================================================

    @staticmethod
    fn initialize_params[INIT: Initializer](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        comptime for i in range(Self.N):
            var pi = LayoutTensor[
                dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
            ](params.ptr + Self._param_offset[i]())
            Self.Inner.initialize_params[INIT](pi)

    # =========================================================================
    # CPU Forward (with cache)
    # =========================================================================

    @staticmethod
    fn forward[
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
        comptime I_OUT = Self.Inner.OUT_DIM

        # Temp buffer for each copy's output
        var tmp_buf = InlineArray[Scalar[dtype], BATCH * I_OUT](
            uninitialized=True
        )

        comptime for i in range(Self.N):
            var i_out = LayoutTensor[
                dtype, Layout.row_major(BATCH, I_OUT), MutAnyOrigin
            ](tmp_buf.unsafe_ptr())
            var pi = LayoutTensor[
                dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
            ](params.ptr + Self._param_offset[i]())
            var ci = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr + BATCH * Self._cache_offset[i]())
            # Rebind input to Inner.IN_DIM layout for type unification
            var input_i = LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.Inner.IN_DIM), MutAnyOrigin
            ](input.ptr)
            Self.Inner.forward[BATCH](input_i, i_out, pi, ci)

            # Copy into concat output
            for b in range(BATCH):
                for j in range(I_OUT):
                    output.ptr[b * Self.OUT_DIM + Self._out_offset[i]() + j] = (
                        tmp_buf[b * I_OUT + j]
                    )

    # =========================================================================
    # CPU Forward (no cache)
    # =========================================================================

    @staticmethod
    fn forward[
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
    ):
        comptime I_OUT = Self.Inner.OUT_DIM

        var tmp_buf = InlineArray[Scalar[dtype], BATCH * I_OUT](
            uninitialized=True
        )

        comptime for i in range(Self.N):
            var i_out = LayoutTensor[
                dtype, Layout.row_major(BATCH, I_OUT), MutAnyOrigin
            ](tmp_buf.unsafe_ptr())
            var pi = LayoutTensor[
                dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
            ](params.ptr + Self._param_offset[i]())
            var input_i = LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.Inner.IN_DIM), MutAnyOrigin
            ](input.ptr)
            Self.Inner.forward[BATCH](input_i, i_out, pi)

            for b in range(BATCH):
                for j in range(I_OUT):
                    output.ptr[b * Self.OUT_DIM + Self._out_offset[i]() + j] = (
                        tmp_buf[b * I_OUT + j]
                    )

    # =========================================================================
    # CPU Backward
    # =========================================================================

    @staticmethod
    fn backward[
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
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        comptime I_OUT = Self.Inner.OUT_DIM

        # Extract per-copy grad_output slice
        var go_buf = InlineArray[Scalar[dtype], BATCH * I_OUT](
            uninitialized=True
        )

        # Temp grad_input for copies after the first
        var gi_tmp_buf = InlineArray[Scalar[dtype], BATCH * Self.IN_DIM](
            uninitialized=True
        )

        comptime for i in range(Self.N):
            # Extract grad_output slice for copy i
            for b in range(BATCH):
                for j in range(I_OUT):
                    go_buf[b * I_OUT + j] = grad_output.ptr[
                        b * Self.OUT_DIM + Self._out_offset[i]() + j
                    ]

            var go_i = LayoutTensor[
                dtype, Layout.row_major(BATCH, I_OUT), MutAnyOrigin
            ](go_buf.unsafe_ptr())
            var pi = LayoutTensor[
                dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
            ](params.ptr + Self._param_offset[i]())
            var ci = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr + BATCH * Self._cache_offset[i]())
            var grads_i = LayoutTensor[
                dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
            ](grads.ptr + Self._param_offset[i]())

            comptime if i == 0:
                # First copy: write directly to grad_input
                Self.Inner.backward[BATCH](
                    go_i, grad_input, pi, ci, grads_i
                )
            else:
                # Subsequent copies: write to temp, then accumulate
                var gi_tmp = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Inner.IN_DIM),
                    MutAnyOrigin,
                ](gi_tmp_buf.unsafe_ptr())
                Self.Inner.backward[BATCH](
                    go_i, gi_tmp, pi, ci, grads_i
                )
                # Accumulate into grad_input
                for k in range(BATCH * Self.IN_DIM):
                    grad_input.ptr[k] = grad_input.ptr[k] + gi_tmp_buf[k]

    # =========================================================================
    # GPU Forward (with cache)
    # =========================================================================

    @staticmethod
    fn forward_gpu[
        BATCH: Int,
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
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        comptime I_OUT = Self.Inner.OUT_DIM

        # Forward each copy into temp buffers, then concat
        comptime I_PS = Self.Inner.PARAM_SIZE
        comptime IP_BLOCKS = (I_PS + TPB - 1) // TPB
        comptime I_CS_TOTAL = BATCH * Self.Inner.CACHE_SIZE

        comptime for i in range(Self.N):
            var i_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * I_OUT)
            var i_out_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, I_OUT), MutAnyOrigin
            ](i_out_buf.unsafe_ptr())

            var input_i = LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.Inner.IN_DIM), MutAnyOrigin
            ](input.ptr)

            comptime if i == 0:
                # Copy 0: offset 0, always aligned
                var pi = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.Inner.PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr)
                var ci = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                    MutAnyOrigin,
                ](cache.ptr)
                Self.Inner.forward_gpu[BATCH](
                    ctx, i_out_t, input_i, pi, ci, workspace
                )
            else:
                # Copy i>0: may be misaligned, copy to aligned buffers
                var pi_buf = ctx.enqueue_create_buffer[dtype](I_PS)
                var pi = LayoutTensor[
                    dtype, Layout.row_major(I_PS), MutAnyOrigin
                ](pi_buf.unsafe_ptr())

                @always_inline
                fn _fo_fwd_copy_params(
                    dst: LayoutTensor[
                        dtype, Layout.row_major(I_PS), MutAnyOrigin
                    ],
                    src: LayoutTensor[
                        dtype,
                        Layout.row_major(Self.PARAM_SIZE),
                        MutAnyOrigin,
                    ],
                ):
                    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                    if idx < I_PS:
                        dst.ptr[idx] = src.ptr[
                            Self._param_offset[i]() + idx
                        ]

                ctx.enqueue_function[
                    _fo_fwd_copy_params, _fo_fwd_copy_params
                ](pi, params, grid_dim=(IP_BLOCKS,), block_dim=(TPB,))

                var ci_buf = ctx.enqueue_create_buffer[dtype](
                    max(1, I_CS_TOTAL)
                )
                var ci = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                    MutAnyOrigin,
                ](ci_buf.unsafe_ptr())

                Self.Inner.forward_gpu[BATCH](
                    ctx, i_out_t, input_i, pi, ci, workspace
                )

                # Copy cache back to parent cache buffer at correct offset
                if I_CS_TOTAL > 0:
                    comptime IC_BLOCKS = (I_CS_TOTAL + TPB - 1) // TPB

                    @always_inline
                    fn _fo_fwd_copy_cache_back(
                        dst: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.CACHE_SIZE),
                            MutAnyOrigin,
                        ],
                        src: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                            MutAnyOrigin,
                        ],
                    ):
                        var idx = Int(
                            block_dim.x * block_idx.x + thread_idx.x
                        )
                        if idx < I_CS_TOTAL:
                            dst.ptr[
                                BATCH * Self._cache_offset[i]() + idx
                            ] = src.ptr[idx]

                    ctx.enqueue_function[
                        _fo_fwd_copy_cache_back, _fo_fwd_copy_cache_back
                    ](
                        cache,
                        ci,
                        grid_dim=(IC_BLOCKS,),
                        block_dim=(TPB,),
                    )

            # Scatter copy i's output into the concat output
            var i_immut = LayoutTensor[
                dtype, Layout.row_major(BATCH, I_OUT), ImmutAnyOrigin
            ](i_out_buf.unsafe_ptr())
            comptime I_TOTAL = BATCH * I_OUT
            var i_grid = (I_TOTAL + TPB - 1) // TPB

            @always_inline
            fn scatter_k(
                dst: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
                ],
                src: LayoutTensor[
                    dtype, Layout.row_major(BATCH, I_OUT), ImmutAnyOrigin
                ],
            ):
                var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                if idx >= I_TOTAL:
                    return
                var row = idx // I_OUT
                var col = idx % I_OUT
                dst.ptr[row * Self.OUT_DIM + Self._out_offset[i]() + col] = (
                    src.ptr[idx]
                )

            ctx.enqueue_function[scatter_k, scatter_k](
                output, i_immut, grid_dim=(i_grid,), block_dim=(TPB,)
            )

    @staticmethod
    fn forward_gpu_no_cache[
        BATCH: Int,
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
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        pass

    @staticmethod
    fn forward_gpu_no_cache_on_stream[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        stream: DeviceStream,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        Self.forward_gpu_no_cache[BATCH](ctx, output, input, params, workspace)

    @staticmethod
    fn backward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        comptime I_OUT = Self.Inner.OUT_DIM

        var go_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)

        comptime I_PS = Self.Inner.PARAM_SIZE
        comptime IP_BLOCKS = (I_PS + TPB - 1) // TPB
        comptime I_CS_TOTAL = BATCH * Self.Inner.CACHE_SIZE

        comptime for i in range(Self.N):
            # Extract grad_output slice for copy i
            var go_buf = ctx.enqueue_create_buffer[dtype](BATCH * I_OUT)
            var go_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, I_OUT), MutAnyOrigin
            ](go_buf.unsafe_ptr())

            comptime I_TOTAL = BATCH * I_OUT
            var i_grid = (I_TOTAL + TPB - 1) // TPB

            @always_inline
            fn extract_k(
                dst: LayoutTensor[
                    dtype, Layout.row_major(BATCH, I_OUT), MutAnyOrigin
                ],
                src: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.OUT_DIM),
                    ImmutAnyOrigin,
                ],
            ):
                var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                if idx >= I_TOTAL:
                    return
                var row = idx // I_OUT
                var col = idx % I_OUT
                dst.ptr[idx] = src.ptr[
                    row * Self.OUT_DIM + Self._out_offset[i]() + col
                ]

            ctx.enqueue_function[extract_k, extract_k](
                go_t, go_immut, grid_dim=(i_grid,), block_dim=(TPB,)
            )

            comptime if i == 0:
                # Copy 0: offset 0, always aligned
                var pi = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.Inner.PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr)
                var ci = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                    MutAnyOrigin,
                ](cache.ptr)
                var grads_i = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.Inner.PARAM_SIZE),
                    MutAnyOrigin,
                ](grads.ptr)
                # First copy: backward directly into grad_input
                Self.Inner.backward_gpu[BATCH](
                    ctx, grad_input, go_t, pi, ci, grads_i, workspace
                )
            else:
                # Copy i>0: may be misaligned, copy to aligned buffers

                # Copy params to aligned buffer
                var pi_buf = ctx.enqueue_create_buffer[dtype](I_PS)
                var pi = LayoutTensor[
                    dtype, Layout.row_major(I_PS), MutAnyOrigin
                ](pi_buf.unsafe_ptr())

                @always_inline
                fn _fo_bwd_copy_params(
                    dst: LayoutTensor[
                        dtype, Layout.row_major(I_PS), MutAnyOrigin
                    ],
                    src: LayoutTensor[
                        dtype,
                        Layout.row_major(Self.PARAM_SIZE),
                        MutAnyOrigin,
                    ],
                ):
                    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                    if idx < I_PS:
                        dst.ptr[idx] = src.ptr[
                            Self._param_offset[i]() + idx
                        ]

                ctx.enqueue_function[
                    _fo_bwd_copy_params, _fo_bwd_copy_params
                ](pi, params, grid_dim=(IP_BLOCKS,), block_dim=(TPB,))

                # Copy cache to aligned buffer
                var ci_buf = ctx.enqueue_create_buffer[dtype](
                    max(1, I_CS_TOTAL)
                )
                var ci = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                    MutAnyOrigin,
                ](ci_buf.unsafe_ptr())
                if I_CS_TOTAL > 0:
                    comptime IC_BLOCKS = (I_CS_TOTAL + TPB - 1) // TPB

                    @always_inline
                    fn _fo_bwd_copy_cache(
                        dst: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                            MutAnyOrigin,
                        ],
                        src: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.CACHE_SIZE),
                            MutAnyOrigin,
                        ],
                    ):
                        var idx = Int(
                            block_dim.x * block_idx.x + thread_idx.x
                        )
                        if idx < I_CS_TOTAL:
                            dst.ptr[idx] = src.ptr[
                                BATCH * Self._cache_offset[i]() + idx
                            ]

                    ctx.enqueue_function[
                        _fo_bwd_copy_cache, _fo_bwd_copy_cache
                    ](ci, cache, grid_dim=(IC_BLOCKS,), block_dim=(TPB,))

                # Aligned grads buffer, zeroed
                var grads_i_buf = ctx.enqueue_create_buffer[dtype](I_PS)
                var grads_i = LayoutTensor[
                    dtype, Layout.row_major(I_PS), MutAnyOrigin
                ](grads_i_buf.unsafe_ptr())
                var zero_gi = ctx.enqueue_create_host_buffer[dtype](I_PS)
                for zi in range(I_PS):
                    zero_gi[zi] = Scalar[dtype](0.0)
                ctx.enqueue_copy(grads_i_buf, zero_gi)

                # Backward into temp grad_input, then add
                var gi_buf = ctx.enqueue_create_buffer[dtype](
                    BATCH * Self.IN_DIM
                )
                var gi_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Inner.IN_DIM),
                    MutAnyOrigin,
                ](gi_buf.unsafe_ptr())
                Self.Inner.backward_gpu[BATCH](
                    ctx, gi_t, go_t, pi, ci, grads_i, workspace
                )

                # Copy grads back to parent grads buffer at correct offset
                @always_inline
                fn _fo_bwd_scatter_grads(
                    dst: LayoutTensor[
                        dtype,
                        Layout.row_major(Self.PARAM_SIZE),
                        MutAnyOrigin,
                    ],
                    src: LayoutTensor[
                        dtype, Layout.row_major(I_PS), MutAnyOrigin
                    ],
                ):
                    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                    if idx < I_PS:
                        dst.ptr[Self._param_offset[i]() + idx] = src.ptr[idx]

                ctx.enqueue_function[
                    _fo_bwd_scatter_grads, _fo_bwd_scatter_grads
                ](grads, grads_i, grid_dim=(IP_BLOCKS,), block_dim=(TPB,))

                # Add to grad_input
                var gi_immut = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.IN_DIM),
                    ImmutAnyOrigin,
                ](gi_buf.unsafe_ptr())
                comptime GI_TOTAL = BATCH * Self.IN_DIM
                var gi_grid = (GI_TOTAL + TPB - 1) // TPB

                @always_inline
                fn add_gi(
                    a: LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.IN_DIM),
                        MutAnyOrigin,
                    ],
                    b: LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.IN_DIM),
                        ImmutAnyOrigin,
                    ],
                ):
                    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                    if idx >= GI_TOTAL:
                        return
                    a.ptr[idx] = a.ptr[idx] + b.ptr[idx]

                ctx.enqueue_function[add_gi, add_gi](
                    grad_input,
                    gi_immut,
                    grid_dim=(gi_grid,),
                    block_dim=(TPB,),
                )
