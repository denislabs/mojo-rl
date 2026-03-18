"""DualPath combinator: runs two Models on the same input, concatenates outputs.

Forward:  output = concat(A(input), B(input))
Backward: grad_input = A.backward(grad_A) + B.backward(grad_B)

This is like Parallel but at the Model level (Parallel works at DiffOp level).
Used to forward through twin critics on the same critic_input:

    critic_input → DualPath[Critic1, Critic2] → [Q1 || Q2]
    [Q1 || Q2] → MinOp → min_Q

The backward automatically routes gradients to the correct critic and sums
the grad_inputs (which are the same critic_input, so the sum is correct).
"""

from ...constants import dtype, TPB
from ...model.model import Model, PerfTimerPtr, NULL_PERF
from ...initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream


@fieldwise_init
struct DualPath[A: Model, B: Model](Model):
    """Run two models on same input, concatenate outputs.

    Requires: A.IN_DIM == B.IN_DIM
    OUT_DIM = A.OUT_DIM + B.OUT_DIM
    PARAM_SIZE = A.PARAM_SIZE + B.PARAM_SIZE
    """

    comptime IN_DIM: Int = Self.A.IN_DIM
    comptime OUT_DIM: Int = Self.A.OUT_DIM + Self.B.OUT_DIM
    comptime PARAM_SIZE: Int = Self.A.PARAM_SIZE + Self.B.PARAM_SIZE
    comptime CACHE_SIZE: Int = Self.A.CACHE_SIZE + Self.B.CACHE_SIZE
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = (
        # Temp buffers for A and B outputs + their workspaces
        Self.A.OUT_DIM
        + Self.B.OUT_DIM
        + Self.A.WORKSPACE_SIZE_PER_SAMPLE
        + Self.B.WORKSPACE_SIZE_PER_SAMPLE
    )

    # =========================================================================
    # Offset helpers
    # =========================================================================

    @staticmethod
    fn _a_param_offset() -> Int:
        return 0

    @staticmethod
    fn _b_param_offset() -> Int:
        return Self.A.PARAM_SIZE

    @staticmethod
    fn _a_cache_offset() -> Int:
        return 0

    @staticmethod
    fn _b_cache_offset() -> Int:
        return Self.A.CACHE_SIZE

    # =========================================================================
    # Initialization
    # =========================================================================

    @staticmethod
    fn initialize_params[INIT: Initializer](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        var pa = LayoutTensor[
            dtype, Layout.row_major(Self.A.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        Self.A.initialize_params[INIT](pa)

        var pb = LayoutTensor[
            dtype, Layout.row_major(Self.B.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self.A.PARAM_SIZE)
        Self.B.initialize_params[INIT](pb)

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
        comptime A_OUT = Self.A.OUT_DIM
        comptime B_OUT = Self.B.OUT_DIM

        # Forward A
        var a_buf = InlineArray[Scalar[dtype], BATCH * A_OUT](
            uninitialized=True
        )
        var a_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, A_OUT), MutAnyOrigin
        ](a_buf.unsafe_ptr())
        var pa = LayoutTensor[
            dtype, Layout.row_major(Self.A.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var ca = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.A.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        Self.A.forward[BATCH](input, a_out, pa, ca)

        # Forward B — rebind input to B.IN_DIM layout (same value, different type path)
        var b_buf = InlineArray[Scalar[dtype], BATCH * B_OUT](
            uninitialized=True
        )
        var b_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, B_OUT), MutAnyOrigin
        ](b_buf.unsafe_ptr())
        var pb = LayoutTensor[
            dtype, Layout.row_major(Self.B.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self.A.PARAM_SIZE)
        var cb = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.B.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr + BATCH * Self.A.CACHE_SIZE)
        var input_b = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.B.IN_DIM), MutAnyOrigin
        ](input.ptr)
        Self.B.forward[BATCH](input_b, b_out, pb, cb)

        # Concat outputs
        for b in range(BATCH):
            for i in range(A_OUT):
                output.ptr[b * Self.OUT_DIM + i] = a_buf[b * A_OUT + i]
            for i in range(B_OUT):
                output.ptr[b * Self.OUT_DIM + A_OUT + i] = b_buf[
                    b * B_OUT + i
                ]

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
        comptime A_OUT = Self.A.OUT_DIM
        comptime B_OUT = Self.B.OUT_DIM

        var a_buf = InlineArray[Scalar[dtype], BATCH * A_OUT](
            uninitialized=True
        )
        var a_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, A_OUT), MutAnyOrigin
        ](a_buf.unsafe_ptr())
        var pa = LayoutTensor[
            dtype, Layout.row_major(Self.A.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        Self.A.forward[BATCH](input, a_out, pa)

        var b_buf = InlineArray[Scalar[dtype], BATCH * B_OUT](
            uninitialized=True
        )
        var b_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, B_OUT), MutAnyOrigin
        ](b_buf.unsafe_ptr())
        var pb = LayoutTensor[
            dtype, Layout.row_major(Self.B.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self.A.PARAM_SIZE)
        var input_b = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.B.IN_DIM), MutAnyOrigin
        ](input.ptr)
        Self.B.forward[BATCH](input_b, b_out, pb)

        for b in range(BATCH):
            for i in range(A_OUT):
                output.ptr[b * Self.OUT_DIM + i] = a_buf[b * A_OUT + i]
            for i in range(B_OUT):
                output.ptr[b * Self.OUT_DIM + A_OUT + i] = b_buf[
                    b * B_OUT + i
                ]

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
        comptime A_OUT = Self.A.OUT_DIM
        comptime B_OUT = Self.B.OUT_DIM

        # Extract per-branch gradients
        var ga_buf = InlineArray[Scalar[dtype], BATCH * A_OUT](
            uninitialized=True
        )
        var gb_buf = InlineArray[Scalar[dtype], BATCH * B_OUT](
            uninitialized=True
        )
        for b in range(BATCH):
            for i in range(A_OUT):
                ga_buf[b * A_OUT + i] = grad_output.ptr[
                    b * Self.OUT_DIM + i
                ]
            for i in range(B_OUT):
                gb_buf[b * B_OUT + i] = grad_output.ptr[
                    b * Self.OUT_DIM + A_OUT + i
                ]

        # Backward A → grad_input
        var ga = LayoutTensor[
            dtype, Layout.row_major(BATCH, A_OUT), MutAnyOrigin
        ](ga_buf.unsafe_ptr())
        var pa = LayoutTensor[
            dtype, Layout.row_major(Self.A.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var ca = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.A.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        var grads_a = LayoutTensor[
            dtype, Layout.row_major(Self.A.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr)
        Self.A.backward[BATCH](ga, grad_input, pa, ca, grads_a)

        # Backward B → temp grad_input, then add
        var gi_b_buf = InlineArray[Scalar[dtype], BATCH * Self.IN_DIM](
            uninitialized=True
        )
        var gi_b = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ](gi_b_buf.unsafe_ptr())
        var gb = LayoutTensor[
            dtype, Layout.row_major(BATCH, B_OUT), MutAnyOrigin
        ](gb_buf.unsafe_ptr())
        var pb = LayoutTensor[
            dtype, Layout.row_major(Self.B.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self.A.PARAM_SIZE)
        var cb = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.B.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr + BATCH * Self.A.CACHE_SIZE)
        var grads_b = LayoutTensor[
            dtype, Layout.row_major(Self.B.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr + Self.A.PARAM_SIZE)
        var gi_b_rb = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.B.IN_DIM), MutAnyOrigin
        ](gi_b_buf.unsafe_ptr())
        Self.B.backward[BATCH](gb, gi_b_rb, pb, cb, grads_b)

        # Sum grad_inputs
        for i in range(BATCH * Self.IN_DIM):
            grad_input.ptr[i] = grad_input.ptr[i] + gi_b_buf[i]

    # =========================================================================
    # GPU Forward/Backward (with cache)
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
        comptime A_OUT = Self.A.OUT_DIM
        comptime B_OUT = Self.B.OUT_DIM

        # Allocate temp output buffers
        var a_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * A_OUT)
        var b_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * B_OUT)

        var a_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, A_OUT), MutAnyOrigin
        ](a_out_buf.unsafe_ptr())
        var b_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, B_OUT), MutAnyOrigin
        ](b_out_buf.unsafe_ptr())

        var pa = LayoutTensor[
            dtype, Layout.row_major(Self.A.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)

        var ca = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.A.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)

        # B params/cache: copy to aligned buffers to avoid misalignment
        # (A.PARAM_SIZE may not be a multiple of 4, causing misaligned matmul)
        var pb_buf = ctx.enqueue_create_buffer[dtype](Self.B.PARAM_SIZE)
        var pb = LayoutTensor[
            dtype, Layout.row_major(Self.B.PARAM_SIZE), MutAnyOrigin
        ](pb_buf.unsafe_ptr())
        comptime B_PS = Self.B.PARAM_SIZE
        comptime BP_BLOCKS = (B_PS + TPB - 1) // TPB

        @always_inline
        fn _dp_copy_params(
            dst: LayoutTensor[
                dtype, Layout.row_major(B_PS), MutAnyOrigin
            ],
            src: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
            ],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i < B_PS:
                dst.ptr[i] = src.ptr[Self.A.PARAM_SIZE + i]

        ctx.enqueue_function[_dp_copy_params, _dp_copy_params](
            pb, params, grid_dim=(BP_BLOCKS,), block_dim=(TPB,)
        )

        var cb_buf = ctx.enqueue_create_buffer[dtype](
            max(1, BATCH * Self.B.CACHE_SIZE)
        )
        var cb = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.B.CACHE_SIZE), MutAnyOrigin
        ](cb_buf.unsafe_ptr())

        # Forward both
        Self.A.forward_gpu[BATCH](ctx, a_out_t, input, pa, ca, workspace)
        var input_b = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.B.IN_DIM), MutAnyOrigin
        ](input.ptr)
        Self.B.forward_gpu[BATCH](ctx, b_out_t, input_b, pb, cb, workspace)

        # Copy B cache back to parent cache buffer at correct offset
        comptime B_CS_TOTAL = BATCH * Self.B.CACHE_SIZE
        if B_CS_TOTAL > 0:
            comptime BC_BLOCKS = (B_CS_TOTAL + TPB - 1) // TPB

            @always_inline
            fn _dp_copy_cache_back(
                dst: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
                ],
                src: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.B.CACHE_SIZE), MutAnyOrigin
                ],
            ):
                var i = Int(block_dim.x * block_idx.x + thread_idx.x)
                if i < B_CS_TOTAL:
                    dst.ptr[BATCH * Self.A.CACHE_SIZE + i] = src.ptr[i]

            ctx.enqueue_function[_dp_copy_cache_back, _dp_copy_cache_back](
                cache, cb, grid_dim=(BC_BLOCKS,), block_dim=(TPB,)
            )

        # Concat into output
        var a_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, A_OUT), ImmutAnyOrigin
        ](a_out_buf.unsafe_ptr())
        var b_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, B_OUT), ImmutAnyOrigin
        ](b_out_buf.unsafe_ptr())
        comptime TOTAL = BATCH * Self.OUT_DIM
        var grid_x = (TOTAL + TPB - 1) // TPB

        @always_inline
        fn concat_k(
            dst: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            sa: LayoutTensor[
                dtype, Layout.row_major(BATCH, A_OUT), ImmutAnyOrigin
            ],
            sb: LayoutTensor[
                dtype, Layout.row_major(BATCH, B_OUT), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= TOTAL:
                return
            var row = idx // Self.OUT_DIM
            var col = idx % Self.OUT_DIM
            if col < A_OUT:
                dst.ptr[idx] = sa.ptr[row * A_OUT + col]
            else:
                dst.ptr[idx] = sb.ptr[row * B_OUT + (col - A_OUT)]

        ctx.enqueue_function[concat_k, concat_k](
            output, a_immut, b_immut, grid_dim=(grid_x,), block_dim=(TPB,)
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
        # Simplified: just use forward_gpu with dummy cache
        # In practice, the no-cache path is rarely used for training
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
        comptime A_OUT = Self.A.OUT_DIM
        comptime B_OUT = Self.B.OUT_DIM

        # Extract per-branch grads
        var ga_buf = ctx.enqueue_create_buffer[dtype](BATCH * A_OUT)
        var gb_buf = ctx.enqueue_create_buffer[dtype](BATCH * B_OUT)

        var ga_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, A_OUT), MutAnyOrigin
        ](ga_buf.unsafe_ptr())
        var gb_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, B_OUT), MutAnyOrigin
        ](gb_buf.unsafe_ptr())

        var go_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)

        # Extract A grad
        comptime A_TOTAL = BATCH * A_OUT
        var a_grid = (A_TOTAL + TPB - 1) // TPB

        @always_inline
        fn extract_a(
            dst: LayoutTensor[
                dtype, Layout.row_major(BATCH, A_OUT), MutAnyOrigin
            ],
            src: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= A_TOTAL:
                return
            var b = idx // A_OUT
            var i = idx % A_OUT
            dst.ptr[idx] = src.ptr[b * Self.OUT_DIM + i]

        ctx.enqueue_function[extract_a, extract_a](
            ga_t, go_immut, grid_dim=(a_grid,), block_dim=(TPB,)
        )

        # Extract B grad
        comptime B_TOTAL = BATCH * B_OUT
        var b_grid = (B_TOTAL + TPB - 1) // TPB

        @always_inline
        fn extract_b(
            dst: LayoutTensor[
                dtype, Layout.row_major(BATCH, B_OUT), MutAnyOrigin
            ],
            src: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= B_TOTAL:
                return
            var b = idx // B_OUT
            var i = idx % B_OUT
            dst.ptr[idx] = src.ptr[b * Self.OUT_DIM + A_OUT + i]

        ctx.enqueue_function[extract_b, extract_b](
            gb_t, go_immut, grid_dim=(b_grid,), block_dim=(TPB,)
        )

        # Backward A
        var pa = LayoutTensor[
            dtype, Layout.row_major(Self.A.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var ca = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.A.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        var grads_a = LayoutTensor[
            dtype, Layout.row_major(Self.A.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr)
        Self.A.backward_gpu[BATCH](
            ctx, grad_input, ga_t, pa, ca, grads_a, workspace
        )

        # Backward B → temp, then add
        # Copy B's params, cache, grads to aligned buffers
        var gi_b_buf = ctx.enqueue_create_buffer[dtype](BATCH * Self.IN_DIM)
        comptime B_PS = Self.B.PARAM_SIZE
        comptime B_CS_TOTAL = BATCH * Self.B.CACHE_SIZE
        comptime BP_BLOCKS = (B_PS + TPB - 1) // TPB

        var pb_buf = ctx.enqueue_create_buffer[dtype](B_PS)
        var pb = LayoutTensor[
            dtype, Layout.row_major(B_PS), MutAnyOrigin
        ](pb_buf.unsafe_ptr())

        @always_inline
        fn _dp_bwd_copy_params(
            dst: LayoutTensor[dtype, Layout.row_major(B_PS), MutAnyOrigin],
            src: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
            ],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i < B_PS:
                dst.ptr[i] = src.ptr[Self.A.PARAM_SIZE + i]

        ctx.enqueue_function[_dp_bwd_copy_params, _dp_bwd_copy_params](
            pb, params, grid_dim=(BP_BLOCKS,), block_dim=(TPB,)
        )

        var cb_buf = ctx.enqueue_create_buffer[dtype](max(1, B_CS_TOTAL))
        var cb = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.B.CACHE_SIZE), MutAnyOrigin
        ](cb_buf.unsafe_ptr())
        # Copy B cache from parent cache buffer
        if B_CS_TOTAL > 0:
            comptime BC_BLOCKS = (B_CS_TOTAL + TPB - 1) // TPB

            @always_inline
            fn _dp_bwd_copy_cache(
                dst: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.B.CACHE_SIZE),
                    MutAnyOrigin,
                ],
                src: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.CACHE_SIZE),
                    MutAnyOrigin,
                ],
            ):
                var i = Int(block_dim.x * block_idx.x + thread_idx.x)
                if i < B_CS_TOTAL:
                    dst.ptr[i] = src.ptr[BATCH * Self.A.CACHE_SIZE + i]

            ctx.enqueue_function[_dp_bwd_copy_cache, _dp_bwd_copy_cache](
                cb, cache, grid_dim=(BC_BLOCKS,), block_dim=(TPB,)
            )

        var grads_b_buf = ctx.enqueue_create_buffer[dtype](B_PS)
        var grads_b = LayoutTensor[
            dtype, Layout.row_major(B_PS), MutAnyOrigin
        ](grads_b_buf.unsafe_ptr())
        # Zero B grads
        var zero_b = ctx.enqueue_create_host_buffer[dtype](B_PS)
        for i in range(B_PS):
            zero_b[i] = Scalar[dtype](0.0)
        ctx.enqueue_copy(grads_b_buf, zero_b)

        var gi_b_rb = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.B.IN_DIM), MutAnyOrigin
        ](gi_b_buf.unsafe_ptr())
        Self.B.backward_gpu[BATCH](
            ctx, gi_b_rb, gb_t, pb, cb, grads_b, workspace
        )

        # Copy B grads back to parent grads buffer at correct offset
        @always_inline
        fn _dp_bwd_scatter_grads(
            dst: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
            ],
            src: LayoutTensor[dtype, Layout.row_major(B_PS), MutAnyOrigin],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i < B_PS:
                dst.ptr[Self.A.PARAM_SIZE + i] = src.ptr[i]

        ctx.enqueue_function[_dp_bwd_scatter_grads, _dp_bwd_scatter_grads](
            grads, grads_b, grid_dim=(BP_BLOCKS,), block_dim=(TPB,)
        )

        # Add B's grad_input to A's
        var gi_b_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](gi_b_buf.unsafe_ptr())
        comptime GI_TOTAL = BATCH * Self.IN_DIM
        var gi_grid = (GI_TOTAL + TPB - 1) // TPB

        @always_inline
        fn add_gi(
            a: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            b: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= GI_TOTAL:
                return
            a.ptr[idx] = a.ptr[idx] + b.ptr[idx]

        ctx.enqueue_function[add_gi, add_gi](
            grad_input,
            gi_b_immut,
            grid_dim=(gi_grid,),
            block_dim=(TPB,),
        )
