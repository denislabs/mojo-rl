"""SplitApply combinator: split input at a boundary, apply different Models.

Forward:  output = concat(A(input[:, :split]), B(input[:, split:]))
Backward: grad_input[:, :split]  = A.backward(grad_A)
          grad_input[:, split:]  = B.backward(grad_B)

This enables routing different slices of a concatenated tensor to different
networks. Key use case for SAC:

    SkipConcat[Actor → RSample] → [obs(17), action(6), log_prob(1)]
                                       ↓                    ↓
    SplitApply[                  TwinCriticMin,          Identity,
               split=23]     →  [min_Q(1),              log_prob(1)]

The split point divides the input into left (dims 0..split-1) and
right (dims split..IN_DIM-1) portions.
"""

from ...constants import dtype, TPB
from ...model.model import Model, PerfTimerPtr, NULL_PERF
from ...initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream


@fieldwise_init
struct SplitApply[Left: Model, Right: Model, split: Int](Model):
    """Split input at `split`, apply Left to [:split] and Right to [split:].

    IN_DIM = Left.IN_DIM + Right.IN_DIM  (= split + remaining)
    OUT_DIM = Left.OUT_DIM + Right.OUT_DIM
    PARAM_SIZE = Left.PARAM_SIZE + Right.PARAM_SIZE
    """

    comptime IN_DIM: Int = Self.Left.IN_DIM + Self.Right.IN_DIM
    comptime OUT_DIM: Int = Self.Left.OUT_DIM + Self.Right.OUT_DIM
    comptime PARAM_SIZE: Int = Self.Left.PARAM_SIZE + Self.Right.PARAM_SIZE
    comptime CACHE_SIZE: Int = Self.Left.CACHE_SIZE + Self.Right.CACHE_SIZE
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = (
        Self.Left.WORKSPACE_SIZE_PER_SAMPLE
        + Self.Right.WORKSPACE_SIZE_PER_SAMPLE
    )

    comptime LEFT_IN: Int = Self.Left.IN_DIM
    comptime RIGHT_IN: Int = Self.Right.IN_DIM
    comptime LEFT_OUT: Int = Self.Left.OUT_DIM
    comptime RIGHT_OUT: Int = Self.Right.OUT_DIM

    # =========================================================================
    # Initialization
    # =========================================================================

    @staticmethod
    fn initialize_params[INIT: Initializer](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        var pl = LayoutTensor[
            dtype, Layout.row_major(Self.Left.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        Self.Left.initialize_params[INIT](pl)
        var pr = LayoutTensor[
            dtype, Layout.row_major(Self.Right.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self.Left.PARAM_SIZE)
        Self.Right.initialize_params[INIT](pr)

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
        # Extract left input slice
        var left_in_buf = InlineArray[Scalar[dtype], BATCH * Self.LEFT_IN](
            uninitialized=True
        )
        var right_in_buf = InlineArray[Scalar[dtype], BATCH * Self.RIGHT_IN](
            uninitialized=True
        )
        for b in range(BATCH):
            for i in range(Self.LEFT_IN):
                left_in_buf[b * Self.LEFT_IN + i] = input.ptr[
                    b * Self.IN_DIM + i
                ]
            for i in range(Self.RIGHT_IN):
                right_in_buf[b * Self.RIGHT_IN + i] = input.ptr[
                    b * Self.IN_DIM + Self.split + i
                ]

        # Forward Left
        var left_in = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_IN), MutAnyOrigin
        ](left_in_buf.unsafe_ptr())
        var left_out_buf = InlineArray[
            Scalar[dtype], BATCH * Self.LEFT_OUT
        ](uninitialized=True)
        var left_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_OUT), MutAnyOrigin
        ](left_out_buf.unsafe_ptr())
        var pl = LayoutTensor[
            dtype, Layout.row_major(Self.Left.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var cl = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Left.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        Self.Left.forward[BATCH](left_in, left_out, pl, cl)

        # Forward Right
        var right_in = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_IN), MutAnyOrigin
        ](right_in_buf.unsafe_ptr())
        var right_out_buf = InlineArray[
            Scalar[dtype], BATCH * Self.RIGHT_OUT
        ](uninitialized=True)
        var right_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_OUT), MutAnyOrigin
        ](right_out_buf.unsafe_ptr())
        var pr = LayoutTensor[
            dtype, Layout.row_major(Self.Right.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self.Left.PARAM_SIZE)
        var cr = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Right.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr + BATCH * Self.Left.CACHE_SIZE)
        Self.Right.forward[BATCH](right_in, right_out, pr, cr)

        # Concat outputs
        for b in range(BATCH):
            for i in range(Self.LEFT_OUT):
                output.ptr[b * Self.OUT_DIM + i] = left_out_buf[
                    b * Self.LEFT_OUT + i
                ]
            for i in range(Self.RIGHT_OUT):
                output.ptr[b * Self.OUT_DIM + Self.LEFT_OUT + i] = (
                    right_out_buf[b * Self.RIGHT_OUT + i]
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
        # Extract slices
        var left_in_buf = InlineArray[Scalar[dtype], BATCH * Self.LEFT_IN](
            uninitialized=True
        )
        var right_in_buf = InlineArray[Scalar[dtype], BATCH * Self.RIGHT_IN](
            uninitialized=True
        )
        for b in range(BATCH):
            for i in range(Self.LEFT_IN):
                left_in_buf[b * Self.LEFT_IN + i] = input.ptr[
                    b * Self.IN_DIM + i
                ]
            for i in range(Self.RIGHT_IN):
                right_in_buf[b * Self.RIGHT_IN + i] = input.ptr[
                    b * Self.IN_DIM + Self.split + i
                ]

        var left_in = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_IN), MutAnyOrigin
        ](left_in_buf.unsafe_ptr())
        var left_out_buf = InlineArray[
            Scalar[dtype], BATCH * Self.LEFT_OUT
        ](uninitialized=True)
        var left_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_OUT), MutAnyOrigin
        ](left_out_buf.unsafe_ptr())
        var pl = LayoutTensor[
            dtype, Layout.row_major(Self.Left.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        Self.Left.forward[BATCH](left_in, left_out, pl)

        var right_in = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_IN), MutAnyOrigin
        ](right_in_buf.unsafe_ptr())
        var right_out_buf = InlineArray[
            Scalar[dtype], BATCH * Self.RIGHT_OUT
        ](uninitialized=True)
        var right_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_OUT), MutAnyOrigin
        ](right_out_buf.unsafe_ptr())
        var pr = LayoutTensor[
            dtype, Layout.row_major(Self.Right.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self.Left.PARAM_SIZE)
        Self.Right.forward[BATCH](right_in, right_out, pr)

        for b in range(BATCH):
            for i in range(Self.LEFT_OUT):
                output.ptr[b * Self.OUT_DIM + i] = left_out_buf[
                    b * Self.LEFT_OUT + i
                ]
            for i in range(Self.RIGHT_OUT):
                output.ptr[b * Self.OUT_DIM + Self.LEFT_OUT + i] = (
                    right_out_buf[b * Self.RIGHT_OUT + i]
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
        # Extract per-branch gradients
        var gl_buf = InlineArray[Scalar[dtype], BATCH * Self.LEFT_OUT](
            uninitialized=True
        )
        var gr_buf = InlineArray[Scalar[dtype], BATCH * Self.RIGHT_OUT](
            uninitialized=True
        )
        for b in range(BATCH):
            for i in range(Self.LEFT_OUT):
                gl_buf[b * Self.LEFT_OUT + i] = grad_output.ptr[
                    b * Self.OUT_DIM + i
                ]
            for i in range(Self.RIGHT_OUT):
                gr_buf[b * Self.RIGHT_OUT + i] = grad_output.ptr[
                    b * Self.OUT_DIM + Self.LEFT_OUT + i
                ]

        # Backward Left → left portion of grad_input
        var gl = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_OUT), MutAnyOrigin
        ](gl_buf.unsafe_ptr())
        var gi_l_buf = InlineArray[Scalar[dtype], BATCH * Self.LEFT_IN](
            uninitialized=True
        )
        var gi_l = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_IN), MutAnyOrigin
        ](gi_l_buf.unsafe_ptr())
        var pl = LayoutTensor[
            dtype, Layout.row_major(Self.Left.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var cl = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Left.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        var grads_l = LayoutTensor[
            dtype, Layout.row_major(Self.Left.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr)
        Self.Left.backward[BATCH](gl, gi_l, pl, cl, grads_l)

        # Backward Right → right portion of grad_input
        var gr = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_OUT), MutAnyOrigin
        ](gr_buf.unsafe_ptr())
        var gi_r_buf = InlineArray[Scalar[dtype], BATCH * Self.RIGHT_IN](
            uninitialized=True
        )
        var gi_r = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_IN), MutAnyOrigin
        ](gi_r_buf.unsafe_ptr())
        var pr = LayoutTensor[
            dtype, Layout.row_major(Self.Right.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self.Left.PARAM_SIZE)
        var cr = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Right.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr + BATCH * Self.Left.CACHE_SIZE)
        var grads_r = LayoutTensor[
            dtype, Layout.row_major(Self.Right.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr + Self.Left.PARAM_SIZE)
        Self.Right.backward[BATCH](gr, gi_r, pr, cr, grads_r)

        # Assemble grad_input from left + right
        for b in range(BATCH):
            for i in range(Self.LEFT_IN):
                grad_input.ptr[b * Self.IN_DIM + i] = gi_l_buf[
                    b * Self.LEFT_IN + i
                ]
            for i in range(Self.RIGHT_IN):
                grad_input.ptr[b * Self.IN_DIM + Self.split + i] = gi_r_buf[
                    b * Self.RIGHT_IN + i
                ]

    # =========================================================================
    # GPU (stubs — delegates to CPU pattern with device buffers)
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
        # Extract left/right slices via kernels
        var left_buf = ctx.enqueue_create_buffer[dtype](BATCH * Self.LEFT_IN)
        var right_buf = ctx.enqueue_create_buffer[dtype](BATCH * Self.RIGHT_IN)

        comptime L_TOTAL = BATCH * Self.LEFT_IN
        comptime R_TOTAL = BATCH * Self.RIGHT_IN
        var l_grid = (L_TOTAL + TPB - 1) // TPB
        var r_grid = (R_TOTAL + TPB - 1) // TPB

        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)

        var left_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_IN), MutAnyOrigin
        ](left_buf.unsafe_ptr())

        @always_inline
        fn extract_left(
            dst: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.LEFT_IN), MutAnyOrigin
            ],
            src: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= L_TOTAL:
                return
            var b = idx // Self.LEFT_IN
            var i = idx % Self.LEFT_IN
            dst.ptr[idx] = src.ptr[b * Self.IN_DIM + i]

        ctx.enqueue_function[extract_left, extract_left](
            left_t, input_immut, grid_dim=(l_grid,), block_dim=(TPB,)
        )

        var right_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_IN), MutAnyOrigin
        ](right_buf.unsafe_ptr())

        @always_inline
        fn extract_right(
            dst: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.RIGHT_IN), MutAnyOrigin
            ],
            src: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= R_TOTAL:
                return
            var b = idx // Self.RIGHT_IN
            var i = idx % Self.RIGHT_IN
            dst.ptr[idx] = src.ptr[b * Self.IN_DIM + Self.split + i]

        ctx.enqueue_function[extract_right, extract_right](
            right_t, input_immut, grid_dim=(r_grid,), block_dim=(TPB,)
        )

        # Forward Left and Right
        var left_out_buf = ctx.enqueue_create_buffer[dtype](
            BATCH * Self.LEFT_OUT
        )
        var right_out_buf = ctx.enqueue_create_buffer[dtype](
            BATCH * Self.RIGHT_OUT
        )
        var left_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_OUT), MutAnyOrigin
        ](left_out_buf.unsafe_ptr())
        var right_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_OUT), MutAnyOrigin
        ](right_out_buf.unsafe_ptr())

        var pl = LayoutTensor[
            dtype, Layout.row_major(Self.Left.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var cl = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Left.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        Self.Left.forward_gpu[BATCH](
            ctx, left_out_t, left_t, pl, cl, workspace
        )

        # Right params/cache: copy to aligned buffers to avoid misalignment
        # (Left.PARAM_SIZE may not be a multiple of 4, causing misaligned matmul)
        comptime R_PS = Self.Right.PARAM_SIZE
        comptime RP_BLOCKS = (R_PS + TPB - 1) // TPB

        var pr_buf = ctx.enqueue_create_buffer[dtype](R_PS)
        var pr = LayoutTensor[
            dtype, Layout.row_major(R_PS), MutAnyOrigin
        ](pr_buf.unsafe_ptr())

        @always_inline
        fn _sa_fwd_copy_params(
            dst: LayoutTensor[dtype, Layout.row_major(R_PS), MutAnyOrigin],
            src: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
            ],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i < R_PS:
                dst.ptr[i] = src.ptr[Self.Left.PARAM_SIZE + i]

        ctx.enqueue_function[_sa_fwd_copy_params, _sa_fwd_copy_params](
            pr, params, grid_dim=(RP_BLOCKS,), block_dim=(TPB,)
        )

        var cr_buf = ctx.enqueue_create_buffer[dtype](
            max(1, BATCH * Self.Right.CACHE_SIZE)
        )
        var cr = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Right.CACHE_SIZE), MutAnyOrigin
        ](cr_buf.unsafe_ptr())

        Self.Right.forward_gpu[BATCH](
            ctx, right_out_t, right_t, pr, cr, workspace
        )

        # Copy Right cache back to parent cache buffer at correct offset
        comptime R_CS_TOTAL = BATCH * Self.Right.CACHE_SIZE
        if R_CS_TOTAL > 0:
            comptime RC_BLOCKS = (R_CS_TOTAL + TPB - 1) // TPB

            @always_inline
            fn _sa_fwd_copy_cache_back(
                dst: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.CACHE_SIZE),
                    MutAnyOrigin,
                ],
                src: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Right.CACHE_SIZE),
                    MutAnyOrigin,
                ],
            ):
                var i = Int(block_dim.x * block_idx.x + thread_idx.x)
                if i < R_CS_TOTAL:
                    dst.ptr[BATCH * Self.Left.CACHE_SIZE + i] = src.ptr[i]

            ctx.enqueue_function[
                _sa_fwd_copy_cache_back, _sa_fwd_copy_cache_back
            ](cache, cr, grid_dim=(RC_BLOCKS,), block_dim=(TPB,))

        # Concat outputs
        var lo_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_OUT), ImmutAnyOrigin
        ](left_out_buf.unsafe_ptr())
        var ro_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_OUT), ImmutAnyOrigin
        ](right_out_buf.unsafe_ptr())
        comptime OUT_TOTAL = BATCH * Self.OUT_DIM
        var o_grid = (OUT_TOTAL + TPB - 1) // TPB

        @always_inline
        fn concat_out(
            dst: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            sl: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.LEFT_OUT), ImmutAnyOrigin
            ],
            sr: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.RIGHT_OUT), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= OUT_TOTAL:
                return
            var b = idx // Self.OUT_DIM
            var c = idx % Self.OUT_DIM
            if c < Self.LEFT_OUT:
                dst.ptr[idx] = sl.ptr[b * Self.LEFT_OUT + c]
            else:
                dst.ptr[idx] = sr.ptr[
                    b * Self.RIGHT_OUT + (c - Self.LEFT_OUT)
                ]

        ctx.enqueue_function[concat_out, concat_out](
            output, lo_immut, ro_immut, grid_dim=(o_grid,), block_dim=(TPB,)
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
        pass

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
        # Extract per-branch grad_outputs
        var gl_buf = ctx.enqueue_create_buffer[dtype](BATCH * Self.LEFT_OUT)
        var gr_buf = ctx.enqueue_create_buffer[dtype](BATCH * Self.RIGHT_OUT)
        var go_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)

        var gl_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_OUT), MutAnyOrigin
        ](gl_buf.unsafe_ptr())
        comptime GL_TOTAL = BATCH * Self.LEFT_OUT
        var gl_grid = (GL_TOTAL + TPB - 1) // TPB

        @always_inline
        fn extract_gl(
            dst: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.LEFT_OUT), MutAnyOrigin
            ],
            src: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= GL_TOTAL:
                return
            var b = idx // Self.LEFT_OUT
            var i = idx % Self.LEFT_OUT
            dst.ptr[idx] = src.ptr[b * Self.OUT_DIM + i]

        ctx.enqueue_function[extract_gl, extract_gl](
            gl_t, go_immut, grid_dim=(gl_grid,), block_dim=(TPB,)
        )

        var gr_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_OUT), MutAnyOrigin
        ](gr_buf.unsafe_ptr())
        comptime GR_TOTAL = BATCH * Self.RIGHT_OUT
        var gr_grid = (GR_TOTAL + TPB - 1) // TPB

        @always_inline
        fn extract_gr(
            dst: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.RIGHT_OUT), MutAnyOrigin
            ],
            src: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= GR_TOTAL:
                return
            var b = idx // Self.RIGHT_OUT
            var i = idx % Self.RIGHT_OUT
            dst.ptr[idx] = src.ptr[b * Self.OUT_DIM + Self.LEFT_OUT + i]

        ctx.enqueue_function[extract_gr, extract_gr](
            gr_t, go_immut, grid_dim=(gr_grid,), block_dim=(TPB,)
        )

        # Backward Left
        var gi_l_buf = ctx.enqueue_create_buffer[dtype](BATCH * Self.LEFT_IN)
        var gi_l_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_IN), MutAnyOrigin
        ](gi_l_buf.unsafe_ptr())
        var pl = LayoutTensor[
            dtype, Layout.row_major(Self.Left.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var cl = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Left.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        var grads_l = LayoutTensor[
            dtype, Layout.row_major(Self.Left.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr)
        Self.Left.backward_gpu[BATCH](
            ctx, gi_l_t, gl_t, pl, cl, grads_l, workspace
        )

        # Backward Right — copy params, cache, grads to aligned buffers
        var gi_r_buf = ctx.enqueue_create_buffer[dtype](BATCH * Self.RIGHT_IN)
        var gi_r_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_IN), MutAnyOrigin
        ](gi_r_buf.unsafe_ptr())

        comptime R_PS = Self.Right.PARAM_SIZE
        comptime R_CS_TOTAL = BATCH * Self.Right.CACHE_SIZE
        comptime RP_BLOCKS = (R_PS + TPB - 1) // TPB

        # Copy Right params to aligned buffer
        var pr_buf = ctx.enqueue_create_buffer[dtype](R_PS)
        var pr = LayoutTensor[
            dtype, Layout.row_major(R_PS), MutAnyOrigin
        ](pr_buf.unsafe_ptr())

        @always_inline
        fn _sa_bwd_copy_params(
            dst: LayoutTensor[dtype, Layout.row_major(R_PS), MutAnyOrigin],
            src: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
            ],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i < R_PS:
                dst.ptr[i] = src.ptr[Self.Left.PARAM_SIZE + i]

        ctx.enqueue_function[_sa_bwd_copy_params, _sa_bwd_copy_params](
            pr, params, grid_dim=(RP_BLOCKS,), block_dim=(TPB,)
        )

        # Copy Right cache to aligned buffer
        var cr_buf = ctx.enqueue_create_buffer[dtype](max(1, R_CS_TOTAL))
        var cr = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Right.CACHE_SIZE), MutAnyOrigin
        ](cr_buf.unsafe_ptr())
        if R_CS_TOTAL > 0:
            comptime RC_BLOCKS = (R_CS_TOTAL + TPB - 1) // TPB

            @always_inline
            fn _sa_bwd_copy_cache(
                dst: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Right.CACHE_SIZE),
                    MutAnyOrigin,
                ],
                src: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.CACHE_SIZE),
                    MutAnyOrigin,
                ],
            ):
                var i = Int(block_dim.x * block_idx.x + thread_idx.x)
                if i < R_CS_TOTAL:
                    dst.ptr[i] = src.ptr[BATCH * Self.Left.CACHE_SIZE + i]

            ctx.enqueue_function[_sa_bwd_copy_cache, _sa_bwd_copy_cache](
                cr, cache, grid_dim=(RC_BLOCKS,), block_dim=(TPB,)
            )

        # Aligned grads buffer for Right, zeroed
        var grads_r_buf = ctx.enqueue_create_buffer[dtype](R_PS)
        var grads_r = LayoutTensor[
            dtype, Layout.row_major(R_PS), MutAnyOrigin
        ](grads_r_buf.unsafe_ptr())
        var zero_r = ctx.enqueue_create_host_buffer[dtype](R_PS)
        for i in range(R_PS):
            zero_r[i] = Scalar[dtype](0.0)
        ctx.enqueue_copy(grads_r_buf, zero_r)

        Self.Right.backward_gpu[BATCH](
            ctx, gi_r_t, gr_t, pr, cr, grads_r, workspace
        )

        # Copy Right grads back to parent grads buffer at correct offset
        @always_inline
        fn _sa_bwd_scatter_grads(
            dst: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
            ],
            src: LayoutTensor[dtype, Layout.row_major(R_PS), MutAnyOrigin],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i < R_PS:
                dst.ptr[Self.Left.PARAM_SIZE + i] = src.ptr[i]

        ctx.enqueue_function[_sa_bwd_scatter_grads, _sa_bwd_scatter_grads](
            grads, grads_r, grid_dim=(RP_BLOCKS,), block_dim=(TPB,)
        )

        # Assemble grad_input
        var gi_l_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_IN), ImmutAnyOrigin
        ](gi_l_buf.unsafe_ptr())
        var gi_r_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_IN), ImmutAnyOrigin
        ](gi_r_buf.unsafe_ptr())
        comptime GI_TOTAL = BATCH * Self.IN_DIM
        var gi_grid = (GI_TOTAL + TPB - 1) // TPB

        @always_inline
        fn assemble_gi(
            dst: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            sl: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.LEFT_IN), ImmutAnyOrigin
            ],
            sr: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.RIGHT_IN), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= GI_TOTAL:
                return
            var b = idx // Self.IN_DIM
            var c = idx % Self.IN_DIM
            if c < Self.split:
                dst.ptr[idx] = sl.ptr[b * Self.LEFT_IN + c]
            else:
                dst.ptr[idx] = sr.ptr[
                    b * Self.RIGHT_IN + (c - Self.split)
                ]

        ctx.enqueue_function[assemble_gi, assemble_gi](
            grad_input,
            gi_l_immut,
            gi_r_immut,
            grid_dim=(gi_grid,),
            block_dim=(TPB,),
        )
