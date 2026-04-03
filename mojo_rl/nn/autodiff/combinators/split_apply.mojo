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

Param layout uses 4-element alignment padding between Left and Right params
to guarantee GPU matmul alignment (16 bytes for float32).
"""

from ...constants import dtype, TPB
from ...model.model import Model, PerfTimerPtr, NULL_PERF
from ...initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream


# GPU matmul requires 16-byte alignment = 4 float32 elements
@always_inline
def _align4(x: Int) -> Int:
    """Round up to next multiple of 4 for GPU alignment."""
    return (x + 3) & ~3


@fieldwise_init
struct SplitApply[Left: Model, Right: Model, split: Int](Model):
    """Split input at `split`, apply Left to [:split] and Right to [split:].

    IN_DIM = Left.IN_DIM + Right.IN_DIM  (= split + remaining)
    OUT_DIM = Left.OUT_DIM + Right.OUT_DIM
    PARAM_SIZE includes alignment padding between Left and Right params.
    """

    comptime IN_DIM: Int = Self.Left.IN_DIM + Self.Right.IN_DIM
    comptime OUT_DIM: Int = Self.Left.OUT_DIM + Self.Right.OUT_DIM

    # Aligned param layout: [Left_params (padded to 4) | Right_params]
    comptime _RIGHT_PARAM_OFF: Int = _align4(Self.Left.PARAM_SIZE)
    comptime PARAM_SIZE: Int = Self._RIGHT_PARAM_OFF + Self.Right.PARAM_SIZE

    comptime CACHE_SIZE: Int = Self.Left.CACHE_SIZE + Self.Right.CACHE_SIZE
    # Own scratch: split/concat temporaries shared between forward and backward
    comptime _OWN_WS: Int = (
        Self.Left.IN_DIM
        + Self.Right.IN_DIM  # split input / grad_input
        + Self.Left.OUT_DIM
        + Self.Right.OUT_DIM  # branch outputs / grad_outputs
    )
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = (
        Self._OWN_WS
        + Self.Left.WORKSPACE_SIZE_PER_SAMPLE
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
    def initialize_params[
        INIT: Initializer, dtype: DType = DType.float32
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        # Zero padding region
        for i in range(Self.Left.PARAM_SIZE, Self._RIGHT_PARAM_OFF):
            params.ptr[i] = Scalar[dtype](0.0)

        var pl = LayoutTensor[
            dtype, Layout.row_major(Self.Left.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        Self.Left.initialize_params[INIT, dtype](pl)
        var pr = LayoutTensor[
            dtype, Layout.row_major(Self.Right.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self._RIGHT_PARAM_OFF)
        Self.Right.initialize_params[INIT, dtype](pr)

    # =========================================================================
    # CPU Forward (with cache)
    # =========================================================================

    @staticmethod
    def forward[
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
        var left_out_buf = InlineArray[Scalar[dtype], BATCH * Self.LEFT_OUT](
            uninitialized=True
        )
        var left_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_OUT), MutAnyOrigin
        ](left_out_buf.unsafe_ptr())
        var pl = LayoutTensor[
            dtype, Layout.row_major(Self.Left.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var cl = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Left.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        Self.Left.forward[BATCH, dtype](left_in, left_out, pl, cl)

        # Forward Right
        var right_in = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_IN), MutAnyOrigin
        ](right_in_buf.unsafe_ptr())
        var right_out_buf = InlineArray[Scalar[dtype], BATCH * Self.RIGHT_OUT](
            uninitialized=True
        )
        var right_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_OUT), MutAnyOrigin
        ](right_out_buf.unsafe_ptr())
        var pr = LayoutTensor[
            dtype, Layout.row_major(Self.Right.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self._RIGHT_PARAM_OFF)
        var cr = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Right.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr + BATCH * Self.Left.CACHE_SIZE)
        Self.Right.forward[BATCH, dtype](right_in, right_out, pr, cr)

        # Concat outputs
        for b in range(BATCH):
            for i in range(Self.LEFT_OUT):
                output.ptr[b * Self.OUT_DIM + i] = left_out_buf[
                    b * Self.LEFT_OUT + i
                ]
            for i in range(Self.RIGHT_OUT):
                output.ptr[
                    b * Self.OUT_DIM + Self.LEFT_OUT + i
                ] = right_out_buf[b * Self.RIGHT_OUT + i]

    # =========================================================================
    # CPU Forward (no cache)
    # =========================================================================

    @staticmethod
    def forward[
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
        var left_out_buf = InlineArray[Scalar[dtype], BATCH * Self.LEFT_OUT](
            uninitialized=True
        )
        var left_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_OUT), MutAnyOrigin
        ](left_out_buf.unsafe_ptr())
        var pl = LayoutTensor[
            dtype, Layout.row_major(Self.Left.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        Self.Left.forward[BATCH, dtype](left_in, left_out, pl)

        var right_in = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_IN), MutAnyOrigin
        ](right_in_buf.unsafe_ptr())
        var right_out_buf = InlineArray[Scalar[dtype], BATCH * Self.RIGHT_OUT](
            uninitialized=True
        )
        var right_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_OUT), MutAnyOrigin
        ](right_out_buf.unsafe_ptr())
        var pr = LayoutTensor[
            dtype, Layout.row_major(Self.Right.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self._RIGHT_PARAM_OFF)
        Self.Right.forward[BATCH, dtype](right_in, right_out, pr)

        for b in range(BATCH):
            for i in range(Self.LEFT_OUT):
                output.ptr[b * Self.OUT_DIM + i] = left_out_buf[
                    b * Self.LEFT_OUT + i
                ]
            for i in range(Self.RIGHT_OUT):
                output.ptr[
                    b * Self.OUT_DIM + Self.LEFT_OUT + i
                ] = right_out_buf[b * Self.RIGHT_OUT + i]

    # =========================================================================
    # CPU Backward
    # =========================================================================

    @staticmethod
    def backward[
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
        Self.Left.backward[BATCH, dtype](gl, gi_l, pl, cl, grads_l)

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
        ](params.ptr + Self._RIGHT_PARAM_OFF)
        var cr = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Right.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr + BATCH * Self.Left.CACHE_SIZE)
        var grads_r = LayoutTensor[
            dtype, Layout.row_major(Self.Right.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr + Self._RIGHT_PARAM_OFF)
        Self.Right.backward[BATCH, dtype](gr, gi_r, pr, cr, grads_r)

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
    def forward_gpu[
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
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        # Slice workspace: [left_in | right_in | left_out | right_out | child_ws]
        var ws_ptr = workspace.unsafe_ptr()
        var left_ptr = ws_ptr
        var right_ptr = ws_ptr + BATCH * Self.LEFT_IN
        var left_out_ptr = right_ptr + BATCH * Self.RIGHT_IN
        var right_out_ptr = left_out_ptr + BATCH * Self.LEFT_OUT
        var child_ws_ptr = right_out_ptr + BATCH * Self.RIGHT_OUT
        comptime CHILD_WS_SIZE = max(
            1,
            BATCH
            * (
                Self.Left.WORKSPACE_SIZE_PER_SAMPLE
                + Self.Right.WORKSPACE_SIZE_PER_SAMPLE
            ),
        )
        var child_ws = DeviceBuffer[dtype](
            ctx, child_ws_ptr, CHILD_WS_SIZE, owning=False
        )

        # Extract left/right slices via kernels
        comptime L_TOTAL = BATCH * Self.LEFT_IN
        comptime R_TOTAL = BATCH * Self.RIGHT_IN
        var l_grid = (L_TOTAL + TPB - 1) // TPB
        var r_grid = (R_TOTAL + TPB - 1) // TPB

        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)

        var left_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_IN), MutAnyOrigin
        ](left_ptr)

        @always_inline
        def extract_left(
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
        ](right_ptr)

        @always_inline
        def extract_right(
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
        var left_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_OUT), MutAnyOrigin
        ](left_out_ptr)
        var right_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_OUT), MutAnyOrigin
        ](right_out_ptr)

        # Left: params at offset 0 (always aligned)
        var pl = LayoutTensor[
            dtype, Layout.row_major(Self.Left.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var cl = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Left.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        Self.Left.forward_gpu[BATCH, dtype](ctx, left_out_t, left_t, pl, cl, child_ws)

        # Right: params at aligned offset — safe for direct pointer access
        comptime R_PS = Self.Right.PARAM_SIZE
        var pr = LayoutTensor[dtype, Layout.row_major(R_PS), MutAnyOrigin](
            params.ptr + Self._RIGHT_PARAM_OFF
        )  # Aligned!
        var cr = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Right.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr + BATCH * Self.Left.CACHE_SIZE)

        Self.Right.forward_gpu[BATCH, dtype](
            ctx, right_out_t, right_t, pr, cr, child_ws
        )

        # Concat outputs
        var lo_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_OUT), ImmutAnyOrigin
        ](left_out_ptr)
        var ro_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_OUT), ImmutAnyOrigin
        ](right_out_ptr)
        comptime OUT_TOTAL = BATCH * Self.OUT_DIM
        var o_grid = (OUT_TOTAL + TPB - 1) // TPB

        @always_inline
        def concat_out(
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
                dst.ptr[idx] = sr.ptr[b * Self.RIGHT_OUT + (c - Self.LEFT_OUT)]

        ctx.enqueue_function[concat_out, concat_out](
            output, lo_immut, ro_immut, grid_dim=(o_grid,), block_dim=(TPB,)
        )

    @staticmethod
    def forward_gpu_no_cache[
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
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        pass

    @staticmethod
    def forward_gpu_no_cache_on_stream[
        BATCH: Int, dtype: DType = DType.float32
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
    def backward_gpu[
        BATCH: Int, dtype: DType = DType.float32
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
        # Slice workspace: reuse same layout as forward
        # [left_in/gi_l | right_in/gi_r | left_out/gl | right_out/gr | child_ws]
        var ws_ptr = workspace.unsafe_ptr()
        var gi_l_ptr = ws_ptr
        var gi_r_ptr = ws_ptr + BATCH * Self.LEFT_IN
        var gl_ptr = gi_r_ptr + BATCH * Self.RIGHT_IN
        var gr_ptr = gl_ptr + BATCH * Self.LEFT_OUT
        var child_ws_ptr = gr_ptr + BATCH * Self.RIGHT_OUT
        comptime CHILD_WS_SIZE = max(
            1,
            BATCH
            * (
                Self.Left.WORKSPACE_SIZE_PER_SAMPLE
                + Self.Right.WORKSPACE_SIZE_PER_SAMPLE
            ),
        )
        var child_ws = DeviceBuffer[dtype](
            ctx, child_ws_ptr, CHILD_WS_SIZE, owning=False
        )

        # Extract per-branch grad_outputs
        var go_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)

        var gl_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_OUT), MutAnyOrigin
        ](gl_ptr)
        comptime GL_TOTAL = BATCH * Self.LEFT_OUT
        var gl_grid = (GL_TOTAL + TPB - 1) // TPB

        @always_inline
        def extract_gl(
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
        ](gr_ptr)
        comptime GR_TOTAL = BATCH * Self.RIGHT_OUT
        var gr_grid = (GR_TOTAL + TPB - 1) // TPB

        @always_inline
        def extract_gr(
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

        # Backward Left — params/cache/grads at offset 0 (always aligned)
        var gi_l_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_IN), MutAnyOrigin
        ](gi_l_ptr)
        var pl = LayoutTensor[
            dtype, Layout.row_major(Self.Left.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var cl = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Left.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        var grads_l = LayoutTensor[
            dtype, Layout.row_major(Self.Left.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr)
        Self.Left.backward_gpu[BATCH, dtype](
            ctx, gi_l_t, gl_t, pl, cl, grads_l, child_ws
        )

        # Backward Right — params/grads at aligned offset
        var gi_r_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_IN), MutAnyOrigin
        ](gi_r_ptr)

        var pr = LayoutTensor[
            dtype, Layout.row_major(Self.Right.PARAM_SIZE), MutAnyOrigin
        ](
            params.ptr + Self._RIGHT_PARAM_OFF
        )  # Aligned!
        var cr = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Right.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr + BATCH * Self.Left.CACHE_SIZE)
        var grads_r = LayoutTensor[
            dtype, Layout.row_major(Self.Right.PARAM_SIZE), MutAnyOrigin
        ](
            grads.ptr + Self._RIGHT_PARAM_OFF
        )  # Aligned!
        Self.Right.backward_gpu[BATCH, dtype](
            ctx, gi_r_t, gr_t, pr, cr, grads_r, child_ws
        )

        # Assemble grad_input
        var gi_l_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.LEFT_IN), ImmutAnyOrigin
        ](gi_l_ptr)
        var gi_r_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.RIGHT_IN), ImmutAnyOrigin
        ](gi_r_ptr)
        comptime GI_TOTAL = BATCH * Self.IN_DIM
        var gi_grid = (GI_TOTAL + TPB - 1) // TPB

        @always_inline
        def assemble_gi(
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
                dst.ptr[idx] = sr.ptr[b * Self.RIGHT_IN + (c - Self.split)]

        ctx.enqueue_function[assemble_gi, assemble_gi](
            grad_input,
            gi_l_immut,
            gi_r_immut,
            grid_dim=(gi_grid,),
            block_dim=(TPB,),
        )
