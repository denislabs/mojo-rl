"""DualPath combinator: runs two Models on the same input, concatenates outputs.

Forward:  output = concat(A(input), B(input))
Backward: grad_input = A.backward(grad_A) + B.backward(grad_B)

This is like Parallel but at the Model level (Parallel works at DiffOp level).
Used to forward through twin critics on the same critic_input:

    critic_input → DualPath[Critic1, Critic2] → [Q1 || Q2]
    [Q1 || Q2] → MinOp → min_Q

The backward automatically routes gradients to the correct critic and sums
the grad_inputs (which are the same critic_input, so the sum is correct).

Param layout uses 4-element alignment padding between A and B params to
guarantee GPU matmul alignment (16 bytes for float32).
"""

from ...constants import dtype, TPB, gpu_align
from ...model.model import Model, PerfTimerPtr, NULL_PERF
from ...initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream


# GPU matmul requires 16-byte alignment = 4 float32 elements
@always_inline
def _align4(x: Int) -> Int:
    """GPU-aligned element count (16-byte aligned for any dtype)."""
    return gpu_align(x)


@fieldwise_init
struct DualPath[A: Model, B: Model](Model):
    """Run two models on same input, concatenate outputs.

    Requires: A.IN_DIM == B.IN_DIM
    OUT_DIM = A.OUT_DIM + B.OUT_DIM
    PARAM_SIZE includes alignment padding between A and B params.
    """

    comptime IN_DIM: Int = Self.A.IN_DIM
    comptime OUT_DIM: Int = Self.A.OUT_DIM + Self.B.OUT_DIM

    # Aligned param layout: [A_params (padded to 4) | B_params]
    comptime _B_PARAM_OFF: Int = _align4(Self.A.PARAM_SIZE)
    comptime PARAM_SIZE: Int = Self._B_PARAM_OFF + Self.B.PARAM_SIZE

    comptime CACHE_SIZE: Int = Self.A.CACHE_SIZE + Self.B.CACHE_SIZE
    comptime STATE_SIZE: Int = Self.A.STATE_SIZE + Self.B.STATE_SIZE
    # Own scratch: shared between forward and backward
    # Forward:  a_out(A.OUT_DIM) + b_out(B.OUT_DIM)
    # Backward: ga(A.OUT_DIM) + gb(B.OUT_DIM) + gi_b(IN_DIM)  ← larger
    comptime _OWN_WS: Int = Self.A.OUT_DIM + Self.B.OUT_DIM + Self.IN_DIM
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = (
        Self._OWN_WS
        + Self.A.WORKSPACE_SIZE_PER_SAMPLE
        + Self.B.WORKSPACE_SIZE_PER_SAMPLE
    )

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
        for i in range(Self.A.PARAM_SIZE, Self._B_PARAM_OFF):
            params.ptr[i] = Scalar[dtype](0.0)

        var pa = LayoutTensor[
            dtype, Layout.row_major(Self.A.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        Self.A.initialize_params[INIT, dtype](pa)

        var pb = LayoutTensor[
            dtype, Layout.row_major(Self.B.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self._B_PARAM_OFF)
        Self.B.initialize_params[INIT, dtype](pb)

    @staticmethod
    def initialize_state[dtype: DType = DType.float32](
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
    ):
        """Recurse into A and B initialize_state (scalar offsets, no padding)."""
        comptime if Self.A.STATE_SIZE > 0:
            var sa = LayoutTensor[
                dtype, Layout.row_major(Self.A.STATE_SIZE), MutAnyOrigin
            ](state.ptr)
            Self.A.initialize_state[dtype](sa)
        comptime if Self.B.STATE_SIZE > 0:
            var sb = LayoutTensor[
                dtype, Layout.row_major(Self.B.STATE_SIZE), MutAnyOrigin
            ](state.ptr + Self.A.STATE_SIZE)
            Self.B.initialize_state[dtype](sb)

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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
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
        var sa = LayoutTensor[
            dtype, Layout.row_major(Self.A.STATE_SIZE), MutAnyOrigin
        ](state.ptr)
        var ca = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.A.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        Self.A.forward[BATCH, dtype](input, a_out, pa, sa, ca)

        # Forward B (params at aligned offset)
        var b_buf = InlineArray[Scalar[dtype], BATCH * B_OUT](
            uninitialized=True
        )
        var b_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, B_OUT), MutAnyOrigin
        ](b_buf.unsafe_ptr())
        var pb = LayoutTensor[
            dtype, Layout.row_major(Self.B.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self._B_PARAM_OFF)
        var sb = LayoutTensor[
            dtype, Layout.row_major(Self.B.STATE_SIZE), MutAnyOrigin
        ](state.ptr + Self.A.STATE_SIZE)
        var cb = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.B.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr + BATCH * Self.A.CACHE_SIZE)
        var input_b = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.B.IN_DIM), MutAnyOrigin
        ](input.ptr)
        Self.B.forward[BATCH, dtype](input_b, b_out, pb, sb, cb)

        # Concat outputs
        for b in range(BATCH):
            for i in range(A_OUT):
                output.ptr[b * Self.OUT_DIM + i] = a_buf[b * A_OUT + i]
            for i in range(B_OUT):
                output.ptr[b * Self.OUT_DIM + A_OUT + i] = b_buf[b * B_OUT + i]

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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
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
        var sa = LayoutTensor[
            dtype, Layout.row_major(Self.A.STATE_SIZE), MutAnyOrigin
        ](state.ptr)
        Self.A.forward[BATCH, dtype](input, a_out, pa, sa)

        var b_buf = InlineArray[Scalar[dtype], BATCH * B_OUT](
            uninitialized=True
        )
        var b_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, B_OUT), MutAnyOrigin
        ](b_buf.unsafe_ptr())
        var pb = LayoutTensor[
            dtype, Layout.row_major(Self.B.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self._B_PARAM_OFF)
        var sb = LayoutTensor[
            dtype, Layout.row_major(Self.B.STATE_SIZE), MutAnyOrigin
        ](state.ptr + Self.A.STATE_SIZE)
        var input_b = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.B.IN_DIM), MutAnyOrigin
        ](input.ptr)
        Self.B.forward[BATCH, dtype](input_b, b_out, pb, sb)

        for b in range(BATCH):
            for i in range(A_OUT):
                output.ptr[b * Self.OUT_DIM + i] = a_buf[b * A_OUT + i]
            for i in range(B_OUT):
                output.ptr[b * Self.OUT_DIM + A_OUT + i] = b_buf[b * B_OUT + i]

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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
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
                ga_buf[b * A_OUT + i] = grad_output.ptr[b * Self.OUT_DIM + i]
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
        var sa = LayoutTensor[
            dtype, Layout.row_major(Self.A.STATE_SIZE), MutAnyOrigin
        ](state.ptr)
        var ca = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.A.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        var grads_a = LayoutTensor[
            dtype, Layout.row_major(Self.A.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr)
        Self.A.backward[BATCH, dtype](ga, grad_input, pa, sa, ca, grads_a)

        # Backward B → temp grad_input, then add
        var gi_b_buf = InlineArray[Scalar[dtype], BATCH * Self.IN_DIM](
            uninitialized=True
        )
        var gb = LayoutTensor[
            dtype, Layout.row_major(BATCH, B_OUT), MutAnyOrigin
        ](gb_buf.unsafe_ptr())
        var pb = LayoutTensor[
            dtype, Layout.row_major(Self.B.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self._B_PARAM_OFF)
        var sb = LayoutTensor[
            dtype, Layout.row_major(Self.B.STATE_SIZE), MutAnyOrigin
        ](state.ptr + Self.A.STATE_SIZE)
        var cb = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.B.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr + BATCH * Self.A.CACHE_SIZE)
        var grads_b = LayoutTensor[
            dtype, Layout.row_major(Self.B.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr + Self._B_PARAM_OFF)
        var gi_b_rb = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.B.IN_DIM), MutAnyOrigin
        ](gi_b_buf.unsafe_ptr())
        Self.B.backward[BATCH, dtype](gb, gi_b_rb, pb, sb, cb, grads_b)

        # Sum grad_inputs
        for i in range(BATCH * Self.IN_DIM):
            grad_input.ptr[i] = grad_input.ptr[i] + gi_b_buf[i]

    # =========================================================================
    # GPU Forward (with cache)
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
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

        # Slice workspace: [a_out(A_OUT) | b_out(B_OUT) | gi_b(IN_DIM) | child_ws]
        var ws_ptr = workspace.unsafe_ptr()
        var a_out_ptr = ws_ptr
        var b_out_ptr = ws_ptr + BATCH * A_OUT
        var child_ws_ptr = b_out_ptr + BATCH * B_OUT + BATCH * Self.IN_DIM
        comptime CHILD_WS_SIZE = max(
            1,
            BATCH
            * (
                Self.A.WORKSPACE_SIZE_PER_SAMPLE
                + Self.B.WORKSPACE_SIZE_PER_SAMPLE
            ),
        )
        var child_ws = DeviceBuffer[dtype](
            ctx, child_ws_ptr, CHILD_WS_SIZE, owning=False
        )

        var a_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, A_OUT), MutAnyOrigin
        ](a_out_ptr)
        var b_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, B_OUT), MutAnyOrigin
        ](b_out_ptr)

        # Params at aligned offsets — safe for direct pointer access
        var pa = LayoutTensor[
            dtype, Layout.row_major(Self.A.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var pb = LayoutTensor[
            dtype, Layout.row_major(Self.B.PARAM_SIZE), MutAnyOrigin
        ](
            params.ptr + Self._B_PARAM_OFF
        )  # Aligned!

        var sa = LayoutTensor[
            dtype, Layout.row_major(Self.A.STATE_SIZE), MutAnyOrigin
        ](state.ptr)
        var sb = LayoutTensor[
            dtype, Layout.row_major(Self.B.STATE_SIZE), MutAnyOrigin
        ](state.ptr + Self.A.STATE_SIZE)

        var ca = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.A.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        var cb = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.B.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr + BATCH * Self.A.CACHE_SIZE)

        # Forward both
        Self.A.forward_gpu[BATCH, dtype](ctx, a_out_t, input, pa, sa, ca, child_ws)
        var input_b = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.B.IN_DIM), MutAnyOrigin
        ](input.ptr)
        Self.B.forward_gpu[BATCH, dtype](ctx, b_out_t, input_b, pb, sb, cb, child_ws)

        # Concat into output
        var a_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, A_OUT), ImmutAnyOrigin
        ](a_out_ptr)
        var b_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, B_OUT), ImmutAnyOrigin
        ](b_out_ptr)
        comptime TOTAL = BATCH * Self.OUT_DIM
        var grid_x = (TOTAL + TPB - 1) // TPB

        @parameter
        @always_inline
        def concat_k(
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        Self.forward_gpu_no_cache[BATCH, dtype](ctx, output, input, params, state, workspace)

    # =========================================================================
    # GPU Backward
    # =========================================================================

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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
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

        # Slice workspace: [ga(A_OUT) | gb(B_OUT) | gi_b(IN_DIM) | child_ws]
        var ws_ptr = workspace.unsafe_ptr()
        var ga_ptr = ws_ptr
        var gb_ptr = ws_ptr + BATCH * A_OUT
        var gi_b_ptr = gb_ptr + BATCH * B_OUT
        var child_ws_ptr = gi_b_ptr + BATCH * Self.IN_DIM
        comptime CHILD_WS_SIZE = max(
            1,
            BATCH
            * (
                Self.A.WORKSPACE_SIZE_PER_SAMPLE
                + Self.B.WORKSPACE_SIZE_PER_SAMPLE
            ),
        )
        var child_ws = DeviceBuffer[dtype](
            ctx, child_ws_ptr, CHILD_WS_SIZE, owning=False
        )

        var ga_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, A_OUT), MutAnyOrigin
        ](ga_ptr)
        var gb_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, B_OUT), MutAnyOrigin
        ](gb_ptr)

        var go_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)

        # Extract A grad
        comptime A_TOTAL = BATCH * A_OUT
        var a_grid = (A_TOTAL + TPB - 1) // TPB

        @parameter
        @always_inline
        def extract_a(
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

        @parameter
        @always_inline
        def extract_b(
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

        # Backward A — params/cache/grads at offset 0 (always aligned)
        var pa = LayoutTensor[
            dtype, Layout.row_major(Self.A.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var sa = LayoutTensor[
            dtype, Layout.row_major(Self.A.STATE_SIZE), MutAnyOrigin
        ](state.ptr)
        var ca = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.A.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        var grads_a = LayoutTensor[
            dtype, Layout.row_major(Self.A.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr)
        Self.A.backward_gpu[BATCH, dtype](
            ctx, grad_input, ga_t, pa, sa, ca, grads_a, child_ws
        )

        # Backward B — params/grads at aligned offset
        var gi_b_rb = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.B.IN_DIM), MutAnyOrigin
        ](gi_b_ptr)
        var pb = LayoutTensor[
            dtype, Layout.row_major(Self.B.PARAM_SIZE), MutAnyOrigin
        ](
            params.ptr + Self._B_PARAM_OFF
        )  # Aligned!
        var sb = LayoutTensor[
            dtype, Layout.row_major(Self.B.STATE_SIZE), MutAnyOrigin
        ](state.ptr + Self.A.STATE_SIZE)
        var cb = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.B.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr + BATCH * Self.A.CACHE_SIZE)
        var grads_b = LayoutTensor[
            dtype, Layout.row_major(Self.B.PARAM_SIZE), MutAnyOrigin
        ](
            grads.ptr + Self._B_PARAM_OFF
        )  # Aligned!
        Self.B.backward_gpu[BATCH, dtype](
            ctx, gi_b_rb, gb_t, pb, sb, cb, grads_b, child_ws
        )

        # Add B's grad_input to A's
        var gi_b_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](gi_b_ptr)
        comptime GI_TOTAL = BATCH * Self.IN_DIM
        var gi_grid = (GI_TOTAL + TPB - 1) // TPB

        @parameter
        @always_inline
        def add_gi(
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
