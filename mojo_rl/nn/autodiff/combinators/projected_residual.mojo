"""ProjectedResidual: residual connection with a learnable projection skip.

ProjectedResidual[Inner, Skip] computes y = Inner(x) + Skip(x).

Unlike `Residual` (identity skip, requires IN_DIM == OUT_DIM), this combinator
takes a Skip Model on the shortcut path. Used for ResNet downsample blocks
where the main path changes spatial dims and/or channel count, so the skip
must apply a 1×1 stride-s conv (option B in He et al., 2016).

Constraints:
    Inner.IN_DIM  == Skip.IN_DIM    (both consume the same input)
    Inner.OUT_DIM == Skip.OUT_DIM   (outputs are added elementwise)

Forward:  output = Inner(input) + Skip(input)
Backward: grad_input = Inner.backward(grad_output) + Skip.backward(grad_output)
          (grad_output flows identically to both paths — no extract step)

Param layout: [Inner_params (4-aligned padded) | Skip_params]
Cache layout: [Inner_cache | Skip_cache]
State layout: [Inner_state | Skip_state]
"""

from ...constants import dtype, TPB, gpu_align
from ...model.model import Model, PerfTimerPtr, NULL_PERF
from ...initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream


@always_inline
def _align4(x: Int) -> Int:
    """GPU-aligned element count (16-byte aligned for any dtype)."""
    return gpu_align(x)


@fieldwise_init
struct ProjectedResidual[Inner: Model, Skip: Model](Model):
    """Residual block with a learnable projection on the skip path.

    Computes y = Inner(x) + Skip(x). Both paths receive the same input;
    their outputs are summed elementwise.

    Use case: ResNet downsample block —
        Inner = Conv3x3-BN-ReLU(stride=2) → Conv3x3-BN(stride=1)
        Skip  = Conv1x1-BN(stride=2)
        ReLU is applied externally on the sum.
    """

    comptime IN_DIM: Int = Self.Inner.IN_DIM
    comptime OUT_DIM: Int = Self.Inner.OUT_DIM

    # Aligned param layout: [Inner_params (padded to 4) | Skip_params]
    comptime _SKIP_PARAM_OFF: Int = _align4(Self.Inner.PARAM_SIZE)
    comptime PARAM_SIZE: Int = Self._SKIP_PARAM_OFF + Self.Skip.PARAM_SIZE

    comptime CACHE_SIZE: Int = Self.Inner.CACHE_SIZE + Self.Skip.CACHE_SIZE
    comptime STATE_SIZE: Int = Self.Inner.STATE_SIZE + Self.Skip.STATE_SIZE

    # Own scratch (per sample): one OUT_DIM slot for skip_out scratch
    # (forward) + one IN_DIM slot for skip's grad_input scratch (backward).
    # Forward and backward never overlap in time; we allocate the union
    # to keep slicing simple.
    comptime _OWN_WS: Int = Self.OUT_DIM + Self.IN_DIM
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = (
        Self._OWN_WS
        + Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
        + Self.Skip.WORKSPACE_SIZE_PER_SAMPLE
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
        # Zero padding region between Inner and Skip params
        for i in range(Self.Inner.PARAM_SIZE, Self._SKIP_PARAM_OFF):
            params.ptr[i] = Scalar[dtype](0.0)

        var p_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        Self.Inner.initialize_params[INIT, dtype](p_inner)

        var p_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self._SKIP_PARAM_OFF)
        Self.Skip.initialize_params[INIT, dtype](p_skip)

    @staticmethod
    def initialize_state[dtype: DType = DType.float32](
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
    ):
        """Recurse into Inner and Skip initialize_state (scalar offsets)."""
        comptime if Self.Inner.STATE_SIZE > 0:
            var s_inner = LayoutTensor[
                dtype, Layout.row_major(Self.Inner.STATE_SIZE), MutAnyOrigin
            ](state.ptr)
            Self.Inner.initialize_state[dtype](s_inner)
        comptime if Self.Skip.STATE_SIZE > 0:
            var s_skip = LayoutTensor[
                dtype, Layout.row_major(Self.Skip.STATE_SIZE), MutAnyOrigin
            ](state.ptr + Self.Inner.STATE_SIZE)
            Self.Skip.initialize_state[dtype](s_skip)

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
        comptime SK_OUT = Self.Skip.OUT_DIM

        # Inner writes directly into output
        var p_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var s_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.STATE_SIZE), MutAnyOrigin
        ](state.ptr)
        var c_inner = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Inner.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        Self.Inner.forward[BATCH, dtype](input, output, p_inner, s_inner, c_inner)

        # Skip writes to a stack scratch buffer
        var skip_buf = InlineArray[Scalar[dtype], BATCH * SK_OUT](
            uninitialized=True
        )
        var skip_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, SK_OUT), MutAnyOrigin
        ](skip_buf.unsafe_ptr())
        var p_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self._SKIP_PARAM_OFF)
        var s_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.STATE_SIZE), MutAnyOrigin
        ](state.ptr + Self.Inner.STATE_SIZE)
        var c_skip = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Skip.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr + BATCH * Self.Inner.CACHE_SIZE)
        var input_skip = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Skip.IN_DIM), MutAnyOrigin
        ](input.ptr)
        Self.Skip.forward[BATCH, dtype](input_skip, skip_out, p_skip, s_skip, c_skip)

        # output += skip
        for i in range(BATCH * Self.OUT_DIM):
            output.ptr[i] = output.ptr[i] + skip_buf[i]

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
        comptime SK_OUT = Self.Skip.OUT_DIM

        var p_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var s_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.STATE_SIZE), MutAnyOrigin
        ](state.ptr)
        Self.Inner.forward[BATCH, dtype](input, output, p_inner, s_inner)

        var skip_buf = InlineArray[Scalar[dtype], BATCH * SK_OUT](
            uninitialized=True
        )
        var skip_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, SK_OUT), MutAnyOrigin
        ](skip_buf.unsafe_ptr())
        var p_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self._SKIP_PARAM_OFF)
        var s_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.STATE_SIZE), MutAnyOrigin
        ](state.ptr + Self.Inner.STATE_SIZE)
        var input_skip = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Skip.IN_DIM), MutAnyOrigin
        ](input.ptr)
        Self.Skip.forward[BATCH, dtype](input_skip, skip_out, p_skip, s_skip)

        for i in range(BATCH * Self.OUT_DIM):
            output.ptr[i] = output.ptr[i] + skip_buf[i]

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
        comptime SK_IN = Self.Skip.IN_DIM

        # Inner backward → grad_input (overwrites)
        var p_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var s_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.STATE_SIZE), MutAnyOrigin
        ](state.ptr)
        var c_inner = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Inner.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        var grads_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr)
        Self.Inner.backward[BATCH, dtype](
            grad_output, grad_input, p_inner, s_inner, c_inner, grads_inner
        )

        # Skip backward → temp grad_input, then add
        var gi_skip_buf = InlineArray[Scalar[dtype], BATCH * SK_IN](
            uninitialized=True
        )
        var gi_skip = LayoutTensor[
            dtype, Layout.row_major(BATCH, SK_IN), MutAnyOrigin
        ](gi_skip_buf.unsafe_ptr())
        var p_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self._SKIP_PARAM_OFF)
        var s_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.STATE_SIZE), MutAnyOrigin
        ](state.ptr + Self.Inner.STATE_SIZE)
        var c_skip = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Skip.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr + BATCH * Self.Inner.CACHE_SIZE)
        var grads_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr + Self._SKIP_PARAM_OFF)
        var go_skip = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Skip.OUT_DIM), MutAnyOrigin
        ](grad_output.ptr)
        Self.Skip.backward[BATCH, dtype](
            go_skip, gi_skip, p_skip, s_skip, c_skip, grads_skip
        )

        for i in range(BATCH * Self.IN_DIM):
            grad_input.ptr[i] = grad_input.ptr[i] + gi_skip_buf[i]

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
        comptime OUT = Self.OUT_DIM
        comptime IN = Self.IN_DIM

        # Slice workspace: [skip_out(OUT) | gi_skip(IN, unused here) | child_ws]
        var ws_ptr = workspace.unsafe_ptr()
        var skip_out_ptr = ws_ptr
        var child_ws_ptr = ws_ptr + BATCH * OUT + BATCH * IN
        comptime CHILD_WS_SIZE = max(
            1,
            BATCH
            * (
                Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
                + Self.Skip.WORKSPACE_SIZE_PER_SAMPLE
            ),
        )
        var child_ws = DeviceBuffer[dtype](
            ctx, child_ws_ptr, CHILD_WS_SIZE, owning=False
        )

        var skip_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Skip.OUT_DIM), MutAnyOrigin
        ](skip_out_ptr)

        var p_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var p_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self._SKIP_PARAM_OFF)  # Aligned

        var s_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.STATE_SIZE), MutAnyOrigin
        ](state.ptr)
        var s_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.STATE_SIZE), MutAnyOrigin
        ](state.ptr + Self.Inner.STATE_SIZE)

        var c_inner = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Inner.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        var c_skip = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Skip.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr + BATCH * Self.Inner.CACHE_SIZE)

        var input_skip = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Skip.IN_DIM), MutAnyOrigin
        ](input.ptr)

        # Inner writes directly to output; Skip writes to scratch
        Self.Inner.forward_gpu[BATCH, dtype](
            ctx, output, input, p_inner, s_inner, c_inner, child_ws
        )
        Self.Skip.forward_gpu[BATCH, dtype](
            ctx, skip_out_t, input_skip, p_skip, s_skip, c_skip, child_ws
        )

        # output += skip_out
        var skip_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT), ImmutAnyOrigin
        ](skip_out_ptr)
        comptime TOTAL = BATCH * OUT
        var grid_x = (TOTAL + TPB - 1) // TPB

        @parameter
        @always_inline
        def add_k(
            a: LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
            ],
            b: LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= TOTAL:
                return
            a.ptr[idx] = a.ptr[idx] + b.ptr[idx]

        ctx.enqueue_function[add_k](
            output, skip_immut, grid_dim=(grid_x,), block_dim=(TPB,)
        )

    # =========================================================================
    # GPU Forward (no cache — inference path used for evaluation)
    # =========================================================================

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
        comptime OUT = Self.OUT_DIM
        comptime IN = Self.IN_DIM

        var ws_ptr = workspace.unsafe_ptr()
        var skip_out_ptr = ws_ptr
        var child_ws_ptr = ws_ptr + BATCH * OUT + BATCH * IN
        comptime CHILD_WS_SIZE = max(
            1,
            BATCH
            * (
                Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
                + Self.Skip.WORKSPACE_SIZE_PER_SAMPLE
            ),
        )
        var child_ws = DeviceBuffer[dtype](
            ctx, child_ws_ptr, CHILD_WS_SIZE, owning=False
        )

        var skip_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Skip.OUT_DIM), MutAnyOrigin
        ](skip_out_ptr)

        var p_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var p_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self._SKIP_PARAM_OFF)

        var s_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.STATE_SIZE), MutAnyOrigin
        ](state.ptr)
        var s_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.STATE_SIZE), MutAnyOrigin
        ](state.ptr + Self.Inner.STATE_SIZE)

        var input_skip = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Skip.IN_DIM), MutAnyOrigin
        ](input.ptr)

        Self.Inner.forward_gpu_no_cache[BATCH, dtype](
            ctx, output, input, p_inner, s_inner, child_ws
        )
        Self.Skip.forward_gpu_no_cache[BATCH, dtype](
            ctx, skip_out_t, input_skip, p_skip, s_skip, child_ws
        )

        var skip_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT), ImmutAnyOrigin
        ](skip_out_ptr)
        comptime TOTAL = BATCH * OUT
        var grid_x = (TOTAL + TPB - 1) // TPB

        @parameter
        @always_inline
        def add_k(
            a: LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
            ],
            b: LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= TOTAL:
                return
            a.ptr[idx] = a.ptr[idx] + b.ptr[idx]

        ctx.enqueue_function[add_k](
            output, skip_immut, grid_dim=(grid_x,), block_dim=(TPB,)
        )

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
        Self.forward_gpu_no_cache[BATCH, dtype](
            ctx, output, input, params, state, workspace
        )

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
        comptime OUT = Self.OUT_DIM
        comptime IN = Self.IN_DIM

        # Slice workspace: [skip_out(OUT, unused) | gi_skip(IN) | child_ws]
        var ws_ptr = workspace.unsafe_ptr()
        var gi_skip_ptr = ws_ptr + BATCH * OUT
        var child_ws_ptr = ws_ptr + BATCH * OUT + BATCH * IN
        comptime CHILD_WS_SIZE = max(
            1,
            BATCH
            * (
                Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
                + Self.Skip.WORKSPACE_SIZE_PER_SAMPLE
            ),
        )
        var child_ws = DeviceBuffer[dtype](
            ctx, child_ws_ptr, CHILD_WS_SIZE, owning=False
        )

        # Inner backward → grad_input (overwrites)
        var p_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var s_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.STATE_SIZE), MutAnyOrigin
        ](state.ptr)
        var c_inner = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Inner.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        var grads_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr)
        Self.Inner.backward_gpu[BATCH, dtype](
            ctx, grad_input, grad_output, p_inner, s_inner, c_inner, grads_inner, child_ws
        )

        # Skip backward → gi_skip scratch
        var gi_skip_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Skip.IN_DIM), MutAnyOrigin
        ](gi_skip_ptr)
        var p_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self._SKIP_PARAM_OFF)
        var s_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.STATE_SIZE), MutAnyOrigin
        ](state.ptr + Self.Inner.STATE_SIZE)
        var c_skip = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Skip.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr + BATCH * Self.Inner.CACHE_SIZE)
        var grads_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr + Self._SKIP_PARAM_OFF)
        var go_skip = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Skip.OUT_DIM), MutAnyOrigin
        ](grad_output.ptr)
        Self.Skip.backward_gpu[BATCH, dtype](
            ctx, gi_skip_t, go_skip, p_skip, s_skip, c_skip, grads_skip, child_ws
        )

        # grad_input += gi_skip
        var gi_skip_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, IN), ImmutAnyOrigin
        ](gi_skip_ptr)
        comptime TOTAL = BATCH * IN
        var grid_x = (TOTAL + TPB - 1) // TPB

        @parameter
        @always_inline
        def add_k(
            a: LayoutTensor[
                dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
            ],
            b: LayoutTensor[
                dtype, Layout.row_major(BATCH, IN), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= TOTAL:
                return
            a.ptr[idx] = a.ptr[idx] + b.ptr[idx]

        ctx.enqueue_function[add_k](
            grad_input, gi_skip_immut, grid_dim=(grid_x,), block_dim=(TPB,)
        )

    # =========================================================================
    # GPU Forward (inference-mode, with cache)
    # =========================================================================

    @staticmethod
    def forward_gpu_inference_with_cache[
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
        comptime OUT = Self.OUT_DIM
        comptime IN = Self.IN_DIM

        var ws_ptr = workspace.unsafe_ptr()
        var skip_out_ptr = ws_ptr
        var child_ws_ptr = ws_ptr + BATCH * OUT + BATCH * IN
        comptime CHILD_WS_SIZE = max(
            1,
            BATCH
            * (
                Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
                + Self.Skip.WORKSPACE_SIZE_PER_SAMPLE
            ),
        )
        var child_ws = DeviceBuffer[dtype](
            ctx, child_ws_ptr, CHILD_WS_SIZE, owning=False
        )

        var skip_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Skip.OUT_DIM), MutAnyOrigin
        ](skip_out_ptr)

        var p_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var p_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self._SKIP_PARAM_OFF)

        var s_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.STATE_SIZE), MutAnyOrigin
        ](state.ptr)
        var s_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.STATE_SIZE), MutAnyOrigin
        ](state.ptr + Self.Inner.STATE_SIZE)

        var c_inner = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Inner.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        var c_skip = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Skip.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr + BATCH * Self.Inner.CACHE_SIZE)

        var input_skip = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Skip.IN_DIM), MutAnyOrigin
        ](input.ptr)

        Self.Inner.forward_gpu_inference_with_cache[BATCH, dtype](
            ctx, output, input, p_inner, s_inner, c_inner, child_ws
        )
        Self.Skip.forward_gpu_inference_with_cache[BATCH, dtype](
            ctx, skip_out_t, input_skip, p_skip, s_skip, c_skip, child_ws
        )

        var skip_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT), ImmutAnyOrigin
        ](skip_out_ptr)
        comptime TOTAL = BATCH * OUT
        var grid_x = (TOTAL + TPB - 1) // TPB

        @parameter
        @always_inline
        def add_k(
            a: LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
            ],
            b: LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= TOTAL:
                return
            a.ptr[idx] = a.ptr[idx] + b.ptr[idx]

        ctx.enqueue_function[add_k](
            output, skip_immut, grid_dim=(grid_x,), block_dim=(TPB,)
        )

    # =========================================================================
    # GPU Backward (inference-mode)
    # =========================================================================

    @staticmethod
    def backward_gpu_inference[
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
        comptime OUT = Self.OUT_DIM
        comptime IN = Self.IN_DIM

        var ws_ptr = workspace.unsafe_ptr()
        var gi_skip_ptr = ws_ptr + BATCH * OUT
        var child_ws_ptr = ws_ptr + BATCH * OUT + BATCH * IN
        comptime CHILD_WS_SIZE = max(
            1,
            BATCH
            * (
                Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
                + Self.Skip.WORKSPACE_SIZE_PER_SAMPLE
            ),
        )
        var child_ws = DeviceBuffer[dtype](
            ctx, child_ws_ptr, CHILD_WS_SIZE, owning=False
        )

        var p_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](params.ptr)
        var s_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.STATE_SIZE), MutAnyOrigin
        ](state.ptr)
        var c_inner = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Inner.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr)
        var grads_inner = LayoutTensor[
            dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr)
        Self.Inner.backward_gpu_inference[BATCH, dtype](
            ctx, grad_input, grad_output, p_inner, s_inner, c_inner, grads_inner, child_ws
        )

        var gi_skip_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Skip.IN_DIM), MutAnyOrigin
        ](gi_skip_ptr)
        var p_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.PARAM_SIZE), MutAnyOrigin
        ](params.ptr + Self._SKIP_PARAM_OFF)
        var s_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.STATE_SIZE), MutAnyOrigin
        ](state.ptr + Self.Inner.STATE_SIZE)
        var c_skip = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Skip.CACHE_SIZE), MutAnyOrigin
        ](cache.ptr + BATCH * Self.Inner.CACHE_SIZE)
        var grads_skip = LayoutTensor[
            dtype, Layout.row_major(Self.Skip.PARAM_SIZE), MutAnyOrigin
        ](grads.ptr + Self._SKIP_PARAM_OFF)
        var go_skip = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.Skip.OUT_DIM), MutAnyOrigin
        ](grad_output.ptr)
        Self.Skip.backward_gpu_inference[BATCH, dtype](
            ctx, gi_skip_t, go_skip, p_skip, s_skip, c_skip, grads_skip, child_ws
        )

        var gi_skip_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, IN), ImmutAnyOrigin
        ](gi_skip_ptr)
        comptime TOTAL = BATCH * IN
        var grid_x = (TOTAL + TPB - 1) // TPB

        @parameter
        @always_inline
        def add_k(
            a: LayoutTensor[
                dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
            ],
            b: LayoutTensor[
                dtype, Layout.row_major(BATCH, IN), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= TOTAL:
                return
            a.ptr[idx] = a.ptr[idx] + b.ptr[idx]

        ctx.enqueue_function[add_k](
            grad_input, gi_skip_immut, grid_dim=(grid_x,), block_dim=(TPB,)
        )
