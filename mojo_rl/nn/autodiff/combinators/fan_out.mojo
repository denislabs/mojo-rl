"""FanOut combinator: runs N copies of the same Model on the same input, concatenates outputs.

Forward:  output = concat(Inner_0(input), Inner_1(input), ..., Inner_{N-1}(input))
Backward: grad_input = sum_i(Inner_i.backward(grad_output_i))
          Each Inner_i has its own params and cache, but shares the same input.

This is a generalization of DualPath — DualPath runs 2 different Models on the
same input. FanOut runs N copies of the SAME Model (with separate params each)
on the same input.

Param layout uses 4-element alignment padding between copies' params to
guarantee GPU matmul alignment (16 bytes for float32).
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
    return x
    # return (x + 3) & ~3


@fieldwise_init
struct FanOut[Inner: Model, N: Int](Model):
    """Run N independent copies of Inner on the same input, concat outputs.

    IN_DIM = Inner.IN_DIM
    OUT_DIM = N * Inner.OUT_DIM
    PARAM_SIZE includes alignment padding between copies' params.
    CACHE_SIZE = N * Inner.CACHE_SIZE
    """

    comptime IN_DIM: Int = Self.Inner.IN_DIM
    comptime OUT_DIM: Int = Self.N * Self.Inner.OUT_DIM

    # Aligned param layout: each copy's params padded to 4, except last
    comptime _ALIGNED_INNER_PS: Int = _align4(Self.Inner.PARAM_SIZE)
    comptime PARAM_SIZE: Int = (
        Self.N - 1
    ) * Self._ALIGNED_INNER_PS + Self.Inner.PARAM_SIZE

    comptime CACHE_SIZE: Int = Self.N * Self.Inner.CACHE_SIZE
    # Own scratch: go_buf (Inner.OUT_DIM) + gi_buf (IN_DIM) reused across iterations
    comptime _OWN_WS: Int = Self.Inner.OUT_DIM + Self.IN_DIM
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = (
        Self._OWN_WS + Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
    )

    # =========================================================================
    # Offset helpers
    # =========================================================================

    @staticmethod
    def _param_offset[i: Int]() -> Int:
        return i * Self._ALIGNED_INNER_PS

    @staticmethod
    def _cache_offset[i: Int]() -> Int:
        return i * Self.Inner.CACHE_SIZE

    @staticmethod
    def _out_offset[i: Int]() -> Int:
        return i * Self.Inner.OUT_DIM

    # =========================================================================
    # Initialization
    # =========================================================================

    @staticmethod
    def initialize_params[
        INIT: Initializer
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        comptime for i in range(Self.N):
            # Zero padding region after copy i (except last)
            comptime if i < Self.N - 1:
                for j in range(Self.Inner.PARAM_SIZE, Self._ALIGNED_INNER_PS):
                    params.ptr[Self._param_offset[i]() + j] = Scalar[dtype](0.0)

            var pi = LayoutTensor[
                dtype, Layout.row_major(Self.Inner.PARAM_SIZE), MutAnyOrigin
            ](params.ptr + Self._param_offset[i]())
            Self.Inner.initialize_params[INIT](pi)

    # =========================================================================
    # CPU Forward (with cache)
    # =========================================================================

    @staticmethod
    def forward[
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
                    output.ptr[
                        b * Self.OUT_DIM + Self._out_offset[i]() + j
                    ] = tmp_buf[b * I_OUT + j]

    # =========================================================================
    # CPU Forward (no cache)
    # =========================================================================

    @staticmethod
    def forward[
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
                    output.ptr[
                        b * Self.OUT_DIM + Self._out_offset[i]() + j
                    ] = tmp_buf[b * I_OUT + j]

    # =========================================================================
    # CPU Backward
    # =========================================================================

    @staticmethod
    def backward[
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
                Self.Inner.backward[BATCH](go_i, grad_input, pi, ci, grads_i)
            else:
                # Subsequent copies: write to temp, then accumulate
                var gi_tmp = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Inner.IN_DIM),
                    MutAnyOrigin,
                ](gi_tmp_buf.unsafe_ptr())
                Self.Inner.backward[BATCH](go_i, gi_tmp, pi, ci, grads_i)
                # Accumulate into grad_input
                for k in range(BATCH * Self.IN_DIM):
                    grad_input.ptr[k] = grad_input.ptr[k] + gi_tmp_buf[k]

    # =========================================================================
    # GPU Forward (with cache)
    # =========================================================================

    @staticmethod
    def forward_gpu[
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

        # Slice workspace: [i_out_buf (I_OUT) | gi_buf (IN_DIM) | child_ws]
        var ws_ptr = workspace.unsafe_ptr()
        var i_out_ptr = ws_ptr  # Reused each iteration
        var child_ws_ptr = ws_ptr + BATCH * Self._OWN_WS
        comptime CHILD_WS_SIZE = max(
            1, BATCH * Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
        )
        var child_ws = DeviceBuffer[dtype](
            ctx, child_ws_ptr, CHILD_WS_SIZE, owning=False
        )

        # Forward each copy — params at aligned offsets, direct pointer access
        comptime for i in range(Self.N):
            var i_out_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, I_OUT), MutAnyOrigin
            ](i_out_ptr)

            var input_i = LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.Inner.IN_DIM), MutAnyOrigin
            ](input.ptr)

            # Params at aligned offset — safe for direct pointer access
            var pi = LayoutTensor[
                dtype,
                Layout.row_major(Self.Inner.PARAM_SIZE),
                MutAnyOrigin,
            ](
                params.ptr + Self._param_offset[i]()
            )  # Aligned!
            var ci = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr + BATCH * Self._cache_offset[i]())
            Self.Inner.forward_gpu[BATCH](
                ctx, i_out_t, input_i, pi, ci, child_ws
            )

            # Scatter copy i's output into the concat output
            var i_immut = LayoutTensor[
                dtype, Layout.row_major(BATCH, I_OUT), ImmutAnyOrigin
            ](i_out_ptr)
            comptime I_TOTAL = BATCH * I_OUT
            var i_grid = (I_TOTAL + TPB - 1) // TPB

            @always_inline
            def scatter_k(
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
                dst.ptr[
                    row * Self.OUT_DIM + Self._out_offset[i]() + col
                ] = src.ptr[idx]

            ctx.enqueue_function[scatter_k, scatter_k](
                output, i_immut, grid_dim=(i_grid,), block_dim=(TPB,)
            )

    @staticmethod
    def forward_gpu_no_cache[
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
    def forward_gpu_no_cache_on_stream[
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
    def backward_gpu[
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

        # Slice workspace: [go_buf (I_OUT) | gi_buf (IN_DIM) | child_ws]
        var ws_ptr = workspace.unsafe_ptr()
        var go_ws_ptr = ws_ptr  # Reused each iteration
        var gi_ws_ptr = ws_ptr + BATCH * I_OUT  # Reused each iteration
        var child_ws_ptr = ws_ptr + BATCH * Self._OWN_WS
        comptime CHILD_WS_SIZE = max(
            1, BATCH * Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
        )
        var child_ws = DeviceBuffer[dtype](
            ctx, child_ws_ptr, CHILD_WS_SIZE, owning=False
        )

        var go_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)

        comptime for i in range(Self.N):
            # Extract grad_output slice for copy i
            var go_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, I_OUT), MutAnyOrigin
            ](go_ws_ptr)

            comptime I_TOTAL = BATCH * I_OUT
            var i_grid = (I_TOTAL + TPB - 1) // TPB

            @always_inline
            def extract_k(
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

            # Params/cache/grads at aligned offsets — direct pointer access
            var pi = LayoutTensor[
                dtype,
                Layout.row_major(Self.Inner.PARAM_SIZE),
                MutAnyOrigin,
            ](
                params.ptr + Self._param_offset[i]()
            )  # Aligned!
            var ci = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.Inner.CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr + BATCH * Self._cache_offset[i]())
            var grads_i = LayoutTensor[
                dtype,
                Layout.row_major(Self.Inner.PARAM_SIZE),
                MutAnyOrigin,
            ](
                grads.ptr + Self._param_offset[i]()
            )  # Aligned!

            comptime if i == 0:
                # First copy: backward directly into grad_input
                Self.Inner.backward_gpu[BATCH](
                    ctx, grad_input, go_t, pi, ci, grads_i, child_ws
                )
            else:
                # Subsequent copies: backward into workspace temp, then accumulate
                var gi_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.Inner.IN_DIM),
                    MutAnyOrigin,
                ](gi_ws_ptr)
                Self.Inner.backward_gpu[BATCH](
                    ctx, gi_t, go_t, pi, ci, grads_i, child_ws
                )

                # Add to grad_input
                var gi_immut = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, Self.IN_DIM),
                    ImmutAnyOrigin,
                ](gi_ws_ptr)
                comptime GI_TOTAL = BATCH * Self.IN_DIM
                var gi_grid = (GI_TOTAL + TPB - 1) // TPB

                @always_inline
                def add_gi(
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
