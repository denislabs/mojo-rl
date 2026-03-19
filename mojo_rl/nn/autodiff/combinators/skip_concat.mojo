"""SkipConcat combinator: concatenates input with Inner(input).

Forward:  output = concat(input, Inner(input))
Backward: grad_inner = Inner.backward(grad_output[:, IN_DIM:])
          grad_input = grad_output[:, :IN_DIM] + grad_inner

This enables the skip-connection pattern needed for actor-critic composition:
    obs → SkipConcat[Actor → RSampleOp] → [obs || action || log_prob]

The obs passes through unchanged while the Actor produces actions.
Downstream ops (Critic) can read [obs, action] from the concatenated output.

Like Residual, but concatenates instead of adding, and the inner model
can change dimensions (OUT_DIM != IN_DIM is allowed).
"""

from ...constants import dtype, TPB
from ...model.model import Model, PerfTimerPtr, NULL_PERF
from ...initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream


@fieldwise_init
struct SkipConcat[Inner: Model](Model):
    """Skip connection with concatenation: y = concat(x, Inner(x)).

    IN_DIM = Inner.IN_DIM
    OUT_DIM = Inner.IN_DIM + Inner.OUT_DIM  (input || inner_output)
    """

    comptime IN_DIM: Int = Self.Inner.IN_DIM
    comptime OUT_DIM: Int = Self.Inner.IN_DIM + Self.Inner.OUT_DIM
    comptime PARAM_SIZE: Int = Self.Inner.PARAM_SIZE
    comptime CACHE_SIZE: Int = Self.Inner.CACHE_SIZE
    # Own scratch: inner_out / grad_inner buffer (shared between fwd/bwd)
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = (
        Self.Inner.OUT_DIM + Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
    )

    # =========================================================================
    # Initialization
    # =========================================================================

    @staticmethod
    fn initialize_params[INIT: Initializer](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        Self.Inner.initialize_params[INIT](params)

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
        # Copy input to first part of output
        for b in range(BATCH):
            for i in range(Self.IN_DIM):
                output.ptr[b * Self.OUT_DIM + i] = input.ptr[
                    b * Self.IN_DIM + i
                ]

        # Forward Inner(input) → second part of output
        comptime INNER_OUT = Self.Inner.OUT_DIM
        # Create a view into the inner-output portion of the output tensor
        # We need to use pointer math since output has stride OUT_DIM
        # but inner expects stride INNER_OUT
        #
        # Strategy: use a temporary buffer for inner output, then copy
        var inner_buf = InlineArray[
            Scalar[dtype], BATCH * INNER_OUT
        ](uninitialized=True)
        var inner_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, INNER_OUT), MutAnyOrigin
        ](inner_buf.unsafe_ptr())

        Self.Inner.forward[BATCH](input, inner_out, params, cache)

        # Copy inner output to second part of output
        for b in range(BATCH):
            for i in range(INNER_OUT):
                output.ptr[b * Self.OUT_DIM + Self.IN_DIM + i] = (
                    inner_buf[b * INNER_OUT + i]
                )

    # =========================================================================
    # CPU Forward (no cache — inference)
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
        # Copy input
        for b in range(BATCH):
            for i in range(Self.IN_DIM):
                output.ptr[b * Self.OUT_DIM + i] = input.ptr[
                    b * Self.IN_DIM + i
                ]

        # Forward Inner
        comptime INNER_OUT = Self.Inner.OUT_DIM
        var inner_buf = InlineArray[
            Scalar[dtype], BATCH * INNER_OUT
        ](uninitialized=True)
        var inner_out = LayoutTensor[
            dtype, Layout.row_major(BATCH, INNER_OUT), MutAnyOrigin
        ](inner_buf.unsafe_ptr())

        Self.Inner.forward[BATCH](input, inner_out, params)

        for b in range(BATCH):
            for i in range(INNER_OUT):
                output.ptr[b * Self.OUT_DIM + Self.IN_DIM + i] = (
                    inner_buf[b * INNER_OUT + i]
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
        comptime INNER_OUT = Self.Inner.OUT_DIM

        # Extract inner gradient from grad_output[:, IN_DIM:]
        var grad_inner_buf = InlineArray[
            Scalar[dtype], BATCH * INNER_OUT
        ](uninitialized=True)
        for b in range(BATCH):
            for i in range(INNER_OUT):
                grad_inner_buf[b * INNER_OUT + i] = grad_output.ptr[
                    b * Self.OUT_DIM + Self.IN_DIM + i
                ]
        var grad_inner = LayoutTensor[
            dtype, Layout.row_major(BATCH, INNER_OUT), MutAnyOrigin
        ](grad_inner_buf.unsafe_ptr())

        # Backward through Inner
        Self.Inner.backward[BATCH](
            grad_inner, grad_input, params, cache, grads
        )

        # Add skip gradient: grad_input += grad_output[:, :IN_DIM]
        for b in range(BATCH):
            for i in range(Self.IN_DIM):
                grad_input.ptr[b * Self.IN_DIM + i] = (
                    grad_input.ptr[b * Self.IN_DIM + i]
                    + grad_output.ptr[b * Self.OUT_DIM + i]
                )

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
        comptime INNER_OUT = Self.Inner.OUT_DIM

        # Slice workspace: [inner_out (BATCH * INNER_OUT) | child_ws]
        var ws_ptr = workspace.unsafe_ptr()
        var inner_out_ptr = ws_ptr  # BATCH * INNER_OUT
        var child_ws_ptr = ws_ptr + BATCH * INNER_OUT
        comptime CHILD_WS_SIZE = max(
            1, BATCH * Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
        )
        var child_ws = DeviceBuffer[dtype](
            ctx, child_ws_ptr, CHILD_WS_SIZE, owning=False
        )

        var inner_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, INNER_OUT), MutAnyOrigin
        ](inner_out_ptr)

        # Forward Inner
        Self.Inner.forward_gpu[BATCH](
            ctx, inner_out_t, input, params, cache, child_ws
        )

        # Copy input + inner_output → output (interleaved)
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)
        var inner_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, INNER_OUT), ImmutAnyOrigin
        ](inner_out_ptr)
        comptime TOTAL = BATCH * Self.OUT_DIM
        var grid_x = (TOTAL + TPB - 1) // TPB

        @always_inline
        fn concat_kernel(
            dst: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            src_skip: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            src_inner: LayoutTensor[
                dtype, Layout.row_major(BATCH, INNER_OUT), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= TOTAL:
                return
            var b = idx // Self.OUT_DIM
            var c = idx % Self.OUT_DIM
            if c < Self.IN_DIM:
                dst.ptr[idx] = src_skip.ptr[b * Self.IN_DIM + c]
            else:
                dst.ptr[idx] = src_inner.ptr[
                    b * INNER_OUT + (c - Self.IN_DIM)
                ]

        ctx.enqueue_function[concat_kernel, concat_kernel](
            output,
            input_immut,
            inner_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # GPU Forward (no cache)
    # =========================================================================

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
        comptime INNER_OUT = Self.Inner.OUT_DIM

        # Slice workspace: [inner_out (BATCH * INNER_OUT) | child_ws]
        var ws_ptr = workspace.unsafe_ptr()
        var inner_out_ptr = ws_ptr  # BATCH * INNER_OUT
        var child_ws_ptr = ws_ptr + BATCH * INNER_OUT
        comptime CHILD_WS_SIZE = max(
            1, BATCH * Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
        )
        var child_ws = DeviceBuffer[dtype](
            ctx, child_ws_ptr, CHILD_WS_SIZE, owning=False
        )

        var inner_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, INNER_OUT), MutAnyOrigin
        ](inner_out_ptr)

        Self.Inner.forward_gpu_no_cache[BATCH](
            ctx, inner_out_t, input, params, child_ws
        )

        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)
        var inner_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, INNER_OUT), ImmutAnyOrigin
        ](inner_out_ptr)
        comptime TOTAL = BATCH * Self.OUT_DIM
        var grid_x = (TOTAL + TPB - 1) // TPB

        @always_inline
        fn concat_kernel(
            dst: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            src_skip: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            src_inner: LayoutTensor[
                dtype, Layout.row_major(BATCH, INNER_OUT), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= TOTAL:
                return
            var b = idx // Self.OUT_DIM
            var c = idx % Self.OUT_DIM
            if c < Self.IN_DIM:
                dst.ptr[idx] = src_skip.ptr[b * Self.IN_DIM + c]
            else:
                dst.ptr[idx] = src_inner.ptr[
                    b * INNER_OUT + (c - Self.IN_DIM)
                ]

        ctx.enqueue_function[concat_kernel, concat_kernel](
            output,
            input_immut,
            inner_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

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

    # =========================================================================
    # GPU Backward
    # =========================================================================

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
        comptime INNER_OUT = Self.Inner.OUT_DIM

        # Slice workspace: [grad_inner (BATCH * INNER_OUT) | child_ws]
        var ws_ptr = workspace.unsafe_ptr()
        var grad_inner_ptr = ws_ptr  # BATCH * INNER_OUT
        var child_ws_ptr = ws_ptr + BATCH * INNER_OUT
        comptime CHILD_WS_SIZE = max(
            1, BATCH * Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
        )
        var child_ws = DeviceBuffer[dtype](
            ctx, child_ws_ptr, CHILD_WS_SIZE, owning=False
        )

        # Extract grad_inner from grad_output[:, IN_DIM:]
        var grad_inner_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, INNER_OUT), MutAnyOrigin
        ](grad_inner_ptr)

        comptime INNER_TOTAL = BATCH * INNER_OUT
        var inner_grid = (INNER_TOTAL + TPB - 1) // TPB

        @always_inline
        fn extract_inner_grad(
            dst: LayoutTensor[
                dtype, Layout.row_major(BATCH, INNER_OUT), MutAnyOrigin
            ],
            src: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= INNER_TOTAL:
                return
            var b = idx // INNER_OUT
            var i = idx % INNER_OUT
            dst.ptr[idx] = src.ptr[b * Self.OUT_DIM + Self.IN_DIM + i]

        var go_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)

        ctx.enqueue_function[extract_inner_grad, extract_inner_grad](
            grad_inner_t,
            go_immut,
            grid_dim=(inner_grid,),
            block_dim=(TPB,),
        )

        # Backward through Inner → grad_input
        Self.Inner.backward_gpu[BATCH](
            ctx,
            grad_input,
            grad_inner_t,
            params,
            cache,
            grads,
            child_ws,
        )

        # Add skip gradient: grad_input += grad_output[:, :IN_DIM]
        comptime SKIP_TOTAL = BATCH * Self.IN_DIM
        var skip_grid = (SKIP_TOTAL + TPB - 1) // TPB

        @always_inline
        fn add_skip_grad(
            gi: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            go: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= SKIP_TOTAL:
                return
            var b = idx // Self.IN_DIM
            var i = idx % Self.IN_DIM
            gi.ptr[idx] = gi.ptr[idx] + go.ptr[b * Self.OUT_DIM + i]

        ctx.enqueue_function[add_skip_grad, add_skip_grad](
            grad_input,
            go_immut,
            grid_dim=(skip_grid,),
            block_dim=(TPB,),
        )
