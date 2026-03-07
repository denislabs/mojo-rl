"""Residual skip connection combinator.

Residual[Inner: Model] computes y = Inner.forward(x) + x.
Requires IN_DIM == OUT_DIM (skip connection needs matching dimensions).

Forward:  output = Inner(input) + input
Backward: grad_input = Inner.backward(grad_output) + grad_output
"""

from ...constants import dtype, TPB
from ...model.model import Model
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer


# GPU kernel impl: a[i] += b[i]
@always_inline
fn _add_kernel_impl[
    BATCH: Int, DIM: Int
](
    a: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), ImmutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * DIM:
        return
    a.ptr[idx] = a.ptr[idx] + b.ptr[idx]


@fieldwise_init
struct Residual[Inner: Model](Model):
    """Skip connection: y = Inner(x) + x.

    The inner model must have IN_DIM == OUT_DIM so that the skip
    addition is dimensionally valid.
    """

    comptime IN_DIM: Int = Self.Inner.IN_DIM
    comptime OUT_DIM: Int = Self.Inner.OUT_DIM
    comptime PARAM_SIZE: Int = Self.Inner.PARAM_SIZE
    comptime CACHE_SIZE: Int = Self.Inner.CACHE_SIZE
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = (
        Self.Inner.WORKSPACE_SIZE_PER_SAMPLE
    )

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
        Self.Inner.forward[BATCH](input, output, params, cache)
        for i in range(BATCH * Self.IN_DIM):
            output.ptr[i] = output.ptr[i] + input.ptr[i]

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
        Self.Inner.forward[BATCH](input, output, params)
        for i in range(BATCH * Self.IN_DIM):
            output.ptr[i] = output.ptr[i] + input.ptr[i]

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
        Self.Inner.backward[BATCH](grad_output, grad_input, params, cache, grads)
        for i in range(BATCH * Self.IN_DIM):
            grad_input.ptr[i] = grad_input.ptr[i] + grad_output.ptr[i]

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
    ) raises:
        Self.Inner.forward_gpu[BATCH](
            ctx, output, input, params, cache, workspace
        )

        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)
        var grid_x = (BATCH * Self.IN_DIM + TPB - 1) // TPB

        @always_inline
        fn wrapper(
            a: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            b: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
        ):
            _add_kernel_impl[BATCH, Self.IN_DIM](a, b)

        ctx.enqueue_function[wrapper, wrapper](
            output, input_immut, grid_dim=(grid_x,), block_dim=(TPB,)
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
    ) raises:
        Self.Inner.forward_gpu_no_cache[BATCH](
            ctx, output, input, params, workspace
        )

        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)
        var grid_x = (BATCH * Self.IN_DIM + TPB - 1) // TPB

        @always_inline
        fn wrapper(
            a: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            b: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
        ):
            _add_kernel_impl[BATCH, Self.IN_DIM](a, b)

        ctx.enqueue_function[wrapper, wrapper](
            output, input_immut, grid_dim=(grid_x,), block_dim=(TPB,)
        )

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
    ) raises:
        Self.Inner.backward_gpu[BATCH](
            ctx, grad_input, grad_output, params, cache, grads, workspace
        )

        var go_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](grad_output.ptr)
        var grid_x = (BATCH * Self.IN_DIM + TPB - 1) // TPB

        @always_inline
        fn wrapper(
            a: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            b: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
        ):
            _add_kernel_impl[BATCH, Self.IN_DIM](a, b)

        ctx.enqueue_function[wrapper, wrapper](
            grad_input, go_immut, grid_dim=(grid_x,), block_dim=(TPB,)
        )
