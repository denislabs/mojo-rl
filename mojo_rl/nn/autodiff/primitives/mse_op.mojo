"""MSEOp: Mean Squared Error loss as a DiffOp.

Takes two values (prediction and target) concatenated as input,
outputs the squared error. The mean is handled by the gradient seed (1/BATCH).

Forward:  output[b, 0] = (input[b, 0] - input[b, 1])^2
Backward: grad_input[b, 0] = 2 * (input[b, 0] - input[b, 1]) * grad_output[b, 0]
          grad_input[b, 1] = 0  (target is frozen, no gradient)

Usage in DQN:
    [Q(s,a), target] → MSEOp → squared_error
    With grad_seed = 1/BATCH, this gives the standard MSE gradient.
"""

from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext


struct MSEOp(DiffOp):
    """Squared error: output = (pred - target)^2.

    IN_DIM = 2 (prediction || target concatenated)
    OUT_DIM = 1
    PARAM_SIZE = 0
    CACHE_SIZE = 1 (caches the residual pred - target for backward)
    """

    comptime OP_ID: Int = OpID.USER_DEFINED._value + 10
    comptime IN_DIM: Int = 2
    comptime OUT_DIM: Int = 1
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = 1
    comptime OP_WORKSPACE_PER_SAMPLE: Int = 0

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
        for b in range(BATCH):
            var pred = rebind[Scalar[dtype]](input[b, 0])
            var target = rebind[Scalar[dtype]](input[b, 1])
            var residual = pred - target
            cache[b, 0] = residual
            output[b, 0] = residual * residual

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
        for b in range(BATCH):
            var g = rebind[Scalar[dtype]](grad_output[b, 0])
            var residual = rebind[Scalar[dtype]](cache[b, 0])
            # d(residual^2)/d(pred) = 2 * residual
            grad_input[b, 0] = Scalar[dtype](2.0) * residual * g
            # target is frozen — no gradient
            grad_input[b, 1] = Scalar[dtype](0.0)

    # =========================================================================
    # GPU kernels
    # =========================================================================

    @always_inline
    @staticmethod
    fn eval_kernel_impl[
        BATCH: Int
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, 2), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ],
    ):
        var b = Int(block_dim.x * block_idx.x + thread_idx.x)
        if b >= BATCH:
            return
        var pred = rebind[Scalar[dtype]](input[b, 0])
        var target = rebind[Scalar[dtype]](input[b, 1])
        var residual = pred - target
        cache[b, 0] = residual
        output[b, 0] = residual * residual

    @always_inline
    @staticmethod
    fn backward_kernel_impl[
        BATCH: Int
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, 2), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
        ],
    ):
        var b = Int(block_dim.x * block_idx.x + thread_idx.x)
        if b >= BATCH:
            return
        var g = rebind[Scalar[dtype]](grad_output[b, 0])
        var residual = rebind[Scalar[dtype]](cache[b, 0])
        grad_input[b, 0] = Scalar[dtype](2.0) * residual * g
        grad_input[b, 1] = Scalar[dtype](0.0)

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
        workspace: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, 2), ImmutAnyOrigin
        ](input.ptr)
        var grid_x = (BATCH + TPB - 1) // TPB

        @always_inline
        fn wrapper(
            o: LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ],
            i: LayoutTensor[
                dtype, Layout.row_major(BATCH, 2), ImmutAnyOrigin
            ],
            c: LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
            ],
        ):
            Self.eval_kernel_impl[BATCH](o, i, c)

        ctx.enqueue_function[wrapper, wrapper](
            output, input_immut, cache,
            grid_dim=(grid_x,), block_dim=(TPB,),
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
        workspace: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        var go_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
        ](grad_output.ptr)
        var c_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
        ](cache.ptr)
        var grid_x = (BATCH + TPB - 1) // TPB

        @always_inline
        fn wrapper(
            gi: LayoutTensor[
                dtype, Layout.row_major(BATCH, 2), MutAnyOrigin
            ],
            go: LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
            ],
            c: LayoutTensor[
                dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
            ],
        ):
            Self.backward_kernel_impl[BATCH](gi, go, c)

        ctx.enqueue_function[wrapper, wrapper](
            grad_input, go_immut, c_immut,
            grid_dim=(grid_x,), block_dim=(TPB,),
        )
