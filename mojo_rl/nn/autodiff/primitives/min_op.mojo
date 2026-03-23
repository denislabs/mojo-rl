"""MinOp: Elementwise minimum of two concatenated inputs (DiffOp).

Takes input [BATCH, 2*dim] = [x1 || x2] and outputs [BATCH, dim] = min(x1, x2).
On backward, routes gradient to the selected input (x1 or x2), zeroing the other.

This is the autodiff equivalent of torch.min(Q1, Q2) used in SAC/TD3 to compute
min(Q1, Q2) for the actor loss. Replaces manual gradient masking code.

Usage in SAC: After forwarding through both critics, concat their outputs and
pass through MinOp to get min(Q1, Q2) with automatic gradient routing.
"""

from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext


struct MinOp[dim: Int](DiffOp):
    """Elementwise minimum: output = min(input[:, :dim], input[:, dim:]).

    IN_DIM = 2 * dim (two inputs concatenated)
    OUT_DIM = dim
    PARAM_SIZE = 0
    CACHE_SIZE = dim (stores mask: 0 if x1 <= x2, 1 if x2 < x1)
    """

    comptime OP_ID: Int = OpID.USER_DEFINED._value + 2
    comptime IN_DIM: Int = 2 * Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = Self.dim
    comptime OP_WORKSPACE_PER_SAMPLE: Int = 0

    def __init__(out self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    # =========================================================================
    # CPU eval / vjp
    # =========================================================================

    @staticmethod
    def eval[
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
            for i in range(Self.dim):
                var x1 = rebind[Scalar[dtype]](input[b, i])
                var x2 = rebind[Scalar[dtype]](input[b, Self.dim + i])
                if x1 <= x2:
                    output[b, i] = x1
                    cache[b, i] = Scalar[dtype](0.0)  # x1 was selected
                else:
                    output[b, i] = x2
                    cache[b, i] = Scalar[dtype](1.0)  # x2 was selected

    @staticmethod
    def vjp[
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
        """Route gradient to the selected input, zero the other."""
        for b in range(BATCH):
            for i in range(Self.dim):
                var g = grad_output[b, i]
                var mask = rebind[Scalar[dtype]](cache[b, i])
                if mask == Scalar[dtype](0.0):
                    # x1 was selected
                    grad_input[b, i] = g
                    grad_input[b, Self.dim + i] = Scalar[dtype](0.0)
                else:
                    # x2 was selected
                    grad_input[b, i] = Scalar[dtype](0.0)
                    grad_input[b, Self.dim + i] = g

    # =========================================================================
    # GPU kernels
    # =========================================================================

    @always_inline
    @staticmethod
    def eval_kernel_impl[
        BATCH: Int
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        var b = idx // Self.dim
        var i = idx % Self.dim
        var x1 = rebind[Scalar[dtype]](input[b, i])
        var x2 = rebind[Scalar[dtype]](input[b, Self.dim + i])
        if x1 <= x2:
            output[b, i] = x1
            cache[b, i] = Scalar[dtype](0.0)
        else:
            output[b, i] = x2
            cache[b, i] = Scalar[dtype](1.0)

    @always_inline
    @staticmethod
    def backward_kernel_impl[
        BATCH: Int
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        var b = idx // Self.dim
        var i = idx % Self.dim
        var g = rebind[Scalar[dtype]](grad_output[b, i])
        var mask = rebind[Scalar[dtype]](cache[b, i])
        var zero = Scalar[dtype](0.0)
        if mask == zero:
            grad_input[b, i] = g
            grad_input[b, Self.dim + i] = zero
        else:
            grad_input[b, i] = zero
            grad_input[b, Self.dim + i] = g

    # =========================================================================
    # GPU launchers
    # =========================================================================

    @staticmethod
    def eval_gpu[
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
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)
        var total = BATCH * Self.dim
        var grid_x = (total + TPB - 1) // TPB

        @always_inline
        def wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
        ):
            Self.eval_kernel_impl[BATCH](output, input, cache)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
            cache,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

    @staticmethod
    def vjp_gpu[
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
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](grad_output.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](cache.ptr)
        var total = BATCH * Self.dim
        var grid_x = (total + TPB - 1) // TPB

        @always_inline
        def wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
        ):
            Self.backward_kernel_impl[BATCH](grad_input, grad_output, cache)

        ctx.enqueue_function[wrapper, wrapper](
            grad_input,
            grad_output_immut,
            cache_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )
