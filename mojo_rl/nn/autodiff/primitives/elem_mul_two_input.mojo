"""ElemMulTwoInputOp: element-wise y = a * b (no learned param).

Two-input DiffOp: input is the concatenation of `a` and `b` (shape
`(BATCH, 2*dim)`), output is `y` (shape `(BATCH, dim)`). When used inside
ComputeGraph, the two source nodes' outputs are concatenated to form this
op's input — same convention as MinOp / DivideOp.

Distinct from the existing `ElemMul[dim]` which multiplies a tensor by a
*learned* per-feature gamma parameter. This op multiplies two
runtime-dynamic tensors with no learnable parameters.

Forward:   y[i] = a[i] * b[i]
Backward:  dy/da[i] = b[i]
           dy/db[i] = a[i]

Cache (size 2*dim per sample):
    cache[..dim)         = a[i]   (for grad_b)
    cache[dim..2*dim)    = b[i]   (for grad_a)

Used in TD-MPC2's `scaled_entropy = -log_prob_post * entropy_scale` path
when expressed as a ComputeGraph (see `docs/TDMPC2_AUDIT.md`, Stage 2/3).

PARAM_SIZE = 0
"""

from ...constants import dtype, TPB
from ..op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext


struct ElemMulTwoInputOp[dim: Int](DiffOp):
    """Element-wise product of two tensor inputs (no params)."""

    # Reuse ELEM_MUL since this is the same OpID family; no learned gamma.
    comptime OP_ID: Int = OpID.ELEM_MUL._value
    comptime IN_DIM: Int = 2 * Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = 2 * Self.dim
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
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        for b in range(BATCH):
            for i in range(Self.dim):
                var a_i = rebind[Scalar[dtype]](input[b, i])
                var b_i = rebind[Scalar[dtype]](input[b, Self.dim + i])
                output[b, i] = a_i * b_i
                cache[b, i] = a_i
                cache[b, Self.dim + i] = b_i

    @staticmethod
    def vjp[
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
        mut grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        for b in range(BATCH):
            for i in range(Self.dim):
                var a_i = rebind[Scalar[dtype]](cache[b, i])
                var b_i = rebind[Scalar[dtype]](cache[b, Self.dim + i])
                var go = rebind[Scalar[dtype]](grad_output[b, i])
                grad_input[b, i] = go * b_i
                grad_input[b, Self.dim + i] = go * a_i

    # =========================================================================
    # GPU kernels
    # =========================================================================

    @always_inline
    @staticmethod
    def eval_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, 2 * Self.dim), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, 2 * Self.dim), MutAnyOrigin
        ],
    ):
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        var row = idx // Self.dim
        var col = idx % Self.dim
        var a_i = rebind[Scalar[dtype]](input[row, col])
        var b_i = rebind[Scalar[dtype]](input[row, Self.dim + col])
        output[row, col] = a_i * b_i
        cache[row, col] = a_i
        cache[row, Self.dim + col] = b_i

    @always_inline
    @staticmethod
    def backward_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, 2 * Self.dim), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, 2 * Self.dim), ImmutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        var row = idx // Self.dim
        var col = idx % Self.dim
        var a_i = rebind[Scalar[dtype]](cache[row, col])
        var b_i = rebind[Scalar[dtype]](cache[row, Self.dim + col])
        var go = rebind[Scalar[dtype]](grad_output[row, col])
        grad_input[row, col] = go * b_i
        grad_input[row, Self.dim + col] = go * a_i

    # =========================================================================
    # GPU launchers
    # =========================================================================

    @staticmethod
    def eval_gpu[
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
        workspace: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, 2 * Self.dim), ImmutAnyOrigin
        ](input.ptr)
        var total_elements = BATCH * Self.dim
        var grid_x = (total_elements + TPB - 1) // TPB

        @parameter
        @always_inline
        def wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, 2 * Self.dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, 2 * Self.dim), MutAnyOrigin
            ],
        ):
            Self.eval_kernel_impl[BATCH, dtype](output, input, cache)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
            cache,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

    @staticmethod
    def vjp_gpu[
        BATCH: Int, dtype: DType = DType.float32
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
            dtype, Layout.row_major(BATCH, 2 * Self.dim), ImmutAnyOrigin
        ](cache.ptr)
        var total_elements = BATCH * Self.dim
        var grid_x = (total_elements + TPB - 1) // TPB

        @parameter
        @always_inline
        def wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, 2 * Self.dim), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, 2 * Self.dim), ImmutAnyOrigin
            ],
        ):
            Self.backward_kernel_impl[BATCH, dtype](
                grad_input, grad_output, cache
            )

        ctx.enqueue_function[wrapper, wrapper](
            grad_input,
            grad_output_immut,
            cache_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )
