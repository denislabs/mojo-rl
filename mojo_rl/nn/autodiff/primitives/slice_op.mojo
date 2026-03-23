"""SliceOp: Extract a contiguous range of dimensions from input (DiffOp).

Forward:  output[b, i] = input[b, start + i]   for i in [0, end-start)
Backward: grad_input[b, j] = grad_output[b, j-start] if start <= j < end else 0

This enables extracting specific fields from concatenated outputs, e.g.:
    SkipConcat[Actor → RSample] → [obs(17), action(6), log_prob(1)]
    SliceOp[24, 0, 23]  → [obs, action] (for critic input)
    SliceOp[24, 23, 24] → [log_prob]    (for entropy term)
"""

from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext


struct SliceOp[in_dim: Int, start: Int, end: Int](DiffOp):
    """Extract dimensions [start:end) from input.

    IN_DIM = in_dim
    OUT_DIM = end - start
    PARAM_SIZE = 0
    CACHE_SIZE = 0
    """

    comptime OP_ID: Int = OpID.USER_DEFINED._value + 3
    comptime IN_DIM: Int = Self.in_dim
    comptime OUT_DIM: Int = Self.end - Self.start
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = 0
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
            for i in range(Self.OUT_DIM):
                output.ptr[b * Self.OUT_DIM + i] = input.ptr[
                    b * Self.IN_DIM + Self.start + i
                ]

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
        # Zero all, then fill the slice
        for i in range(BATCH * Self.IN_DIM):
            grad_input.ptr[i] = Scalar[dtype](0.0)
        for b in range(BATCH):
            for i in range(Self.OUT_DIM):
                grad_input.ptr[
                    b * Self.IN_DIM + Self.start + i
                ] = grad_output.ptr[b * Self.OUT_DIM + i]

    # =========================================================================
    # GPU kernels
    # =========================================================================

    @always_inline
    @staticmethod
    def eval_kernel_impl[
        BATCH: Int
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.OUT_DIM:
            return
        var b = idx // Self.OUT_DIM
        var i = idx % Self.OUT_DIM
        output.ptr[idx] = input.ptr[b * Self.IN_DIM + Self.start + i]

    @always_inline
    @staticmethod
    def backward_kernel_impl[
        BATCH: Int
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
    ):
        # Zero pass
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.IN_DIM:
            return
        var b = idx // Self.IN_DIM
        var col = idx % Self.IN_DIM
        if col >= Self.start and col < Self.end:
            grad_input.ptr[idx] = grad_output.ptr[
                b * Self.OUT_DIM + (col - Self.start)
            ]
        else:
            grad_input.ptr[idx] = Scalar[dtype](0.0)

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
        comptime total = BATCH * Self.OUT_DIM
        var grid_x = (total + TPB - 1) // TPB

        @always_inline
        def wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
        ):
            Self.eval_kernel_impl[BATCH](output, input)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
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
        var go_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)
        # Use IN_DIM grid so every element gets zeroed or filled
        comptime total = BATCH * Self.IN_DIM
        var grid_x = (total + TPB - 1) // TPB

        @always_inline
        def wrapper(
            gi: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            go: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            Self.backward_kernel_impl[BATCH](gi, go)

        ctx.enqueue_function[wrapper, wrapper](
            grad_input,
            go_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )
