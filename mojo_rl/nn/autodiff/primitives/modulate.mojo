"""ModulateOp: AdaLN-zero affine modulation.

Forward:  y[i] = x[i] * (1 + scale[i]) + shift[i]

Three-input DiffOp — input concatenates x, scale, and shift along dim 1:

    input[b, 0:dim]        = x[b, :]
    input[b, dim:2*dim]    = scale[b, :]
    input[b, 2*dim:3*dim]  = shift[b, :]

Used in LeWM's ConditionalTransformerBlock (AdaLN-zero) to inject the
conditioning signal `c` into each transformer block via a SiLU + Linear
that produces (shift, scale, gate) tuples. See
`docs/LEWM_PORT_PLAN.md` §3.1.

Gradients:
    dy/dx[i]     = 1 + scale[i]
    dy/dscale[i] = x[i]
    dy/dshift[i] = 1

Cache (size 2*dim per sample):
    cache[b, 0:dim]     = x[b, :]      (for dscale)
    cache[b, dim:2*dim] = scale[b, :]  (for dx)

PARAM_SIZE = 0  — no learned params; modulation tensors come from upstream
"""

from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.sys import simd_width_of


comptime _CPU_SIMD_W = simd_width_of[dtype]()


struct ModulateOp[dim: Int](DiffOp):
    """AdaLN-zero affine modulation: y = x * (1 + scale) + shift."""

    # LeWM-experimental — use USER_DEFINED range to avoid collisions.
    comptime OP_ID: Int = OpID.USER_DEFINED._value + 20
    comptime IN_DIM: Int = 3 * Self.dim
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
        comptime W = _CPU_SIMD_W
        var in_p = input.ptr
        var out_p = output.ptr
        var c_p = cache.ptr
        var one_v = SIMD[dtype, W](1)
        for b in range(BATCH):
            var in_off = b * 3 * Self.dim
            var out_off = b * Self.dim
            var c_off = b * 2 * Self.dim
            var j = 0
            while j + W <= Self.dim:
                var x = in_p.load[width=W](in_off + j)
                var s = in_p.load[width=W](in_off + Self.dim + j)
                var sh = in_p.load[width=W](in_off + 2 * Self.dim + j)
                c_p.store(c_off + j, x)
                c_p.store(c_off + Self.dim + j, s)
                out_p.store(out_off + j, x * (one_v + s) + sh)
                j += W
            var one = Scalar[dtype](1)
            while j < Self.dim:
                var x = in_p[in_off + j]
                var s = in_p[in_off + Self.dim + j]
                var sh = in_p[in_off + 2 * Self.dim + j]
                c_p[c_off + j] = x
                c_p[c_off + Self.dim + j] = s
                out_p[out_off + j] = x * (one + s) + sh
                j += 1

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
        comptime W = _CPU_SIMD_W
        var go_p = grad_output.ptr
        var gi_p = grad_input.ptr
        var c_p = cache.ptr
        var one_v = SIMD[dtype, W](1)
        for b in range(BATCH):
            var go_off = b * Self.dim
            var gi_off = b * 3 * Self.dim
            var c_off = b * 2 * Self.dim
            var j = 0
            while j + W <= Self.dim:
                var go = go_p.load[width=W](go_off + j)
                var x = c_p.load[width=W](c_off + j)
                var s = c_p.load[width=W](c_off + Self.dim + j)
                gi_p.store(gi_off + j, go * (one_v + s))
                gi_p.store(gi_off + Self.dim + j, go * x)
                gi_p.store(gi_off + 2 * Self.dim + j, go)
                j += W
            var one = Scalar[dtype](1)
            while j < Self.dim:
                var go = go_p[go_off + j]
                var x = c_p[c_off + j]
                var s = c_p[c_off + Self.dim + j]
                gi_p[gi_off + j] = go * (one + s)
                gi_p[gi_off + Self.dim + j] = go * x
                gi_p[gi_off + 2 * Self.dim + j] = go
                j += 1

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
            dtype, Layout.row_major(BATCH, 3 * Self.dim), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, 2 * Self.dim), MutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        var row = idx // Self.dim
        var col = idx % Self.dim
        var x = rebind[Scalar[dtype]](input[row, col])
        var s = rebind[Scalar[dtype]](input[row, Self.dim + col])
        var sh = rebind[Scalar[dtype]](input[row, 2 * Self.dim + col])
        cache[row, col] = x
        cache[row, Self.dim + col] = s
        output[row, col] = x * (Scalar[dtype](1.0) + s) + sh

    @always_inline
    @staticmethod
    def backward_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, 3 * Self.dim), MutAnyOrigin
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
        var go = rebind[Scalar[dtype]](grad_output[row, col])
        var x = rebind[Scalar[dtype]](cache[row, col])
        var s = rebind[Scalar[dtype]](cache[row, Self.dim + col])
        grad_input[row, col] = go * (Scalar[dtype](1.0) + s)
        grad_input[row, Self.dim + col] = go * x
        grad_input[row, 2 * Self.dim + col] = go

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
            dtype, Layout.row_major(BATCH, 3 * Self.dim), ImmutAnyOrigin
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
                dtype, Layout.row_major(BATCH, 3 * Self.dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, 2 * Self.dim), MutAnyOrigin
            ],
        ):
            Self.eval_kernel_impl[BATCH, dtype](output, input, cache)

        ctx.enqueue_function[wrapper](
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
                dtype, Layout.row_major(BATCH, 3 * Self.dim), MutAnyOrigin
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

        ctx.enqueue_function[wrapper](
            grad_input,
            grad_output_immut,
            cache_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )
