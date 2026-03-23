"""HuberOp: Huber (Smooth L1) loss as a DiffOp.

Takes prediction and target concatenated as input, outputs the Huber loss.
More robust to outliers than MSE — useful for DQN with large TD errors.

Forward:
    residual = pred - target
    if |residual| <= delta:
        output = 0.5 * residual^2
    else:
        output = delta * |residual| - 0.5 * delta^2

Backward:
    if |residual| <= delta:
        grad_pred = residual * grad_output
    else:
        grad_pred = delta * sign(residual) * grad_output
    grad_target = 0  (frozen)
"""

from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.math import abs
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext


struct HuberOp[delta: Float64 = 1.0](DiffOp):
    """Huber loss: robust alternative to MSE.

    IN_DIM = 2 (prediction || target concatenated)
    OUT_DIM = 1
    PARAM_SIZE = 0
    CACHE_SIZE = 1 (caches the residual for backward)
    """

    comptime OP_ID: Int = OpID.USER_DEFINED._value + 11
    comptime IN_DIM: Int = 2
    comptime OUT_DIM: Int = 1
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = 1
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
        comptime d = Self.delta
        comptime half_d_sq = 0.5 * d * d
        for b in range(BATCH):
            var pred = rebind[Scalar[dtype]](input[b, 0])
            var target = rebind[Scalar[dtype]](input[b, 1])
            var residual = pred - target
            cache[b, 0] = residual
            var abs_r = abs(residual)
            if Float64(abs_r) <= d:
                output[b, 0] = Scalar[dtype](0.5) * residual * residual
            else:
                output[b, 0] = Scalar[dtype](d) * abs_r - Scalar[dtype](
                    half_d_sq
                )

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
        comptime d = Self.delta
        for b in range(BATCH):
            var g = rebind[Scalar[dtype]](grad_output[b, 0])
            var residual = rebind[Scalar[dtype]](cache[b, 0])
            var abs_r = abs(residual)
            if Float64(abs_r) <= d:
                grad_input[b, 0] = residual * g
            else:
                var sign = Scalar[dtype](1.0) if residual > Scalar[dtype](
                    0.0
                ) else Scalar[dtype](-1.0)
                grad_input[b, 0] = Scalar[dtype](d) * sign * g
            # Target is frozen
            grad_input[b, 1] = Scalar[dtype](0.0)

    # =========================================================================
    # GPU kernels
    # =========================================================================

    @always_inline
    @staticmethod
    def eval_kernel_impl[
        BATCH: Int
    ](
        output: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, 2), ImmutAnyOrigin],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
    ):
        var b = Int(block_dim.x * block_idx.x + thread_idx.x)
        if b >= BATCH:
            return
        var pred = rebind[Scalar[dtype]](input[b, 0])
        var target = rebind[Scalar[dtype]](input[b, 1])
        var residual = pred - target
        cache[b, 0] = residual
        var abs_r = abs(residual)
        var d_s = Scalar[dtype](Self.delta)
        var half_d_sq = Scalar[dtype](0.5 * Self.delta * Self.delta)
        if abs_r <= d_s:
            output[b, 0] = Scalar[dtype](0.5) * residual * residual
        else:
            output[b, 0] = d_s * abs_r - half_d_sq

    @always_inline
    @staticmethod
    def backward_kernel_impl[
        BATCH: Int
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, 2), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
        ],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin],
    ):
        var b = Int(block_dim.x * block_idx.x + thread_idx.x)
        if b >= BATCH:
            return
        var g = rebind[Scalar[dtype]](grad_output[b, 0])
        var residual = rebind[Scalar[dtype]](cache[b, 0])
        var abs_r = abs(residual)
        var d_s = Scalar[dtype](Self.delta)
        if abs_r <= d_s:
            grad_input[b, 0] = residual * g
        else:
            var sign = Scalar[dtype](1.0) if residual > Scalar[dtype](
                0.0
            ) else Scalar[dtype](-1.0)
            grad_input[b, 0] = d_s * sign * g
        grad_input[b, 1] = Scalar[dtype](0.0)

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
            dtype, Layout.row_major(BATCH, 2), ImmutAnyOrigin
        ](input.ptr)
        var grid_x = (BATCH + TPB - 1) // TPB

        @always_inline
        def wrapper(
            o: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
            i: LayoutTensor[dtype, Layout.row_major(BATCH, 2), ImmutAnyOrigin],
            c: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
        ):
            Self.eval_kernel_impl[BATCH](o, i, c)

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
        var go_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
        ](grad_output.ptr)
        var c_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
        ](cache.ptr)
        var grid_x = (BATCH + TPB - 1) // TPB

        @always_inline
        def wrapper(
            gi: LayoutTensor[dtype, Layout.row_major(BATCH, 2), MutAnyOrigin],
            go: LayoutTensor[dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin],
            c: LayoutTensor[dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin],
        ):
            Self.backward_kernel_impl[BATCH](gi, go, c)

        ctx.enqueue_function[wrapper, wrapper](
            grad_input,
            go_immut,
            c_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )
