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
from std.math import abs as math_abs
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.sys import simd_width_of


comptime _CPU_SIMD_W = simd_width_of[dtype]()


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
        comptime d = Self.delta
        comptime half_d_sq = 0.5 * d * d
        comptime W = _CPU_SIMD_W
        var in_p = input.ptr
        var out_p = output.ptr
        var c_p = cache.ptr
        var delta_v = SIMD[dtype, W](d)
        var half_v = SIMD[dtype, W](0.5)
        var hds_v = SIMD[dtype, W](half_d_sq)
        var b = 0
        while b + W <= BATCH:
            var pair = in_p.load[width=2 * W](2 * b).deinterleave()
            # deinterleave() returns SIMD[dtype, (2*W)/2] tuple — Mojo nightly
            # doesn't fold the width back to W. Explicit rebind required.
            var pred_v = rebind[SIMD[dtype, W]](pair[0])
            var target_v = rebind[SIMD[dtype, W]](pair[1])
            var r = pred_v - target_v
            c_p.store(b, r)
            var abs_r = math_abs(r)
            # Both branches computed; select by |r| <= delta mask.
            var quad = half_v * r * r
            var lin = delta_v * abs_r - hds_v
            out_p.store(b, abs_r.le(delta_v).select(quad, lin))
            b += W
        while b < BATCH:
            var pred = in_p[2 * b]
            var target = in_p[2 * b + 1]
            var r = pred - target
            c_p[b] = r
            var ar = r if r >= 0 else -r
            if Float64(ar) <= d:
                out_p[b] = Scalar[dtype](0.5) * r * r
            else:
                out_p[b] = Scalar[dtype](d) * ar - Scalar[dtype](half_d_sq)
            b += 1

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
        comptime d = Self.delta
        comptime W = _CPU_SIMD_W
        var go_p = grad_output.ptr
        var gi_p = grad_input.ptr
        var c_p = cache.ptr
        var delta_v = SIMD[dtype, W](d)
        var pos_v = SIMD[dtype, W](1)
        var neg_v = SIMD[dtype, W](-1)
        var zero_v = SIMD[dtype, W](0)
        var b = 0
        while b + W <= BATCH:
            var g = go_p.load[width=W](b)
            var r = c_p.load[width=W](b)
            var abs_r = math_abs(r)
            var sign = r.ge(zero_v).select(pos_v, neg_v)
            var d_pred = abs_r.le(delta_v).select(r * g, delta_v * sign * g)
            gi_p.store(2 * b, d_pred.interleave(zero_v))
            b += W
        while b < BATCH:
            var g = go_p[b]
            var r = c_p[b]
            var ar = r if r >= 0 else -r
            if Float64(ar) <= d:
                gi_p[2 * b] = r * g
            else:
                var sgn: Scalar[dtype] = 1 if r > 0 else -1
                gi_p[2 * b] = Scalar[dtype](d) * sgn * g
            gi_p[2 * b + 1] = 0
            b += 1

    # =========================================================================
    # GPU kernels
    # =========================================================================

    @always_inline
    @staticmethod
    def eval_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
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
        BATCH: Int, dtype: DType = DType.float32
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
            dtype, Layout.row_major(BATCH, 2), ImmutAnyOrigin
        ](input.ptr)
        var grid_x = (BATCH + TPB - 1) // TPB

        @parameter
        @always_inline
        def wrapper(
            o: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
            i: LayoutTensor[dtype, Layout.row_major(BATCH, 2), ImmutAnyOrigin],
            c: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
        ):
            Self.eval_kernel_impl[BATCH, dtype](o, i, c)

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
        var go_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
        ](grad_output.ptr)
        var c_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin
        ](cache.ptr)
        var grid_x = (BATCH + TPB - 1) // TPB

        @parameter
        @always_inline
        def wrapper(
            gi: LayoutTensor[dtype, Layout.row_major(BATCH, 2), MutAnyOrigin],
            go: LayoutTensor[dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin],
            c: LayoutTensor[dtype, Layout.row_major(BATCH, 1), ImmutAnyOrigin],
        ):
            Self.backward_kernel_impl[BATCH, dtype](gi, go, c)

        ctx.enqueue_function[wrapper](
            grad_input,
            go_immut,
            c_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )
