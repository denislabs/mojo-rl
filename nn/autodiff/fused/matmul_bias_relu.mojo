"""FusedMatMulBiasReLU — thin wrapper delegating to FusedMatMulBiasActivation."""

from ...constants import dtype, TILE, TPB
from ...autodiff.op import DiffOp, FusedOp, OpID
from .activation import ReLUActivation
from .matmul_bias_act import FusedMatMulBiasActivation
from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext

comptime _Impl = FusedMatMulBiasActivation


struct FusedMatMulBiasReLU[in_dim: Int, out_dim: Int](FusedOp):
    """Fused y = relu(x @ W + b). Delegates to FusedMatMulBiasActivation."""

    comptime OP_ID: Int = OpID.FUSED_MATMUL_BIAS_RELU._value
    comptime IN_DIM: Int = Self.in_dim
    comptime OUT_DIM: Int = Self.out_dim
    comptime PARAM_SIZE: Int = Self.in_dim * Self.out_dim + Self.out_dim
    comptime CACHE_SIZE: Int = Self.in_dim + Self.out_dim
    comptime FUSED_COUNT: Int = 3

    fn __init__(out self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

    fn __init__(out self, *, copy: Self):
        pass

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
        _Impl[Self.in_dim, Self.out_dim, ReLUActivation].eval[BATCH](
            input, output, params, cache
        )

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
        _Impl[Self.in_dim, Self.out_dim, ReLUActivation].vjp[BATCH](
            grad_output, grad_input, params, cache, grad_params
        )

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
    ) raises:
        _Impl[Self.in_dim, Self.out_dim, ReLUActivation].eval_gpu[BATCH](
            ctx, output, input, params, cache
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
    ) raises:
        _Impl[Self.in_dim, Self.out_dim, ReLUActivation].vjp_gpu[BATCH](
            ctx, grad_output, grad_input, params, cache, grad_params
        )
