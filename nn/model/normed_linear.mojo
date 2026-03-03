from ..constants import dtype
from .model import Model
from .sequential import Sequential
from .linear import Linear
from .layer_norm import LayerNorm
from .mish import Mish
from layout import LayoutTensor, Layout
from gpu.host import DeviceContext, DeviceBuffer


struct NormedLinear[in_dim: Int, out_dim: Int](Model):
    """NormedLinear block: Linear → LayerNorm → Mish.

    The base building block for all TDMPC2 MLPs.
    Final projection layers use plain Linear, optionally followed by SimNorm
    (dynamics head) or Sigmoid (termination head).

    Equivalent to Sequential[Linear[in, out], LayerNorm[out], Mish[out]].

    Parameters:
        in_dim: Input dimension.
        out_dim: Output dimension.

    PARAM_SIZE  = Linear.PARAM_SIZE + LayerNorm.PARAM_SIZE + Mish.PARAM_SIZE
                = (in*out + out) + (2*out) + 0
    CACHE_SIZE  = Linear.CACHE_SIZE + LayerNorm.CACHE_SIZE + Mish.CACHE_SIZE
                = in + (out+2) + 2*out
    """

    # Inner sequential type
    comptime _Inner = Sequential[
        Linear[Self.in_dim, Self.out_dim],
        LayerNorm[Self.out_dim],
        Mish[Self.out_dim],
    ]

    comptime IN_DIM: Int = Self.in_dim
    comptime OUT_DIM: Int = Self.out_dim
    comptime PARAM_SIZE: Int = Self._Inner.PARAM_SIZE
    comptime CACHE_SIZE: Int = Self._Inner.CACHE_SIZE
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self._Inner.WORKSPACE_SIZE_PER_SAMPLE

    fn __init__(out self):
        pass

    fn __moveinit__(out self, deinit other: Self):
        pass

    fn __copyinit__(out self, other: Self):
        pass

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
        """Forward: Linear → LayerNorm → Mish (with caching)."""
        Self._Inner.forward[BATCH](input, output, params, cache)

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
        """Forward pass without caching (inference)."""
        Self._Inner.forward[BATCH](input, output, params)

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
        """Backward through Linear → LayerNorm → Mish."""
        Self._Inner.backward[BATCH](
            grad_output, grad_input, params, cache, grads
        )

    @staticmethod
    fn forward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU forward with caching."""
        Self._Inner.forward_gpu[BATCH](
            ctx, output_buf, input_buf, params_buf, cache_buf, workspace_buf
        )

    @staticmethod
    fn forward_gpu_no_cache[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU forward without caching (inference)."""
        Self._Inner.forward_gpu_no_cache[BATCH](
            ctx, output_buf, input_buf, params_buf, workspace_buf
        )

    @staticmethod
    fn backward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        grad_input_buf: DeviceBuffer[dtype],
        grad_output_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
        grads_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU backward."""
        Self._Inner.backward_gpu[BATCH](
            ctx,
            grad_input_buf,
            grad_output_buf,
            params_buf,
            cache_buf,
            grads_buf,
            workspace_buf,
        )
