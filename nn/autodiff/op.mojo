from ..constants import dtype, TILE, TPB
from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext


struct OpID:
    """Compile-time enum for DiffOp identification.

    Used by fusion passes to pattern-match on adjacent ops
    via clean integer comparison at compile time.
    """

    var _value: Int

    fn __init__(out self, value: Int):
        self._value = value

    # Arithmetic primitives (1-9)
    comptime MATMUL = OpID(1)
    comptime BIAS_ADD = OpID(2)
    comptime ELEM_ADD = OpID(3)
    comptime ELEM_MUL = OpID(4)
    comptime SCALE = OpID(5)

    # Activations (10-19)
    comptime RELU = OpID(10)
    comptime TANH = OpID(11)
    comptime SIGMOID = OpID(12)
    comptime MISH = OpID(13)
    comptime SOFTMAX = OpID(14)

    # Normalization (20-29)
    comptime LAYER_NORM = OpID(20)
    comptime RMS_NORM = OpID(21)

    # Reduction (30-39)
    comptime REDUCE_SUM = OpID(30)
    comptime REDUCE_MEAN = OpID(31)

    # Regularization (40-49)
    comptime DROPOUT = OpID(40)

    # Spatial (50-59)
    comptime CONV2D = OpID(50)
    comptime MAX_POOL2D = OpID(51)
    comptime AVG_POOL2D = OpID(52)
    comptime FLATTEN = OpID(53)

    # Embedding (60-69)
    comptime EMBEDDING = OpID(60)

    # Attention (70-79)
    comptime SCALED_DOT_PRODUCT_ATTENTION = OpID(70)
    comptime MULTI_HEAD_PROJECTION = OpID(71)

    # Fused ops (100+)
    comptime FUSED_MATMUL_BIAS = OpID(100)
    comptime FUSED_MATMUL_BIAS_RELU = OpID(101)
    comptime FUSED_MATMUL_BIAS_TANH = OpID(102)
    comptime FUSED_MATMUL_BIAS_SIGMOID = OpID(103)
    comptime FUSED_MATMUL_BIAS_MISH = OpID(104)
    comptime FUSED_CONV2D_RELU = OpID(110)
    comptime FUSED_CONV2D_TANH = OpID(111)
    comptime FUSED_CONV2D_SIGMOID = OpID(112)
    comptime FUSED_CONV2D_MISH = OpID(113)

    # Combinators (200+)
    comptime RESIDUAL = OpID(200)
    comptime PARALLEL = OpID(201)

    # User-defined (1000+)
    comptime USER_DEFINED = OpID(1000)


trait DiffOp(Movable & ImplicitlyCopyable):
    """A single differentiable primitive operation.

    Each DiffOp knows:
    - Its OP_ID for compile-time pattern matching (fusion)
    - Its shape signature (IN_DIM -> OUT_DIM)
    - How many parameters it owns
    - What it needs to cache for backward
    - Its forward computation (eval)
    - Its VJP (vector-Jacobian product) for backward
    """

    # Type identity for compile-time fusion pattern matching.
    comptime OP_ID: Int

    comptime IN_DIM: Int
    comptime OUT_DIM: Int
    comptime PARAM_SIZE: Int
    comptime CACHE_SIZE: Int

    # --- CPU ---
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
        ...

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
        ...

    # --- GPU ---
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
        ...

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
        ...


trait FusedOp(DiffOp):
    """A fused operation that replaces multiple sequential DiffOps.

    The fused op must produce identical results to the original sequence,
    but can use a single GPU kernel launch instead of multiple.
    """

    comptime FUSED_COUNT: Int
