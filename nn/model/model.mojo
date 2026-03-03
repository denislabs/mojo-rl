from ..constants import dtype
from layout import LayoutTensor, Layout
from gpu.host import DeviceContext, DeviceBuffer


trait Model(Movable & ImplicitlyCopyable):
    """Base trait for neural network modules (stateless).

    Models are stateless - they describe the computation graph but don't store
    weights or gradients. All state (params, grads, cache) is managed externally
    as LayoutTensor views for zero-copy composition.

    Compile-time constants:
    - IN_DIM: Input dimension per sample
    - OUT_DIM: Output dimension per sample
    - PARAM_SIZE: Total number of parameters (e.g., W + b for Linear)
    - CACHE_SIZE: Elements cached per sample during forward (for backward pass)
    - WORKSPACE_SIZE_PER_SAMPLE: GPU workspace needed per sample for intermediate buffers.
      For leaf layers (Linear, ReLU, etc.) this is 0.
      For Sequential, this includes intermediate activation buffers.
      Total workspace = BATCH * WORKSPACE_SIZE_PER_SAMPLE.

    All CPU and GPU tensors use LayoutTensor for consistent zero-copy views:
    - input/output: [BATCH, DIM] layout
    - params/grads: [PARAM_SIZE] layout (1D)
    - cache: [BATCH, CACHE_SIZE] layout

    GPU workspace is kept as DeviceBuffer (size may be 0 for leaf layers,
    allocated as max(1, BATCH * WORKSPACE_SIZE_PER_SAMPLE) by the caller).
    """

    comptime IN_DIM: Int
    comptime OUT_DIM: Int
    comptime PARAM_SIZE: Int
    comptime CACHE_SIZE: Int
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int

    # =========================================================================
    # Forward passes
    # =========================================================================

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
        """Forward pass with caching (for training).

        Args:
            input: Input tensor [BATCH, IN_DIM].
            output: Output tensor [BATCH, OUT_DIM] (written).
            params: Model parameters [PARAM_SIZE].
            cache: Cache buffer [BATCH, CACHE_SIZE] for backward pass (written).
        """
        ...

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
        """Forward pass without caching (for inference).

        Args:
            input: Input tensor [BATCH, IN_DIM].
            output: Output tensor [BATCH, OUT_DIM] (written).
            params: Model parameters [PARAM_SIZE].
        """
        ...

    # =========================================================================
    # Backward pass
    # =========================================================================

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
        """Backward pass: compute grad_input and accumulate parameter gradients.

        Args:
            grad_output: Gradient of loss w.r.t. output [BATCH, OUT_DIM].
            grad_input: Gradient of loss w.r.t. input [BATCH, IN_DIM] (written).
            params: Model parameters [PARAM_SIZE].
            cache: Cache from forward pass [BATCH, CACHE_SIZE].
            grads: Parameter gradients [PARAM_SIZE] (accumulated, not overwritten).
        """
        ...

    # =========================================================================
    # GPU forward passes
    # =========================================================================
    # Shaped tensors (input, output, params, cache) are passed as LayoutTensor
    # for a uniform API matching the CPU interface.
    # workspace is kept as DeviceBuffer because its size may be 0 for leaf
    # layers (allocated as max(1, BATCH * WORKSPACE_SIZE_PER_SAMPLE) by caller).
    # =========================================================================

    @staticmethod
    fn forward_gpu[
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
        workspace: DeviceBuffer[dtype],
    ) raises:
        """GPU forward pass with caching (for training).

        Args:
            ctx: GPU device context.
            output: Output [BATCH, OUT_DIM] (written).
            input: Input [BATCH, IN_DIM].
            params: Parameters [PARAM_SIZE].
            cache: Cache [BATCH, CACHE_SIZE] (written).
            workspace: Pre-allocated workspace for Sequential intermediate buffers.
        """
        ...

    @staticmethod
    fn forward_gpu_no_cache[
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
        workspace: DeviceBuffer[dtype],
    ) raises:
        """GPU forward pass without caching (for inference).

        Args:
            ctx: GPU device context.
            output: Output [BATCH, OUT_DIM] (written).
            input: Input [BATCH, IN_DIM].
            params: Parameters [PARAM_SIZE].
            workspace: Pre-allocated workspace for Sequential intermediate buffers.
        """
        ...

    # =========================================================================
    # GPU backward pass
    # =========================================================================

    @staticmethod
    fn backward_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
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
        workspace: DeviceBuffer[dtype],
    ) raises:
        """GPU backward pass.

        Args:
            ctx: GPU device context.
            grad_input: Gradient w.r.t. input [BATCH, IN_DIM] (written).
            grad_output: Gradient w.r.t. output [BATCH, OUT_DIM].
            params: Parameters [PARAM_SIZE].
            cache: Cache from forward pass [BATCH, CACHE_SIZE].
            grads: Parameter gradients [PARAM_SIZE] (accumulated).
            workspace: Pre-allocated workspace for Sequential intermediate buffers.
        """
        ...
