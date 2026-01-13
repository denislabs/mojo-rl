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

    All tensors use LayoutTensor for consistent zero-copy views:
    - input/output: [BATCH, DIM] layout
    - params/grads: [PARAM_SIZE] layout (1D)
    - cache: [BATCH, CACHE_SIZE] layout
    - workspace: [BATCH * WORKSPACE_SIZE_PER_SAMPLE] layout (1D, GPU only)
    """

    comptime IN_DIM: Int
    comptime OUT_DIM: Int
    comptime PARAM_SIZE: Int
    comptime CACHE_SIZE: Int
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int

    # =========================================================================
    # Forward passes
    # =========================================================================

    fn forward[
        BATCH: Int
    ](
        self,
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

    fn forward[
        BATCH: Int
    ](
        self,
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

    fn backward[
        BATCH: Int
    ](
        self,
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

    @staticmethod
    fn forward_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU forward pass with caching (for training).

        Args:
            ctx: GPU device context.
            output_buf: Output buffer [BATCH * OUT_DIM].
            input_buf: Input buffer [BATCH * IN_DIM].
            params_buf: Parameters buffer [PARAM_SIZE].
            cache_buf: Cache buffer [BATCH * CACHE_SIZE].
        """
        ...

    @staticmethod
    fn forward_gpu_no_cache[
        BATCH: Int
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU forward pass without caching (for inference).

        Args:
            ctx: GPU device context.
            output_buf: Output buffer [BATCH * OUT_DIM].
            input_buf: Input buffer [BATCH * IN_DIM].
            params_buf: Parameters buffer [PARAM_SIZE].
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
        grad_input_buf: DeviceBuffer[dtype],
        grad_output_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
        grads_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU backward pass: compute grad_input and accumulate parameter gradients.

        Args:
            ctx: GPU device context.
            grad_input_buf: Gradient of loss w.r.t. input [BATCH * IN_DIM] (written).
            grad_output_buf: Gradient of loss w.r.t. output [BATCH * OUT_DIM].
            params_buf: Parameters buffer [PARAM_SIZE].
            cache_buf: Cache from forward pass [BATCH * CACHE_SIZE].
            grads_buf: Parameter gradients [PARAM_SIZE] (accumulated).
        """
        ...

    # =========================================================================
    # GPU methods with workspace (for Sequential - avoids internal allocation)
    # =========================================================================

    @staticmethod
    fn forward_gpu_ws[
        BATCH: Int
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU forward pass with pre-allocated workspace (for training).

        For leaf layers (Linear, ReLU, etc.), this just calls forward_gpu.
        For Sequential, workspace holds intermediate activation buffers.

        Args:
            ctx: GPU device context.
            output_buf: Output buffer [BATCH * OUT_DIM].
            input_buf: Input buffer [BATCH * IN_DIM].
            params_buf: Parameters buffer [PARAM_SIZE].
            cache_buf: Cache buffer [BATCH * CACHE_SIZE].
            workspace_buf: Pre-allocated workspace [BATCH * WORKSPACE_SIZE_PER_SAMPLE].
        """
        ...

    @staticmethod
    fn forward_gpu_no_cache_ws[
        BATCH: Int
    ](
        ctx: DeviceContext,
        output_buf: DeviceBuffer[dtype],
        input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU forward pass without caching, with pre-allocated workspace (for inference).

        Args:
            ctx: GPU device context.
            output_buf: Output buffer [BATCH * OUT_DIM].
            input_buf: Input buffer [BATCH * IN_DIM].
            params_buf: Parameters buffer [PARAM_SIZE].
            workspace_buf: Pre-allocated workspace [BATCH * WORKSPACE_SIZE_PER_SAMPLE].
        """
        ...

    @staticmethod
    fn backward_gpu_ws[
        BATCH: Int
    ](
        ctx: DeviceContext,
        grad_input_buf: DeviceBuffer[dtype],
        grad_output_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
        grads_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU backward pass with pre-allocated workspace.

        For leaf layers (Linear, ReLU, etc.), this just calls backward_gpu.
        For Sequential, workspace holds intermediate gradient buffers.

        Args:
            ctx: GPU device context.
            grad_input_buf: Gradient of loss w.r.t. input [BATCH * IN_DIM] (written).
            grad_output_buf: Gradient of loss w.r.t. output [BATCH * OUT_DIM].
            params_buf: Parameters buffer [PARAM_SIZE].
            cache_buf: Cache from forward pass [BATCH * CACHE_SIZE].
            grads_buf: Parameter gradients [PARAM_SIZE] (accumulated).
            workspace_buf: Pre-allocated workspace [BATCH * WORKSPACE_SIZE_PER_SAMPLE].
        """
        ...
