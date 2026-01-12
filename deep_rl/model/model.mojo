from ..constants import dtype
from layout import LayoutTensor, Layout


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

    All tensors use LayoutTensor for consistent zero-copy views:
    - input/output: [BATCH, DIM] layout
    - params/grads: [PARAM_SIZE] layout (1D)
    - cache: [BATCH, CACHE_SIZE] layout
    """

    comptime IN_DIM: Int
    comptime OUT_DIM: Int
    comptime PARAM_SIZE: Int
    comptime CACHE_SIZE: Int

    # =========================================================================
    # Forward passes
    # =========================================================================

    fn forward[
        BATCH: Int
    ](
        self,
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
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
        input: LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
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
        params: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin],
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
    # GPU kernels (static methods)
    # =========================================================================

    @always_inline
    @staticmethod
    fn forward_kernel[
        BATCH: Int
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        x: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.IN_DIM),
            ImmutAnyOrigin,
        ],
        W: LayoutTensor[
            dtype,
            Layout.row_major(Self.IN_DIM, Self.OUT_DIM),
            ImmutAnyOrigin,
        ],
        b: LayoutTensor[
            dtype,
            Layout.row_major(Self.OUT_DIM),
            ImmutAnyOrigin,
        ],
    ):
        """Forward pass on GPU."""
        ...

    @always_inline
    @staticmethod
    fn backward_dx_kernel[
        BATCH: Int
    ](
        dx: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        dy: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.OUT_DIM),
            ImmutAnyOrigin,
        ],
        W: LayoutTensor[
            dtype,
            Layout.row_major(Self.IN_DIM, Self.OUT_DIM),
            ImmutAnyOrigin,
        ],
    ):
        """Backward pass for input gradient on GPU: dx = dy @ W.T."""
        ...

    @always_inline
    @staticmethod
    fn backward_dW_db_kernel[
        BATCH: Int
    ](
        dW: LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), MutAnyOrigin
        ],
        db: LayoutTensor[dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin],
        x: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.IN_DIM),
            ImmutAnyOrigin,
        ],
        dy: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.OUT_DIM),
            ImmutAnyOrigin,
        ],
    ):
        """Backward pass for weight/bias gradients on GPU: dW = x.T @ dy, db = sum(dy).
        """
        ...
