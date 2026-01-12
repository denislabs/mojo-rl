from ..constants import dtype
from layout import LayoutTensor, Layout


trait Model(Movable & ImplicitlyCopyable):
    """Base trait for neural network modules.

    Modules have:
    - Compile-time dimensions (IN_DIM, OUT_DIM, PARAM_SIZE, CACHE_SIZE)
    - forward() for inference with external cache buffer
    - backward() for gradient computation using cached values
    - zero_grad() to reset gradients
    - get_params/set_params/get_grads for optimizer integration
    - Movable for composition in Sequential containers

    CACHE_SIZE represents the number of elements cached per sample during forward
    pass, needed for backward computation. The caller allocates BATCH * CACHE_SIZE
    elements and passes them to forward/backward.
    """

    comptime IN_DIM: Int
    comptime OUT_DIM: Int
    comptime PARAM_SIZE: Int
    comptime CACHE_SIZE: Int

    fn zero_grad(mut self):
        """Reset all gradients to zero."""
        ...

    fn forward[
        BATCH: Int
    ](
        mut self,
        input: InlineArray[Scalar[dtype], BATCH * Self.IN_DIM],
        mut output: InlineArray[Scalar[dtype], BATCH * Self.OUT_DIM],
        mut cache: InlineArray[Scalar[dtype], BATCH * Self.CACHE_SIZE],
    ):
        """Forward pass with external cache buffer (for training).

        Args:
            input: Input tensor [BATCH, IN_DIM].
            output: Output tensor [BATCH, OUT_DIM] (written by this function).
            cache: Cache buffer [BATCH, CACHE_SIZE] for backward pass.
        """
        ...

    fn forward[
        BATCH: Int
    ](
        self,
        input: InlineArray[Scalar[dtype], BATCH * Self.IN_DIM],
        mut output: InlineArray[Scalar[dtype], BATCH * Self.OUT_DIM],
    ):
        """Forward pass without caching (for inference/evaluation).

        Args:
            input: Input tensor [BATCH, IN_DIM].
            output: Output tensor [BATCH, OUT_DIM] (written by this function).
        """
        ...

    fn backward[
        BATCH: Int
    ](
        mut self,
        grad_output: InlineArray[Scalar[dtype], BATCH * Self.OUT_DIM],
        mut grad_input: InlineArray[Scalar[dtype], BATCH * Self.IN_DIM],
        cache: InlineArray[Scalar[dtype], BATCH * Self.CACHE_SIZE],
    ):
        """Backward pass: compute grad_input and accumulate gradients.

        Args:
            grad_output: Gradient of loss w.r.t. output [BATCH, OUT_DIM].
            grad_input: Gradient of loss w.r.t. input [BATCH, IN_DIM] (written).
            cache: Cache from forward pass [BATCH, CACHE_SIZE].
        """
        ...

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

    fn get_params(self) -> InlineArray[Scalar[dtype], Self.PARAM_SIZE]:
        """Get flattened parameters for optimizer."""
        ...

    fn set_params(
        mut self, params: InlineArray[Scalar[dtype], Self.PARAM_SIZE]
    ):
        """Set parameters from flattened array."""
        ...

    fn get_grads(self) -> InlineArray[Scalar[dtype], Self.PARAM_SIZE]:
        """Get flattened gradients for optimizer."""
        ...
