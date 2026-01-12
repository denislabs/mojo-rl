from ..constants import dtype
from .model import Model
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from ..nn_gpu import relu_forward_kernel, relu_backward_kernel


struct ReLU[dim: Int](Model):
    """ReLU activation: y = max(0, x).

    CACHE_SIZE = dim (caches pre-activation values for backward pass)
    """

    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = Self.dim  # Cache pre-activation for backward

    fn __init__(out self):
        pass

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor for Sequential composition."""
        pass

    fn __copyinit__(out self, other: Self):
        """Copy constructor for Copyable trait."""
        pass

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
        """Forward: y = max(0, x).

        Caches pre-activation values for backward pass.
        Note: params is unused (ReLU has no parameters).
        """
        for batch in range(BATCH):
            for i in range(Self.dim):
                var val = input[batch, i]
                cache[batch, i] = val  # Cache for backward
                output[batch, i] = val if val > 0 else 0

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

        Note: params is unused (ReLU has no parameters).
        """
        for batch in range(BATCH):
            for i in range(Self.dim):
                var val = input[batch, i]
                output[batch, i] = val if val > 0 else 0

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
        """Backward: dx = dy * (x > 0).

        Uses cached pre-activation values from forward pass.
        Note: params and grads are unused (ReLU has no parameters).
        """
        for batch in range(BATCH):
            for i in range(Self.dim):
                var pre = cache[batch, i]
                grad_input[batch, i] = grad_output[batch, i] if pre > 0 else 0

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
        """Forward pass on GPU: y = max(0, x). W and b are unused (no params).
        """
        # ReLU has no weights - forward just applies activation
        # Note: This doesn't cache for backward - use with separate cache buffer
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        var row = idx // Self.dim
        var col = idx % Self.dim
        var val = x[row, col]
        output[row, col] = max(val, 0)

    @always_inline
    @staticmethod
    fn forward_with_cache_kernel[
        BATCH: Int
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        x: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.dim),
            ImmutAnyOrigin,
        ],
        cache: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.dim),
            MutAnyOrigin,
        ],
    ):
        """Forward pass on GPU with caching for backward."""
        relu_forward_kernel[BATCH, Self.dim](output, x, cache)

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
        """Backward pass for input gradient on GPU. W is unused (no params)."""
        # ReLU backward needs cache - this is a placeholder that won't work alone
        # Use backward_with_cache_kernel instead
        pass

    @always_inline
    @staticmethod
    fn backward_with_cache_kernel[
        BATCH: Int
    ](
        dx: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        dy: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.dim),
            ImmutAnyOrigin,
        ],
        cache: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.dim),
            ImmutAnyOrigin,
        ],
    ):
        """Backward pass on GPU: dx = dy * (x > 0)."""
        relu_backward_kernel[BATCH, Self.dim](dx, dy, cache)

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
        """ReLU has no weight gradients - no-op."""
        pass
