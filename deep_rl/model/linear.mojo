from ..constants import dtype
from .model import Model
from layout import LayoutTensor, Layout
from gpu import thread_idx

from ..nn_gpu import (
    generic_matmul_kernel,
    linear_forward_kernel,
    linear_forward_relu_kernel,
    linear_backward_dx_kernel,
    linear_backward_dx_relu_kernel,
    linear_backward_dW_kernel,
    linear_backward_db_kernel,
    linear_backward_dW_db_kernel,
)


struct Linear[in_dim: Int, out_dim: Int](Model):
    """Linear layer: y = x @ W + b (stateless).

    This is a stateless layer - all parameters and gradients are managed externally.
    The caller allocates and passes:
    - params: [W_flat (in_dim * out_dim) | b (out_dim)]
    - grads: [dW_flat (in_dim * out_dim) | db (out_dim)]

    PARAM_SIZE = in_dim * out_dim + out_dim (W flattened + b)
    CACHE_SIZE = in_dim (caches input for weight gradient computation)
    """

    comptime IN_DIM: Int = Self.in_dim
    comptime OUT_DIM: Int = Self.out_dim
    comptime PARAM_SIZE: Int = Self.IN_DIM * Self.OUT_DIM + Self.OUT_DIM
    comptime CACHE_SIZE: Int = Self.IN_DIM  # Cache input for dW computation

    fn __init__(out self):
        """Initialize stateless Linear layer."""
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
        """Forward pass: output = input @ W + b.

        Caches the input for backward pass (needed for weight gradients).

        Args:
            input: Input tensor [BATCH, IN_DIM].
            output: Output tensor [BATCH, OUT_DIM] (written).
            params: Model parameters [W_flat | b].
            cache: Cache buffer [BATCH, IN_DIM] for backward pass (written).
        """
        # Create 2D view of W from params (first in_dim * out_dim elements)
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        # b starts after W in params
        var b_offset = Self.in_dim * Self.out_dim

        # Cache input for backward
        for batch in range(BATCH):
            for i in range(Self.in_dim):
                cache[batch, i] = input[batch, i]

        # Compute y = x @ W + b
        for batch in range(BATCH):
            for j in range(Self.out_dim):
                var acc = params[b_offset + j]  # bias
                for i in range(Self.in_dim):
                    acc += input[batch, i] * W[i, j]
                output[batch, j] = acc

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
            params: Model parameters [W_flat | b].
        """
        # Create 2D view of W from params
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        var b_offset = Self.in_dim * Self.out_dim

        # Compute y = x @ W + b (no caching)
        for batch in range(BATCH):
            for j in range(Self.out_dim):
                var acc = params[b_offset + j]  # bias
                for i in range(Self.in_dim):
                    acc += input[batch, i] * W[i, j]
                output[batch, j] = acc

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
        """Forward pass: y = x @ W + b (no activation)."""
        linear_forward_kernel[BATCH, Self.IN_DIM, Self.OUT_DIM](output, x, W, b)

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
        linear_backward_dx_kernel[BATCH, Self.IN_DIM, Self.OUT_DIM](dx, dy, W)

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
        linear_backward_dW_db_kernel[BATCH, Self.IN_DIM, Self.OUT_DIM](
            dW, db, x, dy
        )

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
        """Backward pass: compute grad_input and accumulate dW, db.

        Uses cached input from forward pass to compute weight gradients.

        Args:
            grad_output: Gradient of loss w.r.t. output [BATCH, OUT_DIM].
            grad_input: Gradient of loss w.r.t. input [BATCH, IN_DIM] (written).
            params: Model parameters [W_flat | b].
            cache: Cached input from forward pass [BATCH, IN_DIM].
            grads: Parameter gradients [dW_flat | db] (accumulated, not overwritten).
        """
        # Create 2D views of W and dW from 1D params/grads
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        var dW = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](grads.ptr)
        var db_offset = Self.in_dim * Self.out_dim

        for batch in range(BATCH):
            # dx = dy @ W.T
            for i in range(Self.in_dim):
                var acc: grad_input.element_type = 0
                for j in range(Self.out_dim):
                    acc += grad_output[batch, j] * W[i, j]
                grad_input[batch, i] = acc

            # dW += x.T @ dy (accumulated)
            for i in range(Self.in_dim):
                for j in range(Self.out_dim):
                    dW[i, j] = (
                        dW[i, j] + cache[batch, i] * grad_output[batch, j]
                    )

            # db += sum(dy, axis=0)
            for j in range(Self.out_dim):
                grads[db_offset + j] = (
                    grads[db_offset + j] + grad_output[batch, j]
                )
