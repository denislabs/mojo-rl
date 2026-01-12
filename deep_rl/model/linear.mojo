from ..constants import dtype
from .model import Model
from layout import LayoutTensor, Layout
from gpu import thread_idx
from math import sqrt
from random import random_float64

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
    """Linear layer: y = x @ W + b.

    Parameters stored as LayoutTensor backed by InlineArray:
    - W: [in_dim, out_dim] weight matrix
    - b: [out_dim] bias vector

    PARAM_SIZE = in_dim * out_dim + out_dim (W flattened + b)
    CACHE_SIZE = in_dim (caches input for weight gradient computation)
    """

    comptime IN_DIM: Int = Self.in_dim
    comptime OUT_DIM: Int = Self.out_dim
    comptime PARAM_SIZE: Int = Self.IN_DIM * Self.OUT_DIM + Self.OUT_DIM
    comptime CACHE_SIZE: Int = Self.IN_DIM  # Cache input for dW computation

    # Storage (InlineArray - stack allocated)
    var W_storage: InlineArray[Scalar[dtype], Self.in_dim * Self.out_dim]
    var b_storage: InlineArray[Scalar[dtype], Self.out_dim]
    var dW_storage: InlineArray[Scalar[dtype], Self.in_dim * Self.out_dim]
    var db_storage: InlineArray[Scalar[dtype], Self.out_dim]

    fn __init__(out self):
        """Initialize with Xavier initialization."""
        # Initialize storage
        self.W_storage = InlineArray[Scalar[dtype], Self.in_dim * Self.out_dim](
            uninitialized=True
        )
        self.b_storage = InlineArray[Scalar[dtype], Self.out_dim](
            uninitialized=True
        )
        self.dW_storage = InlineArray[
            Scalar[dtype], Self.in_dim * Self.out_dim
        ](uninitialized=True)
        self.db_storage = InlineArray[Scalar[dtype], Self.out_dim](
            uninitialized=True
        )

        # Create LayoutTensor views for initialization
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim)
        ](self.W_storage)
        var b = LayoutTensor[dtype, Layout.row_major(Self.out_dim)](
            self.b_storage
        )
        var dW = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim)
        ](self.dW_storage)
        var db = LayoutTensor[dtype, Layout.row_major(Self.out_dim)](
            self.db_storage
        )

        # Xavier initialization for weights
        var std = sqrt(2.0 / Float64(Self.in_dim + Self.out_dim))
        for i in range(Self.in_dim):
            for j in range(Self.out_dim):
                W[i, j] = Scalar[dtype]((random_float64() * 2.0 - 1.0) * std)
                dW[i, j] = 0

        # Zero bias
        for j in range(Self.out_dim):
            b[j] = 0
            db[j] = 0

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor for Sequential composition."""
        self.W_storage = other.W_storage^
        self.b_storage = other.b_storage^
        self.dW_storage = other.dW_storage^
        self.db_storage = other.db_storage^

    fn __copyinit__(out self, other: Self):
        """Copy constructor for Copyable trait."""
        self.W_storage = other.W_storage
        self.b_storage = other.b_storage
        self.dW_storage = other.dW_storage
        self.db_storage = other.db_storage

    fn forward[
        BATCH: Int
    ](
        mut self,
        input: InlineArray[Scalar[dtype], BATCH * Self.IN_DIM],
        mut output: InlineArray[Scalar[dtype], BATCH * Self.OUT_DIM],
        mut cache: InlineArray[Scalar[dtype], BATCH * Self.CACHE_SIZE],
    ):
        """Forward pass: output = input @ W + b.

        Caches the input for backward pass (needed for weight gradients).
        """
        # Create LayoutTensor views for readable indexing
        var x = LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM)](input)
        var y = LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM)](
            output
        )
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim)
        ](self.W_storage)
        var b = LayoutTensor[dtype, Layout.row_major(Self.out_dim)](
            self.b_storage
        )
        var c = LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM)](cache)

        # Cache input for backward
        for batch in range(BATCH):
            for i in range(Self.in_dim):
                c[batch, i] = x[batch, i]

        # Compute y = x @ W + b
        for batch in range(BATCH):
            for j in range(Self.out_dim):
                var acc = b[j]
                for i in range(Self.in_dim):
                    acc += x[batch, i] * W[i, j]

                y[batch, j] = acc

    fn forward[
        BATCH: Int
    ](
        self,
        input: InlineArray[Scalar[dtype], BATCH * Self.IN_DIM],
        mut output: InlineArray[Scalar[dtype], BATCH * Self.OUT_DIM],
    ):
        """Forward pass without caching (for inference)."""
        var x = LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM)](input)
        var y = LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM)](
            output
        )
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim)
        ](self.W_storage)
        var b = LayoutTensor[dtype, Layout.row_major(Self.out_dim)](
            self.b_storage
        )

        # Compute y = x @ W + b (no caching)
        for batch in range(BATCH):
            for j in range(Self.out_dim):
                var acc = b[j]
                for i in range(Self.in_dim):
                    acc += x[batch, i] * W[i, j]
                y[batch, j] = acc

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
        mut self,
        grad_output: InlineArray[Scalar[dtype], BATCH * Self.OUT_DIM],
        mut grad_input: InlineArray[Scalar[dtype], BATCH * Self.IN_DIM],
        cache: InlineArray[Scalar[dtype], BATCH * Self.CACHE_SIZE],
    ):
        """Backward pass: compute grad_input and accumulate dW, db.

        Uses cached input from forward pass to compute weight gradients.
        """
        # Create LayoutTensor views
        var dy = LayoutTensor[dtype, Layout.row_major(BATCH, Self.OUT_DIM)](
            grad_output
        )
        var dx = LayoutTensor[dtype, Layout.row_major(BATCH, Self.IN_DIM)](
            grad_input
        )
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim)
        ](self.W_storage)
        var dW = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim)
        ](self.dW_storage)
        var db = LayoutTensor[dtype, Layout.row_major(Self.out_dim)](
            self.db_storage
        )
        var x_cached = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM)
        ](cache)

        for batch in range(BATCH):
            # dx = dy @ W.T
            for i in range(Self.in_dim):
                var acc: dx.element_type = 0
                for j in range(Self.out_dim):
                    acc += dy[batch, j] * W[i, j]
                dx[batch, i] = acc

            # dW += x.T @ dy (accumulated)
            for i in range(Self.in_dim):
                for j in range(Self.out_dim):
                    dW[i, j] = dW[i, j] + x_cached[batch, i] * dy[batch, j]

            # db += sum(dy, axis=0)
            for j in range(Self.out_dim):
                db[j] = db[j] + dy[batch, j]

    fn zero_grad(mut self):
        """Reset gradients to zero."""
        var dW = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim)
        ](self.dW_storage)
        var db = LayoutTensor[dtype, Layout.row_major(Self.out_dim)](
            self.db_storage
        )

        for i in range(Self.in_dim):
            for j in range(Self.out_dim):
                dW[i, j] = 0
        for j in range(Self.out_dim):
            db[j] = 0

    fn get_params(self) -> InlineArray[Scalar[dtype], Self.PARAM_SIZE]:
        """Get flattened parameters [W_flat, b]."""
        var params = InlineArray[Scalar[dtype], Self.PARAM_SIZE](
            uninitialized=True
        )

        # Copy W (flattened)
        for i in range(Self.in_dim * Self.out_dim):
            params[i] = self.W_storage[i]

        # Copy b
        for i in range(Self.out_dim):
            params[Self.in_dim * Self.out_dim + i] = self.b_storage[i]

        return params^

    fn set_params(
        mut self, params: InlineArray[Scalar[dtype], Self.PARAM_SIZE]
    ):
        """Set parameters from flattened array [W_flat, b]."""
        # Copy W
        for i in range(Self.in_dim * Self.out_dim):
            self.W_storage[i] = params[i]

        # Copy b
        for i in range(Self.out_dim):
            self.b_storage[i] = params[Self.in_dim * Self.out_dim + i]

    fn get_grads(self) -> InlineArray[Scalar[dtype], Self.PARAM_SIZE]:
        """Get flattened gradients [dW_flat, db]."""
        var grads = InlineArray[Scalar[dtype], Self.PARAM_SIZE](
            uninitialized=True
        )

        # Copy dW (flattened)
        for i in range(Self.in_dim * Self.out_dim):
            grads[i] = self.dW_storage[i]

        # Copy db
        for i in range(Self.out_dim):
            grads[Self.in_dim * Self.out_dim + i] = self.db_storage[i]

        return grads^
