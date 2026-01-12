from ..constants import dtype
from .model import Model
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim
from ..nn_gpu import tanh_forward_kernel, tanh_backward_kernel


struct Tanh[dim: Int](Model):
    """Tanh activation: y = tanh(x).

    CACHE_SIZE = dim (caches tanh output for backward pass: dx = dy * (1 - tanh(x)^2))
    """

    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = Self.dim  # Cache tanh output for backward

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
        mut self,
        input: InlineArray[Scalar[dtype], BATCH * Self.dim],
        mut output: InlineArray[Scalar[dtype], BATCH * Self.dim],
        mut cache: InlineArray[Scalar[dtype], BATCH * Self.CACHE_SIZE],
    ):
        """Forward: y = tanh(x).

        Caches tanh output for backward pass (needed for derivative).
        """
        from math import exp

        var x = LayoutTensor[dtype, Layout.row_major(BATCH, Self.dim)](input)
        var y = LayoutTensor[dtype, Layout.row_major(BATCH, Self.dim)](output)
        var c = LayoutTensor[dtype, Layout.row_major(BATCH, Self.dim)](cache)

        for batch in range(BATCH):
            for i in range(Self.dim):
                var val_scalar: Scalar[dtype] = rebind[Scalar[dtype]](
                    x[batch, i]
                )
                var val = Float64(val_scalar)
                # Compute tanh manually: (e^x - e^-x) / (e^x + e^-x)
                var exp_val = exp(val)
                var exp_neg_val = exp(-val)
                var tanh_val = (exp_val - exp_neg_val) / (exp_val + exp_neg_val)
                var t = Scalar[dtype](tanh_val[0])  # Extract scalar from SIMD
                c[batch, i] = t  # Cache tanh output for backward
                y[batch, i] = t

    fn forward[
        BATCH: Int
    ](
        self,
        input: InlineArray[Scalar[dtype], BATCH * Self.dim],
        mut output: InlineArray[Scalar[dtype], BATCH * Self.dim],
    ):
        """Forward pass without caching (for inference)."""
        from math import exp

        var x = LayoutTensor[dtype, Layout.row_major(BATCH, Self.dim)](input)
        var y = LayoutTensor[dtype, Layout.row_major(BATCH, Self.dim)](output)

        for batch in range(BATCH):
            for i in range(Self.dim):
                var val_scalar: Scalar[dtype] = rebind[Scalar[dtype]](
                    x[batch, i]
                )
                var val = Float64(val_scalar)
                var exp_val = exp(val)
                var exp_neg_val = exp(-val)
                var tanh_val = (exp_val - exp_neg_val) / (exp_val + exp_neg_val)
                y[batch, i] = Scalar[dtype](tanh_val[0])

    fn backward[
        BATCH: Int
    ](
        mut self,
        grad_output: InlineArray[Scalar[dtype], BATCH * Self.dim],
        mut grad_input: InlineArray[Scalar[dtype], BATCH * Self.dim],
        cache: InlineArray[Scalar[dtype], BATCH * Self.CACHE_SIZE],
    ):
        """Backward: dx = dy * (1 - tanh(x)^2).

        Uses cached tanh output from forward pass.
        """
        var dy = LayoutTensor[dtype, Layout.row_major(BATCH, Self.dim)](
            grad_output
        )
        var dx = LayoutTensor[dtype, Layout.row_major(BATCH, Self.dim)](
            grad_input
        )
        var c = LayoutTensor[dtype, Layout.row_major(BATCH, Self.dim)](cache)

        for batch in range(BATCH):
            for i in range(Self.dim):
                var t = c[batch, i]  # tanh(x) cached
                dx[batch, i] = dy[batch, i] * (1 - t * t)

    fn zero_grad(mut self):
        """No gradients to zero."""
        pass

    fn get_params(self) -> InlineArray[Scalar[dtype], Self.PARAM_SIZE]:
        """Tanh has no parameters."""
        return InlineArray[Scalar[dtype], 0](uninitialized=True)

    fn set_params(
        mut self, params: InlineArray[Scalar[dtype], Self.PARAM_SIZE]
    ):
        """Tanh has no parameters to set."""
        pass

    fn get_grads(self) -> InlineArray[Scalar[dtype], Self.PARAM_SIZE]:
        """Tanh has no gradients."""
        return InlineArray[Scalar[dtype], 0](uninitialized=True)

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
        """Forward pass on GPU: y = tanh(x). W and b are unused (no params)."""
        # Tanh has no weights - forward just applies activation
        # Note: This doesn't cache for backward - use with separate cache buffer
        from math import exp

        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        var row = idx // Self.dim
        var col = idx % Self.dim
        var val = x[row, col]
        var val_f32 = rebind[Scalar[DType.float32]](val)
        var exp_val = exp(val_f32)
        var exp_neg_val = exp(-val_f32)
        var tanh_val = (exp_val - exp_neg_val) / (exp_val + exp_neg_val)
        output[row, col] = rebind[output.element_type](tanh_val)

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
        tanh_forward_kernel[BATCH, Self.dim](output, x, cache)

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
        # Tanh backward needs cache - this is a placeholder that won't work alone
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
        """Backward pass on GPU: dx = dy * (1 - tanh(x)^2)."""
        tanh_backward_kernel[BATCH, Self.dim](dx, dy, cache)

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
        """Tanh has no weight gradients - no-op."""
        pass
