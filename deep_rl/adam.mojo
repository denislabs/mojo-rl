"""Adam optimizer for neural networks.

Adam (Adaptive Moment Estimation) combines:
- Momentum: Running average of gradients
- RMSprop: Running average of squared gradients

Update rule:
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad^2
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    param = param - lr * m_hat / (sqrt(v_hat) + eps)

Optimized with @always_inline and SIMD vectorization.
"""

from math import sqrt
from sys import simd_width_of


struct AdamState[size: Int, dtype: DType = DType.float64]:
    """Adam optimizer state for a single parameter tensor.

    Stores first and second moment estimates.
    """

    var m: InlineArray[Scalar[Self.dtype], Self.size]  # First moment (mean)
    var v: InlineArray[
        Scalar[Self.dtype], Self.size
    ]  # Second moment (variance)
    var t: Int  # Timestep

    fn __init__(out self):
        """Initialize Adam state with zeros."""
        self.m = InlineArray[Scalar[Self.dtype], Self.size](fill=0)
        self.v = InlineArray[Scalar[Self.dtype], Self.size](fill=0)
        self.t = 0

    @always_inline
    fn step(
        mut self,
        mut params: InlineArray[Scalar[Self.dtype], Self.size],
        grads: InlineArray[Scalar[Self.dtype], Self.size],
        lr: Scalar[Self.dtype] = 0.001,
        beta1: Scalar[Self.dtype] = 0.9,
        beta2: Scalar[Self.dtype] = 0.999,
        eps: Scalar[Self.dtype] = 1e-8,
    ):
        """Perform one Adam update step.

        Args:
            params: Parameters to update (modified in place).
            grads: Gradients for the parameters.
            lr: Learning rate.
            beta1: Exponential decay rate for first moment.
            beta2: Exponential decay rate for second moment.
            eps: Small constant for numerical stability.
        """
        self.t += 1

        # Bias correction factors (precompute once)
        var bias_correction1 = 1.0 - (beta1**self.t)
        var bias_correction2 = 1.0 - (beta2**self.t)
        var one_minus_beta1 = 1.0 - beta1
        var one_minus_beta2 = 1.0 - beta2

        for i in range(Self.size):
            var g = grads[i]
            # Update biased first moment estimate
            self.m[i] = beta1 * self.m[i] + one_minus_beta1 * g

            # Update biased second moment estimate
            self.v[i] = beta2 * self.v[i] + one_minus_beta2 * g * g

            # Compute bias-corrected estimates
            var m_hat = self.m[i] / bias_correction1
            var v_hat = self.v[i] / bias_correction2

            # Update parameters
            params[i] -= lr * m_hat / (sqrt(v_hat) + eps)


struct LinearAdam[
    in_features: Int, out_features: Int, dtype: DType = DType.float64
]:
    """Linear layer with built-in Adam optimizer.

    Combines Linear layer with Adam state for each parameter.
    """

    # Weights and biases
    var W: InlineArray[Scalar[Self.dtype], Self.in_features * Self.out_features]
    var b: InlineArray[Scalar[Self.dtype], Self.out_features]

    # Gradients
    var dW: InlineArray[
        Scalar[Self.dtype], Self.in_features * Self.out_features
    ]
    var db: InlineArray[Scalar[Self.dtype], Self.out_features]

    # Adam state
    var adam_W: AdamState[Self.in_features * Self.out_features, Self.dtype]
    var adam_b: AdamState[Self.out_features, Self.dtype]

    fn __init__(out self):
        """Initialize with Xavier initialization and Adam state."""
        from .tensor import xavier_init, zeros

        self.W = xavier_init[
            Self.in_features * Self.out_features,
            Self.in_features,
            Self.out_features,
            Self.dtype,
        ]()
        self.b = zeros[Self.out_features, Self.dtype]()
        self.dW = zeros[Self.in_features * Self.out_features, Self.dtype]()
        self.db = zeros[Self.out_features, Self.dtype]()
        self.adam_W = AdamState[
            Self.in_features * Self.out_features, Self.dtype
        ]()
        self.adam_b = AdamState[Self.out_features, Self.dtype]()

    @always_inline
    fn forward[
        batch_size: Int
    ](
        mut self,
        x: InlineArray[Scalar[Self.dtype], batch_size * Self.in_features],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.out_features]:
        """Forward pass: y = x @ W + b - SIMD optimized with register blocking.

        Vectorizes along out_features dimension for contiguous W access.
        Uses 4 accumulators to hide memory latency and improve ILP.
        """
        comptime width = simd_width_of[Self.dtype]()
        comptime NUM_ACCUM = 4  # Number of parallel accumulators
        var result = InlineArray[
            Scalar[Self.dtype], batch_size * Self.out_features
        ](fill=0)

        for i in range(batch_size):
            # Process 4 SIMD vectors at once using register blocking
            var j = 0
            while j + NUM_ACCUM * width <= Self.out_features:
                # Initialize 4 accumulators with bias
                var acc0 = self.b.unsafe_ptr().offset(j).load[width=width]()
                var acc1 = self.b.unsafe_ptr().offset(j + width).load[width=width]()
                var acc2 = self.b.unsafe_ptr().offset(j + 2 * width).load[width=width]()
                var acc3 = self.b.unsafe_ptr().offset(j + 3 * width).load[width=width]()

                # Dot product over in_features using 4 accumulators
                for k in range(Self.in_features):
                    var x_val = SIMD[Self.dtype, width](x[i * Self.in_features + k])
                    var w_ptr = self.W.unsafe_ptr().offset(k * Self.out_features + j)
                    acc0 = x_val.fma(w_ptr.load[width=width](), acc0)
                    acc1 = x_val.fma(w_ptr.offset(width).load[width=width](), acc1)
                    acc2 = x_val.fma(w_ptr.offset(2 * width).load[width=width](), acc2)
                    acc3 = x_val.fma(w_ptr.offset(3 * width).load[width=width](), acc3)

                # Store all 4 results
                var r_ptr = result.unsafe_ptr().offset(i * Self.out_features + j)
                r_ptr.store(acc0)
                r_ptr.offset(width).store(acc1)
                r_ptr.offset(2 * width).store(acc2)
                r_ptr.offset(3 * width).store(acc3)
                j += NUM_ACCUM * width

            # Handle remaining SIMD vectors (1-3)
            while j + width <= Self.out_features:
                var acc = self.b.unsafe_ptr().offset(j).load[width=width]()
                for k in range(Self.in_features):
                    var x_val = SIMD[Self.dtype, width](x[i * Self.in_features + k])
                    var w_vec = self.W.unsafe_ptr().offset(k * Self.out_features + j).load[width=width]()
                    acc = x_val.fma(w_vec, acc)
                result.unsafe_ptr().offset(i * Self.out_features + j).store(acc)
                j += width

            # Scalar remainder
            while j < Self.out_features:
                var sum: Scalar[Self.dtype] = self.b[j]
                for k in range(Self.in_features):
                    sum += x[i * Self.in_features + k] * self.W[k * Self.out_features + j]
                result[i * Self.out_features + j] = sum
                j += 1

        return result^

    @always_inline
    fn forward_relu[
        batch_size: Int
    ](
        mut self,
        x: InlineArray[Scalar[Self.dtype], batch_size * Self.in_features],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.out_features]:
        """Fused forward pass with ReLU: y = max(0, x @ W + b).

        Avoids intermediate allocation by applying ReLU inline.
        """
        var result = InlineArray[
            Scalar[Self.dtype], batch_size * Self.out_features
        ](fill=0)

        for i in range(batch_size):
            for j in range(Self.out_features):
                var sum: Scalar[Self.dtype] = self.b[j]

                for k in range(Self.in_features):
                    sum += (
                        x[i * Self.in_features + k]
                        * self.W[k * Self.out_features + j]
                    )
                # Fused ReLU: max(0, sum)
                result[i * Self.out_features + j] = sum if sum > 0 else 0

        return result^

    @always_inline
    fn backward[
        batch_size: Int
    ](
        mut self,
        dy: InlineArray[Scalar[Self.dtype], batch_size * Self.out_features],
        x: InlineArray[Scalar[Self.dtype], batch_size * Self.in_features],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.in_features]:
        """Backward pass: compute gradients - SIMD optimized with register blocking."""
        comptime width = simd_width_of[Self.dtype]()
        comptime NUM_ACCUM = 4  # Number of parallel accumulators

        # Zero gradients
        for i in range(Self.in_features * Self.out_features):
            self.dW[i] = 0
        for i in range(Self.out_features):
            self.db[i] = 0

        # dW = x.T @ dy (SIMD over out_features with register blocking)
        for j in range(Self.in_features):
            var k = 0
            # Process 4 SIMD vectors at once
            while k + NUM_ACCUM * width <= Self.out_features:
                var acc0 = SIMD[Self.dtype, width](0)
                var acc1 = SIMD[Self.dtype, width](0)
                var acc2 = SIMD[Self.dtype, width](0)
                var acc3 = SIMD[Self.dtype, width](0)

                for i in range(batch_size):
                    var x_val = SIMD[Self.dtype, width](x[i * Self.in_features + j])
                    var dy_ptr = dy.unsafe_ptr().offset(i * Self.out_features + k)
                    acc0 = x_val.fma(dy_ptr.load[width=width](), acc0)
                    acc1 = x_val.fma(dy_ptr.offset(width).load[width=width](), acc1)
                    acc2 = x_val.fma(dy_ptr.offset(2 * width).load[width=width](), acc2)
                    acc3 = x_val.fma(dy_ptr.offset(3 * width).load[width=width](), acc3)

                var dw_ptr = self.dW.unsafe_ptr().offset(j * Self.out_features + k)
                dw_ptr.store(acc0)
                dw_ptr.offset(width).store(acc1)
                dw_ptr.offset(2 * width).store(acc2)
                dw_ptr.offset(3 * width).store(acc3)
                k += NUM_ACCUM * width

            # Remaining SIMD vectors
            while k + width <= Self.out_features:
                var acc = SIMD[Self.dtype, width](0)
                for i in range(batch_size):
                    var x_val = SIMD[Self.dtype, width](x[i * Self.in_features + j])
                    var dy_vec = dy.unsafe_ptr().offset(i * Self.out_features + k).load[width=width]()
                    acc = x_val.fma(dy_vec, acc)
                self.dW.unsafe_ptr().offset(j * Self.out_features + k).store(acc)
                k += width

            # Scalar remainder
            while k < Self.out_features:
                var sum: Scalar[Self.dtype] = 0
                for i in range(batch_size):
                    sum += x[i * Self.in_features + j] * dy[i * Self.out_features + k]
                self.dW[j * Self.out_features + k] = sum
                k += 1

        # db = sum(dy, axis=0) - SIMD over out_features with register blocking
        var j = 0
        while j + NUM_ACCUM * width <= Self.out_features:
            var acc0 = SIMD[Self.dtype, width](0)
            var acc1 = SIMD[Self.dtype, width](0)
            var acc2 = SIMD[Self.dtype, width](0)
            var acc3 = SIMD[Self.dtype, width](0)

            for i in range(batch_size):
                var dy_ptr = dy.unsafe_ptr().offset(i * Self.out_features + j)
                acc0 += dy_ptr.load[width=width]()
                acc1 += dy_ptr.offset(width).load[width=width]()
                acc2 += dy_ptr.offset(2 * width).load[width=width]()
                acc3 += dy_ptr.offset(3 * width).load[width=width]()

            var db_ptr = self.db.unsafe_ptr().offset(j)
            db_ptr.store(acc0)
            db_ptr.offset(width).store(acc1)
            db_ptr.offset(2 * width).store(acc2)
            db_ptr.offset(3 * width).store(acc3)
            j += NUM_ACCUM * width

        while j + width <= Self.out_features:
            var acc = SIMD[Self.dtype, width](0)
            for i in range(batch_size):
                var dy_vec = dy.unsafe_ptr().offset(i * Self.out_features + j).load[width=width]()
                acc += dy_vec
            self.db.unsafe_ptr().offset(j).store(acc)
            j += width

        while j < Self.out_features:
            var sum: Scalar[Self.dtype] = 0
            for i in range(batch_size):
                sum += dy[i * Self.out_features + j]
            self.db[j] = sum
            j += 1

        # dx = dy @ W.T - SIMD over out_features with register blocking
        var dx = InlineArray[Scalar[Self.dtype], batch_size * Self.in_features](
            fill=0
        )

        for i in range(batch_size):
            for jj in range(Self.in_features):
                # Use 4 accumulators for dot product
                var acc0 = SIMD[Self.dtype, width](0)
                var acc1 = SIMD[Self.dtype, width](0)
                var acc2 = SIMD[Self.dtype, width](0)
                var acc3 = SIMD[Self.dtype, width](0)

                var k = 0
                while k + NUM_ACCUM * width <= Self.out_features:
                    var dy_ptr = dy.unsafe_ptr().offset(i * Self.out_features + k)
                    var w_ptr = self.W.unsafe_ptr().offset(jj * Self.out_features + k)
                    acc0 = dy_ptr.load[width=width]().fma(w_ptr.load[width=width](), acc0)
                    acc1 = dy_ptr.offset(width).load[width=width]().fma(w_ptr.offset(width).load[width=width](), acc1)
                    acc2 = dy_ptr.offset(2 * width).load[width=width]().fma(w_ptr.offset(2 * width).load[width=width](), acc2)
                    acc3 = dy_ptr.offset(3 * width).load[width=width]().fma(w_ptr.offset(3 * width).load[width=width](), acc3)
                    k += NUM_ACCUM * width

                # Combine accumulators
                var acc = acc0 + acc1 + acc2 + acc3

                # Remainder SIMD iterations
                while k + width <= Self.out_features:
                    var dy_vec = dy.unsafe_ptr().offset(i * Self.out_features + k).load[width=width]()
                    var w_vec = self.W.unsafe_ptr().offset(jj * Self.out_features + k).load[width=width]()
                    acc = dy_vec.fma(w_vec, acc)
                    k += width

                # Reduce SIMD vector to scalar
                var sum: Scalar[Self.dtype] = acc.reduce_add()

                # Scalar remainder
                while k < Self.out_features:
                    sum += dy[i * Self.out_features + k] * self.W[jj * Self.out_features + k]
                    k += 1

                dx[i * Self.in_features + jj] = sum

        return dx^

    @always_inline
    fn update_adam(
        mut self,
        lr: Scalar[Self.dtype] = 0.001,
        beta1: Scalar[Self.dtype] = 0.9,
        beta2: Scalar[Self.dtype] = 0.999,
        eps: Scalar[Self.dtype] = 1e-8,
    ):
        """Update weights using Adam optimizer."""
        self.adam_W.step(self.W, self.dW, lr, beta1, beta2, eps)
        self.adam_b.step(self.b, self.db, lr, beta1, beta2, eps)

    @always_inline
    fn zero_grad(mut self):
        """Reset gradients to zero - SIMD optimized."""
        comptime width = simd_width_of[Self.dtype]()
        comptime W_size = Self.in_features * Self.out_features
        var zero_vec = SIMD[Self.dtype, width](0)

        # Zero dW with SIMD
        var i = 0
        while i + width <= W_size:
            self.dW.unsafe_ptr().offset(i).store(zero_vec)
            i += width
        while i < W_size:
            self.dW[i] = 0
            i += 1

        # Zero db with SIMD
        var j = 0
        while j + width <= Self.out_features:
            self.db.unsafe_ptr().offset(j).store(zero_vec)
            j += width
        while j < Self.out_features:
            self.db[j] = 0
            j += 1

    @always_inline
    fn soft_update_from(
        mut self,
        source: Self,
        tau: Scalar[Self.dtype],
    ):
        """Soft update: self = tau * source + (1 - tau) * self - SIMD optimized.

        Used for target network updates in DDPG/TD3.
        """
        comptime width = simd_width_of[Self.dtype]()
        comptime W_size = Self.in_features * Self.out_features
        var tau_vec = SIMD[Self.dtype, width](tau)
        var one_minus_tau_vec = SIMD[Self.dtype, width](1.0 - tau)

        # Update W with SIMD
        var i = 0
        while i + width <= W_size:
            var src_vec = source.W.unsafe_ptr().offset(i).load[width=width]()
            var self_vec = self.W.unsafe_ptr().offset(i).load[width=width]()
            var result = tau_vec * src_vec + one_minus_tau_vec * self_vec
            self.W.unsafe_ptr().offset(i).store(result)
            i += width
        while i < W_size:
            self.W[i] = tau * source.W[i] + (1.0 - tau) * self.W[i]
            i += 1

        # Update b with SIMD
        var j = 0
        while j + width <= Self.out_features:
            var src_vec = source.b.unsafe_ptr().offset(j).load[width=width]()
            var self_vec = self.b.unsafe_ptr().offset(j).load[width=width]()
            var result = tau_vec * src_vec + one_minus_tau_vec * self_vec
            self.b.unsafe_ptr().offset(j).store(result)
            j += width
        while j < Self.out_features:
            self.b[j] = tau * source.b[j] + (1.0 - tau) * self.b[j]
            j += 1

    @always_inline
    fn copy_from(mut self, source: Self):
        """Hard copy parameters from source."""
        for i in range(Self.in_features * Self.out_features):
            self.W[i] = source.W[i]

        for i in range(Self.out_features):
            self.b[i] = source.b[i]

    fn num_parameters(self) -> Int:
        """Return total number of learnable parameters."""
        return Self.in_features * Self.out_features + Self.out_features
