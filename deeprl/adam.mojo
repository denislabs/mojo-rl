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
"""

from math import sqrt


struct AdamState[size: Int, dtype: DType = DType.float64]:
    """Adam optimizer state for a single parameter tensor.

    Stores first and second moment estimates.
    """

    var m: InlineArray[Scalar[Self.dtype], Self.size]  # First moment (mean)
    var v: InlineArray[Scalar[Self.dtype], Self.size]  # Second moment (variance)
    var t: Int  # Timestep

    fn __init__(out self):
        """Initialize Adam state with zeros."""
        self.m = InlineArray[Scalar[Self.dtype], Self.size](fill=0)
        self.v = InlineArray[Scalar[Self.dtype], Self.size](fill=0)
        self.t = 0

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

        # Bias correction factors
        var bias_correction1 = 1.0 - (beta1 ** self.t)
        var bias_correction2 = 1.0 - (beta2 ** self.t)

        for i in range(Self.size):
            # Update biased first moment estimate
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * grads[i]

            # Update biased second moment estimate
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * grads[i] * grads[i]

            # Compute bias-corrected estimates
            var m_hat = self.m[i] / bias_correction1
            var v_hat = self.v[i] / bias_correction2

            # Update parameters
            params[i] -= lr * m_hat / (sqrt(v_hat) + eps)


struct LinearAdam[in_features: Int, out_features: Int, dtype: DType = DType.float64]:
    """Linear layer with built-in Adam optimizer.

    Combines Linear layer with Adam state for each parameter.
    """

    # Weights and biases
    var W: InlineArray[Scalar[Self.dtype], Self.in_features * Self.out_features]
    var b: InlineArray[Scalar[Self.dtype], Self.out_features]

    # Gradients
    var dW: InlineArray[Scalar[Self.dtype], Self.in_features * Self.out_features]
    var db: InlineArray[Scalar[Self.dtype], Self.out_features]

    # Adam state
    var adam_W: AdamState[Self.in_features * Self.out_features, Self.dtype]
    var adam_b: AdamState[Self.out_features, Self.dtype]

    fn __init__(out self):
        """Initialize with Xavier initialization and Adam state."""
        from .tensor import xavier_init, zeros

        self.W = xavier_init[Self.in_features * Self.out_features, Self.in_features, Self.out_features, Self.dtype]()
        self.b = zeros[Self.out_features, Self.dtype]()
        self.dW = zeros[Self.in_features * Self.out_features, Self.dtype]()
        self.db = zeros[Self.out_features, Self.dtype]()
        self.adam_W = AdamState[Self.in_features * Self.out_features, Self.dtype]()
        self.adam_b = AdamState[Self.out_features, Self.dtype]()

    fn forward[batch_size: Int](
        mut self,
        x: InlineArray[Scalar[Self.dtype], batch_size * Self.in_features],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.out_features]:
        """Forward pass: y = x @ W + b."""
        var result = InlineArray[Scalar[Self.dtype], batch_size * Self.out_features](fill=0)

        for i in range(batch_size):
            for j in range(Self.out_features):
                var sum: Scalar[Self.dtype] = self.b[j]
                for k in range(Self.in_features):
                    sum += x[i * Self.in_features + k] * self.W[k * Self.out_features + j]
                result[i * Self.out_features + j] = sum

        return result^

    fn backward[batch_size: Int](
        mut self,
        dy: InlineArray[Scalar[Self.dtype], batch_size * Self.out_features],
        x: InlineArray[Scalar[Self.dtype], batch_size * Self.in_features],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.in_features]:
        """Backward pass: compute gradients."""
        # Zero gradients
        for i in range(Self.in_features * Self.out_features):
            self.dW[i] = 0
        for i in range(Self.out_features):
            self.db[i] = 0

        # dW = x.T @ dy
        for j in range(Self.in_features):
            for k in range(Self.out_features):
                var sum: Scalar[Self.dtype] = 0
                for i in range(batch_size):
                    sum += x[i * Self.in_features + j] * dy[i * Self.out_features + k]
                self.dW[j * Self.out_features + k] = sum

        # db = sum(dy, axis=0)
        for j in range(Self.out_features):
            var sum: Scalar[Self.dtype] = 0
            for i in range(batch_size):
                sum += dy[i * Self.out_features + j]
            self.db[j] = sum

        # dx = dy @ W.T
        var dx = InlineArray[Scalar[Self.dtype], batch_size * Self.in_features](fill=0)
        for i in range(batch_size):
            for j in range(Self.in_features):
                var sum: Scalar[Self.dtype] = 0
                for k in range(Self.out_features):
                    sum += dy[i * Self.out_features + k] * self.W[j * Self.out_features + k]
                dx[i * Self.in_features + j] = sum

        return dx^

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

    fn zero_grad(mut self):
        """Reset gradients to zero."""
        for i in range(Self.in_features * Self.out_features):
            self.dW[i] = 0
        for i in range(Self.out_features):
            self.db[i] = 0

    fn soft_update_from(
        mut self,
        source: Self,
        tau: Scalar[Self.dtype],
    ):
        """Soft update: self = tau * source + (1 - tau) * self.

        Used for target network updates in DDPG/TD3.
        """
        var one_minus_tau = 1.0 - tau

        for i in range(Self.in_features * Self.out_features):
            self.W[i] = tau * source.W[i] + one_minus_tau * self.W[i]

        for i in range(Self.out_features):
            self.b[i] = tau * source.b[i] + one_minus_tau * self.b[i]

    fn copy_from(mut self, source: Self):
        """Hard copy parameters from source."""
        for i in range(Self.in_features * Self.out_features):
            self.W[i] = source.W[i]
        for i in range(Self.out_features):
            self.b[i] = source.b[i]

    fn num_parameters(self) -> Int:
        """Return total number of learnable parameters."""
        return Self.in_features * Self.out_features + Self.out_features
