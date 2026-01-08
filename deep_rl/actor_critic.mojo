"""Actor and Critic networks for continuous control (DDPG/TD3).

Actor: Maps observations to continuous actions
    obs -> hidden1 (relu) -> hidden2 (relu) -> action (tanh)
    Output is scaled to action bounds.

Critic: Estimates Q-value for (state, action) pairs
    (obs, action) -> hidden1 (relu) -> hidden2 (relu) -> Q-value
"""

from .tensor import (
    tanh_activation,
    relu,
    tanh_grad,
    relu_grad,
    elementwise_mul,
    scale,
    zeros,
)
from .adam import LinearAdam


# =============================================================================
# Actor Network
# =============================================================================


struct Actor[
    obs_dim: Int,
    action_dim: Int,
    hidden1_dim: Int = 256,
    hidden2_dim: Int = 256,
    dtype: DType = DType.float64,
]:
    """Actor network for continuous control.

    Architecture: obs_dim -> hidden1 (relu) -> hidden2 (relu) -> action_dim (tanh).
    Output is bounded to [-1, 1] by tanh, then scaled to action bounds.
    """

    var layer1: LinearAdam[Self.obs_dim, Self.hidden1_dim, Self.dtype]
    var layer2: LinearAdam[Self.hidden1_dim, Self.hidden2_dim, Self.dtype]
    var layer3: LinearAdam[Self.hidden2_dim, Self.action_dim, Self.dtype]

    # Action scaling
    var action_scale: Scalar[Self.dtype]
    var action_bias: Scalar[Self.dtype]

    fn __init__(
        out self,
        action_scale: Scalar[Self.dtype] = 1.0,
        action_bias: Scalar[Self.dtype] = 0.0,
    ):
        """Initialize Actor network.

        Args:
            action_scale: Scale factor for actions (max_action - min_action) / 2.
            action_bias: Bias for actions (max_action + min_action) / 2.
        """
        self.layer1 = LinearAdam[Self.obs_dim, Self.hidden1_dim, Self.dtype]()
        self.layer2 = LinearAdam[Self.hidden1_dim, Self.hidden2_dim, Self.dtype]()
        self.layer3 = LinearAdam[Self.hidden2_dim, Self.action_dim, Self.dtype]()
        self.action_scale = action_scale
        self.action_bias = action_bias

    fn forward[batch_size: Int](
        mut self,
        obs: InlineArray[Scalar[Self.dtype], batch_size * Self.obs_dim],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim]:
        """Forward pass: obs -> action.

        Returns actions bounded to [-action_scale + action_bias, action_scale + action_bias].
        """
        # Layer 1: linear + relu (SIMD-optimized relu)
        var h1_pre = self.layer1.forward[batch_size](obs)
        var h1 = relu[batch_size * Self.hidden1_dim, Self.dtype](h1_pre)

        # Layer 2: linear + relu (SIMD-optimized relu)
        var h2_pre = self.layer2.forward[batch_size](h1)
        var h2 = relu[batch_size * Self.hidden2_dim, Self.dtype](h2_pre)

        # Layer 3: linear + tanh + scale
        var out_pre = self.layer3.forward[batch_size](h2)
        var out_tanh = tanh_activation[batch_size * Self.action_dim, Self.dtype](out_pre)

        # Scale output: action = tanh(out) * action_scale + action_bias
        var actions = InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim](fill=0)
        for i in range(batch_size * Self.action_dim):
            actions[i] = out_tanh[i] * self.action_scale + self.action_bias

        return actions^

    fn forward_with_cache[batch_size: Int](
        mut self,
        obs: InlineArray[Scalar[Self.dtype], batch_size * Self.obs_dim],
        mut h1_out: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden1_dim],
        mut h2_out: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden2_dim],
        mut out_tanh: InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim]:
        """Forward pass with cached activations for backward."""
        var h1_pre = self.layer1.forward[batch_size](obs)
        var h1 = relu[batch_size * Self.hidden1_dim, Self.dtype](h1_pre)

        var h2_pre = self.layer2.forward[batch_size](h1)
        var h2 = relu[batch_size * Self.hidden2_dim, Self.dtype](h2_pre)

        var out_pre = self.layer3.forward[batch_size](h2)
        var out_t = tanh_activation[batch_size * Self.action_dim, Self.dtype](out_pre)

        # Store caches
        for i in range(batch_size * Self.hidden1_dim):
            h1_out[i] = h1[i]
        for i in range(batch_size * Self.hidden2_dim):
            h2_out[i] = h2[i]
        for i in range(batch_size * Self.action_dim):
            out_tanh[i] = out_t[i]

        # Scale output
        var actions = InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim](fill=0)
        for i in range(batch_size * Self.action_dim):
            actions[i] = out_t[i] * self.action_scale + self.action_bias

        return actions^

    fn backward[batch_size: Int](
        mut self,
        dactions: InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
        obs: InlineArray[Scalar[Self.dtype], batch_size * Self.obs_dim],
        h1: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden1_dim],
        h2: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden2_dim],
        out_tanh: InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
    ):
        """Backward pass through Actor.

        Args:
            dactions: Gradient w.r.t. actions (from critic).
            obs: Original observations.
            h1, h2: Cached hidden activations.
            out_tanh: Cached tanh output (before scaling).
        """
        # Backprop through scaling: d_out_tanh = dactions * action_scale
        var d_out_tanh = scale[batch_size * Self.action_dim, Self.dtype](dactions, self.action_scale)

        # Backprop through tanh: d_out_pre = d_out_tanh * tanh_grad(out_tanh)
        var tanh_g = tanh_grad[batch_size * Self.action_dim, Self.dtype](out_tanh)
        var d_out_pre = elementwise_mul[batch_size * Self.action_dim, Self.dtype](d_out_tanh, tanh_g)

        # Backward through layer3
        var dh2 = self.layer3.backward[batch_size](d_out_pre, h2)

        # Backprop through relu2
        var relu_g2 = relu_grad[batch_size * Self.hidden2_dim, Self.dtype](h2)
        var dh2_pre = elementwise_mul[batch_size * Self.hidden2_dim, Self.dtype](dh2, relu_g2)

        # Backward through layer2
        var dh1 = self.layer2.backward[batch_size](dh2_pre, h1)

        # Backprop through relu1
        var relu_g1 = relu_grad[batch_size * Self.hidden1_dim, Self.dtype](h1)
        var dh1_pre = elementwise_mul[batch_size * Self.hidden1_dim, Self.dtype](dh1, relu_g1)

        # Backward through layer1
        _ = self.layer1.backward[batch_size](dh1_pre, obs)

    fn update_adam(
        mut self,
        lr: Scalar[Self.dtype] = 0.001,
        beta1: Scalar[Self.dtype] = 0.9,
        beta2: Scalar[Self.dtype] = 0.999,
    ):
        """Update all layers using Adam."""
        self.layer1.update_adam(lr, beta1, beta2)
        self.layer2.update_adam(lr, beta1, beta2)
        self.layer3.update_adam(lr, beta1, beta2)

    fn zero_grad(mut self):
        """Reset all gradients."""
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.layer3.zero_grad()

    fn soft_update_from(mut self, source: Self, tau: Scalar[Self.dtype]):
        """Soft update from source network."""
        self.layer1.soft_update_from(source.layer1, tau)
        self.layer2.soft_update_from(source.layer2, tau)
        self.layer3.soft_update_from(source.layer3, tau)

    fn copy_from(mut self, source: Self):
        """Hard copy from source network."""
        self.layer1.copy_from(source.layer1)
        self.layer2.copy_from(source.layer2)
        self.layer3.copy_from(source.layer3)

    fn num_parameters(self) -> Int:
        """Total number of parameters."""
        return (
            self.layer1.num_parameters() +
            self.layer2.num_parameters() +
            self.layer3.num_parameters()
        )

    fn print_info(self, name: String = "Actor"):
        """Print network info."""
        print(name + ":")
        print(
            "  Architecture: " + String(Self.obs_dim) +
            " -> " + String(Self.hidden1_dim) + " (relu)" +
            " -> " + String(Self.hidden2_dim) + " (relu)" +
            " -> " + String(Self.action_dim) + " (tanh)"
        )
        print("  Total parameters: " + String(self.num_parameters()))


# =============================================================================
# Critic Network
# =============================================================================


struct Critic[
    obs_dim: Int,
    action_dim: Int,
    hidden1_dim: Int = 256,
    hidden2_dim: Int = 256,
    dtype: DType = DType.float64,
]:
    """Critic network for Q-value estimation.

    Architecture: (obs_dim + action_dim) -> hidden1 (relu) -> hidden2 (relu) -> 1.
    Estimates Q(s, a) for continuous control.
    """

    var layer1: LinearAdam[Self.obs_dim + Self.action_dim, Self.hidden1_dim, Self.dtype]
    var layer2: LinearAdam[Self.hidden1_dim, Self.hidden2_dim, Self.dtype]
    var layer3: LinearAdam[Self.hidden2_dim, 1, Self.dtype]

    fn __init__(out self):
        """Initialize Critic network."""
        self.layer1 = LinearAdam[Self.obs_dim + Self.action_dim, Self.hidden1_dim, Self.dtype]()
        self.layer2 = LinearAdam[Self.hidden1_dim, Self.hidden2_dim, Self.dtype]()
        self.layer3 = LinearAdam[Self.hidden2_dim, 1, Self.dtype]()

    fn forward[batch_size: Int](
        mut self,
        obs: InlineArray[Scalar[Self.dtype], batch_size * Self.obs_dim],
        actions: InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
    ) -> InlineArray[Scalar[Self.dtype], batch_size]:
        """Forward pass: (obs, action) -> Q-value.

        Args:
            obs: Observations of shape (batch_size, obs_dim).
            actions: Actions of shape (batch_size, action_dim).

        Returns:
            Q-values of shape (batch_size,).
        """
        # Concatenate obs and actions
        var x = InlineArray[Scalar[Self.dtype], batch_size * (Self.obs_dim + Self.action_dim)](fill=0)
        for i in range(batch_size):
            for j in range(Self.obs_dim):
                x[i * (Self.obs_dim + Self.action_dim) + j] = obs[i * Self.obs_dim + j]
            for j in range(Self.action_dim):
                x[i * (Self.obs_dim + Self.action_dim) + Self.obs_dim + j] = actions[i * Self.action_dim + j]

        # Layer 1: linear + relu
        var h1_pre = self.layer1.forward[batch_size](x)
        var h1 = relu[batch_size * Self.hidden1_dim, Self.dtype](h1_pre)

        # Layer 2: linear + relu
        var h2_pre = self.layer2.forward[batch_size](h1)
        var h2 = relu[batch_size * Self.hidden2_dim, Self.dtype](h2_pre)

        # Layer 3: linear (no activation)
        var q_values = self.layer3.forward[batch_size](h2)

        # Flatten output
        var result = InlineArray[Scalar[Self.dtype], batch_size](fill=0)
        for i in range(batch_size):
            result[i] = q_values[i]

        return result^

    fn forward_with_cache[batch_size: Int](
        mut self,
        obs: InlineArray[Scalar[Self.dtype], batch_size * Self.obs_dim],
        actions: InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
        mut x_out: InlineArray[Scalar[Self.dtype], batch_size * (Self.obs_dim + Self.action_dim)],
        mut h1_out: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden1_dim],
        mut h2_out: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden2_dim],
    ) -> InlineArray[Scalar[Self.dtype], batch_size]:
        """Forward pass with cached activations."""
        # Concatenate obs and actions
        for i in range(batch_size):
            for j in range(Self.obs_dim):
                x_out[i * (Self.obs_dim + Self.action_dim) + j] = obs[i * Self.obs_dim + j]
            for j in range(Self.action_dim):
                x_out[i * (Self.obs_dim + Self.action_dim) + Self.obs_dim + j] = actions[i * Self.action_dim + j]

        var h1_pre = self.layer1.forward[batch_size](x_out)
        var h1 = relu[batch_size * Self.hidden1_dim, Self.dtype](h1_pre)

        var h2_pre = self.layer2.forward[batch_size](h1)
        var h2 = relu[batch_size * Self.hidden2_dim, Self.dtype](h2_pre)

        # Store caches
        for i in range(batch_size * Self.hidden1_dim):
            h1_out[i] = h1[i]
        for i in range(batch_size * Self.hidden2_dim):
            h2_out[i] = h2[i]

        var q_values = self.layer3.forward[batch_size](h2)

        var result = InlineArray[Scalar[Self.dtype], batch_size](fill=0)
        for i in range(batch_size):
            result[i] = q_values[i]

        return result^

    fn backward[batch_size: Int](
        mut self,
        dq: InlineArray[Scalar[Self.dtype], batch_size],
        x: InlineArray[Scalar[Self.dtype], batch_size * (Self.obs_dim + Self.action_dim)],
        h1: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden1_dim],
        h2: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden2_dim],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim]:
        """Backward pass through Critic.

        Args:
            dq: Gradient w.r.t. Q-values (from TD error).
            x: Cached concatenated input.
            h1: Cached first hidden layer activations.
            h2: Cached second hidden layer activations.

        Returns:
            Gradient w.r.t. actions (for actor update).
        """
        # Backward through layer3 (dq already has correct shape)
        var dh2 = self.layer3.backward[batch_size](dq, h2)

        # Backprop through relu2
        var relu_g2 = relu_grad[batch_size * Self.hidden2_dim, Self.dtype](h2)
        var dh2_pre = elementwise_mul[batch_size * Self.hidden2_dim, Self.dtype](dh2, relu_g2)

        # Backward through layer2
        var dh1 = self.layer2.backward[batch_size](dh2_pre, h1)

        # Backprop through relu1
        var relu_g1 = relu_grad[batch_size * Self.hidden1_dim, Self.dtype](h1)
        var dh1_pre = elementwise_mul[batch_size * Self.hidden1_dim, Self.dtype](dh1, relu_g1)

        # Backward through layer1
        var dx = self.layer1.backward[batch_size](dh1_pre, x)

        # Extract action gradients from dx
        var dactions = InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim](fill=0)
        for i in range(batch_size):
            for j in range(Self.action_dim):
                dactions[i * Self.action_dim + j] = dx[i * (Self.obs_dim + Self.action_dim) + Self.obs_dim + j]

        return dactions^

    fn update_adam(
        mut self,
        lr: Scalar[Self.dtype] = 0.001,
        beta1: Scalar[Self.dtype] = 0.9,
        beta2: Scalar[Self.dtype] = 0.999,
    ):
        """Update all layers using Adam."""
        self.layer1.update_adam(lr, beta1, beta2)
        self.layer2.update_adam(lr, beta1, beta2)
        self.layer3.update_adam(lr, beta1, beta2)

    fn zero_grad(mut self):
        """Reset all gradients."""
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.layer3.zero_grad()

    fn soft_update_from(mut self, source: Self, tau: Scalar[Self.dtype]):
        """Soft update from source network."""
        self.layer1.soft_update_from(source.layer1, tau)
        self.layer2.soft_update_from(source.layer2, tau)
        self.layer3.soft_update_from(source.layer3, tau)

    fn copy_from(mut self, source: Self):
        """Hard copy from source network."""
        self.layer1.copy_from(source.layer1)
        self.layer2.copy_from(source.layer2)
        self.layer3.copy_from(source.layer3)

    fn num_parameters(self) -> Int:
        """Total number of parameters."""
        return (
            self.layer1.num_parameters() +
            self.layer2.num_parameters() +
            self.layer3.num_parameters()
        )

    fn print_info(self, name: String = "Critic"):
        """Print network info."""
        print(name + ":")
        print(
            "  Architecture: (" + String(Self.obs_dim) + " + " + String(Self.action_dim) + ")" +
            " -> " + String(Self.hidden1_dim) + " (relu)" +
            " -> " + String(Self.hidden2_dim) + " (relu)" +
            " -> 1"
        )
        print("  Total parameters: " + String(self.num_parameters()))
