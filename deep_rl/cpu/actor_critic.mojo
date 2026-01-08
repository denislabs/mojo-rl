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


# =============================================================================
# Stochastic Actor Network (for SAC)
# =============================================================================


struct StochasticActor[
    obs_dim: Int,
    action_dim: Int,
    hidden1_dim: Int = 256,
    hidden2_dim: Int = 256,
    dtype: DType = DType.float64,
]:
    """Stochastic Actor network for SAC.

    Architecture: obs_dim -> hidden1 (relu) -> hidden2 (relu) -> (mean, log_std).
    Outputs a Gaussian distribution over actions with learned mean and log_std.
    Actions are sampled using the reparameterization trick: a = tanh(mean + std * noise).
    """

    # Shared layers
    var layer1: LinearAdam[Self.obs_dim, Self.hidden1_dim, Self.dtype]
    var layer2: LinearAdam[Self.hidden1_dim, Self.hidden2_dim, Self.dtype]

    # Separate heads for mean and log_std
    var mean_layer: LinearAdam[Self.hidden2_dim, Self.action_dim, Self.dtype]
    var logstd_layer: LinearAdam[Self.hidden2_dim, Self.action_dim, Self.dtype]

    # Action scaling
    var action_scale: Scalar[Self.dtype]
    var action_bias: Scalar[Self.dtype]

    # Log std bounds for numerical stability
    comptime LOG_STD_MIN = Scalar[Self.dtype](-20.0)
    comptime LOG_STD_MAX = Scalar[Self.dtype](2.0)

    fn __init__(
        out self,
        action_scale: Scalar[Self.dtype] = 1.0,
        action_bias: Scalar[Self.dtype] = 0.0,
    ):
        """Initialize Stochastic Actor network.

        Args:
            action_scale: Scale factor for actions (max_action - min_action) / 2.
            action_bias: Bias for actions (max_action + min_action) / 2.
        """
        self.layer1 = LinearAdam[Self.obs_dim, Self.hidden1_dim, Self.dtype]()
        self.layer2 = LinearAdam[Self.hidden1_dim, Self.hidden2_dim, Self.dtype]()
        self.mean_layer = LinearAdam[Self.hidden2_dim, Self.action_dim, Self.dtype]()
        self.logstd_layer = LinearAdam[Self.hidden2_dim, Self.action_dim, Self.dtype]()
        self.action_scale = action_scale
        self.action_bias = action_bias

    fn forward[batch_size: Int](
        mut self,
        obs: InlineArray[Scalar[Self.dtype], batch_size * Self.obs_dim],
    ) -> Tuple[
        InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
        InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
    ]:
        """Forward pass: obs -> (mean, log_std).

        Returns:
            Tuple of (mean, log_std) arrays for the Gaussian policy.
            Note: log_std is clamped to [LOG_STD_MIN, LOG_STD_MAX].
        """
        # Layer 1: linear + relu
        var h1_pre = self.layer1.forward[batch_size](obs)
        var h1 = relu[batch_size * Self.hidden1_dim, Self.dtype](h1_pre)

        # Layer 2: linear + relu
        var h2_pre = self.layer2.forward[batch_size](h1)
        var h2 = relu[batch_size * Self.hidden2_dim, Self.dtype](h2_pre)

        # Mean head (no activation - applied later with tanh during sampling)
        var mean = self.mean_layer.forward[batch_size](h2)

        # Log std head (clamped for stability)
        var log_std_raw = self.logstd_layer.forward[batch_size](h2)
        var log_std = InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim](fill=0)
        for i in range(batch_size * Self.action_dim):
            var val = log_std_raw[i]
            if val < Self.LOG_STD_MIN:
                val = Self.LOG_STD_MIN
            elif val > Self.LOG_STD_MAX:
                val = Self.LOG_STD_MAX
            log_std[i] = val

        return (mean^, log_std^)

    fn forward_with_cache[batch_size: Int](
        mut self,
        obs: InlineArray[Scalar[Self.dtype], batch_size * Self.obs_dim],
        mut h1_out: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden1_dim],
        mut h2_out: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden2_dim],
    ) -> Tuple[
        InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
        InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
    ]:
        """Forward pass with cached activations for backward."""
        var h1_pre = self.layer1.forward[batch_size](obs)
        var h1 = relu[batch_size * Self.hidden1_dim, Self.dtype](h1_pre)

        var h2_pre = self.layer2.forward[batch_size](h1)
        var h2 = relu[batch_size * Self.hidden2_dim, Self.dtype](h2_pre)

        # Store caches
        for i in range(batch_size * Self.hidden1_dim):
            h1_out[i] = h1[i]
        for i in range(batch_size * Self.hidden2_dim):
            h2_out[i] = h2[i]

        var mean = self.mean_layer.forward[batch_size](h2)

        var log_std_raw = self.logstd_layer.forward[batch_size](h2)
        var log_std = InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim](fill=0)
        for i in range(batch_size * Self.action_dim):
            var val = log_std_raw[i]
            if val < Self.LOG_STD_MIN:
                val = Self.LOG_STD_MIN
            elif val > Self.LOG_STD_MAX:
                val = Self.LOG_STD_MAX
            log_std[i] = val

        return (mean^, log_std^)

    fn sample_action[batch_size: Int](
        mut self,
        obs: InlineArray[Scalar[Self.dtype], batch_size * Self.obs_dim],
        noise: InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
    ) -> Tuple[
        InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
        InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
    ]:
        """Sample actions using reparameterization trick.

        a = tanh(mean + std * noise) * action_scale + action_bias

        Args:
            obs: Observations of shape (batch_size, obs_dim).
            noise: Pre-sampled Gaussian noise of shape (batch_size, action_dim).

        Returns:
            Tuple of (actions, log_probs).
        """
        var mean_logstd = self.forward[batch_size](obs)
        var mean = mean_logstd[0]
        var log_std = mean_logstd[1]

        var actions = InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim](fill=0)
        var log_probs = InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim](fill=0)

        comptime LOG_2PI_HALF = Scalar[Self.dtype](0.9189385332)  # 0.5 * log(2 * pi)

        for i in range(batch_size * Self.action_dim):
            var std = _exp_approx(log_std[i])
            var raw_action = mean[i] + std * noise[i]

            # Tanh squashing
            var tanh_action = _tanh_approx(raw_action)
            actions[i] = tanh_action * self.action_scale + self.action_bias

            # Log probability with tanh correction
            # log π(a|s) = log N(u; μ, σ²) - log(1 - tanh²(u))
            var log_prob = -0.5 * noise[i] * noise[i] - log_std[i] - LOG_2PI_HALF

            # Tanh correction: subtract log(1 - tanh²(u))
            var tanh_sq = tanh_action * tanh_action
            if tanh_sq > 0.999999:
                tanh_sq = 0.999999
            log_prob -= _log_approx(1.0 - tanh_sq + 1e-6)

            log_probs[i] = log_prob

        return (actions^, log_probs^)

    fn sample_action_with_cache[batch_size: Int](
        mut self,
        obs: InlineArray[Scalar[Self.dtype], batch_size * Self.obs_dim],
        noise: InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
        mut h1_out: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden1_dim],
        mut h2_out: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden2_dim],
        mut mean_out: InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
        mut log_std_out: InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
        mut tanh_out: InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
    ) -> Tuple[
        InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
        InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
    ]:
        """Sample actions with cached activations for backward pass."""
        var mean_logstd = self.forward_with_cache[batch_size](obs, h1_out, h2_out)
        var mean = mean_logstd[0]
        var log_std = mean_logstd[1]

        # Store mean and log_std for backward
        for i in range(batch_size * Self.action_dim):
            mean_out[i] = mean[i]
            log_std_out[i] = log_std[i]

        var actions = InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim](fill=0)
        var log_probs = InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim](fill=0)

        comptime LOG_2PI_HALF = Scalar[Self.dtype](0.9189385332)

        for i in range(batch_size * Self.action_dim):
            var std = _exp_approx(log_std[i])
            var raw_action = mean[i] + std * noise[i]

            var tanh_action = _tanh_approx(raw_action)
            tanh_out[i] = tanh_action
            actions[i] = tanh_action * self.action_scale + self.action_bias

            var log_prob = -0.5 * noise[i] * noise[i] - log_std[i] - LOG_2PI_HALF
            var tanh_sq = tanh_action * tanh_action
            if tanh_sq > 0.999999:
                tanh_sq = 0.999999
            log_prob -= _log_approx(1.0 - tanh_sq + 1e-6)
            log_probs[i] = log_prob

        return (actions^, log_probs^)

    fn backward[batch_size: Int](
        mut self,
        d_loss: InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
        obs: InlineArray[Scalar[Self.dtype], batch_size * Self.obs_dim],
        noise: InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
        h1: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden1_dim],
        h2: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden2_dim],
        mean: InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
        log_std: InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
        tanh_actions: InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim],
    ):
        """Backward pass through Stochastic Actor.

        Computes gradients for policy optimization: max E[Q(s,a) - α * log π(a|s)]

        Args:
            d_loss: Gradient w.r.t. loss (combines Q gradient and entropy gradient).
            obs: Original observations.
            noise: Sampled noise used for reparameterization.
            h1, h2: Cached hidden activations.
            mean, log_std: Cached policy parameters.
            tanh_actions: Cached tanh(raw_action) values.
        """
        # d_loss combines: ∂Q/∂a * ∂a/∂θ - α * ∂log_π/∂θ
        # We need to backprop through: a = tanh(mean + std * noise) * scale + bias

        var d_mean = InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim](fill=0)
        var d_log_std = InlineArray[Scalar[Self.dtype], batch_size * Self.action_dim](fill=0)

        for i in range(batch_size * Self.action_dim):
            var tanh_sq = tanh_actions[i] * tanh_actions[i]
            var dtanh = 1.0 - tanh_sq  # Derivative of tanh

            var std = _exp_approx(log_std[i])

            # Gradient through action = tanh(raw) * scale
            # ∂a/∂raw = scale * (1 - tanh²)
            var da_draw = self.action_scale * dtanh

            # ∂raw/∂mean = 1, ∂raw/∂std = noise, ∂std/∂log_std = std
            d_mean[i] = d_loss[i] * da_draw
            d_log_std[i] = d_loss[i] * da_draw * noise[i] * std

        # Backward through mean head
        var dh2_mean = self.mean_layer.backward[batch_size](d_mean, h2)

        # Backward through log_std head
        var dh2_logstd = self.logstd_layer.backward[batch_size](d_log_std, h2)

        # Combine gradients from both heads
        var dh2 = InlineArray[Scalar[Self.dtype], batch_size * Self.hidden2_dim](fill=0)
        for i in range(batch_size * Self.hidden2_dim):
            dh2[i] = dh2_mean[i] + dh2_logstd[i]

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
        self.mean_layer.update_adam(lr, beta1, beta2)
        self.logstd_layer.update_adam(lr, beta1, beta2)

    fn zero_grad(mut self):
        """Reset all gradients."""
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.mean_layer.zero_grad()
        self.logstd_layer.zero_grad()

    fn copy_from(mut self, source: Self):
        """Hard copy from source network."""
        self.layer1.copy_from(source.layer1)
        self.layer2.copy_from(source.layer2)
        self.mean_layer.copy_from(source.mean_layer)
        self.logstd_layer.copy_from(source.logstd_layer)

    fn num_parameters(self) -> Int:
        """Total number of parameters."""
        return (
            self.layer1.num_parameters() +
            self.layer2.num_parameters() +
            self.mean_layer.num_parameters() +
            self.logstd_layer.num_parameters()
        )

    fn print_info(self, name: String = "StochasticActor"):
        """Print network info."""
        print(name + ":")
        print(
            "  Architecture: " + String(Self.obs_dim) +
            " -> " + String(Self.hidden1_dim) + " (relu)" +
            " -> " + String(Self.hidden2_dim) + " (relu)" +
            " -> (" + String(Self.action_dim) + " mean, " +
            String(Self.action_dim) + " log_std)"
        )
        print("  Total parameters: " + String(self.num_parameters()))


# =============================================================================
# Helper functions for StochasticActor
# =============================================================================


@always_inline
fn _exp_approx[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Fast exponential approximation."""
    from math import exp
    return exp(x)


@always_inline
fn _log_approx[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Fast log approximation."""
    from math import log
    return log(x)


@always_inline
fn _tanh_approx[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Tanh using math module."""
    from math import tanh
    return tanh(x)
