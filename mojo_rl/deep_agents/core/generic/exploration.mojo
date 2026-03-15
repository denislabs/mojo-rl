"""Exploration strategies for off-policy agents.

Each implementation owns its exploration state (noise_std, etc.) and provides
both CPU and GPU methods. The generic OffPolicyAgent delegates to these
for action selection during training.

Implementations:
  - GaussianNoise: Additive Gaussian noise with decay (DDPG, TD3)
  - StochasticExploration: Sample from learned distribution (SAC)
"""

from std.random import random_float64
from std.math import sqrt

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.gpu.random import gaussian_noise


# =============================================================================
# GaussianNoise — deterministic policy + additive noise (DDPG, TD3)
# =============================================================================


struct GaussianNoise(Movable, Copyable):
    """Gaussian exploration noise with configurable decay.

    Used by DDPG and TD3: action = actor(obs) + N(0, noise_std * action_scale),
    clipped to [-action_scale, action_scale].

    Decay: noise_std *= noise_decay per episode, clamped to noise_min.
    """

    var noise_std: Float64
    var noise_min: Float64
    var noise_decay: Float64

    fn __init__(
        out self,
        noise_std: Float64 = 0.1,
        noise_min: Float64 = 0.01,
        noise_decay: Float64 = 0.995,
    ):
        self.noise_std = noise_std
        self.noise_min = noise_min
        self.noise_decay = noise_decay

    fn __init__(out self, *, copy: Self):
        self.noise_std = copy.noise_std
        self.noise_min = copy.noise_min
        self.noise_decay = copy.noise_decay

    fn __init__(out self, *, deinit take: Self):
        self.noise_std = take.noise_std
        self.noise_min = take.noise_min
        self.noise_decay = take.noise_decay

    fn explore[
        DTYPE: DType
    ](self, action: List[Scalar[DTYPE]], action_scale: Float64) -> List[
        Scalar[DTYPE]
    ]:
        """Add Gaussian noise to deterministic action and clip.

        Args:
            action: Raw actor output scaled by action_scale.
            action_scale: Half-range of the action space.

        Returns:
            Noisy action clipped to [-action_scale, action_scale].
        """
        var result = List[Scalar[DTYPE]](capacity=len(action))
        for i in range(len(action)):
            var a = Float64(action[i])
            a += self.noise_std * action_scale * gaussian_noise()
            if a > action_scale:
                a = action_scale
            elif a < -action_scale:
                a = -action_scale
            result.append(Scalar[DTYPE](a))
        return result^

    fn decay(mut self):
        """Decay noise_std by noise_decay factor (clamped to noise_min)."""
        self.noise_std *= self.noise_decay
        if self.noise_std < self.noise_min:
            self.noise_std = self.noise_min

    fn get_rate(self) -> Float64:
        """Return current noise std for logging."""
        return self.noise_std


# =============================================================================
# StochasticExploration — sample from learned distribution (SAC)
# =============================================================================


struct StochasticExploration(Movable, Copyable):
    """Entropy-regularized stochastic exploration (SAC).

    No external noise is added — exploration comes from the stochastic policy
    itself (Gaussian with learned mean and log_std). This struct is a marker
    that tells the generic agent to use reparameterized sampling rather than
    deterministic + noise.

    The actual reparameterization logic is in the StochasticTarget and
    MaxEntLoss components, not here.
    """

    fn __init__(out self):
        pass

    fn __init__(out self, *, copy: Self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

    fn decay(mut self):
        """No-op: SAC exploration is automatic via entropy regularization."""
        pass

    fn get_rate(self) -> Float64:
        """Return 0.0 — SAC uses entropy, not an explicit explore rate."""
        return 0.0
