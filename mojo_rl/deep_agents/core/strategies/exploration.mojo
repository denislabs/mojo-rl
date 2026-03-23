"""Exploration strategies for off-policy agents.

Stateless strategy types with compile-time parameters and @staticmethod
methods, following the nn/Model pattern: strategies describe computation,
runtime state (noise_std) lives on the agent.

Fixed hyperparams are compile-time struct parameters (like Adam's LR, BETA1).
Mutable state (noise_std) is passed as function argument (like Adam's lr_scale).

Implementations:
  - GaussianNoise: Additive Gaussian noise with decay (DDPG, TD3)
  - StochasticSample: Marker for entropy-based exploration (SAC)
"""

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.gpu.random import gaussian_noise


trait Explore:
    """Trait for exploration strategies."""

    comptime IS_STOCHASTIC: Bool
    comptime INITIAL_STD: Float64

    @staticmethod
    def explore[
        DTYPE: DType
    ](
        action: List[Scalar[DTYPE]],
        action_scale: Float64,
        noise_std: Float64,
    ) -> List[Scalar[DTYPE]]:
        ...

    @staticmethod
    def decay(mut noise_std: Float64):
        ...

    @staticmethod
    def get_rate(noise_std: Float64) -> Float64:
        ...


# =============================================================================
# GaussianNoise — deterministic policy + additive noise (DDPG, TD3)
# =============================================================================


struct GaussianNoise[
    initial_std: Float64 = 0.1,
    min_std: Float64 = 0.01,
    decay_rate: Float64 = 0.995,
](Explore):
    """Gaussian exploration noise with configurable decay.

    Compile-time params set the decay schedule. Runtime noise_std is
    passed to each method — it lives on the agent, not on this struct.

    Used by DDPG and TD3: action = actor(obs) + N(0, noise_std * action_scale),
    clipped to [-action_scale, action_scale].
    """

    comptime IS_STOCHASTIC: Bool = False
    comptime INITIAL_STD: Float64 = Self.initial_std

    @staticmethod
    def explore[
        DTYPE: DType
    ](
        action: List[Scalar[DTYPE]],
        action_scale: Float64,
        noise_std: Float64,
    ) -> List[Scalar[DTYPE]]:
        """Add Gaussian noise to deterministic action and clip."""
        var result = List[Scalar[DTYPE]](capacity=len(action))
        for i in range(len(action)):
            var a = Float64(action[i])
            a += noise_std * action_scale * gaussian_noise()
            if a > action_scale:
                a = action_scale
            elif a < -action_scale:
                a = -action_scale
            result.append(Scalar[DTYPE](a))
        return result^

    @staticmethod
    def decay(mut noise_std: Float64):
        """Decay noise_std by decay_rate (clamped to min_std)."""
        noise_std *= Self.decay_rate
        if noise_std < Self.min_std:
            noise_std = Self.min_std

    @staticmethod
    def get_rate(noise_std: Float64) -> Float64:
        """Return current noise std for logging."""
        return noise_std


# =============================================================================
# StochasticSample — entropy-based exploration (SAC)
# =============================================================================


struct StochasticSample(Explore):
    """Entropy-regularized stochastic exploration (SAC).

    No external noise — exploration comes from the stochastic policy itself
    (reparameterized Gaussian with learned mean and log_std). This is a
    marker that tells the generic agent to use reparameterized sampling
    rather than deterministic + noise.

    The actual sampling logic lives in target_action (ReparamTarget) and
    actor_loss (MaxEntLoss), not here.
    """

    comptime IS_STOCHASTIC: Bool = True
    comptime INITIAL_STD: Float64 = 0.0

    @staticmethod
    def explore[
        DTYPE: DType
    ](
        action: List[Scalar[DTYPE]],
        action_scale: Float64,
        noise_std: Float64,
    ) -> List[Scalar[DTYPE]]:
        """No-op: SAC exploration is via stochastic policy sampling."""
        return action.copy()

    @staticmethod
    def decay(mut noise_std: Float64):
        """No-op: SAC exploration is via entropy regularization."""
        pass

    @staticmethod
    def get_rate(noise_std: Float64) -> Float64:
        """Return 0.0 — SAC uses entropy, not an explicit explore rate."""
        return 0.0
