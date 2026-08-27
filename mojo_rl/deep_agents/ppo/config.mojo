"""PPO named preset — config descriptor + factory (Design F).

Additive sugar over the primitive
`PPOAgent[train_target, ACTOR, CRITIC, OBS_DIM, ACT_DIM, ROLLOUT_LEN,
          MINIBATCH, N_EPOCHS, N_ENVS]`. The primitive stays the source of
truth for arbitrary actor/critic/rollout combinations; this module names
the canonical clipped-surrogate PPO and bundles its tuned defaults + a
default fused-`LinearTanh` actor/critic.

Same shape as `c51/config.mojo`, `redq/config.mojo`, `sac/config.mojo`:

  1. `PPOConfigT` — a trait bundling the FULL compile-time identity: the
     deployment `TARGET`, the actor + critic nets, the structural shape
     (`OBS_DIM` / `ACT_DIM` / `ROLLOUT_LEN` / `MINIBATCH` / `N_EPOCHS` /
     `N_ENVS`), plus tuned scalar defaults (`DEF_*`).

  2. `PPOConfig` — a zero-field conformer struct parametrized by `target`
     and the structural ints. On-policy, so NO replay block.

  3. `agent_from_config` + the capitalized preset `PPO`, which reads like
     a constructor at the call site:

         var agent = PPO["cpu", OBS, ACT, ROLLOUT, MB, EPOCHS]()
         var agent = PPO["gpu", OBS, ACT, ROLLOUT, MB, EPOCHS, N_ENVS](ctx=ctx)

The default nets use FUSED `LinearTanh` (matmul + bias + tanh in one
kernel) for the hidden layers — same parameter layout as the unfused
`Linear` + `Tanh` pair. The actor's `GaussianHead` and the critic's scalar
head stay plain.
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.linear_tanh import LinearTanh
from mojo_rl.nn.combinators.sequential import Sequential

from ..primitives.gaussian_head import GaussianHead

from .agent import PPOAgent


# ──────────────────────────────────────────────────────────────────────
# Net presets — target-agnostic, parametrized comptime aliases.
# Hidden layers fused (LinearTanh); GaussianHead + value head stay plain.
# ──────────────────────────────────────────────────────────────────────


comptime PPOActorNet[OBS: Int, ACT: Int, HIDDEN: Int] = Sequential[
    LinearTanh[OBS, HIDDEN],
    LinearTanh[HIDDEN, HIDDEN],
    GaussianHead[HIDDEN, ACT],
]
"""2-layer fused-tanh trunk + Gaussian (μ, log σ) head — PPO's unbounded
continuous policy."""


comptime PPOCriticNet[OBS: Int, HIDDEN: Int] = Sequential[
    LinearTanh[OBS, HIDDEN],
    LinearTanh[HIDDEN, HIDDEN],
    Linear[HIDDEN, 1],
]
"""2-layer fused-tanh value net V(s)."""


# ──────────────────────────────────────────────────────────────────────
# Config trait — full compile-time identity + tuned scalar defaults.
# ──────────────────────────────────────────────────────────────────────


trait PPOConfigT(Copyable, Movable, Deinitable):
    """Compile-time descriptor of a PPO-family algorithm. Conformers are
    zero-field comptime tags — never instantiated at runtime; only their
    comptime members are read."""

    comptime TARGET: StaticString
    comptime ACTOR: Module
    comptime CRITIC: Module
    comptime OBS_DIM: Int
    comptime ACT_DIM: Int
    comptime ROLLOUT_LEN: Int
    comptime MINIBATCH: Int
    comptime N_EPOCHS: Int
    comptime N_ENVS: Int

    # Tuned scalar defaults (read into __init__ kwarg defaults).
    comptime DEF_ACTOR_LR: Scalar[DT]
    comptime DEF_CRITIC_LR: Scalar[DT]
    comptime DEF_GAMMA: Scalar[DT]
    comptime DEF_GAE_LAMBDA: Scalar[DT]
    comptime DEF_CLIP_EPS: Scalar[DT]
    comptime DEF_ENTROPY_COEF: Scalar[DT]
    comptime DEF_ACTION_SCALE: Scalar[DT]
    comptime DEF_LOG_STD_INIT: Scalar[DT]
    comptime DEF_MAX_GRAD_NORM: Scalar[DT]


# ──────────────────────────────────────────────────────────────────────
# Conformer — one struct, parametrized by `target` + structural ints.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct PPOConfig[
    target: StaticString,
    OBS: Int, ACT: Int,
    ROLLOUT: Int, MB: Int, EPOCHS: Int,
    N_ENVS_: Int = 1, HIDDEN: Int = 64,
](PPOConfigT):
    """Clipped-surrogate PPO (Schulman et al. 2017, CleanRL-style
    unbounded Gaussian). GAE advantages, multi-epoch minibatch SGD. One
    config covers cpu + gpu (on-policy — no replay backend to select)."""

    comptime TARGET = Self.target
    comptime ACTOR = PPOActorNet[Self.OBS, Self.ACT, Self.HIDDEN]
    comptime CRITIC = PPOCriticNet[Self.OBS, Self.HIDDEN]
    comptime OBS_DIM = Self.OBS
    comptime ACT_DIM = Self.ACT
    comptime ROLLOUT_LEN = Self.ROLLOUT
    comptime MINIBATCH = Self.MB
    comptime N_EPOCHS = Self.EPOCHS
    comptime N_ENVS = Self.N_ENVS_

    comptime DEF_ACTOR_LR = Scalar[DT](3e-4)
    comptime DEF_CRITIC_LR = Scalar[DT](1e-3)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_GAE_LAMBDA = Scalar[DT](0.95)
    comptime DEF_CLIP_EPS = Scalar[DT](0.2)
    comptime DEF_ENTROPY_COEF = Scalar[DT](0.0)
    comptime DEF_ACTION_SCALE = Scalar[DT](1.0)
    comptime DEF_LOG_STD_INIT = Scalar[DT](-0.5)
    comptime DEF_MAX_GRAD_NORM = Scalar[DT](0.0)


# ──────────────────────────────────────────────────────────────────────
# Generic factory — any PPOConfigT → primitive agent, defaults applied.
# ──────────────────────────────────────────────────────────────────────


def agent_from_config[
    CONFIG: PPOConfigT,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = CONFIG.DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = CONFIG.DEF_CRITIC_LR,
    gamma: Scalar[DT] = CONFIG.DEF_GAMMA,
    gae_lambda: Scalar[DT] = CONFIG.DEF_GAE_LAMBDA,
    clip_eps: Scalar[DT] = CONFIG.DEF_CLIP_EPS,
    entropy_coef: Scalar[DT] = CONFIG.DEF_ENTROPY_COEF,
    action_scale: Scalar[DT] = CONFIG.DEF_ACTION_SCALE,
    log_std_init: Scalar[DT] = CONFIG.DEF_LOG_STD_INIT,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1600.0),
    max_grad_norm: Scalar[DT] = CONFIG.DEF_MAX_GRAD_NORM,
) raises -> PPOAgent[
    CONFIG.TARGET,
    CONFIG.ACTOR,
    CONFIG.CRITIC,
    CONFIG.OBS_DIM,
    CONFIG.ACT_DIM,
    CONFIG.ROLLOUT_LEN,
    CONFIG.MINIBATCH,
    CONFIG.N_EPOCHS,
    CONFIG.N_ENVS,
]:
    """Build the primitive `PPOAgent` from any `PPOConfigT`. Every scalar
    defaults to the config's tuned value but stays overridable."""
    return PPOAgent[
        CONFIG.TARGET,
        CONFIG.ACTOR,
        CONFIG.CRITIC,
        CONFIG.OBS_DIM,
        CONFIG.ACT_DIM,
        CONFIG.ROLLOUT_LEN,
        CONFIG.MINIBATCH,
        CONFIG.N_EPOCHS,
        CONFIG.N_ENVS,
    ](
        ctx=ctx,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        entropy_coef=entropy_coef,
        action_scale=action_scale,
        log_std_init=log_std_init,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
        max_grad_norm=max_grad_norm,
    )


# ──────────────────────────────────────────────────────────────────────
# Capitalized preset — single function, `target` as a parameter.
# ──────────────────────────────────────────────────────────────────────


def PPO[
    target: StaticString,
    OBS: Int, ACT: Int,
    ROLLOUT: Int, MB: Int, EPOCHS: Int,
    N_ENVS: Int = 1, HIDDEN: Int = 64,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = PPOConfig[target, OBS, ACT, ROLLOUT, MB, EPOCHS, N_ENVS, HIDDEN].DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = PPOConfig[target, OBS, ACT, ROLLOUT, MB, EPOCHS, N_ENVS, HIDDEN].DEF_CRITIC_LR,
    gamma: Scalar[DT] = PPOConfig[target, OBS, ACT, ROLLOUT, MB, EPOCHS, N_ENVS, HIDDEN].DEF_GAMMA,
    gae_lambda: Scalar[DT] = PPOConfig[target, OBS, ACT, ROLLOUT, MB, EPOCHS, N_ENVS, HIDDEN].DEF_GAE_LAMBDA,
    clip_eps: Scalar[DT] = PPOConfig[target, OBS, ACT, ROLLOUT, MB, EPOCHS, N_ENVS, HIDDEN].DEF_CLIP_EPS,
    entropy_coef: Scalar[DT] = PPOConfig[target, OBS, ACT, ROLLOUT, MB, EPOCHS, N_ENVS, HIDDEN].DEF_ENTROPY_COEF,
    action_scale: Scalar[DT] = PPOConfig[target, OBS, ACT, ROLLOUT, MB, EPOCHS, N_ENVS, HIDDEN].DEF_ACTION_SCALE,
    log_std_init: Scalar[DT] = PPOConfig[target, OBS, ACT, ROLLOUT, MB, EPOCHS, N_ENVS, HIDDEN].DEF_LOG_STD_INIT,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1600.0),
    max_grad_norm: Scalar[DT] = PPOConfig[target, OBS, ACT, ROLLOUT, MB, EPOCHS, N_ENVS, HIDDEN].DEF_MAX_GRAD_NORM,
) raises -> PPOAgent[
    target,
    PPOActorNet[OBS, ACT, HIDDEN],
    PPOCriticNet[OBS, HIDDEN],
    OBS, ACT, ROLLOUT, MB, EPOCHS, N_ENVS,
]:
    """Clipped-surrogate PPO with the canonical fused-`LinearTanh` actor +
    critic. `target` selects cpu/gpu; the structural ints (`ROLLOUT`,
    minibatch `MB`, `EPOCHS`, `N_ENVS`) define the rollout shape; all
    scalars default to the tuned config value but stay overridable."""
    return agent_from_config[
        PPOConfig[target, OBS, ACT, ROLLOUT, MB, EPOCHS, N_ENVS, HIDDEN]
    ](
        ctx=ctx,
        actor_lr=actor_lr, critic_lr=critic_lr,
        gamma=gamma, gae_lambda=gae_lambda,
        clip_eps=clip_eps, entropy_coef=entropy_coef,
        action_scale=action_scale, log_std_init=log_std_init,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
        max_grad_norm=max_grad_norm,
    )
