"""PPO-discrete named preset — config descriptor + factory (Design F).

Additive sugar over the primitive
`PPODiscreteAgent[train_target, ACTOR, CRITIC, OBS_DIM, N_ACTIONS,
                  ROLLOUT_LEN, MINIBATCH, N_EPOCHS, N_ENVS]`. Same Design-F
shape as `ppo/config.mojo`; the only differences from continuous PPO are
the categorical policy head (`Linear[HIDDEN, N_ACTIONS]` logits instead of
a `GaussianHead`) and the discrete-tuned defaults (`entropy_coef=0.01`).

  1. `PPODiscreteConfigT` — trait bundling the compile-time identity.
  2. `PPODiscreteConfig` — conformer parametrized by `target` + ints.
  3. `agent_from_config` + the capitalized preset `PPODiscrete`:

         var agent = PPODiscrete["cpu", OBS, N_ACT, ROLLOUT, MB, EPOCHS]()

Hidden layers use FUSED `LinearTanh`; the logit head + value head are
plain `Linear`.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.linear_tanh import LinearTanh
from mojo_rl.nn2.combinators.sequential import Sequential

from .agent import PPODiscreteAgent


# ──────────────────────────────────────────────────────────────────────
# Net presets — target-agnostic, parametrized comptime aliases.
# ──────────────────────────────────────────────────────────────────────


comptime PPODiscreteActorNet[
    OBS: Int, N_ACTIONS: Int, HIDDEN: Int
] = Sequential[
    LinearTanh[OBS, HIDDEN],
    LinearTanh[HIDDEN, HIDDEN],
    Linear[HIDDEN, N_ACTIONS],
]
"""2-layer fused-tanh trunk + categorical logit head (softmax +
sampling applied inside the trainer)."""


comptime PPODiscreteCriticNet[OBS: Int, HIDDEN: Int] = Sequential[
    LinearTanh[OBS, HIDDEN],
    LinearTanh[HIDDEN, HIDDEN],
    Linear[HIDDEN, 1],
]
"""2-layer fused-tanh value net V(s)."""


# ──────────────────────────────────────────────────────────────────────
# Config trait.
# ──────────────────────────────────────────────────────────────────────


trait PPODiscreteConfigT(Copyable, Movable, ImplicitlyDestructible):
    """Compile-time descriptor of a discrete-PPO algorithm. Conformers are
    zero-field comptime tags."""

    comptime TARGET: StaticString
    comptime ACTOR: Module
    comptime CRITIC: Module
    comptime OBS_DIM: Int
    comptime N_ACTIONS: Int
    comptime ROLLOUT_LEN: Int
    comptime MINIBATCH: Int
    comptime N_EPOCHS: Int
    comptime N_ENVS: Int

    comptime DEF_ACTOR_LR: Scalar[DT]
    comptime DEF_CRITIC_LR: Scalar[DT]
    comptime DEF_GAMMA: Scalar[DT]
    comptime DEF_GAE_LAMBDA: Scalar[DT]
    comptime DEF_CLIP_EPS: Scalar[DT]
    comptime DEF_ENTROPY_COEF: Scalar[DT]
    comptime DEF_MAX_GRAD_NORM: Scalar[DT]


# ──────────────────────────────────────────────────────────────────────
# Conformer.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct PPODiscreteConfig[
    target: StaticString,
    OBS: Int, N_ACT: Int,
    ROLLOUT: Int, MB: Int, EPOCHS: Int,
    N_ENVS_: Int = 1, HIDDEN: Int = 64,
](PPODiscreteConfigT):
    """Clipped-surrogate discrete PPO — softmax categorical policy, GAE
    advantages, multi-epoch minibatch SGD."""

    comptime TARGET = Self.target
    comptime ACTOR = PPODiscreteActorNet[Self.OBS, Self.N_ACT, Self.HIDDEN]
    comptime CRITIC = PPODiscreteCriticNet[Self.OBS, Self.HIDDEN]
    comptime OBS_DIM = Self.OBS
    comptime N_ACTIONS = Self.N_ACT
    comptime ROLLOUT_LEN = Self.ROLLOUT
    comptime MINIBATCH = Self.MB
    comptime N_EPOCHS = Self.EPOCHS
    comptime N_ENVS = Self.N_ENVS_

    comptime DEF_ACTOR_LR = Scalar[DT](3e-4)
    comptime DEF_CRITIC_LR = Scalar[DT](1e-3)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_GAE_LAMBDA = Scalar[DT](0.95)
    comptime DEF_CLIP_EPS = Scalar[DT](0.2)
    comptime DEF_ENTROPY_COEF = Scalar[DT](0.01)
    comptime DEF_MAX_GRAD_NORM = Scalar[DT](0.0)


# ──────────────────────────────────────────────────────────────────────
# Generic factory.
# ──────────────────────────────────────────────────────────────────────


def agent_from_config[
    CONFIG: PPODiscreteConfigT,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = CONFIG.DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = CONFIG.DEF_CRITIC_LR,
    gamma: Scalar[DT] = CONFIG.DEF_GAMMA,
    gae_lambda: Scalar[DT] = CONFIG.DEF_GAE_LAMBDA,
    clip_eps: Scalar[DT] = CONFIG.DEF_CLIP_EPS,
    entropy_coef: Scalar[DT] = CONFIG.DEF_ENTROPY_COEF,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](0.0),
    max_grad_norm: Scalar[DT] = CONFIG.DEF_MAX_GRAD_NORM,
) raises -> PPODiscreteAgent[
    CONFIG.TARGET,
    CONFIG.ACTOR,
    CONFIG.CRITIC,
    CONFIG.OBS_DIM,
    CONFIG.N_ACTIONS,
    CONFIG.ROLLOUT_LEN,
    CONFIG.MINIBATCH,
    CONFIG.N_EPOCHS,
    CONFIG.N_ENVS,
]:
    """Build the primitive `PPODiscreteAgent` from any
    `PPODiscreteConfigT`. Every scalar defaults to the config's tuned
    value but stays overridable."""
    return PPODiscreteAgent[
        CONFIG.TARGET,
        CONFIG.ACTOR,
        CONFIG.CRITIC,
        CONFIG.OBS_DIM,
        CONFIG.N_ACTIONS,
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
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
        max_grad_norm=max_grad_norm,
    )


# ──────────────────────────────────────────────────────────────────────
# Capitalized preset.
# ──────────────────────────────────────────────────────────────────────


def PPODiscrete[
    target: StaticString,
    OBS: Int, N_ACT: Int,
    ROLLOUT: Int, MB: Int, EPOCHS: Int,
    N_ENVS: Int = 1, HIDDEN: Int = 64,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = PPODiscreteConfig[target, OBS, N_ACT, ROLLOUT, MB, EPOCHS, N_ENVS, HIDDEN].DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = PPODiscreteConfig[target, OBS, N_ACT, ROLLOUT, MB, EPOCHS, N_ENVS, HIDDEN].DEF_CRITIC_LR,
    gamma: Scalar[DT] = PPODiscreteConfig[target, OBS, N_ACT, ROLLOUT, MB, EPOCHS, N_ENVS, HIDDEN].DEF_GAMMA,
    gae_lambda: Scalar[DT] = PPODiscreteConfig[target, OBS, N_ACT, ROLLOUT, MB, EPOCHS, N_ENVS, HIDDEN].DEF_GAE_LAMBDA,
    clip_eps: Scalar[DT] = PPODiscreteConfig[target, OBS, N_ACT, ROLLOUT, MB, EPOCHS, N_ENVS, HIDDEN].DEF_CLIP_EPS,
    entropy_coef: Scalar[DT] = PPODiscreteConfig[target, OBS, N_ACT, ROLLOUT, MB, EPOCHS, N_ENVS, HIDDEN].DEF_ENTROPY_COEF,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](0.0),
    max_grad_norm: Scalar[DT] = PPODiscreteConfig[target, OBS, N_ACT, ROLLOUT, MB, EPOCHS, N_ENVS, HIDDEN].DEF_MAX_GRAD_NORM,
) raises -> PPODiscreteAgent[
    target,
    PPODiscreteActorNet[OBS, N_ACT, HIDDEN],
    PPODiscreteCriticNet[OBS, HIDDEN],
    OBS, N_ACT, ROLLOUT, MB, EPOCHS, N_ENVS,
]:
    """Discrete clipped-surrogate PPO with the canonical fused-`LinearTanh`
    actor + critic. `target` selects cpu/gpu; the structural ints define
    the rollout shape; all scalars default to the tuned config value."""
    return agent_from_config[
        PPODiscreteConfig[target, OBS, N_ACT, ROLLOUT, MB, EPOCHS, N_ENVS, HIDDEN]
    ](
        ctx=ctx,
        actor_lr=actor_lr, critic_lr=critic_lr,
        gamma=gamma, gae_lambda=gae_lambda,
        clip_eps=clip_eps, entropy_coef=entropy_coef,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
        max_grad_norm=max_grad_norm,
    )
