"""A2C named presets — config descriptors + factories (Design F).

A2C is the degenerate single-epoch / full-batch PPO (`N_EPOCHS=1`,
`MINIBATCH=ROLLOUT_LEN·N_ENVS`), so this module reuses the fused PPO net
presets and bundles A2C's tuned defaults (higher LR `7e-4`, no clip, a
default `max_grad_norm=0.5`).

  * `A2C` / `A2CConfig` — continuous (Gaussian policy), reuses
    `PPOActorNet` / `PPOCriticNet`.
  * `A2CDiscrete` / `A2CDiscreteConfig` — categorical policy, reuses
    `PPODiscreteActorNet` / `PPODiscreteCriticNet`.

        var agent = A2C["cpu", OBS, ACT, ROLLOUT]()
        var agent = A2CDiscrete["cpu", OBS, N_ACT, ROLLOUT]()
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module

from ..ppo.config import PPOActorNet, PPOCriticNet
from ..ppo_discrete.config import (
    PPODiscreteActorNet,
    PPODiscreteCriticNet,
)

from .agent import A2CAgent, A2CDiscreteAgent


# ──────────────────────────────────────────────────────────────────────
# Continuous A2C — config trait + conformer.
# ──────────────────────────────────────────────────────────────────────


trait A2CConfigT(Copyable, Movable, ImplicitlyDestructible):
    """Compile-time descriptor of a continuous A2C algorithm."""

    comptime TARGET: StaticString
    comptime ACTOR: Module
    comptime CRITIC: Module
    comptime OBS_DIM: Int
    comptime ACT_DIM: Int
    comptime ROLLOUT_LEN: Int
    comptime N_ENVS: Int

    comptime DEF_ACTOR_LR: Scalar[DT]
    comptime DEF_CRITIC_LR: Scalar[DT]
    comptime DEF_GAMMA: Scalar[DT]
    comptime DEF_GAE_LAMBDA: Scalar[DT]
    comptime DEF_ENTROPY_COEF: Scalar[DT]
    comptime DEF_ACTION_SCALE: Scalar[DT]
    comptime DEF_LOG_STD_INIT: Scalar[DT]
    comptime DEF_MAX_GRAD_NORM: Scalar[DT]


@fieldwise_init
struct A2CConfig[
    target: StaticString,
    OBS: Int, ACT: Int, ROLLOUT: Int,
    N_ENVS_: Int = 1, HIDDEN: Int = 64,
](A2CConfigT):
    """Advantage Actor-Critic (continuous) — single-epoch full-batch PPO
    with a Gaussian policy."""

    comptime TARGET = Self.target
    comptime ACTOR = PPOActorNet[Self.OBS, Self.ACT, Self.HIDDEN]
    comptime CRITIC = PPOCriticNet[Self.OBS, Self.HIDDEN]
    comptime OBS_DIM = Self.OBS
    comptime ACT_DIM = Self.ACT
    comptime ROLLOUT_LEN = Self.ROLLOUT
    comptime N_ENVS = Self.N_ENVS_

    comptime DEF_ACTOR_LR = Scalar[DT](7e-4)
    comptime DEF_CRITIC_LR = Scalar[DT](7e-4)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_GAE_LAMBDA = Scalar[DT](0.95)
    comptime DEF_ENTROPY_COEF = Scalar[DT](0.0)
    comptime DEF_ACTION_SCALE = Scalar[DT](1.0)
    comptime DEF_LOG_STD_INIT = Scalar[DT](-0.5)
    comptime DEF_MAX_GRAD_NORM = Scalar[DT](0.5)


# ──────────────────────────────────────────────────────────────────────
# Discrete A2C — config trait + conformer.
# ──────────────────────────────────────────────────────────────────────


trait A2CDiscreteConfigT(Copyable, Movable, ImplicitlyDestructible):
    """Compile-time descriptor of a discrete A2C algorithm."""

    comptime TARGET: StaticString
    comptime ACTOR: Module
    comptime CRITIC: Module
    comptime OBS_DIM: Int
    comptime N_ACTIONS: Int
    comptime ROLLOUT_LEN: Int

    comptime DEF_ACTOR_LR: Scalar[DT]
    comptime DEF_CRITIC_LR: Scalar[DT]
    comptime DEF_GAMMA: Scalar[DT]
    comptime DEF_GAE_LAMBDA: Scalar[DT]
    comptime DEF_ENTROPY_COEF: Scalar[DT]
    comptime DEF_MAX_GRAD_NORM: Scalar[DT]


@fieldwise_init
struct A2CDiscreteConfig[
    target: StaticString,
    OBS: Int, N_ACT: Int, ROLLOUT: Int,
    HIDDEN: Int = 64,
](A2CDiscreteConfigT):
    """Advantage Actor-Critic (discrete) — single-epoch full-batch PPO
    with a categorical policy."""

    comptime TARGET = Self.target
    comptime ACTOR = PPODiscreteActorNet[Self.OBS, Self.N_ACT, Self.HIDDEN]
    comptime CRITIC = PPODiscreteCriticNet[Self.OBS, Self.HIDDEN]
    comptime OBS_DIM = Self.OBS
    comptime N_ACTIONS = Self.N_ACT
    comptime ROLLOUT_LEN = Self.ROLLOUT

    comptime DEF_ACTOR_LR = Scalar[DT](7e-4)
    comptime DEF_CRITIC_LR = Scalar[DT](7e-4)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_GAE_LAMBDA = Scalar[DT](0.95)
    comptime DEF_ENTROPY_COEF = Scalar[DT](0.01)
    comptime DEF_MAX_GRAD_NORM = Scalar[DT](0.5)


# ──────────────────────────────────────────────────────────────────────
# Generic factories.
# ──────────────────────────────────────────────────────────────────────


def agent_from_config[
    CONFIG: A2CConfigT,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = CONFIG.DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = CONFIG.DEF_CRITIC_LR,
    gamma: Scalar[DT] = CONFIG.DEF_GAMMA,
    gae_lambda: Scalar[DT] = CONFIG.DEF_GAE_LAMBDA,
    entropy_coef: Scalar[DT] = CONFIG.DEF_ENTROPY_COEF,
    action_scale: Scalar[DT] = CONFIG.DEF_ACTION_SCALE,
    log_std_init: Scalar[DT] = CONFIG.DEF_LOG_STD_INIT,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1600.0),
    max_grad_norm: Scalar[DT] = CONFIG.DEF_MAX_GRAD_NORM,
) raises -> A2CAgent[
    CONFIG.TARGET,
    CONFIG.ACTOR,
    CONFIG.CRITIC,
    CONFIG.OBS_DIM,
    CONFIG.ACT_DIM,
    CONFIG.ROLLOUT_LEN,
    CONFIG.N_ENVS,
]:
    """Build the primitive continuous `A2CAgent` from any `A2CConfigT`."""
    return A2CAgent[
        CONFIG.TARGET,
        CONFIG.ACTOR,
        CONFIG.CRITIC,
        CONFIG.OBS_DIM,
        CONFIG.ACT_DIM,
        CONFIG.ROLLOUT_LEN,
        CONFIG.N_ENVS,
    ](
        ctx=ctx,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        entropy_coef=entropy_coef,
        action_scale=action_scale,
        log_std_init=log_std_init,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
        max_grad_norm=max_grad_norm,
    )


def agent_from_config_discrete[
    CONFIG: A2CDiscreteConfigT,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = CONFIG.DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = CONFIG.DEF_CRITIC_LR,
    gamma: Scalar[DT] = CONFIG.DEF_GAMMA,
    gae_lambda: Scalar[DT] = CONFIG.DEF_GAE_LAMBDA,
    entropy_coef: Scalar[DT] = CONFIG.DEF_ENTROPY_COEF,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](0.0),
    max_grad_norm: Scalar[DT] = CONFIG.DEF_MAX_GRAD_NORM,
) raises -> A2CDiscreteAgent[
    CONFIG.TARGET,
    CONFIG.ACTOR,
    CONFIG.CRITIC,
    CONFIG.OBS_DIM,
    CONFIG.N_ACTIONS,
    CONFIG.ROLLOUT_LEN,
]:
    """Build the primitive discrete `A2CDiscreteAgent` from any
    `A2CDiscreteConfigT`."""
    return A2CDiscreteAgent[
        CONFIG.TARGET,
        CONFIG.ACTOR,
        CONFIG.CRITIC,
        CONFIG.OBS_DIM,
        CONFIG.N_ACTIONS,
        CONFIG.ROLLOUT_LEN,
    ](
        ctx=ctx,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        entropy_coef=entropy_coef,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
        max_grad_norm=max_grad_norm,
    )


# ──────────────────────────────────────────────────────────────────────
# Capitalized presets.
# ──────────────────────────────────────────────────────────────────────


def A2C[
    target: StaticString,
    OBS: Int, ACT: Int, ROLLOUT: Int,
    N_ENVS: Int = 1, HIDDEN: Int = 64,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = A2CConfig[target, OBS, ACT, ROLLOUT, N_ENVS, HIDDEN].DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = A2CConfig[target, OBS, ACT, ROLLOUT, N_ENVS, HIDDEN].DEF_CRITIC_LR,
    gamma: Scalar[DT] = A2CConfig[target, OBS, ACT, ROLLOUT, N_ENVS, HIDDEN].DEF_GAMMA,
    gae_lambda: Scalar[DT] = A2CConfig[target, OBS, ACT, ROLLOUT, N_ENVS, HIDDEN].DEF_GAE_LAMBDA,
    entropy_coef: Scalar[DT] = A2CConfig[target, OBS, ACT, ROLLOUT, N_ENVS, HIDDEN].DEF_ENTROPY_COEF,
    action_scale: Scalar[DT] = A2CConfig[target, OBS, ACT, ROLLOUT, N_ENVS, HIDDEN].DEF_ACTION_SCALE,
    log_std_init: Scalar[DT] = A2CConfig[target, OBS, ACT, ROLLOUT, N_ENVS, HIDDEN].DEF_LOG_STD_INIT,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1600.0),
    max_grad_norm: Scalar[DT] = A2CConfig[target, OBS, ACT, ROLLOUT, N_ENVS, HIDDEN].DEF_MAX_GRAD_NORM,
) raises -> A2CAgent[
    target,
    PPOActorNet[OBS, ACT, HIDDEN],
    PPOCriticNet[OBS, HIDDEN],
    OBS, ACT, ROLLOUT, N_ENVS,
]:
    """Continuous A2C with the canonical fused-`LinearTanh` Gaussian actor
    + critic. Single-epoch full-batch PPO under the hood."""
    return agent_from_config[
        A2CConfig[target, OBS, ACT, ROLLOUT, N_ENVS, HIDDEN]
    ](
        ctx=ctx,
        actor_lr=actor_lr, critic_lr=critic_lr,
        gamma=gamma, gae_lambda=gae_lambda,
        entropy_coef=entropy_coef,
        action_scale=action_scale, log_std_init=log_std_init,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
        max_grad_norm=max_grad_norm,
    )


def A2CDiscrete[
    target: StaticString,
    OBS: Int, N_ACT: Int, ROLLOUT: Int,
    HIDDEN: Int = 64,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = A2CDiscreteConfig[target, OBS, N_ACT, ROLLOUT, HIDDEN].DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = A2CDiscreteConfig[target, OBS, N_ACT, ROLLOUT, HIDDEN].DEF_CRITIC_LR,
    gamma: Scalar[DT] = A2CDiscreteConfig[target, OBS, N_ACT, ROLLOUT, HIDDEN].DEF_GAMMA,
    gae_lambda: Scalar[DT] = A2CDiscreteConfig[target, OBS, N_ACT, ROLLOUT, HIDDEN].DEF_GAE_LAMBDA,
    entropy_coef: Scalar[DT] = A2CDiscreteConfig[target, OBS, N_ACT, ROLLOUT, HIDDEN].DEF_ENTROPY_COEF,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](0.0),
    max_grad_norm: Scalar[DT] = A2CDiscreteConfig[target, OBS, N_ACT, ROLLOUT, HIDDEN].DEF_MAX_GRAD_NORM,
) raises -> A2CDiscreteAgent[
    target,
    PPODiscreteActorNet[OBS, N_ACT, HIDDEN],
    PPODiscreteCriticNet[OBS, HIDDEN],
    OBS, N_ACT, ROLLOUT,
]:
    """Discrete A2C with the canonical fused-`LinearTanh` categorical actor
    + critic. Single-epoch full-batch discrete PPO under the hood."""
    return agent_from_config_discrete[
        A2CDiscreteConfig[target, OBS, N_ACT, ROLLOUT, HIDDEN]
    ](
        ctx=ctx,
        actor_lr=actor_lr, critic_lr=critic_lr,
        gamma=gamma, gae_lambda=gae_lambda,
        entropy_coef=entropy_coef,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
        max_grad_norm=max_grad_norm,
    )
