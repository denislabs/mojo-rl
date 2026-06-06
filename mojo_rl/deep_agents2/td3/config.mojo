"""TD3 named preset — config descriptor + factory (Design F).

Additive sugar over the primitive
`TD3Agent[train_target, SAMPLE, ACTOR, CRITIC]`, mirroring `sac/config.mojo`
(see it for the full rationale). TD3 = DDPG + twin critics (min-of-2 target),
target-policy smoothing, and a delayed actor update. The `CRITIC` net is a
single Q-net spec; the trainer instantiates the twin pair from it.

  1. `TD3ConfigT` — trait bundling the FULL compile-time identity: the
     deployment `TARGET`, the replay `SAMPLE` block, actor + critic nets,
     plus tuned scalar/int defaults (`DEF_*`).

  2. `TD3Config` — a zero-field conformer struct parametrized by `target`.
     ONE config covers cpu + gpu via the target-generic replay block.

  3. `agent_from_config` + the capitalized preset `TD3`:

         var agent = TD3["cpu", OBS, ACT, BATCH, CAP]()
         var agent = TD3["gpu", OBS, ACT, BATCH, CAP](ctx=ctx)

NOTE: this replaces the former runtime `TD3Config` hyperparameter bag (a
`Saveable` struct that was never referenced — trainer/agent take raw kwargs).
`TD3Config` now means the compile-time descriptor, matching `SACConfig` and
every other family.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.linear_relu import LinearReLU
from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.nn2.combinators.sequential import Sequential

from ..training.blocks import SampleBlock, ReplaySampleStep
from ..data.any_replay import AnyReplay

from .agent import TD3Agent


# ──────────────────────────────────────────────────────────────────────
# Net presets — target-agnostic. Same shapes as DDPG: fused-LinearReLU
# trunk + Tanh actor head; fused-LinearReLU critic with a scalar Linear
# head (the trainer builds the twin pair from this one CRITIC spec).
# ──────────────────────────────────────────────────────────────────────


comptime TD3ActorNet[OBS: Int, ACT: Int, HIDDEN: Int] = Sequential[
    LinearReLU[OBS, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, ACT],
    Tanh[ACT],
]
"""2-layer fused-MLP trunk + Tanh head — TD3's deterministic actor."""


comptime TD3CriticNet[OBS: Int, ACT: Int, HIDDEN: Int] = Sequential[
    LinearReLU[OBS + ACT, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, 1],
]
"""2-layer fused-MLP critic Q(s, a) — instantiated as a twin pair."""


# ──────────────────────────────────────────────────────────────────────
# Config trait — full compile-time identity + tuned defaults.
# ──────────────────────────────────────────────────────────────────────


trait TD3ConfigT(Copyable, Movable, ImplicitlyDestructible):
    """Compile-time descriptor of a TD3-family algorithm. Conformers are
    zero-field comptime tags — never instantiated at runtime; only their
    comptime members are read."""

    comptime TARGET: StaticString
    comptime SAMPLE: SampleBlock
    comptime ACTOR: Module
    comptime CRITIC: Module

    # Tuned scalar/int defaults (read into __init__ kwarg defaults).
    comptime DEF_ACTOR_LR: Scalar[DT]
    comptime DEF_CRITIC_LR: Scalar[DT]
    comptime DEF_GAMMA: Scalar[DT]
    comptime DEF_TAU: Scalar[DT]
    comptime DEF_ACTION_SCALE: Scalar[DT]
    comptime DEF_EXPLORATION_NOISE: Scalar[DT]
    comptime DEF_TARGET_POLICY_NOISE: Scalar[DT]
    comptime DEF_TARGET_NOISE_CLIP: Scalar[DT]
    comptime DEF_POLICY_DELAY: Int
    comptime DEF_LEARNING_STARTS: Int
    comptime DEF_MAX_GRAD_NORM: Scalar[DT]


# ──────────────────────────────────────────────────────────────────────
# Conformer — one struct, parametrized by `target`.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct TD3Config[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 256,
](TD3ConfigT):
    """Twin Delayed DDPG (Fujimoto et al. 2018) — twin critics (min-of-2),
    target-policy smoothing, delayed actor update, uniform replay (1-step).
    One config covers cpu + gpu via the target-generic replay block."""

    comptime TARGET = Self.target
    comptime SAMPLE = ReplaySampleStep[
        AnyReplay[Self.target, Self.OBS, Self.ACT, Self.CAP], Self.BATCH,
    ]
    comptime ACTOR = TD3ActorNet[Self.OBS, Self.ACT, Self.HIDDEN]
    comptime CRITIC = TD3CriticNet[Self.OBS, Self.ACT, Self.HIDDEN]

    comptime DEF_ACTOR_LR = Scalar[DT](3e-4)
    comptime DEF_CRITIC_LR = Scalar[DT](3e-4)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_ACTION_SCALE = Scalar[DT](1.0)
    comptime DEF_EXPLORATION_NOISE = Scalar[DT](0.1)
    comptime DEF_TARGET_POLICY_NOISE = Scalar[DT](0.2)
    comptime DEF_TARGET_NOISE_CLIP = Scalar[DT](0.5)
    comptime DEF_POLICY_DELAY = 2
    comptime DEF_LEARNING_STARTS = 1_000
    comptime DEF_MAX_GRAD_NORM = Scalar[DT](0.0)


# ──────────────────────────────────────────────────────────────────────
# Generic factory — any TD3ConfigT → primitive agent, defaults applied.
# ──────────────────────────────────────────────────────────────────────


def agent_from_config[
    CONFIG: TD3ConfigT,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = CONFIG.DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = CONFIG.DEF_CRITIC_LR,
    gamma: Scalar[DT] = CONFIG.DEF_GAMMA,
    tau: Scalar[DT] = CONFIG.DEF_TAU,
    action_scale: Scalar[DT] = CONFIG.DEF_ACTION_SCALE,
    exploration_noise: Scalar[DT] = CONFIG.DEF_EXPLORATION_NOISE,
    target_policy_noise: Scalar[DT] = CONFIG.DEF_TARGET_POLICY_NOISE,
    target_noise_clip: Scalar[DT] = CONFIG.DEF_TARGET_NOISE_CLIP,
    policy_delay: Int = CONFIG.DEF_POLICY_DELAY,
    learning_starts: Int = CONFIG.DEF_LEARNING_STARTS,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
    max_grad_norm: Scalar[DT] = CONFIG.DEF_MAX_GRAD_NORM,
    use_bf16: Bool = False,
) raises -> TD3Agent[
    CONFIG.TARGET,
    CONFIG.SAMPLE,
    CONFIG.ACTOR,
    CONFIG.CRITIC,
]:
    """Build the primitive `TD3Agent` from any `TD3ConfigT`. Every scalar
    defaults to the config's tuned value but stays overridable. The
    deployment target and replay block are read off the config, so this one
    function serves cpu and gpu."""
    return TD3Agent[
        CONFIG.TARGET,
        CONFIG.SAMPLE,
        CONFIG.ACTOR,
        CONFIG.CRITIC,
    ](
        ctx=ctx,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        tau=tau,
        action_scale=action_scale,
        exploration_noise=exploration_noise,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip,
        policy_delay=policy_delay,
        learning_starts=learning_starts,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
        max_grad_norm=max_grad_norm,
        use_bf16=use_bf16,
    )


# ──────────────────────────────────────────────────────────────────────
# Capitalized preset — single function, `target` as a parameter.
# ──────────────────────────────────────────────────────────────────────


def TD3[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 256,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = TD3Config[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = TD3Config[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_CRITIC_LR,
    gamma: Scalar[DT] = TD3Config[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_GAMMA,
    tau: Scalar[DT] = TD3Config[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TAU,
    action_scale: Scalar[DT] = TD3Config[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_ACTION_SCALE,
    exploration_noise: Scalar[DT] = TD3Config[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_EXPLORATION_NOISE,
    target_policy_noise: Scalar[DT] = TD3Config[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TARGET_POLICY_NOISE,
    target_noise_clip: Scalar[DT] = TD3Config[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TARGET_NOISE_CLIP,
    policy_delay: Int = TD3Config[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_POLICY_DELAY,
    learning_starts: Int = TD3Config[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_LEARNING_STARTS,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
    max_grad_norm: Scalar[DT] = TD3Config[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_MAX_GRAD_NORM,
    use_bf16: Bool = False,
) raises -> TD3Agent[
    target,
    ReplaySampleStep[AnyReplay[target, OBS, ACT, CAP], BATCH],
    TD3ActorNet[OBS, ACT, HIDDEN],
    TD3CriticNet[OBS, ACT, HIDDEN],
]:
    """Twin Delayed DDPG with the canonical fused-`LinearReLU` deterministic
    actor + twin critics, uniform replay, target-policy smoothing, delayed
    actor. `target` selects cpu/gpu; all scalars default to the tuned config
    value but stay overridable."""
    return agent_from_config[
        TD3Config[target, OBS, ACT, BATCH, CAP, HIDDEN]
    ](
        ctx=ctx,
        actor_lr=actor_lr, critic_lr=critic_lr,
        gamma=gamma, tau=tau,
        action_scale=action_scale,
        exploration_noise=exploration_noise,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip,
        policy_delay=policy_delay,
        learning_starts=learning_starts,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
        max_grad_norm=max_grad_norm,
        use_bf16=use_bf16,
    )
