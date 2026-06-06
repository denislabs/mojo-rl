"""DDPG named preset — config descriptor + factory (Design F).

Additive sugar over the primitive
`DDPGAgent[train_target, SAMPLE, ACTOR, CRITIC]`, mirroring `sac/config.mojo`
exactly (see it for the full rationale). The primitive stays the source of
truth for arbitrary actor/critic/replay combinations; this module names the
canonical DDPG setup and bundles its tuned defaults + a default
fused-`LinearReLU` deterministic actor / critic.

  1. `DDPGConfigT` — a trait bundling the FULL compile-time identity of the
     algorithm: the deployment `TARGET`, the replay `SAMPLE` block, the actor
     + critic nets, plus tuned scalar defaults (`DEF_*`).

  2. `DDPGConfig` — a zero-field conformer struct parametrized by `target`.
     ONE config covers both CPU and GPU because the replay block is
     target-generic (`ReplaySampleStep[AnyReplay[target, …]]`).

  3. `agent_from_config` + the capitalized preset `DDPG`:

         var agent = DDPG["cpu", OBS, ACT, BATCH, CAP]()
         var agent = DDPG["gpu", OBS, ACT, BATCH, CAP](ctx=ctx)

The default actor is a fused-`LinearReLU` trunk with a `Tanh` head (bounded
deterministic action); the critic is a fused-`LinearReLU` Q-net. Same
parameter layout as the unfused `Linear` + `ReLU` pair, so checkpoints are
target- and fusion-portable.

NOTE: this replaces the former runtime `DDPGConfig` hyperparameter bag (a
`Saveable` struct that was never referenced — trainer/agent take raw kwargs).
`DDPGConfig` now means the compile-time descriptor, matching `SACConfig` and
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

from .agent import DDPGAgent


# ──────────────────────────────────────────────────────────────────────
# Net presets — target-agnostic, parametrized comptime aliases.
# Hidden layers are FUSED LinearReLU; the actor ends in Tanh (bounded
# deterministic action), the critic's scalar head stays a plain Linear.
# ──────────────────────────────────────────────────────────────────────


comptime DDPGActorNet[OBS: Int, ACT: Int, HIDDEN: Int] = Sequential[
    LinearReLU[OBS, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, ACT],
    Tanh[ACT],
]
"""2-layer fused-MLP trunk + Tanh head — DDPG's deterministic actor."""


comptime DDPGCriticNet[OBS: Int, ACT: Int, HIDDEN: Int] = Sequential[
    LinearReLU[OBS + ACT, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, 1],
]
"""2-layer fused-MLP single critic Q(s, a)."""


# ──────────────────────────────────────────────────────────────────────
# Config trait — full compile-time identity + tuned scalar defaults.
# ──────────────────────────────────────────────────────────────────────


trait DDPGConfigT(Copyable, Movable, ImplicitlyDestructible):
    """Compile-time descriptor of a DDPG-family algorithm. Conformers are
    zero-field comptime tags — never instantiated at runtime; only their
    comptime members are read."""

    comptime TARGET: StaticString
    comptime SAMPLE: SampleBlock
    comptime ACTOR: Module
    comptime CRITIC: Module

    # Tuned scalar defaults (read into __init__ kwarg defaults).
    comptime DEF_ACTOR_LR: Scalar[DT]
    comptime DEF_CRITIC_LR: Scalar[DT]
    comptime DEF_GAMMA: Scalar[DT]
    comptime DEF_TAU: Scalar[DT]
    comptime DEF_ACTION_SCALE: Scalar[DT]
    comptime DEF_NOISE_SCALE: Scalar[DT]
    comptime DEF_LEARNING_STARTS: Int
    comptime DEF_MAX_GRAD_NORM: Scalar[DT]


# ──────────────────────────────────────────────────────────────────────
# Conformer — one struct, parametrized by `target`.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct DDPGConfig[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 256,
](DDPGConfigT):
    """Deep Deterministic Policy Gradient (Lillicrap et al. 2016) — single
    critic, deterministic actor with Gaussian exploration noise, uniform
    replay (1-step). One config covers cpu + gpu via the target-generic
    replay block."""

    comptime TARGET = Self.target
    comptime SAMPLE = ReplaySampleStep[
        AnyReplay[Self.target, Self.OBS, Self.ACT, Self.CAP], Self.BATCH,
    ]
    comptime ACTOR = DDPGActorNet[Self.OBS, Self.ACT, Self.HIDDEN]
    comptime CRITIC = DDPGCriticNet[Self.OBS, Self.ACT, Self.HIDDEN]

    comptime DEF_ACTOR_LR = Scalar[DT](1e-4)
    comptime DEF_CRITIC_LR = Scalar[DT](1e-3)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_ACTION_SCALE = Scalar[DT](1.0)
    comptime DEF_NOISE_SCALE = Scalar[DT](0.1)
    comptime DEF_LEARNING_STARTS = 1_000
    comptime DEF_MAX_GRAD_NORM = Scalar[DT](0.0)


# ──────────────────────────────────────────────────────────────────────
# Generic factory — any DDPGConfigT → primitive agent, defaults applied.
# ──────────────────────────────────────────────────────────────────────


def agent_from_config[
    CONFIG: DDPGConfigT,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = CONFIG.DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = CONFIG.DEF_CRITIC_LR,
    gamma: Scalar[DT] = CONFIG.DEF_GAMMA,
    tau: Scalar[DT] = CONFIG.DEF_TAU,
    action_scale: Scalar[DT] = CONFIG.DEF_ACTION_SCALE,
    noise_scale: Scalar[DT] = CONFIG.DEF_NOISE_SCALE,
    learning_starts: Int = CONFIG.DEF_LEARNING_STARTS,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
    max_grad_norm: Scalar[DT] = CONFIG.DEF_MAX_GRAD_NORM,
    use_bf16: Bool = False,
) raises -> DDPGAgent[
    CONFIG.TARGET,
    CONFIG.SAMPLE,
    CONFIG.ACTOR,
    CONFIG.CRITIC,
]:
    """Build the primitive `DDPGAgent` from any `DDPGConfigT`. Every scalar
    defaults to the config's tuned value but stays overridable. The
    deployment target and replay block are read off the config, so this one
    function serves cpu and gpu."""
    return DDPGAgent[
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
        noise_scale=noise_scale,
        learning_starts=learning_starts,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
        max_grad_norm=max_grad_norm,
        use_bf16=use_bf16,
    )


# ──────────────────────────────────────────────────────────────────────
# Capitalized preset — single function, `target` as a parameter.
# Reads like a constructor. Full tuning surface, defaults from the config.
# ──────────────────────────────────────────────────────────────────────


def DDPG[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 256,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = DDPGConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = DDPGConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_CRITIC_LR,
    gamma: Scalar[DT] = DDPGConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_GAMMA,
    tau: Scalar[DT] = DDPGConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TAU,
    action_scale: Scalar[DT] = DDPGConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_ACTION_SCALE,
    noise_scale: Scalar[DT] = DDPGConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_NOISE_SCALE,
    learning_starts: Int = DDPGConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_LEARNING_STARTS,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
    max_grad_norm: Scalar[DT] = DDPGConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_MAX_GRAD_NORM,
    use_bf16: Bool = False,
) raises -> DDPGAgent[
    target,
    ReplaySampleStep[AnyReplay[target, OBS, ACT, CAP], BATCH],
    DDPGActorNet[OBS, ACT, HIDDEN],
    DDPGCriticNet[OBS, ACT, HIDDEN],
]:
    """Deep Deterministic Policy Gradient with the canonical fused-`LinearReLU`
    deterministic actor + single critic, uniform replay, Gaussian exploration
    noise. `target` selects cpu/gpu; all scalars default to the tuned config
    value but stay overridable."""
    return agent_from_config[
        DDPGConfig[target, OBS, ACT, BATCH, CAP, HIDDEN]
    ](
        ctx=ctx,
        actor_lr=actor_lr, critic_lr=critic_lr,
        gamma=gamma, tau=tau,
        action_scale=action_scale,
        noise_scale=noise_scale,
        learning_starts=learning_starts,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
        max_grad_norm=max_grad_norm,
        use_bf16=use_bf16,
    )
