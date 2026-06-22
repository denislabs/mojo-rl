"""REDQ named presets — config descriptors + factories (Design F).

Additive sugar over the primitive
`REDQTrainer[train_target, SAMPLE, ACTOR, CRITIC,
              N, N_MIN, UTD, POLICY_DELAY, Q_MODE]`. The primitive stays
the source of truth for arbitrary combinations; this module names the
canonical algorithms and bundles their tuned defaults.

Same shape as `c51/config.mojo`:

  1. `REDQConfigT` — a trait bundling the FULL compile-time identity of an
     algorithm: the deployment `TARGET`, the replay `SAMPLE` block, the
     actor + critic nets, the ensemble knobs (`N`, `N_MIN`, `UTD`,
     `POLICY_DELAY`, `Q_MODE`), plus tuned scalar defaults (`DEF_*`).

  2. `REDQConfig` (paper-faithful) / `SmallREDQConfig` (SAC-shape) —
     zero-field conformer structs parametrized by `target`. One config
     covers both CPU and GPU because the replay block is target-generic
     (`ReplaySampleStep[AnyReplay[target, …]]`).

  3. `agent_from_config` + capitalized presets `REDQ` / `SmallREDQ`.
     Each preset is a SINGLE function taking `target` as a parameter:

         var agent = REDQ["cpu", OBS, ACT, BATCH, CAP](ctx=None)
         var agent = SmallREDQ["gpu", OBS, ACT, BATCH, CAP](ctx=ctx)

Defaults table:

                          REDQConfig (paper-faithful)   SmallREDQConfig (SAC-shape)
  N                       10                            2
  N_MIN                   2                             2
  UTD                     20                            1
  POLICY_DELAY            20                            1
  Q_MODE                  MIN                           MIN
  actor/critic LR         3e-4 / 3e-4                   3e-4 / 1e-3
  gamma                   0.99                          0.99
  tau                     0.005                         0.005
  init_alpha              0.2                           0.2
  target_entropy          −ACT                          −ACT
  learning_starts         5_000                         1_000

The `SmallREDQConfig` regime is what you want for quick experiments on
classic-control envs (Pendulum, HalfCheetah toy runs) — it's REDQ's
shape with the cheapest possible compute (no UTD inner loop, no policy
delay, N=2 critics). At those settings REDQ's TD target reduces to
SAC's `min(Q1, Q2)` but the actor loss still averages over the online
critics (the algorithmic difference).
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.module import Module
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.linear_relu import LinearReLU
from mojo_rl.nn.storage.combinators.sequential import Sequential

from ..primitives.stochastic_actor import StochasticActor
from ..training.blocks import SampleBlock, ReplaySampleStep
from ..data.any_replay import AnyReplay

from .agent import REDQAgent
from .kernels import REDQ_TARGET_MIN, REDQ_TARGET_AVE


# ──────────────────────────────────────────────────────────────────────
# Net presets — target-agnostic, parametrized comptime aliases.
# ──────────────────────────────────────────────────────────────────────


comptime REDQActor[OBS: Int, ACT: Int, HIDDEN: Int] = StochasticActor[
    OBS, ACT,
    LinearReLU[OBS, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
]
"""2-layer fused-MLP trunk + (μ, log σ) heads — same shape as SAC's actor."""


comptime REDQCritic[OBS: Int, ACT: Int, HIDDEN: Int] = Sequential[
    LinearReLU[OBS + ACT, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, 1],
]
"""2-layer MLP critic. Each of the N online + target nets in
`CriticEnsemble[CRITIC, N]` is one of these."""


# ──────────────────────────────────────────────────────────────────────
# Config trait — full compile-time identity + tuned scalar defaults.
# ──────────────────────────────────────────────────────────────────────


trait REDQConfigT(Copyable, Movable, ImplicitlyDeletable):
    """Compile-time descriptor of a REDQ-family algorithm. Conformers
    are zero-field comptime tags — never instantiated at runtime; only
    their comptime members are read."""

    comptime TARGET: StaticString
    comptime SAMPLE: SampleBlock
    comptime ACTOR: Module
    comptime CRITIC: Module
    comptime N: Int
    comptime N_MIN: Int
    comptime UTD: Int
    comptime POLICY_DELAY: Int
    comptime Q_MODE: Int

    # Tuned scalar defaults (read into __init__ kwarg defaults).
    comptime DEF_ACTOR_LR: Scalar[DT]
    comptime DEF_CRITIC_LR: Scalar[DT]
    comptime DEF_ALPHA_LR: Scalar[DT]
    comptime DEF_GAMMA: Scalar[DT]
    comptime DEF_TAU: Scalar[DT]
    comptime DEF_ACTION_SCALE: Scalar[DT]
    comptime DEF_INIT_ALPHA: Scalar[DT]
    comptime DEF_TARGET_ENTROPY: Scalar[DT]
    comptime DEF_LEARNING_STARTS: Int
    comptime DEF_MAX_GRAD_NORM: Scalar[DT]


# ──────────────────────────────────────────────────────────────────────
# Conformers — one struct per algorithm, parametrized by `target`.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct REDQConfig[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 256,
](REDQConfigT):
    """Paper-faithful REDQ (Chen et al. 2021): N=10 critics, M=2-subset
    MIN target, UTD=20 critic updates per env step, policy_delay=20
    (actor + α every 20 inner critic updates)."""

    comptime TARGET = Self.target
    comptime SAMPLE = ReplaySampleStep[
        AnyReplay[Self.target, Self.OBS, Self.ACT, Self.CAP], Self.BATCH,
    ]
    comptime ACTOR = REDQActor[Self.OBS, Self.ACT, Self.HIDDEN]
    comptime CRITIC = REDQCritic[Self.OBS, Self.ACT, Self.HIDDEN]
    comptime N = 10
    comptime N_MIN = 2
    comptime UTD = 20
    comptime POLICY_DELAY = 20
    comptime Q_MODE = REDQ_TARGET_MIN

    comptime DEF_ACTOR_LR = Scalar[DT](3e-4)
    comptime DEF_CRITIC_LR = Scalar[DT](3e-4)
    comptime DEF_ALPHA_LR = Scalar[DT](3e-4)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_ACTION_SCALE = Scalar[DT](1.0)
    comptime DEF_INIT_ALPHA = Scalar[DT](0.2)
    comptime DEF_TARGET_ENTROPY = Scalar[DT](-Float64(Self.ACT))
    comptime DEF_LEARNING_STARTS = 5_000
    comptime DEF_MAX_GRAD_NORM = Scalar[DT](0.0)


@fieldwise_init
struct SmallREDQConfig[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 64,
](REDQConfigT):
    """SAC-shape REDQ — N=2/M=2/UTD=1/POLICY_DELAY=1, the cheapest
    REDQ regime. At N=2 the TD target equals SAC's `min(Q1, Q2)`; the
    actor loss still averages the online critics (the algorithmic
    difference). Good fit for classic-control smokes."""

    comptime TARGET = Self.target
    comptime SAMPLE = ReplaySampleStep[
        AnyReplay[Self.target, Self.OBS, Self.ACT, Self.CAP], Self.BATCH,
    ]
    comptime ACTOR = REDQActor[Self.OBS, Self.ACT, Self.HIDDEN]
    comptime CRITIC = REDQCritic[Self.OBS, Self.ACT, Self.HIDDEN]
    comptime N = 2
    comptime N_MIN = 2
    comptime UTD = 1
    comptime POLICY_DELAY = 1
    comptime Q_MODE = REDQ_TARGET_MIN

    comptime DEF_ACTOR_LR = Scalar[DT](3e-4)
    comptime DEF_CRITIC_LR = Scalar[DT](1e-3)
    comptime DEF_ALPHA_LR = Scalar[DT](3e-4)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_ACTION_SCALE = Scalar[DT](1.0)
    comptime DEF_INIT_ALPHA = Scalar[DT](0.2)
    comptime DEF_TARGET_ENTROPY = Scalar[DT](-Float64(Self.ACT))
    comptime DEF_LEARNING_STARTS = 1_000
    comptime DEF_MAX_GRAD_NORM = Scalar[DT](0.0)


# ──────────────────────────────────────────────────────────────────────
# Generic factory — any REDQConfigT → primitive agent, defaults applied.
# ──────────────────────────────────────────────────────────────────────


def agent_from_config[
    CONFIG: REDQConfigT,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = CONFIG.DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = CONFIG.DEF_CRITIC_LR,
    alpha_lr: Scalar[DT] = CONFIG.DEF_ALPHA_LR,
    gamma: Scalar[DT] = CONFIG.DEF_GAMMA,
    tau: Scalar[DT] = CONFIG.DEF_TAU,
    action_scale: Scalar[DT] = CONFIG.DEF_ACTION_SCALE,
    init_alpha: Scalar[DT] = CONFIG.DEF_INIT_ALPHA,
    target_entropy: Scalar[DT] = CONFIG.DEF_TARGET_ENTROPY,
    learning_starts: Int = CONFIG.DEF_LEARNING_STARTS,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
    max_grad_norm: Scalar[DT] = CONFIG.DEF_MAX_GRAD_NORM,
) raises -> REDQAgent[
    CONFIG.TARGET,
    CONFIG.SAMPLE,
    CONFIG.ACTOR,
    CONFIG.CRITIC,
    CONFIG.N,
    CONFIG.N_MIN,
    CONFIG.UTD,
    CONFIG.POLICY_DELAY,
    CONFIG.Q_MODE,
]:
    """Build the primitive `REDQAgent` from any `REDQConfigT`. Every
    scalar defaults to the config's tuned value but stays overridable."""
    return REDQAgent[
        CONFIG.TARGET,
        CONFIG.SAMPLE,
        CONFIG.ACTOR,
        CONFIG.CRITIC,
        CONFIG.N,
        CONFIG.N_MIN,
        CONFIG.UTD,
        CONFIG.POLICY_DELAY,
        CONFIG.Q_MODE,
    ](
        ctx=ctx,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha_lr=alpha_lr,
        gamma=gamma,
        tau=tau,
        action_scale=action_scale,
        init_alpha=init_alpha,
        target_entropy=target_entropy,
        learning_starts=learning_starts,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
        max_grad_norm=max_grad_norm,
    )


# ──────────────────────────────────────────────────────────────────────
# Capitalized presets — read like constructors at the call site.
# ──────────────────────────────────────────────────────────────────────


def REDQ[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 256,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = REDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = REDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_CRITIC_LR,
    alpha_lr: Scalar[DT] = REDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_ALPHA_LR,
    gamma: Scalar[DT] = REDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_GAMMA,
    tau: Scalar[DT] = REDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TAU,
    action_scale: Scalar[DT] = REDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_ACTION_SCALE,
    init_alpha: Scalar[DT] = REDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_INIT_ALPHA,
    target_entropy: Scalar[DT] = REDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TARGET_ENTROPY,
    learning_starts: Int = REDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_LEARNING_STARTS,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
    max_grad_norm: Scalar[DT] = REDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_MAX_GRAD_NORM,
) raises -> REDQAgent[
    target,
    ReplaySampleStep[AnyReplay[target, OBS, ACT, CAP], BATCH],
    REDQActor[OBS, ACT, HIDDEN],
    REDQCritic[OBS, ACT, HIDDEN],
    10, 2, 20, 20, REDQ_TARGET_MIN,
]:
    """Paper-faithful REDQ — N=10 critics, M=2 subset MIN, UTD=20,
    policy_delay=20."""
    return agent_from_config[
        REDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN]
    ](
        ctx=ctx,
        actor_lr=actor_lr, critic_lr=critic_lr, alpha_lr=alpha_lr,
        gamma=gamma, tau=tau,
        action_scale=action_scale,
        init_alpha=init_alpha, target_entropy=target_entropy,
        learning_starts=learning_starts,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
        max_grad_norm=max_grad_norm,
    )


def SmallREDQ[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 64,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = SmallREDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = SmallREDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_CRITIC_LR,
    alpha_lr: Scalar[DT] = SmallREDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_ALPHA_LR,
    gamma: Scalar[DT] = SmallREDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_GAMMA,
    tau: Scalar[DT] = SmallREDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TAU,
    action_scale: Scalar[DT] = SmallREDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_ACTION_SCALE,
    init_alpha: Scalar[DT] = SmallREDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_INIT_ALPHA,
    target_entropy: Scalar[DT] = SmallREDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TARGET_ENTROPY,
    learning_starts: Int = SmallREDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_LEARNING_STARTS,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
    max_grad_norm: Scalar[DT] = SmallREDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_MAX_GRAD_NORM,
) raises -> REDQAgent[
    target,
    ReplaySampleStep[AnyReplay[target, OBS, ACT, CAP], BATCH],
    REDQActor[OBS, ACT, HIDDEN],
    REDQCritic[OBS, ACT, HIDDEN],
    2, 2, 1, 1, REDQ_TARGET_MIN,
]:
    """SAC-shape REDQ — N=2/M=2/UTD=1/POLICY_DELAY=1. Cheapest REDQ
    regime; algorithmic difference vs SAC is the averaged-not-min'd
    actor loss."""
    return agent_from_config[
        SmallREDQConfig[target, OBS, ACT, BATCH, CAP, HIDDEN]
    ](
        ctx=ctx,
        actor_lr=actor_lr, critic_lr=critic_lr, alpha_lr=alpha_lr,
        gamma=gamma, tau=tau,
        action_scale=action_scale,
        init_alpha=init_alpha, target_entropy=target_entropy,
        learning_starts=learning_starts,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
        max_grad_norm=max_grad_norm,
    )
