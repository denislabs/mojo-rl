"""C51 / Rainbow named presets — config descriptors + factories (Design F).

Additive sugar over the primitive
`C51Agent[train_target, SAMPLE, Q_NET, N_ATOMS, NUM_ACTIONS, DOUBLE]`.
The primitive stays the source of truth for arbitrary combinations; this
module names the common algorithms and bundles their tuned defaults.

Three pieces:

  1. `C51ConfigT` — a trait bundling the FULL compile-time identity of an
     algorithm: the deployment `TARGET`, the replay `SAMPLE` block, the
     Q-net `Q_NET`, the distributional flags (`N_ATOMS` / `NUM_ACTIONS` /
     `DOUBLE`), plus tuned scalar defaults (`DEF_*`). Scalars are comptime
     only so they can seed `__init__` kwarg defaults — still overridable.

  2. `C51Config` / `DoubleC51Config` / `RainbowConfig` — conformers,
     parametrized by `target`. Because the replay block is target-generic
     (`ReplaySampleStep[AnyReplay[target,…]]` / `NStepSampleStep[N, AnyPerReplay[target,…]]`),
     ONE config struct covers cpu and gpu — no per-target duplication.

  3. `agent_from_config` + capitalized presets `C51` / `DoubleC51` /
     `Rainbow`. Each preset is a SINGLE function taking `target` as a
     parameter (return type stays non-conditional because the sample
     block is one target-parametrized type). Capitalized names read like
     constructors at the call site:

         var agent = Rainbow["gpu", OBS, ACT, BATCH, CAP](ctx=ctx)
         var agent = C51["cpu", OBS, ACT, BATCH, CAP](lr=2.5e-4)
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.noisy_linear import NoisyLinear
from mojo_rl.nn2.primitives.dueling_head_c51 import DuelingHeadC51
from mojo_rl.nn2.combinators.sequential import Sequential

from ..training.blocks import (
    SampleBlock,
    ReplaySampleStep,
    NStepSampleStep,
)
from ..data.any_replay import AnyReplay
from ..data.any_per_replay import AnyPerReplay

from .agent import C51Agent


# ──────────────────────────────────────────────────────────────────────
# Net presets (target-agnostic, parametrized comptime aliases).
# ──────────────────────────────────────────────────────────────────────

comptime C51Net[OBS: Int, ACT: Int, NA: Int, HIDDEN: Int] = Sequential[
    Linear[OBS, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, ACT * NA],
]
"""Plain categorical Q-net: outputs ACT · NA per-atom logits."""

comptime RainbowNet[OBS: Int, ACT: Int, NA: Int, HIDDEN: Int] = Sequential[
    Linear[OBS, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    NoisyLinear[HIDDEN, (1 + ACT) * NA],
    DuelingHeadC51[ACT, NA],
]
"""Distributional dueling net with NoisyLinear exploration: the wide
projection emits (1 + ACT) · NA, split inside DuelingHeadC51 into a
value stream V[NA] and an advantage stream A[ACT, NA]."""


# ──────────────────────────────────────────────────────────────────────
# Config trait — full compile-time identity + tuned scalar defaults.
# ──────────────────────────────────────────────────────────────────────


trait C51ConfigT(Copyable, Movable, ImplicitlyDestructible):
    """Compile-time descriptor of a C51-family algorithm. Bundles the
    deployment target, the replay block, the Q-net, the distributional
    flags, and tuned scalar defaults. Conformers are zero-field comptime
    tags — never instantiated at runtime; only their comptime members
    are read."""

    comptime TARGET: StaticString
    comptime SAMPLE: SampleBlock
    comptime Q_NET: Module
    comptime N_ATOMS: Int
    comptime NUM_ACTIONS: Int
    comptime DOUBLE: Bool

    # Tuned scalar defaults (read into __init__ kwarg defaults).
    comptime DEF_LR: Scalar[DT]
    comptime DEF_GAMMA: Scalar[DT]
    comptime DEF_TAU: Scalar[DT]
    comptime DEF_EPS: Scalar[DT]
    comptime DEF_EPS_DECAY: Scalar[DT]
    comptime DEF_EPS_MIN: Scalar[DT]
    comptime DEF_LEARNING_STARTS: Int
    comptime DEF_TARGET_UPDATE_FREQ: Int
    comptime DEF_MAX_GRAD_NORM: Scalar[DT]
    comptime DEF_PER_ALPHA: Scalar[DT]
    comptime DEF_PER_BETA: Scalar[DT]
    comptime DEF_PER_EPSILON: Scalar[DT]
    comptime DEF_NSTEP: Int
    comptime DEF_VMIN: Scalar[DT]
    comptime DEF_VMAX: Scalar[DT]


# ──────────────────────────────────────────────────────────────────────
# Conformers — one struct per algorithm, parametrized by target.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct C51Config[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    NA: Int = 51, HIDDEN: Int = 64,
](C51ConfigT):
    """Vanilla C51 (Bellemare et al. 2017). Plain net, single Q-target,
    epsilon-greedy, uniform replay (1-step)."""

    comptime TARGET = Self.target
    comptime SAMPLE = ReplaySampleStep[
        AnyReplay[Self.target, Self.OBS, 1, Self.CAP], Self.BATCH
    ]
    comptime Q_NET = C51Net[Self.OBS, Self.ACT, Self.NA, Self.HIDDEN]
    comptime N_ATOMS = Self.NA
    comptime NUM_ACTIONS = Self.ACT
    comptime DOUBLE = False

    comptime DEF_LR = Scalar[DT](1e-4)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_EPS = Scalar[DT](1.0)
    comptime DEF_EPS_DECAY = Scalar[DT](0.995)
    comptime DEF_EPS_MIN = Scalar[DT](0.05)
    comptime DEF_LEARNING_STARTS = 1_000
    comptime DEF_TARGET_UPDATE_FREQ = 500
    comptime DEF_MAX_GRAD_NORM = Scalar[DT](10.0)
    comptime DEF_PER_ALPHA = Scalar[DT](0.6)
    comptime DEF_PER_BETA = Scalar[DT](0.4)
    comptime DEF_PER_EPSILON = Scalar[DT](1e-6)
    comptime DEF_NSTEP = 1
    comptime DEF_VMIN = Scalar[DT](-10.0)
    comptime DEF_VMAX = Scalar[DT](10.0)


@fieldwise_init
struct DoubleC51Config[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    NA: Int = 51, HIDDEN: Int = 64,
](C51ConfigT):
    """Double C51 — identical to `C51Config` but `DOUBLE=True`
    (online-net argmax, target-net evaluation; van Hasselt 2016)."""

    comptime TARGET = Self.target
    comptime SAMPLE = ReplaySampleStep[
        AnyReplay[Self.target, Self.OBS, 1, Self.CAP], Self.BATCH
    ]
    comptime Q_NET = C51Net[Self.OBS, Self.ACT, Self.NA, Self.HIDDEN]
    comptime N_ATOMS = Self.NA
    comptime NUM_ACTIONS = Self.ACT
    comptime DOUBLE = True

    comptime DEF_LR = Scalar[DT](1e-4)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_EPS = Scalar[DT](1.0)
    comptime DEF_EPS_DECAY = Scalar[DT](0.995)
    comptime DEF_EPS_MIN = Scalar[DT](0.05)
    comptime DEF_LEARNING_STARTS = 1_000
    comptime DEF_TARGET_UPDATE_FREQ = 500
    comptime DEF_MAX_GRAD_NORM = Scalar[DT](10.0)
    comptime DEF_PER_ALPHA = Scalar[DT](0.6)
    comptime DEF_PER_BETA = Scalar[DT](0.4)
    comptime DEF_PER_EPSILON = Scalar[DT](1e-6)
    comptime DEF_NSTEP = 1
    comptime DEF_VMIN = Scalar[DT](-10.0)
    comptime DEF_VMAX = Scalar[DT](10.0)


@fieldwise_init
struct RainbowConfig[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    NA: Int = 51, HIDDEN: Int = 64, NSTEP: Int = 3,
](C51ConfigT):
    """Rainbow (Hessel et al. 2018) — all six components: PER + N-step
    replay (`NStepSampleStep` over `AnyPerReplay`), dueling distributional net,
    `DOUBLE=True`, `ε=0` (Noisy supplies exploration), `nstep=NSTEP`."""

    comptime TARGET = Self.target
    comptime SAMPLE = NStepSampleStep[
        Self.NSTEP, AnyPerReplay[Self.target, Self.OBS, 1, Self.CAP], Self.BATCH
    ]
    comptime Q_NET = RainbowNet[Self.OBS, Self.ACT, Self.NA, Self.HIDDEN]
    comptime N_ATOMS = Self.NA
    comptime NUM_ACTIONS = Self.ACT
    comptime DOUBLE = True

    comptime DEF_LR = Scalar[DT](1e-4)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_EPS = Scalar[DT](0.0)          # Noisy nets → no ε-greedy
    comptime DEF_EPS_DECAY = Scalar[DT](1.0)
    comptime DEF_EPS_MIN = Scalar[DT](0.0)
    comptime DEF_LEARNING_STARTS = 1_000
    comptime DEF_TARGET_UPDATE_FREQ = 500
    comptime DEF_MAX_GRAD_NORM = Scalar[DT](10.0)
    comptime DEF_PER_ALPHA = Scalar[DT](0.5)
    comptime DEF_PER_BETA = Scalar[DT](0.4)
    comptime DEF_PER_EPSILON = Scalar[DT](1e-6)
    comptime DEF_NSTEP = Self.NSTEP
    comptime DEF_VMIN = Scalar[DT](-10.0)
    comptime DEF_VMAX = Scalar[DT](10.0)


# ──────────────────────────────────────────────────────────────────────
# Generic factory — any C51ConfigT → primitive agent, defaults applied.
# ──────────────────────────────────────────────────────────────────────


def agent_from_config[
    CONFIG: C51ConfigT,
](
    ctx: Optional[DeviceContext] = None,
    lr: Scalar[DT] = CONFIG.DEF_LR,
    gamma: Scalar[DT] = CONFIG.DEF_GAMMA,
    tau: Scalar[DT] = CONFIG.DEF_TAU,
    epsilon: Scalar[DT] = CONFIG.DEF_EPS,
    epsilon_decay: Scalar[DT] = CONFIG.DEF_EPS_DECAY,
    epsilon_min: Scalar[DT] = CONFIG.DEF_EPS_MIN,
    learning_starts: Int = CONFIG.DEF_LEARNING_STARTS,
    target_update_freq: Int = CONFIG.DEF_TARGET_UPDATE_FREQ,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = 0.0,
    max_grad_norm: Scalar[DT] = CONFIG.DEF_MAX_GRAD_NORM,
    per_alpha: Scalar[DT] = CONFIG.DEF_PER_ALPHA,
    per_beta: Scalar[DT] = CONFIG.DEF_PER_BETA,
    per_epsilon: Scalar[DT] = CONFIG.DEF_PER_EPSILON,
    nstep: Int = CONFIG.DEF_NSTEP,
    v_min: Scalar[DT] = CONFIG.DEF_VMIN,
    v_max: Scalar[DT] = CONFIG.DEF_VMAX,
) raises -> C51Agent[
    CONFIG.TARGET,
    CONFIG.SAMPLE,
    CONFIG.Q_NET,
    CONFIG.N_ATOMS,
    CONFIG.NUM_ACTIONS,
    CONFIG.DOUBLE,
]:
    """Build the primitive `C51Agent` from any `C51ConfigT`. Every scalar
    defaults to the config's tuned value but stays overridable. The
    deployment target and replay block are read off the config, so this
    one function serves cpu and gpu."""
    return C51Agent[
        CONFIG.TARGET,
        CONFIG.SAMPLE,
        CONFIG.Q_NET,
        CONFIG.N_ATOMS,
        CONFIG.NUM_ACTIONS,
        CONFIG.DOUBLE,
    ](
        ctx=ctx,
        lr=lr,
        gamma=gamma,
        tau=tau,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        learning_starts=learning_starts,
        target_update_freq=target_update_freq,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
        max_grad_norm=max_grad_norm,
        per_alpha=per_alpha,
        per_beta=per_beta,
        per_epsilon=per_epsilon,
        nstep=nstep,
        v_min=v_min,
        v_max=v_max,
    )


# ──────────────────────────────────────────────────────────────────────
# Capitalized presets — single function each, `target` as a parameter.
# Read like constructors. Full tuning surface, defaults from the config.
# ──────────────────────────────────────────────────────────────────────


def C51[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    NA: Int = 51, HIDDEN: Int = 64,
](
    ctx: Optional[DeviceContext] = None,
    lr: Scalar[DT] = C51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_LR,
    gamma: Scalar[DT] = C51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_GAMMA,
    tau: Scalar[DT] = C51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_TAU,
    epsilon: Scalar[DT] = C51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_EPS,
    epsilon_decay: Scalar[DT] = C51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_EPS_DECAY,
    epsilon_min: Scalar[DT] = C51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_EPS_MIN,
    learning_starts: Int = C51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_LEARNING_STARTS,
    target_update_freq: Int = C51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_TARGET_UPDATE_FREQ,
    window_size: Int = 10,
    max_grad_norm: Scalar[DT] = C51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_MAX_GRAD_NORM,
    v_min: Scalar[DT] = C51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_VMIN,
    v_max: Scalar[DT] = C51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_VMAX,
) raises -> C51Agent[
    target,
    ReplaySampleStep[AnyReplay[target, OBS, 1, CAP], BATCH],
    C51Net[OBS, ACT, NA, HIDDEN],
    NA, ACT, False,
]:
    """Vanilla C51 (uniform replay, single Q-target, ε-greedy)."""
    return agent_from_config[C51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN]](
        ctx=ctx, lr=lr, gamma=gamma, tau=tau, epsilon=epsilon,
        epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
        learning_starts=learning_starts, target_update_freq=target_update_freq,
        window_size=window_size, max_grad_norm=max_grad_norm,
        v_min=v_min, v_max=v_max,
    )


def DoubleC51[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    NA: Int = 51, HIDDEN: Int = 64,
](
    ctx: Optional[DeviceContext] = None,
    lr: Scalar[DT] = DoubleC51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_LR,
    gamma: Scalar[DT] = DoubleC51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_GAMMA,
    tau: Scalar[DT] = DoubleC51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_TAU,
    epsilon: Scalar[DT] = DoubleC51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_EPS,
    epsilon_decay: Scalar[DT] = DoubleC51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_EPS_DECAY,
    epsilon_min: Scalar[DT] = DoubleC51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_EPS_MIN,
    learning_starts: Int = DoubleC51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_LEARNING_STARTS,
    target_update_freq: Int = DoubleC51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_TARGET_UPDATE_FREQ,
    window_size: Int = 10,
    max_grad_norm: Scalar[DT] = DoubleC51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_MAX_GRAD_NORM,
    v_min: Scalar[DT] = DoubleC51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_VMIN,
    v_max: Scalar[DT] = DoubleC51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_VMAX,
) raises -> C51Agent[
    target,
    ReplaySampleStep[AnyReplay[target, OBS, 1, CAP], BATCH],
    C51Net[OBS, ACT, NA, HIDDEN],
    NA, ACT, True,
]:
    """Double C51 (uniform replay, online-argmax / target-eval)."""
    return agent_from_config[DoubleC51Config[target, OBS, ACT, BATCH, CAP, NA, HIDDEN]](
        ctx=ctx, lr=lr, gamma=gamma, tau=tau, epsilon=epsilon,
        epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
        learning_starts=learning_starts, target_update_freq=target_update_freq,
        window_size=window_size, max_grad_norm=max_grad_norm,
        v_min=v_min, v_max=v_max,
    )


def Rainbow[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    NA: Int = 51, HIDDEN: Int = 64, NSTEP: Int = 3,
](
    ctx: Optional[DeviceContext] = None,
    lr: Scalar[DT] = RainbowConfig[target, OBS, ACT, BATCH, CAP, NA, HIDDEN, NSTEP].DEF_LR,
    gamma: Scalar[DT] = RainbowConfig[target, OBS, ACT, BATCH, CAP, NA, HIDDEN, NSTEP].DEF_GAMMA,
    tau: Scalar[DT] = RainbowConfig[target, OBS, ACT, BATCH, CAP, NA, HIDDEN, NSTEP].DEF_TAU,
    epsilon: Scalar[DT] = RainbowConfig[target, OBS, ACT, BATCH, CAP, NA, HIDDEN, NSTEP].DEF_EPS,
    learning_starts: Int = RainbowConfig[target, OBS, ACT, BATCH, CAP, NA, HIDDEN, NSTEP].DEF_LEARNING_STARTS,
    target_update_freq: Int = RainbowConfig[target, OBS, ACT, BATCH, CAP, NA, HIDDEN, NSTEP].DEF_TARGET_UPDATE_FREQ,
    window_size: Int = 10,
    max_grad_norm: Scalar[DT] = RainbowConfig[target, OBS, ACT, BATCH, CAP, NA, HIDDEN, NSTEP].DEF_MAX_GRAD_NORM,
    per_alpha: Scalar[DT] = RainbowConfig[target, OBS, ACT, BATCH, CAP, NA, HIDDEN, NSTEP].DEF_PER_ALPHA,
    per_beta: Scalar[DT] = RainbowConfig[target, OBS, ACT, BATCH, CAP, NA, HIDDEN, NSTEP].DEF_PER_BETA,
    per_epsilon: Scalar[DT] = RainbowConfig[target, OBS, ACT, BATCH, CAP, NA, HIDDEN, NSTEP].DEF_PER_EPSILON,
    v_min: Scalar[DT] = RainbowConfig[target, OBS, ACT, BATCH, CAP, NA, HIDDEN, NSTEP].DEF_VMIN,
    v_max: Scalar[DT] = RainbowConfig[target, OBS, ACT, BATCH, CAP, NA, HIDDEN, NSTEP].DEF_VMAX,
) raises -> C51Agent[
    target,
    NStepSampleStep[NSTEP, AnyPerReplay[target, OBS, 1, CAP], BATCH],
    RainbowNet[OBS, ACT, NA, HIDDEN],
    NA, ACT, True,
]:
    """Rainbow — six-of-six (PER + N-step + Dueling + Noisy + Double +
    C51). `nstep` is fixed to `NSTEP` so the replay accumulator and the
    target-Y γ^n bootstrap stay aligned."""
    return agent_from_config[
        RainbowConfig[target, OBS, ACT, BATCH, CAP, NA, HIDDEN, NSTEP]
    ](
        ctx=ctx, lr=lr, gamma=gamma, tau=tau, epsilon=epsilon,
        learning_starts=learning_starts, target_update_freq=target_update_freq,
        window_size=window_size, max_grad_norm=max_grad_norm,
        per_alpha=per_alpha, per_beta=per_beta, per_epsilon=per_epsilon,
        v_min=v_min, v_max=v_max,
    )
