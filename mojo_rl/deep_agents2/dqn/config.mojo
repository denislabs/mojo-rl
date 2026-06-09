"""DQN named presets — config descriptors + factories (Design F).

Additive sugar over the primitive
`DQNAgent[train_target, SAMPLE, Q_NET, DOUBLE]`, mirroring
`c51/config.mojo`. The primitive stays the source of truth for arbitrary
combinations; this module names the common algorithms and bundles their
tuned defaults. Adding a new variant is a ~30-line conformer struct — no
new trainer code (exactly how `RainbowConfig` rides `C51Trainer`).

Three pieces:

  1. `DQNConfigT` — a trait bundling the FULL compile-time identity of an
     algorithm: the deployment `TARGET`, the replay `SAMPLE` block, the
     Q-net `Q_NET`, the Double-DQN flag (`DOUBLE`), plus tuned scalar
     defaults (`DEF_*`). Scalars are comptime only so they can seed
     `__init__` kwarg defaults — still overridable.

  2. `DQNConfig` / `DoubleDQNConfig` — conformers, parametrized by
     `target`. Because the replay block is target-generic
     (`ReplaySampleStep[AnyReplay[target,…]]`), ONE config struct covers
     cpu and gpu — no per-target duplication.

  3. `agent_from_config` + capitalized presets `DQN` / `DoubleDQN` /
     `DuelingDQN` / `NoisyDQN` / `DQNPER` / `RainbowDQN`. Each preset is a
     SINGLE function taking `target` as a parameter. They read like
     constructors at the call site:

         var agent = DQN["gpu", OBS, ACT, BATCH, CAP](ctx=ctx)
         var agent = DoubleDQN["cpu", OBS, ACT, BATCH, CAP](lr=1e-3)
         var agent = RainbowDQN["gpu", OBS, ACT, BATCH, CAP](ctx=ctx)

Coverage: the full single-env DQN family — plain, Double, Dueling,
Noisy, PER, scalar-Q Rainbow (Double+Dueling+Noisy+PER+N-step), and
(Phase 1.6) the Nature-CNN pixel config `DQNCNNConfig` / `DQNCNN`. Each
is a ~30-line conformer over the SAME `DQNTrainer` / `DQNAgent`
primitive; no new trainer code (exactly how `RainbowConfig` rides
`C51Trainer`). The CNN config needed no obs-pipeline change either: nn2
`Conv2D`/`Flatten` expose flat `IN_DIMS`/`OUT_DIM`, so a
`Sequential[Conv2D, …, Flatten, Linear, …]` Q-net flows through the
discrete driver with the obs treated as a flat `FRAMES·84·84` vector
(the same insight Phase 5.2 used for on-policy CNN PPO).
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.linear_relu import LinearReLU
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.noisy_linear import NoisyLinear
from mojo_rl.nn2.primitives.dueling_head import DuelingHead
from mojo_rl.nn2.primitives.conv2d import Conv2D
from mojo_rl.nn2.primitives.flatten import Flatten
from mojo_rl.nn2.combinators.sequential import Sequential

from ..training.blocks import (
    SampleBlock,
    ReplaySampleStep,
    NStepSampleStep,
)
from ..data.any_replay import AnyReplay
from ..data.any_per_replay import AnyPerReplay

from .agent import DQNAgent


# ──────────────────────────────────────────────────────────────────────
# Net presets (target-agnostic, parametrized comptime aliases).
# ──────────────────────────────────────────────────────────────────────

comptime DQNNet[OBS: Int, ACT: Int, HIDDEN: Int] = Sequential[
    LinearReLU[OBS, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, ACT],
]
"""Plain scalar Q-net: outputs one Q-value per action. Hidden layers fused
(LinearReLU); the Q head is a plain Linear."""

comptime DuelingDQNNet[OBS: Int, ACT: Int, HIDDEN: Int] = Sequential[
    LinearReLU[OBS, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, 1 + ACT],
    DuelingHead[ACT],
]
"""Dueling scalar Q-net: the wide projection emits (1 + ACT), split inside
`DuelingHead` into a value scalar V and an advantage stream A[ACT].
Wired in Phase 1 (`DuelingDQNConfig`)."""

comptime NoisyDQNNet[OBS: Int, ACT: Int, HIDDEN: Int] = Sequential[
    LinearReLU[OBS, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    NoisyLinear[HIDDEN, ACT],
]
"""NoisyLinear head supplies exploration in place of ε-greedy."""

comptime RainbowDQNNet[OBS: Int, ACT: Int, HIDDEN: Int] = Sequential[
    LinearReLU[OBS, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    NoisyLinear[HIDDEN, 1 + ACT],
    DuelingHead[ACT],
]
"""Scalar-Q dueling + Noisy net — the non-distributional analogue of
`c51/config.mojo`'s `RainbowNet`. The wide NoisyLinear projection emits
(1 + ACT), split inside `DuelingHead` into V + A streams. Used by
`RainbowDQNConfig`."""

comptime NatureDQNNet[FRAMES: Int, ACT: Int, HIDDEN: Int = 512] = Sequential[
    Conv2D[FRAMES, 32, 8, 4, 0, 84, 84], ReLU[32 * 20 * 20],
    Conv2D[32, 64, 4, 2, 0, 20, 20], ReLU[64 * 9 * 9],
    Conv2D[64, 64, 3, 1, 0, 9, 9], ReLU[64 * 7 * 7],
    Flatten[64 * 7 * 7],
    LinearReLU[64 * 7 * 7, HIDDEN],
    Linear[HIDDEN, ACT],
]
"""Canonical Nature-DQN CNN (Mnih et al. 2015 / CleanRL DQN-Atari) for
`FRAMES`×84×84 stacked-frame pixel obs. Fixed conv geometry:
  conv1 FRAMES→32 (8×8 s4 p0): 84→20   (32·20·20)
  conv2 32→64    (4×4 s2 p0): 20→9    (64·9·9)
  conv3 64→64    (3×3 s1 p0): 9→7     (64·7·7 = 3136)
  Flatten → Linear(3136→HIDDEN) → ReLU → Linear(HIDDEN→ACT).
nn2 `Conv2D`/`Flatten` expose FLAT `IN_DIMS`/`OUT_DIM` (Conv2D's
`IN_DIM_FLAT = FRAMES·84·84`), so this `Sequential` plugs into the
discrete off-policy trainer/driver with the obs treated as a flat
`FRAMES·84·84` vector — ZERO plumbing changes, exactly as Phase 5.2's CNN
PPO did on the on-policy side. Used by `DQNCNNConfig`."""


# ──────────────────────────────────────────────────────────────────────
# Config trait — full compile-time identity + tuned scalar defaults.
# ──────────────────────────────────────────────────────────────────────


trait DQNConfigT(Copyable, Movable, ImplicitlyDestructible):
    """Compile-time descriptor of a DQN-family algorithm. Bundles the
    deployment target, the replay block, the Q-net, the Double flag, and
    tuned scalar defaults. Conformers are zero-field comptime tags — never
    instantiated at runtime; only their comptime members are read.

    The PER / N-step defaults (`DEF_PER_*`, `DEF_NSTEP`) are part of the
    contract for forward-compat with the Phase-1 prioritized / n-step
    configs; the plain + Double conformers set benign values that the
    uniform-replay path ignores."""

    comptime TARGET: StaticString
    comptime SAMPLE: SampleBlock
    comptime Q_NET: Module
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


# ──────────────────────────────────────────────────────────────────────
# Conformers — one struct per algorithm, parametrized by target.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct DQNConfig[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 64,
](DQNConfigT):
    """Vanilla DQN (Mnih et al. 2015). Plain net, single Q-target,
    ε-greedy, uniform replay (1-step)."""

    comptime TARGET = Self.target
    comptime SAMPLE = ReplaySampleStep[
        AnyReplay[Self.target, Self.OBS, 1, Self.CAP], Self.BATCH
    ]
    comptime Q_NET = DQNNet[Self.OBS, Self.ACT, Self.HIDDEN]
    comptime DOUBLE = False

    comptime DEF_LR = Scalar[DT](1e-3)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_EPS = Scalar[DT](1.0)
    comptime DEF_EPS_DECAY = Scalar[DT](0.995)
    comptime DEF_EPS_MIN = Scalar[DT](0.01)
    comptime DEF_LEARNING_STARTS = 1_000
    comptime DEF_TARGET_UPDATE_FREQ = 0
    comptime DEF_MAX_GRAD_NORM = Scalar[DT](0.0)
    comptime DEF_PER_ALPHA = Scalar[DT](0.6)
    comptime DEF_PER_BETA = Scalar[DT](0.4)
    comptime DEF_PER_EPSILON = Scalar[DT](1e-6)
    comptime DEF_NSTEP = 1


@fieldwise_init
struct DoubleDQNConfig[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 64,
](DQNConfigT):
    """Double DQN — identical to `DQNConfig` but `DOUBLE=True` (online-net
    argmax, target-net evaluation; van Hasselt et al. 2016)."""

    comptime TARGET = Self.target
    comptime SAMPLE = ReplaySampleStep[
        AnyReplay[Self.target, Self.OBS, 1, Self.CAP], Self.BATCH
    ]
    comptime Q_NET = DQNNet[Self.OBS, Self.ACT, Self.HIDDEN]
    comptime DOUBLE = True

    comptime DEF_LR = Scalar[DT](1e-3)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_EPS = Scalar[DT](1.0)
    comptime DEF_EPS_DECAY = Scalar[DT](0.995)
    comptime DEF_EPS_MIN = Scalar[DT](0.01)
    comptime DEF_LEARNING_STARTS = 1_000
    comptime DEF_TARGET_UPDATE_FREQ = 0
    comptime DEF_MAX_GRAD_NORM = Scalar[DT](0.0)
    comptime DEF_PER_ALPHA = Scalar[DT](0.6)
    comptime DEF_PER_BETA = Scalar[DT](0.4)
    comptime DEF_PER_EPSILON = Scalar[DT](1e-6)
    comptime DEF_NSTEP = 1


@fieldwise_init
struct DuelingDQNConfig[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 64,
](DQNConfigT):
    """Dueling DQN (Wang et al. 2016). Shared trunk → V(s) + A(s,·)
    streams aggregated by `DuelingHead`; uniform replay, ε-greedy,
    single Q-target."""

    comptime TARGET = Self.target
    comptime SAMPLE = ReplaySampleStep[
        AnyReplay[Self.target, Self.OBS, 1, Self.CAP], Self.BATCH
    ]
    comptime Q_NET = DuelingDQNNet[Self.OBS, Self.ACT, Self.HIDDEN]
    comptime DOUBLE = False

    comptime DEF_LR = Scalar[DT](1e-3)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_EPS = Scalar[DT](1.0)
    comptime DEF_EPS_DECAY = Scalar[DT](0.995)
    comptime DEF_EPS_MIN = Scalar[DT](0.01)
    comptime DEF_LEARNING_STARTS = 1_000
    comptime DEF_TARGET_UPDATE_FREQ = 0
    comptime DEF_MAX_GRAD_NORM = Scalar[DT](0.0)
    comptime DEF_PER_ALPHA = Scalar[DT](0.6)
    comptime DEF_PER_BETA = Scalar[DT](0.4)
    comptime DEF_PER_EPSILON = Scalar[DT](1e-6)
    comptime DEF_NSTEP = 1


@fieldwise_init
struct NoisyDQNConfig[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 64,
](DQNConfigT):
    """Noisy DQN (Fortunato et al. 2018). `NoisyLinear` output head
    supplies state-dependent exploration, so ε-greedy is disabled
    (`DEF_EPS=0`). Uniform replay, single Q-target."""

    comptime TARGET = Self.target
    comptime SAMPLE = ReplaySampleStep[
        AnyReplay[Self.target, Self.OBS, 1, Self.CAP], Self.BATCH
    ]
    comptime Q_NET = NoisyDQNNet[Self.OBS, Self.ACT, Self.HIDDEN]
    comptime DOUBLE = False

    comptime DEF_LR = Scalar[DT](1e-3)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_EPS = Scalar[DT](0.0)          # Noisy nets → no ε-greedy
    comptime DEF_EPS_DECAY = Scalar[DT](1.0)
    comptime DEF_EPS_MIN = Scalar[DT](0.0)
    comptime DEF_LEARNING_STARTS = 1_000
    comptime DEF_TARGET_UPDATE_FREQ = 0
    comptime DEF_MAX_GRAD_NORM = Scalar[DT](0.0)
    comptime DEF_PER_ALPHA = Scalar[DT](0.6)
    comptime DEF_PER_BETA = Scalar[DT](0.4)
    comptime DEF_PER_EPSILON = Scalar[DT](1e-6)
    comptime DEF_NSTEP = 1


@fieldwise_init
struct DQNPERConfig[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 64,
](DQNConfigT):
    """DQN + Prioritized Experience Replay (Schaul et al. 2016). Plain
    net, `DOUBLE=True` (PER is canonically paired with Double-DQN),
    ε-greedy, prioritized 1-step replay over `AnyPerReplay`."""

    comptime TARGET = Self.target
    comptime SAMPLE = ReplaySampleStep[
        AnyPerReplay[Self.target, Self.OBS, 1, Self.CAP], Self.BATCH
    ]
    comptime Q_NET = DQNNet[Self.OBS, Self.ACT, Self.HIDDEN]
    comptime DOUBLE = True

    comptime DEF_LR = Scalar[DT](1e-3)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_EPS = Scalar[DT](1.0)
    comptime DEF_EPS_DECAY = Scalar[DT](0.995)
    comptime DEF_EPS_MIN = Scalar[DT](0.01)
    comptime DEF_LEARNING_STARTS = 1_000
    comptime DEF_TARGET_UPDATE_FREQ = 0
    comptime DEF_MAX_GRAD_NORM = Scalar[DT](10.0)
    comptime DEF_PER_ALPHA = Scalar[DT](0.6)
    comptime DEF_PER_BETA = Scalar[DT](0.4)
    comptime DEF_PER_EPSILON = Scalar[DT](1e-6)
    comptime DEF_NSTEP = 1


@fieldwise_init
struct RainbowDQNConfig[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 64, NSTEP: Int = 3,
](DQNConfigT):
    """Scalar-Q Rainbow — the non-distributional analogue of
    `c51/config.mojo`'s `RainbowConfig`: Double + Dueling + Noisy + PER +
    N-step (five of Rainbow's six; the sixth is C51 distributional, which
    is what lives on the C51 side). `nstep` is fixed to `NSTEP` so the
    replay accumulator and the γ^n target bootstrap stay aligned;
    `DEF_EPS=0` since Noisy supplies exploration."""

    comptime TARGET = Self.target
    comptime SAMPLE = NStepSampleStep[
        Self.NSTEP,
        AnyPerReplay[Self.target, Self.OBS, 1, Self.CAP],
        Self.BATCH,
    ]
    comptime Q_NET = RainbowDQNNet[Self.OBS, Self.ACT, Self.HIDDEN]
    comptime DOUBLE = True

    comptime DEF_LR = Scalar[DT](1e-3)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_EPS = Scalar[DT](0.0)          # Noisy nets → no ε-greedy
    comptime DEF_EPS_DECAY = Scalar[DT](1.0)
    comptime DEF_EPS_MIN = Scalar[DT](0.0)
    comptime DEF_LEARNING_STARTS = 1_000
    comptime DEF_TARGET_UPDATE_FREQ = 0
    comptime DEF_MAX_GRAD_NORM = Scalar[DT](10.0)
    comptime DEF_PER_ALPHA = Scalar[DT](0.5)
    comptime DEF_PER_BETA = Scalar[DT](0.4)
    comptime DEF_PER_EPSILON = Scalar[DT](1e-6)
    comptime DEF_NSTEP = Self.NSTEP


@fieldwise_init
struct DQNCNNConfig[
    target: StaticString,
    ACT: Int, BATCH: Int, CAP: Int,
    FRAMES: Int = 4, HIDDEN: Int = 512,
](DQNConfigT):
    """DQN with the canonical Nature CNN for `FRAMES`×84×84 pixel obs
    (Double DQN, matching Mnih et al. 2015 / CleanRL DQN-Atari). The
    image is carried as a flat `FRAMES·84·84` obs through the SAME
    `DQNTrainer` / `DQNAgent` / discrete off-policy driver — nn2 `Conv2D`/
    `Flatten` expose flat `IN_DIMS`/`OUT_DIM`, so no obs-pipeline change is
    needed (the deferral reason behind Phase 1.6 is dispelled exactly as
    Phase 5.2 dispelled it for on-policy CNN PPO).

    `OBS = FRAMES·84·84`. Uniform replay; `DOUBLE=True` (Nature/CleanRL
    canonically pair DQN-Atari with Double + reward/grad clipping). Tuned
    Atari defaults: `lr=2.5e-4`, `max_grad_norm=10`."""

    comptime OBS = Self.FRAMES * 84 * 84
    comptime TARGET = Self.target
    comptime SAMPLE = ReplaySampleStep[
        AnyReplay[Self.target, Self.OBS, 1, Self.CAP], Self.BATCH
    ]
    comptime Q_NET = NatureDQNNet[Self.FRAMES, Self.ACT, Self.HIDDEN]
    comptime DOUBLE = True

    comptime DEF_LR = Scalar[DT](2.5e-4)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_EPS = Scalar[DT](1.0)
    comptime DEF_EPS_DECAY = Scalar[DT](0.995)
    comptime DEF_EPS_MIN = Scalar[DT](0.01)
    comptime DEF_LEARNING_STARTS = 1_000
    comptime DEF_TARGET_UPDATE_FREQ = 0
    comptime DEF_MAX_GRAD_NORM = Scalar[DT](10.0)
    comptime DEF_PER_ALPHA = Scalar[DT](0.6)
    comptime DEF_PER_BETA = Scalar[DT](0.4)
    comptime DEF_PER_EPSILON = Scalar[DT](1e-6)
    comptime DEF_NSTEP = 1


# ──────────────────────────────────────────────────────────────────────
# Generic factory — any DQNConfigT → primitive agent, defaults applied.
# ──────────────────────────────────────────────────────────────────────


def agent_from_config[
    CONFIG: DQNConfigT,
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
) raises -> DQNAgent[
    CONFIG.TARGET,
    CONFIG.SAMPLE,
    CONFIG.Q_NET,
    CONFIG.DOUBLE,
]:
    """Build the primitive `DQNAgent` from any `DQNConfigT`. Every scalar
    defaults to the config's tuned value but stays overridable. The
    deployment target and replay block are read off the config, so this
    one function serves cpu and gpu. `per_*` / `nstep` are no-ops for
    uniform / single-step configs (the trainer's `configure_per` /
    `configure_gamma` default to no-op for those backends)."""
    return DQNAgent[
        CONFIG.TARGET,
        CONFIG.SAMPLE,
        CONFIG.Q_NET,
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
    )


# ──────────────────────────────────────────────────────────────────────
# Capitalized presets — single function each, `target` as a parameter.
# Read like constructors. Full tuning surface, defaults from the config.
# ──────────────────────────────────────────────────────────────────────


def DQN[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 64,
](
    ctx: Optional[DeviceContext] = None,
    lr: Scalar[DT] = DQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_LR,
    gamma: Scalar[DT] = DQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_GAMMA,
    tau: Scalar[DT] = DQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TAU,
    epsilon: Scalar[DT] = DQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_EPS,
    epsilon_decay: Scalar[DT] = DQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_EPS_DECAY,
    epsilon_min: Scalar[DT] = DQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_EPS_MIN,
    learning_starts: Int = DQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_LEARNING_STARTS,
    target_update_freq: Int = DQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TARGET_UPDATE_FREQ,
    window_size: Int = 10,
    max_grad_norm: Scalar[DT] = DQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_MAX_GRAD_NORM,
) raises -> DQNAgent[
    target,
    ReplaySampleStep[AnyReplay[target, OBS, 1, CAP], BATCH],
    DQNNet[OBS, ACT, HIDDEN],
    False,
]:
    """Vanilla DQN (uniform replay, single Q-target, ε-greedy)."""
    return agent_from_config[DQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN]](
        ctx=ctx, lr=lr, gamma=gamma, tau=tau, epsilon=epsilon,
        epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
        learning_starts=learning_starts, target_update_freq=target_update_freq,
        window_size=window_size, max_grad_norm=max_grad_norm,
    )


def DoubleDQN[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 64,
](
    ctx: Optional[DeviceContext] = None,
    lr: Scalar[DT] = DoubleDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_LR,
    gamma: Scalar[DT] = DoubleDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_GAMMA,
    tau: Scalar[DT] = DoubleDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TAU,
    epsilon: Scalar[DT] = DoubleDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_EPS,
    epsilon_decay: Scalar[DT] = DoubleDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_EPS_DECAY,
    epsilon_min: Scalar[DT] = DoubleDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_EPS_MIN,
    learning_starts: Int = DoubleDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_LEARNING_STARTS,
    target_update_freq: Int = DoubleDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TARGET_UPDATE_FREQ,
    window_size: Int = 10,
    max_grad_norm: Scalar[DT] = DoubleDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_MAX_GRAD_NORM,
) raises -> DQNAgent[
    target,
    ReplaySampleStep[AnyReplay[target, OBS, 1, CAP], BATCH],
    DQNNet[OBS, ACT, HIDDEN],
    True,
]:
    """Double DQN (uniform replay, online-argmax / target-eval)."""
    return agent_from_config[DoubleDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN]](
        ctx=ctx, lr=lr, gamma=gamma, tau=tau, epsilon=epsilon,
        epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
        learning_starts=learning_starts, target_update_freq=target_update_freq,
        window_size=window_size, max_grad_norm=max_grad_norm,
    )


def DuelingDQN[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 64,
](
    ctx: Optional[DeviceContext] = None,
    lr: Scalar[DT] = DuelingDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_LR,
    gamma: Scalar[DT] = DuelingDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_GAMMA,
    tau: Scalar[DT] = DuelingDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TAU,
    epsilon: Scalar[DT] = DuelingDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_EPS,
    epsilon_decay: Scalar[DT] = DuelingDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_EPS_DECAY,
    epsilon_min: Scalar[DT] = DuelingDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_EPS_MIN,
    learning_starts: Int = DuelingDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_LEARNING_STARTS,
    target_update_freq: Int = DuelingDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TARGET_UPDATE_FREQ,
    window_size: Int = 10,
    max_grad_norm: Scalar[DT] = DuelingDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_MAX_GRAD_NORM,
) raises -> DQNAgent[
    target,
    ReplaySampleStep[AnyReplay[target, OBS, 1, CAP], BATCH],
    DuelingDQNNet[OBS, ACT, HIDDEN],
    False,
]:
    """Dueling DQN (V + A streams, uniform replay, ε-greedy)."""
    return agent_from_config[DuelingDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN]](
        ctx=ctx, lr=lr, gamma=gamma, tau=tau, epsilon=epsilon,
        epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
        learning_starts=learning_starts, target_update_freq=target_update_freq,
        window_size=window_size, max_grad_norm=max_grad_norm,
    )


def NoisyDQN[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 64,
](
    ctx: Optional[DeviceContext] = None,
    lr: Scalar[DT] = NoisyDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_LR,
    gamma: Scalar[DT] = NoisyDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_GAMMA,
    tau: Scalar[DT] = NoisyDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TAU,
    learning_starts: Int = NoisyDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_LEARNING_STARTS,
    target_update_freq: Int = NoisyDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TARGET_UPDATE_FREQ,
    window_size: Int = 10,
    max_grad_norm: Scalar[DT] = NoisyDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_MAX_GRAD_NORM,
) raises -> DQNAgent[
    target,
    ReplaySampleStep[AnyReplay[target, OBS, 1, CAP], BATCH],
    NoisyDQNNet[OBS, ACT, HIDDEN],
    False,
]:
    """Noisy DQN (NoisyLinear exploration, ε disabled, uniform replay)."""
    return agent_from_config[NoisyDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN]](
        ctx=ctx, lr=lr, gamma=gamma, tau=tau,
        learning_starts=learning_starts, target_update_freq=target_update_freq,
        window_size=window_size, max_grad_norm=max_grad_norm,
    )


def DQNPER[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 64,
](
    ctx: Optional[DeviceContext] = None,
    lr: Scalar[DT] = DQNPERConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_LR,
    gamma: Scalar[DT] = DQNPERConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_GAMMA,
    tau: Scalar[DT] = DQNPERConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TAU,
    epsilon: Scalar[DT] = DQNPERConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_EPS,
    epsilon_decay: Scalar[DT] = DQNPERConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_EPS_DECAY,
    epsilon_min: Scalar[DT] = DQNPERConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_EPS_MIN,
    learning_starts: Int = DQNPERConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_LEARNING_STARTS,
    target_update_freq: Int = DQNPERConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TARGET_UPDATE_FREQ,
    window_size: Int = 10,
    max_grad_norm: Scalar[DT] = DQNPERConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_MAX_GRAD_NORM,
    per_alpha: Scalar[DT] = DQNPERConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_PER_ALPHA,
    per_beta: Scalar[DT] = DQNPERConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_PER_BETA,
    per_epsilon: Scalar[DT] = DQNPERConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_PER_EPSILON,
) raises -> DQNAgent[
    target,
    ReplaySampleStep[AnyPerReplay[target, OBS, 1, CAP], BATCH],
    DQNNet[OBS, ACT, HIDDEN],
    True,
]:
    """DQN + PER (prioritized 1-step replay, Double target, ε-greedy)."""
    return agent_from_config[DQNPERConfig[target, OBS, ACT, BATCH, CAP, HIDDEN]](
        ctx=ctx, lr=lr, gamma=gamma, tau=tau, epsilon=epsilon,
        epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
        learning_starts=learning_starts, target_update_freq=target_update_freq,
        window_size=window_size, max_grad_norm=max_grad_norm,
        per_alpha=per_alpha, per_beta=per_beta, per_epsilon=per_epsilon,
    )


def RainbowDQN[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 64, NSTEP: Int = 3,
](
    ctx: Optional[DeviceContext] = None,
    lr: Scalar[DT] = RainbowDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN, NSTEP].DEF_LR,
    gamma: Scalar[DT] = RainbowDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN, NSTEP].DEF_GAMMA,
    tau: Scalar[DT] = RainbowDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN, NSTEP].DEF_TAU,
    learning_starts: Int = RainbowDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN, NSTEP].DEF_LEARNING_STARTS,
    target_update_freq: Int = RainbowDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN, NSTEP].DEF_TARGET_UPDATE_FREQ,
    window_size: Int = 10,
    max_grad_norm: Scalar[DT] = RainbowDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN, NSTEP].DEF_MAX_GRAD_NORM,
    per_alpha: Scalar[DT] = RainbowDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN, NSTEP].DEF_PER_ALPHA,
    per_beta: Scalar[DT] = RainbowDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN, NSTEP].DEF_PER_BETA,
    per_epsilon: Scalar[DT] = RainbowDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN, NSTEP].DEF_PER_EPSILON,
) raises -> DQNAgent[
    target,
    NStepSampleStep[NSTEP, AnyPerReplay[target, OBS, 1, CAP], BATCH],
    RainbowDQNNet[OBS, ACT, HIDDEN],
    True,
]:
    """Scalar-Q Rainbow — Double + Dueling + Noisy + PER + N-step.
    `nstep` is fixed to `NSTEP` so the replay accumulator and the γ^n
    target bootstrap stay aligned."""
    return agent_from_config[
        RainbowDQNConfig[target, OBS, ACT, BATCH, CAP, HIDDEN, NSTEP]
    ](
        ctx=ctx, lr=lr, gamma=gamma, tau=tau,
        learning_starts=learning_starts, target_update_freq=target_update_freq,
        window_size=window_size, max_grad_norm=max_grad_norm,
        per_alpha=per_alpha, per_beta=per_beta, per_epsilon=per_epsilon,
        nstep=NSTEP,
    )


def DQNCNN[
    target: StaticString,
    ACT: Int, BATCH: Int, CAP: Int,
    FRAMES: Int = 4, HIDDEN: Int = 512,
](
    ctx: Optional[DeviceContext] = None,
    lr: Scalar[DT] = DQNCNNConfig[target, ACT, BATCH, CAP, FRAMES, HIDDEN].DEF_LR,
    gamma: Scalar[DT] = DQNCNNConfig[target, ACT, BATCH, CAP, FRAMES, HIDDEN].DEF_GAMMA,
    tau: Scalar[DT] = DQNCNNConfig[target, ACT, BATCH, CAP, FRAMES, HIDDEN].DEF_TAU,
    epsilon: Scalar[DT] = DQNCNNConfig[target, ACT, BATCH, CAP, FRAMES, HIDDEN].DEF_EPS,
    epsilon_decay: Scalar[DT] = DQNCNNConfig[target, ACT, BATCH, CAP, FRAMES, HIDDEN].DEF_EPS_DECAY,
    epsilon_min: Scalar[DT] = DQNCNNConfig[target, ACT, BATCH, CAP, FRAMES, HIDDEN].DEF_EPS_MIN,
    learning_starts: Int = DQNCNNConfig[target, ACT, BATCH, CAP, FRAMES, HIDDEN].DEF_LEARNING_STARTS,
    target_update_freq: Int = DQNCNNConfig[target, ACT, BATCH, CAP, FRAMES, HIDDEN].DEF_TARGET_UPDATE_FREQ,
    window_size: Int = 10,
    max_grad_norm: Scalar[DT] = DQNCNNConfig[target, ACT, BATCH, CAP, FRAMES, HIDDEN].DEF_MAX_GRAD_NORM,
) raises -> DQNAgent[
    target,
    ReplaySampleStep[AnyReplay[target, FRAMES * 84 * 84, 1, CAP], BATCH],
    NatureDQNNet[FRAMES, ACT, HIDDEN],
    True,
]:
    """DQN with the canonical Nature CNN for `FRAMES`×84×84 pixel obs
    (Double DQN). The image flows as a flat `FRAMES·84·84` obs through the
    unchanged discrete off-policy path — no obs-pipeline plumbing change."""
    return agent_from_config[
        DQNCNNConfig[target, ACT, BATCH, CAP, FRAMES, HIDDEN]
    ](
        ctx=ctx, lr=lr, gamma=gamma, tau=tau, epsilon=epsilon,
        epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
        learning_starts=learning_starts, target_update_freq=target_update_freq,
        window_size=window_size, max_grad_norm=max_grad_norm,
    )
