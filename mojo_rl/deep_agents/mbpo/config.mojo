"""MBPO named preset — config descriptor + factory (Design F).

Additive sugar over the primitive
`MBPOAgent[train_target, ACTOR, CRITIC, DynNet, OBS_DIM, ACT_DIM, BATCH,
           REPLAY_CAPACITY, SYNTH_CAPACITY, N_ENSEMBLE, NUM_ELITES,
           REAL_RATIO_PCT, LOGVAR_MIN, LOGVAR_MAX]`. MBPO has three nets
(SAC actor + critic + a probabilistic dynamics ensemble) and a large
schedule surface, so this preset bundles all three default nets plus the
tuned defaults that converge on classic-control (the reference 5/95 real
ratio + `LOGVAR_MAX=-2` diverge; the working regime is `REAL_RATIO_PCT=50`
+ `LOGVAR_MAX=-5`).

  1. `MBPOConfigT` — trait bundling the FULL compile-time identity: target,
     the three nets, dims, buffer capacities, ensemble knobs, the
     real-ratio split, the logvar bounds, plus tuned scalar defaults.

  2. `MBPOConfig` — conformer parametrized by `target` + structural ints.

  3. `agent_from_config` + the capitalized preset `MBPO`:

         var agent = MBPO["cpu", OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP]()

The SAC actor/critic hidden layers are FUSED `LinearReLU`; the dynamics
net keeps `Linear` + `Swish` (no fused-Swish layer exists) and a plain
linear mean/logvar head.
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.linear_relu import LinearReLU
from mojo_rl.nn.primitives.activations import Swish
from mojo_rl.nn.combinators.sequential import Sequential

from ..primitives.stochastic_actor import StochasticActor

from .agent import MBPOAgent


# ──────────────────────────────────────────────────────────────────────
# Net presets — target-agnostic, parametrized comptime aliases.
# ──────────────────────────────────────────────────────────────────────


comptime MBPOActorNet[OBS: Int, ACT: Int, HIDDEN: Int] = StochasticActor[
    OBS, ACT,
    LinearReLU[OBS, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
]
"""SAC stochastic actor — 2-layer fused trunk + (μ, log σ) heads."""


comptime MBPOCriticNet[OBS: Int, ACT: Int, HIDDEN: Int] = Sequential[
    LinearReLU[OBS + ACT, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, 1],
]
"""SAC twin-critic net (one of two)."""


comptime MBPODynNet[OBS: Int, ACT: Int, DYN_HIDDEN: Int] = Sequential[
    Linear[OBS + ACT, DYN_HIDDEN], Swish[DYN_HIDDEN],
    Linear[DYN_HIDDEN, DYN_HIDDEN], Swish[DYN_HIDDEN],
    Linear[DYN_HIDDEN, DYN_HIDDEN], Swish[DYN_HIDDEN],
    Linear[DYN_HIDDEN, DYN_HIDDEN], Swish[DYN_HIDDEN],
    Linear[DYN_HIDDEN, 2 * (1 + OBS)],
]
"""Probabilistic dynamics net: 4 Swish hidden layers → mean+logvar head
for [reward, Δobs]. One member of the `DynamicsEnsembleBlock`."""


# ──────────────────────────────────────────────────────────────────────
# Config trait — full compile-time identity + tuned scalar defaults.
# ──────────────────────────────────────────────────────────────────────


trait MBPOConfigT(Copyable, Movable, Deinitable):
    """Compile-time descriptor of an MBPO-family algorithm. Conformers are
    zero-field comptime tags."""

    comptime TARGET: StaticString
    comptime ACTOR: Module
    comptime CRITIC: Module
    comptime DYN_NET: Module
    comptime OBS_DIM: Int
    comptime ACT_DIM: Int
    comptime BATCH: Int
    comptime REPLAY_CAPACITY: Int
    comptime SYNTH_CAPACITY: Int
    comptime N_ENSEMBLE: Int
    comptime NUM_ELITES: Int
    comptime REAL_RATIO_PCT: Int
    comptime LOGVAR_MIN: Float64
    comptime LOGVAR_MAX: Float64

    # Tuned scalar defaults (read into __init__ kwarg defaults).
    comptime DEF_ACTOR_LR: Scalar[DT]
    comptime DEF_CRITIC_LR: Scalar[DT]
    comptime DEF_ALPHA_LR: Scalar[DT]
    comptime DEF_MODEL_LR: Scalar[DT]
    comptime DEF_GAMMA: Scalar[DT]
    comptime DEF_TAU: Scalar[DT]
    comptime DEF_ACTION_SCALE: Scalar[DT]
    comptime DEF_INIT_ALPHA: Scalar[DT]
    comptime DEF_TARGET_ENTROPY: Scalar[DT]
    comptime DEF_LEARNING_STARTS: Int


# ──────────────────────────────────────────────────────────────────────
# Conformer — parametrized by `target` + structural ints.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct MBPOConfig[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH_: Int,
    REPLAY_CAP: Int, SYNTH_CAP: Int,
    N_ENS: Int = 7, N_ELITES: Int = 5,
    REAL_RATIO: Int = 50,
    HIDDEN: Int = 256, DYN_HIDDEN: Int = 200,
    LOGVAR_MIN_: Float64 = -10.0, LOGVAR_MAX_: Float64 = -5.0,
](MBPOConfigT):
    """MBPO (Janner et al. 2019) — SAC over a learned probabilistic
    dynamics ensemble with elite selection + short synthetic rollouts.
    Defaults to the converging classic-control regime (`REAL_RATIO_PCT=50`,
    `LOGVAR_MAX=-5`); the reference 5/95 + `-2` regime diverges."""

    comptime TARGET = Self.target
    comptime ACTOR = MBPOActorNet[Self.OBS, Self.ACT, Self.HIDDEN]
    comptime CRITIC = MBPOCriticNet[Self.OBS, Self.ACT, Self.HIDDEN]
    comptime DYN_NET = MBPODynNet[Self.OBS, Self.ACT, Self.DYN_HIDDEN]
    comptime OBS_DIM = Self.OBS
    comptime ACT_DIM = Self.ACT
    comptime BATCH = Self.BATCH_
    comptime REPLAY_CAPACITY = Self.REPLAY_CAP
    comptime SYNTH_CAPACITY = Self.SYNTH_CAP
    comptime N_ENSEMBLE = Self.N_ENS
    comptime NUM_ELITES = Self.N_ELITES
    comptime REAL_RATIO_PCT = Self.REAL_RATIO
    comptime LOGVAR_MIN = Self.LOGVAR_MIN_
    comptime LOGVAR_MAX = Self.LOGVAR_MAX_

    comptime DEF_ACTOR_LR = Scalar[DT](3e-4)
    comptime DEF_CRITIC_LR = Scalar[DT](3e-4)
    comptime DEF_ALPHA_LR = Scalar[DT](3e-4)
    comptime DEF_MODEL_LR = Scalar[DT](1e-3)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_ACTION_SCALE = Scalar[DT](1.0)
    comptime DEF_INIT_ALPHA = Scalar[DT](0.2)
    comptime DEF_TARGET_ENTROPY = Scalar[DT](-Float64(Self.ACT))
    comptime DEF_LEARNING_STARTS = 1_000


# ──────────────────────────────────────────────────────────────────────
# Generic factory — any MBPOConfigT → primitive agent, defaults applied.
# ──────────────────────────────────────────────────────────────────────


def agent_from_config[
    CONFIG: MBPOConfigT,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = CONFIG.DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = CONFIG.DEF_CRITIC_LR,
    alpha_lr: Scalar[DT] = CONFIG.DEF_ALPHA_LR,
    model_lr: Scalar[DT] = CONFIG.DEF_MODEL_LR,
    gamma: Scalar[DT] = CONFIG.DEF_GAMMA,
    tau: Scalar[DT] = CONFIG.DEF_TAU,
    action_scale: Scalar[DT] = CONFIG.DEF_ACTION_SCALE,
    init_alpha: Scalar[DT] = CONFIG.DEF_INIT_ALPHA,
    target_entropy: Scalar[DT] = CONFIG.DEF_TARGET_ENTROPY,
    learning_starts: Int = CONFIG.DEF_LEARNING_STARTS,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
    model_train_freq: Int = 250,
    dyn_epochs_per_round: Int = 4,
    rollout_length: Int = 1,
    num_rollouts_per_step: Int = 400,
    sac_updates_per_step: Int = 20,
    dyn_batch_size: Int = 256,
    dyn_max_epochs: Int = 40,
    dyn_weight_decay: Scalar[DT] = 5e-5,
    dyn_learnable_bounds: Bool = False,
    use_bf16: Bool = False,
) raises -> MBPOAgent[
    CONFIG.TARGET,
    CONFIG.ACTOR,
    CONFIG.CRITIC,
    CONFIG.DYN_NET,
    CONFIG.OBS_DIM,
    CONFIG.ACT_DIM,
    CONFIG.BATCH,
    CONFIG.REPLAY_CAPACITY,
    CONFIG.SYNTH_CAPACITY,
    CONFIG.N_ENSEMBLE,
    CONFIG.NUM_ELITES,
    CONFIG.REAL_RATIO_PCT,
    CONFIG.LOGVAR_MIN,
    CONFIG.LOGVAR_MAX,
]:
    """Build the primitive `MBPOAgent` from any `MBPOConfigT`. Every scalar
    + schedule knob defaults to the config / reference value but stays
    overridable."""
    return MBPOAgent[
        CONFIG.TARGET,
        CONFIG.ACTOR,
        CONFIG.CRITIC,
        CONFIG.DYN_NET,
        CONFIG.OBS_DIM,
        CONFIG.ACT_DIM,
        CONFIG.BATCH,
        CONFIG.REPLAY_CAPACITY,
        CONFIG.SYNTH_CAPACITY,
        CONFIG.N_ENSEMBLE,
        CONFIG.NUM_ELITES,
        CONFIG.REAL_RATIO_PCT,
        CONFIG.LOGVAR_MIN,
        CONFIG.LOGVAR_MAX,
    ](
        ctx=ctx,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha_lr=alpha_lr,
        model_lr=model_lr,
        gamma=gamma,
        tau=tau,
        action_scale=action_scale,
        init_alpha=init_alpha,
        target_entropy=target_entropy,
        learning_starts=learning_starts,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
        model_train_freq=model_train_freq,
        dyn_epochs_per_round=dyn_epochs_per_round,
        rollout_length=rollout_length,
        num_rollouts_per_step=num_rollouts_per_step,
        sac_updates_per_step=sac_updates_per_step,
        dyn_batch_size=dyn_batch_size,
        dyn_max_epochs=dyn_max_epochs,
        dyn_weight_decay=dyn_weight_decay,
        dyn_learnable_bounds=dyn_learnable_bounds,
        use_bf16=use_bf16,
    )


# ──────────────────────────────────────────────────────────────────────
# Capitalized preset — single function, `target` as a parameter.
# ──────────────────────────────────────────────────────────────────────


def MBPO[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int,
    REPLAY_CAP: Int, SYNTH_CAP: Int,
    N_ENS: Int = 7, N_ELITES: Int = 5,
    REAL_RATIO: Int = 50,
    HIDDEN: Int = 256, DYN_HIDDEN: Int = 200,
    LOGVAR_MIN: Float64 = -10.0, LOGVAR_MAX: Float64 = -5.0,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = MBPOConfig[target, OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP, N_ENS, N_ELITES, REAL_RATIO, HIDDEN, DYN_HIDDEN, LOGVAR_MIN, LOGVAR_MAX].DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = MBPOConfig[target, OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP, N_ENS, N_ELITES, REAL_RATIO, HIDDEN, DYN_HIDDEN, LOGVAR_MIN, LOGVAR_MAX].DEF_CRITIC_LR,
    alpha_lr: Scalar[DT] = MBPOConfig[target, OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP, N_ENS, N_ELITES, REAL_RATIO, HIDDEN, DYN_HIDDEN, LOGVAR_MIN, LOGVAR_MAX].DEF_ALPHA_LR,
    model_lr: Scalar[DT] = MBPOConfig[target, OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP, N_ENS, N_ELITES, REAL_RATIO, HIDDEN, DYN_HIDDEN, LOGVAR_MIN, LOGVAR_MAX].DEF_MODEL_LR,
    gamma: Scalar[DT] = MBPOConfig[target, OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP, N_ENS, N_ELITES, REAL_RATIO, HIDDEN, DYN_HIDDEN, LOGVAR_MIN, LOGVAR_MAX].DEF_GAMMA,
    tau: Scalar[DT] = MBPOConfig[target, OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP, N_ENS, N_ELITES, REAL_RATIO, HIDDEN, DYN_HIDDEN, LOGVAR_MIN, LOGVAR_MAX].DEF_TAU,
    action_scale: Scalar[DT] = MBPOConfig[target, OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP, N_ENS, N_ELITES, REAL_RATIO, HIDDEN, DYN_HIDDEN, LOGVAR_MIN, LOGVAR_MAX].DEF_ACTION_SCALE,
    init_alpha: Scalar[DT] = MBPOConfig[target, OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP, N_ENS, N_ELITES, REAL_RATIO, HIDDEN, DYN_HIDDEN, LOGVAR_MIN, LOGVAR_MAX].DEF_INIT_ALPHA,
    target_entropy: Scalar[DT] = MBPOConfig[target, OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP, N_ENS, N_ELITES, REAL_RATIO, HIDDEN, DYN_HIDDEN, LOGVAR_MIN, LOGVAR_MAX].DEF_TARGET_ENTROPY,
    learning_starts: Int = MBPOConfig[target, OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP, N_ENS, N_ELITES, REAL_RATIO, HIDDEN, DYN_HIDDEN, LOGVAR_MIN, LOGVAR_MAX].DEF_LEARNING_STARTS,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
    model_train_freq: Int = 250,
    dyn_epochs_per_round: Int = 4,
    rollout_length: Int = 1,
    num_rollouts_per_step: Int = 400,
    sac_updates_per_step: Int = 20,
    dyn_batch_size: Int = 256,
    dyn_max_epochs: Int = 40,
    dyn_weight_decay: Scalar[DT] = 5e-5,
    dyn_learnable_bounds: Bool = False,
    use_bf16: Bool = False,
) raises -> MBPOAgent[
    target,
    MBPOActorNet[OBS, ACT, HIDDEN],
    MBPOCriticNet[OBS, ACT, HIDDEN],
    MBPODynNet[OBS, ACT, DYN_HIDDEN],
    OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP,
    N_ENS, N_ELITES, REAL_RATIO, LOGVAR_MIN, LOGVAR_MAX,
]:
    """MBPO with the canonical fused-`LinearReLU` SAC actor/critic + a
    4-layer Swish probabilistic dynamics ensemble. `target` selects
    cpu/gpu; ensemble/buffer ints and the logvar bounds default to the
    converging classic-control regime but stay overridable."""
    return agent_from_config[
        MBPOConfig[
            target, OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP,
            N_ENS, N_ELITES, REAL_RATIO, HIDDEN, DYN_HIDDEN,
            LOGVAR_MIN, LOGVAR_MAX,
        ]
    ](
        ctx=ctx,
        actor_lr=actor_lr, critic_lr=critic_lr, alpha_lr=alpha_lr,
        model_lr=model_lr, gamma=gamma, tau=tau,
        action_scale=action_scale, init_alpha=init_alpha,
        target_entropy=target_entropy, learning_starts=learning_starts,
        window_size=window_size, initial_episode_fill=initial_episode_fill,
        model_train_freq=model_train_freq,
        dyn_epochs_per_round=dyn_epochs_per_round,
        rollout_length=rollout_length,
        num_rollouts_per_step=num_rollouts_per_step,
        sac_updates_per_step=sac_updates_per_step,
        dyn_batch_size=dyn_batch_size, dyn_max_epochs=dyn_max_epochs,
        dyn_weight_decay=dyn_weight_decay,
        dyn_learnable_bounds=dyn_learnable_bounds, use_bf16=use_bf16,
    )
