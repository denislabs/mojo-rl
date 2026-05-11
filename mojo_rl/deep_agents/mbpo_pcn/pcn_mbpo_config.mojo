"""PCN-MBPO compile-time configuration — fork of `MBPOConfig`.

Differs from vanilla `MBPOConfig`:
- No `DynamicsModel` / `DynOpt`. PCN dynamics is described by the PCN
  hidden width + SGLD hyperparameters, not by a Sequential autodiff model.
- No NLL-bounds parameters. PCN is deterministic.
- New comptime knobs: `DYN_HIDDEN_DIM`, `T_INFER`, `LR_X`, `DYN_LR`,
  `DYN_BATCH`, `ROLLOUT_BATCH`.

The SAC side (`ActorModel`, `CriticModel`, strategies, etc.) matches
vanilla `MBPOConfig` exactly so that the forked `PCNMBPOAgent` can reuse
all SAC machinery unchanged.
"""

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import (
    Model,
    Linear,
    LinearReLU,
    LinearTanh,
    Sequential,
    Parallel,
    LayerNorm,
    ReLU,
)
from mojo_rl.nn.optimizer import Optimizer, Adam

from mojo_rl.deep_agents.core.configs.offpolicy_config import OffPolicyConfig
from mojo_rl.deep_agents.core.strategies.exploration import Explore, StochasticSample
from mojo_rl.deep_agents.core.strategies.update_schedule import Schedule, DelayedActorOnly
from mojo_rl.deep_agents.core.strategies.target_value import TargetValue, EntropicTwinQTarget
from mojo_rl.deep_agents.core.strategies.target_action import TargetAction, ReparamTarget
from mojo_rl.deep_agents.core.strategies.actor_loss import ActorLoss, AutodiffMaxEntLoss
from mojo_rl.deep_agents.core.strategies.termination import TerminationFn, NeverTerminate


# =============================================================================
# PCNMBPOConfig trait — extends OffPolicyConfig with PCN-dynamics requirements
# =============================================================================


trait PCNMBPOConfig(OffPolicyConfig):
    """Compile-time configuration for PCN-MBPO agents.

    Extends `OffPolicyConfig` (which carries the SAC actor/critic/strategies)
    with PCN-dynamics-specific knobs and ensemble sizing. No `DynamicsModel`
    or `DynOpt` (PCN doesn't use an autodiff Sequential).
    """

    # PCN dynamics architecture.
    comptime DYN_HIDDEN_DIM: Int

    # PCN training hyperparameters.
    comptime T_INFER: Int                  # SGLD inference iterations per minibatch.
    comptime LR_X: Float64                 # SGLD step size on latent z.
    comptime DYN_LR: Float64               # Adam learning rate on weights.
    comptime DYN_GRAD_CLIP_NORM: Float64

    # Ensemble sizing.
    comptime ENSEMBLE_SIZE: Int
    comptime ELITE_SIZE: Int

    # Batch sizes for dynamics (separate from SAC's BATCH).
    comptime DYN_BATCH: Int
    comptime ROLLOUT_BATCH: Int

    # Synthetic replay buffer capacity.
    comptime SYNTH_CAPACITY: Int

    # Termination function for CPU model rollouts (matches MBPO trait).
    comptime TermFn: TerminationFn


# =============================================================================
# DefaultPCNMBPOConfig — concrete config (SAC policy + PCN dynamics ensemble)
# =============================================================================


struct DefaultPCNMBPOConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 256,
    CAP: Int = 100000,
    SYNTH_CAP: Int = 400000,
    BS: Int = 256,
    NUM_ENSEMBLE: Int = 7,
    NUM_ELITES: Int = 5,
    DYN_HIDDEN: Int = 200,
    DYN_BATCH_SIZE: Int = 256,
    DYN_ROLLOUT_BATCH: Int = 400,
    actor_lr: Float64 = 0.0003,
    critic_lr: Float64 = 0.0003,
    model_lr: Float64 = 0.001,
    t_infer: Int = 10,
    lr_x: Float64 = 0.01,
    dyn_grad_clip: Float64 = 1.0,
    TFn: TerminationFn = NeverTerminate,
    action_scale: Float64 = 1.0,
](PCNMBPOConfig):
    """Default PCN-MBPO config: SAC policy + 2-layer PCN dynamics ensemble.

    SAC actor/critic mirror `DefaultMBPOConfig` exactly. Dynamics is the
    PCN ensemble (`PCDynamics` 2-layer chain, ensemble of NUM_ENSEMBLE
    members, top NUM_ELITES used for imagination).

    Defaults preserve `DefaultMBPOConfig`'s SAC settings; only the
    dynamics half changes.
    """

    comptime NAME: String = "PCN-MBPO"
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    # SAC actor: Parallel output [mean(ACTIONS), log_std(ACTIONS)].
    comptime ActorModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, Self.ACT],
            LinearTanh[Self.HIDDEN, Self.ACT],
        ],
    ]
    # Critic with LayerNorm (matches DefaultMBPOConfig).
    comptime CriticModel = Sequential[
        Linear[Self.OBS + Self.ACT, Self.HIDDEN],
        LayerNorm[Self.HIDDEN],
        ReLU[Self.HIDDEN],
        Linear[Self.HIDDEN, Self.HIDDEN],
        LayerNorm[Self.HIDDEN],
        ReLU[Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]
    comptime ActorOpt = Adam[Self.actor_lr]
    comptime CriticOpt = Adam[Self.critic_lr]

    comptime NUM_CRITICS: Int = 2
    comptime HAS_TARGET_ACTOR: Bool = False

    # SAC strategies — identical to DefaultMBPOConfig.
    comptime Explore = StochasticSample
    comptime Schedule = DelayedActorOnly
    comptime TargetAction = ReparamTarget
    comptime TargetValue = EntropicTwinQTarget
    comptime ActorLoss = AutodiffMaxEntLoss[action_scale=Self.action_scale]

    # =========================================================================
    # PCN dynamics configuration
    # =========================================================================

    comptime DYN_HIDDEN_DIM: Int = Self.DYN_HIDDEN
    comptime T_INFER: Int = Self.t_infer
    comptime LR_X: Float64 = Self.lr_x
    comptime DYN_LR: Float64 = Self.model_lr
    comptime DYN_GRAD_CLIP_NORM: Float64 = Self.dyn_grad_clip
    comptime ENSEMBLE_SIZE: Int = Self.NUM_ENSEMBLE
    comptime ELITE_SIZE: Int = Self.NUM_ELITES
    comptime DYN_BATCH: Int = Self.DYN_BATCH_SIZE
    comptime ROLLOUT_BATCH: Int = Self.DYN_ROLLOUT_BATCH
    comptime SYNTH_CAPACITY: Int = Self.SYNTH_CAP

    comptime TermFn = Self.TFn
