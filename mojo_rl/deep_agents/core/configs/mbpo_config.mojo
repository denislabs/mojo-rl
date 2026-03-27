"""MBPO (Model-Based Policy Optimization) configuration.

MBPO = SAC + probabilistic dynamics ensemble + Dyna-style data augmentation.
The policy learning side is identical to SAC. The dynamics model generates
synthetic rollouts to augment the real replay buffer, achieving ~10x better
sample efficiency.

Reference: Janner et al., "When to Trust Your Model: Model-Based Policy
Optimization" (NeurIPS 2019).
"""

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import (
    Model,
    Linear,
    LinearReLU,
    LinearTanh,
    LinearSwish,
    Sequential,
    Parallel,
)
from mojo_rl.nn.optimizer import Optimizer, Adam

from .offpolicy_config import OffPolicyConfig
from ..strategies.exploration import Explore, StochasticSample
from ..strategies.update_schedule import Schedule, DelayedActorOnly
from ..strategies.target_value import TargetValue, EntropicTwinQTarget
from ..strategies.target_action import TargetAction, ReparamTarget
from ..strategies.actor_loss import ActorLoss, AutodiffMaxEntLoss
from ..strategies.termination import TerminationFn, NeverTerminate


# =============================================================================
# MBPOConfig trait — extends OffPolicyConfig with dynamics model requirements
# =============================================================================


trait MBPOConfig(OffPolicyConfig):
    """Compile-time configuration for MBPO agents.

    Extends OffPolicyConfig with dynamics model architecture, optimizer,
    ensemble sizing, and termination function for model rollouts.
    """

    # Dynamics model
    comptime DynamicsModel: Model
    comptime DynOpt: Optimizer

    # Ensemble configuration
    comptime ENSEMBLE_SIZE: Int
    comptime ELITE_SIZE: Int

    # Synthetic buffer capacity
    comptime SYNTH_CAPACITY: Int

    # Termination function for model rollouts
    comptime TermFn: TerminationFn


# =============================================================================
# DefaultMBPOConfig — concrete config (SAC policy + Swish dynamics ensemble)
# =============================================================================


struct DefaultMBPOConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 256,
    CAP: Int = 100000,
    SYNTH_CAP: Int = 400000,
    BS: Int = 256,
    NUM_ENSEMBLE: Int = 7,
    NUM_ELITES: Int = 5,
    DYN_HIDDEN: Int = 200,
    actor_lr: Float64 = 0.0003,
    critic_lr: Float64 = 0.0003,
    model_lr: Float64 = 0.001,
    TFn: TerminationFn = NeverTerminate,
](MBPOConfig):
    """Default MBPO config: SAC policy + 4-layer Swish dynamics ensemble.

    The SAC side (actor, critics, strategies) is identical to SACConfig.
    Additional compile-time params configure the dynamics ensemble:
    - NUM_ENSEMBLE: Total networks in ensemble (default: 7)
    - NUM_ELITES: Networks selected by holdout loss (default: 5)
    - DYN_HIDDEN: Dynamics network hidden width (default: 200)
    - SYNTH_CAP: Synthetic replay buffer capacity (default: 400000)
    - TFn: Environment-specific termination check for model rollouts
    """

    comptime NAME: String = "MBPO"
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    # SAC actor: Parallel output [mean(ACTIONS), log_std(ACTIONS)]
    comptime ActorModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, Self.ACT],      # mean head
            LinearTanh[Self.HIDDEN, Self.ACT],   # log_std head (tanh-clamped)
        ],
    ]
    comptime CriticModel = Sequential[
        LinearReLU[Self.OBS + Self.ACT, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]
    comptime ActorOpt = Adam[Self.actor_lr]
    comptime CriticOpt = Adam[Self.critic_lr]

    comptime NUM_CRITICS: Int = 2
    comptime HAS_TARGET_ACTOR: Bool = False

    # SAC strategies (reused exactly)
    comptime Explore = StochasticSample
    comptime Schedule = DelayedActorOnly
    comptime TargetAction = ReparamTarget
    comptime TargetValue = EntropicTwinQTarget
    comptime ActorLoss = AutodiffMaxEntLoss[]

    # =========================================================================
    # Dynamics model configuration
    # =========================================================================

    # Dynamics model: (obs + act) -> 2 * (1 + obs)
    # Output = [mean(reward, delta_obs), logvar(reward, delta_obs)]
    comptime DYN_IN: Int = Self.OBS + Self.ACT
    comptime DYN_OUT: Int = 2 * (1 + Self.OBS)

    comptime DynamicsModel = Sequential[
        LinearSwish[Self.DYN_IN, Self.DYN_HIDDEN],
        LinearSwish[Self.DYN_HIDDEN, Self.DYN_HIDDEN],
        LinearSwish[Self.DYN_HIDDEN, Self.DYN_HIDDEN],
        LinearSwish[Self.DYN_HIDDEN, Self.DYN_HIDDEN],
        Linear[Self.DYN_HIDDEN, Self.DYN_OUT],
    ]
    comptime DynOpt = Adam[Self.model_lr]

    # Ensemble size constants
    comptime ENSEMBLE_SIZE: Int = Self.NUM_ENSEMBLE
    comptime ELITE_SIZE: Int = Self.NUM_ELITES
    comptime SYNTH_CAPACITY: Int = Self.SYNTH_CAP

    # Termination function
    comptime TermFn = Self.TFn
