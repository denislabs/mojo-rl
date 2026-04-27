"""REDQ-OFE configuration.

REDQ-OFE = REDQ + OFENet (Ota et al., "Can Increasing Input Dimensionality
Improve Deep RL?", ICML 2020). A DenseNet-style feature extractor is
trained alongside the RL agent via auxiliary next-state prediction loss.
Actor and critics consume OFE features (phi_s, phi_sa) as stop-gradient
inputs — OFE params are updated ONLY by the aux loss, matching the
paper's `teflon/policy/SAC.py:train_for_batch` which computes gradients
against critic.trainable_variables only.

Architecture (DenseNet variant with LayerNorm):
  - num_layers blocks in state branch, num_layers in action branch
  - per_unit = total_units / num_layers new features per block
  - block = Linear(per_unit) -> LayerNorm -> Swish -> concat(input, .)
    (LayerNorm replaces the paper's BN — see composites_ofenet.mojo
     docstring for the rationale.)

Configs from `references/OFENet-main/gins/`:
  - HalfCheetah, Hopper, Walker2d: total_units=240, num_layers=6 -> per_unit=40
  - Ant, Humanoid:                 total_units=240, num_layers=8 -> per_unit=30
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
from mojo_rl.nn.composites_ofenet import StateBranch6, StateBranch8, ActionBranch6, ActionBranch8

from mojo_rl.deep_agents.core.configs.offpolicy_config import OffPolicyConfig
from mojo_rl.deep_agents.core.strategies.exploration import Explore, StochasticSample
from mojo_rl.deep_agents.core.strategies.update_schedule import Schedule, DelayedActorOnly
from mojo_rl.deep_agents.core.strategies.target_value import TargetValue, EntropicTwinQTarget
from mojo_rl.deep_agents.core.strategies.target_action import TargetAction, ReparamTarget
from mojo_rl.deep_agents.core.strategies.actor_loss import ActorLoss, AutodiffMaxEntLoss

from mojo_rl.deep_agents.redq.config import (
    REDQ_TARGET_MIN,
    REDQ_TARGET_AVE,
    REDQ_TARGET_REM,
)


# =============================================================================
# REDQOFEConfig trait — extends REDQ's config with OFE-specific knobs
# =============================================================================


trait REDQOFEConfig(OffPolicyConfig):
    """Compile-time configuration for REDQ-OFE agents.

    Extends OffPolicyConfig (for REDQ-shaped actor/critic machinery) with
    OFENet knobs. The inherited `ActorModel` must accept phi_s (feature
    dim, not raw state dim); `CriticModel` must accept phi_sa.
    """

    # REDQ-specific knobs (mirrored from REDQConfig since we don't inherit)
    comptime NUM_ENSEMBLE: Int
    comptime NUM_MIN: Int
    comptime UTD_RATIO: Int
    comptime POLICY_DELAY: Int
    comptime Q_TARGET_MODE: Int

    # OFE-specific knobs
    comptime OFE_NUM_LAYERS: Int      # 6 or 8 (paper default)
    comptime OFE_PER_UNIT: Int        # total_units / num_layers
    comptime OFE_LR: Float64          # Aux Adam LR (paper: 3e-4)
    # Derived: phi_s_dim = OBS + OFE_NUM_LAYERS * OFE_PER_UNIT
    comptime PHI_S_DIM: Int
    # Derived: phi_sa_dim = PHI_S_DIM + ACT + OFE_NUM_LAYERS * OFE_PER_UNIT
    comptime PHI_SA_DIM: Int

    # OFE sub-networks (Model aliases from nn.composites_ofenet, bound to
    # this config's state_dim / action_dim / per_unit). Agents pick
    # 6-layer or 8-layer variants via the config.
    comptime OFEStateBranchModel: Model     # IN=obs_dim, OUT=phi_s_dim
    comptime OFEActionBranchModel: Model    # IN=phi_s_dim + action_dim, OUT=phi_sa_dim
    comptime OFEPredictorModel: Model       # IN=phi_sa_dim, OUT=obs_dim


# =============================================================================
# DefaultREDQOFEConfig6 — 6-layer variant (HalfCheetah / Hopper / Walker2d)
# =============================================================================


struct DefaultREDQOFEConfig6[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 256,
    CAP: Int = 1_000_000,
    BS: Int = 256,
    N_ENS: Int = 10,
    N_MIN: Int = 2,
    UTD: Int = 20,
    POL_DELAY: Int = 20,
    Q_MODE: Int = REDQ_TARGET_MIN,
    actor_lr: Float64 = 0.0003,
    critic_lr: Float64 = 0.0003,
    ofe_lr: Float64 = 0.0003,
    OFE_TOTAL_UNITS: Int = 240,
    action_scale: Float64 = 1.0,
](REDQOFEConfig):
    """6-layer OFE variant. Matches `references/OFENet-main/gins/HalfCheetah.gin`
    (total_units=240, num_layers=6 -> per_unit=40).
    """

    comptime NAME: String = "REDQ-OFE-6"
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    # OFE dims
    comptime OFE_NUM_LAYERS: Int = 6
    comptime OFE_PER_UNIT: Int = Self.OFE_TOTAL_UNITS // 6   # 40
    comptime PHI_S_DIM: Int = Self.OBS + 6 * Self.OFE_PER_UNIT
    comptime PHI_SA_DIM: Int = Self.PHI_S_DIM + Self.ACT + 6 * Self.OFE_PER_UNIT

    # OFE sub-networks
    comptime OFEStateBranchModel = StateBranch6[Self.OBS, Self.OFE_PER_UNIT]
    comptime OFEActionBranchModel = ActionBranch6[
        Self.PHI_S_DIM, Self.ACT, Self.OFE_PER_UNIT
    ]
    comptime OFEPredictorModel = Linear[Self.PHI_SA_DIM, Self.OBS]

    # Actor takes phi_s (feature-space input)
    comptime ActorModel = Sequential[
        LinearReLU[Self.PHI_S_DIM, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, Self.ACT],
            LinearTanh[Self.HIDDEN, Self.ACT],
        ],
    ]

    # Critic takes phi_sa (feature-space input)
    comptime CriticModel = Sequential[
        LinearReLU[Self.PHI_SA_DIM, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]

    comptime ActorOpt = Adam[Self.actor_lr]
    comptime CriticOpt = Adam[Self.critic_lr]

    comptime NUM_CRITICS: Int = Self.N_ENS
    comptime HAS_TARGET_ACTOR: Bool = False

    comptime Explore = StochasticSample
    comptime Schedule = DelayedActorOnly
    comptime TargetAction = ReparamTarget
    comptime TargetValue = EntropicTwinQTarget
    comptime ActorLoss = AutodiffMaxEntLoss[action_scale=Self.action_scale]

    comptime NUM_ENSEMBLE: Int = Self.N_ENS
    comptime NUM_MIN: Int = Self.N_MIN
    comptime UTD_RATIO: Int = Self.UTD
    comptime POLICY_DELAY: Int = Self.POL_DELAY
    comptime Q_TARGET_MODE: Int = Self.Q_MODE
    comptime OFE_LR: Float64 = Self.ofe_lr


# =============================================================================
# DefaultREDQOFEConfig8 — 8-layer variant (Ant / Humanoid)
# =============================================================================


struct DefaultREDQOFEConfig8[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 256,
    CAP: Int = 1_000_000,
    BS: Int = 256,
    N_ENS: Int = 10,
    N_MIN: Int = 2,
    UTD: Int = 20,
    POL_DELAY: Int = 20,
    Q_MODE: Int = REDQ_TARGET_MIN,
    actor_lr: Float64 = 0.0003,
    critic_lr: Float64 = 0.0003,
    ofe_lr: Float64 = 0.0003,
    OFE_TOTAL_UNITS: Int = 240,
    action_scale: Float64 = 1.0,
](REDQOFEConfig):
    """8-layer OFE variant. Matches `references/OFENet-main/gins/{Ant,Humanoid}.gin`
    (total_units=240, num_layers=8 -> per_unit=30).
    """

    comptime NAME: String = "REDQ-OFE-8"
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP

    comptime OFE_NUM_LAYERS: Int = 8
    comptime OFE_PER_UNIT: Int = Self.OFE_TOTAL_UNITS // 8   # 30
    comptime PHI_S_DIM: Int = Self.OBS + 8 * Self.OFE_PER_UNIT
    comptime PHI_SA_DIM: Int = Self.PHI_S_DIM + Self.ACT + 8 * Self.OFE_PER_UNIT

    comptime OFEStateBranchModel = StateBranch8[Self.OBS, Self.OFE_PER_UNIT]
    comptime OFEActionBranchModel = ActionBranch8[
        Self.PHI_S_DIM, Self.ACT, Self.OFE_PER_UNIT
    ]
    comptime OFEPredictorModel = Linear[Self.PHI_SA_DIM, Self.OBS]

    comptime ActorModel = Sequential[
        LinearReLU[Self.PHI_S_DIM, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, Self.ACT],
            LinearTanh[Self.HIDDEN, Self.ACT],
        ],
    ]

    comptime CriticModel = Sequential[
        LinearReLU[Self.PHI_SA_DIM, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]

    comptime ActorOpt = Adam[Self.actor_lr]
    comptime CriticOpt = Adam[Self.critic_lr]

    comptime NUM_CRITICS: Int = Self.N_ENS
    comptime HAS_TARGET_ACTOR: Bool = False

    comptime Explore = StochasticSample
    comptime Schedule = DelayedActorOnly
    comptime TargetAction = ReparamTarget
    comptime TargetValue = EntropicTwinQTarget
    comptime ActorLoss = AutodiffMaxEntLoss[action_scale=Self.action_scale]

    comptime NUM_ENSEMBLE: Int = Self.N_ENS
    comptime NUM_MIN: Int = Self.N_MIN
    comptime UTD_RATIO: Int = Self.UTD
    comptime POLICY_DELAY: Int = Self.POL_DELAY
    comptime Q_TARGET_MODE: Int = Self.Q_MODE
    comptime OFE_LR: Float64 = Self.ofe_lr
