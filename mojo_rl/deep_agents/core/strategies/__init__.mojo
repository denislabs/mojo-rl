"""Strategy traits and implementations for generic agents."""

from .exploration import Explore, GaussianNoise, StochasticSample
from .update_schedule import Schedule, EveryStep, DelayedAll, DelayedActorOnly
from .target_value import TargetValue, SingleQTarget, TwinQTarget, EntropicTwinQTarget
from .target_action import TargetAction, DeterministicTarget, SmoothedTarget, ReparamTarget
from .actor_loss import ActorLoss, DPGLoss, MaxEntLoss, AutodiffMaxEntLoss, AutodiffDPGLoss, AutodiffTD3Loss
from .policy_gradient import PolicyGradient, VanillaPG, ClippedSurrogate, AutodiffVanillaPG, AutodiffClippedSurrogate
from .epoch_schedule import EpochSchedule, SinglePass, MultiEpochMinibatch
from .q_target import QTarget, StandardQTarget, DoubleQTarget
from .q_output import QOutput, DirectQ, DuelingQ
from .q_gradient import QGradient, ManualQGradient, AutodiffQGradient
