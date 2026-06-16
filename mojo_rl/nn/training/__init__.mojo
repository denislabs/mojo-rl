"""Training-loop coordinators (general-purpose, non-RL)."""

from .trainer import Trainer
from .autoregressive_trainer import AutoregressiveTrainer
from .timer import Timer
from .augmenter import Augmenter, IdentityAugmenter, CIFAR10CropFlipAugmenter
from .lr_scheduler import (
    Scheduler,
    ConstantSchedule,
    LinearWarmupSchedule,
    CosineSchedule,
    WarmupCosineSchedule,
    StepSchedule,
)
