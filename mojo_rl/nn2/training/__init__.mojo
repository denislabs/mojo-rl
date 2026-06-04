"""Training-loop coordinators (general-purpose, non-RL)."""

from .trainer import Trainer
from .timer import Timer
from .augmenter import Augmenter, IdentityAugmenter, CIFAR10CropFlipAugmenter
