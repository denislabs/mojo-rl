"""J.1.g-redesign-v2 — shared (agent-agnostic) TrainerBlocks.

Each block is a small (often stateless) struct that owns its inner
LossBlock if any, and exposes a `step[target]` method that takes
`mut state: TrainerState` plus the trainer fields it needs as `ref` /
`mut` arguments. The trainer's `train_step` body IS the pipeline.

This package now contains ONLY the shared (cross-agent) blocks. The
agent-specific blocks (SAC actor step / DDPG target y / TD3 delayed
actor polyak / PPO blocks / ...) live under each agent's own
`<agent>/blocks/` package.
"""

from .sample_block          import SampleBlock
# Backend-generic sample blocks (the canonical implementations).
from .replay_sample_step     import ReplaySampleStep
from .n_step_sample_step      import NStepSampleStep
# CPU + GPU sample blocks — comptime aliases over the generic blocks
# (steps 4 & 5). The backend `R` selects CPU/GPU and uniform/PER.
from .block_aliases import (
    UniformSampleCpuStep,
    PerSampleCpuStep,
    NStepSampleCpuStep,
    NStepPerSampleCpuStep,
    UniformSampleGpuStep,
    PerSampleGpuStep,
    NStepSampleGpuStep,
    NStepPerSampleGpuStep,
)
from .twin_critic_step       import TwinCriticStep
from .polyak_step            import PolyakStep, SinglePolyakStep
from .single_critic_step     import SingleCriticStep
from .dual_sample_cpu_step    import DualSampleCpuStep
