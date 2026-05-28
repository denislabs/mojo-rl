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
from .uniform_sample_cpu_step import UniformSampleCpuStep
from .uniform_sample_gpu_step import UniformSampleGpuStep
from .per_sample_gpu_step     import PerSampleGpuStep
from .twin_critic_step       import TwinCriticStep
from .polyak_step            import PolyakStep
from .single_critic_step     import SingleCriticStep
from .dual_sample_cpu_step    import DualSampleCpuStep
from .n_step_sample_cpu_step  import NStepSampleCpuStep
from .n_step_sample_gpu_step  import NStepSampleGpuStep
