"""J.1.g-redesign-v2 — ref-based TrainerBlocks (no bind, no graph).

Each block is a small (often stateless) struct that owns its inner
LossBlock if any, and exposes a `step[target]` method that takes
`mut state: TrainerState` plus the trainer fields it needs as `ref` /
`mut` arguments. The trainer's `train_step` body IS the pipeline.

Bit-identical to blocks/* (TrainerGraph-based), validated SAC seed=42
30k Pendulum → -169.04118.
"""

from .sample_block          import SampleBlock
from .uniform_sample_cpu_step import UniformSampleCpuStep
from .uniform_sample_gpu_step import UniformSampleGpuStep
from .per_sample_gpu_step     import PerSampleGpuStep
from .target_y_step          import TargetYStep
from .twin_critic_step       import TwinCriticStep
from .sac_actor_step         import SACActorStep
from .alpha_update_step      import AlphaUpdateStep
from .polyak_step            import PolyakStep
from .single_critic_step     import SingleCriticStep
from .ddpg_target_y_step     import DDPGTargetYStep
from .td3_target_y_step      import TD3TargetYStep
from .ddpg_actor_step        import DDPGActorStep
from .ddpg_polyak_step       import DDPGPolyakStep
from .td3_delayed_actor_polyak_step import TD3DelayedActorPolyakStep
from .dual_sample_cpu_step    import DualSampleCpuStep
