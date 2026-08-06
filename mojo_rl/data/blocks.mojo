# +--------------------------------------------------------------------------+ #
# | Sample-block aliases over the data layer
# +--------------------------------------------------------------------------+ #
"""Drop-in replacements for `deep_agents.training.blocks`' CPU aliases.

Migrating an algorithm is repointing ONE alias at a call site. Nothing inside
SAC/TD3/DQN/C51/DDPG changes: they already take `SAMPLE: SampleBlock` as a
compile-time parameter, and `ReplaySampleStep[R, BATCH]` is already generic
over any `ReplayBuffer`. That is why the migration is per-call-site rather
than per-algorithm-internals.

    # before
    from mojo_rl.deep_agents.training.blocks import UniformSampleCpuStep
    # after
    from mojo_rl.data.blocks import StoreUniformSampleCpuStep

Both produce bit-identical minibatches under the same seed. That was gated
against the legacy buffers until they were deleted; the surviving gates are
`tests/data/test_replay_seam.mojo` (CPU) and
`tests/data/test_replay_gpu_seam.mojo` (device uniform / ERE / PER / uint8 /
`add_batch`), over the goldens in `tests/data/test_sampler_golden.mojo`.

⚠ Note the argument order matches the legacy aliases exactly
(`OBS, ACT, BATCH, CAP`) so a swap is a one-token edit, not a re-ordering.
"""

from mojo_rl.deep_agents.training.blocks.replay_sample_step import (
    ReplaySampleStep,
)
from mojo_rl.nn.constants import DT
from .replay import StoreReplay
from .replay_gpu import StoreReplayGpu


comptime StoreUniformSampleCpuStep[
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int
] = ReplaySampleStep[StoreReplay[OBS, ACT, CAP, False], BATCH]
"""Replaces `UniformSampleCpuStep` (backed by `CPUReplay`)."""

comptime StorePerSampleCpuStep[
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int
] = ReplaySampleStep[StoreReplay[OBS, ACT, CAP, True], BATCH]
"""Replaces `PerSampleCpuStep` (backed by `CPUPrioritizedReplay`)."""


comptime StoreUniformSampleGpuStep[
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int
] = ReplaySampleStep[StoreReplayGpu[OBS, ACT, CAP], BATCH]
"""Replaces `UniformSampleGpuStep` (backed by `GPUReplay`). Supports ERE via
`configure_ere`, bit-identically."""

comptime StorePerSampleGpuStep[
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int, SDT: DType = DT
] = ReplaySampleStep[StoreReplayGpu[OBS, ACT, CAP, True, SDT], BATCH]
"""Replaces `PerSampleGpuStep` (backed by `GPUPrioritizedReplay`)."""
