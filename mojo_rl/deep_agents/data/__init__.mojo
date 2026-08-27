"""Replay buffers.

⚠ The transition buffers moved to `mojo_rl.data` (2026-08-05, 4d): `CPUReplay`,
`GPUReplay`, `CPUPrioritizedReplay` and `GPUPrioritizedReplay` are gone,
replaced by `StoreReplay` / `StoreReplayGpu` — one struct per backend with
`PRIORITIZED` as a comptime flag, because the sum-tree is a sampler rather than
a storage subclass.

    from mojo_rl.data.replay import StoreReplay
    from mojo_rl.data.replay_gpu import StoreReplayGpu

`AnyReplay` / `AnyPerReplay` SURVIVE and now wrap those: Mojo still cannot
select a field type by target, so the carry-both shim is still required.

What remains here is the sequence-replay hierarchy (a separate trait, not yet
migrated) and the n-step ACCUMULATORS — which were never samplers.
"""

from .n_step_replay import NStepTransition, NStepBuffer, GPUNStepBuffer
from .sequence_replay_buffer import SequenceReplayBuffer
from .sequence_replay import SequenceReplay
from .gpu_sequence_replay import GPUSequenceReplay
from .any_replay import AnyReplay
from .any_per_replay import AnyPerReplay
