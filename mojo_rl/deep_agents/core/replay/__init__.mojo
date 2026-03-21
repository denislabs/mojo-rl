"""Deep RL replay buffer package.

Provides replay buffers for experience replay in deep RL algorithms.

Buffer types:
- ReplayBuffer: Stack-allocated, fast but limited by stack size.
  Best for small observation spaces (< 10D) with moderate buffer sizes (< 50k).

- HeapReplayBuffer: Heap-allocated, supports large buffers.
  Use for large observation spaces (24D+) or buffer sizes > 50k.

- PrioritizedReplayBuffer: Stack-allocated with priority sampling.
"""

from .replay_buffer import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    HeapReplayBuffer,
    ReplayBufferTrait,
)
from .sequence_replay_buffer import SequenceReplayBuffer
from .gpu_replay_buffer import GPUReplayBuffer
from .gpu_per_replay_buffer import GPUPrioritizedReplayBuffer
from .gpu_sequence_replay_buffer import GPUSequenceReplayBuffer
from .nstep_buffer import NStepBuffer, NStepTransition, GPUNStepBuffer
