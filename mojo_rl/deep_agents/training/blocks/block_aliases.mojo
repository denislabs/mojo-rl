"""Back-compat aliases for the retired CPU + GPU concrete sample blocks.

Steps 4 & 5 collapsed the eight parallel `…CpuStep` / `…GpuStep` structs
into the two backend-generic blocks:

  * `ReplaySampleStep[R, BATCH]`        — uniform OR PER, single-env
  * `NStepSampleStep[N, R, BATCH]`      — host n-step decorator over R

The backend `R` (a `ReplayBuffer`) now decides CPU vs GPU and uniform vs
PER. The device-batch hooks (`add_batch` / `configure_ere`) are
`ReplayBuffer` trait methods, and `GPUNStepBuffer.store_into` is generic
over `ReplayBuffer`, so the generic blocks forward the GPU multi-env path
too — no GPU-specific block needed.

The old names survive as comptime aliases so every existing call site
(SAC / DDPG / TD3 / DQN / C51 trainers + ~30 tests + examples) keeps
compiling unchanged.
"""

from .replay_sample_step import ReplaySampleStep
from .n_step_sample_step import NStepSampleStep
from ...data.cpu_replay import CPUReplay
from ...data.cpu_per_replay import CPUPrioritizedReplay
from ...data.gpu_replay import GPUReplay
from ...data.per_replay import GPUPrioritizedReplay


# ── CPU (step 4) ──────────────────────────────────────────────────────

comptime UniformSampleCpuStep[
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int
] = ReplaySampleStep[CPUReplay[OBS, ACT, CAP], BATCH]

comptime PerSampleCpuStep[
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int
] = ReplaySampleStep[CPUPrioritizedReplay[OBS, ACT, CAP], BATCH]

comptime NStepSampleCpuStep[
    N: Int, OBS: Int, ACT: Int, BATCH: Int, CAP: Int
] = NStepSampleStep[N, CPUReplay[OBS, ACT, CAP], BATCH]

comptime NStepPerSampleCpuStep[
    N: Int, OBS: Int, ACT: Int, BATCH: Int, CAP: Int
] = NStepSampleStep[N, CPUPrioritizedReplay[OBS, ACT, CAP], BATCH]


# ── GPU (step 5) ──────────────────────────────────────────────────────

comptime UniformSampleGpuStep[
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int
] = ReplaySampleStep[GPUReplay[OBS, ACT, CAP], BATCH]

comptime PerSampleGpuStep[
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int
] = ReplaySampleStep[GPUPrioritizedReplay[OBS, ACT, CAP], BATCH]

comptime NStepSampleGpuStep[
    N: Int, OBS: Int, ACT: Int, BATCH: Int, CAP: Int
] = NStepSampleStep[N, GPUReplay[OBS, ACT, CAP], BATCH]

comptime NStepPerSampleGpuStep[
    N: Int, OBS: Int, ACT: Int, BATCH: Int, CAP: Int
] = NStepSampleStep[N, GPUPrioritizedReplay[OBS, ACT, CAP], BATCH]
