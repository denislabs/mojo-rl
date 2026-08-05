"""Sample-block aliases. **Now backed by `mojo_rl.data`** (2026-08-05, 4d).

Repointing these eight aliases migrated ~40 call sites in one edit, because
every consumer imports the ALIAS, not the buffer type. Safe to do wholesale
only because all five legacy policies (CPU uniform/PER, GPU uniform/ERE/PER)
are gated bit-identical — see `tests/data/test_replay*_parity.mojo`.

⚠ These 4-arg aliases never exposed `OBS_STORE_DT`, so the uint8-obs pixel
replay path does NOT go through here; it names the buffer types directly in
`deep_agents/*/config.mojo` and is migrated separately.

Original note follows.

Back-compat aliases for the retired CPU + GPU concrete sample blocks.

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
from mojo_rl.data.replay import StoreReplay
from mojo_rl.data.replay_gpu import StoreReplayGpu


# ── CPU (step 4) ──────────────────────────────────────────────────────

comptime UniformSampleCpuStep[
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int
] = ReplaySampleStep[StoreReplay[OBS, ACT, CAP, False], BATCH]

comptime PerSampleCpuStep[
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int
] = ReplaySampleStep[StoreReplay[OBS, ACT, CAP, True], BATCH]

comptime NStepSampleCpuStep[
    N: Int, OBS: Int, ACT: Int, BATCH: Int, CAP: Int
] = NStepSampleStep[N, StoreReplay[OBS, ACT, CAP, False], BATCH]

comptime NStepPerSampleCpuStep[
    N: Int, OBS: Int, ACT: Int, BATCH: Int, CAP: Int
] = NStepSampleStep[N, StoreReplay[OBS, ACT, CAP, True], BATCH]


# ── GPU (step 5) ──────────────────────────────────────────────────────

comptime UniformSampleGpuStep[
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int
] = ReplaySampleStep[StoreReplayGpu[OBS, ACT, CAP, False], BATCH]

comptime PerSampleGpuStep[
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int
] = ReplaySampleStep[StoreReplayGpu[OBS, ACT, CAP, True], BATCH]

comptime NStepSampleGpuStep[
    N: Int, OBS: Int, ACT: Int, BATCH: Int, CAP: Int
] = NStepSampleStep[N, StoreReplayGpu[OBS, ACT, CAP, False], BATCH]

comptime NStepPerSampleGpuStep[
    N: Int, OBS: Int, ACT: Int, BATCH: Int, CAP: Int
] = NStepSampleStep[N, StoreReplayGpu[OBS, ACT, CAP, True], BATCH]
