"""DualSampleCpuStep — MBPO mixed-batch sampler.

Owns BOTH the real replay buffer AND the synthetic replay buffer. Each
step samples REAL_BS transitions from real_buf into the first REAL_BS
rows of state.mb_*, and SYNTH_BS transitions from synth_buf into the
remaining rows.

Trainer's `record(...)` delegates to `real_add(...)`; MBPO's synthetic
rollout generator separately calls `synth_add(...)`.
"""

from mojo_rl.nn.constants import DT
from ...data.cpu_replay import CPUReplay
from ..trainer_block import TrainerState


struct DualSampleCpuStep[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    REAL_CAP: Int,
    SYNTH_CAP: Int,
    REAL_BS: Int,
    SYNTH_BS: Int,
](Defaultable & Movable & ImplicitlyDeletable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_

    var real_buf:  Optional[CPUReplay[Self.OBS, Self.ACT, Self.REAL_CAP]]
    var synth_buf: Optional[CPUReplay[Self.OBS, Self.ACT, Self.SYNTH_CAP]]
    var learning_starts: Int

    def __init__(out self):
        self.real_buf = None
        self.synth_buf = None
        self.learning_starts = 0

    def setup(mut self, learning_starts: Int) raises:
        self.real_buf = CPUReplay[Self.OBS, Self.ACT, Self.REAL_CAP].new()
        self.synth_buf = CPUReplay[
            Self.OBS, Self.ACT, Self.SYNTH_CAP
        ].new()
        self.learning_starts = learning_starts

    def real_add(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ):
        self.real_buf.value().add(obs, action, reward, next_obs, done)

    def synth_add(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ):
        self.synth_buf.value().add(obs, action, reward, next_obs, done)

    # Uniform readiness accessors (mirror DualSampleGpuStep) so the
    # trainer can gate on `real_count()`/`synth_count()` regardless of
    # the backend.
    def real_count(self) -> Int:
        if not self.real_buf:
            return 0
        return self.real_buf.value().size

    def synth_count(self) -> Int:
        if not self.synth_buf:
            return 0
        return self.synth_buf.value().size

    def step(
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        comptime assert (
            Self.REAL_BS + Self.SYNTH_BS == Self.BATCH
        ), "DualSampleCpuStep: REAL_BS + SYNTH_BS must equal BATCH"
        if state.step_idx < self.learning_starts:
            state.did_step = False
            return
        if self.real_buf.value().size < Self.REAL_BS:
            state.did_step = False
            return
        if self.synth_buf.value().size < Self.SYNTH_BS:
            state.did_step = False
            return

        # Real partition: rows [0, REAL_BS).
        self.real_buf.value().sample(
            Self.REAL_BS,
            state.mb_s.data,
            state.mb_a.data,
            state.mb_r.data,
            state.mb_sp.data,
            state.mb_d.data,
        )
        # Synth partition: rows [REAL_BS, BATCH) via `row_offset`.
        self.synth_buf.value().sample(
            Self.SYNTH_BS,
            state.mb_s.data,
            state.mb_a.data,
            state.mb_r.data,
            state.mb_sp.data,
            state.mb_d.data,
            row_offset=Self.REAL_BS,
        )
