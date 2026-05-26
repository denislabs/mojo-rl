"""UniformSampleGpuStep — GPU variant of UniformSampleCpuStep.

Owns Optional[GPUReplay]. `setup(ctx, learning_starts)` allocates; `add`
pushes via the trainer-supplied ctx. `step` reads device-side state.mb_*
buffers and samples in-place.
"""

from std.gpu.host import DeviceContext

from ...constants import DT
from ...data.gpu_replay import GPUReplay
from ..trainer_block import TrainerState


struct UniformSampleGpuStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, CAP: Int,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_

    var buf: Optional[GPUReplay[Self.OBS, Self.ACT, Self.CAP]]
    var learning_starts: Int

    def __init__(out self):
        self.buf = None
        self.learning_starts = 0

    def setup(mut self, ctx: DeviceContext, learning_starts: Int) raises:
        self.buf = GPUReplay[Self.OBS, Self.ACT, Self.CAP].new(
            ctx, batch_capacity=Self.BATCH
        )
        self.learning_starts = learning_starts

    def add(
        mut self,
        ctx: DeviceContext,
        s: UnsafePointer[Scalar[DT], MutAnyOrigin],
        a: UnsafePointer[Scalar[DT], MutAnyOrigin],
        r: Scalar[DT],
        sp: UnsafePointer[Scalar[DT], MutAnyOrigin],
        d: Scalar[DT],
    ) raises:
        self.buf.value().add(ctx, s, a, r, sp, d)

    def step(
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        if state.step_idx < self.learning_starts:
            state.did_step = False
            return
        if self.buf.value().size < Self.BATCH:
            state.did_step = False
            return
        self.buf.value().sample[Self.BATCH](
            state.ctx.value(),
            state.mb_s.dev.value(),
            state.mb_a.dev.value(),
            state.mb_r.dev.value(),
            state.mb_sp.dev.value(),
            state.mb_d.dev.value(),
        )
