"""UniformSampleCpuStep — owns a CPUReplay, samples a uniform minibatch.

Trainer's `record(...)` delegates to `self.sample_blk.add(...)`. Sets
`state.did_step = False` when buffer is under-filled or before
learning_starts.
"""

from ...constants import DT
from ...data.cpu_replay import CPUReplay
from ..trainer_block import TrainerState


struct UniformSampleCpuStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, CAP: Int,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_

    var buf: CPUReplay[Self.OBS, Self.ACT, Self.CAP]
    var learning_starts: Int

    def __init__(out self):
        var null_p = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self.buf = CPUReplay[Self.OBS, Self.ACT, Self.CAP](
            obs=null_p, act=null_p, rew=null_p, nxt=null_p, dne=null_p,
            size=0, pos=0,
        )
        self.learning_starts = 0

    def setup(mut self, learning_starts: Int) raises:
        self.buf = CPUReplay[Self.OBS, Self.ACT, Self.CAP].new()
        self.learning_starts = learning_starts

    def add(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ):
        self.buf.add(obs, action, reward, next_obs, done)

    def step(
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        if state.step_idx < self.learning_starts:
            state.did_step = False
            return
        if self.buf.size < Self.BATCH:
            state.did_step = False
            return
        self.buf.sample(
            Self.BATCH,
            state.mb_s.cpu_ptr(),
            state.mb_a.cpu_ptr(),
            state.mb_r.cpu_ptr(),
            state.mb_sp.cpu_ptr(),
            state.mb_d.cpu_ptr(),
        )
