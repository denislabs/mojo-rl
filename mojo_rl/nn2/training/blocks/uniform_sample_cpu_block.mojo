"""UniformSampleCpuBlock — owns CPUReplay, samples a uniform minibatch.

Per J.1.a Q1(b), the sample block OWNS the replay buffer. Trainer's
`record(...)` delegates to `self.blocks[0].add(...)`. The block sets
`state.did_step = False` when the buffer is under-filled or before
learning_starts."""

from ...data.cpu_replay import CPUReplay
from ...constants import DT
from ..trainer_block import TrainerBlock, TrainerState


struct UniformSampleCpuBlock[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    CAP: Int,
](TrainerBlock):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_

    var buf: CPUReplay[Self.OBS, Self.ACT, Self.CAP]
    var learning_starts: Int

    def __init__(out self):
        self.buf = CPUReplay[Self.OBS, Self.ACT, Self.CAP](
            obs=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            act=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            rew=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            nxt=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            dne=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
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

    def step_via[target: StaticString = "cpu"](
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
