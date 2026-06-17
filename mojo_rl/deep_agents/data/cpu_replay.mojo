"""CPUReplay[OBS, ACT, CAP] — circular CPU replay buffer.

Phase 8.1. Stores transitions `(s, a, r, s', done)` in five flat
allocations; `add` writes at `pos` and wraps; `sample(n, ...)` does
uniform-with-replacement sampling into caller-provided output buffers.

OBS / ACT / CAP are comptime so the per-slot strides are constants
the compiler can fold. `done` is stored as a Scalar[DT] (1.0 / 0.0)
rather than Bool — matches the `nonterm` arithmetic SAC's critic-target
computation does anyway.

Conforms to `ReplayBuffer` (the unifying trait over CPU/GPU storage):
the `make` / `add` / `sample_into` / `count` members are the trait
surface; the legacy `new` / `sample` methods are retained for callers
that pre-date the trait. `ctx` args are ignored (CPU-only).

`OBS` and `ACT` are dimensions (not buffer sizes), so e.g.
`CPUReplay[3, 1, 50000]` for Pendulum.
"""

from std.memory import alloc
from std.random import random_float64
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from ..training.replay_buffer import ReplayBuffer
from ..training.trainer_block import TrainerState


@fieldwise_init
struct CPUReplay[OBS_: Int, ACT_: Int, CAP_: Int](ReplayBuffer):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime CAP = Self.CAP_

    var obs: UnsafePointer[Scalar[DT], MutUntrackedOrigin]
    var act: UnsafePointer[Scalar[DT], MutUntrackedOrigin]
    var rew: UnsafePointer[Scalar[DT], MutUntrackedOrigin]
    var nxt: UnsafePointer[Scalar[DT], MutUntrackedOrigin]
    var dne: UnsafePointer[Scalar[DT], MutUntrackedOrigin]
    var size: Int
    var pos: Int

    @staticmethod
    def new() -> Self:
        return Self(
            obs=alloc[Scalar[DT]](Self.CAP * Self.OBS),
            act=alloc[Scalar[DT]](Self.CAP * Self.ACT),
            rew=alloc[Scalar[DT]](Self.CAP),
            nxt=alloc[Scalar[DT]](Self.CAP * Self.OBS),
            dne=alloc[Scalar[DT]](Self.CAP),
            size=0,
            pos=0,
        )

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        batch_capacity: Int = 4096,
    ) raises -> Self:
        # CPU-only; ctx / batch_capacity ignored.
        return Self.new()

    def add(
        mut self,
        ref s: List[Scalar[DT]],
        ref a: List[Scalar[DT]],
        r: Scalar[DT],
        ref sp: List[Scalar[DT]],
        d: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ):
        # ctx ignored (CPU-only).
        var p = self.pos
        for i in range(Self.OBS):
            self.obs[p * Self.OBS + i] = s[i]
            self.nxt[p * Self.OBS + i] = sp[i]
        for j in range(Self.ACT):
            self.act[p * Self.ACT + j] = a[j]
        self.rew[p] = r
        self.dne[p] = d
        self.pos = (self.pos + 1) % Self.CAP
        if self.size < Self.CAP:
            self.size += 1

    def sample(
        mut self,
        n: Int,
        s_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        a_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        r_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        sp_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        d_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ):
        """Uniform random sampling with replacement, n items."""
        for k in range(n):
            var idx = Int(random_float64() * Float64(self.size))
            if idx >= self.size:
                idx = self.size - 1
            for i in range(Self.OBS):
                s_out[k * Self.OBS + i] = self.obs[idx * Self.OBS + i]
                sp_out[k * Self.OBS + i] = self.nxt[idx * Self.OBS + i]
            for j in range(Self.ACT):
                a_out[k * Self.ACT + j] = self.act[idx * Self.ACT + j]
            r_out[k] = self.rew[idx]
            d_out[k] = self.dne[idx]

    def sample_into[
        BATCH: Int
    ](mut self, mut state: TrainerState[Self.OBS, Self.ACT, BATCH],) raises:
        """Trait-surface sampling: write a uniform minibatch into the
        host mirrors of `state.mb_*`."""
        self.sample(
            BATCH,
            state.mb_s.cpu_ptr(),
            state.mb_a.cpu_ptr(),
            state.mb_r.cpu_ptr(),
            state.mb_sp.cpu_ptr(),
            state.mb_d.cpu_ptr(),
        )

    def count(self) -> Int:
        return self.size
