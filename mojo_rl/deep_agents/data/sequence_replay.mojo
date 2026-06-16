"""SequenceReplay[OBS, ACT, CAP] — circular CPU sequence-replay buffer.

Used by DreamerV3 / TD-MPC2 world-model training: stores per-step
transitions in a flat ring, samples contiguous length-T windows.

Storage (4 flat allocations, each holding CAP slots):
  obs  [CAP, OBS]
  act  [CAP, ACT]
  rew  [CAP]
  dne  [CAP]        — 1.0 if the transition's `done` flag was set
                       (treated as truncation/termination — caller is
                        responsible for the semantics).

Record API (per-env-step):
  buf.record(obs_ptr, act_ptr, r, d)

`obs_ptr` is the *current* obs, not the next obs — the next-obs of a
record is implicitly the obs of the next record. For a window of T+1
observation frames + T transitions, the sample writer pulls indices
s, s+1, ..., s+T from `obs`, and indices s, s+1, ..., s+T-1 from
`act`/`rew`/`dne`.

Sample API:
  buf.sample_batch[B, T](mut obs_out[B, T+1, OBS], mut act_out[B, T, ACT],
                          mut rew_out[B, T], mut dne_out[B, T])

`B` and `T` are comptime so the per-slot strides fold. When `size < CAP`
the buffer is non-wrapping; once full, the oldest element is at
`(pos + 1) mod CAP`. We pick a random logical start `s ∈ [0, size - T - 1]`
and resolve to physical indices via `(_origin + s + k) mod CAP`.
"""

from std.memory import alloc
from std.random import random_float64
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from .sequence_replay_buffer import SequenceReplayBuffer


@fieldwise_init
struct SequenceReplay[OBS_: Int, ACT_: Int, CAP_: Int](SequenceReplayBuffer):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime CAP = Self.CAP_

    var obs: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var act: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var rew: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var dne: UnsafePointer[Scalar[DT], MutAnyOrigin]
    # Per-transition task id (DT-encoded), [CAP]. Written only by the
    # multi-task `record_task` path; the single-task `record` never touches it
    # (so the single-task RNG/compute stream is byte-identical). Allocated
    # always — a tiny [CAP] buffer that costs nothing when unused.
    var task: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var size: Int
    var pos: Int

    @staticmethod
    def new() -> Self:
        return Self(
            obs=alloc[Scalar[DT]](Self.CAP * Self.OBS),
            act=alloc[Scalar[DT]](Self.CAP * Self.ACT),
            rew=alloc[Scalar[DT]](Self.CAP),
            dne=alloc[Scalar[DT]](Self.CAP),
            task=alloc[Scalar[DT]](Self.CAP),
            size=0,
            pos=0,
        )

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Trait factory. CPU backend — `ctx` is ignored."""
        comptime assert target == "cpu", (
            "SequenceReplay is the CPU backend; use GPUSequenceReplay for"
            " target == \"gpu\""
        )
        return Self.new()

    def count(self) -> Int:
        return self.size

    def record(
        mut self,
        s: UnsafePointer[Scalar[DT], MutAnyOrigin],
        a: UnsafePointer[Scalar[DT], MutAnyOrigin],
        r: Scalar[DT],
        d: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ):
        var p = self.pos
        for i in range(Self.OBS):
            self.obs[p * Self.OBS + i] = s[i]
        for j in range(Self.ACT):
            self.act[p * Self.ACT + j] = a[j]
        self.rew[p] = r
        self.dne[p] = d
        self.pos = (self.pos + 1) % Self.CAP
        if self.size < Self.CAP:
            self.size += 1

    def record_task(
        mut self,
        s: UnsafePointer[Scalar[DT], MutAnyOrigin],
        a: UnsafePointer[Scalar[DT], MutAnyOrigin],
        r: Scalar[DT],
        d: Scalar[DT],
        task_id: Int,
    ):
        """Multi-task variant of `record`: stores the same transition AND the
        window's `task_id` (one env per window → constant task across the
        window). Additive — not part of the `SequenceReplayBuffer` trait; the
        single-task `record` is left untouched so its path is bit-identical."""
        var p = self.pos
        for i in range(Self.OBS):
            self.obs[p * Self.OBS + i] = s[i]
        for j in range(Self.ACT):
            self.act[p * Self.ACT + j] = a[j]
        self.rew[p] = r
        self.dne[p] = d
        self.task[p] = Scalar[DT](task_id)
        self.pos = (self.pos + 1) % Self.CAP
        if self.size < Self.CAP:
            self.size += 1

    def can_sample[T: Int](self) -> Bool:
        """Need at least T+1 elements to extract a length-T window
        (T+1 obs frames + T transitions)."""
        return self.size >= T + 1

    def _origin(self) -> Int:
        """Physical index of the oldest valid slot. When the buffer is
        not yet full the data sits at [0, size); once full it's the
        ring starting at (pos) — `pos` is where the NEXT write goes, so
        the oldest slot is at `pos` itself once we wrap."""
        if self.size < Self.CAP:
            return 0
        return self.pos

    def sample_batch[
        B: Int, T: Int,
    ](
        mut self,
        obs_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        rew_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        dne_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Draw `B` random length-`T` windows.

        Output strides:
          obs_out  shape [B, T+1, OBS] flat
          act_out  shape [B, T, ACT]   flat
          rew_out  shape [B, T]
          dne_out  shape [B, T]
        """
        if self.size < T + 1:
            raise "SequenceReplay.sample_batch: not enough data to sample a length-T window"

        # Valid logical-start range: [0, size - T - 1]   (inclusive)
        var n_valid = self.size - T  # number of valid start indices
        var origin = self._origin()

        for b in range(B):
            var s = Int(random_float64() * Float64(n_valid))
            if s >= n_valid:
                s = n_valid - 1

            # Copy T+1 observation frames.
            for k in range(T + 1):
                var phys = (origin + s + k) % Self.CAP
                var src = phys * Self.OBS
                var dst = b * (T + 1) * Self.OBS + k * Self.OBS
                for i in range(Self.OBS):
                    obs_out[dst + i] = self.obs[src + i]

            # Copy T action/reward/done frames.
            for k in range(T):
                var phys = (origin + s + k) % Self.CAP
                var src_a = phys * Self.ACT
                var dst_a = b * T * Self.ACT + k * Self.ACT
                for j in range(Self.ACT):
                    act_out[dst_a + j] = self.act[src_a + j]
                rew_out[b * T + k] = self.rew[phys]
                dne_out[b * T + k] = self.dne[phys]

    def sample_batch_task[
        B: Int, T: Int,
    ](
        mut self,
        obs_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        rew_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        dne_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        task_out: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B] one task/window
    ) raises:
        """Multi-task variant of `sample_batch`: identical window sampling, plus
        one `task_id` per window written to `task_out[b]` (read at the window's
        start frame — one env per window so the task is constant across it).
        Additive; not part of the trait."""
        if self.size < T + 1:
            raise "SequenceReplay.sample_batch_task: not enough data to sample a length-T window"

        var n_valid = self.size - T
        var origin = self._origin()

        for b in range(B):
            var s = Int(random_float64() * Float64(n_valid))
            if s >= n_valid:
                s = n_valid - 1

            for k in range(T + 1):
                var phys = (origin + s + k) % Self.CAP
                var src = phys * Self.OBS
                var dst = b * (T + 1) * Self.OBS + k * Self.OBS
                for i in range(Self.OBS):
                    obs_out[dst + i] = self.obs[src + i]

            for k in range(T):
                var phys = (origin + s + k) % Self.CAP
                var src_a = phys * Self.ACT
                var dst_a = b * T * Self.ACT + k * Self.ACT
                for j in range(Self.ACT):
                    act_out[dst_a + j] = self.act[src_a + j]
                rew_out[b * T + k] = self.rew[phys]
                dne_out[b * T + k] = self.dne[phys]

            # Window task = task of the window-start frame.
            var phys0 = (origin + s) % Self.CAP
            task_out[b] = self.task[phys0]
