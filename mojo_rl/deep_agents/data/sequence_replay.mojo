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

from std.random import random_float64
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from .sequence_replay_buffer import SequenceReplayBuffer


@fieldwise_init
struct SequenceReplay[OBS_: Int, ACT_: Int, CAP_: Int](SequenceReplayBuffer):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime CAP = Self.CAP_

    # Owning RAII `List` rings (host-indexed only). Replaces the raw `alloc`'d
    # `MutUntrackedOrigin` pointers, which — with no `__deinit__` and a trait that
    # is `Deinitable` — were never freed (a genuine leak). `List`
    # destruction frees them automatically.
    var obs: List[Scalar[DT]]
    var act: List[Scalar[DT]]
    var rew: List[Scalar[DT]]
    var dne: List[Scalar[DT]]
    # Per-frame `is_first` ring [CAP]: 1.0 if this slot's obs is the first frame
    # of an episode (carry must reset here). Tracked INTERNALLY via `pending_first`
    # so the WM can store a terminal observation (`record_terminal`) and still
    # mark the *following* reset frame as the episode start. DreamerV3 needs this
    # separate from `dne` so a genuine terminal obs gets cont=0 while the reset
    # frame that follows it gets cont=1 + a carry reset. Read only by the additive
    # `sample_batch_fst`; the trait `sample_batch` ignores it (other users unaffected).
    var fst: List[Scalar[DT]]
    # Per-transition task id (DT-encoded), [CAP]. Written only by the
    # multi-task `record_task` path; the single-task `record` never touches it
    # (so the single-task RNG/compute stream is byte-identical). Allocated
    # always — a tiny [CAP] buffer that costs nothing when unused.
    var task: List[Scalar[DT]]
    var size: Int
    var pos: Int
    # True ⇒ the next `record`ed frame begins a new episode (fst=1). Starts True
    # (first frame ever) and is re-armed after any `record` whose done flag is
    # set; `record_terminal` leaves it armed so the post-terminal reset frame
    # (not the stored terminal obs) is the one flagged first.
    var pending_first: Bool
    # Online queue (reference replay `online: True`): every `online_every`-th
    # appended frame enqueues the newest window's END slot (physical index);
    # `sample_batch_fst` pops these into batch rows before falling back to
    # uniform draws — every fresh window is trained on exactly once, promptly
    # (a recency-coverage guarantee, NOT a heavy freshness bias). 0 = off
    # (default; all existing consumers byte-identical).
    var online_every: Int
    var online_tick: Int
    var online_q: List[Int]  # FIFO of physical end-slot indices (small)
    # Number of INTERLEAVED env streams sharing this ring (1 = single-env, the
    # default and the byte-identical path). A batched collector records env 0,
    # 1, … N-1 back-to-back every iteration, so slot `p` and slot `p + N` are
    # CONSECUTIVE FRAMES OF THE SAME ENV and slot `p + 1` is a different env
    # entirely. With `env_stride = N` the samplers walk the ring in steps of N
    # from an env-aligned start, so a window is one env's trajectory.
    #
    # ⚠ Leaving this at 1 while recording N envs does NOT fail — it silently
    # produces windows that hop between envs every frame, i.e. a world model
    # trained on transitions that never happened. There is no way to detect
    # that from the loss curve, so the batched driver MUST call
    # `set_env_stride(N_ENVS)`.
    var env_stride: Int

    @staticmethod
    def new() -> Self:
        return Self(
            obs=List[Scalar[DT]](length=Self.CAP * Self.OBS, fill=Scalar[DT](0)),
            act=List[Scalar[DT]](length=Self.CAP * Self.ACT, fill=Scalar[DT](0)),
            rew=List[Scalar[DT]](length=Self.CAP, fill=Scalar[DT](0)),
            dne=List[Scalar[DT]](length=Self.CAP, fill=Scalar[DT](0)),
            fst=List[Scalar[DT]](length=Self.CAP, fill=Scalar[DT](0)),
            task=List[Scalar[DT]](length=Self.CAP, fill=Scalar[DT](0)),
            size=0,
            pos=0,
            pending_first=True,
            online_every=0,
            online_tick=0,
            online_q=List[Int](),
            env_stride=1,
        )

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Trait factory. CPU backend — `ctx` is ignored."""
        comptime assert target == "cpu", (
            "SequenceReplay is the CPU backend; use GPUSequenceReplay for"
            ' target == "gpu"'
        )
        return Self.new()

    def count(self) -> Int:
        return self.size

    def set_online(mut self, every: Int):
        """Enable the online queue: each `every`-th appended frame enqueues the
        freshest length-`every` window for guaranteed prompt sampling. Pass the
        training window length T. 0 disables (the default)."""
        self.online_every = every

    def _online_tick_append(mut self):
        """Called after every ring append. Enqueues the just-completed fresh
        window (its END slot, physical) once per `online_every` appends."""
        if self.online_every <= 0:
            return
        self.online_tick += 1
        if (
            self.online_tick >= self.online_every
            and self.size >= self.online_every + 1
        ):
            if len(self.online_q) >= 32:  # backlog cap: drop oldest
                _ = self.online_q.pop(0)
            self.online_q.append((self.pos - 1 + Self.CAP) % Self.CAP)
            self.online_tick = 0

    def _online_pop_start(mut self, T: Int) -> Int:
        """Pop the oldest queued fresh window; return its LOGICAL start index,
        or -1 if the queue is empty (or the window was evicted)."""
        var origin = self._origin()
        while len(self.online_q) > 0:
            var phys_end = self.online_q.pop(0)
            var end_logical = (phys_end - origin + Self.CAP) % Self.CAP
            var s = end_logical - T
            if s >= 0 and end_logical < self.size:
                return s
        return -1

    def record(
        mut self,
        s: Pointer[Scalar[DT], MutAnyOrigin],
        a: Pointer[Scalar[DT], MutAnyOrigin],
        r: Scalar[DT],
        d: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ):
        var p = self.pos
        for i in range(Self.OBS):
            self.obs[p * Self.OBS + i] = s[unsafe_offset=i]
        for j in range(Self.ACT):
            self.act[p * Self.ACT + j] = a[unsafe_offset=j]
        self.rew[p] = r
        self.dne[p] = d
        self.fst[p] = Scalar[DT](1.0) if self.pending_first else Scalar[DT](0.0)
        # A done frame ends the episode → the *next* recorded frame is first.
        self.pending_first = d >= Scalar[DT](0.5)
        self.pos = (self.pos + 1) % Self.CAP
        if self.size < Self.CAP:
            self.size += 1
        self._online_tick_append()

    def record_terminal(
        mut self,
        s: Pointer[Scalar[DT], MutAnyOrigin],
    ):
        """Store a genuine TERMINAL observation as its own frame (no outgoing
        transition: act=0, rew=0, dne=0). Called right after a `record(done=1)`
        so the window's obs frame *after* the terminal transition is the real
        terminal state (→ the cont head can learn `latent(terminal)→0`), instead
        of the next episode's reset obs. `pending_first` is left armed (set by the
        preceding done=1 record) so the FOLLOWING reset frame — not this terminal
        frame — is flagged `is_first`."""
        var p = self.pos
        for i in range(Self.OBS):
            self.obs[p * Self.OBS + i] = s[unsafe_offset=i]
        for j in range(Self.ACT):
            self.act[p * Self.ACT + j] = Scalar[DT](0.0)
        self.rew[p] = Scalar[DT](0.0)
        self.dne[p] = Scalar[DT](0.0)
        self.fst[p] = Scalar[DT](0.0)   # terminal frame is NOT an episode start
        self.pos = (self.pos + 1) % Self.CAP
        if self.size < Self.CAP:
            self.size += 1
        self._online_tick_append()

    def record_task(
        mut self,
        s: Pointer[Scalar[DT], MutAnyOrigin],
        a: Pointer[Scalar[DT], MutAnyOrigin],
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
            self.obs[p * Self.OBS + i] = s[unsafe_offset=i]
        for j in range(Self.ACT):
            self.act[p * Self.ACT + j] = a[unsafe_offset=j]
        self.rew[p] = r
        self.dne[p] = d
        self.fst[p] = Scalar[DT](1.0) if self.pending_first else Scalar[DT](0.0)
        self.pending_first = d >= Scalar[DT](0.5)
        self.task[p] = Scalar[DT](task_id)
        self.pos = (self.pos + 1) % Self.CAP
        if self.size < Self.CAP:
            self.size += 1

    def set_env_stride(mut self, n_envs: Int) raises:
        """Declare that `n_envs` env streams are interleaved in this ring.

        Call ONCE before a batched collector starts recording (and never
        mid-run — the frames already stored were laid down under the old
        stride). `CAP` must be a multiple of `n_envs`, otherwise the ring's
        wrap shifts every env's lane by `CAP % n_envs` and the strided walk
        starts crossing streams at exactly the point the buffer fills — a
        corruption that only appears after CAP steps."""
        if n_envs < 1:
            raise Error("SequenceReplay.set_env_stride: n_envs must be >= 1")
        if self.size > 0 and n_envs != self.env_stride:
            # Re-striding a ring that already holds data reinterprets every
            # stored frame as belonging to a different env. Raising here is
            # what turns "single-env `train` after batched `train_batched` on
            # the same agent" from silent corruption into an error.
            raise Error(
                "SequenceReplay.set_env_stride: the buffer already holds"
                " frames recorded at a different stride — a single replay"
                " cannot mix single-env and batched collection"
            )
        if Self.CAP % n_envs != 0:
            raise Error(
                "SequenceReplay.set_env_stride: CAP must be a multiple of"
                " n_envs (else the ring wrap misaligns the per-env lanes)"
            )
        self.env_stride = n_envs

    def can_sample[T: Int](self) -> Bool:
        """Need at least T+1 elements to extract a length-T window
        (T+1 obs frames + T transitions). With `env_stride = N` each env
        holds `size / N` frames, so the requirement scales by N."""
        if self.env_stride <= 1:
            return self.size >= T + 1
        return self.size >= (T + 1) * self.env_stride

    def _draw_start(mut self, n_valid: Int) -> Int:
        """Uniform logical start in [0, n_valid). Extracted so the strided and
        contiguous paths draw identically."""
        var s = Int(random_float64() * Float64(n_valid))
        if s >= n_valid:
            s = n_valid - 1
        return s

    def _origin(self) -> Int:
        """Physical index of the oldest valid slot. When the buffer is
        not yet full the data sits at [0, size); once full it's the
        ring starting at (pos) — `pos` is where the NEXT write goes, so
        the oldest slot is at `pos` itself once we wrap."""
        if self.size < Self.CAP:
            return 0
        return self.pos

    def sample_batch[
        B: Int,
        T: Int,
    ](
        mut self,
        obs_out: Pointer[Scalar[DT], MutAnyOrigin],
        act_out: Pointer[Scalar[DT], MutAnyOrigin],
        rew_out: Pointer[Scalar[DT], MutAnyOrigin],
        dne_out: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Draw `B` random length-`T` windows.

        Output strides:
          obs_out  shape [B, T+1, OBS] flat
          act_out  shape [B, T, ACT]   flat
          rew_out  shape [B, T]
          dne_out  shape [B, T]

        With `env_stride = N > 1` each window is drawn from ONE env's
        interleaved lane (frames N apart), so a window is a real trajectory
        instead of a round-robin of N different envs. Rows still spread over
        lanes — the lane is drawn per row.
        """
        if self.env_stride > 1:
            self._sample_batch_strided[B, T](
                obs_out, act_out, rew_out, dne_out
            )
            return
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
                    obs_out[unsafe_offset=dst + i] = self.obs[src + i]

            # Copy T action/reward/done frames.
            for k in range(T):
                var phys = (origin + s + k) % Self.CAP
                var src_a = phys * Self.ACT
                var dst_a = b * T * Self.ACT + k * Self.ACT
                for j in range(Self.ACT):
                    act_out[unsafe_offset=dst_a + j] = self.act[src_a + j]
                rew_out[unsafe_offset=b * T + k] = self.rew[phys]
                dne_out[unsafe_offset=b * T + k] = self.dne[phys]

    def _sample_batch_strided[
        B: Int,
        T: Int,
    ](
        mut self,
        obs_out: Pointer[Scalar[DT], MutAnyOrigin],
        act_out: Pointer[Scalar[DT], MutAnyOrigin],
        rew_out: Pointer[Scalar[DT], MutAnyOrigin],
        dne_out: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """`sample_batch` over an N-env interleaved ring (`env_stride = N`).

        Lane `e` of the ring holds env `e`'s trajectory at slots
        `origin + e, origin + N + e, origin + 2N + e, …` — so the walk is
        `phys = (origin + (s + k)·N + e) mod CAP` instead of `origin + s + k`.
        `origin` is env-aligned because `CAP % N == 0` (enforced by
        `set_env_stride`) and `pos` advances by exactly N per collector
        iteration from 0.

        Output layout is identical to `sample_batch` — the consumer cannot
        tell the difference, which is the point: `train_step` is unchanged.
        """
        var n = self.env_stride
        var per_env = self.size // n          # frames held by each env
        var n_valid = per_env - T             # valid per-env starts
        if n_valid < 1:
            raise (
                "SequenceReplay._sample_batch_strided: not enough per-env data"
                " to sample a length-T window"
            )
        var origin = self._origin()

        for b in range(B):
            # Draw the LANE per row (not per batch) so one batch mixes envs
            # across rows while each row stays within one env.
            var e = self._draw_start(n)
            var s = self._draw_start(n_valid)

            for k in range(T + 1):
                var phys = (origin + (s + k) * n + e) % Self.CAP
                var src = phys * Self.OBS
                var dst = b * (T + 1) * Self.OBS + k * Self.OBS
                for i in range(Self.OBS):
                    obs_out[unsafe_offset=dst + i] = self.obs[src + i]

            for k in range(T):
                var phys = (origin + (s + k) * n + e) % Self.CAP
                var src_a = phys * Self.ACT
                var dst_a = b * T * Self.ACT + k * Self.ACT
                for j in range(Self.ACT):
                    act_out[unsafe_offset=dst_a + j] = self.act[src_a + j]
                rew_out[unsafe_offset=b * T + k] = self.rew[phys]
                dne_out[unsafe_offset=b * T + k] = self.dne[phys]

    def sample_batch_fst[
        B: Int,
        T: Int,
    ](
        mut self,
        obs_out: Pointer[Scalar[DT], MutAnyOrigin],   # [B, T+1, OBS]
        act_out: Pointer[Scalar[DT], MutAnyOrigin],   # [B, T, ACT]
        rew_out: Pointer[Scalar[DT], MutAnyOrigin],   # [B, T]
        dne_out: Pointer[Scalar[DT], MutAnyOrigin],   # [B, T]
        fst_out: Pointer[Scalar[DT], MutAnyOrigin],   # [B, T+1] per obs frame
    ) raises:
        """`sample_batch` + the per-obs-frame `is_first` flags ([B, T+1]). The WM
        keys its carry-reset mask on `fst` (frame t+1) instead of `dne` (transition
        t) so a stored terminal frame doesn't reset the carry, but the reset frame
        after it does. For sequences with no inserted terminal frame this is
        identical to keying on `dne`, so non-terminating envs are unaffected.

        ⚠ Single-env only. `pending_first` is one global flag, so with N envs
        interleaved a done in env 3 flags env 4's next frame as the episode
        start. Raising is better than the silent mislabel."""
        if self.env_stride > 1:
            raise (
                "SequenceReplay.sample_batch_fst: is_first tracking is"
                " single-stream; env_stride > 1 would mislabel episode starts"
            )
        if self.size < T + 1:
            raise "SequenceReplay.sample_batch_fst: not enough data to sample a length-T window"
        var n_valid = self.size - T
        var origin = self._origin()
        for b in range(B):
            # Online queue first (reference `online: True`): fresh windows are
            # served exactly once, remaining rows draw uniform.
            var s = self._online_pop_start(T)
            if s < 0:
                s = Int(random_float64() * Float64(n_valid))
                if s >= n_valid:
                    s = n_valid - 1
            for k in range(T + 1):
                var phys = (origin + s + k) % Self.CAP
                var src = phys * Self.OBS
                var dst = b * (T + 1) * Self.OBS + k * Self.OBS
                for i in range(Self.OBS):
                    obs_out[unsafe_offset=dst + i] = self.obs[src + i]
                fst_out[unsafe_offset=b * (T + 1) + k] = self.fst[phys]
            for k in range(T):
                var phys = (origin + s + k) % Self.CAP
                var src_a = phys * Self.ACT
                var dst_a = b * T * Self.ACT + k * Self.ACT
                for j in range(Self.ACT):
                    act_out[unsafe_offset=dst_a + j] = self.act[src_a + j]
                rew_out[unsafe_offset=b * T + k] = self.rew[phys]
                dne_out[unsafe_offset=b * T + k] = self.dne[phys]

    def sample_batch_task[
        B: Int,
        T: Int,
    ](
        mut self,
        obs_out: Pointer[Scalar[DT], MutAnyOrigin],
        act_out: Pointer[Scalar[DT], MutAnyOrigin],
        rew_out: Pointer[Scalar[DT], MutAnyOrigin],
        dne_out: Pointer[Scalar[DT], MutAnyOrigin],
        task_out: Pointer[
            Scalar[DT], MutAnyOrigin
        ],  # [B] one task/window
    ) raises:
        """Multi-task variant of `sample_batch`: identical window sampling, plus
        one `task_id` per window written to `task_out[b]` (read at the window's
        start frame — one env per window so the task is constant across it).
        Additive; not part of the trait.

        ⚠ Single-stream only: the strided walk lives in
        `_sample_batch_strided`, which does not carry the task column. A
        BATCHED multi-task collector has to extend that first — this raise is
        the marker for it."""
        if self.env_stride > 1:
            raise (
                "SequenceReplay.sample_batch_task: env_stride > 1 is not"
                " supported on the task path yet — extend"
                " _sample_batch_strided with the task column"
            )
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
                    obs_out[unsafe_offset=dst + i] = self.obs[src + i]

            for k in range(T):
                var phys = (origin + s + k) % Self.CAP
                var src_a = phys * Self.ACT
                var dst_a = b * T * Self.ACT + k * Self.ACT
                for j in range(Self.ACT):
                    act_out[unsafe_offset=dst_a + j] = self.act[src_a + j]
                rew_out[unsafe_offset=b * T + k] = self.rew[phys]
                dne_out[unsafe_offset=b * T + k] = self.dne[phys]

            # Window task = task of the window-start frame.
            var phys0 = (origin + s) % Self.CAP
            task_out[unsafe_offset=b] = self.task[phys0]
