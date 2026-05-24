"""N-step transition buffer — accumulates rewards across N env steps
and emits compressed transitions `(s_t, a_t, R_n, s_{t+n}, done_any)`
where `R_n = r_t + γ·r_{t+1} + … + γ^{n−1}·r_{t+n−1}`.

Phase C.2 — primitive for n-step replay. Base replay buffers
(`CPUReplay`, `GPUReplay`) store these compressed transitions; the
agent's TD target uses `γ^N` (not `γ`) for the bootstrap term:
`y = R_n + γ^N · Q(s_{t+n})`. Useful for Rainbow-style returns and
SAC variants that benefit from longer-horizon targets.

Mirrors `mojo_rl/deep_agents/core/replay/nstep_buffer.mojo` but
extended for continuous-action support: actions are `[ACT]` vectors
(not scalars), tracked as `InlineArray[Scalar[DT], ACT]` on CPU and
as `[N_ENVS * N * ACT]` device buffer rings on GPU.

Two surfaces:

  - `NStepBuffer[N, OBS, ACT]` — single-env CPU streaming. Caller
    invokes `add(s, a, r, s', done)` and receives an
    `NStepTransition` result; when `result.valid == True`, push it
    into the base replay via the standard `add(...)` API.

  - `GPUNStepBuffer[N, OBS, ACT, N_ENVS]` — per-env parallel
    accumulation on GPU. `process(ctx, obs, act, rew, nobs, done)`
    runs one kernel over all `N_ENVS` envs; the internal `out_*`
    device buffers receive emitted compressed transitions plus an
    `out_valid` flag. `store_into[CAP](ctx, buf)` blindly pushes
    all `N_ENVS` slots into a base `GPUReplay[OBS, ACT, CAP]` via
    `add_batch[N_ENVS]` — invalid slots are zero-padded and get
    overwritten as the circular buffer wraps (matches deep_agents).

Done semantics: when `done` is set, the kernel flushes the current
partial chain immediately and resets the ring. Subsequent steps
start fresh from the new episode.

Trainer integration is left to the caller: nn2 keeps the primitive
narrow and lets `SACTrainer.record` / `record_batch_gpu` users wrap
the call as needed. A `use_n_step: Bool` Saveable config flag in
`SACConfig` is a future C.2b chunk.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from ..constants import DT
from .gpu_replay import GPUReplay


# ──────────────────────────────────────────────────────────────────────
# CPU surface: NStepTransition + NStepBuffer
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct NStepTransition[OBS: Int, ACT: Int](
    Movable & ImplicitlyDestructible
):
    """Compressed n-step transition returned by `NStepBuffer.add`.

    `valid=False` means the buffer accumulated but didn't emit (ring
    not yet full and not done). `valid=True` means the caller should
    push this transition into the base replay via the standard
    `add(obs, action, reward, next_obs, done)` API.
    """

    var valid: Bool
    var obs: InlineArray[Scalar[DT], Self.OBS]
    var action: InlineArray[Scalar[DT], Self.ACT]
    var reward: Scalar[DT]
    var next_obs: InlineArray[Scalar[DT], Self.OBS]
    var done: Bool

    @staticmethod
    def empty() -> Self:
        return Self(
            valid=False,
            obs=InlineArray[Scalar[DT], Self.OBS](fill=Scalar[DT](0.0)),
            action=InlineArray[Scalar[DT], Self.ACT](fill=Scalar[DT](0.0)),
            reward=Scalar[DT](0.0),
            next_obs=InlineArray[Scalar[DT], Self.OBS](fill=Scalar[DT](0.0)),
            done=False,
        )


@fieldwise_init
struct NStepBuffer[N: Int, OBS: Int, ACT: Int](
    Movable & ImplicitlyDestructible
):
    """CPU streaming n-step buffer for a single env.

    Accumulates up to `N` transitions; on emit (ring full or `done`),
    computes the discounted n-step return via Horner's rule and
    produces an `NStepTransition` for the caller to push into a base
    replay buffer. `done` flushes the current partial chain
    immediately and resets the ring.

    Storage: flat host `List` allocations sized at construction.
    `count` tracks current ring fill (0..N).
    """

    var obs: List[Scalar[DT]]      # [N * OBS]
    var actions: List[Scalar[DT]]  # [N * ACT]
    var rewards: List[Scalar[DT]]  # [N]
    var gamma: Scalar[DT]
    var count: Int

    @staticmethod
    def new(gamma: Scalar[DT] = Scalar[DT](0.99)) -> Self:
        return Self(
            obs=List[Scalar[DT]](
                length=Self.N * Self.OBS, fill=Scalar[DT](0.0),
            ),
            actions=List[Scalar[DT]](
                length=Self.N * Self.ACT, fill=Scalar[DT](0.0),
            ),
            rewards=List[Scalar[DT]](
                length=Self.N, fill=Scalar[DT](0.0),
            ),
            gamma=gamma,
            count=0,
        )

    def _compute_return(self, n: Int) -> Scalar[DT]:
        """R = r_0 + γ·r_1 + … + γ^{n−1}·r_{n−1} via Horner's rule."""
        var r = Scalar[DT](0.0)
        for i in range(n - 1, -1, -1):
            r = r * self.gamma + self.rewards[i]
        return r

    def _shift_left(mut self):
        """Drop the oldest slot, shift everything left by one."""
        for i in range(Self.N - 1):
            for d in range(Self.OBS):
                self.obs[i * Self.OBS + d] = (
                    self.obs[(i + 1) * Self.OBS + d]
                )
            for j in range(Self.ACT):
                self.actions[i * Self.ACT + j] = (
                    self.actions[(i + 1) * Self.ACT + j]
                )
            self.rewards[i] = self.rewards[i + 1]

    def add(
        mut self,
        obs_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward: Scalar[DT],
        next_obs_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done: Bool,
    ) -> NStepTransition[Self.OBS, Self.ACT]:
        """Push one env-step transition. Returns an emitted n-step
        transition (`valid=True`) once ready, otherwise `valid=False`.

        Args:
            obs_p: pointer to current obs of length OBS.
            action_p: pointer to action of length ACT.
            reward: per-step reward.
            next_obs_p: pointer to next obs of length OBS.
            done: episode-done flag.
        """
        var c = self.count
        for d in range(Self.OBS):
            self.obs[c * Self.OBS + d] = obs_p[d]
        for j in range(Self.ACT):
            self.actions[c * Self.ACT + j] = action_p[j]
        self.rewards[c] = reward
        c += 1
        self.count = c

        if done or c == Self.N:
            var r_n = self._compute_return(c)
            var s0 = InlineArray[Scalar[DT], Self.OBS](
                fill=Scalar[DT](0.0)
            )
            var a0 = InlineArray[Scalar[DT], Self.ACT](
                fill=Scalar[DT](0.0)
            )
            for d in range(Self.OBS):
                s0[d] = self.obs[d]
            for j in range(Self.ACT):
                a0[j] = self.actions[j]
            var sn = InlineArray[Scalar[DT], Self.OBS](
                fill=Scalar[DT](0.0)
            )
            for d in range(Self.OBS):
                sn[d] = next_obs_p[d]

            if done:
                self.count = 0
            else:
                self._shift_left()
                self.count = Self.N - 1

            return NStepTransition[Self.OBS, Self.ACT](
                valid=True,
                obs=s0, action=a0, reward=r_n,
                next_obs=sn, done=done,
            )
        return NStepTransition[Self.OBS, Self.ACT].empty()

    def reset(mut self):
        """Reset the ring (e.g. on episode boundary if the caller is
        feeding `done=False` transitions and needs an explicit
        flush)."""
        self.count = 0


# ──────────────────────────────────────────────────────────────────────
# GPU surface: kernel + GPUNStepBuffer.
# ──────────────────────────────────────────────────────────────────────


def _nstep_process_kernel[
    N_ENVS: Int, N: Int, OBS: Int, ACT: Int,
](
    in_obs: LayoutTensor[
        DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin,
    ],
    in_act: LayoutTensor[
        DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin,
    ],
    in_rew: LayoutTensor[
        DT, Layout.row_major(N_ENVS), MutAnyOrigin,
    ],
    in_nobs: LayoutTensor[
        DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin,
    ],
    in_done: LayoutTensor[
        DT, Layout.row_major(N_ENVS), MutAnyOrigin,
    ],
    ring_obs: LayoutTensor[
        DT, Layout.row_major(N_ENVS, N * OBS), MutAnyOrigin,
    ],
    ring_act: LayoutTensor[
        DT, Layout.row_major(N_ENVS, N * ACT), MutAnyOrigin,
    ],
    ring_rew: LayoutTensor[
        DT, Layout.row_major(N_ENVS, N), MutAnyOrigin,
    ],
    counts: LayoutTensor[
        DType.int32, Layout.row_major(N_ENVS), MutAnyOrigin,
    ],
    out_obs: LayoutTensor[
        DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin,
    ],
    out_act: LayoutTensor[
        DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin,
    ],
    out_rew: LayoutTensor[
        DT, Layout.row_major(N_ENVS), MutAnyOrigin,
    ],
    out_nobs: LayoutTensor[
        DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin,
    ],
    out_done: LayoutTensor[
        DT, Layout.row_major(N_ENVS), MutAnyOrigin,
    ],
    out_valid: LayoutTensor[
        DType.int32, Layout.row_major(N_ENVS), MutAnyOrigin,
    ],
    gamma: Scalar[DT],
):
    """Per-env parallel n-step accumulation.

    Thread `e` (one per env):
      1. Append `(in_obs, in_act, in_rew)` at ring slot `count`; ++count.
      2. If `in_done` is set OR `count == N`: compute `R_n` via Horner,
         emit `(ring_obs[0], ring_act[0], R_n, in_nobs, in_done)` and
         either reset count to 0 (done) or shift-left + count=N−1.
      3. Else: `out_valid[e] = 0`; no emit.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var c = Int(counts[e])

    for d in range(OBS):
        ring_obs[e, c * OBS + d] = in_obs[e, d]
    for j in range(ACT):
        ring_act[e, c * ACT + j] = in_act[e, j]
    ring_rew[e, c] = in_rew[e]
    c += 1

    var is_done = in_done[e] > Scalar[DT](0.5)

    if is_done or c == N:
        var r_n = Scalar[DT](0.0)
        for i in range(c - 1, -1, -1):
            r_n = r_n * gamma + rebind[Scalar[DT]](ring_rew[e, i])

        for d in range(OBS):
            out_obs[e, d] = rebind[Scalar[DT]](ring_obs[e, d])
            out_nobs[e, d] = rebind[Scalar[DT]](in_nobs[e, d])
        for j in range(ACT):
            out_act[e, j] = rebind[Scalar[DT]](ring_act[e, j])
        out_rew[e] = r_n
        out_done[e] = (
            Scalar[DT](1.0) if is_done else Scalar[DT](0.0)
        )
        out_valid[e] = Int32(1)

        if is_done:
            counts[e] = Int32(0)
        else:
            for i in range(N - 1):
                for d in range(OBS):
                    ring_obs[e, i * OBS + d] = (
                        ring_obs[e, (i + 1) * OBS + d]
                    )
                for j in range(ACT):
                    ring_act[e, i * ACT + j] = (
                        ring_act[e, (i + 1) * ACT + j]
                    )
                ring_rew[e, i] = ring_rew[e, i + 1]
            counts[e] = Int32(N - 1)
    else:
        out_valid[e] = Int32(0)
        counts[e] = Int32(c)


@fieldwise_init
struct GPUNStepBuffer[N: Int, OBS: Int, ACT: Int, N_ENVS: Int](
    Movable & ImplicitlyDestructible
):
    """GPU per-env streaming n-step buffer for `N_ENVS` parallel envs.

    Internal storage:
      * `ring_obs`, `ring_act`, `ring_rew` — per-env rings of N slots.
      * `counts` — per-env step counter (0..N) on device.
      * `out_*` — compressed emit buffers, overwritten each call.
      * `out_valid` — 1 if env emitted this call, 0 otherwise.

    Usage:
      ```
      var ns = GPUNStepBuffer[3, 3, 1, 8].new(ctx, gamma=Scalar[DT](0.99))
      # per loop iter:
      ns.process(ctx, obs, act, rew, nobs, done)
      ns.store_into[CAP](ctx, replay_buf)
      ```

    Blind-store semantics in `store_into`: pushes ALL N_ENVS slots
    regardless of `out_valid`. Slots where the env wasn't ready hold
    zero-padded data; the circular buffer overwrites them as it wraps.
    Matches `deep_agents`' `GPUNStepBuffer.store_into`. For exact
    compaction (skip invalid), the caller can D2H `out_valid` and call
    `buf.add` per-valid; nn2 keeps the simpler path here.
    """

    var ring_obs: DeviceBuffer[DT]
    var ring_act: DeviceBuffer[DT]
    var ring_rew: DeviceBuffer[DT]
    var counts: DeviceBuffer[DType.int32]

    var out_obs: DeviceBuffer[DT]
    var out_act: DeviceBuffer[DT]
    var out_rew: DeviceBuffer[DT]
    var out_nobs: DeviceBuffer[DT]
    var out_done: DeviceBuffer[DT]
    var out_valid: DeviceBuffer[DType.int32]

    var gamma: Scalar[DT]

    @staticmethod
    def new(
        ctx: DeviceContext,
        gamma: Scalar[DT] = Scalar[DT](0.99),
    ) raises -> Self:
        var ring_obs = ctx.enqueue_create_buffer[DT](
            Self.N_ENVS * Self.N * Self.OBS
        )
        var ring_act = ctx.enqueue_create_buffer[DT](
            Self.N_ENVS * Self.N * Self.ACT
        )
        var ring_rew = ctx.enqueue_create_buffer[DT](Self.N_ENVS * Self.N)
        var counts = ctx.enqueue_create_buffer[DType.int32](Self.N_ENVS)
        ring_obs.enqueue_fill(Scalar[DT](0.0))
        ring_act.enqueue_fill(Scalar[DT](0.0))
        ring_rew.enqueue_fill(Scalar[DT](0.0))
        counts.enqueue_fill(Int32(0))

        var out_obs = ctx.enqueue_create_buffer[DT](
            Self.N_ENVS * Self.OBS
        )
        var out_act = ctx.enqueue_create_buffer[DT](
            Self.N_ENVS * Self.ACT
        )
        var out_rew = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        var out_nobs = ctx.enqueue_create_buffer[DT](
            Self.N_ENVS * Self.OBS
        )
        var out_done = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        var out_valid = ctx.enqueue_create_buffer[DType.int32](Self.N_ENVS)
        out_obs.enqueue_fill(Scalar[DT](0.0))
        out_act.enqueue_fill(Scalar[DT](0.0))
        out_rew.enqueue_fill(Scalar[DT](0.0))
        out_nobs.enqueue_fill(Scalar[DT](0.0))
        out_done.enqueue_fill(Scalar[DT](0.0))
        out_valid.enqueue_fill(Int32(0))

        return Self(
            ring_obs=ring_obs^, ring_act=ring_act^, ring_rew=ring_rew^,
            counts=counts^,
            out_obs=out_obs^, out_act=out_act^, out_rew=out_rew^,
            out_nobs=out_nobs^, out_done=out_done^, out_valid=out_valid^,
            gamma=gamma,
        )

    def process(
        mut self,
        ctx: DeviceContext,
        obs: DeviceBuffer[DT],
        act: DeviceBuffer[DT],
        rew: DeviceBuffer[DT],
        nobs: DeviceBuffer[DT],
        done: DeviceBuffer[DT],
    ) raises:
        """One kernel: ring update + emit decision for all N_ENVS envs.
        Outputs land in the internal `out_*` device buffers;
        `out_valid[e]` is 1 if env `e` emitted this call."""
        var in_obs_lt = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.OBS), MutAnyOrigin,
        ](obs.unsafe_ptr())
        var in_act_lt = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.ACT), MutAnyOrigin,
        ](act.unsafe_ptr())
        var in_rew_lt = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS), MutAnyOrigin,
        ](rew.unsafe_ptr())
        var in_nobs_lt = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.OBS), MutAnyOrigin,
        ](nobs.unsafe_ptr())
        var in_done_lt = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS), MutAnyOrigin,
        ](done.unsafe_ptr())

        var ring_obs_lt = LayoutTensor[
            DT,
            Layout.row_major(Self.N_ENVS, Self.N * Self.OBS),
            MutAnyOrigin,
        ](self.ring_obs.unsafe_ptr())
        var ring_act_lt = LayoutTensor[
            DT,
            Layout.row_major(Self.N_ENVS, Self.N * Self.ACT),
            MutAnyOrigin,
        ](self.ring_act.unsafe_ptr())
        var ring_rew_lt = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.N), MutAnyOrigin,
        ](self.ring_rew.unsafe_ptr())
        var counts_lt = LayoutTensor[
            DType.int32, Layout.row_major(Self.N_ENVS), MutAnyOrigin,
        ](self.counts.unsafe_ptr())

        var out_obs_lt = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.OBS), MutAnyOrigin,
        ](self.out_obs.unsafe_ptr())
        var out_act_lt = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.ACT), MutAnyOrigin,
        ](self.out_act.unsafe_ptr())
        var out_rew_lt = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS), MutAnyOrigin,
        ](self.out_rew.unsafe_ptr())
        var out_nobs_lt = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.OBS), MutAnyOrigin,
        ](self.out_nobs.unsafe_ptr())
        var out_done_lt = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS), MutAnyOrigin,
        ](self.out_done.unsafe_ptr())
        var out_valid_lt = LayoutTensor[
            DType.int32, Layout.row_major(Self.N_ENVS), MutAnyOrigin,
        ](self.out_valid.unsafe_ptr())

        comptime TPB = 128
        comptime n_blocks = (Self.N_ENVS + TPB - 1) // TPB
        comptime kernel = _nstep_process_kernel[
            Self.N_ENVS, Self.N, Self.OBS, Self.ACT,
        ]
        ctx.enqueue_function[kernel](
            in_obs_lt, in_act_lt, in_rew_lt, in_nobs_lt, in_done_lt,
            ring_obs_lt, ring_act_lt, ring_rew_lt, counts_lt,
            out_obs_lt, out_act_lt, out_rew_lt, out_nobs_lt,
            out_done_lt, out_valid_lt,
            self.gamma,
            grid_dim=n_blocks, block_dim=TPB,
        )

    def store_into[CAP: Int](
        self,
        ctx: DeviceContext,
        mut buf: GPUReplay[Self.OBS, Self.ACT, CAP],
    ) raises:
        """Blind-store all N_ENVS slots into a base GPUReplay via
        `add_batch[N_ENVS]`. Invalid slots (`out_valid[e] == 0`)
        contain zero-padded data and get overwritten as the buffer
        wraps. Matches deep_agents semantics."""
        buf.add_batch[Self.N_ENVS](
            ctx,
            self.out_obs, self.out_act, self.out_rew,
            self.out_nobs, self.out_done,
        )

    def reset(mut self, ctx: DeviceContext) raises:
        """Reset all per-env counts to 0 (e.g. on episode boundary
        when the kernel's auto-reset-on-done path isn't used)."""
        self.counts.enqueue_fill(Int32(0))
