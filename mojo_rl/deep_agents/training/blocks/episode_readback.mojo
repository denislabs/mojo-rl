"""Shared per-env episode-return tracking for the batched off-policy drivers.

ONE copy of two blocks the continuous (`driver_offpolicy.mojo`) and
discrete (`driver_offpolicy_discrete.mojo`) batched drivers previously
carried as near-identical inline code:

  * `accumulate_episode_returns` — the host-side accumulate loop: add
    this step's reward into each env's running return; on done, move the
    finished return into `completed` (in env order) and zero the
    accumulator. Used directly on CPU-env branches / hybrids where
    reward+done are already host-side.
  * `EpisodeReturnRing[N_ENVS]` — the deferred GPU readback: a ring of
    pinned host buffers (one (reward, done) pair per slot) lets the
    driver enqueue the small per-iteration D2H copies WITHOUT
    synchronizing, then drain the buffered iterations with a single
    `synchronize` when the ring fills or an emit boundary is imminent.
    `sync_every == 1` reproduces a per-iteration sync exactly; higher
    values trade logging granularity for far fewer host↔device stalls —
    important once the train step is a captured CUDA graph. Returns are
    drained in order, so `mean_return` / `ep_count` stay exact at every
    emit boundary.

The caller consumes the drained `completed` list with its own trainer
hooks (`add_complete_return` + `ep_returns.append(mean_return())`), so
this block stays trait-free.
"""

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.nn.constants import DT


def accumulate_episode_returns[
    N_ENVS: Int,
](
    rewards_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    dones_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    mut per_env: List[Scalar[DT]],
    mut completed: List[Scalar[DT]],
):
    """Accumulate one env-step of host-side rewards into the per-env
    return accumulators; on done (> 0.5), append the finished return to
    `completed` (env order) and zero that env's accumulator."""
    for e in range(N_ENVS):
        per_env[e] = per_env[e] + rewards_ptr[unsafe_offset=e]
        if dones_ptr[unsafe_offset=e] > Scalar[DT](0.5):
            completed.append(per_env[e])
            per_env[e] = Scalar[DT](0.0)


@fieldwise_init
struct EpisodeReturnRing[N_ENVS: Int](Movable):
    """Ring of pinned host (reward, done) buffers for deferred episode
    readback on GPU-env batched drivers. See module docstring."""

    var ring_reward: List[HostBuffer[DT]]
    var ring_done: List[HostBuffer[DT]]
    var pending: Int
    var sync_every: Int
    var per_env: List[Scalar[DT]]

    @staticmethod
    def make(ctx: DeviceContext, episode_sync_every: Int) raises -> Self:
        var se = episode_sync_every if episode_sync_every >= 1 else 1
        var rr = List[HostBuffer[DT]]()
        var rd = List[HostBuffer[DT]]()
        for _ in range(se):
            rr.append(ctx.enqueue_create_host_buffer[DT](Self.N_ENVS))
            rd.append(ctx.enqueue_create_host_buffer[DT](Self.N_ENVS))
        return Self(
            ring_reward=rr^,
            ring_done=rd^,
            pending=0,
            sync_every=se,
            per_env=List[Scalar[DT]](length=Self.N_ENVS, fill=Scalar[DT](0.0)),
        )

    def enqueue(
        mut self,
        ctx: DeviceContext,
        reward_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        done_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Enqueue this iteration's reward+done D2H into the next ring
        slot WITHOUT synchronizing. Caller must `drain` before the ring
        fills (`due()` handles the boundary logic)."""
        var reward_view = DeviceBuffer[DT](
            ctx, reward_ptr, Self.N_ENVS, owning=False
        )
        var done_view = DeviceBuffer[DT](
            ctx, done_ptr, Self.N_ENVS, owning=False
        )
        ctx.enqueue_copy(self.ring_reward[self.pending], reward_view)
        ctx.enqueue_copy(self.ring_done[self.pending], done_view)
        self.pending += 1

    def due(self, emit_now: Bool) -> Bool:
        """True when the ring is full or the caller reached an emit
        boundary (print/diag/end/... — the caller decides `emit_now`)."""
        return self.pending >= self.sync_every or emit_now

    def drain(mut self, ctx: DeviceContext) raises -> List[Scalar[DT]]:
        """ONE `synchronize`, then fold every buffered iteration through
        the per-env accumulators in order. Returns the completed episode
        returns (oldest first). No-op (and NO sync) when nothing is
        pending — safe as a defensive final drain."""
        var completed = List[Scalar[DT]]()
        if self.pending == 0:
            return completed^
        ctx.synchronize()
        for s in range(self.pending):
            accumulate_episode_returns[Self.N_ENVS](
                self.ring_reward[s].unsafe_ptr().as_unsafe_any_origin(),
                self.ring_done[s].unsafe_ptr().as_unsafe_any_origin(),
                self.per_env,
                completed,
            )
        self.pending = 0
        return completed^
