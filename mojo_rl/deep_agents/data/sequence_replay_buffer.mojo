"""SequenceReplayBuffer — the sibling trait over CPU and GPU *sequence*
replay storage.

The `ReplayBuffer` trait (`training/replay_buffer.mojo`) is about single
transitions: it samples `BATCH` independent `(s, a, r, s', done)` into a
`TrainerState`. World-model training (DreamerV3 / TD-MPC2) needs something
genuinely different — contiguous length-`T` *windows* — so it gets its own
trait rather than being shoehorned under `ReplayBuffer`. The two share no
sampling contract (one fills `[BATCH, OBS]`, the other `[B, T+1, OBS]`), so
unifying them would mean a union surface with raising stubs everywhere; a
clean second trait is the honest shape.

`SequenceReplay` (host pointers, loop record/sample) and `GPUSequenceReplay`
(device buffers, kernel record/sample) are genuinely different *compute* — as
with `CPUReplay` / `GPUReplay` they stay separate structs but conform to one
trait, so a consumer can hold `comptime RepT = …` and flip backend by
re-binding the target.

Sample surface — two entry points, each backend native on one side:

  - `sample_batch[B, T]`     → writes the four **host** output pointers. The
    CPU contract DreamerV3Trainer uses today. GPU implements it as a
    device-sample-then-copy-to-host bridge (stored `ctx`), so a GPU buffer
    can still feed the current CPU world-model pipeline.
  - `sample_batch_dev[B, T]` → writes four **device** buffers, no host
    round-trip. The fast path for when the GPU world-model consumer lands
    (DreamerV3 PR5c Step 5). Default raises; only GPU backends implement it.

Mirrors the `ReplayBuffer` device-hook idiom (`add_batch` / `configure_ere`):
device-only capabilities live on the trait with raising defaults so the CPU
backend inherits them and a generic consumer never branches on backend.

Window semantics (match the CPU `SequenceReplay`): a length-`T` window is
`T+1` observation frames + `T` action/reward/done frames; the next-obs of a
record is implicitly the obs of the next record. Episode-boundary rejection
(refusing windows that span a reset) is intentionally NOT done here — both
backends sample uniformly over starts and rely on the consumer's `dne`
masking, so CPU and GPU produce the same window distribution. Adding
rejection + a separate term/trunc flag is a future enhancement to apply to
both backends together.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import DT


trait SequenceReplayBuffer(ImplicitlyDeletable, Movable):
    """Uniform surface over a sequence (window) replay buffer. `count` is
    named so as not to collide with conformers' `size` *field*."""

    comptime OBS: Int
    comptime ACT: Int
    comptime CAP: Int

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Allocate the buffer. `target == "cpu"` ignores `ctx`; `"gpu"`
        requires it (raise if None) and allocates device storage."""
        ...

    def record(
        mut self,
        s: UnsafePointer[Scalar[DT], MutAnyOrigin],
        a: UnsafePointer[Scalar[DT], MutAnyOrigin],
        r: Scalar[DT],
        d: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Push one transition `(s, a, r, done)` at the ring head. `s` is
        the *current* obs (next-obs is the next record's obs). CPU stores
        directly and ignores `ctx`; GPU stages the host pointers to device
        (requires `ctx`)."""
        ...

    def count(self) -> Int:
        """Number of transitions currently stored (saturates at CAP)."""
        ...

    def can_sample[T: Int](self) -> Bool:
        """Need at least T+1 stored elements to extract a length-T window
        (T+1 obs frames + T transitions)."""
        ...

    def sample_batch[
        B: Int,
        T: Int,
    ](
        mut self,
        obs_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        rew_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        dne_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Draw `B` random length-`T` windows into the four **host** output
        regions:
          obs_out  [B, T+1, OBS] flat
          act_out  [B, T, ACT]   flat
          rew_out  [B, T]
          dne_out  [B, T]
        CPU writes these directly; GPU samples on-device then copies out."""
        ...

    # ─── Device-batch hooks (default raise; GPU backends override) ─────────
    #
    # The capabilities the GPU multi-env / device-resident path needs.
    # Lifting them onto the trait (with raising defaults) mirrors
    # `ReplayBuffer.add_batch` — a generic `S`-typed buffer can only call
    # *trait* methods, and CPU backends never see the device path.

    def record_batch[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        src_obs: DeviceBuffer[DT],
        src_act: DeviceBuffer[DT],
        src_rew: DeviceBuffer[DT],
        src_dne: DeviceBuffer[DT],
    ) raises:
        """Push `N_ENVS` device-resident transitions in one kernel launch
        (lockstep multi-env collection). Default raises — only device
        backends implement it."""
        raise Error(
            "record_batch not supported by this SequenceReplayBuffer"
            " (CPU backend)"
        )

    def sample_batch_dev[
        B: Int,
        T: Int,
    ](
        mut self,
        ctx: DeviceContext,
        obs_dev: DeviceBuffer[DT],
        act_dev: DeviceBuffer[DT],
        rew_dev: DeviceBuffer[DT],
        dne_dev: DeviceBuffer[DT],
    ) raises:
        """Draw `B` length-`T` windows into caller-provided **device**
        buffers (no host round-trip). Same layout as `sample_batch`.
        Default raises — only device backends implement it."""
        raise Error(
            "sample_batch_dev not supported by this SequenceReplayBuffer"
            " (CPU backend)"
        )
