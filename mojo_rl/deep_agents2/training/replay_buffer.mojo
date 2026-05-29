"""ReplayBuffer — the unifying trait over CPU and GPU replay storage.

The composable seam that lets one generic sample block work over either
backend, instead of parallel `…CpuStep` / `…GpuStep` structs. `CPUReplay`
(host pointers, loop add/sample) and `GPUReplay` (device buffers, kernel
add/sample) are genuinely different *compute*, so they stay separate
structs — but as principled conformers of one trait, like every other
strategy in deep_agents2.

The one signature that has to span both backends is sampling: CPU writes
into `state.mb_*.cpu_ptr()`, GPU launches a gather into
`state.mb_*.dev` using `state.ctx`. Both destinations already live on
`TrainerState`, so `sample_into[BATCH](state)` lets each conformer reach
for the side it needs — the trait surface stays identical.

Construction goes through the static `make` (CPU ignores `ctx`; GPU
requires it), mirroring the nn2 `Module.make[target](ctx)` idiom, so a
generic block can build its buffer without branching on target.

Layering: this module imports only `TrainerState` (a near-leaf that
depends on nn2 alone). The buffers in `data/` import this trait to
declare conformance — `blocks → data → replay_buffer → trainer_block`,
no cycle.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT
from .trainer_block import TrainerState


trait ReplayBuffer(Movable, ImplicitlyDestructible):
    """Uniform surface over a transition replay buffer. `count` is named
    so as not to collide with conformers' `size` *field*."""

    comptime OBS: Int
    comptime ACT: Int
    comptime CAP: Int

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        batch_capacity: Int = 4096,
    ) raises -> Self:
        """Allocate the buffer. CPU backends ignore both args; GPU
        backends require `ctx` (raise if None) and size their sample-side
        index scratch to `batch_capacity`."""
        ...

    def add(
        mut self,
        ref s: List[Scalar[DT]],
        ref a: List[Scalar[DT]],
        r: Scalar[DT],
        ref sp: List[Scalar[DT]],
        d: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Push one transition `(s, a, r, s', done)`. GPU backends do the
        List→device staging internally; CPU backends ignore `ctx`."""
        ...

    def sample_into[BATCH: Int](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, BATCH],
    ) raises:
        """Sample `BATCH` transitions into `state.mb_s/mb_a/mb_r/mb_sp/
        mb_d`. CPU writes host pointers; GPU launches a gather into the
        device mirrors using `state.ctx`."""
        ...

    def count(self) -> Int:
        """Number of transitions currently stored (saturates at CAP)."""
        ...

    # ─── Prioritized-replay hooks (default no-op; PER backends override) ─
    #
    # Uniform backends inherit the no-op defaults, so the generic sample
    # block can forward these unconditionally without branching on
    # whether `R` is prioritized. `sample_into` is where a PER backend
    # additionally fills `state.mb_w` and flips `state.has_per`.

    def configure_per(
        mut self,
        alpha: Scalar[DT] = Scalar[DT](0.6),
        beta: Scalar[DT] = Scalar[DT](0.4),
        epsilon: Scalar[DT] = Scalar[DT](1e-6),
    ):
        """Set the PER exponents before any `add`. No-op for uniform
        backends."""
        pass

    def set_beta(mut self, beta: Scalar[DT]):
        """Anneal the IS-weight exponent β (0.4 → 1.0 over training).
        No-op for uniform backends."""
        pass

    def update_priorities[BATCH: Int](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, BATCH],
    ) raises:
        """Refresh sum-tree leaves from the signed TD residuals in
        `state.td_residuals` for the slice sampled by the most recent
        `sample_into`. No-op for uniform backends."""
        pass

    # ─── Device-batch hooks (default raise / no-op; GPU backends override) ─
    #
    # These are the capabilities the GPU multi-env path needs. Lifting
    # them onto the trait (rather than leaving them on concrete GPU
    # structs) is what lets the generic `ReplaySampleStep[R]` /
    # `NStepSampleStep[N, R]` blocks subsume the former GPU-only sample
    # blocks: a generic `R`-typed buffer can only call *trait* methods,
    # so `add_batch` and `configure_ere` have to live here. CPU backends
    # inherit the raising / no-op defaults (their drivers never call the
    # device path), mirroring `SampleBlock.store_via_block_gpu`.

    def add_batch[N_ENVS: Int](
        mut self,
        ctx: DeviceContext,
        src_obs: DeviceBuffer[DT],
        src_act: DeviceBuffer[DT],
        src_rew: DeviceBuffer[DT],
        src_nxt: DeviceBuffer[DT],
        src_dne: DeviceBuffer[DT],
    ) raises:
        """Push `N_ENVS` device-resident transitions in one kernel
        launch. The seam `GPUNStepBuffer.store_into` drives through this.
        Default raises — only device backends (GPUReplay /
        GPUPrioritizedReplay) implement it; CPU backends never see the
        device multi-env path."""
        raise Error(
            "add_batch not supported by this ReplayBuffer (CPU backend)"
        )

    def configure_ere(
        mut self,
        enable: Bool = False,
        eta: Scalar[DT] = Scalar[DT](0.996),
        c_min: Int = 1,
        k_max: Int = 1000,
    ) raises:
        """Enable ERE recency-biased sampling (Wang & Ross 2019). No-op
        for backends without an ERE path (CPU, GPU PER); GPUReplay
        overrides to flip its ERE state. Call after `make`."""
        pass
