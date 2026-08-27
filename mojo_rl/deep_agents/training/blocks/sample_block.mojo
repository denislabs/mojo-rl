"""SampleBlock — uniform trait surface for replay-buffer-owning sample
blocks consumed by the unified SAC trainer (and future off-policy
trainers).

Each conforming block owns its replay buffer (uniform CPU, uniform GPU,
or PER+GPU today; PER+CPU + sequence/NStep future) and exposes:

  - `setup(learning_starts, ctx?)`            — allocate buffers
  - `add(obs, action, r, nxt, d, ctx?)`       — push transition
  - `step(state)`                             — sample into state.mb_*
                                                  (PER also writes mb_w
                                                  + flips state.has_per)
  - `update_priorities(state)`  default pass  — refresh sum-tree (PER)
  - `set_beta(beta)`           default no-op  — IS β anneal (PER)

The trait's `step` / `update_priorities` are deliberately NOT comptime-
parameterised on `target`: every conforming block has a fixed kernel
target (CPU-only or GPU-only). The trainer's own `target` selects which
block type to instantiate at the type level; the block's `step` then
does the right thing internally. This keeps the trait minimal and lets
unified trainers stay agnostic to the sample block's compute target.

`Optional[DeviceContext]` on `setup`/`add` mirrors the matmul-stdlib
idiom: CPU blocks ignore `ctx`, GPU blocks require it (raise if None).
Callers can pass `ctx=trainer.ctx` (an `Optional[DeviceContext]` field
on GPU trainers) unconditionally — Mojo's implicit promotion handles
the wrapping when needed.

`Defaultable` IS a parent trait — the unified SAC trainer calls
`Self.SAMPLE()` to default-construct the sample block in its `__init__`,
which requires the trait to advertise a no-arg constructor.
"""

from max.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import DT
from ...data.n_step_replay import GPUNStepBuffer
from ..trainer_block import TrainerState


trait SampleBlock(Defaultable, Deinitable, Movable):
    comptime OBS: Int
    comptime ACT: Int
    comptime BATCH: Int

    def setup(
        mut self,
        learning_starts: Int,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ...

    def add(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ...

    def step(
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        ...

    def update_priorities(
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        pass

    def set_beta(mut self, beta: Scalar[DT]):
        pass

    def configure_per(
        mut self,
        alpha: Scalar[DT] = 0.6,
        beta: Scalar[DT] = 0.4,
        epsilon: Scalar[DT] = 1e-6,
    ):
        """Override PER hyperparameters before `setup()`. No-op for
        uniform blocks. Allows the unified trainer's `make()` to apply
        PER args unconditionally without comptime-branching on the
        block type."""
        pass

    def configure_gamma(mut self, gamma: Scalar[DT]):
        """Align the sample block's discount with the trainer's γ. Used
        by n-step blocks to ensure the per-step n-step return is computed
        with the same γ the target-Y bootstrap uses. No-op for blocks
        that don't accumulate multi-step returns (uniform, PER)."""
        pass

    def configure_ere(
        mut self,
        enable: Bool = False,
        eta: Scalar[DT] = 0.996,
        c_min: Int = 1,
        k_max: Int = 1000,
    ) raises:
        """Enable ERE recency-biased sampling (Wang & Ross 2019). No-op
        for blocks that don't own a GPUReplay (CPU / PER). Called from
        the unified trainer's `make()` after `setup()` when the trainer
        was built with `use_ere=True`."""
        pass

    def add_batch_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        """Push N_ENVS transitions into the device-resident replay in
        one kernel launch. Required for the N_ENVS multi-env GPU
        driver. Default raises — CPU blocks + n-step wrappers don't
        support this yet."""
        raise Error("add_batch_gpu not supported by this SampleBlock")

    def store_via_block_gpu[
        N_ENVS: Int, NS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut nstep_buf: GPUNStepBuffer[NS, Self.OBS, Self.ACT, N_ENVS],
    ) raises:
        """Route GPUNStepBuffer.store_into through this block's owned
        GPU replay buffer. Used by the unified trainer's
        record_batch_gpu_nstep[N_ENVS, NS] path so the multi-env +
        n-step combination doesn't need direct access to the replay.

        Default raises — only GPU sample blocks implement this. The
        block selects the right GPUReplay vs GPUPrioritizedReplay
        overload of store_into internally."""
        raise Error("store_via_block_gpu not supported by this SampleBlock")
