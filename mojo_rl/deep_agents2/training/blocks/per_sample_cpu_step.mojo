"""PerSampleCpuStep — CPU PER sampler, owns a CPUPrioritizedReplay.

Conforms to `SampleBlock`. Mirrors the GPU version's surface
(`PerSampleGpuStep`) but with host buffers and no DeviceContext.

Pre-step: stratified PER sampling populates `state.mb_*` and copies
IS weights into `state.mb_w`; sets `state.has_per = True` so downstream
critic blocks pick them up.

Post-step: trainer calls `update_priorities(state)` after the critic
step has captured signed TD residuals into `state.td_residuals`.

`set_beta(beta)` is the annealed-IS schedule hook (β: 0.4 → 1.0).
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT
from ...data.cpu_per_replay import CPUPrioritizedReplay
from ...data.n_step_replay import GPUNStepBuffer
from ..trainer_block import TrainerState
from .sample_block import SampleBlock


struct PerSampleCpuStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, CAP: Int,
](SampleBlock, Defaultable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_

    var buf: Optional[CPUPrioritizedReplay[Self.OBS, Self.ACT, Self.CAP]]
    var learning_starts: Int

    # PER hyperparameters (overridable before setup() via configure_per()).
    var per_alpha: Scalar[DT]
    var per_beta: Scalar[DT]
    var per_epsilon: Scalar[DT]

    def __init__(out self):
        self.buf = None
        self.learning_starts = 0
        self.per_alpha = Scalar[DT](0.6)
        self.per_beta = Scalar[DT](0.4)
        self.per_epsilon = Scalar[DT](1e-6)

    def configure_per(
        mut self,
        alpha: Scalar[DT] = Scalar[DT](0.6),
        beta: Scalar[DT] = Scalar[DT](0.4),
        epsilon: Scalar[DT] = Scalar[DT](1e-6),
    ):
        self.per_alpha = alpha
        self.per_beta = beta
        self.per_epsilon = epsilon

    def setup(
        mut self,
        learning_starts: Int,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """CPU block — `ctx` ignored. Allocates the CPUPrioritizedReplay."""
        self.buf = CPUPrioritizedReplay[
            Self.OBS, Self.ACT, Self.CAP,
        ].new(
            alpha=self.per_alpha,
            beta=self.per_beta,
            epsilon=self.per_epsilon,
            batch_capacity=Self.BATCH,
        )
        self.learning_starts = learning_starts

    def set_beta(mut self, beta: Scalar[DT]):
        self.per_beta = beta
        if self.buf:
            self.buf.value().set_beta(beta)

    def add(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """CPU add — `ctx` ignored."""
        self.buf.value().add(obs, action, reward, next_obs, done)

    def step(
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        """Stratified PER sample into `state.mb_*`, host IS weights
        into `state.mb_w`. Flips `state.has_per = True`."""
        if state.step_idx < self.learning_starts:
            state.did_step = False
            return
        if not self.buf.value().is_ready[Self.BATCH]():
            state.did_step = False
            return

        self.buf.value().sample[Self.BATCH](
            state.mb_s.cpu_ptr(),
            state.mb_a.cpu_ptr(),
            state.mb_r.cpu_ptr(),
            state.mb_sp.cpu_ptr(),
            state.mb_d.cpu_ptr(),
        )

        # Copy normalised IS weights into state.mb_w (host).
        var w_dst = state.mb_w.cpu_ptr()
        var w_src = self.buf.value()._host_weights.unsafe_ptr()
        for i in range(Self.BATCH):
            w_dst[i] = w_src[i]
        state.has_per = True

    def update_priorities(
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        """Called by the trainer at end of train_step. Reads per-sample
        signed TD residuals from `state.td_residuals` and refreshes the
        sum-tree leaves. No-op when `state.did_step` is False."""
        if not state.did_step:
            return
        self.buf.value().update_priorities[Self.BATCH](
            state.td_residuals.cpu_ptr(),
        )

    # ── Cross-target hooks (raise on this CPU-only block) ──────────

    def add_batch_gpu[N_ENVS: Int](
        mut self,
        ctx: DeviceContext,
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        raise Error("PerSampleCpuStep.add_batch_gpu: CPU-only block")

    def store_via_block_gpu[N_ENVS: Int, NS: Int](
        mut self,
        ctx: DeviceContext,
        mut nstep_buf: GPUNStepBuffer[NS, Self.OBS, Self.ACT, N_ENVS],
    ) raises:
        raise Error("PerSampleCpuStep.store_via_block_gpu: CPU-only block")
