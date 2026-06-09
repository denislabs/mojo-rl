"""ReplaySampleStep[R, BATCH] — single-env sample block, generic over
any `ReplayBuffer`.

ONE block for both uniform and prioritized single-env replay: the
backend `R` decides which. Uniform when `R` is `CPUReplay`/`GPUReplay`/
`AnyReplay`; prioritized when `R` is `CPUPrioritizedReplay`/
`GPUPrioritizedReplay`/`AnyPerReplay`. This subsumes the parallel
`UniformSampleCpuStep`/`UniformSampleGpuStep`/`PerSampleCpuStep`/
`PerSampleGpuStep` structs (the PER hooks forward to `R`, which no-ops
for uniform backends).

Implements the core `SampleBlock` surface plus the PER hooks
(configure_per / set_beta / update_priorities) AND the GPU-batch hooks
(configure_ere / add_batch_gpu / store_via_block_gpu). The latter forward
to `R`'s trait methods — `add_batch` / `configure_ere` are
`ReplayBuffer` methods (default raise / no-op for CPU backends), and
`store_via_block_gpu` routes `GPUNStepBuffer.store_into` through the owned
`R`. This is what lets ONE block subsume the former GPU-only
`UniformSampleGpuStep` / `PerSampleGpuStep` as well. `configure_gamma`
inherits its `SampleBlock` default — the host n-step decorator
(`NStepSampleStep`) handles γ.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT
from ..replay_buffer import ReplayBuffer
from ..trainer_block import TrainerState
from ...data.n_step_replay import GPUNStepBuffer
from .sample_block import SampleBlock


struct ReplaySampleStep[R: ReplayBuffer, BATCH_: Int](
    SampleBlock, Defaultable
):
    comptime OBS = Self.R.OBS
    comptime ACT = Self.R.ACT
    comptime BATCH = Self.BATCH_

    var buf: Optional[Self.R]
    var learning_starts: Int

    # PER hyperparams — stored on configure_per(), applied to the buffer
    # at setup() (after make()). No-op for uniform backends.
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

    def set_beta(mut self, beta: Scalar[DT]):
        self.per_beta = beta
        if self.buf:
            self.buf.value().set_beta(beta)

    def setup(
        mut self,
        learning_starts: Int,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.buf = Self.R.make(ctx=ctx, batch_capacity=Self.BATCH)
        self.buf.value().configure_per(
            alpha=self.per_alpha,
            beta=self.per_beta,
            epsilon=self.per_epsilon,
        )
        self.learning_starts = learning_starts

    def add(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.buf.value().add(obs, action, reward, next_obs, done, ctx=ctx)

    def step(
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        if state.step_idx < self.learning_starts:
            state.did_step = False
            return
        if self.buf.value().count() < Self.BATCH:
            state.did_step = False
            return
        self.buf.value().sample_into[Self.BATCH](state)

    def update_priorities(
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        if not state.did_step:
            return
        self.buf.value().update_priorities[Self.BATCH](state)

    # ── GPU-batch hooks — forward to R's trait methods. No-op / raise
    #    for CPU backends (their drivers never reach the device path),
    #    so the same block instance is correct on CPU and GPU. ──────────

    def configure_ere(
        mut self,
        enable: Bool = False,
        eta: Scalar[DT] = Scalar[DT](0.996),
        c_min: Int = 1,
        k_max: Int = 1000,
    ) raises:
        if not self.buf:
            if enable:
                raise Error(
                    "ReplaySampleStep.configure_ere: call setup() first"
                )
            return
        self.buf.value().configure_ere(
            enable=enable, eta=eta, c_min=c_min, k_max=k_max
        )

    def add_batch_gpu[N_ENVS: Int](
        mut self,
        ctx: DeviceContext,
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        self.buf.value().add_batch[N_ENVS](
            ctx, prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev,
        )

    def store_via_block_gpu[N_ENVS: Int, NS: Int](
        mut self,
        ctx: DeviceContext,
        mut nstep_buf: GPUNStepBuffer[NS, Self.OBS, Self.ACT, N_ENVS],
    ) raises:
        nstep_buf.store_into(ctx, self.buf.value())
