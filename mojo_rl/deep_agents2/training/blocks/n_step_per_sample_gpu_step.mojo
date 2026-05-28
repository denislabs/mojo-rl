"""NStepPerSampleGpuStep — n-step + PER on GPU.

GPU sibling of `NStepPerSampleCpuStep`. Wraps a `PerSampleGpuStep`
(host-emitted n-step transitions go straight into the on-device PER
buffer via inner.add). All sample/update/IS-weight bookkeeping happens
in the inner block; this struct only owns the host-side NStepBuffer.

Required by Rainbow 6/6 on GPU (PER + N-step both active).

The trainer's target-Y block bakes γ^N as the bootstrap discount via
its `nstep` param — caller's responsibility to keep `N` here aligned
with the trainer's `nstep`.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT
from ...data.n_step_replay import NStepBuffer, GPUNStepBuffer
from ..trainer_block import TrainerState
from .sample_block import SampleBlock
from .per_sample_gpu_step import PerSampleGpuStep


struct NStepPerSampleGpuStep[
    N: Int,
    OBS_: Int, ACT_: Int, BATCH_: Int, CAP: Int,
](SampleBlock, Defaultable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime N_STEP = Self.N

    var inner: PerSampleGpuStep[Self.OBS, Self.ACT, Self.BATCH, Self.CAP]
    var nstep: NStepBuffer[Self.N, Self.OBS, Self.ACT]

    def __init__(out self):
        self.inner = PerSampleGpuStep[
            Self.OBS, Self.ACT, Self.BATCH, Self.CAP,
        ]()
        self.nstep = NStepBuffer[Self.N, Self.OBS, Self.ACT].new()

    def setup(
        mut self,
        learning_starts: Int,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.inner.setup(learning_starts, ctx=ctx)

    def configure_per(
        mut self,
        alpha: Scalar[DT] = Scalar[DT](0.6),
        beta: Scalar[DT] = Scalar[DT](0.4),
        epsilon: Scalar[DT] = Scalar[DT](1e-6),
    ):
        self.inner.configure_per(alpha=alpha, beta=beta, epsilon=epsilon)

    def configure_gamma(mut self, gamma: Scalar[DT]):
        self.nstep.gamma = gamma

    def set_beta(mut self, beta: Scalar[DT]):
        self.inner.set_beta(beta)

    def add(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        var done_b = done > Scalar[DT](0.5)
        var t = self.nstep.add(obs, action, reward, next_obs, done_b)
        if not t.valid:
            return
        var s0 = List[Scalar[DT]](length=Self.OBS, fill=Scalar[DT](0.0))
        var a0 = List[Scalar[DT]](length=Self.ACT, fill=Scalar[DT](0.0))
        var sn = List[Scalar[DT]](length=Self.OBS, fill=Scalar[DT](0.0))
        for d in range(Self.OBS):
            s0[d] = t.obs[d]
            sn[d] = t.next_obs[d]
        for j in range(Self.ACT):
            a0[j] = t.action[j]
        var done_f = Scalar[DT](1.0) if t.done else Scalar[DT](0.0)
        self.inner.add(s0, a0, t.reward, sn, done_f, ctx=ctx)

    def step(
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        self.inner.step(state)

    def update_priorities(
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        self.inner.update_priorities(state)

    # ── Cross-target hooks (delegate to inner) ─────────────────────

    def add_batch_gpu[N_ENVS: Int](
        mut self,
        ctx: DeviceContext,
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        # Multi-env n-step accumulation lives in the driver's
        # GPUNStepBuffer path (`store_via_block_gpu`). The single-env
        # `add` above handles N_ENVS=1 record_batch_cpu callers.
        self.inner.add_batch_gpu[N_ENVS](
            ctx, prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev,
        )

    def store_via_block_gpu[N_ENVS: Int, NS: Int](
        mut self,
        ctx: DeviceContext,
        mut nstep_buf: GPUNStepBuffer[NS, Self.OBS, Self.ACT, N_ENVS],
    ) raises:
        self.inner.store_via_block_gpu[N_ENVS, NS](ctx, nstep_buf)
