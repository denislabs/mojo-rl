"""NStepSampleGpuStep — n-step wrapper over UniformSampleGpuStep.

Single-env GPU n-step: uses the CPU `NStepBuffer` to accumulate
host-side, only pushing into the GPU replay (via inner.add) when the
ring emits. The GPU replay sees one-step transitions with γ^N-discounted
returns; trainer's target_y block applies the matching γ^N bootstrap.

Multi-env N_ENVS n-step lives in the driver layer (separate port,
uses GPUNStepBuffer for batched on-device ring buffering).
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from ...data.n_step_replay import NStepBuffer
from ..trainer_block import TrainerState
from .sample_block import SampleBlock
from .uniform_sample_gpu_step import UniformSampleGpuStep


struct NStepSampleGpuStep[
    N: Int,
    OBS_: Int, ACT_: Int, BATCH_: Int, CAP: Int,
](SampleBlock, Defaultable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime N_STEP = Self.N

    var inner: UniformSampleGpuStep[Self.OBS, Self.ACT, Self.BATCH, Self.CAP]
    var nstep:  NStepBuffer[Self.N, Self.OBS, Self.ACT]

    def __init__(out self):
        self.inner = UniformSampleGpuStep[
            Self.OBS, Self.ACT, Self.BATCH, Self.CAP,
        ]()
        self.nstep = NStepBuffer[Self.N, Self.OBS, Self.ACT].new()

    def setup(
        mut self,
        learning_starts: Int,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.inner.setup(learning_starts, ctx=ctx)

    def configure_gamma(mut self, gamma: Scalar[DT]):
        """Override the n-step discount used by NStepBuffer (caller
        passes the same γ as the trainer's target_y block, so the n-step
        return and bootstrap discount stay aligned)."""
        self.nstep.gamma = gamma

    def configure_ere(
        mut self,
        enable: Bool = False,
        eta: Scalar[DT] = Scalar[DT](0.996),
        c_min: Int = 1,
        k_max: Int = 1000,
    ) raises:
        self.inner.configure_ere(
            enable=enable, eta=eta, c_min=c_min, k_max=k_max,
        )

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
