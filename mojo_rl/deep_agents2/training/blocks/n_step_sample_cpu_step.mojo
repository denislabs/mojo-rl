"""NStepSampleCpuStep — n-step wrapper over UniformSampleCpuStep.

`add()` accumulates env-step transitions through an internal
`NStepBuffer[N]`; only emits to the inner uniform block when the ring
fills (N transitions accumulated) or `done`. `step()` / `setup()` /
`configure_*()` delegate to the inner block unchanged.

The trainer's `target_y` block uses γ^N as the bootstrap discount,
comptime-baked via the trainer's `N_STEP` param — caller's
responsibility to keep `N_STEP` here aligned with that.

The inner block is a `UniformSampleCpuStep`; PER+n-step combination
would need its own wrapper (or a generic NStepSample wrapping any
inner SampleBlock — punted, no caller yet).
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from ...data.n_step_replay import NStepBuffer
from ..trainer_block import TrainerState
from .sample_block import SampleBlock
from .uniform_sample_cpu_step import UniformSampleCpuStep


struct NStepSampleCpuStep[
    N: Int,
    OBS_: Int, ACT_: Int, BATCH_: Int, CAP: Int,
](SampleBlock, Defaultable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime N_STEP = Self.N

    var inner: UniformSampleCpuStep[Self.OBS, Self.ACT, Self.BATCH, Self.CAP]
    var nstep:  NStepBuffer[Self.N, Self.OBS, Self.ACT]

    def __init__(out self):
        self.inner = UniformSampleCpuStep[
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
        """Override the n-step discount used by NStepBuffer (caller passes
        the same γ as the trainer's target_y block, so the n-step return
        and bootstrap discount stay aligned)."""
        self.nstep.gamma = gamma

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
        # Convert InlineArray → List for the inner.add surface.
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
