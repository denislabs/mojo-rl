"""UniformSampleGpuStep — GPU variant of UniformSampleCpuStep.

Conforms to SampleBlock (Step 1 of SAC unification). `ctx` is required
on `setup`/`add` — raises if None. Internal H2D conversion (List →
UnsafePointer for GPUReplay.add) moved here from the trainer's record
path so the SampleBlock surface stays uniform across CPU/GPU.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT
from ...data.gpu_replay import GPUReplay
from ...data.n_step_replay import GPUNStepBuffer
from ..trainer_block import TrainerState
from .sample_block import SampleBlock


struct UniformSampleGpuStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, CAP: Int,
](SampleBlock, Defaultable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_

    var buf: Optional[GPUReplay[Self.OBS, Self.ACT, Self.CAP]]
    var learning_starts: Int

    def __init__(out self):
        self.buf = None
        self.learning_starts = 0

    def setup(
        mut self,
        learning_starts: Int,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        if not ctx:
            raise Error("UniformSampleGpuStep.setup: ctx required for GPU")
        self.buf = GPUReplay[Self.OBS, Self.ACT, Self.CAP].new(
            ctx.value(), batch_capacity=Self.BATCH
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
        if not ctx:
            raise Error("UniformSampleGpuStep.add: ctx required for GPU")
        var obs_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            obs.unsafe_ptr()
        )
        var act_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            action.unsafe_ptr()
        )
        var nxt_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            next_obs.unsafe_ptr()
        )
        self.buf.value().add(ctx.value(), obs_p, act_p, reward, nxt_p, done)

    def step(
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        if state.step_idx < self.learning_starts:
            state.did_step = False
            return
        if self.buf.value().size < Self.BATCH:
            state.did_step = False
            return
        self.buf.value().sample[Self.BATCH](
            state.ctx.value(),
            state.mb_s.dev.value(),
            state.mb_a.dev.value(),
            state.mb_r.dev.value(),
            state.mb_sp.dev.value(),
            state.mb_d.dev.value(),
        )

    def configure_ere(
        mut self,
        enable: Bool = False,
        eta: Scalar[DT] = Scalar[DT](0.996),
        c_min: Int = 1,
        k_max: Int = 1000,
    ) raises:
        """Delegate to GPUReplay.enable_ere. Must be called AFTER setup()
        (which creates self.buf). When `enable=False`, no-op (default
        uniform sampling stays in effect)."""
        if not enable:
            return
        if not self.buf:
            raise Error(
                "UniformSampleGpuStep.configure_ere: call setup() first"
            )
        self.buf.value().enable_ere(eta=eta, c_min=c_min, k_max=k_max)

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
            ctx,
            prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev,
        )

    def store_via_block_gpu[N_ENVS: Int, NS: Int](
        mut self,
        ctx: DeviceContext,
        mut nstep_buf: GPUNStepBuffer[NS, Self.OBS, Self.ACT, N_ENVS],
    ) raises:
        nstep_buf.store_into[Self.CAP](ctx, self.buf.value())
