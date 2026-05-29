"""NStepSampleStep[N, R, BATCH] — host n-step accumulator over any
`ReplayBuffer`-backed sample block.

Decorates `ReplaySampleStep[R, BATCH]` with an `NStepBuffer[N]` host
accumulator: `add` feeds each env-step transition through the ring and
only forwards a compressed n-step transition to the inner block when N
have accumulated (or on `done`). Everything else (setup / step /
configure_per / set_beta / update_priorities) forwards to the inner
block; `configure_gamma` sets the accumulator's discount.

Generic over `R`, so it subsumes BOTH n-step-over-uniform and
n-step-over-PER: `NStepSampleStep[N, AnyReplay[target, …], BATCH]` and
`NStepSampleStep[N, AnyPerReplay[target, …], BATCH]` (the latter is the
replay half of Rainbow). The trainer's target-Y bakes γ^N as the
bootstrap discount via its `nstep` param — keep `N` aligned with it.

The host ring drives single-env `add`; the device-side multi-env n-step
path (`GPUNStepBuffer`) is forwarded through `store_via_block_gpu` to the
inner block — `store_into` is now generic over `ReplayBuffer`, so the
same decorator subsumes the former GPU-only `NStepSampleGpuStep` /
`NStepPerSampleGpuStep`.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT
from ...data.n_step_replay import NStepBuffer, GPUNStepBuffer
from ..replay_buffer import ReplayBuffer
from ..trainer_block import TrainerState
from .sample_block import SampleBlock
from .replay_sample_step import ReplaySampleStep


struct NStepSampleStep[N: Int, R: ReplayBuffer, BATCH_: Int](
    SampleBlock, Defaultable
):
    comptime OBS = Self.R.OBS
    comptime ACT = Self.R.ACT
    comptime BATCH = Self.BATCH_
    comptime N_STEP = Self.N

    var inner: ReplaySampleStep[Self.R, Self.BATCH_]
    var nstep: NStepBuffer[Self.N, Self.R.OBS, Self.R.ACT]

    def __init__(out self):
        self.inner = ReplaySampleStep[Self.R, Self.BATCH_]()
        self.nstep = NStepBuffer[Self.N, Self.R.OBS, Self.R.ACT].new()

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

    def setup(
        mut self,
        learning_starts: Int,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.inner.setup(learning_starts, ctx=ctx)

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

    # ── GPU-batch hooks — forward to the inner block. The host n-step
    #    ring is bypassed on the device multi-env path (accumulation
    #    happens in the driver's GPUNStepBuffer); add_batch_gpu is the
    #    non-n-step direct device store. ────────────────────────────────

    def configure_ere(
        mut self,
        enable: Bool = False,
        eta: Scalar[DT] = Scalar[DT](0.996),
        c_min: Int = 1,
        k_max: Int = 1000,
    ) raises:
        self.inner.configure_ere(
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
        self.inner.add_batch_gpu[N_ENVS](
            ctx, prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev,
        )

    def store_via_block_gpu[N_ENVS: Int, NS: Int](
        mut self,
        ctx: DeviceContext,
        mut nstep_buf: GPUNStepBuffer[NS, Self.OBS, Self.ACT, N_ENVS],
    ) raises:
        self.inner.store_via_block_gpu[N_ENVS, NS](ctx, nstep_buf)
