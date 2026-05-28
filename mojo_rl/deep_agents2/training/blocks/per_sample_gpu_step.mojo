"""PerSampleGpuStep — GPU PER sampler, owns a GPUPrioritizedReplay.

Conforms to SampleBlock (Step 1 of SAC unification). The `setup` surface
mirrors uniform blocks (`learning_starts`, `ctx?`); PER-specific
hyperparameters (α / β / ε) live on the struct and are set before
`setup` is called (the trainer's `make` factory wires them via
field-default-overrides on the block instance).

Pre-step: stratified PER sampling populates `state.mb_*` and writes IS
weights into `state.mb_w`; sets `state.has_per = True` so downstream
critic blocks pick them up.

Post-step: trainer calls `update_priorities(state)` after the critic
step has captured signed TD residuals into `state.td_residuals`.

`set_beta(beta)` is the annealed-IS schedule hook callers ramp 0.4 → 1.0.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT
from ...data.per_replay import GPUPrioritizedReplay
from ...data.n_step_replay import GPUNStepBuffer
from ..trainer_block import TrainerState
from .sample_block import SampleBlock


struct PerSampleGpuStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, CAP: Int,
](SampleBlock, Defaultable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_

    var buf: Optional[GPUPrioritizedReplay[Self.OBS, Self.ACT, Self.CAP]]
    var learning_starts: Int

    # PER hyperparameters (callers can override before setup() if needed;
    # otherwise stays at Schaul defaults).
    var per_alpha:   Scalar[DT]
    var per_beta:    Scalar[DT]
    var per_epsilon: Scalar[DT]

    def __init__(out self):
        self.buf = None
        self.learning_starts = 0
        self.per_alpha   = Scalar[DT](0.6)
        self.per_beta    = Scalar[DT](0.4)
        self.per_epsilon = Scalar[DT](1e-6)

    def configure_per(
        mut self,
        alpha: Scalar[DT] = Scalar[DT](0.6),
        beta: Scalar[DT] = Scalar[DT](0.4),
        epsilon: Scalar[DT] = Scalar[DT](1e-6),
    ):
        """Override PER hyperparameters before calling setup()."""
        self.per_alpha = alpha
        self.per_beta = beta
        self.per_epsilon = epsilon

    def setup(
        mut self,
        learning_starts: Int,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        if not ctx:
            raise Error("PerSampleGpuStep.setup: ctx required for GPU")
        self.buf = GPUPrioritizedReplay[
            Self.OBS, Self.ACT, Self.CAP,
        ].new(
            ctx.value(), alpha=self.per_alpha,
            beta=self.per_beta, epsilon=self.per_epsilon,
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
        if not ctx:
            raise Error("PerSampleGpuStep.add: ctx required for GPU")
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
        if not self.buf.value().is_ready[Self.BATCH]():
            state.did_step = False
            return
        var ctx = state.ctx.value()
        self.buf.value().sample[Self.BATCH](
            ctx,
            state.mb_s.dev.value(),
            state.mb_a.dev.value(),
            state.mb_r.dev.value(),
            state.mb_sp.dev.value(),
            state.mb_d.dev.value(),
        )
        # After sample[BATCH], the PER buffer's `_host_weights` holds the
        # normalised IS weights for the sampled slice (length = BATCH).
        # H2D-copy them into state.mb_w so TwinCriticStep can pick them up
        # via state.has_per.
        ctx.enqueue_copy(
            state.mb_w.dev.value(),
            self.buf.value()._host_weights.unsafe_ptr(),
        )
        state.has_per = True

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

    def update_priorities(
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        """Called by the trainer at end of train_step. Pulls per-sample
        signed TD residuals (`Q1 − y`) from `state.td_residuals` and
        refreshes sum-tree leaves at the indices captured by the most
        recent `step()`. No-op when `state.did_step` is False (e.g.
        before `learning_starts`)."""
        if not state.did_step:
            return
        var ctx = state.ctx.value()
        self.buf.value().update_priorities[Self.BATCH](
            ctx, state.td_residuals.dev.value(),
        )
