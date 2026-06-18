"""DualSampleStep — MBPO mixed-batch sampler (CPU + GPU, dual storage).

Owns BOTH a real replay buffer AND a synthetic replay buffer, and carries
BOTH backends (host `CPUReplay` + device `Optional[GPUReplay]`) in one
struct so the MBPO trainer can hold a single concrete field type for both
`train_target`s — sidestepping the "ternary over two struct types"
limitation (a `T_gpu if cond else T_cpu` alias does not type-unify). Only
the matching backend is allocated in `setup[target]`; the other stays a
zero-cost placeholder (CPU: null pointers, size 0; GPU: `None`). This is
the documented nn target-unification pattern (see
`feedback_mojo_nn2_target_unification`).

Each step gathers REAL_BS transitions from the real buffer into the first
REAL_BS rows of `state.mb_*`, and SYNTH_BS from the synth buffer into the
rest. On GPU the synth partition lands at the REAL_BS offset via non-owning
`DeviceBuffer` views (`GPUReplay.sample[N]` gathers rows `[0, N)`).

Real transitions arrive host-side via `real_add` (CPU: direct; GPU: H2D);
synthetic transitions arrive host-side via `synth_add` (CPU rollout) or
device-batched via `synth_add_batch` (GPU rollout).
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import DT
from ...data.cpu_replay import CPUReplay
from ...data.gpu_replay import GPUReplay
from ..trainer_block import TrainerState


struct DualSampleStep[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    REAL_CAP: Int,
    SYNTH_CAP: Int,
    REAL_BS: Int,
    SYNTH_BS: Int,
](Defaultable & Movable & ImplicitlyDeletable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_

    var real_cpu: Optional[CPUReplay[Self.OBS, Self.ACT, Self.REAL_CAP]]
    var synth_cpu: Optional[CPUReplay[Self.OBS, Self.ACT, Self.SYNTH_CAP]]
    var real_gpu: Optional[GPUReplay[Self.OBS, Self.ACT, Self.REAL_CAP]]
    var synth_gpu: Optional[GPUReplay[Self.OBS, Self.ACT, Self.SYNTH_CAP]]
    var learning_starts: Int

    def __init__(out self):
        self.real_cpu = None
        self.synth_cpu = None
        self.real_gpu = None
        self.synth_gpu = None
        self.learning_starts = 0

    def setup[target: StaticString](
        mut self, learning_starts: Int, ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            self.real_cpu = CPUReplay[Self.OBS, Self.ACT, Self.REAL_CAP].new()
            self.synth_cpu = CPUReplay[Self.OBS, Self.ACT, Self.SYNTH_CAP].new()
        else:
            var c = ctx.value()
            self.real_gpu = GPUReplay[Self.OBS, Self.ACT, Self.REAL_CAP].new(
                c, batch_capacity=Self.BATCH,
            )
            self.synth_gpu = GPUReplay[Self.OBS, Self.ACT, Self.SYNTH_CAP].new(
                c, batch_capacity=Self.BATCH,
            )
        self.learning_starts = learning_starts

    def real_add[target: StaticString](
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            self.real_cpu.value().add(obs, action, reward, next_obs, done)
        else:
            self.real_gpu.value().add(
                obs, action, reward, next_obs, done, ctx=ctx,
            )

    def synth_add(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ):
        """CPU rollout synthetic store (host list)."""
        self.synth_cpu.value().add(obs, action, reward, next_obs, done)

    def synth_add_batch[
        N: Int
    ](
        mut self,
        ctx: DeviceContext,
        src_obs: DeviceBuffer[DT],
        src_act: DeviceBuffer[DT],
        src_rew: DeviceBuffer[DT],
        src_nxt: DeviceBuffer[DT],
        src_dne: DeviceBuffer[DT],
    ) raises:
        """GPU rollout synthetic store (one device batch)."""
        self.synth_gpu.value().add_batch[N](
            ctx, src_obs, src_act, src_rew, src_nxt, src_dne,
        )

    def real_sample[
        N: Int
    ](
        mut self,
        ctx: DeviceContext,
        mb_s: DeviceBuffer[DT],
        mb_a: DeviceBuffer[DT],
        mb_r: DeviceBuffer[DT],
        mb_sp: DeviceBuffer[DT],
        mb_d: DeviceBuffer[DT],
    ) raises:
        """Draw N transitions from the real buffer into caller device
        buffers (GPU dynamics-train bootstrap + rollout start states)."""
        self.real_gpu.value().sample[N](ctx, mb_s, mb_a, mb_r, mb_sp, mb_d)

    def real_sample_range[
        N: Int
    ](
        mut self,
        ctx: DeviceContext,
        lo: Int,
        hi: Int,
        mb_s: DeviceBuffer[DT],
        mb_a: DeviceBuffer[DT],
        mb_r: DeviceBuffer[DT],
        mb_sp: DeviceBuffer[DT],
        mb_d: DeviceBuffer[DT],
    ) raises:
        """Draw N transitions from the real-buffer index range `[lo, hi)` —
        the MBPO dyn-train train/holdout split."""
        self.real_gpu.value().sample_range[N](
            ctx, lo, hi, mb_s, mb_a, mb_r, mb_sp, mb_d
        )

    def real_count[target: StaticString](self) -> Int:
        comptime if target == "cpu":
            if not self.real_cpu:
                return 0
            return self.real_cpu.value().size
        else:
            if not self.real_gpu:
                return 0
            return self.real_gpu.value().count()

    def synth_count[target: StaticString](self) -> Int:
        comptime if target == "cpu":
            if not self.synth_cpu:
                return 0
            return self.synth_cpu.value().size
        else:
            if not self.synth_gpu:
                return 0
            return self.synth_gpu.value().count()

    def step[target: StaticString](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        comptime assert (
            Self.REAL_BS + Self.SYNTH_BS == Self.BATCH
        ), "DualSampleStep: REAL_BS + SYNTH_BS must equal BATCH"
        if state.step_idx < self.learning_starts:
            state.did_step = False
            return

        comptime if target == "cpu":
            if self.real_cpu.value().size < Self.REAL_BS:
                state.did_step = False
                return
            if self.synth_cpu.value().size < Self.SYNTH_BS:
                state.did_step = False
                return
            # Real partition → rows [0, REAL_BS); synth → [REAL_BS, BATCH).
            # `row_offset` stacks both into the same `state.mb_*.cpu` lists
            # (replaces the old pointer-offset writes).
            self.real_cpu.value().sample(
                Self.REAL_BS,
                state.mb_s.cpu,
                state.mb_a.cpu,
                state.mb_r.cpu,
                state.mb_sp.cpu,
                state.mb_d.cpu,
            )
            self.synth_cpu.value().sample(
                Self.SYNTH_BS,
                state.mb_s.cpu,
                state.mb_a.cpu,
                state.mb_r.cpu,
                state.mb_sp.cpu,
                state.mb_d.cpu,
                row_offset=Self.REAL_BS,
            )
        else:
            if self.real_gpu.value().count() < Self.REAL_BS:
                state.did_step = False
                return
            if self.synth_gpu.value().count() < Self.SYNTH_BS:
                state.did_step = False
                return
            var ctx = state.ctx.value()
            # Real partition → rows [0, REAL_BS).
            self.real_gpu.value().sample[Self.REAL_BS](
                ctx,
                state.mb_s.dev.value(),
                state.mb_a.dev.value(),
                state.mb_r.dev.value(),
                state.mb_sp.dev.value(),
                state.mb_d.dev.value(),
            )
            # Synth partition → rows [REAL_BS, BATCH) via offset views.
            var s_off = state.mb_s.dev.value().unsafe_ptr() + Self.REAL_BS * Self.OBS
            var a_off = state.mb_a.dev.value().unsafe_ptr() + Self.REAL_BS * Self.ACT
            var r_off = state.mb_r.dev.value().unsafe_ptr() + Self.REAL_BS
            var sp_off = state.mb_sp.dev.value().unsafe_ptr() + Self.REAL_BS * Self.OBS
            var d_off = state.mb_d.dev.value().unsafe_ptr() + Self.REAL_BS
            var vs = DeviceBuffer[DT](ctx, s_off, Self.SYNTH_BS * Self.OBS, owning=False)
            var va = DeviceBuffer[DT](ctx, a_off, Self.SYNTH_BS * Self.ACT, owning=False)
            var vr = DeviceBuffer[DT](ctx, r_off, Self.SYNTH_BS, owning=False)
            var vsp = DeviceBuffer[DT](ctx, sp_off, Self.SYNTH_BS * Self.OBS, owning=False)
            var vd = DeviceBuffer[DT](ctx, d_off, Self.SYNTH_BS, owning=False)
            self.synth_gpu.value().sample[Self.SYNTH_BS](ctx, vs, va, vr, vsp, vd)
