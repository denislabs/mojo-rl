"""EnsembleCriticStep — N-critic gradient step against a shared target y.

Phase R.1 (CPU). Replaces SAC's `TwinCriticStep` for the REDQ ensemble:
loops over the N online critics in `CriticEnsemble[CRITIC, N]` and
runs one `CriticUpdateBlock.step` per critic against `state.mb_s`,
`state.mb_a`, `state.mb_y`. Sums the per-critic losses into
`state.critic_loss` (matches SAC's `loss1 + loss2` convention).

Design — ONE `CriticUpdateBlock` instance, reused per critic:
  SAC's `TwinCriticUpdateBlock` owns TWO `CriticUpdateBlock`s, each
  with its own `_mb_q`/`_mb_grad_q`/`_mb_grad_sa` scratch. For N=10
  ensembles that would 5× the per-block scratch with no functional
  win — the scratches are pure intermediates (overwritten each
  `CriticUpdateBlock.step` call, never read between calls). We hold
  ONE block and reuse it; each loop iteration overwrites the
  intermediates and `opt.step` consumes them before the next critic
  is touched.

Scratch ownership at this block level: only `_mb_sa`, the shared
concat(s, a) buffer. The N (critic, opt) pairs live in
`CriticEnsemble` (R.0) and are passed by `mut ensemble`.

R.1 is CPU-only. GPU comes with the full REDQTrainer GPU port (R.5
or later); the surface here is deliberately shaped so the GPU path
will just be a comptime-if branch in `step` and a `concat_sa_gpu`
swap, mirroring how SAC's TwinCriticUpdateBlock evolved.
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.scratch import Scratch
from mojo_rl.nn.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn.core.target_storage import TargetStorage, assert_tag_for

from ...loss.critic_update_block import CriticUpdateBlock
from ...training.off_policy_critic import concat_sa, concat_sa_gpu
from ...training.trainer_block import TrainerState
from ..ensemble import CriticEnsemble


struct EnsembleCriticStep[
    CRITIC: Module,
    N: Int,
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
](Defaultable & Movable & ImplicitlyDeletable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime SA_DIM = Self.OBS + Self.ACT

    var member_step: CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA_DIM]
    var _mb_sa: Scratch["mb_sa", Self.BATCH * Self.SA_DIM]
    var ts: TargetStorage

    def __init__(out self):
        self.member_step = CriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.SA_DIM,
        ]()
        self._mb_sa = Scratch["mb_sa", Self.BATCH * Self.SA_DIM]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """CPU + GPU factory. R.1 was CPU-only; R.5 adds the GPU branch
        (concat_sa_gpu + inner CriticUpdateBlock.step["gpu"], which is
        already GPU-capable from the SAC port)."""
        comptime assert target == "cpu" or target == "gpu", (
            "EnsembleCriticStep: target must be 'cpu' or 'gpu'"
        )
        var blk = Self()
        blk.member_step = CriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.SA_DIM,
        ].make[target](ctx=ctx)
        blk.ts = TargetStorage.make[target](ctx=ctx)
        init_scratch_auto[Self, target](blk, ctx)
        return blk^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut ensemble: CriticEnsemble[Self.CRITIC, Self.N],
    ) raises:
        """One ensemble-wide critic gradient step. Reads
        `state.mb_s` / `state.mb_a` / `state.mb_y`, writes
        `state.critic_loss = Σᵢ loss_i`."""
        assert_tag_for["EnsembleCriticStep", target](self.ts.target_tag)

        var sa_p = self._mb_sa.target_ptr[target]()
        comptime if target == "cpu":
            concat_sa[Self.OBS, Self.ACT, Self.BATCH](
                state.mb_s.target_ptr[target](),
                state.mb_a.target_ptr[target](),
                sa_p,
            )
        else:
            concat_sa_gpu[Self.OBS, Self.ACT, Self.BATCH](
                self.ts.ctx.value(),
                state.mb_s.target_ptr[target](),
                state.mb_a.target_ptr[target](),
                sa_p,
            )
        var sa_t = TileTensor(sa_p, row_major[Self.BATCH, Self.SA_DIM]())
        var mb_y_t = TileTensor(
            state.mb_y.target_ptr[target](),
            row_major[Self.BATCH, 1](),
        )

        var loss_sum: Scalar[DT] = Scalar[DT](0.0)
        for i in range(Self.N):
            var loss = self.member_step.step[target, POLICY](
                ensemble.pairs[i].online,
                ensemble.opts[i],
                sa_t,
                mb_y_t,
            )
            loss_sum += loss
        state.critic_loss = loss_sum
