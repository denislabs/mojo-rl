"""TrainerState — shared per-step flow object for ref-based block trainers.

History: this file originally defined both a `TrainerBlock` trait + a
`TrainerState` struct (J.1.a) used by the TrainerGraph[*BLOCKS] pipeline
walker. After the J.1.g-redesign-v2 port to ref-based block calls, the
trait surface and walker were deleted; only TrainerState survives as
the canonical inter-block flow container.

Held by every trainer (SAC/SAC-GPU/DDPG/TD3/MBPO) and passed by
`mut state` into every block's `step[target]` method. Carries:
  - the canonical minibatch six-pack (mb_s/a/r/sp/d/y)
  - inter-block scalars (alpha, log_prob_mean, critic_loss, actor_loss)
  - step bookkeeping (step_idx, ctx, did_step)
"""

from std.gpu.host import DeviceContext

from ..constants import DT
from ..core.scratch import Scratch


struct TrainerState[
    OBS: Int,
    ACT: Int,
    BATCH: Int,
](Defaultable & Movable & ImplicitlyDestructible):
    var mb_s:  Scratch["mb_s",  Self.BATCH * Self.OBS, True]
    var mb_a:  Scratch["mb_a",  Self.BATCH * Self.ACT, True]
    var mb_r:  Scratch["mb_r",  Self.BATCH, True]
    var mb_sp: Scratch["mb_sp", Self.BATCH * Self.OBS, True]
    var mb_d:  Scratch["mb_d",  Self.BATCH, True]
    var mb_y:  Scratch["mb_y",  Self.BATCH, True]

    # PER hooks. `mb_w` carries per-sample IS weights into the critic update;
    # `td_residuals` carries per-sample `Q1(s,a) − y` back out of the critic
    # update so the sampler block can refresh sum-tree priorities. Both are
    # always allocated (BATCH * 4 B each) so opt-in PER trainers can read/write
    # them without conditional allocation; uniform trainers simply ignore them.
    var mb_w:         Scratch["mb_w",         Self.BATCH, True]
    var td_residuals: Scratch["td_residuals", Self.BATCH, True]
    var has_per: Bool

    var alpha:         Scalar[DT]
    var log_prob_mean: Scalar[DT]
    var critic_loss:   Scalar[DT]
    var actor_loss:    Scalar[DT]

    var step_idx: Int
    var ctx:      Optional[DeviceContext]
    var did_step: Bool

    def __init__(out self):
        self.mb_s  = Scratch["mb_s",  Self.BATCH * Self.OBS, True]()
        self.mb_a  = Scratch["mb_a",  Self.BATCH * Self.ACT, True]()
        self.mb_r  = Scratch["mb_r",  Self.BATCH, True]()
        self.mb_sp = Scratch["mb_sp", Self.BATCH * Self.OBS, True]()
        self.mb_d  = Scratch["mb_d",  Self.BATCH, True]()
        self.mb_y  = Scratch["mb_y",  Self.BATCH, True]()
        self.mb_w         = Scratch["mb_w",         Self.BATCH, True]()
        self.td_residuals = Scratch["td_residuals", Self.BATCH, True]()
        self.has_per = False
        self.alpha = Scalar[DT](0.0)
        self.log_prob_mean = Scalar[DT](0.0)
        self.critic_loss = Scalar[DT](0.0)
        self.actor_loss  = Scalar[DT](0.0)
        self.step_idx = 0
        self.ctx = None
        self.did_step = True

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "TrainerState: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        comptime if target == "cpu":
            s.mb_s  = Scratch["mb_s",  Self.BATCH * Self.OBS, True].make_cpu()
            s.mb_a  = Scratch["mb_a",  Self.BATCH * Self.ACT, True].make_cpu()
            s.mb_r  = Scratch["mb_r",  Self.BATCH, True].make_cpu()
            s.mb_sp = Scratch["mb_sp", Self.BATCH * Self.OBS, True].make_cpu()
            s.mb_d  = Scratch["mb_d",  Self.BATCH, True].make_cpu()
            s.mb_y  = Scratch["mb_y",  Self.BATCH, True].make_cpu()
            s.mb_w         = Scratch["mb_w",         Self.BATCH, True].make_cpu()
            s.td_residuals = Scratch["td_residuals", Self.BATCH, True].make_cpu()
        else:
            if not ctx:
                raise Error("TrainerState.make[target='gpu']: ctx required")
            var ctx_v = ctx.value()
            s.mb_s  = Scratch["mb_s",  Self.BATCH * Self.OBS, True].make_gpu(ctx_v)
            s.mb_a  = Scratch["mb_a",  Self.BATCH * Self.ACT, True].make_gpu(ctx_v)
            s.mb_r  = Scratch["mb_r",  Self.BATCH, True].make_gpu(ctx_v)
            s.mb_sp = Scratch["mb_sp", Self.BATCH * Self.OBS, True].make_gpu(ctx_v)
            s.mb_d  = Scratch["mb_d",  Self.BATCH, True].make_gpu(ctx_v)
            s.mb_y  = Scratch["mb_y",  Self.BATCH, True].make_gpu(ctx_v)
            s.mb_w         = Scratch["mb_w",         Self.BATCH, True].make_gpu(ctx_v)
            s.td_residuals = Scratch["td_residuals", Self.BATCH, True].make_gpu(ctx_v)
            s.ctx = ctx_v
        return s^
