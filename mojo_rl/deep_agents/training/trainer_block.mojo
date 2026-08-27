"""TrainerState — shared per-step flow object for ref-based block trainers.

History: this file originally defined both a `TrainerBlock` trait + a
`TrainerState` struct (J.1.a) used by the TrainerGraph[*BLOCKS] pipeline
walker. After the J.1.g-redesign-v2 port to ref-based block calls, the
trait surface and walker were deleted; only TrainerState survives as
the canonical inter-block flow container.

Held by every off-policy trainer (SAC/DDPG/TD3/MBPO/REDQ/DQN/C51) and passed
by `mut state` into every block's `step[target]` method. Carries:
  - the canonical minibatch six-pack (mb_s/a/r/sp/d/y)
  - inter-block scalars (alpha, log_prob_mean, critic_loss, actor_loss)
  - step bookkeeping (step_idx, ctx, did_step)

STORAGE migration (Stage 5): the minibatch buffers are owned `nn.storage`
`Tensor`s (was legacy `Scratch`). Each holds a CPU `List` + optional GPU
`DeviceBuffer`; the replay `sample_into(state)` fills them, and blocks build
their typed views internally (`mb.lt[target, layout]()`) and pass `mb` by ref
into the storage Module `forward`/`vjp` via `TensorRefs`. `make[target]`
allocates on the chosen target (CPU list / GPU buffer).
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor


struct TrainerState[
    OBS: Int,
    ACT: Int,
    BATCH: Int,
](Defaultable & Movable & Deinitable):
    # Canonical minibatch six-pack (owned storage Tensors).
    var mb_s:  Tensor   # [BATCH * OBS]
    var mb_a:  Tensor   # [BATCH * ACT]
    var mb_r:  Tensor   # [BATCH]
    var mb_sp: Tensor   # [BATCH * OBS]
    var mb_d:  Tensor   # [BATCH]
    var mb_y:  Tensor   # [BATCH]

    # PER hooks. `mb_w` carries per-sample IS weights into the critic update;
    # `td_residuals` carries per-sample `Q1(s,a) − y` back out of the critic
    # update so the sampler block can refresh sum-tree priorities. Both are
    # always allocated (BATCH each) so opt-in PER trainers can read/write them
    # without conditional allocation; uniform trainers simply ignore them.
    var mb_w:         Tensor   # [BATCH]
    var td_residuals: Tensor   # [BATCH]
    var has_per: Bool

    var alpha:         Scalar[DT]
    var log_prob_mean: Scalar[DT]
    var critic_loss:   Scalar[DT]
    var actor_loss:    Scalar[DT]

    var step_idx: Int
    var ctx:      Optional[DeviceContext]
    var did_step: Bool

    def __init__(out self):
        self.mb_s  = Tensor()
        self.mb_a  = Tensor()
        self.mb_r  = Tensor()
        self.mb_sp = Tensor()
        self.mb_d  = Tensor()
        self.mb_y  = Tensor()
        self.mb_w         = Tensor()
        self.td_residuals = Tensor()
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
        # Unified allocator: `make[target]` dispatches alloc/alloc_gpu (and
        # raises if gpu w/o ctx), so the CPU and GPU minibatch staging buffers
        # collapse into ONE branch-free block — the CPU/GPU-path unification.
        s.mb_s  = Tensor.make[target](Self.BATCH * Self.OBS, ctx)
        s.mb_a  = Tensor.make[target](Self.BATCH * Self.ACT, ctx)
        s.mb_r  = Tensor.make[target](Self.BATCH, ctx)
        s.mb_sp = Tensor.make[target](Self.BATCH * Self.OBS, ctx)
        s.mb_d  = Tensor.make[target](Self.BATCH, ctx)
        s.mb_y  = Tensor.make[target](Self.BATCH, ctx)
        s.mb_w         = Tensor.make[target](Self.BATCH, ctx)
        s.td_residuals = Tensor.make[target](Self.BATCH, ctx)
        comptime if target == "gpu":
            s.ctx = ctx
        return s^
