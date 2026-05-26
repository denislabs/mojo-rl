"""J.1.a — TrainerBlock trait + TrainerState struct.

Mirrors `LossBlock` / `GraphNode` but operates on a flowing TrainerState
instead of returning a loss. Each block reads inter-block data from
state, mutates models (held via typed UnsafePointers), and writes back
scalars (loss, log_prob_mean) into state.

Per J.1.a decisions:
  - Q1(b): SampleBlock owns the replay buffer. Trainer's `record(...)`
    delegates to `self.blocks[0].add(...)`.
  - Q2: `state.did_step = False` from any block short-circuits the walker.
  - Q3: Trainer is pinned (no __moveinit__) so model pointers held by
    blocks stay valid for the trainer's lifetime.
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
        self.alpha = Scalar[DT](0.0)
        self.log_prob_mean = Scalar[DT](0.0)
        self.critic_loss = Scalar[DT](0.0)
        self.actor_loss  = Scalar[DT](0.0)
        self.step_idx = 0
        self.ctx = None
        self.did_step = True

    @staticmethod
    def make_cpu() raises -> Self:
        var s = Self()
        s.mb_s  = Scratch["mb_s",  Self.BATCH * Self.OBS, True].make_cpu()
        s.mb_a  = Scratch["mb_a",  Self.BATCH * Self.ACT, True].make_cpu()
        s.mb_r  = Scratch["mb_r",  Self.BATCH, True].make_cpu()
        s.mb_sp = Scratch["mb_sp", Self.BATCH * Self.OBS, True].make_cpu()
        s.mb_d  = Scratch["mb_d",  Self.BATCH, True].make_cpu()
        s.mb_y  = Scratch["mb_y",  Self.BATCH, True].make_cpu()
        return s^

    @staticmethod
    def make_gpu(ctx: DeviceContext) raises -> Self:
        var s = Self()
        s.mb_s  = Scratch["mb_s",  Self.BATCH * Self.OBS, True].make_gpu(ctx)
        s.mb_a  = Scratch["mb_a",  Self.BATCH * Self.ACT, True].make_gpu(ctx)
        s.mb_r  = Scratch["mb_r",  Self.BATCH, True].make_gpu(ctx)
        s.mb_sp = Scratch["mb_sp", Self.BATCH * Self.OBS, True].make_gpu(ctx)
        s.mb_d  = Scratch["mb_d",  Self.BATCH, True].make_gpu(ctx)
        s.mb_y  = Scratch["mb_y",  Self.BATCH, True].make_gpu(ctx)
        s.ctx = ctx
        return s^


trait TrainerBlock(Defaultable & Movable & ImplicitlyDestructible):
    """One composable step in a training iteration. Reads/mutates
    flow state via TrainerState. Models are held via typed
    UnsafePointers set at trainer init (see e.g. TargetYStepBlock.bind).

    Each block declares OBS/ACT/BATCH comptime members so the trait
    signature matches the trainer's TrainerState shape. All blocks in
    a single TrainerGraph must share the same (OBS, ACT, BATCH) — this
    is enforced by TrainerGraph's comptime asserts.
    """
    comptime OBS: Int
    comptime ACT: Int
    comptime BATCH: Int

    def step_via[target: StaticString = "cpu"](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        ...
