"""CriticUpdateBlock / TwinCriticUpdateBlock — Phase 10F.

Self-contained critic-update blocks that absorb the scratch buffers
(`mb_q`, `mb_grad_q`, `mb_grad_sa`) the trainer would otherwise own.
Mirrors the SACActorLossCG (Phase 10E) pattern: block owns its
intermediates, public `step` method does forward/MSE/backward/opt-step.

Why "block" not "graph" — the critic-update chain is a *linear* MSE
regression (one Module forward, one Loss, no fan-out, no Slice/Min).
ComputeGraph v2's name-resolution machinery would add overhead without
benefit. The win here is **scratch ownership**, not DAG topology.

Free helpers `critic_update_step` / `twin_critic_update_step` in
`training/off_policy_critic.mojo` remain available for prototyping
new algorithms that don't want the block plumbing. SACTrainer migrates
to the block.

CPU only (Phase 10F). GPU lands together with CG v2's GPU path.

Surface:
    CriticUpdateBlock[CRITIC, BATCH, SA_DIM]
        - `make[target]() raises -> Self`
        - `step[target](mut critic, mut opt, sa_t, y_t) raises -> Scalar[DT]`
            zero_grad → critic.forward → mse.forward → mse.backward →
            critic.backward → opt.step; returns scalar loss.

    TwinCriticUpdateBlock[CRITIC, BATCH, OBS, ACT]
        - owns 2× CriticUpdateBlock + `_mb_sa` scratch
        - `make[target]() raises -> Self`
        - `step[target](mut c1, mut c1_opt, mut c2, mut c2_opt,
                        mb_s_ptr, mb_a_ptr, mb_y_t) raises -> Scalar[DT]`
            concat_sa → c1.step + c2.step; returns sum of losses.
"""

from layout import TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import (
    Module,
    TARGET_UNINIT,
    TARGET_CPU,
    target_tag_for,
)
from ..optimizer.adam import Adam
from .mse import MSELoss
from ..training.off_policy_critic import concat_sa


struct CriticUpdateBlock[
    CRITIC: Module,
    BATCH: Int,
    SA_DIM: Int,
](Movable & ImplicitlyDestructible):
    """Single-critic MSE update step. Owns all intermediate scratch."""

    var mse_loss: MSELoss[1]
    var _mb_q: List[Scalar[DT]]          # [BATCH, 1]  critic forward output
    var _mb_grad_q: List[Scalar[DT]]     # [BATCH, 1]  MSE backward output
    var _mb_grad_sa: List[Scalar[DT]]    # [BATCH, SA_DIM]  critic backward output (discarded)
    var _target_tag: Int8

    def __init__(out self):
        self.mse_loss = MSELoss[1]()
        self._mb_q = List[Scalar[DT]]()
        self._mb_grad_q = List[Scalar[DT]]()
        self._mb_grad_sa = List[Scalar[DT]]()
        self._target_tag = TARGET_UNINIT

    @staticmethod
    def make[target: StaticString]() raises -> Self:
        comptime assert target == "cpu", (
            "CriticUpdateBlock.make[target='gpu'] not yet implemented (Phase 10F CPU only)"
        )
        comptime assert Self.CRITIC.IN_DIM == Self.SA_DIM, (
            "CriticUpdateBlock: CRITIC.IN_DIM must equal SA_DIM"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "CriticUpdateBlock: CRITIC.OUT_DIM must equal 1"
        )
        var blk = Self()
        blk.mse_loss = MSELoss[1].make[target="cpu"]()
        var zero: Scalar[DT] = 0.0
        blk._mb_q.resize(Self.BATCH, zero)
        blk._mb_grad_q.resize(Self.BATCH, zero)
        blk._mb_grad_sa.resize(Self.BATCH * Self.SA_DIM, zero)
        blk._target_tag = TARGET_CPU
        return blk^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "CriticUpdateBlock: method called with [target='"
                + String(target)
                + "'] but block was make'd for a different target (tag="
                + String(Int(self._target_tag)) + ")"
            )

    def step[
        target: StaticString,
        LSA: TensorLayout, LY: TensorLayout,
        OSA: MutOrigin, OY: MutOrigin,
    ](
        mut self,
        mut critic: Self.CRITIC,
        mut opt: Adam,
        sa_t: TileTensor[DT, LSA, OSA],
        y_t: TileTensor[DT, LY, OY],
    ) raises -> Scalar[DT]:
        comptime assert target == "cpu", (
            "CriticUpdateBlock.step: GPU path not yet implemented"
        )
        self._assert_tag[target]()

        var mb_q_t = TileTensor(
            self._mb_q.unsafe_ptr(), row_major[Self.BATCH, 1]()
        )
        opt.zero_grad["cpu", M=Self.CRITIC](critic)
        critic.forward["cpu", Self.BATCH](sa_t, mb_q_t)
        var loss = self.mse_loss.forward["cpu", Self.BATCH](mb_q_t, y_t)

        var mb_grad_q_t = TileTensor(
            self._mb_grad_q.unsafe_ptr(), row_major[Self.BATCH, 1]()
        )
        self.mse_loss.backward["cpu", Self.BATCH](y_t, mb_grad_q_t)

        var mb_grad_sa_t = TileTensor(
            self._mb_grad_sa.unsafe_ptr(), row_major[Self.BATCH, Self.SA_DIM]()
        )
        critic.backward["cpu", Self.BATCH](mb_grad_q_t, mb_grad_sa_t)
        opt.step["cpu", M=Self.CRITIC](critic)
        return loss


struct TwinCriticUpdateBlock[
    CRITIC: Module,
    BATCH: Int,
    OBS: Int,
    ACT: Int,
](Movable & ImplicitlyDestructible):
    """Twin-critic update against shared target `y`. Owns two
    `CriticUpdateBlock`s + a shared `_mb_sa` scratch."""

    comptime SA_DIM = Self.OBS + Self.ACT

    var c1: CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA_DIM]
    var c2: CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA_DIM]
    var _mb_sa: List[Scalar[DT]]  # [BATCH, SA_DIM]  concat(s, a)
    var _target_tag: Int8

    def __init__(out self):
        self.c1 = CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA_DIM]()
        self.c2 = CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA_DIM]()
        self._mb_sa = List[Scalar[DT]]()
        self._target_tag = TARGET_UNINIT

    @staticmethod
    def make[target: StaticString]() raises -> Self:
        comptime assert target == "cpu", (
            "TwinCriticUpdateBlock.make[target='gpu'] not yet implemented (Phase 10F CPU only)"
        )
        var blk = Self()
        blk.c1 = CriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.SA_DIM
        ].make[target="cpu"]()
        blk.c2 = CriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.SA_DIM
        ].make[target="cpu"]()
        blk._mb_sa.resize(Self.BATCH * Self.SA_DIM, Scalar[DT](0.0))
        blk._target_tag = TARGET_CPU
        return blk^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "TwinCriticUpdateBlock: method called with [target='"
                + String(target)
                + "'] but block was make'd for a different target (tag="
                + String(Int(self._target_tag)) + ")"
            )

    def step[
        target: StaticString,
        LY: TensorLayout, OY: MutOrigin,
    ](
        mut self,
        mut critic1: Self.CRITIC,
        mut critic1_opt: Adam,
        mut critic2: Self.CRITIC,
        mut critic2_opt: Adam,
        mb_s_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_a_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_y_t: TileTensor[DT, LY, OY],
    ) raises -> Scalar[DT]:
        comptime assert target == "cpu", (
            "TwinCriticUpdateBlock.step: GPU path not yet implemented"
        )
        self._assert_tag[target]()

        var sa_p = self._mb_sa.unsafe_ptr()
        concat_sa[Self.OBS, Self.ACT, Self.BATCH](mb_s_ptr, mb_a_ptr, sa_p)
        var sa_t = TileTensor(sa_p, row_major[Self.BATCH, Self.SA_DIM]())

        var loss1 = self.c1.step["cpu"](critic1, critic1_opt, sa_t, mb_y_t)
        var loss2 = self.c2.step["cpu"](critic2, critic2_opt, sa_t, mb_y_t)
        return loss1 + loss2
