"""CriticUpdateBlock / TwinCriticUpdateBlock — critic-update LossBlocks.

Phase 2 Track B migration: pre-Phase-2 each block declared a mix of
CPU `List` + GPU `Optional[DeviceBuffer]` + `_n` size-tracking fields
and hand-wrote a 3-arm runtime-tag-branching `_mb_X_ptr()` helper per
buffer. Post-migration, scratch is a `Scratch[NAME, SIZE]` field and
the ptr helpers collapse to compile-time `cpu_ptr()` / `dev_ptr()`
inside a `comptime if target == "cpu"` branch.

Self-contained: each block absorbs the scratch buffers (`mb_q`,
`mb_grad_q`, `mb_grad_sa`) the trainer would otherwise own. Mirrors
the SACActorLossCG ownership pattern but stays linear — the chain is
a single Module forward + MSELoss + backward + opt step, no fan-out,
no Slice/Min, so a full ComputeGraph would just add overhead. The win
here is **scratch ownership**, not DAG topology.

CPU + GPU.

Free helpers `critic_update_step` / `twin_critic_update_step` in
`training/off_policy_critic.mojo` stay available for prototyping
algorithms that don't want the block plumbing.

Surface:
    CriticUpdateBlock[CRITIC, BATCH, SA_DIM]
        - `make[target]() raises -> Self`                 (CPU)
        - `make[target](ctx) raises -> Self`              (GPU)
        - `step[target](mut critic, mut opt, sa_t, y_t) raises -> Scalar[DT]`
            zero_grad → critic.forward → mse.forward → mse.backward →
            critic.backward → opt.step; returns scalar loss.

    TwinCriticUpdateBlock[CRITIC, BATCH, OBS, ACT]
        - owns 2× CriticUpdateBlock + `_mb_sa` scratch
        - `make[target]() raises -> Self`                 (CPU)
        - `make[target](ctx) raises -> Self`              (GPU)
        - `step[target](mut c1, mut c1_opt, mut c2, mut c2_opt,
                        mb_s_ptr, mb_a_ptr, mb_y_t) raises -> Scalar[DT]`
            concat_sa → c1.step + c2.step; returns sum of losses.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT
from ..core.module import Module
from ..core.scratch import Scratch
from ..core.scratch_walkers import init_scratch_auto
from ..core.target_storage import TargetStorage, assert_tag_for
from ..optimizer.adam import Adam
from .loss_block import LossBlock
from .mse import MSELoss
from ..training.off_policy_critic import concat_sa, concat_sa_gpu


struct CriticUpdateBlock[
    CRITIC: Module,
    BATCH: Int,
    SA_DIM: Int,
](LossBlock):
    """Single-critic MSE update step. Owns all intermediate scratch."""

    var mse_loss: MSELoss[1]

    var _mb_q: Scratch["mb_q", Self.BATCH]
    var _mb_grad_q: Scratch["mb_grad_q", Self.BATCH]
    var _mb_grad_sa: Scratch["mb_grad_sa", Self.BATCH * Self.SA_DIM]

    var ts: TargetStorage

    def __init__(out self):
        self.mse_loss = MSELoss[1]()
        self._mb_q = Scratch["mb_q", Self.BATCH]()
        self._mb_grad_q = Scratch["mb_grad_q", Self.BATCH]()
        self._mb_grad_sa = Scratch["mb_grad_sa", Self.BATCH * Self.SA_DIM]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString]() raises -> Self:
        comptime assert target == "cpu", (
            "CriticUpdateBlock.make[target='gpu'] requires a DeviceContext"
        )
        comptime assert Self.CRITIC.IN_DIM == Self.SA_DIM, (
            "CriticUpdateBlock: CRITIC.IN_DIM must equal SA_DIM"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "CriticUpdateBlock: CRITIC.OUT_DIM must equal 1"
        )
        var blk = Self()
        blk.mse_loss = MSELoss[1].make[target="cpu"]()
        blk.ts = TargetStorage.make_cpu()
        init_scratch_auto[Self, target="cpu"](blk)
        return blk^

    @staticmethod
    def make[target: StaticString](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "CriticUpdateBlock.make[target='cpu'](ctx) — drop ctx for CPU"
        )
        comptime assert Self.CRITIC.IN_DIM == Self.SA_DIM, (
            "CriticUpdateBlock: CRITIC.IN_DIM must equal SA_DIM"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "CriticUpdateBlock: CRITIC.OUT_DIM must equal 1"
        )
        var blk = Self()
        blk.mse_loss = MSELoss[1].make[target="gpu"](ctx)
        blk.ts = TargetStorage.make_gpu(ctx)
        init_scratch_auto[Self, target="gpu"](blk, Optional[DeviceContext](ctx))
        return blk^

    def step[
        target: StaticString,
    ](
        mut self,
        mut critic: Self.CRITIC,
        mut opt: Adam,
        sa_t: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        y_t: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
    ) raises -> Scalar[DT]:
        assert_tag_for["CriticUpdateBlock", target](self.ts.target_tag)

        var mb_q_p: UnsafePointer[Scalar[DT], MutAnyOrigin]
        var mb_grad_q_p: UnsafePointer[Scalar[DT], MutAnyOrigin]
        var mb_grad_sa_p: UnsafePointer[Scalar[DT], MutAnyOrigin]
        comptime if target == "cpu":
            mb_q_p = self._mb_q.cpu_ptr()
            mb_grad_q_p = self._mb_grad_q.cpu_ptr()
            mb_grad_sa_p = self._mb_grad_sa.cpu_ptr()
        else:
            mb_q_p = self._mb_q.dev_ptr()
            mb_grad_q_p = self._mb_grad_q.dev_ptr()
            mb_grad_sa_p = self._mb_grad_sa.dev_ptr()

        # Launder caller-supplied tiles to MutAnyOrigin — Module's variadic
        # forward/vjp surface requires it.
        var sa_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](sa_t.ptr)
        var sa_t_rb = TileTensor(sa_p, row_major[Self.BATCH, Self.SA_DIM]())

        var mb_q_t = TileTensor(mb_q_p, row_major[Self.BATCH, 1]())
        opt.zero_grad[target, M=Self.CRITIC](critic)
        critic.forward[target, Self.BATCH](sa_t_rb, output=mb_q_t)
        var loss = self.mse_loss.forward[target, Self.BATCH](mb_q_t, y_t)

        var mb_grad_q_t = TileTensor(mb_grad_q_p, row_major[Self.BATCH, 1]())
        self.mse_loss.vjp[target, Self.BATCH](y_t, mb_grad_q_t)

        var mb_grad_sa_t = TileTensor(
            mb_grad_sa_p,
            row_major[Self.BATCH, Self.SA_DIM](),
        )
        critic.vjp[target, Self.BATCH](mb_grad_q_t, mb_grad_sa_t)
        opt.step[target, M=Self.CRITIC](critic)
        return loss


struct TwinCriticUpdateBlock[
    CRITIC: Module,
    BATCH: Int,
    OBS: Int,
    ACT: Int,
](LossBlock):
    """Twin-critic update against shared target `y`. Owns two
    `CriticUpdateBlock`s + a shared `_mb_sa` scratch."""

    comptime SA_DIM = Self.OBS + Self.ACT

    var c1: CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA_DIM]
    var c2: CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA_DIM]

    var _mb_sa: Scratch["mb_sa", Self.BATCH * Self.SA_DIM]

    var ts: TargetStorage

    def __init__(out self):
        self.c1 = CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA_DIM]()
        self.c2 = CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA_DIM]()
        self._mb_sa = Scratch["mb_sa", Self.BATCH * Self.SA_DIM]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString]() raises -> Self:
        comptime assert target == "cpu", (
            "TwinCriticUpdateBlock.make[target='gpu'] requires a DeviceContext"
        )
        var blk = Self()
        blk.c1 = CriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.SA_DIM
        ].make[target="cpu"]()
        blk.c2 = CriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.SA_DIM
        ].make[target="cpu"]()
        blk.ts = TargetStorage.make_cpu()
        init_scratch_auto[Self, target="cpu"](blk)
        return blk^

    @staticmethod
    def make[target: StaticString](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "TwinCriticUpdateBlock.make[target='cpu'](ctx) — drop ctx for CPU"
        )
        var blk = Self()
        blk.c1 = CriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.SA_DIM
        ].make[target="gpu"](ctx)
        blk.c2 = CriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.SA_DIM
        ].make[target="gpu"](ctx)
        blk.ts = TargetStorage.make_gpu(ctx)
        init_scratch_auto[Self, target="gpu"](blk, Optional[DeviceContext](ctx))
        return blk^

    def step[
        target: StaticString,
    ](
        mut self,
        mut critic1: Self.CRITIC,
        mut critic1_opt: Adam,
        mut critic2: Self.CRITIC,
        mut critic2_opt: Adam,
        mb_s_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_a_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_y_t: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
    ) raises -> Scalar[DT]:
        assert_tag_for["TwinCriticUpdateBlock", target](self.ts.target_tag)

        var sa_p: UnsafePointer[Scalar[DT], MutAnyOrigin]
        comptime if target == "cpu":
            sa_p = self._mb_sa.cpu_ptr()
            concat_sa[Self.OBS, Self.ACT, Self.BATCH](
                mb_s_ptr, mb_a_ptr, sa_p
            )
        else:
            sa_p = self._mb_sa.dev_ptr()
            concat_sa_gpu[Self.OBS, Self.ACT, Self.BATCH](
                self.ts.ctx.value(), mb_s_ptr, mb_a_ptr, sa_p
            )
        var sa_t = TileTensor(sa_p, row_major[Self.BATCH, Self.SA_DIM]())

        var loss1 = self.c1.step[target](critic1, critic1_opt, sa_t, mb_y_t)
        var loss2 = self.c2.step[target](critic2, critic2_opt, sa_t, mb_y_t)
        return loss1 + loss2
