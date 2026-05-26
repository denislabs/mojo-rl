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

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT
from ..core.amp import AMPPolicy, NoAMP
from ..core.module import Module
from ..core.scratch import Scratch
from ..core.scratch_walkers import init_scratch_auto
from ..core.target_storage import TargetStorage, assert_tag_for
from ..optimizer.adam import Adam
from .loss_block import LossBlock
from .mse import MSELoss
from ..training.off_policy_critic import concat_sa, concat_sa_gpu


# ──────────────────────────────────────────────────────────────────────
# Phase C.3c — IS-weighted MSE gradient scaling kernel.
# ──────────────────────────────────────────────────────────────────────


def _scale_grad_by_weights_kernel[BATCH: Int](
    mb_grad_q: LayoutTensor[
        DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
    ],
    weights: LayoutTensor[
        DT, Layout.row_major(BATCH), MutAnyOrigin,
    ],
):
    """Per-lane in-place scaling: `mb_grad_q[i, 0] *= weights[i]`.

    Phase C.3c. Inserted between MSE.vjp and Critic.vjp inside
    `CriticUpdateBlock.step` when PER passes a non-null IS-weights
    pointer. The weighted gradient flows through Critic.vjp unchanged
    afterwards, giving the per-sample PER correction
    `grad_θ ∝ Σ_b w_b · (Q_b − y_b) · ∂Q/∂θ` that Schaul et al. §3.4
    prescribes.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    mb_grad_q[i, 0] = mb_grad_q[i, 0] * weights[i]


def _capture_td_residuals_kernel[BATCH: Int](
    mb_grad_q: LayoutTensor[
        DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
    ],
    out_residuals: LayoutTensor[
        DT, Layout.row_major(BATCH), MutAnyOrigin,
    ],
):
    """Per-lane recovery of the unscaled TD residual `Q − y` from
    `mb_grad_q = (Q − y) / BATCH` (the value MSE.vjp wrote). Used by
    PER to refresh sum-tree priorities after the critic step. Run
    BEFORE the IS-weight scaling so the captured residuals are the
    raw signed TD error, not the IS-weighted gradient.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    out_residuals[i] = mb_grad_q[i, 0] * Scalar[DT](BATCH)


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
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "CriticUpdateBlock: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.CRITIC.IN_DIMS[0] == Self.SA_DIM, (
            "CriticUpdateBlock: CRITIC.IN_DIM must equal SA_DIM"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "CriticUpdateBlock: CRITIC.OUT_DIM must equal 1"
        )
        var blk = Self()
        comptime if target == "cpu":
            blk.mse_loss = MSELoss[1].make[target="cpu"]()
            blk.ts = TargetStorage.make_cpu()
            init_scratch_auto[Self, target="cpu"](blk)
        else:
            if not ctx:
                raise Error("CriticUpdateBlock.make[target='gpu']: ctx required")
            var ctx_v = ctx.value()
            blk.mse_loss = MSELoss[1].make[target="gpu"](ctx)
            blk.ts = TargetStorage.make_gpu(ctx_v)
            init_scratch_auto[Self, target="gpu"](blk, ctx)
        return blk^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
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
        weights_p: UnsafePointer[
            Scalar[DT], MutAnyOrigin,
        ] = UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
        td_residuals_p: UnsafePointer[
            Scalar[DT], MutAnyOrigin,
        ] = UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
    ) raises -> Scalar[DT]:
        """Phase C.3c — `weights_p` (optional, default null sentinel)
        is a `[BATCH]` per-sample IS weight vector. When non-null, the
        gradient `mb_grad_q` produced by `mse.vjp` is scaled in-place
        by `weights_p[i]` before flowing into `critic.vjp`. CPU path
        does a sequential loop; GPU path launches
        `_scale_grad_by_weights_kernel`. Null pointer → unweighted
        MSE → bit-identical to pre-C.3c.

        `td_residuals_p` (optional, default null sentinel) is a
        `[BATCH]` output vector. When non-null, the unscaled signed TD
        residual `(Q − y)` is captured between `mse.vjp` and the IS-
        weight scaling and written here. Used by PER to refresh sum-
        tree priorities. Captured BEFORE IS scaling so priorities
        reflect the true error magnitude, not the IS-weighted gradient.
        """
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
        critic.forward[target, Self.BATCH, POLICY](sa_t_rb, output=mb_q_t)
        var loss = self.mse_loss.forward[target, Self.BATCH, POLICY](
            mb_q_t, y_t,
        )

        var mb_grad_q_t = TileTensor(mb_grad_q_p, row_major[Self.BATCH, 1]())
        self.mse_loss.vjp[target, Self.BATCH, POLICY](y_t, mb_grad_q_t)

        # PER residual capture (raw signed TD `Q − y = mb_grad_q · BATCH`),
        # taken BEFORE the IS-weight scaling below so priorities reflect
        # error magnitude not weighted gradient. Null pointer → no capture.
        if Int(td_residuals_p) != 0:
            comptime if target == "cpu":
                var scale = Scalar[DT](Self.BATCH)
                for i in range(Self.BATCH):
                    td_residuals_p[i] = mb_grad_q_p[i] * scale
            else:
                var grad_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH, 1), MutAnyOrigin,
                ](mb_grad_q_p)
                var out_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
                ](td_residuals_p)
                comptime TPB = 128
                comptime n_blocks = (Self.BATCH + TPB - 1) // TPB
                comptime capture_kernel = _capture_td_residuals_kernel[
                    Self.BATCH
                ]
                var ctx = self.ts.ctx.value()
                ctx.enqueue_function[capture_kernel](
                    grad_lt, out_lt,
                    grid_dim=n_blocks, block_dim=TPB,
                )

        # Phase C.3c — IS-weight scaling, gated on non-null sentinel.
        if Int(weights_p) != 0:
            comptime if target == "cpu":
                for i in range(Self.BATCH):
                    mb_grad_q_p[i] = mb_grad_q_p[i] * weights_p[i]
            else:
                var grad_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH, 1), MutAnyOrigin,
                ](mb_grad_q_p)
                var w_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
                ](weights_p)
                comptime TPB = 128
                comptime n_blocks = (Self.BATCH + TPB - 1) // TPB
                comptime scale_kernel = _scale_grad_by_weights_kernel[
                    Self.BATCH
                ]
                var ctx = self.ts.ctx.value()
                ctx.enqueue_function[scale_kernel](
                    grad_lt, w_lt,
                    grid_dim=n_blocks, block_dim=TPB,
                )

        var mb_grad_sa_t = TileTensor(
            mb_grad_sa_p,
            row_major[Self.BATCH, Self.SA_DIM](),
        )
        critic.vjp[target, Self.BATCH, POLICY](mb_grad_q_t, mb_grad_sa_t)
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
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "TwinCriticUpdateBlock: target must be 'cpu' or 'gpu'"
        )
        var blk = Self()
        comptime if target == "cpu":
            blk.c1 = CriticUpdateBlock[
                Self.CRITIC, Self.BATCH, Self.SA_DIM
            ].make[target="cpu"]()
            blk.c2 = CriticUpdateBlock[
                Self.CRITIC, Self.BATCH, Self.SA_DIM
            ].make[target="cpu"]()
            blk.ts = TargetStorage.make_cpu()
            init_scratch_auto[Self, target="cpu"](blk)
        else:
            if not ctx:
                raise Error("TwinCriticUpdateBlock.make[target='gpu']: ctx required")
            var ctx_v = ctx.value()
            blk.c1 = CriticUpdateBlock[
                Self.CRITIC, Self.BATCH, Self.SA_DIM
            ].make[target="gpu"](ctx)
            blk.c2 = CriticUpdateBlock[
                Self.CRITIC, Self.BATCH, Self.SA_DIM
            ].make[target="gpu"](ctx)
            blk.ts = TargetStorage.make_gpu(ctx_v)
            init_scratch_auto[Self, target="gpu"](blk, ctx)
        return blk^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
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
        weights_p: UnsafePointer[
            Scalar[DT], MutAnyOrigin,
        ] = UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
        td_residuals_p: UnsafePointer[
            Scalar[DT], MutAnyOrigin,
        ] = UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
    ) raises -> Scalar[DT]:
        """Phase C.3c — `weights_p` (optional, default null) flows
        through both sub-block updates so both critics receive the
        same per-sample PER weighting. Bit-identical when null.

        `td_residuals_p` (optional, default null) captures the signed
        TD residual from critic1 only — canonical choice for PER
        priority refresh; critic2 sees the same target so |Q1−y| is
        a representative single-critic proxy (Schaul et al. §3.1).
        """
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

        var loss1 = self.c1.step[target, POLICY](
            critic1, critic1_opt, sa_t, mb_y_t,
            weights_p=weights_p,
            td_residuals_p=td_residuals_p,
        )
        var loss2 = self.c2.step[target, POLICY](
            critic2, critic2_opt, sa_t, mb_y_t,
            weights_p=weights_p,
        )
        return loss1 + loss2
