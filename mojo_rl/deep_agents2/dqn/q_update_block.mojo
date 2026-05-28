"""DQNQUpdateBlock — DQN online-Q gradient step.

DQN's single-Q analogue of `loss/critic_update_block.mojo`'s
`CriticUpdateBlock`. Differences from the SAC critic block:

  - Q-net takes observation only (`mb_s`, OBS dims), not `sa = concat(s,a)`.
  - Gather step between Q.forward and MSE: extract `Q(s, a_taken)` from
    `[B, NA]` via the action indices in `mb_a` ([B, 1]). Done with
    `GatherCols[NA].forward` — on-device, no CPU shim
    (replaces trainer.mojo:339-351 today).
  - Scatter step before Q.vjp: only the `a_taken` slot of `grad_q_all`
    has gradient, the rest must be zero (replaces trainer.mojo:369-375
    today). Done via a block-owned kernel `_scatter_action_grad` that
    takes `mb_a` directly — cheaper than going through `GatherCols.vjp`
    which doesn't have access to indices on the standard
    `Module.vjp(grad_output, *grad_inputs)` surface.

Self-contained scratch ownership (init_scratch_auto). CPU + GPU.

PER hooks deferred to a later commit (see plan §3 commit 3). The block
exposes `weights_p`/`td_residuals_p` sentinel parameters for parity
with `CriticUpdateBlock`, but they're untested for DQN yet; pass null
for uniform replay.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.loss.mse import MSELoss
from mojo_rl.nn2.primitives.gather_cols import GatherCols


# ──────────────────────────────────────────────────────────────────────
# Block-owned kernels.
# ──────────────────────────────────────────────────────────────────────


def _capture_td_residuals_kernel[BATCH: Int](
    mb_grad_q: LayoutTensor[
        DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
    ],
    out_residuals: LayoutTensor[
        DT, Layout.row_major(BATCH), MutAnyOrigin,
    ],
):
    """`out_residuals[i] = mb_grad_q[i, 0] * BATCH` — recovers the raw
    signed TD residual `Q − y` from the value MSE.vjp wrote
    (`mb_grad_q = (Q − y) / BATCH`). Used by PER to refresh sum-tree
    priorities AFTER the gather and BEFORE the IS-weight scaling so the
    captured residuals are the raw signed TD error, not the IS-weighted
    gradient. Mirrors loss/critic_update_block.mojo:85-102."""
    var i = Int(global_idx.x)
    if i >= BATCH:
        return
    out_residuals[i] = mb_grad_q[i, 0] * Scalar[DT](BATCH)


def _scale_grad_by_weights_kernel[BATCH: Int](
    mb_grad_q: LayoutTensor[
        DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
    ],
    weights: LayoutTensor[
        DT, Layout.row_major(BATCH), MutAnyOrigin,
    ],
):
    """In-place: `mb_grad_q[i, 0] *= weights[i]`. The PER IS-weight
    scaling step. Mirrors loss/critic_update_block.mojo:62-82."""
    var i = Int(global_idx.x)
    if i >= BATCH:
        return
    mb_grad_q[i, 0] = mb_grad_q[i, 0] * weights[i]


def _scatter_action_grad_kernel[
    BATCH: Int, NA: Int,
](
    grad_q_gath: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    mb_a: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    grad_q_all: LayoutTensor[DT, Layout.row_major(BATCH, NA), MutAnyOrigin],
):
    """`grad_q_all[b, k] = grad_q_gath[b, 0]` if `k == Int(mb_a[b, 0])`
    else `0`. One thread per (BATCH * NA) element; per-row write is
    unique so no atomics needed."""
    var lin = Int(global_idx.x)
    var total = BATCH * NA
    if lin < total:
        var b = lin // NA
        var k = lin % NA
        var a = Int(rebind[Scalar[DT]](mb_a[b, 0]))
        if k == a:
            grad_q_all[b, k] = rebind[Scalar[DT]](grad_q_gath[b, 0])
        else:
            grad_q_all[b, k] = Scalar[DT](0.0)


struct DQNQUpdateBlock[
    Q_NET: Module,
    BATCH: Int,
    OBS: Int,
    NA: Int,
](Defaultable & Movable & ImplicitlyDestructible):
    var mse_loss: MSELoss[1]
    var gather_cols: GatherCols[Self.NA]

    var _mb_q_all: Scratch["mb_q_all", Self.BATCH * Self.NA]
    var _mb_q_gath: Scratch["mb_q_gath", Self.BATCH]
    var _mb_grad_q: Scratch["mb_grad_q", Self.BATCH]
    var _mb_grad_q_all: Scratch["mb_grad_q_all", Self.BATCH * Self.NA]
    var _mb_grad_obs: Scratch["mb_grad_obs", Self.BATCH * Self.OBS]

    var ts: TargetStorage

    def __init__(out self):
        self.mse_loss = MSELoss[1]()
        self.gather_cols = GatherCols[Self.NA]()
        self._mb_q_all = Scratch["mb_q_all", Self.BATCH * Self.NA]()
        self._mb_q_gath = Scratch["mb_q_gath", Self.BATCH]()
        self._mb_grad_q = Scratch["mb_grad_q", Self.BATCH]()
        self._mb_grad_q_all = Scratch["mb_grad_q_all", Self.BATCH * Self.NA]()
        self._mb_grad_obs = Scratch["mb_grad_obs", Self.BATCH * Self.OBS]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "DQNQUpdateBlock: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.Q_NET.IN_DIMS[0] == Self.OBS, (
            "DQNQUpdateBlock: Q_NET.IN_DIM must equal OBS"
        )
        comptime assert Self.Q_NET.OUT_DIM == Self.NA, (
            "DQNQUpdateBlock: Q_NET.OUT_DIM must equal NA"
        )
        var b = Self()
        b.mse_loss = MSELoss[1].make[target](ctx=ctx)
        b.gather_cols = GatherCols[Self.NA].make[target, INIT=Zero](ctx=ctx)
        b.ts = TargetStorage.make[target](ctx=ctx)
        init_scratch_auto[Self, target=target](b, ctx)
        return b^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut q_online: Self.Q_NET,
        mut q_opt: Adam,
        mb_s_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_a_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        weights_p: UnsafePointer[
            Scalar[DT], MutAnyOrigin,
        ] = UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
        td_residuals_p: UnsafePointer[
            Scalar[DT], MutAnyOrigin,
        ] = UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
    ) raises -> Scalar[DT]:
        """Run zero_grad → Q.forward → gather → MSE forward+vjp → (PER hooks)
        → scatter → Q.vjp → opt.step. Returns scalar loss.

        PER hooks (gated on non-null sentinels; null = uniform, bit-
        identical to pre-PER):
          - `td_residuals_p` ([BATCH]) : captures raw signed TD residual
            `(Q − y) = mb_grad_q · BATCH` AFTER mse.vjp, BEFORE IS scaling.
          - `weights_p`     ([BATCH]) : scales `mb_grad_q[i] *= weights[i]`
            in-place so the scatter+Q.vjp gradient is IS-weighted."""
        assert_tag_for["DQNQUpdateBlock", target](self.ts.target_tag)

        var q_all_p = self._mb_q_all.target_ptr[target]()
        var q_gath_p = self._mb_q_gath.target_ptr[target]()
        var grad_q_p = self._mb_grad_q.target_ptr[target]()
        var grad_q_all_p = self._mb_grad_q_all.target_ptr[target]()
        var grad_obs_p = self._mb_grad_obs.target_ptr[target]()

        # 1. Zero grads.
        q_opt.zero_grad[target, M=Self.Q_NET](q_online)

        # 2. Q_online(s) → q_all.
        var s_t = TileTensor(mb_s_ptr, row_major[Self.BATCH, Self.OBS]())
        var q_all_t = TileTensor(q_all_p, row_major[Self.BATCH, Self.NA]())
        q_online.forward[target, Self.BATCH, POLICY](s_t, output=q_all_t)

        # 3. GatherCols(q_all, mb_a) → q_gath. Hetero-variadic: both
        # carriers use row_major[BATCH, NA]. mb_a is allocated as
        # [BATCH] scalars; typed_view inside the leaf reinterprets as
        # [BATCH, 1] which keeps the offset arithmetic correct.
        var q_all_carrier = TileTensor(
            q_all_p, row_major[Self.BATCH, Self.NA](),
        )
        var mb_a_carrier = TileTensor(
            mb_a_ptr, row_major[Self.BATCH, Self.NA](),
        )
        var q_gath_t = TileTensor(q_gath_p, row_major[Self.BATCH, 1]())
        self.gather_cols.forward[target, Self.BATCH, POLICY](
            q_all_carrier, mb_a_carrier, output=q_gath_t,
        )

        # 4. MSE(q_gath, y).
        var y_t = TileTensor(mb_y_ptr, row_major[Self.BATCH, 1]())
        var loss = self.mse_loss.forward[target, Self.BATCH, POLICY](
            q_gath_t, y_t,
        )

        # 5. MSE.vjp → grad_q.
        var grad_q_t = TileTensor(grad_q_p, row_major[Self.BATCH, 1]())
        self.mse_loss.vjp[target, Self.BATCH, POLICY](y_t, grad_q_t)

        # 5a. PER residual capture (raw signed TD `Q − y = mb_grad_q · BATCH`),
        # taken BEFORE the IS-weight scaling below so priorities reflect
        # error magnitude not weighted gradient. Null pointer → no capture.
        if Int(td_residuals_p) != 0:
            comptime if target == "cpu":
                var scale = Scalar[DT](Self.BATCH)
                for i in range(Self.BATCH):
                    td_residuals_p[i] = grad_q_p[i] * scale
            else:
                var grad_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH, 1), MutAnyOrigin,
                ](grad_q_p)
                var out_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
                ](td_residuals_p)
                comptime n_blocks = (Self.BATCH + TPB - 1) // TPB
                comptime capture_kernel = _capture_td_residuals_kernel[
                    Self.BATCH,
                ]
                var ctx = self.ts.ctx.value()
                ctx.enqueue_function[capture_kernel](
                    grad_lt, out_lt,
                    grid_dim=n_blocks, block_dim=TPB,
                )

        # 5b. PER IS-weight scaling. Null pointer → no scaling.
        if Int(weights_p) != 0:
            comptime if target == "cpu":
                for i in range(Self.BATCH):
                    grad_q_p[i] = grad_q_p[i] * weights_p[i]
            else:
                var grad_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH, 1), MutAnyOrigin,
                ](grad_q_p)
                var w_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
                ](weights_p)
                comptime n_blocks = (Self.BATCH + TPB - 1) // TPB
                comptime scale_kernel = _scale_grad_by_weights_kernel[
                    Self.BATCH,
                ]
                var ctx = self.ts.ctx.value()
                ctx.enqueue_function[scale_kernel](
                    grad_lt, w_lt,
                    grid_dim=n_blocks, block_dim=TPB,
                )

        # 6. Scatter grad_q into grad_q_all at the taken-action slot.
        comptime if target == "cpu":
            for b in range(Self.BATCH):
                var a = Int(mb_a_ptr[b])
                for k in range(Self.NA):
                    if k == a:
                        grad_q_all_p[b * Self.NA + k] = grad_q_p[b]
                    else:
                        grad_q_all_p[b * Self.NA + k] = Scalar[DT](0.0)
        else:
            var gq_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, 1), MutAnyOrigin,
            ](grad_q_p)
            var mba_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, 1), MutAnyOrigin,
            ](mb_a_ptr)
            var gqa_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.NA), MutAnyOrigin,
            ](grad_q_all_p)
            comptime total = Self.BATCH * Self.NA
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _scatter_action_grad_kernel[Self.BATCH, Self.NA]
            self.ts.ctx.value().enqueue_function[kernel](
                gq_lt, mba_lt, gqa_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )

        # 7. Q.vjp(grad_q_all) → grad_obs (discarded).
        var grad_q_all_t = TileTensor(
            grad_q_all_p, row_major[Self.BATCH, Self.NA](),
        )
        var grad_obs_t = TileTensor(
            grad_obs_p, row_major[Self.BATCH, Self.OBS](),
        )
        q_online.vjp[target, Self.BATCH, POLICY](grad_q_all_t, grad_obs_t)

        # 8. opt.step.
        q_opt.step[target, M=Self.Q_NET](q_online)

        return loss
