"""C51QUpdateBlock — distributional Q-net gradient step (cross-entropy).

Mirrors `dqn/q_update_block.mojo` but the loss is a cross-entropy
against the per-batch target distribution `m [B, N_ATOMS]` computed
by `C51TargetYBlock` instead of MSE against a scalar y.

Pipeline:
  1. opt.zero_grad
  2. Q_online(s) → _logits_all                       [B, NA · N_ATOMS]
  3. GatherActionSlice(_logits_all, mb_a) → _logits_a [B, N_ATOMS]
  4. CrossEntropyLoss[N_ATOMS](_logits_a, m) → scalar loss
     (computes softmax+log_softmax internally; numerically stable)
  5. CE.vjp(m, grad_logits_a)  →  grad_logits_a = (softmax − m) / BATCH
  6. Scatter grad_logits_a into _grad_logits_all at slot `a_taken · N_ATOMS`
     (block-owned kernel — mirrors DQNQUpdateBlock's scatter).
  7. Q_online.vjp(_grad_logits_all) → _grad_obs (discarded)
  8. opt.step

PER plumbing: identical sentinel pattern as `DQNQUpdateBlock` —
`weights_p` / `td_residuals_p` default null. PER scale: multiplies
`grad_logits_a` per-row by `weights[i]` after CE.vjp; td_residual
capture for C51 uses cross-entropy loss as the priority signal
(common Rainbow practice: per-sample CE = -Σ_k m·log_softmax). For the
first port we capture the per-sample CE before IS scaling; weights are
applied AFTER capture so priorities reflect un-weighted error.

CPU + GPU. GPU branch lifts the host scatter / PER capture / PER
scale loops into kernels.
"""

from std.math import exp as fexp, log as flog
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.loss.cross_entropy import CrossEntropyLoss
from mojo_rl.nn2.primitives.gather_action_slice import GatherActionSlice


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — module-level (one thread per BATCH row except scatter
# which is one thread per (b, c)).
# ──────────────────────────────────────────────────────────────────────


def _c51_per_residual_kernel[
    BATCH: Int, N_ATOMS: Int
](
    logits_a: LayoutTensor[
        DT,
        Layout.row_major(BATCH, N_ATOMS),
        MutAnyOrigin,
    ],
    m: LayoutTensor[DT, Layout.row_major(BATCH, N_ATOMS), MutAnyOrigin],
    td_out: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    """Per-row cross-entropy used as the PER priority signal."""
    var b = Int(global_idx.x)
    if b < BATCH:
        var mx = rebind[Scalar[DT]](logits_a[b, 0])
        for i in range(1, N_ATOMS):
            var v = rebind[Scalar[DT]](logits_a[b, i])
            if v > mx:
                mx = v
        var s_exp: Scalar[DT] = Scalar[DT](0.0)
        for i in range(N_ATOMS):
            s_exp = s_exp + fexp(rebind[Scalar[DT]](logits_a[b, i]) - mx)
        var lse = mx + flog(s_exp)
        var ce: Scalar[DT] = Scalar[DT](0.0)
        for i in range(N_ATOMS):
            var log_p = rebind[Scalar[DT]](logits_a[b, i]) - lse
            if log_p < Scalar[DT](-20.0):
                log_p = Scalar[DT](-20.0)
            ce = ce - rebind[Scalar[DT]](m[b, i]) * log_p
        td_out[b] = ce


def _c51_per_scale_kernel[
    BATCH: Int, N_ATOMS: Int
](
    grad_logits_a: LayoutTensor[
        DT,
        Layout.row_major(BATCH, N_ATOMS),
        MutAnyOrigin,
    ],
    weights: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    """`grad_logits_a[b, i] *= weights[b]`."""
    var idx = Int(global_idx.x)
    var total = BATCH * N_ATOMS
    if idx < total:
        var b = idx // N_ATOMS
        var i = idx % N_ATOMS
        grad_logits_a[b, i] = rebind[Scalar[DT]](grad_logits_a[b, i]) * rebind[
            Scalar[DT]
        ](weights[b])


def _c51_scatter_grad_kernel[
    BATCH: Int, NA: Int, N_ATOMS: Int
](
    grad_logits_a: LayoutTensor[
        DT,
        Layout.row_major(BATCH, N_ATOMS),
        MutAnyOrigin,
    ],
    mb_a: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    grad_logits_all: LayoutTensor[
        DT,
        Layout.row_major(BATCH, NA * N_ATOMS),
        MutAnyOrigin,
    ],
):
    """`grad_logits_all[b, c] = 0`, then `grad_logits_all[b, a·N + i] =
    grad_logits_a[b, i]` for `i ∈ [0, N_ATOMS)` and `a = int(mb_a[b])`."""
    var idx = Int(global_idx.x)
    comptime ROW = NA * N_ATOMS
    var total = BATCH * ROW
    if idx < total:
        var b = idx // ROW
        var c = idx % ROW
        var a = Int(rebind[Scalar[DT]](mb_a[b]))
        var lo = a * N_ATOMS
        var hi = lo + N_ATOMS
        if c >= lo and c < hi:
            grad_logits_all[b, c] = rebind[Scalar[DT]](grad_logits_a[b, c - lo])
        else:
            grad_logits_all[b, c] = Scalar[DT](0.0)


struct C51QUpdateBlock[
    Q_NET: Module,
    BATCH: Int,
    OBS: Int,
    NA: Int,
    N_ATOMS: Int,
](Defaultable & Movable & ImplicitlyDestructible):
    var ce_loss: CrossEntropyLoss[Self.N_ATOMS]
    var gather_slice: GatherActionSlice[Self.NA, Self.N_ATOMS]

    var _logits_all: Scratch["logits_all", Self.BATCH * Self.NA * Self.N_ATOMS]
    var _logits_a: Scratch["logits_a", Self.BATCH * Self.N_ATOMS]
    var _grad_logits_a: Scratch["grad_logits_a", Self.BATCH * Self.N_ATOMS]
    var _grad_logits_all: Scratch[
        "grad_logits_all",
        Self.BATCH * Self.NA * Self.N_ATOMS,
    ]
    var _grad_obs: Scratch["grad_obs", Self.BATCH * Self.OBS]

    var ts: TargetStorage

    def __init__(out self):
        self.ce_loss = CrossEntropyLoss[Self.N_ATOMS]()
        self.gather_slice = GatherActionSlice[Self.NA, Self.N_ATOMS]()
        self._logits_all = Scratch[
            "logits_all",
            Self.BATCH * Self.NA * Self.N_ATOMS,
        ]()
        self._logits_a = Scratch[
            "logits_a",
            Self.BATCH * Self.N_ATOMS,
        ]()
        self._grad_logits_a = Scratch[
            "grad_logits_a",
            Self.BATCH * Self.N_ATOMS,
        ]()
        self._grad_logits_all = Scratch[
            "grad_logits_all",
            Self.BATCH * Self.NA * Self.N_ATOMS,
        ]()
        self._grad_obs = Scratch["grad_obs", Self.BATCH * Self.OBS]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None,) raises -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "C51QUpdateBlock: target must be 'cpu' or 'gpu'"
        comptime assert (
            Self.Q_NET.IN_DIMS[0] == Self.OBS
        ), "C51QUpdateBlock: Q_NET.IN_DIM must equal OBS"
        comptime assert (
            Self.Q_NET.OUT_DIM == Self.NA * Self.N_ATOMS
        ), "C51QUpdateBlock: Q_NET.OUT_DIM must equal NA · N_ATOMS"
        var b = Self()
        b.ce_loss = CrossEntropyLoss[Self.N_ATOMS].make[target](ctx=ctx)
        b.gather_slice = GatherActionSlice[Self.NA, Self.N_ATOMS].make[
            target,
            INIT=Zero,
        ](ctx=ctx)
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
        mb_m_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        weights_p: UnsafePointer[
            Scalar[DT],
            MutAnyOrigin,
        ] = UnsafePointer[
            Scalar[DT], MutAnyOrigin
        ](unsafe_from_address=0),
        td_residuals_p: UnsafePointer[
            Scalar[DT],
            MutAnyOrigin,
        ] = UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
    ) raises -> Scalar[DT]:
        """Zero_grad → Q.forward → gather slice → CE forward+vjp →
        (PER hooks) → scatter → Q.vjp → opt.step. Returns scalar loss."""
        assert_tag_for["C51QUpdateBlock", target](self.ts.target_tag)
        comptime ROW = Self.NA * Self.N_ATOMS

        var logits_all_p = self._logits_all.target_ptr[target]()
        var logits_a_p = self._logits_a.target_ptr[target]()
        var grad_logits_a_p = self._grad_logits_a.target_ptr[target]()
        var grad_logits_all_p = self._grad_logits_all.target_ptr[target]()
        var grad_obs_p = self._grad_obs.target_ptr[target]()

        # 1. Zero grads.
        q_opt.zero_grad[target, M=Self.Q_NET](q_online)

        # 2. Q_online(s) → logits_all.
        var s_t = TileTensor(mb_s_ptr, row_major[Self.BATCH, Self.OBS]())
        var la_t = TileTensor(logits_all_p, row_major[Self.BATCH, ROW]())
        q_online.forward[target, Self.BATCH, POLICY](s_t, output=la_t)

        # 3. Gather slice at a_taken → logits_a [B, N_ATOMS].
        # Hetero-variadic: both carriers use row_major[BATCH, NA*N_ATOMS].
        var la_carrier = TileTensor(
            logits_all_p,
            row_major[Self.BATCH, ROW](),
        )
        var mb_a_carrier = TileTensor(
            mb_a_ptr,
            row_major[Self.BATCH, ROW](),
        )
        var la_slice_t = TileTensor(
            logits_a_p,
            row_major[Self.BATCH, Self.N_ATOMS](),
        )
        self.gather_slice.forward[target, Self.BATCH, POLICY](
            la_carrier,
            mb_a_carrier,
            output=la_slice_t,
        )

        # 4. CE(logits_a, m) → scalar loss.
        var m_t = TileTensor(mb_m_ptr, row_major[Self.BATCH, Self.N_ATOMS]())
        var loss = self.ce_loss.forward[target, Self.BATCH, POLICY](
            la_slice_t,
            m_t,
        )

        # 5. CE.vjp → grad_logits_a = (softmax(logits_a) − m) / BATCH.
        var grad_la_t = TileTensor(
            grad_logits_a_p,
            row_major[Self.BATCH, Self.N_ATOMS](),
        )
        self.ce_loss.vjp[target, Self.BATCH, POLICY](m_t, grad_la_t)

        comptime if target == "cpu":
            # 5a. PER residual capture — per-sample cross-entropy.
            if Int(td_residuals_p) != 0:
                for b in range(Self.BATCH):
                    var off = b * Self.N_ATOMS
                    var mx = logits_a_p[off]
                    for i in range(1, Self.N_ATOMS):
                        if logits_a_p[off + i] > mx:
                            mx = logits_a_p[off + i]
                    var s_exp: Scalar[DT] = 0.0
                    for i in range(Self.N_ATOMS):
                        s_exp = s_exp + fexp(logits_a_p[off + i] - mx)
                    var lse = mx + flog(s_exp)
                    var ce: Scalar[DT] = 0.0
                    for i in range(Self.N_ATOMS):
                        var log_p = logits_a_p[off + i] - lse
                        if log_p < Scalar[DT](-20.0):
                            log_p = Scalar[DT](-20.0)
                        ce = ce - mb_m_ptr[off + i] * log_p
                    td_residuals_p[b] = ce

            # 5b. PER IS-weight scaling on grad_logits_a (per-row scale).
            if Int(weights_p) != 0:
                for b in range(Self.BATCH):
                    var w = weights_p[b]
                    for i in range(Self.N_ATOMS):
                        grad_logits_a_p[b * Self.N_ATOMS + i] = (
                            grad_logits_a_p[b * Self.N_ATOMS + i] * w
                        )

            # 6. Scatter grad_logits_a → grad_logits_all at a_taken slot.
            for b in range(Self.BATCH):
                var a = Int(mb_a_ptr[b])
                var dst_base = b * ROW
                for c in range(ROW):
                    grad_logits_all_p[dst_base + c] = Scalar[DT](0.0)
                var src_base = b * Self.N_ATOMS
                var dst_slice = dst_base + a * Self.N_ATOMS
                for i in range(Self.N_ATOMS):
                    grad_logits_all_p[dst_slice + i] = grad_logits_a_p[
                        src_base + i
                    ]
        else:
            var ctx = self.ts.ctx.value()

            # 5a. PER residual capture on device.
            if Int(td_residuals_p) != 0:
                var logits_a_lt = LayoutTensor[
                    DT,
                    Layout.row_major(Self.BATCH, Self.N_ATOMS),
                    MutAnyOrigin,
                ](logits_a_p)
                var m_lt = LayoutTensor[
                    DT,
                    Layout.row_major(Self.BATCH, Self.N_ATOMS),
                    MutAnyOrigin,
                ](mb_m_ptr)
                var td_lt = LayoutTensor[
                    DT,
                    Layout.row_major(Self.BATCH),
                    MutAnyOrigin,
                ](td_residuals_p)
                comptime per_res_kernel = _c51_per_residual_kernel[
                    Self.BATCH,
                    Self.N_ATOMS,
                ]
                comptime n_blocks_pr = (Self.BATCH + TPB - 1) // TPB
                ctx.enqueue_function[per_res_kernel](
                    logits_a_lt,
                    m_lt,
                    td_lt,
                    grid_dim=n_blocks_pr,
                    block_dim=TPB,
                )

            # 5b. PER IS-weight scaling on device.
            if Int(weights_p) != 0:
                var grad_la_lt = LayoutTensor[
                    DT,
                    Layout.row_major(Self.BATCH, Self.N_ATOMS),
                    MutAnyOrigin,
                ](grad_logits_a_p)
                var w_lt = LayoutTensor[
                    DT,
                    Layout.row_major(Self.BATCH),
                    MutAnyOrigin,
                ](weights_p)
                comptime per_scl_kernel = _c51_per_scale_kernel[
                    Self.BATCH,
                    Self.N_ATOMS,
                ]
                comptime total_scl = Self.BATCH * Self.N_ATOMS
                comptime n_blocks_ps = (total_scl + TPB - 1) // TPB
                ctx.enqueue_function[per_scl_kernel](
                    grad_la_lt,
                    w_lt,
                    grid_dim=n_blocks_ps,
                    block_dim=TPB,
                )

            # 6. Scatter grad_logits_a → grad_logits_all at a_taken slot.
            var grad_la_lt2 = LayoutTensor[
                DT,
                Layout.row_major(Self.BATCH, Self.N_ATOMS),
                MutAnyOrigin,
            ](grad_logits_a_p)
            var mb_a_lt = LayoutTensor[
                DT,
                Layout.row_major(Self.BATCH),
                MutAnyOrigin,
            ](mb_a_ptr)
            var grad_la_all_lt = LayoutTensor[
                DT,
                Layout.row_major(Self.BATCH, ROW),
                MutAnyOrigin,
            ](grad_logits_all_p)
            comptime sc_kernel = _c51_scatter_grad_kernel[
                Self.BATCH,
                Self.NA,
                Self.N_ATOMS,
            ]
            comptime total_sc = Self.BATCH * ROW
            comptime n_blocks_sc = (total_sc + TPB - 1) // TPB
            ctx.enqueue_function[sc_kernel](
                grad_la_lt2,
                mb_a_lt,
                grad_la_all_lt,
                grid_dim=n_blocks_sc,
                block_dim=TPB,
            )

        # 7. Q_online.vjp(grad_logits_all) → grad_obs (discarded).
        var grad_la_all_t = TileTensor(
            grad_logits_all_p,
            row_major[Self.BATCH, ROW](),
        )
        var grad_obs_t = TileTensor(
            grad_obs_p,
            row_major[Self.BATCH, Self.OBS](),
        )
        q_online.vjp[target, Self.BATCH, POLICY](grad_la_all_t, grad_obs_t)

        # 8. opt.step.
        q_opt.step[target, M=Self.Q_NET](q_online)

        return loss
