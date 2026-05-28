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

CPU-only initial port. GPU follow-up.
"""

from std.math import exp as fexp, log as flog
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.loss.cross_entropy import CrossEntropyLoss
from mojo_rl.nn2.primitives.gather_action_slice import GatherActionSlice


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
        "grad_logits_all", Self.BATCH * Self.NA * Self.N_ATOMS,
    ]
    var _grad_obs: Scratch["grad_obs", Self.BATCH * Self.OBS]

    var ts: TargetStorage

    def __init__(out self):
        self.ce_loss = CrossEntropyLoss[Self.N_ATOMS]()
        self.gather_slice = GatherActionSlice[Self.NA, Self.N_ATOMS]()
        self._logits_all = Scratch[
            "logits_all", Self.BATCH * Self.NA * Self.N_ATOMS,
        ]()
        self._logits_a = Scratch[
            "logits_a", Self.BATCH * Self.N_ATOMS,
        ]()
        self._grad_logits_a = Scratch[
            "grad_logits_a", Self.BATCH * Self.N_ATOMS,
        ]()
        self._grad_logits_all = Scratch[
            "grad_logits_all", Self.BATCH * Self.NA * Self.N_ATOMS,
        ]()
        self._grad_obs = Scratch["grad_obs", Self.BATCH * Self.OBS]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert (
            target == "cpu"
        ), "C51QUpdateBlock: GPU target not yet supported (CPU-only port)"
        comptime assert (
            Self.Q_NET.IN_DIMS[0] == Self.OBS
        ), "C51QUpdateBlock: Q_NET.IN_DIM must equal OBS"
        comptime assert (
            Self.Q_NET.OUT_DIM == Self.NA * Self.N_ATOMS
        ), "C51QUpdateBlock: Q_NET.OUT_DIM must equal NA · N_ATOMS"
        var b = Self()
        b.ce_loss = CrossEntropyLoss[Self.N_ATOMS].make[target](ctx=ctx)
        b.gather_slice = GatherActionSlice[Self.NA, Self.N_ATOMS].make[
            target, INIT=Zero,
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
            Scalar[DT], MutAnyOrigin,
        ] = UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
        td_residuals_p: UnsafePointer[
            Scalar[DT], MutAnyOrigin,
        ] = UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
    ) raises -> Scalar[DT]:
        """zero_grad → Q.forward → gather slice → CE forward+vjp →
        (PER hooks) → scatter → Q.vjp → opt.step. Returns scalar loss."""
        assert_tag_for["C51QUpdateBlock", target](self.ts.target_tag)
        comptime ROW = Self.NA * Self.N_ATOMS

        var logits_all_p = self._logits_all.cpu_ptr()
        var logits_a_p = self._logits_a.cpu_ptr()
        var grad_logits_a_p = self._grad_logits_a.cpu_ptr()
        var grad_logits_all_p = self._grad_logits_all.cpu_ptr()
        var grad_obs_p = self._grad_obs.cpu_ptr()

        # 1. Zero grads.
        q_opt.zero_grad[target, M=Self.Q_NET](q_online)

        # 2. Q_online(s) → logits_all.
        var s_t = TileTensor(mb_s_ptr, row_major[Self.BATCH, Self.OBS]())
        var la_t = TileTensor(logits_all_p, row_major[Self.BATCH, ROW]())
        q_online.forward[target, Self.BATCH, POLICY](s_t, output=la_t)

        # 3. Gather slice at a_taken → logits_a [B, N_ATOMS].
        # Hetero-variadic: both carriers use row_major[BATCH, NA*N_ATOMS].
        var la_carrier = TileTensor(
            logits_all_p, row_major[Self.BATCH, ROW](),
        )
        var mb_a_carrier = TileTensor(
            mb_a_ptr, row_major[Self.BATCH, ROW](),
        )
        var la_slice_t = TileTensor(
            logits_a_p, row_major[Self.BATCH, Self.N_ATOMS](),
        )
        self.gather_slice.forward[target, Self.BATCH, POLICY](
            la_carrier, mb_a_carrier, output=la_slice_t,
        )

        # 4. CE(logits_a, m) → scalar loss.
        var m_t = TileTensor(mb_m_ptr, row_major[Self.BATCH, Self.N_ATOMS]())
        var loss = self.ce_loss.forward[target, Self.BATCH, POLICY](
            la_slice_t, m_t,
        )

        # 5. CE.vjp → grad_logits_a = (softmax(logits_a) − m) / BATCH.
        var grad_la_t = TileTensor(
            grad_logits_a_p, row_major[Self.BATCH, Self.N_ATOMS](),
        )
        self.ce_loss.vjp[target, Self.BATCH, POLICY](m_t, grad_la_t)

        # 5a. PER residual capture — per-sample cross-entropy. Computed
        #     directly from logits_a + m before any scaling. (Common
        #     Rainbow choice; alternatives include L1 of the projected
        #     distribution diff.)
        if Int(td_residuals_p) != 0:
            for b in range(Self.BATCH):
                var off = b * Self.N_ATOMS
                # Recompute log-softmax for this row (cheap; could be
                # cached but the kernel is already CPU-bound for N_ATOMS
                # in the dozens).
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

        # 7. Q_online.vjp(grad_logits_all) → grad_obs (discarded).
        var grad_la_all_t = TileTensor(
            grad_logits_all_p, row_major[Self.BATCH, ROW](),
        )
        var grad_obs_t = TileTensor(
            grad_obs_p, row_major[Self.BATCH, Self.OBS](),
        )
        q_online.vjp[target, Self.BATCH, POLICY](grad_la_all_t, grad_obs_t)

        # 8. opt.step.
        q_opt.step[target, M=Self.Q_NET](q_online)

        return loss
