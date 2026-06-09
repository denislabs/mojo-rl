"""DDPGActorLoss — deterministic policy gradient block (Block E-4).

Maximizes E_s[critic(s, π_φ(s))] via:

  loss(φ)        = -mean_b critic(s, π_φ(s))
  ∂loss/∂a       = -1/B
  ∂a/∂φ          via π.backward
  ∂critic/∂a     via critic.vjp[mode="input_only"]
                  (NO critic param-grad accumulation — actor update only)

CPU + GPU. The CPU path is bit-identical to the pre-Phase-4 block (host
loops for the mean, the −1/B seed, and the grad-column slice). The GPU
path mirrors `SACActorLoss` minus the rsample/entropy machinery:

  * scratch lives in device `Scratch` buffers (`init_scratch_auto`),
  * the concat is `concat_sa_gpu`,
  * the loss `−mean_b q` is reduced + accumulated on-device into a
    `[Σ, count]` buffer the trainer drains at flush cadence (no per-step
    D2H — the per-step path is CUDA-graph capturable),
  * the constant `−1/B` seed and the `grad_sa[:, OBS:] → grad_a` column
    slice are one-thread-per-lane kernels.

Owns (all `Scratch`):
  _mb_a       [BATCH, ACT]
  _mb_sa      [BATCH, OBS+ACT]
  _mb_q       [BATCH, 1]
  _mb_grad_q  [BATCH, 1]
  _mb_grad_sa [BATCH, OBS+ACT]
  _mb_grad_a  [BATCH, ACT]          (GPU: distinct from _mb_a; CPU reuses
                                     _mb_a as before for bit-identity)
  _mb_grad_s_unused [BATCH, OBS]

Shared by TD3 (uses critic1 only, identical math).
"""

from std.gpu import global_idx, thread_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn2.constants import DT, TPB, TPB_REDUCE
from mojo_rl.nn2.core import Module, Optimizer
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for
from ..training.off_policy_critic import concat_sa, concat_sa_gpu
from ..loss.loss_block import LossBlock


# ──────────────────────────────────────────────────────────────────────
# GPU glue kernels (no per-step D2H; CUDA-graph capturable).
# ──────────────────────────────────────────────────────────────────────


def _neg_mean_acc_kernel[BATCH: Int](
    q: UnsafePointer[Scalar[DT], MutAnyOrigin],
    acc: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """`acc[0] += -mean(q); acc[1] += 1` (accumulate). Single-block
    `block.sum` over [BATCH]; the host reads `acc` once per flush. Mirrors
    SACActorLoss's `_reduce_mean_acc_kernel`, negated for the DPG loss."""
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < BATCH:
        my_sum += q[k]
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[0] = acc[0] - total[0] / Scalar[DT](BATCH)
        acc[1] = acc[1] + Scalar[DT](1.0)


def _fill_const_kernel[N: Int](
    buf: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    value: Scalar[DT],
):
    """`buf[idx] = value` — used to seed grad_q with −1/B."""
    var idx = Int(global_idx.x)
    if idx < N:
        buf[idx] = value


def _slice_grad_a_kernel[
    BATCH: Int, OBS: Int, ACT: Int
](
    grad_sa: LayoutTensor[DT, Layout.row_major(BATCH, OBS + ACT), MutAnyOrigin],
    grad_a: LayoutTensor[DT, Layout.row_major(BATCH, ACT), MutAnyOrigin],
):
    """`grad_a[b, j] = grad_sa[b, OBS + j]` — extract the action-grad tail
    columns of the concatenated grad_sa into a contiguous [BATCH, ACT]
    tile for `actor.vjp`."""
    var idx = Int(global_idx.x)
    var total = BATCH * ACT
    if idx < total:
        var b = idx // ACT
        var j = idx % ACT
        grad_a[b, j] = rebind[Scalar[DT]](grad_sa[b, OBS + j])


struct DDPGActorLoss[
    ACTOR: Module,
    CRITIC: Module,
    BATCH: Int,
](LossBlock):
    comptime OBS_DIM = Self.ACTOR.IN_DIMS[0]
    comptime ACT_DIM = Self.ACTOR.OUT_DIM
    comptime SA_DIM = Self.OBS_DIM + Self.ACT_DIM

    var _mb_a: Scratch["mb_a", Self.BATCH * Self.ACT_DIM]
    var _mb_sa: Scratch["mb_sa", Self.BATCH * Self.SA_DIM]
    var _mb_q: Scratch["mb_q", Self.BATCH]
    var _mb_grad_q: Scratch["mb_grad_q", Self.BATCH]
    var _mb_grad_sa: Scratch["mb_grad_sa", Self.BATCH * Self.SA_DIM]
    var _mb_grad_a: Scratch["mb_grad_a", Self.BATCH * Self.ACT_DIM]
    var _mb_grad_s_unused: Scratch["mb_grad_s_unused", Self.BATCH * Self.OBS_DIM]

    # GPU-only device loss accumulator ([Σ(-mean q), count]); the host
    # reads it at flush cadence. None on CPU.
    var _loss_acc_dev: Optional[DeviceBuffer[DT]]

    var ts: TargetStorage

    def __init__(out self):
        comptime assert (
            Self.CRITIC.IN_DIMS[0] == Self.SA_DIM
        ), "DDPGActorLoss: CRITIC.IN_DIM must equal OBS+ACT"
        comptime assert (
            Self.CRITIC.OUT_DIM == 1
        ), "DDPGActorLoss: CRITIC.OUT_DIM must equal 1"
        self._mb_a = Scratch["mb_a", Self.BATCH * Self.ACT_DIM]()
        self._mb_sa = Scratch["mb_sa", Self.BATCH * Self.SA_DIM]()
        self._mb_q = Scratch["mb_q", Self.BATCH]()
        self._mb_grad_q = Scratch["mb_grad_q", Self.BATCH]()
        self._mb_grad_sa = Scratch["mb_grad_sa", Self.BATCH * Self.SA_DIM]()
        self._mb_grad_a = Scratch["mb_grad_a", Self.BATCH * Self.ACT_DIM]()
        self._mb_grad_s_unused = Scratch[
            "mb_grad_s_unused", Self.BATCH * Self.OBS_DIM
        ]()
        self._loss_acc_dev = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "DDPGActorLoss: target must be 'cpu' or 'gpu'"
        )
        var b = Self()
        b.ts = TargetStorage.make[target](ctx=ctx)
        init_scratch_auto[Self, target](b, ctx)
        comptime if target == "gpu":
            var ctx_v = ctx.value()
            var acc = ctx_v.enqueue_create_buffer[DT](2)
            acc.enqueue_fill(0.0)
            b._loss_acc_dev = acc^
        return b^

    # ── GPU loss-accumulator accessors (flush cadence) ───────────────
    def reset_loss_accum(mut self) raises:
        """Zero the device (Σ, count) loss accumulator. GPU only."""
        self._loss_acc_dev.value().enqueue_fill(0.0)

    def read_loss_accum(mut self) raises -> Scalar[DT]:
        """D2H the device loss accumulator once and return its window mean
        (Σ / count). 0 if no steps. GPU only."""
        var ctx = self.ts.ctx.value()
        var h = ctx.enqueue_create_host_buffer[DT](2)
        ctx.enqueue_copy(h, self._loss_acc_dev.value())
        ctx.synchronize()
        var s = h.unsafe_ptr()[0]
        var n = h.unsafe_ptr()[1]
        if n == Scalar[DT](0.0):
            return Scalar[DT](0.0)
        return s / n

    def forward_backward[
        target: StaticString,
        OPT: Optimizer,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut actor: Self.ACTOR,
        mut actor_opt: OPT,
        mut critic: Self.CRITIC,
        mb_s_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises -> Scalar[DT]:
        """Single DPG update step. Returns the scalar actor loss
        (= -mean_b q) on CPU; a 0 sentinel on GPU (the real metric is
        drained from `read_loss_accum` at flush).

        Caller must hold `mut critic` for the `vjp[mode="input_only"]`
        call; critic params are NOT updated by this method."""
        assert_tag_for["DDPGActorLoss", target](self.ts.target_tag)
        comptime BB = Self.BATCH
        comptime OBS = Self.OBS_DIM
        comptime ACT = Self.ACT_DIM
        comptime SA = Self.SA_DIM

        # Zero actor grads. Critic grads remain whatever the caller set them
        # to (we won't touch critic param grads — input_only backward).
        actor_opt.zero_grad[target, M=Self.ACTOR](actor)

        var mb_a_p = self._mb_a.target_ptr[target]()
        var mb_sa_p = self._mb_sa.target_ptr[target]()
        var mb_q_p = self._mb_q.target_ptr[target]()
        var mb_grad_q_p = self._mb_grad_q.target_ptr[target]()
        var mb_grad_sa_p = self._mb_grad_sa.target_ptr[target]()
        var mb_grad_s_unused_p = self._mb_grad_s_unused.target_ptr[target]()

        # Forward: a = actor(s); sa = concat(s, a); q = critic(sa).
        var mb_s_t = TileTensor(mb_s_ptr, row_major[BB, OBS]())
        var mb_a_t = TileTensor(mb_a_p, row_major[BB, ACT]())
        actor.forward[target, BB, POLICY](mb_s_t, output=mb_a_t)
        comptime if target == "cpu":
            concat_sa[OBS, ACT, BB](mb_s_ptr, mb_a_p, mb_sa_p)
        else:
            concat_sa_gpu[OBS, ACT, BB](
                self.ts.ctx.value(), mb_s_ptr, mb_a_p, mb_sa_p
            )
        var mb_sa_t = TileTensor(mb_sa_p, row_major[BB, SA]())
        var mb_q_t = TileTensor(mb_q_p, row_major[BB, 1]())
        critic.forward[target, BB, POLICY](mb_sa_t, output=mb_q_t)

        var loss: Scalar[DT] = 0.0
        var mb_grad_q_t = TileTensor(mb_grad_q_p, row_major[BB, 1]())
        var mb_grad_sa_t = TileTensor(mb_grad_sa_p, row_major[BB, SA]())

        comptime if target == "cpu":
            # ── Loss = -mean_b q (host sum; CPU bit-identity path).
            var q_sum: Scalar[DT] = 0.0
            for b in range(BB):
                q_sum += mb_q_p[b]
            loss = -q_sum / Scalar[DT](BB)

            # ∂loss/∂q[b] = -1/B (broadcast).
            var inv_B = Scalar[DT](1.0) / Scalar[DT](BB)
            for b in range(BB):
                mb_grad_q_p[b] = -inv_B

            # critic.vjp[input_only]: ∂q/∂sa, skip critic params.
            critic.vjp[target, BB, mode="input_only"](
                mb_grad_q_t, mb_grad_sa_t,
            )

            # Route ∂q/∂a (= grad_sa[:, OBS:]) into actor's grad-out. Reuse
            # `_mb_a` to hold grad_a (forward cache already consumed by the
            # actor.vjp below — bit-identical to the pre-Phase-4 block).
            for b in range(BB):
                for j in range(ACT):
                    mb_a_p[b * ACT + j] = mb_grad_sa_p[b * SA + OBS + j]
            var mb_grad_a_t = TileTensor(mb_a_p, row_major[BB, ACT]())
            var mb_grad_s_unused_t = TileTensor(
                mb_grad_s_unused_p, row_major[BB, OBS](),
            )
            actor.vjp[target, BB](mb_grad_a_t, mb_grad_s_unused_t)
        else:
            var ctx = self.ts.ctx.value()

            # ── Loss = -mean_b q, accumulated on-device (no per-step D2H).
            comptime neg_mean = _neg_mean_acc_kernel[BB]
            ctx.enqueue_function[neg_mean](
                mb_q_p, self._loss_acc_dev.value().unsafe_ptr(),
                grid_dim=1, block_dim=TPB_REDUCE,
            )
            loss = Scalar[DT](0.0)

            # ∂loss/∂q[b] = -1/B (device fill).
            var inv_B = Scalar[DT](1.0) / Scalar[DT](BB)
            var grad_q_lt = LayoutTensor[
                DT, Layout.row_major(BB), MutAnyOrigin,
            ](mb_grad_q_p)
            comptime fill_blocks = (BB + TPB - 1) // TPB
            comptime fill_kernel = _fill_const_kernel[BB]
            ctx.enqueue_function[fill_kernel](
                grad_q_lt, -inv_B, grid_dim=fill_blocks, block_dim=TPB,
            )

            # critic.vjp[input_only]: ∂q/∂sa, skip critic params.
            critic.vjp[target, BB, mode="input_only"](
                mb_grad_q_t, mb_grad_sa_t,
            )

            # grad_a = grad_sa[:, OBS:OBS+ACT] (device column slice into a
            # distinct buffer — no aliasing with the actor forward cache).
            var mb_grad_a_p = self._mb_grad_a.target_ptr[target]()
            var grad_sa_lt = LayoutTensor[
                DT, Layout.row_major(BB, SA), MutAnyOrigin,
            ](mb_grad_sa_p)
            var grad_a_lt = LayoutTensor[
                DT, Layout.row_major(BB, ACT), MutAnyOrigin,
            ](mb_grad_a_p)
            comptime slice_total = BB * ACT
            comptime slice_blocks = (slice_total + TPB - 1) // TPB
            comptime slice_kernel = _slice_grad_a_kernel[BB, OBS, ACT]
            ctx.enqueue_function[slice_kernel](
                grad_sa_lt, grad_a_lt,
                grid_dim=slice_blocks, block_dim=TPB,
            )
            var mb_grad_a_t = TileTensor(mb_grad_a_p, row_major[BB, ACT]())
            var mb_grad_s_unused_t = TileTensor(
                mb_grad_s_unused_p, row_major[BB, OBS](),
            )
            actor.vjp[target, BB](mb_grad_a_t, mb_grad_s_unused_t)

        # Step actor only.
        actor_opt.step[target, M=Self.ACTOR](actor)
        return loss
