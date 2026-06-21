"""SACActorLoss — SAC actor loss on the storage ComputeGraph (named DX).

  loss_per_b[b] = α·log π(a|s) − min(Q1(s,a), Q2(s,a)),   a ~ π(·|s)
  loss          = mean_b(loss_per_b);   ∂loss/∂loss_per_b = 1/BATCH

STORAGE migration (Stage 5): the graph is declared with the legacy NAME-wired DX
(`InputSlot`/`Node`/`ExternalNode`, edges by predecessor name — no runtime edge
list). The online actor + the two ONLINE critics are `ExternalNode`s threaded as
tracked `mut` refs into `graph.forward`/`vjp` (the actor accumulates param grads
and is stepped; the critics' grads are computed-then-discarded — the critic
block zero_grads before its own update). The `Scale` node's runtime `multiplier`
carries the moving α (host scalar). Mean loss + mean log_prob are host reductions
(per-step D2H on GPU; cheap at SAC scales).

  graph: s → actor → rsample → {action, logp} ; (s, action) → concat →
         q1, q2 → min_q ;  α·logp = Scale(logp) ;  loss = α·logp − min_q (output)
"""

from std.gpu import thread_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB_REDUCE
from mojo_rl.nn.storage.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.storage.core.module import Module
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.initializer import Zero
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.nn.storage.primitives.rsample import RSample
from mojo_rl.nn.storage.primitives.slice import Slice
from mojo_rl.nn.storage.primitives.concat import Concat2
from mojo_rl.nn.storage.primitives.scale import Scale
from mojo_rl.nn.storage.primitives.binary_elementwise import (
    BinaryElemMin, BinarySub,
)
from mojo_rl.nn.storage.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.storage.combinators.graph_decl import InputSlot, Node, ExternalNode
from ..loss.loss_block import LossBlock


# ── Device reductions (single-block; launch grid=1, block=TPB_REDUCE) ──
# `reduce_mean_write_kernel` writes mean(src) -> dst[0,0] (overwrite); the
# device ScalarAdam reads it as the entropy grad this same step (no D2H).
# `reduce_mean_acc_kernel` accumulates (Sum-of-means, count) into acc[0,0]/
# acc[1,0]; the host drains it once per flush.
def reduce_mean_write_kernel[
    B: Int
](
    src: LayoutTensor[DT, Layout.row_major(B, 1), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(1, 1), MutAnyOrigin],
):
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < B:
        my_sum += rebind[Scalar[DT]](src[k, 0])
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        dst[0, 0] = total[0] / Scalar[DT](B)


def reduce_mean_acc_kernel[
    B: Int
](
    src: LayoutTensor[DT, Layout.row_major(B, 1), MutAnyOrigin],
    acc: LayoutTensor[DT, Layout.row_major(1, 2), MutAnyOrigin],
):
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < B:
        my_sum += rebind[Scalar[DT]](src[k, 0])
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[0, 0] = rebind[Scalar[DT]](acc[0, 0]) + total[0] / Scalar[DT](B)
        acc[0, 1] = rebind[Scalar[DT]](acc[0, 1]) + Scalar[DT](1.0)


@fieldwise_init
struct SACActorLossOut(Movable & ImplicitlyDeletable):
    var loss: Scalar[DT]
    var log_prob_mean: Scalar[DT]


struct SACActorLoss[
    ACTOR: Module,
    CRITIC: Module,
    BATCH: Int,
](LossBlock):
    comptime OBS_DIM = Self.ACTOR.IN_DIMS[0]
    comptime ACT_DIM = Self.ACTOR.OUT_DIM // 2
    comptime ALP_DIM = Self.ACT_DIM + 1
    comptime SA_DIM = Self.OBS_DIM + Self.ACT_DIM

    comptime Graph = ComputeGraph[
        InputSlot["s", Self.OBS_DIM],                        # s
        ExternalNode["actor", Self.ACTOR, "s"],              # → [mu|ls]
        Node["rsample", RSample[Self.ACT_DIM], "actor"],     # → [a|logp]
        Node["action", Slice[Self.ALP_DIM, 0, Self.ACT_DIM], "rsample"],
        Node["logp", Slice[Self.ALP_DIM, Self.ACT_DIM, Self.ALP_DIM], "rsample"],
        Node["concat", Concat2[Self.OBS_DIM, Self.ACT_DIM], "s", "action"],
        ExternalNode["q1", Self.CRITIC, "concat"],
        ExternalNode["q2", Self.CRITIC, "concat"],
        Node["min_q", BinaryElemMin[1], "q1", "q2"],
        Node["alogp", Scale[1], "logp"],                     # α·logp
        Node["loss", BinarySub[1], "alogp", "min_q"],        # loss_per_b (output)
    ]

    var graph: Self.Graph
    var _loss_out: Tensor   # [B] loss_per_b (graph output)
    var _grad_seed: Tensor  # [B] = 1/BATCH (backward seed)
    # Device-α path (GPU only): `_lp_mean` [1] holds mean(log_prob), read by
    # the device ScalarAdam this same step; `_loss_acc` [2] = (Σmean, count)
    # actor-loss metric accumulator drained at flush cadence. Empty on CPU.
    var _lp_mean: Tensor
    var _loss_acc: Tensor

    def __init__(out self):
        self.graph = Self.Graph()
        self._loss_out = Tensor()
        self._grad_seed = Tensor()
        self._lp_mean = Tensor()
        self._loss_acc = Tensor()

    @staticmethod
    def make[
        target: StaticString
    ](
        ctx: Optional[DeviceContext] = None,
        action_scale: Scalar[DT] = Scalar[DT](1.0),
    ) raises -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "SACActorLoss: target must be 'cpu' or 'gpu'"
        comptime assert (
            Self.ACTOR.OUT_DIM == 2 * Self.ACT_DIM
        ), "SACActorLoss: ACTOR.OUT_DIM must equal 2·ACT_DIM"
        comptime assert (
            Self.CRITIC.IN_DIMS[0] == Self.SA_DIM
        ), "SACActorLoss: CRITIC.IN_DIM must equal OBS_DIM + ACT_DIM"
        comptime assert (
            Self.CRITIC.OUT_DIM == 1
        ), "SACActorLoss: CRITIC.OUT_DIM must equal 1"
        var blk = Self()
        blk.graph = Self.Graph.make[target, Zero](ctx)
        blk.graph.set_node_attr["rsample", "action_scale"](action_scale)
        comptime if target == "cpu":
            blk._loss_out = Tensor.alloc(Self.BATCH)
            blk._grad_seed = Tensor.alloc(Self.BATCH)
            for b in range(Self.BATCH):
                blk._grad_seed.data[b] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
        else:
            var c = ctx.value()
            blk._loss_out = Tensor.alloc_gpu(c, Self.BATCH)
            blk._grad_seed = Tensor.alloc(Self.BATCH)
            for b in range(Self.BATCH):
                blk._grad_seed.data[b] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
            blk._grad_seed.upload(c)
            blk._lp_mean = Tensor.alloc_gpu(c, 1)
            blk._lp_mean.dev.value().enqueue_fill(Scalar[DT](0))
            blk._loss_acc = Tensor.alloc_gpu(c, 2)
            blk._loss_acc.dev.value().enqueue_fill(Scalar[DT](0))
        return blk^

    # ── Device-α accessors (GPU only) ────────────────────────────────
    def lp_mean_dev(mut self) -> DeviceBuffer[DT]:
        """The device `lp_mean` [1] buffer — the device ScalarAdam reads it as
        the per-step entropy grad."""
        return self._lp_mean.dev.value()

    def set_alpha_ptr(
        mut self, p: UnsafePointer[Scalar[DT], MutAnyOrigin]
    ):
        """One-time GPU wiring: point the `alogp` Scale node at SAC's on-device
        α buffer. After this the actor-loss forward/vjp read α on-device, so
        `forward_backward` skips the per-step host α bake."""
        self.graph.set_node_attr_ptr["alogp", "multiplier"](p)

    def reset_loss_accum(mut self) raises:
        """Zero the device (Σmean, count) loss accumulator — flush cadence."""
        self._loss_acc.dev.value().enqueue_fill(Scalar[DT](0))

    def read_loss_accum(mut self, ctx: DeviceContext) raises -> Scalar[DT]:
        """D2H the device loss accumulator once (flush cadence) and return its
        window mean (Σmean / count). 0 if no steps."""
        self._loss_acc.download(ctx)
        var s = self._loss_acc.data[0]
        var n = self._loss_acc.data[1]
        if n == Scalar[DT](0.0):
            return Scalar[DT](0.0)
        return s / n

    def forward_backward[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut actor: Self.ACTOR,
        mut actor_opt: Adam,
        mut critic1: Self.CRITIC,
        mut critic2: Self.CRITIC,
        mut mb_s: Tensor,
        alpha: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises -> SACActorLossOut:
        comptime BB = Self.BATCH
        actor.zero_grad[target](ctx)
        # CPU bakes the host α scalar into the `alogp` Scale node per step. On
        # GPU α lives on-device (wired once at make via `set_alpha_ptr`) and is
        # refreshed by the device ScalarAdam — no per-step host work.
        comptime if target == "cpu":
            self.graph.set_node_attr["alogp", "multiplier"](alpha)  # α

        # Seed the graph input slot with s (a COPY into the graph pool), then
        # forward (actor + online critics threaded as tracked refs).
        self.graph.set_input["s", BB](mb_s, ctx)
        self.graph.forward[BB, target, POLICY](
            self._loss_out, ctx, actor, critic1, critic2
        )

        var loss_mean: Scalar[DT] = 0.0
        var lp_mean: Scalar[DT] = 0.0
        comptime if target == "cpu":
            # CPU: host reduction over the graph buffers (bit-identity path).
            var loss_sum: Scalar[DT] = 0.0
            var lp_sum: Scalar[DT] = 0.0
            for b in range(BB):
                loss_sum += self._loss_out.data[b]
                lp_sum += self.graph.node_output["logp"]().data[b]
            var inv_b = Scalar[DT](1.0) / Scalar[DT](BB)
            loss_mean = loss_sum * inv_b
            lp_mean = lp_sum * inv_b
        else:
            # GPU: device-reduce both, NO D2H. lp_mean → `_lp_mean` (read by the
            # device ScalarAdam this step); loss → `_loss_acc` (drained at flush).
            # Returned host scalars stay 0 sentinels — the trainer drains the
            # device buffers instead.
            var c = ctx.value()
            # Materialize the detached (MutAnyOrigin) device views first, so no
            # `self` field borrow is held across the launch.
            var logp_v = self.graph.node_output["logp"]().lt[
                "gpu", Layout.row_major(BB, 1)
            ]()
            var lp_mean_v = self._lp_mean.lt["gpu", Layout.row_major(1, 1)]()
            var loss_v = self._loss_out.lt["gpu", Layout.row_major(BB, 1)]()
            var loss_acc_v = self._loss_acc.lt["gpu", Layout.row_major(1, 2)]()
            c.enqueue_function[reduce_mean_write_kernel[BB]](
                logp_v,
                lp_mean_v,
                grid_dim=1,
                block_dim=TPB_REDUCE,
            )
            c.enqueue_function[reduce_mean_acc_kernel[BB]](
                loss_v,
                loss_acc_v,
                grid_dim=1,
                block_dim=TPB_REDUCE,
            )

        # backward (seed = 1/BATCH) + actor step. Grad flows through the
        # critics (param grads discarded) into the actor (stepped).
        self.graph.vjp[BB, target, POLICY](
            self._grad_seed, ctx, actor, critic1, critic2
        )
        actor_opt.step[target, M=Self.ACTOR](actor, ctx)

        return SACActorLossOut(loss=loss_mean, log_prob_mean=lp_mean)
