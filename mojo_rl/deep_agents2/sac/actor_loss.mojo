"""SACActorLoss — SAC actor loss as a single ComputeGraph.

Phase 3 FullGraph. The 11-node graph captures the full actor-loss
computation; the loss block degenerates to "bind externals, set input,
forward, seed, backward, step". No more inline GPU glue kernels — the
existing primitive GPU paths (RSample / Slice / Concat /
BinaryElemMin / Scale / BinarySub) compose to express the same math.

Graph topology (§8.6.1):

    InputSlot ["s",          OBS]
    ExternalNode ["actor_out", ACTOR,                      "s"]
    ExternalNode ["alp",       RSample[ACT],               "actor_out"]
    Node        ["action",     Slice[ACT+1, 0, ACT],       "alp"]
    Node        ["log_prob",   Slice[ACT+1, ACT, ACT+1],   "alp"]
    Node       ["sa",         Concat[OBS, ACT],           "s", "action"]
    ExternalNode ["q1",        CRITIC, "sa", MODE="input_only"]
    ExternalNode ["q2",        CRITIC, "sa", MODE="input_only"]
    Node       ["min_q",      BinaryElemMin[1],           "q1", "q2"]
    Node        ["alpha_lp",   Scale[1],                   "log_prob"]
    Node       ["loss_per_b", BinarySub[1],               "alpha_lp", "min_q"]

ACTOR, RSample, and CRITIC are external — owned by the trainer (actor +
critics) or the loss block (rsample, kept here so the trainer's
`select_action` path can reuse it). The graph references them via
ExternalNode + per-call `set_external`. Critic backward runs with
`MODE="input_only"` so the actor-loss path never accumulates critic
param grads (the same intent the spec captured with `StopGradParams`,
expressed inline without the wrapper).

Forward / backward semantics match the pre-Phase-3 hand-orchestrated
block exactly:
  loss_per_b[b] = α · log_prob(a|s)  −  min(Q1(s, a), Q2(s, a))
  loss          = mean_b(loss_per_b)
  d loss / d loss_per_b[b] = 1/BATCH

Mean loss + mean log_prob are computed by host-side reduction over
BATCH scalars (cheap at SAC scales; GPU uses one device→host copy each).

Public surface: `make[target]`, `forward_backward[target, OPT]`, plus a
public `rsample` field (the trainer's `select_action` reuses it).
"""

from std.gpu import thread_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT, TPB_REDUCE
from mojo_rl.nn2.core import Module, Optimizer, Initializer
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.combinators.compute_graph import ComputeGraph
from mojo_rl.nn2.combinators.graph_nodes import (
    InputSlot,
    Node,
    ExternalNode,
)
from ..primitives.rsample import RSample
from mojo_rl.nn2.primitives.scale import Scale
from mojo_rl.nn2.primitives.slice import Slice
from mojo_rl.nn2.primitives.binary_elem_min import BinaryElemMin
from mojo_rl.nn2.primitives.binary_sub import BinarySub
from mojo_rl.nn2.primitives.concat import Concat
from ..loss.loss_block import LossBlock
from ..loss.seed_grad_inv_batch import seed_grad_inv_batch


# ──────────────────────────────────────────────────────────────────────
# Slice 4c — device reductions (CUDA-graph capturable; no per-step D2H).
# Both are single-block `block.sum` reduces over the [BATCH] graph output.
# Launch grid=1, block=TPB_REDUCE. Mirror MSELoss's `_mse_reduce_add_kernel`.
# ──────────────────────────────────────────────────────────────────────


def _reduce_mean_write_kernel[BATCH: Int](
    src: UnsafePointer[Scalar[DT], MutAnyOrigin],
    dst: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """`dst[0] = mean(src[0..BATCH])` (overwrite). Used for `lp_mean` —
    consumed in-place by the device ScalarAdam this same step."""
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < BATCH:
        my_sum += src[k]
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        dst[0] = total[0] / Scalar[DT](BATCH)


def _reduce_mean_acc_kernel[BATCH: Int](
    src: UnsafePointer[Scalar[DT], MutAnyOrigin],
    acc: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """`acc[0] += mean(src); acc[1] += 1` (accumulate). Used for the actor
    loss metric — the host reads `acc` once per flush, never per step."""
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < BATCH:
        my_sum += src[k]
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[0] = acc[0] + total[0] / Scalar[DT](BATCH)
        acc[1] = acc[1] + Scalar[DT](1.0)


@fieldwise_init
struct SACActorLossOut(Movable & ImplicitlyDestructible):
    """Result of one `forward_backward` call.

    `loss` is the mean-batch scalar value (for logging).
    `log_prob_mean` is the mean of log_prob over the batch — caller passes
    `-(log_prob_mean + target_entropy)` to its α optimizer.
    """
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

    # The 11-node FullGraph. §8.6.1.
    comptime ActorGraph = ComputeGraph[
        1,
        InputSlot["s",          Self.OBS_DIM],
        ExternalNode["actor_out", Self.ACTOR,        "s"],
        ExternalNode["alp",       RSample[Self.ACT_DIM], "actor_out"],
        Node ["action",    Slice[Self.ALP_DIM, 0, Self.ACT_DIM], "alp"],
        Node ["log_prob",  Slice[Self.ALP_DIM, Self.ACT_DIM, Self.ALP_DIM], "alp"],
        Node["sa",        Concat[Self.OBS_DIM, Self.ACT_DIM], "s", "action"],
        ExternalNode["q1", Self.CRITIC, "sa", MODE="input_only"],
        ExternalNode["q2", Self.CRITIC, "sa", MODE="input_only"],
        Node["min_q",     BinaryElemMin[1],         "q1", "q2"],
        Node ["alpha_lp",  Scale[1],                 "log_prob"],
        Node["loss_per_b", BinarySub[1],            "alpha_lp", "min_q"],
    ]

    var graph: Self.ActorGraph
    # Trainer reuses this for env-step `select_action`; kept here so
    # there's exactly one RSample instance (deterministic RNG sequence).
    var rsample: RSample[Self.ACT_DIM]

    # Scratch for graph IO. loss_per_b is [BATCH, 1] (the graph output);
    # grad_seed is [BATCH, 1] of 1/BATCH (the backward seed).
    var _loss_out: Scratch["loss_out", Self.BATCH]
    var _grad_seed: Scratch["grad_seed", Self.BATCH]

    # Slice 4c — device reduction outputs (GPU only, no per-step D2H).
    # `_lp_mean_dev` [1] holds mean(log_prob); the device ScalarAdam reads
    # it as the entropy grad this same step. `_loss_acc_dev` [2] is the
    # (Σmean, count) actor-loss metric accumulator the trainer drains at
    # flush cadence (same shape as MSELoss's `loss_acc_dev`).
    var _lp_mean_dev: Optional[DeviceBuffer[DT]]
    var _loss_acc_dev: Optional[DeviceBuffer[DT]]

    var ts: TargetStorage

    def __init__(out self):
        self.graph = Self.ActorGraph()
        self.rsample = RSample[Self.ACT_DIM]()
        self._loss_out = Scratch["loss_out", Self.BATCH]()
        self._grad_seed = Scratch["grad_seed", Self.BATCH]()
        self._lp_mean_dev = None
        self._loss_acc_dev = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
        action_scale: Scalar[DT] = Scalar[DT](1.0),
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU.

        Inner ctors (`ComputeGraph.make`, `RSample.make`) raise if
        `target='gpu'` but `ctx is None`, so by the time we reach the
        GPU-only host buffer creation, `ctx.value()` is safe.
        """
        comptime assert target == "cpu" or target == "gpu", (
            "SACActorLoss: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT_DIM, (
            "SACActorLoss: ACTOR.OUT_DIM must equal 2·ACT_DIM"
        )
        comptime assert Self.CRITIC.IN_DIMS[0] == Self.SA_DIM, (
            "SACActorLoss: CRITIC.IN_DIM must equal OBS_DIM + ACT_DIM"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "SACActorLoss: CRITIC.OUT_DIM must equal 1"
        )

        var blk = Self()
        blk.graph = Self.ActorGraph.make[target, INIT=Zero](ctx=ctx)
        blk.rsample = RSample[Self.ACT_DIM].make[target, INIT=Zero](ctx=ctx)
        blk.rsample.action_scale = action_scale
        blk.ts = TargetStorage.make[target](ctx=ctx)
        init_scratch_auto[Self, target](blk, ctx)
        comptime if target == "gpu":
            var ctx_v = ctx.value()
            blk._lp_mean_dev = ctx_v.enqueue_create_buffer[DT](1)
            var acc = ctx_v.enqueue_create_buffer[DT](2)
            acc.enqueue_fill(0.0)
            blk._loss_acc_dev = acc^
        return blk^

    # ── Slice 4c accessors ───────────────────────────────────────────
    def lp_mean_dev_ptr(
        mut self,
    ) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Pointer to the device `lp_mean` [1] buffer — the device
        ScalarAdam reads it as the per-step entropy grad. GPU only."""
        return self._lp_mean_dev.value().unsafe_ptr()

    def set_alpha_ptr(
        mut self, p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ):
        """One-time GPU wiring: point the `alpha_lp` Scale node at the
        device α buffer (SAC's on-device temperature). After this the
        actor-loss forward/backward read α on-device, so `forward_backward`
        skips the per-step `set_node_attr` host bake. Caller invokes once
        at trainer make."""
        self.graph.set_node_attr_ptr["alpha_lp", "multiplier"](p)

    def reset_loss_accum(mut self) raises:
        """Zero the device (Σmean, count) loss accumulator — flush cadence."""
        self._loss_acc_dev.value().enqueue_fill(0.0)

    def read_loss_accum(mut self) raises -> Scalar[DT]:
        """D2H the device loss accumulator once (flush cadence) and return
        its window mean (Σmean / count). 0 if no steps. GPU only."""
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
        mut critic1: Self.CRITIC,
        mut critic2: Self.CRITIC,
        mb_s_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alpha: Scalar[DT],
    ) raises -> SACActorLossOut:
        assert_tag_for["SACActorLoss", target](self.ts.target_tag)
        comptime BB = Self.BATCH
        comptime OBS = Self.OBS_DIM

        actor_opt.zero_grad[target, M=Self.ACTOR](actor)

        # ── Bind externals (must repeat each call: trainer may move
        # the field, e.g. via Tuple indexing inside OptimizerBundle).
        self.graph.set_external["actor_out", Self.ACTOR](actor)
        self.graph.set_external["alp", RSample[Self.ACT_DIM]](self.rsample)
        self.graph.set_external["q1", Self.CRITIC](critic1)
        self.graph.set_external["q2", Self.CRITIC](critic2)

        # ── Set graph input + α attribute. CPU bakes the host α scalar per
        # call; GPU reads α on-device via the `alpha_lp` multiplier_ptr
        # wired once at make (`set_alpha_ptr`) — no per-step host work, so
        # the actor-loss forward/backward are CUDA-graph capturable.
        var mb_s_t = TileTensor(mb_s_ptr, row_major[BB, OBS]())
        self.graph.set_input["s", BB](mb_s_t)
        comptime if target == "cpu":
            self.graph.set_node_attr["alpha_lp", "multiplier"](alpha)

        # ── Forward.
        var loss_p = self._loss_out.target_ptr[target]()
        var loss_t = TileTensor(loss_p, row_major[BB, 1]())
        self.graph.forward[target, BB, POLICY](loss_t)

        var lp_p = self.graph.node_out_ptr["log_prob"]()
        var loss_mean: Scalar[DT] = 0.0
        var lp_mean: Scalar[DT] = 0.0

        # ── Mean reduction.
        comptime if target == "cpu":
            # CPU reads the graph buffers directly + host sum (unchanged —
            # the SAC CPU bit-identity path).
            var loss_sum: Scalar[DT] = 0.0
            var lp_sum: Scalar[DT] = 0.0
            for b in range(BB):
                loss_sum += loss_p[b]
                lp_sum += lp_p[b]
            var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](BB)
            loss_mean = loss_sum * inv_b
            lp_mean = lp_sum * inv_b
        else:
            # GPU: device-reduce both, NO D2H. lp_mean → `_lp_mean_dev`
            # (read by the device ScalarAdam this step); loss → device
            # accumulator (read at flush). Returned host scalars stay 0
            # sentinels — the trainer drains the device buffers instead.
            var ctx = self.ts.ctx.value()
            comptime red_lp = _reduce_mean_write_kernel[BB]
            ctx.enqueue_function[red_lp](
                lp_p, self._lp_mean_dev.value().unsafe_ptr(),
                grid_dim=1, block_dim=TPB_REDUCE,
            )
            comptime red_loss = _reduce_mean_acc_kernel[BB]
            ctx.enqueue_function[red_loss](
                loss_p, self._loss_acc_dev.value().unsafe_ptr(),
                grid_dim=1, block_dim=TPB_REDUCE,
            )

        # ── Seed grad_out = 1/BATCH, then backward + step.
        var grad_p = self._grad_seed.target_ptr[target]()
        seed_grad_inv_batch[target, BB](grad_p, ctx=self.ts.ctx)
        var grad_t = TileTensor(grad_p, row_major[BB, 1]())
        self.graph.vjp[target, BB, POLICY](grad_t)

        actor_opt.step[target, M=Self.ACTOR](actor)
        return SACActorLossOut(loss=loss_mean, log_prob_mean=lp_mean)
