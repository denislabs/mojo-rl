"""SACActorLossCG — SAC actor loss as a single ComputeGraph.

Phase 3 FullGraph. The 11-node graph captures the full actor-loss
computation; the loss block degenerates to "bind externals, set input,
forward, seed, backward, step". No more inline GPU glue kernels — the
existing primitive GPU paths (RSample / Slice / BinaryConcat /
BinaryElemMin / Scale / BinarySub) compose to express the same math.

Graph topology (§8.6.1):

    InputSlot ["s",          OBS]
    ExternalUnaryNode ["actor_out", ACTOR,                      "s"]
    ExternalUnaryNode ["alp",       RSample[ACT],               "actor_out"]
    UnaryNode        ["action",     Slice[ACT+1, 0, ACT],       "alp"]
    UnaryNode        ["log_prob",   Slice[ACT+1, ACT, ACT+1],   "alp"]
    BinaryNode       ["sa",         BinaryConcat[OBS, ACT],     "s", "action"]
    ExternalUnaryNode ["q1",        CRITIC, "sa", MODE="input_only"]
    ExternalUnaryNode ["q2",        CRITIC, "sa", MODE="input_only"]
    BinaryNode       ["min_q",      BinaryElemMin[1],           "q1", "q2"]
    UnaryNode        ["alpha_lp",   Scale[1],                   "log_prob"]
    BinaryNode       ["loss_per_b", BinarySub[1],               "alpha_lp", "min_q"]

ACTOR, RSample, and CRITIC are external — owned by the trainer (actor +
critics) or the loss block (rsample, kept here so the trainer's
`select_action` path can reuse it). The graph references them via
ExternalUnaryNode + per-call `set_external`. Critic backward runs with
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

from std.gpu.host import DeviceContext, HostBuffer
from layout import TileTensor, row_major

from ..constants import DT
from ..core import Module, Optimizer, Initializer
from ..core.scratch import Scratch
from ..core.scratch_walkers import init_scratch_auto
from ..core.target_storage import TargetStorage, assert_tag_for
from ..initializer import Zero
from ..combinators.compute_graph import ComputeGraph
from ..combinators.graph_nodes import (
    InputSlot,
    UnaryNode,
    BinaryNode,
    ExternalUnaryNode,
)
from ..primitives.rsample import RSample
from ..primitives.scale import Scale
from ..primitives.slice import Slice
from ..primitives.binary_elem_min import BinaryElemMin
from ..primitives.binary_sub import BinarySub
from ..primitives.binary_concat import BinaryConcat
from .loss_block import LossBlock
from .seed_grad_inv_batch import seed_grad_inv_batch


@fieldwise_init
struct SACActorLossOut(Movable & ImplicitlyDestructible):
    """Result of one `forward_backward` call.

    `loss` is the mean-batch scalar value (for logging).
    `log_prob_mean` is the mean of log_prob over the batch — caller passes
    `-(log_prob_mean + target_entropy)` to its α optimizer.
    """
    var loss: Scalar[DT]
    var log_prob_mean: Scalar[DT]


struct SACActorLossCG[
    ACTOR: Module,
    CRITIC: Module,
    BATCH: Int,
](LossBlock):
    comptime OBS_DIM = Self.ACTOR.IN_DIM
    comptime ACT_DIM = Self.ACTOR.OUT_DIM // 2
    comptime ALP_DIM = Self.ACT_DIM + 1
    comptime SA_DIM = Self.OBS_DIM + Self.ACT_DIM

    # The 11-node FullGraph. §8.6.1.
    comptime ActorGraph = ComputeGraph[
        1,
        InputSlot["s",          Self.OBS_DIM],
        ExternalUnaryNode["actor_out", Self.ACTOR,        "s"],
        ExternalUnaryNode["alp",       RSample[Self.ACT_DIM], "actor_out"],
        UnaryNode ["action",    Slice[Self.ALP_DIM, 0, Self.ACT_DIM], "alp"],
        UnaryNode ["log_prob",  Slice[Self.ALP_DIM, Self.ACT_DIM, Self.ALP_DIM], "alp"],
        BinaryNode["sa",        BinaryConcat[Self.OBS_DIM, Self.ACT_DIM], "s", "action"],
        ExternalUnaryNode["q1", Self.CRITIC, "sa", MODE="input_only"],
        ExternalUnaryNode["q2", Self.CRITIC, "sa", MODE="input_only"],
        BinaryNode["min_q",     BinaryElemMin[1],         "q1", "q2"],
        UnaryNode ["alpha_lp",  Scale[1],                 "log_prob"],
        BinaryNode["loss_per_b", BinarySub[1],            "alpha_lp", "min_q"],
    ]

    var graph: Self.ActorGraph
    # Trainer reuses this for env-step `select_action`; kept here so
    # there's exactly one RSample instance (deterministic RNG sequence).
    var rsample: RSample[Self.ACT_DIM]

    # Scratch for graph IO. loss_per_b is [BATCH, 1] (the graph output);
    # grad_seed is [BATCH, 1] of 1/BATCH (the backward seed).
    var _loss_out: Scratch["loss_out", Self.BATCH]
    var _grad_seed: Scratch["grad_seed", Self.BATCH]

    # Host staging for GPU mean reduction (one [BATCH] buffer per side).
    var _loss_host: Optional[HostBuffer[DT]]
    var _lp_host: Optional[HostBuffer[DT]]

    var ts: TargetStorage

    def __init__(out self):
        self.graph = Self.ActorGraph()
        self.rsample = RSample[Self.ACT_DIM]()
        self._loss_out = Scratch["loss_out", Self.BATCH]()
        self._grad_seed = Scratch["grad_seed", Self.BATCH]()
        self._loss_host = None
        self._lp_host = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
    ) raises -> Self:
        comptime assert target == "cpu", (
            "SACActorLossCG.make[target='gpu'] requires a DeviceContext"
        )
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT_DIM, (
            "SACActorLossCG: ACTOR.OUT_DIM must equal 2·ACT_DIM"
        )
        comptime assert Self.CRITIC.IN_DIM == Self.SA_DIM, (
            "SACActorLossCG: CRITIC.IN_DIM must equal OBS_DIM + ACT_DIM"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "SACActorLossCG: CRITIC.OUT_DIM must equal 1"
        )

        var blk = Self()
        blk.graph = Self.ActorGraph.make[target="cpu", INIT=Zero]()
        blk.rsample = RSample[Self.ACT_DIM].make[target="cpu", INIT=Zero]()
        blk.rsample.action_scale = action_scale
        blk.ts = TargetStorage.make_cpu()
        init_scratch_auto[Self, target="cpu"](blk)
        return blk^

    @staticmethod
    def make[target: StaticString](
        ctx: DeviceContext,
        action_scale: Scalar[DT] = Scalar[DT](1.0),
    ) raises -> Self:
        """GPU factory."""
        comptime assert target == "gpu", (
            "SACActorLossCG.make[target='cpu'](ctx) — drop ctx for CPU"
        )
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT_DIM, (
            "SACActorLossCG: ACTOR.OUT_DIM must equal 2·ACT_DIM"
        )
        comptime assert Self.CRITIC.IN_DIM == Self.SA_DIM, (
            "SACActorLossCG: CRITIC.IN_DIM must equal OBS_DIM + ACT_DIM"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "SACActorLossCG: CRITIC.OUT_DIM must equal 1"
        )

        var blk = Self()
        blk.graph = Self.ActorGraph.make[target="gpu", INIT=Zero](ctx)
        blk.rsample = RSample[Self.ACT_DIM].make[target="gpu", INIT=Zero](ctx)
        blk.rsample.action_scale = action_scale
        blk.ts = TargetStorage.make_gpu(ctx)
        init_scratch_auto[Self, target="gpu"](blk, Optional[DeviceContext](ctx))
        blk._loss_host = ctx.enqueue_create_host_buffer[DT](Self.BATCH)
        blk._lp_host = ctx.enqueue_create_host_buffer[DT](Self.BATCH)
        return blk^

    def forward_backward[
        target: StaticString,
        OPT: Optimizer,
    ](
        mut self,
        mut actor: Self.ACTOR,
        mut actor_opt: OPT,
        mut critic1: Self.CRITIC,
        mut critic2: Self.CRITIC,
        mb_s_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alpha: Scalar[DT],
    ) raises -> SACActorLossOut:
        assert_tag_for["SACActorLossCG", target](self.ts.target_tag)
        comptime BB = Self.BATCH
        comptime OBS = Self.OBS_DIM

        actor_opt.zero_grad[target, M=Self.ACTOR](actor)

        # ── Bind externals (must repeat each call: trainer may move
        # the field, e.g. via Tuple indexing inside OptimizerBundle).
        self.graph.set_external["actor_out", Self.ACTOR](actor)
        self.graph.set_external["alp", RSample[Self.ACT_DIM]](self.rsample)
        self.graph.set_external["q1", Self.CRITIC](critic1)
        self.graph.set_external["q2", Self.CRITIC](critic2)

        # ── Set graph input + α attribute.
        var mb_s_t = TileTensor(mb_s_ptr, row_major[BB, OBS]())
        self.graph.set_input["s", BB](mb_s_t)
        self.graph.set_node_attr["alpha_lp", "multiplier"](alpha)

        # ── Forward.
        comptime if target == "cpu":
            var loss_p = self._loss_out.cpu_ptr()
            var loss_t = TileTensor(loss_p, row_major[BB, 1]())
            self.graph.forward["cpu", BB](loss_t)

            # Mean loss + mean log_prob, read directly from the graph.
            var lp_p = self.graph.node_out_ptr["log_prob"]()
            var loss_sum: Scalar[DT] = 0.0
            var lp_sum: Scalar[DT] = 0.0
            for b in range(BB):
                loss_sum += loss_p[b]
                lp_sum += lp_p[b]
            var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](BB)
            var loss_mean = loss_sum * inv_b
            var lp_mean = lp_sum * inv_b

            # Seed grad_out = 1/BATCH, then backward.
            var grad_p = self._grad_seed.cpu_ptr()
            seed_grad_inv_batch["cpu", BB](grad_p)
            var grad_t = TileTensor(grad_p, row_major[BB, 1]())
            self.graph.backward["cpu", BB](grad_t)

            actor_opt.step["cpu", M=Self.ACTOR](actor)
            return SACActorLossOut(loss=loss_mean, log_prob_mean=lp_mean)
        else:
            var ctx = self.ts.ctx.value()
            var loss_p = self._loss_out.dev_ptr()
            var loss_t = TileTensor(loss_p, row_major[BB, 1]())
            self.graph.forward["gpu", BB](loss_t)

            # Host-side mean reduction: copy loss_per_b + log_prob to
            # host, sum, divide. Cheaper than launching a reduction
            # kernel + scalar download at SAC batch sizes (≤ 1024).
            var lp_dev_p = self.graph.node_out_ptr["log_prob"]()
            ctx.enqueue_copy(self._loss_host.value(), loss_p)
            ctx.enqueue_copy(self._lp_host.value(), lp_dev_p)
            ctx.synchronize()
            var loss_hp = self._loss_host.value().unsafe_ptr()
            var lp_hp = self._lp_host.value().unsafe_ptr()
            var loss_sum: Scalar[DT] = 0.0
            var lp_sum: Scalar[DT] = 0.0
            for b in range(BB):
                loss_sum += loss_hp[b]
                lp_sum += lp_hp[b]
            var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](BB)
            var loss_mean = loss_sum * inv_b
            var lp_mean = lp_sum * inv_b

            var grad_p = self._grad_seed.dev_ptr()
            seed_grad_inv_batch["gpu", BB](
                grad_p, Optional[DeviceContext](ctx)
            )
            var grad_t = TileTensor(grad_p, row_major[BB, 1]())
            self.graph.backward["gpu", BB](grad_t)

            actor_opt.step["gpu", M=Self.ACTOR](actor)
            return SACActorLossOut(loss=loss_mean, log_prob_mean=lp_mean)
