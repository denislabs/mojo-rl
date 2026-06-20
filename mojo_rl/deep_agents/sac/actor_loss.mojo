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

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
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

    def __init__(out self):
        self.graph = Self.Graph()
        self._loss_out = Tensor()
        self._grad_seed = Tensor()

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
        return blk^

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
        self.graph.set_node_attr["alogp", "multiplier"](alpha)  # α

        # Seed the graph input slot with s (a COPY into the graph pool), then
        # forward (actor + online critics threaded as tracked refs).
        self.graph.set_input["s", BB](mb_s, ctx)
        self.graph.forward[BB, target, POLICY](
            self._loss_out, ctx, actor, critic1, critic2
        )

        # mean loss + mean log_prob (host reduction; D2H on GPU).
        comptime if target == "gpu":
            self._loss_out.download(ctx.value())
            self.graph.node_output["logp"]().download(ctx.value())
        var loss_sum: Scalar[DT] = 0.0
        var lp_sum: Scalar[DT] = 0.0
        for b in range(BB):
            loss_sum += self._loss_out.data[b]
            lp_sum += self.graph.node_output["logp"]().data[b]
        var inv_b = Scalar[DT](1.0) / Scalar[DT](BB)

        # backward (seed = 1/BATCH) + actor step. Grad flows through the
        # critics (param grads discarded) into the actor (stepped).
        self.graph.vjp[BB, target, POLICY](
            self._grad_seed, ctx, actor, critic1, critic2
        )
        actor_opt.step[target, M=Self.ACTOR](actor, ctx)

        return SACActorLossOut(
            loss=loss_sum * inv_b, log_prob_mean=lp_sum * inv_b
        )
