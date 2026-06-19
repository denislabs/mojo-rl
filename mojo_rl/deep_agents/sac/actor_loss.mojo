"""SACActorLoss — SAC actor loss on storage ComputeGraph + ExternalRef.

  loss_per_b[b] = α·log π(a|s) − min(Q1(s,a), Q2(s,a)),   a ~ π(·|s)
  loss          = mean_b(loss_per_b);   ∂loss/∂loss_per_b = 1/BATCH

STORAGE migration (Stage 5): rebuilt on `ComputeGraph[NUM_IN, *NODES]` — the
online actor + the two ONLINE critics are `ExternalRef` markers threaded as
tracked `mut` refs into `graph.forward`/`vjp` (the actor accumulates param grads
and is stepped; the critics' grads are computed-then-discarded — the critic
block zero_grads before its own update, so the legacy MODE="input_only" is just a
skipped-compute optimization). The `Scale` node's runtime `multiplier` carries the
moving α (host scalar; device-α/CUDA-graph capture deferred). Mean loss + mean
log_prob are host reductions (per-step D2H on GPU; cheap at SAC scales).

  graph: ExternalRef[ACTOR] → RSample → {Slice(action), Slice(logp)} →
         Concat2(s,action) → ExternalRef[CRITIC]×2 → BinaryElemMin = min_q;
         Scale(logp)=α·logp ; BinarySub(α·logp, min_q) = loss_per_b (output)
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.storage.core.module import Module
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_pack import TensorPack
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
from mojo_rl.nn.storage.combinators.external_ref import ExternalRef
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
        1,
        ExternalRef[Self.ACTOR],                       # 0 s → [mu|ls]
        RSample[Self.ACT_DIM],                         # 1 → [a|logp]
        Slice[Self.ALP_DIM, 0, Self.ACT_DIM],          # 2 action
        Slice[Self.ALP_DIM, Self.ACT_DIM, Self.ALP_DIM], # 3 log_prob
        Concat2[Self.OBS_DIM, Self.ACT_DIM],           # 4 (s, a)
        ExternalRef[Self.CRITIC],                      # 5 q1
        ExternalRef[Self.CRITIC],                      # 6 q2
        BinaryElemMin[1],                              # 7 min_q
        Scale[1],                                      # 8 α·logp
        BinarySub[1],                                  # 9 loss_per_b (output)
    ]

    var graph: Self.Graph
    var _edges: List[List[Int]]
    var _inp: TensorPack[1]
    var _loss_out: Tensor   # [B] loss_per_b (graph output)
    var _grad_seed: Tensor  # [B] = 1/BATCH (backward seed)
    var _gin: TensorPack[1] # grad wrt s (discarded)

    def __init__(out self):
        self.graph = Self.Graph()
        self._edges = Self._build_edges()
        self._inp = TensorPack[1]()
        self._loss_out = Tensor()
        self._grad_seed = Tensor()
        self._gin = TensorPack[1]()

    @staticmethod
    def _build_edges() -> List[List[Int]]:
        var e = List[List[Int]]()
        e.append([0])      # 0 actor(s=slot0)
        e.append([1])      # 1 rsample(actor_out=slot1)
        e.append([2])      # 2 slice action(alp=slot2)
        e.append([2])      # 3 slice logp(alp=slot2)
        e.append([0, 3])   # 4 concat(s=slot0, action=slot3)
        e.append([5])      # 5 critic1(sa=slot5)
        e.append([5])      # 6 critic2(sa=slot5)
        e.append([6, 7])   # 7 min(q1=slot6, q2=slot7)
        e.append([4])      # 8 scale(logp=slot4)
        e.append([9, 8])   # 9 sub(α·logp=slot9, min_q=slot8) = α·logp − min_q
        return e^

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
        blk.graph.children[1].action_scale = action_scale  # the RSample node
        comptime if target == "cpu":
            blk._inp[0].ensure(Self.BATCH * Self.OBS_DIM)
            blk._loss_out = Tensor.alloc(Self.BATCH)
            blk._grad_seed = Tensor.alloc(Self.BATCH)
            for b in range(Self.BATCH):
                blk._grad_seed.data[b] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
        else:
            var c = ctx.value()
            blk._inp[0].ensure_gpu(c, Self.BATCH * Self.OBS_DIM)
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
        self.graph.children[8].set_multiplier(alpha)  # Scale node α

        # seed the graph input slot with s.
        comptime if target == "cpu":
            self._inp[0].ensure(BB * Self.OBS_DIM)
            for q in range(BB * Self.OBS_DIM):
                self._inp[0].data[q] = mb_s.data[q]
        else:
            var c = ctx.value()
            self._inp[0].ensure_gpu(c, BB * Self.OBS_DIM)
            c.enqueue_copy(self._inp[0].dev.value(), mb_s.dev.value())

        self.graph.forward[BB, target, POLICY](
            self._edges, self._inp, self._loss_out, ctx, actor, critic1, critic2
        )

        # mean loss + mean log_prob (host reduction; D2H on GPU).
        comptime if target == "gpu":
            self._loss_out.download(ctx.value())
            self.graph.node_output(3).download(ctx.value())
        var loss_sum: Scalar[DT] = 0.0
        var lp_sum: Scalar[DT] = 0.0
        for b in range(BB):
            loss_sum += self._loss_out.data[b]
            lp_sum += self.graph.node_output(3).data[b]
        var inv_b = Scalar[DT](1.0) / Scalar[DT](BB)

        # backward (seed = 1/BATCH) + actor step. Grad flows through the
        # critics (param grads discarded) into the actor (stepped).
        self.graph.vjp[BB, target, POLICY](
            self._edges, self._grad_seed, self._gin, ctx, actor, critic1, critic2
        )
        actor_opt.step[target, M=Self.ACTOR](actor, ctx)

        return SACActorLossOut(
            loss=loss_sum * inv_b, log_prob_mean=lp_sum * inv_b
        )
