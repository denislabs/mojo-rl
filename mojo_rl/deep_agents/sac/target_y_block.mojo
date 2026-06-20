"""TargetYBlock — SAC target-value computation (storage ComputeGraph, named DX).

Computes  y[b] = r[b] + γ·(1−term[b])·( min(Q1_t,Q2_t)(s',a') − α·log π(a'|s') )
with a' ~ π(·|s') from the ONLINE actor and Q from the two TARGET critics.

STORAGE migration (Stage 5): the graph is declared with the legacy NAME-wired DX
(`InputSlot`/`Node`/`ExternalNode`, edges by predecessor name — no runtime edge
list). The actor + the two target critics are `ExternalNode`s, threaded as
tracked `mut` refs into `graph.forward`. The graph computes `min_q` (its output)
and `log_prob` (read via `node_output["logp"]`); the reward add + terminal mask +
α/γ arithmetic fold into the `sac_target_y` helper (host α). Forward-only — `y`
is a target, no grad flows here.

  graph: s' → actor → rsample → {action, logp} ; (s', action) → concat →
         q1, q2 → min_q

Surface:
    TargetYBlock[ACTOR, CRITIC, BATCH, OBS, ACT]
        make[target](action_scale, gamma, ctx) -> Self
        step[target, POLICY](mut state, mut actor, mut tgt1, mut tgt2)
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.storage.core.module import Module
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.initializer import Zero
from mojo_rl.nn.storage.primitives.rsample import RSample
from mojo_rl.nn.storage.primitives.slice import Slice
from mojo_rl.nn.storage.primitives.concat import Concat2
from mojo_rl.nn.storage.primitives.binary_elementwise import BinaryElemMin
from mojo_rl.nn.storage.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.storage.combinators.graph_decl import InputSlot, Node, ExternalNode
from mojo_rl.nn.storage.loss.sac import sac_target_y
from ..loss.loss_block import LossBlock
from ..training.trainer_block import TrainerState


struct TargetYBlock[
    ACTOR: Module,
    CRITIC: Module,
    BATCH: Int,
    OBS: Int,
    ACT: Int,
](LossBlock):
    comptime SA_DIM = Self.OBS + Self.ACT
    comptime ALP_DIM = Self.ACT + 1

    # Owns the rsample/slice/concat/min nodes; the actor + 2 target critics are
    # ExternalNodes, supplied at forward by the trainer (tracked refs). min_q is
    # the output; log_prob is read by name via node_output["logp"].
    comptime Graph = ComputeGraph[
        InputSlot["s", Self.OBS],                            # s'
        ExternalNode["actor", Self.ACTOR, "s"],              # → [mu|ls]
        Node["rsample", RSample[Self.ACT], "actor"],         # → [a'|logp']
        Node["action", Slice[Self.ALP_DIM, 0, Self.ACT], "rsample"],
        Node["logp", Slice[Self.ALP_DIM, Self.ACT, Self.ALP_DIM], "rsample"],
        Node["concat", Concat2[Self.OBS, Self.ACT], "s", "action"],
        ExternalNode["q1", Self.CRITIC, "concat"],
        ExternalNode["q2", Self.CRITIC, "concat"],
        Node["min_q", BinaryElemMin[1], "q1", "q2"],         # output
    ]

    var graph: Self.Graph
    var action_scale: Scalar[DT]
    var gamma: Scalar[DT]
    var _min_q: Tensor        # graph output

    def __init__(out self):
        self.graph = Self.Graph()
        self.action_scale = Scalar[DT](1.0)
        self.gamma = Scalar[DT](0.99)
        self._min_q = Tensor()

    @staticmethod
    def make[
        target: StaticString
    ](
        action_scale: Scalar[DT] = 1.0,
        gamma: Scalar[DT] = 0.99,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "TargetYBlock: target must be 'cpu' or 'gpu'"
        comptime assert (
            Self.ACTOR.IN_DIMS[0] == Self.OBS
        ), "TargetYBlock: ACTOR.IN_DIM must equal OBS"
        comptime assert (
            Self.ACTOR.OUT_DIM == 2 * Self.ACT
        ), "TargetYBlock: ACTOR.OUT_DIM must equal 2·ACT"
        comptime assert (
            Self.CRITIC.IN_DIMS[0] == Self.SA_DIM
        ), "TargetYBlock: CRITIC.IN_DIM must equal OBS + ACT"
        comptime assert (
            Self.CRITIC.OUT_DIM == 1
        ), "TargetYBlock: CRITIC.OUT_DIM must equal 1"
        var blk = Self()
        blk.graph = Self.Graph.make[target, Zero](ctx)
        # the RSample node carries the action bound.
        blk.graph.set_node_attr["rsample", "action_scale"](action_scale)
        blk.action_scale = action_scale
        blk.gamma = gamma
        comptime if target == "cpu":
            blk._min_q = Tensor.alloc(Self.BATCH)
        else:
            blk._min_q = Tensor.alloc_gpu(ctx.value(), Self.BATCH)
        return blk^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut actor: Self.ACTOR,
        mut tgt1: Self.CRITIC,
        mut tgt2: Self.CRITIC,
    ) raises:
        var ctx = state.ctx
        # Seed the named input slot with s' (a COPY into the graph pool).
        self.graph.set_input["s", Self.BATCH](state.mb_sp, ctx)

        # Forward the graph (actor + target critics threaded as tracked refs).
        self.graph.forward[Self.BATCH, target, POLICY](
            self._min_q, ctx, actor, tgt1, tgt2
        )

        # y = r + γ·(1−done)·(min_q − α·logp). min_q = graph output;
        # logp = node_output["logp"] (the Slice(logp) branch, [B]). α = host scalar.
        sac_target_y[target, Self.BATCH](
            state.mb_r,
            state.mb_d,
            self._min_q,
            self.graph.node_output["logp"](),
            self.gamma,
            state.alpha,
            state.mb_y,
            ctx,
        )
