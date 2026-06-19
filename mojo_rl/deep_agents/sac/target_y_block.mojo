"""TargetYBlock — SAC target-value computation (storage ComputeGraph + ExternalRef).

Computes  y[b] = r[b] + γ·(1−term[b])·( min(Q1_t,Q2_t)(s',a') − α·log π(a'|s') )
with a' ~ π(·|s') from the ONLINE actor and Q from the two TARGET critics.

STORAGE migration (Stage 5): the legacy named-node graph (InputSlot/ExternalNode/
set_external) is rebuilt on the storage `ComputeGraph[NUM_IN, *NODES]` with
`ExternalRef` marker nodes for the shared actor + target critics (threaded as
tracked `mut` refs into `graph.forward`). The graph computes `min_q` (its output)
and `log_prob` (read via `node_output`); the reward add + terminal mask + α/γ
arithmetic fold into the `sac_target_y` helper (host α — no device-α pointer /
CUDA-graph capture, which is deferred project-wide). Forward-only — `y` is a
target, no grad flows here.

  graph: ExternalRef[ACTOR] → RSample → {Slice(action), Slice(logp)} →
         Concat2(s', action) → ExternalRef[CRITIC]×2 → BinaryElemMin = min_q

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
from mojo_rl.nn.storage.core.tensor_pack import TensorPack
from mojo_rl.nn.storage.core.initializer import Zero
from mojo_rl.nn.storage.primitives.rsample import RSample
from mojo_rl.nn.storage.primitives.slice import Slice
from mojo_rl.nn.storage.primitives.concat import Concat2
from mojo_rl.nn.storage.primitives.binary_elementwise import BinaryElemMin
from mojo_rl.nn.storage.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.storage.combinators.external_ref import ExternalRef
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
    # ExternalRef markers, supplied at forward by the trainer (tracked refs).
    # min_q is node 7 (the output); log_prob is node 3 (read via node_output).
    comptime Graph = ComputeGraph[
        1,
        ExternalRef[Self.ACTOR],                       # 0 s' → [mu|ls]
        RSample[Self.ACT],                             # 1 → [a'|logp']
        Slice[Self.ALP_DIM, 0, Self.ACT],              # 2 action a'
        Slice[Self.ALP_DIM, Self.ACT, Self.ALP_DIM],   # 3 log_prob
        Concat2[Self.OBS, Self.ACT],                   # 4 (s', a')
        ExternalRef[Self.CRITIC],                      # 5 q1
        ExternalRef[Self.CRITIC],                      # 6 q2
        BinaryElemMin[1],                              # 7 min_q (output)
    ]

    var graph: Self.Graph
    var action_scale: Scalar[DT]
    var gamma: Scalar[DT]
    var _edges: List[List[Int]]
    var _inp: TensorPack[1]   # holds s' for the graph input slot
    var _min_q: Tensor        # graph output

    def __init__(out self):
        self.graph = Self.Graph()
        self.action_scale = Scalar[DT](1.0)
        self.gamma = Scalar[DT](0.99)
        self._edges = Self._build_edges()
        self._inp = TensorPack[1]()
        self._min_q = Tensor()

    @staticmethod
    def _build_edges() -> List[List[Int]]:
        var e = List[List[Int]]()
        e.append([0])      # 0 actor(s')         slot0 = input s'
        e.append([1])      # 1 rsample(actor_out=slot1)
        e.append([2])      # 2 slice action(alp=slot2)
        e.append([2])      # 3 slice logp(alp=slot2)
        e.append([0, 3])   # 4 concat(s'=slot0, action=slot3)
        e.append([5])      # 5 critic1(sa=slot5)
        e.append([5])      # 6 critic2(sa=slot5)
        e.append([6, 7])   # 7 min(q1=slot6, q2=slot7)
        return e^

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
        # the RSample node (index 1) carries the action bound.
        blk.graph.children[1].action_scale = action_scale
        blk.action_scale = action_scale
        blk.gamma = gamma
        comptime if target == "cpu":
            blk._inp[0].ensure(Self.BATCH * Self.OBS)
            blk._min_q = Tensor.alloc(Self.BATCH)
        else:
            var c = ctx.value()
            blk._inp[0].ensure_gpu(c, Self.BATCH * Self.OBS)
            blk._min_q = Tensor.alloc_gpu(c, Self.BATCH)
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
        # Seed the graph input slot with s'.
        comptime if target == "cpu":
            self._inp[0].ensure(Self.BATCH * Self.OBS)
            for q in range(Self.BATCH * Self.OBS):
                self._inp[0].data[q] = state.mb_sp.data[q]
        else:
            var c = ctx.value()
            self._inp[0].ensure_gpu(c, Self.BATCH * Self.OBS)
            c.enqueue_copy(self._inp[0].dev.value(), state.mb_sp.dev.value())

        # Forward the graph (actor + target critics threaded as tracked refs).
        self.graph.forward[Self.BATCH, target, POLICY](
            self._edges, self._inp, self._min_q, ctx, actor, tgt1, tgt2
        )

        # y = r + γ·(1−done)·(min_q − α·logp). min_q = graph output;
        # logp = node_output(3) (the Slice(logp) branch, [B]). α = host scalar.
        sac_target_y[target, Self.BATCH](
            state.mb_r,
            state.mb_d,
            self._min_q,
            self.graph.node_output(3),
            self.gamma,
            state.alpha,
            state.mb_y,
            ctx,
        )
