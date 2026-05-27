"""PPOActorLoss — PPO actor loss as a single ComputeGraph.

Phase I.2 (graph shape) + Phase I.2.5 (quaternary cleanup). The 6-node
graph captures the actor-side loss end-to-end via four separate
InputSlots — one per data input — now that I.2.5's GraphNode N-ary
refactor lifted the Node ARITY cap from 2 to 4.

Graph topology:

    InputSlot ["s",          OBS_DIM]
    InputSlot ["a",          ACT_DIM]
    InputSlot ["olp",        1]
    InputSlot ["adv",        1]
    ExternalNode ["actor_out", ACTOR,                                  "s"]
    Node       ["loss_per_b", PPOObjective[ACT_DIM],   "actor_out","a","olp","adv"]

ACTOR is external (owned by the trainer). PPOObjective is the
quaternary leaf that does the clipped-surrogate + entropy math (see
`nn2/primitives/ppo_objective.mojo`).

The graph forward returns `loss_per_b` of shape [BATCH, 1]; the loss
block sums it host-side for logging, then seeds `1/BATCH` and walks
backward. PPOObjective's vjp writes zeros into the three rollout-time
grad slots (action / olp / adv are non-differentiable), so the only
meaningful gradient flowing into the trainer is `grad_actor_output`
accumulated through the actor.

Public surface mirrors `SACActorLoss`:
  - `make[target="cpu"]()` factory.
  - `forward_backward[target, OPT](actor, actor_opt, s_ptr, a_ptr, olp_ptr, adv_ptr) -> Scalar[DT]`
    runs the whole step + returns the mean per-batch loss for logging.
  - `set_clip_eps(value)` / `set_entropy_coef(value)` runtime knobs.

I.2.5 cleanup: the previous version of this block packed
(action | old_log_prob | advantage) into a single InputSlot["aux"]
because Node/ExternalNode were capped at ARITY ≤ 2. That workaround is
gone — each data input now flows through its own named slot.
"""

from layout import TileTensor, row_major

from ..constants import DT
from ..core import Module, Optimizer
from ..core.amp import AMPPolicy, NoAMP
from ..core.scratch import Scratch
from ..core.scratch_walkers import init_scratch_auto
from ..core.target_storage import TargetStorage, assert_tag_for
from ..initializer import Zero
from ..combinators.compute_graph import ComputeGraph
from ..combinators.graph_nodes import (
    InputSlot,
    Node,
    ExternalNode,
)
from ..primitives.ppo_objective import PPOObjective
from .loss_block import LossBlock
from .seed_grad_inv_batch import seed_grad_inv_batch


struct PPOActorLoss[
    ACTOR: Module,
    BATCH: Int,
](LossBlock):
    comptime OBS_DIM = Self.ACTOR.IN_DIMS[0]
    comptime ACT_DIM = Self.ACTOR.OUT_DIM // 2

    # The 6-node FullGraph — four InputSlots, one ExternalNode, one Node.
    comptime ActorGraph = ComputeGraph[
        1,
        InputSlot["s",   Self.OBS_DIM],
        InputSlot["a",   Self.ACT_DIM],
        InputSlot["olp", 1],
        InputSlot["adv", 1],
        ExternalNode["actor_out", Self.ACTOR, "s"],
        Node[
            "loss_per_b", PPOObjective[Self.ACT_DIM],
            "actor_out", "a", "olp", "adv",
        ],
    ]

    var graph: Self.ActorGraph

    var _loss_out: Scratch["loss_out", Self.BATCH]
    var _grad_seed: Scratch["grad_seed", Self.BATCH]

    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT_DIM, (
            "PPOActorLoss: ACTOR.OUT_DIM must equal 2·ACT_DIM"
        )
        self.graph = Self.ActorGraph()
        self._loss_out = Scratch["loss_out", Self.BATCH]()
        self._grad_seed = Scratch["grad_seed", Self.BATCH]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        clip_eps: Scalar[DT] = Scalar[DT](0.2),
        entropy_coef: Scalar[DT] = Scalar[DT](0.0),
    ) raises -> Self:
        comptime assert target == "cpu", (
            "PPOActorLoss.make[target='gpu'] not implemented yet "
            "(Phase I.2 CPU only)"
        )
        var blk = Self()
        blk.graph = Self.ActorGraph.make[target="cpu", INIT=Zero]()
        blk.ts = TargetStorage.make_cpu()
        init_scratch_auto[Self, target="cpu"](blk)
        # Push runtime hyperparameters into the PPOObjective node.
        blk.graph.set_node_attr["loss_per_b", "clip_eps"](clip_eps)
        blk.graph.set_node_attr["loss_per_b", "entropy_coef"](entropy_coef)
        return blk^

    def set_clip_eps(mut self, value: Scalar[DT]):
        self.graph.set_node_attr["loss_per_b", "clip_eps"](value)

    def set_entropy_coef(mut self, value: Scalar[DT]):
        self.graph.set_node_attr["loss_per_b", "entropy_coef"](value)

    def forward_backward[
        target: StaticString,
        OPT: Optimizer,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut actor: Self.ACTOR,
        mut actor_opt: OPT,
        s_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        a_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        olp_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        adv_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises -> Scalar[DT]:
        """Run one minibatch PPO actor update.

        `s_ptr`    points to [BATCH, OBS_DIM] observations.
        `a_ptr`    points to [BATCH, ACT_DIM] actions (unbounded rollout samples).
        `olp_ptr`  points to [BATCH] (or [BATCH, 1]) old log-probs.
        `adv_ptr`  points to [BATCH] (or [BATCH, 1]) normalised advantages.

        Returns the mean per-batch loss for logging.
        """
        assert_tag_for["PPOActorLoss", target](self.ts.target_tag)
        comptime BB = Self.BATCH
        comptime OBS = Self.OBS_DIM
        comptime ACT = Self.ACT_DIM

        actor_opt.zero_grad[target, M=Self.ACTOR](actor)

        # Bind external actor (re-bind every call — the trainer may move
        # the field across optimizer-bundle indexing).
        self.graph.set_external["actor_out", Self.ACTOR](actor)

        # Set graph inputs. Hetero-variadic workaround: pass each tile
        # with the IN0_DIM-shaped Layout so the variadic pack unifies;
        # PPOObjective recovers the real per-input shape via typed_view.
        # The set_input plumbing uses each TileTensor's .ptr directly so
        # the Layout type here only matters for compile-time unification.
        var s_t = TileTensor(s_ptr, row_major[BB, OBS]())
        var a_t = TileTensor(a_ptr, row_major[BB, ACT]())
        var olp_t = TileTensor(olp_ptr, row_major[BB, 1]())
        var adv_t = TileTensor(adv_ptr, row_major[BB, 1]())
        self.graph.set_input["s", BB](s_t)
        self.graph.set_input["a", BB](a_t)
        self.graph.set_input["olp", BB](olp_t)
        self.graph.set_input["adv", BB](adv_t)

        comptime if target == "cpu":
            var loss_p = self._loss_out.cpu_ptr()
            var loss_t = TileTensor(loss_p, row_major[BB, 1]())
            self.graph.forward["cpu", BB, POLICY](loss_t)

            var loss_sum: Scalar[DT] = 0.0
            for b in range(BB):
                loss_sum += loss_p[b]
            var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](BB)
            var loss_mean = loss_sum * inv_b

            # Seed backward with 1/BATCH per row, then walk vjp.
            var grad_p = self._grad_seed.cpu_ptr()
            seed_grad_inv_batch["cpu", BB](grad_p)
            var grad_t = TileTensor(grad_p, row_major[BB, 1]())
            self.graph.vjp["cpu", BB, POLICY](grad_t)

            actor_opt.step["cpu", M=Self.ACTOR](actor)
            return loss_mean
        else:
            comptime assert False, (
                "PPOActorLoss.forward_backward GPU not implemented "
                "(Phase I.2 CPU only)"
            )
            return Scalar[DT](0.0)
