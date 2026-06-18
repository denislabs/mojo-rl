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
`nn/primitives/ppo_objective.mojo`).

The graph forward returns `loss_per_b` of shape [BATCH, 1]; the loss
block sums it host-side for logging, then seeds `1/BATCH` and walks
backward. PPOObjective's vjp writes zeros into the three rollout-time
grad slots (action / olp / adv are non-differentiable), so the only
meaningful gradient flowing into the trainer is `grad_actor_output`
accumulated through the actor.

Public surface mirrors `SACActorLoss`:
  - `make[target](ctx=...)` unified CPU/GPU factory (ctx required for GPU).
  - `forward_backward[target, OPT](actor, actor_opt, s_ptr, a_ptr, olp_ptr, adv_ptr) -> Scalar[DT]`
    runs the whole step + returns the mean per-batch loss for logging.
  - `set_clip_eps(value)` / `set_entropy_coef(value)` runtime knobs.

I.2.5 cleanup: the previous version of this block packed
(action | old_log_prob | advantage) into a single InputSlot["aux"]
because Node/ExternalNode were capped at ARITY ≤ 2. That workaround is
gone — each data input now flows through its own named slot.

GPU path: depends on PPOObjective GPU forward + vjp kernels. The
forward_backward GPU branch mirrors SACActorLoss — device buffer for
loss_per_b, host buffer for the mean reduction (one device→host copy
+ sync per step).
"""

from std.gpu.host import DeviceContext, HostBuffer
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core import Module, Optimizer
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.scratch import Scratch
from mojo_rl.nn.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.combinators.graph_nodes import (
    InputSlot,
    Node,
    ExternalNode,
)
from .objective import PPOObjective
from ..loss.loss_block import LossBlock
from ..loss.seed_grad_inv_batch import seed_grad_inv_batch


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

    # GPU-only host staging buffer for the host-side mean reduction.
    var _loss_host: Optional[HostBuffer[DT]]

    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT_DIM, (
            "PPOActorLoss: ACTOR.OUT_DIM must equal 2·ACT_DIM"
        )
        self.graph = Self.ActorGraph()
        self._loss_out = Scratch["loss_out", Self.BATCH]()
        self._grad_seed = Scratch["grad_seed", Self.BATCH]()
        self._loss_host = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
        clip_eps: Scalar[DT] = Scalar[DT](0.2),
        entropy_coef: Scalar[DT] = Scalar[DT](0.0),
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "PPOActorLoss: target must be 'cpu' or 'gpu'"
        )
        var blk = Self()
        blk.graph = Self.ActorGraph.make[target, INIT=Zero](ctx=ctx)
        blk.ts = TargetStorage.make[target](ctx=ctx)
        init_scratch_auto[Self, target](blk, ctx)
        comptime if target == "gpu":
            var ctx_v = ctx.value()
            blk._loss_host = ctx_v.enqueue_create_host_buffer[DT](Self.BATCH)
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

        # ── Forward.
        var loss_p = self._loss_out.target_ptr[target]()
        var loss_t = TileTensor(loss_p, row_major[BB, 1]())
        self.graph.forward[target, BB, POLICY](loss_t)

        # ── Host-side mean reduction. CPU reads loss_per_b directly;
        # GPU stages it through a host buffer first.
        var loss_read_p = loss_p
        comptime if target == "gpu":
            var ctx = self.ts.ctx.value()
            ctx.enqueue_copy(self._loss_host.value(), loss_p)
            ctx.synchronize()
            loss_read_p = self._loss_host.value().unsafe_ptr()
        var loss_sum: Scalar[DT] = 0.0
        for b in range(BB):
            loss_sum += loss_read_p[b]
        var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](BB)
        var loss_mean = loss_sum * inv_b

        # ── Seed backward with 1/BATCH per row, then walk vjp.
        var grad_p = self._grad_seed.target_ptr[target]()
        seed_grad_inv_batch[target, BB](LayoutTensor[DT, Layout.row_major(BB, 1), MutAnyOrigin](grad_p), ctx=self.ts.ctx)
        var grad_t = TileTensor(grad_p, row_major[BB, 1]())
        self.graph.vjp[target, BB, POLICY](grad_t)

        actor_opt.step[target, M=Self.ACTOR](actor)
        return loss_mean
