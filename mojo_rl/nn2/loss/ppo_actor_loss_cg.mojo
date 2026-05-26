"""PPOActorLossCG — PPO actor loss as a single ComputeGraph.

Phase I.2.c FullGraph. The 4-node graph captures the actor-side loss
end-to-end; the loss block degenerates to "bind external actor, set
inputs, forward, seed, backward, step".

Graph topology:

    InputSlot ["s",          OBS_DIM]
    InputSlot ["aux",        ACT_DIM + 2]                      # action | old_log_prob | advantage
    ExternalNode ["actor_out", ACTOR,                  "s"]
    Node       ["loss_per_b", PPOObjective[ACT_DIM],   "actor_out", "aux"]

ACTOR is external (owned by the trainer). PPOObjective handles the
clipped-surrogate + entropy-bonus math (see `nn2/primitives/ppo_objective.mojo`).

The graph forward returns `loss_per_b` of shape [BATCH, 1]; the loss
block sums it host-side for logging, then seeds `1/BATCH` and walks
backward. PPOObjective's vjp writes `grad_aux = 0` (action / old_log_prob
/ advantage are non-differentiable), so the only meaningful gradient
flowing into the trainer is `grad_actor_output` accumulated through the
actor.

Public surface mirrors `SACActorLossCG`:
  - `make[target="cpu"]()` factory.
  - `forward_backward[target, OPT](actor, actor_opt, s_ptr, aux_ptr) -> Scalar[DT]`
    runs the whole step + returns the mean per-batch loss for logging.
  - `set_clip_eps(value)` / `set_entropy_coef(value)` runtime knobs.

Bespoke `PPOActorLoss[ACT]` produces per-element grad identical to this
form up to 3.7e-9 (see `tests/nn2/test_ppo_objective.mojo`).
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


struct PPOActorLossCG[
    ACTOR: Module,
    BATCH: Int,
](LossBlock):
    comptime OBS_DIM = Self.ACTOR.IN_DIM
    comptime ACT_DIM = Self.ACTOR.OUT_DIM // 2
    comptime AUX_DIM = Self.ACT_DIM + 2

    # The 4-node FullGraph.
    comptime ActorGraph = ComputeGraph[
        1,
        InputSlot["s",   Self.OBS_DIM],
        InputSlot["aux", Self.AUX_DIM],
        ExternalNode["actor_out", Self.ACTOR, "s"],
        Node["loss_per_b", PPOObjective[Self.ACT_DIM], "actor_out", "aux"],
    ]

    var graph: Self.ActorGraph

    # Scratch for the graph output (loss_per_b [BATCH, 1]) and the seed
    # gradient (1/BATCH [BATCH, 1]).
    var _loss_out: Scratch["loss_out", Self.BATCH]
    var _grad_seed: Scratch["grad_seed", Self.BATCH]

    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT_DIM, (
            "PPOActorLossCG: ACTOR.OUT_DIM must equal 2·ACT_DIM"
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
            "PPOActorLossCG.make[target='gpu'] not implemented yet "
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
        aux_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises -> Scalar[DT]:
        """Run one minibatch PPO actor update.

        `s_ptr` points to [BATCH, OBS_DIM] observations.
        `aux_ptr` points to [BATCH, ACT_DIM+2] packed
        `[action | old_log_prob | advantage]`.

        Returns the mean per-batch loss for logging.
        """
        assert_tag_for["PPOActorLossCG", target](self.ts.target_tag)
        comptime BB = Self.BATCH
        comptime OBS = Self.OBS_DIM
        comptime AUX = Self.AUX_DIM

        actor_opt.zero_grad[target, M=Self.ACTOR](actor)

        # Bind external actor (re-bind every call — the trainer may move
        # the field across optimizer-bundle indexing).
        self.graph.set_external["actor_out", Self.ACTOR](actor)

        # Set graph inputs. Hetero-variadic workaround: pass aux with
        # OBS_DIM-shaped Layout for the variadic pack to unify; the slot
        # uses its declared OUT_DIM internally.
        var s_t = TileTensor(s_ptr, row_major[BB, OBS]())
        var aux_t = TileTensor(aux_ptr, row_major[BB, AUX]())
        self.graph.set_input["s", BB](s_t)
        self.graph.set_input["aux", BB](aux_t)

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
                "PPOActorLossCG.forward_backward GPU not implemented "
                "(Phase I.2 CPU only)"
            )
            return Scalar[DT](0.0)
