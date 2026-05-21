"""SACActorLossCG — Phase 10E port of SACActorLoss to ComputeGraph v2.

Drop-in replacement for `SACActorLoss` (Phase 9A) with the same public
surface (`make[target]`, `forward_backward[target, OPT]`, public
`rsample` field) but built around a small ComputeGraph v2 DAG that
absorbs the post-critic chain.

The original chain split into two parts at the trainer-owned externals:

  pre-critic   (manual orchestration in this block):
      actor.forward(s)            →  actor_out [BATCH, 2·ACT]
      rsample.forward(actor_out)  →  [action | log_prob] [BATCH, ACT+1]
      (split action / log_prob)
      concat(s, action)           →  sa [BATCH, OBS+ACT]
      critic1.forward(sa), critic2.forward(sa) → q1, q2 [BATCH, 1] each

  post-critic  (one CG v2 graph, `_post_graph`):
      input = [q1 | q2 | log_prob] [BATCH, 3]
        Slice "q1_in"        cols [0, 1) → q1_in
        Slice "q2_in"        cols [1, 2) → q2_in
        Slice "log_prob_in"  cols [2, 3) → log_prob_in
        BinaryElemMin "min_q" (q1_in, q2_in) → min_q
        Scale "alpha_lp"      (log_prob_in) → α · log_prob
        BinarySub "loss_per_b" (alpha_lp, min_q) → loss_per_b
      output = loss_per_b [BATCH, 1]

Backward symmetrically:
      seed dL/dL_per_b = 1/BATCH
      _post_graph.backward(seed) → grad over [q1 | q2 | log_prob]
      split → grad_q1, grad_q2, grad_log_prob
      critic1.backward[mode="input_only"](grad_q1) → grad_sa1
      critic2.backward[mode="input_only"](grad_q2) → grad_sa2
      grad_action = grad_sa1[OBS:] + grad_sa2[OBS:]
      pack [grad_action | grad_log_prob] → grad_alp [BATCH, ACT+1]
      rsample.backward(grad_alp) → grad_ao [BATCH, 2·ACT]
      actor.backward(grad_ao) — accumulates actor param grads

Bit-identical to `SACActorLoss` (Phase 9A) — same RNG order via the
single shared `rsample`, same fp32 add order in the critic-grad sum
(slice + scatter-add cancels into the same sequence), same element-wise
ops (Scale/Sub/ElemMin/Slice are pure memory copy + arithmetic without
reduction). Verified by `tests/nn2/test_sac_actor_loss_cg.mojo`.

CPU only (Phase 10E). GPU lands when CG v2's GPU path lights up
(Phase 10F or later).
"""

from std.memory import alloc
from layout import TileTensor, row_major

from ..constants import DT
from ..core import (
    Module,
    Optimizer,
    Initializer,
)
from ..core.target_storage import TargetStorage, assert_tag_for
from ..initializer import Zero
from ..combinators.compute_graph import ComputeGraph
from ..combinators.graph_nodes import UnaryNode, BinaryNode
from ..primitives.rsample import RSample
from ..primitives.scale import Scale
from ..primitives.slice import Slice
from ..primitives.binary_elem_min import BinaryElemMin
from ..primitives.binary_sub import BinarySub


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
](Movable & ImplicitlyDestructible):
    comptime OBS_DIM = Self.ACTOR.IN_DIM
    comptime ACT_DIM = Self.ACTOR.OUT_DIM // 2
    comptime SA_DIM = Self.OBS_DIM + Self.ACT_DIM

    # Post-critic graph: [q1 | q2 | log_prob] → loss_per_b.
    # Node order matters only for compile-time backward dispatch (reverse).
    # alpha_lp's Scale is at index 4 — `set_alpha` writes its `.multiplier`.
    comptime PostGraph = ComputeGraph[
        3, 1,
        UnaryNode["q1_in",       Slice[3, 0, 1],       "input"],
        UnaryNode["q2_in",       Slice[3, 1, 2],       "input"],
        UnaryNode["log_prob_in", Slice[3, 2, 3],       "input"],
        BinaryNode["min_q",      BinaryElemMin[1],     "q1_in", "q2_in"],
        UnaryNode["alpha_lp",    Scale[1],             "log_prob_in"],
        BinaryNode["loss_per_b", BinarySub[1],         "alpha_lp", "min_q"],
    ]

    # rsample is public so the trainer reuses it for env-action sampling.
    var rsample: RSample[Self.ACT_DIM]
    var _post_graph: Self.PostGraph

    # Minimal scratch — five buffers for the manual pre/post-graph glue.
    var _mb_ao: List[Scalar[DT]]            # [BATCH, 2·ACT]  actor.forward output
    var _mb_alp: List[Scalar[DT]]           # [BATCH, ACT+1]  rsample output [action | log_prob]
    var _mb_sa: List[Scalar[DT]]            # [BATCH, SA]     concat(s, action)
    var _mb_q1: List[Scalar[DT]]            # [BATCH, 1]      critic1 output
    var _mb_q2: List[Scalar[DT]]            # [BATCH, 1]      critic2 output
    var _mb_post_in: List[Scalar[DT]]       # [BATCH, 3]      [q1 | q2 | log_prob]
    var _mb_post_out: List[Scalar[DT]]      # [BATCH, 1]      loss_per_b
    var _mb_grad_post_out: List[Scalar[DT]] # [BATCH, 1]      seed = 1/BATCH
    var _mb_grad_post_in: List[Scalar[DT]]  # [BATCH, 3]      grad over [q1 | q2 | log_prob]
    var _mb_grad_q1: List[Scalar[DT]]       # [BATCH, 1]
    var _mb_grad_q2: List[Scalar[DT]]       # [BATCH, 1]
    var _mb_grad_sa1: List[Scalar[DT]]      # [BATCH, SA]
    var _mb_grad_sa2: List[Scalar[DT]]      # [BATCH, SA]
    var _mb_grad_alp: List[Scalar[DT]]      # [BATCH, ACT+1]  [grad_action | grad_log_prob]
    var _mb_grad_ao: List[Scalar[DT]]       # [BATCH, 2·ACT]
    var _mb_grad_obs_unused: List[Scalar[DT]]  # [BATCH, OBS]  thrown away

    var ts: TargetStorage

    def __init__(out self):
        self.rsample = RSample[Self.ACT_DIM]()
        self._post_graph = Self.PostGraph()
        self._mb_ao = List[Scalar[DT]]()
        self._mb_alp = List[Scalar[DT]]()
        self._mb_sa = List[Scalar[DT]]()
        self._mb_q1 = List[Scalar[DT]]()
        self._mb_q2 = List[Scalar[DT]]()
        self._mb_post_in = List[Scalar[DT]]()
        self._mb_post_out = List[Scalar[DT]]()
        self._mb_grad_post_out = List[Scalar[DT]]()
        self._mb_grad_post_in = List[Scalar[DT]]()
        self._mb_grad_q1 = List[Scalar[DT]]()
        self._mb_grad_q2 = List[Scalar[DT]]()
        self._mb_grad_sa1 = List[Scalar[DT]]()
        self._mb_grad_sa2 = List[Scalar[DT]]()
        self._mb_grad_alp = List[Scalar[DT]]()
        self._mb_grad_ao = List[Scalar[DT]]()
        self._mb_grad_obs_unused = List[Scalar[DT]]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
    ) raises -> Self:
        comptime assert target == "cpu", (
            "SACActorLossCG.make[target='gpu'] not yet implemented (Phase 10E CPU only)"
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
        blk.rsample = RSample[Self.ACT_DIM].make[target="cpu", INIT=Zero]()
        blk.rsample.action_scale = action_scale
        blk._post_graph = Self.PostGraph.make[target="cpu", INIT=Zero]()

        var zero: Scalar[DT] = 0.0
        blk._mb_ao.resize(Self.BATCH * 2 * Self.ACT_DIM, zero)
        blk._mb_alp.resize(Self.BATCH * (Self.ACT_DIM + 1), zero)
        blk._mb_sa.resize(Self.BATCH * Self.SA_DIM, zero)
        blk._mb_q1.resize(Self.BATCH, zero)
        blk._mb_q2.resize(Self.BATCH, zero)
        blk._mb_post_in.resize(Self.BATCH * 3, zero)
        blk._mb_post_out.resize(Self.BATCH, zero)
        blk._mb_grad_post_out.resize(Self.BATCH, zero)
        blk._mb_grad_post_in.resize(Self.BATCH * 3, zero)
        blk._mb_grad_q1.resize(Self.BATCH, zero)
        blk._mb_grad_q2.resize(Self.BATCH, zero)
        blk._mb_grad_sa1.resize(Self.BATCH * Self.SA_DIM, zero)
        blk._mb_grad_sa2.resize(Self.BATCH * Self.SA_DIM, zero)
        blk._mb_grad_alp.resize(Self.BATCH * (Self.ACT_DIM + 1), zero)
        blk._mb_grad_ao.resize(Self.BATCH * 2 * Self.ACT_DIM, zero)
        blk._mb_grad_obs_unused.resize(Self.BATCH * Self.OBS_DIM, zero)

        blk.ts = TargetStorage.make_cpu()
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
        comptime assert target == "cpu", (
            "SACActorLossCG.forward_backward: GPU path not yet implemented"
        )
        assert_tag_for["SACActorLossCG", target](self.ts.target_tag)
        comptime BB = Self.BATCH
        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM
        comptime SA = Self.SA_DIM

        actor_opt.zero_grad["cpu", M=Self.ACTOR](actor)

        # ── Pre-critic: actor → rsample → concat(s, action) ───────────────
        var mb_s_t = TileTensor(mb_s_ptr, row_major[BB, OBS]())
        var mb_ao_p = self._mb_ao.unsafe_ptr()
        var mb_ao_t = TileTensor(mb_ao_p, row_major[BB, 2 * ACT]())
        actor.forward["cpu", BB](mb_s_t, mb_ao_t)

        var mb_alp_p = self._mb_alp.unsafe_ptr()
        var mb_alp_t = TileTensor(mb_alp_p, row_major[BB, ACT + 1]())
        self.rsample.forward["cpu", BB](mb_ao_t, mb_alp_t)

        # Manual concat(s, action), with action sourced from mb_alp[:, 0:ACT].
        var mb_sa_p = self._mb_sa.unsafe_ptr()
        for b in range(BB):
            for d in range(OBS):
                mb_sa_p[b * SA + d] = mb_s_ptr[b * OBS + d]
            for j in range(ACT):
                mb_sa_p[b * SA + OBS + j] = mb_alp_p[b * (ACT + 1) + j]

        # Twin critic forwards.
        var mb_sa_t = TileTensor(mb_sa_p, row_major[BB, SA]())
        var mb_q1_p = self._mb_q1.unsafe_ptr()
        var mb_q2_p = self._mb_q2.unsafe_ptr()
        var mb_q1_t = TileTensor(mb_q1_p, row_major[BB, 1]())
        var mb_q2_t = TileTensor(mb_q2_p, row_major[BB, 1]())
        critic1.forward["cpu", BB](mb_sa_t, mb_q1_t)
        critic2.forward["cpu", BB](mb_sa_t, mb_q2_t)

        # ── Post-critic: pack [q1 | q2 | log_prob] → post_graph → loss ────
        var mb_post_in_p = self._mb_post_in.unsafe_ptr()
        for b in range(BB):
            mb_post_in_p[b * 3]     = mb_q1_p[b]
            mb_post_in_p[b * 3 + 1] = mb_q2_p[b]
            mb_post_in_p[b * 3 + 2] = mb_alp_p[b * (ACT + 1) + ACT]  # log_prob

        # Set α on alpha_lp (node index 4 = "alpha_lp" Scale).
        self._post_graph.nodes[4].op.multiplier = alpha

        var mb_post_in_t = TileTensor(mb_post_in_p, row_major[BB, 3]())
        var mb_post_out_p = self._mb_post_out.unsafe_ptr()
        var mb_post_out_t = TileTensor(mb_post_out_p, row_major[BB, 1]())
        self._post_graph.forward["cpu", BB](mb_post_in_t, mb_post_out_t)

        # Mean loss + mean log_prob.
        var loss_sum: Scalar[DT] = 0.0
        var lp_sum: Scalar[DT] = 0.0
        for b in range(BB):
            loss_sum += mb_post_out_p[b]
            lp_sum += mb_alp_p[b * (ACT + 1) + ACT]
        var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](BB)

        # ── Backward: seed → post_graph.backward → critics → rsample → actor ──
        var g_post_out_p = self._mb_grad_post_out.unsafe_ptr()
        for b in range(BB):
            g_post_out_p[b] = inv_b
        var g_post_out_t = TileTensor(g_post_out_p, row_major[BB, 1]())

        var g_post_in_p = self._mb_grad_post_in.unsafe_ptr()
        var g_post_in_t = TileTensor(g_post_in_p, row_major[BB, 3]())
        self._post_graph.backward["cpu", BB](g_post_out_t, g_post_in_t)

        # Unpack grad over [q1 | q2 | log_prob].
        var g_q1_p = self._mb_grad_q1.unsafe_ptr()
        var g_q2_p = self._mb_grad_q2.unsafe_ptr()
        for b in range(BB):
            g_q1_p[b] = g_post_in_p[b * 3]
            g_q2_p[b] = g_post_in_p[b * 3 + 1]
            # g_log_prob = g_post_in_p[b * 3 + 2] — packed into grad_alp below

        # Critic backward in input-only mode (frozen params, Phase 8.2
        # contract): slim Module trait collapsed backward_input into
        # backward[mode="input_only"] (audit Follow-up #7).
        var g_q1_t = TileTensor(g_q1_p, row_major[BB, 1]())
        var g_q2_t = TileTensor(g_q2_p, row_major[BB, 1]())
        var g_sa1_p = self._mb_grad_sa1.unsafe_ptr()
        var g_sa2_p = self._mb_grad_sa2.unsafe_ptr()
        var g_sa1_t = TileTensor(g_sa1_p, row_major[BB, SA]())
        var g_sa2_t = TileTensor(g_sa2_p, row_major[BB, SA]())
        critic1.backward["cpu", BB, mode="input_only"](g_q1_t, g_sa1_t)
        critic2.backward["cpu", BB, mode="input_only"](g_q2_t, g_sa2_t)

        # Sum action portions; pack [grad_action | grad_log_prob] → grad_alp.
        var g_alp_p = self._mb_grad_alp.unsafe_ptr()
        for b in range(BB):
            for j in range(ACT):
                g_alp_p[b * (ACT + 1) + j] = (
                    g_sa1_p[b * SA + OBS + j] + g_sa2_p[b * SA + OBS + j]
                )
            g_alp_p[b * (ACT + 1) + ACT] = g_post_in_p[b * 3 + 2]

        var g_alp_t = TileTensor(g_alp_p, row_major[BB, ACT + 1]())
        var g_ao_p = self._mb_grad_ao.unsafe_ptr()
        var g_ao_t = TileTensor(g_ao_p, row_major[BB, 2 * ACT]())
        self.rsample.backward["cpu", BB](g_alp_t, g_ao_t)

        var g_obs_p = self._mb_grad_obs_unused.unsafe_ptr()
        var g_obs_t = TileTensor(g_obs_p, row_major[BB, OBS]())
        actor.backward["cpu", BB](g_ao_t, g_obs_t)

        actor_opt.step["cpu", M=Self.ACTOR](actor)

        return SACActorLossOut(
            loss=loss_sum * inv_b,
            log_prob_mean=lp_sum * inv_b,
        )
