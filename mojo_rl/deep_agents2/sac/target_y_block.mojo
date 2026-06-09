"""TargetYBlock — SAC target-y computation as a single ComputeGraph.

Phase 3.2 FullGraph migration. The block now owns a 14-node graph that
captures the full target-value formula; `step` collapses to "bind
externals, set inputs, set α/γ, forward". No inline GPU kernels.

Graph topology (computes only the BOOTSTRAP `γ·soft_v`; the reward add and
terminal mask are applied in `step`):

    InputSlot         ["sp",          OBS]
    ExternalNode ["actor_out",   ACTOR,                          "sp"]
    ExternalNode ["alp",         RSample[ACT],                   "actor_out"]
    Node         ["action",      Slice[ALP, 0, ACT],             "alp"]
    Node         ["log_prob",    Slice[ALP, ACT, ALP],           "alp"]
    Node        ["sa",          Concat[OBS, ACT],               "sp", "action"]
    ExternalNode ["q1",          CRITIC, "sa", MODE="input_only"]
    ExternalNode ["q2",          CRITIC, "sa", MODE="input_only"]
    Node        ["min_q",       BinaryElemMin[1],               "q1", "q2"]
    Node         ["alpha_lp",    Scale[1],                       "log_prob"]  # multiplier=α per call
    Node        ["soft_v",      BinarySub[1],                   "min_q", "alpha_lp"]
    Node         ["gamma_softv", Scale[1],                       "soft_v"]    # multiplier=γ, set at make()  (terminal)

`step` then writes `y[b] = r[b] + (1 − term[b])·gamma_softv[b]` (host loop
on CPU, `_mask_bootstrap_kernel` on GPU); `term` is the per-sample
natural-termination flag (drop bootstrap on termination, keep on
truncation — CleanRL semantics).

ACTOR, RSample, CRITIC are external. The trainer owns the actor and the
two target critics; this block owns its own RSample instance (separate
RNG state from the SAC actor loss's rsample, matching the pre-Phase-3
behavior). `MODE="input_only"` on the critics: target_y is a target,
not a loss, so no gradient flows through these critics on this path.

Forward-only — `y` is a target for critic update, not a loss. Backward
is never called on this graph. We still implement `Module.backward` on
all the nodes (the trait requires it) but it's dead code on this path.

The TD bootstrap is masked per-sample by the natural-termination flag
(`term`): kept on time-limit truncation, dropped on real termination (see
`feedback_ppo_pendulum_timelimit_gae`). For truncation-only envs (`term ≡
0`) the masked add reduces to `r + γ·soft_v` — bit-identical to the prior
in-graph `Add(r, γ·soft_v)`.

Surface:
    TargetYBlock[ACTOR, CRITIC, BATCH, OBS, ACT]
        - `make[target](action_scale, gamma) raises -> Self`            (CPU)
        - `make[target](ctx, action_scale, gamma) raises -> Self`       (GPU)
        - `step[target](mut actor, mut critic1_target, mut critic2_target,
                        mb_sp_ptr, mb_r_ptr, mb_term_ptr, alpha, mb_y_ptr)`
            Writes `mb_y_ptr` ([BATCH, 1] interpreted as [BATCH]) in-place.
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.target_storage import (
    TargetStorage, assert_tag_for, require_ctx,
)
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
from mojo_rl.nn2.primitives.concat import Concat
from mojo_rl.nn2.primitives.binary_elem_min import BinaryElemMin
from mojo_rl.nn2.primitives.binary_sub import BinarySub
from ..loss.loss_block import LossBlock
from ..training.terminal_mask import apply_terminal_mask
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

    # The graph computes the BOOTSTRAP term `gamma_softv = γ·(min_q − α·logp)`
    # only. The reward add and the terminal mask `y = r + (1−term)·gamma_softv`
    # happen in `step` (host loop / GPU kernel) because the mask is per-sample
    # data, not a graph parameter. `r` is therefore no longer a graph input.
    comptime TargetYGraph = ComputeGraph[
        1,
        InputSlot["sp", Self.OBS],
        ExternalNode["actor_out", Self.ACTOR, "sp"],
        ExternalNode["alp", RSample[Self.ACT], "actor_out"],
        Node["action", Slice[Self.ALP_DIM, 0, Self.ACT], "alp"],
        Node["log_prob", Slice[Self.ALP_DIM, Self.ACT, Self.ALP_DIM], "alp"],
        Node["sa", Concat[Self.OBS, Self.ACT], "sp", "action"],
        ExternalNode["q1", Self.CRITIC, "sa", MODE="input_only"],
        ExternalNode["q2", Self.CRITIC, "sa", MODE="input_only"],
        Node["min_q", BinaryElemMin[1], "q1", "q2"],
        Node["alpha_lp", Scale[1], "log_prob"],
        Node["soft_v", BinarySub[1], "min_q", "alpha_lp"],
        Node["gamma_softv", Scale[1], "soft_v"],
    ]

    var graph: Self.TargetYGraph
    var rsample: RSample[Self.ACT]  # owned — separate RNG from SAC actor loss

    var action_scale: Scalar[DT]
    var gamma: Scalar[DT]
    var ts: TargetStorage

    def __init__(out self):
        self.graph = Self.TargetYGraph()
        self.rsample = RSample[Self.ACT]()
        self.action_scale = Scalar[DT](1.0)
        self.gamma = Scalar[DT](0.99)
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString
    ](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory (absorbed the former TargetYStep wrapper).
        `ctx=None` on CPU; required on GPU (matmul-style Optional ctx)."""
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
        comptime if target == "cpu":
            blk.graph = Self.TargetYGraph.make[target="cpu", INIT=Zero]()
            blk.rsample = RSample[Self.ACT].make[target="cpu", INIT=Zero]()
            blk.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["TargetYBlock.make[target='gpu']"](ctx)
            blk.graph = Self.TargetYGraph.make[target="gpu", INIT=Zero](ctx_v)
            blk.rsample = RSample[Self.ACT].make[target="gpu", INIT=Zero](ctx_v)
            blk.ts = TargetStorage.make_gpu(ctx_v)
        blk.rsample.action_scale = action_scale
        blk.action_scale = action_scale
        blk.gamma = gamma
        # γ on the gamma_softv Scale node is constant across calls; set once at
        # make. (α on alpha_lp varies per step — set inside `step` from the
        # caller's α.)
        blk.graph.set_node_attr["gamma_softv", "multiplier"](gamma)
        return blk^

    def set_alpha_ptr(
        mut self, p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ):
        """One-time GPU wiring: point the `alpha_lp` Scale node at the
        device α buffer so the target-y forward reads α on-device. After
        this, `step` skips the per-step `set_node_attr` host bake."""
        self.graph.set_node_attr_ptr["alpha_lp", "multiplier"](p)

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut actor: Self.ACTOR,
        mut critic1_target: Self.CRITIC,
        mut critic2_target: Self.CRITIC,
        mb_sp_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_r_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_term_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alpha: Scalar[DT],
        mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Compute `mb_y[b] = r[b] + (1−term[b])·γ·(min(Q1_t, Q2_t)(sp, a')
        − α·log_prob(a'|sp))` in-place into `mb_y_ptr`.

        `mb_term_ptr` holds the per-sample natural-termination flag
        (1.0/0.0): the TD bootstrap is dropped on real termination and kept
        on time-limit truncation (CleanRL semantics). For envs that never
        terminate (`term ≡ 0`) this is exactly `r + γ·soft_v` — bit-identical
        to the previous unmasked target.

        The graph forward computes only the bootstrap `γ·soft_v` into
        `mb_y_ptr`; the reward add and mask are applied below.

        `POLICY` (Phase C.5) is threaded into the underlying
        `graph.forward` so the target-y compute can run with
        Bf16Compute when the trainer opts in. Default `NoAMP` is
        bit-identical to pre-C.5."""
        assert_tag_for["TargetYBlock", target](self.ts.target_tag)

        # Bind externals.
        self.graph.set_external["actor_out", Self.ACTOR](actor)
        self.graph.set_external["alp", RSample[Self.ACT]](self.rsample)
        self.graph.set_external["q1", Self.CRITIC](critic1_target)
        self.graph.set_external["q2", Self.CRITIC](critic2_target)

        # Set inputs (rank-2 view over the rank-1 caller buffer).
        var mb_sp_t = TileTensor(mb_sp_ptr, row_major[Self.BATCH, Self.OBS]())
        self.graph.set_input["sp", Self.BATCH](mb_sp_t)

        # α: CPU bakes the host scalar per call; γ was baked in at make().
        # On GPU α is read on-device via the `alpha_lp` multiplier_ptr wired
        # once at make (`set_alpha_ptr`) so the target-y forward is
        # CUDA-graph capturable — no per-step host work here.
        comptime if target == "cpu":
            self.graph.set_node_attr["alpha_lp", "multiplier"](alpha)

        # Forward writes the bootstrap `γ·soft_v` into mb_y (graph's last
        # node is `gamma_softv`, OUT_DIM=1).
        var mb_y_t = TileTensor(mb_y_ptr, row_major[Self.BATCH, 1]())
        self.graph.forward[target, Self.BATCH, POLICY](mb_y_t)

        # Reward add + terminal mask: mb_y[b] = r[b] + (1−term[b])·mb_y[b].
        apply_terminal_mask[target, Self.BATCH](
            self.ts.ctx, mb_r_ptr, mb_term_ptr, mb_y_ptr,
        )

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
        """State-driven overload (absorbed the former TargetYStep): unpacks
        the minibatch pointers from `state` and delegates to the positional
        `step`. Writes `state.mb_y` in-place."""
        self.step[target, POLICY](
            actor, tgt1, tgt2,
            state.mb_sp.target_ptr[target](),
            state.mb_r.target_ptr[target](),
            state.mb_d.target_ptr[target](),
            state.alpha,
            state.mb_y.target_ptr[target](),
        )
