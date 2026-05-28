"""TargetYBlock — SAC target-y computation as a single ComputeGraph.

Phase 3.2 FullGraph migration. The block now owns a 14-node graph that
captures the full target-value formula; `step` collapses to "bind
externals, set inputs, set α/γ, forward". No inline GPU kernels.

Graph topology:

    InputSlot         ["sp",          OBS]
    InputSlot         ["r",           1]
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
    Node         ["gamma_softv", Scale[1],                       "soft_v"]    # multiplier=γ, set at make()
    Node        ["y",           Add[1, 2],                      "r", "gamma_softv"]

ACTOR, RSample, CRITIC are external. The trainer owns the actor and the
two target critics; this block owns its own RSample instance (separate
RNG state from the SAC actor loss's rsample, matching the pre-Phase-3
behavior). `MODE="input_only"` on the critics: target_y is a target,
not a loss, so no gradient flows through these critics on this path.

Forward-only — `y` is a target for critic update, not a loss. Backward
is never called on this graph. We still implement `Module.backward` on
all the nodes (the trait requires it) but it's dead code on this path.

`nonterm=1.0` is hardcoded for Pendulum-style truncation envs (see
`feedback_ppo_pendulum_timelimit_gae`). Phase 5 will lift this.

Surface (unchanged from pre-Phase-3.2):
    TargetYBlock[ACTOR, CRITIC, BATCH, OBS, ACT]
        - `make[target](action_scale, gamma) raises -> Self`            (CPU)
        - `make[target](ctx, action_scale, gamma) raises -> Self`       (GPU)
        - `step[target](mut actor, mut critic1_target, mut critic2_target,
                        mb_sp_ptr, mb_r_ptr, alpha, mb_y_ptr) raises`
            Writes `mb_y_ptr` ([BATCH, 1] interpreted as [BATCH]) in-place.
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for
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
from mojo_rl.nn2.primitives.add import Add
from ..loss.loss_block import LossBlock


struct TargetYBlock[
    ACTOR: Module,
    CRITIC: Module,
    BATCH: Int,
    OBS: Int,
    ACT: Int,
](LossBlock):
    comptime SA_DIM = Self.OBS + Self.ACT
    comptime ALP_DIM = Self.ACT + 1

    comptime TargetYGraph = ComputeGraph[
        1,
        InputSlot["sp", Self.OBS],
        InputSlot["r", 1],
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
        Node["y", Add[1, 2], "r", "gamma_softv"],
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
    ) raises -> Self:
        comptime assert (
            target == "cpu"
        ), "TargetYBlock.make[target='gpu'] requires a DeviceContext"
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
        blk.graph = Self.TargetYGraph.make[target="cpu", INIT=Zero]()
        blk.rsample = RSample[Self.ACT].make[target="cpu", INIT=Zero]()
        blk.rsample.action_scale = action_scale
        blk.ts = TargetStorage.make_cpu()
        blk.action_scale = action_scale
        blk.gamma = gamma
        # γ on the gamma_softv Scale node is constant across calls; set once at make.
        # (α on alpha_lp varies per step — set inside `step` from the caller's α.)
        blk.graph.set_node_attr["gamma_softv", "multiplier"](gamma)
        return blk^

    @staticmethod
    def make[
        target: StaticString
    ](
        ctx: DeviceContext,
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        gamma: Scalar[DT] = Scalar[DT](0.99),
    ) raises -> Self:
        """GPU factory."""
        comptime assert (
            target == "gpu"
        ), "TargetYBlock.make[target='cpu'](ctx) — drop ctx for CPU"
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
        blk.graph = Self.TargetYGraph.make[target="gpu", INIT=Zero](ctx)
        blk.rsample = RSample[Self.ACT].make[target="gpu", INIT=Zero](ctx)
        blk.rsample.action_scale = action_scale
        blk.ts = TargetStorage.make_gpu(ctx)
        blk.action_scale = action_scale
        blk.gamma = gamma
        blk.graph.set_node_attr["gamma_softv", "multiplier"](gamma)
        return blk^

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
        alpha: Scalar[DT],
        mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Compute `mb_y[b] = r[b] + γ·(min(Q1_t, Q2_t)(sp, a') − α·log_prob(a'|sp))`
        in-place into `mb_y_ptr`. nonterm=1.0 baked in (Pendulum-style).

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

        # Set inputs (rank-2 views over the rank-1 caller buffers).
        var mb_sp_t = TileTensor(mb_sp_ptr, row_major[Self.BATCH, Self.OBS]())
        var mb_r_t = TileTensor(mb_r_ptr, row_major[Self.BATCH, 1]())
        self.graph.set_input["sp", Self.BATCH](mb_sp_t)
        self.graph.set_input["r", Self.BATCH](mb_r_t)

        # α varies per call; γ was baked in at make().
        self.graph.set_node_attr["alpha_lp", "multiplier"](alpha)

        # Forward into mb_y (graph's last node is `y`, OUT_DIM=1).
        var mb_y_t = TileTensor(mb_y_ptr, row_major[Self.BATCH, 1]())
        self.graph.forward[target, Self.BATCH, POLICY](mb_y_t)
