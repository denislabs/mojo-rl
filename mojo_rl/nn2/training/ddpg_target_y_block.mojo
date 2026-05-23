"""DDPGTargetYBlock — deterministic target-value compute (FullGraph).

Phase 4.5 FullGraph migration. The block now owns a 6-node graph that
captures the full DDPG target-value formula; `step` collapses to "bind
externals, set inputs, set γ, forward". No inline GPU kernels, no manual
clamp loops, no `concat_sa` helper.

Graph topology:

    InputSlot         ["sp",          OBS]
    InputSlot         ["r",           1]
    ExternalNode ["a_sp",        ACTOR,                          "sp"]
    Node         ["a_clipped",   Clamp[ACT],                     "a_sp"]
    Node        ["sa",          Concat[OBS, ACT],               "sp", "a_clipped"]
    ExternalNode ["q",           CRITIC, "sa", MODE="input_only"]
    Node         ["gamma_q",     Scale[1],                       "q"]
    Node        ["y",           Add[1, 2],                      "r", "gamma_q"]

`MODE="input_only"` on the critic: target_y is a target, not a loss, so
no gradient flows through this critic on this path.

Forward-only — `y` is a target for critic update, not a loss. Backward
is never called on this graph (Module trait still requires `vjp` but it's
dead code here).

Formula:
    a'    = actor_target(s')          (deterministic)
    a'    = clamp(a', -action_scale, action_scale)
    sa'   = concat(s', a')
    q'    = critic_target(sa')
    y[b]  = r[b] + γ·q'[b]

`nonterm=1.0` is hardcoded for Pendulum-style truncation envs (see
`feedback_ppo_pendulum_timelimit_gae`). Phase 5 will lift this.

Sibling of `TargetYBlock` (SAC) and `TD3TargetYBlock` (TD3) — DDPG-specific
shape (deterministic actor, single critic, no log_prob/min reduction).

Surface (unchanged from pre-Phase-4.5):
    DDPGTargetYBlock[ACTOR, CRITIC, BATCH, OBS, ACT]
        - `make[target](action_scale, gamma) raises -> Self`            (CPU)
        - `make[target](ctx, action_scale, gamma) raises -> Self`       (GPU)
        - `step[target](mut actor_target, mut critic_target,
                        mb_sp_ptr, mb_r_ptr, mb_y_ptr) raises`
            Writes `mb_y_ptr` ([BATCH, 1] interpreted as [BATCH]) in-place.
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from ..constants import DT
from ..core.module import Module
from ..core.target_storage import TargetStorage, assert_tag_for
from ..initializer import Zero
from ..combinators.compute_graph import ComputeGraph
from ..combinators.graph_nodes import (
    InputSlot,
    Node,
    ExternalNode,
)
from ..primitives.clamp import Clamp
from ..primitives.scale import Scale
from ..primitives.concat import Concat
from ..primitives.add import Add
from ..loss.loss_block import LossBlock


struct DDPGTargetYBlock[
    ACTOR: Module,
    CRITIC: Module,
    BATCH: Int,
    OBS: Int,
    ACT: Int,
](LossBlock):
    comptime SA_DIM = Self.OBS + Self.ACT

    comptime DDPGTargetYGraph = ComputeGraph[
        1,
        InputSlot["sp", Self.OBS],
        InputSlot["r", 1],
        ExternalNode["a_sp", Self.ACTOR, "sp"],
        Node["a_clipped", Clamp[Self.ACT], "a_sp"],
        Node["sa", Concat[Self.OBS, Self.ACT], "sp", "a_clipped"],
        ExternalNode["q", Self.CRITIC, "sa", MODE="input_only"],
        Node["gamma_q", Scale[1], "q"],
        Node["y", Add[1, 2], "r", "gamma_q"],
    ]

    var graph: Self.DDPGTargetYGraph
    var action_scale: Scalar[DT]
    var gamma: Scalar[DT]
    var ts: TargetStorage

    def __init__(out self):
        self.graph = Self.DDPGTargetYGraph()
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
        ), "DDPGTargetYBlock.make[target='gpu'] requires a DeviceContext"
        comptime assert (
            Self.ACTOR.IN_DIM == Self.OBS
        ), "DDPGTargetYBlock: ACTOR.IN_DIM must equal OBS"
        comptime assert (
            Self.ACTOR.OUT_DIM == Self.ACT
        ), "DDPGTargetYBlock: ACTOR.OUT_DIM must equal ACT"
        comptime assert (
            Self.CRITIC.IN_DIM == Self.SA_DIM
        ), "DDPGTargetYBlock: CRITIC.IN_DIM must equal OBS+ACT"
        comptime assert (
            Self.CRITIC.OUT_DIM == 1
        ), "DDPGTargetYBlock: CRITIC.OUT_DIM must equal 1"
        var blk = Self()
        blk.graph = Self.DDPGTargetYGraph.make[target="cpu", INIT=Zero]()
        blk.ts = TargetStorage.make_cpu()
        blk.action_scale = action_scale
        blk.gamma = gamma
        # Bake action-scale clamp + γ into the graph; constant across calls.
        blk.graph.set_node_attr["a_clipped", "min_val"](-action_scale)
        blk.graph.set_node_attr["a_clipped", "max_val"](action_scale)
        blk.graph.set_node_attr["gamma_q", "multiplier"](gamma)
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
        ), "DDPGTargetYBlock.make[target='cpu'](ctx) — drop ctx for CPU"
        comptime assert (
            Self.ACTOR.IN_DIM == Self.OBS
        ), "DDPGTargetYBlock: ACTOR.IN_DIM must equal OBS"
        comptime assert (
            Self.ACTOR.OUT_DIM == Self.ACT
        ), "DDPGTargetYBlock: ACTOR.OUT_DIM must equal ACT"
        comptime assert (
            Self.CRITIC.IN_DIM == Self.SA_DIM
        ), "DDPGTargetYBlock: CRITIC.IN_DIM must equal OBS+ACT"
        comptime assert (
            Self.CRITIC.OUT_DIM == 1
        ), "DDPGTargetYBlock: CRITIC.OUT_DIM must equal 1"
        var blk = Self()
        blk.graph = Self.DDPGTargetYGraph.make[target="gpu", INIT=Zero](ctx)
        blk.ts = TargetStorage.make_gpu(ctx)
        blk.action_scale = action_scale
        blk.gamma = gamma
        blk.graph.set_node_attr["a_clipped", "min_val"](-action_scale)
        blk.graph.set_node_attr["a_clipped", "max_val"](action_scale)
        blk.graph.set_node_attr["gamma_q", "multiplier"](gamma)
        return blk^

    def step[
        target: StaticString
    ](
        mut self,
        mut actor_target: Self.ACTOR,
        mut critic_target: Self.CRITIC,
        mb_sp_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_r_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Compute `mb_y[b] = r[b] + γ·Q_t(sp, clamp(actor_t(sp), ±scale))`
        in-place into `mb_y_ptr`. nonterm=1.0 baked in (Pendulum-style)."""
        assert_tag_for["DDPGTargetYBlock", target](self.ts.target_tag)

        # Bind externals.
        self.graph.set_external["a_sp", Self.ACTOR](actor_target)
        self.graph.set_external["q", Self.CRITIC](critic_target)

        # Set inputs (rank-2 views over the rank-1 caller buffers).
        var mb_sp_t = TileTensor(mb_sp_ptr, row_major[Self.BATCH, Self.OBS]())
        var mb_r_t = TileTensor(mb_r_ptr, row_major[Self.BATCH, 1]())
        self.graph.set_input["sp", Self.BATCH](mb_sp_t)
        self.graph.set_input["r", Self.BATCH](mb_r_t)

        # Forward into mb_y (graph's last node is `y`, OUT_DIM=1).
        var mb_y_t = TileTensor(mb_y_ptr, row_major[Self.BATCH, 1]())
        self.graph.forward[target, Self.BATCH](mb_y_t)
