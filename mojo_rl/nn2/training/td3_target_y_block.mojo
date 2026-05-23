"""TD3TargetYBlock — twin-critic target-y with clipped noise (FullGraph).

Phase 4.5 FullGraph migration. The block now owns a 10-node graph that
captures the full TD3 target-value formula with target-policy smoothing;
`step` collapses to "sample noise into scratch, set inputs, forward". No
inline GPU kernels, no manual clamp/add loops, no `concat_sa` helper.

Graph topology:

    InputSlot         ["sp",          OBS]
    InputSlot         ["r",           1]
    InputSlot         ["noise",       ACT]                                  # sigma-scaled host-side
    ExternalNode ["a_sp",        ACTOR,                          "sp"]
    Node         ["noise_clip",  Clamp[ACT],                     "noise"]
    Node        ["a_plus_n",    Add[ACT, 2],                    "a_sp", "noise_clip"]
    Node         ["a_smoothed",  Clamp[ACT],                     "a_plus_n"]
    Node        ["sa",          Concat[OBS, ACT],               "sp", "a_smoothed"]
    ExternalNode ["q1",          CRITIC, "sa", MODE="input_only"]
    ExternalNode ["q2",          CRITIC, "sa", MODE="input_only"]
    Node        ["min_q",       BinaryElemMin[1],               "q1", "q2"]
    Node         ["gamma_q",     Scale[1],                       "min_q"]
    Node        ["y",           Add[1, 2],                      "r", "gamma_q"]

`MODE="input_only"` on both critics: target_y is a target, not a loss, so
no gradient flows through these critics on this path.

TD3 target-policy smoothing (Fujimoto et al., 2018):
    a'    = clamp(actor_target(s') + clamp(ε, -c, c), -action_scale, action_scale)
            with ε ~ N(0, σ_target^2)
    sa'   = concat(s', a')
    qmin  = min(critic1_target(sa'), critic2_target(sa'))
    y[b]  = r[b] + γ·nonterm·qmin

Differences vs DDPG target-y:
  - Clipped Gaussian noise added to target action (smoothing → reduces
    overestimation from sharp critic peaks).
  - Min over twin target critics (the SAC trick, also used here).

Differences vs SAC target-y:
  - No α·log_prob term (deterministic policy).
  - Noise is clipped (not unclipped squashed-Gaussian).

Forward-only — `y` is a target for critic update, not a loss. CPU only
(noise sampling uses `box_muller_normal`; GPU enablement is future work
and would slot in a Philox-based noise node before `noise_clip`).

`nonterm=1.0` is hardcoded for Pendulum-style truncation envs (see
`feedback_ppo_pendulum_timelimit_gae`). Phase 5 will lift this.
"""

from layout import TileTensor, row_major

from ..constants import DT
from ..core.module import Module
from ..core.scratch import Scratch
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
from ..primitives.binary_elem_min import BinaryElemMin
from ..random.box_muller import box_muller_normal
from ..loss.loss_block import LossBlock


struct TD3TargetYBlock[
    ACTOR: Module,
    CRITIC: Module,
    BATCH: Int,
    OBS: Int,
    ACT: Int,
](LossBlock):
    comptime SA_DIM = Self.OBS + Self.ACT

    comptime TD3TargetYGraph = ComputeGraph[
        1,
        InputSlot["sp", Self.OBS],
        InputSlot["r", 1],
        InputSlot["noise", Self.ACT],
        ExternalNode["a_sp", Self.ACTOR, "sp"],
        Node["noise_clip", Clamp[Self.ACT], "noise"],
        Node["a_plus_n", Add[Self.ACT, 2], "a_sp", "noise_clip"],
        Node["a_smoothed", Clamp[Self.ACT], "a_plus_n"],
        Node["sa", Concat[Self.OBS, Self.ACT], "sp", "a_smoothed"],
        ExternalNode["q1", Self.CRITIC, "sa", MODE="input_only"],
        ExternalNode["q2", Self.CRITIC, "sa", MODE="input_only"],
        Node["min_q", BinaryElemMin[1], "q1", "q2"],
        Node["gamma_q", Scale[1], "min_q"],
        Node["y", Add[1, 2], "r", "gamma_q"],
    ]

    var graph: Self.TD3TargetYGraph
    var noise_buf: Scratch["noise", Self.BATCH * Self.ACT]

    var action_scale: Scalar[DT]
    var gamma: Scalar[DT]
    var noise_std: Scalar[DT]  # σ for target-policy smoothing
    var noise_clip: Scalar[DT]  # c — noise clamped to ±c·action_scale
    var ts: TargetStorage

    def __init__(out self):
        self.graph = Self.TD3TargetYGraph()
        self.noise_buf = Scratch["noise", Self.BATCH * Self.ACT]()
        self.action_scale = Scalar[DT](1.0)
        self.gamma = Scalar[DT](0.99)
        self.noise_std = Scalar[DT](0.2)
        self.noise_clip = Scalar[DT](0.5)
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString
    ](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        noise_std: Scalar[DT] = Scalar[DT](0.2),
        noise_clip: Scalar[DT] = Scalar[DT](0.5),
    ) raises -> Self:
        comptime assert target == "cpu", "TD3TargetYBlock: CPU only"
        comptime assert (
            Self.ACTOR.IN_DIM == Self.OBS
        ), "TD3TargetYBlock: ACTOR.IN_DIM must equal OBS"
        comptime assert (
            Self.ACTOR.OUT_DIM == Self.ACT
        ), "TD3TargetYBlock: ACTOR.OUT_DIM must equal ACT"
        comptime assert (
            Self.CRITIC.IN_DIM == Self.SA_DIM
        ), "TD3TargetYBlock: CRITIC.IN_DIM must equal OBS+ACT"
        comptime assert (
            Self.CRITIC.OUT_DIM == 1
        ), "TD3TargetYBlock: CRITIC.OUT_DIM must equal 1"
        var blk = Self()
        blk.graph = Self.TD3TargetYGraph.make[target="cpu", INIT=Zero]()
        blk.noise_buf = Scratch["noise", Self.BATCH * Self.ACT].make_cpu()
        blk.ts = TargetStorage.make_cpu()
        blk.action_scale = action_scale
        blk.gamma = gamma
        blk.noise_std = noise_std
        blk.noise_clip = noise_clip
        # Bake noise-clip + action-clamp + γ into the graph; constant across calls.
        var clip_lim = noise_clip * action_scale
        blk.graph.set_node_attr["noise_clip", "min_val"](-clip_lim)
        blk.graph.set_node_attr["noise_clip", "max_val"](clip_lim)
        blk.graph.set_node_attr["a_smoothed", "min_val"](-action_scale)
        blk.graph.set_node_attr["a_smoothed", "max_val"](action_scale)
        blk.graph.set_node_attr["gamma_q", "multiplier"](gamma)
        return blk^

    def step[
        target: StaticString,
    ](
        mut self,
        mut actor_target: Self.ACTOR,
        mut critic1_target: Self.CRITIC,
        mut critic2_target: Self.CRITIC,
        mb_sp_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_r_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        comptime assert target == "cpu", "TD3TargetYBlock: CPU only"
        assert_tag_for["TD3TargetYBlock", target](self.ts.target_tag)

        # Sample standard-normal noise host-side, then σ-scale in place.
        # The graph's `noise_clip` node clamps to ±(noise_clip · action_scale).
        var noise_p = self.noise_buf.cpu_ptr()
        box_muller_normal(noise_p, Self.BATCH * Self.ACT)
        var sigma = self.noise_std * self.action_scale
        for k in range(Self.BATCH * Self.ACT):
            noise_p[k] = noise_p[k] * sigma

        # Bind externals.
        self.graph.set_external["a_sp", Self.ACTOR](actor_target)
        self.graph.set_external["q1", Self.CRITIC](critic1_target)
        self.graph.set_external["q2", Self.CRITIC](critic2_target)

        # Set inputs (rank-2 views over rank-1 caller / scratch buffers).
        var mb_sp_t = TileTensor(mb_sp_ptr, row_major[Self.BATCH, Self.OBS]())
        var mb_r_t = TileTensor(mb_r_ptr, row_major[Self.BATCH, 1]())
        var noise_t = TileTensor(noise_p, row_major[Self.BATCH, Self.ACT]())
        self.graph.set_input["sp", Self.BATCH](mb_sp_t)
        self.graph.set_input["r", Self.BATCH](mb_r_t)
        self.graph.set_input["noise", Self.BATCH](noise_t)

        # Forward into mb_y (graph's last node is `y`, OUT_DIM=1).
        var mb_y_t = TileTensor(mb_y_ptr, row_major[Self.BATCH, 1]())
        self.graph.forward[target, Self.BATCH](mb_y_t)
