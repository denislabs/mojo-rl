"""DDPGTargetYBlock — deterministic target-value compute (FullGraph).

Phase 4.5 FullGraph migration. The block now owns a 6-node graph that
captures the full DDPG target-value formula; `step` collapses to "bind
externals, set inputs, set γ, forward". No inline GPU kernels, no manual
clamp loops, no `concat_sa` helper.

Graph topology (computes only the BOOTSTRAP `γ·Q'`; the reward add and
terminal mask are applied in `step`):

    InputSlot         ["sp",          OBS]
    ExternalNode ["a_sp",        ACTOR,                          "sp"]
    Node         ["a_clipped",   Clamp[ACT],                     "a_sp"]
    Node        ["sa",          Concat[OBS, ACT],               "sp", "a_clipped"]
    ExternalNode ["q",           CRITIC, "sa", MODE="input_only"]
    Node         ["gamma_q",     Scale[1],                       "q"]   (terminal)

`step` then writes `y[b] = r[b] + (1 − term[b])·gamma_q[b]` via the shared
`apply_terminal_mask`.

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
    y[b]  = r[b] + (1 − term[b])·γ·q'[b]

The TD bootstrap is masked per-sample by the natural-termination flag
(`term`): dropped on termination, kept on time-limit truncation (see
`feedback_ppo_pendulum_timelimit_gae`). For truncation-only envs (`term ≡
0`) this is exactly `r + γ·q'` — bit-identical to the prior in-graph
`Add(r, γ·q')`.

Sibling of `TargetYBlock` (SAC) and `TD3TargetYBlock` (TD3) — DDPG-specific
shape (deterministic actor, single critic, no log_prob/min reduction).

Surface:
    DDPGTargetYBlock[ACTOR, CRITIC, BATCH, OBS, ACT]
        - `make[target](action_scale, gamma) raises -> Self`            (CPU)
        - `make[target](ctx, action_scale, gamma) raises -> Self`       (GPU)
        - `step[target](mut actor_target, mut critic_target,
                        mb_sp_ptr, mb_r_ptr, mb_term_ptr, mb_y_ptr) raises`
            Writes `mb_y_ptr` ([BATCH, 1] interpreted as [BATCH]) in-place.
"""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.target_storage import (
    TargetStorage, assert_tag_for, require_ctx,
)
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.combinators.graph_nodes import (
    InputSlot,
    Node,
    ExternalNode,
)
from mojo_rl.nn.primitives.clamp import Clamp
from mojo_rl.nn.primitives.scale import Scale
from mojo_rl.nn.primitives.concat import Concat
from ..loss.loss_block import LossBlock
from ..training.terminal_mask import apply_terminal_mask
from ..training.trainer_block import TrainerState


struct DDPGTargetYBlock[
    ACTOR: Module,
    CRITIC: Module,
    BATCH: Int,
    OBS: Int,
    ACT: Int,
](LossBlock):
    comptime SA_DIM = Self.OBS + Self.ACT

    # Graph computes only the BOOTSTRAP `gamma_q = γ·Q'`; the reward add and
    # terminal mask `y = r + (1−term)·gamma_q` happen in `step` (per-sample
    # data, not a graph parameter). `r` is no longer a graph input.
    comptime DDPGTargetYGraph = ComputeGraph[
        1,
        InputSlot["sp", Self.OBS],
        ExternalNode["a_sp", Self.ACTOR, "sp"],
        Node["a_clipped", Clamp[Self.ACT], "a_sp"],
        Node["sa", Concat[Self.OBS, Self.ACT], "sp", "a_clipped"],
        ExternalNode["q", Self.CRITIC, "sa", MODE="input_only"],
        Node["gamma_q", Scale[1], "q"],
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
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory (absorbed the former DDPGTargetYStep).
        `ctx=None` on CPU; required on GPU."""
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "DDPGTargetYBlock: target must be 'cpu' or 'gpu'"
        comptime assert (
            Self.ACTOR.IN_DIMS[0] == Self.OBS
        ), "DDPGTargetYBlock: ACTOR.IN_DIM must equal OBS"
        comptime assert (
            Self.ACTOR.OUT_DIM == Self.ACT
        ), "DDPGTargetYBlock: ACTOR.OUT_DIM must equal ACT"
        comptime assert (
            Self.CRITIC.IN_DIMS[0] == Self.SA_DIM
        ), "DDPGTargetYBlock: CRITIC.IN_DIM must equal OBS+ACT"
        comptime assert (
            Self.CRITIC.OUT_DIM == 1
        ), "DDPGTargetYBlock: CRITIC.OUT_DIM must equal 1"
        var blk = Self()
        comptime if target == "cpu":
            blk.graph = Self.DDPGTargetYGraph.make[target="cpu", INIT=Zero]()
            blk.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["DDPGTargetYBlock.make[target='gpu']"](ctx)
            blk.graph = Self.DDPGTargetYGraph.make[target="gpu", INIT=Zero](
                ctx_v
            )
            blk.ts = TargetStorage.make_gpu(ctx_v)
        blk.action_scale = action_scale
        blk.gamma = gamma
        # Bake action-scale clamp + γ into the graph; constant across calls.
        blk.graph.set_node_attr["a_clipped", "min_val"](-action_scale)
        blk.graph.set_node_attr["a_clipped", "max_val"](action_scale)
        blk.graph.set_node_attr["gamma_q", "multiplier"](gamma)
        return blk^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut actor_target: Self.ACTOR,
        mut critic_target: Self.CRITIC,
        mb_sp_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_r_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_term_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Compute `mb_y[b] = r[b] + (1−term[b])·γ·Q_t(sp, clamp(actor_t(sp),
        ±scale))` in-place into `mb_y_ptr`. The TD bootstrap is dropped on
        natural termination, kept on truncation (`term ≡ 0` → `r + γ·Q'`,
        bit-identical to the prior unmasked target)."""
        assert_tag_for["DDPGTargetYBlock", target](self.ts.target_tag)

        # Bind externals.
        self.graph.set_external["a_sp", Self.ACTOR](actor_target)
        self.graph.set_external["q", Self.CRITIC](critic_target)

        # Set inputs (rank-2 view over the rank-1 caller buffer).
        var mb_sp_t = TileTensor(mb_sp_ptr, row_major[Self.BATCH, Self.OBS]())
        self.graph.set_input["sp", Self.BATCH](mb_sp_t)

        # Forward writes the bootstrap `γ·Q'` into mb_y (terminal node
        # `gamma_q`, OUT_DIM=1); then add reward + apply the terminal mask.
        var mb_y_t = TileTensor(mb_y_ptr, row_major[Self.BATCH, 1]())
        self.graph.forward[target, Self.BATCH, POLICY](mb_y_t)

        apply_terminal_mask[target, Self.BATCH](
            self.ts.ctx,
            LayoutTensor[DT, Layout.row_major(Self.BATCH), MutAnyOrigin](mb_r_ptr),
            LayoutTensor[DT, Layout.row_major(Self.BATCH), MutAnyOrigin](mb_term_ptr),
            LayoutTensor[DT, Layout.row_major(Self.BATCH, 1), MutAnyOrigin](mb_y_ptr),
        )

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut actor_t: Self.ACTOR,
        mut critic_t: Self.CRITIC,
    ) raises:
        """State-driven overload (absorbed the former DDPGTargetYStep):
        unpacks the minibatch pointers from `state` and delegates to the
        positional `step`. Writes `state.mb_y` in-place."""
        self.step[target, POLICY](
            actor_t, critic_t,
            state.mb_sp.target_ptr[target](),
            state.mb_r.target_ptr[target](),
            state.mb_d.target_ptr[target](),
            state.mb_y.target_ptr[target](),
        )
