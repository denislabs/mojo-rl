"""TD3TargetYBlock — twin-critic target-y with clipped noise (FullGraph).

Phase 4.5 FullGraph migration. The block now owns a 10-node graph that
captures the full TD3 target-value formula with target-policy smoothing;
`step` collapses to "sample noise into scratch, set inputs, forward". No
inline GPU kernels, no manual clamp/add loops, no `concat_sa` helper.

Graph topology (computes only the BOOTSTRAP `γ·min(Q1',Q2')`; the reward
add and terminal mask are applied in `step`):

    InputSlot         ["sp",          OBS]
    InputSlot         ["noise",       ACT]                                  # sigma-scaled host-side
    ExternalNode ["a_sp",        ACTOR,                          "sp"]
    Node         ["noise_clip",  Clamp[ACT],                     "noise"]
    Node        ["a_plus_n",    Add[ACT, 2],                    "a_sp", "noise_clip"]
    Node         ["a_smoothed",  Clamp[ACT],                     "a_plus_n"]
    Node        ["sa",          Concat[OBS, ACT],               "sp", "a_smoothed"]
    ExternalNode ["q1",          CRITIC, "sa", MODE="input_only"]
    ExternalNode ["q2",          CRITIC, "sa", MODE="input_only"]
    Node        ["min_q",       BinaryElemMin[1],               "q1", "q2"]
    Node         ["gamma_q",     Scale[1],                       "min_q"]   (terminal)

`step` then writes `y[b] = r[b] + (1 − term[b])·gamma_q[b]` via the shared
`apply_terminal_mask`.

`MODE="input_only"` on both critics: target_y is a target, not a loss, so
no gradient flows through these critics on this path.

TD3 target-policy smoothing (Fujimoto et al., 2018):
    a'    = clamp(actor_target(s') + clamp(ε, -c, c), -action_scale, action_scale)
            with ε ~ N(0, σ_target^2)
    sa'   = concat(s', a')
    qmin  = min(critic1_target(sa'), critic2_target(sa'))
    y[b]  = r[b] + (1 − term[b])·γ·qmin

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

The TD bootstrap is masked per-sample by the natural-termination flag
(`term`): dropped on termination, kept on time-limit truncation (see
`feedback_ppo_pendulum_timelimit_gae`). For truncation-only envs (`term ≡
0`) this is exactly `r + γ·qmin` — bit-identical to the prior in-graph
`Add(r, γ·qmin)`.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.scratch import Scratch
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
from mojo_rl.nn.primitives.add import Add
from mojo_rl.nn.primitives.binary_elem_min import BinaryElemMin
from mojo_rl.nn.random.box_muller import box_muller_normal, box_muller_normal_gpu
from ..loss.loss_block import LossBlock
from ..training.terminal_mask import apply_terminal_mask
from ..training.trainer_block import TrainerState


def _scale_inplace_kernel[N: Int](
    buf: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    s: Scalar[DT],
):
    """`buf[idx] *= s` — σ-scale the device standard-normal noise buffer
    (TD3 target-policy smoothing). One thread per element."""
    var idx = Int(global_idx.x)
    if idx < N:
        buf[idx] = buf[idx] * s


struct TD3TargetYBlock[
    ACTOR: Module,
    CRITIC: Module,
    BATCH: Int,
    OBS: Int,
    ACT: Int,
](LossBlock):
    comptime SA_DIM = Self.OBS + Self.ACT

    # Graph computes only the BOOTSTRAP `gamma_q = γ·min(Q1',Q2')`; the reward
    # add and terminal mask `y = r + (1−term)·gamma_q` happen in `step`
    # (per-sample data, not a graph parameter). `r` is no longer a graph input.
    comptime TD3TargetYGraph = ComputeGraph[
        1,
        InputSlot["sp", Self.OBS],
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
    ]

    var graph: Self.TD3TargetYGraph
    var noise_buf: Scratch["noise", Self.BATCH * Self.ACT]

    var action_scale: Scalar[DT]
    var gamma: Scalar[DT]
    var noise_std: Scalar[DT]  # σ for target-policy smoothing
    var noise_clip: Scalar[DT]  # c — noise clamped to ±c·action_scale
    # Philox state for the GPU target-smoothing noise (gpu path only).
    var _noise_rng_seed: UInt64
    var _noise_rng_offset: UInt64
    var ts: TargetStorage

    def __init__(out self):
        self.graph = Self.TD3TargetYGraph()
        self.noise_buf = Scratch["noise", Self.BATCH * Self.ACT]()
        self.action_scale = Scalar[DT](1.0)
        self.gamma = Scalar[DT](0.99)
        self.noise_std = Scalar[DT](0.2)
        self.noise_clip = Scalar[DT](0.5)
        self._noise_rng_seed = UInt64(0x7D3_5EED_C0DE)
        self._noise_rng_offset = UInt64(0)
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString
    ](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        noise_std: Scalar[DT] = Scalar[DT](0.2),
        noise_clip: Scalar[DT] = Scalar[DT](0.5),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory (absorbed the former TD3TargetYStep).
        `ctx=None` on CPU; required on GPU. GPU samples target-policy noise
        on-device via Philox box-muller + a σ-scale kernel."""
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "TD3TargetYBlock: target must be 'cpu' or 'gpu'"
        comptime assert (
            Self.ACTOR.IN_DIMS[0] == Self.OBS
        ), "TD3TargetYBlock: ACTOR.IN_DIM must equal OBS"
        comptime assert (
            Self.ACTOR.OUT_DIM == Self.ACT
        ), "TD3TargetYBlock: ACTOR.OUT_DIM must equal ACT"
        comptime assert (
            Self.CRITIC.IN_DIMS[0] == Self.SA_DIM
        ), "TD3TargetYBlock: CRITIC.IN_DIM must equal OBS+ACT"
        comptime assert (
            Self.CRITIC.OUT_DIM == 1
        ), "TD3TargetYBlock: CRITIC.OUT_DIM must equal 1"
        var blk = Self()
        comptime if target == "cpu":
            blk.graph = Self.TD3TargetYGraph.make[target="cpu", INIT=Zero]()
            blk.noise_buf = Scratch["noise", Self.BATCH * Self.ACT].make_cpu()
            blk.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["TD3TargetYBlock.make[target='gpu']"](ctx)
            blk.graph = Self.TD3TargetYGraph.make[target="gpu", INIT=Zero](
                ctx_v
            )
            blk.noise_buf = Scratch["noise", Self.BATCH * Self.ACT].make_gpu(
                ctx_v
            )
            blk.ts = TargetStorage.make_gpu(ctx_v)
        blk.action_scale = action_scale
        blk.gamma = gamma
        blk.noise_std = noise_std
        blk.noise_clip = noise_clip
        # Bake noise-clip + action-clamp + γ into the graph; constant across
        # calls.
        var clip_lim = noise_clip * action_scale
        blk.graph.set_node_attr["noise_clip", "min_val"](-clip_lim)
        blk.graph.set_node_attr["noise_clip", "max_val"](clip_lim)
        blk.graph.set_node_attr["a_smoothed", "min_val"](-action_scale)
        blk.graph.set_node_attr["a_smoothed", "max_val"](action_scale)
        blk.graph.set_node_attr["gamma_q", "multiplier"](gamma)
        return blk^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut actor_target: Self.ACTOR,
        mut critic1_target: Self.CRITIC,
        mut critic2_target: Self.CRITIC,
        mb_sp_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_r_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_term_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Compute `mb_y[b] = r[b] + (1−term[b])·γ·min(Q1,Q2)(sp, smoothed a')`
        in-place into `mb_y_ptr`. The TD bootstrap is dropped on natural
        termination, kept on truncation (`term ≡ 0` → `r + γ·min(Q1,Q2)`,
        bit-identical to the prior unmasked target)."""
        assert_tag_for["TD3TargetYBlock", target](self.ts.target_tag)
        comptime N = Self.BATCH * Self.ACT

        # Sample standard-normal noise, then σ-scale in place. The graph's
        # `noise_clip` node clamps to ±(noise_clip · action_scale). CPU uses
        # std.random box-muller (bit-identity path); GPU uses Philox box-muller
        # + a σ-scale kernel (separate baseline, same math).
        var noise_p = self.noise_buf.target_ptr[target]()
        var sigma = self.noise_std * self.action_scale
        comptime if target == "cpu":
            box_muller_normal(noise_p, N)
            for k in range(N):
                noise_p[k] = noise_p[k] * sigma
        else:
            var ctx = self.ts.ctx.value()
            box_muller_normal_gpu[N](
                ctx, noise_p, self._noise_rng_seed, self._noise_rng_offset,
            )
            self._noise_rng_offset += UInt64(((N + 1) // 2) * 2)
            var noise_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](noise_p)
            comptime n_blocks = (N + TPB - 1) // TPB
            comptime scale_kernel = _scale_inplace_kernel[N]
            ctx.enqueue_function[scale_kernel](
                noise_lt, sigma, grid_dim=n_blocks, block_dim=TPB,
            )

        # Bind externals.
        self.graph.set_external["a_sp", Self.ACTOR](actor_target)
        self.graph.set_external["q1", Self.CRITIC](critic1_target)
        self.graph.set_external["q2", Self.CRITIC](critic2_target)

        # Set inputs (rank-2 views over rank-1 caller / scratch buffers).
        var mb_sp_t = TileTensor(mb_sp_ptr, row_major[Self.BATCH, Self.OBS]())
        var noise_t = TileTensor(noise_p, row_major[Self.BATCH, Self.ACT]())
        self.graph.set_input["sp", Self.BATCH](mb_sp_t)
        self.graph.set_input["noise", Self.BATCH](noise_t)

        # Forward writes the bootstrap `γ·min(Q1',Q2')` into mb_y (terminal
        # node `gamma_q`, OUT_DIM=1); then add reward + apply the terminal mask.
        var mb_y_t = TileTensor(mb_y_ptr, row_major[Self.BATCH, 1]())
        self.graph.forward[target, Self.BATCH, POLICY](mb_y_t)

        apply_terminal_mask[target, Self.BATCH](
            self.ts.ctx, mb_r_ptr, mb_term_ptr, mb_y_ptr,
        )

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut actor_t: Self.ACTOR,
        mut critic1_t: Self.CRITIC,
        mut critic2_t: Self.CRITIC,
    ) raises:
        """State-driven overload (absorbed the former TD3TargetYStep):
        unpacks the minibatch pointers from `state` and delegates to the
        positional `step`. Writes `state.mb_y` in-place."""
        self.step[target, POLICY](
            actor_t, critic1_t, critic2_t,
            state.mb_sp.target_ptr[target](),
            state.mb_r.target_ptr[target](),
            state.mb_d.target_ptr[target](),
            state.mb_y.target_ptr[target](),
        )
