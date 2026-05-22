"""SACActorLossCG — SAC actor loss with a ComputeGraph post-critic head.

Public surface: `make[target]`, `forward_backward[target, OPT]`, plus a
public `rsample` field. CPU + GPU.

The chain splits in two at the trainer-owned externals:

  pre-critic   (manual orchestration in this block):
      actor.forward(s)            →  actor_out [BATCH, 2·ACT]
      rsample.forward(actor_out)  →  [action | log_prob] [BATCH, ACT+1]
      (split action / log_prob)
      concat(s, action)           →  sa [BATCH, OBS+ACT]
      critic1.forward(sa), critic2.forward(sa) → q1, q2 [BATCH, 1] each

  post-critic  (one ComputeGraph, `_post_graph`):
      input = [q1 | q2 | log_prob] [BATCH, 3]
        Slice "q1_in"        cols [0, 1) → q1_in
        Slice "q2_in"        cols [1, 2) → q2_in
        Slice "log_prob_in"  cols [2, 3) → log_prob_in
        BinaryElemMin "min_q" (q1_in, q2_in) → min_q
        Scale "alpha_lp"      (log_prob_in) → α · log_prob
        BinarySub "loss_per_b" (alpha_lp, min_q) → loss_per_b
      output = loss_per_b [BATCH, 1]

Backward symmetrically. GPU mirrors the CPU pipeline via the kernels
`_concat_s_action_from_alp_kernel`, `_pack_post_in_kernel`,
`_unpack_grad_post_in_kernel`, `_combine_grad_alp_kernel` interleaved
with `actor`/`critic`/`rsample`/`_post_graph` device dispatches.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.memory import alloc
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT
from ..core import (
    Module,
    Optimizer,
    Initializer,
)
from ..core.target_tag import TARGET_GPU
from ..core.target_storage import TargetStorage, assert_tag_for
from ..initializer import Zero
from ..combinators.compute_graph import ComputeGraph
from ..combinators.graph_nodes import InputSlot, UnaryNode, BinaryNode
from ..primitives.rsample import RSample
from ..primitives.scale import Scale
from ..primitives.slice import Slice
from ..primitives.binary_elem_min import BinaryElemMin
from ..primitives.binary_sub import BinarySub
from .loss_block import LossBlock


# ──────────────────────────────────────────────────────────────────────
# Inline GPU glue kernels — block D.
# ──────────────────────────────────────────────────────────────────────


def _concat_s_action_from_alp_kernel[
    OBS: Int, ACT: Int, BATCH: Int,
](
    s: LayoutTensor[DT, Layout.row_major(BATCH, OBS), MutAnyOrigin],
    alp: LayoutTensor[
        DT, Layout.row_major(BATCH, ACT + 1), MutAnyOrigin,
    ],
    sa: LayoutTensor[
        DT, Layout.row_major(BATCH, OBS + ACT), MutAnyOrigin,
    ],
):
    """sa[b, :OBS] = s[b, :OBS]; sa[b, OBS:] = alp[b, :ACT]. One thread
    per [b, d] over sa's full shape."""
    var idx = Int(global_idx.x)
    comptime SA = OBS + ACT
    var total = BATCH * SA
    if idx < total:
        var b = idx // SA
        var d = idx % SA
        if d < OBS:
            sa[b, d] = rebind[Scalar[DT]](s[b, d])
        else:
            sa[b, d] = rebind[Scalar[DT]](alp[b, d - OBS])


def _pack_post_in_kernel[
    ACT: Int, BATCH: Int,
](
    q1: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    q2: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    alp: LayoutTensor[
        DT, Layout.row_major(BATCH, ACT + 1), MutAnyOrigin,
    ],
    post_in: LayoutTensor[
        DT, Layout.row_major(BATCH, 3), MutAnyOrigin,
    ],
):
    """post_in[b] = (q1[b], q2[b], alp[b, ACT])."""
    var b = Int(global_idx.x)
    if b < BATCH:
        post_in[b, 0] = rebind[Scalar[DT]](q1[b, 0])
        post_in[b, 1] = rebind[Scalar[DT]](q2[b, 0])
        post_in[b, 2] = rebind[Scalar[DT]](alp[b, ACT])


def _unpack_grad_post_in_kernel[
    BATCH: Int,
](
    grad_post_in: LayoutTensor[
        DT, Layout.row_major(BATCH, 3), MutAnyOrigin,
    ],
    grad_q1: LayoutTensor[
        DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
    ],
    grad_q2: LayoutTensor[
        DT, Layout.row_major(BATCH, 1), MutAnyOrigin,
    ],
):
    """grad_q1[b, 0] = grad_post_in[b, 0]; grad_q2[b, 0] = grad_post_in[b, 1].
    grad_log_prob (col 2) is consumed by `_combine_grad_alp_kernel` directly."""
    var b = Int(global_idx.x)
    if b < BATCH:
        grad_q1[b, 0] = rebind[Scalar[DT]](grad_post_in[b, 0])
        grad_q2[b, 0] = rebind[Scalar[DT]](grad_post_in[b, 1])


def _combine_grad_alp_kernel[
    OBS: Int, ACT: Int, BATCH: Int,
](
    grad_sa1: LayoutTensor[
        DT, Layout.row_major(BATCH, OBS + ACT), MutAnyOrigin,
    ],
    grad_sa2: LayoutTensor[
        DT, Layout.row_major(BATCH, OBS + ACT), MutAnyOrigin,
    ],
    grad_post_in: LayoutTensor[
        DT, Layout.row_major(BATCH, 3), MutAnyOrigin,
    ],
    grad_alp: LayoutTensor[
        DT, Layout.row_major(BATCH, ACT + 1), MutAnyOrigin,
    ],
):
    """grad_alp[b, j] = grad_sa1[b, OBS+j] + grad_sa2[b, OBS+j] for j < ACT;
    grad_alp[b, ACT] = grad_post_in[b, 2] (the log_prob gradient)."""
    var idx = Int(global_idx.x)
    comptime ALP = ACT + 1
    var total = BATCH * ALP
    if idx < total:
        var b = idx // ALP
        var j = idx % ALP
        if j < ACT:
            grad_alp[b, j] = (
                rebind[Scalar[DT]](grad_sa1[b, OBS + j])
                + rebind[Scalar[DT]](grad_sa2[b, OBS + j])
            )
        else:
            grad_alp[b, j] = rebind[Scalar[DT]](grad_post_in[b, 2])


def _fill_constant_kernel[N: Int](
    buf: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    value: Scalar[DT],
):
    var idx = Int(global_idx.x)
    if idx < N:
        buf[idx] = value


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
](LossBlock):
    comptime OBS_DIM = Self.ACTOR.IN_DIM
    comptime ACT_DIM = Self.ACTOR.OUT_DIM // 2
    comptime SA_DIM = Self.OBS_DIM + Self.ACT_DIM

    # InputSlot["input", 3] is nodes[0]; q1_in/q2_in/log_prob_in at 1/2/3,
    # min_q at 4, alpha_lp at 5, loss_per_b at 6.
    comptime PostGraph = ComputeGraph[
        1,
        InputSlot["input",       3],
        UnaryNode["q1_in",       Slice[3, 0, 1],       "input"],
        UnaryNode["q2_in",       Slice[3, 1, 2],       "input"],
        UnaryNode["log_prob_in", Slice[3, 2, 3],       "input"],
        BinaryNode["min_q",      BinaryElemMin[1],     "q1_in", "q2_in"],
        UnaryNode["alpha_lp",    Scale[1],             "log_prob_in"],
        BinaryNode["loss_per_b", BinarySub[1],         "alpha_lp", "min_q"],
    ]

    var rsample: RSample[Self.ACT_DIM]
    var _post_graph: Self.PostGraph

    # CPU scratch.
    var _mb_ao: List[Scalar[DT]]
    var _mb_alp: List[Scalar[DT]]
    var _mb_sa: List[Scalar[DT]]
    var _mb_q1: List[Scalar[DT]]
    var _mb_q2: List[Scalar[DT]]
    var _mb_post_in: List[Scalar[DT]]
    var _mb_post_out: List[Scalar[DT]]
    var _mb_grad_post_out: List[Scalar[DT]]
    var _mb_grad_post_in: List[Scalar[DT]]
    var _mb_grad_q1: List[Scalar[DT]]
    var _mb_grad_q2: List[Scalar[DT]]
    var _mb_grad_sa1: List[Scalar[DT]]
    var _mb_grad_sa2: List[Scalar[DT]]
    var _mb_grad_alp: List[Scalar[DT]]
    var _mb_grad_ao: List[Scalar[DT]]
    var _mb_grad_obs_unused: List[Scalar[DT]]

    # GPU scratch (block D).
    var _mb_ao_dev: Optional[DeviceBuffer[DT]]
    var _mb_alp_dev: Optional[DeviceBuffer[DT]]
    var _mb_sa_dev: Optional[DeviceBuffer[DT]]
    var _mb_q1_dev: Optional[DeviceBuffer[DT]]
    var _mb_q2_dev: Optional[DeviceBuffer[DT]]
    var _mb_post_in_dev: Optional[DeviceBuffer[DT]]
    var _mb_post_out_dev: Optional[DeviceBuffer[DT]]
    var _mb_grad_post_out_dev: Optional[DeviceBuffer[DT]]
    var _mb_grad_post_in_dev: Optional[DeviceBuffer[DT]]
    var _mb_grad_q1_dev: Optional[DeviceBuffer[DT]]
    var _mb_grad_q2_dev: Optional[DeviceBuffer[DT]]
    var _mb_grad_sa1_dev: Optional[DeviceBuffer[DT]]
    var _mb_grad_sa2_dev: Optional[DeviceBuffer[DT]]
    var _mb_grad_alp_dev: Optional[DeviceBuffer[DT]]
    var _mb_grad_ao_dev: Optional[DeviceBuffer[DT]]
    var _mb_grad_obs_unused_dev: Optional[DeviceBuffer[DT]]
    # Host-side staging for mean loss/log_prob (BATCH scalars each).
    var _mb_post_out_host: Optional[HostBuffer[DT]]
    var _mb_alp_host: Optional[HostBuffer[DT]]

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
        self._mb_ao_dev = None
        self._mb_alp_dev = None
        self._mb_sa_dev = None
        self._mb_q1_dev = None
        self._mb_q2_dev = None
        self._mb_post_in_dev = None
        self._mb_post_out_dev = None
        self._mb_grad_post_out_dev = None
        self._mb_grad_post_in_dev = None
        self._mb_grad_q1_dev = None
        self._mb_grad_q2_dev = None
        self._mb_grad_sa1_dev = None
        self._mb_grad_sa2_dev = None
        self._mb_grad_alp_dev = None
        self._mb_grad_ao_dev = None
        self._mb_grad_obs_unused_dev = None
        self._mb_post_out_host = None
        self._mb_alp_host = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
    ) raises -> Self:
        comptime assert target == "cpu", (
            "SACActorLossCG.make[target='gpu'] requires a DeviceContext"
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

    @staticmethod
    def make[target: StaticString](
        ctx: DeviceContext,
        action_scale: Scalar[DT] = Scalar[DT](1.0),
    ) raises -> Self:
        """GPU factory — allocates all device scratch and inner sub-graphs."""
        comptime assert target == "gpu", (
            "SACActorLossCG.make[target='cpu'](ctx) — drop ctx for CPU"
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
        blk.rsample = RSample[Self.ACT_DIM].make[target="gpu", INIT=Zero](ctx)
        blk.rsample.action_scale = action_scale
        blk._post_graph = Self.PostGraph.make[target="gpu", INIT=Zero](ctx)

        blk._mb_ao_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * 2 * Self.ACT_DIM)
        blk._mb_alp_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * (Self.ACT_DIM + 1))
        blk._mb_sa_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.SA_DIM)
        blk._mb_q1_dev = ctx.enqueue_create_buffer[DT](Self.BATCH)
        blk._mb_q2_dev = ctx.enqueue_create_buffer[DT](Self.BATCH)
        blk._mb_post_in_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * 3)
        blk._mb_post_out_dev = ctx.enqueue_create_buffer[DT](Self.BATCH)
        blk._mb_grad_post_out_dev = ctx.enqueue_create_buffer[DT](Self.BATCH)
        blk._mb_grad_post_in_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * 3)
        blk._mb_grad_q1_dev = ctx.enqueue_create_buffer[DT](Self.BATCH)
        blk._mb_grad_q2_dev = ctx.enqueue_create_buffer[DT](Self.BATCH)
        blk._mb_grad_sa1_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.SA_DIM)
        blk._mb_grad_sa2_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.SA_DIM)
        blk._mb_grad_alp_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * (Self.ACT_DIM + 1))
        blk._mb_grad_ao_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * 2 * Self.ACT_DIM)
        blk._mb_grad_obs_unused_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.OBS_DIM)
        blk._mb_post_out_host = ctx.enqueue_create_host_buffer[DT](Self.BATCH)
        blk._mb_alp_host = ctx.enqueue_create_host_buffer[DT](Self.BATCH * (Self.ACT_DIM + 1))

        blk.ts = TargetStorage.make_gpu(ctx)
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
        assert_tag_for["SACActorLossCG", target](self.ts.target_tag)

        comptime if target == "cpu":
            return self._forward_backward_cpu[OPT](
                actor, actor_opt, critic1, critic2, mb_s_ptr, alpha,
            )
        else:
            return self._forward_backward_gpu[OPT](
                actor, actor_opt, critic1, critic2, mb_s_ptr, alpha,
            )

    def _forward_backward_cpu[OPT: Optimizer](
        mut self,
        mut actor: Self.ACTOR,
        mut actor_opt: OPT,
        mut critic1: Self.CRITIC,
        mut critic2: Self.CRITIC,
        mb_s_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alpha: Scalar[DT],
    ) raises -> SACActorLossOut:
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

        # Set α on alpha_lp (Block B: InputSlot now at index 0, shifting
        # alpha_lp from 4 to 5).
        self._post_graph.nodes[5].op.multiplier = alpha

        var mb_post_in_t = TileTensor(mb_post_in_p, row_major[BB, 3]())
        var mb_post_out_p = self._mb_post_out.unsafe_ptr()
        var mb_post_out_t = TileTensor(mb_post_out_p, row_major[BB, 1]())
        self._post_graph.set_input["input", BB](mb_post_in_t)
        self._post_graph.forward["cpu", BB](mb_post_out_t)

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

        self._post_graph.backward["cpu", BB](g_post_out_t)
        # Block B: read the grad-of-input directly from the slot's
        # accumulator. Replaces the old external _mb_grad_post_in path.
        var g_post_in_p = self._post_graph.grad_input_ptr["input"]()

        # Unpack grad over [q1 | q2 | log_prob].
        var g_q1_p = self._mb_grad_q1.unsafe_ptr()
        var g_q2_p = self._mb_grad_q2.unsafe_ptr()
        for b in range(BB):
            g_q1_p[b] = g_post_in_p[b * 3]
            g_q2_p[b] = g_post_in_p[b * 3 + 1]

        var g_q1_t = TileTensor(g_q1_p, row_major[BB, 1]())
        var g_q2_t = TileTensor(g_q2_p, row_major[BB, 1]())
        var g_sa1_p = self._mb_grad_sa1.unsafe_ptr()
        var g_sa2_p = self._mb_grad_sa2.unsafe_ptr()
        var g_sa1_t = TileTensor(g_sa1_p, row_major[BB, SA]())
        var g_sa2_t = TileTensor(g_sa2_p, row_major[BB, SA]())
        critic1.backward["cpu", BB, mode="input_only"](g_q1_t, g_sa1_t)
        critic2.backward["cpu", BB, mode="input_only"](g_q2_t, g_sa2_t)

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

    def _forward_backward_gpu[OPT: Optimizer](
        mut self,
        mut actor: Self.ACTOR,
        mut actor_opt: OPT,
        mut critic1: Self.CRITIC,
        mut critic2: Self.CRITIC,
        mb_s_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alpha: Scalar[DT],
    ) raises -> SACActorLossOut:
        comptime BB = Self.BATCH
        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM
        comptime SA = Self.SA_DIM
        var ctx = self.ts.ctx.value()

        # Resolve device pointers from scratch.
        var ao_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_ao_dev.value().unsafe_ptr()
        )
        var alp_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_alp_dev.value().unsafe_ptr()
        )
        var sa_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_sa_dev.value().unsafe_ptr()
        )
        var q1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_q1_dev.value().unsafe_ptr()
        )
        var q2_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_q2_dev.value().unsafe_ptr()
        )
        var post_in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_post_in_dev.value().unsafe_ptr()
        )
        var post_out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_post_out_dev.value().unsafe_ptr()
        )
        var g_post_out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_grad_post_out_dev.value().unsafe_ptr()
        )
        var g_post_in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_grad_post_in_dev.value().unsafe_ptr()
        )
        var g_q1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_grad_q1_dev.value().unsafe_ptr()
        )
        var g_q2_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_grad_q2_dev.value().unsafe_ptr()
        )
        var g_sa1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_grad_sa1_dev.value().unsafe_ptr()
        )
        var g_sa2_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_grad_sa2_dev.value().unsafe_ptr()
        )
        var g_alp_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_grad_alp_dev.value().unsafe_ptr()
        )
        var g_ao_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_grad_ao_dev.value().unsafe_ptr()
        )
        var g_obs_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_grad_obs_unused_dev.value().unsafe_ptr()
        )

        actor_opt.zero_grad["gpu", M=Self.ACTOR](actor)

        # ── Pre-critic ─────────────────────────────────────────────────
        var mb_s_t = TileTensor(mb_s_ptr, row_major[BB, OBS]())
        var ao_t = TileTensor(ao_p, row_major[BB, 2 * ACT]())
        actor.forward["gpu", BB](mb_s_t, ao_t)

        var alp_t = TileTensor(alp_p, row_major[BB, ACT + 1]())
        self.rsample.forward["gpu", BB](ao_t, alp_t)

        # concat(s, action) where action comes from alp[:, 0:ACT].
        var s_lt = LayoutTensor[
            DT, Layout.row_major(BB, OBS), MutAnyOrigin,
        ](mb_s_ptr)
        var alp_lt = LayoutTensor[
            DT, Layout.row_major(BB, ACT + 1), MutAnyOrigin,
        ](alp_p)
        var sa_lt = LayoutTensor[
            DT, Layout.row_major(BB, SA), MutAnyOrigin,
        ](sa_p)
        comptime TPB = 128
        comptime n_blocks_sa = (BB * SA + TPB - 1) // TPB
        comptime concat_kernel = _concat_s_action_from_alp_kernel[OBS, ACT, BB]
        ctx.enqueue_function[concat_kernel](
            s_lt, alp_lt, sa_lt,
            grid_dim=n_blocks_sa, block_dim=TPB,
        )

        # Twin critic forwards.
        var sa_t = TileTensor(sa_p, row_major[BB, SA]())
        var q1_t = TileTensor(q1_p, row_major[BB, 1]())
        var q2_t = TileTensor(q2_p, row_major[BB, 1]())
        critic1.forward["gpu", BB](sa_t, q1_t)
        critic2.forward["gpu", BB](sa_t, q2_t)

        # Pack [q1 | q2 | log_prob] into post_in.
        var q1_lt = LayoutTensor[
            DT, Layout.row_major(BB, 1), MutAnyOrigin,
        ](q1_p)
        var q2_lt = LayoutTensor[
            DT, Layout.row_major(BB, 1), MutAnyOrigin,
        ](q2_p)
        var post_in_lt = LayoutTensor[
            DT, Layout.row_major(BB, 3), MutAnyOrigin,
        ](post_in_p)
        comptime n_blocks_b = (BB + TPB - 1) // TPB
        comptime pack_kernel = _pack_post_in_kernel[ACT, BB]
        ctx.enqueue_function[pack_kernel](
            q1_lt, q2_lt, alp_lt, post_in_lt,
            grid_dim=n_blocks_b, block_dim=TPB,
        )

        # Set α on alpha_lp (Block B: index 5 after InputSlot insertion).
        self._post_graph.nodes[5].op.multiplier = alpha

        var post_in_t = TileTensor(post_in_p, row_major[BB, 3]())
        var post_out_t = TileTensor(post_out_p, row_major[BB, 1]())
        self._post_graph.set_input["input", BB](post_in_t)
        self._post_graph.forward["gpu", BB](post_out_t)

        # Mean loss + mean log_prob via host-side reduction (BATCH scalars
        # each — at SAC scales (≤ 1024) this is much cheaper than launching
        # a reduction kernel + scalar download).
        ctx.enqueue_copy(self._mb_post_out_host.value(), post_out_p)
        ctx.enqueue_copy(self._mb_alp_host.value(), alp_p)
        ctx.synchronize()
        var loss_sum: Scalar[DT] = 0.0
        var lp_sum: Scalar[DT] = 0.0
        var post_out_hp = self._mb_post_out_host.value().unsafe_ptr()
        var alp_hp = self._mb_alp_host.value().unsafe_ptr()
        for b in range(BB):
            loss_sum += post_out_hp[b]
            lp_sum += alp_hp[b * (ACT + 1) + ACT]
        var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](BB)

        # Backward: seed grad_post_out = 1/BATCH.
        var g_post_out_lt = LayoutTensor[
            DT, Layout.row_major(BB), MutAnyOrigin,
        ](g_post_out_p)
        comptime fill_kernel = _fill_constant_kernel[BB]
        ctx.enqueue_function[fill_kernel](
            g_post_out_lt, inv_b,
            grid_dim=n_blocks_b, block_dim=TPB,
        )

        var g_post_out_t = TileTensor(g_post_out_p, row_major[BB, 1]())
        self._post_graph.backward["gpu", BB](g_post_out_t)
        # Block B: pull the grad-of-input pointer from the slot's
        # accumulator and rebind it as the existing g_post_in_p alias so
        # the downstream unpack kernel below reads the same data.
        g_post_in_p = self._post_graph.grad_input_ptr["input"]()

        # Unpack grad over [q1 | q2 | log_prob].
        var g_post_in_lt = LayoutTensor[
            DT, Layout.row_major(BB, 3), MutAnyOrigin,
        ](g_post_in_p)
        var g_q1_lt = LayoutTensor[
            DT, Layout.row_major(BB, 1), MutAnyOrigin,
        ](g_q1_p)
        var g_q2_lt = LayoutTensor[
            DT, Layout.row_major(BB, 1), MutAnyOrigin,
        ](g_q2_p)
        comptime unpack_kernel = _unpack_grad_post_in_kernel[BB]
        ctx.enqueue_function[unpack_kernel](
            g_post_in_lt, g_q1_lt, g_q2_lt,
            grid_dim=n_blocks_b, block_dim=TPB,
        )

        # Twin critic backward in input-only mode.
        var g_q1_t = TileTensor(g_q1_p, row_major[BB, 1]())
        var g_q2_t = TileTensor(g_q2_p, row_major[BB, 1]())
        var g_sa1_t = TileTensor(g_sa1_p, row_major[BB, SA]())
        var g_sa2_t = TileTensor(g_sa2_p, row_major[BB, SA]())
        critic1.backward["gpu", BB, mode="input_only"](g_q1_t, g_sa1_t)
        critic2.backward["gpu", BB, mode="input_only"](g_q2_t, g_sa2_t)

        # Combine: grad_alp[b, j] = grad_sa1[b, OBS+j] + grad_sa2[b, OBS+j]
        # (for j < ACT); grad_alp[b, ACT] = grad_post_in[b, 2].
        var g_sa1_lt = LayoutTensor[
            DT, Layout.row_major(BB, SA), MutAnyOrigin,
        ](g_sa1_p)
        var g_sa2_lt = LayoutTensor[
            DT, Layout.row_major(BB, SA), MutAnyOrigin,
        ](g_sa2_p)
        var g_alp_lt = LayoutTensor[
            DT, Layout.row_major(BB, ACT + 1), MutAnyOrigin,
        ](g_alp_p)
        comptime n_blocks_alp = (BB * (ACT + 1) + TPB - 1) // TPB
        comptime combine_kernel = _combine_grad_alp_kernel[OBS, ACT, BB]
        ctx.enqueue_function[combine_kernel](
            g_sa1_lt, g_sa2_lt, g_post_in_lt, g_alp_lt,
            grid_dim=n_blocks_alp, block_dim=TPB,
        )

        var g_alp_t = TileTensor(g_alp_p, row_major[BB, ACT + 1]())
        var g_ao_t = TileTensor(g_ao_p, row_major[BB, 2 * ACT]())
        self.rsample.backward["gpu", BB](g_alp_t, g_ao_t)

        var g_obs_t = TileTensor(g_obs_p, row_major[BB, OBS]())
        actor.backward["gpu", BB](g_ao_t, g_obs_t)
        actor_opt.step["gpu", M=Self.ACTOR](actor)

        return SACActorLossOut(
            loss=loss_sum * inv_b,
            log_prob_mean=lp_sum * inv_b,
        )
