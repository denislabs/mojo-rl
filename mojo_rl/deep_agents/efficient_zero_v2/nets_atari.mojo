"""EfficientZeroV2 **Atari** networks (nn) — the spatial-latent model.

Full-parity port of the official EZv2 Atari backbone
(`references/EfficientZeroV2-main/ez/agents/models/base_model.py`,
`ez/config/exp/atari.yaml`). Unlike the MLP / pixel MuZero configs in this repo
— which collapse the encoder to a **vector** latent — EZv2 Atari keeps a
**spatial** latent ``[64, 6, 6]`` (``num_channels=64``) all the way through the
representation, dynamics and heads. ``state_norm=False`` (no MinMaxNorm on the
latent), ``num_blocks=1``, ``action_embedding=True`` (dim 16),
``reduced_channels=16``, value-prefix LSTM reward, 601-atom value/reward support.

Implementation lever (see ``docs/EZV2_ATARI_PARITY.md`` §B): the spatial latent
rides through the planner / replay / MCTS adapters as a **flat** ``LATENT = 64·6·6
= 2304`` vector — every ``Conv2D`` here already consumes a CHW-flat input via its
``H``/``W`` comptime params (channel-concat of two ``[C,H,W]`` maps with equal
``H,W`` is exactly flat vector concat). So ``GumbelGPUMCTS`` /
``GPUMCTSSequenceReplay`` / the ``MZ*GPU`` adapters stay untouched; only
``LATENT`` changes (128 → 2304).

This file currently provides the **representation** tower; the convolutional
dynamics (action-plane embed + residual-to-state) and the conv value/policy /
value-prefix-LSTM reward heads land alongside once the planner-integration design
(action encoding, scale-hidden) is pinned — see the parity doc.

Spatial-shape convention (matches ``Conv2D`` / the ResNet composites):
    OH = (H + 2*P - K) // S + 1

Note on bias: the reference convs are ``bias=False`` (every conv is BN-followed,
so the bias is redundant — BN re-centres). nn ``Conv2D`` carries a bias; for the
one conv whose BN is dropped (the no-BN downsample skip below) this is a benign
extra parameter, logged in the parity doc.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from mojo_rl.nn.core.module import Module, mptr
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.combinators.parallel import Parallel
from mojo_rl.nn.combinators.projected_residual import ProjectedResidual
from mojo_rl.nn.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.combinators.graph_nodes import InputSlot, Node
from mojo_rl.nn.primitives.conv2d import Conv2D
from mojo_rl.nn.primitives.batch_norm_2d import BatchNorm2D
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.primitives.avg_pool_2d import AvgPool2D
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.slice import Slice
from mojo_rl.nn.primitives.concat import Concat
from mojo_rl.nn.primitives.add import Add
from mojo_rl.nn.primitives.broadcast_tokens import BroadcastTokens
from mojo_rl.nn.primitives.layer_norm import LayerNorm
from mojo_rl.nn.models.resnet import ResBlockConv2DBN
from mojo_rl.deep_agents.dreamerv3.zero_init import (
    scale_output_module, scale_output_graph,
)


# ──────────────────────────────────────────────────────────────────────
# Downsample ResNet block — 3×3-stride-2 skip with NO BatchNorm
# ──────────────────────────────────────────────────────────────────────
#
# The EZv2 `DownSample.downsample_block` is a post-activated ResidualBlock with
# `downsample = conv3x3(IC→OC, stride=2)` (base_model.py:37, layer.py:11-35):
#
#   main:  Conv3×3-s2 → BN → ReLU → Conv3×3-s1 → BN
#   skip:  Conv3×3-s2                         (NO BN — differs from our
#                                              `ResBlockDownsampleBN`, whose skip
#                                              is 1×1-s2 + BN)
#   y   :  ReLU(main(x) + skip(x))
#
# Canonical K=3, P=1, S=2: both paths map H → (H-1)//2 + 1.
comptime EZDownBlockNoBN[
    IC: Int, OC: Int, H: Int, W: Int,
] = Sequential[
    ProjectedResidual[
        Sequential[
            Conv2D[IC, OC, 3, 2, 1, H, W],
            BatchNorm2D[OC, (H - 1) // 2 + 1, (W - 1) // 2 + 1],
            ReLU[OC * ((H - 1) // 2 + 1) * ((W - 1) // 2 + 1)],
            Conv2D[
                OC, OC, 3, 1, 1,
                (H - 1) // 2 + 1, (W - 1) // 2 + 1,
            ],
            BatchNorm2D[OC, (H - 1) // 2 + 1, (W - 1) // 2 + 1],
        ],
        # skip: bare 3×3-stride-2 conv, no BN (reference `downsample=conv2`).
        Conv2D[IC, OC, 3, 2, 1, H, W],
    ],
    ReLU[OC * ((H - 1) // 2 + 1) * ((W - 1) // 2 + 1)],
]


# ──────────────────────────────────────────────────────────────────────
# h (Atari) — representation: stacked RGB obs → spatial latent [64, 6, 6]
# ──────────────────────────────────────────────────────────────────────
#
# `DownSample(in=IN_CH, out=C)` (base_model.py:14-60) then the
# `RepresentationNetwork`'s `num_blocks` identity ResBlocks (base_model.py:86-99,
# num_blocks=1 for Atari). For Atari `IN_CH = n_stack·3 = 12` (RGB), input 96×96,
# `C = num_channels = 64`. No `state_norm` ⇒ no MinMaxNorm tail.
#
# Spatial collapse (K=3,P=1 throughout; s2 conv/pool halve, s1 preserve):
#   96 →[conv1 s2] 48 →[resblocks1 s1] 48 →[down s2] 24 →[resblocks2 s1] 24
#      →[pool1 s2] 12 →[resblocks3 s1] 12 →[pool2 s2] 6 →[rep resblock s1] 6
#   ⇒ [64, 6, 6] = 2304.
comptime EZRepNetResNetAtari[
    IN_CH: Int, C: Int,
] = Sequential[
    # ── DownSample ───────────────────────────────────────────────────
    # conv1: Conv(IN_CH→C/2, k3,s2,p1) → BN → ReLU      (96 → 48)
    Conv2D[IN_CH, C // 2, 3, 2, 1, 96, 96],
    BatchNorm2D[C // 2, 48, 48],
    ReLU[(C // 2) * 48 * 48],
    # resblocks1: 1× identity ResBlock(C/2) at 48×48
    ResBlockConv2DBN[C // 2, 3, 1, 48, 48],
    # downsample_block: ResBlock(C/2→C, s2) at 48 → 24 (no-BN skip)
    EZDownBlockNoBN[C // 2, C, 48, 48],
    # resblocks2: 1× identity ResBlock(C) at 24×24
    ResBlockConv2DBN[C, 3, 1, 24, 24],
    # pooling1: AvgPool(k3,s2,p1) 24 → 12
    AvgPool2D[C, 3, 2, 1, 24, 24],
    # resblocks3: 1× identity ResBlock(C) at 12×12
    ResBlockConv2DBN[C, 3, 1, 12, 12],
    # pooling2: AvgPool(k3,s2,p1) 12 → 6
    AvgPool2D[C, 3, 2, 1, 12, 12],
    # ── RepresentationNetwork: num_blocks=1 identity ResBlock(C) at 6×6 ──
    ResBlockConv2DBN[C, 3, 1, 6, 6],
]


# ──────────────────────────────────────────────────────────────────────
# Atari spatial-latent fixed geometry (num_channels=64, 6×6, reduced=16)
# ──────────────────────────────────────────────────────────────────────
comptime EZ_C = 64                     # num_channels
comptime EZ_HW = 6                     # spatial side of the latent
comptime EZ_LATENT = EZ_C * EZ_HW * EZ_HW          # 2304 = [64,6,6]
comptime EZ_REDC = 16                  # reduced_channels (heads + action embed)
comptime EZ_PLANE = EZ_HW * EZ_HW                   # 36  = [1,6,6]
comptime EZ_EMB = EZ_REDC * EZ_HW * EZ_HW          # 576 = [16,6,6]
comptime EZ_CAT = (EZ_C + EZ_REDC) * EZ_HW * EZ_HW # 2880 = [80,6,6]


# ──────────────────────────────────────────────────────────────────────
# Action → embedding plane.  onehot(ACT) → normalized scalar → [1,6,6]
# plane (broadcast) → Conv1×1(1→16) → LayerNorm([16,6,6]) → ReLU.
#
# Reference (`base_model.py:134-150`, action_embedding=True, dim=16) fills a
# single plane with `action_idx / action_space_size`, then conv1×1/LN/relu.
# Here the onehot→scalar map is a learnable `Linear[ACT,1]` (DEVIATION: the
# reference uses a FIXED `a/A` ramp — a learnable per-action scalar is a
# superset; the frozen-ramp init is a documented follow-up). The scalar is
# then broadcast to the 36-cell plane by `BroadcastTokens`.
# ──────────────────────────────────────────────────────────────────────
comptime EZActionPlane[ACT: Int] = Sequential[
    Linear[ACT, 1],
    BroadcastTokens[EZ_PLANE, 1],          # scalar → [1,6,6] = 36
]


# ──────────────────────────────────────────────────────────────────────
# g (Atari) — convolutional dynamics DAG (ComputeGraph).
#   input  [z(2304) | onehot(ACT)]   (the flat MZDyn adapter contract)
#   output [z'(2304) | reward_logits(BINS)]
#
# z → state [64,6,6]; action → [16,6,6] embed; channel-concat → [80,6,6] →
# Conv3×3(80→64)→BN → += z (residual to input state) → ReLU → 1×ResBlock(64)
# = z'. Reward branch off z': Conv1×1(64→16)→BN→ReLU→MLP[32]→BINS (a fused,
# stateless stand-in for the value-prefix LSTM — see docs/EZV2_ATARI_PARITY.md
# §D; honoring the LSTM needs a shared-planner (h,c)-pool rewrite).
# ──────────────────────────────────────────────────────────────────────
comptime EZDynAtariGraph[ACT: Int, BINS: Int] = ComputeGraph[
    EZ_LATENT + BINS,
    InputSlot["in", EZ_LATENT + ACT],
    # split the flat input into the latent state and the action onehot
    Node["z",    Slice[EZ_LATENT + ACT, 0, EZ_LATENT],            "in"],
    Node["aoh",  Slice[EZ_LATENT + ACT, EZ_LATENT, EZ_LATENT + ACT], "in"],
    # action → [1,6,6] plane → Conv1×1(1→16) → LN([16,6,6]) → ReLU
    Node["apl",  EZActionPlane[ACT],                              "aoh"],
    Node["aemb", Sequential[
                     Conv2D[1, EZ_REDC, 1, 1, 0, EZ_HW, EZ_HW],
                     LayerNorm[EZ_EMB],
                     ReLU[EZ_EMB],
                 ],                                               "apl"],
    # channel-concat state(64) ⊕ action-embed(16) = [80,6,6]
    Node["cat",  Concat[EZ_LATENT, EZ_EMB],                       "z", "aemb"],
    # Conv3×3(80→64) → BN  (P=1 keeps 6×6)
    Node["c1",   Sequential[
                     Conv2D[EZ_C + EZ_REDC, EZ_C, 3, 1, 1, EZ_HW, EZ_HW],
                     BatchNorm2D[EZ_C, EZ_HW, EZ_HW],
                 ],                                               "cat"],
    # residual to the INPUT state, then ReLU
    Node["res",  Add[EZ_LATENT, 2],                               "c1", "z"],
    Node["rl",   ReLU[EZ_LATENT],                                 "res"],
    # 1× post-activated ResBlock(64) → next state z'
    Node["zp",   ResBlockConv2DBN[EZ_C, 3, 1, EZ_HW, EZ_HW],      "rl"],
    # reward branch off z': Conv1×1(64→16)→BN→ReLU→MLP[32]→BINS
    Node["rew",  Sequential[
                     Conv2D[EZ_C, EZ_REDC, 1, 1, 0, EZ_HW, EZ_HW],
                     BatchNorm2D[EZ_REDC, EZ_HW, EZ_HW],
                     ReLU[EZ_EMB],
                     Linear[EZ_EMB, 32],
                     ReLU[32],
                     Linear[32, BINS],
                 ],                                               "zp"],
    # pack [z' | reward_logits]
    Node["out",  Concat[EZ_LATENT, BINS],                         "zp", "rew"],
]


# ──────────────────────────────────────────────────────────────────────
# GPU/CPU raw-pointer copy (vjp grad copy-back). N is comptime.
# ──────────────────────────────────────────────────────────────────────
def _ezdyn_copy_kernel[N: Int](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = rebind[Scalar[DT]](src[idx])


# ──────────────────────────────────────────────────────────────────────
# EZDynNetAtari — single-input Module wrapper over the dynamics graph.
#
# The `MZDynGPU` adapter calls `forward["gpu",B](in_t, output=out_t)` with one
# concatenated `[z|onehot]` tile — but `ComputeGraph` is driven via
# `set_input`/`forward(output)`. This thin wrapper bridges the two: forward
# feeds the slot then runs the graph; vjp runs the graph backward then copies
# the slot's accumulated input-gradient into `grad_inputs[0]` (the unroll's
# BPTT needs ∂/∂z). Param/state walks + the BatchNorm train/eval toggle
# delegate to the graph. Contract: IN_DIMS[0]=LATENT+ACT, OUT_DIM=LATENT+BINS.
# ──────────────────────────────────────────────────────────────────────
struct EZDynNetAtari[ACT: Int, BINS: Int](Module):
    comptime ARITY: Int = 1
    comptime LATENT = EZ_LATENT
    comptime IN_DIM = Self.LATENT + Self.ACT
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN_DIM)
    comptime OUT_DIM = Self.LATENT + Self.BINS
    comptime Graph = EZDynAtariGraph[Self.ACT, Self.BINS]

    var graph: Self.Graph
    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.Graph.OUT_DIM == Self.OUT_DIM, (
            "EZDynNetAtari: graph OUT_DIM must equal LATENT+BINS"
        )
        self.graph = Self.Graph()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "EZDynNetAtari: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        s.graph = Self.Graph.make[target, INIT](ctx=ctx)
        comptime if target == "cpu":
            s.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("EZDynNetAtari.make[target='gpu']: ctx required")
            s.ts = TargetStorage.make_gpu(ctx.value())
        return s^

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["EZDynNetAtari", target](self.ts.target_tag)
        self.graph.set_input["in", BATCH](
            inputs.tile[0, BATCH, Self.IN_DIM]()
        )
        self.graph.forward[target, BATCH, POLICY=POLICY](output)

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        assert_tag_for["EZDynNetAtari", target](self.ts.target_tag)
        self.graph.vjp[target, BATCH, POLICY=POLICY, mode=mode](grad_output)
        # copy slot["in"].grad_out → grad_inputs[0]  (∂/∂[z|action])
        comptime N = BATCH * Self.IN_DIM
        var dst = mptr(grad_inputs.tile[0, BATCH, Self.IN_DIM]().ptr)
        var src = self.graph.grad_input_ptr["in"]()
        comptime if target == "cpu":
            for i in range(N):
                dst[i] = src[i]
        else:
            var dst_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](dst)
            var src_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](src)
            comptime nb = (N + TPB - 1) // TPB
            self.ts.ctx.value().enqueue_function[_ezdyn_copy_kernel[N]](
                dst_lt, src_lt, grid_dim=nb, block_dim=TPB,
            )

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["EZDynNetAtari", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        self.graph.for_each_param[target, V](prefix + sep + "graph", visitor)

    def for_each_state[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["EZDynNetAtari", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        self.graph.for_each_state[target, V](prefix + sep + "graph", visitor)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        self.graph.set_attr[ATTR](value)


# ──────────────────────────────────────────────────────────────────────
# f (Atari) — convolutional prediction: z → [policy_logits | value_logits]
#
# `ValuePolicyNetwork` (base_model.py:166-213): a shared ResBlock(64), then two
# conv heads — Conv1×1(64→16)→BN→ReLU→flatten 576→MLP[32]→(ACT policy / BINS
# value categorical). `init_zero=True` zeros the last layer of each head (stable
# zero outputs at init) — a DOCUMENTED FOLLOW-UP here (needs per-layer zero
# init; the dreamerv3 `zero_init` visitor is the mechanism). Activation in the
# head MLP is ELU in the reference; nn has no ELU, so ReLU is substituted
# (logged). Output packing `[policy(ACT) | value(BINS)]` matches `MZPredNet`, so
# the planner's prediction adapter slices it unchanged.
# ──────────────────────────────────────────────────────────────────────
comptime EZPredNetAtari[ACT: Int, BINS: Int] = Sequential[
    # shared num_blocks=1 ResBlock(64)
    ResBlockConv2DBN[EZ_C, 3, 1, EZ_HW, EZ_HW],
    Parallel[
        # policy head: Conv1×1(64→16)→BN→ReLU→MLP[32]→ACT
        Sequential[
            Conv2D[EZ_C, EZ_REDC, 1, 1, 0, EZ_HW, EZ_HW],
            BatchNorm2D[EZ_REDC, EZ_HW, EZ_HW],
            ReLU[EZ_EMB],
            Linear[EZ_EMB, 32],
            ReLU[32],
            Linear[32, ACT],
        ],
        # value head: Conv1×1(64→16)→BN→ReLU→MLP[32]→BINS
        Sequential[
            Conv2D[EZ_C, EZ_REDC, 1, 1, 0, EZ_HW, EZ_HW],
            BatchNorm2D[EZ_REDC, EZ_HW, EZ_HW],
            ReLU[EZ_EMB],
            Linear[EZ_EMB, 32],
            ReLU[32],
            Linear[32, BINS],
        ],
    ],
]


# ──────────────────────────────────────────────────────────────────────
# init_zero (EZv2 `init_zero=True`) — scale each head's OUTPUT layer at init.
#
# base_model.py zeros the last Linear of the value, policy, and reward heads
# so the model starts with neutral value/reward predictions and a uniform
# policy prior (stable MCTS targets before the heads have learned). We reuse
# the DreamerV3 `scale_output_*` visitors (scale=0.0 == exact zero-init).
#
# Param names follow the combinator naming (Sequential→`.{i}`, Parallel→
# `.a`/`.b`, ComputeGraph→`.{node}`, Linear leaf→`.weight`/`.bias`):
#   • EZPredNetAtari = Sequential[ResBlock(0), Parallel(1)[policy=a, value=b]];
#     each head's output Linear is Sequential child 5 → `1.a.5.*` / `1.b.5.*`.
#   • EZDynAtariGraph reward branch is node `rew` (a 6-child Sequential); its
#     output Linear is child 5 → `rew.5.*`.
# Only the OUTPUT layer is scaled — scaling the whole head would choke the
# hidden layers' gradient (see zero_init.mojo).
# ──────────────────────────────────────────────────────────────────────
def ez_atari_init_zero_pred[
    target: StaticString, ACT: Int, BINS: Int
](
    mut pred: EZPredNetAtari[ACT, BINS],
    ctx: Optional[DeviceContext] = None,
    scale: Scalar[DT] = Scalar[DT](0.0),
) raises:
    """Zero (or `scale`) the policy + value head output Linears of the
    prediction net. scale=0.0 → uniform policy + neutral value at init."""
    scale_output_module[target, EZPredNetAtari[ACT, BINS]](
        pred, "1.a.5.weight", "1.a.5.bias", scale, ctx
    )
    scale_output_module[target, EZPredNetAtari[ACT, BINS]](
        pred, "1.b.5.weight", "1.b.5.bias", scale, ctx
    )


def ez_atari_init_zero_dyn[
    target: StaticString, ACT: Int, BINS: Int
](
    mut dyn: EZDynNetAtari[ACT, BINS],
    ctx: Optional[DeviceContext] = None,
    scale: Scalar[DT] = Scalar[DT](0.0),
) raises:
    """Zero (or `scale`) the reward head output Linear of the dynamics net
    (the fused value-prefix stand-in). scale=0.0 → neutral reward at init."""
    scale_output_graph[target](
        dyn.graph, "rew.5.weight", "rew.5.bias", scale, ctx
    )
