"""Spatial-latent MuZero nets for Connect Four — conv h / g / f.

The flat-latent `MZRepNet`/`MZDynNet`/`MZPredNet` hit a value-fit ceiling on C4:
the latent is a flat vector and the dynamics is an MLP, so it cannot model the
board's tactics (piece-drop + threat propagation), the n-step value targets stay
noisy, and `value_mse` plateaus regardless of width. This module keeps the latent
**spatial** — a `[C, H, W]` feature map (flat-encoded as `LATENT = C·H·W`, the
planner is agnostic to the internal shape) — so every h/g/f operates with conv
weight-sharing over the 6×7 grid. This is the EZv2/AlphaZero "spatial model"
applied to MuZero, mirroring `efficient_zero_v2/nets_atari.mojo` but for the 6×7
board and **BatchNorm-free** (LayerNorm / MinMaxNorm only) so the arena's
params-only `hard_copy_params` promotion stays correct.

Contracts (so the planner adapters + the 2p arena driver are unchanged):
  * `MZRepNetC4Spatial[C, H, W]`        — IN = 3·H·W (= OBS 126), OUT = C·H·W (= LATENT)
  * `MZDynNetC4Spatial[C, ACT, BINS, H, W]` — IN = LATENT+ACT, OUT = LATENT+BINS
  * `MZPredNetC4Spatial[C, ACT, BINS, H, W]` — IN = LATENT, OUT = ACT+BINS

`LATENT = C·H·W` (e.g. C=32 on 6×7 → 1344). The dynamics is a `ComputeGraph`:
the flat `[z | onehot(a)]` input is split, the action is encoded as a
**column-marked** `[1,H,W]` plane (the played column lit across all rows — the
standard board action plane; requires ACT == COLS) then `Conv1×1 → LayerNorm →
ReLU` to reduced planes, channel-concatenated with the state, a 3×3 conv +
residual-to-state gives the next latent (MinMaxNorm'd to match the rep's
scaling), and a conv reward head reads it. The wrapper bridges the graph's
`set_input`/`forward(out)` to the single-tile `Module` `forward`/`vjp` the
`MZDynGPU` adapter + the unroll call expect.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.storage.core.initializer import Initializer
from mojo_rl.nn.storage.core.module import Module
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.param import ParamVisitor
from mojo_rl.nn.storage.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.storage.core.walkers import join_name
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.combinators.parallel import Parallel
from mojo_rl.nn.storage.combinators.residual import Residual
from mojo_rl.nn.storage.combinators.repeat import Repeat
from mojo_rl.nn.storage.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.storage.combinators.graph_decl import InputSlot, Node
from mojo_rl.nn.storage.primitives.conv2d import Conv2D
from mojo_rl.nn.storage.primitives.activations import ReLU
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.min_max_norm import MinMaxNorm
from mojo_rl.nn.storage.primitives.layer_norm import LayerNorm
from mojo_rl.nn.storage.primitives.slice import Slice
from mojo_rl.nn.storage.primitives.concat import Concat
from mojo_rl.nn.storage.primitives.add import Add
from mojo_rl.nn.storage.primitives.broadcast_tokens import BroadcastTokens
from mojo_rl.deep_agents.dreamerv3.zero_init import (
    scale_output_module, scale_output_graph,
)


# ── BN-free identity-skip residual conv block (3×3, pad 1 → H×W preserved) ──
comptime C4ResBlock[C: Int, H: Int, W: Int] = Residual[
    Sequential[
        Conv2D[C, C, 3, 1, 1, H, W],
        ReLU[C * H * W],
        Conv2D[C, C, 3, 1, 1, H, W],
    ]
]


# ── post-activation residual TOWER: `NB` independent [ResBlock, ReLU] units
#    chained (`Repeat`, own weights per copy). One unit = a 3×3 residual block
#    whose summed output is ReLU'd before the next block. This is the single
#    depth knob shared by h / g / f — the muzero-general `blocks` parameter.
#    NB=1 is one block + trailing ReLU; deeper stacks add tactical capacity for
#    the value-fit ceiling. Repeat names children `.{i}`, so the tower's params
#    live under one extra index vs. the old hand-unrolled blocks.
comptime C4ResTower[NB: Int, C: Int, H: Int, W: Int] = Repeat[
    NB, Sequential[C4ResBlock[C, H, W], ReLU[C * H * W]]
]


# ──────────────────────────────────────────────────────────────────────
# h — representation: 3×H×W board planes → spatial latent [C,H,W]
# ──────────────────────────────────────────────────────────────────────
comptime MZRepNetC4Spatial[C: Int, H: Int, W: Int, NB: Int = 2] = Sequential[
    Conv2D[3, C, 3, 1, 1, H, W],
    ReLU[C * H * W],
    # NB residual blocks (each followed by ReLU). Default 2 reproduces the
    # original depth; bump NB for more capacity (muzero-general `blocks`).
    C4ResTower[NB, C, H, W],
    # MinMaxNorm scales the flat latent to [0,1] (idempotent under the
    # planner's scale-hidden kernel; stays inside the autodiff graph so
    # training gets the scaling gradient — same tail every MZRepNet ends in).
    MinMaxNorm[C * H * W],
]


# ── action → COLUMN-MARKED plane: onehot(ACT=COLS) → [1,H,W] where the played
#    column is 1 across all rows. Requires ACT == W (= COLS), which holds for
#    Connect Four (7 columns = 7 actions). `BroadcastTokens[H, ACT]` computes
#    out[r*ACT + c] = onehot[c] = out[r*W + c] — i.e. the played column lit down
#    every row, in the latent's exact row-major [H,W] layout.
#
#    This is the standard AlphaZero/MuZero board action encoding: it tells the
#    conv dynamics *which column* the move is in SPATIALLY, so it can apply the
#    move at the right location. The conv1×1 that follows learns the embedding
#    from this marking.
comptime C4ActionPlane[ACT: Int, H: Int, W: Int] = BroadcastTokens[H, ACT]


# ──────────────────────────────────────────────────────────────────────
# g — convolutional dynamics DAG (ComputeGraph).
#   input  [z(C·H·W) | onehot(ACT)]   output [z'(C·H·W) | reward_logits(BINS)]
# z → state[C,H,W]; action → column-marked [1,H,W] → Conv1×1 → [REDC,H,W] embed;
# channel-concat → Conv3×3(→C) += residual-to-z → ReLU → NB×ResBlock = z'
# (MinMaxNorm'd). Reward branch off z': Conv1×1(C→REDC)→ReLU→MLP[32]→BINS. All
# BatchNorm-free (LayerNorm on the action embed, MinMaxNorm on the next latent).
# Storage ComputeGraph: OUT_DIM is inferred from the last node ("out"), so there
# is no leading output-count param (vs. the legacy `ComputeGraph[OUT, ...]`).
# ──────────────────────────────────────────────────────────────────────
comptime MZDynC4SpatialGraph[
    C: Int, ACT: Int, BINS: Int, H: Int, W: Int, REDC: Int, NB: Int,
] = ComputeGraph[
    InputSlot["in", C * H * W + ACT],
    Node["z", Slice[C * H * W + ACT, 0, C * H * W], "in"],
    Node["aoh", Slice[C * H * W + ACT, C * H * W, C * H * W + ACT], "in"],
    Node["apl", C4ActionPlane[ACT, H, W], "aoh"],
    Node[
        "aemb",
        Sequential[
            Conv2D[1, REDC, 1, 1, 0, H, W],
            LayerNorm[REDC * H * W],
            ReLU[REDC * H * W],
        ],
        "apl",
    ],
    Node["cat", Concat[C * H * W, REDC * H * W], "z", "aemb"],
    Node["c1", Conv2D[C + REDC, C, 3, 1, 1, H, W], "cat"],
    Node["res", Add[C * H * W], "c1", "z"],
    Node["rl", ReLU[C * H * W], "res"],
    Node["zpre", C4ResTower[NB, C, H, W], "rl"],
    Node["zp", MinMaxNorm[C * H * W], "zpre"],
    Node[
        "rew",
        Sequential[
            Conv2D[C, REDC, 1, 1, 0, H, W],
            ReLU[REDC * H * W],
            Linear[REDC * H * W, 32],
            ReLU[32],
            Linear[32, BINS],
        ],
        "zp",
    ],
    Node["out", Concat[C * H * W, BINS], "zp", "rew"],
]


# GPU LayoutTensor copy for the vjp input-grad copy-back (dst, src both real
# graph-pool / grad-pool buffers).
def _mzc4dyn_copy_kernel[N: Int](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = rebind[Scalar[DT]](src[idx])


# ──────────────────────────────────────────────────────────────────────
# MZDynNetC4Spatial — single-input Module wrapper over the dynamics graph.
# forward feeds the slot then runs the graph; vjp runs the graph backward then
# copies the slot's accumulated input-grad into grad_inputs[0] (the unroll's
# BPTT needs ∂/∂[z|action]). Param/state walks delegate to the graph. Storage:
# no TargetStorage (ctx is arg-threaded); BN-free so set_attr is the trait no-op.
# Contract: IN_DIMS[0]=LATENT+ACT, OUT_DIM=LATENT+BINS.
# ──────────────────────────────────────────────────────────────────────
struct MZDynNetC4Spatial[
    C: Int, ACT: Int, BINS: Int, H: Int, W: Int, NB: Int = 1, REDC: Int = 16,
](Module):
    comptime ARITY: Int = 1
    comptime LATENT = Self.C * Self.H * Self.W
    comptime IN_DIM = Self.LATENT + Self.ACT
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN_DIM)
    comptime OUT_DIM = Self.LATENT + Self.BINS
    comptime Graph = MZDynC4SpatialGraph[
        Self.C, Self.ACT, Self.BINS, Self.H, Self.W, Self.REDC, Self.NB
    ]

    var graph: Self.Graph

    def __init__(out self):
        comptime assert Self.Graph.OUT_DIM == Self.OUT_DIM, (
            "MZDynNetC4Spatial: graph OUT_DIM must equal LATENT+BINS"
        )
        comptime assert Self.ACT == Self.W, (
            "MZDynNetC4Spatial: the column-marked action plane requires"
            " ACT == W (actions == board columns), e.g. Connect Four 7==7"
        )
        self.graph = Self.Graph()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "MZDynNetC4Spatial: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        s.graph = Self.Graph.make[target, INIT](ctx)
        return s^

    def forward[
        target: StaticString,
        B: Int,
        o: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.graph.set_input["in", B](inputs[0], ctx)
        self.graph.forward[B, target, POLICY=POLICY](out, ctx)

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.graph.vjp[B, target, POLICY=POLICY](grad_output, ctx)
        comptime N = B * Self.IN_DIM
        comptime if target == "cpu":
            ref gin = self.graph.grad_input["in"]()
            ref dst = grad_inputs[0]
            for i in range(N):
                dst.data[i] = gin.data[i]
        else:
            var c = ctx.value()
            comptime nb = (N + TPB - 1) // TPB
            c.enqueue_function[_mzc4dyn_copy_kernel[N]](
                grad_inputs[0].lt["gpu", Layout.row_major(N)](),
                self.graph.grad_input["in"]().lt["gpu", Layout.row_major(N)](),
                grid_dim=nb, block_dim=TPB,
            )

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.graph.for_each_param[target, V](
            visitor, ctx, join_name(prefix, String("graph"))
        )

    def for_each_state[
        target: StaticString,
        V: ParamVisitor,
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.graph.for_each_state[target, V](
            visitor, ctx, join_name(prefix, String("graph"))
        )


# ──────────────────────────────────────────────────────────────────────
# f — convolutional prediction: spatial latent → [policy_logits | value_logits]
# Shared ResBlock(C), then two conv heads: Conv1×1(C→REDC)→ReLU→flatten→MLP[FC]
# →(ACT policy / BINS value categorical). Output packing [policy | value] matches
# `MZPredNet`, so the prediction adapter slices it unchanged.
# ──────────────────────────────────────────────────────────────────────
comptime MZPredNetC4Spatial[
    C: Int, ACT: Int, BINS: Int, H: Int, W: Int,
    NB: Int = 1, REDC: Int = 16, FC: Int = 64,
] = Sequential[
    # Shared NB-block residual torso, then the two conv heads. The tower is one
    # Sequential child (index 0), so the Parallel heads sit at index 1.
    C4ResTower[NB, C, H, W],
    Parallel[
        Sequential[
            Conv2D[C, REDC, 1, 1, 0, H, W],
            ReLU[REDC * H * W],
            Linear[REDC * H * W, FC],
            ReLU[FC],
            Linear[FC, ACT],
        ],
        Sequential[
            Conv2D[C, REDC, 1, 1, 0, H, W],
            ReLU[REDC * H * W],
            Linear[REDC * H * W, FC],
            ReLU[FC],
            Linear[FC, BINS],
        ],
    ],
]


# ──────────────────────────────────────────────────────────────────────
# init_zero (EZv2 `init_zero=True`) — scale each head's OUTPUT Linear at init.
#
# Zeros the last Linear of the policy + value heads (prediction) and the reward
# head (dynamics), so the model starts with a uniform policy prior and neutral
# value/reward predictions (stable MCTS targets before the heads have learned,
# and more early self-play exploration from the uniform prior). `scale=0.0` is
# exact zero-init; a small scale keeps some Kaiming asymmetry.
#
# Param names follow the STORAGE combinator naming (Sequential→`.{i}`, Parallel→
# `.{i}` by branch INDEX [NOT `.a`/`.b` as in legacy nn], Repeat→`.{i}`,
# ComputeGraph node→`.{node}`, Linear leaf→`.weight`/`.bias`). These torsos are
# BN-FREE, so each head's output Linear is one index earlier than EZv2's BN heads:
#   • MZPredNetC4Spatial = Sequential[C4ResTower(0), Parallel(1)[0, 1]] — the
#     shared torso is ONE tower child, so the Parallel sits at index 1 (NB
#     does not move it). Each head is Sequential[Conv(0), ReLU(1), Linear(2),
#     ReLU(3), Linear(4)], so the output Linears are `1.0.4.*` (policy) /
#     `1.1.4.*` (value).
#   • MZDynC4SpatialGraph reward branch is node `rew` (a 5-child Sequential); its
#     output Linear is child 4 → `rew.4.*` (independent of NB / the zpre tower).
# Only the OUTPUT layer is scaled — scaling the whole head chokes the hidden
# layers' gradient (see zero_init.mojo).
# ──────────────────────────────────────────────────────────────────────
def mzc4_init_zero_pred[
    target: StaticString,
    C: Int, ACT: Int, BINS: Int, H: Int, W: Int,
    NB: Int = 1, REDC: Int = 16, FC: Int = 64,
](
    mut pred: MZPredNetC4Spatial[C, ACT, BINS, H, W, NB, REDC, FC],
    ctx: Optional[DeviceContext] = None,
    scale: Scalar[DT] = Scalar[DT](0.0),
) raises:
    """Zero (or `scale`) the policy + value head output Linears → uniform policy
    + neutral value at init."""
    scale_output_module[
        target, MZPredNetC4Spatial[C, ACT, BINS, H, W, NB, REDC, FC]
    ](pred, "1.0.4.weight", "1.0.4.bias", scale, ctx)
    scale_output_module[
        target, MZPredNetC4Spatial[C, ACT, BINS, H, W, NB, REDC, FC]
    ](pred, "1.1.4.weight", "1.1.4.bias", scale, ctx)


def mzc4_init_zero_dyn[
    target: StaticString,
    C: Int, ACT: Int, BINS: Int, H: Int, W: Int, NB: Int = 1, REDC: Int = 16,
](
    mut dyn: MZDynNetC4Spatial[C, ACT, BINS, H, W, NB, REDC],
    ctx: Optional[DeviceContext] = None,
    scale: Scalar[DT] = Scalar[DT](0.0),
) raises:
    """Zero (or `scale`) the reward head output Linear → neutral reward at init.
    """
    scale_output_graph[target](
        dyn.graph, "rew.4.weight", "rew.4.bias", scale, ctx
    )
