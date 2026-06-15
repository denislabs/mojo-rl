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
the flat `[z | onehot(a)]` input is split, the action is embedded into reduced
planes (`Linear[ACT,1] → broadcast → [1,H,W] → Conv1×1 → LayerNorm → ReLU`),
channel-concatenated with the state, a 3×3 conv + residual-to-state gives the
next latent (MinMaxNorm'd to match the rep's scaling), and a conv reward head
reads it. The wrapper bridges the graph's `set_input`/`forward(output)` to the
single-tile `Module` `forward`/`vjp` the `MZDynGPU` adapter + the unroll call.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from mojo_rl.nn2.core.module import Module, mptr
from mojo_rl.nn2.core.tensor_pack import TensorPack
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.combinators.parallel import Parallel
from mojo_rl.nn2.combinators.residual import Residual
from mojo_rl.nn2.combinators.compute_graph import ComputeGraph
from mojo_rl.nn2.combinators.graph_nodes import InputSlot, Node
from mojo_rl.nn2.primitives.conv2d import Conv2D
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.min_max_norm import MinMaxNorm
from mojo_rl.nn2.primitives.layer_norm import LayerNorm
from mojo_rl.nn2.primitives.slice import Slice
from mojo_rl.nn2.primitives.concat import Concat
from mojo_rl.nn2.primitives.add import Add
from mojo_rl.nn2.primitives.broadcast_tokens import BroadcastTokens


# ── BN-free identity-skip residual conv block (3×3, pad 1 → H×W preserved) ──
comptime C4ResBlock[C: Int, H: Int, W: Int] = Residual[
    Sequential[
        Conv2D[C, C, 3, 1, 1, H, W],
        ReLU[C * H * W],
        Conv2D[C, C, 3, 1, 1, H, W],
    ]
]


# ──────────────────────────────────────────────────────────────────────
# h — representation: 3×H×W board planes → spatial latent [C,H,W]
# ──────────────────────────────────────────────────────────────────────
comptime MZRepNetC4Spatial[C: Int, H: Int, W: Int] = Sequential[
    Conv2D[3, C, 3, 1, 1, H, W],
    ReLU[C * H * W],
    C4ResBlock[C, H, W],
    ReLU[C * H * W],
    C4ResBlock[C, H, W],
    ReLU[C * H * W],
    # MinMaxNorm scales the flat latent to [0,1] (idempotent under the
    # planner's scale-hidden kernel; stays inside the autodiff graph so
    # training gets the scaling gradient — same tail every MZRepNet ends in).
    MinMaxNorm[C * H * W],
]


# ── action → embedding planes: onehot(ACT) → learnable scalar → [1,H,W] ──
comptime C4ActionPlane[ACT: Int, H: Int, W: Int] = Sequential[
    Linear[ACT, 1],
    BroadcastTokens[H * W, 1],
]


# ──────────────────────────────────────────────────────────────────────
# g — convolutional dynamics DAG (ComputeGraph).
#   input  [z(C·H·W) | onehot(ACT)]   output [z'(C·H·W) | reward_logits(BINS)]
# z → state[C,H,W]; action → [REDC,H,W] embed; channel-concat → Conv3×3(→C) +=
# residual-to-z → ReLU → 1×ResBlock = z' (MinMaxNorm'd). Reward branch off z':
# Conv1×1(C→REDC)→ReLU→MLP[32]→BINS. All BatchNorm-free (LayerNorm on the action
# embed, MinMaxNorm on the next latent).
# ──────────────────────────────────────────────────────────────────────
comptime MZDynC4SpatialGraph[
    C: Int, ACT: Int, BINS: Int, H: Int, W: Int, REDC: Int,
] = ComputeGraph[
    C * H * W + BINS,
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
    Node["res", Add[C * H * W, 2], "c1", "z"],
    Node["rl", ReLU[C * H * W], "res"],
    Node["zpre", C4ResBlock[C, H, W], "rl"],
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


# GPU/CPU raw-pointer copy for the vjp input-grad copy-back.
def _mzc4dyn_copy_kernel[N: Int](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = rebind[Scalar[DT]](src[idx])


# ──────────────────────────────────────────────────────────────────────
# MZDynNetC4Spatial — single-input Module wrapper over the dynamics graph.
# Mirrors `EZDynNetAtari`: forward feeds the slot then runs the graph; vjp runs
# the graph backward then copies the slot's accumulated input-grad into
# grad_inputs[0] (the unroll's BPTT needs ∂/∂[z|action]). Param/state walks +
# the train/eval toggle delegate to the graph.
# Contract: IN_DIMS[0]=LATENT+ACT, OUT_DIM=LATENT+BINS.
# ──────────────────────────────────────────────────────────────────────
struct MZDynNetC4Spatial[
    C: Int, ACT: Int, BINS: Int, H: Int, W: Int, REDC: Int = 16,
](Module):
    comptime ARITY: Int = 1
    comptime LATENT = Self.C * Self.H * Self.W
    comptime IN_DIM = Self.LATENT + Self.ACT
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN_DIM)
    comptime OUT_DIM = Self.LATENT + Self.BINS
    comptime Graph = MZDynC4SpatialGraph[
        Self.C, Self.ACT, Self.BINS, Self.H, Self.W, Self.REDC
    ]

    var graph: Self.Graph
    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.Graph.OUT_DIM == Self.OUT_DIM, (
            "MZDynNetC4Spatial: graph OUT_DIM must equal LATENT+BINS"
        )
        self.graph = Self.Graph()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "MZDynNetC4Spatial: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        s.graph = Self.Graph.make[target, INIT](ctx=ctx)
        comptime if target == "cpu":
            s.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("MZDynNetC4Spatial.make[gpu]: ctx required")
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
        assert_tag_for["MZDynNetC4Spatial", target](self.ts.target_tag)
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
        assert_tag_for["MZDynNetC4Spatial", target](self.ts.target_tag)
        self.graph.vjp[target, BATCH, POLICY=POLICY, mode=mode](grad_output)
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
            self.ts.ctx.value().enqueue_function[_mzc4dyn_copy_kernel[N]](
                dst_lt, src_lt, grid_dim=nb, block_dim=TPB,
            )

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["MZDynNetC4Spatial", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        self.graph.for_each_param[target, V](prefix + sep + "graph", visitor)

    def for_each_state[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["MZDynNetC4Spatial", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        self.graph.for_each_state[target, V](prefix + sep + "graph", visitor)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        self.graph.set_attr[ATTR](value)


# ──────────────────────────────────────────────────────────────────────
# f — convolutional prediction: spatial latent → [policy_logits | value_logits]
# Shared ResBlock(C), then two conv heads: Conv1×1(C→REDC)→ReLU→flatten→MLP[FC]
# →(ACT policy / BINS value categorical). Output packing [policy | value] matches
# `MZPredNet`, so the prediction adapter slices it unchanged.
# ──────────────────────────────────────────────────────────────────────
comptime MZPredNetC4Spatial[
    C: Int, ACT: Int, BINS: Int, H: Int, W: Int, REDC: Int = 16, FC: Int = 64,
] = Sequential[
    C4ResBlock[C, H, W],
    ReLU[C * H * W],
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
