"""ResNet-18 (ImageNet variant) feature stack — the ACT / DETR image backbone.

`models/resnet.mojo` has the CIFAR ResNet-20 building blocks; this is the
torchvision `resnet18` topology those blocks compose into. The two differ in
more than depth: the ImageNet stem is a **7x7 stride-2** convolution followed by
a **3x3 stride-2 max-pool** (a 4x reduction before any residual block), where
the CIFAR stem is a single 3x3 stride-1 conv. Feeding 480x640 camera frames
through a CIFAR stem would be a different model, not a smaller one.

Head-less by construction — no avg-pool, no flatten, no classifier. ACT takes
the `layer4` feature map and projects it to the transformer width with a 1x1
convolution (`detr_vae.py:56 self.input_proj`), so a classifier would be dead
weight. This matches `Backbone`'s `return_layers = {'layer4': "0"}`
(`backbone.py:66`).

## Shapes

Total stride 32:

    conv1  7x7/2 p3   H     -> H/2
    maxpool 3x3/2 p1  H/2   -> H/4
    layer1 2x basic   H/4      (stride 1)
    layer2 2x basic   H/4   -> H/8
    layer3 2x basic   H/8   -> H/16
    layer4 2x basic   H/16  -> H/32

    480x640 -> 15x20 = 300 tokens per camera   (the paper's rig)
    240x320 ->  8x10 =  80 tokens per camera   (the CPU-gate default)

Output is `[BATCH, 512 * OH * OW]` in NCHW.

## ⚠ BatchNorm, not FrozenBatchNorm2d

`backbone.py:88` passes `norm_layer=FrozenBatchNorm2d` — batch statistics and
affine parameters held fixed at their ImageNet values. That is the right choice
for a PRETRAINED backbone fine-tuned on a small batch, and meaningless for a
randomly-initialized one: there are no pretrained statistics to freeze. This
uses ordinary trainable `BatchNorm2D`. A deliberate, labelled deviation —
revisit it together with any pretrained-weight loading, never separately.
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT, LAYOUT_NCHW
from ..core.amp import AMPPolicy, NoAMP
from ..core.initializer import Initializer
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..primitives.max_pool_2d import MaxPool2D
from ..combinators.sequential import Sequential
from .conv import Conv2DBatchNormReLU
from .resnet import ResBlockConv2DBN, ResBlockDownsampleBN


# Spatial reductions, spelled out so each stage's H/W arguments are readable
# rather than a nest of `(H + 2*P - K) // S + 1`.
comptime _S2[X: Int] = (X + 2 * 3 - 7) // 2 + 1  # conv1: 7x7 stride 2 pad 3
comptime _MP[X: Int] = (X + 2 * 1 - 3) // 2 + 1  # maxpool: 3x3 stride 2 pad 1
comptime _D[X: Int] = (X + 2 * 1 - 3) // 2 + 1  # basic block, stride 2 pad 1


comptime ResNet18Stem[
    IN_CH: Int, H: Int, W: Int, LAYOUT: Int = LAYOUT_NCHW
] = Sequential[
    Conv2DBatchNormReLU[IN_CH, 64, 7, 2, 3, H, W, LAYOUT=LAYOUT],
    MaxPool2D[64, 3, 2, 1, _S2[H], _S2[W], LAYOUT=LAYOUT],
]


comptime ResNet18Seq[
    IN_CH: Int, H: Int, W: Int, LAYOUT: Int = LAYOUT_NCHW
] = Sequential[
    ResNet18Stem[IN_CH, H, W, LAYOUT],
    # layer1 — 64 channels, stride 1 (the max-pool already did the reduction)
    ResBlockConv2DBN[64, 3, 1, _MP[_S2[H]], _MP[_S2[W]], LAYOUT=LAYOUT],
    ResBlockConv2DBN[64, 3, 1, _MP[_S2[H]], _MP[_S2[W]], LAYOUT=LAYOUT],
    # layer2 — 64 -> 128, stride 2
    ResBlockDownsampleBN[
        64, 128, 3, 1, _MP[_S2[H]], _MP[_S2[W]], LAYOUT=LAYOUT
    ],
    ResBlockConv2DBN[
        128, 3, 1, _D[_MP[_S2[H]]], _D[_MP[_S2[W]]], LAYOUT=LAYOUT
    ],
    # layer3 — 128 -> 256, stride 2
    ResBlockDownsampleBN[
        128, 256, 3, 1, _D[_MP[_S2[H]]], _D[_MP[_S2[W]]], LAYOUT=LAYOUT
    ],
    ResBlockConv2DBN[
        256, 3, 1, _D[_D[_MP[_S2[H]]]], _D[_D[_MP[_S2[W]]]], LAYOUT=LAYOUT
    ],
    # layer4 — 256 -> 512, stride 2
    ResBlockDownsampleBN[
        256,
        512,
        3,
        1,
        _D[_D[_MP[_S2[H]]]],
        _D[_D[_MP[_S2[W]]]],
        LAYOUT=LAYOUT,
    ],
    ResBlockConv2DBN[
        512,
        3,
        1,
        _D[_D[_D[_MP[_S2[H]]]]],
        _D[_D[_D[_MP[_S2[W]]]]],
        LAYOUT=LAYOUT,
    ],
]


# Feature-map geometry, for callers that need the token count.
comptime ResNet18OutH[H: Int] = _D[_D[_D[_MP[_S2[H]]]]]
comptime ResNet18OutW[W: Int] = _D[_D[_D[_MP[_S2[W]]]]]
comptime RESNET18_OUT_CH: Int = 512


# ══════════════════════════════════════════════════════════════════════════
# ResNet18Backbone — the same stack, behind a NAMED struct
# ══════════════════════════════════════════════════════════════════════════
"""⚠ A struct, not a `comptime` alias, and that is the whole point.

A parametric alias is pure substitution: every type that mentions
`ResNet18Seq[3, 240, 320]` carries its FULL expansion — 20 `Conv2D`, 20
`BatchNorm2D`, 22 `Sequential`, each with its own spelled-out spatial
parameters. `ComputeGraph[*DECLS]` mangles its whole decl list into
`__init__`'s symbol, so embedding the alias there produced a **4.5 MB mangled
name** — 70x over Apple's linker limit, which is why these builds needed
`-Xlinker -ld_classic`.

Holding the `Sequential` as an INTERNAL `comptime` member instead means the
enclosing type sees only `ResNet18Backbone,IN_CH=3,H=240,W=320`. The expansion
happens once, inside this struct, rather than at every use site. Same
mechanism `primitives/decoder_block.mojo` uses for its `ComputeGraph`.

Behaviour is unchanged — every method delegates to the same `Sequential`.
"""


struct ResNet18Backbone[
    IN_CH: Int, H: Int, W: Int, LAYOUT: Int = LAYOUT_NCHW
](Module):
    comptime Net = ResNet18Seq[Self.IN_CH, Self.H, Self.W, Self.LAYOUT]
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](
        fill=Self.IN_CH * Self.H * Self.W
    )
    comptime OUT_DIM: Int = Self.Net.OUT_DIM

    var net: Self.Net

    def __init__(out self):
        self.net = Self.Net()

    def __init__(out self, *, deinit move: Self):
        self.net = move.net^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var b = Self()
        b.net = Self.Net.make[target, INIT](ctx)
        return b^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.net.forward[target, B, POLICY=POLICY](inputs, out, ctx)

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.net.vjp[target, B, POLICY=POLICY](
            forward_input, grad_output, grad_inputs, ctx
        )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.net.for_each_param[target](visitor, ctx, prefix)

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.net.for_each_state[target](visitor, ctx, prefix)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.net.zero_grad[target](ctx)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        self.net.set_attr[ATTR](value)

    def polyak_from[
        target: StaticString
    ](mut self, mut src: Self, tau: Scalar[DT],
      ctx: Optional[DeviceContext]) raises:
        self.net.polyak_from[target](src.net, tau, ctx)
