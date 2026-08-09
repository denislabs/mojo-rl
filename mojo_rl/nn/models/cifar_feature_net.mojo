"""CIFAR ResNet-20 feature backbone (head-less) for perceptual losses.

`CifarBackbone[H, W]` is the ResNet-20 (CIFAR variant) convolutional feature
stack WITHOUT the classifier head (no avg-pool / flatten / linear) — exactly the
layers of `examples/nn/resnet/resnet20_cifar10_training_storage_gpu.mojo` up to
the final 64-channel stage. It is a plain `Sequential`, so forward / vjp / param
& state walkers come for free.

Spatial dims are parameterized by the input `H, W` (require `H % 4 == 0`,
`W % 4 == 0`): the two stride-2 downsamples take `H → H/2 → H/4`. The
convolution / BatchNorm parameter SIZES are independent of `H, W` (conv weights
are spatially shared; BN params are per-channel), so a checkpoint trained at
32×32 loads into a `CifarBackbone[64, 64]` instance unchanged — only the feature
map resolution differs. Used as a FROZEN (BN-eval) feature extractor:
`backbone.set_attr["training"](Scalar[DT](0.0))`.

Output: `64 * (H // 4) * (W // 4)` features per sample (NCHW, 64 × H/4 × W/4).
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.initializer import Initializer
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.walkers import join_name
from mojo_rl.nn.models.conv import Conv2DBatchNormReLU
from mojo_rl.nn.models.resnet import ResBlockConv2DBN, ResBlockDownsampleBN
from mojo_rl.nn.primitives.avg_pool_2d import AvgPool2D
from mojo_rl.nn.primitives.flatten import Flatten
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.combinators.repeat import Repeat


# ResNet-20 (6n+2, n=3) feature stack: 3×3 BN stem (3→16) then 3 stages of 3
# residual blocks (16 → 32 → 64, stages 2–3 downsampling). Head-less.
comptime CifarBackbone[H: Int, W: Int] = Sequential[
    Conv2DBatchNormReLU[3, 16, 3, 1, 1, H, W],                 # stem → 16×H×W
    # Stage 1: 3 identity blocks @ 16ch, H×W
    Repeat[3, ResBlockConv2DBN[16, 3, 1, H, W], shared=False],
    # Stage 2: downsample 16→32 (H×W → H/2×W/2) + 2 identity blocks
    ResBlockDownsampleBN[16, 32, 3, 1, H, W],
    Repeat[2, ResBlockConv2DBN[32, 3, 1, H // 2, W // 2], shared=False],
    # Stage 3: downsample 32→64 (H/2×W/2 → H/4×W/4) + 2 identity blocks
    ResBlockDownsampleBN[32, 64, 3, 1, H // 2, W // 2],
    Repeat[2, ResBlockConv2DBN[64, 3, 1, H // 4, W // 4], shared=False],
]


# ── Trainable classifier: backbone + global-avg-pool head ──────────────────
# A bespoke 2-field Module so the backbone is a NAMED field — after training the
# CIFAR classifier we save JUST `classifier.backbone` (a clean backbone-only
# checkpoint) via `save_params(trainer.model.backbone, path)`, which then loads
# straight into a frozen `CifarBackbone` for the perceptual loss. (A monolithic
# `Sequential[backbone, head]` would force the checkpoint to include head params,
# and the sequential checkpoint reader can't skip a tail of unwanted params.)
struct CifarFeatureClassifier[NC: Int, H: Int = 32, W: Int = 32](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=3 * Self.H * Self.W)
    comptime OUT_DIM = Self.NC
    comptime FEAT = 64 * (Self.H // 4) * (Self.W // 4)

    comptime BACKBONE = CifarBackbone[Self.H, Self.W]
    comptime HEAD = Sequential[
        AvgPool2D[64, Self.H // 4, Self.W // 4, 0, Self.H // 4, Self.W // 4],
        Flatten[64],
        Linear[64, Self.NC],
    ]

    var backbone: Self.BACKBONE
    var head: Self.HEAD
    var feat: Tensor       # scratch [BATCH*FEAT]; set in forward, reused in vjp
    var grad_feat: Tensor

    def __init__(out self):
        self.backbone = Self.BACKBONE()
        self.head = Self.HEAD()
        self.feat = Tensor()
        self.grad_feat = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "CifarFeatureClassifier: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.backbone = Self.BACKBONE.make[target=target, INIT=INIT](ctx)
        m.head = Self.HEAD.make[target=target, INIT=INIT](ctx)
        return m^

    @staticmethod
    def display_label() -> String:
        return String("CifarFeatureClassifier")

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            self.feat.ensure(B * Self.FEAT)
        else:
            self.feat.ensure_gpu(ctx.value(), B * Self.FEAT)
        self.backbone.forward[target, B, POLICY=POLICY](inputs, self.feat, ctx)
        self.head.forward[target, B, POLICY=POLICY](
            TensorRefs[Self.ARITY](self.feat), out, ctx
        )

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
        comptime if target == "cpu":
            self.grad_feat.ensure(B * Self.FEAT)
        else:
            self.grad_feat.ensure_gpu(ctx.value(), B * Self.FEAT)
        self.head.vjp[target, B, POLICY=POLICY](
            TensorRefs[Self.ARITY](self.feat),
            grad_output,
            TensorRefs[Self.ARITY](self.grad_feat),
            ctx,
        )
        self.backbone.vjp[target, B, POLICY=POLICY](
            forward_input, self.grad_feat, grad_inputs, ctx
        )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.backbone.for_each_param[target, V](
            visitor, ctx, join_name(prefix, "backbone")
        )
        self.head.for_each_param[target, V](
            visitor, ctx, join_name(prefix, "head")
        )

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.backbone.for_each_state[target, V](
            visitor, ctx, join_name(prefix, "backbone")
        )
        self.head.for_each_state[target, V](
            visitor, ctx, join_name(prefix, "head")
        )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.backbone.zero_grad[target](ctx)
        self.head.zero_grad[target](ctx)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        # Propagate the BN train/eval toggle (model.set_attr["training"]) into
        # both children (the Trainer flips this each epoch).
        self.backbone.set_attr[ATTR](value)
        self.head.set_attr[ATTR](value)
