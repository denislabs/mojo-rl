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

from mojo_rl.nn.constants import DT
from mojo_rl.nn.models.conv import Conv2DBatchNormReLU
from mojo_rl.nn.models.resnet import ResBlockConv2DBN, ResBlockDownsampleBN
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
