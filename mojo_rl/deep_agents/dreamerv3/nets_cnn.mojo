"""DreamerV3 CNN encoder / decoder for pixel observations (P1 of the pixel arc).

The MLP `DreamerEncoder`/`DreamerDecoder` in `nets.mojo` only handle vector obs.
Pixel obs (CarRacing: stacked grayscale frames) need a convolutional encoder
(image → tokens) and a transposed-conv decoder (feature → image). Like the rest
of the Dreamer nets these are pure `comptime` `Sequential` aliases over existing
primitives — `Conv2D` (down) and the new `Conv2DTranspose` (up) — so they get
forward / vjp / param-walkers for free.

In this storage framework a tensor is always a flat `[B, DIM]` slab; `Conv2D` /
`Conv2DTranspose` interpret the flat `DIM` as `[C,H,W]` (or `[H,W,C]` for NHWC)
via index math. So there is NO reshape/flatten module between the conv stack and
the Linear bottleneck — the flat conv output IS the Linear input, and the flat
Linear output IS the deconv input (as long as the LAYOUT is consistent).

Architecture = the DreamerV3 image backbone: a fixed 4-layer stride-2 stack
(kernel 4, stride 2, pad 1) with channels `BASE·{1,2,4,8}`. Each stride-2 conv
HALVES an even spatial dim; each transposed conv DOUBLES it. So:

    encoder  C×H×W → BASE×H/2 → 2B×H/4 → 4B×H/8 → 8B×H/16 → Linear → tokens[TOKEN]
    decoder  feat[FEATIN] → Linear → 8B×H/16 → 4B×H/8 → 2B×H/4 → B×H/2 → C×H×W

`H` (== `W`) MUST be divisible by 16 (so every intermediate dim stays even).
For CarRacing pick a 16-divisible resolution: 64 (→ minres 4) or 96 (→ minres 6).

Norm: the reference inserts a LayerNorm after each conv. v1 omits it (Conv→act
only) to keep the shape/gradient surface minimal; adding per-layer norm is a P5
convergence lever, not a shape change.

Activation `A` defaults to `GELUOp` (matching `nets.mojo`); the training config
passes `SwishOp` (SiLU). The final encoder Linear and final decoder deconv are
LINEAR (no activation) — raw tokens / raw pixel logits.
"""

from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.conv2d import Conv2D
from mojo_rl.nn.primitives.conv2d_transpose import Conv2DTranspose
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.elementwise import Elementwise
from mojo_rl.nn.primitives.ops.gelu_op import GELUOp
from mojo_rl.nn.core.element_op import ElementOp
from mojo_rl.nn.constants import DT, LAYOUT_NCHW


# ── stride-2 building blocks (k=4, s=2, p=1) ──────────────────────────────────
# Down: even H,W → H/2, W/2.  OH = (H + 2 - 4)//2 + 1 = H//2 (even H).
comptime _ConvDown[
    IC: Int, OC: Int, H: Int, W: Int, A: ElementOp, LAYOUT: Int
] = Sequential[
    Conv2D[IC, OC, 4, 2, 1, H, W, DT, LAYOUT],
    Elementwise[OC * (H // 2) * (W // 2), A],
]

# Up: H,W → 2H, 2W.  OHt = (H-1)*2 - 2 + 4 = 2H.
comptime _ConvUp[
    IC: Int, OC: Int, H: Int, W: Int, A: ElementOp, LAYOUT: Int
] = Sequential[
    Conv2DTranspose[IC, OC, 4, 2, 1, H, W, 0, LAYOUT],
    Elementwise[OC * (2 * H) * (2 * W), A],
]


# ── Encoder: image[C·H·W] → 4× stride-2 conv → Linear → tokens[TOKEN] ─────────
comptime DreamerEncoderCNN[
    C: Int, H: Int, W: Int, BASE: Int, TOKEN: Int,
    A: ElementOp = GELUOp, LAYOUT: Int = LAYOUT_NCHW,
] = Sequential[
    _ConvDown[C, BASE, H, W, A, LAYOUT],                       # H   → H/2
    _ConvDown[BASE, 2 * BASE, H // 2, W // 2, A, LAYOUT],      # H/2 → H/4
    _ConvDown[2 * BASE, 4 * BASE, H // 4, W // 4, A, LAYOUT],  # H/4 → H/8
    _ConvDown[4 * BASE, 8 * BASE, H // 8, W // 8, A, LAYOUT],  # H/8 → H/16
    Linear[8 * BASE * (H // 16) * (W // 16), TOKEN],
]


# ── Decoder: feature[FEATIN] → Linear → 4× transposed-conv → image[C·H·W] ─────
comptime DreamerDecoderCNN[
    FEATIN: Int, C: Int, H: Int, W: Int, BASE: Int,
    A: ElementOp = GELUOp, LAYOUT: Int = LAYOUT_NCHW,
] = Sequential[
    Linear[FEATIN, 8 * BASE * (H // 16) * (W // 16)],
    Elementwise[8 * BASE * (H // 16) * (W // 16), A],
    _ConvUp[8 * BASE, 4 * BASE, H // 16, W // 16, A, LAYOUT],  # H/16 → H/8
    _ConvUp[4 * BASE, 2 * BASE, H // 8, W // 8, A, LAYOUT],    # H/8  → H/4
    _ConvUp[2 * BASE, BASE, H // 4, W // 4, A, LAYOUT],        # H/4  → H/2
    Conv2DTranspose[BASE, C, 4, 2, 1, H // 2, W // 2, 0, LAYOUT],  # H/2 → H (raw)
]
