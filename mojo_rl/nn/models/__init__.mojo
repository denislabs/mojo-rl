"""Models — pre-built architectures composed from storage leaves.

Storage-surface port of `nn/models/`. The transformer/ViT/ResNet/Conv pieces are
pure `comptime` compositions (no structs/kernels). GPT adds two weight-surgery
construction ops (`gpt_scale_residual_proj`, `gpt_wire_tie`).
"""

from .transformer import (
    MultiHeadAttention,
    MultiHeadAttentionXL,
    TransformerFFN,
    TransformerBlock,
)
from .conv import Conv2DReLU, Conv2DBatchNormReLU
from .resnet import ResBlockConv2DBN, ResBlockDownsampleBN
from .vit import PatchEmbed, ViT
from .gpt import (
    GPT,
    GPTDrop,
    GPTDropTied,
    MultiHeadAttentionDrop,
    TransformerFFNDrop,
    TransformerBlockDrop,
    gpt_scale_residual_proj,
    gpt_wire_tie,
)
