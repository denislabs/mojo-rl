"""Pre-built model / block compositions for nn.

Each model here is a *compositional* alias — it expands to a `Sequential`
/ `Residual` / `ProjectedResidual` of existing primitives, so it gets
correct forward / vjp / walkers / `set_attr` propagation for free and
needs no bespoke kernels.

Navigation (import from the specific submodule, e.g.
`from mojo_rl.nn.models.gpt import GPT`):

  - `conv`        — `Conv2DReLU`, `Conv2DBatchNormReLU`
  - `resnet`      — `ResBlockConv2DBN`, `ResBlockDownsampleBN`
  - `transformer` — `MultiHeadAttentionXL`, `MultiHeadAttention`,
                    `TransformerFFN`, `TransformerBlock` (shared pieces)
  - `gpt`         — `GPT`, the nanoGPT dropout variants
                    (`MultiHeadAttentionDrop`, `TransformerFFNDrop`,
                    `TransformerBlockDrop`, `GPTDrop`, `GPTDropTied`) and
                    the GPT construction ops (`gpt_scale_residual_proj`,
                    `gpt_wire_tie`)
  - `vit`         — `PatchEmbed`, `ViT`
"""
