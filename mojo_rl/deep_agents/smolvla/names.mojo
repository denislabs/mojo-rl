# +--------------------------------------------------------------------------+ #
# | SmolVLA <-> our parameter names
# +--------------------------------------------------------------------------+ #
"""`TorchNameMap` builders for `lerobot/smolvla_base`.

Built in loops, not written out 197 times: the tower is twelve identical layers
and a hand-listed table would be twelve chances to typo an index in a way that
loads real weights into the wrong slot. The loop makes the per-layer rule
appear once, where it can be read.

## Where our names come from

They are STRUCTURE-DETERMINED, not chosen. `Sequential.for_each_param` names
children by index (`join_name(prefix, String(i))`), so composed trees produce
paths like `1.0.1.0.1.2.0.weight`. The one place they read semantically is
`SigLIPAttention`, which is a named struct precisely because its four
projections are all `[768, 768]` and therefore mutually substitutable by size:

    1.{i}.0.0.1.q.0.weight     <- ...self_attn.q_proj.weight
                ^ not '.1.'

Reading a positional index wrong there is a silent wrong-weights-right-shape
load; reading `q` wrong is a diff you can see.

## Layout

`nn.Linear.weight` is `[out, in]` and ours is `[in, out]` -> `TN_TRANSPOSE`
(`add_linear` does both sides). Conv2D agrees on `[OC, IC, KH, KW]` -> plain.
`LayerNorm`'s `gamma`/`beta` are their `weight`/`bias` -> plain, rename only.

⚠ The position embedding is `[1024, 768]` on their side and our `BiasAdd`'s
flat `bias` on ours, and it is **plain, not transposed**: both are row-major
position-major, so the flattening already agrees. Marking it TN_TRANSPOSE would
load a shape-legal, silently scrambled table.
"""

from mojo_rl.nn.core.torch_names import (
    TorchNameMap, TN_PLAIN, TN_TRANSPOSE, TN_ZEROS,
)
from .vision import (
    SIGLIP_DIM, SIGLIP_FF, SIGLIP_LAYERS, SIGLIP_PATCH, SIGLIP_TOKENS,
)
from .text import (
    SMOLLM_DIM, SMOLLM_FF, SMOLLM_HEAD_DIM, SMOLLM_KV_HEADS, SMOLLM_LAYERS,
)


comptime SMOLVLA_VLM = String("model.vlm_with_expert.vlm.")
comptime SMOLVLA_VISION = SMOLVLA_VLM + "model.vision_model."
comptime SMOLVLA_TEXT = SMOLVLA_VLM + "model.text_model."


def vision_name_map[
    DIM: Int = SIGLIP_DIM,
    FF: Int = SIGLIP_FF,
    LAYERS: Int = SIGLIP_LAYERS,
    PATCH: Int = SIGLIP_PATCH,
    TOKENS: Int = SIGLIP_TOKENS,
]() raises -> TorchNameMap:
    """The 197 tensors of the SigLIP tower, ours <-> theirs.

    `ours` are relative to the tower's own walk root, so a caller loading the
    tower as a subtree passes its prefix as `LoadTorchNamed`'s GRAPH_PREFIX.
    """
    var m = TorchNameMap()
    var T = SMOLVLA_VISION

    # ── embeddings: patch conv + position table ──────────────────────────
    var conv_shape: List[Int] = [DIM, 3, PATCH, PATCH]
    m.add(String("0.0.weight"), T + "embeddings.patch_embedding.weight",
          conv_shape, TN_PLAIN)
    var conv_b: List[Int] = [DIM]
    m.add(String("0.0.bias"), T + "embeddings.patch_embedding.bias",
          conv_b, TN_PLAIN)
    # ⚠ plain: see the header. Both sides are row-major [TOKENS, DIM].
    var pos_shape: List[Int] = [TOKENS, DIM]
    m.add(String("0.2.bias"), T + "embeddings.position_embedding.weight",
          pos_shape, TN_PLAIN)

    # ── the twelve encoder layers ────────────────────────────────────────
    var d1: List[Int] = [DIM]
    var ff1: List[Int] = [FF]
    for i in range(LAYERS):
        var ours = String("1.") + String(i) + "."
        var theirs = T + "encoder.layers." + String(i) + "."

        # attention branch: LayerNorm then q/k/v/out
        m.add(ours + "0.0.0.0.gamma", theirs + "layer_norm1.weight", d1)
        m.add(ours + "0.0.0.0.beta", theirs + "layer_norm1.bias", d1)
        m.add_linear(ours + "0.0.1.q.0.weight",
                     theirs + "self_attn.q_proj.weight", DIM, DIM)
        m.add(ours + "0.0.1.q.0.bias", theirs + "self_attn.q_proj.bias", d1)
        m.add_linear(ours + "0.0.1.k.0.weight",
                     theirs + "self_attn.k_proj.weight", DIM, DIM)
        m.add(ours + "0.0.1.k.0.bias", theirs + "self_attn.k_proj.bias", d1)
        m.add_linear(ours + "0.0.1.v.0.weight",
                     theirs + "self_attn.v_proj.weight", DIM, DIM)
        m.add(ours + "0.0.1.v.0.bias", theirs + "self_attn.v_proj.bias", d1)
        m.add_linear(ours + "0.0.1.o.0.weight",
                     theirs + "self_attn.out_proj.weight", DIM, DIM)
        m.add(ours + "0.0.1.o.0.bias", theirs + "self_attn.out_proj.bias", d1)

        # MLP branch: LayerNorm then fc1 -> GELU -> fc2
        m.add(ours + "1.0.0.0.gamma", theirs + "layer_norm2.weight", d1)
        m.add(ours + "1.0.0.0.beta", theirs + "layer_norm2.bias", d1)
        m.add_linear(ours + "1.0.1.0.0.weight", theirs + "mlp.fc1.weight",
                     FF, DIM)
        m.add(ours + "1.0.1.0.0.bias", theirs + "mlp.fc1.bias", ff1)
        m.add_linear(ours + "1.0.1.2.0.weight", theirs + "mlp.fc2.weight",
                     DIM, FF)
        m.add(ours + "1.0.1.2.0.bias", theirs + "mlp.fc2.bias", d1)

    # ── post-LayerNorm ───────────────────────────────────────────────────
    m.add(String("2.0.gamma"), T + "post_layernorm.weight", d1)
    m.add(String("2.0.beta"), T + "post_layernorm.bias", d1)
    return m^


def text_name_map[
    DIM: Int = SMOLLM_DIM,
    FF: Int = SMOLLM_FF,
    LAYERS: Int = SMOLLM_LAYERS,
    KV_W: Int = SMOLLM_KV_HEADS * SMOLLM_HEAD_DIM,
]() raises -> TorchNameMap:
    """The SmolLM2 tower: 145 checkpoint tensors + 112 zero-filled biases.

    ⚠ **The biases are the interesting half.** SmolLM2 is bias-free
    (`attention_bias=False`, `mlp_bias=False`) while our `Linear` ALWAYS carries
    a `bias` Param — seven per layer. Left at their random initialisation the
    loaded model is a DIFFERENT FUNCTION from the published one, at a magnitude
    that reads as a numerical disagreement rather than as the missing tensors it
    is. `TN_ZEROS` fills them and skips them on save; `theirs` is empty because
    there is nothing on the other side to name. Exactly the torchvision ResNet18
    conv-bias case this flag was introduced for.

    `ours` are relative to the tower's walk root (`SmolLMTextTower`), which
    covers `layers.*` and the final `norm`. `embed_tokens`, `lm_head` and the
    connector are separate modules and are mapped with them.
    """
    var m = TorchNameMap()
    var T = SMOLVLA_TEXT
    var d1: List[Int] = [DIM]
    var kv1: List[Int] = [KV_W]
    var ff1: List[Int] = [FF]

    for i in range(LAYERS):
        var o = String("0.") + String(i) + "."
        var t = T + "layers." + String(i) + "."

        m.add(o + "0.0.0.0.gamma", t + "input_layernorm.weight", d1)
        m.add_linear(o + "0.0.1.q.0.weight", t + "self_attn.q_proj.weight",
                     DIM, DIM)
        m.add(o + "0.0.1.q.0.bias", String(""), d1, TN_ZEROS)
        m.add_linear(o + "0.0.1.k.0.weight", t + "self_attn.k_proj.weight",
                     KV_W, DIM)
        m.add(o + "0.0.1.k.0.bias", String(""), kv1, TN_ZEROS)
        m.add_linear(o + "0.0.1.v.0.weight", t + "self_attn.v_proj.weight",
                     KV_W, DIM)
        m.add(o + "0.0.1.v.0.bias", String(""), kv1, TN_ZEROS)
        m.add_linear(o + "0.0.1.o.0.weight", t + "self_attn.o_proj.weight",
                     DIM, DIM)
        m.add(o + "0.0.1.o.0.bias", String(""), d1, TN_ZEROS)

        m.add(o + "1.0.0.0.gamma", t + "post_attention_layernorm.weight", d1)
        m.add_linear(o + "1.0.1.gate.0.weight", t + "mlp.gate_proj.weight",
                     FF, DIM)
        m.add(o + "1.0.1.gate.0.bias", String(""), ff1, TN_ZEROS)
        m.add_linear(o + "1.0.1.up.0.weight", t + "mlp.up_proj.weight",
                     FF, DIM)
        m.add(o + "1.0.1.up.0.bias", String(""), ff1, TN_ZEROS)
        m.add_linear(o + "1.0.1.down.0.weight", t + "mlp.down_proj.weight",
                     DIM, FF)
        m.add(o + "1.0.1.down.0.bias", String(""), d1, TN_ZEROS)

    m.add(String("1.0.gamma"), T + "norm.weight", d1)
    return m^
