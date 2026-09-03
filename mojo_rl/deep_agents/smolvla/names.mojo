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
from .expert import (
    EXPERT_W, EXPERT_FF, EXPERT_LAYERS, EXPERT_SELF_EVERY, VLM_W, VLM_KV_W,
)
from .text import (
    SMOLLM_DIM, SMOLLM_FF, SMOLLM_HEAD_DIM, SMOLLM_KV_HEADS, SMOLLM_LAYERS,
)


comptime SMOLVLA_VLM = String("model.vlm_with_expert.vlm.")
comptime SMOLVLA_VISION = SMOLVLA_VLM + "model.vision_model."
comptime SMOLVLA_TEXT = SMOLVLA_VLM + "model.text_model."
comptime SMOLVLA_EXPERT = String("model.vlm_with_expert.lm_expert.")


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


def misc_name_map[
    DIM: Int = 960,
    VOCAB: Int = 49280,
    CONN_IN: Int = 12288,
    STATE: Int = 32,
    ACT: Int = 32,
    W: Int = 720,
]() raises -> TorchNameMap:
    """Connector + token embedding + LM head + the five action heads.

    `ours` are prefixed per component (`connector.`, `embed.`, `lm_head.`,
    `state_proj.`, ...) because these are separate small modules rather than one
    walked tree; a caller loads each with its own `GRAPH_PREFIX`.

    ⚠ Bias handling is NOT uniform here and that is the point. The three BF16
    tensors (connector, embed, lm_head) are bias-free in the checkpoint, so our
    `Linear`'s always-present bias is `TN_ZEROS`. All five F32 action heads DO
    ship a trained bias, mapped plainly. Applying either rule to the other set
    silently discards or invents a bias.
    """
    var m = TorchNameMap()
    var R = String("model.")
    var d1: List[Int] = [DIM]
    var v1: List[Int] = [VOCAB]
    var w1: List[Int] = [W]
    var a1: List[Int] = [ACT]

    # connector: [960, 12288], no bias in the file.
    # ⚠ It lives under the VLM subtree (`…vlm.model.connector.…`), unlike the
    # action heads, which are top-level `model.…`. Two different roots in one
    # map function; the coverage gate caught this exact slip.
    m.add_linear(String("connector.weight"),
                 SMOLVLA_VLM + "model.connector.modality_projection.proj.weight",
                 DIM, CONN_IN)
    m.add(String("connector.bias"), String(""), d1, TN_ZEROS)

    # embedding: torch nn.Embedding.weight is [num_embeddings, dim] and ours is
    # the same [VOCAB, DIM] — a rename, NOT a transpose.
    var emb: List[Int] = [VOCAB, DIM]
    m.add(String("embed.weight"), SMOLVLA_TEXT + "embed_tokens.weight", emb,
          TN_PLAIN)

    # lm_head: a Linear, so [out, in] -> transposed. No bias in the file.
    m.add_linear(String("lm_head.weight"), SMOLVLA_VLM + "lm_head.weight",
                 VOCAB, DIM)
    m.add(String("lm_head.bias"), String(""), v1, TN_ZEROS)

    # the five action heads — all F32, all WITH a real bias
    m.add_linear(String("state_proj.weight"), R + "state_proj.weight",
                 DIM, STATE)
    m.add(String("state_proj.bias"), R + "state_proj.bias", d1)
    m.add_linear(String("action_in.weight"), R + "action_in_proj.weight",
                 W, ACT)
    m.add(String("action_in.bias"), R + "action_in_proj.bias", w1)
    m.add_linear(String("action_out.weight"), R + "action_out_proj.weight",
                 ACT, W)
    m.add(String("action_out.bias"), R + "action_out_proj.bias", a1)
    m.add_linear(String("time_mlp_in.weight"), R + "action_time_mlp_in.weight",
                 W, 2 * W)
    m.add(String("time_mlp_in.bias"), R + "action_time_mlp_in.bias", w1)
    m.add_linear(String("time_mlp_out.weight"),
                 R + "action_time_mlp_out.weight", W, W)
    m.add(String("time_mlp_out.bias"), R + "action_time_mlp_out.bias", w1)
    return m^


def expert_name_map[
    W: Int = EXPERT_W,
    FF: Int = EXPERT_FF,
    LAYERS: Int = EXPERT_LAYERS,
    QW: Int = VLM_W,
    KVW: Int = VLM_KV_W,
    SELF_EVERY: Int = EXPERT_SELF_EVERY,
]() raises -> TorchNameMap:
    """The action expert: 145 checkpoint tensors + 112 zero-filled biases.

    ⚠ **k and v change shape with the layer's parity.** Even layers project
    their own 720-wide stream (`Linear[720 -> 320]`); odd layers project the
    VLM's cached 320-wide K/V (`Linear[320 -> 320]`). Both produce a 320-wide
    result, so a map that used one rule everywhere would be right about the
    OUTPUT shape and wrong about the input on half the layers — and the shape
    check in `LoadTorchNamed` is what would catch it, which is why the map
    declares full 2-D shapes rather than element counts.

    `ours` mirror the checkpoint's own layer indexing (`layers.7.self_attn.k`),
    because `SmolVLAExpert` holds the two kinds in two lists and a positional
    name would number each 0..7 twice.
    """
    var m = TorchNameMap()
    var T = SMOLVLA_EXPERT
    var w1: List[Int] = [W]
    var q1: List[Int] = [QW]
    var kv1: List[Int] = [KVW]
    var ff1: List[Int] = [FF]

    for i in range(LAYERS):
        var o = String("layers.") + String(i) + "."
        var t = T + "layers." + String(i) + "."
        var kv_in = W if (i % SELF_EVERY == 0) else KVW

        m.add(o + "input_layernorm.gamma", t + "input_layernorm.weight", w1)
        m.add_linear(o + "self_attn.q.weight", t + "self_attn.q_proj.weight",
                     QW, W)
        m.add(o + "self_attn.q.bias", String(""), q1, TN_ZEROS)
        m.add_linear(o + "self_attn.k.weight", t + "self_attn.k_proj.weight",
                     KVW, kv_in)
        m.add(o + "self_attn.k.bias", String(""), kv1, TN_ZEROS)
        m.add_linear(o + "self_attn.v.weight", t + "self_attn.v_proj.weight",
                     KVW, kv_in)
        m.add(o + "self_attn.v.bias", String(""), kv1, TN_ZEROS)
        m.add_linear(o + "self_attn.o.weight", t + "self_attn.o_proj.weight",
                     W, QW)
        m.add(o + "self_attn.o.bias", String(""), w1, TN_ZEROS)

        m.add(o + "post_attention_layernorm.gamma",
              t + "post_attention_layernorm.weight", w1)
        m.add_linear(o + "mlp.gate.weight", t + "mlp.gate_proj.weight", FF, W)
        m.add(o + "mlp.gate.bias", String(""), ff1, TN_ZEROS)
        m.add_linear(o + "mlp.up.weight", t + "mlp.up_proj.weight", FF, W)
        m.add(o + "mlp.up.bias", String(""), ff1, TN_ZEROS)
        m.add_linear(o + "mlp.down.weight", t + "mlp.down_proj.weight", W, FF)
        m.add(o + "mlp.down.bias", String(""), w1, TN_ZEROS)

    m.add(String("norm.gamma"), T + "norm.weight", w1)
    return m^
