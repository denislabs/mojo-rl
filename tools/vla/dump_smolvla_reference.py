#!/usr/bin/env python3
"""Dump SmolVLA tower activations from HuggingFace's OWN SmolVLM implementation.

    pixi run -e act-ref python tools/vla/dump_smolvla_reference.py --out /tmp/vla_ref

Every brick of the Mojo port is already checked against something outside itself
— RoPE against MAX's kernel, the block mask against the reference's documented
examples, PixelShuffle against numpy replaying the torch ops, the time embedding
against numpy, the loaded weights against a hand-written decode. What none of
that covers is whether the bricks are WIRED TOGETHER correctly.

A torch forward written here from the same reading of the architecture could not
see a misreading — it would restate it. So this drives `transformers`' own
`SmolVLMVisionTransformer` and the text tower it ships, loaded with the real
checkpoint's weights, and dumps their outputs.

⚠ **The text tower is truncated to 16 layers**, matching `num_vlm_layers: 16` in
SmolVLA's config. The backbone's own config says 32; using 32 here would produce
a reference the port is right to disagree with.

Output is `refload.mojo`'s format, so the Mojo gate reads it with the existing
`RefDump`:

    <out>/manifest.txt      `name<TAB>d0,d1,...` per array
    <out>/<name>.bin        raw little-endian float32, C order
"""

import argparse
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

VLM = "model.vlm_with_expert.vlm."
VIS = VLM + "model.vision_model."
TXT = VLM + "model.text_model."

DEFAULT_WEIGHTS = os.path.expanduser(
    "~/.cache/mojo_rl/hub/lerobot__smolvla_base/main/model.safetensors"
)


def _seeded(shape, scale, mod):
    """The same deterministic pattern the Mojo gates build, so both sides feed
    the towers identical inputs without shipping an input file."""
    n = int(np.prod(shape))
    idx = np.arange(n, dtype=np.int64)
    return (((idx * mod[0]) % mod[1]) - mod[2]).astype(np.float32).reshape(shape) * scale


class Dump:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.lines = []

    def add(self, name, arr):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
        a.tofile(os.path.join(self.root, name + ".bin"))
        self.lines.append(f"{name}\t{','.join(str(d) for d in a.shape)}")
        print(f"  {name:28s} {str(a.shape):22s} "
              f"min {a.min():+.6f} max {a.max():+.6f}")

    def close(self):
        with open(os.path.join(self.root, "manifest.txt"), "w") as f:
            f.write("\n".join(self.lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-seq", type=int, default=32)
    ap.add_argument("--layers", type=int, default=16)
    a = ap.parse_args()

    from transformers.models.smolvlm.configuration_smolvlm import SmolVLMVisionConfig
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMVisionTransformer

    torch.manual_seed(0)
    sd = load_file(a.weights)
    d = Dump(a.out)

    # ── vision tower ─────────────────────────────────────────────────────
    vcfg = SmolVLMVisionConfig(
        hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
        num_attention_heads=12, patch_size=16, image_size=512,
    )
    vis = SmolVLMVisionTransformer(vcfg).eval()
    vsd = {k[len(VIS):]: v.float() for k, v in sd.items() if k.startswith(VIS)}
    missing, unexpected = vis.load_state_dict(vsd, strict=False)
    print(f"vision: loaded {len(vsd)} tensors, missing {len(missing)},"
          f" unexpected {len(unexpected)}")
    if unexpected:
        raise SystemExit(f"unexpected vision keys: {unexpected[:5]}")
    if missing:
        raise SystemExit(f"missing vision keys: {missing[:5]}")

    px = torch.from_numpy(_seeded((1, 3, 512, 512), 0.03, (31, 23, 11)))
    with torch.no_grad():
        # Bisection points: if the embeddings already disagree the fault is in
        # the patch conv / transpose / position table, not in twelve layers of
        # attention. Localising beats guessing.
        # All-True: the image fills the 32x32 grid. SmolVLM's embeddings bucket
        # FRACTIONAL coordinates into position ids to support ragged images, but
        # at a full grid `bucketize` returns exactly [0..31] per axis, so
        # pos_ids == h*32 + w — the plain raster order. Worth knowing: at a
        # PARTIAL grid it would not, and a flat position table would be wrong.
        pam = torch.ones(1, 32, 32, dtype=torch.bool)
        emb = vis.embeddings(pixel_values=px, patch_attention_mask=pam)
        l0 = vis.encoder.layers[0](emb, attention_mask=None)
        l0 = l0[0] if isinstance(l0, tuple) else l0
        vout = vis(pixel_values=px).last_hidden_state
    d.add("vision_in", px.numpy())
    d.add("vision_emb", emb.numpy())
    d.add("vision_l0", l0.numpy())
    d.add("vision_out", vout.numpy())

    # ── text tower, TRUNCATED to SmolVLA's 16 layers ─────────────────────
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel

    tcfg = LlamaConfig(
        hidden_size=960, intermediate_size=2560, num_hidden_layers=a.layers,
        num_attention_heads=15, num_key_value_heads=5, head_dim=64,
        rms_norm_eps=1e-5, rope_theta=100000.0, vocab_size=49280,
        attention_bias=False, mlp_bias=False, tie_word_embeddings=False,
        _attn_implementation="eager",
    )
    txt = LlamaModel(tcfg).eval()
    tsd = {}
    for k, v in sd.items():
        if not k.startswith(TXT):
            continue
        rel = k[len(TXT):]
        if rel.startswith("layers."):
            li = int(rel.split(".")[1])
            if li >= a.layers:
                continue
        tsd[rel] = v.float()
    missing, unexpected = txt.load_state_dict(tsd, strict=False)
    print(f"text: loaded {len(tsd)} tensors, missing {len(missing)},"
          f" unexpected {len(unexpected)}")
    if unexpected:
        raise SystemExit(f"unexpected text keys: {unexpected[:5]}")

    S = a.text_seq
    emb = torch.from_numpy(_seeded((1, S, 960), 0.05, (29, 17, 8)))
    pos = torch.arange(S)[None, :]
    # prefix-LM block mask: 8 image + (S-9) language as ONE bidirectional block,
    # then state as its own — SmolVLA's `ar`, additive 4-D as LlamaModel wants.
    ar = np.array([0] * (S - 1) + [1], dtype=np.int64)
    cs = np.cumsum(ar)
    allow = cs[None, :] <= cs[:, None]
    m4 = torch.zeros(1, 1, S, S)
    m4[0, 0][~torch.from_numpy(allow)] = torch.finfo(torch.float32).min
    with torch.no_grad():
        tout = txt(
            inputs_embeds=emb, attention_mask=m4, position_ids=pos,
            use_cache=False,
        ).last_hidden_state
    d.add("text_in", emb.numpy())
    d.add("text_out", tout.numpy())
    d.add("text_ar", ar.astype(np.float32))

    d.close()
    print(f"\n{a.out}: {len(d.lines)} arrays")
    return 0


if __name__ == "__main__":
    sys.exit(main())
