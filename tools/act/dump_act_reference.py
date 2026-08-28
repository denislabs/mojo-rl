#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | ACT reference dumps — the numbers the Mojo port is gated against
# +--------------------------------------------------------------------------+ #
"""Run pieces of `references/act-main/` under PyTorch and dump every
intermediate to a flat binary directory the Mojo gates read.

    pixi run -e act-ref python tools/act/dump_act_reference.py --out /tmp/act_ref

⚠ Runs in the `act-ref` pixi environment ONLY. Nothing under `mojo_rl/` imports
torch; this exists so the port is checked against the reference rather than
against itself.

## Output format

One directory of `<name>.bin` (raw little-endian float32, C order) plus a
`manifest.txt` of `name<TAB>d0,d1,...` lines. Deliberately not `.npy`: the Mojo
side has no numpy, and a flat blob plus a shape line needs no parser.

## Sections

`--only` selects one:

* `xattn` — `torch.nn.functional.scaled_dot_product_attention` driven exactly
  the way `nn.MultiheadAttention` drives it internally, at `Q_LEN != KV_LEN`,
  with and without a key padding mask, plus the q/k/v gradients of a scalar
  objective. Gates `mojo_rl/nn/primitives/cross_attention.mojo`.
* `layers` — the reference's own `TransformerEncoderLayer` /
  `TransformerDecoderLayer` (imported from `references/act-main/`, not
  reimplemented), in eval mode, with every parameter emitted under the Mojo
  side's `for_each_param` name. Gates `deep_agents/act/layers.mojo`.

More sections land with the milestones they gate (M3 ResNet18 + position
embeddings, M4 CVAE/losses, M5 the whole `DETRVAE`).

## ⚠ The weight-layout transpose

torch `nn.Linear` stores `weight` as `[out_features, in_features]` and computes
`x @ Wᵀ`. This framework's `Linear` computes `y = x @ W + b`, so its `W` is
`[in, out]`. `emit_linear` transposes, in ONE place, and `test_act_linear_layout`
gates that single decision on a single `Linear` before any composite is
believed. A transpose error inside a square `[DIM, DIM]` projection produces a
plausible wrong number, not a shape error.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch


class Dump:
    """A directory of float32 blobs + a shape manifest."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries: list[tuple[str, tuple[int, ...]]] = []

    def add(self, name: str, t):
        a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
        a = np.ascontiguousarray(a, dtype=np.float32)
        (self.root / f"{name}.bin").write_bytes(a.tobytes())
        self.entries.append((name, a.shape))
        return a

    def close(self):
        """Write the manifest, MERGING with whatever is already there.

        ⚠ It used to overwrite. `--only frozen_bn` into an existing dump then
        left a manifest naming 8 arrays, the other ~200 `.bin` files still on
        disk and unreachable, and five ACT gates failing with
        `RefDump: no array named 'ens_chunks'` — which reads as a code
        regression and is not one. `--only` exists to avoid re-running the slow
        sections; silently invalidating them defeats it.

        Names written this run win, so re-dumping one section refreshes it.
        """
        merged: dict[str, str] = {}
        existing = self.root / "manifest.txt"
        if existing.is_file():
            for line in existing.read_text().splitlines():
                if "\t" in line:
                    name, shape = line.split("\t", 1)
                    # Drop entries whose blob has since been removed, so a
                    # hand-cleaned directory does not resurrect them.
                    if (self.root / f"{name}.bin").is_file():
                        merged[name] = shape
        for n, sh in self.entries:
            merged[n] = ",".join(str(d) for d in sh)
        existing.write_text(
            "\n".join(f"{n}\t{sh}" for n, sh in merged.items()) + "\n"
        )
        kept = len(merged) - len(self.entries)
        print(
            f"  wrote {len(self.entries)} arrays to {self.root}"
            + (f" ({kept} kept from a previous run)" if kept > 0 else "")
        )


def mha(q, k, v, n_heads, key_valid=None):
    """Multi-head attention exactly as `nn.MultiheadAttention` computes it.

    Written out rather than called so the dump has no hidden reshape: the Mojo
    leaf is gated on this arithmetic, and `nn.MultiheadAttention` would also
    bring its own in-projection weights, which the Mojo leaf does not have (in
    DETR the projections are separate `Linear`s applied before the attention).

    `q  (B, Q, D)`, `k`/`v  (B, KV, D)`, `key_valid (B, KV)` with 1 = attend
    (INVERSE of torch's `key_padding_mask`, matching the Mojo leaf).
    """
    b, ql, d = q.shape
    kl = k.shape[1]
    hd = d // n_heads

    def split(x, ln):
        return x.reshape(b, ln, n_heads, hd).permute(0, 2, 1, 3)

    qs, ks, vs = split(q, ql), split(k, kl), split(v, kl)
    scores = qs @ ks.transpose(-1, -2) / (hd ** 0.5)  # (B, H, Q, KV)
    if key_valid is not None:
        bias = torch.where(
            key_valid.bool(), 0.0, float("-inf")
        )  # (B, KV)
        scores = scores + bias[:, None, None, :]
    attn = scores.softmax(dim=-1)
    out = attn @ vs  # (B, H, Q, HD)
    return out.permute(0, 2, 1, 3).reshape(b, ql, d), attn


def section_xattn(dump: Dump, seed: int):
    """Cross-attention forward + q/k/v grads, masked and unmasked."""
    torch.manual_seed(seed)
    B, H, D = 3, 4, 16
    QL, KL = 5, 7

    q = torch.randn(B, QL, D, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, KL, D, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, KL, D, dtype=torch.float64, requires_grad=True)

    # Per-sample key padding: sample 0 keeps all keys, sample 1 keeps 4 of 7,
    # sample 2 keeps 1. Different lengths per row is the case `MaskedAttention`
    # structurally cannot represent.
    valid = torch.ones(B, KL, dtype=torch.float64)
    valid[1, 4:] = 0.0
    valid[2, 1:] = 0.0

    dump.add("xattn_q", q)
    dump.add("xattn_k", k)
    dump.add("xattn_v", v)
    dump.add("xattn_valid", valid)

    # An arbitrary but FIXED downstream weighting, so the backward pass is a
    # real VJP rather than a sum (a sum would hide any per-position error).
    torch.manual_seed(seed + 1)
    w = torch.randn(B, QL, D, dtype=torch.float64)
    dump.add("xattn_gout", w)

    for tag, mask in (("plain", None), ("masked", valid)):
        for t in (q, k, v):
            if t.grad is not None:
                t.grad = None
        out, attn = mha(q, k, v, H, mask)
        dump.add(f"xattn_out_{tag}", out)
        dump.add(f"xattn_attn_{tag}", attn)
        (out * w).sum().backward()
        dump.add(f"xattn_dq_{tag}", q.grad)
        dump.add(f"xattn_dk_{tag}", k.grad)
        dump.add(f"xattn_dv_{tag}", v.grad)

    # Self-attention shape (QL == KL) with pos on q,k but not v — the DETR
    # encoder layer's attention, which the existing self-attention leaf cannot
    # express because it derives q, k and v from one tensor.
    torch.manual_seed(seed + 2)
    S = 6
    x = torch.randn(B, S, D, dtype=torch.float64, requires_grad=True)
    pos = torch.randn(1, S, D, dtype=torch.float64).expand(B, S, D)
    dump.add("xattn_selfx", x)
    dump.add("xattn_selfpos", pos)
    out, _ = mha(x + pos, x + pos, x, H, None)
    dump.add("xattn_self_out", out)


# ── DETR layers ─────────────────────────────────────────────────────────


def emit_linear(dump: Dump, name: str, lin):
    """One `nn.Linear` in THIS framework's layout.

    torch: `weight [out, in]`, `y = x @ Wᵀ`.  here: `W [in, out]`, `y = x @ W`.
    The transpose lives here and nowhere else.
    """
    dump.add(f"{name}.weight", lin.weight.detach().t().contiguous())
    dump.add(f"{name}.bias", lin.bias)


def emit_mha(dump: Dump, prefix: dict, mha):
    """`nn.MultiheadAttention` -> four separate projections.

    torch packs q/k/v into one `in_proj_weight [3D, D]` when the three inputs
    share an embed dim. Splitting it is the whole mapping: rows [0:D] are q,
    [D:2D] k, [2D:3D] v — and getting that order wrong is invisible in every
    shape and in most magnitudes.
    """
    d = mha.embed_dim
    w = mha.in_proj_weight.detach()  # [3D, D]
    b = mha.in_proj_bias.detach()  # [3D]
    for i, key in enumerate(("q", "k", "v")):
        dump.add(f"{prefix[key]}.weight", w[i * d : (i + 1) * d].t().contiguous())
        dump.add(f"{prefix[key]}.bias", b[i * d : (i + 1) * d])
    dump.add(f"{prefix['o']}.weight", mha.out_proj.weight.detach().t().contiguous())
    dump.add(f"{prefix['o']}.bias", mha.out_proj.bias)


def emit_ln(dump: Dump, name: str, ln):
    dump.add(f"{name}.gamma", ln.weight)
    dump.add(f"{name}.beta", ln.bias)


def section_linear(dump: Dump, seed: int):
    """One bare `nn.Linear`. Gates the transpose convention on its own, before
    any composite depends on it."""
    torch.manual_seed(seed + 100)
    IN, OUT, B = 5, 3, 4
    lin = torch.nn.Linear(IN, OUT).double()
    x = torch.randn(B, IN, dtype=torch.float64)
    dump.add("lin_x", x)
    dump.add("lin_out", lin(x))
    emit_linear(dump, "lin", lin)


def import_reference():
    """Put `references/act-main/` on the path and return its `models` package.

    Every reference module ends with `import IPython; e = IPython.embed` — a
    debugger hook that is never called. Stubbing it beats adding IPython to the
    gate environment, and beats editing the reference tree, which must stay
    exactly as published for a comparison against it to mean anything.
    """
    import sys
    import types

    if "IPython" not in sys.modules:
        stub = types.ModuleType("IPython")
        stub.embed = lambda *a, **k: None  # never invoked
        sys.modules["IPython"] = stub

    root = Path(__file__).resolve().parents[2] / "references/act-main"
    for p in (str(root), str(root / "detr")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def section_layers(dump: Dump, seed: int):
    """The REFERENCE's own layer classes, imported rather than reimplemented."""
    import_reference()
    from models.transformer import (  # type: ignore
        TransformerDecoderLayer,
        TransformerEncoderLayer,
    )

    torch.manual_seed(seed + 200)
    B, D, H, FF = 2, 16, 4, 32
    SEQ, QL, KL = 6, 5, 7

    # ── encoder layer, unmasked ──
    enc = TransformerEncoderLayer(D, H, FF, dropout=0.1, normalize_before=False)
    enc.eval()  # dropout off — the Mojo side runs set_attr["training"](0)
    x = torch.randn(SEQ, B, D)
    pos = torch.randn(SEQ, B, D)
    dump.add("enc_x", x.permute(1, 0, 2))  # -> (B, SEQ, D), our flat layout
    dump.add("enc_pos", pos.permute(1, 0, 2))
    with torch.no_grad():
        out = enc(x, pos=pos)
    dump.add("enc_out", out.permute(1, 0, 2))
    emit_mha(dump, {"q": "enc.q.0", "k": "enc.k.0", "v": "enc.v.0", "o": "enc.ao.0"},
             enc.self_attn)
    emit_ln(dump, "enc.n1.0", enc.norm1)
    emit_linear(dump, "enc.f1.0", enc.linear1)
    emit_linear(dump, "enc.f2.0", enc.linear2)
    emit_ln(dump, "enc.out.0", enc.norm2)

    # ── encoder layer, per-sample key padding ──
    # torch's `src_key_padding_mask` is True = IGNORE; our `key_valid` is
    # 1.0 = ATTEND. Emitting `valid` (not `is_pad`) keeps the polarity
    # decision in one place instead of at each gate.
    encm = TransformerEncoderLayer(D, H, FF, dropout=0.1, normalize_before=False)
    encm.eval()
    xm = torch.randn(SEQ, B, D)
    posm = torch.randn(SEQ, B, D)
    is_pad = torch.zeros(B, SEQ, dtype=torch.bool)
    is_pad[0, 4:] = True  # sample 0 keeps 4 of 6
    is_pad[1, 2:] = True  # sample 1 keeps 2 of 6
    dump.add("encm_x", xm.permute(1, 0, 2))
    dump.add("encm_pos", posm.permute(1, 0, 2))
    dump.add("encm_valid", (~is_pad).float())
    with torch.no_grad():
        outm = encm(xm, pos=posm, src_key_padding_mask=is_pad)
    dump.add("encm_out", outm.permute(1, 0, 2))
    emit_mha(dump,
             {"q": "encm.q.0", "k": "encm.k.0", "v": "encm.v.0", "o": "encm.ao.0"},
             encm.self_attn)
    emit_ln(dump, "encm.n1.0", encm.norm1)
    emit_linear(dump, "encm.f1.0", encm.linear1)
    emit_linear(dump, "encm.f2.0", encm.linear2)
    emit_ln(dump, "encm.out.0", encm.norm2)

    # ── decoder layer ──
    dec = TransformerDecoderLayer(D, H, FF, dropout=0.1, normalize_before=False)
    dec.eval()
    tgt = torch.randn(QL, B, D)
    mem = torch.randn(KL, B, D)
    mpos = torch.randn(KL, B, D)
    qpos = torch.randn(QL, B, D)
    dump.add("dec_tgt", tgt.permute(1, 0, 2))
    dump.add("dec_mem", mem.permute(1, 0, 2))
    dump.add("dec_mpos", mpos.permute(1, 0, 2))
    dump.add("dec_qpos", qpos.permute(1, 0, 2))
    with torch.no_grad():
        dout = dec(tgt, mem, pos=mpos, query_pos=qpos)
    dump.add("dec_out", dout.permute(1, 0, 2))
    emit_mha(dump,
             {"q": "dec.sq.0", "k": "dec.sk.0", "v": "dec.sv.0", "o": "dec.sao.0"},
             dec.self_attn)
    emit_ln(dump, "dec.n1.0", dec.norm1)
    emit_mha(dump,
             {"q": "dec.cq.0", "k": "dec.ck.0", "v": "dec.cv.0", "o": "dec.cao.0"},
             dec.multihead_attn)
    emit_ln(dump, "dec.n2.0", dec.norm2)
    emit_linear(dump, "dec.f1.0", dec.linear1)
    emit_linear(dump, "dec.f2.0", dec.linear2)
    emit_ln(dump, "dec.out.0", dec.norm3)


# ── position embeddings + ResNet18 ──────────────────────────────────────


def section_pos(dump: Dump, seed: int):
    """Both position tables, from the reference's OWN code.

    The 1-D table comes from `detr_vae.get_sinusoid_encoding_table`; the 2-D one
    from `position_encoding.PositionEmbeddingSine`, driven exactly as
    `build_position_encoding` builds it (`num_pos_feats = hidden_dim // 2`,
    `normalize=True`).

    Both are re-emitted in TOKEN-MAJOR `[N, DIM]`, which is what the Mojo side
    produces — the reference keeps the 2-D one in NCHW and permutes it inside
    `Transformer.forward`.
    """
    import_reference()
    from models.detr_vae import get_sinusoid_encoding_table  # type: ignore
    from models.position_encoding import PositionEmbeddingSine  # type: ignore

    SEQ, DIM = 9, 16
    t = get_sinusoid_encoding_table(SEQ, DIM)  # (1, SEQ, DIM)
    dump.add("pos1d_table", t[0])

    OH, OW, D2 = 3, 5, 8
    pe = PositionEmbeddingSine(D2 // 2, normalize=True)
    # `forward` reads only `x[0, [0]]`'s spatial shape; the values are unused.
    x = torch.zeros(1, 1, OH, OW)
    with torch.no_grad():
        p = pe(x)  # (1, D2, OH, OW)
    # NCHW -> token-major (OH*OW, D2), the layout SinusoidalPos2DTokens emits.
    dump.add("pos2d_table", p[0].flatten(1).permute(1, 0).contiguous())


def resnet18_trunk(net, x):
    """torchvision `resnet18` forward, truncated at `layer4` — the ACT backbone.

    One definition, because the pretrained-weight dump must run EXACTLY the
    same truncation as the gate does or the two disagree for a reason that has
    nothing to do with the weights.
    """
    with torch.no_grad():
        y = net.conv1(x)
        y = net.bn1(y)
        y = net.relu(y)
        y = net.maxpool(y)
        y = net.layer1(y)
        y = net.layer2(y)
        y = net.layer3(y)
        y = net.layer4(y)
    return y


def emit_resnet18(dump: Dump, net, prefix: str = "rn18"):
    """Every `resnet18` weight + BN statistic under the Mojo side's
    `for_each_param` names, which come from the `Sequential` child indices of
    `models/resnet18.mojo`.

    ⚠ ONE COPY OF THIS MAPPING. `dump_resnet18_imagenet.py` calls it too. A
    second transcription is a second thing to keep in step with a topology
    change, and this file has already paid for that lesson elsewhere.
    """

    def emit_conv(name, conv):
        # Conv weights are [OC, IC, KH, KW] on both sides — NO transpose. Only
        # `nn.Linear` differs, and that is handled in `emit_linear`.
        dump.add(f"{name}.weight", conv.weight)
        # ⚠ torchvision's ResNet convs are `bias=False` (a BatchNorm follows, and
        # its beta subsumes any bias). This framework's `Conv2D` ALWAYS has one.
        # Emitting zeros makes the two models the same function; without it our
        # conv bias would keep its random init and the comparison would read as
        # a small numerical disagreement rather than as a structural one.
        assert conv.bias is None, f"{name}: expected bias=False"
        dump.add(f"{name}.bias", torch.zeros(conv.out_channels))

    def emit_bn(name, bn):
        dump.add(f"{name}.gamma", bn.weight)
        dump.add(f"{name}.beta", bn.bias)
        dump.add(f"{name}.running_mean", bn.running_mean)
        dump.add(f"{name}.running_var", bn.running_var)

    def emit_basic(pre, blk, downsample: bool):
        emit_conv(f"{pre}.0.0", blk.conv1)
        emit_bn(f"{pre}.0.1", blk.bn1)
        emit_conv(f"{pre}.0.3", blk.conv2)
        emit_bn(f"{pre}.0.4", blk.bn2)
        if downsample:
            emit_conv(f"{pre}.1.0", blk.downsample[0])
            emit_bn(f"{pre}.1.1", blk.downsample[1])

    # Stem: Sequential[Conv2DBatchNormReLU[...], MaxPool2D] -> "0.0" / "0.1"
    emit_conv(f"{prefix}.0.0.0", net.conv1)
    emit_bn(f"{prefix}.0.0.1", net.bn1)

    # Blocks. ResBlockConv2DBN  = Sequential[Residual[Seq[conv,bn,relu,conv,bn]], relu]
    #         ResBlockDownsampleBN = Sequential[ProjectedResidual[main, skip], relu]
    idx = 1  # Sequential child 0 is the stem
    for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
        for blk in layer:
            emit_basic(f"{prefix}.{idx}.0", blk, blk.downsample is not None)
            idx += 1


def section_resnet(dump: Dump, seed: int):
    """torchvision `resnet18` truncated at `layer4`, on RANDOM weights."""
    import torchvision

    torch.manual_seed(seed + 300)
    IN_H, IN_W, B = 64, 96, 2
    net = torchvision.models.resnet18(weights=None)
    net.eval()  # BN in eval -> running stats (init: mean 0, var 1) => identity-ish

    x = torch.randn(B, 3, IN_H, IN_W)
    y = resnet18_trunk(net, x)
    dump.add("rn18_x", x)
    dump.add("rn18_out", y)
    print(f"      resnet18 {IN_H}x{IN_W} -> {tuple(y.shape)}")
    emit_resnet18(dump, net, "rn18")


def section_frozen_bn(dump: Dump, seed: int):
    """torchvision `FrozenBatchNorm2d` — what BOTH ACT implementations wrap the
    ResNet backbone's normalization in.

    Statistics AND affine are `register_buffer`, so all four are constants and
    none of them takes a gradient. Dumped with statistics that are DELIBERATELY
    far from the init values (running_var != 1, running_mean != 0): at the init
    values frozen BN is near-identity and a gate that ignored the statistics
    entirely would still pass.

    `grad_input` is dumped too. A frozen BatchNorm still PASSES gradient — it is
    a fixed affine map, `gi = gamma * inv_std * dy` — and a "freeze" that also
    stopped the gradient reaching the convolutions below would train nothing and
    look like a learning-rate problem.
    """
    from torchvision.ops.misc import FrozenBatchNorm2d

    torch.manual_seed(seed + 700)
    B, C, H, W = 2, 6, 5, 4
    bn = FrozenBatchNorm2d(C)
    with torch.no_grad():
        bn.weight.copy_(torch.randn(C) * 0.5 + 1.0)
        bn.bias.copy_(torch.randn(C) * 0.3)
        bn.running_mean.copy_(torch.randn(C) * 0.7)
        bn.running_var.copy_(torch.rand(C) * 2.0 + 0.5)  # far from 1.0

    x = torch.randn(B, C, H, W, requires_grad=True)
    y = bn(x)
    go = torch.randn(B, C, H, W)
    y.backward(go)

    dump.add("fbn.gamma", bn.weight)
    dump.add("fbn.beta", bn.bias)
    dump.add("fbn.running_mean", bn.running_mean)
    dump.add("fbn.running_var", bn.running_var)
    dump.add("fbn_x", x)
    dump.add("fbn_out", y)
    dump.add("fbn_go", go)
    dump.add("fbn_gin", x.grad)
    print(f"      FrozenBatchNorm2d {B}x{C}x{H}x{W}, running_var in "
          f"[{bn.running_var.min():.3f}, {bn.running_var.max():.3f}]")


# ── CVAE + losses ───────────────────────────────────────────────────────


def section_cvae(dump: Dump, seed: int):
    """The reference's OWN `kl_divergence` and masked-L1 expression, plus the
    reparameterization run on an INJECTED noise draw.

    Injecting `eps` (rather than letting either side sample) is what makes the
    reparameterization comparable at all: two RNGs will never agree, and gating
    only the mean/variance would not catch a wrong `exp(logvar/2)`.
    """
    import_reference()
    import torch.nn.functional as F  # noqa: N812
    from policy import kl_divergence  # type: ignore

    torch.manual_seed(seed + 400)
    B, L = 4, 6
    mu = torch.randn(B, L, dtype=torch.float64, requires_grad=True)
    logvar = torch.randn(B, L, dtype=torch.float64, requires_grad=True) * 0.5
    logvar = logvar.detach().requires_grad_(True)
    eps = torch.randn(B, L, dtype=torch.float64)

    dump.add("cvae_packed", torch.cat([mu, logvar], dim=1))
    dump.add("cvae_eps", eps)

    # reparametrize, with eps supplied rather than drawn (detr_vae.py:16).
    z = mu + logvar.div(2).exp() * eps
    dump.add("cvae_z", z)
    gz = torch.randn(B, L, dtype=torch.float64)
    dump.add("cvae_gz", gz)
    (z * gz).sum().backward()
    dump.add("cvae_dpacked_reparam", torch.cat([mu.grad, logvar.grad], dim=1))

    # KL — the reference function, unmodified. It returns
    # `(total_kld, dim_wise, mean_kld)` where total_kld is `klds.sum(1).mean(0)`;
    # we want the PER-SAMPLE sum, so recompute the row sums the same way and
    # check they agree with the reference's batch mean.
    mu.grad = None
    logvar.grad = None
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    per_sample = klds.sum(1)
    total_kld, _, _ = kl_divergence(mu, logvar)
    assert torch.allclose(per_sample.mean(), total_kld[0]), "KL reduction drift"
    dump.add("cvae_kl", per_sample)
    gk = torch.randn(B, dtype=torch.float64)
    dump.add("cvae_gk", gk)
    (per_sample * gk).sum().backward()
    dump.add("cvae_dpacked_kl", torch.cat([mu.grad, logvar.grad], dim=1))

    # Masked L1 — `policy.py:29`, verbatim including the `.mean()` denominator.
    torch.manual_seed(seed + 401)
    K, D = 5, 3
    pred = torch.randn(B, K, D, dtype=torch.float64, requires_grad=True)
    tgt = torch.randn(B, K, D, dtype=torch.float64, requires_grad=True)
    is_pad = torch.zeros(B, K, dtype=torch.bool)
    is_pad[1, 3:] = True  # 3 of 5 valid
    is_pad[2, 1:] = True  # 1 of 5 valid
    is_pad[3, :] = True  # NOTHING valid — the degenerate row
    dump.add("l1_pred", pred)
    dump.add("l1_tgt", tgt)
    dump.add("l1_valid", (~is_pad).double())

    all_l1 = F.l1_loss(tgt, pred, reduction="none")
    masked = all_l1 * ~is_pad.unsqueeze(-1)
    per_row = masked.reshape(B, -1).mean(dim=1)  # the per-sample form
    assert torch.allclose(per_row.mean(), masked.mean()), "L1 reduction drift"
    dump.add("l1_out", per_row)
    gl = torch.randn(B, dtype=torch.float64)
    dump.add("l1_g", gl)
    (per_row * gl).sum().backward()
    dump.add("l1_dpred", pred.grad)
    dump.add("l1_dtgt", tgt.grad)


# ── the whole DETRVAE ───────────────────────────────────────────────────


class Args:
    """The subset of `detr/main.py`'s argparse namespace that `build()` reads."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def section_detrvae(dump: Dump, seed: int):
    """The reference `DETRVAE`, end to end, with its loss.

    ⚠ `state_dim` is a parameter of `build()` but 14 is HARDCODED into three
    layers (`input_proj_robot_state`, `encoder_action_proj`,
    `encoder_joint_proj`, `detr_vae.py:53,71,72`). So the gate runs at 14 rather
    than at the SO-101's 6 — patching the reference to accept 6 would mean
    comparing against something we edited, which is not comparing against the
    reference. Our graph is parameterized, so 14 exercises the same code.

    ⚠ `reparametrize` is patched to return `mu` (i.e. eps = 0) for the duration.
    Two RNG streams cannot agree, and the Mojo side has the matching
    `set_attr["deterministic"]` seam. The sampling arithmetic itself is already
    gated in `section_cvae` against an injected eps.
    """
    import_reference()
    import torch.nn.functional as F  # noqa: N812
    import models.detr_vae as dv  # type: ignore
    from models.detr_vae import build as build_vae  # type: ignore
    from policy import kl_divergence  # type: ignore

    B, QPOS, ADIM = 2, 14, 14
    N_CAM, IMG_H, IMG_W = 2, 64, 96
    K, DIM, HEADS, FF = 6, 16, 2, 32
    N_ENC, N_DEC = 1, 1

    args = Args(
        hidden_dim=DIM,
        dropout=0.1,
        nheads=HEADS,
        dim_feedforward=FF,
        enc_layers=N_ENC,
        dec_layers=N_DEC,
        pre_norm=False,
        num_queries=K,
        camera_names=["a", "b"][:N_CAM],
        backbone="resnet18",
        dilation=False,
        position_embedding="sine",
        masks=False,
        lr_backbone=1e-5,
    )

    torch.manual_seed(seed + 500)
    orig_reparam = dv.reparametrize
    dv.reparametrize = lambda mu, logvar: mu  # deterministic z = mu
    try:
        model = build_vae(args)
        model.eval()  # dropout off, BN on running stats

        qpos = torch.randn(B, QPOS)
        image = torch.randn(B, N_CAM, 3, IMG_H, IMG_W)
        actions = torch.randn(B, K, ADIM)
        is_pad = torch.zeros(B, K, dtype=torch.bool)
        is_pad[1, 4:] = True  # sample 1 keeps 4 of 6

        with torch.no_grad():
            a_hat, _, (mu, logvar) = model(qpos, image, None, actions, is_pad)
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1_rows = (all_l1 * ~is_pad.unsqueeze(-1)).reshape(B, -1).mean(1)
            klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_rows = klds.sum(1)

        dump.add("vae_qpos", qpos)
        # (B, N_CAM, 3, H, W) is already our flat `images` layout.
        dump.add("vae_images", image)
        dump.add("vae_actions", actions)
        dump.add("vae_valid", (~is_pad).float())
        dump.add("vae_ahat", a_hat)
        dump.add("vae_latinfo", torch.cat([mu, logvar], dim=1))

        # ── intermediates, for bisecting a whole-model disagreement ──
        # Recomputed by walking `DETRVAE.forward` (detr_vae.py:87) rather than
        # by hooking, so each tensor is named by the graph node it should equal.
        with torch.no_grad():
            ae = model.encoder_action_proj(actions)
            qe = model.encoder_joint_proj(qpos).unsqueeze(1)
            ce = model.cls_embed.weight.unsqueeze(0).repeat(B, 1, 1)
            einp = torch.cat([ce, qe, ae], axis=1)  # (B, K+2, DIM)
            dump.add("vae_einp", einp)
            dump.add("vae_epos", model.pos_table[0])  # (K+2, DIM)
            eo = model.encoder(
                einp.permute(1, 0, 2),
                pos=model.pos_table.clone().detach().permute(1, 0, 2),
                src_key_padding_mask=torch.cat(
                    [torch.full((B, 2), False), is_pad], axis=1
                ),
            )
            dump.add("vae_cvae", eo.permute(1, 0, 2))  # (B, K+2, DIM)
            dump.add("vae_clsout", eo[0])  # (B, DIM)

            feats = model.backbones[0][0](image[:, 0])["0"]
            dump.add("vae_feat0", feats)  # (B, 512, OH, OW), camera 0
            src0 = model.input_proj(feats)  # (B, DIM, OH, OW)
            dump.add(
                "vae_src0", src0.flatten(2).permute(0, 2, 1).contiguous()
            )  # token-major (B, OH*OW, DIM)
        dump.add("vae_l1", l1_rows)
        dump.add("vae_kl", kl_rows)

        # ── parameters, under the Mojo graph's `for_each_param` names ──
        m = model
        dump.add("vae.cls.queries", m.cls_embed.weight)  # (1, DIM)
        emit_linear(dump, "vae.qenc", m.encoder_joint_proj)
        emit_linear(dump, "vae.aenc.0", m.encoder_action_proj)
        emit_linear(dump, "vae.latinfo", m.latent_proj)
        emit_linear(dump, "vae.lattok", m.latent_out_proj)
        emit_linear(dump, "vae.prop", m.input_proj_robot_state)
        dump.add("vae.addpos.queries", m.additional_pos_embed.weight)
        dump.add("vae.qpe.queries", m.query_embed.weight)
        emit_linear(dump, "vae.ahat.0", m.action_head)

        # `input_proj` is a 1x1 Conv2d == a per-token Linear. [OC, IC, 1, 1] ->
        # [IC, OC], the same transpose `emit_linear` applies.
        ip = m.input_proj
        dump.add(
            "vae.src.0.weight",
            ip.weight.detach().reshape(ip.out_channels, ip.in_channels)
            .t()
            .contiguous(),
        )
        dump.add("vae.src.0.bias", ip.bias)

        # CVAE encoder stack (`model.encoder`) and the transformer's own
        # encoder/decoder stacks.
        for i, lyr in enumerate(m.encoder.layers):
            emit_mha(
                dump,
                {
                    "q": f"vae.cvae.{i}.q.0",
                    "k": f"vae.cvae.{i}.k.0",
                    "v": f"vae.cvae.{i}.v.0",
                    "o": f"vae.cvae.{i}.ao.0",
                },
                lyr.self_attn,
            )
            emit_ln(dump, f"vae.cvae.{i}.n1.0", lyr.norm1)
            emit_linear(dump, f"vae.cvae.{i}.f1.0", lyr.linear1)
            emit_linear(dump, f"vae.cvae.{i}.f2.0", lyr.linear2)
            emit_ln(dump, f"vae.cvae.{i}.out.0", lyr.norm2)

        for i, lyr in enumerate(m.transformer.encoder.layers):
            emit_mha(
                dump,
                {
                    "q": f"vae.memory.{i}.q.0",
                    "k": f"vae.memory.{i}.k.0",
                    "v": f"vae.memory.{i}.v.0",
                    "o": f"vae.memory.{i}.ao.0",
                },
                lyr.self_attn,
            )
            emit_ln(dump, f"vae.memory.{i}.n1.0", lyr.norm1)
            emit_linear(dump, f"vae.memory.{i}.f1.0", lyr.linear1)
            emit_linear(dump, f"vae.memory.{i}.f2.0", lyr.linear2)
            emit_ln(dump, f"vae.memory.{i}.out.0", lyr.norm2)

        for i, lyr in enumerate(m.transformer.decoder.layers):
            emit_mha(
                dump,
                {
                    "q": f"vae.hs.{i}.sq.0",
                    "k": f"vae.hs.{i}.sk.0",
                    "v": f"vae.hs.{i}.sv.0",
                    "o": f"vae.hs.{i}.sao.0",
                },
                lyr.self_attn,
            )
            emit_ln(dump, f"vae.hs.{i}.n1.0", lyr.norm1)
            emit_mha(
                dump,
                {
                    "q": f"vae.hs.{i}.cq.0",
                    "k": f"vae.hs.{i}.ck.0",
                    "v": f"vae.hs.{i}.cv.0",
                    "o": f"vae.hs.{i}.cao.0",
                },
                lyr.multihead_attn,
            )
            emit_ln(dump, f"vae.hs.{i}.n2.0", lyr.norm2)
            emit_linear(dump, f"vae.hs.{i}.f1.0", lyr.linear1)
            emit_linear(dump, f"vae.hs.{i}.f2.0", lyr.linear2)
            emit_ln(dump, f"vae.hs.{i}.out.0", lyr.norm3)
        emit_ln(dump, "vae.hsn.0", m.transformer.decoder.norm)

        # The backbone. `Joiner[0]` is the Backbone; `.body` is the
        # IntermediateLayerGetter over the torchvision resnet18.
        body = m.backbones[0][0].body

        def emit_conv(name, conv):
            dump.add(f"{name}.weight", conv.weight)
            assert conv.bias is None, f"{name}: expected bias=False"
            dump.add(f"{name}.bias", torch.zeros(conv.out_channels))

        def emit_fbn(name, bn):
            # ⚠ FrozenBatchNorm2d, NOT nn.BatchNorm2d — it holds weight/bias/
            # running_mean/running_var as BUFFERS and applies them with a
            # hardcoded eps of 1e-5, matching this framework's BN in eval.
            dump.add(f"{name}.gamma", bn.weight)
            dump.add(f"{name}.beta", bn.bias)
            dump.add(f"{name}.running_mean", bn.running_mean)
            dump.add(f"{name}.running_var", bn.running_var)

        # ⚠ the extra ".0": the backbone sits inside `Tokenwise[N_CAM, ...]`,
        # which contributes one naming level of its own before the ResNet's.
        emit_conv("vae.feat.0.0.0.0", body.conv1)
        emit_fbn("vae.feat.0.0.0.1", body.bn1)
        idx = 1
        for layer in (body.layer1, body.layer2, body.layer3, body.layer4):
            for blk in layer:
                pre = f"vae.feat.0.{idx}.0"
                emit_conv(f"{pre}.0.0", blk.conv1)
                emit_fbn(f"{pre}.0.1", blk.bn1)
                emit_conv(f"{pre}.0.3", blk.conv2)
                emit_fbn(f"{pre}.0.4", blk.bn2)
                if blk.downsample is not None:
                    emit_conv(f"{pre}.1.0", blk.downsample[0])
                    emit_fbn(f"{pre}.1.1", blk.downsample[1])
                idx += 1
        print(
            f"      DETRVAE {IMG_H}x{IMG_W} x{N_CAM}cam, K={K}, DIM={DIM}"
            f" -> a_hat {tuple(a_hat.shape)}"
        )
    finally:
        dv.reparametrize = orig_reparam


def section_ensemble(dump: Dump, seed: int):
    """Temporal ensembling, computed by the REFERENCE's own storage scheme.

    `imitate_episodes.py:248` verbatim: a `[T, T+K, A]` buffer, occupancy by
    `torch.all(actions != 0, axis=1)`, `exp_weights = exp(-k * arange(n))`
    normalized, gathered in ascending query order. The Mojo side uses a `K x K`
    ring and explicit occupancy; this dump is what makes those equivalent
    rather than merely plausible.
    """
    torch.manual_seed(seed + 600)
    T, K, A = 12, 4, 3
    kk = 0.01

    # One distinct chunk per query step, values far from zero so the
    # reference's `!= 0` occupancy test behaves (see the Mojo header).
    chunks = torch.randn(T, K, A, dtype=torch.float64) + 3.0
    dump.add("ens_chunks", chunks)

    all_time = torch.zeros(T, T + K, A, dtype=torch.float64)
    outs = []
    for t in range(T):
        all_time[[t], t : t + K] = chunks[t]
        cur = all_time[:, t]
        populated = torch.all(cur != 0, axis=1)
        cur = cur[populated]
        w = np.exp(-kk * np.arange(len(cur)))
        w = w / w.sum()
        outs.append((cur * torch.from_numpy(w)[:, None]).sum(dim=0))
    dump.add("ens_out", torch.stack(outs))
    print(f"      ensemble T={T} K={K} A={A} k={kk}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/act_ref")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="all")
    args = ap.parse_args()

    torch.set_grad_enabled(True)
    dump = Dump(Path(args.out))

    if args.only in ("all", "xattn"):
        print("[xattn] cross-attention forward + grads")
        section_xattn(dump, args.seed)
    if args.only in ("all", "linear"):
        print("[linear] one nn.Linear (the transpose convention)")
        section_linear(dump, args.seed)
    if args.only in ("all", "layers"):
        print("[layers] DETR encoder / masked encoder / decoder layers")
        section_layers(dump, args.seed)
    if args.only in ("all", "pos"):
        print("[pos] 1-D ACT table + 2-D DETR sine table")
        section_pos(dump, args.seed)
    if args.only in ("all", "frozen_bn"):
        print("[frozen_bn] torchvision FrozenBatchNorm2d")
        section_frozen_bn(dump, args.seed)

    if args.only in ("all", "resnet"):
        print("[resnet] torchvision resnet18 through layer4")
        section_resnet(dump, args.seed)
    if args.only in ("all", "cvae"):
        print("[cvae] reparameterize / KL / masked L1")
        section_cvae(dump, args.seed)
    if args.only in ("all", "detrvae"):
        print("[detrvae] the whole model + loss")
        section_detrvae(dump, args.seed)
    if args.only in ("all", "ensemble"):
        print("[ensemble] temporal ensembling")
        section_ensemble(dump, args.seed)

    dump.close()


if __name__ == "__main__":
    main()
