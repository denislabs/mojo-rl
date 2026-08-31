#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | torchvision ResNet18, ImageNet weights  ->  a dump the ACT backbone loads
# +--------------------------------------------------------------------------+ #
"""ImageNet-pretrained ResNet18 weights, in this framework's parameter names.

    pixi run -e act-ref python tools/act/dump_resnet18_imagenet.py \
        --out ~/.cache/mojo_rl/act_so101/resnet18_imagenet

Run ONCE. The result is ~45 MB of float32 blobs plus a manifest, read by
`ACTTrainer.load_backbone` (`deep_agents/act/refload.mojo`).

⚠ THIS IS NO LONGER THE ONLY WAY TO GET THESE WEIGHTS, and no longer the
default one. `ACT_PRETRAINED=hub` fetches `timm/resnet18.tv_in1k` — the same
torchvision `IMAGENET1K_V1` weights, republished as a `.safetensors` — and
`ACTTrainer.load_backbone_safetensors` reads it with no `torch`, no
`torchvision` and no `-e act-ref`.

What this script is FOR now is being the oracle that makes that trustworthy:
`tests/nn/test_safetensors_resnet18_torch.mojo` compares every value of the
Hub file against this dump and requires BIT equality (11,190,912 of them). A
repo name is a claim; that gate is the check. Keep this script.

## Why this exists

The 50-episode run overfits: validation L1 bottoms out at epoch 15 and rises
for the next 33 while training L1 keeps falling. The backbone is random at step
0 and has to learn vision from 12,411 frames, which is the one thing the paper
does NOT do — `detr/main.py` builds `resnet18(weights=IMAGENET1K_V1)` and
fine-tunes it at `lr_backbone=1e-5`. ImageNet pretraining is the regularizer
this dataset cannot supply from its own frames.

## What is dumped, and what is deliberately not

Weights and BN running statistics for the trunk through `layer4` — the ACT
backbone is `IntermediateLayerGetter(..., {"layer4": "0"})`, so `avgpool` and
`fc` are never reached and are not emitted. **The running statistics are the
point**, not an afterthought: pretrained weights with the framework's init
statistics (mean 0, var 1) are not the pretrained network, they are a different
function that happens to share its convolutions.

The name mapping is `emit_resnet18` in `dump_act_reference.py` — imported, not
re-transcribed, so the gate and this script cannot disagree about it.

## The gate

A reference forward on a FIXED input at the SO-101 resolution is dumped
alongside (`rn18in_x`, `rn18in_out`), so
`tests/deep_agents/act/test_act_pretrained_backbone.mojo` can check that our
ResNet18 carrying these weights computes what torchvision computes. Loading
weights that produce a different function is exactly the failure this has to
exclude, and a checksum could not see it.

⚠ The input is `torch.randn`, NOT an ImageNet-normalized photo. What is being
gated is the weight conversion, and for that a fixed arbitrary input is
strictly better: a natural image would leave most of the network in a typical
regime, while noise exercises every channel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_act_reference import Dump, emit_resnet18, resnet18_trunk  # noqa: E402


# The SO-101 store's resolution — `deep_agents/act/config.mojo`. Gating at the
# resolution actually trained on means the dumped feature map has the shape the
# training graph will produce (8x10), so a geometry mistake shows up here.
IN_H, IN_W, B = 240, 320, 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=str(
            Path.home() / ".cache/mojo_rl/act_so101/resnet18_imagenet"
        ),
    )
    ap.add_argument(
        "--weights",
        default="IMAGENET1K_V1",
        help="torchvision weight enum; the only one resnet18 ships",
    )
    args = ap.parse_args()

    import torchvision

    print(f"[1/3] fetching torchvision resnet18 weights={args.weights} ...")
    net = torchvision.models.resnet18(weights=args.weights)
    net.eval()

    # A trained network must not look like an untrained one. `bn1.running_var`
    # is exactly 1.0 at init and is not after ImageNet, so this separates "the
    # weights arrived" from "an enum silently resolved to None".
    rv = net.bn1.running_var.detach().numpy()
    if abs(float(rv.mean()) - 1.0) < 1e-6:
        raise SystemExit(
            "bn1.running_var is still at its init value — these are NOT"
            f" pretrained weights (--weights {args.weights} resolved to"
            " nothing?)"
        )
    print(
        f"      bn1.running_var mean={rv.mean():.4f}"
        f" (init would be exactly 1.0)"
    )

    out = Path(args.out)
    dump = Dump(out)

    print(f"[2/3] reference forward at {IN_H}x{IN_W} ...")
    torch.manual_seed(0)
    x = torch.randn(B, 3, IN_H, IN_W)
    y = resnet18_trunk(net, x)
    dump.add("rn18in_x", x)
    dump.add("rn18in_out", y)
    print(f"      {tuple(x.shape)} -> {tuple(y.shape)}")

    print("[3/3] emitting weights + BN running statistics ...")
    emit_resnet18(dump, net, "rn18in")
    dump.close()

    n_bytes = sum(f.stat().st_size for f in out.glob("*.bin"))
    print(f"\nwrote {out}\n  {n_bytes / 1e6:.1f} MB")
    print(
        "\nuse it with:\n"
        f"  export ACT_PRETRAINED={out}\n"
        "  ...then run examples/so101/act_so101_train_gpu.mojo"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
