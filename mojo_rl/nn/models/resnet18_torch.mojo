# +--------------------------------------------------------------------------+ #
# | ResNet18: our parameter names <-> torchvision's
# +--------------------------------------------------------------------------+ #
"""The `TorchNameMap` for `models/resnet18.mojo`, in both directions.

    from mojo_rl.nn.models.resnet18_torch import resnet18_torch_map
    from mojo_rl.nn.core.torch_names import LoadTorchNamed

    var v = LoadTorchNamed["feat.0."](
        SafeTensors(path), resnet18_torch_map(3)
    )
    model.for_each_param[target, ...](v, ctx)
    model.for_each_state[target, ...](v, ctx)
    v.report(String("load_backbone"))

## Where the weights come from

`timm/resnet18.tv_in1k` on the Hub is torchvision's `IMAGENET1K_V1` republished
as `model.safetensors` (46.8 MB, ungated), and its keys ARE torchvision's:
`conv1.weight`, `layer2.0.downsample.1.running_var`. Which means this map plus
`io/safetensors.mojo` replaces `tools/act/dump_resnet18_imagenet.py` and the
`torch` + `torchvision` install behind it — on a TRAINING path, not a gate.

⚠ "tv_in1k" says the weights came from torchvision; it does not PROVE they are
bit-identical to what `resnet18(weights=IMAGENET1K_V1)` loads. We are unusually
well placed to find out rather than assume, because the dump this replaces is
still here: `tests/deep_agents/act/test_act_pretrained_backbone.mojo` compares
a backbone carrying these weights against a torchvision reference forward.

## The two names being mapped

    ours    `for_each_param` on `ResNet18Seq`, so `Sequential` child indices:
            child 0 is the stem (`Conv2DBatchNormReLU` + `MaxPool2D`), children
            1..8 are the eight `BasicBlock`s in order.
    theirs  torchvision attribute paths: `conv1`, `bn1`, `layerL.B.conv1`, ...

⚠ THIS IS THE SECOND TRANSCRIPTION OF THIS TOPOLOGY. The first is
`tools/act/dump_act_reference.py:emit_resnet18`, which writes the torch
reference dump. Two copies of a mapping is exactly the shape this repo's most
frequent defect takes, so they are not left to agree by inspection:
`test_act_pretrained_backbone.mojo` checks that the NAME SETS are identical
before it compares a single number. If you change one, that check fails before
anything subtle does.

## Nothing here transposes

ResNet18 has no `nn.Linear` before `fc`, and `fc` is past `layer4` — the ACT
backbone is `IntermediateLayerGetter(..., {"layer4": "0"})` and never reaches
it, so it is not mapped. Convolution weights are `[OC, IC, KH, KW]` on both
sides. The `TN_ZEROS` conv biases are the only entry that is not a plain
rename, and they are the reason a "just rename it" loader would be wrong.
"""

from mojo_rl.nn.core.torch_names import TorchNameMap, TN_PLAIN, TN_ZEROS


comptime RESNET18_TV_REPO = "timm/resnet18.tv_in1k"
"""torchvision's ImageNet-1k weights, republished with a safetensors file."""
comptime RESNET18_TV_FILE = "model.safetensors"

comptime RESNET18_MAP_ENTRIES = 120
"""20 convolutions (weight + a zero bias) + 20 BatchNorms (4 tensors each).

An exact count, checked by `resnet18_torch_map`. The map is built by a loop
over a topology table, and both of that loop's failure modes — a block emitted
twice, a block not emitted at all — leave a map that still works for whatever
it does contain."""


def _conv(
    mut m: TorchNameMap, ours: String, theirs: String, oc: Int, ic: Int, k: Int
) raises:
    var ws: List[Int] = [oc, ic, k, k]
    m.add(ours + ".weight", theirs + ".weight", ws)
    # ⚠ torchvision's ResNet convolutions are `bias=False` — a BatchNorm
    # follows and its beta subsumes any bias — while this framework's `Conv2D`
    # ALWAYS has one. Zeroing it makes the two models the same function.
    # Leaving it at its random init makes them different ones, at a magnitude
    # that reads as a numerical disagreement rather than as a missing tensor.
    var bs: List[Int] = [oc]
    m.add(ours + ".bias", String(""), bs, TN_ZEROS)


def _bn(mut m: TorchNameMap, ours: String, theirs: String, c: Int) raises:
    var s: List[Int] = [c]
    m.add(ours + ".gamma", theirs + ".weight", s)
    m.add(ours + ".beta", theirs + ".bias", s)
    # ⚠ Running statistics are STATE, not parameters, and they are not
    # optional. Pretrained convolutions carrying this framework's init
    # statistics (mean 0, var 1) are not the pretrained network; they are a
    # different function that happens to share its weights.
    m.add(ours + ".running_mean", theirs + ".running_mean", s)
    m.add(ours + ".running_var", theirs + ".running_var", s)
    # `num_batches_tracked` is deliberately NOT mapped. It is torchvision's
    # momentum bookkeeping, it is I64 with shape [], and we have no parameter
    # for it — so it stays in the file, unread.


def resnet18_torch_map(in_ch: Int = 3) raises -> TorchNameMap:
    """The full ResNet18 trunk map, backbone-local on our side.

    `in_ch` sizes the stem convolution only; everything downstream is fixed by
    the architecture."""
    var m = TorchNameMap()

    # Sequential child 0 — the stem. `MaxPool2D` has no parameters, so the
    # `.1` of `Conv2DBatchNormReLU` + `MaxPool2D` contributes no names.
    _conv(m, String("0.0.0"), String("conv1"), 64, in_ch, 7)
    _bn(m, String("0.0.1"), String("bn1"), 64)

    # Children 1..8 — the eight BasicBlocks, two per torchvision `layer`.
    #   ResBlockConv2DBN     = Sequential[Residual[Seq[c,bn,relu,c,bn]], relu]
    #   ResBlockDownsampleBN = Sequential[ProjectedResidual[main, skip], relu]
    # so inside a block, `0.0`/`0.1` are conv1/bn1, `0.3`/`0.4` are conv2/bn2
    # (`0.2` is the ReLU), and `1.0`/`1.1` are the projection when there is one.
    var layers: List[String] = [
        String("layer1"), String("layer1"), String("layer2"), String("layer2"),
        String("layer3"), String("layer3"), String("layer4"), String("layer4"),
    ]
    var blocks: List[Int] = [0, 1, 0, 1, 0, 1, 0, 1]
    var in_c: List[Int] = [64, 64, 64, 128, 128, 256, 256, 512]
    var out_c: List[Int] = [64, 64, 128, 128, 256, 256, 512, 512]
    var downsample: List[Bool] = [
        False, False, True, False, True, False, True, False,
    ]

    for j in range(8):
        var po = String(j + 1) + ".0"
        var pt = layers[j] + "." + String(blocks[j])
        _conv(m, po + ".0.0", pt + ".conv1", out_c[j], in_c[j], 3)
        _bn(m, po + ".0.1", pt + ".bn1", out_c[j])
        _conv(m, po + ".0.3", pt + ".conv2", out_c[j], out_c[j], 3)
        _bn(m, po + ".0.4", pt + ".bn2", out_c[j])
        if downsample[j]:
            _conv(m, po + ".1.0", pt + ".downsample.0", out_c[j], in_c[j], 1)
            _bn(m, po + ".1.1", pt + ".downsample.1", out_c[j])

    if m.size() != RESNET18_MAP_ENTRIES:
        raise Error(
            "resnet18_torch_map: built " + String(m.size()) + " entries,"
            " expected " + String(RESNET18_MAP_ENTRIES)
        )
    return m^
