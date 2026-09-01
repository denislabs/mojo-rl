# +--------------------------------------------------------------------------+ #
# | The Hub backbone, checkable on a box with no PyTorch
# +--------------------------------------------------------------------------+ #
"""Gate `timm/resnet18.tv_in1k` — the file `ACT_PRETRAINED=hub` downloads.

    pixi run mojo run -I . tests/nn/test_resnet18_hub_weights.mojo

⚠ **THIS EXISTS BECAUSE THE OTHER BACKBONE GATE CANNOT RUN WHERE IT MATTERS.**
`tests/nn/test_safetensors_resnet18_torch.mojo` compares all 11,190,912 values
against a `dump_resnet18_imagenet.py` dump — the right check, and it needs
PyTorch and the `act-ref` environment. The whole point of the `hub` route is a
box that has neither. So on exactly the machine that uses these weights,
nothing verified them before a multi-hour training run.

This is the check that runs anywhere: a pinned SHA-256 of the downloaded file,
and every tensor the ACT backbone asks for present at the expected shape.

## ⚠ WHAT A RED GATE HERE MEANS, AND WHAT IT DOES NOT

Red means **the upstream file changed**. `timm/resnet18.tv_in1k` is someone
else's repository; nothing stops it being re-uploaded.

**DO NOT JUST UPDATE THE HASH.** That defeats the entire gate — it turns
"upstream silently changed the weights your policies were trained on" into a
one-line commit. The response is:

  1. re-run `tools/act/dump_resnet18_imagenet.py` under `-e act-ref`;
  2. run `tests/nn/test_safetensors_resnet18_torch.mojo`, which compares every
     value against torchvision;
  3. only if THAT passes, re-pin the hash here, and say in the commit message
     that upstream re-uploaded and what the diff was.

The SHA is the strong check — it covers all 11,699,132 values exactly. The
shape walk exists so a mismatch says WHICH tensor is wrong instead of only
"the file differs", and so a missing tensor is reported as a missing tensor.

⚠ NEEDS THE NETWORK on its first run, then the file is cached
(`~/.cache/mojo_rl/hub/`). It is therefore NOT in the smoke tier, which is
defined as self-contained. It needs no PyTorch, no dump and no dataset.
"""

from mojo_rl.io.hf import HF_MODEL, hf_download_file
from mojo_rl.io.safetensors import SafeTensors
from mojo_rl.io.sha256 import sha256_file
from mojo_rl.nn.core.torch_names import TN_ZEROS
from mojo_rl.nn.models.resnet18_torch import (
    RESNET18_TV_REPO, resnet18_torch_map,
)


comptime EXPECT_SHA = "694f673df6520a3158624e8a89af086f59923ee4cd7436fe5bc3bc71d295ad81"
"""SHA-256 of `timm/resnet18.tv_in1k/model.safetensors`, pinned 2026-09-01.

Verified transitively: the same bytes were compared value for value against a
torchvision `IMAGENET1K_V1` dump by `test_safetensors_resnet18_torch.mojo`.
See the header before touching this."""

comptime EXPECT_TENSORS = 122
comptime EXPECT_VALUES = 11699132
"""Every value in the file, INCLUDING the `fc` classifier ACT does not use.
The backbone's own total is 11,190,912 — the number the torchvision gate
compares — and the difference is `fc` plus the `num_batches_tracked` counters."""


def main() raises:
    print("[resnet18-hub] gate")

    var path = hf_download_file(
        String(RESNET18_TV_REPO), String("model.safetensors"), HF_MODEL,
        verbose=False,
    )
    print("  file  " + path)

    # ── the strong check ─────────────────────────────────────────────
    var sha = sha256_file(path)
    if sha != EXPECT_SHA:
        raise Error(
            "the Hub file changed.\n  expected " + String(EXPECT_SHA)
            + "\n  got      " + sha
            + "\nDO NOT simply re-pin this hash — read the header. Re-dump"
            " under `-e act-ref` and re-run"
            " tests/nn/test_safetensors_resnet18_torch.mojo first."
        )
    print("  sha256 matches the pinned value")

    # ── the readable check ───────────────────────────────────────────
    var st = SafeTensors(String(path))
    if len(st.names) != EXPECT_TENSORS:
        raise Error(
            "the file holds " + String(len(st.names)) + " tensors, expected "
            + String(EXPECT_TENSORS)
        )
    var total = 0
    for i in range(len(st.names)):
        var e = 1
        for k in range(st.shape_rank[i]):
            e *= st.shape_data[st.shape_start[i] + k]
        total += e
    if total != EXPECT_VALUES:
        raise Error(
            "the file holds " + String(total) + " values, expected "
            + String(EXPECT_VALUES)
        )
    print(
        "  " + String(len(st.names)) + " tensors, " + String(total)
        + " values"
    )

    # ── every tensor the ACT backbone will ask for ───────────────────
    # ⚠ FROM THE MAP, NOT A LIST WRITTEN HERE. `resnet18_torch_map` is what
    # `load_backbone_safetensors` walks at load time, so asking it what it
    # wants is the only version of this check that cannot drift from the
    # loader. A hand-written list of names would pass forever while the
    # loader asked for something else.
    var m = resnet18_torch_map(3)
    var checked = 0
    var missing = 0
    var zeroed = 0
    for i in range(len(m.theirs)):
        # ⚠ AN EMPTY `theirs` IS NOT A MISSING TENSOR. torchvision's ResNet
        # convolutions are `bias=False` — the following BatchNorm's beta
        # subsumes any bias — while this framework's `Conv2D` always has one,
        # so the map carries it as `TN_ZEROS`: "zero this, it has no
        # counterpart". Counting those as missing reported 20 absent tensors
        # for a file that is completely correct.
        if m.kind[i] == TN_ZEROS:
            zeroed += 1
            continue
        var want_name = m.theirs[i]
        var want_elems = 1
        for k in range(m.shape_rank[i]):
            want_elems *= m.shape_data[m.shape_start[i] + k]

        var found = -1
        for j in range(len(st.names)):
            if st.names[j] == want_name:
                found = j
                break
        if found < 0:
            print("    MISSING: " + want_name)
            missing += 1
            continue

        var got_elems = 1
        for k in range(st.shape_rank[found]):
            got_elems *= st.shape_data[st.shape_start[found] + k]
        if got_elems != want_elems:
            raise Error(
                "shape mismatch for '" + want_name + "': the backbone wants "
                + String(want_elems) + " values, the file has "
                + String(got_elems)
            )
        checked += 1

    if missing != 0:
        raise Error(
            String(missing) + " tensor(s) the ACT backbone loads are not in"
            " the Hub file — it is not the network this loader expects"
        )
    # ⚠ VACUITY. An empty map would sail through the loop above and report a
    # clean run over nothing at all.
    if checked < 60:
        raise Error(
            "only " + String(checked) + " tensors were checked; the name map"
            " is empty or truncated, so this gate proved nothing"
        )
    # ⚠ PIN THE ZEROED COUNT TOO. ResNet18 has exactly 20 convolutions
    # (stem + 16 in blocks + 3 downsample projections), so 20 biases are
    # zeroed. If that number moves, the map changed shape and this gate's
    # skip rule needs re-reading rather than silently covering less.
    if zeroed != 20:
        raise Error(
            String(zeroed) + " zero-filled entries, expected 20 (one bias per"
            " ResNet18 convolution). The name map changed."
        )
    print(
        "  " + String(checked) + " mapped + " + String(zeroed)
        + " zero-filled = " + String(len(m.theirs))
        + " backbone tensors, all at the right shape"
    )

    # A value the loader will actually read, so the gate touches DATA and not
    # only the header it just validated.
    var w = st.read_f32(String("layer4.1.conv2.weight"))
    if len(w) != 512 * 512 * 3 * 3:
        raise Error(
            "layer4.1.conv2.weight has " + String(len(w)) + " values"
        )
    print("  read " + String(len(w)) + " values from the deepest conv")
    print("[PASS] resnet18-hub")
