# +--------------------------------------------------------------------------+ #
# | Is `timm/resnet18.tv_in1k` really torchvision's IMAGENET1K_V1?
# +--------------------------------------------------------------------------+ #
"""Gates `nn/models/resnet18_torch.mojo` against the torch dump it replaces.

    pixi run -e act-ref python tools/act/dump_resnet18_imagenet.py \\
        --out ~/.cache/mojo_rl/act_so101/resnet18_imagenet
    pixi run mojo run -I . tests/nn/test_safetensors_resnet18_torch.mojo
    pixi run -e act-ref python tools/nn/dump_safetensors_reference.py \\
        --verify-resnet18 /tmp/rn18_export.safetensors

## The claim under test

`ACTTrainer.load_backbone_safetensors` fetches `timm/resnet18.tv_in1k` from the
Hub and fills the vision backbone from it, with no `torch` anywhere. Two things
have to be true for that to be the same load as
`tools/act/dump_resnet18_imagenet.py`:

  1. "tv_in1k" means what it says — the file holds torchvision's
     `IMAGENET1K_V1` weights and not some other ImageNet recipe. That is a
     CLAIM IN A REPO NAME, and a repo can be re-uploaded.
  2. `resnet18_torch_map` pairs the right tensors. Every BasicBlock holds two
     convolutions of identical shape, and `layer2`-`layer4` each hold a
     downsample convolution too — so a map that swapped `conv1` and `conv2`
     inside a block, or attached a stage's statistics to the wrong stage, loads
     cleanly, reports full coverage, and computes a different function.

Both are settled the same way: compare EVERY VALUE against the dump, bitwise.
11 million floats agreeing to the last bit is not a coincidence, and no
tolerance is appropriate — nothing in either path does arithmetic.

## The name-set check comes first, and is not redundant

`emit_resnet18` in `tools/act/dump_act_reference.py` is the OTHER transcription
of this topology. Two copies of a mapping is the shape this repo's most
frequent defect takes, so the two name sets are compared as SETS, both
directions, before a single value is. If they drift, this says so in one line
instead of as eleven million mismatches whose cause has to be inferred.
"""

from std.memory import bitcast
from std.os import getenv
from std.os.path import isdir
from std.sys import argv

from mojo_rl.io.hf import HF_MODEL, hf_download_file
from mojo_rl.io.safetensors import SafeTensors
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.torch_names import (
    LoadTorchNamed,
    SaveTorchNamed,
    TN_ZEROS,
)
from mojo_rl.nn.models.resnet18 import ResNet18Backbone
from mojo_rl.nn.models.resnet18_torch import (
    RESNET18_MAP_ENTRIES,
    RESNET18_TV_FILE,
    RESNET18_TV_REPO,
    resnet18_torch_map,
)
from mojo_rl.deep_agents.act.refload import RefDump


# Parameters do not depend on the input geometry, so the map is checked at a
# resolution that compiles in seconds rather than at 240x320.
comptime NET = ResNet18Backbone[3, 64, 96]
comptime LOADER = LoadTorchNamed[""]
comptime SAVER = SaveTorchNamed[""]
comptime DUMP_PREFIX = "rn18in."
comptime EXPORT = "/tmp/rn18_export.safetensors"
comptime GEN = (
    "pixi run -e act-ref python tools/act/dump_resnet18_imagenet.py --out"
    " ~/.cache/mojo_rl/act_so101/resnet18_imagenet"
)


def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


def main() raises:
    var args = argv()
    var dump_dir: String
    if len(args) > 1:
        dump_dir = String(args[1])
    else:
        var home = getenv("HOME")
        if home == "":
            raise Error("$HOME is unset; pass the dump directory as an argument")
        dump_dir = home + "/.cache/mojo_rl/act_so101/resnet18_imagenet"
    if not isdir(dump_dir):
        raise Error(
            "no ResNet18 dump at " + dump_dir + " — generate it with:\n    "
            + GEN
        )

    var fails = 0
    print("timm/resnet18.tv_in1k vs the torchvision dump")
    print("  dump: " + dump_dir)

    var path = hf_download_file(
        String(RESNET18_TV_REPO), String(RESNET18_TV_FILE), HF_MODEL
    )
    var f = SafeTensors(path)
    var m = resnet18_torch_map(3)
    var d = RefDump(String(dump_dir))
    print("")
    print(
        "  file " + String(f.size()) + " tensors, map "
        + String(m.size()) + " entries, dump " + String(len(d.names))
        + " arrays"
    )
    print("")

    check(fails, "the map has its declared size",
          m.size() == RESNET18_MAP_ENTRIES, String(m.size()))

    # ── name sets, both directions ────────────────────────────────────────
    var ours_missing_from_dump = 0
    var first_missing = String("")
    for i in range(m.size()):
        var k = String(DUMP_PREFIX) + String(m.ours[i])
        if not d.has(k):
            ours_missing_from_dump += 1
            if first_missing == "":
                first_missing = k
    check(fails, "every map entry is in the dump",
          ours_missing_from_dump == 0,
          "first absent: " + first_missing if first_missing else
          String(m.size()) + " names")

    # The reverse. The dump also carries the reference forward's input and
    # output (`rn18in_x`, `rn18in_out`), which are not parameters — they are
    # named without the trailing dot, so the prefix test excludes them.
    var dump_missing_from_map = 0
    var first_extra = String("")
    var dump_params = 0
    for i in range(len(d.names)):
        var nm = String(d.names[i])
        if not nm.startswith(String(DUMP_PREFIX)):
            continue
        dump_params += 1
        var local = String(nm[byte = String(DUMP_PREFIX).byte_length() :])
        if m.index_of_ours(local) < 0:
            dump_missing_from_map += 1
            if first_extra == "":
                first_extra = nm
    check(fails, "every dump parameter is in the map",
          dump_missing_from_map == 0 and dump_params == RESNET18_MAP_ENTRIES,
          String(dump_params) + " dump parameters"
          + ("" if first_extra == "" else ", first unmapped: " + first_extra))

    # ── every value, bitwise ──────────────────────────────────────────────
    var compared = 0
    var tensors = 0
    var bad = 0
    var first_bad = String("")
    var zero_bad = 0
    for i in range(m.size()):
        var want = d.get(String(DUMP_PREFIX) + String(m.ours[i]))
        if m.kind[i] == TN_ZEROS:
            # The conv biases torchvision does not have. The dump writes zeros
            # for them; if it ever stopped, our TN_ZEROS fill would silently
            # stop matching the reference model.
            for k in range(len(want)):
                if Float64(want[k]) != 0.0:
                    zero_bad += 1
            tensors += 1
            compared += len(want)
            continue
        var key = String(m.theirs[i])
        if not f.has(key):
            fails += 1
            print("  FAIL  the file has no '" + key + "'")
            continue
        var got = f.read_f32(key)
        if len(got) != len(want):
            fails += 1
            print(
                "  FAIL  " + key + ": " + String(len(got)) + " values, dump"
                " has " + String(len(want))
            )
            continue
        for k in range(len(got)):
            if bitcast[DType.uint32](got[k]) != bitcast[DType.uint32](
                Float32(want[k])
            ):
                bad += 1
                if first_bad == "":
                    first_bad = (
                        key + "[" + String(k) + "]: " + String(got[k])
                        + " vs " + String(want[k])
                    )
        compared += len(got)
        tensors += 1

    check(fails, "the conv biases the dump writes are zero", zero_bad == 0,
          String(zero_bad) + " non-zero")
    check(fails, "every weight is BIT-identical to torchvision's",
          bad == 0 and compared > 11000000,
          String(compared) + " values over " + String(tensors)
          + " tensors compared, " + String(bad) + " differ"
          + ("" if first_bad == "" else "; first " + first_bad))
    if compared == 0:
        raise Error("gate: compared nothing")

    # ── the model walk agrees with the map ────────────────────────────────
    print("")
    var net = NET.make["cpu", Kaiming](None)
    var v = LOADER(SafeTensors(path), resnet18_torch_map(3))
    net.for_each_param["cpu", LOADER](v, None)
    net.for_each_state["cpu", LOADER](v, None)
    check(fails, "the backbone walk is fully covered",
          len(v.loaded) == 100 and len(v.zeroed) == 20
          and len(v.missing) == 0 and len(v.unmapped) == 0
          and v.skipped == 0,
          String(len(v.loaded)) + " loaded, " + String(len(v.zeroed))
          + " zeroed, " + String(len(v.missing)) + " missing, "
          + String(len(v.unmapped)) + " unmapped, " + String(v.skipped)
          + " skipped")
    v.report(String("gate"))

    # ── export, and compare it back against the source file ───────────────
    var sv = SAVER(resnet18_torch_map(3))
    net.for_each_param["cpu", SAVER](sv, None)
    net.for_each_state["cpu", SAVER](sv, None)
    sv.report(String("gate"))
    sv.writer.add_metadata(String("producer"), String("mojo-rl"))
    sv.writer.save(String(EXPORT))

    var ex = SafeTensors(String(EXPORT))
    check(fails, "the export holds only torchvision's 100 tensors",
          ex.size() == 100, String(ex.size()))
    var ebad = 0
    var ecmp = 0
    var eshape = 0
    for i in range(ex.size()):
        var nm = String(ex.names[i])
        if not f.has(nm):
            fails += 1
            print("  FAIL  exported '" + nm + "' is not a torchvision name")
            continue
        if ex.shape_str(nm) != f.shape_str(nm):
            eshape += 1
            continue
        var a = ex.read_f32(nm)
        var bvals = f.read_f32(nm)
        for k in range(len(a)):
            if bitcast[DType.uint32](a[k]) != bitcast[DType.uint32](bvals[k]):
                ebad += 1
        ecmp += len(a)
    check(fails, "the export's shapes are torchvision's", eshape == 0,
          String(eshape) + " differ")
    # ⚠ The whole load -> save path, end to end. The map is used in both
    # directions here, so a transpose applied on one side and not the other
    # shows up as a mismatch rather than cancelling out.
    check(fails, "the export is bit-identical to the source file",
          ebad == 0 and ecmp > 11000000,
          String(ecmp) + " values compared, " + String(ebad) + " differ")

    print("")
    print("  wrote " + EXPORT)
    print("  NOW RUN, or the `load_state_dict` half is untested:")
    print(
        "    pixi run -e act-ref python"
        " tools/nn/dump_safetensors_reference.py --verify-resnet18 " + EXPORT
    )
    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("resnet18 torch-name gate failed")
