# +--------------------------------------------------------------------------+ #
# | ImageNet ResNet18 weights — does our backbone compute torchvision's function?
# +--------------------------------------------------------------------------+ #
"""Gates the pretrained-weight conversion at the resolution it will be used at.

    pixi run -e act-ref python tools/act/dump_resnet18_imagenet.py \\
        --out ~/.cache/mojo_rl/act_so101/resnet18_imagenet
    pixi run mojo run -I . tests/deep_agents/act/test_act_pretrained_backbone.mojo

## What this catches that a load count cannot

`ACTTrainer.load_backbone` reports "120 tensors" whether or not those tensors
landed in the right slots. Every BasicBlock in a ResNet stage holds two convs
of IDENTICAL shape, and `layer2`-`layer4` each hold a downsample conv too, so
a mapping that swapped `conv1` and `conv2` inside a block, or attached a
stage's statistics to the wrong stage, would load cleanly, report full
coverage, and produce a different function. The only check that sees that is
running both networks on the same input.

The M3 gate (`test_act_backbone_vs_reference.mojo`) already compares our
ResNet18 against torchvision — on RANDOM weights at 64x96. This one exists for
the two things that differ in the pretrained path and nowhere else:

  * **The BatchNorm running statistics carry information.** At random init they
    are exactly mean 0 / var 1, so in eval mode BN is near-identity and a gate
    that dropped them entirely would still pass. ImageNet statistics are not
    (`bn1.running_var` mean 3.02), so here they are load-bearing.
  * **240x320, the training resolution.** 64x96 exercises the same layers but
    a different feature-map geometry (2x3 vs 8x10). A stride or padding mistake
    that happens to be invisible at one is not at the other.
"""

from std.python import Python, PythonObject

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.models.resnet18 import (
    RESNET18_OUT_CH,
    ResNet18Backbone,
    ResNet18OutH,
    ResNet18OutW,
)
from mojo_rl.deep_agents.act.config import (
    SO101_IMG_H,
    SO101_IMG_W,
)
from mojo_rl.deep_agents.act.refload import (
    ListParams,
    LoadRefParams,
    RefDump,
)


comptime IMG_H = SO101_IMG_H
comptime IMG_W = SO101_IMG_W
comptime OH = ResNet18OutH[IMG_H]
comptime OW = ResNet18OutW[IMG_W]
comptime B = 2

comptime IN_N = B * 3 * IMG_H * IMG_W
comptime OUT_N = B * RESNET18_OUT_CH * OH * OW

comptime TOL = 1e-4
"""MEASURED, not guessed: 20 convolutions and 20 BatchNorms of fp32
accumulation against oneDNN's different reduction orders agree to **9.1e-6** on
a signal whose maximum is 9.1 — parts per million. 1e-4 is 11x that.

It was written as 2e-3 first, by analogy with the M3 gate's random-weight
comparison, which would have accepted a 200x degradation without comment. A
tolerance chosen by analogy is a tolerance nobody has measured."""


def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


def dump_dir() raises -> String:
    var os = Python.import_module("os")
    var env = String(
        os.environ.get(PythonObject("ACT_PRETRAINED"), PythonObject(""))
    )
    if env.byte_length() > 0:
        return env
    var home = String(os.path.expanduser(PythonObject("~")))
    return home + "/.cache/mojo_rl/act_so101/resnet18_imagenet"


def main() raises:
    var fails = 0
    var d_dir = dump_dir()
    var os = Python.import_module("os")
    if not Bool(os.path.exists(PythonObject(d_dir + "/manifest.txt"))):
        print("MISSING DUMP: " + d_dir)
        print(
            "  pixi run -e act-ref python tools/act/dump_resnet18_imagenet.py"
        )
        raise Error("pretrained dump not found")

    print("ImageNet ResNet18 backbone gate")
    print("  dump      " + d_dir)
    print(
        "  geometry  " + String(IMG_H) + "x" + String(IMG_W) + " -> "
        + String(RESNET18_OUT_CH) + "x" + String(OH) + "x" + String(OW)
    )
    print("")

    var net = ResNet18Backbone[3, IMG_H, IMG_W].make["cpu", Kaiming]()
    net.set_attr["training"](Scalar[DT](0.0))  # BN eval -> running statistics

    var wl = LoadRefParams["rn18in."](RefDump(String(d_dir)))
    net.for_each_param["cpu"](wl, None, String(""))
    var pl = ListParams()
    net.for_each_param["cpu"](pl, None, String(""))
    check(
        fails,
        "every weight loaded",
        len(wl.missing) == 0 and len(wl.loaded) == len(pl.names),
        String(len(wl.loaded)) + "/" + String(len(pl.names))
        + (", first missing: " + wl.missing[0] if len(wl.missing) > 0 else ""),
    )

    var sl = LoadRefParams["rn18in."](RefDump(String(d_dir)))
    net.for_each_state["cpu"](sl, None, String(""))
    var slist = ListParams()
    net.for_each_state["cpu"](slist, None, String(""))
    check(
        fails,
        "every BN running statistic loaded",
        len(sl.missing) == 0 and len(sl.loaded) == len(slist.names),
        String(len(sl.loaded)) + "/" + String(len(slist.names))
        + (", first missing: " + sl.missing[0] if len(sl.missing) > 0 else ""),
    )

    # The statistics must be TRAINED ones. At init `running_var` is exactly 1
    # and `running_mean` exactly 0, which in eval makes BatchNorm an identity —
    # so a dump of an untrained network would satisfy every check above and the
    # forward comparison too, while carrying no ImageNet information at all.
    var d = RefDump(String(d_dir))
    var rv = d.get(String("rn18in.0.0.1.running_var"))
    var rv_off = Float64(0.0)
    for i in range(len(rv)):
        rv_off = max(rv_off, abs(Float64(rv[i]) - 1.0))
    check(
        fails,
        "the running statistics are TRAINED, not init values",
        rv_off > 0.1,
        "max|running_var - 1| = " + String(rv_off),
    )

    # ── the forward ──────────────────────────────────────────────────────
    var xp = TensorPack[1]()
    var xr = d.get(String("rn18in_x"))
    if len(xr) != IN_N:
        raise Error(
            "gate: dump input is " + String(len(xr)) + " values, expected "
            + String(IN_N) + " — the dump was written at another resolution"
        )
    xp[0].ensure(IN_N)
    for i in range(IN_N):
        xp[0].data[i] = xr[i]

    var yo = Tensor()
    net.forward["cpu", B](TensorRefs[1, MutAnyOrigin](xp[0]), yo)

    var yr = d.get(String("rn18in_out"))
    var worst = Float64(0.0)
    var mag = Float64(0.0)
    for i in range(OUT_N):
        worst = max(worst, abs(Float64(yo.data[i]) - Float64(yr[i])))
        mag = max(mag, abs(Float64(yr[i])))
    check(
        fails,
        "layer4 output vs torchvision, ImageNet weights",
        worst < TOL,
        "max|diff| = " + String(worst),
    )
    check(
        fails,
        "the reference output is non-trivial",
        mag > 0.1,
        "max|ref| = " + String(mag),
    )

    # A ResNet stage holds two convolutions of identical shape per block, so a
    # swapped pair loads cleanly. That is what the forward above is for — this
    # states the margin it passed by, so a future loosening has to argue with a
    # number.
    check(
        fails,
        "the comparison has margin (diff is a small fraction of the signal)",
        worst < 0.01 * mag,
        String(worst) + " vs " + String(0.01 * mag),
    )

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("pretrained backbone gate failed")
