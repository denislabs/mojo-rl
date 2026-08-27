# +--------------------------------------------------------------------------+ #
# | M3 gate — ResNet18 backbone + both position tables
# +--------------------------------------------------------------------------+ #
"""Gates `nn/models/resnet18.mojo` against **torchvision's** `resnet18`, and
`nn/primitives/sinusoidal_pos_tokens.mojo` against the ACT reference's own
`get_sinusoid_encoding_table` / `PositionEmbeddingSine`.

    pixi run -e act-ref python tools/act/dump_act_reference.py --out /tmp/act_ref
    pixi run mojo build -I . -Xlinker -ld_classic -o /tmp/t \\
        tests/deep_agents/act/test_act_backbone_vs_reference.mojo && /tmp/t

⚠ **`-Xlinker -ld_classic` is required.** The fully-expanded `ResNet18Backbone`
alias mangles to a symbol longer than Apple's new linker accepts
(`ld: Assertion failed: (name.size() <= maxLength)`). The source is healthy —
this is the same toolchain limit recorded in
`feedback_mojo_build_failure_can_be_link_only`, reached here through Sequential
nesting depth rather than through an embedded MJCF string. `mojo run` JITs past
it and never invokes ld.

BatchNorm runs in EVAL on both sides, so the comparison is over the running
statistics (freshly initialized: mean 0, var 1) rather than over batch
statistics, which would make the result depend on the batch composition. Weights
AND running stats are loaded by name — `for_each_param` and `for_each_state` are
separate walks and loading only the first would leave BN's statistics at their
init while every weight matched, which reads as a small numerical disagreement.
"""

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
from mojo_rl.nn.primitives.sinusoidal_pos_tokens import (
    SinusoidalPos1DTokens,
    SinusoidalPos2DTokens,
)
from mojo_rl.deep_agents.act.refload import ListParams, LoadRefParams, RefDump


comptime REF_DIR = "/tmp/act_ref"

# Must match `dump_act_reference.py:section_resnet` / `section_pos`.
comptime B = 2
comptime IMG_H = 64
comptime IMG_W = 96
comptime OH = ResNet18OutH[IMG_H]
comptime OW = ResNet18OutW[IMG_W]

comptime POS1D_SEQ = 9
comptime POS1D_DIM = 16
comptime POS2D_OH = 3
comptime POS2D_OW = 5
comptime POS2D_DIM = 8

comptime TOL_POS = 1e-6
comptime TOL_RN = 2e-4


def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


def worst(mut t: Tensor, ref b: List[Scalar[DT]], n: Int) -> Float64:
    var w = Float64(0.0)
    for i in range(n):
        w = max(w, abs(Float64(t.data[i]) - Float64(b[i])))
    return w


def main() raises:
    var fails = 0
    print("ResNet18 + position-table gate (reference: " + String(REF_DIR) + ")")
    print("")

    var d = RefDump(String(REF_DIR))

    # ── 1. ACT's 1-D sinusoid table ──────────────────────────────────────
    comptime N1 = POS1D_SEQ * POS1D_DIM
    var p1 = SinusoidalPos1DTokens[4, POS1D_SEQ, POS1D_DIM].make[
        "cpu", Kaiming
    ]()
    var carrier = TensorPack[1]()
    carrier[0].ensure(B * 4)
    for i in range(B * 4):
        carrier[0].data[i] = Scalar[DT](0.0)
    var o1 = Tensor()
    p1.forward["cpu", B](
        TensorRefs[1, MutAnyOrigin](carrier[0]), o1
    )
    var r1 = d.get(String("pos1d_table"))
    # The node broadcasts the table to every batch row; compare row 0, then
    # confirm the rows are identical (a per-row table would be a different bug).
    var w1 = worst(o1, r1, N1)
    check(
        fails,
        "1-D table vs get_sinusoid_encoding_table",
        w1 < TOL_POS,
        "max|diff| = " + String(w1),
    )
    var row_spread = Float64(0.0)
    for i in range(N1):
        row_spread = max(
            row_spread, abs(Float64(o1.data[N1 + i]) - Float64(o1.data[i]))
        )
    check(fails, "1-D table is identical across the batch", row_spread == 0.0)

    # The table must not be the separable `SinusoidalPosAdd[T, 1, D]` — that one
    # carries a constant +1 on every odd index. Position 0 of ACT's table is
    # sin(0)=0 / cos(0)=1, so check a NON-zero position instead, where the two
    # actually differ.
    var nondegenerate = False
    for j in range(POS1D_DIM):
        if abs(Float64(o1.data[3 * POS1D_DIM + j])) > 1e-6:
            nondegenerate = True
    check(fails, "1-D table row 3 is non-zero", nondegenerate)

    # ── 2. DETR's 2-D sine table ─────────────────────────────────────────
    comptime N2 = POS2D_OH * POS2D_OW * POS2D_DIM
    var p2 = SinusoidalPos2DTokens[4, POS2D_DIM, POS2D_OH, POS2D_OW].make[
        "cpu", Kaiming
    ]()
    var o2 = Tensor()
    p2.forward["cpu", B](
        TensorRefs[1, MutAnyOrigin](carrier[0]), o2
    )
    var r2 = d.get(String("pos2d_table"))
    var w2 = worst(o2, r2, N2)
    check(
        fails,
        "2-D table vs PositionEmbeddingSine(normalize=True)",
        w2 < TOL_POS,
        "max|diff| = " + String(w2),
    )

    # Row half vs column half must actually differ per axis: token (0,1) and
    # token (1,0) share nothing if y-before-x is right, and would be swapped if
    # the concatenation order were reversed.
    var tok01 = 0 * POS2D_OW + 1
    var tok10 = 1 * POS2D_OW + 0
    var half = POS2D_DIM // 2
    var y_same = Float64(0.0)
    for j in range(half):
        y_same = max(
            y_same,
            abs(
                Float64(o2.data[tok01 * POS2D_DIM + j])
                - Float64(o2.data[0 * POS2D_DIM + j])
            ),
        )
    check(
        fails,
        "2-D table: the ROW half is constant along a row (y-before-x order)",
        y_same < 1e-6,
        "max|y(0,1) - y(0,0)| = " + String(y_same),
    )
    var y_moves = Float64(0.0)
    for j in range(half):
        y_moves = max(
            y_moves,
            abs(
                Float64(o2.data[tok10 * POS2D_DIM + j])
                - Float64(o2.data[0 * POS2D_DIM + j])
            ),
        )
    check(
        fails,
        "2-D table: the ROW half changes down a column",
        y_moves > 1e-3,
        "max|y(1,0) - y(0,0)| = " + String(y_moves),
    )

    # ── 3. ResNet18 through layer4 ───────────────────────────────────────
    print("")
    print(
        "  geometry: " + String(IMG_H) + "x" + String(IMG_W) + " -> "
        + String(RESNET18_OUT_CH) + "x" + String(OH) + "x" + String(OW)
    )
    var net = ResNet18Backbone[3, IMG_H, IMG_W].make["cpu", Kaiming]()
    net.set_attr["training"](Scalar[DT](0.0))  # BN eval -> running stats

    var wl = LoadRefParams["rn18."](RefDump(String(REF_DIR)))
    net.for_each_param["cpu"](wl, None, String(""))
    var pl = ListParams()
    net.for_each_param["cpu"](pl, None, String(""))
    var w_ok = len(wl.missing) == 0 and len(wl.loaded) == len(pl.names)
    check(
        fails,
        "every weight loaded from torchvision",
        w_ok,
        String(len(wl.loaded)) + "/" + String(len(pl.names))
        + (", first missing: " + wl.missing[0] if len(wl.missing) > 0 else ""),
    )

    var sl = LoadRefParams["rn18."](RefDump(String(REF_DIR)))
    net.for_each_state["cpu"](sl, None, String(""))
    var slist = ListParams()
    net.for_each_state["cpu"](slist, None, String(""))
    var s_ok = len(sl.missing) == 0 and len(sl.loaded) == len(slist.names)
    check(
        fails,
        "every BN running statistic loaded",
        s_ok,
        String(len(sl.loaded)) + "/" + String(len(slist.names))
        + (", first missing: " + sl.missing[0] if len(sl.missing) > 0 else ""),
    )

    comptime IN_N = B * 3 * IMG_H * IMG_W
    comptime OUT_N = B * RESNET18_OUT_CH * OH * OW
    var xp = TensorPack[1]()
    var xr = d.get(String("rn18_x"))
    xp[0].ensure(IN_N)
    for i in range(IN_N):
        xp[0].data[i] = xr[i]
    var yo = Tensor()
    net.forward["cpu", B](TensorRefs[1, MutAnyOrigin](xp[0]), yo)

    var yr = d.get(String("rn18_out"))
    var wr = worst(yo, yr, OUT_N)
    # Tolerance is looser than the layer gates': 20 convolutions and 20
    # BatchNorms of fp32 accumulation, against a reference that ran in fp32 too
    # but with different reduction orders inside cuDNN/oneDNN.
    check(
        fails,
        "ResNet18 layer4 output vs torchvision",
        wr < TOL_RN,
        "max|diff| = " + String(wr),
    )

    # A near-zero output would pass the comparison and mean nothing.
    var mag = Float64(0.0)
    for i in range(OUT_N):
        mag = max(mag, abs(Float64(yr[i])))
    check(
        fails,
        "the reference output is non-trivial",
        mag > 0.1,
        "max|ref| = " + String(mag),
    )

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("act backbone gate failed")
