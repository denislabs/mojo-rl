# +--------------------------------------------------------------------------+ #
# | ResNet18 backbone — GPU vs CPU
# +--------------------------------------------------------------------------+ #
"""The vision tower's GPU path, on its own.

    pixi run -e apple mojo run -I . tests/nn/test_resnet18_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_resnet18_gpu.mojo

⚠ Deliberately separate from `test_act_gpu_vs_cpu.mojo`. That gate instantiates
the whole ACT graph twice — once per target — and ResNet18 is 20 Conv2D + 20
BatchNorm2D, so including it there meant 80 kernel instantiations on top of
everything else and a build that never completed on CUDA. It runs a two-conv
stub instead, because what it checks (the graph, the optimizer, the host/device
boundary) does not depend on which backbone is attached. This file keeps the
real backbone's GPU coverage, at one model type instead of a whole graph.

Chained, not re-derived: the CPU path is gated against torchvision in
`tests/deep_agents/act/test_act_backbone_vs_reference.mojo`, so CPU-vs-GPU here
covers the whole tower.

The FORWARD runs with BatchNorm in EVAL — on running statistics — because in
training mode each side computes its own batch statistics and the comparison
would measure two reduction orders rather than the convolutions. Weights and
running statistics are copied from the CPU model so the two are the same
function.

The BACKWARD also runs in eval, and covers BatchNorm's affine gradients as well
as the convolutions. Those affine gradients used to be dropped entirely on GPU
in eval mode — this gate is what found it; see
`tests/nn/test_batch_norm_2d_eval_backward_gpu.mojo` for the isolated case.
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.models.resnet18 import (
    RESNET18_OUT_CH,
    ResNet18Backbone,
    ResNet18OutH,
    ResNet18OutW,
)


comptime B = 2
comptime IMG_H = 64
comptime IMG_W = 96
comptime OH = ResNet18OutH[IMG_H]
comptime OW = ResNet18OutW[IMG_W]
comptime IN_N = B * 3 * IMG_H * IMG_W
comptime OUT_N = B * RESNET18_OUT_CH * OH * OW

# ── CPU/GPU parity statistic ─────────────────────────────────────────────
# ⚠ `max|a-b|` compared against `max|a|` is NOT a relative error — the two
# maxima come from different elements. What matters is elementwise
# `|a-b| <= ATOL + RTOL*|a|` (numpy `allclose` semantics), reported as the
# worst ratio so a failure says HOW far out it is.
#
# ⚠⚠ RTOL is set by the HARDWARE, not by taste. NVIDIA runs fp32 matmuls on
# TF32 tensor cores — a 10-bit mantissa, so ~1e-3 relative per matmul, and it
# compounds with depth. Apple has no TF32 and sits at ~1e-7. A tolerance
# calibrated on Metal therefore FAILS on CUDA for a correct kernel; this is
# `feedback_fd_gradcheck_tf32`, which cost three false bug reports before.
# Elementwise ops (BatchNorm alone) stay at ~1e-8 on BOTH — the split between
# "has a matmul" and "does not" is the discriminator.
comptime PARITY_ATOL: Float64 = 1e-5
comptime PARITY_RTOL: Float64 = 2e-2
"""2e-2 covers TF32 compounded through a 20-layer conv stack. It is loose
enough that it can only catch a STRUCTURAL error — a wrong index, a dropped
term, a missing accumulation — which is exactly what a CPU/GPU parity gate is
for. Numerical accuracy against the reference is gated on CPU, in fp32."""


def parity(ref a: List[Scalar[DT]], ref b: List[Scalar[DT]]) raises -> Float64:
    """Worst `|a-b| / (ATOL + RTOL*|a|)`. < 1.0 means every element is within
    tolerance."""
    if len(a) != len(b):
        raise Error(
            "parity: length mismatch " + String(len(a)) + " vs "
            + String(len(b))
        )
    var w = Float64(0.0)
    for i in range(len(a)):
        var x = Float64(a[i])
        var d = abs(x - Float64(b[i]))
        w = max(w, d / (PARITY_ATOL + PARITY_RTOL * abs(x)))
    return w


def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


struct _Collect(ParamVisitor):
    """Snapshot every param — or every GRADIENT — in walk order."""

    var vals: List[Scalar[DT]]
    var grads: Bool
    var names: List[String]
    var starts: List[Int]

    def __init__(out self, grads: Bool = False):
        self.vals = List[Scalar[DT]]()
        self.grads = grads
        self.names = List[String]()
        self.starts = List[Int]()

    def __init__(out self, *, deinit move: Self):
        self.vals = move.vals^
        self.grads = move.grads
        self.names = move.names^
        self.starts = move.starts^

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target != "cpu":
            if self.grads:
                grad.download(ctx.value())
            else:
                param.download(ctx.value())
        self.names.append(String(name))
        self.starts.append(len(self.vals))
        for i in range(N):
            self.vals.append(grad.data[i] if self.grads else param.data[i])


struct _Inject(ParamVisitor):
    var vals: List[Scalar[DT]]
    var pos: Int

    def __init__(out self, var vals: List[Scalar[DT]]):
        self.vals = vals^
        self.pos = 0

    def __init__(out self, *, deinit move: Self):
        self.vals = move.vals^
        self.pos = move.pos

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        param.ensure(N)
        for i in range(N):
            param.data[i] = self.vals[self.pos + i]
        self.pos += N
        comptime if target != "cpu":
            param.upload(ctx.value())


def main() raises:
    var fails = 0
    var ctx = DeviceContext()
    print("ResNet18 GPU-vs-CPU gate")
    print("  device: " + String(ctx.name()))
    print(
        "  " + String(IMG_H) + "x" + String(IMG_W) + " -> "
        + String(RESNET18_OUT_CH) + "x" + String(OH) + "x" + String(OW)
    )
    print("")

    var nc = ResNet18Backbone[3, IMG_H, IMG_W].make["cpu", Kaiming]()
    var ng = ResNet18Backbone[3, IMG_H, IMG_W].make["gpu", Kaiming](ctx)
    nc.set_attr["training"](Scalar[DT](0.0))
    ng.set_attr["training"](Scalar[DT](0.0))

    # Two independent Kaiming inits do NOT agree — copy the CPU model's.
    var wp = _Collect()
    nc.for_each_param["cpu"](wp, None, String(""))
    var wi = _Inject(wp.vals.copy())
    ng.for_each_param["gpu"](wi, ctx, String(""))
    var sp = _Collect()
    nc.for_each_state["cpu"](sp, None, String(""))
    var si = _Inject(sp.vals.copy())
    ng.for_each_state["gpu"](si, ctx, String(""))
    check(
        fails,
        "weights + BatchNorm statistics transferred",
        wi.pos > 0 and si.pos > 0,
        String(wi.pos) + " params, " + String(si.pos) + " state",
    )

    var pc = TensorPack[1]()
    var pg = TensorPack[1]()
    pc[0].ensure(IN_N)
    pg[0].ensure(IN_N)
    for i in range(IN_N):
        var v = Scalar[DT](0.05 * Float64(i % 23) - 0.5)
        pc[0].data[i] = v
        pg[0].data[i] = v
    pg[0].upload(ctx)

    var oc = Tensor()
    var og = Tensor()
    nc.forward["cpu", B](TensorRefs[1, MutAnyOrigin](pc[0]), oc)
    ng.forward["gpu", B](TensorRefs[1, MutAnyOrigin](pg[0]), og, ctx)
    ctx.synchronize()
    og.download(ctx)

    var lc = List[Scalar[DT]]()
    var lg = List[Scalar[DT]]()
    var mag = Float64(0.0)
    for i in range(OUT_N):
        lc.append(oc.data[i])
        lg.append(og.data[i])
        mag = max(mag, abs(Float64(oc.data[i])))
    var w = parity(lc, lg)
    check(
        fails,
        "layer4 output",
        w < 1.0,
        "worst |d|/(atol+rtol|a|) = " + String(w),
    )
    # A dead output would satisfy the comparison and mean nothing.
    check(
        fails,
        "the output is non-trivial",
        mag > 0.05,
        "max|cpu| = " + String(mag),
    )

    # ── backward, in EVAL (see the header) ───────────────────────────────
    var gc = Tensor()
    var gg = Tensor()
    gc.ensure(OUT_N)
    gg.ensure(OUT_N)
    for i in range(OUT_N):
        var v = Scalar[DT](0.01 * Float64((i * 7) % 13) - 0.06)
        gc.data[i] = v
        gg.data[i] = v
    gg.upload(ctx)

    var ggc = TensorPack[1]()
    var ggg = TensorPack[1]()
    ggc[0].ensure(IN_N)
    ggg[0].ensure_gpu(ctx, IN_N)
    nc.zero_grad["cpu"](None)
    ng.zero_grad["gpu"](ctx)
    nc.vjp["cpu", B](
        TensorRefs[1, MutAnyOrigin](pc[0]), gc,
        TensorRefs[1, MutAnyOrigin](ggc[0]),
    )
    ng.vjp["gpu", B](
        TensorRefs[1, MutAnyOrigin](pg[0]), gg,
        TensorRefs[1, MutAnyOrigin](ggg[0]), ctx,
    )
    ctx.synchronize()
    ggg[0].download(ctx)

    var gic = List[Scalar[DT]]()
    var gig = List[Scalar[DT]]()
    var gmag = Float64(0.0)
    for i in range(IN_N):
        gic.append(ggc[0].data[i])
        gig.append(ggg[0].data[i])
        gmag = max(gmag, abs(Float64(ggc[0].data[i])))
    var gw = parity(gic, gig)
    check(
        fails,
        "grad_input",
        gw < 1.0,
        "worst |d|/(atol+rtol|a|) = " + String(gw),
    )
    check(
        fails,
        "the input gradient is non-trivial",
        gmag > 1e-6,
        "max|cpu| = " + String(gmag),
    )

    # The PARAMETER gradients matter more than grad_input — that is where the
    # 40 layers actually learn, and a wrong conv/BN backward kernel shows up
    # here while grad_input can still look plausible.
    var pgc = _Collect(grads=True)
    nc.for_each_param["cpu"](pgc, None, String(""))
    var pgg = _Collect(grads=True)
    ng.for_each_param["gpu"](pgg, ctx, String(""))

    var pw = Float64(0.0)
    var pmag = Float64(0.0)
    var n_nonzero = 0
    if len(pgc.vals) != len(pgg.vals):
        raise Error(
            "gate: param walks differ — " + String(len(pgc.vals)) + " vs "
            + String(len(pgg.vals))
        )
    for i in range(len(pgc.vals)):
        var a = Float64(pgc.vals[i])
        pw = max(pw, abs(a - Float64(pgg.vals[i])))
        pmag = max(pmag, abs(a))
        if a != 0.0:
            n_nonzero += 1
    # Which parameter carries the disagreement? A per-layer breakdown is the
    # difference between "the backward is wrong" and knowing WHICH backward.
    var worst_name = String("(none)")
    var worst_val = Float64(0.0)
    for k in range(len(pgc.names)):
        var lo = pgc.starts[k]
        var hi = pgc.starts[k + 1] if k + 1 < len(pgc.starts) else len(
            pgc.vals
        )
        var d = Float64(0.0)
        for i in range(lo, hi):
            var x = Float64(pgc.vals[i])
            d = max(
                d,
                abs(x - Float64(pgg.vals[i]))
                / (PARITY_ATOL + PARITY_RTOL * abs(x)),
            )
        if d > worst_val:
            worst_val = d
            worst_name = pgc.names[k]
    print("    worst parameter: " + worst_name + "  " + String(worst_val))

    # Split conv from BatchNorm affine, because they used to behave differently
    # (the GPU dropped the affine gradients in eval) and a single aggregate
    # number would not have shown which. Both are now checked.
    var conv_w = Float64(0.0)
    var conv_mag = Float64(0.0)
    var n_conv = 0
    var bn_dropped = 0
    var bn_w = Float64(0.0)
    var bn_mag = Float64(0.0)
    for k in range(len(pgc.names)):
        var nm = pgc.names[k]
        var is_bn = nm.endswith(".gamma") or nm.endswith(".beta")
        var lo = pgc.starts[k]
        var hi = pgc.starts[k + 1] if k + 1 < len(pgc.starts) else len(
            pgc.vals
        )
        if is_bn:
            for i in range(lo, hi):
                var xb = Float64(pgc.vals[i])
                bn_w = max(
                    bn_w,
                    abs(xb - Float64(pgg.vals[i]))
                    / (PARITY_ATOL + PARITY_RTOL * abs(xb)),
                )
                bn_mag = max(bn_mag, abs(xb))
            bn_dropped += 1
            continue
        n_conv += 1
        for i in range(lo, hi):
            var x = Float64(pgc.vals[i])
            var d = abs(x - Float64(pgg.vals[i]))
            conv_w = max(conv_w, d / (PARITY_ATOL + PARITY_RTOL * abs(x)))
            conv_mag = max(conv_mag, abs(x))
    check(
        fails,
        "convolution parameter gradients (" + String(n_conv) + " tensors)",
        conv_w < 1.0,
        "worst |d|/(atol+rtol|a|) = " + String(conv_w),
    )
    check(
        fails,
        "the convolution gradients are non-trivial",
        conv_mag > 1e-6,
        "max|cpu| = " + String(conv_mag),
    )
    check(
        fails,
        "BatchNorm affine gradients in eval ("
        + String(bn_dropped) + " tensors)",
        bn_w < 1.0,
        "worst |d|/(atol+rtol|a|) = " + String(bn_w),
    )
    # These were EXACTLY ZERO on GPU before the fix, so a non-trivial check is
    # the one that matters: agreement on two zeros would prove nothing.
    check(
        fails,
        "the BatchNorm affine gradients are non-trivial",
        bn_mag > 1e-6,
        "max|cpu| = " + String(bn_mag),
    )

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("resnet18 GPU gate failed")
