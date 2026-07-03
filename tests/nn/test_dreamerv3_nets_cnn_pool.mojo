"""Phase B-2 reference-geometry CNN gates (pool encoder / bspace decoder).

1. Upsample2x: forward = exact nearest-neighbor map; vjp = exact 2×2 grad sum.
2. DreamerDecoderStem: analytic input-gradient vs central finite differences
   (the hand-written split/branches/add/merge plumbing is the risk surface;
   the child modules are already gated elsewhere).
3. DreamerEncoderCNNPool / DreamerDecoderCNNPool: forward shapes + vjp runs +
   nonzero param grads at tiny dims (CPU).

Run: pixi run mojo run -I . tests/nn/test_dreamerv3_nets_cnn_pool.mojo
"""

from std.math import isfinite
from std.random import seed, random_float64
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.initializer import Kaiming, TruncNormalIn
from mojo_rl.nn.primitives.upsample2x import Upsample2x
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents.dreamerv3.nets_cnn import (
    DreamerDecoderStem,
    DreamerEncoderCNNPool,
    DreamerDecoderCNNPool,
)
from std.gpu.host import DeviceContext


struct _GradAbsSum(ParamVisitor):
    var total: Scalar[DT]

    def __init__(out self):
        self.total = Scalar[DT](0.0)

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            if grad.n >= N:
                grad.download(ctx.value())
        for i in range(min(grad.n, N)):
            var g = grad.data[i]
            self.total += g if g >= Scalar[DT](0.0) else -g


def test_upsample2x() raises:
    print("test_upsample2x ...")
    comptime C = 2
    comptime H = 3
    comptime W = 2
    comptime B = 2
    var up = Upsample2x[C, H, W].make["cpu", Kaiming](None)
    var x = Tensor.alloc(B * C * H * W)
    for i in range(B * C * H * W):
        x.data[i] = Scalar[DT](i) * 0.25 - 1.0
    var out = Tensor()
    up.forward["cpu", B](TensorRefs[1](x), out, None)
    # forward exactness
    for b in range(B):
        for c in range(C):
            for h in range(2 * H):
                for w in range(2 * W):
                    var got = out.data[
                        (b * C + c) * (4 * H * W) + h * (2 * W) + w
                    ]
                    var want = x.data[
                        (b * C + c) * (H * W) + (h // 2) * W + (w // 2)
                    ]
                    assert_true(got == want, "up2x forward exact")
    # vjp exactness: gin = sum of the 2×2 cell
    var go = Tensor.alloc(B * C * 4 * H * W)
    for i in range(B * C * 4 * H * W):
        go.data[i] = Scalar[DT]((i % 5)) * 0.1
    var gi = Tensor()
    up.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), None)
    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    var base = (b * C + c) * (4 * H * W)
                    var want = (
                        go.data[base + (2 * h) * (2 * W) + 2 * w]
                        + go.data[base + (2 * h) * (2 * W) + 2 * w + 1]
                        + go.data[base + (2 * h + 1) * (2 * W) + 2 * w]
                        + go.data[base + (2 * h + 1) * (2 * W) + 2 * w + 1]
                    )
                    var got = gi.data[(b * C + c) * (H * W) + h * W + w]
                    assert_true(got == want, "up2x vjp exact")
    print("  ok")


def test_stem_fd() raises:
    print("test_stem_fd (input-grad vs central differences) ...")
    comptime DETER = 16
    comptime SC = 8
    comptime UNITS = 6
    comptime U = 16
    comptime B = 3
    comptime F = DETER + SC
    var stem = DreamerDecoderStem[DETER, SC, UNITS, U, SwishOp].make[
        "cpu", TruncNormalIn
    ](None)

    var feat = Tensor.alloc(B * F)
    for i in range(B * F):
        feat.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    # fixed random cotangent: L = Σ out ⊙ cot
    var cot = Tensor.alloc(B * U)
    for i in range(B * U):
        cot.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)

    var out = Tensor()
    stem.forward["cpu", B](TensorRefs[1](feat), out, None)
    var gin = Tensor()
    stem.vjp["cpu", B](TensorRefs[1](feat), cot, TensorRefs[1](gin), None)

    # central differences on a spread of coordinates
    var eps = Scalar[DT](1e-2)
    var checked = 0
    var i = 0
    while i < B * F:
        var orig = feat.data[i]
        feat.data[i] = orig + eps
        var op = Tensor()
        stem.forward["cpu", B](TensorRefs[1](feat), op, None)
        feat.data[i] = orig - eps
        var om = Tensor()
        stem.forward["cpu", B](TensorRefs[1](feat), om, None)
        feat.data[i] = orig
        var lp = Scalar[DT](0.0)
        var lm = Scalar[DT](0.0)
        for j in range(B * U):
            lp += op.data[j] * cot.data[j]
            lm += om.data[j] * cot.data[j]
        var fd = (lp - lm) / (2.0 * eps)
        var an = gin.data[i]
        var d = fd - an
        d = d if d >= Scalar[DT](0.0) else -d
        var mag = fd if fd >= Scalar[DT](0.0) else -fd
        assert_true(
            d <= Scalar[DT](2e-2) * (mag + Scalar[DT](0.1)),
            "stem FD mismatch at " + String(i) + ": fd=" + String(fd)
            + " an=" + String(an),
        )
        checked += 1
        i += 5  # every 5th coordinate (both deter + stoch ranges covered)
    print("  ok (", checked, "coords )")


def test_enc_dec_pool() raises:
    print("test_enc_dec_pool (shapes + vjp + nonzero param grads) ...")
    comptime C = 1
    comptime IMG = 16
    comptime BASE = 4
    comptime DETER = 16
    comptime SC = 8
    comptime UNITS = 6
    comptime FEATIN = DETER + SC
    comptime B = 2
    comptime TOKEN = 4 * BASE * (IMG // 16) * (IMG // 16)  # 16
    comptime Enc = DreamerEncoderCNNPool[C, IMG, IMG, BASE, SwishOp]
    comptime Dec = DreamerDecoderCNNPool[
        FEATIN, DETER, C, IMG, IMG, BASE, UNITS, SwishOp
    ]
    var enc = Enc.make["cpu", TruncNormalIn](None)
    var dec = Dec.make["cpu", TruncNormalIn](None)

    var img = Tensor.alloc(B * C * IMG * IMG)
    for i in range(B * C * IMG * IMG):
        img.data[i] = Scalar[DT](random_float64())
    var tok = Tensor()
    enc.forward["cpu", B](TensorRefs[1](img), tok, None)
    assert_true(tok.n == B * TOKEN, "encoder token width")
    for i in range(B * TOKEN):
        assert_true(isfinite(Float64(tok.data[i])), "encoder finite")

    var feat = Tensor.alloc(B * FEATIN)
    for i in range(B * FEATIN):
        feat.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    var rec = Tensor()
    dec.forward["cpu", B](TensorRefs[1](feat), rec, None)
    assert_true(rec.n == B * C * IMG * IMG, "decoder output size")
    for i in range(B * C * IMG * IMG):
        assert_true(isfinite(Float64(rec.data[i])), "decoder finite")

    # vjp both + nonzero param grads
    var gtok = Tensor.alloc(B * TOKEN)
    for i in range(B * TOKEN):
        gtok.data[i] = Scalar[DT](random_float64() - 0.5)
    var gimg = Tensor()
    enc.zero_grad["cpu"](None)
    enc.vjp["cpu", B](TensorRefs[1](img), gtok, TensorRefs[1](gimg), None)
    var ce = _GradAbsSum()
    enc.for_each_param["cpu"](ce, None)
    assert_true(ce.total > Scalar[DT](0.0), "encoder param grads nonzero")

    var grec = Tensor.alloc(B * C * IMG * IMG)
    for i in range(B * C * IMG * IMG):
        grec.data[i] = Scalar[DT](random_float64() - 0.5)
    var gfeat = Tensor()
    dec.zero_grad["cpu"](None)
    dec.vjp["cpu", B](TensorRefs[1](feat), grec, TensorRefs[1](gfeat), None)
    var cd = _GradAbsSum()
    dec.for_each_param["cpu"](cd, None)
    assert_true(cd.total > Scalar[DT](0.0), "decoder param grads nonzero")
    var gsum = Scalar[DT](0.0)
    for i in range(B * FEATIN):
        var g = gfeat.data[i]
        gsum += g if g >= Scalar[DT](0.0) else -g
    assert_true(gsum > Scalar[DT](0.0), "decoder input grads nonzero")
    print("  ok")


def main() raises:
    seed(11)
    print("DreamerV3 reference-geometry CNN (Phase B-2) gates")
    test_upsample2x()
    test_stem_fd()
    test_enc_dec_pool()
    print("NETS CNN POOL OK")
