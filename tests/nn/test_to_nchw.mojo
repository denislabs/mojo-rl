"""ToNCHW channels-last→first latent adapter — correctness (CPU + GPU).

NCHW LAYOUT = pure identity copy (passthrough). NHWC LAYOUT = [H,W,C]→[C,H,W]
transpose; its vjp is the inverse scatter. Checks: NCHW forward/vjp are identity;
NHWC forward matches a hand transpose and round-trips through vjp; GPU == CPU.
"""

from std.math import abs
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT, LAYOUT_NCHW, LAYOUT_NHWC
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.to_nchw import ToNCHW

comptime C = 4
comptime H = 3
comptime W = 3
comptime HW = H * W
comptime DIM = C * HW
comptime B = 2


def test_nchw_identity() raises:
    var m = ToNCHW[C, H, W].make["cpu", Deterministic]()
    var x = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        x.data[i] = Scalar[DT](i) * 0.1 - 2.0
    var out = Tensor.alloc(B * DIM)
    m.forward["cpu", B](TensorRefs[1](x), out, None)
    var bad = 0
    for i in range(B * DIM):
        if out.data[i] != x.data[i]:
            bad += 1
    assert_true(bad == 0, "NCHW ToNCHW forward is identity")
    print("  NCHW identity: ok")


def test_nhwc_transpose() raises:
    var m = ToNCHW[C, H, W, LAYOUT_NHWC].make["cpu", Deterministic]()
    var x = Tensor.alloc(B * DIM)  # NHWC-laid-out input: [b, hw*C + c]
    for i in range(B * DIM):
        x.data[i] = Scalar[DT](i) * 0.1 - 2.0
    var out = Tensor.alloc(B * DIM)
    m.forward["cpu", B](TensorRefs[1](x), out, None)
    # out must be NCHW [c*HW + hw] == in[hw*C + c]
    var bad = 0
    for b in range(B):
        for c in range(C):
            for hw in range(HW):
                var got = out.data[b * DIM + c * HW + hw]
                var exp = x.data[b * DIM + hw * C + c]
                if got != exp:
                    bad += 1
    assert_true(bad == 0, "NHWC ToNCHW forward = [H,W,C]->[C,H,W]")

    # vjp: round-trip — gin in NHWC layout must equal a grad fed in NCHW.
    var go = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        go.data[i] = Scalar[DT]((i % 7) - 3) * 0.5
    var gin = Tensor.alloc(B * DIM)
    m.zero_grad["cpu"](None)
    m.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gin), None)
    var bad2 = 0
    for b in range(B):
        for c in range(C):
            for hw in range(HW):
                var got = gin.data[b * DIM + hw * C + c]  # NHWC position
                var exp = go.data[b * DIM + c * HW + hw]  # NCHW grad
                if got != exp:
                    bad2 += 1
    assert_true(bad2 == 0, "NHWC ToNCHW vjp = inverse scatter")
    print("  NHWC transpose fwd+vjp: ok")


def test_gpu_vs_cpu_nhwc() raises:
    var c = DeviceContext()
    var mc = ToNCHW[C, H, W, LAYOUT_NHWC].make["cpu", Deterministic]()
    var mg = ToNCHW[C, H, W, LAYOUT_NHWC].make["gpu", Deterministic](Optional(c))
    var x = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        x.data[i] = Scalar[DT](i) * 0.07 - 1.0
    var oc = Tensor.alloc(B * DIM)
    mc.forward["cpu", B](TensorRefs[1](x), oc, None)
    var xg = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        xg.data[i] = x.data[i]
    xg.upload(c)
    var og = Tensor.alloc(B * DIM)
    mg.forward["gpu", B](TensorRefs[1](xg), og, Optional(c))
    og.download(c)
    var bad = 0
    for i in range(B * DIM):
        if abs(Float64(oc.data[i] - og.data[i])) > 1e-7:
            bad += 1
    assert_true(bad == 0, "ToNCHW GPU == CPU (NHWC)")
    print("  GPU==CPU NHWC: ok")


def main() raises:
    print("=" * 60)
    print("ToNCHW latent adapter (NCHW identity / NHWC transpose)")
    print("=" * 60)
    test_nchw_identity()
    test_nhwc_transpose()
    test_gpu_vs_cpu_nhwc()
    print("TO_NCHW GATE PASSED")
