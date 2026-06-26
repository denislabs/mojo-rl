"""MaxPool2D + AvgPool2D NCHW-vs-NHWC layout parity (CPU, exact).

Phase 1 of the channels_last migration: the `LAYOUT` param wires an NHWC path
into both pools (reusing conv2d's _in_off/_out_off/_out_decode/_in_decode). This
gate proves each pool computes the SAME logical result in both layouts — set one
logical (input, grad_output) in both layouts and assert forward + grad_input
agree. CPU is deterministic → exact. (No prior pool golden existed, so this is
also the pools' first parity gate; the conv-stack smoke test guards gross NCHW
breakage.)
"""

from std.math import abs
from std.testing import assert_true

from mojo_rl.nn.constants import DT, LAYOUT_NCHW, LAYOUT_NHWC
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.max_pool_2d import MaxPool2D
from mojo_rl.nn.primitives.avg_pool_2d import AvgPool2D
from mojo_rl.nn.primitives.conv2d import _in_off, _out_off

comptime C = 3
comptime K = 2
comptime S = 2
comptime P = 0
comptime H = 6
comptime W = 6
comptime B = 2
comptime OH = (H + 2 * P - K) // S + 1
comptime OW = (W + 2 * P - K) // S + 1
comptime IN = C * H * W
comptime OUT = C * OH * OW
comptime OSP = OH * OW


def _xl(b: Int, c: Int, ih: Int, iw: Int) -> Scalar[DT]:
    # distinct values → no argmax ties to disambiguate
    return Scalar[DT](((b * C + c) * H + ih) * W + iw) * 0.01 - 3.0


def _gol(b: Int, c: Int, oh: Int, ow: Int) -> Scalar[DT]:
    return Scalar[DT](((((b * C + c) * OH + oh) * OW + ow) % 7) - 3) * 0.3


def _fill[
    LAYOUT: Int
](mut x: Tensor, mut go: Tensor):
    for b in range(B):
        for c in range(C):
            for ih in range(H):
                for iw in range(W):
                    x.data[
                        b * IN + _in_off[LAYOUT, C, H, W](c, ih, iw)
                    ] = _xl(b, c, ih, iw)
            for oh in range(OH):
                for ow in range(OW):
                    go.data[
                        b * OUT + _out_off[LAYOUT, C, OSP](c, oh * OW + ow)
                    ] = _gol(b, c, oh, ow)


def _maxd_out_gi(
    outn: Tensor, outh: Tensor, gin: Tensor, gih: Tensor
) -> Tuple[Float64, Float64]:
    var d_out: Float64 = 0.0
    var d_gi: Float64 = 0.0
    for b in range(B):
        for c in range(C):
            for oh in range(OH):
                for ow in range(OW):
                    var a = outn.data[b * OUT + _out_off[LAYOUT_NCHW, C, OSP](c, oh * OW + ow)]
                    var d = outh.data[b * OUT + _out_off[LAYOUT_NHWC, C, OSP](c, oh * OW + ow)]
                    d_out = max(d_out, abs(Float64(a - d)))
            for ih in range(H):
                for iw in range(W):
                    var a = gin.data[b * IN + _in_off[LAYOUT_NCHW, C, H, W](c, ih, iw)]
                    var d = gih.data[b * IN + _in_off[LAYOUT_NHWC, C, H, W](c, ih, iw)]
                    d_gi = max(d_gi, abs(Float64(a - d)))
    return (d_out, d_gi)


def test_maxpool_parity() raises:
    var mn = MaxPool2D[C, K, S, P, H, W].make["cpu", Deterministic]()
    var xn = Tensor.alloc(B * IN)
    var gon = Tensor.alloc(B * OUT)
    _fill[LAYOUT_NCHW](xn, gon)
    var outn = Tensor.alloc(B * OUT)
    mn.forward["cpu", B](TensorRefs[1](xn), outn, None)
    var gin = Tensor.alloc(B * IN)
    mn.vjp["cpu", B](TensorRefs[1](xn), gon, TensorRefs[1](gin), None)

    var mh = MaxPool2D[C, K, S, P, H, W, LAYOUT_NHWC].make["cpu", Deterministic]()
    var xh = Tensor.alloc(B * IN)
    var goh = Tensor.alloc(B * OUT)
    _fill[LAYOUT_NHWC](xh, goh)
    var outh = Tensor.alloc(B * OUT)
    mh.forward["cpu", B](TensorRefs[1](xh), outh, None)
    var gih = Tensor.alloc(B * IN)
    mh.vjp["cpu", B](TensorRefs[1](xh), goh, TensorRefs[1](gih), None)

    var d = _maxd_out_gi(outn, outh, gin, gih)
    print("  MaxPool NCHW↔NHWC max|Δ|: out", d[0], "grad_x", d[1])
    assert_true(d[0] < 1e-5 and d[1] < 1e-5, "MaxPool2D NCHW vs NHWC parity")


def test_avgpool_parity() raises:
    var mn = AvgPool2D[C, K, S, P, H, W].make["cpu", Deterministic]()
    var xn = Tensor.alloc(B * IN)
    var gon = Tensor.alloc(B * OUT)
    _fill[LAYOUT_NCHW](xn, gon)
    var outn = Tensor.alloc(B * OUT)
    mn.forward["cpu", B](TensorRefs[1](xn), outn, None)
    var gin = Tensor.alloc(B * IN)
    mn.vjp["cpu", B](TensorRefs[1](xn), gon, TensorRefs[1](gin), None)

    var mh = AvgPool2D[C, K, S, P, H, W, DT, LAYOUT_NHWC].make[
        "cpu", Deterministic
    ]()
    var xh = Tensor.alloc(B * IN)
    var goh = Tensor.alloc(B * OUT)
    _fill[LAYOUT_NHWC](xh, goh)
    var outh = Tensor.alloc(B * OUT)
    mh.forward["cpu", B](TensorRefs[1](xh), outh, None)
    var gih = Tensor.alloc(B * IN)
    mh.vjp["cpu", B](TensorRefs[1](xh), goh, TensorRefs[1](gih), None)

    var d = _maxd_out_gi(outn, outh, gin, gih)
    print("  AvgPool NCHW↔NHWC max|Δ|: out", d[0], "grad_x", d[1])
    assert_true(d[0] < 1e-5 and d[1] < 1e-5, "AvgPool2D NCHW vs NHWC parity")


def main() raises:
    print("=" * 60)
    print("MaxPool2D + AvgPool2D NCHW-vs-NHWC layout parity (CPU)")
    print("=" * 60)
    test_maxpool_parity()
    test_avgpool_parity()
    print("POOL NHWC PARITY GATE PASSED")
