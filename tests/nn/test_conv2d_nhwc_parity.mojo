"""Conv2D NCHW-vs-NHWC layout parity (CPU, exact).

Phase 1 of the channels_last migration: the `LAYOUT` param wires an NHWC code
path into Conv2D behind `comptime if LAYOUT`. This gate proves that path computes
the SAME logical convolution as NCHW — set up one logical (input, weight) in both
layouts (permuted via the shared offset helpers) and assert forward + every
gradient agree. CPU is deterministic, so the match is exact (fp32 reduction-order
only); a mismatch means the index math in one layout is wrong.

NCHW is the default and is covered bit-identically by test_conv2d_bf16_flow; this
file only exercises the new NHWC path against it.
"""

from std.math import abs
from std.testing import assert_true

from mojo_rl.nn.constants import DT, LAYOUT_NCHW, LAYOUT_NHWC
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.conv2d import Conv2D, _in_off, _col_off, _out_off

comptime IC = 3
comptime OC = 4
comptime K = 3
comptime S = 1
comptime P = 1
comptime H = 5
comptime W = 5
comptime B = 2

comptime CN = Conv2D[IC, OC, K, S, P, H, W]                  # NCHW (default)
comptime CH = Conv2D[IC, OC, K, S, P, H, W, DT, LAYOUT_NHWC]  # NHWC
comptime IN = CN.IN_FLAT
comptime OUT = CN.OUT_FLAT
comptime COL = CN.COL
comptime SO = CN.SO


# Distinct logical values so any wrong gather/scatter shows up as a mismatch.
def _wl(oc: Int, ic: Int, kh: Int, kw: Int) -> Scalar[DT]:
    return Scalar[DT](((((oc * IC + ic) * K + kh) * K + kw) % 13) - 6) * 0.1


def _xl(b: Int, ic: Int, ih: Int, iw: Int) -> Scalar[DT]:
    return Scalar[DT](((((b * IC + ic) * H + ih) * W + iw) % 11) - 5) * 0.1


def _gl(b: Int, oc: Int, oh: Int, ow: Int) -> Scalar[DT]:
    return Scalar[DT](((((b * OC + oc) * H + oh) * W + ow) % 7) - 3) * 0.2


def _setup[
    LAYOUT: Int
](mut m: Conv2D[IC, OC, K, S, P, H, W, DT, LAYOUT], mut x: Tensor, mut go: Tensor):
    """Write the shared logical weight / input / grad_output into `m`'s LAYOUT."""
    for oc in range(OC):
        for ic in range(IC):
            for kh in range(K):
                for kw in range(K):
                    m.weight.val.data[
                        oc * COL + _col_off[LAYOUT, IC, K](ic, kh, kw)
                    ] = _wl(oc, ic, kh, kw)
    for b in range(B):
        for ic in range(IC):
            for ih in range(H):
                for iw in range(W):
                    x.data[
                        b * IN + _in_off[LAYOUT, IC, H, W](ic, ih, iw)
                    ] = _xl(b, ic, ih, iw)
    for b in range(B):
        for oc in range(OC):
            for oh in range(CN.OH):
                for ow in range(CN.OW):
                    go.data[
                        b * OUT
                        + _out_off[LAYOUT, OC, SO](oc, oh * CN.OW + ow)
                    ] = _gl(b, oc, oh, ow)


def test_nchw_nhwc_parity() raises:
    # ── NCHW ──
    var mn = CN.make["cpu", Deterministic]()
    var xn = Tensor.alloc(B * IN)
    var gon = Tensor.alloc(B * OUT)
    _setup[LAYOUT_NCHW](mn, xn, gon)
    var outn = Tensor.alloc(B * OUT)
    mn.forward["cpu", B](TensorRefs[1](xn), outn, None)
    var gin = Tensor.alloc(B * IN)
    mn.zero_grad["cpu"](None)
    mn.vjp["cpu", B](TensorRefs[1](xn), gon, TensorRefs[1](gin), None)

    # ── NHWC ──
    var mh = CH.make["cpu", Deterministic]()
    var xh = Tensor.alloc(B * IN)
    var goh = Tensor.alloc(B * OUT)
    _setup[LAYOUT_NHWC](mh, xh, goh)
    var outh = Tensor.alloc(B * OUT)
    mh.forward["cpu", B](TensorRefs[1](xh), outh, None)
    var gih = Tensor.alloc(B * IN)
    mh.zero_grad["cpu"](None)
    mh.vjp["cpu", B](TensorRefs[1](xh), goh, TensorRefs[1](gih), None)

    # ── compare logical tensors across the two layouts ──
    var d_out: Float64 = 0.0
    var d_gi: Float64 = 0.0
    for b in range(B):
        for oc in range(OC):
            for s in range(SO):
                var a = outn.data[b * OUT + _out_off[LAYOUT_NCHW, OC, SO](oc, s)]
                var c = outh.data[b * OUT + _out_off[LAYOUT_NHWC, OC, SO](oc, s)]
                d_out = max(d_out, abs(Float64(a - c)))
        for ic in range(IC):
            for ih in range(H):
                for iw in range(W):
                    var a = gin.data[
                        b * IN + _in_off[LAYOUT_NCHW, IC, H, W](ic, ih, iw)
                    ]
                    var c = gih.data[
                        b * IN + _in_off[LAYOUT_NHWC, IC, H, W](ic, ih, iw)
                    ]
                    d_gi = max(d_gi, abs(Float64(a - c)))
    var d_gw: Float64 = 0.0
    for oc in range(OC):
        for ic in range(IC):
            for kh in range(K):
                for kw in range(K):
                    var a = mn.weight.grd.data[
                        oc * COL + _col_off[LAYOUT_NCHW, IC, K](ic, kh, kw)
                    ]
                    var c = mh.weight.grd.data[
                        oc * COL + _col_off[LAYOUT_NHWC, IC, K](ic, kh, kw)
                    ]
                    d_gw = max(d_gw, abs(Float64(a - c)))
    var d_gb: Float64 = 0.0
    for oc in range(OC):
        d_gb = max(d_gb, abs(Float64(mn.bias.grd.data[oc] - mh.bias.grd.data[oc])))

    print("  NCHW↔NHWC max|Δ|: out", d_out, "grad_x", d_gi, "grad_w", d_gw,
          "grad_b", d_gb)
    var tol = 1e-5
    assert_true(
        d_out < tol and d_gi < tol and d_gw < tol and d_gb < tol,
        "Conv2D NCHW vs NHWC logical parity (exact)",
    )


def main() raises:
    print("=" * 60)
    print("Conv2D NCHW-vs-NHWC layout parity (CPU)")
    print("=" * 60)
    test_nchw_nhwc_parity()
    print("CONV2D NHWC PARITY GATE PASSED")
