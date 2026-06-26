"""BatchNorm2D NCHW-vs-NHWC layout parity (CPU, exact).

Phase 1 of the channels_last migration: the `LAYOUT` param wires an NHWC code
path into BatchNorm2D (single `_bn_off` offset swap — the per-channel stats are
layout-agnostic). This gate proves that path computes the SAME logical BN as
NCHW: set one logical (input, grad_output) in both layouts and assert forward,
grad_input, grad_gamma, grad_beta agree. CPU is deterministic → exact match.

NCHW is the default and is covered bit-identically by test_batch_norm_2d_storage;
this file only exercises the new NHWC path against it.
"""

from std.math import abs
from std.sys import has_accelerator
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT, LAYOUT_NCHW, LAYOUT_NHWC
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.batch_norm_2d import BatchNorm2D, _bn_off

comptime C = 3
comptime HH = 4
comptime WW = 5
comptime B = 6
comptime SP = HH * WW
comptime FLAT = C * SP


def _xl(b: Int, c: Int, s: Int) -> Scalar[DT]:
    return Scalar[DT](((((b * C + c) * SP + s) % 17) - 8)) * 0.13


def _gol(b: Int, c: Int, s: Int) -> Scalar[DT]:
    return Scalar[DT](((((b * C + c) * SP + s) % 9) - 4)) * 0.25


def _setup[
    LAYOUT: Int
](mut m: BatchNorm2D[C, HH, WW, ADT=DT, LAYOUT=LAYOUT], mut x: Tensor, mut go: Tensor):
    """Same logical input / grad_output written into `m`'s LAYOUT; γ/β are per-
    channel (layout-agnostic) so set them identically."""
    for k in range(C):
        m.gamma.val.data[k] = Scalar[DT](0.7 + 0.1 * Float64(k))
        m.beta.val.data[k] = Scalar[DT](-0.3 + 0.05 * Float64(k))
    for b in range(B):
        var bb = b * FLAT
        for c in range(C):
            for s in range(SP):
                x.data[bb + _bn_off[LAYOUT, C, SP](c, s)] = _xl(b, c, s)
                go.data[bb + _bn_off[LAYOUT, C, SP](c, s)] = _gol(b, c, s)


def test_bn2d_nchw_nhwc_parity() raises:
    # ── NCHW ──
    var mn = BatchNorm2D[C, HH, WW].make["cpu", Deterministic]()
    var xn = Tensor.alloc(B * FLAT)
    var gon = Tensor.alloc(B * FLAT)
    _setup[LAYOUT_NCHW](mn, xn, gon)
    var outn = Tensor.alloc(B * FLAT)
    mn.forward["cpu", B](TensorRefs[1](xn), outn, None)
    var gin = Tensor.alloc(B * FLAT)
    mn.zero_grad["cpu"](None)
    mn.vjp["cpu", B](TensorRefs[1](xn), gon, TensorRefs[1](gin), None)

    # ── NHWC ──
    var mh = BatchNorm2D[C, HH, WW, ADT=DT, LAYOUT=LAYOUT_NHWC].make[
        "cpu", Deterministic
    ]()
    var xh = Tensor.alloc(B * FLAT)
    var goh = Tensor.alloc(B * FLAT)
    _setup[LAYOUT_NHWC](mh, xh, goh)
    var outh = Tensor.alloc(B * FLAT)
    mh.forward["cpu", B](TensorRefs[1](xh), outh, None)
    var gih = Tensor.alloc(B * FLAT)
    mh.zero_grad["cpu"](None)
    mh.vjp["cpu", B](TensorRefs[1](xh), goh, TensorRefs[1](gih), None)

    # ── compare logical tensors ──
    var d_out: Float64 = 0.0
    var d_gi: Float64 = 0.0
    for b in range(B):
        var bb = b * FLAT
        for c in range(C):
            for s in range(SP):
                var on = outn.data[bb + _bn_off[LAYOUT_NCHW, C, SP](c, s)]
                var oh = outh.data[bb + _bn_off[LAYOUT_NHWC, C, SP](c, s)]
                d_out = max(d_out, abs(Float64(on - oh)))
                var an = gin.data[bb + _bn_off[LAYOUT_NCHW, C, SP](c, s)]
                var ah = gih.data[bb + _bn_off[LAYOUT_NHWC, C, SP](c, s)]
                d_gi = max(d_gi, abs(Float64(an - ah)))
    var d_dg: Float64 = 0.0
    var d_db: Float64 = 0.0
    for k in range(C):
        d_dg = max(d_dg, abs(Float64(mn.gamma.grd.data[k] - mh.gamma.grd.data[k])))
        d_db = max(d_db, abs(Float64(mn.beta.grd.data[k] - mh.beta.grd.data[k])))

    print("  NCHW↔NHWC max|Δ|: out", d_out, "grad_x", d_gi, "grad_gamma", d_dg,
          "grad_beta", d_db)
    var tol = 1e-5
    assert_true(
        d_out < tol and d_gi < tol and d_dg < tol and d_db < tol,
        "BatchNorm2D NCHW vs NHWC logical parity (exact)",
    )


def test_bn2d_nhwc_gpu_vs_cpu() raises:
    """The GPU NHWC path uses the COALESCED transposed-reduction kernels (vs the
    CPU NHWC offset-swap loops). Both must agree → validates the new GPU kernels.
    fp32 reduction-order (chunked GPU vs sequential CPU) → loose tol."""
    var ctx = DeviceContext()
    comptime L = LAYOUT_NHWC
    # CPU NHWC reference
    var mc = BatchNorm2D[C, HH, WW, ADT=DT, LAYOUT=L].make["cpu", Deterministic]()
    var xc = Tensor.alloc(B * FLAT)
    var goc = Tensor.alloc(B * FLAT)
    _setup[L](mc, xc, goc)
    var outc = Tensor.alloc(B * FLAT)
    mc.forward["cpu", B](TensorRefs[1](xc), outc, None)
    var gic = Tensor.alloc(B * FLAT)
    mc.zero_grad["cpu"](None)
    mc.vjp["cpu", B](TensorRefs[1](xc), goc, TensorRefs[1](gic), None)

    # GPU NHWC (transposed coalesced kernels)
    var mg = BatchNorm2D[C, HH, WW, ADT=DT, LAYOUT=L].make[
        "gpu", Deterministic
    ](Optional(ctx))
    var xg = Tensor.alloc(B * FLAT)
    var gog = Tensor.alloc(B * FLAT)
    _setup[L](mg, xg, gog)  # writes host gamma/beta + x/go
    mg.gamma.val.upload(ctx)
    mg.beta.val.upload(ctx)
    xg.upload(ctx)
    gog.upload(ctx)
    var outg = Tensor.alloc(B * FLAT)
    mg.forward["gpu", B](TensorRefs[1](xg), outg, Optional(ctx))
    var gig = Tensor.alloc(B * FLAT)
    mg.zero_grad["gpu"](Optional(ctx))
    mg.vjp["gpu", B](TensorRefs[1](xg), gog, TensorRefs[1](gig), Optional(ctx))
    outg.download(ctx)
    gig.download(ctx)
    mg.gamma.grd.download(ctx)
    mg.beta.grd.download(ctx)
    mg.running_var.t.download(ctx)

    var d_out: Float64 = 0.0
    var d_gi: Float64 = 0.0
    for i in range(B * FLAT):
        d_out = max(d_out, abs(Float64(outc.data[i] - outg.data[i])))
        d_gi = max(d_gi, abs(Float64(gic.data[i] - gig.data[i])))
    var d_dg: Float64 = 0.0
    var d_rv: Float64 = 0.0
    for k in range(C):
        d_dg = max(d_dg, abs(Float64(mc.gamma.grd.data[k] - mg.gamma.grd.data[k])))
        d_rv = max(
            d_rv, abs(Float64(mc.running_var.t.data[k] - mg.running_var.t.data[k]))
        )
    print("  GPU↔CPU (NHWC) max|Δ|: out", d_out, "grad_x", d_gi, "grad_gamma",
          d_dg, "rvar", d_rv)
    var tol = 1e-3
    assert_true(
        d_out < tol and d_gi < tol and d_dg < tol and d_rv < tol,
        "BN2D NHWC GPU (transposed kernels) == CPU NHWC",
    )


def main() raises:
    print("=" * 60)
    print("BatchNorm2D NCHW-vs-NHWC layout parity (CPU)")
    print("=" * 60)
    test_bn2d_nchw_nhwc_parity()
    print("  CPU parity: ok")
    comptime if has_accelerator():
        test_bn2d_nhwc_gpu_vs_cpu()
    print("BN2D NHWC PARITY GATE PASSED")
