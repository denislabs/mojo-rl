"""ConvRMSNorm (channel-wise RMSNorm for NCHW conv maps) — correctness.

Three checks:
  1. CPU forward golden — independent channel-wise RMS formula (catches the
     strided-channel indexing).
  2. CPU finite-difference gradcheck — numeric vs analytic grad_input and
     grad_gamma (catches an analytic backward bug that CPU↔GPU parity alone
     would NOT, since both paths could share it).
  3. GPU vs CPU parity — forward + grad_input + grad_gamma.

Run:
  pixi run -e apple mojo run -I . tests/nn/test_conv_rms_norm.mojo
"""

from std.math import sqrt
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.conv_rms_norm import ConvRMSNorm


comptime C = 3
comptime HW = 4
comptime DIM = C * HW
comptime B = 2
comptime EPS = Scalar[DT](1e-4)


def _fill(mut x: Tensor):
    for i in range(B * DIM):
        x.data[i] = Scalar[DT]((i % 11) - 5) * 0.21 + 0.13


def _set_gamma(mut m: ConvRMSNorm[C, HW]):
    for c in range(C):
        m.gamma.val.data[c] = Scalar[DT](0.7 + 0.3 * Float64(c))


def _loss(mut m: ConvRMSNorm[C, HW], mut x: Tensor, go: Tensor) raises -> Scalar[DT]:
    var out = Tensor.alloc(B * DIM)
    m.forward["cpu", B](TensorRefs[1](x), out, None)
    var L: Scalar[DT] = 0
    for i in range(B * DIM):
        L += out.data[i] * go.data[i]
    return L


def test_forward_golden() raises:
    print("test_forward_golden ...")
    var m = ConvRMSNorm[C, HW].make["cpu", Deterministic]()
    _set_gamma(m)
    var x = Tensor.alloc(B * DIM)
    _fill(x)
    var out = Tensor.alloc(B * DIM)
    m.forward["cpu", B](TensorRefs[1](x), out, None)

    var ok = True
    for b in range(B):
        for p in range(HW):
            var sumsq: Scalar[DT] = 0
            for c in range(C):
                var v = x.data[b * DIM + c * HW + p]
                sumsq += v * v
            var inv_rms = Scalar[DT](1.0) / sqrt(sumsq / Float32(C) + EPS)
            for c in range(C):
                var idx = b * DIM + c * HW + p
                var exp = x.data[idx] * inv_rms * m.gamma.val.data[c]
                if abs(out.data[idx] - exp) > 1e-5:
                    ok = False
    assert_true(ok, "ConvRMSNorm forward golden (channel-wise)")
    print("  ok")


def test_gradcheck() raises:
    print("test_gradcheck (finite-diff vs analytic) ...")
    comptime H = Scalar[DT](3e-3)
    comptime TOL = Scalar[DT](2e-2)
    var m = ConvRMSNorm[C, HW].make["cpu", Deterministic]()
    _set_gamma(m)
    var x = Tensor.alloc(B * DIM)
    _fill(x)
    var go = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        go.data[i] = Scalar[DT]((i % 5) - 2) * 0.3 + 0.1

    # analytic: forward (populates cache) → vjp
    var out = Tensor.alloc(B * DIM)
    m.forward["cpu", B](TensorRefs[1](x), out, None)
    m.zero_grad["cpu"](None)
    var gi = Tensor.alloc(B * DIM)
    m.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), None)

    # numeric grad_input
    var ok = True
    var maxerr_x: Scalar[DT] = 0
    for j in range(B * DIM):
        var saved = x.data[j]
        x.data[j] = saved + H
        var lp = _loss(m, x, go)
        x.data[j] = saved - H
        var lm = _loss(m, x, go)
        x.data[j] = saved
        var num = (lp - lm) / (2 * H)
        var err = abs(num - gi.data[j])
        if err > maxerr_x:
            maxerr_x = err
    # restore cache for any later use (re-forward pristine)
    m.forward["cpu", B](TensorRefs[1](x), out, None)

    # numeric grad_gamma
    var maxerr_g: Scalar[DT] = 0
    for c in range(C):
        var saved = m.gamma.val.data[c]
        m.gamma.val.data[c] = saved + H
        var lp = _loss(m, x, go)
        m.gamma.val.data[c] = saved - H
        var lm = _loss(m, x, go)
        m.gamma.val.data[c] = saved
        var num = (lp - lm) / (2 * H)
        var err = abs(num - m.gamma.grd.data[c])
        if err > maxerr_g:
            maxerr_g = err

    print("  max|num-analytic|: grad_input", maxerr_x, " grad_gamma", maxerr_g)
    ok = maxerr_x < TOL and maxerr_g < TOL
    assert_true(ok, "ConvRMSNorm finite-diff gradcheck")
    print("  ok")


def test_gpu_vs_cpu() raises:
    print("test_gpu_vs_cpu ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = ConvRMSNorm[C, HW].make["cpu", Deterministic]()
    var gpu = ConvRMSNorm[C, HW].make["gpu", Deterministic](Optional(c))
    for ch in range(C):
        cpu.gamma.val.data[ch] = Scalar[DT](0.7 + 0.3 * Float64(ch))
        gpu.gamma.val.data[ch] = cpu.gamma.val.data[ch]
    gpu.gamma.val.upload(c)

    var x = Tensor.alloc(B * DIM)
    var go = Tensor.alloc(B * DIM)
    _fill(x)
    for i in range(B * DIM):
        go.data[i] = Scalar[DT]((i % 5) - 2) * 0.3 + 0.1

    var c_out = Tensor.alloc(B * DIM)
    var c_gi = Tensor.alloc(B * DIM)
    cpu.forward["cpu", B](TensorRefs[1](x), c_out, None)
    cpu.zero_grad["cpu"](None)
    cpu.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(B * DIM)
    var ggo = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        gx.data[i] = x.data[i]
        ggo.data[i] = go.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(B * DIM)
    var g_gi = Tensor.alloc(B * DIM)
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.zero_grad["gpu"](Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)
    gpu.gamma.grd.download(c)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * DIM):
        if abs(g_out.data[i] - c_out.data[i]) > mo:
            mo = abs(g_out.data[i] - c_out.data[i])
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi:
            mgi = abs(g_gi.data[i] - c_gi.data[i])
    var mdg: Scalar[DT] = 0
    for ch in range(C):
        if abs(gpu.gamma.grd.data[ch] - cpu.gamma.grd.data[ch]) > mdg:
            mdg = abs(gpu.gamma.grd.data[ch] - cpu.gamma.grd.data[ch])
    print("  max Δ: out", mo, " gi", mgi, " dg", mdg)
    assert_true(mo < TOL and mgi < TOL and mdg < TOL, "ConvRMSNorm GPU vs CPU")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("ConvRMSNorm (channel-wise RMSNorm, NCHW) — correctness")
    print("=" * 70)
    test_forward_golden()
    test_gradcheck()
    test_gpu_vs_cpu()
    print("ALL PASSED")
