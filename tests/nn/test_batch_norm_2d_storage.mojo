"""BatchNorm2D storage primitive — CPU correctness (golden) + GPU vs CPU.

Standalone storage test (no legacy oracle — converted from the former
`_storage_parity` test in legacy-removal Phase 0b). The CPU check asserts the
storage forward/backward (train mode) + eval-mode forward against golden
fingerprints (S = Σ vᵢ, W = Σ vᵢ·(i+1) — the weight catches sign/position errors
that a plain sum would cancel), captured from the bit-identical legacy↔storage
run the parity test used to verify. The GPU check is storage-only (GPU vs CPU
consistency). Run:
  pixi run -e apple mojo run -I . tests/nn/test_batch_norm_2d_storage.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.batch_norm_2d import BatchNorm2D


comptime C = 3
comptime HH = 4
comptime WW = 4
comptime B = 6
comptime FLAT = C * HH * WW


def _check(name: String, data: Tensor, n: Int,
           es: Scalar[DT], ew: Scalar[DT], tol: Scalar[DT]) -> Bool:
    """Assert tensor fingerprint (Σ vᵢ, Σ vᵢ·(i+1)) matches golden (es, ew)."""
    var s: Scalar[DT] = 0
    var w: Scalar[DT] = 0
    for i in range(n):
        s += data.data[i]
        w += data.data[i] * Scalar[DT](i + 1)
    var ok = abs(s - es) < tol and abs(w - ew) < tol
    print("  ", name, "S", s, "(exp", es, ") W", w, "(exp", ew, ")", "OK" if ok else "FAIL")
    return ok


def test_bn2d_cpu_golden() raises:
    print("test_bn2d_cpu_golden (storage CPU vs golden) ...")
    comptime TOL = Scalar[DT](5e-3)
    var st = BatchNorm2D[C, HH, WW].make["cpu", Deterministic]()
    for k in range(C):
        st.gamma.val.data[k] = Scalar[DT](0.7 + 0.1 * Float64(k))
        st.beta.val.data[k] = Scalar[DT](-0.3 + 0.05 * Float64(k))
    var sx = Tensor.alloc(B * FLAT)
    var sgo = Tensor.alloc(B * FLAT)
    var sout = Tensor.alloc(B * FLAT)
    var sgi = Tensor.alloc(B * FLAT)
    var soute = Tensor.alloc(B * FLAT)
    for i in range(B * FLAT):
        sx.data[i] = Scalar[DT]((i % 17) - 8) * 0.13
        sgo.data[i] = Scalar[DT]((i % 9) - 4) * 0.25
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.zero_grad["cpu"](None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)
    st.set_training(False)
    st.forward["cpu", B](TensorRefs[1](sx), soute, None)

    var ok = _check("out", sout, B * FLAT, -72.000015, -9333.076, TOL)
    ok = _check("gi", sgi, B * FLAT, -1.1920929e-07, 560.82605, TOL) and ok
    ok = _check("eval", soute, B * FLAT, -72.83249, -9779.975, TOL) and ok
    ok = _check("dgamma", st.gamma.grd, C, -2.6456983, -7.3035164, TOL) and ok
    ok = _check("dbeta", st.beta.grd, C, 0.0, 3.0, TOL) and ok
    ok = _check("rmean", st.running_mean.t, C, -0.0010833312, -0.0028437467, TOL) and ok
    ok = _check("rvar", st.running_var.t, C, 2.8209608, 5.6419993, TOL) and ok
    assert_true(ok, "BatchNorm2D CPU golden")
    print("  ok")


def test_bn2d_gpu_vs_cpu() raises:
    print("test_bn2d_gpu_vs_cpu (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](3e-5)
    var c = DeviceContext()
    var cpu = BatchNorm2D[C, HH, WW].make["cpu", Deterministic]()
    var gpu = BatchNorm2D[C, HH, WW].make["gpu", Deterministic](Optional(c))
    for k in range(C):
        cpu.gamma.val.data[k] = Scalar[DT](0.7 + 0.1 * Float64(k))
        cpu.beta.val.data[k] = Scalar[DT](-0.3 + 0.05 * Float64(k))
        gpu.gamma.val.data[k] = cpu.gamma.val.data[k]
        gpu.beta.val.data[k] = cpu.beta.val.data[k]
    gpu.gamma.val.upload(c)
    gpu.beta.val.upload(c)

    var sx = Tensor.alloc(B * FLAT)
    var sgo = Tensor.alloc(B * FLAT)
    for i in range(B * FLAT):
        sx.data[i] = Scalar[DT]((i % 17) - 8) * 0.13
        sgo.data[i] = Scalar[DT]((i % 9) - 4) * 0.25
    var c_out = Tensor.alloc(B * FLAT)
    var c_gi = Tensor.alloc(B * FLAT)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
    cpu.zero_grad["cpu"](None)
    cpu.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(B * FLAT)
    var ggo = Tensor.alloc(B * FLAT)
    for i in range(B * FLAT):
        gx.data[i] = sx.data[i]
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(B * FLAT)
    var g_gi = Tensor.alloc(B * FLAT)
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.zero_grad["gpu"](Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)
    gpu.running_var.t.download(c)
    gpu.gamma.grd.download(c)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * FLAT):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    var mrv: Scalar[DT] = 0
    var mdg: Scalar[DT] = 0
    for f in range(C):
        if abs(gpu.running_var.t.data[f] - cpu.running_var.t.data[f]) > mrv:
            mrv = abs(gpu.running_var.t.data[f] - cpu.running_var.t.data[f])
        if abs(gpu.gamma.grd.data[f] - cpu.gamma.grd.data[f]) > mdg:
            mdg = abs(gpu.gamma.grd.data[f] - cpu.gamma.grd.data[f])
    print("  max Δ: out", mo, " gi", mgi, " rv", mrv, " dg", mdg)
    assert_true(mo < TOL and mgi < TOL and mrv < TOL and mdg < TOL,
                "BN2D GPU vs CPU")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("BatchNorm2D storage primitive (CPU golden + GPU vs CPU)")
    print("=" * 70)
    test_bn2d_cpu_golden()
    test_bn2d_gpu_vs_cpu()
    print("ALL PASSED")
