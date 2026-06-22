"""LayerNorm storage primitive — CPU correctness (golden) + GPU vs CPU.

Standalone storage test (no legacy oracle — converted from the former
`_storage_parity` test in legacy-removal Phase 0b). The CPU check asserts the
storage forward/backward against golden fingerprints (S = Σ vᵢ, W = Σ vᵢ·(i+1) —
the weight catches sign/position errors that a plain sum would cancel), captured
from the bit-identical legacy↔storage run the parity test used to verify. The GPU
check is storage-only (GPU vs CPU consistency). Run:
  pixi run -e apple mojo run -I . tests/nn/test_layer_norm_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.layer_norm import LayerNorm


comptime DIM = 10
comptime B = 6


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


def test_ln_cpu_golden() raises:
    print("test_ln_cpu_golden (storage CPU vs golden) ...")
    comptime TOL = Scalar[DT](5e-3)
    var st = LayerNorm[DIM].make["cpu", Deterministic]()
    for k in range(DIM):
        st.gamma.val.data[k] = Scalar[DT](0.6 + 0.07 * Float64(k))
        st.beta.val.data[k] = Scalar[DT](-0.15 + 0.04 * Float64(k))
    var sx = Tensor.alloc(B * DIM)
    var sgo = Tensor.alloc(B * DIM)
    var sout = Tensor.alloc(B * DIM)
    var sgi = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.zero_grad["cpu"](None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var ok = _check("out", sout, B * DIM, 3.7573538, 154.60642, TOL)
    ok = _check("gi", sgi, B * DIM, 0.0, -3.592206, TOL) and ok
    ok = _check("dgamma", st.gamma.grd, DIM, -2.1457193, -17.76936, TOL) and ok
    ok = _check("dbeta", st.beta.grd, DIM, -1.3199999, -9.24, TOL) and ok
    assert_true(ok, "LayerNorm CPU golden")
    print("  ok")


def test_ln_gpu_vs_cpu() raises:
    print("test_ln_gpu_vs_cpu (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = LayerNorm[DIM].make["cpu", Deterministic]()
    var gpu = LayerNorm[DIM].make["gpu", Deterministic](Optional(c))
    for k in range(DIM):
        cpu.gamma.val.data[k] = Scalar[DT](0.6 + 0.07 * Float64(k))
        cpu.beta.val.data[k] = Scalar[DT](-0.15 + 0.04 * Float64(k))
        gpu.gamma.val.data[k] = cpu.gamma.val.data[k]
        gpu.beta.val.data[k] = cpu.beta.val.data[k]
    gpu.gamma.val.upload(c)
    gpu.beta.val.upload(c)

    var sx = Tensor.alloc(B * DIM)
    var sgo = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    var c_out = Tensor.alloc(B * DIM)
    var c_gi = Tensor.alloc(B * DIM)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
    cpu.zero_grad["cpu"](None)
    cpu.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(B * DIM)
    var ggo = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        gx.data[i] = sx.data[i]
        ggo.data[i] = sgo.data[i]
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
    gpu.beta.grd.download(c)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * DIM):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    var mdg: Scalar[DT] = 0
    for k in range(DIM):
        if abs(gpu.gamma.grd.data[k] - cpu.gamma.grd.data[k]) > mdg:
            mdg = abs(gpu.gamma.grd.data[k] - cpu.gamma.grd.data[k])
    print("  max Δ: out", mo, " gi", mgi, " dg", mdg)
    assert_true(mo < TOL and mgi < TOL and mdg < TOL, "LayerNorm GPU vs CPU")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("LayerNorm storage primitive (CPU golden + GPU vs CPU)")
    print("=" * 70)
    test_ln_cpu_golden()
    test_ln_gpu_vs_cpu()
    print("ALL PASSED")
