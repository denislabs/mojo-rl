"""BlockLinear storage primitive — CPU correctness (golden) + GPU vs CPU.

Standalone storage test (no legacy oracle — converted from the former
`_storage_parity` test in legacy-removal Phase 0b). The CPU check asserts the
storage forward/backward against golden fingerprints (S = Σ vᵢ, W = Σ vᵢ·(i+1) —
the weight catches sign/position errors that a plain sum would cancel), captured
from the bit-identical legacy↔storage run the parity test used to verify. The GPU
check is storage-only (GPU vs CPU consistency). Run:
  pixi run -e apple mojo run -I . tests/nn/test_block_linear_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.block_linear import BlockLinear


comptime IN = 12
comptime OUT = 8
comptime BLOCKS = 4
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


def test_bl_cpu_golden() raises:
    print("test_bl_cpu_golden (storage CPU vs golden) ...")
    comptime TOL = Scalar[DT](5e-3)
    comptime W = BlockLinear[IN, OUT, BLOCKS].W_SIZE

    var st = BlockLinear[IN, OUT, BLOCKS].make["cpu", Deterministic]()
    for k in range(W):
        st.weight.val.data[k] = Scalar[DT](0.3 - 0.013 * Float64(k % 11))
    for k in range(OUT):
        st.bias.val.data[k] = Scalar[DT](-0.2 + 0.05 * Float64(k))
    var sx = Tensor.alloc(B * IN)
    var sgo = Tensor.alloc(B * OUT)
    var sout = Tensor.alloc(B * OUT)
    var sgi = Tensor.alloc(B * IN)
    for i in range(B * IN):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(B * OUT):
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.zero_grad["cpu"](None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var ok = _check("out", sout, B * OUT, -2.5319993, -35.17026, TOL)
    ok = _check("gi", sgi, B * IN, -0.5139198, 12.668703, TOL) and ok
    ok = _check("dweight", st.weight.grd, W, 1.7820005, -35.3628, TOL) and ok
    ok = _check("dbias", st.bias.grd, OUT, -0.65999997, -6.82, TOL) and ok
    assert_true(ok, "BlockLinear CPU golden")
    print("  ok")


def test_bl_gpu_vs_cpu() raises:
    print("test_bl_gpu_vs_cpu (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    comptime W = BlockLinear[IN, OUT, BLOCKS].W_SIZE
    var c = DeviceContext()
    var cpu = BlockLinear[IN, OUT, BLOCKS].make["cpu", Deterministic]()
    var gpu = BlockLinear[IN, OUT, BLOCKS].make["gpu", Deterministic](Optional(c))
    for k in range(W):
        cpu.weight.val.data[k] = Scalar[DT](0.3 - 0.013 * Float64(k % 11))
        gpu.weight.val.data[k] = cpu.weight.val.data[k]
    for k in range(OUT):
        cpu.bias.val.data[k] = Scalar[DT](-0.2 + 0.05 * Float64(k))
        gpu.bias.val.data[k] = cpu.bias.val.data[k]
    gpu.weight.val.upload(c)
    gpu.bias.val.upload(c)

    var sx = Tensor.alloc(B * IN)
    var sgo = Tensor.alloc(B * OUT)
    for i in range(B * IN):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(B * OUT):
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    var c_out = Tensor.alloc(B * OUT)
    var c_gi = Tensor.alloc(B * IN)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
    cpu.zero_grad["cpu"](None)
    cpu.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(B * IN)
    var ggo = Tensor.alloc(B * OUT)
    for i in range(B * IN):
        gx.data[i] = sx.data[i]
    for i in range(B * OUT):
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(B * OUT)
    var g_gi = Tensor.alloc(B * IN)
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.zero_grad["gpu"](Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)
    gpu.weight.grd.download(c)
    gpu.bias.grd.download(c)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * OUT):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
    for i in range(B * IN):
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    var mdw: Scalar[DT] = 0
    for k in range(W):
        if abs(gpu.weight.grd.data[k] - cpu.weight.grd.data[k]) > mdw:
            mdw = abs(gpu.weight.grd.data[k] - cpu.weight.grd.data[k])
    var mdb: Scalar[DT] = 0
    for k in range(OUT):
        if abs(gpu.bias.grd.data[k] - cpu.bias.grd.data[k]) > mdb:
            mdb = abs(gpu.bias.grd.data[k] - cpu.bias.grd.data[k])
    print("  max Δ: out", mo, " gi", mgi, " dw", mdw, " db", mdb)
    assert_true(mo < TOL and mgi < TOL and mdw < TOL and mdb < TOL,
                "BlockLinear GPU vs CPU")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("BlockLinear storage primitive (CPU golden + GPU vs CPU)")
    print("=" * 70)
    test_bl_cpu_golden()
    test_bl_gpu_vs_cpu()
    print("ALL PASSED")
