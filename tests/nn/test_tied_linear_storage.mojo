"""TiedLinear storage primitive — CPU correctness (golden) + GPU vs CPU.

Standalone storage test (no legacy oracle — converted from the former
`_storage_parity` test in legacy-removal Phase 0b). The weight is BORROWED
(tied) from an external owner cell. The CPU check asserts the storage
forward/backward against golden fingerprints (S = Σ vᵢ, W = Σ vᵢ·(i+1) — the
weight catches sign/position errors that a plain sum would cancel), captured
from the bit-identical legacy↔storage run the parity test used to verify. The
GPU check is storage-only (GPU vs CPU consistency). Run:
  pixi run -e apple mojo run -I . tests/nn/test_tied_linear_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.tied_linear import TiedLinear


comptime IN = 5    # EMBED
comptime OUT = 7   # VOCAB
comptime B = 4


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


def test_tied_cpu_golden() raises:
    print("test_tied_cpu_golden (storage CPU vs golden) ...")
    comptime TOL = Scalar[DT](5e-3)

    # Shared source weight values (laid out [OUT, IN]) and inputs.
    var Wv = List[Scalar[DT]](length=OUT * IN, fill=0.0)
    var xv = List[Scalar[DT]](length=B * IN, fill=0.0)
    var gov = List[Scalar[DT]](length=B * OUT, fill=0.0)
    for k in range(OUT * IN):
        Wv[k] = Scalar[DT]((k % 13) - 6) * 0.11
    for i in range(B * IN):
        xv[i] = Scalar[DT]((i % 11) - 5) * 0.17
    for i in range(B * OUT):
        gov[i] = Scalar[DT]((i % 7) - 3) * 0.23

    # ── Storage: tie to owner Tensor cells ───────────────────────────────
    var sW = Tensor.alloc(OUT * IN)
    var sgW = Tensor.alloc(OUT * IN)
    for k in range(OUT * IN):
        sW.data[k] = Wv[k]
    var st = TiedLinear[IN, OUT].make["cpu", Deterministic]()
    st.tie_to(sW, sgW)

    var sx = Tensor.alloc(B * IN)
    var sgo = Tensor.alloc(B * OUT)
    var sout = Tensor.alloc(B * OUT)
    var sgi = Tensor.alloc(B * IN)
    for i in range(B * IN):
        sx.data[i] = xv[i]
    for i in range(B * OUT):
        sgo.data[i] = gov[i]
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var ok = _check("out", sout, B * OUT, 1.3089999, 11.4631, TOL)
    ok = _check("gi", sgi, B * IN, 3.7444, 38.000603, TOL) and ok
    ok = _check("dW", sgW, OUT * IN, 5.9604645e-08, -49.266003, TOL) and ok
    assert_true(ok, "TiedLinear CPU golden")
    _ = sW.n  # keep borrowed owner cell alive past the tied calls
    print("  ok")


def test_tied_gpu_vs_cpu() raises:
    print("test_tied_gpu_vs_cpu (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()

    # Shared source weight + inputs.
    var Wv = List[Scalar[DT]](length=OUT * IN, fill=0.0)
    var xv = List[Scalar[DT]](length=B * IN, fill=0.0)
    var gov = List[Scalar[DT]](length=B * OUT, fill=0.0)
    for k in range(OUT * IN):
        Wv[k] = Scalar[DT]((k % 13) - 6) * 0.11
    for i in range(B * IN):
        xv[i] = Scalar[DT]((i % 11) - 5) * 0.17
    for i in range(B * OUT):
        gov[i] = Scalar[DT]((i % 7) - 3) * 0.23

    # ── CPU ──────────────────────────────────────────────────────────────
    var cW = Tensor.alloc(OUT * IN)
    var cgW = Tensor.alloc(OUT * IN)
    for k in range(OUT * IN):
        cW.data[k] = Wv[k]
    var cpu = TiedLinear[IN, OUT].make["cpu", Deterministic]()
    cpu.tie_to(cW, cgW)
    var cx = Tensor.alloc(B * IN)
    var cgo = Tensor.alloc(B * OUT)
    var c_out = Tensor.alloc(B * OUT)
    var c_gi = Tensor.alloc(B * IN)
    for i in range(B * IN):
        cx.data[i] = xv[i]
    for i in range(B * OUT):
        cgo.data[i] = gov[i]
    cpu.forward["cpu", B](TensorRefs[1](cx), c_out, None)
    cpu.vjp["cpu", B](TensorRefs[1](cx), cgo, TensorRefs[1](c_gi), None)

    # ── GPU ──────────────────────────────────────────────────────────────
    var gW = Tensor.alloc(OUT * IN)
    var ggW = Tensor.alloc(OUT * IN)
    for k in range(OUT * IN):
        gW.data[k] = Wv[k]
    gW.upload(c)
    ggW.upload(c)  # zeroed grad cell on device
    var gpu = TiedLinear[IN, OUT].make["gpu", Deterministic](Optional(c))
    gpu.tie_to(gW, ggW)
    var gx = Tensor.alloc(B * IN)
    var ggo = Tensor.alloc(B * OUT)
    for i in range(B * IN):
        gx.data[i] = xv[i]
    for i in range(B * OUT):
        ggo.data[i] = gov[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(B * OUT)
    var g_gi = Tensor.alloc(B * IN)
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)
    ggW.download(c)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * OUT):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
    for i in range(B * IN):
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    var mdw: Scalar[DT] = 0
    for k in range(OUT * IN):
        if abs(ggW.data[k] - cgW.data[k]) > mdw: mdw = abs(ggW.data[k] - cgW.data[k])
    print("  max Δ: out", mo, " gi", mgi, " dW", mdw)
    assert_true(mo < TOL and mgi < TOL and mdw < TOL, "TiedLinear GPU vs CPU")
    # keep the borrowed owner cells alive past the tied calls (the tie holds a
    # raw pointer to them; in real use the owner Param lives inside the model).
    _ = gW.n
    _ = cW.n
    print("  ok")


def main() raises:
    print("=" * 70)
    print("TiedLinear storage primitive (CPU golden + GPU vs CPU)")
    print("=" * 70)
    test_tied_cpu_golden()
    test_tied_gpu_vs_cpu()
    print("ALL PASSED")
