"""TiedLinear legacy ↔ storage parity (CPU) + storage GPU vs CPU.

The weight is BORROWED (tied) from an external owner cell. Both legacy and
storage tie to the SAME source weight values + the same input, and we compare
out + grad_input + the SHARED weight grad (max|Δ| < 1e-6 CPU). GPU compares
storage GPU vs storage CPU (TOL ~2e-5).

Run:
  rm -f mojo_rl.mojoc && pixi run mojo run -I . tests/nn/test_tied_linear_storage_parity.mojo
  rm -f mojo_rl.mojoc && pixi run -e apple mojo run -I . tests/nn/test_tied_linear_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.tied_linear import TiedLinear as LegacyTiedLinear
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.core.module import mptr
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.tied_linear import TiedLinear


comptime IN = 5    # EMBED
comptime OUT = 7   # VOCAB
comptime B = 4


def test_tied_cpu_parity() raises:
    print("test_tied_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)

    # Shared source weight + grad values (laid out [OUT, IN]) and inputs.
    var Wv = List[Scalar[DT]](length=OUT * IN, fill=0.0)
    var xv = List[Scalar[DT]](length=B * IN, fill=0.0)
    var gov = List[Scalar[DT]](length=B * OUT, fill=0.0)
    for k in range(OUT * IN):
        Wv[k] = Scalar[DT]((k % 13) - 6) * 0.11
    for i in range(B * IN):
        xv[i] = Scalar[DT]((i % 11) - 5) * 0.17
    for i in range(B * OUT):
        gov[i] = Scalar[DT]((i % 7) - 3) * 0.23

    # ── Legacy: tie to raw scalar buffers ────────────────────────────────
    var legW: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OUT * IN)
    var leggW: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OUT * IN)
    var legx: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * IN)
    var legout: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * OUT)
    var leggo: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * OUT)
    var leggi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * IN)
    for k in range(OUT * IN):
        legW[k] = Wv[k]
        leggW[k] = 0.0
    for i in range(B * IN):
        legx[i] = xv[i]
    for i in range(B * OUT):
        leggo[i] = gov[i]

    var leg = LegacyTiedLinear[IN, OUT].make[target="cpu", INIT=Zero]()
    leg.tie_to(mptr(legW), mptr(leggW))
    var lx_t = TileTensor(legx, row_major[B, IN]())
    var lout_t = TileTensor(legout, row_major[B, OUT]())
    var lgo_t = TileTensor(leggo, row_major[B, OUT]())
    var lgi_t = TileTensor(leggi, row_major[B, IN]())
    leg.forward["cpu", B](lx_t, output=lout_t)
    leg.vjp["cpu", B](lgo_t, lgi_t)

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

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * OUT):
        if abs(sout.data[i] - legout[i]) > mo: mo = abs(sout.data[i] - legout[i])
    for i in range(B * IN):
        if abs(sgi.data[i] - leggi[i]) > mgi: mgi = abs(sgi.data[i] - leggi[i])
    var mdw: Scalar[DT] = 0
    for k in range(OUT * IN):
        if abs(sgW.data[k] - leggW[k]) > mdw: mdw = abs(sgW.data[k] - leggW[k])
    print("  max Δ: out", mo, " gi", mgi, " dW", mdw)
    assert_true(mo < TOL and mgi < TOL and mdw < TOL, "TiedLinear CPU parity")
    _ = sW.n  # keep borrowed owner cell alive past the tied calls
    print("  ok")


def test_tied_gpu_parity() raises:
    print("test_tied_gpu_parity (storage GPU vs storage CPU) ...")
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
    print("TiedLinear legacy ↔ storage parity")
    print("=" * 70)
    test_tied_cpu_parity()
    test_tied_gpu_parity()
    print("ALL PASSED")
