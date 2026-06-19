"""Pool2D legacy ↔ storage parity + storage GPU-vs-CPU.

For MaxPool2D and AvgPool2D: run the LEGACY `nn.primitives.*Pool2D` and the
storage port with identical input/grad_output, compare forward + grad_input
(CPU bit-parity vs legacy, then storage GPU vs storage CPU).

Run:
  pixi run mojo run -I . tests/nn/test_pool_2d_storage_parity.mojo
  pixi run -e apple mojo run -I . tests/nn/test_pool_2d_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.max_pool_2d import MaxPool2D as LegacyMaxPool2D
from mojo_rl.nn.primitives.avg_pool_2d import AvgPool2D as LegacyAvgPool2D
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.max_pool_2d import MaxPool2D
from mojo_rl.nn.storage.primitives.avg_pool_2d import AvgPool2D


comptime C = 2
comptime H = 4
comptime W = 4
comptime K = 2
comptime S = 2
comptime P = 0
comptime B = 3
comptime OH = (H + 2 * P - K) // S + 1
comptime OW = (W + 2 * P - K) // S + 1
comptime IN_FLAT = C * H * W
comptime OUT_FLAT = C * OH * OW


def _fill_in(i: Int) -> Scalar[DT]:
    return Scalar[DT]((i % 11) - 5) * 0.17


def _fill_go(i: Int) -> Scalar[DT]:
    return Scalar[DT]((i % 7) - 3) * 0.2


# ── MaxPool ────────────────────────────────────────────────────────────
def test_max_cpu_parity() raises:
    print("test_max_cpu_parity (legacy MaxPool2D vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)
    var leg = LegacyMaxPool2D[C, K, S, P, H, W].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * IN_FLAT)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * OUT_FLAT)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * OUT_FLAT)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * IN_FLAT)
    for i in range(B * IN_FLAT):
        x[i] = _fill_in(i)
    for i in range(B * OUT_FLAT):
        go[i] = _fill_go(i)

    var x_t = TileTensor(x, row_major[B, IN_FLAT]())
    var y_t = TileTensor(y, row_major[B, OUT_FLAT]())
    var go_t = TileTensor(go, row_major[B, OUT_FLAT]())
    var gi_t = TileTensor(gi, row_major[B, IN_FLAT]())
    leg.forward["cpu", B](x_t, output=y_t)
    leg.zero_grad["cpu"]()
    leg.vjp["cpu", B](go_t, gi_t)

    var st = MaxPool2D[C, K, S, P, H, W].make["cpu", Deterministic]()
    var sx = Tensor.alloc(B * IN_FLAT)
    var sgo = Tensor.alloc(B * OUT_FLAT)
    var sout = Tensor.alloc(B * OUT_FLAT)
    var sgi = Tensor.alloc(B * IN_FLAT)
    for i in range(B * IN_FLAT):
        sx.data[i] = x[i]
    for i in range(B * OUT_FLAT):
        sgo.data[i] = go[i]
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.zero_grad["cpu"](None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * OUT_FLAT):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
    for i in range(B * IN_FLAT):
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "MaxPool2D CPU parity")
    print("  ok")


def test_max_gpu_parity() raises:
    print("test_max_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = MaxPool2D[C, K, S, P, H, W].make["cpu", Deterministic]()
    var gpu = MaxPool2D[C, K, S, P, H, W].make["gpu", Deterministic](Optional(c))

    var sx = Tensor.alloc(B * IN_FLAT)
    var sgo = Tensor.alloc(B * OUT_FLAT)
    for i in range(B * IN_FLAT):
        sx.data[i] = _fill_in(i)
    for i in range(B * OUT_FLAT):
        sgo.data[i] = _fill_go(i)
    var c_out = Tensor.alloc(B * OUT_FLAT)
    var c_gi = Tensor.alloc(B * IN_FLAT)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
    cpu.zero_grad["cpu"](None)
    cpu.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(B * IN_FLAT)
    var ggo = Tensor.alloc(B * OUT_FLAT)
    for i in range(B * IN_FLAT):
        gx.data[i] = sx.data[i]
    for i in range(B * OUT_FLAT):
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(B * OUT_FLAT)
    var g_gi = Tensor.alloc(B * IN_FLAT)
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.zero_grad["gpu"](Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * OUT_FLAT):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
    for i in range(B * IN_FLAT):
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "MaxPool2D GPU vs CPU")
    print("  ok")


# ── AvgPool ────────────────────────────────────────────────────────────
def test_avg_cpu_parity() raises:
    print("test_avg_cpu_parity (legacy AvgPool2D vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)
    var leg = LegacyAvgPool2D[C, K, S, P, H, W].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * IN_FLAT)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * OUT_FLAT)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * OUT_FLAT)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * IN_FLAT)
    for i in range(B * IN_FLAT):
        x[i] = _fill_in(i)
    for i in range(B * OUT_FLAT):
        go[i] = _fill_go(i)

    var x_t = TileTensor(x, row_major[B, IN_FLAT]())
    var y_t = TileTensor(y, row_major[B, OUT_FLAT]())
    var go_t = TileTensor(go, row_major[B, OUT_FLAT]())
    var gi_t = TileTensor(gi, row_major[B, IN_FLAT]())
    leg.forward["cpu", B](x_t, output=y_t)
    leg.zero_grad["cpu"]()
    leg.vjp["cpu", B](go_t, gi_t)

    var st = AvgPool2D[C, K, S, P, H, W].make["cpu", Deterministic]()
    var sx = Tensor.alloc(B * IN_FLAT)
    var sgo = Tensor.alloc(B * OUT_FLAT)
    var sout = Tensor.alloc(B * OUT_FLAT)
    var sgi = Tensor.alloc(B * IN_FLAT)
    for i in range(B * IN_FLAT):
        sx.data[i] = x[i]
    for i in range(B * OUT_FLAT):
        sgo.data[i] = go[i]
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.zero_grad["cpu"](None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * OUT_FLAT):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
    for i in range(B * IN_FLAT):
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "AvgPool2D CPU parity")
    print("  ok")


def test_avg_gpu_parity() raises:
    print("test_avg_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = AvgPool2D[C, K, S, P, H, W].make["cpu", Deterministic]()
    var gpu = AvgPool2D[C, K, S, P, H, W].make["gpu", Deterministic](Optional(c))

    var sx = Tensor.alloc(B * IN_FLAT)
    var sgo = Tensor.alloc(B * OUT_FLAT)
    for i in range(B * IN_FLAT):
        sx.data[i] = _fill_in(i)
    for i in range(B * OUT_FLAT):
        sgo.data[i] = _fill_go(i)
    var c_out = Tensor.alloc(B * OUT_FLAT)
    var c_gi = Tensor.alloc(B * IN_FLAT)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
    cpu.zero_grad["cpu"](None)
    cpu.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(B * IN_FLAT)
    var ggo = Tensor.alloc(B * OUT_FLAT)
    for i in range(B * IN_FLAT):
        gx.data[i] = sx.data[i]
    for i in range(B * OUT_FLAT):
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(B * OUT_FLAT)
    var g_gi = Tensor.alloc(B * IN_FLAT)
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.zero_grad["gpu"](Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * OUT_FLAT):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
    for i in range(B * IN_FLAT):
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "AvgPool2D GPU vs CPU")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Pool2D legacy ↔ storage parity")
    print("=" * 70)
    test_max_cpu_parity()
    test_avg_cpu_parity()
    test_max_gpu_parity()
    test_avg_gpu_parity()
    print("ALL PASSED")
