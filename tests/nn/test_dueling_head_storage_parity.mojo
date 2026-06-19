"""DuelingHead / DuelingHeadC51 legacy ↔ storage parity (CPU bit-identical)
+ storage GPU vs CPU.

Both leaves are pure no-param aggregations (Q = V + (A − mean_a A), per-atom for
the C51 variant), so legacy↔storage CPU is bit-identical on out + grad_input
(no param grads — the heads have no Params). Run:
  pixi run mojo run -I . tests/nn/test_dueling_head_storage_parity.mojo
  pixi run -e apple mojo run -I . tests/nn/test_dueling_head_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.dueling_head import DuelingHead as LegacyDuelingHead
from mojo_rl.nn.primitives.dueling_head_c51 import (
    DuelingHeadC51 as LegacyDuelingHeadC51,
)
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.dueling_head import DuelingHead
from mojo_rl.nn.storage.primitives.dueling_head_c51 import DuelingHeadC51


# ════════════════════════════════════════════════════════════════════════
# DuelingHead[NA]
# ════════════════════════════════════════════════════════════════════════
comptime NA = 5
comptime B = 6


def test_dueling_cpu_parity() raises:
    print("test_dueling_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)
    comptime IN = NA + 1
    comptime OUT = NA

    var leg = LegacyDuelingHead[NA].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * IN)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * OUT)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * OUT)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * IN)
    for i in range(B * IN):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(B * OUT):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.22

    var x_t = TileTensor(x, row_major[B, IN]())
    var y_t = TileTensor(y, row_major[B, OUT]())
    var go_t = TileTensor(go, row_major[B, OUT]())
    var gi_t = TileTensor(gi, row_major[B, IN]())
    leg.forward["cpu", B](x_t, output=y_t)
    leg.vjp["cpu", B](go_t, gi_t)

    var st = DuelingHead[NA].make["cpu", Deterministic]()
    var sx = Tensor.alloc(B * IN)
    var sgo = Tensor.alloc(B * OUT)
    var sout = Tensor.alloc(B * OUT)
    var sgi = Tensor.alloc(B * IN)
    for i in range(B * IN):
        sx.data[i] = x[i]
    for i in range(B * OUT):
        sgo.data[i] = go[i]
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * OUT):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
    for i in range(B * IN):
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "DuelingHead CPU parity")
    print("  ok")


def test_dueling_gpu_parity() raises:
    print("test_dueling_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    comptime IN = NA + 1
    comptime OUT = NA
    var c = DeviceContext()
    var cpu = DuelingHead[NA].make["cpu", Deterministic]()
    var gpu = DuelingHead[NA].make["gpu", Deterministic](Optional(c))

    var sx = Tensor.alloc(B * IN)
    var sgo = Tensor.alloc(B * OUT)
    for i in range(B * IN):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(B * OUT):
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    var c_out = Tensor.alloc(B * OUT)
    var c_gi = Tensor.alloc(B * IN)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
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
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * OUT):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
    for i in range(B * IN):
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "DuelingHead GPU vs CPU")
    print("  ok")


# ════════════════════════════════════════════════════════════════════════
# DuelingHeadC51[NA, N_ATOMS]
# ════════════════════════════════════════════════════════════════════════
comptime C_NA = 4
comptime C_ATOMS = 3
comptime C_B = 5


def test_dueling_c51_cpu_parity() raises:
    print("test_dueling_c51_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)
    comptime IN = (1 + C_NA) * C_ATOMS
    comptime OUT = C_NA * C_ATOMS

    var leg = LegacyDuelingHeadC51[C_NA, C_ATOMS].make[
        target="cpu", INIT=Zero
    ]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](C_B * IN)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](C_B * OUT)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](C_B * OUT)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](C_B * IN)
    for i in range(C_B * IN):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(C_B * OUT):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.22

    var x_t = TileTensor(x, row_major[C_B, IN]())
    var y_t = TileTensor(y, row_major[C_B, OUT]())
    var go_t = TileTensor(go, row_major[C_B, OUT]())
    var gi_t = TileTensor(gi, row_major[C_B, IN]())
    leg.forward["cpu", C_B](x_t, output=y_t)
    leg.vjp["cpu", C_B](go_t, gi_t)

    var st = DuelingHeadC51[C_NA, C_ATOMS].make["cpu", Deterministic]()
    var sx = Tensor.alloc(C_B * IN)
    var sgo = Tensor.alloc(C_B * OUT)
    var sout = Tensor.alloc(C_B * OUT)
    var sgi = Tensor.alloc(C_B * IN)
    for i in range(C_B * IN):
        sx.data[i] = x[i]
    for i in range(C_B * OUT):
        sgo.data[i] = go[i]
    st.forward["cpu", C_B](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", C_B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(C_B * OUT):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
    for i in range(C_B * IN):
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "DuelingHeadC51 CPU parity")
    print("  ok")


def test_dueling_c51_gpu_parity() raises:
    print("test_dueling_c51_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    comptime IN = (1 + C_NA) * C_ATOMS
    comptime OUT = C_NA * C_ATOMS
    var c = DeviceContext()
    var cpu = DuelingHeadC51[C_NA, C_ATOMS].make["cpu", Deterministic]()
    var gpu = DuelingHeadC51[C_NA, C_ATOMS].make["gpu", Deterministic](
        Optional(c)
    )

    var sx = Tensor.alloc(C_B * IN)
    var sgo = Tensor.alloc(C_B * OUT)
    for i in range(C_B * IN):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(C_B * OUT):
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    var c_out = Tensor.alloc(C_B * OUT)
    var c_gi = Tensor.alloc(C_B * IN)
    cpu.forward["cpu", C_B](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", C_B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(C_B * IN)
    var ggo = Tensor.alloc(C_B * OUT)
    for i in range(C_B * IN):
        gx.data[i] = sx.data[i]
    for i in range(C_B * OUT):
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(C_B * OUT)
    var g_gi = Tensor.alloc(C_B * IN)
    gpu.forward["gpu", C_B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", C_B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(C_B * OUT):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
    for i in range(C_B * IN):
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "DuelingHeadC51 GPU vs CPU")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("DuelingHead / DuelingHeadC51 legacy ↔ storage parity")
    print("=" * 70)
    test_dueling_cpu_parity()
    test_dueling_c51_cpu_parity()
    test_dueling_gpu_parity()
    test_dueling_c51_gpu_parity()
    print("ALL PASSED")
