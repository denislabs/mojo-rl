"""MinMaxNorm + SimNorm legacy ↔ storage parity (CPU) + storage GPU vs CPU.

Both leaves are param-less, so legacy↔storage CPU should be bit-identical on out
+ grad_input (same per-sample math). Run:
  pixi run mojo run -I . tests/nn/test_min_max_sim_norm_storage_parity.mojo
  pixi run -e apple mojo run -I . tests/nn/test_min_max_sim_norm_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.min_max_norm import MinMaxNorm as LegacyMinMaxNorm
from mojo_rl.nn.primitives.sim_norm import SimNorm as LegacySimNorm
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.min_max_norm import MinMaxNorm
from mojo_rl.nn.storage.primitives.sim_norm import SimNorm


comptime DIM = 10
comptime GROUPS = 2
comptime B = 6


# ──────────────────────────────────────────────────────────────────────
# MinMaxNorm
# ──────────────────────────────────────────────────────────────────────


def test_mmn_cpu_parity() raises:
    print("test_mmn_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)
    var leg = LegacyMinMaxNorm[DIM].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DIM)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DIM)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DIM)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DIM)
    for i in range(B * DIM):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.18
        go[i] = Scalar[DT]((i % 7) - 3) * 0.22

    var x_t = TileTensor(x, row_major[B, DIM]())
    var y_t = TileTensor(y, row_major[B, DIM]())
    var go_t = TileTensor(go, row_major[B, DIM]())
    var gi_t = TileTensor(gi, row_major[B, DIM]())
    leg.forward["cpu", B](x_t, output=y_t)
    leg.vjp["cpu", B](go_t, gi_t)

    var st = MinMaxNorm[DIM].make["cpu", Deterministic]()
    var sx = Tensor.alloc(B * DIM)
    var sgo = Tensor.alloc(B * DIM)
    var sout = Tensor.alloc(B * DIM)
    var sgi = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        sx.data[i] = x[i]
        sgo.data[i] = go[i]
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * DIM):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "MinMaxNorm CPU parity")
    print("  ok")


def test_mmn_gpu_parity() raises:
    print("test_mmn_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = MinMaxNorm[DIM].make["cpu", Deterministic]()
    var gpu = MinMaxNorm[DIM].make["gpu", Deterministic](Optional(c))

    var sx = Tensor.alloc(B * DIM)
    var sgo = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    var c_out = Tensor.alloc(B * DIM)
    var c_gi = Tensor.alloc(B * DIM)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
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
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * DIM):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "MinMaxNorm GPU vs CPU")
    print("  ok")


# ──────────────────────────────────────────────────────────────────────
# SimNorm
# ──────────────────────────────────────────────────────────────────────


def test_sn_cpu_parity() raises:
    print("test_sn_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)
    var leg = LegacySimNorm[DIM, GROUPS].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DIM)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DIM)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DIM)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DIM)
    for i in range(B * DIM):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.18
        go[i] = Scalar[DT]((i % 7) - 3) * 0.22

    var x_t = TileTensor(x, row_major[B, DIM]())
    var y_t = TileTensor(y, row_major[B, DIM]())
    var go_t = TileTensor(go, row_major[B, DIM]())
    var gi_t = TileTensor(gi, row_major[B, DIM]())
    leg.forward["cpu", B](x_t, output=y_t)
    leg.vjp["cpu", B](go_t, gi_t)

    var st = SimNorm[DIM, GROUPS].make["cpu", Deterministic]()
    var sx = Tensor.alloc(B * DIM)
    var sgo = Tensor.alloc(B * DIM)
    var sout = Tensor.alloc(B * DIM)
    var sgi = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        sx.data[i] = x[i]
        sgo.data[i] = go[i]
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * DIM):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "SimNorm CPU parity")
    print("  ok")


def test_sn_gpu_parity() raises:
    print("test_sn_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = SimNorm[DIM, GROUPS].make["cpu", Deterministic]()
    var gpu = SimNorm[DIM, GROUPS].make["gpu", Deterministic](Optional(c))

    var sx = Tensor.alloc(B * DIM)
    var sgo = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    var c_out = Tensor.alloc(B * DIM)
    var c_gi = Tensor.alloc(B * DIM)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
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
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * DIM):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "SimNorm GPU vs CPU")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("MinMaxNorm + SimNorm legacy ↔ storage parity")
    print("=" * 70)
    test_mmn_cpu_parity()
    test_sn_cpu_parity()
    test_mmn_gpu_parity()
    test_sn_gpu_parity()
    print("ALL PASSED")
