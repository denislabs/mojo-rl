"""Slice / Sum / Mean / ReduceMax legacy ↔ storage parity (CPU) + storage GPU vs CPU.

Each leaf: legacy↔storage CPU is bit-identical (same loop/kernel math carried
over), and storage GPU matches storage CPU (TOL ~2e-5). ReduceMax is
forward-only — the vjp zero-fill is asserted on both surfaces. Run:
  pixi run mojo run -I . tests/nn/test_slice_reduce_storage_parity.mojo
  pixi run -e apple mojo run -I . tests/nn/test_slice_reduce_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.slice import Slice as LegacySlice
from mojo_rl.nn.primitives.reduce import Sum as LegacySum
from mojo_rl.nn.primitives.reduce import Mean as LegacyMean
from mojo_rl.nn.primitives.reduce_max import ReduceMax as LegacyReduceMax
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.slice import Slice
from mojo_rl.nn.storage.primitives.reduce import Sum, Mean
from mojo_rl.nn.storage.primitives.reduce_max import ReduceMax


comptime DIM = 10
comptime B = 6


# ════════════════════════════════════════════════════════════════════════
# Slice
# ════════════════════════════════════════════════════════════════════════
comptime S_START = 3
comptime S_END = 8
comptime S_OUT = S_END - S_START


def test_slice_cpu_parity() raises:
    print("test_slice_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)

    var leg = LegacySlice[DIM, S_START, S_END].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DIM)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * S_OUT)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * S_OUT)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DIM)
    for i in range(B * DIM):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(B * S_OUT):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.22
    # prefill gi with junk to confirm the zero-fill happens.
    for i in range(B * DIM):
        gi[i] = Scalar[DT](42.0)

    var x_t = TileTensor(x, row_major[B, DIM]())
    var y_t = TileTensor(y, row_major[B, S_OUT]())
    var go_t = TileTensor(go, row_major[B, S_OUT]())
    var gi_t = TileTensor(gi, row_major[B, DIM]())
    leg.forward["cpu", B](x_t, output=y_t)
    leg.vjp["cpu", B](go_t, gi_t)

    var st = Slice[DIM, S_START, S_END].make["cpu", Deterministic]()
    var sx = Tensor.alloc(B * DIM)
    var sgo = Tensor.alloc(B * S_OUT)
    var sout = Tensor.alloc(B * S_OUT)
    var sgi = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        sx.data[i] = x[i]
    for i in range(B * S_OUT):
        sgo.data[i] = go[i]
    for i in range(B * DIM):
        sgi.data[i] = Scalar[DT](42.0)
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    for i in range(B * S_OUT):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
    var mgi: Scalar[DT] = 0
    for i in range(B * DIM):
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "Slice CPU parity")
    print("  ok")


def test_slice_gpu_parity() raises:
    print("test_slice_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = Slice[DIM, S_START, S_END].make["cpu", Deterministic]()
    var gpu = Slice[DIM, S_START, S_END].make["gpu", Deterministic](Optional(c))

    var sx = Tensor.alloc(B * DIM)
    var sgo = Tensor.alloc(B * S_OUT)
    for i in range(B * DIM):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(B * S_OUT):
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    var c_out = Tensor.alloc(B * S_OUT)
    var c_gi = Tensor.alloc(B * DIM)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(B * DIM)
    var ggo = Tensor.alloc(B * S_OUT)
    for i in range(B * DIM):
        gx.data[i] = sx.data[i]
    for i in range(B * S_OUT):
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(B * S_OUT)
    var g_gi = Tensor.alloc(B * DIM)
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)

    var mo: Scalar[DT] = 0
    for i in range(B * S_OUT):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
    var mgi: Scalar[DT] = 0
    for i in range(B * DIM):
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "Slice GPU vs CPU")
    print("  ok")


# ════════════════════════════════════════════════════════════════════════
# Sum  (Reduce[DIM, SumOp])
# ════════════════════════════════════════════════════════════════════════
def test_sum_cpu_parity() raises:
    print("test_sum_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)

    var leg = LegacySum[DIM].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DIM)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DIM)
    for i in range(B * DIM):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(B):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.22

    var x_t = TileTensor(x, row_major[B, DIM]())
    var y_t = TileTensor(y, row_major[B, 1]())
    var go_t = TileTensor(go, row_major[B, 1]())
    var gi_t = TileTensor(gi, row_major[B, DIM]())
    leg.forward["cpu", B](x_t, output=y_t)
    leg.vjp["cpu", B](go_t, gi_t)

    var st = Sum[DIM].make["cpu", Deterministic]()
    var sx = Tensor.alloc(B * DIM)
    var sgo = Tensor.alloc(B)
    var sout = Tensor.alloc(B)
    var sgi = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        sx.data[i] = x[i]
    for i in range(B):
        sgo.data[i] = go[i]
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    for i in range(B):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
    var mgi: Scalar[DT] = 0
    for i in range(B * DIM):
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "Sum CPU parity")
    print("  ok")


def test_sum_gpu_parity() raises:
    print("test_sum_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = Sum[DIM].make["cpu", Deterministic]()
    var gpu = Sum[DIM].make["gpu", Deterministic](Optional(c))

    var sx = Tensor.alloc(B * DIM)
    var sgo = Tensor.alloc(B)
    for i in range(B * DIM):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(B):
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    var c_out = Tensor.alloc(B)
    var c_gi = Tensor.alloc(B * DIM)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(B * DIM)
    var ggo = Tensor.alloc(B)
    for i in range(B * DIM):
        gx.data[i] = sx.data[i]
    for i in range(B):
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(B)
    var g_gi = Tensor.alloc(B * DIM)
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)

    var mo: Scalar[DT] = 0
    for i in range(B):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
    var mgi: Scalar[DT] = 0
    for i in range(B * DIM):
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "Sum GPU vs CPU")
    print("  ok")


# ════════════════════════════════════════════════════════════════════════
# Mean  (Reduce[DIM, MeanOp])
# ════════════════════════════════════════════════════════════════════════
def test_mean_cpu_parity() raises:
    print("test_mean_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)

    var leg = LegacyMean[DIM].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DIM)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * DIM)
    for i in range(B * DIM):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(B):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.22

    var x_t = TileTensor(x, row_major[B, DIM]())
    var y_t = TileTensor(y, row_major[B, 1]())
    var go_t = TileTensor(go, row_major[B, 1]())
    var gi_t = TileTensor(gi, row_major[B, DIM]())
    leg.forward["cpu", B](x_t, output=y_t)
    leg.vjp["cpu", B](go_t, gi_t)

    var st = Mean[DIM].make["cpu", Deterministic]()
    var sx = Tensor.alloc(B * DIM)
    var sgo = Tensor.alloc(B)
    var sout = Tensor.alloc(B)
    var sgi = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        sx.data[i] = x[i]
    for i in range(B):
        sgo.data[i] = go[i]
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    for i in range(B):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
    var mgi: Scalar[DT] = 0
    for i in range(B * DIM):
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "Mean CPU parity")
    print("  ok")


def test_mean_gpu_parity() raises:
    print("test_mean_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = Mean[DIM].make["cpu", Deterministic]()
    var gpu = Mean[DIM].make["gpu", Deterministic](Optional(c))

    var sx = Tensor.alloc(B * DIM)
    var sgo = Tensor.alloc(B)
    for i in range(B * DIM):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(B):
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    var c_out = Tensor.alloc(B)
    var c_gi = Tensor.alloc(B * DIM)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(B * DIM)
    var ggo = Tensor.alloc(B)
    for i in range(B * DIM):
        gx.data[i] = sx.data[i]
    for i in range(B):
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(B)
    var g_gi = Tensor.alloc(B * DIM)
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)

    var mo: Scalar[DT] = 0
    for i in range(B):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
    var mgi: Scalar[DT] = 0
    for i in range(B * DIM):
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "Mean GPU vs CPU")
    print("  ok")


# ════════════════════════════════════════════════════════════════════════
# ReduceMax (forward-only; vjp zero-fills)
# ════════════════════════════════════════════════════════════════════════
comptime RNA = 9
comptime RB = 7


def test_reduce_max_cpu_parity() raises:
    print("test_reduce_max_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)

    var leg = LegacyReduceMax[RNA].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](RB * RNA)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](RB)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](RB)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](RB * RNA)
    for i in range(RB * RNA):
        x[i] = Scalar[DT]((i % 17) - 8) * 0.31
    for i in range(RB):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.22
    for i in range(RB * RNA):
        gi[i] = Scalar[DT](99.0)

    var x_t = TileTensor(x, row_major[RB, RNA]())
    var y_t = TileTensor(y, row_major[RB, 1]())
    var go_t = TileTensor(go, row_major[RB, 1]())
    var gi_t = TileTensor(gi, row_major[RB, RNA]())
    leg.forward["cpu", RB](x_t, output=y_t)
    leg.vjp["cpu", RB](go_t, gi_t)

    var st = ReduceMax[RNA].make["cpu", Deterministic]()
    var sx = Tensor.alloc(RB * RNA)
    var sgo = Tensor.alloc(RB)
    var sout = Tensor.alloc(RB)
    var sgi = Tensor.alloc(RB * RNA)
    for i in range(RB * RNA):
        sx.data[i] = x[i]
    for i in range(RB):
        sgo.data[i] = go[i]
    for i in range(RB * RNA):
        sgi.data[i] = Scalar[DT](99.0)
    st.forward["cpu", RB](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", RB](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    for i in range(RB):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
    var mgi: Scalar[DT] = 0
    for i in range(RB * RNA):
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    print("  max Δ: out", mo, " grad_input(zero)", mgi)
    assert_true(mo < TOL and mgi < TOL, "ReduceMax CPU parity")
    for i in range(RB * RNA):
        assert_true(sgi.data[i] == Scalar[DT](0.0), "grad_input zero")
    print("  ok")


def test_reduce_max_gpu_parity() raises:
    print("test_reduce_max_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = ReduceMax[RNA].make["cpu", Deterministic]()
    var gpu = ReduceMax[RNA].make["gpu", Deterministic](Optional(c))

    var sx = Tensor.alloc(RB * RNA)
    var sgo = Tensor.alloc(RB)
    for i in range(RB * RNA):
        sx.data[i] = Scalar[DT]((i % 17) - 8) * 0.31
    for i in range(RB):
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    var c_out = Tensor.alloc(RB)
    var c_gi = Tensor.alloc(RB * RNA)
    cpu.forward["cpu", RB](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", RB](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(RB * RNA)
    var ggo = Tensor.alloc(RB)
    for i in range(RB * RNA):
        gx.data[i] = sx.data[i]
    for i in range(RB):
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(RB)
    var g_gi = Tensor.alloc(RB * RNA)
    gpu.forward["gpu", RB](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", RB](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)

    var mo: Scalar[DT] = 0
    for i in range(RB):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
    var mgi: Scalar[DT] = 0
    for i in range(RB * RNA):
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    print("  max Δ: out", mo, " grad_input(zero)", mgi)
    assert_true(mo < TOL and mgi < TOL, "ReduceMax GPU vs CPU")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Slice / Sum / Mean / ReduceMax legacy ↔ storage parity")
    print("=" * 70)
    test_slice_cpu_parity()
    test_slice_gpu_parity()
    test_sum_cpu_parity()
    test_sum_gpu_parity()
    test_mean_cpu_parity()
    test_mean_gpu_parity()
    test_reduce_max_cpu_parity()
    test_reduce_max_gpu_parity()
    print("ALL PASSED")
