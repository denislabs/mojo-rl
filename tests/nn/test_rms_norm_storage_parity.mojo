"""RMSNorm legacy ↔ storage parity (CPU bit-identical) + storage GPU vs CPU.

Run: pixi run -e apple mojo run -I . tests/nn/test_rms_norm_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.rms_norm import RMSNorm as LegacyRMSNorm
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.rms_norm import RMSNorm


comptime DIM = 10
comptime B = 6


def test_rms_cpu_parity() raises:
    print("test_rms_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)
    var leg = LegacyRMSNorm[DIM].make[target="cpu", INIT=Zero]()
    var lg = leg.gamma.value_unsafe_ptr_cpu()
    for k in range(DIM):
        lg[k] = Scalar[DT](0.6 + 0.07 * Float64(k))

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
    leg.zero_grad["cpu"]()
    leg.vjp["cpu", B](go_t, gi_t)

    var st = RMSNorm[DIM].make["cpu", Deterministic]()
    for k in range(DIM):
        st.gamma.val.data[k] = lg[k]
    var sx = Tensor.alloc(B * DIM)
    var sgo = Tensor.alloc(B * DIM)
    var sout = Tensor.alloc(B * DIM)
    var sgi = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        sx.data[i] = x[i]
        sgo.data[i] = go[i]
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.zero_grad["cpu"](None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * DIM):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    var mdg: Scalar[DT] = 0
    for k in range(DIM):
        if abs(st.gamma.grd.data[k] - leg.gamma.grd.cpu[k]) > mdg:
            mdg = abs(st.gamma.grd.data[k] - leg.gamma.grd.cpu[k])
    print("  max Δ: out", mo, " gi", mgi, " dg", mdg)
    assert_true(mo < TOL and mgi < TOL and mdg < TOL, "RMSNorm CPU parity")
    print("  ok")


def test_rms_gpu_parity() raises:
    print("test_rms_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = RMSNorm[DIM].make["cpu", Deterministic]()
    var gpu = RMSNorm[DIM].make["gpu", Deterministic](Optional(c))
    for k in range(DIM):
        cpu.gamma.val.data[k] = Scalar[DT](0.6 + 0.07 * Float64(k))
        gpu.gamma.val.data[k] = cpu.gamma.val.data[k]
    gpu.gamma.val.upload(c)

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
    assert_true(mo < TOL and mgi < TOL and mdg < TOL, "RMSNorm GPU vs CPU")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("RMSNorm legacy ↔ storage parity")
    print("=" * 70)
    test_rms_cpu_parity()
    test_rms_gpu_parity()
    print("ALL PASSED")
