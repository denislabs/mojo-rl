"""BatchNorm2D legacy ↔ storage parity (CPU bit-identical) + storage GPU vs CPU.

Exercises the State (running stats), train/eval split, and the multi-block GPU
reduction (partial→finalize→scatter). Run:
  pixi run -e apple mojo run -I . tests/nn/test_batch_norm_2d_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.batch_norm_2d import BatchNorm2D as LegacyBatchNorm2D
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.batch_norm_2d import BatchNorm2D


comptime C = 3
comptime HH = 4
comptime WW = 4
comptime B = 6
comptime FLAT = C * HH * WW


def test_bn2d_cpu_parity() raises:
    print("test_bn2d_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)
    var leg = LegacyBatchNorm2D[C, HH, WW].make[target="cpu", INIT=Zero]()
    var lg = leg.gamma.value_unsafe_ptr_cpu()
    var lb = leg.beta.value_unsafe_ptr_cpu()
    for k in range(C):
        lg[k] = Scalar[DT](0.7 + 0.1 * Float64(k))
        lb[k] = Scalar[DT](-0.3 + 0.05 * Float64(k))

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * FLAT)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * FLAT)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * FLAT)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * FLAT)
    var ye: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * FLAT)
    for i in range(B * FLAT):
        x[i] = Scalar[DT]((i % 17) - 8) * 0.13
        go[i] = Scalar[DT]((i % 9) - 4) * 0.25

    var x_t = TileTensor(x, row_major[B, FLAT]())
    var y_t = TileTensor(y, row_major[B, FLAT]())
    var go_t = TileTensor(go, row_major[B, FLAT]())
    var gi_t = TileTensor(gi, row_major[B, FLAT]())
    var ye_t = TileTensor(ye, row_major[B, FLAT]())
    leg.forward["cpu", B](x_t, output=y_t)
    leg.zero_grad["cpu"]()
    leg.vjp["cpu", B](go_t, gi_t)
    leg.set_attr["training"](Scalar[DT](0.0))
    leg.forward["cpu", B](x_t, output=ye_t)

    var st = BatchNorm2D[C, HH, WW].make["cpu", Deterministic]()
    for k in range(C):
        st.gamma.val.data[k] = lg[k]
        st.beta.val.data[k] = lb[k]
    var sx = Tensor.alloc(B * FLAT)
    var sgo = Tensor.alloc(B * FLAT)
    var sout = Tensor.alloc(B * FLAT)
    var sgi = Tensor.alloc(B * FLAT)
    var soute = Tensor.alloc(B * FLAT)
    for i in range(B * FLAT):
        sx.data[i] = x[i]
        sgo.data[i] = go[i]
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.zero_grad["cpu"](None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)
    st.set_training(False)
    st.forward["cpu", B](TensorRefs[1](sx), soute, None)

    var mo: Scalar[DT] = 0
    var me: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * FLAT):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
        if abs(soute.data[i] - ye[i]) > me: me = abs(soute.data[i] - ye[i])
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    var mrm: Scalar[DT] = 0
    var mrv: Scalar[DT] = 0
    var mdg: Scalar[DT] = 0
    var mdb: Scalar[DT] = 0
    for f in range(C):
        if abs(st.running_mean.data[f] - leg.running_mean.t.cpu[f]) > mrm:
            mrm = abs(st.running_mean.data[f] - leg.running_mean.t.cpu[f])
        if abs(st.running_var.data[f] - leg.running_var.t.cpu[f]) > mrv:
            mrv = abs(st.running_var.data[f] - leg.running_var.t.cpu[f])
        if abs(st.gamma.grd.data[f] - leg.gamma.grd.cpu[f]) > mdg:
            mdg = abs(st.gamma.grd.data[f] - leg.gamma.grd.cpu[f])
        if abs(st.beta.grd.data[f] - leg.beta.grd.cpu[f]) > mdb:
            mdb = abs(st.beta.grd.data[f] - leg.beta.grd.cpu[f])
    print("  max Δ: out", mo, " gi", mgi, " rm", mrm, " rv", mrv,
          " dg", mdg, " db", mdb, " eval", me)
    assert_true(mo < TOL and me < TOL and mgi < TOL and mrm < TOL
                and mrv < TOL and mdg < TOL and mdb < TOL, "BN2D CPU parity")
    print("  ok")


def test_bn2d_gpu_parity() raises:
    print("test_bn2d_gpu_parity (storage GPU vs storage CPU) ...")
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
    gpu.running_var.download(c)
    gpu.gamma.grd.download(c)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * FLAT):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    var mrv: Scalar[DT] = 0
    var mdg: Scalar[DT] = 0
    for f in range(C):
        if abs(gpu.running_var.data[f] - cpu.running_var.data[f]) > mrv:
            mrv = abs(gpu.running_var.data[f] - cpu.running_var.data[f])
        if abs(gpu.gamma.grd.data[f] - cpu.gamma.grd.data[f]) > mdg:
            mdg = abs(gpu.gamma.grd.data[f] - cpu.gamma.grd.data[f])
    print("  max Δ: out", mo, " gi", mgi, " rv", mrv, " dg", mdg)
    assert_true(mo < TOL and mgi < TOL and mrv < TOL and mdg < TOL,
                "BN2D GPU vs CPU parity")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("BatchNorm2D legacy ↔ storage parity")
    print("=" * 70)
    test_bn2d_cpu_parity()
    test_bn2d_gpu_parity()
    print("ALL PASSED")
