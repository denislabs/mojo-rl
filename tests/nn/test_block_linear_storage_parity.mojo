"""BlockLinear legacy ↔ storage parity (CPU) + storage GPU vs CPU.

CPU: legacy BlockLinear vs storage BlockLinear with identical weights/bias/
input — max|Δ| < 1e-6 on out + grad_input + weight.grd + bias.grd.
GPU: storage GPU vs storage CPU, TOL ~2e-5. Run both:
  rm -f mojo_rl.mojoc && pixi run mojo run -I . tests/nn/test_block_linear_storage_parity.mojo
  rm -f mojo_rl.mojoc && pixi run -e apple mojo run -I . tests/nn/test_block_linear_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.block_linear import BlockLinear as LegacyBlockLinear
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.block_linear import BlockLinear


comptime IN = 12
comptime OUT = 8
comptime BLOCKS = 4
comptime B = 6


def test_bl_cpu_parity() raises:
    print("test_bl_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)
    comptime W = BlockLinear[IN, OUT, BLOCKS].W_SIZE

    var leg = LegacyBlockLinear[IN, OUT, BLOCKS].make[target="cpu", INIT=Zero]()
    var lw = leg.weight.value_unsafe_ptr_cpu()
    var lb = leg.bias.value_unsafe_ptr_cpu()
    for k in range(W):
        lw[k] = Scalar[DT](0.3 - 0.013 * Float64(k % 11))
    for k in range(OUT):
        lb[k] = Scalar[DT](-0.2 + 0.05 * Float64(k))

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
    leg.zero_grad["cpu"]()
    leg.vjp["cpu", B](go_t, gi_t)

    var st = BlockLinear[IN, OUT, BLOCKS].make["cpu", Deterministic]()
    for k in range(W):
        st.weight.val.data[k] = lw[k]
    for k in range(OUT):
        st.bias.val.data[k] = lb[k]
    var sx = Tensor.alloc(B * IN)
    var sgo = Tensor.alloc(B * OUT)
    var sout = Tensor.alloc(B * OUT)
    var sgi = Tensor.alloc(B * IN)
    for i in range(B * IN):
        sx.data[i] = x[i]
    for i in range(B * OUT):
        sgo.data[i] = go[i]
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.zero_grad["cpu"](None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * OUT):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
    for i in range(B * IN):
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    var mdw: Scalar[DT] = 0
    for k in range(W):
        if abs(st.weight.grd.data[k] - leg.weight.grd.cpu[k]) > mdw:
            mdw = abs(st.weight.grd.data[k] - leg.weight.grd.cpu[k])
    var mdb: Scalar[DT] = 0
    for k in range(OUT):
        if abs(st.bias.grd.data[k] - leg.bias.grd.cpu[k]) > mdb:
            mdb = abs(st.bias.grd.data[k] - leg.bias.grd.cpu[k])
    print("  max Δ: out", mo, " gi", mgi, " dw", mdw, " db", mdb)
    assert_true(mo < TOL and mgi < TOL and mdw < TOL and mdb < TOL,
                "BlockLinear CPU parity")
    print("  ok")


def test_bl_gpu_parity() raises:
    print("test_bl_gpu_parity (storage GPU vs storage CPU) ...")
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
    print("BlockLinear legacy ↔ storage parity")
    print("=" * 70)
    test_bl_cpu_parity()
    test_bl_gpu_parity()
    print("ALL PASSED")
