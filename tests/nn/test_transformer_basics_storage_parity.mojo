"""BiasAdd / Transpose2D / TokenMean legacy ↔ storage parity (CPU) + storage GPU vs CPU.

Each leaf: legacy↔storage CPU is bit-identical (same loop/kernel math carried
over), and storage GPU matches storage CPU (TOL ~2e-5). BiasAdd additionally
checks the bias grad. Run:
  pixi run mojo run -I . tests/nn/test_transformer_basics_storage_parity.mojo
  pixi run -e apple mojo run -I . tests/nn/test_transformer_basics_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.bias_add import BiasAdd as LegacyBiasAdd
from mojo_rl.nn.primitives.transpose_2d import Transpose2D as LegacyTranspose2D
from mojo_rl.nn.primitives.token_mean import TokenMean as LegacyTokenMean
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.bias_add import BiasAdd
from mojo_rl.nn.storage.primitives.transpose_2d import Transpose2D
from mojo_rl.nn.storage.primitives.token_mean import TokenMean


comptime B = 6


# ════════════════════════════════════════════════════════════════════════
# BiasAdd[DIM]  (param: bias)
# ════════════════════════════════════════════════════════════════════════
comptime BA_DIM = 11


def _fill_bias(mut p: Tensor, n: Int):
    # Non-zero bias so forward + grad-input copy are exercised meaningfully.
    for i in range(n):
        p.data[i] = Scalar[DT]((i % 5) - 2) * 0.3


def test_bias_add_cpu_parity() raises:
    print("test_bias_add_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)

    var leg = LegacyBiasAdd[BA_DIM].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * BA_DIM)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * BA_DIM)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * BA_DIM)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * BA_DIM)
    for i in range(B * BA_DIM):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(B * BA_DIM):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.22
    for i in range(B * BA_DIM):
        gi[i] = Scalar[DT](42.0)
    # set legacy bias to the same pattern as the storage leaf
    for i in range(BA_DIM):
        leg.bias.val.cpu[i] = Scalar[DT]((i % 5) - 2) * 0.3

    var x_t = TileTensor(x, row_major[B, BA_DIM]())
    var y_t = TileTensor(y, row_major[B, BA_DIM]())
    var go_t = TileTensor(go, row_major[B, BA_DIM]())
    var gi_t = TileTensor(gi, row_major[B, BA_DIM]())
    leg.forward["cpu", B](x_t, output=y_t)
    leg.zero_grad["cpu"]()
    leg.vjp["cpu", B](go_t, gi_t)

    var st = BiasAdd[BA_DIM].make["cpu", Deterministic]()
    _fill_bias(st.bias.val, BA_DIM)
    var sx = Tensor.alloc(B * BA_DIM)
    var sgo = Tensor.alloc(B * BA_DIM)
    var sout = Tensor.alloc(B * BA_DIM)
    var sgi = Tensor.alloc(B * BA_DIM)
    for i in range(B * BA_DIM):
        sx.data[i] = x[i]
    for i in range(B * BA_DIM):
        sgo.data[i] = go[i]
    for i in range(B * BA_DIM):
        sgi.data[i] = Scalar[DT](42.0)
    st.zero_grad["cpu"](None)
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    for i in range(B * BA_DIM):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
    var mgi: Scalar[DT] = 0
    for i in range(B * BA_DIM):
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    var mgb: Scalar[DT] = 0
    for i in range(BA_DIM):
        var d = abs(st.bias.grd.data[i] - leg.bias.grd.cpu[i])
        if d > mgb: mgb = d
    print("  max Δ: out", mo, " gi", mgi, " gbias", mgb)
    assert_true(mo < TOL and mgi < TOL and mgb < TOL, "BiasAdd CPU parity")
    print("  ok")


def test_bias_add_gpu_parity() raises:
    print("test_bias_add_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = BiasAdd[BA_DIM].make["cpu", Deterministic]()
    _fill_bias(cpu.bias.val, BA_DIM)
    var gpu = BiasAdd[BA_DIM].make["gpu", Deterministic](Optional(c))
    _fill_bias(gpu.bias.val, BA_DIM)
    gpu.bias.val.upload(c)

    var sx = Tensor.alloc(B * BA_DIM)
    var sgo = Tensor.alloc(B * BA_DIM)
    for i in range(B * BA_DIM):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(B * BA_DIM):
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    var c_out = Tensor.alloc(B * BA_DIM)
    var c_gi = Tensor.alloc(B * BA_DIM)
    cpu.zero_grad["cpu"](None)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(B * BA_DIM)
    var ggo = Tensor.alloc(B * BA_DIM)
    for i in range(B * BA_DIM):
        gx.data[i] = sx.data[i]
    for i in range(B * BA_DIM):
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(B * BA_DIM)
    var g_gi = Tensor.alloc(B * BA_DIM)
    gpu.zero_grad["gpu"](Optional(c))
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)
    gpu.bias.grd.download(c)

    var mo: Scalar[DT] = 0
    for i in range(B * BA_DIM):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
    var mgi: Scalar[DT] = 0
    for i in range(B * BA_DIM):
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    var mgb: Scalar[DT] = 0
    for i in range(BA_DIM):
        var d = abs(gpu.bias.grd.data[i] - cpu.bias.grd.data[i])
        if d > mgb: mgb = d
    print("  max Δ: out", mo, " gi", mgi, " gbias", mgb)
    assert_true(mo < TOL and mgi < TOL and mgb < TOL, "BiasAdd GPU vs CPU")
    print("  ok")


# ════════════════════════════════════════════════════════════════════════
# Transpose2D[A, B]
# ════════════════════════════════════════════════════════════════════════
comptime TA = 4
comptime TB = 7
comptime T_AB = TA * TB


def test_transpose_cpu_parity() raises:
    print("test_transpose_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)

    var leg = LegacyTranspose2D[TA, TB].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * T_AB)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * T_AB)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * T_AB)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * T_AB)
    for i in range(B * T_AB):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(B * T_AB):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.22

    var x_t = TileTensor(x, row_major[B, T_AB]())
    var y_t = TileTensor(y, row_major[B, T_AB]())
    var go_t = TileTensor(go, row_major[B, T_AB]())
    var gi_t = TileTensor(gi, row_major[B, T_AB]())
    leg.forward["cpu", B](x_t, output=y_t)
    leg.vjp["cpu", B](go_t, gi_t)

    var st = Transpose2D[TA, TB].make["cpu", Deterministic]()
    var sx = Tensor.alloc(B * T_AB)
    var sgo = Tensor.alloc(B * T_AB)
    var sout = Tensor.alloc(B * T_AB)
    var sgi = Tensor.alloc(B * T_AB)
    for i in range(B * T_AB):
        sx.data[i] = x[i]
    for i in range(B * T_AB):
        sgo.data[i] = go[i]
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    for i in range(B * T_AB):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
    var mgi: Scalar[DT] = 0
    for i in range(B * T_AB):
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "Transpose2D CPU parity")
    print("  ok")


def test_transpose_gpu_parity() raises:
    print("test_transpose_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = Transpose2D[TA, TB].make["cpu", Deterministic]()
    var gpu = Transpose2D[TA, TB].make["gpu", Deterministic](Optional(c))

    var sx = Tensor.alloc(B * T_AB)
    var sgo = Tensor.alloc(B * T_AB)
    for i in range(B * T_AB):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(B * T_AB):
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    var c_out = Tensor.alloc(B * T_AB)
    var c_gi = Tensor.alloc(B * T_AB)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(B * T_AB)
    var ggo = Tensor.alloc(B * T_AB)
    for i in range(B * T_AB):
        gx.data[i] = sx.data[i]
    for i in range(B * T_AB):
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(B * T_AB)
    var g_gi = Tensor.alloc(B * T_AB)
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)

    var mo: Scalar[DT] = 0
    for i in range(B * T_AB):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
    var mgi: Scalar[DT] = 0
    for i in range(B * T_AB):
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "Transpose2D GPU vs CPU")
    print("  ok")


# ════════════════════════════════════════════════════════════════════════
# TokenMean[SEQ_LEN, DIM]
# ════════════════════════════════════════════════════════════════════════
comptime TM_SEQ = 5
comptime TM_DIM = 8
comptime TM_IN = TM_SEQ * TM_DIM


def test_token_mean_cpu_parity() raises:
    print("test_token_mean_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)

    var leg = LegacyTokenMean[TM_SEQ, TM_DIM].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * TM_IN)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * TM_DIM)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * TM_DIM)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * TM_IN)
    for i in range(B * TM_IN):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(B * TM_DIM):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.22

    var x_t = TileTensor(x, row_major[B, TM_IN]())
    var y_t = TileTensor(y, row_major[B, TM_DIM]())
    var go_t = TileTensor(go, row_major[B, TM_DIM]())
    var gi_t = TileTensor(gi, row_major[B, TM_IN]())
    leg.forward["cpu", B](x_t, output=y_t)
    leg.vjp["cpu", B](go_t, gi_t)

    var st = TokenMean[TM_SEQ, TM_DIM].make["cpu", Deterministic]()
    var sx = Tensor.alloc(B * TM_IN)
    var sgo = Tensor.alloc(B * TM_DIM)
    var sout = Tensor.alloc(B * TM_DIM)
    var sgi = Tensor.alloc(B * TM_IN)
    for i in range(B * TM_IN):
        sx.data[i] = x[i]
    for i in range(B * TM_DIM):
        sgo.data[i] = go[i]
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    for i in range(B * TM_DIM):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
    var mgi: Scalar[DT] = 0
    for i in range(B * TM_IN):
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "TokenMean CPU parity")
    print("  ok")


def test_token_mean_gpu_parity() raises:
    print("test_token_mean_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = TokenMean[TM_SEQ, TM_DIM].make["cpu", Deterministic]()
    var gpu = TokenMean[TM_SEQ, TM_DIM].make["gpu", Deterministic](Optional(c))

    var sx = Tensor.alloc(B * TM_IN)
    var sgo = Tensor.alloc(B * TM_DIM)
    for i in range(B * TM_IN):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(B * TM_DIM):
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    var c_out = Tensor.alloc(B * TM_DIM)
    var c_gi = Tensor.alloc(B * TM_IN)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(B * TM_IN)
    var ggo = Tensor.alloc(B * TM_DIM)
    for i in range(B * TM_IN):
        gx.data[i] = sx.data[i]
    for i in range(B * TM_DIM):
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(B * TM_DIM)
    var g_gi = Tensor.alloc(B * TM_IN)
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)

    var mo: Scalar[DT] = 0
    for i in range(B * TM_DIM):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
    var mgi: Scalar[DT] = 0
    for i in range(B * TM_IN):
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "TokenMean GPU vs CPU")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("BiasAdd / Transpose2D / TokenMean legacy ↔ storage parity")
    print("=" * 70)
    test_bias_add_cpu_parity()
    test_bias_add_gpu_parity()
    test_transpose_cpu_parity()
    test_transpose_gpu_parity()
    test_token_mean_cpu_parity()
    test_token_mean_gpu_parity()
    print("ALL PASSED")
