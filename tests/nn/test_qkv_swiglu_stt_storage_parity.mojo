"""QKVToMajor / SwiGLU / SpaceTimeTranspose legacy ↔ storage parity.

Each leaf carries the same permutation / gated-activation kernels VERBATIM, so
legacy↔storage CPU is bit-identical (out + grad_inputs), and storage GPU↔CPU
agrees to a small tolerance. All three are ARITY-1, param-free. Run:
  rm -f mojo_rl.mojoc && pixi run mojo run -I . tests/nn/test_qkv_swiglu_stt_storage_parity.mojo
  rm -f mojo_rl.mojoc && pixi run -e apple mojo run -I . tests/nn/test_qkv_swiglu_stt_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.primitives.qkv_to_major import QKVToMajor as LegacyQKV
from mojo_rl.nn.primitives.swiglu import SwiGLU as LegacySwiGLU
from mojo_rl.nn.primitives.space_time_transpose import (
    SpaceTimeTranspose as LegacySTT,
)

from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.qkv_to_major import QKVToMajor
from mojo_rl.nn.storage.primitives.swiglu import SwiGLU
from mojo_rl.nn.storage.primitives.space_time_transpose import SpaceTimeTranspose


comptime CPU_TOL = Scalar[DT](1e-6)
comptime GPU_TOL = Scalar[DT](2e-5)


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _maxabs(a: List[Scalar[DT]], b: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) -> Scalar[DT]:
    var m: Scalar[DT] = 0
    for i in range(n):
        if abs(a[i] - b[i]) > m:
            m = abs(a[i] - b[i])
    return m


def _maxabs2(a: List[Scalar[DT]], b: List[Scalar[DT]], n: Int) -> Scalar[DT]:
    var m: Scalar[DT] = 0
    for i in range(n):
        if abs(a[i] - b[i]) > m:
            m = abs(a[i] - b[i])
    return m


# ════════════════════════════════════════════════════════════════════════
# QKVToMajor   (SEQ, DIM ; IN=OUT=3*SEQ*DIM)
# ════════════════════════════════════════════════════════════════════════
comptime Q_SEQ = 3
comptime Q_DIM = 4
comptime Q_B = 2
comptime Q_W = 3 * Q_SEQ * Q_DIM


def test_qkv_cpu_parity() raises:
    print("  qkv cpu parity ...")
    var leg = LegacyQKV[Q_SEQ, Q_DIM].make[target="cpu", INIT=Zero]()
    var x = _alloc(Q_B * Q_W)
    var y = _alloc(Q_B * Q_W)
    var go = _alloc(Q_B * Q_W)
    var gi = _alloc(Q_B * Q_W)
    for i in range(Q_B * Q_W):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.13
        go[i] = Scalar[DT]((i % 7) - 3) * 0.21
    var x_t = TileTensor(x, row_major[Q_B, Q_W]())
    var y_t = TileTensor(y, row_major[Q_B, Q_W]())
    var go_t = TileTensor(go, row_major[Q_B, Q_W]())
    var gi_t = TileTensor(gi, row_major[Q_B, Q_W]())
    leg.forward["cpu", Q_B](x_t, output=y_t)
    leg.vjp["cpu", Q_B](go_t, gi_t)

    var st = QKVToMajor[Q_SEQ, Q_DIM].make["cpu", Deterministic]()
    var sx = Tensor.alloc(Q_B * Q_W)
    var sgo = Tensor.alloc(Q_B * Q_W)
    var sout = Tensor.alloc(Q_B * Q_W)
    var sgi = Tensor.alloc(Q_B * Q_W)
    for i in range(Q_B * Q_W):
        sx.data[i] = x[i]
        sgo.data[i] = go[i]
    st.forward["cpu", Q_B](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", Q_B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo = _maxabs(sout.data, y, Q_B * Q_W)
    var mgi = _maxabs(sgi.data, gi, Q_B * Q_W)
    print("    max Δ: out", mo, " gi", mgi)
    assert_true(mo < CPU_TOL and mgi < CPU_TOL, "QKVToMajor CPU parity")
    x.free(); y.free(); go.free(); gi.free()
    print("    ok")


def test_qkv_gpu_parity() raises:
    print("  qkv gpu vs cpu ...")
    var c = DeviceContext()
    var cpu = QKVToMajor[Q_SEQ, Q_DIM].make["cpu", Deterministic]()
    var gpu = QKVToMajor[Q_SEQ, Q_DIM].make["gpu", Deterministic](Optional(c))

    var sx = Tensor.alloc(Q_B * Q_W)
    var sgo = Tensor.alloc(Q_B * Q_W)
    for i in range(Q_B * Q_W):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.13
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.21
    var c_out = Tensor.alloc(Q_B * Q_W)
    var c_gi = Tensor.alloc(Q_B * Q_W)
    cpu.forward["cpu", Q_B](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", Q_B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(Q_B * Q_W)
    var ggo = Tensor.alloc(Q_B * Q_W)
    for i in range(Q_B * Q_W):
        gx.data[i] = sx.data[i]
        ggo.data[i] = sgo.data[i]
    gx.upload(c); ggo.upload(c)
    var g_out = Tensor.alloc(Q_B * Q_W)
    var g_gi = Tensor.alloc(Q_B * Q_W)
    g_out.upload(c); g_gi.upload(c)
    gpu.forward["gpu", Q_B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", Q_B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c); g_gi.download(c)

    var mo = _maxabs2(g_out.data, c_out.data, Q_B * Q_W)
    var mgi = _maxabs2(g_gi.data, c_gi.data, Q_B * Q_W)
    print("    max Δ: out", mo, " gi", mgi)
    assert_true(mo < GPU_TOL and mgi < GPU_TOL, "QKVToMajor GPU vs CPU")
    print("    ok")


# ════════════════════════════════════════════════════════════════════════
# SwiGLU   (HIDDEN ; IN=2*HIDDEN, OUT=HIDDEN)
# ════════════════════════════════════════════════════════════════════════
comptime SG_H = 5
comptime SG_B = 3
comptime SG_IN = 2 * SG_H
comptime SG_OUT = SG_H


def test_swiglu_cpu_parity() raises:
    print("  swiglu cpu parity ...")
    var leg = LegacySwiGLU[SG_H].make[target="cpu", INIT=Zero]()
    var x = _alloc(SG_B * SG_IN)
    var y = _alloc(SG_B * SG_OUT)
    var go = _alloc(SG_B * SG_OUT)
    var gi = _alloc(SG_B * SG_IN)
    for i in range(SG_B * SG_IN):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.17
    for i in range(SG_B * SG_OUT):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.23
    var x_t = TileTensor(x, row_major[SG_B, SG_IN]())
    var y_t = TileTensor(y, row_major[SG_B, SG_OUT]())
    var go_t = TileTensor(go, row_major[SG_B, SG_OUT]())
    var gi_t = TileTensor(gi, row_major[SG_B, SG_IN]())
    leg.forward["cpu", SG_B](x_t, output=y_t)
    leg.vjp["cpu", SG_B](go_t, gi_t)

    var st = SwiGLU[SG_H].make["cpu", Deterministic]()
    var sx = Tensor.alloc(SG_B * SG_IN)
    var sgo = Tensor.alloc(SG_B * SG_OUT)
    var sout = Tensor.alloc(SG_B * SG_OUT)
    var sgi = Tensor.alloc(SG_B * SG_IN)
    for i in range(SG_B * SG_IN):
        sx.data[i] = x[i]
    for i in range(SG_B * SG_OUT):
        sgo.data[i] = go[i]
    st.forward["cpu", SG_B](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", SG_B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo = _maxabs(sout.data, y, SG_B * SG_OUT)
    var mgi = _maxabs(sgi.data, gi, SG_B * SG_IN)
    print("    max Δ: out", mo, " gi", mgi)
    assert_true(mo < CPU_TOL and mgi < CPU_TOL, "SwiGLU CPU parity")
    x.free(); y.free(); go.free(); gi.free()
    print("    ok")


def test_swiglu_gpu_parity() raises:
    print("  swiglu gpu vs cpu ...")
    var c = DeviceContext()
    var cpu = SwiGLU[SG_H].make["cpu", Deterministic]()
    var gpu = SwiGLU[SG_H].make["gpu", Deterministic](Optional(c))

    var sx = Tensor.alloc(SG_B * SG_IN)
    var sgo = Tensor.alloc(SG_B * SG_OUT)
    for i in range(SG_B * SG_IN):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.17
    for i in range(SG_B * SG_OUT):
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.23
    var c_out = Tensor.alloc(SG_B * SG_OUT)
    var c_gi = Tensor.alloc(SG_B * SG_IN)
    cpu.forward["cpu", SG_B](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", SG_B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(SG_B * SG_IN)
    var ggo = Tensor.alloc(SG_B * SG_OUT)
    for i in range(SG_B * SG_IN):
        gx.data[i] = sx.data[i]
    for i in range(SG_B * SG_OUT):
        ggo.data[i] = sgo.data[i]
    gx.upload(c); ggo.upload(c)
    var g_out = Tensor.alloc(SG_B * SG_OUT)
    var g_gi = Tensor.alloc(SG_B * SG_IN)
    g_out.upload(c); g_gi.upload(c)
    gpu.forward["gpu", SG_B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", SG_B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c); g_gi.download(c)

    var mo = _maxabs2(g_out.data, c_out.data, SG_B * SG_OUT)
    var mgi = _maxabs2(g_gi.data, c_gi.data, SG_B * SG_IN)
    print("    max Δ: out", mo, " gi", mgi)
    assert_true(mo < GPU_TOL and mgi < GPU_TOL, "SwiGLU GPU vs CPU")
    print("    ok")


# ════════════════════════════════════════════════════════════════════════
# SpaceTimeTranspose   (T, S, D ; IN=OUT=T*S*D)
# ════════════════════════════════════════════════════════════════════════
comptime ST_T = 3
comptime ST_S = 4
comptime ST_D = 2
comptime ST_B = 2
comptime ST_W = ST_T * ST_S * ST_D


def test_stt_cpu_parity() raises:
    print("  stt cpu parity ...")
    var leg = LegacySTT[ST_T, ST_S, ST_D].make[target="cpu", INIT=Zero]()
    var x = _alloc(ST_B * ST_W)
    var y = _alloc(ST_B * ST_W)
    var go = _alloc(ST_B * ST_W)
    var gi = _alloc(ST_B * ST_W)
    for i in range(ST_B * ST_W):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.13
        go[i] = Scalar[DT]((i % 7) - 3) * 0.21
    var x_t = TileTensor(x, row_major[ST_B, ST_W]())
    var y_t = TileTensor(y, row_major[ST_B, ST_W]())
    var go_t = TileTensor(go, row_major[ST_B, ST_W]())
    var gi_t = TileTensor(gi, row_major[ST_B, ST_W]())
    leg.forward["cpu", ST_B](x_t, output=y_t)
    leg.vjp["cpu", ST_B](go_t, gi_t)

    var st = SpaceTimeTranspose[ST_T, ST_S, ST_D].make["cpu", Deterministic]()
    var sx = Tensor.alloc(ST_B * ST_W)
    var sgo = Tensor.alloc(ST_B * ST_W)
    var sout = Tensor.alloc(ST_B * ST_W)
    var sgi = Tensor.alloc(ST_B * ST_W)
    for i in range(ST_B * ST_W):
        sx.data[i] = x[i]
        sgo.data[i] = go[i]
    st.forward["cpu", ST_B](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", ST_B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo = _maxabs(sout.data, y, ST_B * ST_W)
    var mgi = _maxabs(sgi.data, gi, ST_B * ST_W)
    print("    max Δ: out", mo, " gi", mgi)
    assert_true(mo < CPU_TOL and mgi < CPU_TOL, "SpaceTimeTranspose CPU parity")
    x.free(); y.free(); go.free(); gi.free()
    print("    ok")


def test_stt_gpu_parity() raises:
    print("  stt gpu vs cpu ...")
    var c = DeviceContext()
    var cpu = SpaceTimeTranspose[ST_T, ST_S, ST_D].make["cpu", Deterministic]()
    var gpu = SpaceTimeTranspose[ST_T, ST_S, ST_D].make["gpu", Deterministic](
        Optional(c)
    )

    var sx = Tensor.alloc(ST_B * ST_W)
    var sgo = Tensor.alloc(ST_B * ST_W)
    for i in range(ST_B * ST_W):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.13
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.21
    var c_out = Tensor.alloc(ST_B * ST_W)
    var c_gi = Tensor.alloc(ST_B * ST_W)
    cpu.forward["cpu", ST_B](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", ST_B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(ST_B * ST_W)
    var ggo = Tensor.alloc(ST_B * ST_W)
    for i in range(ST_B * ST_W):
        gx.data[i] = sx.data[i]
        ggo.data[i] = sgo.data[i]
    gx.upload(c); ggo.upload(c)
    var g_out = Tensor.alloc(ST_B * ST_W)
    var g_gi = Tensor.alloc(ST_B * ST_W)
    g_out.upload(c); g_gi.upload(c)
    gpu.forward["gpu", ST_B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", ST_B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c); g_gi.download(c)

    var mo = _maxabs2(g_out.data, c_out.data, ST_B * ST_W)
    var mgi = _maxabs2(g_gi.data, c_gi.data, ST_B * ST_W)
    print("    max Δ: out", mo, " gi", mgi)
    assert_true(mo < GPU_TOL and mgi < GPU_TOL, "SpaceTimeTranspose GPU vs CPU")
    print("    ok")


def main() raises:
    print("=" * 70)
    print("QKVToMajor / SwiGLU / SpaceTimeTranspose legacy ↔ storage parity")
    print("=" * 70)
    print("CPU parity (legacy vs storage):")
    test_qkv_cpu_parity()
    test_swiglu_cpu_parity()
    test_stt_cpu_parity()
    print("GPU vs CPU (storage):")
    test_qkv_gpu_parity()
    test_swiglu_gpu_parity()
    test_stt_gpu_parity()
    print("ALL PASSED")
