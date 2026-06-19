"""SinusoidalPosAdd / SinusoidalPosAddBT / BroadcastTokens legacy ↔ storage parity.

Each leaf: legacy↔storage CPU is bit-identical (same kernel math carried over,
max|Δ| < 1e-6 on out + grad_input), and storage GPU matches storage CPU
(TOL ~2e-5). Run:
  rm -f mojo_rl.mojoc && pixi run mojo run -I . tests/nn/test_pos_broadcast_storage_parity.mojo
  rm -f mojo_rl.mojoc && pixi run -e apple mojo run -I . tests/nn/test_pos_broadcast_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.primitives.sinusoidal_pos import (
    SinusoidalPosAdd as LegacyPos,
)
from mojo_rl.nn.primitives.sinusoidal_pos_bt import (
    SinusoidalPosAddBT as LegacyPosBT,
)
from mojo_rl.nn.primitives.broadcast_tokens import (
    BroadcastTokens as LegacyBroadcastTokens,
)
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.sinusoidal_pos import SinusoidalPosAdd
from mojo_rl.nn.storage.primitives.sinusoidal_pos_bt import SinusoidalPosAddBT
from mojo_rl.nn.storage.primitives.broadcast_tokens import BroadcastTokens


# ════════════════════════════════════════════════════════════════════════
# SinusoidalPosAdd[T, S, D, SCALE] — per-sample (T*S*D) grid, BATCH = B
# ════════════════════════════════════════════════════════════════════════
comptime PT = 3
comptime PS = 4
comptime PD = 6
comptime PN = PT * PS * PD
comptime PB = 5


def test_pos_cpu_parity() raises:
    print("test_pos_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)

    var leg = LegacyPos[PT, PS, PD, True].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](PB * PN)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](PB * PN)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](PB * PN)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](PB * PN)
    for i in range(PB * PN):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(PB * PN):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.22

    var x_t = TileTensor(x, row_major[PB, PN]())
    var y_t = TileTensor(y, row_major[PB, PN]())
    var go_t = TileTensor(go, row_major[PB, PN]())
    var gi_t = TileTensor(gi, row_major[PB, PN]())
    leg.forward["cpu", PB](x_t, output=y_t)
    leg.vjp["cpu", PB](go_t, gi_t)

    var st = SinusoidalPosAdd[PT, PS, PD, True].make["cpu", Deterministic]()
    var sx = Tensor.alloc(PB * PN)
    var sgo = Tensor.alloc(PB * PN)
    var sout = Tensor.alloc(PB * PN)
    var sgi = Tensor.alloc(PB * PN)
    for i in range(PB * PN):
        sx.data[i] = x[i]
        sgo.data[i] = go[i]
    st.forward["cpu", PB](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", PB](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(PB * PN):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "SinusoidalPosAdd CPU parity")
    print("  ok")


def test_pos_gpu_parity() raises:
    print("test_pos_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = SinusoidalPosAdd[PT, PS, PD, True].make["cpu", Deterministic]()
    var gpu = SinusoidalPosAdd[PT, PS, PD, True].make["gpu", Deterministic](
        Optional(c)
    )

    var sx = Tensor.alloc(PB * PN)
    var sgo = Tensor.alloc(PB * PN)
    for i in range(PB * PN):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    var c_out = Tensor.alloc(PB * PN)
    var c_gi = Tensor.alloc(PB * PN)
    cpu.forward["cpu", PB](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", PB](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(PB * PN)
    var ggo = Tensor.alloc(PB * PN)
    for i in range(PB * PN):
        gx.data[i] = sx.data[i]
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(PB * PN)
    var g_gi = Tensor.alloc(PB * PN)
    gpu.forward["gpu", PB](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", PB](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(PB * PN):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "SinusoidalPosAdd GPU vs CPU")
    print("  ok")


# ════════════════════════════════════════════════════════════════════════
# SinusoidalPosAddBT[T, S, D, SCALE] — BATCH = B (rows index B*T), per-row SD
# ════════════════════════════════════════════════════════════════════════
comptime BT_T = 3
comptime BT_S = 4
comptime BT_D = 5
comptime BT_SD = BT_S * BT_D
comptime BTB = 9  # B*T rows (T=3 → 3 outer; row t = bt % T)


def test_pos_bt_cpu_parity() raises:
    print("test_pos_bt_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)

    var leg = LegacyPosBT[BT_T, BT_S, BT_D, True].make[
        target="cpu", INIT=Zero
    ]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BTB * BT_SD
    )
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BTB * BT_SD
    )
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BTB * BT_SD
    )
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BTB * BT_SD
    )
    for i in range(BTB * BT_SD):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.18
        go[i] = Scalar[DT]((i % 7) - 3) * 0.22

    var x_t = TileTensor(x, row_major[BTB, BT_SD]())
    var y_t = TileTensor(y, row_major[BTB, BT_SD]())
    var go_t = TileTensor(go, row_major[BTB, BT_SD]())
    var gi_t = TileTensor(gi, row_major[BTB, BT_SD]())
    leg.forward["cpu", BTB](x_t, output=y_t)
    leg.vjp["cpu", BTB](go_t, gi_t)

    var st = SinusoidalPosAddBT[BT_T, BT_S, BT_D, True].make[
        "cpu", Deterministic
    ]()
    var sx = Tensor.alloc(BTB * BT_SD)
    var sgo = Tensor.alloc(BTB * BT_SD)
    var sout = Tensor.alloc(BTB * BT_SD)
    var sgi = Tensor.alloc(BTB * BT_SD)
    for i in range(BTB * BT_SD):
        sx.data[i] = x[i]
        sgo.data[i] = go[i]
    st.forward["cpu", BTB](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", BTB](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(BTB * BT_SD):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "SinusoidalPosAddBT CPU parity")
    print("  ok")


def test_pos_bt_gpu_parity() raises:
    print("test_pos_bt_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = SinusoidalPosAddBT[BT_T, BT_S, BT_D, True].make[
        "cpu", Deterministic
    ]()
    var gpu = SinusoidalPosAddBT[BT_T, BT_S, BT_D, True].make[
        "gpu", Deterministic
    ](Optional(c))

    var sx = Tensor.alloc(BTB * BT_SD)
    var sgo = Tensor.alloc(BTB * BT_SD)
    for i in range(BTB * BT_SD):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    var c_out = Tensor.alloc(BTB * BT_SD)
    var c_gi = Tensor.alloc(BTB * BT_SD)
    cpu.forward["cpu", BTB](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", BTB](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(BTB * BT_SD)
    var ggo = Tensor.alloc(BTB * BT_SD)
    for i in range(BTB * BT_SD):
        gx.data[i] = sx.data[i]
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(BTB * BT_SD)
    var g_gi = Tensor.alloc(BTB * BT_SD)
    gpu.forward["gpu", BTB](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", BTB](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(BTB * BT_SD):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "SinusoidalPosAddBT GPU vs CPU")
    print("  ok")


# ════════════════════════════════════════════════════════════════════════
# BroadcastTokens[N, DIM] — fan one DIM-vector to N tokens; bwd sums
# ════════════════════════════════════════════════════════════════════════
comptime BN = 4
comptime BDIM = 7
comptime BOUT = BN * BDIM
comptime BB = 6


def test_broadcast_cpu_parity() raises:
    print("test_broadcast_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)

    var leg = LegacyBroadcastTokens[BN, BDIM].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BB * BDIM)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BB * BOUT)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BB * BOUT
    )
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BB * BDIM
    )
    for i in range(BB * BDIM):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(BB * BOUT):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.22

    var x_t = TileTensor(x, row_major[BB, BDIM]())
    var y_t = TileTensor(y, row_major[BB, BOUT]())
    var go_t = TileTensor(go, row_major[BB, BOUT]())
    var gi_t = TileTensor(gi, row_major[BB, BDIM]())
    leg.forward["cpu", BB](x_t, output=y_t)
    leg.vjp["cpu", BB](go_t, gi_t)

    var st = BroadcastTokens[BN, BDIM].make["cpu", Deterministic]()
    var sx = Tensor.alloc(BB * BDIM)
    var sgo = Tensor.alloc(BB * BOUT)
    var sout = Tensor.alloc(BB * BOUT)
    var sgi = Tensor.alloc(BB * BDIM)
    for i in range(BB * BDIM):
        sx.data[i] = x[i]
    for i in range(BB * BOUT):
        sgo.data[i] = go[i]
    st.forward["cpu", BB](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", BB](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    for i in range(BB * BOUT):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
    var mgi: Scalar[DT] = 0
    for i in range(BB * BDIM):
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "BroadcastTokens CPU parity")
    print("  ok")


def test_broadcast_gpu_parity() raises:
    print("test_broadcast_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = BroadcastTokens[BN, BDIM].make["cpu", Deterministic]()
    var gpu = BroadcastTokens[BN, BDIM].make["gpu", Deterministic](Optional(c))

    var sx = Tensor.alloc(BB * BDIM)
    var sgo = Tensor.alloc(BB * BOUT)
    for i in range(BB * BDIM):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(BB * BOUT):
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22
    var c_out = Tensor.alloc(BB * BOUT)
    var c_gi = Tensor.alloc(BB * BDIM)
    cpu.forward["cpu", BB](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", BB](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(BB * BDIM)
    var ggo = Tensor.alloc(BB * BOUT)
    for i in range(BB * BDIM):
        gx.data[i] = sx.data[i]
    for i in range(BB * BOUT):
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(BB * BOUT)
    var g_gi = Tensor.alloc(BB * BDIM)
    gpu.forward["gpu", BB](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", BB](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)

    var mo: Scalar[DT] = 0
    for i in range(BB * BOUT):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
    var mgi: Scalar[DT] = 0
    for i in range(BB * BDIM):
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "BroadcastTokens GPU vs CPU")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("SinusoidalPosAdd / SinusoidalPosAddBT / BroadcastTokens parity")
    print("=" * 70)
    test_pos_cpu_parity()
    test_pos_bt_cpu_parity()
    test_broadcast_cpu_parity()
    test_pos_gpu_parity()
    test_pos_bt_gpu_parity()
    test_broadcast_gpu_parity()
    print("ALL PASSED")
