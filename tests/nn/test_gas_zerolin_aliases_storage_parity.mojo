"""GatherActionSlice / ZeroLinear / SiLU / StopGrad legacy ↔ storage parity.

Real leaves (GatherActionSlice, ZeroLinear): legacy↔storage CPU is bit-identical
(same kernels/math carried over), and storage GPU matches storage CPU (TOL
~2e-5). ZeroLinear also checks weight/bias param grads. The aliases (SiLU,
StopGrad) are instantiated and forward/vjp-parity-checked vs legacy. Run:
  pixi run mojo run -I . tests/nn/test_gas_zerolin_aliases_storage_parity.mojo
  pixi run -e apple mojo run -I . tests/nn/test_gas_zerolin_aliases_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.primitives.gather_action_slice import (
    GatherActionSlice as LegacyGAS,
)
from mojo_rl.nn.primitives.zero_linear import ZeroLinear as LegacyZeroLinear
from mojo_rl.nn.primitives.silu import SiLU as LegacySiLU
from mojo_rl.nn.primitives.stop_grad import StopGrad as LegacyStopGrad
from mojo_rl.nn.initializer import Zero

from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.tensor_pack import TensorPack as STensorPack
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.gather_action_slice import GatherActionSlice
from mojo_rl.nn.storage.primitives.zero_linear import ZeroLinear
from mojo_rl.nn.storage.primitives.silu import SiLU
from mojo_rl.nn.storage.primitives.stop_grad import StopGrad


# ════════════════════════════════════════════════════════════════════════
# GatherActionSlice (forward-only; vjp zero-fills)
# ════════════════════════════════════════════════════════════════════════
comptime GNA = 4
comptime GK = 3
comptime GB = 8


def test_gas_cpu_parity() raises:
    print("test_gas_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)
    comptime NK = GNA * GK

    var v: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](GB * NK)
    var idx: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](GB)
    var out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](GB * GK)
    for b in range(GB):
        for c in range(NK):
            v[b * NK + c] = Scalar[DT](100.0 * Float64(b) + Float64(c))
        idx[b] = Scalar[DT]((b * 3 + 1) % GNA)

    var leg = LegacyGAS[GNA, GK].make[target="cpu", INIT=Zero]()
    # `of` is single-type variadic → all carrier tiles share `row_major[GB, NK]`.
    var v_t = TileTensor(v, row_major[GB, NK]())
    var i_t = TileTensor(idx, row_major[GB, NK]())  # hetero-variadic carrier
    var o_t = TileTensor(out, row_major[GB, GK]())
    leg.forward["cpu", GB](TensorPack[2].of(v_t, i_t), output=o_t)

    # Legacy vjp zero-fill: prefill grads with junk, vjp must zero.
    var lgv: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](GB * NK)
    var lgi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](GB)
    var lgo: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](GB * GK)
    for i in range(GB * NK): lgv[i] = Scalar[DT](42.0)
    for b in range(GB): lgi[b] = Scalar[DT](99.0)
    var lgo_t = TileTensor(lgo, row_major[GB, GK]())
    var lgv_t = TileTensor(lgv, row_major[GB, NK]())
    var lgi_t = TileTensor(lgi, row_major[GB, NK]())  # carrier
    leg.vjp["cpu", GB](lgo_t, TensorPack[2].of(lgv_t, lgi_t))

    # Storage.
    var st = GatherActionSlice[GNA, GK].make["cpu", Deterministic]()
    var ins = STensorPack[2]()
    ins[0].ensure(GB * NK)
    ins[1].ensure(GB)
    for b in range(GB):
        for c in range(NK):
            ins[0].data[b * NK + c] = v[b * NK + c]
        ins[1].data[b] = idx[b]
    var sout = Tensor.alloc(GB * GK)
    st.forward["cpu", GB](TensorRefs[2](ins[0], ins[1]), sout, None)

    var gpk = STensorPack[2]()
    gpk[0].ensure(GB * NK)
    gpk[1].ensure(GB)
    var sgo = Tensor.alloc(GB * GK)
    for i in range(GB * NK): gpk[0].data[i] = Scalar[DT](42.0)
    for b in range(GB): gpk[1].data[b] = Scalar[DT](99.0)
    st.vjp["cpu", GB](
        TensorRefs[2](ins[0], ins[1]), sgo, TensorRefs[2](gpk[0], gpk[1]), None
    )

    var mo: Scalar[DT] = 0
    for i in range(GB * GK):
        if abs(sout.data[i] - out[i]) > mo: mo = abs(sout.data[i] - out[i])
    var mgv: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(GB * NK):
        if abs(gpk[0].data[i] - lgv[i]) > mgv: mgv = abs(gpk[0].data[i] - lgv[i])
    for b in range(GB):
        if abs(gpk[1].data[b] - lgi[b]) > mgi: mgi = abs(gpk[1].data[b] - lgi[b])
    print("  max Δ: out", mo, " grad_values(zero)", mgv, " grad_idx(zero)", mgi)
    assert_true(mo < TOL and mgv < TOL and mgi < TOL, "GAS CPU parity")
    for i in range(GB * NK):
        assert_true(gpk[0].data[i] == Scalar[DT](0.0), "grad_values zero")
    for b in range(GB):
        assert_true(gpk[1].data[b] == Scalar[DT](0.0), "grad_idx zero")
    print("  ok")


def test_gas_gpu_parity() raises:
    print("test_gas_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    comptime NK = GNA * GK
    var c = DeviceContext()
    var cpu = GatherActionSlice[GNA, GK].make["cpu", Deterministic]()
    var gpu = GatherActionSlice[GNA, GK].make["gpu", Deterministic](Optional(c))

    var cins = STensorPack[2]()
    cins[0].ensure(GB * NK)
    cins[1].ensure(GB)
    for b in range(GB):
        for cc in range(NK):
            cins[0].data[b * NK + cc] = Scalar[DT](100.0 * Float64(b) + Float64(cc))
        cins[1].data[b] = Scalar[DT]((b * 3 + 1) % GNA)
    var c_out = Tensor.alloc(GB * GK)
    cpu.forward["cpu", GB](TensorRefs[2](cins[0], cins[1]), c_out, None)
    var cgpk = STensorPack[2]()
    cgpk[0].ensure(GB * NK)
    cgpk[1].ensure(GB)
    var c_go = Tensor.alloc(GB * GK)
    cpu.vjp["cpu", GB](
        TensorRefs[2](cins[0], cins[1]), c_go,
        TensorRefs[2](cgpk[0], cgpk[1]), None,
    )

    var gins = STensorPack[2]()
    gins[0].ensure(GB * NK)
    gins[1].ensure(GB)
    for b in range(GB):
        for cc in range(NK):
            gins[0].data[b * NK + cc] = cins[0].data[b * NK + cc]
        gins[1].data[b] = cins[1].data[b]
    gins[0].upload(c)
    gins[1].upload(c)
    var g_out = Tensor.alloc(GB * GK)
    gpu.forward["gpu", GB](TensorRefs[2](gins[0], gins[1]), g_out, Optional(c))
    g_out.download(c)

    var ggpk = STensorPack[2]()
    ggpk[0].ensure(GB * NK)
    ggpk[1].ensure(GB)
    var g_go = Tensor.alloc(GB * GK)
    g_go.upload(c)
    gpu.vjp["gpu", GB](
        TensorRefs[2](gins[0], gins[1]), g_go,
        TensorRefs[2](ggpk[0], ggpk[1]), Optional(c),
    )
    ggpk[0].download(c)
    ggpk[1].download(c)

    var mo: Scalar[DT] = 0
    for i in range(GB * GK):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
    var mgv: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(GB * NK):
        if abs(ggpk[0].data[i] - cgpk[0].data[i]) > mgv: mgv = abs(ggpk[0].data[i] - cgpk[0].data[i])
    for b in range(GB):
        if abs(ggpk[1].data[b] - cgpk[1].data[b]) > mgi: mgi = abs(ggpk[1].data[b] - cgpk[1].data[b])
    print("  max Δ: out", mo, " grad_values(zero)", mgv, " grad_idx(zero)", mgi)
    assert_true(mo < TOL and mgv < TOL and mgi < TOL, "GAS GPU vs CPU")
    print("  ok")


# ════════════════════════════════════════════════════════════════════════
# ZeroLinear (owns inner Linear params; zero-init weight + bias)
# ════════════════════════════════════════════════════════════════════════
comptime ZIN = 5
comptime ZOUT = 4
comptime ZB = 6


def test_zerolin_cpu_parity() raises:
    print("test_zerolin_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](ZB * ZIN)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](ZB * ZOUT)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](ZB * ZOUT)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](ZB * ZIN)
    for i in range(ZB * ZIN):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(ZB * ZOUT):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.22

    var leg = LegacyZeroLinear[ZIN, ZOUT].make[target="cpu", INIT=Zero]()
    var x_t = TileTensor(x, row_major[ZB, ZIN]())
    var y_t = TileTensor(y, row_major[ZB, ZOUT]())
    var go_t = TileTensor(go, row_major[ZB, ZOUT]())
    var gi_t = TileTensor(gi, row_major[ZB, ZIN]())
    leg.zero_grad["cpu"]()
    leg.forward["cpu", ZB](TensorPack[1].of(x_t), output=y_t)
    leg.vjp["cpu", ZB](go_t, TensorPack[1].of(gi_t))

    var st = ZeroLinear[ZIN, ZOUT].make["cpu", Deterministic]()
    var sx = Tensor.alloc(ZB * ZIN)
    var sgo = Tensor.alloc(ZB * ZOUT)
    var sout = Tensor.alloc(ZB * ZOUT)
    var sgi = Tensor.alloc(ZB * ZIN)
    for i in range(ZB * ZIN): sx.data[i] = x[i]
    for i in range(ZB * ZOUT): sgo.data[i] = go[i]
    st.zero_grad["cpu"](None)
    st.forward["cpu", ZB](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", ZB](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    # out should be all zero (weight=bias=0) → check parity + literal zero.
    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(ZB * ZOUT):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
    for i in range(ZB * ZIN):
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])

    var lgw = leg.inner.weight.grad_unsafe_ptr_cpu()
    var lgb = leg.inner.bias.grad_unsafe_ptr_cpu()
    var mgw: Scalar[DT] = 0
    var mgb: Scalar[DT] = 0
    for i in range(ZIN * ZOUT):
        var d = abs(st.inner.weight.grd.data[i] - lgw[i])
        if d > mgw: mgw = d
    for i in range(ZOUT):
        var d = abs(st.inner.bias.grd.data[i] - lgb[i])
        if d > mgb: mgb = d

    print("  max Δ: out", mo, " gi", mgi, " gw", mgw, " gb", mgb)
    assert_true(mo < TOL and mgi < TOL and mgw < TOL and mgb < TOL,
                "ZeroLinear CPU parity")
    for i in range(ZB * ZOUT):
        assert_true(sout.data[i] == Scalar[DT](0.0), "zero-init out is zero")
    print("  ok")


def test_zerolin_gpu_parity() raises:
    print("test_zerolin_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = ZeroLinear[ZIN, ZOUT].make["cpu", Deterministic]()
    var gpu = ZeroLinear[ZIN, ZOUT].make["gpu", Deterministic](Optional(c))

    var sx = Tensor.alloc(ZB * ZIN)
    var sgo = Tensor.alloc(ZB * ZOUT)
    for i in range(ZB * ZIN): sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
    for i in range(ZB * ZOUT): sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.22

    var c_out = Tensor.alloc(ZB * ZOUT)
    var c_gi = Tensor.alloc(ZB * ZIN)
    cpu.zero_grad["cpu"](None)
    cpu.forward["cpu", ZB](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", ZB](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(ZB * ZIN)
    var ggo = Tensor.alloc(ZB * ZOUT)
    for i in range(ZB * ZIN): gx.data[i] = sx.data[i]
    for i in range(ZB * ZOUT): ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(ZB * ZOUT)
    var g_gi = Tensor.alloc(ZB * ZIN)
    gpu.zero_grad["gpu"](Optional(c))
    gpu.forward["gpu", ZB](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", ZB](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)
    gpu.inner.weight.grd.download(c)
    gpu.inner.bias.grd.download(c)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(ZB * ZOUT):
        if abs(g_out.data[i] - c_out.data[i]) > mo: mo = abs(g_out.data[i] - c_out.data[i])
    for i in range(ZB * ZIN):
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi: mgi = abs(g_gi.data[i] - c_gi.data[i])
    var mgw: Scalar[DT] = 0
    var mgb: Scalar[DT] = 0
    for i in range(ZIN * ZOUT):
        var d = abs(gpu.inner.weight.grd.data[i] - cpu.inner.weight.grd.data[i])
        if d > mgw: mgw = d
    for i in range(ZOUT):
        var d = abs(gpu.inner.bias.grd.data[i] - cpu.inner.bias.grd.data[i])
        if d > mgb: mgb = d
    print("  max Δ: out", mo, " gi", mgi, " gw", mgw, " gb", mgb)
    assert_true(mo < TOL and mgi < TOL and mgw < TOL and mgb < TOL,
                "ZeroLinear GPU vs CPU")
    print("  ok")


# ════════════════════════════════════════════════════════════════════════
# SiLU / StopGrad aliases — instantiate + forward/vjp parity vs legacy (CPU)
# ════════════════════════════════════════════════════════════════════════
comptime ADIM = 7
comptime ABATCH = 5


def test_silu_alias() raises:
    print("test_silu_alias (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)
    comptime N = ABATCH * ADIM

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT]((i % 11) - 5) * 0.21
        go[i] = Scalar[DT]((i % 7) - 3) * 0.19

    var leg = LegacySiLU[ADIM].make[target="cpu", INIT=Zero]()
    var x_t = TileTensor(x, row_major[ABATCH, ADIM]())
    var y_t = TileTensor(y, row_major[ABATCH, ADIM]())
    var go_t = TileTensor(go, row_major[ABATCH, ADIM]())
    var gi_t = TileTensor(gi, row_major[ABATCH, ADIM]())
    leg.forward["cpu", ABATCH](TensorPack[1].of(x_t), output=y_t)
    leg.vjp["cpu", ABATCH](go_t, TensorPack[1].of(gi_t))

    var st = SiLU[ADIM].make["cpu", Deterministic]()
    var sx = Tensor.alloc(N)
    var sgo = Tensor.alloc(N)
    var sout = Tensor.alloc(N)
    var sgi = Tensor.alloc(N)
    for i in range(N):
        sx.data[i] = x[i]
        sgo.data[i] = go[i]
    st.forward["cpu", ABATCH](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", ABATCH](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(N):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "SiLU alias parity")
    print("  ok")


def test_stopgrad_alias() raises:
    print("test_stopgrad_alias (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)
    comptime N = ABATCH * ADIM

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for i in range(N):
        x[i] = Scalar[DT]((i % 11) - 5) * 0.21
        go[i] = Scalar[DT]((i % 7) - 3) * 0.19
        gi[i] = Scalar[DT](42.0)

    var leg = LegacyStopGrad[ADIM].make[target="cpu", INIT=Zero]()
    var x_t = TileTensor(x, row_major[ABATCH, ADIM]())
    var y_t = TileTensor(y, row_major[ABATCH, ADIM]())
    var go_t = TileTensor(go, row_major[ABATCH, ADIM]())
    var gi_t = TileTensor(gi, row_major[ABATCH, ADIM]())
    leg.forward["cpu", ABATCH](TensorPack[1].of(x_t), output=y_t)
    leg.vjp["cpu", ABATCH](go_t, TensorPack[1].of(gi_t))

    var st = StopGrad[ADIM].make["cpu", Deterministic]()
    var sx = Tensor.alloc(N)
    var sgo = Tensor.alloc(N)
    var sout = Tensor.alloc(N)
    var sgi = Tensor.alloc(N)
    for i in range(N):
        sx.data[i] = x[i]
        sgo.data[i] = go[i]
        sgi.data[i] = Scalar[DT](42.0)
    st.forward["cpu", ABATCH](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", ABATCH](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(N):
        if abs(sout.data[i] - y[i]) > mo: mo = abs(sout.data[i] - y[i])
        if abs(sgi.data[i] - gi[i]) > mgi: mgi = abs(sgi.data[i] - gi[i])
    print("  max Δ: out", mo, " gi(zero)", mgi)
    # identity forward → out == x; zero backward → gi == 0
    assert_true(mo < TOL and mgi < TOL, "StopGrad alias parity")
    for i in range(N):
        assert_true(sgi.data[i] == Scalar[DT](0.0), "StopGrad gi is zero")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("GatherActionSlice / ZeroLinear / SiLU / StopGrad parity")
    print("=" * 70)
    test_gas_cpu_parity()
    test_gas_gpu_parity()
    test_zerolin_cpu_parity()
    test_zerolin_gpu_parity()
    test_silu_alias()
    test_stopgrad_alias()
    print("ALL PASSED")
