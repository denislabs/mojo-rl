"""Scale / Clamp / GatherCols legacy ↔ storage parity (CPU) + storage GPU vs CPU.

Each leaf: legacy↔storage CPU is bit-identical (same SIMD/kernel math carried
over), and storage GPU matches storage CPU (TOL ~2e-5). GatherCols is
forward-only — the vjp zero-fill is asserted on both surfaces. Run:
  pixi run mojo run -I . tests/nn/test_scale_clamp_gather_storage_parity.mojo
  pixi run -e apple mojo run -I . tests/nn/test_scale_clamp_gather_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.primitives.scale import Scale as LegacyScale
from mojo_rl.nn.primitives.clamp import Clamp as LegacyClamp
from mojo_rl.nn.primitives.gather_cols import GatherCols as LegacyGatherCols
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.tensor_pack import TensorPack as STensorPack
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.scale import Scale
from mojo_rl.nn.storage.primitives.clamp import Clamp
from mojo_rl.nn.storage.primitives.gather_cols import GatherCols


comptime DIM = 10
comptime B = 6


# ════════════════════════════════════════════════════════════════════════
# Scale
# ════════════════════════════════════════════════════════════════════════
def test_scale_cpu_parity() raises:
    print("test_scale_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)
    comptime M = Scalar[DT](1.37)

    var leg = LegacyScale[DIM].make[target="cpu", INIT=Zero]()
    leg.set_attr["multiplier"](M)

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

    var st = Scale[DIM].make["cpu", Deterministic]()
    st.set_multiplier(M)
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
    assert_true(mo < TOL and mgi < TOL, "Scale CPU parity")
    print("  ok")


def test_scale_gpu_parity() raises:
    print("test_scale_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    comptime M = Scalar[DT](1.37)
    var c = DeviceContext()
    var cpu = Scale[DIM].make["cpu", Deterministic]()
    var gpu = Scale[DIM].make["gpu", Deterministic](Optional(c))
    cpu.set_multiplier(M)
    gpu.set_multiplier(M)

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
    assert_true(mo < TOL and mgi < TOL, "Scale GPU vs CPU")
    print("  ok")


# ════════════════════════════════════════════════════════════════════════
# Clamp
# ════════════════════════════════════════════════════════════════════════
def test_clamp_cpu_parity() raises:
    print("test_clamp_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)
    comptime MN = Scalar[DT](-0.4)
    comptime MX = Scalar[DT](0.5)

    var leg = LegacyClamp[DIM].make[target="cpu", INIT=Zero]()
    leg.set_attr["min_val"](MN)
    leg.set_attr["max_val"](MX)

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

    var st = Clamp[DIM].make["cpu", Deterministic]()
    st.set_min_max(MN, MX)
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
    assert_true(mo < TOL and mgi < TOL, "Clamp CPU parity")
    print("  ok")


def test_clamp_gpu_parity() raises:
    print("test_clamp_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    comptime MN = Scalar[DT](-0.4)
    comptime MX = Scalar[DT](0.5)
    var c = DeviceContext()
    var cpu = Clamp[DIM].make["cpu", Deterministic]()
    var gpu = Clamp[DIM].make["gpu", Deterministic](Optional(c))
    cpu.set_min_max(MN, MX)
    gpu.set_min_max(MN, MX)

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
    assert_true(mo < TOL and mgi < TOL, "Clamp GPU vs CPU")
    print("  ok")


# ════════════════════════════════════════════════════════════════════════
# GatherCols (forward-only; vjp zero-fills)
# ════════════════════════════════════════════════════════════════════════
comptime GNA = 5
comptime GB = 8


def test_gather_cpu_parity() raises:
    print("test_gather_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)

    var v: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](GB * GNA)
    var idx: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](GB)
    var out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](GB)
    for b in range(GB):
        for a in range(GNA):
            v[b * GNA + a] = Scalar[DT](100.0 * Float64(b) + Float64(a))
        idx[b] = Scalar[DT]((b * 2 + 1) % GNA)

    var leg = LegacyGatherCols[GNA].make[target="cpu", INIT=Zero]()
    var v_t = TileTensor(v, row_major[GB, GNA]())
    var i_t = TileTensor(idx, row_major[GB, GNA]())
    var o_t = TileTensor(out, row_major[GB, 1]())
    leg.forward["cpu", GB](TensorPack[2].of(v_t, i_t), output=o_t)

    # Legacy vjp zero-fill: prefill grads with junk, vjp must zero.
    var lgv: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](GB * GNA)
    var lgi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](GB)
    var lgo: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](GB)
    for i in range(GB * GNA): lgv[i] = Scalar[DT](42.0)
    for b in range(GB): lgi[b] = Scalar[DT](99.0)
    var lgo_t = TileTensor(lgo, row_major[GB, 1]())
    var lgv_t = TileTensor(lgv, row_major[GB, GNA]())
    var lgi_t = TileTensor(lgi, row_major[GB, GNA]())
    leg.vjp["cpu", GB](lgo_t, TensorPack[2].of(lgv_t, lgi_t))

    # Storage. Inputs + grad_inputs live in TensorPacks (shared origin).
    var st = GatherCols[GNA].make["cpu", Deterministic]()
    var ins = STensorPack[2]()
    ins[0].ensure(GB * GNA)
    ins[1].ensure(GB)
    for b in range(GB):
        for a in range(GNA):
            ins[0].data[b * GNA + a] = v[b * GNA + a]
        ins[1].data[b] = idx[b]
    var sout = Tensor.alloc(GB)
    st.forward["cpu", GB](TensorRefs[2](ins[0], ins[1]), sout, None)

    var gpk = STensorPack[2]()
    gpk[0].ensure(GB * GNA)
    gpk[1].ensure(GB)
    var sgo = Tensor.alloc(GB)
    for i in range(GB * GNA): gpk[0].data[i] = Scalar[DT](42.0)
    for b in range(GB): gpk[1].data[b] = Scalar[DT](99.0)
    st.vjp["cpu", GB](
        TensorRefs[2](ins[0], ins[1]), sgo, TensorRefs[2](gpk[0], gpk[1]), None
    )

    var mo: Scalar[DT] = 0
    for b in range(GB):
        if abs(sout.data[b] - out[b]) > mo: mo = abs(sout.data[b] - out[b])
    var mgv: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(GB * GNA):
        if abs(gpk[0].data[i] - lgv[i]) > mgv: mgv = abs(gpk[0].data[i] - lgv[i])
    for b in range(GB):
        if abs(gpk[1].data[b] - lgi[b]) > mgi: mgi = abs(gpk[1].data[b] - lgi[b])
    print("  max Δ: out", mo, " grad_values(zero)", mgv, " grad_idx(zero)", mgi)
    assert_true(mo < TOL and mgv < TOL and mgi < TOL, "GatherCols CPU parity")
    # Also assert the zero-fill is exact zero.
    for i in range(GB * GNA):
        assert_true(gpk[0].data[i] == Scalar[DT](0.0), "grad_values zero")
    for b in range(GB):
        assert_true(gpk[1].data[b] == Scalar[DT](0.0), "grad_idx zero")
    print("  ok")


def test_gather_gpu_parity() raises:
    print("test_gather_gpu_parity (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = GatherCols[GNA].make["cpu", Deterministic]()
    var gpu = GatherCols[GNA].make["gpu", Deterministic](Optional(c))

    var cins = STensorPack[2]()
    cins[0].ensure(GB * GNA)
    cins[1].ensure(GB)
    for b in range(GB):
        for a in range(GNA):
            cins[0].data[b * GNA + a] = Scalar[DT](100.0 * Float64(b) + Float64(a))
        cins[1].data[b] = Scalar[DT]((b * 2 + 1) % GNA)
    var c_out = Tensor.alloc(GB)
    cpu.forward["cpu", GB](TensorRefs[2](cins[0], cins[1]), c_out, None)
    var cgpk = STensorPack[2]()
    cgpk[0].ensure(GB * GNA)
    cgpk[1].ensure(GB)
    var c_go = Tensor.alloc(GB)
    cpu.vjp["cpu", GB](
        TensorRefs[2](cins[0], cins[1]), c_go,
        TensorRefs[2](cgpk[0], cgpk[1]), None,
    )

    var gins = STensorPack[2]()
    gins[0].ensure(GB * GNA)
    gins[1].ensure(GB)
    for b in range(GB):
        for a in range(GNA):
            gins[0].data[b * GNA + a] = cins[0].data[b * GNA + a]
        gins[1].data[b] = cins[1].data[b]
    gins[0].upload(c)
    gins[1].upload(c)
    var g_out = Tensor.alloc(GB)
    gpu.forward["gpu", GB](TensorRefs[2](gins[0], gins[1]), g_out, Optional(c))
    g_out.download(c)

    var ggpk = STensorPack[2]()
    ggpk[0].ensure(GB * GNA)
    ggpk[1].ensure(GB)
    var g_go = Tensor.alloc(GB)
    g_go.upload(c)
    gpu.vjp["gpu", GB](
        TensorRefs[2](gins[0], gins[1]), g_go,
        TensorRefs[2](ggpk[0], ggpk[1]), Optional(c),
    )
    ggpk[0].download(c)
    ggpk[1].download(c)

    var mo: Scalar[DT] = 0
    for b in range(GB):
        if abs(g_out.data[b] - c_out.data[b]) > mo: mo = abs(g_out.data[b] - c_out.data[b])
    var mgv: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(GB * GNA):
        if abs(ggpk[0].data[i] - cgpk[0].data[i]) > mgv: mgv = abs(ggpk[0].data[i] - cgpk[0].data[i])
    for b in range(GB):
        if abs(ggpk[1].data[b] - cgpk[1].data[b]) > mgi: mgi = abs(ggpk[1].data[b] - cgpk[1].data[b])
    print("  max Δ: out", mo, " grad_values(zero)", mgv, " grad_idx(zero)", mgi)
    assert_true(mo < TOL and mgv < TOL and mgi < TOL, "GatherCols GPU vs CPU")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Scale / Clamp / GatherCols legacy ↔ storage parity")
    print("=" * 70)
    test_scale_cpu_parity()
    test_scale_gpu_parity()
    test_clamp_cpu_parity()
    test_clamp_gpu_parity()
    test_gather_cpu_parity()
    test_gather_gpu_parity()
    print("ALL PASSED")
