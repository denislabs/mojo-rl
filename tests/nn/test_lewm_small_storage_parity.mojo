"""LeWM small leaves: legacy ↔ storage parity (CPU) + storage GPU-vs-CPU.

Covers the 4 ported LeWM nn primitives:
  * LayerNormNoAffine  (ARITY-1, param-less)
  * MSEPerSample       (ARITY-2, param-less, OUT_DIM=1)
  * Gate               (ARITY-3, param-less)
  * Modulate           (ARITY-3, param-less)

Per leaf: legacy CPU vs storage CPU bit-identical (out + grad_inputs), then
storage GPU vs storage CPU (~2e-5). All param-free → no param-grad checks.
ARITY≥2 inputs/grads are sourced from one storage `TensorPack` so the
`TensorRefs` share the §B0 origin.

Run:
  rm -f mojo_rl.mojoc && pixi run mojo run -I . tests/nn/test_lewm_small_storage_parity.mojo
  rm -f mojo_rl.mojoc && pixi run -e apple mojo run -I . tests/nn/test_lewm_small_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor_pack import TensorPack as LegacyPack
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.primitives.layer_norm_no_affine import (
    LayerNormNoAffine as LegacyLNNA,
)
from mojo_rl.nn.primitives.mse_per_sample import MSEPerSample as LegacyMPS
from mojo_rl.nn.primitives.gate import Gate as LegacyGate
from mojo_rl.nn.primitives.modulate import Modulate as LegacyModulate

from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.tensor_pack import TensorPack
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.layer_norm_no_affine import LayerNormNoAffine
from mojo_rl.nn.storage.primitives.mse_per_sample import MSEPerSample
from mojo_rl.nn.storage.primitives.gate import Gate
from mojo_rl.nn.storage.primitives.modulate import Modulate


comptime B = 6
comptime DIM = 8
comptime CPU_TOL = Scalar[DT](1e-6)
comptime GPU_TOL = Scalar[DT](2e-5)


def _det(i: Int, scale: Float64) -> Scalar[DT]:
    var v = (Float64((i * 2654435761) % 1000) / 500.0) - 1.0
    return Scalar[DT](v * scale)


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _maxd(a: Tensor, b: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) -> Scalar[DT]:
    var m: Scalar[DT] = 0
    for i in range(n):
        if abs(a.data[i] - b[i]) > m:
            m = abs(a.data[i] - b[i])
    return m


def _maxd2(a: Tensor, b: Tensor, n: Int) -> Scalar[DT]:
    var m: Scalar[DT] = 0
    for i in range(n):
        if abs(a.data[i] - b.data[i]) > m:
            m = abs(a.data[i] - b.data[i])
    return m


# ──────────────────────────────────────────────────────────────────────
# LayerNormNoAffine (ARITY-1)
# ──────────────────────────────────────────────────────────────────────
def test_lnna_cpu() raises:
    print("test_lnna_cpu (legacy vs storage) ...")
    comptime N = B * DIM
    var x = _a(N)
    var y = _a(N)
    var go = _a(N)
    var gi = _a(N)
    for i in range(N):
        x[i] = _det(i + 1, 2.0)
        go[i] = _det(i + 7, 1.0)
    var leg = LegacyLNNA[DIM].make[target="cpu", INIT=Kaiming]()
    var x_t = TileTensor(x, row_major[B, DIM]())
    var y_t = TileTensor(y, row_major[B, DIM]())
    var go_t = TileTensor(go, row_major[B, DIM]())
    var gi_t = TileTensor(gi, row_major[B, DIM]())
    leg.forward["cpu", B](x_t, output=y_t)
    leg.vjp["cpu", B](go_t, gi_t)

    var st = LayerNormNoAffine[DIM].make["cpu", Deterministic]()
    var sx = Tensor.alloc(N)
    var sgo = Tensor.alloc(N)
    var sout = Tensor.alloc(N)
    var sgi = Tensor.alloc(N)
    for i in range(N):
        sx.data[i] = x[i]
        sgo.data[i] = go[i]
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo = _maxd(sout, y, N)
    var mgi = _maxd(sgi, gi, N)
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < CPU_TOL and mgi < CPU_TOL, "LNNA CPU parity")
    x.free(); y.free(); go.free(); gi.free()
    print("  ok")


def test_lnna_gpu() raises:
    print("test_lnna_gpu (storage GPU vs CPU) ...")
    comptime N = B * DIM
    var c = DeviceContext()
    var cpu = LayerNormNoAffine[DIM].make["cpu", Deterministic]()
    var gpu = LayerNormNoAffine[DIM].make["gpu", Deterministic](Optional(c))
    var sx = Tensor.alloc(N)
    var sgo = Tensor.alloc(N)
    for i in range(N):
        sx.data[i] = _det(i + 1, 2.0)
        sgo.data[i] = _det(i + 7, 1.0)
    var c_out = Tensor.alloc(N)
    var c_gi = Tensor.alloc(N)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(N)
    var ggo = Tensor.alloc(N)
    for i in range(N):
        gx.data[i] = sx.data[i]
        ggo.data[i] = sgo.data[i]
    gx.upload(c); ggo.upload(c)
    var g_out = Tensor.alloc(N)
    var g_gi = Tensor.alloc(N)
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c); g_gi.download(c)

    var mo = _maxd2(g_out, c_out, N)
    var mgi = _maxd2(g_gi, c_gi, N)
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < GPU_TOL and mgi < GPU_TOL, "LNNA GPU vs CPU")
    print("  ok")


# ──────────────────────────────────────────────────────────────────────
# MSEPerSample (ARITY-2, OUT_DIM=1)
# ──────────────────────────────────────────────────────────────────────
def test_mps_cpu() raises:
    print("test_mps_cpu (legacy vs storage) ...")
    comptime N = B * DIM
    var a = _a(N)
    var bb = _a(N)
    var y = _a(B)
    var go = _a(B)
    var ga = _a(N)
    var gb = _a(N)
    for i in range(N):
        a[i] = _det(i + 1, 1.5)
        bb[i] = _det(i + 5, 1.3)
    for i in range(B):
        go[i] = _det(i + 3, 1.0)
    var leg = LegacyMPS[DIM].make[target="cpu", INIT=Kaiming]()
    var a_t = TileTensor(a, row_major[B, DIM]())
    var b_t = TileTensor(bb, row_major[B, DIM]())
    var y_t = TileTensor(y, row_major[B, 1]())
    var go_t = TileTensor(go, row_major[B, 1]())
    var ga_t = TileTensor(ga, row_major[B, DIM]())
    var gb_t = TileTensor(gb, row_major[B, DIM]())
    leg.forward["cpu", B](LegacyPack[2].of(a_t, b_t), output=y_t)
    leg.vjp["cpu", B](go_t, LegacyPack[2].of(ga_t, gb_t))

    var st = MSEPerSample[DIM].make["cpu", Deterministic]()
    var sin = TensorPack[2]()
    sin[0].ensure(N); sin[1].ensure(N)
    var sgo = Tensor.alloc(B)
    var sout = Tensor.alloc(B)
    var sg = TensorPack[2]()
    for i in range(N):
        sin[0].data[i] = a[i]; sin[1].data[i] = bb[i]
    for i in range(B):
        sgo.data[i] = go[i]
    st.forward["cpu", B](TensorRefs[2](sin[0], sin[1]), sout, None)
    st.vjp["cpu", B](
        TensorRefs[2](sin[0], sin[1]), sgo,
        TensorRefs[2](sg[0], sg[1]), None,
    )

    var mo = _maxd(sout, y, B)
    var mga = _maxd(sg[0], ga, N)
    var mgb = _maxd(sg[1], gb, N)
    print("  max Δ: out", mo, " ga", mga, " gb", mgb)
    assert_true(mo < CPU_TOL and mga < CPU_TOL and mgb < CPU_TOL,
                "MSEPerSample CPU parity")
    a.free(); bb.free(); y.free(); go.free(); ga.free(); gb.free()
    print("  ok")


def test_mps_gpu() raises:
    print("test_mps_gpu (storage GPU vs CPU) ...")
    comptime N = B * DIM
    var c = DeviceContext()
    var cpu = MSEPerSample[DIM].make["cpu", Deterministic]()
    var gpu = MSEPerSample[DIM].make["gpu", Deterministic](Optional(c))
    var cin = TensorPack[2]()
    cin[0].ensure(N); cin[1].ensure(N)
    var cgo = Tensor.alloc(B)
    for i in range(N):
        cin[0].data[i] = _det(i + 1, 1.5); cin[1].data[i] = _det(i + 5, 1.3)
    for i in range(B):
        cgo.data[i] = _det(i + 3, 1.0)
    var c_out = Tensor.alloc(B)
    var c_g = TensorPack[2]()
    cpu.forward["cpu", B](TensorRefs[2](cin[0], cin[1]), c_out, None)
    cpu.vjp["cpu", B](
        TensorRefs[2](cin[0], cin[1]), cgo,
        TensorRefs[2](c_g[0], c_g[1]), None,
    )

    var gin = TensorPack[2]()
    gin[0].ensure(N); gin[1].ensure(N)
    var ggo = Tensor.alloc(B)
    for i in range(N):
        gin[0].data[i] = cin[0].data[i]; gin[1].data[i] = cin[1].data[i]
    for i in range(B):
        ggo.data[i] = cgo.data[i]
    gin[0].upload(c); gin[1].upload(c); ggo.upload(c)
    var g_out = Tensor.alloc(B)
    var g_g = TensorPack[2]()
    gpu.forward["gpu", B](TensorRefs[2](gin[0], gin[1]), g_out, Optional(c))
    gpu.vjp["gpu", B](
        TensorRefs[2](gin[0], gin[1]), ggo,
        TensorRefs[2](g_g[0], g_g[1]), Optional(c),
    )
    g_out.download(c); g_g[0].download(c); g_g[1].download(c)

    var mo = _maxd2(g_out, c_out, B)
    var mga = _maxd2(g_g[0], c_g[0], N)
    var mgb = _maxd2(g_g[1], c_g[1], N)
    print("  max Δ: out", mo, " ga", mga, " gb", mgb)
    assert_true(mo < GPU_TOL and mga < GPU_TOL and mgb < GPU_TOL,
                "MSEPerSample GPU vs CPU")
    print("  ok")


# ──────────────────────────────────────────────────────────────────────
# Gate (ARITY-3)
# ──────────────────────────────────────────────────────────────────────
def test_gate_cpu() raises:
    print("test_gate_cpu (legacy vs storage) ...")
    comptime N = B * DIM
    var x = _a(N)
    var g = _a(N)
    var br = _a(N)
    var y = _a(N)
    var go = _a(N)
    var gx = _a(N)
    var gg = _a(N)
    var gbr = _a(N)
    for i in range(N):
        x[i] = _det(i + 1, 1.5); g[i] = _det(i + 5, 0.8)
        br[i] = _det(i + 9, 1.2); go[i] = _det(i + 13, 1.0)
    var leg = LegacyGate[DIM].make[target="cpu", INIT=Kaiming]()
    var x_t = TileTensor(x, row_major[B, DIM]())
    var g_t = TileTensor(g, row_major[B, DIM]())
    var br_t = TileTensor(br, row_major[B, DIM]())
    var y_t = TileTensor(y, row_major[B, DIM]())
    var go_t = TileTensor(go, row_major[B, DIM]())
    var gx_t = TileTensor(gx, row_major[B, DIM]())
    var gg_t = TileTensor(gg, row_major[B, DIM]())
    var gbr_t = TileTensor(gbr, row_major[B, DIM]())
    leg.forward["cpu", B](LegacyPack[3].of(x_t, g_t, br_t), output=y_t)
    leg.vjp["cpu", B](go_t, LegacyPack[3].of(gx_t, gg_t, gbr_t))

    var st = Gate[DIM].make["cpu", Deterministic]()
    var sin = TensorPack[3]()
    sin[0].ensure(N); sin[1].ensure(N); sin[2].ensure(N)
    var sgo = Tensor.alloc(N)
    var sout = Tensor.alloc(N)
    var sg = TensorPack[3]()
    for i in range(N):
        sin[0].data[i] = x[i]; sin[1].data[i] = g[i]; sin[2].data[i] = br[i]
        sgo.data[i] = go[i]
    st.forward["cpu", B](TensorRefs[3](sin[0], sin[1], sin[2]), sout, None)
    st.vjp["cpu", B](
        TensorRefs[3](sin[0], sin[1], sin[2]), sgo,
        TensorRefs[3](sg[0], sg[1], sg[2]), None,
    )

    var mo = _maxd(sout, y, N)
    var mgx = _maxd(sg[0], gx, N)
    var mgg = _maxd(sg[1], gg, N)
    var mgbr = _maxd(sg[2], gbr, N)
    print("  max Δ: out", mo, " gx", mgx, " gg", mgg, " gbr", mgbr)
    assert_true(
        mo < CPU_TOL and mgx < CPU_TOL and mgg < CPU_TOL and mgbr < CPU_TOL,
        "Gate CPU parity",
    )
    x.free(); g.free(); br.free(); y.free(); go.free()
    gx.free(); gg.free(); gbr.free()
    print("  ok")


def test_gate_gpu() raises:
    print("test_gate_gpu (storage GPU vs CPU) ...")
    comptime N = B * DIM
    var c = DeviceContext()
    var cpu = Gate[DIM].make["cpu", Deterministic]()
    var gpu = Gate[DIM].make["gpu", Deterministic](Optional(c))
    var cin = TensorPack[3]()
    cin[0].ensure(N); cin[1].ensure(N); cin[2].ensure(N)
    var cgo = Tensor.alloc(N)
    for i in range(N):
        cin[0].data[i] = _det(i + 1, 1.5); cin[1].data[i] = _det(i + 5, 0.8)
        cin[2].data[i] = _det(i + 9, 1.2); cgo.data[i] = _det(i + 13, 1.0)
    var c_out = Tensor.alloc(N)
    var c_g = TensorPack[3]()
    cpu.forward["cpu", B](TensorRefs[3](cin[0], cin[1], cin[2]), c_out, None)
    cpu.vjp["cpu", B](
        TensorRefs[3](cin[0], cin[1], cin[2]), cgo,
        TensorRefs[3](c_g[0], c_g[1], c_g[2]), None,
    )

    var gin = TensorPack[3]()
    gin[0].ensure(N); gin[1].ensure(N); gin[2].ensure(N)
    var ggo = Tensor.alloc(N)
    for i in range(N):
        gin[0].data[i] = cin[0].data[i]; gin[1].data[i] = cin[1].data[i]
        gin[2].data[i] = cin[2].data[i]; ggo.data[i] = cgo.data[i]
    gin[0].upload(c); gin[1].upload(c); gin[2].upload(c); ggo.upload(c)
    var g_out = Tensor.alloc(N)
    var g_g = TensorPack[3]()
    gpu.forward["gpu", B](
        TensorRefs[3](gin[0], gin[1], gin[2]), g_out, Optional(c)
    )
    gpu.vjp["gpu", B](
        TensorRefs[3](gin[0], gin[1], gin[2]), ggo,
        TensorRefs[3](g_g[0], g_g[1], g_g[2]), Optional(c),
    )
    g_out.download(c); g_g[0].download(c); g_g[1].download(c); g_g[2].download(c)

    var mo = _maxd2(g_out, c_out, N)
    var mgx = _maxd2(g_g[0], c_g[0], N)
    var mgg = _maxd2(g_g[1], c_g[1], N)
    var mgbr = _maxd2(g_g[2], c_g[2], N)
    print("  max Δ: out", mo, " gx", mgx, " gg", mgg, " gbr", mgbr)
    assert_true(
        mo < GPU_TOL and mgx < GPU_TOL and mgg < GPU_TOL and mgbr < GPU_TOL,
        "Gate GPU vs CPU",
    )
    print("  ok")


# ──────────────────────────────────────────────────────────────────────
# Modulate (ARITY-3)
# ──────────────────────────────────────────────────────────────────────
def test_modulate_cpu() raises:
    print("test_modulate_cpu (legacy vs storage) ...")
    comptime N = B * DIM
    var x = _a(N)
    var sc = _a(N)
    var sh = _a(N)
    var y = _a(N)
    var go = _a(N)
    var gx = _a(N)
    var gs = _a(N)
    var gsh = _a(N)
    for i in range(N):
        x[i] = _det(i + 1, 1.5); sc[i] = _det(i + 5, 0.8)
        sh[i] = _det(i + 9, 0.5); go[i] = _det(i + 13, 1.0)
    var leg = LegacyModulate[DIM].make[target="cpu", INIT=Kaiming]()
    var x_t = TileTensor(x, row_major[B, DIM]())
    var sc_t = TileTensor(sc, row_major[B, DIM]())
    var sh_t = TileTensor(sh, row_major[B, DIM]())
    var y_t = TileTensor(y, row_major[B, DIM]())
    var go_t = TileTensor(go, row_major[B, DIM]())
    var gx_t = TileTensor(gx, row_major[B, DIM]())
    var gs_t = TileTensor(gs, row_major[B, DIM]())
    var gsh_t = TileTensor(gsh, row_major[B, DIM]())
    leg.forward["cpu", B](LegacyPack[3].of(x_t, sc_t, sh_t), output=y_t)
    leg.vjp["cpu", B](go_t, LegacyPack[3].of(gx_t, gs_t, gsh_t))

    var st = Modulate[DIM].make["cpu", Deterministic]()
    var sin = TensorPack[3]()
    sin[0].ensure(N); sin[1].ensure(N); sin[2].ensure(N)
    var sgo = Tensor.alloc(N)
    var sout = Tensor.alloc(N)
    var sg = TensorPack[3]()
    for i in range(N):
        sin[0].data[i] = x[i]; sin[1].data[i] = sc[i]; sin[2].data[i] = sh[i]
        sgo.data[i] = go[i]
    st.forward["cpu", B](TensorRefs[3](sin[0], sin[1], sin[2]), sout, None)
    st.vjp["cpu", B](
        TensorRefs[3](sin[0], sin[1], sin[2]), sgo,
        TensorRefs[3](sg[0], sg[1], sg[2]), None,
    )

    var mo = _maxd(sout, y, N)
    var mgx = _maxd(sg[0], gx, N)
    var mgs = _maxd(sg[1], gs, N)
    var mgsh = _maxd(sg[2], gsh, N)
    print("  max Δ: out", mo, " gx", mgx, " gs", mgs, " gsh", mgsh)
    assert_true(
        mo < CPU_TOL and mgx < CPU_TOL and mgs < CPU_TOL and mgsh < CPU_TOL,
        "Modulate CPU parity",
    )
    x.free(); sc.free(); sh.free(); y.free(); go.free()
    gx.free(); gs.free(); gsh.free()
    print("  ok")


def test_modulate_gpu() raises:
    print("test_modulate_gpu (storage GPU vs CPU) ...")
    comptime N = B * DIM
    var c = DeviceContext()
    var cpu = Modulate[DIM].make["cpu", Deterministic]()
    var gpu = Modulate[DIM].make["gpu", Deterministic](Optional(c))
    var cin = TensorPack[3]()
    cin[0].ensure(N); cin[1].ensure(N); cin[2].ensure(N)
    var cgo = Tensor.alloc(N)
    for i in range(N):
        cin[0].data[i] = _det(i + 1, 1.5); cin[1].data[i] = _det(i + 5, 0.8)
        cin[2].data[i] = _det(i + 9, 0.5); cgo.data[i] = _det(i + 13, 1.0)
    var c_out = Tensor.alloc(N)
    var c_g = TensorPack[3]()
    cpu.forward["cpu", B](TensorRefs[3](cin[0], cin[1], cin[2]), c_out, None)
    cpu.vjp["cpu", B](
        TensorRefs[3](cin[0], cin[1], cin[2]), cgo,
        TensorRefs[3](c_g[0], c_g[1], c_g[2]), None,
    )

    var gin = TensorPack[3]()
    gin[0].ensure(N); gin[1].ensure(N); gin[2].ensure(N)
    var ggo = Tensor.alloc(N)
    for i in range(N):
        gin[0].data[i] = cin[0].data[i]; gin[1].data[i] = cin[1].data[i]
        gin[2].data[i] = cin[2].data[i]; ggo.data[i] = cgo.data[i]
    gin[0].upload(c); gin[1].upload(c); gin[2].upload(c); ggo.upload(c)
    var g_out = Tensor.alloc(N)
    var g_g = TensorPack[3]()
    gpu.forward["gpu", B](
        TensorRefs[3](gin[0], gin[1], gin[2]), g_out, Optional(c)
    )
    gpu.vjp["gpu", B](
        TensorRefs[3](gin[0], gin[1], gin[2]), ggo,
        TensorRefs[3](g_g[0], g_g[1], g_g[2]), Optional(c),
    )
    g_out.download(c); g_g[0].download(c); g_g[1].download(c); g_g[2].download(c)

    var mo = _maxd2(g_out, c_out, N)
    var mgx = _maxd2(g_g[0], c_g[0], N)
    var mgs = _maxd2(g_g[1], c_g[1], N)
    var mgsh = _maxd2(g_g[2], c_g[2], N)
    print("  max Δ: out", mo, " gx", mgx, " gs", mgs, " gsh", mgsh)
    assert_true(
        mo < GPU_TOL and mgx < GPU_TOL and mgs < GPU_TOL and mgsh < GPU_TOL,
        "Modulate GPU vs CPU",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("LeWM small leaves — legacy↔storage parity + storage GPU-vs-CPU")
    print("=" * 70)
    test_lnna_cpu()
    test_lnna_gpu()
    test_mps_cpu()
    test_mps_gpu()
    test_gate_cpu()
    test_gate_gpu()
    test_modulate_cpu()
    test_modulate_gpu()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
