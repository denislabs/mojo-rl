"""GRUCell storage primitive — CPU correctness (golden) + GPU vs CPU.

Standalone storage test (no legacy oracle — converted from the former
`_storage_parity` test in legacy-removal Phase 0b). GRU is the HARD recurrent
leaf: the storage surface folds the two-phase backward into ONE single-phase
`vjp` (recompute gate grads from `forward_input` + caches — no aliasing between
the forward_input and grad_inputs packs). The CPU check drives BOTH a single
cell step and a short 3-step unroll (re-feeding the new hidden), then asserts
the storage forward/backward against golden fingerprints (S = Σ vᵢ,
W = Σ vᵢ·(i+1) — the weight catches sign/position errors a plain sum cancels),
captured from the bit-identical legacy↔storage run the parity test verified.
The GPU check is storage-only (GPU vs CPU consistency).

The two inputs (x, h) live in a storage `TensorPack[2]` so the pair shares one
origin (the §B0 constraint on `TensorRefs[2]`). Run both:
  rm -f mojo_rl.mojoc && pixi run mojo run -I . tests/nn/test_gru_storage.mojo
  rm -f mojo_rl.mojoc && pixi run -e apple mojo run -I . tests/nn/test_gru_storage.mojo
"""

from std.math import abs as fabs
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.gru_cell import GRUCell


comptime IN = 4
comptime H = 3
comptime B = 5


def _wval(k: Int) -> Scalar[DT]:
    return Scalar[DT](0.25 - 0.017 * Float64(k % 11))


def _bval(k: Int) -> Scalar[DT]:
    return Scalar[DT](-0.13 + 0.041 * Float64(k % 7))


def _check(name: String, data: Tensor, n: Int,
           es: Scalar[DT], ew: Scalar[DT], tol: Scalar[DT]) -> Bool:
    """Assert tensor fingerprint (Σ vᵢ, Σ vᵢ·(i+1)) matches golden (es, ew)."""
    var s: Scalar[DT] = 0
    var w: Scalar[DT] = 0
    for i in range(n):
        s += data.data[i]
        w += data.data[i] * Scalar[DT](i + 1)
    var ok = abs(s - es) < tol and abs(w - ew) < tol
    print("  ", name, "S", s, "(exp", es, ") W", w, "(exp", ew, ")", "OK" if ok else "FAIL")
    return ok


def test_gru_cpu_golden() raises:
    print("test_gru_cpu_golden (storage CPU vs golden, single + unroll) ...")
    comptime TOL = Scalar[DT](5e-3)
    comptime WIH = GRUCell[IN, H].W_IH_SIZE
    comptime WHH = GRUCell[IN, H].W_HH_SIZE
    comptime BIH = GRUCell[IN, H].B_IH_SIZE

    var st = GRUCell[IN, H].make["cpu", Deterministic]()
    for k in range(WIH):
        st.W_ih.val.data[k] = _wval(k)
    for k in range(WHH):
        st.W_hh.val.data[k] = _wval(k + 3)
    for k in range(BIH):
        st.b_ih.val.data[k] = _bval(k)
        st.b_hh.val.data[k] = _bval(k + 2)

    var sin = TensorPack[2]()
    sin[0].ensure(B * IN)
    sin[1].ensure(B * H)
    var sgo = Tensor.alloc(B * H)
    var sout = Tensor.alloc(B * H)
    var sgrad = TensorPack[2]()
    sgrad[0].ensure(B * IN)
    sgrad[1].ensure(B * H)
    for i in range(B * IN):
        sin[0].data[i] = Scalar[DT]((i % 13) - 6) * 0.11
    for i in range(B * H):
        sin[1].data[i] = Scalar[DT]((i % 9) - 4) * 0.07
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.19

    st.forward["cpu", B](TensorRefs[2](sin[0], sin[1]), sout, None)
    st.zero_grad["cpu"](None)
    st.vjp["cpu", B](
        TensorRefs[2](sin[0], sin[1]), sgo,
        TensorRefs[2](sgrad[0], sgrad[1]), None,
    )

    var ok = _check("out", sout, B * H, -1.4211716, -9.119543, TOL)
    ok = _check("dx", sgrad[0], B * IN, -0.22898662, 1.4642927, TOL) and ok
    ok = _check("dh", sgrad[1], B * H, -0.2893071, 0.77405244, TOL) and ok
    ok = _check("dWih", st.W_ih.grd, WIH, 0.6631763, 11.810107, TOL) and ok
    ok = _check("dWhh", st.W_hh.grd, WHH, 0.03733717, 0.38114747, TOL) and ok
    ok = _check("dbih", st.b_ih.grd, BIH, -0.34344143, -2.8744161, TOL) and ok
    ok = _check("dbhh", st.b_hh.grd, BIH, -0.154402, -1.2920494, TOL) and ok
    assert_true(ok, "GRU single-step CPU golden")

    # ----- short 3-step unroll (re-feed hidden) -----
    st.zero_grad["cpu"](None)
    for i in range(B * H):
        sin[1].data[i] = Scalar[DT]((i % 9) - 4) * 0.07
    var unroll_out = Tensor.alloc(B * H)
    for step in range(3):
        for i in range(B * IN):
            sin[0].data[i] = Scalar[DT](((i + step) % 13) - 6) * 0.11
        st.forward["cpu", B](TensorRefs[2](sin[0], sin[1]), unroll_out, None)
        for i in range(B * H):
            sin[1].data[i] = unroll_out.data[i]
    var oku = _check("unroll_out", unroll_out, B * H, -1.3588077, -10.277919, TOL)
    assert_true(oku, "GRU unroll CPU golden")
    print("  ok")


def test_gru_gpu_vs_cpu() raises:
    print("test_gru_gpu_vs_cpu (storage GPU vs storage CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    comptime WIH = GRUCell[IN, H].W_IH_SIZE
    comptime WHH = GRUCell[IN, H].W_HH_SIZE
    comptime BIH = GRUCell[IN, H].B_IH_SIZE
    var c = DeviceContext()

    var cpu = GRUCell[IN, H].make["cpu", Deterministic]()
    var gpu = GRUCell[IN, H].make["gpu", Deterministic](Optional(c))
    for k in range(WIH):
        cpu.W_ih.val.data[k] = _wval(k)
        gpu.W_ih.val.data[k] = cpu.W_ih.val.data[k]
    for k in range(WHH):
        cpu.W_hh.val.data[k] = _wval(k + 3)
        gpu.W_hh.val.data[k] = cpu.W_hh.val.data[k]
    for k in range(BIH):
        cpu.b_ih.val.data[k] = _bval(k)
        cpu.b_hh.val.data[k] = _bval(k + 2)
        gpu.b_ih.val.data[k] = cpu.b_ih.val.data[k]
        gpu.b_hh.val.data[k] = cpu.b_hh.val.data[k]
    gpu.W_ih.val.upload(c)
    gpu.W_hh.val.upload(c)
    gpu.b_ih.val.upload(c)
    gpu.b_hh.val.upload(c)

    # ----- CPU run -----
    var cin = TensorPack[2]()
    cin[0].ensure(B * IN)
    cin[1].ensure(B * H)
    var sgo = Tensor.alloc(B * H)
    for i in range(B * IN):
        cin[0].data[i] = Scalar[DT]((i % 13) - 6) * 0.11
    for i in range(B * H):
        cin[1].data[i] = Scalar[DT]((i % 9) - 4) * 0.07
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.19
    var c_out = Tensor.alloc(B * H)
    var cgrad = TensorPack[2]()
    cgrad[0].ensure(B * IN)
    cgrad[1].ensure(B * H)
    cpu.forward["cpu", B](TensorRefs[2](cin[0], cin[1]), c_out, None)
    cpu.zero_grad["cpu"](None)
    cpu.vjp["cpu", B](
        TensorRefs[2](cin[0], cin[1]), sgo,
        TensorRefs[2](cgrad[0], cgrad[1]), None,
    )

    # ----- GPU run -----
    var gin = TensorPack[2]()
    gin[0].ensure(B * IN)
    gin[1].ensure(B * H)
    var ggo = Tensor.alloc(B * H)
    for i in range(B * IN):
        gin[0].data[i] = cin[0].data[i]
    for i in range(B * H):
        gin[1].data[i] = cin[1].data[i]
        ggo.data[i] = sgo.data[i]
    gin[0].upload(c)
    gin[1].upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(B * H)
    var ggrad = TensorPack[2]()
    ggrad[0].ensure(B * IN)
    ggrad[1].ensure(B * H)
    gpu.forward["gpu", B](TensorRefs[2](gin[0], gin[1]), g_out, Optional(c))
    gpu.zero_grad["gpu"](Optional(c))
    gpu.vjp["gpu", B](
        TensorRefs[2](gin[0], gin[1]), ggo,
        TensorRefs[2](ggrad[0], ggrad[1]), Optional(c),
    )
    g_out.download(c)
    ggrad[0].download(c)
    ggrad[1].download(c)
    gpu.W_ih.grd.download(c)
    gpu.W_hh.grd.download(c)
    gpu.b_ih.grd.download(c)
    gpu.b_hh.grd.download(c)

    var mo: Scalar[DT] = 0
    var mdx: Scalar[DT] = 0
    var mdh: Scalar[DT] = 0
    for i in range(B * H):
        if fabs(g_out.data[i] - c_out.data[i]) > mo:
            mo = fabs(g_out.data[i] - c_out.data[i])
        if fabs(ggrad[1].data[i] - cgrad[1].data[i]) > mdh:
            mdh = fabs(ggrad[1].data[i] - cgrad[1].data[i])
    for i in range(B * IN):
        if fabs(ggrad[0].data[i] - cgrad[0].data[i]) > mdx:
            mdx = fabs(ggrad[0].data[i] - cgrad[0].data[i])
    var m_wih: Scalar[DT] = 0
    for k in range(WIH):
        if fabs(gpu.W_ih.grd.data[k] - cpu.W_ih.grd.data[k]) > m_wih:
            m_wih = fabs(gpu.W_ih.grd.data[k] - cpu.W_ih.grd.data[k])
    var m_whh: Scalar[DT] = 0
    for k in range(WHH):
        if fabs(gpu.W_hh.grd.data[k] - cpu.W_hh.grd.data[k]) > m_whh:
            m_whh = fabs(gpu.W_hh.grd.data[k] - cpu.W_hh.grd.data[k])
    var m_bih: Scalar[DT] = 0
    var m_bhh: Scalar[DT] = 0
    for k in range(BIH):
        if fabs(gpu.b_ih.grd.data[k] - cpu.b_ih.grd.data[k]) > m_bih:
            m_bih = fabs(gpu.b_ih.grd.data[k] - cpu.b_ih.grd.data[k])
        if fabs(gpu.b_hh.grd.data[k] - cpu.b_hh.grd.data[k]) > m_bhh:
            m_bhh = fabs(gpu.b_hh.grd.data[k] - cpu.b_hh.grd.data[k])
    print("  out", mo, " dx", mdx, " dh", mdh)
    print("  dWih", m_wih, " dWhh", m_whh, " dbih", m_bih, " dbhh", m_bhh)
    assert_true(
        mo < TOL and mdx < TOL and mdh < TOL and m_wih < TOL
        and m_whh < TOL and m_bih < TOL and m_bhh < TOL,
        "GRU GPU vs CPU",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("GRUCell storage primitive (CPU golden + GPU vs CPU)")
    print("=" * 70)
    test_gru_cpu_golden()
    test_gru_gpu_vs_cpu()
    print("ALL PASSED")
