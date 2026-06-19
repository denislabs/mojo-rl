"""GRUCell legacy ↔ storage parity (CPU) + storage GPU vs CPU.

GRU is the HARD recurrent leaf: legacy used a SPLIT two-phase vjp; the storage
surface folds it into ONE single-phase `vjp` (recompute gate grads from
`forward_input` + caches — no aliasing between the forward_input and grad_inputs
packs). This test drives BOTH a single cell step and a short 3-step unroll
(re-feeding the new hidden) so the recurrent path is exercised.

The two inputs (x, h) live in a storage `TensorPack[2]` so the pair shares one
origin (the §B0 constraint on `TensorRefs[2]`). legacy GRU is driven with the
legacy nn `TensorPack`.

CPU: legacy GRUCell vs storage GRUCell with identical weights/x/h — max|Δ| < 1e-6
on out + grad_x + grad_h + all four param grads after backward.
GPU: storage GPU vs storage CPU, TOL ~2e-5. Run both:
  rm -f mojo_rl.mojoc && pixi run mojo run -I . tests/nn/test_gru_storage_parity.mojo
  rm -f mojo_rl.mojoc && pixi run -e apple mojo run -I . tests/nn/test_gru_storage_parity.mojo
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor_pack import TensorPack as LegacyTensorPack
from mojo_rl.nn.core.module import mptr
from mojo_rl.nn.primitives.gru_cell import GRUCell as LegacyGRUCell
from mojo_rl.nn.initializer import Zero as LegacyZero
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.tensor_pack import TensorPack
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.gru_cell import GRUCell


comptime IN = 4
comptime H = 3
comptime B = 5


def _wval(k: Int) -> Scalar[DT]:
    return Scalar[DT](0.25 - 0.017 * Float64(k % 11))


def _bval(k: Int) -> Scalar[DT]:
    return Scalar[DT](-0.13 + 0.041 * Float64(k % 7))


def _legpack2(
    p0: UnsafePointer[Scalar[DT], MutAnyOrigin],
    p1: UnsafePointer[Scalar[DT], MutAnyOrigin],
) -> LegacyTensorPack[2]:
    """Build a legacy 2-pack from two raw base pointers (the variadic
    `of()` requires identical layouts → unusable when IN != H)."""
    var ps = InlineArray[
        UnsafePointer[Scalar[DT], MutAnyOrigin], 2
    ](uninitialized=True)
    ps[0] = mptr(p0)
    ps[1] = mptr(p1)
    return LegacyTensorPack[2](ps^)


def test_gru_cpu_parity() raises:
    print("test_gru_cpu_parity (legacy vs storage, CPU, single + unroll) ...")
    comptime TOL = Scalar[DT](1e-6)
    comptime WIH = GRUCell[IN, H].W_IH_SIZE
    comptime WHH = GRUCell[IN, H].W_HH_SIZE
    comptime BIH = GRUCell[IN, H].B_IH_SIZE

    # ----- legacy cell -----
    var leg = LegacyGRUCell[IN, H].make[target="cpu", INIT=LegacyZero]()
    var lwih = leg.W_ih.value_unsafe_ptr_cpu()
    var lwhh = leg.W_hh.value_unsafe_ptr_cpu()
    var lbih = leg.b_ih.value_unsafe_ptr_cpu()
    var lbhh = leg.b_hh.value_unsafe_ptr_cpu()
    for k in range(WIH):
        lwih[k] = _wval(k)
    for k in range(WHH):
        lwhh[k] = _wval(k + 3)
    for k in range(BIH):
        lbih[k] = _bval(k)
        lbhh[k] = _bval(k + 2)

    # ----- storage cell (copy identical weights) -----
    var st = GRUCell[IN, H].make["cpu", Deterministic]()
    for k in range(WIH):
        st.W_ih.val.data[k] = lwih[k]
    for k in range(WHH):
        st.W_hh.val.data[k] = lwhh[k]
    for k in range(BIH):
        st.b_ih.val.data[k] = lbih[k]
        st.b_hh.val.data[k] = lbhh[k]

    # ----- shared inputs -----
    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * IN)
    var h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
    for i in range(B * IN):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.11
    for i in range(B * H):
        h[i] = Scalar[DT]((i % 9) - 4) * 0.07
    for i in range(B * H):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.19

    # ----- legacy fwd + bwd (single step) -----
    var ly: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
    var ldx: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * IN)
    var ldh: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
    var ly_t = TileTensor(ly, row_major[B, H]())
    var lgo_t = TileTensor(go, row_major[B, H]())
    leg.forward["cpu", B](_legpack2(x, h), output=ly_t)
    leg.zero_grad["cpu"]()
    leg.vjp["cpu", B](lgo_t, _legpack2(ldx, ldh))

    # ----- storage fwd + bwd (single step) -----
    var sin = TensorPack[2]()
    sin[0].ensure(B * IN)
    sin[1].ensure(B * H)
    var sgo = Tensor.alloc(B * H)
    var sout = Tensor.alloc(B * H)
    var sgrad = TensorPack[2]()
    sgrad[0].ensure(B * IN)
    sgrad[1].ensure(B * H)
    for i in range(B * IN):
        sin[0].data[i] = x[i]
    for i in range(B * H):
        sin[1].data[i] = h[i]
        sgo.data[i] = go[i]
    st.forward["cpu", B](TensorRefs[2](sin[0], sin[1]), sout, None)
    st.zero_grad["cpu"](None)
    st.vjp["cpu", B](
        TensorRefs[2](sin[0], sin[1]), sgo,
        TensorRefs[2](sgrad[0], sgrad[1]), None,
    )

    var mo: Scalar[DT] = 0
    var mdx: Scalar[DT] = 0
    var mdh: Scalar[DT] = 0
    for i in range(B * H):
        if fabs(sout.data[i] - ly[i]) > mo: mo = fabs(sout.data[i] - ly[i])
        if fabs(sgrad[1].data[i] - ldh[i]) > mdh:
            mdh = fabs(sgrad[1].data[i] - ldh[i])
    for i in range(B * IN):
        if fabs(sgrad[0].data[i] - ldx[i]) > mdx:
            mdx = fabs(sgrad[0].data[i] - ldx[i])
    var m_wih: Scalar[DT] = 0
    for k in range(WIH):
        if fabs(st.W_ih.grd.data[k] - leg.W_ih.grd.cpu[k]) > m_wih:
            m_wih = fabs(st.W_ih.grd.data[k] - leg.W_ih.grd.cpu[k])
    var m_whh: Scalar[DT] = 0
    for k in range(WHH):
        if fabs(st.W_hh.grd.data[k] - leg.W_hh.grd.cpu[k]) > m_whh:
            m_whh = fabs(st.W_hh.grd.data[k] - leg.W_hh.grd.cpu[k])
    var m_bih: Scalar[DT] = 0
    var m_bhh: Scalar[DT] = 0
    for k in range(BIH):
        if fabs(st.b_ih.grd.data[k] - leg.b_ih.grd.cpu[k]) > m_bih:
            m_bih = fabs(st.b_ih.grd.data[k] - leg.b_ih.grd.cpu[k])
        if fabs(st.b_hh.grd.data[k] - leg.b_hh.grd.cpu[k]) > m_bhh:
            m_bhh = fabs(st.b_hh.grd.data[k] - leg.b_hh.grd.cpu[k])
    print("  single: out", mo, " dx", mdx, " dh", mdh)
    print("          dWih", m_wih, " dWhh", m_whh, " dbih", m_bih, " dbhh", m_bhh)
    assert_true(
        mo < TOL and mdx < TOL and mdh < TOL and m_wih < TOL
        and m_whh < TOL and m_bih < TOL and m_bhh < TOL,
        "GRU single-step CPU parity",
    )

    # ----- short 3-step unroll (re-feed hidden) -----
    leg.zero_grad["cpu"]()
    st.zero_grad["cpu"](None)
    var lh_run: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
    for i in range(B * H):
        lh_run[i] = h[i]
        sin[1].data[i] = h[i]
    var max_unroll: Scalar[DT] = 0
    for step in range(3):
        for i in range(B * IN):
            x[i] = Scalar[DT](((i + step) % 13) - 6) * 0.11
            sin[0].data[i] = x[i]
        var lyr_t = TileTensor(ly, row_major[B, H]())
        leg.forward["cpu", B](_legpack2(x, lh_run), output=lyr_t)
        st.forward["cpu", B](TensorRefs[2](sin[0], sin[1]), sout, None)
        for i in range(B * H):
            if fabs(sout.data[i] - ly[i]) > max_unroll:
                max_unroll = fabs(sout.data[i] - ly[i])
            lh_run[i] = ly[i]
            sin[1].data[i] = sout.data[i]
    print("  unroll(3) max out Δ", max_unroll)
    assert_true(max_unroll < TOL, "GRU unroll CPU parity")

    x.free(); h.free(); go.free(); ly.free(); ldx.free(); ldh.free()
    lh_run.free()
    print("  ok")


def test_gru_gpu_parity() raises:
    print("test_gru_gpu_parity (storage GPU vs storage CPU) ...")
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
        "GRU GPU vs CPU parity",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("GRUCell legacy ↔ storage parity")
    print("=" * 70)
    test_gru_cpu_parity()
    test_gru_gpu_parity()
    print("ALL PASSED")
