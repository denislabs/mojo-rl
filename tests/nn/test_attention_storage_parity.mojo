"""ScaledDotProductAttention legacy ↔ storage parity (CPU) + storage GPU vs CPU.

Both paths carry the same QKᵀ/softmax/attn·V kernels + bmm glue verbatim, so
legacy↔storage CPU is bit-identical (out + grad_inputs). storage GPU↔CPU to a
small tolerance. Covers non-causal and causal, multi-head, BATCH>1. Run:
  rm -f mojo_rl.mojoc && pixi run mojo run -I . tests/nn/test_attention_storage_parity.mojo
  rm -f mojo_rl.mojoc && pixi run -e apple mojo run -I . tests/nn/test_attention_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.attention import (
    ScaledDotProductAttention as LegacyAttention,
)
from mojo_rl.nn.primitives.masked_attention import (
    MaskedAttention as LegacyMasked,
    causal_mask as legacy_causal_mask,
    build_modality_mask as legacy_modality_mask,
)
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.attention import ScaledDotProductAttention
from mojo_rl.nn.storage.primitives.masked_attention import (
    MaskedAttention,
    causal_mask,
    build_modality_mask,
)


comptime DIM = 8          # N_HEADS=2, head_dim=4
comptime N_HEADS = 2
comptime SEQ = 3
comptime BATCH = 2
comptime IN = SEQ * DIM * 3
comptime OUT = SEQ * DIM


def _cpu_parity[CAUSAL: Bool]() raises:
    print("  cpu parity CAUSAL=", CAUSAL, " ...")
    comptime TOL = Scalar[DT](1e-6)
    # Legacy (bmm path, USE_MAX_KERNELS default True).
    var leg = LegacyAttention[
        DIM, N_HEADS, SEQ, CAUSAL
    ].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * IN
    )
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * OUT
    )
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * OUT
    )
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * IN
    )
    for i in range(BATCH * IN):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.13
    for i in range(BATCH * OUT):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.21

    var x_t = TileTensor(x, row_major[BATCH, IN]())
    var y_t = TileTensor(y, row_major[BATCH, OUT]())
    var go_t = TileTensor(go, row_major[BATCH, OUT]())
    var gi_t = TileTensor(gi, row_major[BATCH, IN]())
    leg.forward["cpu", BATCH](x_t, output=y_t)
    leg.vjp["cpu", BATCH](go_t, gi_t)

    # Storage.
    var st = ScaledDotProductAttention[
        DIM, N_HEADS, SEQ, CAUSAL
    ].make["cpu", Deterministic]()
    var sx = Tensor.alloc(BATCH * IN)
    var sgo = Tensor.alloc(BATCH * OUT)
    var sout = Tensor.alloc(BATCH * OUT)
    var sgi = Tensor.alloc(BATCH * IN)
    for i in range(BATCH * IN):
        sx.data[i] = x[i]
    for i in range(BATCH * OUT):
        sgo.data[i] = go[i]
    st.forward["cpu", BATCH](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", BATCH](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    for i in range(BATCH * OUT):
        if abs(sout.data[i] - y[i]) > mo:
            mo = abs(sout.data[i] - y[i])
    var mgi: Scalar[DT] = 0
    for i in range(BATCH * IN):
        if abs(sgi.data[i] - gi[i]) > mgi:
            mgi = abs(sgi.data[i] - gi[i])
    print("    max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "Attention CPU parity")
    print("    ok")


def _gpu_parity[CAUSAL: Bool]() raises:
    print("  gpu vs cpu CAUSAL=", CAUSAL, " ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = ScaledDotProductAttention[
        DIM, N_HEADS, SEQ, CAUSAL
    ].make["cpu", Deterministic]()
    var gpu = ScaledDotProductAttention[
        DIM, N_HEADS, SEQ, CAUSAL
    ].make["gpu", Deterministic](Optional(c))

    var sx = Tensor.alloc(BATCH * IN)
    var sgo = Tensor.alloc(BATCH * OUT)
    for i in range(BATCH * IN):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.13
    for i in range(BATCH * OUT):
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.21
    var c_out = Tensor.alloc(BATCH * OUT)
    var c_gi = Tensor.alloc(BATCH * IN)
    cpu.forward["cpu", BATCH](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", BATCH](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(BATCH * IN)
    var ggo = Tensor.alloc(BATCH * OUT)
    for i in range(BATCH * IN):
        gx.data[i] = sx.data[i]
    for i in range(BATCH * OUT):
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(BATCH * OUT)
    var g_gi = Tensor.alloc(BATCH * IN)
    g_out.upload(c)
    g_gi.upload(c)
    gpu.forward["gpu", BATCH](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", BATCH](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)

    var mo: Scalar[DT] = 0
    for i in range(BATCH * OUT):
        if abs(g_out.data[i] - c_out.data[i]) > mo:
            mo = abs(g_out.data[i] - c_out.data[i])
    var mgi: Scalar[DT] = 0
    for i in range(BATCH * IN):
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi:
            mgi = abs(g_gi.data[i] - c_gi.data[i])
    print("    max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "Attention GPU vs CPU")
    print("    ok")


# ----- Masked attention -------------------------------------------------
# kind: 0=all-allow, 1=causal, 2=modality("encoder").


def _mask_for(kind: Int) raises -> List[Scalar[DT]]:
    if kind == 1:
        return causal_mask(SEQ)
    elif kind == 2:
        var mods = [0, 0, 1]
        return build_modality_mask["encoder"](mods, n_latents=1)
    else:
        var m = List[Scalar[DT]]()
        for _ in range(SEQ * SEQ):
            m.append(Scalar[DT](0.0))
        return m^


def _legacy_mask_for(kind: Int) raises -> List[Scalar[DT]]:
    if kind == 1:
        return legacy_causal_mask(SEQ)
    elif kind == 2:
        var mods = [0, 0, 1]
        return legacy_modality_mask["encoder"](mods, n_latents=1)
    else:
        var m = List[Scalar[DT]]()
        for _ in range(SEQ * SEQ):
            m.append(Scalar[DT](0.0))
        return m^


def _masked_cpu_parity(kind: Int) raises:
    print("  masked cpu parity kind=", kind, " ...")
    comptime TOL = Scalar[DT](1e-6)
    var leg = LegacyMasked[DIM, N_HEADS, SEQ].make[target="cpu", INIT=Zero]()
    leg.set_mask(_legacy_mask_for(kind))

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * IN
    )
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * OUT
    )
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * OUT
    )
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * IN
    )
    for i in range(BATCH * IN):
        x[i] = Scalar[DT]((i % 13) - 6) * 0.13
    for i in range(BATCH * OUT):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.21

    var x_t = TileTensor(x, row_major[BATCH, IN]())
    var y_t = TileTensor(y, row_major[BATCH, OUT]())
    var go_t = TileTensor(go, row_major[BATCH, OUT]())
    var gi_t = TileTensor(gi, row_major[BATCH, IN]())
    leg.forward["cpu", BATCH](x_t, output=y_t)
    leg.vjp["cpu", BATCH](go_t, gi_t)

    var st = MaskedAttention[DIM, N_HEADS, SEQ].make["cpu", Deterministic]()
    st.set_mask(_mask_for(kind))
    var sx = Tensor.alloc(BATCH * IN)
    var sgo = Tensor.alloc(BATCH * OUT)
    var sout = Tensor.alloc(BATCH * OUT)
    var sgi = Tensor.alloc(BATCH * IN)
    for i in range(BATCH * IN):
        sx.data[i] = x[i]
    for i in range(BATCH * OUT):
        sgo.data[i] = go[i]
    st.forward["cpu", BATCH](TensorRefs[1](sx), sout, None)
    st.vjp["cpu", BATCH](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    for i in range(BATCH * OUT):
        if abs(sout.data[i] - y[i]) > mo:
            mo = abs(sout.data[i] - y[i])
    var mgi: Scalar[DT] = 0
    for i in range(BATCH * IN):
        if abs(sgi.data[i] - gi[i]) > mgi:
            mgi = abs(sgi.data[i] - gi[i])
    print("    max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "MaskedAttention CPU parity")
    print("    ok")


def _masked_gpu_parity(kind: Int) raises:
    print("  masked gpu vs cpu kind=", kind, " ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = MaskedAttention[DIM, N_HEADS, SEQ].make["cpu", Deterministic]()
    cpu.set_mask(_mask_for(kind))
    var gpu = MaskedAttention[DIM, N_HEADS, SEQ].make["gpu", Deterministic](
        Optional(c)
    )
    gpu.set_mask(_mask_for(kind), Optional(c))

    var sx = Tensor.alloc(BATCH * IN)
    var sgo = Tensor.alloc(BATCH * OUT)
    for i in range(BATCH * IN):
        sx.data[i] = Scalar[DT]((i % 13) - 6) * 0.13
    for i in range(BATCH * OUT):
        sgo.data[i] = Scalar[DT]((i % 7) - 3) * 0.21
    var c_out = Tensor.alloc(BATCH * OUT)
    var c_gi = Tensor.alloc(BATCH * IN)
    cpu.forward["cpu", BATCH](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", BATCH](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(BATCH * IN)
    var ggo = Tensor.alloc(BATCH * OUT)
    for i in range(BATCH * IN):
        gx.data[i] = sx.data[i]
    for i in range(BATCH * OUT):
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(BATCH * OUT)
    var g_gi = Tensor.alloc(BATCH * IN)
    g_out.upload(c)
    g_gi.upload(c)
    gpu.forward["gpu", BATCH](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", BATCH](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)

    var mo: Scalar[DT] = 0
    for i in range(BATCH * OUT):
        if abs(g_out.data[i] - c_out.data[i]) > mo:
            mo = abs(g_out.data[i] - c_out.data[i])
    var mgi: Scalar[DT] = 0
    for i in range(BATCH * IN):
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi:
            mgi = abs(g_gi.data[i] - c_gi.data[i])
    print("    max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "MaskedAttention GPU vs CPU")
    print("    ok")


def test_cpu_parity() raises:
    print("test_attention_cpu_parity (legacy vs storage) ...")
    _cpu_parity[False]()
    _cpu_parity[True]()


def test_gpu_parity() raises:
    print("test_attention_gpu_parity (storage GPU vs CPU) ...")
    _gpu_parity[False]()
    _gpu_parity[True]()


def test_masked_cpu_parity() raises:
    print("test_masked_attention_cpu_parity (legacy vs storage) ...")
    _masked_cpu_parity(0)
    _masked_cpu_parity(1)
    _masked_cpu_parity(2)


def test_masked_gpu_parity() raises:
    print("test_masked_attention_gpu_parity (storage GPU vs CPU) ...")
    _masked_gpu_parity(0)
    _masked_gpu_parity(1)
    _masked_gpu_parity(2)


def main() raises:
    print("=" * 70)
    print("ScaledDotProductAttention legacy ↔ storage parity")
    print("=" * 70)
    test_cpu_parity()
    test_gpu_parity()
    test_masked_cpu_parity()
    test_masked_gpu_parity()
    print("ALL PASSED")
