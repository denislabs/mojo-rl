"""TimeAttentionLatents legacy ↔ storage parity (CPU) + storage GPU vs CPU.

Both paths carry the same gather/scatter kernels + inner causal MHA verbatim, so
legacy↔storage CPU is ~bit-identical (out + grad_input). storage GPU↔CPU to a
small tolerance. Also re-checks the structural invariants (non-latent outputs
and non-latent input grads are exactly 0). Run:
  rm -f mojo_rl.mojoc && pixi run mojo run -I . tests/nn/test_time_attention_latents_storage_parity.mojo
  rm -f mojo_rl.mojoc && pixi run -e apple mojo run -I . tests/nn/test_time_attention_latents_storage_parity.mojo
"""

from std.memory import alloc
from std.math import abs
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.time_attention_latents import (
    TimeAttentionLatents as LegacyTimeAttentionLatents,
)
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.time_attention_latents import (
    TimeAttentionLatents,
)


# Both leaves get the SAME two Linear weights via the shared pattern
# `(i % 7 - 3) * 0.1` (drilled into the inner MHA's two Linear leaves) so CPU
# parity exercises real (nonzero) attention math, not the all-zero case.


comptime D = 4
comptime NH = 2
comptime T = 3
comptime S = 5
comptime L = 2
comptime Bn = 2
comptime BATCH = Bn * T
comptime IN_N = BATCH * S * D
comptime OUT_N = BATCH * S * D


def _spread(i: Int, seed: Float64) -> Scalar[DT]:
    var x = seed + 0.7 * Float64(i)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return Scalar[DT](0.5 * (t - (t * t * t) / 6.0))


def test_cpu_parity() raises:
    print("test_tal_cpu_parity (legacy vs storage, CPU) ...")
    comptime TOL = Scalar[DT](1e-6)

    # Legacy. INIT=Zero, then overwrite params with the shared deterministic
    # pattern so the inner MHA projections match the storage leaf exactly.
    var leg = LegacyTimeAttentionLatents[D, NH, T, S, L].make[
        target="cpu", INIT=Zero
    ]()
    # Drill into the inner MHA (Sequential[Tokenwise[Linear], QKVToMajor, SDPA,
    # Tokenwise[Linear]]) and set both Linear weights to the SAME deterministic
    # pattern the storage `Deterministic` init produces.
    comptime W0 = D * (3 * D)   # first proj  Linear[D, 3*D]
    comptime W1 = D * D         # out  proj   Linear[D, D]
    var w0p = leg.mha.children[0].inner.weight.value_unsafe_ptr_cpu()
    for i in range(W0):
        w0p[i] = Scalar[DT]((i % 7) - 3) * 0.1
    var w1p = leg.mha.children[3].inner.weight.value_unsafe_ptr_cpu()
    for i in range(W1):
        w1p[i] = Scalar[DT]((i % 7) - 3) * 0.1
    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](IN_N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OUT_N)
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](OUT_N)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](IN_N)
    for i in range(IN_N):
        x[i] = _spread(i, 1.3)
    for i in range(OUT_N):
        go[i] = _spread(i, 4.1)

    var xt = TileTensor(x, row_major[BATCH, S * D]())
    var yt = TileTensor(y, row_major[BATCH, S * D]())
    var got = TileTensor(go, row_major[BATCH, S * D]())
    var git = TileTensor(gi, row_major[BATCH, S * D]())
    leg.forward["cpu", BATCH](xt, output=yt)
    leg.zero_grad["cpu"]()
    leg.vjp["cpu", BATCH](got, git)

    # Storage. Set the same two Linear weights by drilling the identical tree.
    var st = TimeAttentionLatents[D, NH, T, S, L].make["cpu", Deterministic]()
    for i in range(W0):
        st.mha.children[0].inner.weight.val.data[i] = (
            Scalar[DT]((i % 7) - 3) * 0.1
        )
    for i in range(W1):
        st.mha.children[3].inner.weight.val.data[i] = (
            Scalar[DT]((i % 7) - 3) * 0.1
        )
    var sx = Tensor.alloc(IN_N)
    var sgo = Tensor.alloc(OUT_N)
    var sout = Tensor.alloc(OUT_N)
    var sgi = Tensor.alloc(IN_N)
    for i in range(IN_N):
        sx.data[i] = x[i]
    for i in range(OUT_N):
        sgo.data[i] = go[i]
    st.forward["cpu", BATCH](TensorRefs[1](sx), sout, None)
    st.zero_grad["cpu"](None)
    st.vjp["cpu", BATCH](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    for i in range(OUT_N):
        if abs(sout.data[i] - y[i]) > mo:
            mo = abs(sout.data[i] - y[i])
    var mgi: Scalar[DT] = 0
    for i in range(IN_N):
        if abs(sgi.data[i] - gi[i]) > mgi:
            mgi = abs(sgi.data[i] - gi[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "TimeAttentionLatents CPU parity")

    # invariants on the storage output.
    var max_nl_out: Float64 = 0.0
    var max_nl_g: Float64 = 0.0
    for b in range(Bn):
        for t in range(T):
            for s in range(S):
                if s >= L:
                    for d in range(D):
                        var oo = abs(
                            Float64(sout.data[(b * T + t) * S * D + s * D + d])
                        )
                        if oo > max_nl_out:
                            max_nl_out = oo
                        var gg = abs(
                            Float64(sgi.data[(b * T + t) * S * D + s * D + d])
                        )
                        if gg > max_nl_g:
                            max_nl_g = gg
    print("  non-latent out max", max_nl_out, " gi max", max_nl_g)
    assert_true(max_nl_out == 0.0, "non-latent outputs must be 0")
    assert_true(max_nl_g == 0.0, "non-latent input grads must be 0")
    print("  ok")


def test_gpu_parity() raises:
    print("test_tal_gpu_parity (storage GPU vs CPU) ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = TimeAttentionLatents[D, NH, T, S, L].make["cpu", Deterministic]()
    var gpu = TimeAttentionLatents[D, NH, T, S, L].make["gpu", Deterministic](
        Optional(c)
    )

    var sx = Tensor.alloc(IN_N)
    var sgo = Tensor.alloc(OUT_N)
    for i in range(IN_N):
        sx.data[i] = _spread(i, 1.3)
    for i in range(OUT_N):
        sgo.data[i] = _spread(i, 4.1)
    var c_out = Tensor.alloc(OUT_N)
    var c_gi = Tensor.alloc(IN_N)
    cpu.forward["cpu", BATCH](TensorRefs[1](sx), c_out, None)
    cpu.vjp["cpu", BATCH](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(IN_N)
    var ggo = Tensor.alloc(OUT_N)
    for i in range(IN_N):
        gx.data[i] = sx.data[i]
    for i in range(OUT_N):
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(OUT_N)
    var g_gi = Tensor.alloc(IN_N)
    g_out.upload(c)
    g_gi.upload(c)
    gpu.forward["gpu", BATCH](TensorRefs[1](gx), g_out, Optional(c))
    gpu.vjp["gpu", BATCH](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)

    var mo: Scalar[DT] = 0
    for i in range(OUT_N):
        if abs(g_out.data[i] - c_out.data[i]) > mo:
            mo = abs(g_out.data[i] - c_out.data[i])
    var mgi: Scalar[DT] = 0
    for i in range(IN_N):
        if abs(g_gi.data[i] - c_gi.data[i]) > mgi:
            mgi = abs(g_gi.data[i] - c_gi.data[i])
    print("  max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "TimeAttentionLatents GPU vs CPU")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("TimeAttentionLatents legacy ↔ storage parity")
    print("=" * 70)
    test_cpu_parity()
    test_gpu_parity()
    print("ALL PASSED")
