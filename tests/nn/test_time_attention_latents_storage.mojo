"""TimeAttentionLatents storage primitive — CPU correctness (golden) + GPU vs CPU.

Standalone storage test (no legacy oracle — converted from the former
`_storage_parity` test in legacy-removal Phase 0b). The CPU check asserts the
storage forward/backward against golden fingerprints (S = Σ vᵢ, W = Σ vᵢ·(i+1) —
the weight catches sign/position errors that a plain sum would cancel), captured
from the bit-identical legacy↔storage run the parity test used to verify. It also
re-checks the structural invariants (non-latent outputs and non-latent input
grads are exactly 0). The GPU check is storage-only (GPU vs CPU consistency).
Run:
  rm -f mojo_rl.mojoc && pixi run -e apple mojo run -I . tests/nn/test_time_attention_latents_storage.mojo
"""

from std.math import abs
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.time_attention_latents import (
    TimeAttentionLatents,
)


# The two inner-MHA Linear weights are set via the shared deterministic pattern
# `(i % 7 - 3) * 0.1` (drilled into the inner MHA's two Linear leaves) so the CPU
# check exercises real (nonzero) attention math, not the all-zero case.


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


def test_tal_cpu_golden() raises:
    print("test_tal_cpu_golden (storage CPU vs golden) ...")
    comptime TOL = Scalar[DT](5e-3)

    comptime W0 = D * (3 * D)   # first proj  Linear[D, 3*D]
    comptime W1 = D * D         # out  proj   Linear[D, D]

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
        sx.data[i] = _spread(i, 1.3)
    for i in range(OUT_N):
        sgo.data[i] = _spread(i, 4.1)
    st.forward["cpu", BATCH](TensorRefs[1](sx), sout, None)
    st.zero_grad["cpu"](None)
    st.vjp["cpu", BATCH](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var ok = _check("out", sout, OUT_N, 3.9190578, 129.1099, TOL)
    ok = _check("gi", sgi, IN_N, 6.2534704, 321.60965, TOL) and ok
    assert_true(ok, "TimeAttentionLatents CPU golden")

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


def test_tal_gpu_vs_cpu() raises:
    print("test_tal_gpu_vs_cpu (storage GPU vs CPU) ...")
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
    print("TimeAttentionLatents storage primitive (CPU golden + GPU vs CPU)")
    print("=" * 70)
    test_tal_cpu_golden()
    test_tal_gpu_vs_cpu()
    print("ALL PASSED")
